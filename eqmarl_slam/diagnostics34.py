"""Stage 34: unified confirmatory evaluation on a sealed test split opened exactly once.

Every method is fixed before the test split is read:
  1. tune the bid-price rule on the validation catalogs only (CE-DP needs no tuning at K=0) and write
     the selection to disk;
  2. train fresh PPO and DQN models on the 12 training catalogs only and save their checkpoints;
  3. only then load the sealed qip_v2 test split (hash-checked), certify its request types, and evaluate
     every method once on the same paired sequences.
"""
from __future__ import annotations

import hashlib
import json
import multiprocessing as mp
from pathlib import Path

import numpy as np
import torch

from .budgeted_entanglement import certify_request
from .robust_routing import RobustRoutingMDP, clairvoyant_value, greedy_action
from .routing_baselines import BidPricePolicy, PlannerPolicy, mean_self_loop, prior_transition, reference_slot_gains
from .routing_learners import evaluate_model, train_ppo, train_q

ROOT = Path(__file__).resolve().parents[1]


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _load(ref):
    path = ROOT / ref["path"]
    if _sha(path) != ref["sha256"]:
        raise ValueError(f"hash mismatch: {ref['path']}")
    return json.loads(path.read_text())


def bootstrap(values, seed, replicates, level):
    values = np.asarray(values, float)
    rng = np.random.default_rng(seed)
    means = values[rng.integers(0, len(values), size=(replicates, len(values)))].mean(axis=1)
    tail = (100 - level) / 2
    return [float(np.percentile(means, tail)), float(np.percentile(means, 100 - tail))]


def rule_episodes(mdp, factory, seeds):
    out = []
    for s in seeds:
        result = mdp.evaluate(mdp.sample_sequence(s), factory(), seed=s + 810000)
        if result["violations"]:
            raise ValueError("budget violation")
        out.append(float(result["surplus"]))
    return out


def tune_bid_price(dev, val, grid, seeds):
    reference = reference_slot_gains(dev)
    rows = []
    for alpha in grid["alpha"]:
        for weight in grid["prior_weight"]:
            means = {m.catalog["catalog_id"]: float(np.mean(rule_episodes(m, lambda: BidPricePolicy(reference, alpha, weight), seeds))) for m in val}
            rows.append({"alpha": alpha, "prior_weight": weight, "catalog_means": means, "mean": float(np.mean(list(means.values())))})
    best = max(rows, key=lambda r: (r["mean"], -r["alpha"], -r["prior_weight"]))
    return rows, best


def train_job(job):
    torch.set_num_threads(1)
    method, cfg, seed = job["method"], job["config"], job["seed"]
    if method == "ppo":
        model, info = train_ppo(job["maps"], seed, cfg["episodes"], cfg["hidden"], cfg["learning_rate"], recurrent=False, calibrated_value=True)
    elif method == "dqn":
        model, info = train_q(job["maps"], seed, cfg["episodes"], cfg["hidden"], cfg["learning_rate"], double=False)
    else:
        raise ValueError(method)
    torch.save({"method": method, "seed": seed, "config": cfg, "state_dict": model.state_dict(), "training": info}, job["out"])
    return {"method": method, "seed": seed, "training": info, "checkpoint": job["out"].name, "checkpoint_sha256": _sha(job["out"])}


def load_model(path):
    from .routing_learners import ActionScore, FeedForwardActorCritic
    payload = torch.load(path, map_location="cpu", weights_only=False)
    hidden = payload["config"]["hidden"]
    model = ActionScore(hidden) if payload["method"] == "dqn" else FeedForwardActorCritic(hidden)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model


def eval_job(job):
    torch.set_num_threads(1)
    model = load_model(job["checkpoint"])
    return {"method": job["method"], "seed": job["seed"],
            "per_catalog": {m.catalog["catalog_id"]: [float(v) for v in evaluate_model(model, job["method"], m, job["seeds"])] for m in job["test"]}}


def run_diagnostics(config, output, emit, progress=None):
    output = Path(output)
    (output / "checkpoints").mkdir()
    freeze = _load(config["v1_freeze"])
    certs = {r["type_id"]: r for r in _load(freeze["non_test_certificate_bundle"])["certificates"]}
    dev = [RobustRoutingMDP(_load(r), certs) for r in freeze["training_catalogs"]["development"]]
    val = [RobustRoutingMDP(_load(r), certs) for r in freeze["training_catalogs"]["validation"]]
    if any(m.catalog["split"] == "test" for m in dev + val):
        raise ValueError("test catalog among training catalogs")
    decisions = 0

    # ---- 1. validation-only tuning --------------------------------------------------------------
    tune_seeds = list(range(config["validation_seed_start"], config["validation_seed_start"] + config["validation_sequences"]))
    rows, best = tune_bid_price(dev, val, config["bid_price"], tune_seeds)
    decisions += len(rows) * len(tune_seeds) * sum(int(m.catalog["horizon"]) for m in val)
    for r in rows:
        emit({"kind": "tuning_bid_price", **r})
    rho = mean_self_loop(dev + val)
    selection = {"bid_price": {"alpha": best["alpha"], "prior_weight": best["prior_weight"], "validation_mean": best["mean"]},
                 "certainty_equivalent": {"warmup_episodes": 0, "rho_training": rho}, "test_data_loaded": False}
    (output / "tuning_selection.json").write_text(json.dumps(selection, indent=2) + "\n")
    emit({"kind": "tuning_selection", **selection})

    # ---- 2. train fresh learners on the 12 training catalogs only -------------------------------
    maps = dev + val
    jobs = []
    for method in config["methods"]:
        for seed in config["training_seeds"][method]:
            jobs.append({"method": method, "config": config["learner_configs"][method], "seed": seed, "maps": maps,
                         "out": output / "checkpoints" / f"{method}-seed-{seed}.pt"})
    trained, updates, train_decisions = [], 0, 0
    with mp.get_context("fork").Pool(config["workers"]) as pool:
        for result in pool.imap_unordered(train_job, jobs):
            trained.append(result); updates += result["training"]["updates"]; train_decisions += result["training"]["decisions"]
            emit({"kind": "training_complete", **result})
    if len(trained) != len(jobs):
        raise RuntimeError("not every model trained")
    (output / "training_complete.json").write_text(json.dumps({"models": sorted((t["method"], t["seed"], t["checkpoint_sha256"]) for t in trained),
                                                               "test_data_loaded": False}, indent=2) + "\n")

    # ---- 3. open the sealed test split once ------------------------------------------------------
    lock = _load(config["test_split_lock"])
    types = _load(lock["test"]["request_types"])["types"]
    test_certs = []
    for index, request in enumerate(types):
        row = certify_request(request, config["certificate_seed"] + 32 * index)
        row["split"] = "test"
        if abs(row["sdp_gap"]) > config["certificate_tolerance"]:
            raise ValueError(f"certificate gap too large for {row['type_id']}")
        test_certs.append(row)
    (output / "test_certificates.json").write_text(json.dumps({"certificates": test_certs}, indent=2) + "\n")
    cert_map = {r["type_id"]: r for r in test_certs}
    test = [RobustRoutingMDP(_load(r), cert_map) for r in lock["test"]["catalogs"]]
    if any(m.catalog["split"] != "test" for m in test):
        raise ValueError("unexpected split in the sealed test lock")
    seeds = list(range(config["evaluation_seed_start"], config["evaluation_seed_start"] + config["evaluation_sequences"]))

    full_reference = reference_slot_gains(dev + val)
    rules = {}
    for m in test:
        n = len(m.catalog["states"])
        true_p = np.asarray(m.catalog["transition"], float)
        online = PlannerPolicy(m, true_p)
        ce = PlannerPolicy(m, prior_transition(n, rho))
        cid = m.catalog["catalog_id"]
        rules[cid] = {
            "greedy": rule_episodes(m, lambda: greedy_action, seeds),
            "bid_price": rule_episodes(m, lambda: BidPricePolicy(full_reference, best["alpha"], best["prior_weight"]), seeds),
            "ce_dp_k0": rule_episodes(m, lambda: ce, seeds),
            "online_dp": rule_episodes(m, lambda: online, seeds),
            "future_clairvoyant": [float(clairvoyant_value(m, m.sample_sequence(s))) for s in seeds],
        }
        decisions += 5 * len(seeds) * int(m.catalog["horizon"])
    eval_jobs = [{"method": t["method"], "seed": t["seed"], "checkpoint": str(output / "checkpoints" / t["checkpoint"]),
                  "test": test, "seeds": seeds} for t in sorted(trained, key=lambda t: (t["method"], t["seed"]))]
    with mp.get_context("fork").Pool(config["workers"]) as pool:
        learned = list(pool.imap_unordered(eval_job, eval_jobs))
    decisions += len(learned) * len(seeds) * sum(int(m.catalog["horizon"]) for m in test)
    with (output / "episode_results.jsonl").open("w") as stream:
        for cid, by_method in rules.items():
            for method, values in by_method.items():
                for s, v in zip(seeds, values):
                    stream.write(json.dumps({"kind": "rule_episode", "method": method, "catalog_id": cid, "sequence_seed": s, "surplus": v}) + "\n")
        for row in sorted(learned, key=lambda r: (r["method"], r["seed"])):
            for cid, values in row["per_catalog"].items():
                for s, v in zip(seeds, values):
                    stream.write(json.dumps({"kind": "learned_episode", "method": row["method"], "training_seed": row["seed"],
                                             "catalog_id": cid, "sequence_seed": s, "surplus": v}) + "\n")

    # ---- 4. registered analysis ------------------------------------------------------------------
    cids = [m.catalog["catalog_id"] for m in test]
    rule_mean = {k: float(np.mean([np.mean(rules[c][k]) for c in cids])) for k in rules[cids[0]]}
    seed_score = {}
    for row in learned:
        seed_score.setdefault(row["method"], {})[row["seed"]] = float(np.mean([np.mean(row["per_catalog"][c]) for c in cids]))
    online_mean = rule_mean["online_dp"]
    reps, bseed = config["bootstrap_replicates"], config["bootstrap_seed"]

    def contrast(method, against, k):
        diffs = np.asarray([seed_score[method][s] - rule_mean[against] for s in sorted(seed_score[method])])
        return {"mean": float(diffs.mean()), "ci95": bootstrap(diffs, bseed + k, reps, 95), "seeds_above": int((diffs > 0).sum())}

    contrasts = {"ppo_minus_bid_price": contrast("ppo", "bid_price", 0), "ppo_minus_greedy": contrast("ppo", "greedy", 1),
                 "dqn_minus_bid_price": contrast("dqn", "bid_price", 2), "dqn_minus_greedy": contrast("dqn", "greedy", 3),
                 "ppo_minus_ce_dp_k0": contrast("ppo", "ce_dp_k0", 4)}
    paired = np.asarray([seed_score["ppo"][a] - seed_score["dqn"][b]
                         for a, b in zip(sorted(seed_score["ppo"]), sorted(seed_score["dqn"]))])
    margin = config["equivalence_margin_fraction_online"] * online_mean
    equivalence = {"mean": float(paired.mean()), "ci90": bootstrap(paired, bseed + 5, reps, 90), "ci95": bootstrap(paired, bseed + 6, reps, 95),
                   "margin": margin}
    equivalence["equivalent"] = bool(-margin < equivalence["ci90"][0] and equivalence["ci90"][1] < margin)
    p = contrasts["ppo_minus_bid_price"]["ci95"]
    decision = "bid_price_superior" if p[1] < 0 else "learned_superior" if p[0] > 0 else "inconclusive"
    fractions = {k: v / online_mean for k, v in rule_mean.items()}
    for method in seed_score:
        fractions[method] = float(np.mean(list(seed_score[method].values()))) / online_mean
    per_catalog = {c: {**{k: float(np.mean(rules[c][k])) for k in rules[c]},
                       **{m: float(np.mean([np.mean(r["per_catalog"][c]) for r in learned if r["method"] == m])) for m in seed_score}}
                   for c in cids}
    summary = {"decision_primary": decision, "ppo_superior_to_greedy": contrasts["ppo_minus_greedy"]["ci95"][0] > 0,
               "dqn_vs_bid_price": ("bid_price_superior" if contrasts["dqn_minus_bid_price"]["ci95"][1] < 0 else
                                    "learned_superior" if contrasts["dqn_minus_bid_price"]["ci95"][0] > 0 else "inconclusive"),
               "ppo_dqn_equivalent": equivalence["equivalent"], "contrasts": contrasts, "ppo_minus_dqn": equivalence,
               "fraction_online_dp": fractions, "rule_means": rule_mean, "online_dp_mean": online_mean,
               "per_catalog_means": per_catalog, "bid_price_selection": selection["bid_price"], "rho": rho}
    (output / "confirmatory_analysis.json").write_text(json.dumps({**summary, "seed_scores": seed_score}, indent=2) + "\n")
    emit({"kind": "analysis", **summary})
    gates = {"all_models_trained": len(trained) == len(jobs), "all_models_evaluated": len(learned) == len(jobs),
             "certificates_within_tolerance": True, "selection_written_before_test": True,
             "finite": bool(np.isfinite(list(rule_mean.values())).all())}
    return {"passed": all(gates.values()), "gates": gates,
            "resources": {"primitive_steps": decisions + train_decisions, "decisions": decisions + train_decisions,
                          "updates": updates, "circuit_evaluations": 0, "shots": 0},
            "summary": {k: v for k, v in summary.items() if k in ("decision_primary", "ppo_superior_to_greedy", "dqn_vs_bid_price", "ppo_dqn_equivalent")}}
