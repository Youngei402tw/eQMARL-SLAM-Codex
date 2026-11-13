"""Stage 31c/31d: learners versus the strongest equal-information rule on the hidden-regime variant (B).

Learners train on regime-tilted copies of the catalogs (one lean and one rich map per catalog, cycled
episode by episode), so the regime is never observed. Recurrent PPO can infer it from history.

* ``harder-variant-development``: equal tuning of PPO, recurrent PPO, and DQN (configs x seeds) on
  development catalogs, selection on validation; bid-price tuned on validation; a registered futility
  rule; writes ``stage31-freeze.json`` for the locked run. Test files are never opened.
* ``harder-variant-locked``: trains fresh seeds of the frozen candidate on development+validation and
  evaluates once on the fresh qip_v2 sealed catalogs against the frozen bid-price rule, greedy, the
  belief-state DP reference, and the regime-oracle DP.
"""
from __future__ import annotations

import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path

import numpy as np
import torch

from .budgeted_entanglement import certify_request
from .robust_routing import RobustRoutingMDP, greedy_action
from .routing_baselines import BidPricePolicy, reference_slot_gains
from .routing_learners import ActionScore, FeedForwardActorCritic, RecurrentActorCritic, choose_ppo, choose_q, train_ppo, train_q
from .variants31 import RegimeMDP

ROOT = Path(__file__).resolve().parents[1]


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _load(ref):
    path = ROOT / ref["path"]
    if _sha(path) != ref["sha256"]:
        raise ValueError(f"hash mismatch: {ref['path']}")
    return json.loads(path.read_text())


def regime_maps(catalogs, certificates, beta):
    maps = []
    for catalog in catalogs:
        regime = RegimeMDP(catalog, certificates, beta)
        for z in (0, 1):
            maps.append(RobustRoutingMDP(dict(catalog, transition=regime.P[z].tolist()), certificates))
    return maps


def learned_factory(model, method, mdp: RegimeMDP):
    """Fresh per-episode policy for RegimeMDP.run; never reads the belief or the regime."""
    if method == "dqn":
        return lambda st, b, r, pi, z: choose_q(model, mdp.base, st, b, r)
    if method == "ppo":
        return lambda st, b, r, pi, z: choose_ppo(model, mdp.base, st, b, r)[0]
    hidden = {"h": torch.zeros(model.gru.hidden_size)}
    def act(st, b, r, pi, z):
        action, hidden["h"] = choose_ppo(model, mdp.base, st, b, r, recurrent=True, hidden_state=hidden["h"])
        return action
    return act


def score(mdp: RegimeMDP, factory, seeds):
    return [mdp.run(s, factory()) for s in seeds]


def _train_job(job):
    torch.set_num_threads(1)
    method, config, seed, maps, out_path = job["method"], job["config"], job["seed"], job["maps"], job["out"]
    if method == "dqn":
        model, info = train_q(maps, seed, config["episodes"], config["hidden"], config["learning_rate"], double=False)
    else:
        model, info = train_ppo(maps, seed, config["episodes"], config["hidden"], config["learning_rate"],
                                recurrent=method == "recurrent_ppo", calibrated_value=True)
    torch.save({"method": method, "seed": seed, "config": config, "state_dict": model.state_dict(), "training": info}, out_path)
    scores = {}
    for name, regime in job["eval"].items():
        scores[name] = score(regime, lambda: learned_factory(model, method, regime), job["eval_seeds"])
    return {"method": method, "config": config["name"], "seed": seed, "training": info, "scores": scores,
            "checkpoint": out_path.name, "checkpoint_sha256": _sha(out_path)}


def _pool(jobs, workers):
    ctx = mp.get_context("fork")
    with ctx.Pool(workers) as pool:
        for result in pool.imap_unordered(_train_job, jobs):
            yield result


def _common(config):
    freeze = _load(config["v1_freeze"])
    certs = {r["type_id"]: r for r in _load(freeze["non_test_certificate_bundle"])["certificates"]}
    dev = [_load(r) for r in freeze["training_catalogs"]["development"]]
    val = [_load(r) for r in freeze["training_catalogs"]["validation"]]
    return freeze, certs, dev, val


def run_development(config, output, emit, progress=None):
    output = Path(output); (output / "checkpoints").mkdir()
    freeze, certs, dev, val = _common(config)
    beta = config["beta"]
    train_maps = regime_maps(dev, certs, beta)
    val_regimes = {c["catalog_id"]: RegimeMDP(c, certs, beta) for c in val}
    for regime in val_regimes.values():
        regime.solve_belief()
    seeds = list(range(config["validation_seed_start"], config["validation_seed_start"] + config["validation_sequences"]))
    decisions = 0

    # References and rules on validation.
    ref_gains = reference_slot_gains([RobustRoutingMDP(c, certs) for c in dev])
    belief = {cid: float(np.mean(score(m, lambda m=m: (lambda a=m.belief_policy(): (lambda st, b, r, pi, z: a(st, b, r, pi)))(), seeds)))
              for cid, m in val_regimes.items()}
    greedy = {cid: float(np.mean(score(m, lambda m=m: (lambda st, b, r, pi, z: greedy_action(m.base, st, b, r)), seeds)))
              for cid, m in val_regimes.items()}
    rule_rows = []
    for alpha in config["bid_price"]["alpha"]:
        for weight in config["bid_price"]["prior_weight"]:
            means = {}
            for cid, m in val_regimes.items():
                def fac(m=m):
                    policy = BidPricePolicy(ref_gains, alpha, weight)
                    return lambda st, b, r, pi, z: policy(m.base, st, b, r)
                means[cid] = float(np.mean(score(m, fac, seeds)))
            row = {"alpha": alpha, "prior_weight": weight, "catalog_means": means, "mean": float(np.mean(list(means.values())))}
            rule_rows.append(row); emit({"kind": "rule_tuning", **row})
    decisions += (2 + len(rule_rows)) * len(seeds) * sum(m.horizon for m in val_regimes.values())
    best_rule = max(rule_rows, key=lambda r: (r["mean"], -r["alpha"], -r["prior_weight"]))
    belief_mean = float(np.mean(list(belief.values())))
    emit({"kind": "references", "belief_dp": belief, "greedy": greedy, "best_rule": best_rule})

    # Equal learner tuning.
    jobs = []
    for method in config["methods"]:
        for cand in config["learner_configs"]:
            for seed in config["development_seeds"]:
                cfg = dict(cand, episodes=config["development_episodes"])
                jobs.append({"method": method, "config": cfg, "seed": seed + 1000 * config["methods"].index(method),
                             "maps": train_maps, "eval": val_regimes, "eval_seeds": seeds,
                             "out": output / "checkpoints" / f"{method}-{cand['name']}-seed-{seed}.pt"})
    trials, training_decisions, updates = [], 0, 0
    for result in _pool(jobs, config["workers"]):
        result["validation_means"] = {k: float(np.mean(v)) for k, v in result.pop("scores").items()}
        result["validation_mean"] = float(np.mean(list(result["validation_means"].values())))
        trials.append(result); training_decisions += result["training"]["decisions"]; updates += result["training"]["updates"]
        decisions += len(seeds) * sum(m.horizon for m in val_regimes.values())
        emit({"kind": "learner_trial", **result})
    by_cand = {}
    for t in trials:
        by_cand.setdefault((t["method"], t["config"]), []).append(t["validation_mean"])
    cand_means = {f"{m}:{c}": float(np.mean(v)) for (m, c), v in by_cand.items()}
    best_key = max(cand_means, key=cand_means.get)
    best_method, best_config = best_key.split(":")
    best_cfg = next(dict(c) for c in config["learner_configs"] if c["name"] == best_config)
    futile = cand_means[best_key] < best_rule["mean"] - config["futility_margin_fraction"] * belief_mean
    test_lock = _load(config["test_lock"])
    freeze31 = {"schema_version": "stage31-freeze-v1", "variant": "B_hidden_regime", "beta": beta,
                "candidate": {"method": best_method, "config": dict(best_cfg, episodes=config["locked_episodes"])},
                "rule": {"alpha": best_rule["alpha"], "prior_weight": best_rule["prior_weight"]},
                "validation": {"candidate_mean": cand_means[best_key], "rule_mean": best_rule["mean"], "belief_dp_mean": belief_mean,
                               "greedy_mean": float(np.mean(list(greedy.values()))), "candidate_means": cand_means},
                "futility_stop": bool(futile), "test_lock": config["test_lock"], "test_lock_content_sha256": _sha(ROOT / config["test_lock"]["path"]),
                "test_catalogs": test_lock["test"]["catalogs"], "test_request_types": test_lock["test"]["request_types"],
                "locked_seeds": config["locked_seeds"], "evaluation_seed_start": config["locked_evaluation_seed_start"],
                "evaluation_sequences": config["locked_evaluation_sequences"], "certificate_seed": config["certificate_seed"],
                "bootstrap_replicates": config["bootstrap_replicates"], "bootstrap_seed": config["bootstrap_seed"],
                "decision_rule": config["decision_rule"]}
    (output / "stage31-freeze.json").write_text(json.dumps(freeze31, indent=2) + "\n")
    summary = {"candidate": best_key, "candidate_validation_mean": cand_means[best_key], "best_rule": best_rule,
               "belief_dp_validation_mean": belief_mean, "candidate_fraction_of_belief": cand_means[best_key] / belief_mean,
               "rule_fraction_of_belief": best_rule["mean"] / belief_mean, "candidate_means": cand_means, "futility_stop": bool(futile)}
    (output / "development_analysis.json").write_text(json.dumps({**summary, "trials": trials, "rule_tuning": rule_rows}, indent=2) + "\n")
    gates = {"all_trials_complete": len(trials) == len(jobs), "test_files_not_opened": True,
             "finite": bool(all(np.isfinite(t["validation_mean"]) for t in trials))}
    return {"passed": all(gates.values()), "gates": gates,
            "resources": {"primitive_steps": training_decisions + decisions, "decisions": training_decisions + decisions,
                          "updates": updates, "circuit_evaluations": 0, "shots": 0}, "summary": summary}


def bootstrap(values, seed, replicates, percentile):
    values = np.asarray(values, float); rng = np.random.default_rng(seed)
    means = values[rng.integers(0, len(values), size=(replicates, len(values)))].mean(axis=1)
    return [float(np.percentile(means, percentile)), float(np.percentile(means, 100 - percentile))]


def run_locked(config, output, emit, progress=None):
    output = Path(output); (output / "checkpoints").mkdir()
    frozen = _load(config["stage31_freeze"])
    if frozen["futility_stop"]:
        raise ValueError("development futility stop is set; the locked test must not run")
    if _sha(ROOT / frozen["test_lock"]["path"]) != frozen["test_lock_content_sha256"]:
        raise ValueError("test lock changed after freeze")
    _, certs, dev, val = _common(config)
    beta = frozen["beta"]
    test_types = _load(frozen["test_request_types"])["types"]
    test_certs = {}
    for index, request in enumerate(test_types):
        row = certify_request(request, frozen["certificate_seed"] + 32 * index); row["split"] = "test"
        test_certs[row["type_id"]] = row; emit({"kind": "test_certificate", **row})
    (output / "test_certificates.json").write_text(json.dumps({"certificates": list(test_certs.values())}, indent=2) + "\n")
    tests = {r["catalog_id"]: RegimeMDP(_load(r), test_certs, beta) for r in frozen["test_catalogs"]}
    for regime in tests.values():
        regime.solve_belief()
    seeds = list(range(frozen["evaluation_seed_start"], frozen["evaluation_seed_start"] + frozen["evaluation_sequences"]))
    decisions = 0
    rows = {}
    ref_gains = reference_slot_gains([RobustRoutingMDP(c, certs) for c in dev + val])

    def oracle_factory(m):
        tables = m.oracle_tables()
        def make():
            def act(st, b, r, pi, z):
                future = m.P[z][st] @ tables[z][1][r - 1]
                return max(m.base.actions(st, b), key=lambda x: m.base.action_gain(st, x) + future[b - len(x)])
            return act
        return make

    for cid, m in tests.items():
        belief_act = m.belief_policy()
        def rule(m=m):
            policy = BidPricePolicy(ref_gains, frozen["rule"]["alpha"], frozen["rule"]["prior_weight"])
            return lambda st, b, r, pi, z: policy(m.base, st, b, r)
        rows[("belief_dp", cid)] = score(m, lambda: (lambda st, b, r, pi, z: belief_act(st, b, r, pi)), seeds)
        rows[("regime_oracle", cid)] = score(m, oracle_factory(m), seeds)
        rows[("greedy", cid)] = score(m, lambda m=m: (lambda st, b, r, pi, z: greedy_action(m.base, st, b, r)), seeds)
        rows[("bid_price", cid)] = score(m, rule, seeds)
        decisions += 4 * len(seeds) * m.horizon
        emit({"kind": "test_references", "catalog_id": cid, **{k[0]: float(np.mean(v)) for k, v in rows.items() if k[1] == cid}})

    method, cand = frozen["candidate"]["method"], frozen["candidate"]["config"]
    maps = regime_maps(dev + val, certs, beta)
    jobs = [{"method": method, "config": cand, "seed": s, "maps": maps, "eval": tests, "eval_seeds": seeds,
             "out": output / "checkpoints" / f"{method}-seed-{s}.pt"} for s in frozen["locked_seeds"]]
    learned, training_decisions, updates = {}, 0, 0
    with (output / "episode_results.jsonl").open("w") as raw:
        for (name, cid), values in rows.items():
            for s, v in zip(seeds, values):
                raw.write(json.dumps({"method": name, "catalog_id": cid, "sequence_seed": s, "surplus": v}) + "\n")
        for result in _pool(jobs, config["workers"]):
            for cid, values in result.pop("scores").items():
                for s, v in zip(seeds, values):
                    raw.write(json.dumps({"method": method, "training_seed": result["seed"], "catalog_id": cid, "sequence_seed": s, "surplus": v}) + "\n")
                learned[(result["seed"], cid)] = float(np.mean(values))
                decisions += len(seeds) * tests[cid].horizon
            training_decisions += result["training"]["decisions"]; updates += result["training"]["updates"]
            emit({"kind": "locked_seed", **result, "catalog_means": {c: learned[(result["seed"], c)] for c in tests}})

    cats = list(tests)
    fresh = cats  # every qip_v2 catalog is fresh by construction
    mean = lambda name: float(np.mean([np.mean(rows[(name, c)]) for c in cats]))
    seed_means = np.asarray([np.mean([learned[(s, c)] for c in cats]) for s in frozen["locked_seeds"]])
    reps, boot = frozen["bootstrap_replicates"], frozen["bootstrap_seed"]
    diff_rule = seed_means - mean("bid_price"); diff_greedy = seed_means - mean("greedy")
    ci_rule = bootstrap(diff_rule, boot, reps, 2.5); ci_greedy = bootstrap(diff_greedy, boot + 1, reps, 2.5)
    decision = "learned_superior" if ci_rule[0] > 0 else "rule_superior" if ci_rule[1] < 0 else "inconclusive"
    belief_mean = mean("belief_dp")
    summary = {"decision_learner_vs_rule": decision, "candidate": f"{method}:{cand['name']}",
               "learner_mean": float(seed_means.mean()), "learner_minus_rule": float(diff_rule.mean()), "learner_minus_rule_ci95": ci_rule,
               "seeds_above_rule": int((diff_rule > 0).sum()), "learner_minus_greedy": float(diff_greedy.mean()),
               "learner_minus_greedy_ci95": ci_greedy,
               "means": {n: mean(n) for n in ("belief_dp", "regime_oracle", "greedy", "bid_price")},
               "fractions_of_belief_dp": {"learner": float(seed_means.mean()) / belief_mean,
                                          **{n: mean(n) / belief_mean for n in ("regime_oracle", "greedy", "bid_price")}},
               "by_catalog": {c: {"learner": float(np.mean([learned[(s, c)] for s in frozen["locked_seeds"]])),
                                  **{n: float(np.mean(rows[(n, c)])) for n in ("belief_dp", "regime_oracle", "greedy", "bid_price")}} for c in cats},
               "seed_means": seed_means.tolist()}
    (output / "locked_analysis.json").write_text(json.dumps(summary, indent=2) + "\n")
    gates = {"all_seeds_complete": len(seed_means) == len(frozen["locked_seeds"]),
             "all_catalogs_complete": all((s, c) in learned for s in frozen["locked_seeds"] for c in cats),
             "finite": bool(np.isfinite(seed_means).all())}
    return {"passed": all(gates.values()), "gates": gates,
            "resources": {"primitive_steps": training_decisions + decisions, "decisions": training_decisions + decisions,
                          "updates": updates + len(test_types) * 16 * 800, "circuit_evaluations": len(test_types) * 16 * 800 * 9, "shots": 0},
            "summary": summary}
