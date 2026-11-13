"""Stage 30: stronger non-learned comparators for the locked routing benchmark.

Order of operations is fixed and enforced in code:

1. Tune the bid-price heuristic and certainty-equivalent DP on the validation catalogs only, using
   development catalogs as the heuristic's reference distribution. Write ``tuning_selection.json``.
2. Only then load the sealed test catalogs and evaluate the selected configurations once on the
   500 paired Stage-28 evaluation sequences per catalog.
3. Compare against the archived Stage-28 learned rows (read-only; no checkpoint is loaded or retrained).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from .robust_routing import RobustRoutingMDP, greedy_action
from .routing_baselines import (BidPricePolicy, PlannerPolicy, estimate_transition, mean_self_loop,
                                reference_slot_gains)

ROOT = Path(__file__).resolve().parents[1]


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _load_checked(ref):
    path = ROOT / ref["path"]
    if _sha(path) != ref["sha256"]:
        raise ValueError(f"hash mismatch: {ref['path']}")
    return json.loads(path.read_text())


def bootstrap(values, seed, replicates, percentile):
    values = np.asarray(values, dtype=float); rng = np.random.default_rng(seed)
    means = values[rng.integers(0, len(values), size=(replicates, len(values)))].mean(axis=1)
    return [float(np.percentile(means, percentile)), float(np.percentile(means, 100 - percentile))]


def _episodes(mdp, policy_factory, seeds, counter):
    values = []
    for seed in seeds:
        result = mdp.evaluate(mdp.sample_sequence(seed), policy_factory(), seed=seed + 810000)
        if result["violations"]:
            raise ValueError("budget violation")
        values.append(result["surplus"]); counter["decisions"] += len(result["actions"])
    return values


def _warmup(mdp, count, seed_base):
    return [mdp.sample_sequence(seed_base + j) for j in range(count)]


def run_diagnostics(config, output, emit, progress=None):
    output = Path(output)
    freeze = _load_checked(config["freeze"])
    stage28 = ROOT / config["stage28_bundle"]
    manifest = json.loads((stage28 / "artifact_manifest.json").read_text())
    hashes = {row["path"]: row["sha256"] for row in manifest["artifacts"]}
    for name in ("episode_results.jsonl", "test_certificates.json"):
        if _sha(stage28 / name) != hashes[name]:
            raise ValueError(f"Stage 28 artifact hash mismatch: {name}")
    non_test = {row["type_id"]: row for row in _load_checked(freeze["non_test_certificate_bundle"])["certificates"]}
    dev = [RobustRoutingMDP(_load_checked(r), non_test) for r in freeze["training_catalogs"]["development"]]
    val = [RobustRoutingMDP(_load_checked(r), non_test) for r in freeze["training_catalogs"]["validation"]]
    counter = {"decisions": 0, "dp_solves": 0, "episodes": 0}
    tuning = config["tuning"]
    val_seeds = list(range(tuning["validation_seed_start"], tuning["validation_seed_start"] + tuning["validation_sequences_per_catalog"]))

    # ---- Phase 1: validation-only tuning -------------------------------------------------------
    dev_reference = reference_slot_gains(dev)
    bid_rows = []
    for alpha in tuning["bid_price"]["alpha"]:
        for weight in tuning["bid_price"]["prior_weight"]:
            means = {m.catalog["catalog_id"]: float(np.mean(_episodes(
                m, lambda: BidPricePolicy(dev_reference, alpha, weight), val_seeds, counter))) for m in val}
            row = {"alpha": alpha, "prior_weight": weight, "catalog_means": means, "mean": float(np.mean(list(means.values())))}
            bid_rows.append(row); emit({"kind": "tuning_bid_price", **row})
    bid_best = max(bid_rows, key=lambda r: (r["mean"], -r["alpha"], -r["prior_weight"]))

    rho_dev = mean_self_loop(dev)
    ce_rows = []
    for k in tuning["certainty_equivalent"]["warmup_episodes"]:
        kappas = [None] if k == 0 else tuning["certainty_equivalent"]["kappa"]
        for kappa in kappas:
            means = {}
            for index, m in enumerate(val):
                sequences = _warmup(m, k, tuning["validation_warmup_seed_start"] + 1000 * index)
                transition = estimate_transition(len(m.catalog["states"]), sequences, rho_dev, kappa or 1.0)
                planner = PlannerPolicy(m, transition); counter["dp_solves"] += 1
                means[m.catalog["catalog_id"]] = float(np.mean(_episodes(m, lambda: planner, val_seeds, counter)))
            row = {"warmup_episodes": k, "kappa": kappa, "catalog_means": means, "mean": float(np.mean(list(means.values())))}
            ce_rows.append(row); emit({"kind": "tuning_certainty_equivalent", **row})
    ce_best = {}
    for k in tuning["certainty_equivalent"]["warmup_episodes"]:
        best = max((r for r in ce_rows if r["warmup_episodes"] == k), key=lambda r: (r["mean"], -(r["kappa"] or 0)))
        ce_best[k] = best["kappa"]
    selection = {"bid_price": {"alpha": bid_best["alpha"], "prior_weight": bid_best["prior_weight"],
                               "validation_mean": bid_best["mean"]},
                 "certainty_equivalent": {"rho_development": rho_dev, "kappa_by_warmup": {str(k): v for k, v in ce_best.items()}},
                 "test_data_loaded": False}
    (output / "tuning_selection.json").write_text(json.dumps(selection, indent=2) + "\n")
    emit({"kind": "tuning_selection", **selection})

    # ---- Phase 2: one evaluation on the sealed catalogs ----------------------------------------
    test_certs = {row["type_id"]: row for row in json.loads((stage28 / "test_certificates.json").read_text())["certificates"]}
    test = [RobustRoutingMDP(_load_checked(r), test_certs) for r in freeze["test_catalogs"]]
    seeds = list(range(freeze["evaluation_seed_start"], freeze["evaluation_seed_start"] + freeze["sequences_per_catalog"]))
    full_reference = reference_slot_gains(dev + val)
    rho_full = mean_self_loop(dev + val)
    stage28_rows = {}
    with (stage28 / "episode_results.jsonl").open() as stream:
        for line in stream:
            row = json.loads(line)
            if row["kind"] == "baseline_episode":
                stage28_rows[(row["method"], None, row["catalog_id"], row["sequence_seed"])] = row["surplus"]
            elif row.get("condition") == "ideal":
                stage28_rows[(row["method"], row["training_seed"], row["catalog_id"], row["sequence_seed"])] = row["surplus"]
    results = {}
    reproduction_error = 0.0
    with (output / "episode_results.jsonl").open("w") as raw:
        def record(method, mdp, values):
            results.setdefault(method, {})[mdp.catalog["catalog_id"]] = values
            for seed, value in zip(seeds, values):
                raw.write(json.dumps({"method": method, "catalog_id": mdp.catalog["catalog_id"],
                                      "sequence_seed": seed, "surplus": value}) + "\n")
        for index, mdp in enumerate(test):
            cid = mdp.catalog["catalog_id"]
            for method, factory in (("greedy_recomputed", lambda: greedy_action),
                                    ("online_dp_recomputed", (lambda p: (lambda: p))(PlannerPolicy(mdp, np.asarray(mdp.catalog["transition"], float))))):
                values = _episodes(mdp, factory, seeds, counter); record(method, mdp, values)
                archived = "greedy" if method == "greedy_recomputed" else "online_reference"
                reproduction_error = max(reproduction_error, max(abs(v - stage28_rows[(archived, None, cid, s)]) for v, s in zip(values, seeds)))
            counter["dp_solves"] += 1
            sel = selection["bid_price"]
            record("bid_price", mdp, _episodes(mdp, lambda: BidPricePolicy(full_reference, sel["alpha"], sel["prior_weight"]), seeds, counter))
            for k, kappa in ce_best.items():
                sequences = _warmup(mdp, k, config["test_warmup_seed_start"] + 1000 * index)
                planner = PlannerPolicy(mdp, estimate_transition(len(mdp.catalog["states"]), sequences, rho_full, kappa or 1.0))
                counter["dp_solves"] += 1
                record(f"ce_dp_k{k}", mdp, _episodes(mdp, lambda: planner, seeds, counter))
            emit({"kind": "test_catalog", "catalog_id": cid,
                  "means": {m: float(np.mean(v[cid])) for m, v in results.items()}})
    counter["episodes"] = sum(len(v) for cats in results.values() for v in cats.values()) + len(bid_rows) * len(val) * len(val_seeds) + len(ce_rows) * len(val) * len(val_seeds)

    # ---- Phase 3: comparisons with the archived learned allocators ------------------------------
    catalogs = [m.catalog["catalog_id"] for m in test]
    fresh = config["fresh_catalogs"]
    reps, boot = config["bootstrap_replicates"], config["bootstrap_seed"]

    def learned_seed_means(method, subset):
        return np.asarray([np.mean([np.mean([stage28_rows[(method, t, c, s)] for s in seeds]) for c in subset])
                           for t in freeze["training_seeds"]])

    def baseline_mean(method, subset):
        return float(np.mean([np.mean(results[method][c]) for c in subset]))

    online = {name: baseline_mean("online_dp_recomputed", subset) for name, subset in (("all", catalogs), ("fresh", fresh))}
    contrasts = {}
    for scope, subset in (("all", catalogs), ("fresh", fresh)):
        for learner in ("ppo", "dqn"):
            seed_means = learned_seed_means(learner, subset)
            for baseline in ["bid_price", "greedy_recomputed", *[f"ce_dp_k{k}" for k in ce_best]]:
                difference = seed_means - baseline_mean(baseline, subset)
                offset = len(contrasts)
                contrasts[f"{scope}:{learner}-{baseline}"] = {
                    "mean": float(difference.mean()), "ci95": bootstrap(difference, boot + offset, reps, 2.5),
                    "seeds_above": int((difference > 0).sum()), "learner_mean": float(seed_means.mean()),
                    "baseline_mean": baseline_mean(baseline, subset)}
    primary = contrasts["all:ppo-bid_price"]
    decision = ("learned_superior" if primary["ci95"][0] > 0 else "bid_price_superior" if primary["ci95"][1] < 0 else "inconclusive")
    ppo_mean = contrasts["all:ppo-bid_price"]["learner_mean"]
    breakeven = next((k for k in ce_best if baseline_mean(f"ce_dp_k{k}", catalogs) >= ppo_mean), None)
    method_means = {m: {"all": baseline_mean(m, catalogs), "fresh": baseline_mean(m, fresh),
                        "fraction_online_all": baseline_mean(m, catalogs) / online["all"],
                        "fraction_online_fresh": baseline_mean(m, fresh) / online["fresh"],
                        "by_catalog": {c: float(np.mean(results[m][c])) for c in catalogs}} for m in results}
    for learner in ("ppo", "dqn"):
        method_means[learner] = {"all": float(learned_seed_means(learner, catalogs).mean()),
                                 "fresh": float(learned_seed_means(learner, fresh).mean()),
                                 "fraction_online_all": float(learned_seed_means(learner, catalogs).mean()) / online["all"],
                                 "fraction_online_fresh": float(learned_seed_means(learner, fresh).mean()) / online["fresh"],
                                 "by_catalog": {c: float(learned_seed_means(learner, [c]).mean()) for c in catalogs}}
    gates = {
        "inputs_hash_verified": True,
        "tuning_completed_before_test_load": not selection["test_data_loaded"] and (output / "tuning_selection.json").is_file(),
        "evaluator_reproduces_stage28_greedy_and_online_dp": reproduction_error <= 1e-9,
        "all_methods_all_catalogs_complete": all(len(results[m][c]) == len(seeds) for m in results for c in catalogs),
        "paired_learned_rows_complete": all((l, t, c, s) in stage28_rows for l in ("ppo", "dqn") for t in freeze["training_seeds"] for c in catalogs for s in seeds),
        "finite_values": bool(all(np.isfinite(v).all() for cats in results.values() for v in cats.values())),
    }
    summary = {"decision_ppo_vs_bid_price": decision, "primary": primary, "contrasts": contrasts,
               "method_means": method_means, "online_dp_mean": online,
               "ce_dp_breakeven_warmup_episodes_vs_ppo": breakeven, "selection": selection,
               "evaluator_reproduction_max_abs_error": reproduction_error,
               "validation_tuning": {"bid_price": bid_rows, "certainty_equivalent": ce_rows}}
    (output / "baseline_analysis.json").write_text(json.dumps(summary, indent=2) + "\n")
    return {"passed": all(gates.values()), "gates": gates,
            "resources": {"primitive_steps": counter["decisions"], "decisions": counter["decisions"], "updates": 0,
                          "circuit_evaluations": 0, "shots": 0},
            "summary": {k: v for k, v in summary.items() if k != "validation_tuning"}}
