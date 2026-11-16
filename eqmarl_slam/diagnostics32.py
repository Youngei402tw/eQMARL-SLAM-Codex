"""Stage 32: learner-strength check on BellRoute v1 (development and validation data only).

Question: does the bid-price advantage persist when the learners get a wider grid (past the old grid
edge), 2.4-3.2x longer training, a calibrated PPO value baseline, and more seeds?

Phase 1: PPO and DQN, 6 configs x 2 seeds, trained on the 8 development catalogs, with learning-curve
checkpoints; each scored on validation *selection* sequences. Phase 2: 4 further seeds of each method's
selected config. The bid-price rule is tuned on the same selection sequences. All final comparisons use
fresh validation *comparison* sequences that no selection step has seen. No test data is read.
"""
from __future__ import annotations

import hashlib
import json
import multiprocessing as mp
from pathlib import Path

import numpy as np
import torch

from .robust_routing import RobustRoutingMDP, greedy_action
from .routing_baselines import BidPricePolicy, PlannerPolicy, reference_slot_gains
from .routing_learners import evaluate_model, train_ppo, train_q

ROOT = Path(__file__).resolve().parents[1]


def _load(ref):
    path = ROOT / ref["path"]
    if hashlib.sha256(path.read_bytes()).hexdigest() != ref["sha256"]:
        raise ValueError(f"hash mismatch: {ref['path']}")
    return json.loads(path.read_text())


def _mean_over(model, method, mdps, seeds):
    by = {m.catalog["catalog_id"]: float(np.mean(evaluate_model(model, method, m, seeds))) for m in mdps}
    return by, float(np.mean(list(by.values())))


def _job(job):
    torch.set_num_threads(1)
    method, cfg, seed = job["method"], job["config"], job["seed"]
    curve = []
    def on_checkpoint(model, done):
        if done < cfg["episodes"]:
            curve.append({"episodes": done, "selection_mean": _mean_over(model, method, job["val"], job["selection_seeds"])[1]})
    common = dict(on_checkpoint=on_checkpoint, checkpoints=tuple(job["checkpoints"]))
    if method == "dqn":
        model, info = train_q(job["maps"], seed, cfg["episodes"], cfg["hidden"], cfg["learning_rate"], double=False, **common)
    else:
        model, info = train_ppo(job["maps"], seed, cfg["episodes"], cfg["hidden"], cfg["learning_rate"],
                                recurrent=False, calibrated_value=True, **common)
    sel_by, sel = _mean_over(model, method, job["val"], job["selection_seeds"])
    cmp_by, cmp = _mean_over(model, method, job["val"], job["comparison_seeds"])
    curve.append({"episodes": cfg["episodes"], "selection_mean": sel})
    out = job["out"]
    torch.save({"method": method, "seed": seed, "config": cfg, "state_dict": model.state_dict(), "training": info}, out)
    return {"method": method, "config": cfg["name"], "seed": seed, "phase": job["phase"], "training": info,
            "curve": curve, "selection_mean": sel, "comparison_mean": cmp, "comparison_by_catalog": cmp_by,
            "checkpoint": out.name, "checkpoint_sha256": hashlib.sha256(out.read_bytes()).hexdigest()}


def _run(jobs, workers):
    with mp.get_context("fork").Pool(workers) as pool:
        yield from pool.imap_unordered(_job, jobs)


def bootstrap(values, seed, replicates, percentile):
    values = np.asarray(values, float); rng = np.random.default_rng(seed)
    means = values[rng.integers(0, len(values), size=(replicates, len(values)))].mean(axis=1)
    return [float(np.percentile(means, percentile)), float(np.percentile(means, 100 - percentile))]


def run_diagnostics(config, output, emit, progress=None):
    output = Path(output); (output / "checkpoints").mkdir()
    freeze = _load(config["v1_freeze"])
    certs = {r["type_id"]: r for r in _load(freeze["non_test_certificate_bundle"])["certificates"]}
    dev = [RobustRoutingMDP(_load(r), certs) for r in freeze["training_catalogs"]["development"]]
    val = [RobustRoutingMDP(_load(r), certs) for r in freeze["training_catalogs"]["validation"]]
    if any(m.catalog["split"] == "test" for m in dev + val):
        raise ValueError("test catalog reached Stage 32")
    sel_seeds = list(range(config["selection_seed_start"], config["selection_seed_start"] + config["selection_sequences"]))
    cmp_seeds = list(range(config["comparison_seed_start"], config["comparison_seed_start"] + config["comparison_sequences"]))
    if set(sel_seeds) & set(cmp_seeds):
        raise ValueError("selection and comparison sequences overlap")
    decisions = 0

    # ---- non-learned rules and references ----
    ref = reference_slot_gains(dev)
    def rule_mean(alpha, weight, seeds):
        nonlocal decisions
        by = {}
        for m in val:
            vals = [m.evaluate(m.sample_sequence(s), BidPricePolicy(ref, alpha, weight), seed=s + 810000)["surplus"] for s in seeds]
            by[m.catalog["catalog_id"]] = float(np.mean(vals)); decisions += len(seeds) * m.catalog["horizon"]
        return by, float(np.mean(list(by.values())))
    rule_rows = []
    for alpha in config["bid_price"]["alpha"]:
        for weight in config["bid_price"]["prior_weight"]:
            by, mean = rule_mean(alpha, weight, sel_seeds)
            rule_rows.append({"alpha": alpha, "prior_weight": weight, "selection_mean": mean, "by_catalog": by})
            emit({"kind": "rule_tuning", **rule_rows[-1]})
    rule = max(rule_rows, key=lambda r: (r["selection_mean"], -r["alpha"], -r["prior_weight"]))
    rule_cmp_by, rule_cmp = rule_mean(rule["alpha"], rule["prior_weight"], cmp_seeds)
    refs = {}
    for name in ("greedy", "online_dp"):
        by = {}
        for m in val:
            policy = greedy_action if name == "greedy" else PlannerPolicy(m, np.asarray(m.catalog["transition"], float))
            by[m.catalog["catalog_id"]] = float(np.mean([m.evaluate(m.sample_sequence(s), policy, seed=s + 810000)["surplus"] for s in cmp_seeds]))
            decisions += len(cmp_seeds) * m.catalog["horizon"]
        refs[name] = {"by_catalog": by, "mean": float(np.mean(list(by.values())))}
    emit({"kind": "references", "rule": {k: rule[k] for k in ("alpha", "prior_weight", "selection_mean")},
          "rule_comparison_mean": rule_cmp, "greedy": refs["greedy"]["mean"], "online_dp": refs["online_dp"]["mean"]})

    # ---- learners ----
    def make_job(method, cand, seed, phase):
        cfg = dict(cand, episodes=config["episodes"][method])
        return {"method": method, "config": cfg, "seed": seed + config["method_seed_offset"][method], "phase": phase,
                "maps": dev, "val": val, "selection_seeds": sel_seeds, "comparison_seeds": cmp_seeds,
                "checkpoints": [c for c in config["curve_checkpoints"] if c < cfg["episodes"]] + [cfg["episodes"]],
                "out": output / "checkpoints" / f"{method}-{cand['name']}-seed-{seed}.pt"}
    trials, train_decisions, updates = [], 0, 0
    def collect(jobs):
        nonlocal train_decisions, updates, decisions
        for result in _run(jobs, config["workers"]):
            trials.append(result); train_decisions += result["training"]["decisions"]; updates += result["training"]["updates"]
            decisions += (len(sel_seeds) * len(result["curve"]) + len(cmp_seeds)) * sum(m.catalog["horizon"] for m in val)
            emit({"kind": "learner_trial", **result})
    collect([make_job(m, c, s, "grid") for m in config["methods"] for c in config["learner_configs"] for s in config["grid_seeds"]])
    selected = {}
    for method in config["methods"]:
        scores = {c["name"]: float(np.mean([t["selection_mean"] for t in trials if t["method"] == method and t["config"] == c["name"]]))
                  for c in config["learner_configs"]}
        selected[method] = max(scores, key=scores.get)
        emit({"kind": "selection", "method": method, "selected": selected[method], "config_selection_means": scores})
    collect([make_job(m, next(c for c in config["learner_configs"] if c["name"] == selected[m]), s, "extra")
             for m in config["methods"] for s in config["extra_seeds"]])

    # ---- analysis ----
    reps, boot = config["bootstrap_replicates"], config["bootstrap_seed"]
    per_method = {}
    for i, method in enumerate(config["methods"]):
        rows = sorted((t for t in trials if t["method"] == method and t["config"] == selected[method]), key=lambda t: t["seed"])
        cmp_vals = np.asarray([t["comparison_mean"] for t in rows])
        diff = cmp_vals - rule_cmp
        per_method[method] = {"config": selected[method], "seeds": len(rows), "selection_mean": float(np.mean([t["selection_mean"] for t in rows])),
                              "comparison_mean": float(cmp_vals.mean()), "minus_rule": float(diff.mean()),
                              "minus_rule_ci95": bootstrap(diff, boot + i, reps, 2.5), "seeds_above_rule": int((diff > 0).sum()),
                              "fraction_online_dp": float(cmp_vals.mean() / refs["online_dp"]["mean"]),
                              "curve": {str(e): float(np.mean([next(p["selection_mean"] for p in t["curve"] if p["episodes"] == e) for t in rows]))
                                        for e in sorted({p["episodes"] for t in rows for p in t["curve"]})}}
    best = max(per_method, key=lambda m: per_method[m]["selection_mean"])
    ci = per_method[best]["minus_rule_ci95"]
    decision = "gap_persists" if ci[1] < 0 else "learner_superior" if ci[0] > 0 else "inconclusive"
    old_edge = config["previous_best"]
    summary = {"decision": decision, "best_method": best, "per_method": per_method,
               "rule": {"alpha": rule["alpha"], "prior_weight": rule["prior_weight"], "comparison_mean": rule_cmp,
                        "fraction_online_dp": rule_cmp / refs["online_dp"]["mean"]},
               "greedy_fraction_online_dp": refs["greedy"]["mean"] / refs["online_dp"]["mean"],
               "online_dp_mean": refs["online_dp"]["mean"], "previous_best_validation": old_edge,
               "selected_at_grid_edge": {m: selected[m].endswith(config["grid_edge_suffix"]) for m in config["methods"]}}
    (output / "strength_analysis.json").write_text(json.dumps({**summary, "trials": trials, "rule_tuning": rule_rows,
                                                               "references": refs, "rule_comparison_by_catalog": rule_cmp_by}, indent=2) + "\n")
    gates = {"all_trials_complete": len(trials) == len(config["methods"]) * (len(config["learner_configs"]) * len(config["grid_seeds"]) + len(config["extra_seeds"])),
             "no_test_data": True, "selection_comparison_disjoint": True,
             "finite": bool(all(np.isfinite(t["comparison_mean"]) for t in trials))}
    return {"passed": all(gates.values()), "gates": gates,
            "resources": {"primitive_steps": decisions + train_decisions, "decisions": decisions + train_decisions,
                          "updates": updates, "circuit_evaluations": 0, "shots": 0}, "summary": summary}
