"""Stage 33: robustness of the Stage-30 conclusions to the transition matrix (BellRoute v1, non-test data).

Question: the Stage-30 catalogs all come from one generator with fast-mixing dynamics (self-loop
probability 0.31-0.37). Do the two Stage-30 findings hold when the transition matrix changes?
  (H1) a budget-pacing (bid-price) rule captures most of the optimum;
  (H2) dynamics barely matter: CE-DP with the generic default guess (no logged data) matches online DP.
Each validation catalog is rebuilt under registered transition-matrix variants, keeping its state table.
Rules are tuned per variant on tuning sequences and scored on disjoint evaluation sequences. No learner
is trained and no test data are read. An unregistered exploratory version of this sweep was run on the
development catalogs before registration; this run uses the validation catalogs only for evaluation.
"""
from __future__ import annotations

import copy
import hashlib
import json
import multiprocessing as mp
from pathlib import Path

import numpy as np

from .robust_routing import RobustRoutingMDP, greedy_action
from .routing_baselines import BidPricePolicy, PlannerPolicy, estimate_transition, prior_transition, reference_slot_gains

ROOT = Path(__file__).resolve().parents[1]


def _load(ref):
    path = ROOT / ref["path"]
    if hashlib.sha256(path.read_bytes()).hexdigest() != ref["sha256"]:
        raise ValueError(f"hash mismatch: {ref['path']}")
    return json.loads(path.read_text())


def _offdiag(P):
    Q = P.copy()
    np.fill_diagonal(Q, 0.0)
    return Q / Q.sum(axis=1, keepdims=True)


def _catalog_rng(catalog_id, salt):
    return np.random.default_rng(int(hashlib.sha256(f"{catalog_id}|{salt}".encode()).hexdigest()[:16], 16))


def transform(catalog, certs, spec):
    """Return a copy of ``catalog`` whose transition matrix follows the registered variant ``spec``."""
    P = np.asarray(catalog["transition"], dtype=float)
    n = len(P)
    kind = spec["kind"]
    if kind == "as_generated":
        Q = P
    elif kind == "stay":
        Q = spec["stay"] * np.eye(n) + (1.0 - spec["stay"]) * _offdiag(P)
    elif kind == "successor":
        rng = _catalog_rng(catalog["catalog_id"], "successor")
        perm = rng.permutation(n)
        while np.any(perm == np.arange(n)):
            perm = rng.permutation(n)
        S = np.zeros((n, n))
        S[np.arange(n), perm] = 1.0
        Q = spec["strength"] * S + (1.0 - spec["strength"]) * P
    elif kind == "rare_jackpot":
        base = RobustRoutingMDP(catalog, certs)
        budget = int(catalog["bell_pair_budget"])
        j = int(np.argmax(base.oracle_table()[1, :, budget]))
        Q = P.copy()
        Q[:, j] *= spec["entry_scale"]
        Q[j, j] = 0.0
        Q /= Q.sum(axis=1, keepdims=True)
        Q[j] *= (1.0 - spec["jackpot_stay"])
        Q[j, j] = spec["jackpot_stay"]
    else:
        raise ValueError(kind)
    if not np.allclose(Q.sum(axis=1), 1.0) or np.any(Q < -1e-12):
        raise ValueError(f"invalid transition matrix for {spec['name']}")
    out = copy.deepcopy(catalog)
    out["transition"] = np.clip(Q, 0.0, None).tolist()
    return out


def _surplus(mdp, factory, seeds):
    return [float(mdp.evaluate(mdp.sample_sequence(s), factory(), seed=s + 810000)["surplus"]) for s in seeds]


def _logged(mdp, seeds):
    return [mdp.sample_sequence(s) for s in seeds]


def variant_job(job):
    config, spec = job["config"], job["spec"]
    certs, val = job["certs"], [transform(c, job["certs"], spec) for c in job["val"]]
    mdps = [RobustRoutingMDP(c, certs) for c in val]
    tune = list(range(config["tuning_seed_start"], config["tuning_seed_start"] + config["tuning_sequences"]))
    evals = list(range(config["evaluation_seed_start"], config["evaluation_seed_start"] + config["evaluation_sequences"]))
    ref = reference_slot_gains(mdps)
    decisions = 0
    grid = [(a, w) for a in config["bid_price"]["alpha"] for w in config["bid_price"]["prior_weight"]]
    tuning = []
    for a, w in grid:
        mean = float(np.mean([np.mean(_surplus(m, lambda: BidPricePolicy(ref, a, w), tune)) for m in mdps]))
        tuning.append({"alpha": a, "prior_weight": w, "tuning_mean": mean})
        decisions += len(tune) * sum(int(m.catalog["horizon"]) for m in mdps)
    best = max(tuning, key=lambda r: (r["tuning_mean"], -r["alpha"], -r["prior_weight"]))
    frozen = config["frozen_bid_price"]
    rho = config["default_rho"]
    per_catalog = {}
    for m in mdps:
        n = len(m.catalog["states"])
        P = np.asarray(m.catalog["transition"], dtype=float)
        policies = {
            "online_dp": (lambda pol=PlannerPolicy(m, P): pol),
            "greedy": (lambda: greedy_action),
            "bid_price_frozen": (lambda: BidPricePolicy(ref, frozen["alpha"], frozen["prior_weight"])),
            "bid_price_retuned": (lambda: BidPricePolicy(ref, best["alpha"], best["prior_weight"])),
            "ce_dp_default": (lambda pol=PlannerPolicy(m, prior_transition(n, rho)): pol),
            "ce_dp_matched_stay": (lambda pol=PlannerPolicy(m, prior_transition(n, float(np.mean(np.diag(P))))): pol),
        }
        for k, kappa in config["ce_dp_logged"].items():
            k = int(k)
            logs = _logged(m, range(config["logging_seed_start"], config["logging_seed_start"] + k))
            est = estimate_transition(n, logs, rho, kappa)
            policies[f"ce_dp_k{k}"] = (lambda pol=PlannerPolicy(m, est): pol)
        values = {name: _surplus(m, factory, evals) for name, factory in policies.items()}
        decisions += len(policies) * len(evals) * int(m.catalog["horizon"])
        per_catalog[m.catalog["catalog_id"]] = {"mean_surplus": {k: float(np.mean(v)) for k, v in values.items()},
                                                "episodes": values, "self_loop_mean": float(np.mean(np.diag(P)))}
    return {"variant": spec["name"], "spec": spec, "bid_price_tuning": tuning, "bid_price_selected": best,
            "per_catalog": per_catalog, "decisions": decisions}


def summarize(result, methods, bootstrap_seed, replicates):
    """Overall fraction of online DP (ratio of catalog-averaged means) with a sequence-level percentile bootstrap."""
    cats = list(result["per_catalog"].values())
    rng = np.random.default_rng(bootstrap_seed)
    n = len(cats[0]["episodes"]["online_dp"])
    idx = rng.integers(0, n, size=(replicates, n))
    out = {}
    opt_boot = np.mean([np.asarray(c["episodes"]["online_dp"])[idx].mean(axis=1) for c in cats], axis=0)
    opt = float(np.mean([c["mean_surplus"]["online_dp"] for c in cats]))
    for method in methods:
        mean = float(np.mean([c["mean_surplus"][method] for c in cats]))
        boot = np.mean([np.asarray(c["episodes"][method])[idx].mean(axis=1) for c in cats], axis=0) / opt_boot
        worst = min(c["mean_surplus"][method] / c["mean_surplus"]["online_dp"] for c in cats)
        out[method] = {"fraction_online_dp": mean / opt, "ci95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
                       "worst_catalog_fraction": float(worst)}
    return out


def run_diagnostics(config, output, emit, progress=None):
    output = Path(output)
    freeze = _load(config["v1_freeze"])
    certs = {r["type_id"]: r for r in _load(freeze["non_test_certificate_bundle"])["certificates"]}
    val = [_load(r) for r in freeze["training_catalogs"]["validation"]]
    if any(c["split"] != "validation" for c in val):
        raise ValueError("Stage 33 evaluates validation catalogs only")
    jobs = [{"config": config, "spec": spec, "certs": certs, "val": val} for spec in config["variants"]]
    with mp.get_context("fork").Pool(config["workers"]) as pool:
        results = pool.map(variant_job, jobs)
    methods = ["greedy", "bid_price_frozen", "bid_price_retuned", "ce_dp_default", "ce_dp_matched_stay"] + \
              [f"ce_dp_k{int(k)}" for k in config["ce_dp_logged"]]
    table, decisions = [], 0
    for i, r in enumerate(results):
        decisions += r["decisions"]
        s = summarize(r, methods, config["bootstrap_seed"] + i, config["bootstrap_replicates"])
        row = {"variant": r["variant"], "self_loop_mean": float(np.mean([c["self_loop_mean"] for c in r["per_catalog"].values()])),
               "bid_price_selected": {k: r["bid_price_selected"][k] for k in ("alpha", "prior_weight")}, "methods": s}
        table.append(row)
        emit({"kind": "variant_summary", **row})
    rule = config["decision_rule"]
    h1 = all(row["methods"]["bid_price_retuned"]["fraction_online_dp"] >= rule["h1_threshold"] for row in table)
    h2_fail = [row["variant"] for row in table if row["methods"]["ce_dp_default"]["fraction_online_dp"] < rule["h2_threshold"]]
    logged_key = f"ce_dp_k{int(rule['h3_logged_episodes'])}"
    h3_rows = [row for row in table if row["variant"] in h2_fail]
    h3 = all(row["methods"][logged_key]["fraction_online_dp"] >= rule["h2_threshold"] for row in h3_rows) if h3_rows else None
    summary = {"h1_pacing_generalizes": h1, "h2_dynamics_irrelevant_everywhere": not h2_fail,
               "h2_failing_variants": h2_fail, "h3_logged_data_recovers": h3, "table": table}
    episodes = output / "episode_means.json"
    episodes.write_text(json.dumps([{"variant": r["variant"], "per_catalog": {k: {"mean_surplus": v["mean_surplus"], "self_loop_mean": v["self_loop_mean"]}
                                                                         for k, v in r["per_catalog"].items()},
                                     "bid_price_tuning": r["bid_price_tuning"]} for r in results], indent=1) + "\n")
    (output / "robustness_analysis.json").write_text(json.dumps(summary, indent=2) + "\n")
    gates = {"all_variants_complete": len(table) == len(config["variants"]), "no_test_data": True,
             "tuning_evaluation_disjoint": config["tuning_seed_start"] + config["tuning_sequences"] <= config["evaluation_seed_start"],
             "finite": bool(all(np.isfinite(m["fraction_online_dp"]) for row in table for m in row["methods"].values()))}
    return {"passed": all(gates.values()), "gates": gates,
            "resources": {"primitive_steps": decisions, "decisions": decisions, "updates": 0, "circuit_evaluations": 0, "shots": 0},
            "summary": {k: v for k, v in summary.items() if k != "table"}}
