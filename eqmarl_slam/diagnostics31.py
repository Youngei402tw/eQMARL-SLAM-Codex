"""Stage 31a: registered headroom diagnostic for harder BellRoute variants (no training, no test data).

For each variant, equal-information rules are tuned with one shared parameter set on the development
catalogs, then scored on the held-out validation catalogs against the variant's optimal (A) or
near-optimal belief-state (B) reference. Gate: a variant qualifies for learner development only if the
best development-selected rule achieves at most ``gate_fraction`` of the reference on validation.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from .robust_routing import RobustRoutingMDP, greedy_action
from .routing_baselines import BidPricePolicy, reference_slot_gains
from .variants31 import RegimeMDP, StoreBidPrice, StoreMDP, store_greedy, store_reference_gains

ROOT = Path(__file__).resolve().parents[1]


def _load(ref):
    path = ROOT / ref["path"]
    if hashlib.sha256(path.read_bytes()).hexdigest() != ref["sha256"]:
        raise ValueError(f"hash mismatch: {ref['path']}")
    return json.loads(path.read_text())


def run_diagnostics(config, output, emit, progress=None):
    output = Path(output)
    freeze = _load(config["freeze"])
    certs = {r["type_id"]: r for r in _load(freeze["non_test_certificate_bundle"])["certificates"]}
    dev = [_load(r) for r in freeze["training_catalogs"]["development"]]
    val = [_load(r) for r in freeze["training_catalogs"]["validation"]]
    for catalog in dev + val:
        if catalog["split"] == "test":
            raise ValueError("test catalog reached the Stage 31a pilot")
    tune_seeds = list(range(config["tuning_seed_start"], config["tuning_seed_start"] + config["tuning_sequences"]))
    eval_seeds = list(range(config["evaluation_seed_start"], config["evaluation_seed_start"] + config["evaluation_sequences"]))
    vmin = min((c["classical_value"] - 0.5) / (c["bell_value"] - 0.5) for c in certs.values()
               if c["bell_value"] - c["classical_value"] > 1e-12)
    decisions = 0
    results = {}

    # ---------------- Variant A: generate-and-store ----------------
    a = config["variant_a"]
    make_a = lambda cats: [StoreMDP(c, certs, a["capacity"], a["gen_prob"], a["decay"], vmin) for c in cats]
    dev_a, val_a = make_a(dev), make_a(val)
    ref_a = store_reference_gains(dev_a)
    for m in dev_a + val_a:
        m.solve()

    def score_a(mdps, factory, seeds):
        nonlocal decisions
        out = []
        for m in mdps:
            out.append(float(np.mean([m.run(s, factory(m)) for s in seeds]))); decisions += len(seeds) * m.horizon
        return out

    grid_a = []
    for alpha in a["bid_price"]["alpha"]:
        for weight in a["bid_price"]["prior_weight"]:
            opt = score_a(dev_a, lambda m: m.optimal_policy(), tune_seeds) if not grid_a else grid_a[0]["optimal"]
            vals = score_a(dev_a, lambda m: StoreBidPrice(m, ref_a, alpha, weight), tune_seeds)
            row = {"alpha": alpha, "prior_weight": weight, "optimal": opt, "means": vals,
                   "fraction": float(np.mean(vals) / np.mean(opt))}
            grid_a.append(row); emit({"kind": "variant_a_tuning", **{k: v for k, v in row.items() if k != "optimal"}})
    sel_a = max(grid_a, key=lambda r: r["fraction"])
    opt_a = score_a(val_a, lambda m: m.optimal_policy(), eval_seeds)
    greedy_a = score_a(val_a, store_greedy, eval_seeds)
    bp_a = score_a(val_a, lambda m: StoreBidPrice(m, ref_a, sel_a["alpha"], sel_a["prior_weight"]), eval_seeds)
    results["A_generate_and_store"] = {
        "settings": a, "selected": {"alpha": sel_a["alpha"], "prior_weight": sel_a["prior_weight"], "dev_fraction": sel_a["fraction"]},
        "validation": {c["catalog_id"]: {"optimal": o, "greedy": g, "bid_price": b} for c, o, g, b in zip(val, opt_a, greedy_a, bp_a)},
        "greedy_fraction": float(np.mean(greedy_a) / np.mean(opt_a)),
        "best_rule_fraction": float(np.mean(bp_a) / np.mean(opt_a)),
        "min_catalog_rule_fraction": float(min(b / o for b, o in zip(bp_a, opt_a)))}
    emit({"kind": "variant_result", "variant": "A", **{k: v for k, v in results["A_generate_and_store"].items() if k != "validation"}})

    # ---------------- Variant B: hidden regime ----------------
    b = config["variant_b"]
    dev_b = [RegimeMDP(c, certs, b["beta"]) for c in dev]
    val_b = [RegimeMDP(c, certs, b["beta"]) for c in val]
    for m in dev_b + val_b:
        m.solve_belief()
    ref_b = reference_slot_gains([RobustRoutingMDP(c, certs) for c in dev])

    def score_b(mdps, factory, seeds):
        nonlocal decisions
        out = []
        for m in mdps:
            vals = []
            for s in seeds:
                vals.append(m.run(s, factory(m))); decisions += m.horizon
            out.append(float(np.mean(vals)))
        return out

    def belief(m):
        act = m.belief_policy()
        return lambda st, bud, rem, pi, z: act(st, bud, rem, pi)

    def bid(alpha, weight):
        def factory(m):
            policy = BidPricePolicy(ref_b, alpha, weight)
            return lambda st, bud, rem, pi, z: policy(m.base, st, bud, rem)
        return factory

    def oracle(m):
        tables = m.oracle_tables()
        def act(st, bud, rem, pi, z):
            future = m.P[z][st] @ tables[z][1][rem - 1]
            return max(m.base.actions(st, bud), key=lambda x: m.base.action_gain(st, x) + future[bud - len(x)])
        return act

    opt_dev_b = score_b(dev_b, belief, tune_seeds)
    grid_b = []
    for alpha in b["bid_price"]["alpha"]:
        for weight in b["bid_price"]["prior_weight"]:
            vals = score_b(dev_b, bid(alpha, weight), tune_seeds)
            row = {"alpha": alpha, "prior_weight": weight, "means": vals, "fraction": float(np.mean(vals) / np.mean(opt_dev_b))}
            grid_b.append(row); emit({"kind": "variant_b_tuning", **row})
    sel_b = max(grid_b, key=lambda r: r["fraction"])
    opt_b = score_b(val_b, belief, eval_seeds)
    greedy_b = score_b(val_b, lambda m: (lambda st, bud, rem, pi, z: greedy_action(m.base, st, bud, rem)), eval_seeds)
    bp_b = score_b(val_b, bid(sel_b["alpha"], sel_b["prior_weight"]), eval_seeds)
    oracle_b = score_b(val_b, oracle, eval_seeds)
    results["B_hidden_regime"] = {
        "settings": b, "selected": {"alpha": sel_b["alpha"], "prior_weight": sel_b["prior_weight"], "dev_fraction": sel_b["fraction"]},
        "validation": {c["catalog_id"]: {"belief_dp": o, "greedy": g, "bid_price": p, "regime_oracle": r}
                       for c, o, g, p, r in zip(val, opt_b, greedy_b, bp_b, oracle_b)},
        "greedy_fraction": float(np.mean(greedy_b) / np.mean(opt_b)),
        "best_rule_fraction": float(np.mean(bp_b) / np.mean(opt_b)),
        "min_catalog_rule_fraction": float(min(p / o for p, o in zip(bp_b, opt_b))),
        "regime_oracle_over_belief": float(np.mean(oracle_b) / np.mean(opt_b))}
    emit({"kind": "variant_result", "variant": "B", **{k: v for k, v in results["B_hidden_regime"].items() if k != "validation"}})

    gate = config["gate_fraction"]
    qualifies = {name: r["best_rule_fraction"] <= gate for name, r in results.items()}
    summary = {"gate_fraction": gate, "qualifies": qualifies,
               "decision": "proceed_with:" + ",".join(k for k, v in qualifies.items() if v) if any(qualifies.values()) else "stop_null_no_headroom",
               "variants": results, "tuning": {"A": grid_a, "B": grid_b}}
    (output / "headroom_analysis.json").write_text(json.dumps(summary, indent=2) + "\n")
    gates = {"no_test_data": True, "references_dominate_rules": all(
                 r["best_rule_fraction"] <= 1.0 + 1e-9 for r in results.values()),
             "finite": bool(all(np.isfinite(r["best_rule_fraction"]) for r in results.values()))}
    return {"passed": all(gates.values()), "gates": gates,
            "resources": {"primitive_steps": decisions, "decisions": decisions, "updates": 0, "circuit_evaluations": 0, "shots": 0},
            "summary": {k: v for k, v in summary.items() if k != "tuning"}}
