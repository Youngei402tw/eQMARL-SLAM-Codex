#!/usr/bin/env python3
"""Emit LaTeX macros for every number quoted in the manuscript, from the Stage 29 audit outputs.

Usage: python docs/paper/make_numbers.py [--audit DIR]
Writes docs/paper/generated/numbers.tex. Prose and tables use only these macros for results,
so the manuscript cannot drift from the audited raw-row analysis.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DEFAULT = ROOT / "experiments/29/29-qip-release-audit_attempt-01"
CAT = {"qip-test-id-a4-complete": "Cmp", "qip-test-topology-a6-ring": "Ring", "qip-test-scale-a8-sparse": "Sparse",
       "qip-test-horizon-a4-path": "Path"}
METHOD = {"ppo": "PPO", "dqn": "DQN", "random": "Random", "static": "Static", "greedy": "Greedy",
          "online_reference": "Online", "future_clairvoyant": "Future", "message_oracle": "Message", "classical_exact": "Classical"}


def f(x, digits=4, sign=False):
    return f"{x:+.{digits}f}".replace("+", "+") if sign else f"{x:.{digits}f}"


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--audit", type=Path, default=DEFAULT)
    args = parser.parse_args(); a = args.audit
    r = json.loads((a / "reproduced_analysis.json").read_text())
    c = json.loads((a / "certificate_audit.json").read_text())
    res = json.loads((a / "resource_audit.json").read_text())
    e = json.loads((a / "earlier_stage_audit.json").read_text())
    out = {}
    m = r["method_means"]
    for key, name in METHOD.items():
        out[f"mean{name}"] = f(m[key], 4)
    for cat, short in CAT.items():
        for key, name in METHOD.items():
            if key in ("ppo", "dqn"):
                val = sum(r["learned_seed_catalog_means"][key][s][cat] for s in r["learned_seed_catalog_means"][key]) / len(r["learned_seed_catalog_means"][key])
            else:
                val = r["baseline_by_catalog"][key][cat]
            out[f"cat{short}{name}"] = f(val, 3)
        out[f"frac{short}PPO"] = f(100 * r["fraction_of_online_reference"]["ppo"][cat], 1)
        out[f"frac{short}DQN"] = f(100 * r["fraction_of_online_reference"]["dqn"][cat], 1)
        out[f"frac{short}Greedy"] = f(100 * r["fraction_of_online_reference"]["greedy"][cat], 1)
    for t in r["catalog_tests"]:
        s = CAT[t["catalog_id"]]
        out[f"diff{s}"] = f(t["mean_difference"], 3, True); out[f"wins{s}"] = str(t["ppo_wins"])
        out[f"rawp{s}"] = f(t["raw_p"], 3); out[f"holmp{s}"] = f(t["holm_p"], 3)
    out["ppoMinusDqn"] = f(r["ppo_minus_dqn"], 4, True)
    out["ppoMinusDqnLoNinetyFive"], out["ppoMinusDqnHiNinetyFive"] = (f(x, 4, True) for x in r["ppo_minus_dqn_95"])
    out["ppoMinusDqnLoNinety"], out["ppoMinusDqnHiNinety"] = (f(x, 4, True) for x in r["ppo_minus_dqn_90"])
    out["ppoMinusGreedy"] = f(r["ppo_minus_greedy"], 3, True)
    out["ppoMinusGreedyLo"], out["ppoMinusGreedyHi"] = (f(x, 3, True) for x in r["ppo_minus_greedy_95"])
    out["dqnMinusGreedy"] = f(m["dqn"] - m["greedy"], 3, True)
    out["eqMargin"] = f(r["equivalence_margin"], 4)
    out["onlineMean"] = f(r["online_reference_mean"], 4)
    out["ppoSeedsAboveDqn"] = str(r["ppo_seeds_above_dqn"]); out["ppoSeedsAboveGreedy"] = str(r["ppo_seeds_above_greedy"])
    out["dqnSeedsAboveGreedy"] = str(r["dqn_seeds_above_greedy"])
    out["fracPPO"] = f(100 * r["learned_fraction_of_online_reference_overall"]["ppo"], 1)
    out["fracDQN"] = f(100 * r["learned_fraction_of_online_reference_overall"]["dqn"], 1)
    out["fracGreedy"] = f(100 * m["greedy"] / m["online_reference"], 1)
    out["fracFuture"] = f(100 * m["future_clairvoyant"] / m["online_reference"], 1)
    ppo_seed = r["learned_seed_means"]["ppo"]
    out["ppoDistinctSeedValues"] = str(len({round(x, 12) for x in ppo_seed}))
    sec = r["secondary_condition_means"]
    out["noiseIdealHundred"] = f(r["ppo_ideal_first_100_sequences"], 4)
    out["noiseHerSevenFive"] = f(sec["heralded-0.75"], 4); out["noiseHerFive"] = f(sec["heralded-0.5"], 4)
    out["noiseVisEight"] = f(sec["visibility-0.8"], 4); out["noiseVisSix"] = f(sec["visibility-0.6"], 4)
    out["ratioHerSevenFive"] = f(sec["heralded-0.75"] / r["ppo_ideal_first_100_sequences"], 3)
    out["ratioHerFive"] = f(sec["heralded-0.5"] / r["ppo_ideal_first_100_sequences"], 3)
    th = c["threshold_summary"]
    out["thrTestMin"], out["thrTestMedian"], out["thrTestMean"] = f(th["min"], 4), f(th["median"], 4), f(th["mean"], 4)
    out["testPositiveTypes"] = str(c["positive_gain_types"]); out["testTypes"] = str(c["types"]); out["testOpportunities"] = str(c["test_opportunities"])
    pf = c["positive_opportunity_fraction_by_visibility"]
    out["posOppIdeal"] = f(100 * pf["1.0"], 1); out["posOppNine"] = f(100 * pf["0.9"], 1); out["posOppEight"] = f(100 * pf["0.8"], 1)
    out["posOppNineFive"] = f(100 * pf["0.95"], 1)
    nt = e["stage27"]["non_test_visibility_thresholds"]
    out["thrNonTestMin"], out["thrNonTestMedian"] = f(nt["min"], 4), f(nt["median"], 4)
    s27 = e["stage27"]["means"]
    for meth, name in (("dqn", "DQN"), ("double_dqn", "DDQN"), ("ppo", "PPO"), ("recurrent_ppo", "RPPO")):
        for cond, cname in (("ideal", "Ideal"), ("heralded-0.9", "HerNine"), ("heralded-0.75", "HerSevenFive"), ("heralded-0.5", "HerFive"),
                            ("visibility-0.9", "VisNine"), ("visibility-0.8", "VisEight"), ("visibility-0.7", "VisSeven"), ("visibility-0.6", "VisSix")):
            out[f"valid{name}{cname}"] = f(s27[meth][cond], 4)
    out["validPosOppNine"] = f(100 * e["stage27"]["positive_gain_fractions"]["visibility-0.9"], 2)
    out["validPosOppIdeal"] = f(100 * e["stage27"]["positive_gain_fractions"]["ideal"], 1)
    s26 = e["stage26"]
    for meth, name in (("dqn", "DQN"), ("double_dqn", "DDQN"), ("ppo", "PPO"), ("recurrent_ppo", "RPPO")):
        out[f"devel{name}"] = f(s26["methods"][meth]["score"], 4)
    out["develDDQNMinusPPO"] = f(s26["double_dqn_minus_ppo"], 4, True)
    for k, name in (("no_budget", "NoBudget"), ("no_horizon", "NoHorizon"), ("binary_gain", "Binary"), ("shuffled_gain", "Shuffled")):
        out[f"abl{name}"] = f(s26["ablations"][k], 4)
        out[f"ablDrop{name}"] = f(100 * (1 - s26["ablations"][k] / s26["methods"]["double_dqn"]["score"]), 0)
    s23 = e["stage23"]
    out["sOneBer"], out["sOneGreedy"], out["sOneRandom"], out["sOneStatic"] = (f(s23[k], 4) for k in ("ber_marl_mean", "greedy", "random", "static"))
    out["sOneDP"], out["sOneFuture"] = f(s23["online_dp"], 4), f(s23["future_clairvoyant"], 4)
    out["sOneFrac"] = f(100 * s23["fraction_of_online_dp_mean"], 2); out["sOneFracLo"] = f(100 * s23["fraction_one_sided_95_lower"], 2)
    out["sOneDiff"] = f(s23["ber_marl_mean"] - s23["greedy"], 4, True)
    out["sOneDiffLo"], out["sOneDiffHi"] = (f(x, 4, True) for x in s23["difference_vs_greedy_95"])
    out["sOneSeedsAbove"] = str(s23["seeds_above_greedy"])
    out["resDecisions"] = f"{res['total_allocator_decisions']:,}".replace(",", "{,}")
    out["resTrainDecisions"] = f"{res['allocator_training_decisions']:,}".replace(",", "{,}")
    out["resEvalPrimary"] = f"{res['learned_evaluation_decisions_primary']:,}".replace(",", "{,}")
    out["resEvalSecondary"] = f"{res['learned_evaluation_decisions_secondary']:,}".replace(",", "{,}")
    out["resBaseline"] = f"{res['baseline_policy_decisions']:,}".replace(",", "{,}")
    out["resAllocUpdates"] = f"{res['allocator_optimizer_updates']:,}".replace(",", "{,}")
    out["resCertUpdates"] = f"{res['certification_optimizer_updates']:,}".replace(",", "{,}")
    out["resCircuits"] = f"{res['certification_circuit_evaluations']:,}".replace(",", "{,}")
    out["resWall"] = f"{res['wall_seconds']:,.0f}".replace(",", "{,}"); out["resWallHours"] = f(res["wall_seconds"] / 3600, 2)
    out["resCPU"] = f(res["cpu_core_hours"], 2)
    out["resDqnTrain"] = f(res["training_wall_seconds_by_method"]["dqn"] / 12, 0); out["resPpoTrain"] = f(res["training_wall_seconds_by_method"]["ppo"] / 12, 0)
    out["paramsDQN"] = f"{res['parameters_by_method']['dqn']:,}".replace(",", "{,}"); out["paramsPPO"] = f"{res['parameters_by_method']['ppo']:,}".replace(",", "{,}")
    tot = res["qip_program_totals"]
    out["progDecisions"] = f"{int(tot['decisions']):,}".replace(",", "{,}"); out["progWallHours"] = f(tot["wall_seconds"] / 3600, 2)
    out["progCPU"] = f(tot["cpu_core_hours"], 2); out["progUpdates"] = f"{int(tot['updates']):,}".replace(",", "{,}")
    ph = r["posthoc_fresh_dynamics_sensitivity"]
    out["phPPOMinusGreedy"] = f(ph["ppo_minus_greedy"], 3, True)
    out["phPPOMinusGreedyLo"], out["phPPOMinusGreedyHi"] = (f(x, 3, True) for x in ph["ppo_minus_greedy_95"])
    out["phPPOMinusDqn"] = f(ph["ppo_minus_dqn"], 4, True)
    out["phPPOMinusDqnLoNinety"], out["phPPOMinusDqnHiNinety"] = (f(x, 4, True) for x in ph["ppo_minus_dqn_90"])
    out["phMargin"] = f(ph["five_percent_margin"], 4)
    out["phFracPPO"] = f(100 * ph["ppo_fraction_online"], 1); out["phFracDQN"] = f(100 * ph["dqn_fraction_online"], 1)
    out["phFracGreedy"] = f(100 * ph["greedy_fraction_online"], 1)
    lines = ["% Generated by make_numbers.py from the Stage 29 audit; do not edit by hand."]
    for key in sorted(out):
        if any(ch.isdigit() for ch in key):
            raise SystemExit(f"macro name contains a digit: {key}")
        lines.append(f"\\newcommand{{\\{key}}}{{{out[key]}}}")
    (HERE / "generated").mkdir(exist_ok=True)
    (HERE / "generated/numbers.tex").write_text("\n".join(lines) + "\n")
    print(f"Wrote {len(out)} macros")


if __name__ == "__main__":
    main()
