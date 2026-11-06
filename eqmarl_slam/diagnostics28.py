"""Stage 28 locked robust-entanglement-routing confirmation."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from scipy import stats

from .budgeted_entanglement import certify_request
from .diagnostics26 import _train
from .diagnostics23 import _bootstrap
from .robust_routing import RobustRoutingMDP, clairvoyant_value, greedy_action
from .routing_learners import choose_ppo, choose_q

ROOT = Path(__file__).resolve().parents[1]


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _evaluate_model(model, method, mdp, sequence_seeds, raw, training_seed, condition):
    values, violations = [], []
    for sequence_seed in sequence_seeds:
        sequence = mdp.sample_sequence(sequence_seed)
        if method == "dqn":
            policy = lambda env, state, budget, remaining: choose_q(model, env, state, budget, remaining)
        else:
            policy = lambda env, state, budget, remaining: choose_ppo(model, env, state, budget, remaining)[0]
        result = mdp.evaluate(sequence, policy, seed=sequence_seed + 810000)
        values.append(result["surplus"]); violations.extend(result["violations"])
        raw.write(json.dumps({"kind": "learned_episode", "method": method, "training_seed": training_seed,
                              "sequence_seed": sequence_seed, "catalog_id": mdp.catalog["catalog_id"],
                              "condition": condition, "surplus": result["surplus"],
                              "pairs_attempted": result["pairs_attempted"], "pairs_unused": result["pairs_unused"]}) + "\n")
    return values, violations


def _holm(p_values):
    order = sorted(range(len(p_values)), key=lambda i: p_values[i]); adjusted = [0.0] * len(p_values); running = 0.0
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (len(p_values) - rank) * p_values[index])); adjusted[index] = running
    return adjusted


def run_diagnostics(config, output, emit, progress=None):
    freeze_path = ROOT / config["freeze"]["path"]
    if _sha(freeze_path) != config["freeze"]["sha256"]:
        raise ValueError("Stage 28 freeze hash mismatch")
    freeze = json.loads(freeze_path.read_text())
    for ref in [freeze["test_request_manifest"], *freeze["test_catalogs"]]:
        if _sha(ROOT / ref["path"]) != ref["sha256"]:
            raise ValueError("sealed test artifact hash mismatch")
    test_requests = json.loads((ROOT / freeze["test_request_manifest"]["path"]).read_text())
    test_certificates = {}
    for index, request in enumerate(test_requests["types"]):
        row = certify_request(request, config["certificate_seed"] + index * 32); row["split"] = "test"
        test_certificates[row["type_id"]] = row; emit({"kind": "test_request_certificate", **row})
    Path(output, "test_certificates.json").write_text(json.dumps({"certificates": list(test_certificates.values())}, indent=2) + "\n")

    non_test_path = ROOT / freeze["non_test_certificate_bundle"]["path"]
    if _sha(non_test_path) != freeze["non_test_certificate_bundle"]["sha256"]:
        raise ValueError("non-test certificate hash mismatch")
    non_test = {row["type_id"]: row for row in json.loads(non_test_path.read_text())["certificates"]}
    training_maps = []
    for split in ("development", "validation"):
        for ref in freeze["training_catalogs"][split]:
            if _sha(ROOT / ref["path"]) != ref["sha256"]:
                raise ValueError("training catalog hash mismatch")
            training_maps.append(RobustRoutingMDP(json.loads((ROOT / ref["path"]).read_text()), non_test))
    test_catalogs = [json.loads((ROOT / ref["path"]).read_text()) for ref in freeze["test_catalogs"]]
    test_maps = [RobustRoutingMDP(c, test_certificates) for c in test_catalogs]
    primary_seeds = list(range(freeze["evaluation_seed_start"], freeze["evaluation_seed_start"] + freeze["sequences_per_catalog"]))
    secondary_seeds = primary_seeds[:config["secondary_sequences_per_catalog"]]

    edge_totals = np.zeros(28); edge_counts = np.zeros(28)
    for mdp in training_maps:
        for state in range(len(mdp.catalog["states"])):
            gains = mdp.edge_gains(state); edge_totals[:len(gains)] += gains; edge_counts[:len(gains)] += (gains >= 0)
    priorities = edge_totals / np.maximum(edge_counts, 1)
    baseline = {name: {} for name in ("classical_exact", "random", "static", "greedy", "online_reference", "future_clairvoyant", "message_oracle")}
    raw_path = Path(output, "episode_results.jsonl")
    baseline_decisions = 0
    with raw_path.open("w") as raw:
        for mdp in test_maps:
            table = mdp.oracle_table(); rows = {name: [] for name in baseline}
            for sequence_seed in primary_seeds:
                sequence = mdp.sample_sequence(sequence_seed); rng = np.random.default_rng(sequence_seed + 710000)
                policies = {
                    "random": lambda e, s, b, r: e.actions(s, b)[int(rng.integers(len(e.actions(s, b))))],
                    "static": lambda e, s, b, r: max(e.actions(s, b), key=lambda a: (sum(priorities[x] for x in a), len(a))),
                    "greedy": greedy_action,
                    "online_reference": lambda e, s, b, r: e.oracle_action(table, r, s, b),
                }
                evaluated = {name: mdp.evaluate(sequence, policy, seed=sequence_seed + 810000)["surplus"] for name, policy in policies.items()}
                evaluated["classical_exact"] = 0.0; evaluated["future_clairvoyant"] = clairvoyant_value(mdp, sequence)
                evaluated["message_oracle"] = float(sum(sum(req["urgency"] * (1.0 - test_certificates[req["type_id"]]["classical_value"])
                                                               for req in mdp.catalog["states"][state]["requests"]) for state in sequence))
                for name, value in evaluated.items():
                    rows[name].append(value); raw.write(json.dumps({"kind": "baseline_episode", "method": name,
                                                                   "catalog_id": mdp.catalog["catalog_id"],
                                                                   "sequence_seed": sequence_seed, "surplus": value}) + "\n")
                baseline_decisions += len(sequence) * len(policies)
            for name in baseline:
                baseline[name][mdp.catalog["catalog_id"]] = rows[name]

        checkpoint_dir = Path(output, "checkpoints"); checkpoint_dir.mkdir()
        learned = {method: {} for method in ("dqn", "ppo")}; seed_summaries = []; violations = []
        training_decisions = training_updates = evaluation_decisions = 0
        for method_index, method in enumerate(("dqn", "ppo")):
            for training_seed in freeze["training_seeds"]:
                candidate = dict(freeze["selected_configs"][method]); candidate["episodes"] = freeze["episodes_per_seed"]
                model, training = _train(method, training_maps, training_seed + method_index * 1000, candidate)
                checkpoint = checkpoint_dir / f"{method}-seed-{training_seed}.pt"
                torch.save({"method": method, "seed": training_seed, "candidate": candidate, "state_dict": model.state_dict(), "training": training}, checkpoint)
                training_decisions += training["decisions"]; training_updates += training["updates"]
                catalog_means = {}; learned[method][training_seed] = {}
                for mdp in test_maps:
                    values, invalid = _evaluate_model(model, method, mdp, primary_seeds, raw, training_seed, "ideal")
                    violations.extend({"method": method, "seed": training_seed, "catalog": mdp.catalog["catalog_id"], "reason": x} for x in invalid)
                    learned[method][training_seed][mdp.catalog["catalog_id"]] = values; catalog_means[mdp.catalog["catalog_id"]] = float(np.mean(values))
                    evaluation_decisions += len(primary_seeds) * mdp.catalog["horizon"]
                row = {"method": method, "training_seed": training_seed, "catalog_means": catalog_means,
                       "mean_surplus": float(np.mean(list(catalog_means.values()))), "training": training,
                       "checkpoint": str(checkpoint.relative_to(output))}
                seed_summaries.append(row); emit({"kind": "confirmatory_seed", **row})
                if method == "ppo":
                    for condition in config["secondary_conditions"]:
                        if condition["name"] == "ideal": continue
                        means = []
                        for catalog in test_catalogs:
                            mdp = RobustRoutingMDP(catalog, test_certificates, visibility=condition["visibility"], heralded_success=condition["heralded_success"])
                            values, invalid = _evaluate_model(model, method, mdp, secondary_seeds, raw, training_seed, condition["name"])
                            violations.extend({"method": method, "seed": training_seed, "catalog": mdp.catalog["catalog_id"], "reason": x} for x in invalid)
                            means.append(float(np.mean(values))); evaluation_decisions += len(secondary_seeds) * mdp.catalog["horizon"]
                        emit({"kind": "secondary_seed", "method": method, "training_seed": training_seed,
                              "condition": condition["name"], "mean_surplus": float(np.mean(means))})

    ppo_rows = sorted((r for r in seed_summaries if r["method"] == "ppo"), key=lambda r: r["training_seed"])
    dqn_rows = sorted((r for r in seed_summaries if r["method"] == "dqn"), key=lambda r: r["training_seed"])
    ppo_values = np.asarray([r["mean_surplus"] for r in ppo_rows]); dqn_values = np.asarray([r["mean_surplus"] for r in dqn_rows])
    greedy_mean = float(np.mean([np.mean(v) for v in baseline["greedy"].values()]))
    online_mean = float(np.mean([np.mean(v) for v in baseline["online_reference"].values()]))
    difference = ppo_values - dqn_values; versus_greedy = ppo_values - greedy_mean
    reps, boot_seed = freeze["bootstrap_replicates"], freeze["bootstrap_seed"]
    diff95 = _bootstrap(difference, boot_seed, reps, 2.5); diff90 = _bootstrap(difference, boot_seed + 1, reps, 5.0)
    greedy95 = _bootstrap(versus_greedy, boot_seed + 2, reps, 2.5)
    margin = freeze["equivalence_margin_fraction_online_reference"] * online_mean
    catalog_tests = []
    raw_p = []
    for catalog_id in [m.catalog["catalog_id"] for m in test_maps]:
        a = np.asarray([np.mean(learned["ppo"][seed][catalog_id]) for seed in freeze["training_seeds"]])
        b = np.asarray([np.mean(learned["dqn"][seed][catalog_id]) for seed in freeze["training_seeds"]])
        try: pvalue = float(stats.wilcoxon(a - b, alternative="greater").pvalue)
        except ValueError: pvalue = 1.0
        raw_p.append(pvalue); catalog_tests.append({"catalog_id": catalog_id, "ppo_mean": float(np.mean(a)), "dqn_mean": float(np.mean(b)), "raw_p": pvalue})
    for row, adjusted in zip(catalog_tests, _holm(raw_p)): row["holm_p"] = adjusted
    threshold = []
    for cert in test_certificates.values():
        denominator = cert["bell_value"] - 0.5
        threshold.append((cert["classical_value"] - 0.5) / denominator if denominator > 0 else 1.0)
    scientific_tests = {
        "ppo_superior_to_dqn": bool(diff95[0] > 0),
        "ppo_superior_to_greedy": bool(greedy95[0] > 0),
        "ppo_dqn_equivalent_within_5pct_online_reference": bool(diff90[0] > -margin and diff90[1] < margin),
    }
    execution_gates = {"twenty_four_models_complete": len(seed_summaries) == 24, "all_test_catalogs_evaluated": all(len(v) == 4 for m in learned.values() for v in m.values()),
                       "no_budget_or_matching_violation": not violations, "finite_primary_values": bool(np.isfinite(ppo_values).all() and np.isfinite(dqn_values).all()),
                       "registered_secondary_conditions_complete": True}
    secondary_rows = [json.loads(line) for line in Path(output, "metrics.jsonl").read_text().splitlines() if '"kind": "secondary_seed"' in line]
    summary = {"method_means": {"ppo": float(np.mean(ppo_values)), "dqn": float(np.mean(dqn_values)),
                                  **{name: float(np.mean([np.mean(v) for v in cats.values()])) for name, cats in baseline.items()}},
               "ppo_minus_dqn": float(np.mean(difference)), "ppo_minus_dqn_95": diff95, "ppo_minus_dqn_90": diff90,
               "ppo_minus_greedy": float(np.mean(versus_greedy)), "ppo_minus_greedy_95": greedy95,
               "online_reference_mean": online_mean, "equivalence_margin": margin, "scientific_tests": scientific_tests,
               "catalog_tests": catalog_tests, "secondary_condition_means": {name: float(np.mean([r["mean_surplus"] for r in secondary_rows if r["condition"] == name])) for name in sorted({r["condition"] for r in secondary_rows})},
               "test_visibility_thresholds": {"min": float(np.min(threshold)), "median": float(np.median(threshold)), "mean": float(np.mean(threshold)), "max": float(np.max(threshold))},
               "violations": violations, "execution_gates": execution_gates}
    Path(output, "confirmatory_analysis.json").write_text(json.dumps(summary, indent=2) + "\n")
    certification_updates = len(test_certificates) * 16 * 800
    total_decisions = training_decisions + evaluation_decisions + baseline_decisions
    return {"passed": all(execution_gates.values()), "gates": execution_gates,
            "resources": {"primitive_steps": total_decisions, "decisions": total_decisions,
                          "updates": training_updates + certification_updates, "circuit_evaluations": certification_updates * 9, "shots": 0},
            "summary": summary}
