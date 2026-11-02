"""Locked held-out study of learned budgeted entanglement routing."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from .ber_marl import dqn_action, greedy_action, train_dqn
from .budgeted_entanglement import BudgetedAllocationMDP, certify_request

ROOT = Path(__file__).resolve().parents[1]


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _clairvoyant(mdp, sequence):
    horizon = len(sequence); maximum = mdp.catalog["bell_pair_budget"]
    value = np.zeros((horizon + 1, maximum + 1))
    for step in range(horizon - 1, -1, -1):
        state = sequence[step]
        for budget in range(maximum + 1):
            value[step, budget] = max(mdp.action_gain(state, action) + value[step + 1, budget - len(action)]
                                      for action in mdp.actions(state, budget))
    return float(value[0, maximum])


def _bootstrap(values, seed, replicates, percentile):
    values = np.asarray(values, dtype=float); rng = np.random.default_rng(seed)
    means = np.empty(replicates)
    for index in range(replicates):
        means[index] = np.mean(values[rng.integers(0, len(values), size=len(values))])
    return float(np.percentile(means, percentile)), float(np.percentile(means, 100 - percentile))


def run_diagnostics(config, output, emit, progress=None):
    freeze_path = ROOT / config["freeze"]["path"]
    if _sha(freeze_path) != config["freeze"]["sha256"]:
        raise ValueError("confirmatory freeze hash mismatch")
    freeze = json.loads(freeze_path.read_text())
    for key in ("test_request_manifest", "test_public_catalog"):
        reference = freeze[key]
        if _sha(ROOT / reference["path"]) != reference["sha256"]:
            raise ValueError(f"{key} hash mismatch")

    test_manifest = json.loads((ROOT / freeze["test_request_manifest"]["path"]).read_text())
    test_certificates = {}
    for index, request in enumerate(test_manifest["types"]):
        row = certify_request(request, 23100 + index * 32); row["split"] = "test"
        test_certificates[row["type_id"]] = row
        emit({"kind": "test_request_certificate", **row})
    Path(output, "test_request_certificates.json").write_text(
        json.dumps({"certificates": list(test_certificates.values())}, indent=2) + "\n")

    non_test_rows = json.loads((ROOT / config["non_test_certificate_bundle"]).read_text())["certificates"]
    non_test_certificates = {row["type_id"]: row for row in non_test_rows}
    training_mdps = []
    for split in ("development", "validation"):
        training_mdps.append(BudgetedAllocationMDP(
            json.loads((ROOT / config["training_catalogs"][split]).read_text()), non_test_certificates))
    test_mdp = BudgetedAllocationMDP(
        json.loads((ROOT / freeze["test_public_catalog"]["path"]).read_text()), test_certificates)
    sequence_seeds = list(range(freeze["evaluation"]["sequence_seed_start"],
                                freeze["evaluation"]["sequence_seed_start"] + freeze["evaluation"]["sequences_per_policy"]))
    sequences = [test_mdp.sample_sequence(seed) for seed in sequence_seeds]

    edge_totals = np.zeros(6); edge_counts = np.zeros(6)
    for mdp in training_mdps:
        for state in range(len(mdp.catalog["states"])):
            for request in mdp.catalog["states"][state]["requests"]:
                edge = request["edge_index"]; edge_totals[edge] += mdp.action_gain(state, (edge,)); edge_counts[edge] += 1
    edge_priority = edge_totals / np.maximum(edge_counts, 1)
    oracle_table = test_mdp.oracle_table()
    baselines = {name: [] for name in ("classical_exact", "quantum_random", "quantum_static", "quantum_greedy",
                                        "dp_online_oracle", "future_clairvoyant", "message_oracle")}
    raw_path = Path(output, "episode_results.jsonl")
    with raw_path.open("w") as raw:
        for episode, (sequence_seed, sequence) in enumerate(zip(sequence_seeds, sequences)):
            greedy = test_mdp.evaluate(sequence, greedy_action)
            oracle = test_mdp.evaluate(sequence, lambda env, state, budget, remaining: env.oracle_action(oracle_table, remaining, state, budget))
            rng = np.random.default_rng(sequence_seed + 700000)
            random_result = test_mdp.evaluate(sequence, lambda env, state, budget, remaining: env.actions(state, budget)[int(rng.integers(len(env.actions(state, budget))))])
            static_result = test_mdp.evaluate(sequence, lambda env, state, budget, remaining: max(
                env.actions(state, budget), key=lambda action: (sum(edge_priority[edge] for edge in action), len(action))))
            clairvoyant = _clairvoyant(test_mdp, sequence)
            classical = greedy["classical_return"]
            message = sum(sum(request["urgency"] * (1.0 - test_certificates[request["type_id"]]["classical_value"])
                              for request in test_mdp.catalog["states"][state]["requests"]) for state in sequence)
            rows = {
                "classical_exact": {"surplus": 0.0, "total_return": classical, "pairs_used": 0},
                "quantum_random": {"surplus": random_result["entanglement_surplus"], "total_return": random_result["total_return"], "pairs_used": random_result["pairs_used"]},
                "quantum_static": {"surplus": static_result["entanglement_surplus"], "total_return": static_result["total_return"], "pairs_used": static_result["pairs_used"]},
                "quantum_greedy": {"surplus": greedy["entanglement_surplus"], "total_return": greedy["total_return"], "pairs_used": greedy["pairs_used"]},
                "dp_online_oracle": {"surplus": oracle["entanglement_surplus"], "total_return": oracle["total_return"], "pairs_used": oracle["pairs_used"]},
                "future_clairvoyant": {"surplus": clairvoyant, "total_return": classical + clairvoyant, "pairs_used": 6},
                "message_oracle": {"surplus": message, "total_return": classical + message, "pairs_used": 0},
            }
            for method, result in rows.items():
                baselines[method].append(result["surplus"])
                raw.write(json.dumps({"kind": "baseline_episode", "sequence_seed": sequence_seed, "method": method, **result}) + "\n")

        seed_rows = []
        violations = []
        for seed in freeze["training"]["seeds"]:
            model, training = train_dqn(training_mdps, seed, episodes=freeze["training"]["episodes_per_seed"],
                                        hidden=freeze["development_selection"]["selected"]["hidden"],
                                        learning_rate=freeze["development_selection"]["selected"]["learning_rate"],
                                        batch_size=freeze["training"]["batch_size"], target_interval=freeze["training"]["target_interval"])
            checkpoint = Path(output, f"ber-marl-seed-{seed}.pt")
            torch.save({"state_dict": model.state_dict(), "hidden": 64, "seed": seed, "training": training}, checkpoint)
            values = []
            for sequence_seed, sequence in zip(sequence_seeds, sequences):
                result = test_mdp.evaluate(sequence, lambda env, state, budget, remaining: dqn_action(model, env, state, budget, remaining))
                values.append(result["entanglement_surplus"])
                if result["pairs_used"] > 6 or any(action["budget_after"] < 0 for action in result["actions"]):
                    violations.append({"seed": seed, "sequence_seed": sequence_seed, "reason": "budget"})
                raw.write(json.dumps({"kind": "ber_marl_episode", "training_seed": seed, "sequence_seed": sequence_seed,
                                      "surplus": result["entanglement_surplus"], "total_return": result["total_return"],
                                      "pairs_used": result["pairs_used"], "pairs_unused": result["pairs_unused"]}) + "\n")
            row = {"training_seed": seed, "mean_surplus": float(np.mean(values)),
                   "mean_greedy_difference": float(np.mean(np.asarray(values) - np.asarray(baselines["quantum_greedy"]))),
                   "dp_fraction": float(np.mean(values) / np.mean(baselines["dp_online_oracle"])), "training": training}
            seed_rows.append(row); emit({"kind": "confirmatory_seed", **row})

    differences = [row["mean_greedy_difference"] for row in seed_rows]
    fractions = [row["dp_fraction"] for row in seed_rows]
    surpluses = [row["mean_surplus"] for row in seed_rows]
    reps = freeze["inference"]["bootstrap_replicates"]; bootstrap_seed = freeze["inference"]["bootstrap_seed"]
    difference_ci = _bootstrap(differences, bootstrap_seed, reps, 2.5)
    fraction_ci = _bootstrap(fractions, bootstrap_seed + 1, reps, 5.0)
    surplus_ci = _bootstrap(surpluses, bootstrap_seed + 2, reps, 5.0)
    dp_mean = float(np.mean(baselines["dp_online_oracle"])); greedy_mean = float(np.mean(baselines["quantum_greedy"]))
    gates = {
        "paired_95_lower_above_zero": difference_ci[0] > 0,
        "improvement_at_least_5pct_dp": float(np.mean(differences)) >= 0.05 * dp_mean,
        "one_sided_dp_fraction_lower_above_0_90": fraction_ci[0] > 0.90,
        "one_sided_surplus_lower_above_zero": surplus_ci[0] > 0,
        "budget_and_matching_exact": not violations,
    }
    request_gaps = [row["quantum_gap"] for row in test_certificates.values()]
    summary = {"method_means": {method: float(np.mean(values)) for method, values in baselines.items()},
               "ber_marl_mean_surplus": float(np.mean(surpluses)), "ber_marl_seed_rows": seed_rows,
               "greedy_mean_surplus": greedy_mean, "dp_mean_surplus": dp_mean,
               "mean_greedy_difference": float(np.mean(differences)), "mean_dp_fraction": float(np.mean(fractions)),
               "difference_bootstrap_95": difference_ci, "dp_fraction_one_sided_95": fraction_ci,
               "surplus_one_sided_95": surplus_ci, "test_request_gap_summary": {"min": min(request_gaps), "mean": float(np.mean(request_gaps)), "max": max(request_gaps)},
               "violations": violations, "gates": gates}
    Path(output, "confirmatory_analysis.json").write_text(json.dumps(summary, indent=2) + "\n")
    certification_updates = len(test_certificates) * 16 * 800
    training_updates = sum(row["training"]["updates"] for row in seed_rows)
    decisions = sum(row["training"]["decisions"] for row in seed_rows) + len(seed_rows) * len(sequences) * 10 + len(sequences) * 40
    return {"passed": all(gates.values()), "gates": gates,
            "resources": {"primitive_steps": decisions, "decisions": decisions, "updates": training_updates + certification_updates,
                          "circuit_evaluations": certification_updates * 9, "shots": 0},
            "summary": summary}
