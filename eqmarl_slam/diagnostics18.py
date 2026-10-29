"""Locked confirmatory learned-policy experiment for nonlocal coordination."""
from __future__ import annotations

import hashlib
import json
import math
import statistics
from pathlib import Path

import numpy as np
import pennylane as qml
import torch

from .nonlocal_game import CONTEXTS, deterministic_strategy_scores, reward
from .nonlocal_training import METHODS, expected_win, restore, train


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sample_quantum(parameters, entangled, decisions, seed):
    if decisions % 4:
        raise ValueError("finite-shot decisions must divide equally across four contexts")
    per_context = decisions // 4
    wins = 0
    executions = 0
    for index, (x, y) in enumerate(CONTEXTS):
        device = qml.device("default.qubit", wires=2, shots=per_context, seed=seed + index)

        @qml.qnode(device)
        def circuit():
            if entangled:
                qml.Hadamard(0); qml.CNOT([0, 1])
            qml.RY(float(parameters["alpha"][x]), 0)
            qml.RY(float(parameters["beta"][y]), 1)
            return qml.sample(wires=[0, 1])

        samples = np.asarray(circuit(), dtype=int)
        wins += int(np.sum((samples[:, 0] ^ samples[:, 1]) == (x & y)))
        executions += 1
    return wins, executions, decisions


def _sample_classical(method, parameters, decisions, seed):
    if decisions % 4:
        raise ValueError("sampled decisions must divide equally across four contexts")
    rng = np.random.default_rng(seed)
    per_context = decisions // 4
    wins = 0
    if method == "classical_shared":
        logits = np.asarray(parameters["strategy_logits"], dtype=float)
        weights = np.exp(logits - logits.max()); weights /= weights.sum()
        strategies = deterministic_strategy_scores()
    for x, y in CONTEXTS:
        if method == "classical_local":
            pa = 1 / (1 + np.exp(-float(parameters["alice_logits"][x])))
            pb = 1 / (1 + np.exp(-float(parameters["bob_logits"][y])))
            actions_a = rng.binomial(1, pa, per_context); actions_b = rng.binomial(1, pb, per_context)
            wins += int(np.sum((actions_a ^ actions_b) == (x & y)))
        else:
            selected = rng.choice(len(strategies), size=per_context, p=weights)
            wins += sum(reward(x, y, strategies[index]["alice"][x], strategies[index]["bob"][y]) for index in selected)
    return wins


def _wilson(wins, trials, z=1.959963984540054):
    p = wins / trials; denominator = 1 + z * z / trials
    center = (p + z * z / (2 * trials)) / denominator
    half = z * math.sqrt(p * (1 - p) / trials + z * z / (4 * trials * trials)) / denominator
    return [center - half, center + half]


def run_diagnostics(config, output, emit, progress=None):
    freeze_path = Path(config["freeze_path"])
    freeze = json.loads(freeze_path.read_text())
    if _sha256(freeze_path) != config["freeze_sha256"]:
        raise ValueError("confirmatory freeze hash mismatch")
    if freeze["protocol_id"] != config["protocol_id"] or tuple(freeze["methods"]) != METHODS:
        raise ValueError("confirmatory config differs from frozen protocol")
    seeds = tuple(freeze["training_seeds"])
    updates = int(freeze["updates"])
    cells = []
    checkpoints = Path(output, "checkpoints"); checkpoints.mkdir()
    total_updates = 0
    circuit_evaluations = 0
    for method in METHODS:
        for seed in seeds:
            row = train(method, seed, float(freeze["learning_rates"][method]), updates)
            total_updates += updates
            if method.startswith("quantum_"):
                circuit_evaluations += updates * 4
            checkpoint = checkpoints / f"{method}_seed-{seed}.json"
            checkpoint.write_text(json.dumps(row, indent=2) + "\n")
            cells.append({"method": method, "seed": seed, "exact_expected_win": row["final_expected_win"],
                          "checkpoint": str(checkpoint.relative_to(output)), "gradients_finite": row["gradients_finite"]})
            emit({"kind": "confirmatory_final", **cells[-1]})

    exact = {(row["method"], row["seed"]): row["exact_expected_win"] for row in cells}
    comparator = [max(exact[("classical_shared", seed)], exact[("quantum_separable", seed)]) for seed in seeds]
    entangled = [exact[("quantum_entangled", seed)] for seed in seeds]
    differences = np.asarray([entangled[i] - comparator[i] for i in range(len(seeds))])
    rng = np.random.default_rng(freeze["bootstrap"]["rng_seed"])
    indices = rng.integers(0, len(seeds), size=(freeze["bootstrap"]["paired_resamples"], len(seeds)))
    difference_draws = differences[indices].mean(axis=1)
    entangled_array = np.asarray(entangled)
    entangled_draws = entangled_array[indices].mean(axis=1)
    difference_interval = [float(np.percentile(difference_draws, 2.5)), float(np.percentile(difference_draws, 97.5))]
    entangled_lower = float(np.percentile(entangled_draws, 5.0))
    estimate = float(differences.mean())
    entangled_mean = statistics.fmean(entangled)
    positive_conditions = {
        "paired_interval_lower_above_zero": difference_interval[0] > 0,
        "entangled_one_sided_lower_above_classical_ceiling": entangled_lower > freeze["classical_ceiling"],
        "minimum_effect_over_ceiling_met": entangled_mean - freeze["classical_ceiling"] >= freeze["minimum_effect_over_ceiling"],
    }

    shot_rows = []
    total_shots = total_decisions = sampled_wins = sampled_trials = 0
    for seed, shot_seed in zip(seeds, freeze["finite_shot"]["rng_seeds"]):
        for method in METHODS:
            checkpoint = json.loads((checkpoints / f"{method}_seed-{seed}.json").read_text())
            parameters = restore(method, checkpoint["parameters"])
            decisions = int(freeze["finite_shot"]["decisions_per_trained_seed"])
            if method.startswith("quantum_"):
                wins, executions, shots = _sample_quantum(parameters, method == "quantum_entangled", decisions, shot_seed)
                circuit_evaluations += executions; total_shots += shots
            else:
                wins = _sample_classical(method, parameters, decisions, shot_seed); executions = shots = 0
            row = {"kind": "finite_shot_evaluation", "method": method, "seed": seed, "sampling_seed": shot_seed,
                   "decisions": decisions, "wins": wins, "win_rate": wins / decisions,
                   "circuit_evaluations": executions, "simulated_quantum_shots": shots}
            shot_rows.append(row); emit(row); total_decisions += decisions
            if method == "quantum_entangled": sampled_wins += wins; sampled_trials += decisions

    noise_rows = []
    for probability in freeze["noise_probabilities"]:
        values = []
        for seed in seeds:
            checkpoint = json.loads((checkpoints / f"quantum_entangled_seed-{seed}.json").read_text())
            parameters = restore("quantum_entangled", checkpoint["parameters"])
            values.append(float(expected_win("quantum_entangled", parameters, noise=probability)))
        row = {"kind": "noise_sensitivity", "probability": probability, "mean_exact_win": statistics.fmean(values), "seed_values": values}
        noise_rows.append(row); emit(row)

    analysis = {
        "analysis_version": "nonlocal-confirmatory-v1",
        "complete": len(cells) == len(METHODS) * len(seeds) and all(row["gradients_finite"] for row in cells),
        "primary": {"estimate_entangled_minus_paired_strongest_control": estimate,
                    "paired_difference_interval_95": difference_interval,
                    "entangled_mean": entangled_mean,
                    "entangled_mean_one_sided_lower_95": entangled_lower,
                    "classical_ceiling": freeze["classical_ceiling"],
                    "minimum_effect_over_ceiling": freeze["minimum_effect_over_ceiling"],
                    "seed_differences": differences.tolist(), "positive_conditions": positive_conditions,
                    "positive_gate_passed": all(positive_conditions.values())},
        "method_summary": {method: {"mean": statistics.fmean(exact[(method, seed)] for seed in seeds),
                                    "min": min(exact[(method, seed)] for seed in seeds),
                                    "max": max(exact[(method, seed)] for seed in seeds)} for method in METHODS},
        "finite_shot_entangled": {"wins": sampled_wins, "decisions": sampled_trials,
                                  "win_rate": sampled_wins / sampled_trials,
                                  "wilson_interval_95": _wilson(sampled_wins, sampled_trials)},
        "noise_sensitivity": noise_rows,
        "cell_table": cells,
    }
    Path(output, "confirmatory_analysis.json").write_text(json.dumps(analysis, indent=2) + "\n")
    gates = {"complete_locked_matrix": analysis["complete"], "frozen_protocol_hash": True,
             "final_checkpoint_only": True, "all_raw_cells_emitted": len(cells) == 64,
             "finite_shot_complete": len(shot_rows) == 64,
             "primary_positive_gate": analysis["primary"]["positive_gate_passed"]}
    return {"passed": all(gates.values()), "gates": gates,
            "resources": {"primitive_steps": total_decisions, "decisions": total_decisions,
                          "updates": total_updates, "circuit_evaluations": circuit_evaluations, "shots": total_shots},
            "summary": analysis}
