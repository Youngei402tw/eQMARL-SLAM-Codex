"""Bounded development and confirmatory freeze for the nonlocal study."""
from __future__ import annotations

import json
import statistics
from pathlib import Path

import numpy as np

from .nonlocal_training import METHODS, restore, train, expected_win


def run_diagnostics(config, output, emit, progress=None):
    learning_rates = tuple(config["learning_rates"])
    tuning_seeds = tuple(config["tuning_seeds"])
    pilot_seeds = tuple(config["pilot_seeds"])
    tuning_updates = int(config["tuning_updates"])
    pilot_updates = int(config["pilot_updates"])
    tuning = []
    selected = {}
    updates = 0
    for method in METHODS:
        candidates = []
        for learning_rate in learning_rates:
            rows = []
            for seed in tuning_seeds:
                row = train(method, seed, learning_rate, tuning_updates)
                updates += tuning_updates
                rows.append(row); tuning.append(row)
                emit({"kind": "tuning_result", **{key: value for key, value in row.items() if key != "curve"}})
            candidates.append({"learning_rate": learning_rate,
                               "mean_final_expected_win": statistics.fmean(row["final_expected_win"] for row in rows),
                               "min_final_expected_win": min(row["final_expected_win"] for row in rows)})
        # Deterministic tie-break toward the smaller rate.
        choice = max(candidates, key=lambda row: (round(row["mean_final_expected_win"], 12), -row["learning_rate"]))
        selected[method] = choice

    pilots = []
    for method in METHODS:
        for seed in pilot_seeds:
            row = train(method, seed, selected[method]["learning_rate"], pilot_updates)
            updates += pilot_updates
            replay = float(expected_win(method, restore(method, row["parameters"])))
            row["serialization_replay_error"] = abs(replay - row["final_expected_win"])
            pilots.append(row)
            emit({"kind": "pilot_result", **{key: value for key, value in row.items() if key != "curve"}})

    by_method = {method: [row for row in pilots if row["method"] == method] for method in METHODS}
    pilot_summary = {method: {"mean": statistics.fmean(row["final_expected_win"] for row in rows),
                              "min": min(row["final_expected_win"] for row in rows),
                              "max": max(row["final_expected_win"] for row in rows)}
                     for method, rows in by_method.items()}
    freeze = {
        "protocol_id": "nonlocal-confirmatory-v1",
        "methods": list(METHODS),
        "training_seeds": list(range(18001, 18017)),
        "learning_rates": {method: selected[method]["learning_rate"] for method in METHODS},
        "updates": pilot_updates,
        "checkpoint_rule": "final update only",
        "primary_metric": "exact expected CHSH win probability under uniform contexts",
        "primary_comparator": "paired seed maximum of classical_shared and quantum_separable",
        "classical_ceiling": 0.75,
        "minimum_effect_over_ceiling": 0.05,
        "bootstrap": {"paired_resamples": 10000, "rng_seed": 19001,
                      "difference_interval": "two-sided percentile 95%",
                      "entangled_mean_lower_bound": "one-sided percentile 95%"},
        "positive_gate": ["paired difference interval lower bound > 0",
                          "entangled mean one-sided lower bound > 0.75",
                          "entangled mean - 0.75 >= 0.05"],
        "finite_shot": {"decisions_per_trained_seed": 8192,
                        "rng_seeds": list(range(28001, 28017)),
                        "scope": "secondary operational simulation"},
        "noise_probabilities": [0.0, 0.02, 0.05],
        "noise_scope": "secondary descriptive exact local-depolarizing sensitivity",
        "failure_handling": "retain every run; no seed replacement; primary incomplete if any planned cell lacks a final outcome",
    }
    Path(output, "confirmatory_freeze.json").write_text(json.dumps(freeze, indent=2) + "\n")
    Path(output, "development_checkpoints.json").write_text(json.dumps({"tuning": tuning, "pilots": pilots}, indent=2) + "\n")
    gates = {
        "equal_tuning_opportunity": len(tuning) == len(METHODS) * len(learning_rates) * len(tuning_seeds),
        "all_gradients_finite": all(row["gradients_finite"] for row in tuning + pilots),
        "serialization_exact": max(row["serialization_replay_error"] for row in pilots) <= 1e-15,
        "controls_respect_ceiling": max(row["final_expected_win"] for method in ("classical_local", "classical_shared", "quantum_separable") for row in by_method[method]) <= 0.750000001,
        "entangled_pilot_testable": pilot_summary["quantum_entangled"]["min"] >= 0.82,
        "fresh_confirmatory_seeds": not set(freeze["training_seeds"]) & (set(tuning_seeds) | set(pilot_seeds)),
    }
    return {"passed": all(gates.values()), "gates": gates,
            "resources": {"primitive_steps": 0, "decisions": 0, "updates": updates,
                          "circuit_evaluations": sum(row["updates"] * 4 for row in tuning + pilots if row["method"].startswith("quantum_")), "shots": 0},
            "summary": {"selected": selected, "pilot_summary": pilot_summary, "tuning_runs": len(tuning),
                        "pilot_runs": len(pilots), "freeze": freeze}}
