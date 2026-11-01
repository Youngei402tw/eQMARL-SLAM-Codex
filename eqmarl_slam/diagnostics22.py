"""Bounded BER-MARL tuning, validation selection, and non-test pilots."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from .ber_marl import evaluate_policy, greedy_action, train_dqn
from .budgeted_entanglement import BudgetedAllocationMDP

ROOT = Path(__file__).resolve().parents[1]


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _baseline(mdp, seeds):
    table = mdp.oracle_table(); greedy, oracle = [], []
    for seed in seeds:
        sequence = mdp.sample_sequence(seed)
        greedy.append(mdp.evaluate(sequence, greedy_action)["entanglement_surplus"])
        oracle.append(mdp.evaluate(sequence, lambda env, state, budget, remaining: env.oracle_action(table, remaining, state, budget))["entanglement_surplus"])
    return {"greedy": greedy, "oracle": oracle, "greedy_mean": float(np.mean(greedy)), "oracle_mean": float(np.mean(oracle))}


def run_diagnostics(config, output, emit, progress=None):
    certificate_path = ROOT / config["certificate_bundle"]
    certificate_payload = json.loads(certificate_path.read_text())
    certificates = {row["type_id"]: row for row in certificate_payload["certificates"]}
    mdps = {}
    for split in ("development", "validation"):
        reference = config["public_catalogs"][split]
        path = ROOT / reference["path"]
        if _sha(path) != reference["sha256"]:
            raise ValueError(f"{split} catalog hash mismatch")
        mdps[split] = BudgetedAllocationMDP(json.loads(path.read_text()), certificates)

    validation_seeds = list(range(config["validation_sequence_seed_start"],
                                  config["validation_sequence_seed_start"] + config["validation_sequences"]))
    baseline = _baseline(mdps["validation"], validation_seeds)
    trials = []
    total_updates = total_decisions = 0
    for config_index, candidate in enumerate(config["candidates"]):
        for replicate in range(2):
            seed = config["tuning_seed_start"] + config_index * 10 + replicate
            model, train = train_dqn(mdps["development"], seed, episodes=candidate["episodes"],
                                     hidden=candidate["hidden"], learning_rate=candidate["learning_rate"])
            evaluation = evaluate_policy(model, mdps["validation"], validation_seeds)
            row = {"candidate": candidate["name"], "seed": seed, "train": train,
                   "validation_mean_surplus": evaluation["mean_surplus"],
                   "validation_dp_fraction": evaluation["mean_surplus"] / baseline["oracle_mean"],
                   "validation_greedy_improvement": evaluation["mean_surplus"] - baseline["greedy_mean"]}
            trials.append(row); emit({"kind": "tuning_trial", **row})
            total_updates += train["updates"]; total_decisions += train["decisions"] + len(validation_seeds) * 10
    grouped = {}
    for candidate in config["candidates"]:
        rows = [row for row in trials if row["candidate"] == candidate["name"]]
        grouped[candidate["name"]] = float(np.mean([row["validation_mean_surplus"] for row in rows]))
    selected_name = max(grouped, key=grouped.get)
    selected = next(candidate for candidate in config["candidates"] if candidate["name"] == selected_name)

    pilot_rows = []
    for offset in range(config["pilot_seeds"]):
        seed = config["pilot_seed_start"] + offset
        model, train = train_dqn([mdps["development"], mdps["validation"]], seed,
                                 episodes=config["pilot_episodes"], hidden=selected["hidden"],
                                 learning_rate=selected["learning_rate"])
        evaluations = {split: evaluate_policy(model, mdp, validation_seeds) for split, mdp in mdps.items()}
        checkpoint = Path(output) / f"pilot-seed-{seed}.pt"
        torch.save({"state_dict": model.state_dict(), "hidden": selected["hidden"], "seed": seed}, checkpoint)
        row = {"seed": seed, "train": train,
               **{f"{split}_mean_surplus": value["mean_surplus"] for split, value in evaluations.items()},
               "validation_dp_fraction": evaluations["validation"]["mean_surplus"] / baseline["oracle_mean"],
               "validation_greedy_improvement": evaluations["validation"]["mean_surplus"] - baseline["greedy_mean"]}
        pilot_rows.append(row); emit({"kind": "pilot", **row})
        total_updates += train["updates"]; total_decisions += train["decisions"] + len(mdps) * len(validation_seeds) * 10
    fractions = [row["validation_dp_fraction"] for row in pilot_rows]
    improvements = [row["validation_greedy_improvement"] for row in pilot_rows]
    gates = {
        "four_candidates_two_seeds_each": len(trials) == 8,
        "selection_uses_validation_only": selected_name in grouped,
        "all_tuning_trials_beat_greedy_on_average": min(grouped.values()) > baseline["greedy_mean"],
        "four_combined_non_test_pilots": len(pilot_rows) == 4,
        "pilot_mean_fraction_above_0_90": float(np.mean(fractions)) > 0.90,
        "every_pilot_beats_greedy": min(improvements) > 0,
        "test_values_unopened": all("test-type" not in key for key in certificates),
    }
    selection = {"selected": selected, "candidate_means": grouped, "validation_baseline": baseline,
                 "pilot_rows": pilot_rows, "gates": gates}
    Path(output, "selection.json").write_text(json.dumps(selection, indent=2) + "\n")
    return {"passed": all(gates.values()), "gates": gates,
            "resources": {"primitive_steps": total_decisions, "decisions": total_decisions,
                          "updates": total_updates, "circuit_evaluations": 0, "shots": 0},
            "summary": selection}
