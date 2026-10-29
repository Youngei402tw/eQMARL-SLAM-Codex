"""Independent release audit and paper artifacts for the nonlocal study."""
from __future__ import annotations

import json
import statistics
from pathlib import Path

import numpy as np

from .archive import validate_bundle
from .nonlocal_training import METHODS, expected_win, restore

ROOT = Path(__file__).resolve().parents[1]


def run_diagnostics(config, output, emit, progress=None):
    bundle = ROOT / config["confirmatory_bundle"]
    validation = validate_bundle(bundle, expected_source_commit=config["confirmatory_source_commit"], repository_root=ROOT)
    freeze = json.loads((ROOT / config["freeze_path"]).read_text())
    published = json.loads((bundle / "confirmatory_analysis.json").read_text())
    raw = [json.loads(line) for line in (bundle / "metrics.jsonl").read_text().splitlines()]
    raw_cells = {(row["method"], row["seed"]): row for row in raw if row.get("kind") == "confirmatory_final"}
    checkpoints = sorted((bundle / "checkpoints").glob("*.json"))
    replay = {}
    replay_errors = []
    for checkpoint in checkpoints:
        payload = json.loads(checkpoint.read_text())
        method, seed = payload["method"], int(payload["seed"])
        value = float(expected_win(method, restore(method, payload["parameters"])))
        replay[(method, seed)] = value
        raw_value = float(raw_cells[(method, seed)]["exact_expected_win"])
        replay_errors.append(abs(value - float(payload["final_expected_win"])))
        replay_errors.append(abs(value - raw_value))
        emit({"kind": "checkpoint_replay", "method": method, "seed": seed, "exact_expected_win": value,
              "max_abs_error": max(replay_errors[-2:])})

    seeds = tuple(freeze["training_seeds"])
    comparator = np.asarray([max(replay[("classical_shared", seed)], replay[("quantum_separable", seed)]) for seed in seeds])
    entangled = np.asarray([replay[("quantum_entangled", seed)] for seed in seeds])
    differences = entangled - comparator
    rng = np.random.default_rng(freeze["bootstrap"]["rng_seed"])
    indices = rng.integers(0, len(seeds), size=(freeze["bootstrap"]["paired_resamples"], len(seeds)))
    difference_draws = differences[indices].mean(axis=1)
    entangled_draws = entangled[indices].mean(axis=1)
    reproduced = {
        "estimate_entangled_minus_paired_strongest_control": float(differences.mean()),
        "paired_difference_interval_95": [float(np.percentile(difference_draws, 2.5)), float(np.percentile(difference_draws, 97.5))],
        "entangled_mean": float(entangled.mean()),
        "entangled_mean_one_sided_lower_95": float(np.percentile(entangled_draws, 5.0)),
        "seed_differences": differences.tolist(),
    }
    keys = ("estimate_entangled_minus_paired_strongest_control", "paired_difference_interval_95", "entangled_mean", "entangled_mean_one_sided_lower_95", "seed_differences")
    analysis_exact = all(reproduced[key] == published["primary"][key] for key in keys)
    shot_rows = [row for row in raw if row.get("kind") == "finite_shot_evaluation" and row["method"] == "quantum_entangled"]
    shot_wins = sum(row["wins"] for row in shot_rows); shot_decisions = sum(row["decisions"] for row in shot_rows)
    shot_exact = shot_wins == published["finite_shot_entangled"]["wins"] and shot_decisions == published["finite_shot_entangled"]["decisions"]

    tables = {
        "primary": reproduced,
        "method_summary": {method: {"mean": statistics.fmean(replay[(method, seed)] for seed in seeds),
                                    "min": min(replay[(method, seed)] for seed in seeds),
                                    "max": max(replay[(method, seed)] for seed in seeds)} for method in METHODS},
        "finite_shot_entangled": published["finite_shot_entangled"],
        "noise_sensitivity": published["noise_sensitivity"],
        "resources": json.loads((bundle / "manifest.json").read_text())["resources_consumed"],
    }
    claims = [
        {"claim": "A learned Bell-entangled decentralized policy exceeds the classical no-communication CHSH ceiling.",
         "supported": published["primary"]["positive_gate_passed"],
         "scope": "two agents, uniform CHSH contexts, no post-input communication, analytic training and simulated finite shots"},
        {"claim": "The advantage remains visible at 2% and 5% registered local depolarizing noise in exact simulation.",
         "supported": all(row["mean_exact_win"] > 0.75 for row in published["noise_sensitivity"]),
         "scope": "specified final local-depolarizing model; no hardware claim"},
        {"claim": "The result establishes speedup or active-SLAM advantage.", "supported": False,
         "scope": "not tested; the independent active-SLAM critic studies remain null"},
        {"claim": "The result survives free classical communication.", "supported": False,
         "scope": "one communicated input bit permits a classical perfect policy and changes the information contract"},
    ]
    Path(output, "reproduced_analysis.json").write_text(json.dumps(reproduced, indent=2) + "\n")
    Path(output, "paper_tables.json").write_text(json.dumps(tables, indent=2) + "\n")
    Path(output, "claim_limitations.json").write_text(json.dumps(claims, indent=2) + "\n")
    gates = {"confirmatory_bundle_valid": validation["valid"], "all_64_checkpoints_replayed": len(checkpoints) == 64,
             "checkpoint_replay_exact": max(replay_errors) <= 1e-15, "raw_cell_count_exact": len(raw_cells) == 64,
             "primary_analysis_exact": analysis_exact, "finite_shot_totals_exact": shot_exact,
             "positive_claim_scoped": claims[0]["supported"] and not claims[2]["supported"] and not claims[3]["supported"]}
    return {"passed": all(gates.values()), "gates": gates,
            "resources": {"primitive_steps": 0, "decisions": 0, "updates": 0, "circuit_evaluations": 0, "shots": 0},
            "summary": {"validation": validation, "reproduced": reproduced, "tables": tables, "claims": claims}}
