"""Independent raw-row reconstruction of the Stage 13 primary result.

This deliberately does not import the Stage 13/14 analysis or experiment code.
"""
from __future__ import annotations

import hashlib
import json
import statistics
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
METHODS = ("central_classical", "bilinear", "flexible_interaction", "quantum_noent", "eqmarl")
CONDITIONS = ("additive", "low", "high")
SEEDS = tuple(range(13001, 13009))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    cells: dict[tuple[str, str, int], float] = {}
    scenarios: dict[tuple[str, str, int], tuple[str, ...]] = {}
    errors: list[str] = []
    checked_artifacts = 0
    folders = sorted((ROOT / "experiments/13").glob("13-confirmatory-*"))
    for folder in folders:
        manifest = json.loads((folder / "manifest.json").read_text())
        config = json.loads((folder / "resolved_config.yaml").read_text())
        diagnostics = json.loads((folder / "diagnostics.json").read_text())
        artifact_manifest = json.loads((folder / "artifact_manifest.json").read_text())
        if manifest["status"] != "completed":
            errors.append(f"{folder.name}: status={manifest['status']}")
        if sha256(folder / "resolved_config.yaml") != manifest["config_sha256"]:
            errors.append(f"{folder.name}: config hash")
        for artifact in artifact_manifest["artifacts"]:
            checked_artifacts += 1
            path = folder / artifact["path"]
            if not path.is_file() or path.stat().st_size != artifact["size_bytes"] or sha256(path) != artifact["sha256"]:
                errors.append(f"{folder.name}: artifact {artifact['path']}")
        rows = [json.loads(line) for line in (folder / "metrics.jsonl").read_text().splitlines()]
        tests = [row for row in rows if row.get("kind") == "test_episode"]
        if len(tests) != 100:
            errors.append(f"{folder.name}: {len(tests)} test rows")
            continue
        key = (config["condition"]["name"], config["method"], int(config["actor_seed"]))
        raw_mean = statistics.fmean(float(row["physical_coverage"]) for row in tests)
        diagnostic_tests = diagnostics["summary"]["test"]
        diagnostic_mean = statistics.fmean(float(row["physical_coverage"]) for row in diagnostic_tests)
        payload_tests = [{key: value for key, value in row.items() if key not in {"run_id", "record", "kind", "checkpoint_sha256"}} for row in tests]
        if raw_mean != diagnostic_mean or payload_tests != diagnostic_tests:
            errors.append(f"{folder.name}: diagnostics/raw mismatch")
        checkpoint_hashes = {row["checkpoint_sha256"] for row in tests}
        if checkpoint_hashes != {sha256(folder / "final.pt")}:
            errors.append(f"{folder.name}: test checkpoint hash")
        cells[key] = raw_mean
        scenarios[key] = tuple(row["scenario_id"] for row in tests)

    expected = {(condition, method, seed) for condition in CONDITIONS for method in METHODS for seed in SEEDS}
    if set(cells) != expected:
        errors.append(f"cell mismatch missing={len(expected-set(cells))} extra={len(set(cells)-expected)}")
    reference_scenarios = scenarios.get(("additive", "central_classical", 13001), ())
    if any(value != reference_scenarios for value in scenarios.values()):
        errors.append("test scenario order differs across cells")

    effects = {
        condition: [cells[(condition, "eqmarl", seed)] - cells[(condition, "quantum_noent", seed)] for seed in SEEDS]
        for condition in CONDITIONS
    }
    interactions = np.asarray([effects["high"][i] - effects["additive"][i] for i in range(len(SEEDS))])
    rng = np.random.default_rng(14001)
    draws = interactions[rng.integers(0, len(interactions), size=(10000, len(interactions)))].mean(axis=1)
    reconstructed = {
        "estimate": float(interactions.mean()),
        "interval_95": [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))],
        "seed_interactions": interactions.tolist(),
    }
    published = json.loads((ROOT / "analysis/13/confirmatory-analysis.json").read_text())["primary"]
    exact_match = reconstructed["estimate"] == published["estimate"] and reconstructed["interval_95"] == published["interval_95"] and reconstructed["seed_interactions"] == published["seed_interactions"]
    if not exact_match:
        errors.append("primary analysis differs")

    output = {
        "audit_version": "stage13-independent-raw-v1",
        "imports_prior_analysis_code": False,
        "bundles": len(folders),
        "cells": len(cells),
        "raw_test_rows": len(cells) * 100,
        "artifacts_hash_checked": checked_artifacts,
        "paired_scenario_order_identical": not any("scenario order" in error for error in errors),
        "reconstructed": reconstructed,
        "published_exact_match": exact_match,
        "errors": errors,
        "passed": not errors,
    }
    destination = ROOT / "analysis/15/stage13-independent-audit.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(output, indent=2) + "\n")
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
