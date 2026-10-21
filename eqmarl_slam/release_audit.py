"""Rebuild the release inventory and typed resource ledger from immutable bundles."""

import json
from pathlib import Path

from .archive import validate_bundle


RESOURCE_KEYS = [
    "primitive_steps",
    "decisions",
    "updates",
    "wall_seconds",
    "cpu_core_hours",
    "gpu_device_hours",
    "circuit_evaluations",
    "shots",
]


def audit(root, output):
    root = Path(root)
    output = Path(output)
    output.mkdir(exist_ok=True, parents=True)
    stages = {}
    checks = []
    failures = []
    source_commits = set()
    oversized = []
    for path in sorted((root / "experiments").glob("??/*/manifest.json")):
        manifest = json.loads(path.read_text())
        check = validate_bundle(path.parent, repository_root=root)
        checks.append(check)
        stage = manifest["stage"]
        resources = stages.setdefault(stage, {key: 0 for key in RESOURCE_KEYS})
        for key in RESOURCE_KEYS:
            resources[key] += manifest["resources_consumed"][key]
        source_commits.add(manifest["source_commit"])
        if manifest["status"] != "completed":
            failures.append(
                {
                    "run_id": manifest["run_id"],
                    "status": manifest["status"],
                    "failure": manifest["failure"],
                }
            )
        inventory = json.loads((path.parent / "artifact_manifest.json").read_text())
        for artifact in inventory["artifacts"]:
            if artifact["size_bytes"] > 10_485_760:
                oversized.append({"run_id": manifest["run_id"], **artifact})

    reconciliation = json.loads(
        (root / "configs/stage04/resource-reconciliation-v1.json").read_text()
    )
    for key in ["primitive_steps", "decisions"]:
        stages["04"][key] += reconciliation["reconciled"][key] - reconciliation["reported"][key]

    planned = []
    for path in sorted((root / "experiments").glob("??/index.json")):
        for run in json.loads(path.read_text())["runs"]:
            if run["status"] == "planned":
                planned.append(run["run_id"])

    test_resources = []
    for path in sorted((root / "experiments/08").glob("*/manifest.json")):
        manifest = json.loads(path.read_text())
        diagnostics = json.loads((path.parent / "diagnostics.json").read_text())["summary"]
        rows = [json.loads(line) for line in (path.parent / "metrics.jsonl").read_text().splitlines()]
        test_rows = [row for row in rows if row["kind"] == "test_episode"]
        test_part = {
            "primitive_steps": sum(row["primitive_steps"] for row in test_rows),
            "decisions": 2 * sum(row["macro_decisions"] for row in test_rows),
            "updates": 0,
            "wall_seconds": diagnostics["test_wall_seconds"],
            "cpu_core_hours": diagnostics["test_cpu_core_hours"],
            "gpu_device_hours": 0,
            "circuit_evaluations": 0,
            "shots": 0,
        }
        training_part = {
            key: manifest["resources_consumed"][key] - test_part[key]
            for key in RESOURCE_KEYS
        }
        test_resources.append(
            {
                "run_id": manifest["run_id"],
                "training_validation_and_run_overhead": training_part,
                "frozen_actor_plus_slam_test": test_part,
            }
        )

    result = {
        "analysis_version": "release-raw-inventory-v1",
        "run_count": len(checks),
        "checks": checks,
        "retained_failed_runs": failures,
        "planned_without_outcomes": planned,
        "source_commits": sorted(source_commits),
        "stage_experiment_resources": stages,
        "stage04_reconciliation": reconciliation,
        "confirmatory_phase_resources": test_resources,
        "files_above_review_threshold": oversized,
        "scope": (
            "Measured scientific-run CPU core-hours and GPU device-hours are distinct. "
            "Setup/development active envelopes live in project_state; setup CPU/circuit "
            "costs were not fully instrumented. Simulated shots are not physical hardware "
            "shots. Test time includes actor and SLAM, with no critic or optimizer."
        ),
    }
    (output / "release-audit.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = audit(args.root, args.output)
    print(
        json.dumps(
            {
                "runs": result["run_count"],
                "failed_retained": len(result["retained_failed_runs"]),
                "pending": result["planned_without_outcomes"],
            },
            indent=2,
        )
    )
