#!/usr/bin/env python3
"""Read-only frozen-policy replay; no training, source edits, or archive overwrite."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from eqmarl_slam.archive import validate_bundle
from eqmarl_slam.classical import SharedGRUActor
from eqmarl_slam.policy_evaluation import evaluate_actor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    args = parser.parse_args()
    root = Path(args.root).resolve()
    plan = json.loads((root / "configs/stage08/analysis-plan.json").read_text())
    method = plan["methods"][0]
    seed = plan["seeds"][0]
    folder = root / f"experiments/08/08-confirmatory-{method}_seed-{seed}_attempt-01"
    integrity = validate_bundle(folder, repository_root=root)
    cfg = json.loads((folder / "resolved_config.yaml").read_text())
    checkpoint = folder / "final.pt"
    before = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    actor = SharedGRUActor(27, cfg["actor_hidden"], 5).double()
    actor.load_state_dict(
        {
            key.removeprefix("actor."): value
            for key, value in payload["model"].items()
            if key.startswith("actor.")
        }
    )
    actor.eval()
    torch.set_num_threads(1)
    resources = {"primitive_steps": 0, "decisions": 0}
    scenario = cfg["validation_scenarios"][0]
    actual = evaluate_actor(actor, cfg["task"], scenario, resources)
    expected = next(
        json.loads(line)
        for line in (folder / "metrics.jsonl").read_text().splitlines()
        if json.loads(line)["kind"] == "validation_episode"
        and json.loads(line)["scenario_id"] == scenario["scenario_id"]
    )
    fields = [
        "physical_coverage",
        "fixed_landmark_accuracy",
        "pose_error_m",
        "trajectory_ate_m",
        "return",
        "communication_bytes",
        "primitive_steps",
    ]
    errors = {key: abs(float(actual[key]) - float(expected[key])) for key in fields}
    passed = (
        max(errors.values()) < 1e-8
        and before == hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    )
    print(
        json.dumps(
            {
                "passed": passed,
                "run_id_replayed": folder.name,
                "checkpoint_sha256": before,
                "integrity": integrity,
                "absolute_errors": errors,
                "resources": resources,
                "optimizer_updates": 0,
                "critic_executions": 0,
            },
            indent=2,
        )
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
