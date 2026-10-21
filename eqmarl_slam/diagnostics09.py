"""Fresh-runtime frozen-policy replay and deterministic raw-only regeneration."""

import hashlib
import json
from pathlib import Path

import torch

from .archive import validate_bundle
from .confirmatory import evaluate_actor
from .critics import make_critic
from .final_analysis import analyze
from .mappo import ActorCritic


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run_diagnostics(config, output, emit, progress=None):
    torch.set_num_threads(1)
    root = Path(__file__).resolve().parents[1]
    output = Path(output)
    resources = {
        "primitive_steps": 0,
        "decisions": 0,
        "updates": 0,
        "circuit_evaluations": 0,
        "shots": 0,
    }
    if progress is not None:
        progress["resources"] = resources
    gates = {}
    checks = []
    index = json.loads((root / "experiments/08/index.json").read_text())
    if "checkpoint_registry_path" in config:
        registry_path = root / config["checkpoint_registry_path"]
        gates["checkpoint_registry_file_hash"] = _sha256(registry_path) == config["checkpoint_registry_sha256"]
        expected_checkpoints = json.loads(registry_path.read_text())["checkpoint_sha256"]
    else:
        expected_checkpoints = config["checkpoint_sha256"]
    indexed_ids = {run["run_id"] for run in index["runs"]}
    gates["checkpoint_registry_is_complete"] = set(expected_checkpoints) == indexed_ids

    for run in index["runs"]:
        folder = root / "experiments/08" / run["run_id"]
        verified = validate_bundle(folder, repository_root=root)
        cfg = json.loads((folder / "resolved_config.yaml").read_text())
        checkpoint = folder / "final.pt"
        actual_checkpoint_sha = _sha256(checkpoint)
        gates[run["run_id"] + "-registered_hash"] = (
            actual_checkpoint_sha == expected_checkpoints.get(run["run_id"])
        )
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        model = ActorCritic(
            critic=make_critic(cfg["critic_method"], cfg["critic_seed"]),
            actor_seed=cfg["actor_seed"],
            critic_seed=cfg["critic_seed"],
            hidden=cfg["actor_hidden"],
        )
        model.load_state_dict(payload["model"])
        model.eval()
        with torch.no_grad():
            obs = torch.zeros((2, 27), dtype=torch.float64)
            dist, _ = model.actor(
                obs,
                model.actor.initial_state(agents=2),
                torch.ones((2, 5), dtype=torch.bool),
            )
        gates[run["run_id"] + "-checkpoint_load"] = bool(
            torch.isfinite(dist.probs).all() and verified["valid"]
        )
        row = {
            "kind": "checkpoint_audit",
            "run_id_audited": run["run_id"],
            "checkpoint_sha256": actual_checkpoint_sha,
            "registered_checkpoint_sha256": expected_checkpoints.get(run["run_id"]),
            "zero_input_probabilities": dist.probs.tolist(),
            "artifact_verification": verified,
        }
        emit(row)
        checks.append(row)

        if cfg["actor_seed"] == config["replay_seed"]:
            scenario = cfg["validation_scenarios"][0]
            actual = evaluate_actor(model.actor, cfg["task"], scenario, resources)
            original = next(
                json.loads(line)
                for line in (folder / "metrics.jsonl").read_text().splitlines()
                if json.loads(line)["kind"] == "validation_episode"
                and json.loads(line)["scenario_id"] == scenario["scenario_id"]
            )
            fields = [
                "physical_coverage",
                "estimated_coverage",
                "fixed_landmark_accuracy",
                "pose_error_m",
                "trajectory_ate_m",
                "trajectory_rpe_m",
                "travel_distance_m",
                "collisions",
                "communication_bytes",
                "messages",
                "success",
                "return",
                "primitive_steps",
            ]
            errors = {key: abs(float(actual[key]) - float(original[key])) for key in fields}
            gates[cfg["critic_method"] + "-fresh_runtime_replay"] = (
                max(errors.values()) < config["replay_atol"]
            )
            emit(
                {
                    "kind": "frozen_validation_replay",
                    "method": cfg["critic_method"],
                    "seed": cfg["actor_seed"],
                    "scenario_id": scenario["scenario_id"],
                    "actual": actual,
                    "original": {key: original[key] for key in fields},
                    "absolute_errors": errors,
                    "optimizer_updates": 0,
                    "critic_executions": 0,
                }
            )

    analyze(root, output / "regenerated-analysis")
    expected = root / "analysis/08/confirmatory-analysis.json"
    actual = output / "regenerated-analysis/confirmatory-analysis.json"
    expected_analysis_sha = _sha256(expected)
    actual_analysis_sha = _sha256(actual)
    gates["registered_analysis_hash_matches"] = (
        expected_analysis_sha == config["analysis_sha256"]
    )
    gates["raw_analysis_regenerates_exactly"] = expected.read_bytes() == actual.read_bytes()
    emit(
        {
            "kind": "analysis_regeneration",
            "expected_sha256": expected_analysis_sha,
            "registered_sha256": config["analysis_sha256"],
            "actual_sha256": actual_analysis_sha,
            "identical": gates["raw_analysis_regenerates_exactly"],
            "training_updates": 0,
        }
    )
    return {
        "passed": all(gates.values()),
        "gates": gates,
        "resources": resources,
        "summary": {
            "checkpoint_count": len(checks),
            "scope": (
                "New dependency environment; all frozen checkpoints loaded, one "
                "validation scenario replayed per method, and final analysis regenerated "
                "from raw data without retraining."
            ),
            "analysis_sha256": actual_analysis_sha,
        },
    }
