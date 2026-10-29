"""Source-verified, bounded diagnostic execution and immutable local archives."""
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import resource
import re
import shutil
import subprocess
import sys
import time
import traceback

import yaml
from .archive import sha256_file, validate_bundle

ROOT = Path(__file__).resolve().parents[1]


def load_config(path):
    """Use JSON numeric semantics for JSON-formatted YAML documents."""
    content = Path(path).read_text()
    return json.loads(content) if content.lstrip().startswith("{") else yaml.safe_load(content)


def write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False, default=numeric_json) + "\n")


def numeric_json(value):
    import numpy as np
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Unsupported JSON type: {type(value).__name__}")


def git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def runtime_manifest(lock="requirements.lock"):
    return {"python": sys.version, "lockfile_sha256": sha256_file(ROOT / lock),
            "platform": platform.platform(),
            "hardware": {"device": "CPU", "available_cpus": len(os.sched_getaffinity(0)),
                         "max_job_threads": 1},
            "dependencies": {dist.metadata["Name"]: dist.version for dist in importlib.metadata.distributions()}}


def preflight_stage(stage, profile, *, allow_detached=False):
    state = json.loads((ROOT / "project_state.json").read_text())
    stages = json.loads((ROOT / "stages.json").read_text())["stages"]
    entry = next(s for s in stages if s["id"] == stage)
    branch = git("branch", "--show-current")
    if branch != entry["branch"] and not (allow_detached and not branch and state["current_branch"] == entry["branch"]):
        raise ValueError("Current branch differs from registered stage branch")
    if state["active_stage"] != stage:
        raise ValueError("Project active stage mismatch")
    for dependency in entry["depends_on"]:
        dep = next(s for s in stages if s["id"] == dependency)
        if dep.get("status") not in ("passed", "completed_null_result") or not dep.get("completion_commit"):
            raise ValueError(f"Dependency {dependency} has no passed completion identity")
        if subprocess.run(["git", "merge-base", "--is-ancestor", dep["completion_commit"], "HEAD"], cwd=ROOT).returncode:
            raise ValueError("Recorded dependency is not an ancestor")
    return {"passed": True, "stage": stage, "profile": profile,
            "source_commit": git("rev-parse", "HEAD"), "dirty_tree": bool(git("status", "--porcelain")),
            "dependency_commit": state["dependency_commit"], "runtime": runtime_manifest(),
            "stage_wall_cap_seconds": 86400}


def _cpu_seconds():
    return sum(getattr(resource.getrusage(who), field)
               for who in (resource.RUSAGE_SELF, resource.RUSAGE_CHILDREN)
               for field in ("ru_utime", "ru_stime"))


def execute(config_path, source_receipt):
    config_path = Path(config_path).resolve()
    config = load_config(config_path)
    stage, run_id = config["stage"], config["run_id"]
    identity_pattern = json.loads((ROOT/'schemas/run_manifest.schema.json').read_text())['properties']['run_id']['pattern']
    if not re.fullmatch(identity_pattern, run_id):
        raise ValueError('Run ID violates archive schema; no experiment started')
    receipt = json.loads(Path(source_receipt).read_text())
    preflight_stage(stage, config["profile"], allow_detached=bool(receipt.get("allow_detached")))
    if git("status", "--porcelain"):
        raise ValueError("Reportable execution requires a clean committed tree")
    source = git("rev-parse", "HEAD")
    actual_branch = git("branch", "--show-current") or json.loads((ROOT/"project_state.json").read_text())["current_branch"]
    if receipt.get("source_commit") != source or receipt.get("branch") != actual_branch or not receipt.get("verified"):
        raise ValueError("Published source receipt does not identify the clean current source")
    index = json.loads((ROOT / "experiments" / stage / "index.json").read_text())
    planned = next((r for r in index["runs"] if r["run_id"] == run_id), None)
    if not planned or planned["status"] != "planned" or planned["config_sha256"] != sha256_file(config_path):
        raise ValueError("Run was not preregistered with this exact config")
    output_root = Path(config.get("artifact_output_root", str(ROOT/"experiments"))).resolve()
    if not output_root.is_relative_to(Path("/workspace")):
        raise ValueError("Artifact output root must remain in the authorized workspace")
    prior = {}
    for folder in {ROOT/"experiments"/stage, output_root/stage}:
        for p in folder.glob("*/manifest.json"):
            item = json.loads(p.read_text()); prior[item["run_id"]] = item
    previous_wall = sum(item["resources_consumed"]["wall_seconds"] for item in prior.values())
    state = json.loads((ROOT / "project_state.json").read_text())
    previous_wall += state.get("stage_budget", {}).get("setup_wall_seconds_charged", 0)
    wall_limit = min(config["wall_seconds"], 86400 - previous_wall)
    if wall_limit <= 0:
        raise ValueError("Cumulative stage budget exhausted before execution")
    output = output_root / stage / run_id
    output.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(config_path, output / "resolved_config.yaml")
    shutil.copyfile(source_receipt, output / "source_publication.json")
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[key] = "1"
    # Constrain native backends as well as Python-level worker counts.
    if hasattr(os, "sched_setaffinity"):
        available = sorted(os.sched_getaffinity(0))
        os.sched_setaffinity(0, set(available[:min(8, int(config.get("max_cpu_threads", 1)))]))
    started_utc = datetime.now(timezone.utc).isoformat()
    started, cpu_start = time.perf_counter(), _cpu_seconds()
    rows = 0
    metrics_path = output / "metrics.jsonl"
    metrics_path.touch()

    def emit(row):
        nonlocal rows
        if time.perf_counter() - started > wall_limit:
            raise TimeoutError("Registered wall cap reached")
        with metrics_path.open("a") as stream:
            stream.write(json.dumps({"run_id": run_id, "record": rows, **row}, allow_nan=False, default=numeric_json) + "\n")
            stream.flush()
        rows += 1

    diagnostics, failure, progress = {}, None, {}
    status = "completed"
    try:
        if config["experiment"] == "reference-circuit":
            from .diagnostics01 import run_diagnostics
            diagnostics = run_diagnostics(config, emit, progress=progress)
        elif config["experiment"] == "model-contracts":
            from .diagnostics_contracts import run_contract_diagnostics
            diagnostics = run_contract_diagnostics(config, output, emit)
        elif config["experiment"] == "exploration-controls":
            from .diagnostics02 import run_diagnostics
            diagnostics = run_diagnostics(config, output, emit, progress=progress)
        elif config["experiment"] == "belief-slam":
            from .diagnostics03 import run_diagnostics
            diagnostics = run_diagnostics(config, output, emit, progress=progress)
        elif config["experiment"] == "controller-preflight":
            from .diagnostics04 import run_diagnostics
            diagnostics = run_diagnostics(config, output, emit, progress=progress)
        elif config["experiment"] == "controller-training":
            from .mappo import train
            diagnostics = train(config, output, emit, progress=progress)
        elif config["experiment"] == "critic-fitting":
            from .diagnostics05 import run_diagnostics
            diagnostics = run_diagnostics(config, output, emit, progress=progress)
        elif config["experiment"] == "critic-training":
            from .critics import train_critic
            diagnostics = train_critic(config, output, emit, progress=progress)
        elif config["experiment"] == "controller-replay":
            from .diagnostics_replay import run_diagnostics
            diagnostics = run_diagnostics(config, output, emit, progress=progress)
        elif config["experiment"] == "interaction-study":
            from .diagnostics06 import run_diagnostics
            diagnostics = run_diagnostics(config, output, emit, progress=progress)
        elif config["experiment"] == "shot-noise":
            from .diagnostics07 import run_diagnostics
            diagnostics = run_diagnostics(config, output, emit, progress=progress)
        elif config["experiment"] == "confirmatory-training":
            from .confirmatory import train_and_test
            diagnostics = train_and_test(config, output, emit, progress=progress)
        elif config["experiment"] == "release-audit":
            from .diagnostics09 import run_diagnostics
            diagnostics = run_diagnostics(config, output, emit, progress=progress)
        elif config["experiment"] == "mechanism-task-diagnostic":
            from .diagnostics11 import run_diagnostics
            diagnostics = run_diagnostics(config, output, emit, progress=progress)
        elif config["experiment"] == "mechanism-release-audit":
            from .diagnostics14 import run_diagnostics
            diagnostics = run_diagnostics(config, output, emit, progress=progress)
        elif config["experiment"] == "nonlocal-correctness":
            from .diagnostics16 import run_diagnostics
            diagnostics = run_diagnostics(config, output, emit, progress=progress)
        elif config["experiment"] == "nonlocal-development":
            from .diagnostics17 import run_diagnostics
            diagnostics = run_diagnostics(config, output, emit, progress=progress)
        elif config["experiment"] == "nonlocal-confirmatory":
            from .diagnostics18 import run_diagnostics
            diagnostics = run_diagnostics(config, output, emit, progress=progress)
        elif config["experiment"] == "legacy-reference":
            command = [config["interpreter"], str(ROOT / "eqmarl_slam/reference_probe.py"),
                       "--fixtures", str(ROOT / config["fixtures"]),
                       "--output", str(output / "legacy_results.json")]
            with (output / "execution.log").open("w") as log:
                process = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=wall_limit,
                                         env={**os.environ, "TF_NUM_INTRAOP_THREADS": "1", "TF_NUM_INTEROP_THREADS": "1"})
            if (output / "legacy_results.json").exists():
                diagnostics = json.loads((output / "legacy_results.json").read_text())
                for row in diagnostics.get("cases", []):
                    emit(row)
            if process.returncode:
                raise RuntimeError(f"Legacy reference returned {process.returncode}; see execution.log")
        else:
            raise ValueError("Unknown registered experiment")
        if diagnostics.get("resources", {}).get("primitive_steps", 0) > config.get("max_primitive_steps", 0):
            raise TimeoutError("Registered primitive-step cap exceeded")
        if not diagnostics.get("passed"):
            status, failure = "failed", {"reason": "One or more registered correctness gates failed", "attempt": 1}
    except Exception as exc:
        status = "stopped_budget" if isinstance(exc, (TimeoutError, subprocess.TimeoutExpired)) else "failed"
        failure = {"reason": str(exc), "type": type(exc).__name__, "traceback": "failure.txt", "attempt": 1}
        (output / "failure.txt").write_text(traceback.format_exc())
        diagnostics.update(passed=False, exception=failure)
        if progress.get("resources"):
            diagnostics["resources"] = progress["resources"]
    resources = diagnostics.get("resources", {})
    consumed = {"primitive_steps": resources.get("primitive_steps", 0),
                "decisions": resources.get("decisions", 0),
                "updates": resources.get("updates", resources.get("synthetic_optimization_updates", 0)),
                "wall_seconds": time.perf_counter() - started,
                "cpu_core_hours": (_cpu_seconds() - cpu_start) / 3600,
                "gpu_device_hours": 0, "circuit_evaluations": diagnostics.get("resources", {}).get("circuit_evaluations", 0),
                "shots": resources.get("shots", 0)}
    runtime = runtime_manifest()
    if config["experiment"] == "legacy-reference":
        runtime = {"python": diagnostics.get("python", "legacy child unavailable; see execution log"),
                   "lockfile_sha256": sha256_file(ROOT / "configs/stage01/legacy/requirements-legacy.lock"),
                   "platform": diagnostics.get("platform", platform.platform()),
                   "hardware": runtime["hardware"], "dependencies": diagnostics.get("packages", {}),
                   "driver_runtime": runtime}
    if config["experiment"] == "belief-slam":
        runtime["estimator_backend_runtime"] = json.loads(
            (ROOT / "configs/stage03/runtime/environment-primary.json").read_text())
    manifest = {
        "schema_version": "0.1.0", "run_id": run_id, "stage": stage, "protocol_id": config["protocol_id"],
        "hypothesis": config["hypothesis"], "method": config["method"], "status": status,
        "source_commit": source, "dependency_completion_commit": config["dependency_completion_commit"],
        "dirty_tree": False, "config_sha256": sha256_file(output / "resolved_config.yaml"),
        "seed_streams": config["seed_streams"], "split_manifests": config["split_manifests"],
        "runtime": runtime, "model": config["model"], "quantum": config["quantum"],
        "budget": {"profile": config["profile"], "primitive_steps": config.get("max_primitive_steps", 0), "wall_seconds": wall_limit},
        "resources_consumed": consumed,
        "execution_segments": [{"started_utc": started_utc, "ended_utc": datetime.now(timezone.utc).isoformat(), **consumed}],
        "evaluation": {"policy_mode": config.get("action_modes"),
                       "checkpoint_selection": config.get('checkpoint_selection', config.get('fixed_task_conventions',{}).get('checkpoint_rule','fixture_only_no_policy_selection')),
                       "metrics_protocol": config["protocol_id"]},
        "artifacts": "artifact_manifest.json", "failure": failure,
        "publication": {"status": "pending", "source_remote_verified": True}}
    write_json(output / "manifest.json", manifest)
    write_json(output / "diagnostics.json", diagnostics)
    write_json(output / "summary.json", {
        "run_id": run_id, "source_commit": source, "config_sha256": manifest["config_sha256"],
        "metrics_sha256": sha256_file(metrics_path), "metrics_rows": rows,
        "analysis_version": config.get('analysis_version','bundle-summary-v1'), "passed": diagnostics.get("passed", False),
        "gates": diagnostics.get("gates", {}), "resources_consumed": consumed,
        "interpretation": config.get("interpretation", "Numerical and runtime correctness only; no training or task-performance claim."),
        "analysis": diagnostics.get("summary", {})})
    (output / "report.md").write_text(
        f"# {run_id}\n\nHypothesis: {config['hypothesis']}\n\n"
        f"Outcome: {status}. Gates passed: {diagnostics.get('passed', False)}.\n\n"
        f"Raw records and diagnostics are authoritative. {config.get('interpretation', 'Fixed numerical diagnostic; no training or policy-performance claim.')} "
        "No physical quantum shots were sampled. Failed outcomes remain archived.\n\n"
        f"Reproduce from source {source} with the committed config after publishing a fresh source receipt; "
        "use a new attempt ID to retain this bundle. Verify with `python -m eqmarl_slam.cli verify --run "
        f"experiments/{stage}/{run_id}`.\n")
    inventory = [{"path": str(p.relative_to(output)), "size_bytes": p.stat().st_size,
                  "sha256": sha256_file(p), "purpose": p.name,
                  "retention": "permanent_git_history"} for p in sorted(output.rglob("*")) if p.is_file()]
    write_json(output / "artifact_manifest.json", {"artifacts": inventory})
    validation = validate_bundle(output, expected_source_commit=source, repository_root=ROOT)
    return {"passed": status == "completed", "run": str(output),
            "status": status, "validation": validation, "resources_consumed": consumed}
