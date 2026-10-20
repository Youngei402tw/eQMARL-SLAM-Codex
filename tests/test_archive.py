"""Integrity checks reject valid-looking bundles with inconsistent evidence."""

import json
from pathlib import Path

import pytest
import yaml

from eqmarl_slam.archive import ArchiveValidationError, sha256_file, validate_bundle


SCHEMA = Path(__file__).resolve().parents[1] / "schemas/run_manifest.schema.json"
SOURCE = "a" * 40


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def refresh_inventory(bundle):
    write_json(bundle / "artifact_manifest.json", {
        "artifacts": [
            {"path": path.name, "size_bytes": path.stat().st_size,
             "sha256": sha256_file(path), "purpose": "test evidence", "retention": "permanent"}
            for path in sorted(bundle.iterdir())
            if path.name != "artifact_manifest.json" and path.is_file()
        ]
    })


@pytest.fixture
def bundle(tmp_path):
    root = tmp_path / "experiments" / "01" / "01-fixture_seed-0001"
    root.mkdir(parents=True)
    write_json(tmp_path / "development.json", {"split": "development", "fixture_ids": ["bell"]})
    config = {"run_id": root.name, "stage": "01", "protocol_id": "reference-circuit-v1", "shots": None}
    (root / "resolved_config.yaml").write_text(yaml.safe_dump(config))
    manifest = {
        "schema_version": "0.1.0", "run_id": root.name, "stage": "01",
        "protocol_id": "reference-circuit-v1", "hypothesis": "Oracle and backend agree",
        "method": "reference-fixture", "status": "completed", "source_commit": SOURCE,
        "dirty_tree": False, "config_sha256": sha256_file(root / "resolved_config.yaml"),
        "seed_streams": {name: None for name in ["model_init", "critic_init", "map", "evaluation_map", "sensor", "action", "quantum_shots", "communication"]},
        "split_manifests": {"train": None, "development": {"path": "development.json", "sha256": sha256_file(tmp_path / "development.json")}, "validation": None, "test": None},
        "runtime": {"python": "3.11", "lockfile_sha256": "b" * 64, "platform": "linux", "hardware": {"cpu": 1}, "dependencies": {}},
        "model": {"actor": None, "encoder": None, "critic": "fixture", "parameter_counts": {}, "optimizer": None},
        "quantum": {"shots": None}, "budget": {"profile": "diagnostic", "primitive_steps": 0, "wall_seconds": 3600},
        "resources_consumed": {name: 0 for name in ["primitive_steps", "decisions", "updates", "wall_seconds", "circuit_evaluations", "shots", "cpu_core_hours", "gpu_device_hours"]},
        "evaluation": {"policy_mode": None, "checkpoint_selection": None, "metrics_protocol": "fixture absolute errors"},
        "artifacts": "artifact_manifest.json", "failure": None,
        "publication": {"status": "pending", "source_remote_verified": True},
    }
    write_json(root / "manifest.json", manifest)
    (root / "metrics.jsonl").write_text(json.dumps({"run_id": root.name, "expectation_error": 0.0}) + "\n")
    write_json(root / "summary.json", {
        "run_id": root.name, "source_commit": SOURCE, "config_sha256": manifest["config_sha256"],
        "metrics_sha256": sha256_file(root / "metrics.jsonl"), "metrics_rows": 1,
    })
    write_json(root / "diagnostics.json", {"run_id": root.name, "passed": True})
    (root / "report.md").write_text("Fixture passed; no performance claim.\n")
    refresh_inventory(root)
    return root


def verify(bundle, **kwargs):
    return validate_bundle(bundle, schema_path=SCHEMA, repository_root=bundle.parents[2], **kwargs)


def edit_json(path, key, value):
    data = json.loads(path.read_text())
    data[key] = value
    write_json(path, data)


def test_valid_bundle(bundle):
    result = verify(bundle, expected_source_commit=SOURCE)
    assert result["valid"] is True
    assert result["metrics_rows"] == 1
    assert result["verified_files"] == 6


def test_tampered_artifact_hash(bundle):
    (bundle / "report.md").write_text("Changed report of equal byte length?\n")
    with pytest.raises(ArchiveValidationError, match="Artifact (byte size|SHA-256) mismatch"):
        verify(bundle)


def test_missing_artifact(bundle):
    (bundle / "diagnostics.json").unlink()
    with pytest.raises(ArchiveValidationError, match="Missing required artifact"):
        verify(bundle)


def test_required_file_omitted_from_inventory(bundle):
    inventory = json.loads((bundle / "artifact_manifest.json").read_text())
    inventory["artifacts"] = [item for item in inventory["artifacts"] if item["path"] != "report.md"]
    write_json(bundle / "artifact_manifest.json", inventory)
    with pytest.raises(ArchiveValidationError, match="Required artifacts absent"):
        verify(bundle)


@pytest.mark.parametrize("filename,key,value", [
    ("summary.json", "run_id", "01-another"),
    ("summary.json", "source_commit", "c" * 40),
    ("diagnostics.json", "run_id", "01-another"),
])
def test_cross_file_identity(bundle, filename, key, value):
    edit_json(bundle / filename, key, value)
    refresh_inventory(bundle)
    with pytest.raises(ArchiveValidationError, match="identity mismatch"):
        verify(bundle)


def test_external_source_identity(bundle):
    with pytest.raises(ArchiveValidationError, match="source_commit identity mismatch"):
        verify(bundle, expected_source_commit="c" * 40)


def test_config_hash(bundle):
    with (bundle / "resolved_config.yaml").open("a") as stream:
        stream.write("layers: 99\n")
    refresh_inventory(bundle)
    with pytest.raises(ArchiveValidationError, match="Resolved config SHA-256 mismatch"):
        verify(bundle)


def test_raw_metrics_summary_hash(bundle):
    with (bundle / "metrics.jsonl").open("a") as stream:
        stream.write('{"expectation_error": 1.0}\n')
    refresh_inventory(bundle)
    with pytest.raises(ArchiveValidationError, match="metrics_sha256"):
        verify(bundle)


def test_schema(bundle):
    manifest = json.loads((bundle / "manifest.json").read_text())
    del manifest["runtime"]
    write_json(bundle / "manifest.json", manifest)
    refresh_inventory(bundle)
    with pytest.raises(ArchiveValidationError, match="Manifest schema"):
        verify(bundle)


def test_split_hash(bundle):
    (bundle.parents[2] / "development.json").write_text('{"split":"development","fixture_ids":[]}')
    with pytest.raises(ArchiveValidationError, match="development split SHA-256 mismatch"):
        verify(bundle)


@pytest.mark.parametrize("path", ["../outside.txt", "/tmp/outside.txt", "https://example.com/outside", "..\\outside.txt"])
def test_unsafe_artifact_path(bundle, path):
    inventory = json.loads((bundle / "artifact_manifest.json").read_text())
    inventory["artifacts"][0]["path"] = path
    write_json(bundle / "artifact_manifest.json", inventory)
    with pytest.raises(ArchiveValidationError, match="Unsafe artifact path"):
        verify(bundle)


def test_symlink_escape(bundle):
    external = bundle.parents[2] / "external.txt"
    external.write_text("outside")
    (bundle / "external.txt").symlink_to(external)
    refresh_inventory(bundle)
    with pytest.raises(ArchiveValidationError, match="Artifact escapes"):
        verify(bundle)


def test_failed_run_requires_reason(bundle):
    edit_json(bundle / "manifest.json", "status", "failed")
    refresh_inventory(bundle)
    with pytest.raises(ArchiveValidationError, match="failure reason"):
        verify(bundle)


def test_failed_run_can_retain_no_metric_rows(bundle):
    edit_json(bundle / "manifest.json", "status", "failed")
    edit_json(bundle / "manifest.json", "failure", {"reason": "backend unavailable", "valid_data": False})
    (bundle / "metrics.jsonl").write_text("")
    edit_json(bundle / "summary.json", "metrics_sha256", sha256_file(bundle / "metrics.jsonl"))
    edit_json(bundle / "summary.json", "metrics_rows", 0)
    refresh_inventory(bundle)
    assert verify(bundle)["status"] == "failed"


def test_nonfinite_counter(bundle):
    manifest = json.loads((bundle / "manifest.json").read_text())
    manifest["resources_consumed"]["wall_seconds"] = -1
    write_json(bundle / "manifest.json", manifest)
    refresh_inventory(bundle)
    with pytest.raises(ArchiveValidationError, match="Invalid resource counter"):
        verify(bundle)
