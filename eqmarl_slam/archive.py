"""Validate durable experiment bundles without importing learning backends.

Hashes identify byte-for-byte artifacts. Validation establishes local integrity;
remote availability must be checked separately and recorded in a publication receipt.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import jsonschema
import yaml


class ArchiveValidationError(ValueError):
    """A run bundle is incomplete or inconsistent."""


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _fail(message: str) -> None:
    raise ArchiveValidationError(message)


def _safe_file(root: Path, relative: Any) -> Path:
    if not isinstance(relative, str) or not relative:
        _fail("Artifact path must be a nonempty relative path")
    candidate = Path(relative)
    if candidate.is_absolute() or ".." in candidate.parts or ":" in relative or "\\" in relative:
        _fail(f"Unsafe artifact path: {relative!r}")
    resolved = (root / candidate).resolve()
    if not resolved.is_relative_to(root.resolve()):
        _fail(f"Artifact escapes its root: {relative!r}")
    if not resolved.is_file():
        _fail(f"Missing required artifact: {relative}")
    return resolved


def _json(path: Path) -> Any:
    try:
        def invalid_constant(value: str) -> None:
            raise ValueError(f"Nonfinite JSON number: {value}")
        return json.loads(path.read_text(), parse_constant=invalid_constant)
    except (OSError, ValueError) as exc:
        raise ArchiveValidationError(f"Invalid JSON in {path.name}: {exc}") from exc


def _object(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        _fail(f"{label} must be an object")
    return value


def _identity(value: dict[str, Any], manifest: dict[str, Any], label: str, *, required: bool) -> None:
    for key in ("run_id", "source_commit", "config_sha256"):
        if key not in value:
            if required:
                _fail(f"{label} missing {key}")
        elif value[key] != manifest[key]:
            _fail(f"{label} {key} identity mismatch")
    for key in ("stage", "protocol_id"):
        if key in value and value[key] != manifest[key]:
            _fail(f"{label} {key} identity mismatch")


def _repository_root(bundle: Path) -> Path:
    for parent in (bundle, *bundle.parents):
        if (parent / "schemas/run_manifest.schema.json").is_file():
            return parent
    _fail("Cannot find repository schema; provide schema_path and repository_root")


def validate_bundle(
    bundle_dir: str | Path,
    *,
    schema_path: str | Path | None = None,
    expected_source_commit: str | None = None,
    repository_root: str | Path | None = None,
) -> dict[str, Any]:
    """Validate schema, identities, raw data, resources, and every local artifact.

    ``resolved_config.yaml`` must declare run_id, stage, and protocol_id. Summary
    identities include run_id, source_commit, config_sha256, and metrics_sha256.
    The inventory is {"artifacts": [{"path", "size_bytes", "sha256", "purpose",
    "retention"}, ...]}; it lists every required bundle file except itself.
    Split paths are repository-relative and must carry their SHA-256. The optional
    expected source SHA is useful when an external index supplies run provenance.
    No network or training operations are performed.
    """
    bundle = Path(bundle_dir).resolve()
    repo = Path(repository_root).resolve() if repository_root is not None else _repository_root(bundle)
    schema = Path(schema_path) if schema_path is not None else repo / "schemas/run_manifest.schema.json"
    required_files = {
        "manifest.json", "resolved_config.yaml", "metrics.jsonl", "summary.json",
        "diagnostics.json", "artifact_manifest.json", "report.md",
    }
    paths = {name: _safe_file(bundle, name) for name in required_files}
    manifest = _object(_json(paths["manifest.json"]), "manifest")
    try:
        jsonschema.Draft202012Validator(_json(schema)).validate(manifest)
    except jsonschema.ValidationError as exc:
        raise ArchiveValidationError(f"Manifest schema: {exc.message}") from exc
    if expected_source_commit is not None and manifest["source_commit"] != expected_source_commit:
        _fail("Manifest source_commit identity mismatch with external index")
    if manifest["run_id"] != bundle.name:
        _fail("Manifest run_id identity mismatch with bundle directory")
    if not manifest["run_id"].startswith(manifest["stage"] + "-"):
        _fail("Manifest run_id stage prefix mismatch")
    if sha256_file(paths["resolved_config.yaml"]) != manifest["config_sha256"]:
        _fail("Resolved config SHA-256 mismatch")
    try:
        config = _object(yaml.safe_load(paths["resolved_config.yaml"].read_text()), "resolved config")
    except yaml.YAMLError as exc:
        raise ArchiveValidationError(f"Invalid resolved config YAML: {exc}") from exc
    for key in ("run_id", "stage", "protocol_id"):
        if config.get(key) != manifest[key]:
            _fail(f"Resolved config {key} identity mismatch")
    _identity(config, manifest, "resolved config", required=False)
    summary = _object(_json(paths["summary.json"]), "summary")
    diagnostics = _object(_json(paths["diagnostics.json"]), "diagnostics")
    _identity(summary, manifest, "summary", required=True)
    _identity(diagnostics, manifest, "diagnostics", required=False)
    metrics_hash = sha256_file(paths["metrics.jsonl"])
    if summary.get("metrics_sha256") != metrics_hash:
        _fail("Summary metrics_sha256 does not identify raw metrics")
    rows = 0
    for line_number, line in enumerate(paths["metrics.jsonl"].read_text().splitlines(), 1):
        if not line.strip():
            _fail(f"Blank metrics row at line {line_number}")
        try:
            def invalid_constant(value: str) -> None:
                raise ValueError(f"Nonfinite JSON number: {value}")
            row = _object(json.loads(line, parse_constant=invalid_constant), "metrics row")
        except ValueError as exc:
            raise ArchiveValidationError(f"Invalid metrics row {line_number}: {exc}") from exc
        _identity(row, manifest, f"metrics row {line_number}", required=False)
        rows += 1
    if "metrics_rows" in summary and summary["metrics_rows"] != rows:
        _fail("Summary metrics_rows mismatch")
    if manifest["status"] in {"failed", "stopped_budget", "abandoned", "interrupted", "invalidated"}:
        failure = manifest["failure"]
        if not isinstance(failure, dict) or not any(
            isinstance(failure.get(key), str) and failure[key].strip()
            for key in ("reason", "message", "failure_mode")
        ):
            _fail("Unsuccessful run requires a recorded failure reason")
    if manifest["status"] == "completed" and manifest["failure"] is not None:
        _fail("Completed run cannot contain an unresolved failure")
    if manifest["status"] == "completed" and not manifest["publication"]["source_remote_verified"]:
        _fail("Completed run requires remotely verified source provenance")
    for key, value in manifest["resources_consumed"].items():
        if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or value < 0:
            _fail(f"Invalid resource counter: {key}")
    for split, reference in manifest["split_manifests"].items():
        if reference is None:
            continue
        reference = _object(reference, f"{split} split reference")
        split_path = _safe_file(repo, reference.get("path"))
        if sha256_file(split_path) != reference.get("sha256"):
            _fail(f"{split} split SHA-256 mismatch")
        split_manifest = _json(split_path)
        if isinstance(split_manifest, dict):
            if "split" in split_manifest and split_manifest["split"] != split:
                _fail(f"{split} split identity mismatch")
        elif not (isinstance(split_manifest, list) and split_manifest
                  and all(isinstance(fixture, dict) for fixture in split_manifest)):
            _fail(f"{split} split manifest must be an object or a nonempty array of fixture objects")
    inventory = _object(_json(paths["artifact_manifest.json"]), "artifact inventory")
    artifacts = inventory.get("artifacts")
    if not isinstance(artifacts, list):
        _fail("Artifact inventory must contain an artifacts list")
    verified = set()
    for artifact in artifacts:
        artifact = _object(artifact, "artifact")
        relative = artifact.get("path")
        path = _safe_file(bundle, relative)
        if relative in verified:
            _fail(f"Duplicate artifact: {relative}")
        if relative == "artifact_manifest.json":
            _fail("Artifact inventory cannot hash itself")
        if not isinstance(artifact.get("purpose"), str) or not artifact["purpose"].strip():
            _fail(f"Artifact purpose missing: {relative}")
        if not isinstance(artifact.get("retention"), str) or not artifact["retention"].strip():
            _fail(f"Artifact retention missing: {relative}")
        if artifact.get("size_bytes") != path.stat().st_size:
            _fail(f"Artifact byte size mismatch: {relative}")
        if artifact.get("sha256") != sha256_file(path):
            _fail(f"Artifact SHA-256 mismatch: {relative}")
        verified.add(relative)
    missing = required_files - {"artifact_manifest.json"} - verified
    if missing:
        _fail(f"Required artifacts absent from inventory: {', '.join(sorted(missing))}")
    return {
        "valid": True,
        "run_id": manifest["run_id"],
        "status": manifest["status"],
        "verified_files": len(verified),
        "metrics_rows": rows,
        "metrics_sha256": metrics_hash,
    }
