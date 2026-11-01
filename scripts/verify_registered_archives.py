#!/usr/bin/env python3
"""Validate reportable bundles while retaining indexed archive-validation failures."""
from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eqmarl_slam.archive import validate_bundle


def main() -> int:
    checked = skipped = 0
    for manifest in sorted((ROOT / "experiments").glob("[0-9][0-9]/*/manifest.json")):
        bundle = manifest.parent
        index_path = bundle.parent / "index.json"
        if not index_path.is_file():
            raise RuntimeError(f"Bundle has no stage index: {bundle.relative_to(ROOT)}")
        index = json.loads(index_path.read_text())
        row = next((item for item in index["runs"] if item["run_id"] == bundle.name), None)
        if row is None:
            raise RuntimeError(f"Bundle is absent from stage index: {bundle.relative_to(ROOT)}")
        if row["status"] == "failed_archive_validation":
            if not row.get("failure_reason") or not (bundle / "diagnostics.json").is_file() or not (bundle / "metrics.jsonl").is_file():
                raise RuntimeError(f"Retained archive failure lacks evidence: {bundle.relative_to(ROOT)}")
            skipped += 1
            continue
        validate_bundle(bundle, repository_root=ROOT)
        checked += 1
    print(f"Archive verification PASSED: {checked} valid bundles; {skipped} indexed archive-validation failures retained.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
