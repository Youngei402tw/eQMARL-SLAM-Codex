#!/usr/bin/env python3
"""Generate the Stage 31 fresh sealed test split (qip_v2) with collision-free seeds.

The v1 generator derived catalog seeds as split_seed + 100 + offset, which collided across splits and
leaked public dynamics into two v1 test catalogs. Here every seed is derived from a SHA-256 of a unique
label, and generation fails if any v2 catalog repeats the transition matrix, active-edge pattern, or
urgencies of any v1 catalog, or if any v2 request type repeats a v1 (predicate, weights) pair.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from generate_qip_splits import catalog, request_types, write_json  # noqa: E402

OUT = ROOT / "configs" / "data" / "qip_v2"
SPECS = [("a4-complete", 4, "complete", 10, 6), ("a5-ring", 5, "ring", 16, 6),
         ("a8-sparse", 8, "sparse_random", 24, 10), ("a6-path-h24", 6, "path", 24, 8)]


def label_seed(label: str) -> int:
    return int(hashlib.sha256(f"bellroute-qip-v2/{label}".encode()).hexdigest()[:8], 16)


def dynamics_key(c: dict):
    return (np.round(np.asarray(c["transition"]), 12).tobytes(),
            tuple(tuple((r["edge_index"], r["urgency"]) for r in s["requests"]) for s in c["states"]))


def main() -> None:
    v1 = [json.loads(p.read_text()) for p in sorted((ROOT / "configs/data/qip_v1").glob("*-catalog.json"))]
    v1_types = {json.dumps([t["predicate"], t["weights"]]) for p in sorted((ROOT / "configs/data/qip_v1").glob("*-request-types.json"))
                for t in json.loads(p.read_text())["types"]}
    requests = request_types("v2test", label_seed("test/request-types"), 16)
    for t in requests["types"]:
        t["type_id"] = t["type_id"].replace("qip-v2test-", "qip2-test-")
        if json.dumps([t["predicate"], t["weights"]]) in v1_types:
            raise SystemExit(f"request type collides with v1: {t['type_id']}")
    requests["split"] = "test"
    request_hash = write_json(OUT / "test-request-types.json", requests)
    type_ids = [t["type_id"] for t in requests["types"]]
    seen = {dynamics_key(c) for c in v1}
    catalogs = []
    for name, agents, topology, horizon, budget in SPECS:
        payload = catalog("test", name, label_seed(f"test/catalog/{name}"), type_ids, agents, topology, horizon, budget)
        payload["catalog_id"] = f"qip2-test-{name}"; payload["schema_version"] = "qip-public-catalog-v2"
        key = dynamics_key(payload)
        if key in seen:
            raise SystemExit(f"catalog dynamics collide: {name}")
        seen.add(key)
        path = OUT / f"test-{name}-catalog.json"
        catalogs.append({"catalog_id": payload["catalog_id"], "path": str(path.relative_to(ROOT)), "sha256": write_json(path, payload)})
    lock = {"schema_version": "qip-split-lock-v2", "protocol_id": "bellroute-harder-variants-v1",
            "note": "Fresh Stage-31 sealed test split; development and validation reuse qip_v1.",
            "test": {"request_types": {"path": "configs/data/qip_v2/test-request-types.json", "sha256": request_hash}, "catalogs": catalogs}}
    write_json(OUT / "split-lock.json", lock)
    print(json.dumps(lock, indent=1))


if __name__ == "__main__":
    main()
