#!/usr/bin/env python3
"""Generate the prospectively locked robust-routing data manifests."""
from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "configs" / "data" / "qip_v1"


def write_json(path: Path, payload: dict) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def request_types(split: str, seed: int, count: int) -> dict:
    rng = np.random.default_rng(seed)
    rows = []
    for index in range(count):
        rows.append({
            "type_id": f"qip-{split}-type-{index:03d}",
            "predicate": rng.integers(0, 2, size=(3, 3)).tolist(),
            "weights": rng.integers(1, 5, size=(3, 3)).tolist(),
        })
    return {
        "schema_version": "qip-request-types-v1",
        "split": split,
        "generator_seed": seed,
        "contexts_per_agent": 3,
        "binary_actions": True,
        "types": rows,
    }


def topology_edges(agents: int, topology: str, seed: int) -> list[tuple[int, int]]:
    if topology == "path":
        return [(i, i + 1) for i in range(agents - 1)]
    if topology == "ring":
        return sorted({tuple(sorted((i, (i + 1) % agents))) for i in range(agents)})
    if topology == "star":
        return [(0, i) for i in range(1, agents)]
    if topology == "complete":
        return list(itertools.combinations(range(agents), 2))
    if topology == "sparse_random":
        rng = np.random.default_rng(seed)
        edges = set((i, i + 1) for i in range(agents - 1))
        candidates = [e for e in itertools.combinations(range(agents), 2) if e not in edges]
        rng.shuffle(candidates)
        edges.update(candidates[: max(1, agents // 2)])
        return sorted(edges)
    raise ValueError(topology)


def catalog(split: str, name: str, seed: int, type_ids: list[str], agents: int,
            topology: str, horizon: int, budget: int, states: int = 14) -> dict:
    rng = np.random.default_rng(seed)
    edges = topology_edges(agents, topology, seed + 1)
    rows = []
    for state_id in range(states):
        minimum = min(2, len(edges))
        count = int(rng.integers(minimum, len(edges) + 1))
        active = sorted(rng.choice(len(edges), size=count, replace=False).tolist())
        requests = []
        for edge_index in active:
            requests.append({
                "edge_index": edge_index,
                "agents": list(edges[edge_index]),
                "type_id": str(rng.choice(type_ids)),
                "urgency": int(rng.choice([1, 2, 3, 5], p=[0.45, 0.30, 0.15, 0.10])),
            })
        rows.append({"state_id": state_id, "requests": requests})
    raw = rng.gamma(shape=1.0, scale=1.0, size=(states, states))
    for i in range(states):
        raw[i, i] += 7.0
        raw[i, (i + 1) % states] += 2.5
    transition = raw / raw.sum(axis=1, keepdims=True)
    return {
        "schema_version": "qip-public-catalog-v1",
        "split": split,
        "catalog_id": f"qip-{split}-{name}",
        "generator_seed": seed,
        "agents": agents,
        "topology": topology,
        "edges": [list(e) for e in edges],
        "maximum_simultaneous_pairs": 2,
        "horizon": horizon,
        "bell_pair_budget": budget,
        "states": rows,
        "transition": transition.tolist(),
    }


SPECS = {
    "development": [
        ("a3-path", 3, "path", 8, 2), ("a3-ring", 3, "ring", 10, 4),
        ("a4-complete", 4, "complete", 10, 6), ("a4-path", 4, "path", 16, 6),
        ("a4-ring", 4, "ring", 10, 4), ("a4-star", 4, "star", 12, 4),
        ("a6-path", 6, "path", 16, 6), ("a6-star", 6, "star", 10, 4),
    ],
    "validation": [
        ("a3-ring", 3, "ring", 8, 2), ("a4-sparse", 4, "sparse_random", 12, 4),
        ("a6-path", 6, "path", 16, 6), ("a6-star", 6, "star", 10, 4),
    ],
    "test": [
        ("id-a4-complete", 4, "complete", 10, 6),
        ("topology-a6-ring", 6, "ring", 16, 6),
        ("scale-a8-sparse", 8, "sparse_random", 24, 10),
        ("horizon-a4-path", 4, "path", 24, 6),
    ],
}


def main() -> None:
    seeds = {"development": 25101, "validation": 25102, "test": 25103}
    counts = {"development": 18, "validation": 12, "test": 16}
    manifest = {"schema_version": "qip-split-lock-v1", "protocol_id": "robust-entanglement-routing-v1", "splits": {}}
    for split in ("development", "validation", "test"):
        requests = request_types(split, seeds[split], counts[split])
        request_path = OUT / f"{split}-request-types.json"
        request_hash = write_json(request_path, requests)
        type_ids = [r["type_id"] for r in requests["types"]]
        catalogs = []
        for offset, (name, agents, topology, horizon, budget) in enumerate(SPECS[split]):
            payload = catalog(split, name, seeds[split] + 100 + offset, type_ids, agents, topology, horizon, budget)
            path = OUT / f"{split}-{name}-catalog.json"
            catalogs.append({"catalog_id": payload["catalog_id"], "path": str(path.relative_to(ROOT)), "sha256": write_json(path, payload)})
        manifest["splits"][split] = {
            "request_types": {"path": str(request_path.relative_to(ROOT)), "sha256": request_hash},
            "catalogs": catalogs,
        }
    lock_path = OUT / "split-lock.json"
    write_json(lock_path, manifest)
    print(lock_path.relative_to(ROOT))


if __name__ == "__main__":
    main()
