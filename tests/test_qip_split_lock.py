import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_qip_split_lock_hashes_and_disjoint_request_ids():
    lock = json.loads((ROOT / "configs/data/qip_v1/split-lock.json").read_text())
    seen = set()
    for split, entry in lock["splits"].items():
        request_ref = entry["request_types"]
        request_path = ROOT / request_ref["path"]
        assert hashlib.sha256(request_path.read_bytes()).hexdigest() == request_ref["sha256"]
        payload = json.loads(request_path.read_text())
        ids = {row["type_id"] for row in payload["types"]}
        assert not (seen & ids)
        seen |= ids
        for catalog_ref in entry["catalogs"]:
            path = ROOT / catalog_ref["path"]
            assert hashlib.sha256(path.read_bytes()).hexdigest() == catalog_ref["sha256"]
            catalog = json.loads(path.read_text())
            assert catalog["split"] == split
            assert 3 <= catalog["agents"] <= 8
            assert catalog["maximum_simultaneous_pairs"] == 2
            assert all(request["type_id"] in ids for state in catalog["states"] for request in state["requests"])


def test_qip_catalog_edges_are_valid_and_connected():
    lock = json.loads((ROOT / "configs/data/qip_v1/split-lock.json").read_text())
    for entry in lock["splits"].values():
        for reference in entry["catalogs"]:
            catalog = json.loads((ROOT / reference["path"]).read_text())
            edges = [tuple(edge) for edge in catalog["edges"]]
            assert len(edges) == len(set(edges))
            assert all(0 <= a < b < catalog["agents"] for a, b in edges)
            reached = {0}
            while True:
                expanded = reached | {b for a, b in edges if a in reached} | {a for a, b in edges if b in reached}
                if expanded == reached:
                    break
                reached = expanded
            assert len(reached) == catalog["agents"]
