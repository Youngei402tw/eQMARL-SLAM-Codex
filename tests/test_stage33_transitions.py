"""Correctness checks for the Stage 33 transition-matrix variants (development data only)."""
import json
from pathlib import Path

import numpy as np

from eqmarl_slam.diagnostics33 import transform

ROOT = Path(__file__).resolve().parents[1]
FREEZE = json.loads((ROOT / "experiments/27/27-qip-robustness-scaling_attempt-01/stage28-freeze.json").read_text())
CERTS = {r["type_id"]: r for r in json.loads((ROOT / FREEZE["non_test_certificate_bundle"]["path"]).read_text())["certificates"]}
DEV = [json.loads((ROOT / r["path"]).read_text()) for r in FREEZE["training_catalogs"]["development"]]
CONFIG = json.loads((ROOT / "configs/stage33/33-transition-robustness_attempt-01.yaml").read_text())


def test_variants_are_stochastic_and_keep_state_table():
    for catalog in DEV[:3]:
        for spec in CONFIG["variants"]:
            out = transform(catalog, CERTS, spec)
            P = np.asarray(out["transition"])
            assert np.allclose(P.sum(1), 1.0) and (P >= 0).all()
            assert out["states"] == catalog["states"] and out["horizon"] == catalog["horizon"]


def test_variant_shapes():
    catalog = DEV[0]
    P = np.asarray(catalog["transition"])
    assert np.allclose(transform(catalog, CERTS, {"name": "g", "kind": "as_generated"})["transition"], P)
    for stay in (0.05, 0.9):
        assert np.allclose(np.diag(transform(catalog, CERTS, {"name": "s", "kind": "stay", "stay": stay})["transition"]), stay)
    succ = np.asarray(transform(catalog, CERTS, {"name": "x", "kind": "successor", "strength": 0.8})["transition"])
    assert (succ.max(1) >= 0.8).all() and np.allclose(succ, transform(catalog, CERTS, {"name": "x", "kind": "successor", "strength": 0.8})["transition"])
    jack = np.asarray(transform(catalog, CERTS, {"name": "j", "kind": "rare_jackpot", "entry_scale": 0.08, "jackpot_stay": 0.5})["transition"])
    j = int(np.argmax(np.diag(jack) == 0.5))
    assert jack[j, j] == 0.5 and np.delete(jack[:, j], j).max() < np.delete(P[:, j], j).max()


def test_config_never_names_test_data():
    text = json.dumps(CONFIG)
    assert "test_catalogs" not in text and CONFIG["tuning_seed_start"] + CONFIG["tuning_sequences"] <= CONFIG["evaluation_seed_start"]
