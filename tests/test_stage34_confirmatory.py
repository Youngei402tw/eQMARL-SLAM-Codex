"""Stage 34 runner pieces, checked on development data only (the sealed split is never read here)."""
import json
import tempfile
from pathlib import Path

import numpy as np

from eqmarl_slam import diagnostics34 as d
from eqmarl_slam.robust_routing import RobustRoutingMDP
from eqmarl_slam.routing_learners import evaluate_model

ROOT = Path(__file__).resolve().parents[1]
FREEZE = json.loads((ROOT / "experiments/27/27-qip-robustness-scaling_attempt-01/stage28-freeze.json").read_text())
CERTS = {r["type_id"]: r for r in json.loads((ROOT / FREEZE["non_test_certificate_bundle"]["path"]).read_text())["certificates"]}
DEV = [RobustRoutingMDP(json.loads((ROOT / r["path"]).read_text()), CERTS) for r in FREEZE["training_catalogs"]["development"]]


def test_saved_models_reload_to_identical_decisions():
    with tempfile.TemporaryDirectory() as tmp:
        for method in ("ppo", "dqn"):
            out = Path(tmp) / f"{method}.pt"
            d.train_job({"method": method, "config": {"hidden": 16, "learning_rate": 1e-3, "episodes": 8}, "seed": 3,
                         "maps": DEV[:2], "out": out})
            model = d.load_model(out)
            a = evaluate_model(model, method, DEV[0], [0, 1, 2])
            b = evaluate_model(d.load_model(out), method, DEV[0], [0, 1, 2])
            assert a == b and all(np.isfinite(a))


def test_bid_price_tuning_picks_best_row():
    rows, best = d.tune_bid_price(DEV[:4], DEV[4:6], {"alpha": [0.75, 1.5], "prior_weight": [0, 80]}, [0, 1, 2, 3])
    assert len(rows) == 4 and best["mean"] == max(r["mean"] for r in rows)


def test_config_reads_sealed_split_only_through_lock():
    config = json.loads((ROOT / "configs/stage34/34-unified-confirmatory_attempt-01.yaml").read_text())
    assert config["test_split_lock"]["path"] == "configs/data/qip_v2/split-lock.json"
    assert "test_catalogs" not in json.dumps(config)
    assert config["validation_seed_start"] + config["validation_sequences"] <= config["evaluation_seed_start"]
