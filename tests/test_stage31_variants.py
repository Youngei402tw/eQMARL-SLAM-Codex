"""Correctness checks for the Stage 31a variants (development data only)."""
import json
from pathlib import Path

import numpy as np
import pytest

from eqmarl_slam.robust_routing import RobustRoutingMDP, greedy_action
from eqmarl_slam.variants31 import RegimeMDP, StoreBidPrice, StoreMDP, store_greedy, store_reference_gains

ROOT = Path(__file__).resolve().parents[1]
FREEZE = json.loads((ROOT / "experiments/27/27-qip-robustness-scaling_attempt-01/stage28-freeze.json").read_text())
CERTS = {r["type_id"]: r for r in json.loads((ROOT / FREEZE["non_test_certificate_bundle"]["path"]).read_text())["certificates"]}
DEV = [json.loads((ROOT / r["path"]).read_text()) for r in FREEZE["training_catalogs"]["development"]]
VMIN = min((c["classical_value"] - .5) / (c["bell_value"] - .5) for c in CERTS.values() if c["bell_value"] - c["classical_value"] > 1e-12)


def test_store_gains_decay_and_expire():
    m = StoreMDP(DEV[0], CERTS, 3, 0.5, 0.97, VMIN)
    assert m.max_age >= 1 and 0.97 ** m.max_age <= VMIN < 0.97 ** (m.max_age - 1)
    for state, rows in enumerate(m.requests):
        for edge in rows:
            gains = [m.gain(state, edge, a) for a in range(m.max_age + 1)]
            assert all(x >= y - 1e-15 for x, y in zip(gains, gains[1:])) and gains[m.max_age] == 0.0
    assert m.after((0, 2, 5), (2,)) == (1,)  # age 6 expires at max_age 6
    assert m.generate((0, 1, 2)) == (0, 1, 2) and m.generate((1,)) == (0, 1)


def test_store_optimal_dominates_rules_and_matches_dp_value():
    m = StoreMDP(DEV[2], CERTS, 3, 0.5, 0.97, VMIN)
    m.solve(); policy = m.optimal_policy()
    seeds = range(3000)
    opt = np.mean([m.run(s, policy) for s in seeds])
    ref = store_reference_gains([m])
    assert opt >= np.mean([m.run(s, store_greedy(m)) for s in seeds]) - 1e-9
    assert opt >= np.mean([m.run(s, StoreBidPrice(m, ref, 1.0, 20)) for s in seeds]) - 1e-9
    # Monte Carlo mean matches the exact DP value averaged over uniform start states.
    n = len(m.transition)
    expected = np.mean([m.gen_prob * m._post(m.horizon, s, m.generate(m.initial_store()))
                        + (1 - m.gen_prob) * m._post(m.horizon, s, m.initial_store()) for s in range(n)])
    assert opt == pytest.approx(expected, rel=0.03)


def test_regime_with_zero_tilt_reduces_to_v1_dp():
    m = RegimeMDP(DEV[1], CERTS, 0.0)
    m.solve_belief()
    base = RobustRoutingMDP(DEV[1], CERTS)
    table = base.oracle_table()
    for k in (1, m.horizon):
        for s in range(m.n):
            assert np.allclose(m.V[k, s, :, 20], table[k, s], atol=1e-6)


def test_regime_belief_update_and_reference_ordering():
    m = RegimeMDP(DEV[3], CERTS, 2.0)
    assert np.allclose([p.sum(1) for p in m.P], 1.0)
    s, t = 0, int(np.argmax(m.P[1][0] / m.P[0][0]))
    assert m.posterior(0.5, s, t) > 0.5
    m.solve_belief(); act = m.belief_policy()
    seeds = range(400)
    belief = np.mean([m.run(x, lambda st, b, r, pi, z: act(st, b, r, pi)) for x in seeds])
    greedy = np.mean([m.run(x, lambda st, b, r, pi, z: greedy_action(m.base, st, b, r)) for x in seeds])
    assert belief >= greedy


def test_regime_maps_and_learner_wrappers_run():
    from pathlib import Path as _P
    import tempfile
    from eqmarl_slam import diagnostics31b as d
    maps = d.regime_maps(DEV[:1], CERTS, 1.0)
    assert len(maps) == 2 and all(np.allclose(np.asarray(m.catalog["transition"]).sum(1), 1) for m in maps)
    assert not np.allclose(maps[0].catalog["transition"], maps[1].catalog["transition"])
    regime = RegimeMDP(DEV[0], CERTS, 1.0)
    with tempfile.TemporaryDirectory() as tmp:
        for method in ("dqn", "ppo", "recurrent_ppo"):
            result = d._train_job({"method": method, "config": {"name": "t", "hidden": 16, "learning_rate": 1e-3, "episodes": 4},
                                   "seed": 0, "maps": maps, "eval": {"x": regime}, "eval_seeds": [0, 1],
                                   "out": _P(tmp) / f"{method}.pt"})
            assert len(result["scores"]["x"]) == 2 and all(np.isfinite(result["scores"]["x"]))
