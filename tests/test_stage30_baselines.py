"""Correctness checks for the Stage 30 comparators and the Gymnasium wrapper (development data only)."""
import json
from pathlib import Path

import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from eqmarl_slam.bellroute_env import ACTION_INDEX, BellRouteEnv
from eqmarl_slam.robust_routing import RobustRoutingMDP, greedy_action
from eqmarl_slam.routing_baselines import (BidPricePolicy, PlannerPolicy, dp_table, estimate_transition,
                                           prior_transition, reference_slot_gains, weighted_quantile)

ROOT = Path(__file__).resolve().parents[1]
FREEZE = json.loads((ROOT / "experiments/27/27-qip-robustness-scaling_attempt-01/stage28-freeze.json").read_text())


def _dev_mdps():
    certs = {row["type_id"]: row for row in json.loads((ROOT / FREEZE["non_test_certificate_bundle"]["path"]).read_text())["certificates"]}
    return [RobustRoutingMDP(json.loads((ROOT / ref["path"]).read_text()), certs)
            for ref in FREEZE["training_catalogs"]["development"]]


@pytest.fixture(scope="module")
def dev():
    return _dev_mdps()


def test_dp_with_true_transition_matches_oracle(dev):
    for mdp in dev[:3]:
        transition = np.asarray(mdp.catalog["transition"], float)
        assert np.array_equal(dp_table(mdp, transition), mdp.oracle_table())
        planner, table = PlannerPolicy(mdp, transition), mdp.oracle_table()
        for seed in range(5):
            sequence = mdp.sample_sequence(seed)
            ours = mdp.evaluate(sequence, planner, seed=seed)["surplus"]
            ref = mdp.evaluate(sequence, lambda e, s, b, r: e.oracle_action(table, r, s, b), seed=seed)["surplus"]
            assert ours == ref


def test_transition_estimate_is_stochastic_and_converges(dev):
    mdp = dev[0]; n = len(mdp.catalog["states"])
    prior = prior_transition(n, 0.6)
    assert np.allclose(prior.sum(1), 1) and np.allclose(np.diag(prior), 0.6)
    assert np.allclose(estimate_transition(n, [], 0.6, 2.0), prior)
    sequences = [mdp.sample_sequence(seed) for seed in range(3000)]
    estimate = estimate_transition(n, sequences, 0.6, 1.0)
    assert np.allclose(estimate.sum(1), 1)
    assert np.abs(estimate - np.asarray(mdp.catalog["transition"])).max() < 0.05


def test_weighted_quantile():
    values = np.asarray([0.0, 1.0, 2.0, 3.0])
    assert weighted_quantile(values, np.ones(4), 0.5) == 1.0
    assert weighted_quantile(values, np.asarray([0, 0, 0, 1.0]), 0.1) == 3.0
    assert weighted_quantile(values, np.ones(4), 0.0) == 0.0


def test_bid_price_is_feasible_and_reduces_to_positive_greedy_with_ample_budget(dev):
    reference = reference_slot_gains(dev)
    assert len(reference) == 2 * sum(len(m.catalog["states"]) for m in dev)
    for mdp in dev:
        for seed in range(10):
            result = mdp.evaluate(mdp.sample_sequence(seed), BidPricePolicy(reference, 1.0, 10.0), seed=seed)
            assert not result["violations"] and result["pairs_unused"] >= 0
    mdp = dev[0]
    policy = BidPricePolicy(reference, 1e6, 0.0)  # share >= 1 => threshold 0
    for state in range(len(mdp.catalog["states"])):
        chosen = policy(mdp, state, 2, 1)
        assert mdp.action_gain(state, chosen) == pytest.approx(mdp.action_gain(state, greedy_action(mdp, state, 2, 1)))
        assert all(mdp.edge_gains(state)[e] > 0 for e in chosen)


def test_bid_price_threshold_rises_as_budget_tightens(dev):
    policy = BidPricePolicy(reference_slot_gains(dev), 1.0, 10.0)
    assert policy.threshold(1, 20) >= policy.threshold(10, 20) >= policy.threshold(40, 20) == 0.0


def test_gym_env_passes_checker_and_reproduces_evaluator(dev):
    mdp = dev[3]
    check_env(BellRouteEnv(mdp.catalog, mdp.certificates, invalid_action="noop"), skip_render_check=True)
    env = BellRouteEnv(mdp.catalog, mdp.certificates)
    for seed in range(5):
        obs, info = env.reset(seed=seed)
        total, done, steps = 0.0, False, 0
        while not done:
            state, budget, remaining = info["public_state"], int(obs["budget"][0]), int(obs["remaining"][0])
            action = ACTION_INDEX[tuple(sorted(greedy_action(mdp, state, budget, remaining)))]
            assert info["action_mask"][action] == 1
            obs, reward, done, truncated, info = env.step(action)
            total += reward; steps += 1
        reference = mdp.evaluate(mdp.sample_sequence(seed), greedy_action, seed=seed + 810000)["surplus"]
        assert steps == mdp.catalog["horizon"] and total == pytest.approx(reference, abs=1e-12)


def test_gym_env_rejects_infeasible_action(dev):
    env = BellRouteEnv(dev[0].catalog, dev[0].certificates)
    _, info = env.reset(seed=0)
    infeasible = int(np.flatnonzero(info["action_mask"] == 0)[0])
    with pytest.raises(ValueError):
        env.step(infeasible)
