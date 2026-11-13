import json
from pathlib import Path

import numpy as np
import torch

from eqmarl_slam.robust_routing import RobustRoutingMDP, greedy_action
from eqmarl_slam.routing_learners import ActionScore, action_features, choose_q, train_ppo, train_q

ROOT = Path(__file__).resolve().parents[1]


def fixture():
    catalog = json.loads((ROOT / "configs/data/qip_v1/development-a4-complete-catalog.json").read_text())
    ids = {r["type_id"] for s in catalog["states"] for r in s["requests"]}
    certs = {i: {"classical_value": 0.7, "bell_value": 0.75} for i in ids}
    return RobustRoutingMDP(catalog, certs)


def test_actions_enforce_matching_and_budget():
    mdp = fixture()
    for state in range(len(mdp.catalog["states"])):
        for budget in range(3):
            for action in mdp.actions(state, budget):
                assert len(action) <= budget
                endpoints = [v for e in action for v in mdp.edges[e]]
                assert len(endpoints) == len(set(endpoints))


def test_visibility_reduces_and_eventually_removes_gain():
    mdp = fixture(); noisy = RobustRoutingMDP(mdp.catalog, mdp.certificates, visibility=0.5)
    assert np.all(noisy.edge_gains(0) <= mdp.edge_gains(0) + 1e-12)
    assert np.all(noisy.edge_gains(0) == 0)


def test_action_features_and_untrained_policy_are_valid():
    mdp = fixture(); actions = mdp.actions(0, 6)
    edge, global_features = action_features(mdp, 0, 6, 10, actions[-1])
    assert edge.shape == (28, 3) and global_features.shape == (8,)
    model = ActionScore(8)
    assert choose_q(model, mdp, 0, 6, 10) in actions


def test_tiny_trainers_execute_and_return_valid_models():
    mdp = fixture()
    q, qlog = train_q([mdp], 1, 4, 8, 1e-3, double=True, batch_size=4, target_interval=2)
    assert qlog["decisions"] == 40 and choose_q(q, mdp, 0, 6, 10) in mdp.actions(0, 6)
    ppo, plog = train_ppo([mdp], 2, 2, 8, 1e-3, recurrent=False, rollout_episodes=2, epochs=1)
    assert plog["decisions"] == 20


def test_oracle_dominates_greedy_in_expectation_fixture_sequences():
    mdp = fixture(); table = mdp.oracle_table()
    for seed in range(3):
        seq = mdp.sample_sequence(seed)
        greedy = mdp.evaluate(seq, greedy_action, seed=seed)["surplus"]
        oracle = mdp.evaluate(seq, lambda e, s, b, r: e.oracle_action(table, r, s, b), seed=seed)["surplus"]
        assert oracle + 1e-10 >= greedy
