import json
from pathlib import Path

from eqmarl_slam.ber_marl import GLOBAL_ACTIONS, RoutingQ, dqn_action, features
from eqmarl_slam.budgeted_entanglement import BudgetedAllocationMDP
from eqmarl_slam.diagnostics23 import _clairvoyant


ROOT = Path(__file__).resolve().parents[1]


def test_routing_features_and_mask_respect_budget():
    catalog = json.loads((ROOT / "configs/data/development-budgeted-public-catalog-v1.json").read_text())
    rows = json.loads((ROOT / "experiments/21/21-budgeted-entanglement-task_attempt-03/request_certificates.json").read_text())["certificates"]
    mdp = BudgetedAllocationMDP(catalog, {row["type_id"]: row for row in rows})
    model = RoutingQ(16)
    assert features(mdp, 0, 6, 10).shape == (14,)
    assert len(GLOBAL_ACTIONS) == 10
    assert dqn_action(model, mdp, 0, 0, 5) == ()


def test_clairvoyant_bound_dominates_online_action_sequence():
    catalog = json.loads((ROOT / "configs/data/development-budgeted-public-catalog-v1.json").read_text())
    rows = json.loads((ROOT / "experiments/21/21-budgeted-entanglement-task_attempt-03/request_certificates.json").read_text())["certificates"]
    mdp = BudgetedAllocationMDP(catalog, {row["type_id"]: row for row in rows})
    sequence = mdp.sample_sequence(9)
    table = mdp.oracle_table()
    online = mdp.evaluate(sequence, lambda env, state, budget, remaining: env.oracle_action(table, remaining, state, budget))
    assert _clairvoyant(mdp, sequence) + 1e-12 >= online["entanglement_surplus"]
