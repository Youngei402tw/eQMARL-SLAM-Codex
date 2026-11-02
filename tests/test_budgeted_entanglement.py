import numpy as np

from eqmarl_slam.budgeted_entanglement import (
    BudgetedAllocationMDP, classical_value, generate_public_catalog,
    generate_request_types, valid_matchings,
)


def test_classical_enumeration_and_matchings():
    request = {"type_id":"unit", "predicate":[[0,0,0],[0,0,0],[0,0,0]], "weights":[[1]*3 for _ in range(3)]}
    value, strategy = classical_value(request)
    assert abs(value - 1.0) <= 1e-12
    assert set(valid_matchings(list(range(6)), 2))
    assert all(len({v for e in matching for v in ((0,1),(0,2),(0,3),(1,2),(1,3),(2,3))[e]}) == 2*len(matching) for matching in valid_matchings(list(range(6)),2))


def test_generator_replay_and_budget_invariants():
    one = generate_request_types(9, 3, "unit")
    two = generate_request_types(9, 3, "unit")
    assert one == two
    certificates = {row["type_id"]:{"classical_value":.75,"quantum_gap":.1} for row in one["types"]}
    catalog = generate_public_catalog(10,"unit",list(certificates),states=4)
    mdp = BudgetedAllocationMDP(catalog,certificates)
    table = mdp.oracle_table(); sequence = mdp.sample_sequence(11)
    result = mdp.evaluate(sequence,lambda env,s,b,r:env.oracle_action(table,r,s,b))
    assert result["pairs_used"] <= catalog["bell_pair_budget"]
    assert all(row["budget_after"] >= 0 for row in result["actions"])
    assert result["total_return"] == result["classical_return"] + result["entanglement_surplus"]
