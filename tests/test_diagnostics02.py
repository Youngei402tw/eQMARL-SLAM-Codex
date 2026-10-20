"""Stage 02 diagnostic wiring checks; no development policy matrix is run."""

import pytest

from eqmarl_slam.diagnostics02 import _env_config, correctness_fixtures, scenario_specs


def test_fixed_falsification_fixtures_pass_and_emit_each_gate():
    records = []
    resources = {}
    result = correctness_fixtures({}, records.append, resources=resources)
    assert result["passed"]
    assert all(result["gates"].values())
    assert len(records) == len(result["gates"]) == 11
    assert {row["fixture"] for row in records} == set(result["gates"])
    assert result["primitive_steps"] == 7
    assert resources == {"primitive_steps": 7, "decisions": 14}
    corridor = next(row for row in records if row["fixture"] == "corridor_physical_coverage")
    assert corridor["before"]["physical_coverage"] == 4 / 5
    assert corridor["after"]["physical_coverage"] == 1


def test_environment_contract_forwards_registered_reward_unchanged():
    reward = {"physical_coverage_gain": 1.0, "step_cost": 0.0,
              "collision_cost": 0.0, "communication_cost": 0.0}
    config = {"horizon": 80, "sensor_range": 2, "coverage_target": 1.0, "reward": reward}
    local = _env_config(config)
    shared = _env_config(config, shared=True)
    assert local["reward"] == shared["reward"] == reward
    assert local["information_condition"] == "local-only"
    assert shared["information_condition"] == "shared-map"
    assert {key: value for key, value in local.items() if key != "information_condition"} == {
        key: value for key, value in shared.items() if key != "information_condition"}
    local["reward"]["physical_coverage_gain"] = 99
    assert config["reward"]["physical_coverage_gain"] == 1


def test_scenarios_are_copied_and_have_fixed_shapes():
    source = {"scenarios": [{"scenario_id": "open", "grid": ["....", "...."], "starts": [[0, 0], [1, 3]]}]}
    copied = scenario_specs(source)
    copied[0]["starts"][0][0] = 1
    assert source["scenarios"][0]["starts"][0][0] == 0


@pytest.mark.parametrize("scenarios", [[],
    [{"scenario_id": "same", "grid": [".."], "starts": [[0, 0], [0, 1]]}] * 2,
    [{"scenario_id": "bad", "grid": ["..", "."], "starts": [[0, 0], [0, 1]]}],
])
def test_malformed_scenario_identity_or_shape_rejected(scenarios):
    with pytest.raises(ValueError):
        scenario_specs({"scenarios": scenarios})
