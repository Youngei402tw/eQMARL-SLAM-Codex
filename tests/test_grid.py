"""Known-pose exploration falsifiers; no learning or performance runs."""

import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from eqmarl_slam.grid import DEFAULT_SCENARIOS, EAST, NORTH, SOUTH, WAIT, WEST, GridExplorationEnv


def corridor(starts=((1, 1), (1, 5))):
    return {"scenario_id": "corridor-v1", "grid": ["#######", "#.....#", "#######"], "starts": starts}


def test_independent_physical_and_estimated_coverage_and_fixed_domain():
    env = GridExplorationEnv({"sensor_range": 1}, corridor())
    observations, info = env.reset(seed=123)
    assert env.observation_space.contains(observations)
    initial = env.metrics()
    assert initial["domain_cell_count"] == 5
    assert initial["physical_coverage"] == initial["estimated_coverage"] == 4 / 5
    domain_hash = initial["domain_sha256"]
    env.known_maps[0, 1, 3] = 0
    corrupt = env.metrics()
    assert corrupt["estimated_coverage"] == 1 and corrupt["physical_coverage"] == 4 / 5
    assert corrupt["domain_sha256"] == domain_hash
    obs, reward, terminated, truncated, info = env.step([EAST, WEST])
    assert env.observation_space.contains(obs) and isinstance(reward, float)
    assert terminated and not truncated
    assert env.metrics()["physical_coverage"] == 1
    assert env.metrics()["travel_distance"] == 2
    assert env.metrics()["map_accuracy"] < 1  # unseen boundary cells remain wrong
    assert info["evaluation_only"]["reward_uses_simulator_truth"]


def test_occlusion_first_wall_observed_beyond_wall_unknown():
    scenario = {"scenario_id": "occlusion-v1", "grid": ["#######", "#.#...#", "#.....#", "#######"], "starts": [[1, 1], [2, 5]]}
    env = GridExplorationEnv({"sensor_range": 5}, scenario)
    env.reset()
    own = env.observe_agent(0)
    assert own["map"][1, 2] == 1
    assert own["map"][1, 3] == -1
    assert not env.physical_seen_by_agent[0, 1, 3]


@pytest.mark.parametrize(
    "starts,actions,expected_collisions",
    [(((1, 1), (1, 3)), (EAST, WEST), 2),  # same destination
     (((1, 1), (1, 2)), (EAST, WEST), 2),  # head-on swap
     (((1, 1), (1, 2)), (EAST, WAIT), 1),  # waiting occupant
     (((1, 1), (1, 5)), (WEST, EAST), 2)], # walls
)
def test_simultaneous_collisions_leave_legal_unique_positions(starts, actions, expected_collisions):
    env = GridExplorationEnv({"sensor_range": 1}, corridor(starts))
    env.reset()
    previous = env.positions.copy()
    env.step(actions)
    np.testing.assert_array_equal(env.positions, previous)
    assert env.metrics()["collisions_total"] == expected_collisions
    assert env.metrics()["travel_distance"] == 0


def test_moves_into_vacated_cells_and_boundary_resolution():
    env = GridExplorationEnv({"sensor_range": 1}, corridor(((1, 1), (1, 2))))
    env.reset()
    env.step([EAST, EAST])
    np.testing.assert_array_equal(env.positions, [[1, 2], [1, 3]])
    assert env.metrics()["collisions_total"] == 0
    borderless = {"grid": ["....", "...."], "starts": [[0, 0], [1, 3]]}
    env = GridExplorationEnv({"sensor_range": 1}, borderless)
    env.reset()
    previous = env.positions.copy()
    env.step([NORTH, SOUTH])
    np.testing.assert_array_equal(env.positions, previous)
    assert env.metrics()["collisions_total"] == 2


def test_local_information_isolation_and_shared_delivery_cost():
    local = GridExplorationEnv({"sensor_range": 1}, corridor())
    local.reset()
    before = local.observe_agent(0)
    local.known_maps[1, 1, 3] = 0
    after = local.observe_agent(0)
    for key in before:
        np.testing.assert_array_equal(before[key], after[key])
    assert local.metrics()["communication_bytes"] == 0
    assert "messages" not in after
    shared = GridExplorationEnv({"sensor_range": 1, "information_condition": "shared-map"}, corridor())
    shared.reset()
    old = shared.observe_agent(0)["map"]
    bytes_before = shared.metrics()["communication_bytes"]
    assert bytes_before == 2 * (3 * 7 + 2 * 8)
    shared.known_maps[1, 1, 3] = 0
    np.testing.assert_array_equal(old, shared.observe_agent(0)["map"])
    assert shared.deliver_shared_messages() == bytes_before
    new = shared.observe_agent(0)["map"]
    assert new[1, 3] == 0 and old[1, 3] == -1
    assert shared.metrics()["communication_bytes"] == 2 * bytes_before
    assert shared.observe_agent(0)["messages"].shape == (1, 2)


def test_intrinsic_horizon_and_gymnasium_api():
    env = GridExplorationEnv({"sensor_range": 1, "horizon": 1}, DEFAULT_SCENARIOS[0])
    obs, _ = env.reset(seed=123)
    assert np.all(obs["remaining_budget"] == 1)
    obs, _, terminated, truncated, _ = env.step([WAIT, WAIT])
    assert terminated and not truncated
    assert np.all(obs["remaining_budget"] == 0)
    assert env.metrics()["termination_reason"] == "intrinsic_horizon"
    with pytest.raises(RuntimeError, match="Episode complete"):
        env.step([WAIT, WAIT])
    check_env(GridExplorationEnv({}, DEFAULT_SCENARIOS[0]), skip_render_check=True)


def test_domain_includes_only_union_of_start_reachable_free_cells():
    scenario = {"grid": ["#########", "#...#...#", "#########"], "starts": [[1, 1], [1, 3]]}
    env = GridExplorationEnv({"sensor_range": 1}, scenario)
    env.reset()
    assert env.metrics()["domain_cell_count"] == 3
    assert not env.reachable_free[1, 5]


def test_observations_do_not_alias_mutable_simulator_or_mapper():
    env = GridExplorationEnv({"sensor_range": 1}, corridor())
    obs, info = env.reset()
    snapshot = env.observe_agent(0)
    obs["map"][:] = 1
    info["evaluation_only"]["physical_seen"][:] = False
    snapshot["pose"][:] = 0
    assert env.observe_agent(0)["map"][1, 1] == 0
    assert env.physical_seen.any()
    np.testing.assert_array_equal(env.positions[0], [1, 1])
