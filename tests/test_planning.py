import numpy as np
import pytest

from eqmarl_slam.planning import SharedFrontierAssignment, action_for_path, bfs_path, greedy_local_frontier, random_action, reactive_action, reachable_frontiers


def test_bfs_includes_current_and_handles_current_target():
    occupancy = np.array([[0, 0, 1], [1, 0, 0], [-1, 1, 0]], dtype=np.int8)
    assert bfs_path(occupancy, (0, 0), (2, 2)) == [(0, 0), (0, 1), (1, 1), (1, 2), (2, 2)]
    assert bfs_path(occupancy, (0, 0), (0, 0)) == [(0, 0)]
    assert action_for_path([(0, 0)]) == 0
    assert action_for_path(None) == 0
    assert action_for_path([(0, 0), (0, 1)]) == 2
    with pytest.raises(ValueError, match="adjacent"):
        action_for_path([(0, 0), (2, 2)])


def test_unknown_occupied_and_disconnected_targets_are_unreachable():
    occupancy = np.array([[0, 1, 0], [0, 1, 0], [-1, 1, 0]], dtype=np.int8)
    for target in [(0, 1), (0, 2), (2, 0)]:
        assert bfs_path(occupancy, (0, 0), target) is None
    assert bfs_path(occupancy, (2, 0), (2, 0)) is None
    with pytest.raises(ValueError, match="outside"):
        bfs_path(occupancy, (0, 0), (-1, 0))


def test_frontiers_are_reachable_known_free_not_unknown_cells():
    occupancy = np.array([[0, 0, -1], [0, 1, -1], [0, 1, 0]], dtype=np.int8)
    assert reachable_frontiers(occupancy, (0, 0)) == [(0, 1)]
    assert greedy_local_frontier(occupancy, (0, 0)) == 2
    mask = np.array([True, True, False, True, True])
    assert greedy_local_frontier(occupancy, (0, 0), mask) == 0
    only_current = np.array([[0, -1]], dtype=np.int8)
    assert reachable_frontiers(only_current, (0, 0)) == [(0, 0)]
    assert greedy_local_frontier(only_current, (0, 0)) == 0


def test_random_and_reactive_sampling_have_explicit_independent_rngs():
    mask = np.array([True, True, False, True, True])
    first = np.random.default_rng(1701)
    second = np.random.default_rng(1701)
    a = [random_action(mask, rng=first) for _ in range(30)]
    b = [random_action(mask, rng=second) for _ in range(30)]
    assert a == b and len(set(a)) > 1 and 2 not in a
    with pytest.raises(ValueError, match="sampled"):
        random_action(mask, rng=first, action_mode="greedy")
    occupancy = np.zeros((3, 3), dtype=np.int8)
    visited = np.ones((3, 3), dtype=bool)
    visited[1, 2] = False
    assert reactive_action(occupancy, (1, 1), visited, np.ones(5, dtype=bool), rng=first) == 2
    visited[0, 1] = False
    assert reactive_action(occupancy, (1, 1), visited, np.ones(5, dtype=bool), rng=first, action_mode="greedy") == 1


def test_shared_assignment_has_distinct_targets_and_no_private_state_reads():
    occupancy = np.array([[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [-1, -1, -1, -1, -1]], dtype=np.int8)
    poses = np.array([[0, 0], [0, 4]])
    controller = SharedFrontierAssignment()
    actions = controller.actions(occupancy, poses)
    np.testing.assert_array_equal(actions, [3, 3])
    assert controller.last_targets == [(1, 0), (1, 4)]
    assert len(set(controller.last_targets)) == len(poses)
    np.testing.assert_array_equal(controller.actions(occupancy.copy(), poses.copy()), actions)
    assert controller.action_mode == "greedy"


def test_shared_plans_do_not_enter_occupied_robot_cells_or_duplicate_next_cells():
    occupancy = np.array([[-1, -1, -1], [0, 0, 0], [0, 0, 0]], dtype=np.int8)
    poses = np.array([[2, 0], [2, 2]])
    controller = SharedFrontierAssignment()
    actions = controller.actions(occupancy, poses)
    deltas = np.array([[0, 0], [-1, 0], [0, 1], [1, 0], [0, -1]])
    destinations = poses + deltas[actions]
    assert len({tuple(cell) for cell in destinations}) == 2
    for i, cell in enumerate(destinations):
        assert not np.array_equal(cell, poses[1 - i])
