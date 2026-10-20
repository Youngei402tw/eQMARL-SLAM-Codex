"""Known-map grid planning and explicitly permitted-information heuristics.

Cells are ``(row, column)``. Maps contain -1 unknown, 0 known free, and 1 known
occupied. Actions are wait, north, east, south, west at indices 0 through 4.
No function accepts a ground-truth grid or reads environment internals.
"""

from __future__ import annotations

from collections import deque
from typing import Sequence

import numpy as np


UNKNOWN, FREE, OCCUPIED = -1, 0, 1
ACTION_DELTAS = ((0, 0), (-1, 0), (0, 1), (1, 0), (0, -1))


def _map(known_map) -> np.ndarray:
    result = np.asarray(known_map)
    if result.ndim != 2 or min(result.shape) < 1 or not np.isin(result, [UNKNOWN, FREE, OCCUPIED]).all():
        raise ValueError("known_map must be a nonempty 2D array with -1/0/1 occupancy values")
    return result


def _cell(value, shape) -> tuple[int, int]:
    array = np.asarray(value)
    if array.shape != (2,) or not np.issubdtype(array.dtype, np.integer):
        raise ValueError("cell must contain two integer row/column coordinates")
    row, column = int(array[0]), int(array[1])
    if not (0 <= row < shape[0] and 0 <= column < shape[1]):
        raise ValueError("cell lies outside the map")
    return row, column


def _neighbors(cell, shape):
    for dr, dc in ACTION_DELTAS[1:]:
        other = cell[0] + dr, cell[1] + dc
        if 0 <= other[0] < shape[0] and 0 <= other[1] < shape[1]:
            yield other


def _parents(known_map: np.ndarray, start: tuple[int, int]):
    if known_map[start] != FREE:
        return {}
    queue = deque([start])
    parents = {start: None}
    while queue:
        current = queue.popleft()
        for neighbor in _neighbors(current, known_map.shape):
            if known_map[neighbor] == FREE and neighbor not in parents:
                parents[neighbor] = current
                queue.append(neighbor)
    return parents


def _restore_path(parents, target):
    if target not in parents:
        return None
    path = []
    while target is not None:
        path.append(target)
        target = parents[target]
    return path[::-1]


def bfs_path(known_map, start, target) -> list[tuple[int, int]] | None:
    """Shortest 4-neighbor path on known free cells, including both endpoints.

    Unknown or occupied endpoints are unreachable. In particular, a free
    current-cell target has path ``[start]`` and requires a wait action, rather
    than an indexing failure or an invented movement toward another cell.
    Invalid/out-of-bounds coordinates raise ``ValueError``.
    """
    occupancy = _map(known_map)
    current = _cell(start, occupancy.shape)
    destination = _cell(target, occupancy.shape)
    if occupancy[current] != FREE or occupancy[destination] != FREE:
        return None
    return _restore_path(_parents(occupancy, current), destination)


def action_for_path(path: Sequence[tuple[int, int]] | None) -> int:
    """First primitive action for an inclusive path; absent/trivial paths wait."""
    if path is None or len(path) < 2:
        return 0
    delta = tuple(np.asarray(path[1]) - np.asarray(path[0]))
    if delta not in ACTION_DELTAS[1:]:
        raise ValueError("path's first edge must join cardinally adjacent cells")
    return ACTION_DELTAS.index(delta)


def reachable_frontiers(known_map, start) -> list[tuple[int, int]]:
    """Sorted reachable known-free cells adjacent to at least one unknown cell."""
    occupancy = _map(known_map)
    current = _cell(start, occupancy.shape)
    parents = _parents(occupancy, current)
    return sorted(cell for cell in parents if any(occupancy[n] == UNKNOWN for n in _neighbors(cell, occupancy.shape)))


def _mask(action_mask) -> np.ndarray:
    mask = np.asarray(action_mask)
    if mask.shape != (5,) or mask.dtype != np.bool_ or not mask.any():
        raise ValueError("action_mask must have five boolean entries and at least one feasible action")
    return mask


def _choice(actions: list[int], *, rng: np.random.Generator, action_mode: str) -> int:
    if not isinstance(rng, np.random.Generator):
        raise ValueError("a caller-owned seeded numpy Generator is required")
    if action_mode == "greedy":
        return int(actions[0])
    if action_mode == "sampled":
        return int(rng.choice(actions))
    raise ValueError("action_mode must be 'greedy' or 'sampled'")


def random_action(action_mask, *, rng: np.random.Generator, action_mode: str = "sampled") -> int:
    """Uniformly sample the explicitly feasible actions, including feasible wait.

    A random policy has only the registered sampled evaluation mode. Callers
    own its generator so action sampling is separate from maps/sensors and
    repeated calls advance rather than resetting a seed.
    """
    if action_mode != "sampled":
        raise ValueError("the uniform random policy requires action_mode='sampled'")
    return _choice(np.flatnonzero(_mask(action_mask)).tolist(), rng=rng, action_mode=action_mode)


def reactive_action(own_map, own_pose, visited, action_mask, *, rng: np.random.Generator, action_mode: str = "greedy") -> int:
    """Prefer feasible known-free unvisited neighbors, then any known-free move.

    This is a local reactive diagnostic, with no path planning or teammate
    information. Greedy ties follow north/east/south/west order; sampled ties
    use the independent action generator. If no known-free move exists, wait
    when feasible, otherwise choose among the declared feasible actions.
    """
    occupancy = _map(own_map)
    current = _cell(own_pose, occupancy.shape)
    visits = np.asarray(visited)
    if visits.shape != occupancy.shape or visits.dtype != np.bool_:
        raise ValueError("visited must be a boolean map with the own_map shape")
    mask = _mask(action_mask)
    moving = []
    unseen = []
    for action, (dr, dc) in enumerate(ACTION_DELTAS[1:], start=1):
        cell = current[0] + dr, current[1] + dc
        if mask[action] and 0 <= cell[0] < occupancy.shape[0] and 0 <= cell[1] < occupancy.shape[1] and occupancy[cell] == FREE:
            moving.append(action)
            if not visits[cell]:
                unseen.append(action)
    candidates = unseen or moving or ([0] if mask[0] else np.flatnonzero(mask).tolist())
    return _choice(candidates, rng=rng, action_mode=action_mode)


def _frontier_plan(occupancy, current):
    parents = _parents(occupancy, current)
    options = []
    for target in reachable_frontiers(occupancy, current):
        path = _restore_path(parents, target)
        options.append((len(path) - 1, target, path))
    # A current-cell frontier does not justify an arbitrary step into unknown
    # space; prefer a reachable frontier requiring movement when one exists.
    moving_options = [option for option in options if option[0] > 0]
    if moving_options:
        options = moving_options
    return min(options, key=lambda option: (option[0], option[1])) if options else None


def greedy_local_frontier(own_map, own_pose, action_mask=None) -> int:
    """Choose the nearest reachable frontier using only the actor's own map.

    Equal distances break by row/column. There is no hidden stochastic mode.
    No frontier, a current-only frontier, or an infeasible first edge waits.
    A missing action mask is appropriate only for a map-only planner fixture.
    """
    occupancy = _map(own_map)
    current = _cell(own_pose, occupancy.shape)
    plan = _frontier_plan(occupancy, current)
    action = 0 if plan is None else action_for_path(plan[2])
    if action_mask is not None:
        mask = _mask(action_mask)
        if not mask[action]:
            return 0 if mask[0] else int(np.flatnonzero(mask)[0])
    return action


class SharedFrontierAssignment:
    """Deterministic nearest-pair allocation under an explicit shared condition.

    The caller supplies a delivered fused map and delivered poses; acquiring
    them must be charged by the environment's communication protocol. This
    controller does not construct a shared map from unreceived private state.
    Greedy agent/frontier matching avoids duplicate targets whenever an
    unassigned reachable frontier exists. Other robots' current cells are
    treated as planning obstacles and conflicting next cells are held, making
    no assumption that one robot can pass through another simultaneously.
    """

    action_mode = "greedy"

    def __init__(self):
        self.last_targets: list[tuple[int, int] | None] = []
        self.last_paths: list[list[tuple[int, int]] | None] = []

    def actions(self, shared_map, delivered_poses, action_masks=None) -> np.ndarray:
        occupancy = _map(shared_map)
        poses_array = np.asarray(delivered_poses)
        if poses_array.ndim != 2 or poses_array.shape[1] != 2 or len(poses_array) == 0:
            raise ValueError("delivered_poses must have shape (agents, 2)")
        poses = [_cell(pose, occupancy.shape) for pose in poses_array]
        if len(set(poses)) != len(poses):
            raise ValueError("delivered poses must identify distinct robot cells")
        n_agents = len(poses)
        if action_masks is None:
            masks = np.ones((n_agents, 5), dtype=bool)
        else:
            masks = np.asarray(action_masks)
            if masks.shape != (n_agents, 5):
                raise ValueError("action_masks must have shape (agents, 5)")
            for mask in masks:
                _mask(mask)
        options = []
        paths_by_agent = []
        for agent, pose in enumerate(poses):
            local_plan_map = occupancy.copy()
            for other, other_pose in enumerate(poses):
                if other != agent:
                    local_plan_map[other_pose] = OCCUPIED
            parents = _parents(local_plan_map, pose)
            paths = {}
            for frontier in reachable_frontiers(local_plan_map, pose):
                path = _restore_path(parents, frontier)
                paths[frontier] = path
                options.append((len(path) - 1, agent, frontier))
            paths_by_agent.append(paths)
        targets = [None] * n_agents
        used = set()
        # Moving targets take precedence over current-only sensing/wait cases.
        for distance, agent, target in sorted(options, key=lambda item: (item[0] == 0, item[0], item[1], item[2])):
            if targets[agent] is None and target not in used:
                targets[agent] = target
                used.add(target)
        selected_paths = [paths_by_agent[i].get(target) for i, target in enumerate(targets)]
        actions = np.asarray([action_for_path(path) for path in selected_paths], dtype=np.int64)
        for i, action in enumerate(actions):
            if not masks[i, action]:
                if not masks[i, 0]:
                    raise ValueError("shared assignment requires wait to be feasible when a planned action is masked")
                actions[i] = 0
        destinations = [tuple(np.asarray(poses[i]) + ACTION_DELTAS[action]) for i, action in enumerate(actions)]
        used_destinations = {}
        for i, destination in enumerate(destinations):
            if destination in used_destinations:
                # Current-cell obstacles preclude occupied destinations; only
                # two movers can propose the same empty cell here.
                actions[i] = 0
            else:
                used_destinations[destination] = i
        self.last_targets = targets
        self.last_paths = selected_paths
        return actions
