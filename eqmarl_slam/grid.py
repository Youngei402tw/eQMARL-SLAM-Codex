"""Known-pose grid exploration control with explicit information boundaries.

This is a supplied-localization mapping task, not SLAM. Coordinates are
``(row, column)`` in a fixed common grid frame; one cell has unit length. The
sensor observes the current cell and four cardinal rays, including the first
wall on each ray. Unknown space behind a wall is not revealed. Actors receive
their own map and supplied pose, or explicitly delivered shared-map snapshots.

The simulator's physical visibility record is independent of the mapper. Truth
is used for sensing, scalar team reward, termination, and evaluation only.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Mapping
import hashlib
import json

import gymnasium as gym
from gymnasium import spaces
import numpy as np

UNKNOWN, FREE, WALL = -1, 0, 1
WAIT, NORTH, EAST, SOUTH, WEST = range(5)
ACTION_DELTAS = np.asarray([(0, 0), (-1, 0), (0, 1), (1, 0), (0, -1)], dtype=np.int64)


DEFAULT_SCENARIOS = (
    {"scenario_id": "development-open-room-v1", "grid": ["#######", "#.....#", "#.....#", "#.....#", "#.....#", "#.....#", "#######"], "starts": [[1, 1], [5, 5]]},
    {"scenario_id": "development-bottleneck-v1", "grid": ["#########", "#...#...#", "#.......#", "#...#...#", "#########"], "starts": [[1, 1], [3, 7]]},
    {"scenario_id": "development-dead-end-v1", "grid": ["#########", "#.......#", "###.#.###", "#...#...#", "#########"], "starts": [[1, 1], [3, 7]]},
)


def _mask_hash(mask: np.ndarray) -> str:
    shape = json.dumps(list(mask.shape), separators=(",", ":")).encode()
    return hashlib.sha256(shape + b"\n" + np.asarray(mask, dtype=np.uint8).tobytes(order="C")).hexdigest()


def _reachable_free(walls: np.ndarray, starts: np.ndarray) -> np.ndarray:
    reachable = np.zeros(walls.shape, dtype=bool)
    queue = deque(tuple(int(x) for x in position) for position in starts)
    for position in queue:
        reachable[position] = True
    height, width = walls.shape
    while queue:
        row, column = queue.popleft()
        for dr, dc in ACTION_DELTAS[1:]:
            neighbor = row + int(dr), column + int(dc)
            if (0 <= neighbor[0] < height and 0 <= neighbor[1] < width
                    and not walls[neighbor] and not reachable[neighbor]):
                reachable[neighbor] = True
                queue.append(neighbor)
    return reachable


def merge_known_maps(maps: np.ndarray) -> np.ndarray:
    """Merge exact common-frame measurements; reject contradictory evidence."""
    maps = np.asarray(maps)
    if maps.ndim != 3 or not np.isin(maps, [UNKNOWN, FREE, WALL]).all():
        raise ValueError("Known maps must have agent/row/column axes and values -1,0,1")
    free = np.any(maps == FREE, axis=0)
    wall = np.any(maps == WALL, axis=0)
    if np.any(free & wall):
        raise ValueError("Contradictory shared map cells in a noiseless known-pose protocol")
    result = np.full(maps.shape[1:], UNKNOWN, dtype=np.int8)
    result[free] = FREE
    result[wall] = WALL
    return result


class GridExplorationEnv(gym.Env):
    """Synchronous grid control with intrinsic finite-horizon termination.

    Primitive actions are wait/north/east/south/west. Wall or boundary moves,
    simultaneous same-cell moves, and head-on swaps are blocked. A move into
    another agent's cell succeeds only if that agent vacates it legally. Moves
    into a waiting or newly blocked agent propagate blocking. Collision counts
    charge attempted movers that were blocked, not an agent that chose to wait.

    Communication in ``shared-map`` sends each agent's complete int8 local map
    and two int64 pose coordinates to every other agent at reset and after each
    transition. Cost is the actual array payload byte count, without protocol
    headers. ``local-only`` sends no messages. An actor never gets an evaluator
    denominator or hidden reachable mask in its observation.
    """

    metadata = {"render_modes": []}

    def __init__(self, config: Mapping | None = None, scenario: Mapping | None = None):
        super().__init__()
        config = dict(config or {})
        scenario = dict(scenario or DEFAULT_SCENARIOS[0])
        grid = scenario.get("grid")
        if not isinstance(grid, (tuple, list)) or not grid or not all(isinstance(row, str) for row in grid):
            raise ValueError("Scenario grid must be a nonempty list of '#'/'.' rows")
        if not len(grid[0]) or any(len(row) != len(grid[0]) or set(row) - {"#", "."} for row in grid):
            raise ValueError("Scenario grid must be rectangular and contain only '#' and '.'")
        self._walls = np.asarray([[cell == "#" for cell in row] for row in grid], dtype=bool)
        self.height, self.width = self._walls.shape
        starts = np.asarray(scenario.get("starts"))
        if starts.ndim != 2 or starts.shape[1] != 2 or starts.shape[0] < 1 or not np.issubdtype(starts.dtype, np.integer):
            raise ValueError("Scenario starts must be integer (row,column) pairs")
        self.n_agents = int(starts.shape[0])
        if config.get("n_agents", self.n_agents) != self.n_agents:
            raise ValueError("Config n_agents differs from the scenario")
        if len(set(map(tuple, starts))) != self.n_agents:
            raise ValueError("Agent starts must be distinct")
        for position in starts:
            if not self._inside(position) or self._walls[tuple(position)]:
                raise ValueError("Agent starts must be legal free cells")
        self._starts = starts.astype(np.int64, copy=True)
        self.sensor_range = int(config.get("sensor_range", 2))
        self.horizon = int(config.get("horizon", 50))
        self.coverage_target = float(config.get("coverage_target", 1.0))
        self.information_condition = config.get("information_condition", "local-only")
        if self.sensor_range < 1 or self.horizon < 1 or not 0 < self.coverage_target <= 1:
            raise ValueError("Positive sensor range/horizon and coverage target in (0,1] required")
        if self.information_condition not in {"local-only", "shared-map"}:
            raise ValueError("information_condition must be local-only or shared-map")
        reward = {"physical_coverage_gain": 1.0, "step_cost": 0.0, "collision_cost": 0.0, "communication_cost": 0.0}
        supplied_reward = dict(config.get("reward", {}))
        if set(supplied_reward) - set(reward):
            raise ValueError("Unknown reward term")
        reward.update({key: float(value) for key, value in supplied_reward.items()})
        if not all(np.isfinite(value) and value >= 0 for value in reward.values()):
            raise ValueError("Reward coefficients must be finite and nonnegative")
        self.config = {
            "n_agents": self.n_agents, "sensor_range": self.sensor_range, "horizon": self.horizon,
            "coverage_target": self.coverage_target, "information_condition": self.information_condition,
            "reward": reward,
        }
        self.scenario_id = str(scenario.get("scenario_id", "explicit-grid-v1"))
        self.reachable_free = _reachable_free(self._walls, self._starts)
        self.domain_cell_count = int(self.reachable_free.sum())
        self.domain_sha256 = _mask_hash(self.reachable_free)
        self.map_sha256 = _mask_hash(self._walls)
        self.action_space = spaces.MultiDiscrete(np.full(self.n_agents, 5, dtype=np.int64))
        obs_spaces = {
            "map": spaces.Box(UNKNOWN, WALL, (self.n_agents, self.height, self.width), dtype=np.int8),
            "visited": spaces.MultiBinary((self.n_agents, self.height, self.width)),
            "pose": spaces.Box(np.zeros((self.n_agents, 2), dtype=np.int64), np.tile([self.height - 1, self.width - 1], (self.n_agents, 1)), dtype=np.int64),
            "remaining_budget": spaces.Box(0.0, 1.0, (self.n_agents, 1), dtype=np.float32),
            "action_mask": spaces.MultiBinary((self.n_agents, 5)),
        }
        if self.information_condition == "shared-map":
            obs_spaces["messages"] = spaces.Box(
                np.zeros((self.n_agents, self.n_agents - 1, 2), dtype=np.int64),
                np.tile([self.height - 1, self.width - 1], (self.n_agents, self.n_agents - 1, 1)), dtype=np.int64,
            )
        self.observation_space = spaces.Dict(obs_spaces)
        self._episode_done = True

    def _inside(self, position) -> bool:
        return 0 <= position[0] < self.height and 0 <= position[1] < self.width

    def _sensor_cells(self, position) -> np.ndarray:
        """Simulator truth is consulted only to generate this physical sensor."""
        mask = np.zeros((self.height, self.width), dtype=bool)
        origin = tuple(int(x) for x in position)
        mask[origin] = True
        for dr, dc in ACTION_DELTAS[1:]:
            for distance in range(1, self.sensor_range + 1):
                target = origin[0] + distance * int(dr), origin[1] + distance * int(dc)
                if not self._inside(target):
                    break
                mask[target] = True
                if self._walls[target]:
                    break
        return mask

    def _sense(self) -> None:
        # Physical measurement provenance is written before the mapper update.
        # Never reconstruct physical coverage from known_maps.
        for agent, position in enumerate(self.positions):
            visible = self._sensor_cells(position)
            self.current_visible[agent] = visible
            self.physical_seen_by_agent[agent] |= visible
            self.physical_seen |= visible
            self.known_maps[agent, visible] = self._walls[visible].astype(np.int8)
            self.visited_maps[agent, tuple(position)[0], tuple(position)[1]] = True

    def deliver_shared_messages(self) -> int:
        """Deliver a fresh snapshot and return charged payload bytes.

        Manual changes to another agent's private map cannot affect a previous
        delivery. This explicit method also supports counterfactual boundary
        tests without making observation reads send hidden messages.
        """
        if self.information_condition != "shared-map":
            return 0
        payload = 0
        for receiver in range(self.n_agents):
            senders = [sender for sender in range(self.n_agents) if sender != receiver]
            self.delivered_maps[receiver] = self.known_maps[senders].copy()
            self.delivered_poses[receiver] = self.positions[senders].copy()
            payload += self.delivered_maps[receiver].nbytes + self.delivered_poses[receiver].nbytes
        self.communication_bytes += payload
        self.message_count += self.n_agents * (self.n_agents - 1)
        return payload

    def _actor_map(self, agent: int) -> np.ndarray:
        if self.information_condition == "local-only":
            return self.known_maps[agent].copy()
        return merge_known_maps(np.concatenate([self.known_maps[agent:agent + 1], self.delivered_maps[agent]], axis=0))

    def observe_agent(self, agent: int) -> dict[str, np.ndarray]:
        if not 0 <= agent < self.n_agents:
            raise ValueError("Unknown agent index")
        known = self._actor_map(agent)
        action_mask = np.ones(5, dtype=bool)
        for action, delta in enumerate(ACTION_DELTAS[1:], 1):
            target = self.positions[agent] + delta
            action_mask[action] = bool(self._inside(target) and known[tuple(target)] != WALL)
        observation = {
            "map": known,
            "visited": self.visited_maps[agent].copy(),
            "pose": self.positions[agent].copy(),
            "remaining_budget": np.asarray([max(0.0, 1 - self.elapsed_steps / self.horizon)], dtype=np.float32),
            "action_mask": action_mask,
        }
        if self.information_condition == "shared-map":
            observation["messages"] = self.delivered_poses[agent].copy()
        return observation

    def _observations(self) -> dict[str, np.ndarray]:
        observations = [self.observe_agent(agent) for agent in range(self.n_agents)]
        return {key: np.stack([observation[key] for observation in observations]) for key in observations[0]}

    def metrics(self) -> dict:
        mapped = merge_known_maps(self.known_maps)
        physical_count = int(np.count_nonzero(self.physical_seen & self.reachable_free))
        estimated_count = int(np.count_nonzero((mapped == FREE) & self.reachable_free))
        physical_coverage = physical_count / self.domain_cell_count
        known = mapped != UNKNOWN
        correct = known & (mapped == self._walls.astype(np.int8))
        return {
            "scenario_id": self.scenario_id,
            "task_kind": "known_pose_grid_exploration",
            "information_condition": self.information_condition,
            "horizon_semantics": "intrinsic_finite_horizon_true_terminal",
            "elapsed_steps": self.elapsed_steps,
            "primitive_steps": self.elapsed_steps,
            "physical_coverage": physical_coverage,
            "coverage": physical_coverage,
            "estimated_coverage": estimated_count / self.domain_cell_count,
            "physical_free_cells_seen": physical_count,
            "estimated_known_free_domain_cells": estimated_count,
            "domain_cell_count": self.domain_cell_count,
            "domain_sha256": self.domain_sha256,
            "map_sha256": self.map_sha256,
            "map_accuracy": float(correct.sum() / self._walls.size),
            "map_accuracy_domain_cells": int(self._walls.size),
            "observed_only_map_accuracy": float(correct.sum() / known.sum()) if known.any() else 0.0,
            "unknown_cells": int((~known).sum()),
            "travel_distance": float(self.travel_distance),
            "collisions_total": self.collisions_total,
            "collision_rate": self.collisions_total / max(1, self.elapsed_steps * self.n_agents),
            "move_attempts": self.move_attempts,
            "communication_bytes": self.communication_bytes,
            "message_count": self.message_count,
            "success": bool(physical_coverage >= self.coverage_target),
            "threshold_step": self.threshold_step,
            "termination_reason": self.termination_reason,
        }

    def _info(self) -> dict:
        return {"evaluation_only": {
            "metrics": self.metrics(), "physical_seen": self.physical_seen.copy(),
            "reachable_free": self.reachable_free.copy(), "truth_wall_map": self._walls.copy(),
            "physical_seen_by_agent": self.physical_seen_by_agent.copy(),
        }}

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if options:
            raise ValueError("Scenario/config changes require a new environment identity")
        self.positions = self._starts.copy()
        shape = (self.n_agents, self.height, self.width)
        self.known_maps = np.full(shape, UNKNOWN, dtype=np.int8)
        self.visited_maps = np.zeros(shape, dtype=bool)
        self.current_visible = np.zeros(shape, dtype=bool)
        self.physical_seen_by_agent = np.zeros(shape, dtype=bool)
        self.physical_seen = np.zeros((self.height, self.width), dtype=bool)
        self.delivered_maps = np.full((self.n_agents, self.n_agents - 1, self.height, self.width), UNKNOWN, dtype=np.int8)
        self.delivered_poses = np.zeros((self.n_agents, self.n_agents - 1, 2), dtype=np.int64)
        self.elapsed_steps = self.collisions_total = self.move_attempts = 0
        self.travel_distance = 0.0
        self.communication_bytes = self.message_count = 0
        self.threshold_step = None
        self.termination_reason = None
        self._episode_done = False
        self._sense()
        self.deliver_shared_messages()
        if np.count_nonzero(self.physical_seen & self.reachable_free) / self.domain_cell_count >= self.coverage_target:
            self.threshold_step = 0
        return self._observations(), self._info()

    def _resolve_movement(self, actions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        proposed = self.positions + ACTION_DELTAS[actions]
        moving = actions != WAIT
        blocked = np.zeros(self.n_agents, dtype=bool)
        for agent, target in enumerate(proposed):
            if moving[agent] and (not self._inside(target) or self._walls[tuple(target)]):
                blocked[agent] = True
        for first in range(self.n_agents):
            for second in range(first + 1, self.n_agents):
                if np.array_equal(proposed[first], proposed[second]):
                    blocked[first] |= moving[first]
                    blocked[second] |= moving[second]
                if (moving[first] and moving[second]
                        and np.array_equal(proposed[first], self.positions[second])
                        and np.array_equal(proposed[second], self.positions[first])):
                    blocked[first] = blocked[second] = True
        # Rejected movement can make the source cell unavailable to another
        # move; propagate that occupancy without making update order matter.
        while True:
            previous = blocked.copy()
            for agent in range(self.n_agents):
                if moving[agent] and not blocked[agent]:
                    for other in range(self.n_agents):
                        if other != agent and (not moving[other] or blocked[other]) and np.array_equal(proposed[agent], self.positions[other]):
                            blocked[agent] = True
                            break
            if np.array_equal(previous, blocked):
                break
        actual = np.where(blocked[:, None], self.positions, proposed)
        return actual, blocked

    def step(self, actions):
        if self._episode_done:
            raise RuntimeError("Episode complete; reset before further transitions")
        actions = np.asarray(actions)
        if not self.action_space.contains(actions):
            raise ValueError("Actions must contain one integer index in [0,4] per agent")
        actions = actions.astype(np.int64, copy=False)
        old_seen = int(np.count_nonzero(self.physical_seen & self.reachable_free))
        positions, collisions = self._resolve_movement(actions)
        self.travel_distance += float(np.count_nonzero(np.any(positions != self.positions, axis=1)))
        self.positions = positions
        self.collisions_total += int(collisions.sum())
        self.move_attempts += int(np.count_nonzero(actions != WAIT))
        self.elapsed_steps += 1
        self._sense()
        bytes_sent = self.deliver_shared_messages()
        new_seen = int(np.count_nonzero(self.physical_seen & self.reachable_free))
        success = new_seen / self.domain_cell_count >= self.coverage_target
        if success and self.threshold_step is None:
            self.threshold_step = self.elapsed_steps
        terminated = bool(success or self.elapsed_steps >= self.horizon)
        self.termination_reason = "coverage_success" if success else "intrinsic_horizon" if terminated else None
        self._episode_done = terminated
        coefficients = self.config["reward"]
        components = {
            "physical_coverage_gain": coefficients["physical_coverage_gain"] * (new_seen - old_seen) / self.domain_cell_count,
            "step_cost": -coefficients["step_cost"],
            "collision_cost": -coefficients["collision_cost"] * float(collisions.mean()),
            "communication_cost": -coefficients["communication_cost"] * bytes_sent,
        }
        info = self._info()
        info["evaluation_only"].update(
            reward_components=components, collisions=collisions.copy(), bytes_sent=bytes_sent,
            reward_uses_simulator_truth=True,
        )
        return self._observations(), float(sum(components.values())), terminated, False, info
