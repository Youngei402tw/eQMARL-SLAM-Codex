"""Gymnasium interface to the budgeted Bell-pair routing benchmark (Stage 30).

The environment wraps ``RobustRoutingMDP`` without changing its semantics:

* Observation (public information only): per-edge certified gains of the current public state,
  edge endpoints and presence masks (padded to the eight-agent limit), remaining budget, remaining rounds.
* Action: an index into a fixed enumeration of matchings over catalog edge slots
  (0 = serve nothing, then single edges, then edge pairs). ``info["action_mask"]`` marks the feasible
  indices. By default an infeasible action raises ``ValueError`` exactly as ``RobustRoutingMDP.evaluate``
  does; ``invalid_action="noop"`` instead serves nothing and sets ``info["invalid_action"]``.
* Reward: realised surplus of the round (expected surplus scaled by heralded successes).
* Episodes end after ``horizon`` rounds (``terminated=True``; time remaining is observed).

Seeding follows the archived evaluator: ``reset(seed=s)`` reproduces ``mdp.sample_sequence(s)`` and the
heralding draws of ``mdp.evaluate(..., seed=s + 810000)``.
"""
from __future__ import annotations

import itertools

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .robust_routing import MAX_AGENTS, MAX_EDGES, RobustRoutingMDP

HERALD_SEED_OFFSET = 810000
ACTION_TABLE: tuple[tuple[int, ...], ...] = ((),) + tuple((e,) for e in range(MAX_EDGES)) + tuple(
    itertools.combinations(range(MAX_EDGES), 2))
ACTION_INDEX = {action: index for index, action in enumerate(ACTION_TABLE)}


class BellRouteEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, catalog: dict, certificates: dict[str, dict], visibility: float = 1.0,
                 heralded_success: float = 1.0, invalid_action: str = "raise"):
        if invalid_action not in ("raise", "noop"):
            raise ValueError("invalid_action must be 'raise' or 'noop'")
        self.invalid_action = invalid_action
        self.mdp = RobustRoutingMDP(catalog, certificates, visibility=visibility, heralded_success=heralded_success)
        if int(catalog.get("maximum_simultaneous_pairs", 2)) > 2:
            raise ValueError("action enumeration supports at most two simultaneous pairs")
        self.horizon = int(catalog["horizon"])
        self.initial_budget = int(catalog["bell_pair_budget"])
        edges = np.full((MAX_EDGES, 2), -1, dtype=np.int64)
        edges[:len(self.mdp.edges)] = np.asarray(self.mdp.edges, dtype=np.int64)
        self._edges = edges
        self._present = np.zeros(MAX_EDGES, dtype=np.int8)
        self._present[:len(self.mdp.edges)] = 1
        self.observation_space = spaces.Dict({
            # Gains are urgency (<= 5) times a win-probability gap (< 0.5).
            "edge_gains": spaces.Box(0.0, 2.5, shape=(MAX_EDGES,), dtype=np.float32),
            "edge_active": spaces.MultiBinary(MAX_EDGES),
            "edge_present": spaces.MultiBinary(MAX_EDGES),
            "edges": spaces.Box(-1, MAX_AGENTS - 1, shape=(MAX_EDGES, 2), dtype=np.int64),
            "budget": spaces.Box(0.0, float(self.initial_budget), shape=(1,), dtype=np.float32),
            "remaining": spaces.Box(0.0, float(self.horizon), shape=(1,), dtype=np.float32),
        })
        self.action_space = spaces.Discrete(len(ACTION_TABLE))
        self._state = self._budget = self._step = 0
        self._herald_rng = np.random.default_rng(0)

    def _observation(self):
        active = np.zeros(MAX_EDGES, dtype=np.int8)
        for request in self.mdp.catalog["states"][self._state]["requests"]:
            active[request["edge_index"]] = 1
        gains = np.zeros(MAX_EDGES, dtype=np.float32)
        current = self.mdp.edge_gains(self._state)
        gains[:len(current)] = current
        return {"edge_gains": gains, "edge_active": active, "edge_present": self._present.copy(),
                "edges": self._edges.copy(), "budget": np.asarray([self._budget], dtype=np.float32),
                "remaining": np.asarray([self.horizon - self._step], dtype=np.float32)}

    def action_mask(self) -> np.ndarray:
        mask = np.zeros(len(ACTION_TABLE), dtype=np.int8)
        for action in self.mdp.actions(self._state, self._budget):
            mask[ACTION_INDEX[tuple(sorted(action))]] = 1
        return mask

    def _info(self):
        return {"action_mask": self.action_mask(), "public_state": self._state}

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        transition = np.asarray(self.mdp.catalog["transition"], dtype=float)
        self._state = int(self.np_random.integers(len(transition)))
        self._budget, self._step = self.initial_budget, 0
        self._herald_rng = np.random.default_rng(None if seed is None else seed + HERALD_SEED_OFFSET)
        return self._observation(), self._info()

    def step(self, action):
        matching = ACTION_TABLE[int(action)]
        invalid = matching not in {tuple(sorted(a)) for a in self.mdp.actions(self._state, self._budget)}
        if invalid and self.invalid_action == "raise":
            raise ValueError("infeasible matching for the current public state and budget")
        if invalid:
            matching = ()
        base = self.mdp.action_gain(self._state, matching, expected=False)
        successes = int(self._herald_rng.binomial(len(matching), self.mdp.heralded_success)) if matching else 0
        reward = base if not matching else base * successes / len(matching)
        self._budget -= len(matching)
        self._step += 1
        transition = np.asarray(self.mdp.catalog["transition"], dtype=float)
        self._state = int(self.np_random.choice(len(transition), p=transition[self._state]))
        terminated = self._step >= self.horizon
        info = self._info()
        info.update({"attempts": len(matching), "successes": successes, "invalid_action": invalid})
        return self._observation(), float(reward), terminated, False, info
