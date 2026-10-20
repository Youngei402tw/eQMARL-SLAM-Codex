"""Classical observation and Gymnasium boundary fixtures.

``TinyContractEnv`` is a synthetic known-state API fixture. It is neither an
exploration benchmark nor an estimator or SLAM result.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
from gymnasium import spaces


@dataclass(frozen=True)
class LocalObservationAdapter:
    """Build an actor vector from that actor's declared local information only.

    The adapter accepts a local belief vector and, under ``message-limited``, an
    already delivered fixed-size packet vector. It has no global-map or truth
    input. Packet selection/delivery belongs to the simulator and must be
    accounted for by a later communication protocol.
    """

    belief_dim: int
    message_dim: int = 0
    include_budget: bool = True
    condition: str = "local-only"

    def __post_init__(self) -> None:
        if self.belief_dim < 1 or self.message_dim < 0:
            raise ValueError("Belief dimension must be positive and message dimension nonnegative")
        if self.condition not in {"local-only", "message-limited"}:
            raise ValueError("Only local-only or message-limited actor information is supported")
        if self.condition == "local-only" and self.message_dim:
            raise ValueError("Local-only observations cannot declare teammate messages")

    @property
    def observation_dim(self) -> int:
        return self.belief_dim + self.message_dim + int(self.include_budget)

    @staticmethod
    def _vector(value: Sequence[float], size: int, name: str) -> np.ndarray:
        array = np.asarray(value, dtype=np.float32)
        if array.shape != (size,) or not np.all(np.isfinite(array)):
            raise ValueError(f"{name} must be a finite vector of shape ({size},)")
        return array.copy()

    def build(
        self,
        own_belief: Sequence[float],
        delivered_messages: Sequence[float] | None = None,
        *,
        remaining_budget: float = 1.0,
    ) -> np.ndarray:
        pieces = [self._vector(own_belief, self.belief_dim, "own_belief")]
        if self.condition == "local-only":
            if delivered_messages is not None and np.asarray(delivered_messages).size:
                raise ValueError("Local-only actor cannot receive messages")
        else:
            if delivered_messages is None:
                if self.message_dim:
                    raise ValueError("Message-limited actor needs its declared delivered-message vector")
                delivered_messages = []
            pieces.append(self._vector(delivered_messages, self.message_dim, "delivered_messages"))
        if self.include_budget:
            if not np.isfinite(remaining_budget) or not 0 <= remaining_budget <= 1:
                raise ValueError("remaining_budget must lie in [0,1]")
            pieces.append(np.asarray([remaining_budget], dtype=np.float32))
        return np.concatenate(pieces)

    def from_fields(self, fields: Mapping[str, object]) -> np.ndarray:
        """Reject undeclared fields, including convenient truth/map additions."""
        allowed = {"own_belief", "delivered_messages", "remaining_budget"}
        if set(fields) - allowed or "own_belief" not in fields:
            raise ValueError("Observation fields must contain only declared local information")
        return self.build(**fields)

    def from_agent(
        self,
        agent_id: int,
        private_beliefs: Mapping[int, Sequence[float]],
        delivered_messages_by_agent: Mapping[int, Sequence[float]] | None = None,
        *,
        remaining_budget: float = 1.0,
    ) -> np.ndarray:
        """Select only the named agent; other private entries cannot affect it."""
        own_messages = None
        if delivered_messages_by_agent is not None:
            own_messages = delivered_messages_by_agent.get(agent_id)
        return self.build(
            private_beliefs[agent_id], own_messages, remaining_budget=remaining_budget
        )


class TinyContractEnv(gym.Env):
    """Two-agent line-world fixture for shapes and boundary semantics.

    Each actor receives its own known position, displacement to the publicly
    declared goal, and remaining finite-horizon budget. All agents reaching +2
    is a true terminal. The external step cap is a truncation, kept separate so
    a learner can explicitly choose its bootstrap rule. No performance metrics
    from this environment establish exploration or SLAM capability.
    """

    metadata = {"render_modes": []}

    def __init__(self, n_agents: int = 2, max_steps: int = 4):
        super().__init__()
        if n_agents < 1 or max_steps < 1:
            raise ValueError("n_agents and max_steps must be positive")
        self.n_agents = n_agents
        self.max_steps = max_steps
        self.observation_space = spaces.Box(-1.0, 1.0, (n_agents, 3), dtype=np.float32)
        self.action_space = spaces.MultiDiscrete(np.full(n_agents, 3))
        self.adapter = LocalObservationAdapter(2)
        self.positions = np.zeros(n_agents, dtype=np.int64)
        self.elapsed_steps = 0
        self._episode_done = True

    def _observations(self) -> np.ndarray:
        budget = max(0.0, 1.0 - self.elapsed_steps / self.max_steps)
        return np.stack(
            [
                self.adapter.build(
                    [float(position / 2), float((2 - position) / 4)],
                    remaining_budget=budget,
                )
                for position in self.positions
            ]
        )

    def _info(self) -> dict:
        return {
            "evaluation_only": {"positions": self.positions.copy()},
            "primitive_steps": self.elapsed_steps,
            "task_kind": "synthetic_known_state_contract_fixture",
        }

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.positions = np.zeros(self.n_agents, dtype=np.int64)
        self.elapsed_steps = 0
        self._episode_done = False
        return self._observations(), self._info()

    def step(self, action):
        if self._episode_done:
            raise RuntimeError("reset must precede stepping a completed episode")
        if not self.action_space.contains(action):
            raise ValueError("Joint action must contain one index in [0,2] per agent")
        self.positions = np.clip(self.positions + np.asarray(action, dtype=np.int64) - 1, -2, 2)
        self.elapsed_steps += 1
        terminated = bool(np.all(self.positions == 2))
        truncated = bool(self.elapsed_steps >= self.max_steps and not terminated)
        self._episode_done = terminated or truncated
        reward = float(np.mean(self.positions == 2))
        return self._observations(), reward, terminated, truncated, self._info()
