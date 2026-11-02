"""Graph-size-independent sequential routing of certified entanglement resources."""
from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np

MAX_AGENTS = 8
MAX_EDGES = MAX_AGENTS * (MAX_AGENTS - 1) // 2


@dataclass
class RobustRoutingMDP:
    catalog: dict
    certificates: dict[str, dict]
    visibility: float = 1.0
    heralded_success: float = 1.0

    def __post_init__(self):
        if not 0.0 <= self.visibility <= 1.0 or not 0.0 <= self.heralded_success <= 1.0:
            raise ValueError("resource probabilities must be in [0,1]")
        self.edges = tuple(tuple(edge) for edge in self.catalog["edges"])
        if len(self.edges) > MAX_EDGES:
            raise ValueError("catalog exceeds registered eight-agent limit")

    def actions(self, state: int, budget: int) -> list[tuple[int, ...]]:
        active = [row["edge_index"] for row in self.catalog["states"][state]["requests"]]
        maximum = min(int(self.catalog.get("maximum_simultaneous_pairs", 2)), budget)
        out = [()]
        for size in range(1, maximum + 1):
            for action in itertools.combinations(active, size):
                endpoints = [node for edge in action for node in self.edges[edge]]
                if len(endpoints) == len(set(endpoints)):
                    out.append(tuple(action))
        return out

    def edge_gains(self, state: int) -> np.ndarray:
        gains = np.zeros(len(self.edges), dtype=np.float32)
        for request in self.catalog["states"][state]["requests"]:
            cert = self.certificates[request["type_id"]]
            noisy_bell = 0.5 + self.visibility * (float(cert["bell_value"]) - 0.5)
            gain = max(0.0, noisy_bell - float(cert["classical_value"]))
            gains[request["edge_index"]] = float(request["urgency"]) * gain
        return gains

    def action_gain(self, state: int, action: tuple[int, ...], *, expected: bool = True) -> float:
        value = float(self.edge_gains(state)[list(action)].sum()) if action else 0.0
        return value * self.heralded_success if expected else value

    def sample_sequence(self, seed: int) -> list[int]:
        rng = np.random.default_rng(seed)
        transition = np.asarray(self.catalog["transition"], dtype=float)
        state = int(rng.integers(len(transition)))
        out = []
        for _ in range(int(self.catalog["horizon"])):
            out.append(state)
            state = int(rng.choice(len(transition), p=transition[state]))
        return out

    def evaluate(self, sequence: list[int], policy, seed: int | None = None) -> dict:
        rng = np.random.default_rng(seed)
        budget = int(self.catalog["bell_pair_budget"])
        rewards, actions = [], []
        violations = []
        for step, state in enumerate(sequence):
            remaining = len(sequence) - step
            action = tuple(policy(self, state, budget, remaining))
            if action not in self.actions(state, budget):
                raise ValueError("policy returned infeasible action")
            base = self.action_gain(state, action, expected=False)
            successes = int(rng.binomial(len(action), self.heralded_success)) if action else 0
            reward = base if not action else base * successes / len(action)
            budget -= len(action)
            if budget < 0:
                violations.append("negative_budget")
            rewards.append(reward)
            actions.append({"state": state, "matching": list(action), "reward": reward,
                            "attempts": len(action), "successes": successes, "budget_after": budget})
        return {"surplus": float(sum(rewards)), "pairs_attempted": int(self.catalog["bell_pair_budget"] - budget),
                "pairs_unused": budget, "violations": violations, "actions": actions}

    def oracle_table(self) -> np.ndarray:
        horizon = int(self.catalog["horizon"])
        maximum = int(self.catalog["bell_pair_budget"])
        states = len(self.catalog["states"])
        transition = np.asarray(self.catalog["transition"], dtype=float)
        value = np.zeros((horizon + 1, states, maximum + 1), dtype=float)
        for remaining in range(1, horizon + 1):
            for state in range(states):
                future = transition[state] @ value[remaining - 1]
                for budget in range(maximum + 1):
                    value[remaining, state, budget] = max(
                        self.action_gain(state, action) + future[budget - len(action)]
                        for action in self.actions(state, budget))
        return value

    def oracle_action(self, table: np.ndarray, remaining: int, state: int, budget: int) -> tuple[int, ...]:
        transition = np.asarray(self.catalog["transition"], dtype=float)
        future = transition[state] @ table[remaining - 1]
        return max(self.actions(state, budget), key=lambda action:
                   self.action_gain(state, action) + future[budget - len(action)])


def greedy_action(mdp: RobustRoutingMDP, state: int, budget: int, remaining: int) -> tuple[int, ...]:
    return max(mdp.actions(state, budget), key=lambda action: (mdp.action_gain(state, action), len(action)))


def clairvoyant_value(mdp: RobustRoutingMDP, sequence: list[int]) -> float:
    horizon, maximum = len(sequence), int(mdp.catalog["bell_pair_budget"])
    value = np.zeros((horizon + 1, maximum + 1), dtype=float)
    for step in range(horizon - 1, -1, -1):
        for budget in range(maximum + 1):
            value[step, budget] = max(mdp.action_gain(sequence[step], action) + value[step + 1, budget - len(action)]
                                      for action in mdp.actions(sequence[step], budget))
    return float(value[0, maximum])
