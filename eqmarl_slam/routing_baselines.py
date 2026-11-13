"""Stronger non-learned allocators for the robust routing benchmark (Stage 30).

Two families are provided, both written against the public ``RobustRoutingMDP`` interface:

* ``BidPricePolicy`` -- a budget-pacing threshold ("bid-price") heuristic. In each round it serves
  the best feasible matching whose every edge has a gain at least ``tau(b, k)``, where ``tau`` is the
  ``1 - alpha * b / (slots * k)`` quantile of a reference distribution of per-slot gains. The reference
  mixes gains pooled from training catalogs (total weight ``prior_weight``) with the gains observed so
  far in the current episode (weight one each). It never sees the transition matrix.
* Certainty-equivalent dynamic programming -- estimate the public transition matrix from ``K`` logged
  public sequences with a Dirichlet prior centred on a self-loop probability ``rho``, then plan with
  exact finite-horizon DP on the estimate. ``K = 0`` plans on the prior alone.
"""
from __future__ import annotations

import numpy as np

from .robust_routing import RobustRoutingMDP

SLOTS = 2


class ActionCache:
    """Memoised feasible actions and expected gains for one MDP."""

    def __init__(self, mdp: RobustRoutingMDP):
        self.mdp = mdp
        self._actions: dict[tuple[int, int], list[tuple[int, ...]]] = {}
        self._gains: dict[int, np.ndarray] = {}

    def actions(self, state: int, budget: int) -> list[tuple[int, ...]]:
        key = (state, min(int(budget), SLOTS + 1))
        if key not in self._actions:
            self._actions[key] = self.mdp.actions(state, budget)
        return self._actions[key]

    def edge_gains(self, state: int) -> np.ndarray:
        if state not in self._gains:
            self._gains[state] = self.mdp.edge_gains(state)  # float32, exactly as the evaluator
        return self._gains[state]

    def action_gain(self, state: int, action: tuple[int, ...]) -> float:
        # Same float32 summation as RobustRoutingMDP.action_gain so ties resolve identically.
        value = float(self.edge_gains(state)[list(action)].sum()) if action else 0.0
        return value * self.mdp.heralded_success

    def slot_gains(self, state: int) -> list[float]:
        """Per-edge gains of the best unconstrained-budget matching, padded to ``SLOTS``."""
        best = max(self.actions(state, SLOTS), key=lambda a: (self.action_gain(state, a), -len(a)))
        gains = sorted((float(self.edge_gains(state)[e]) for e in best), reverse=True)
        return (gains + [0.0] * SLOTS)[:SLOTS]


def reference_slot_gains(mdps: list[RobustRoutingMDP]) -> np.ndarray:
    """Pool per-slot gains over every public state of the given (training) catalogs."""
    out = []
    for mdp in mdps:
        cache = ActionCache(mdp)
        for state in range(len(mdp.catalog["states"])):
            out.extend(cache.slot_gains(state))
    return np.asarray(out, dtype=float)


def weighted_quantile(values: np.ndarray, weights: np.ndarray, level: float) -> float:
    if level <= 0.0 or len(values) == 0:
        return 0.0
    order = np.argsort(values, kind="stable")
    values, weights = values[order], weights[order]
    cumulative = np.cumsum(weights)
    index = int(np.searchsorted(cumulative, level * cumulative[-1], side="left"))
    return float(values[min(index, len(values) - 1)])


class BidPricePolicy:
    """Stateful per-episode pacing-threshold allocator; construct a fresh instance for every episode."""

    def __init__(self, reference: np.ndarray, alpha: float, prior_weight: float):
        if alpha <= 0 or prior_weight < 0:
            raise ValueError("alpha must be positive and prior_weight non-negative")
        self.reference = np.asarray(reference, dtype=float)
        self.alpha, self.prior_weight = float(alpha), float(prior_weight)
        self.observed: list[float] = []
        self.cache: ActionCache | None = None
        self.thresholds: list[float] = []

    def threshold(self, budget: int, remaining: int) -> float:
        share = self.alpha * budget / (SLOTS * remaining)
        if share >= 1.0:
            return 0.0
        observed = np.asarray(self.observed, dtype=float)
        if self.prior_weight > 0 and len(self.reference):
            values = np.concatenate([self.reference, observed])
            weights = np.concatenate([np.full(len(self.reference), self.prior_weight / len(self.reference)),
                                      np.ones(len(observed))])
        else:
            values, weights = observed, np.ones(len(observed))
        return weighted_quantile(values, weights, 1.0 - share)

    def __call__(self, mdp: RobustRoutingMDP, state: int, budget: int, remaining: int) -> tuple[int, ...]:
        if self.cache is None or self.cache.mdp is not mdp:
            self.cache = ActionCache(mdp)
        cache = self.cache
        self.observed.extend(cache.slot_gains(state))
        tau = self.threshold(budget, remaining)
        self.thresholds.append(tau)
        gains = cache.edge_gains(state)
        eligible = [a for a in cache.actions(state, budget)
                    if all(gains[e] > 0.0 and gains[e] >= tau for e in a)]
        return max(eligible, key=lambda a: (cache.action_gain(state, a), -len(a)))


def prior_transition(states: int, rho: float) -> np.ndarray:
    if states == 1:
        return np.ones((1, 1))
    prior = np.full((states, states), (1.0 - rho) / (states - 1))
    np.fill_diagonal(prior, rho)
    return prior


def estimate_transition(states: int, sequences: list[list[int]], rho: float, kappa: float) -> np.ndarray:
    """Posterior-mean transition matrix under a Dirichlet prior with mean ``prior_transition`` and mass ``kappa``."""
    counts = np.zeros((states, states))
    for sequence in sequences:
        for current, following in zip(sequence[:-1], sequence[1:]):
            counts[current, following] += 1.0
    prior = prior_transition(states, rho)
    return (counts + kappa * prior) / (counts.sum(axis=1, keepdims=True) + kappa)


def dp_table(mdp: RobustRoutingMDP, transition: np.ndarray, cache: ActionCache | None = None) -> np.ndarray:
    """Exact finite-horizon DP under ``transition`` (identical recursion to ``oracle_table``)."""
    cache = cache or ActionCache(mdp)
    horizon, maximum = int(mdp.catalog["horizon"]), int(mdp.catalog["bell_pair_budget"])
    states = len(mdp.catalog["states"])
    value = np.zeros((horizon + 1, states, maximum + 1))
    for remaining in range(1, horizon + 1):
        for state in range(states):
            future = transition[state] @ value[remaining - 1]
            for budget in range(maximum + 1):
                value[remaining, state, budget] = max(cache.action_gain(state, a) + future[budget - len(a)]
                                                      for a in cache.actions(state, budget))
    return value


class PlannerPolicy:
    """Greedy-in-value policy for a DP table computed under a (possibly estimated) transition matrix."""

    def __init__(self, mdp: RobustRoutingMDP, transition: np.ndarray):
        self.mdp, self.transition = mdp, np.asarray(transition, dtype=float)
        self.cache = ActionCache(mdp)
        self.table = dp_table(mdp, self.transition, self.cache)

    def __call__(self, mdp: RobustRoutingMDP, state: int, budget: int, remaining: int) -> tuple[int, ...]:
        future = self.transition[state] @ self.table[remaining - 1]
        return max(self.cache.actions(state, budget),
                   key=lambda a: self.cache.action_gain(state, a) + future[budget - len(a)])


def mean_self_loop(mdps: list[RobustRoutingMDP]) -> float:
    return float(np.mean([np.mean(np.diag(np.asarray(m.catalog["transition"], dtype=float))) for m in mdps]))
