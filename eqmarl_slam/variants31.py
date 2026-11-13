"""Stage 31a: harder BellRoute variants, their exact references, and strong equal-information rules.

Variant A, generate-and-store (``StoreMDP``):
  A central store holds at most ``capacity`` Bell pairs. At the start of every round a fresh pair is
  generated with probability ``gen_prob`` (if the store is not full). Stored pairs age by one per round
  and a pair of age ``a`` has Werner visibility ``decay**a``. Serving edge ``e`` with a pair of age ``a``
  yields ``u_e * max(0, 1/2 + decay**a (V_Q - 1/2) - V_C)``. Pairs that can no longer help any request
  type (visibility at or below the smallest certified threshold) are discarded. The allocator observes
  the public state, the ages of stored pairs, and the remaining time, and chooses a matching plus an
  assignment of stored pairs to its edges. Exact DP over (state, age multiset) is the optimum.

Variant B, hidden regime (``RegimeMDP``):
  Each episode draws a latent regime z in {lean, rich} with probability 1/2. The public chain is tilted
  towards high-gain states (rich) or low-gain states (lean): P_z(s'|s) ~ P(s'|s) exp(+-beta g(s')/mean g).
  The budget is fixed as in v1. The near-optimal non-anticipative reference is belief-state DP on a
  41-point posterior grid with linear interpolation (acting on the exact posterior); the regime-oracle
  DP that knows z is an upper bound.
"""
from __future__ import annotations

import itertools
from functools import lru_cache

import numpy as np

from .robust_routing import RobustRoutingMDP, greedy_action
from .routing_baselines import BidPricePolicy, dp_table, weighted_quantile


# ------------------------------------------------------------------------------------------------
# Variant A: generate-and-store with decoherence
# ------------------------------------------------------------------------------------------------
class StoreMDP:
    def __init__(self, catalog: dict, certificates: dict, capacity: int, gen_prob: float, decay: float,
                 min_threshold: float):
        self.base = RobustRoutingMDP(catalog, certificates)
        self.catalog, self.capacity, self.gen_prob, self.decay = catalog, int(capacity), float(gen_prob), float(decay)
        # Smallest age at which no request type with a positive gap can still gain.
        self.max_age = int(np.ceil(np.log(min_threshold) / np.log(decay) - 1e-12))
        self.transition = np.asarray(catalog["transition"], dtype=float)
        self.horizon = int(catalog["horizon"])
        self.requests = []
        for state in catalog["states"]:
            rows = {}
            for request in state["requests"]:
                cert = certificates[request["type_id"]]
                rows[request["edge_index"]] = (float(request["urgency"]), float(cert["bell_value"]), float(cert["classical_value"]))
            self.requests.append(rows)
        self._gain = {}

    def gain(self, state: int, edge: int, age: int) -> float:
        key = (state, edge, age)
        if key not in self._gain:
            u, vq, vc = self.requests[state][edge]
            self._gain[key] = u * max(0.0, 0.5 + self.decay ** age * (vq - 0.5) - vc)
        return self._gain[key]

    def matchings(self, state: int, available: int):
        return self.base.actions(state, available)

    def options(self, state: int, store: tuple[int, ...]):
        """All (matching, assigned ages, reward) choices; ages drawn from the store multiset."""
        out = [((), (), 0.0)]
        for matching in self.matchings(state, len(store)):
            if not matching:
                continue
            for ages in set(itertools.permutations(store, len(matching))):
                reward = sum(self.gain(state, e, a) for e, a in zip(matching, ages))
                out.append((matching, ages, reward))
        return out

    def after(self, store: tuple[int, ...], used: tuple[int, ...]) -> tuple[int, ...]:
        remaining = list(store)
        for age in used:
            remaining.remove(age)
        return tuple(sorted(a + 1 for a in remaining if a + 1 < self.max_age))

    def generate(self, store: tuple[int, ...]) -> tuple[int, ...]:
        return tuple(sorted(store + (0,))) if len(store) < self.capacity else store

    def initial_store(self) -> tuple[int, ...]:
        return (0,) * self.capacity

    def sample(self, seed: int):
        rng = np.random.default_rng(seed)
        state = int(rng.integers(len(self.transition)))
        states, coins = [], []
        for _ in range(self.horizon):
            states.append(state); coins.append(bool(rng.random() < self.gen_prob))
            state = int(rng.choice(len(self.transition), p=self.transition[state]))
        return states, coins

    def solve(self):
        """Exact DP: value before generation, V[k][(state, store)]."""
        transition, cache = self.transition, {}

        @lru_cache(maxsize=None)
        def value(k: int, state: int, store: tuple[int, ...]) -> float:
            if k == 0:
                return 0.0
            return (self.gen_prob * post(k, state, self.generate(store))
                    + (1 - self.gen_prob) * post(k, state, store))

        @lru_cache(maxsize=None)
        def post(k: int, state: int, store: tuple[int, ...]) -> float:
            return max(r + continuation(k, state, self.after(store, ages)) for _, ages, r in self.options(state, store))

        @lru_cache(maxsize=None)
        def continuation(k: int, state: int, store: tuple[int, ...]) -> float:
            if k == 1:
                return 0.0
            return float(sum(p * value(k - 1, s, store) for s, p in enumerate(transition[state]) if p > 0))

        self._post, self._continuation = post, continuation
        return value(self.horizon, 0, self.initial_store())  # warm the cache for one start state

    def optimal_policy(self):
        if not hasattr(self, "_continuation"):
            self.solve()
        def policy(state, store, remaining, observed=None):
            return max(self.options(state, store), key=lambda o: o[2] + self._continuation(remaining, state, self.after(store, o[1])))
        return policy

    def run(self, seed: int, policy) -> float:
        states, coins = self.sample(seed)
        store, total = self.initial_store(), 0.0
        for step, (state, coin) in enumerate(zip(states, coins)):
            if coin:
                store = self.generate(store)
            matching, ages, reward = policy(state, store, self.horizon - step)
            if len(set(n for e in matching for n in self.base.edges[e])) != 2 * len(matching) or len(ages) != len(matching):
                raise ValueError("infeasible store action")
            total += reward
            store = self.after(store, ages)
        return total


def store_greedy(mdp: StoreMDP):
    def policy(state, store, remaining):
        # Maximise immediate reward; among ties use older pairs and fewer pairs.
        return max(mdp.options(state, store), key=lambda o: (round(o[2], 12), sum(o[1]), -len(o[1])))
    return policy


class StoreBidPrice:
    """Age-aware bid price: pair of age a may be spent on edge e only if gain(e,a) >= theta,
    theta = (1 - alpha * supply / (2k)) quantile of the reference gain mixture; supply counts stored
    pairs plus expected generation. A pair that expires next round has threshold 0."""

    def __init__(self, mdp: StoreMDP, reference: np.ndarray, alpha: float, prior_weight: float):
        self.mdp, self.reference, self.alpha, self.prior_weight = mdp, np.asarray(reference, float), alpha, prior_weight
        self.observed: list[float] = []

    def __call__(self, state, store, remaining):
        mdp = self.mdp
        fresh = sorted((mdp.gain(state, e, 0) for e in mdp.requests[state]), reverse=True)[:2]
        self.observed.extend(fresh + [0.0] * (2 - len(fresh)))
        supply = len(store) + mdp.gen_prob * (remaining - 1)
        share = self.alpha * supply / (2 * remaining)
        if share >= 1.0:
            theta = 0.0
        else:
            obs = np.asarray(self.observed)
            if self.prior_weight > 0:
                values = np.concatenate([self.reference, obs])
                weights = np.concatenate([np.full(len(self.reference), self.prior_weight / len(self.reference)), np.ones(len(obs))])
            else:
                values, weights = obs, np.ones(len(obs))
            theta = weighted_quantile(values, weights, 1.0 - share)
        best = ((), (), 0.0)
        for option in mdp.options(state, store):
            matching, ages, reward = option
            if not matching:
                continue
            ok = all(mdp.gain(state, e, a) > 0 and (mdp.gain(state, e, a) >= theta or a + 1 >= mdp.max_age)
                     for e, a in zip(matching, ages))
            if ok and (reward, sum(ages)) > (best[2], sum(best[1])):
                best = option
        return best


def store_reference_gains(mdps: list[StoreMDP]) -> np.ndarray:
    out = []
    for mdp in mdps:
        for state in range(len(mdp.requests)):
            fresh = sorted((mdp.gain(state, e, 0) for e in mdp.requests[state]), reverse=True)[:2]
            out.extend(fresh + [0.0] * (2 - len(fresh)))
    return np.asarray(out)


# ------------------------------------------------------------------------------------------------
# Variant B: hidden regime
# ------------------------------------------------------------------------------------------------
class RegimeMDP:
    GRID = np.linspace(0.0, 1.0, 41)

    def __init__(self, catalog: dict, certificates: dict, beta: float):
        self.base = RobustRoutingMDP(catalog, certificates)
        self.catalog, self.beta = catalog, float(beta)
        P = np.asarray(catalog["transition"], dtype=float)
        g = np.asarray([self.base.action_gain(s, greedy_action(self.base, s, 2, 1)) for s in range(len(P))])
        tilt = g / max(g.mean(), 1e-12)
        self.P = []
        for sign in (-1.0, 1.0):  # z=0 lean, z=1 rich
            Q = P * np.exp(sign * self.beta * tilt)[None, :]
            self.P.append(Q / Q.sum(axis=1, keepdims=True))
        self.horizon, self.budget, self.n = int(catalog["horizon"]), int(catalog["bell_pair_budget"]), len(P)

    def sample(self, seed: int):
        rng = np.random.default_rng(seed)
        z = int(rng.integers(2)); state = int(rng.integers(self.n)); out = []
        for _ in range(self.horizon):
            out.append(state); state = int(rng.choice(self.n, p=self.P[z][state]))
        return z, out

    def posterior(self, prior: float, state: int, following: int) -> float:
        a = prior * self.P[1][state, following]; b = (1 - prior) * self.P[0][state, following]
        return float(a / (a + b)) if a + b > 0 else prior

    def solve_belief(self):
        grid, H, B, n = self.GRID, self.horizon, self.budget, self.n
        V = np.zeros((H + 1, n, B + 1, len(grid)))
        self.future = np.zeros((H + 1, n, B + 1, len(grid)))  # future[k, s, b'] = E[V_{k-1}(s', b', pi')]
        for k in range(1, H + 1):
            for s in range(n):
                p_next = grid[:, None] * self.P[1][s][None, :] + (1 - grid[:, None]) * self.P[0][s][None, :]
                post = np.where(p_next > 0, grid[:, None] * self.P[1][s][None, :] / np.maximum(p_next, 1e-300), grid[:, None])
                for b in range(B + 1):
                    if k > 1:
                        self.future[k, s, b] = sum(p_next[:, s2] * np.interp(post[:, s2], grid, V[k - 1, s2, b]) for s2 in range(n))
                for b in range(B + 1):
                    V[k, s, b] = np.max([self.base.action_gain(s, a) + self.future[k, s, b - len(a)] for a in self.base.actions(s, b)], axis=0)
        self.V = V

    def belief_policy(self):
        if not hasattr(self, "V"):
            self.solve_belief()
        def act(state, budget, remaining, belief):
            return max(self.base.actions(state, budget), key=lambda a: self.base.action_gain(state, a)
                       + np.interp(belief, self.GRID, self.future[remaining, state, budget - len(a)]))
        return act

    def oracle_tables(self):
        tables = []
        for z in (0, 1):
            m = RobustRoutingMDP(dict(self.catalog, transition=self.P[z].tolist()), self.base.certificates)
            tables.append((m, dp_table(m, self.P[z])))
        return tables

    def run(self, seed: int, policy) -> float:
        """policy(state, budget, remaining, belief, z) -> action; belief is the exact posterior of rich."""
        z, states = self.sample(seed)
        budget, belief, total = self.budget, 0.5, 0.0
        for step, state in enumerate(states):
            if step > 0:
                belief = self.posterior(belief, states[step - 1], state)
            action = tuple(policy(state, budget, self.horizon - step, belief, z))
            if action not in self.base.actions(state, budget):
                raise ValueError("infeasible action")
            total += self.base.action_gain(state, action); budget -= len(action)
        return total
