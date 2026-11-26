r"""Tabular Q-learning for the execution problem (Nevmyvaka, Feng & Kearns 2006, with a queue-aware replay).

State (``StateSpec``). Private part, as in Nevmyvaka et al.:
  time bucket  tb = ceil(5 * steps_left / N) - 1          (0 = last decision, 4 = start)
  inventory    ib = ceil(5 * remaining / Q) - 1            (0 = at most 20% left, 4 = more than 80% left)
Market part (any subset, 3 buckets each, cut points from the *training* days):
  ofi    5 s order-flow imbalance signed towards our side, x = side * ofi_z: buyers dominating is adverse for
         a buyer (the price tends to rise before we fill). 0 favourable (x < -c), 1 neutral, 2 adverse (x > c),
         with c the 1/3 quantile of |ofi_z|, so each tail holds about 1/3 of decisions. No trades in the last
         5 s (ofi_z undefined) counts as neutral, x = 0: no flow, no imbalance.
  queue  displayed size at our touch / typical size (30-minute median): 0 short, 1 medium, 2 long (terciles)
  sigma  trailing volatility sigma_hat: 0 calm, 1, 2 volatile (terciles)

Learning. Costs are minimised. After each step (state s, action a, shaped cost c, next state s'):
    Q(s, a) <- Q(s, a) + alpha * (c + min_a' Q(s', a') - Q(s, a))      (target = c at the end of the episode)
Undiscounted, finite horizon: time remaining is part of the state, so Q(s, a) estimates the expected cost
(bps of the order) of everything still to come, measured from the current mid. Exploration: epsilon-greedy,
epsilon falling linearly from 1 to ``eps_end`` over a fraction of the training budget. Q starts at 0, below
any real cost (costs are mostly positive), which also pushes the agent to try every action.

Why Q-learning rather than backward induction over time steps. With queue keeping, an order's place in the
queue depends on how long it has been resting, which the tabular state does not contain; so the one-step
outcome of an action at time t depends on the actions before t. Backward induction (as in Nevmyvaka et al.,
who re-placed the order every step, so had no such memory) would have to assume a fresh order at every step.
Q-learning learns from trajectories the agent actually takes, so the queue place it inherits from its own
earlier actions is in the data it learns from.

Greedy policy (evaluation). argmin over the actions tried in that state. A state never visited in training
falls back to the visit-weighted average of Q over the market buckets of the same (time, inventory) cell;
these fallbacks are counted and reported, never silent.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from mms.execution.env import N_ACTIONS, TOUCH

MARKET_VARS = ("ofi", "queue", "sigma")


@dataclass(frozen=True)
class StateSpec:
    market: tuple = ()                  # subset of MARKET_VARS, in this order
    ofi_c: float = 0.5
    queue_cuts: tuple = (0.5, 1.5)
    sigma_cuts: tuple = (2.0, 3.0)
    n_time: int = 5
    n_inv: int = 5

    def __post_init__(self):
        if any(m not in MARKET_VARS for m in self.market):
            raise ValueError(f"market variables must be in {MARKET_VARS}")

    @property
    def shape(self) -> tuple:
        return (self.n_time, self.n_inv) + (3,) * len(self.market)

    @property
    def name(self) -> str:
        return "private" if not self.market else "+".join(self.market)

    def time_bucket(self, steps_left: int, n_steps: int) -> int:
        return min(self.n_time - 1, max(0, math.ceil(self.n_time * steps_left / n_steps - 1e-9) - 1))

    def inv_bucket(self, remaining_frac: float) -> int:
        return min(self.n_inv - 1, max(0, math.ceil(self.n_inv * remaining_frac - 1e-9) - 1))

    def ofi_bucket(self, side: int, ofi_z: float) -> int:
        x = 0.0 if not np.isfinite(ofi_z) else side * ofi_z
        return 0 if x < -self.ofi_c else (2 if x > self.ofi_c else 1)

    @staticmethod
    def _tercile(x: float, cuts: tuple) -> int:
        if not np.isfinite(x):
            raise ValueError("state variable is not finite")
        return 0 if x < cuts[0] else (2 if x >= cuts[1] else 1)

    def encode(self, obs: dict) -> tuple:
        s = [self.time_bucket(obs["steps_left"], obs["n_steps"]), self.inv_bucket(obs["remaining_frac"])]
        for m in self.market:
            if m == "ofi":
                s.append(self.ofi_bucket(obs["side"], obs["ofi_z"]))
            elif m == "queue":
                s.append(self._tercile(obs["own_size"] / obs["typical_size"], self.queue_cuts))
            else:
                s.append(self._tercile(obs["sigma_hat"], self.sigma_cuts))
        return tuple(s)


@dataclass
class TabularQ:
    shape: tuple
    Q: np.ndarray = field(default=None)
    N: np.ndarray = field(default=None)       # visit counts per (state, action)
    fallbacks: int = 0
    decisions: int = 0

    def __post_init__(self):
        if self.Q is None:
            self.Q = np.zeros(self.shape + (N_ACTIONS,))
        if self.N is None:
            self.N = np.zeros(self.shape + (N_ACTIONS,), dtype=np.int64)

    def act(self, s: tuple, eps: float, rng: np.random.Generator) -> int:
        """epsilon-greedy (training); ties broken at random."""
        if rng.random() < eps:
            return int(rng.integers(N_ACTIONS))
        q = self.Q[s]
        best = np.flatnonzero(q == q.min())
        return int(best[0] if len(best) == 1 else rng.choice(best))

    def update(self, s: tuple, a: int, cost: float, s_next: tuple | None, alpha: float) -> None:
        target = cost if s_next is None else cost + float(self.Q[s_next].min())
        self.Q[s + (a,)] += alpha * (target - self.Q[s + (a,)])
        self.N[s + (a,)] += 1

    def greedy(self, s: tuple) -> int:
        """Evaluation policy (module docstring): only actions tried in s; unvisited s -> (time, inv) average."""
        self.decisions += 1
        n, q = self.N[s], self.Q[s]
        if n.sum() > 0:
            q = np.where(n > 0, q, np.inf)
            return int(np.argmin(q))
        self.fallbacks += 1
        t, i = s[0], s[1]
        n2 = self.N[t, i].reshape(-1, N_ACTIONS)
        q2 = self.Q[t, i].reshape(-1, N_ACTIONS)
        w = n2.sum(axis=0)
        if w.sum() == 0:
            return TOUCH                       # (time, inventory) cell never seen at all: counted as fallback
        qbar = np.where(w > 0, (n2 * q2).sum(axis=0) / np.maximum(w, 1), np.inf)
        return int(np.argmin(qbar))

    def policy_table(self) -> np.ndarray:
        """Greedy action per state (-1 = never visited)."""
        tried = self.N > 0
        q = np.where(tried, self.Q, np.inf)
        a = np.argmin(q, axis=-1)
        return np.where(tried.any(axis=-1), a, -1)


def epsilon(ep: int, n_budget: int, decay_frac: float, eps_end: float = 0.05) -> float:
    """Linear decay from 1 to eps_end over the first decay_frac * n_budget episodes, then flat."""
    k = decay_frac * n_budget
    if ep >= k:
        return eps_end
    return 1.0 - (1.0 - eps_end) * ep / k


class GreedyPolicy:
    """Wrap a trained table as ``policy(obs) -> action`` for ``run_execution``."""

    def __init__(self, agent: TabularQ, spec: StateSpec):
        self.agent, self.spec = agent, spec

    def __call__(self, obs: dict) -> int:
        return self.agent.greedy(self.spec.encode(obs))
