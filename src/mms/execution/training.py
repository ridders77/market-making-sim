"""Training loop for the tabular execution agent, and episode sets for training, validation and test.

Episodes. Training: a day drawn uniformly from the training days and a start drawn uniformly on the 100 ms grid
between 01:00 UTC (the first hour warms up sigma_hat and the typical size) and the last start that leaves the
episode plus 60 s for markouts inside the day. With probability ``explore_p`` the episode is an "exploring
start": it begins with a random number of steps left (1..N) and a random number of lots left (1..Q/lot), so
every (time, inventory) cell of the table is visited, not only those the current policy happens to reach.

Validation and test: stratified, non-overlapping. The day from 01:00 is cut into slots of length 2H and one
episode starts uniformly in the first half of each slot (``stratified_starts``). Every strategy, side, fee and
cancel rule is run on the same list of episodes, so all comparisons are paired.
"""

from __future__ import annotations

import logging
from collections.abc import Callable

import numpy as np

from mms.execution.env import ExecConfig, ExecutionEnv
from mms.execution.qlearning import GreedyPolicy, StateSpec, TabularQ, epsilon
from mms.replay.data import BUCKETS_PER_S

log = logging.getLogger(__name__)
FIRST_START = 3600 * BUCKETS_PER_S
DAY_BUCKETS = 86400 * BUCKETS_PER_S
MARKOUT_PAD = 60 * BUCKETS_PER_S + 2


def last_start(n_buckets_episode: int, day_buckets: int = DAY_BUCKETS) -> int:
    return day_buckets - n_buckets_episode - MARKOUT_PAD


def stratified_starts(days: list[str], cfg: ExecConfig, seed: int) -> list[tuple[str, int]]:
    rng = np.random.default_rng(seed)
    ep = cfg.n_steps * cfg.k
    out = []
    for d in days:
        s = FIRST_START
        while s + ep <= last_start(0):
            st = s + int(rng.integers(0, ep))
            if st <= last_start(ep):
                out.append((d, st))
            s += 2 * ep
    return out


def run_greedy(agent: TabularQ, spec: StateSpec, cfg: ExecConfig, get_day: Callable, episodes) -> np.ndarray:
    """Mean-able array of greedy-policy costs (bps, at cfg's fees) over a list of (day, start)."""
    pol = GreedyPolicy(agent, spec)
    costs = np.empty(len(episodes))
    envs = {}
    for n, (d, st) in enumerate(episodes):
        env = envs.get(d)
        if env is None:
            env = envs[d] = ExecutionEnv(get_day(d), cfg)
        obs = env.reset(st)
        done = False
        while not done:
            obs, _, done, _ = env.step(pol(obs))
        costs[n] = env.result().cost_bps
    return costs


def train(spec: StateSpec, cfg: ExecConfig, get_day: Callable, train_days: list[str], n_budget: int,
          alpha: float, decay_frac: float, seed: int, checkpoints: list[int], evaluate: Callable | None = None,
          explore_p: float = 0.5, eps_end: float = 0.05, stop_at: int | None = None) -> dict:
    """Q-learning over ``n_budget`` episodes (epsilon schedule defined on the budget), stopping at ``stop_at``
    (default: the budget). At each checkpoint ``evaluate(agent) -> dict`` is called and a copy of the table kept.
    Returns {"agent", "curve": [{"episodes", ...evaluate output, "train_cost_running"}], "snapshots": {ep: (Q, N)}}."""
    rng = np.random.default_rng(seed)
    agent = TabularQ(spec.shape)
    envs = {d: ExecutionEnv(get_day(d), cfg) for d in train_days}
    N, k, lots = cfg.n_steps, cfg.k, int(round(cfg.qty / cfg.lot))
    stop = n_budget if stop_at is None else stop_at
    curve, snaps, running = [], {}, []
    cps = sorted(c for c in checkpoints if c <= stop)
    for ep in range(1, stop + 1):
        eps = epsilon(ep - 1, n_budget, decay_frac, eps_end)
        d = train_days[int(rng.integers(len(train_days)))]
        if rng.random() < explore_p:
            steps_left = int(rng.integers(1, N + 1))
            remaining = int(rng.integers(1, lots + 1)) * cfg.lot
        else:
            steps_left, remaining = N, cfg.qty
        st = int(rng.integers(FIRST_START, last_start(steps_left * k) + 1))
        env = envs[d]
        obs = env.reset(st, steps_left=steps_left, remaining=remaining)
        s = spec.encode(obs)
        done, total = False, 0.0
        while not done:
            a = agent.act(s, eps, rng)
            obs, c, done, _ = env.step(a)
            s2 = None if done else spec.encode(obs)
            agent.update(s, a, c, s2, alpha)
            total += c
            s = s2
        if steps_left == N and remaining == cfg.qty:
            running.append(total)
        if cps and ep == cps[0]:
            cps.pop(0)
            row = {"episodes": ep, "epsilon": eps,
                   "train_cost_running": float(np.mean(running[-2000:])) if running else np.nan}
            if evaluate is not None:
                row.update(evaluate(agent))
            curve.append(row)
            snaps[ep] = (agent.Q.copy(), agent.N.copy())
            log.info("%s seed %d ep %d: %s", spec.name, seed, ep, {k_: round(v, 3) for k_, v in row.items()
                                                                      if isinstance(v, float)})
    return {"agent": agent, "curve": curve, "snapshots": snaps}
