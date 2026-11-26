r"""Stage 5 environment: the replay backtester behind a reset(seed) / step(action) API.

    env = ReplayRLEnv(split="calibration")          # training episodes; split="test" for evaluation
    obs = env.reset(seed)                           # seed -> a day of the split and a random 1-hour start
    obs, reward, done, info = env.step(action)      # action: int 0..15, or a (bid_tick, ask_tick) pair

Episodes. A seed draws (with numpy's default_rng(seed)) one day of the split uniformly and a start time
uniformly on the 100 ms grid between 01:00 and 23:00 UTC, so the episode (1 hour) ends by midnight and the
trailing features (30-minute sigma_hat and typical size) are warm. Calibration days are for training;
test days are kept separate and only used for evaluation.

Actions. For each side one of {touch, 1 tick behind, 3 ticks behind, off} (``LEVELS``), 16 joint actions:
action = 4 * bid_level + ask_level. "Touch" is the best price on that side at decision time; "behind" moves
away from the mid (bid lower, ask higher). Prices requested at the touch keep the order's queue place while
the touch does not move (see ``ReplayEnv``). A (bid_tick, ask_tick) pair is also accepted, so any baseline
can run through the same API.

Observation (``vector`` gives them as a float array in this order):
  inventory        lots (positive = long)
  time_remaining   seconds left in the episode
  queue_bid        queue ahead of our bid / typical size if we have a bid whose queue is known; otherwise the
                   displayed size at the best bid / typical size (the queue we would join). inf when our order
                   sits behind the touch (its queue is unseen).
  queue_ask        the same for the ask
  spread_ticks     best ask - best bid, in ticks
  ofi_z            5 s order-flow imbalance in units of its calibration std (NaN if nothing traded)
  sigma_hat        trailing volatility (USDT / sqrt(s))
The full replay observation (book, our orders, ...) is in ``obs["raw"]``.

Reward. Per step: change in mark-to-market wealth (cash + inventory at the mid, USDT, fees included) minus
the inventory penalty (gamma/2) * lot * q_lots^2 * sigma_hat^2 * dt, with gamma = 1e-4 per lot
(``mms.replay.env.inventory_penalty``). Without the penalty the rewards sum to the episode P&L.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from mms.replay.data import BUCKETS_PER_S, load_day
from mms.replay.env import ReplayConfig, ReplayEnv

LEVELS = ("touch", "1 behind", "3 behind", "off")
OFFSETS = (0, 1, 3, None)
N_ACTIONS = 16
OBS_KEYS = ("inventory", "time_remaining", "queue_bid", "queue_ask", "spread_ticks", "ofi_z", "sigma_hat")
FIRST_START_S = 3600            # 01:00 UTC
LAST_END_S = 86400              # 24:00 UTC


def split_days(split: str) -> list[str]:
    from mms.replay.data import REPO
    sel = pd.read_csv(REPO / "results" / "stage2" / "selected_days.csv")
    days = sel.loc[sel["split"] == split, "day"].tolist()
    if not days:
        raise ValueError(f"no days for split {split!r} (use 'calibration' or 'test')")
    return days


def decode_action(action: int) -> tuple[int | None, int | None]:
    """action -> (bid offset, ask offset) in ticks away from the touch, None = off."""
    if not 0 <= int(action) < N_ACTIONS:
        raise ValueError(f"action must be in 0..{N_ACTIONS - 1}")
    return OFFSETS[int(action) // 4], OFFSETS[int(action) % 4]


def encode_action(bid_level: str, ask_level: str) -> int:
    return 4 * LEVELS.index(bid_level) + LEVELS.index(ask_level)


def action_to_prices(action: int, raw_obs: dict) -> tuple[int | None, int | None]:
    ob, oa = decode_action(action)
    bid = None if ob is None else raw_obs["best_bid"] - ob
    ask = None if oa is None else raw_obs["best_ask"] + oa
    return bid, ask


def rl_observation(raw: dict) -> dict:
    typ = raw["typical_size"]

    def queue(order, size):
        if order is not None:
            return order[1] / typ                     # inf stays inf (unseen queue behind the touch)
        return size / typ

    obs = {"inventory": raw["inventory"], "time_remaining": raw["time_remaining"],
           "queue_bid": queue(raw["bid_order"], raw["bid_size"]), "queue_ask": queue(raw["ask_order"], raw["ask_size"]),
           "spread_ticks": raw["spread_ticks"], "ofi_z": raw["ofi_z"], "sigma_hat": raw["sigma_hat"]}
    obs["vector"] = np.array([obs[k] for k in OBS_KEYS], dtype=float)
    obs["raw"] = raw
    return obs


class ReplayRLEnv:
    def __init__(self, split: str = "calibration", config: ReplayConfig = ReplayConfig(),
                 days: Sequence[str] | None = None):
        self.split = split
        self.days = list(days) if days is not None else split_days(split)
        self.cfg = config
        self.env = None

    def episode_start(self, seed: int) -> tuple[str, int]:
        """(day, start bucket) for a seed: uniform day, uniform start on the 100 ms grid in [01:00, 23:00]."""
        rng = np.random.default_rng(int(seed))
        day = self.days[int(rng.integers(len(self.days)))]
        lo = FIRST_START_S * BUCKETS_PER_S
        hi = LAST_END_S * BUCKETS_PER_S - self.cfg.n_steps          # last allowed start
        return day, int(rng.integers(lo, hi + 1))

    def reset(self, seed: int | None = None, day: str | None = None, start: int | None = None) -> dict:
        """reset(seed) for a random episode, or reset(day=..., start=...) for a specific one."""
        if seed is not None:
            day, start = self.episode_start(seed)
        if day is None or start is None:
            raise ValueError("give a seed, or a day and a start")
        if day not in self.days:
            raise ValueError(f"{day} is not a {self.split} day")
        self.env = ReplayEnv(load_day(day), self.cfg)
        self._raw = self.env.reset(start)
        self.day, self.start = day, start
        return rl_observation(self._raw)

    def step(self, action):
        if isinstance(action, (int, np.integer)):
            prices = action_to_prices(int(action), self._raw)
        else:
            prices = tuple(action)
        raw, reward, done, info = self.env.step(prices)
        self._raw = raw
        info = {**info, "prices": prices}
        return (None if done else rl_observation(raw)), reward, done, info

    def result(self):
        return self.env.result()

