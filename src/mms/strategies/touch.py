r"""Quoting rules for the replay backtester (stage 4). Each is a callable ``policy(obs) -> (bid, ask)`` in integer
ticks, None meaning "no order on that side". ``obs`` is ``mms.replay.env.ReplayEnv.observe()``.

All five are deliberately simple baselines for stage 5's RL agent:

1. AlwaysTouch: join the best bid and the best ask; when the touch moves, cancel and re-join at the back.
2. TouchSkew: always-touch plus inventory control. A-S's reservation price is r = S - q gamma sigma^2 (T - t)
   (q in lots). Discretised to ticks, n = round(q gamma sigma_hat^2 (T - t) / tick): when long (n > 0) the bid
   is moved n ticks behind the touch (we buy less eagerly) and the ask may improve inside the spread by up to n
   ticks without crossing (we sell more eagerly); mirror image when short. Beyond a threshold |q| >= m lots
   the side that would grow |q| is pulled.
3. TouchQueueAware: join a side only if the queue we would join (displayed touch size) is below
   theta * typical size. Rationale: with a short queue our order is reached by ordinary flow before the next
   sweep, whereas at the back of a long queue we are mostly filled by the sweeps that go through the level,
   just before the price moves against us. An order we already hold at the touch is kept (it has its place).
4. TouchOFIAware: always-touch, but pull the side that the 5 s order-flow imbalance says is about to be swept:
   z > z* (buyers aggressive) pulls the ask, z < -z* pulls the bid. No signal (NaN: nothing traded) = no pull.
5. ReplayMarkoutAwareAS: stage 3's markout-aware A-S (fee-aware A-S plus the markout allowance, no signal),
   quotes kept >= 0.1 USDT from the mid and rounded outward to the tick (bid down, ask up).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from mms.replay.data import TICK
from mms.strategies.flow_aware import FlowAwareAvellanedaStoikov


@dataclass
class AlwaysTouch:
    name: str = "touch"

    def __call__(self, obs):
        return obs["best_bid"], obs["best_ask"]


@dataclass
class TouchSkew:
    max_lots: float = 5.0         # pull the growing side when |q| >= max_lots
    gamma: float = 1e-4           # risk aversion per lot for the skew (0: threshold only)
    name: str = "touch_skew"

    def skew_ticks(self, obs) -> int:
        """n = round(q gamma sigma_hat^2 (T - t) / tick): the A-S reservation-price shift in ticks."""
        q = obs["inventory"]
        return int(round(q * self.gamma * obs["sigma_hat"] ** 2 * obs["time_remaining"] / TICK))

    def __call__(self, obs):
        bb, ba, q = obs["best_bid"], obs["best_ask"], obs["inventory"]
        n = self.skew_ticks(obs)
        bid = bb - max(n, 0) if n >= 0 else min(bb - n, ba - 1)      # short (n < 0): bid improves, never crosses
        ask = ba - min(n, 0) if n <= 0 else max(ba - n, bb + 1)      # long (n > 0): ask improves, never crosses
        if q >= self.max_lots - 1e-9:
            bid = None
        if q <= -self.max_lots + 1e-9:
            ask = None
        return bid, ask


@dataclass
class TouchQueueAware:
    theta: float = 0.5            # join only if displayed touch size < theta * typical size
    name: str = "touch_queue"

    def _side(self, touch, size, typical, live):
        if live is not None and live[0] == touch:
            return touch                                             # keep our place
        if np.isfinite(typical) and size < self.theta * typical:
            return touch
        return None

    def __call__(self, obs):
        typ = obs["typical_size"]
        return (self._side(obs["best_bid"], obs["bid_size"], typ, obs["bid_order"]),
                self._side(obs["best_ask"], obs["ask_size"], typ, obs["ask_order"]))


@dataclass
class TouchOFIAware:
    z_star: float = 1.0           # pull the threatened side when |z| > z_star
    name: str = "touch_ofi"

    def __call__(self, obs):
        z = obs["ofi_z"]
        bid, ask = obs["best_bid"], obs["best_ask"]
        if math.isfinite(z):
            if z > self.z_star:
                ask = None
            elif z < -self.z_star:
                bid = None
        return bid, ask


class ReplayMarkoutAwareAS:
    """Stage 3 markout-aware A-S in the replay. Parameters are fixed per episode from information available at
    its start: sigma = trailing sigma_hat, k = c sigma_hat^d (model (b)), J = the stage 3 markout curve (delta grid
    in USDT as in stage 3, jump sizes in bps of the mid, times the current mid); T = the episode length.
    Inventory is passed in lots, as in stage 3."""

    name = "markout_as"

    def __init__(self, gamma: float, model_b: dict, jump_delta, jump_mean_bps, fee_bps: float,
                 min_distance: float = 0.1):
        self.gamma, self.mb, self.fee_bps, self.min_distance = gamma, model_b, fee_bps, min_distance
        self.jd, self.jm_bps = np.asarray(jump_delta, float), np.asarray(jump_mean_bps, float)
        self.strat = None

    def params(self, sigma: float, s0: float) -> dict:
        k = self.mb["c"] * sigma ** self.mb["d"]
        A = self.mb["a"] * sigma ** self.mb["b"]
        return {"sigma": sigma, "k": k, "A": A, "s0": s0, "jump_delta": tuple(self.jd),
                "jump_mean": tuple(self.jm_bps * s0 / 1e4)}

    def reset(self, obs) -> None:
        p = self.params(obs["sigma_hat"], obs["mid"])
        self.p = p
        self.strat = FlowAwareAvellanedaStoikov(
            gamma=self.gamma, sigma=p["sigma"], k=p["k"], T=obs["time_remaining"], fee_bps=self.fee_bps,
            jump_delta=p["jump_delta"], jump_mean=p["jump_mean"], s_ref=p["s0"], use_signal=False,
            name="markout_aware_as")

    def __call__(self, obs):
        S = obs["mid"]
        o = {"mid": np.array([S]), "inventory": np.array([obs["inventory"]]), "signal": np.array([0.0]),
             "t": obs["t"]}
        bid, ask = self.strat.act(o)
        bid = min(float(bid[0]), S - self.min_distance)
        ask = max(float(ask[0]), S + self.min_distance)
        return int(math.floor(bid / TICK + 1e-9)), int(math.ceil(ask / TICK - 1e-9))
