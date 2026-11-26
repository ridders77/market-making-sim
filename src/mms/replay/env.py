r"""Replay backtester: our small orders placed into the recorded BTCUSDT book and trades, 100 ms at a time.

API (one episode at a time; stage 5's RL wrapper is ``mms.replay.rl_env``):

    env = ReplayEnv(day_data, ReplayConfig(latency_ms=100, cancel_rule="proportional"))
    obs = env.reset(start)                       # start = decision index (bucket) of the episode's first step
    obs, reward, done, info = env.step((bid_tick, ask_tick))   # None = no order on that side

One step, at decision time t_i (index i):
1. The strategy has seen the book in force at t_i and the trades before t_i (``obs``) and asks for a bid and
   an ask price (integer ticks) or None. The request takes effect at t_i + latency (latency is a multiple of
   100 ms): cancels and new orders alike.
2. Requests due at t_i are applied: a side whose requested price equals its live order's price keeps the
   order (and its queue place); otherwise the live order is cancelled and, if a price is requested, a new
   order joins at the back of that level (queue ahead = displayed size if it is the best price, 0 if it
   improves on the best, unknown = inf if it is behind the best; see ``mms.replay.queue``). Post-only: a bid
   at or above the best ask (ask at or below the best bid) is rejected and counted. Order size = one lot,
   cut so that a fill cannot take |inventory| beyond the limit (10 lots); at the limit that side is rejected.
3. Bucket i (t_i to t_{i+1}) is replayed against the live orders (``update_order``: trades, crossing,
   cancellations under the chosen rule, size cap).
4. Fills are booked at our limit price with the maker fee c on notional: buy x BTC at p: q += x,
   cash -= x p (1 + c); sell: q -= x, cash += x p (1 - c).
5. Mark to market at the mid at t_{i+1}: W = cash + q * mid. reward = W_{i+1} - W_i - penalty, with the
   inventory penalty (gamma / 2) * lot * (q / lot)^2 * sigma_hat^2 * dt (see ``inventory_penalty``).
   Without the penalty the rewards sum exactly to the episode's P&L (tested).

The episode ends after ``episode_s`` seconds; the P&L marks any remaining inventory at the final mid (no
liquidation cost), as in stages 1-3. Strategies see the recorded book, which does not contain our own orders.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

import numpy as np

from mms.replay.data import BUCKET_MS, TICK, ReplayDay
from mms.replay.queue import ASK, BID, BucketBook, Order, RULES, update_order

_EMPTY_P = np.array([], dtype=np.int64)
_EMPTY_Q = np.array([], dtype=float)
_EMPTY_S = np.array([], dtype=np.int8)


@dataclass(frozen=True)
class ReplayConfig:
    latency_ms: int = 100
    cancel_rule: str = "proportional"
    lot: float = 0.01                 # BTC per order
    fee_bps: float = 0.0              # maker fee (negative = rebate)
    inv_limit_lots: float = 10.0
    unseen_depth: str = "back"        # queue ahead when joining behind the touch: "back" (inf) or "front" (0)
    episode_s: float = 3600.0
    gamma: float = 1e-4               # risk aversion per lot, for the reward's inventory penalty
    record: bool = False              # keep per-step arrays (for charts)
    queue_frac: float = 1.0           # stage 5 what-if: share of the displayed queue ahead of us when we join the touch

    def __post_init__(self):
        if self.latency_ms % BUCKET_MS or self.latency_ms < 0:
            raise ValueError("latency must be a non-negative multiple of 100 ms")
        if self.cancel_rule not in RULES:
            raise ValueError(f"cancel_rule must be one of {RULES}")
        if self.unseen_depth not in ("back", "front"):
            raise ValueError("unseen_depth must be 'back' or 'front'")
        if not 0.0 <= self.queue_frac <= 1.0:
            raise ValueError("queue_frac must be in [0, 1]")

    @property
    def n_steps(self) -> int:
        return int(round(self.episode_s * 1000 / BUCKET_MS))

    @property
    def dt(self) -> float:
        return BUCKET_MS / 1000.0


def inventory_penalty(q_btc: float, lot: float, gamma: float, sigma: float, dt: float) -> float:
    r"""Per-step inventory penalty matched to the A-S risk aversion gamma (per lot).

    A-S maximises E[-exp(-gamma W)] with W counted per lot (stage 3: gamma is risk aversion per lot, i.e. on
    wealth measured in USDT / lot). For small risks this is mean - (gamma/2) variance. Holding q lots for dt
    adds variance (q lot)^2 sigma^2 dt in USDT^2, i.e. q^2 sigma^2 dt in (USDT/lot)^2. So the certainty
    equivalent in USDT loses lot * (gamma/2) q^2 sigma^2 dt per step:

        penalty = (gamma / 2) * lot * (q_btc / lot)^2 * sigma^2 * dt.

    Summed over an episode this is the running version of A-S's (gamma/2) q^2 sigma^2 (T - t) term."""
    q_lots = q_btc / lot
    return 0.5 * gamma * lot * q_lots * q_lots * sigma * sigma * dt


@dataclass
class EpisodeResult:
    day: str
    start: int
    pnl: float                     # USDT, fee included, inventory marked at the final mid
    cash: float
    q: float                       # final inventory, BTC
    notional: float                # USDT traded
    fill_qty: float                # BTC filled (both sides)
    n_fills: int                   # fill events (a partial fill counts once per bucket)
    max_abs_q: float               # BTC
    fills: dict                    # arrays: step, side, price (ticks), qty, mid0, how
    n_rejected_post_only: int
    n_rejected_limit: int
    steps_quoted: np.ndarray       # [bid, ask]: steps with a live order
    steps_at_touch: np.ndarray     # [bid, ask]: steps with a live order at the best price
    penalty: float                 # sum of inventory penalties (USDT)
    fee_bps: float
    lot: float
    record: dict = field(default_factory=dict)

    def pnl_with_fee(self, fee_bps: float) -> float:
        """Fees change only cash, not fills: P&L(f) = P&L(f0) - (f - f0) * notional (exact)."""
        return self.pnl - (fee_bps - self.fee_bps) * 1e-4 * self.notional


class ReplayEnv:
    def __init__(self, data: ReplayDay, config: ReplayConfig = ReplayConfig()):
        self.d = data
        self.cfg = config
        self._L = config.latency_ms // BUCKET_MS
        # plain Python lists: element access in the step loop is several times faster than numpy scalars.
        # Built once per day and cached on the data object.
        lists = getattr(data, "_lists", None)
        if lists is None:
            lists = {k: getattr(data, k).tolist() for k in ("bid", "ask", "bid_qty", "ask_qty", "bid_qty_min",
                                                          "ask_qty_min", "mid", "sell_min", "buy_max", "tr_start")}
            data._lists = lists
        self._bid, self._ask = lists["bid"], lists["ask"]
        self._bq, self._aq = lists["bid_qty"], lists["ask_qty"]
        self._bqmin, self._aqmin = lists["bid_qty_min"], lists["ask_qty_min"]
        self._mid = lists["mid"]
        self._sell_min, self._buy_max = lists["sell_min"], lists["buy_max"]
        self._trs = lists["tr_start"]
        self.i = None

    # ------------------------------------------------------------------ reset / observe
    def reset(self, start: int) -> dict:
        cfg, d = self.cfg, self.d
        end = start + cfg.n_steps
        if start < 1 or end > d.n_buckets:
            raise ValueError(f"episode [{start}, {end}) does not fit in the day (1 .. {d.n_buckets})")
        if np.any(d.bid[start - 1:end] < 0) or not np.isfinite(d.sigma_hat[start]):
            raise ValueError("episode overlaps the start of the day without a book or sigma_hat (warm-up)")
        self.start, self.end, self.i = start, end, start
        self.cash, self.q = 0.0, 0.0
        self.orders = {BID: None, ASK: None}
        self.pending = deque()
        self.fee = cfg.fee_bps * 1e-4
        self.limit = cfg.inv_limit_lots * cfg.lot
        self.notional = self.fill_qty = self.max_abs_q = self.penalty = 0.0
        self.n_rej_post = self.n_rej_limit = 0
        self.fl = {k: [] for k in ("step", "side", "price", "qty", "mid0", "how")}
        self.steps_quoted = np.zeros(2, dtype=np.int64)
        self.steps_at_touch = np.zeros(2, dtype=np.int64)
        self.W = 0.0
        if cfg.record:
            self.rec = {k: [] for k in ("best_bid", "best_ask", "bid_px", "ask_px", "Q_bid", "Q_ask", "q", "mid",
                                        "bid_size", "ask_size")}
        return self.observe()

    def observe(self) -> dict:
        """What a strategy may use at decision time t_i: the book in force at t_i (row i - 1), features
        computed from data before t_i, our inventory and our own live orders."""
        i, r, d = self.i, self.i - 1, self.d
        ob, oa = self.orders[BID], self.orders[ASK]
        return {
            "i": i, "t": (i - self.start) * self.cfg.dt, "time_remaining": (self.end - i) * self.cfg.dt,
            "best_bid": self._bid[r], "best_ask": self._ask[r], "bid_size": self._bq[r], "ask_size": self._aq[r],
            "mid": self._mid[r], "spread_ticks": self._ask[r] - self._bid[r],
            "inventory": self.q / self.cfg.lot, "q_btc": self.q,
            "ofi_z": float(d.ofi_z[i]), "sigma_hat": float(d.sigma_hat[i]), "typical_size": float(d.typical_size[i]),
            "bid_order": None if ob is None else (ob.price, ob.Q), "ask_order": None if oa is None else (oa.price, oa.Q),
            "lot": self.cfg.lot,
        }

    # ------------------------------------------------------------------ orders
    def _apply(self, side: int, price) -> None:
        live = self.orders[side]
        if live is not None and price is not None and live.price == price:
            return                                     # same price: keep the order and its place
        self.orders[side] = None                       # cancel (if any)
        if price is None:
            return
        r = self.i - 1
        best, opp = (self._bid[r], self._ask[r]) if side == BID else (self._ask[r], self._bid[r])
        if (side == BID and price >= opp) or (side == ASK and price <= opp):
            self.n_rej_post += 1                       # post-only: would cross the book
            return
        room = self.limit - side * self.q              # BTC we may still buy (bid) or sell (ask)
        qty = min(self.cfg.lot, room)
        if qty <= 1e-12:
            self.n_rej_limit += 1
            return
        if price == best:
            Q = self.cfg.queue_frac * (self._bq[r] if side == BID else self._aq[r])
        elif (side == BID and price > best) or (side == ASK and price < best):
            Q = 0.0                                    # improves on the best price: alone at a new level
        else:
            Q = math.inf if self.cfg.unseen_depth == "back" else 0.0
        self.orders[side] = Order(side=side, price=int(price), qty=qty, Q=Q)

    # ------------------------------------------------------------------ step
    def step(self, action) -> tuple[dict, float, bool, dict]:
        if self.i is None or self.i >= self.end:
            raise RuntimeError("call reset() before step(), and not after the episode is done")
        cfg, i = self.cfg, self.i
        bid_px, ask_px = action
        self.pending.append((i + self._L, bid_px, ask_px))
        while self.pending and self.pending[0][0] <= i:
            _, b, a = self.pending.popleft()
            self._apply(BID, b)
            self._apply(ASK, a)

        prev, r = i - 1, i
        fills_now = []
        for side in (BID, ASK):
            o = self.orders[side]
            if o is None:
                continue
            self.steps_quoted[0 if side == BID else 1] += 1
            if side == BID:
                if o.price == self._bid[prev]:
                    self.steps_at_touch[0] += 1
                relevant = self._sell_min[i] <= o.price
                bk = BucketBook(self._bid[prev], self._bq[prev], self._bid[r], self._bqmin[r], self._ask[r])
            else:
                if o.price == self._ask[prev]:
                    self.steps_at_touch[1] += 1
                relevant = self._buy_max[i] >= o.price
                bk = BucketBook(self._ask[prev], self._aq[prev], self._ask[r], self._aqmin[r], self._bid[r])
            if relevant:
                a, b = self._trs[i], self._trs[i + 1]
                res = update_order(o, self.d.tr_price[a:b], self.d.tr_qty[a:b], self.d.tr_side[a:b], bk, cfg.cancel_rule)
            else:
                res = update_order(o, _EMPTY_P, _EMPTY_Q, _EMPTY_S, bk, cfg.cancel_rule)
            if res.filled > 0:
                fills_now.append((side, o.price, res.filled, res.how))
            if o.qty <= 1e-12:
                self.orders[side] = None
        mid0 = self._mid[prev]
        for side, p, x, how in fills_now:
            px = p * TICK
            self.cash -= side * x * px + self.fee * x * px
            self.q += side * x
            self.notional += x * px
            self.fill_qty += x
            for k, v in zip(("step", "side", "price", "qty", "mid0", "how"), (i, side, p, x, mid0, how)):
                self.fl[k].append(v)
        if abs(self.q) > self.max_abs_q:
            self.max_abs_q = abs(self.q)

        mid1 = self._mid[r]
        W1 = self.cash + self.q * mid1
        pen = inventory_penalty(self.q, cfg.lot, cfg.gamma, float(self.d.sigma_hat[i]), cfg.dt)
        self.penalty += pen
        reward = W1 - self.W - pen
        self.W = W1
        if cfg.record:
            ob, oa = self.orders[BID], self.orders[ASK]
            for k, v in (("best_bid", self._bid[r]), ("best_ask", self._ask[r]), ("bid_px", ob.price if ob else np.nan),
                         ("ask_px", oa.price if oa else np.nan), ("Q_bid", ob.Q if ob else np.nan),
                         ("Q_ask", oa.Q if oa else np.nan), ("q", self.q), ("mid", mid1),
                         ("bid_size", self._bq[r]), ("ask_size", self._aq[r])):
                self.rec[k].append(v)
        self.i = i + 1
        done = self.i == self.end
        info = {"fills": fills_now, "penalty": pen, "mtm": W1}
        return (None if done else self.observe()), reward, done, info

    # ------------------------------------------------------------------ result
    def result(self) -> EpisodeResult:
        if self.i != self.end:
            raise RuntimeError("episode not finished")
        fills = {k: np.asarray(v) for k, v in self.fl.items()}
        return EpisodeResult(
            day=self.d.day, start=self.start, pnl=self.W, cash=self.cash, q=self.q, notional=self.notional,
            fill_qty=self.fill_qty, n_fills=len(self.fl["step"]), max_abs_q=self.max_abs_q, fills=fills,
            n_rejected_post_only=self.n_rej_post, n_rejected_limit=self.n_rej_limit,
            steps_quoted=self.steps_quoted.copy(), steps_at_touch=self.steps_at_touch.copy(), penalty=self.penalty,
            fee_bps=self.cfg.fee_bps, lot=self.cfg.lot,
            record={k: np.asarray(v) for k, v in self.rec.items()} if self.cfg.record else {})


def run_episode(policy, data: ReplayDay, start: int, config: ReplayConfig) -> EpisodeResult:
    """Run one episode: ``policy(obs) -> (bid_tick | None, ask_tick | None)``."""
    env = ReplayEnv(data, config)
    obs = env.reset(start)
    if hasattr(policy, "reset"):
        policy.reset(obs)
    done = False
    while not done:
        obs, _, done, _ = env.step(policy(obs))
    return env.result()
