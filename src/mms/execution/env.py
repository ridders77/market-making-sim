r"""Optimal execution in the replay: buy (or sell) Q BTC within H seconds, deciding every few seconds.

This is the stage 5 environment (Nevmyvaka, Feng & Kearns 2006, "Reinforcement Learning for Optimized Trade
Execution", replayed on BTCUSDT with stage 4's queue model).

Episode. At bucket ``start`` (decision time t_0) the trader must buy (side = +1) or sell (side = -1) Q BTC
by t_0 + H. Decisions are taken every ``decision_s`` seconds: N = H / decision_s steps. At each decision the
trader picks one action for the *whole remaining quantity*:

    0 "cross"     take it now: a marketable (taker) order for everything left; the episode ends
    1 "touch"     rest a limit order at the best price on our side (best bid for a buyer)
    2 "1 behind"  rest it 1 tick behind the touch (a buyer bids best bid - 1 tick)
    3 "3 behind"  rest it 3 ticks behind

Queue rule (our improvement over Nevmyvaka et al., who re-placed the order every step). An order keeps its
place in the queue when the action does not ask for a different price:
  * if the live order is already at the price the action asks for, it is kept (whatever the previous action);
  * if the action is unchanged and the live order is now *closer to the touch* than the action's price
    (e.g. "1 behind" and the touch fell onto our level), it is kept: we got there first;
  * otherwise (new action at another price, or the market moved away from the order) it is cancelled and a new
    order joins at the back of the queue of the new price.
Orders and cancels take effect ``latency_ms`` after the decision (stage 4: 100 ms). Joining the queue uses
stage 4's rule (queue ahead = displayed size at the touch, unknown = inf behind the touch) times
``queue_frac``; the queue then evolves by ``mms.replay.queue.update_order`` with the size cap (a big order is
only filled by as much volume as could have reached it).

Taker fills (``taker_fill``). A taker order executes against the best opposite quote in force when it arrives
(after the latency). Only the best quote's size is recorded; any quantity beyond it is assumed to fill at
successively worse ticks, each holding ``depth_frac`` (0.5) of the displayed best size, i.e. depth beyond the
touch is assumed thinner than the touch. These episodes are counted ("beyond display").

Deadline. At t_0 + H any remainder is cancelled and bought (sold) as a taker against the quote in force then.

Cost: implementation shortfall against the arrival mid m_0 (the mid in force at t_0), in bps of the order's
notional Q m_0, fees included. For fills (x_j BTC at price p_j, fee f_j: maker or taker rate):

    IS = 1e4 / (Q m_0) * sum_j [ side * x_j (p_j - m_0) + f_j x_j p_j ]

(a buyer pays p_j (1 + f_j) per BTC against a benchmark of m_0; a seller receives p_j (1 - f_j)). Positive is
a cost. Fees only enter through the passive and taker notionals, so IS can be repriced exactly for any fees.

Per-step cost for learning (shaped). With R_t the BTC still to trade at decision t and m_t the mid then,
let Phi_t = side * R_t (m_t - m_0) (the paper loss on what is left). The step cost
    c_t = (IS of the fills in step t) + 1e4/(Q m_0) * (Phi_{t+1} - Phi_t)
sums over the episode to IS exactly (Phi_0 = 0, and Phi = 0 once everything is traded). It equals the fills'
cost against the *next* mid plus side * R_t (m_{t+1} - m_t): the trader is charged for the price moving while
it still has inventory left to trade, so a signal that predicts the move can be credited at once.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from mms.replay.data import BUCKET_MS, TICK, ReplayDay
from mms.replay.queue import ASK, BID, BucketBook, Order, RULES, update_order

BUY, SELL = BID, ASK
ACTIONS = ("cross", "touch", "1 behind", "3 behind")
OFFSETS = (None, 0, 1, 3)          # ticks behind the touch; None = cross
CROSS, TOUCH, BEHIND1, BEHIND3 = 0, 1, 2, 3
N_ACTIONS = 4
_EP = np.array([], dtype=np.int64)
_EQ = np.array([], dtype=float)
_ES = np.array([], dtype=np.int8)
EPS = 1e-9


@dataclass(frozen=True)
class ExecConfig:
    side: int = BUY
    qty: float = 0.5                   # BTC to trade
    horizon_s: float = 300.0           # H
    decision_s: float = 10.0
    latency_ms: int = 100
    cancel_rule: str = "proportional"
    maker_bps: float = 2.0             # retail: Binance USD-M regular tier
    taker_bps: float = 5.0
    lot: float = 0.01
    depth_frac: float = 0.5            # depth per tick beyond the best quote, as a fraction of the best size
    size_cap: bool = True
    queue_frac: float = 1.0            # fraction of the displayed queue ahead of us when we join the touch

    def __post_init__(self):
        if self.side not in (BUY, SELL):
            raise ValueError("side must be +1 (buy) or -1 (sell)")
        if self.cancel_rule not in RULES:
            raise ValueError(f"cancel_rule must be one of {RULES}")
        if self.latency_ms % BUCKET_MS or self.latency_ms < 0:
            raise ValueError("latency must be a non-negative multiple of 100 ms")
        if abs(self.qty / self.lot - round(self.qty / self.lot)) > 1e-9 or self.qty <= 0:
            raise ValueError("qty must be a positive whole number of lots")
        k = self.decision_s * 1000 / BUCKET_MS
        if abs(k - round(k)) > 1e-9 or k < 1:
            raise ValueError("decision_s must be a positive multiple of 100 ms")
        if abs(self.horizon_s / self.decision_s - round(self.horizon_s / self.decision_s)) > 1e-9:
            raise ValueError("horizon_s must be a whole number of decision steps")
        if self.latency_ms >= self.decision_s * 1000:
            raise ValueError("latency must be shorter than the decision interval")
        if not 0 <= self.queue_frac <= 1:
            raise ValueError("queue_frac must be in [0, 1]")

    @property
    def n_steps(self) -> int:
        return int(round(self.horizon_s / self.decision_s))

    @property
    def k(self) -> int:
        """buckets per decision step"""
        return int(round(self.decision_s * 1000 / BUCKET_MS))

    @property
    def L(self) -> int:
        return self.latency_ms // BUCKET_MS


def taker_fill(qty: float, best_tick: int, best_size: float, side: int, depth_frac: float = 0.5
               ) -> tuple[list[tuple[int, float]], bool]:
    """Fills of a taker order for ``qty`` BTC against the best opposite quote (``best_tick``, ``best_size``).

    Up to the displayed size fills at the best price. Beyond it the depth is not recorded; we assume each
    further tick holds depth_frac * best_size (thinner than the touch, a conservative assumption), so the
    excess fills in tranches of that size at 1, 2, 3, ... ticks worse (higher for a buyer, lower for a seller).
    Returns ([(price_tick, qty), ...], whether the order went beyond the displayed size)."""
    if qty <= EPS:
        return [], False
    if not best_size > 0:
        raise ValueError("best quote has no size")
    first = min(qty, best_size)
    out = [(int(best_tick), first)]
    rest = qty - first
    tranche = depth_frac * best_size
    j = 0
    while rest > EPS:
        j += 1
        x = min(rest, tranche)
        out.append((int(best_tick + side * j), x))
        rest -= x
    return out, len(out) > 1


def shortfall_bps(side: int, prices: np.ndarray, qtys: np.ndarray, fees_bps: np.ndarray, m0: float,
                  Q: float) -> float:
    """Implementation shortfall in bps of Q m0 for fills (USDT prices, BTC quantities, fee per fill in bps)."""
    prices, qtys, fees = (np.asarray(a, float) for a in (prices, qtys, fees_bps))
    usdt = np.sum(side * qtys * (prices - m0) + fees * 1e-4 * qtys * prices)
    return float(1e4 * usdt / (Q * m0))


@dataclass
class ExecResult:
    day: str
    start: int
    side: int
    m0: float
    qty: float
    cost_bps: float                 # IS, fees included
    price_usdt: float               # sum side * x (p - m0), USDT (no fees)
    passive_qty: float
    passive_notional: float
    taker_qty: float                # crosses + TWAP slices + deadline clean-up
    taker_notional: float
    cleanup_qty: float              # bought at the deadline
    beyond_display: int             # taker orders larger than the displayed best size (0/1/2...)
    beyond_cleanup: bool            # the deadline clean-up was larger than the displayed size
    n_post_reject: int
    steps_used: int
    fills: dict                     # arrays: bucket, price (ticks), qty, how, mid_pre
    actions: list = field(default_factory=list)
    decisions: list = field(default_factory=list)   # per step (state dict), if recorded

    def cost_at(self, maker_bps: float, taker_bps: float) -> float:
        """IS for other fee rates: fills do not depend on the fee (exact repricing)."""
        usdt = self.price_usdt + 1e-4 * (maker_bps * self.passive_notional + taker_bps * self.taker_notional)
        return 1e4 * usdt / (self.qty * self.m0)


class ExecutionEnv:
    """One execution episode at a time on one day of replay data (see the module docstring)."""

    def __init__(self, data: ReplayDay, config: ExecConfig = ExecConfig()):
        self.d = data
        self.cfg = config

    # ------------------------------------------------------------------ reset
    def reset(self, start: int, steps_left: int | None = None, remaining: float | None = None) -> dict:
        """Start an episode at decision bucket ``start``. ``steps_left`` and ``remaining`` (BTC) default to a
        full episode (N steps, Q BTC); smaller values give a shorter "exploring start" used in training, whose
        cost is still in bps of Q m_0 with m_0 the mid at ``start``."""
        cfg, d = self.cfg, self.d
        self.N = cfg.n_steps if steps_left is None else int(steps_left)
        self.R = cfg.qty if remaining is None else float(remaining)
        if not 1 <= self.N <= cfg.n_steps or not EPS < self.R <= cfg.qty + EPS:
            raise ValueError("steps_left must be in 1..N and remaining in (0, Q]")
        end = start + self.N * cfg.k
        if start < 1 or end + 1 > d.n_buckets:
            raise ValueError(f"episode [{start}, {end}] does not fit in the day")
        if np.any(d.bid[start - 1:end] < 0) or not np.isfinite(d.sigma_hat[start]):
            raise ValueError("episode overlaps the start of the day without a book or sigma_hat")
        self.start, self.end = start, end
        sl = slice(start - 1, end + 1)          # local index j_loc = j - (start - 1)
        self._o = start - 1
        self._bid, self._ask = d.bid[sl].tolist(), d.ask[sl].tolist()
        self._bq, self._aq = d.bid_qty[sl].tolist(), d.ask_qty[sl].tolist()
        self._bqmin, self._aqmin = d.bid_qty_min[sl].tolist(), d.ask_qty_min[sl].tolist()
        self._mid = d.mid[sl].tolist()
        self._smin, self._bmax = d.sell_min[sl].tolist(), d.buy_max[sl].tolist()
        self.m0 = self._mid[0]
        self.scale = 1e4 / (cfg.qty * self.m0)
        self.step_idx = 0
        self.i = start
        self.order: Order | None = None
        self.prev_action: int | None = None
        self.price_usdt = self.passive_notional = self.taker_notional = 0.0
        self.passive_qty = self.taker_qty = self.cleanup_qty = 0.0
        self.beyond = self.n_post_reject = 0
        self.beyond_cleanup = False
        self.fl = {k: [] for k in ("bucket", "price", "qty", "how", "mid_pre")}
        self.actions = []
        self.done = False
        self.phi = 0.0                           # Phi_0 = side * R (m_0 - m_0) = 0
        return self.observe()

    # ------------------------------------------------------------------ observation
    def observe(self) -> dict:
        """What the trader knows at the decision time t_i: the book in force (row i - 1), features computed
        from data before t_i, the remaining quantity, the steps left and its own live order."""
        i, d, s = self.i, self.d, self.cfg.side
        r = i - 1 - self._o
        o = self.order
        own_size = self._bq[r] if s == BUY else self._aq[r]
        return {
            "i": i, "step": self.step_idx, "steps_left": self.N - self.step_idx, "n_steps": self.cfg.n_steps,
            "remaining": self.R, "remaining_frac": self.R / self.cfg.qty,
            "best_bid": self._bid[r], "best_ask": self._ask[r], "mid": self._mid[r],
            "own_size": own_size, "typical_size": float(d.typical_size[i]),
            "ofi_z": float(d.ofi_z[i]), "sigma_hat": float(d.sigma_hat[i]), "side": s,
            "order": None if o is None else (o.price, o.Q, o.qty), "prev_action": self.prev_action,
        }

    # ------------------------------------------------------------------ helpers
    def target_price(self, action: int, r: int) -> int:
        """Limit price (ticks) for a passive action, from the book row r (local index)."""
        off = OFFSETS[action]
        return self._bid[r] - off if self.cfg.side == BUY else self._ask[r] + off

    def keeps(self, action: int, target: int) -> bool:
        """The queue rule (module docstring): is the live order kept for this action?"""
        o = self.order
        if o is None or action == CROSS:
            return False
        if o.price == target:
            return True
        closer = (o.price > target) if self.cfg.side == BUY else (o.price < target)
        return action == self.prev_action and closer

    def _book_fill(self, x: float, price_tick: int, how: str, j: int, maker: bool) -> None:
        s, px = self.cfg.side, price_tick * TICK
        self.price_usdt += s * x * (px - self.m0)
        if maker:
            self.passive_qty += x
            self.passive_notional += x * px
        else:
            self.taker_qty += x
            self.taker_notional += x * px
        self.R -= x
        if self.R < EPS:
            self.R = 0.0
        self._step_fee_usdt += (self.cfg.maker_bps if maker else self.cfg.taker_bps) * 1e-4 * x * px
        self._step_price_usdt += s * x * (px - self.m0)
        for k, v in zip(("bucket", "price", "qty", "how", "mid_pre"), (j, price_tick, x, how, self._mid[j - 1 - self._o])):
            self.fl[k].append(v)

    def _take(self, qty: float, j: int, how: str) -> None:
        """Taker order arriving at bucket j: executes against the book in force at t_j (row j - 1)."""
        r = j - 1 - self._o
        s = self.cfg.side
        best, size = (self._ask[r], self._aq[r]) if s == BUY else (self._bid[r], self._bq[r])
        fills, beyond = taker_fill(min(qty, self.R), best, size, s, self.cfg.depth_frac)
        self.beyond += int(beyond)
        if how == "cleanup":
            self.beyond_cleanup = beyond
        for p, x in fills:
            self._book_fill(x, p, how, j, maker=False)

    def _place(self, price: int, j: int) -> None:
        """New passive order for the remaining quantity at ``price``, arriving at bucket j (book row j - 1)."""
        r = j - 1 - self._o
        s = self.cfg.side
        best, opp, size = ((self._bid[r], self._ask[r], self._bq[r]) if s == BUY
                           else (self._ask[r], self._bid[r], self._aq[r]))
        if (s == BUY and price >= opp) or (s == SELL and price <= opp):
            self.n_post_reject += 1            # post-only: it would cross the book now
            return
        if price == best:
            Q = self.cfg.queue_frac * size
        elif (s == BUY and price > best) or (s == SELL and price < best):
            Q = 0.0
        else:
            Q = math.inf                       # behind the touch: unseen depth, "back" convention
        self.order = Order(side=s, price=int(price), qty=self.R, Q=Q)

    def _replay(self, j: int) -> None:
        o = self.order
        if o is None:
            return
        jl, pl = j - self._o, j - 1 - self._o
        s = self.cfg.side
        if s == BUY:
            bk = BucketBook(self._bid[pl], self._bq[pl], self._bid[jl], self._bqmin[jl], self._ask[jl], self._aq[jl])
            relevant = self._smin[jl] <= o.price
        else:
            bk = BucketBook(self._ask[pl], self._aq[pl], self._ask[jl], self._aqmin[jl], self._bid[jl], self._bq[jl])
            relevant = self._bmax[jl] >= o.price
        if relevant:
            a, b = self.d.tr_start[j], self.d.tr_start[j + 1]
            res = update_order(o, self.d.tr_price[a:b], self.d.tr_qty[a:b], self.d.tr_side[a:b], bk,
                               self.cfg.cancel_rule, size_cap=self.cfg.size_cap)
        else:
            res = update_order(o, _EP, _EQ, _ES, bk, self.cfg.cancel_rule, size_cap=self.cfg.size_cap)
        if res.filled > 0:
            self._book_fill(min(res.filled, self.R), o.price, res.how, j, maker=True)
        if o.qty <= EPS or self.R <= 0:
            self.order = None

    # ------------------------------------------------------------------ step
    def step(self, action: int, take_qty: float = 0.0, passive: bool = True) -> tuple[dict | None, float, bool, dict]:
        """One decision step. ``action`` in 0..3 (see ACTIONS). For baselines: ``passive=False`` with a
        ``take_qty`` sends a taker order for that quantity and no passive order (TWAP slices).
        Returns (obs, shaped cost in bps of Q m_0, done, info)."""
        if self.done:
            raise RuntimeError("episode is over: call reset()")
        cfg, i = self.cfg, self.i
        r = i - 1 - self._o
        self._step_fee_usdt = self._step_price_usdt = 0.0
        R_t = self.R
        if passive:
            if not 0 <= int(action) < N_ACTIONS:
                raise ValueError("action must be in 0..3")
            action = int(action)
            target = None if action == CROSS else self.target_price(action, r)
            keep = self.keeps(action, target) if target is not None else False
        else:
            action, target, keep = None, None, False
        self.actions.append(action)
        act_at = i + cfg.L
        stop = i + cfg.k
        for j in range(i, stop):
            if j == act_at:
                if passive and action == CROSS:
                    self.order = None
                    self._take(self.R, j, "cross")
                elif passive:
                    if not keep:
                        self.order = None
                        self._place(target, j)
                else:
                    self.order = None
                    if take_qty > EPS:
                        self._take(take_qty, j, "slice")
            self._replay(j)
            if self.R <= 0:
                break
        self.prev_action = action
        self.step_idx += 1
        self.i = stop
        if self.R > 0 and self.step_idx == self.N:           # deadline: clean up as a taker at t_end
            self.order = None
            before = self.R
            self._take(self.R, stop, "cleanup")
            self.cleanup_qty += before
        self.done = self.R <= 0
        if self.done:
            phi_next = 0.0
        else:
            phi_next = cfg.side * self.R * (self._mid[stop - 1 - self._o] - self.m0)
        cost = self.scale * (self._step_price_usdt + self._step_fee_usdt + phi_next - self.phi)
        self.phi = phi_next
        info = {"R_before": R_t, "R_after": self.R}
        return (None if self.done else self.observe()), cost, self.done, info

    # ------------------------------------------------------------------ result
    def result(self) -> ExecResult:
        if not self.done:
            raise RuntimeError("episode not finished")
        cfg = self.cfg
        fills = {k: np.asarray(v) for k, v in self.fl.items()}
        fee_usdt = 1e-4 * (cfg.maker_bps * self.passive_notional + cfg.taker_bps * self.taker_notional)
        return ExecResult(day=self.d.day, start=self.start, side=cfg.side, m0=self.m0, qty=cfg.qty,
                          cost_bps=self.scale * (self.price_usdt + fee_usdt), price_usdt=self.price_usdt,
                          passive_qty=self.passive_qty, passive_notional=self.passive_notional,
                          taker_qty=self.taker_qty, taker_notional=self.taker_notional, cleanup_qty=self.cleanup_qty,
                          beyond_display=self.beyond, beyond_cleanup=self.beyond_cleanup, n_post_reject=self.n_post_reject, steps_used=self.step_idx,
                          fills=fills, actions=list(self.actions))


def run_execution(policy, data: ReplayDay, start: int, config: ExecConfig) -> ExecResult:
    """Run one full episode. ``policy(obs) -> action`` (0..3), or an object with ``act(obs)`` returning
    (action, take_qty, passive) for baselines that slice."""
    env = ExecutionEnv(data, config)
    obs = env.reset(start)
    done = False
    while not done:
        if hasattr(policy, "act"):
            a, x, passive = policy.act(obs)
            obs, _, done, _ = env.step(a, take_qty=x, passive=passive)
        else:
            obs, _, done, _ = env.step(policy(obs))
    return env.result()
