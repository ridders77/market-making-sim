"""Discrete-time market-making simulator (vectorised across paths).

One step, from t_i to t_{i+1} = t_i + dt, for every path at once:

1. The strategy sees (t_i, S_i, q_i) and posts (bid, ask).
   If a tick size is set, the bid is rounded *down* and the ask *up*.
2. Distances from the mid: delta_b = S_i - bid, delta_a = ask - S_i.
   Each side fills (one unit) with probability A exp(-k delta) dt (stage 1,
   ``fill_rule="linear"``) or 1 - exp(-A exp(-k delta) dt) (stage 2,
   ``fill_rule="exact"``), decided by the path's pre-drawn uniforms u_bid, u_ask.
3. Accounting at the quoted prices, with a maker fee charged on notional:
      bid hit:  q += 1, X -= bid * (1 + fee)
      ask hit:  q -= 1, X += ask * (1 - fee)
4. The mid moves: S_{i+1} = S_i + sigma sqrt(dt) Z_i.

Fills are decided against S_i (the mid when the quotes were posted), as in the
paper. Final P&L is wealth X_T + q_T S_T: any leftover inventory is marked at
the final mid with no liquidation cost (the paper's convention).
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from dataclasses import dataclass, field

import numpy as np

from mms.sim.fills import bernoulli_fills, check_dt, fill_probability
from mms.sim.price import draw_shocks, mid_price_paths
from mms.strategies.base import Strategy

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class MarketParams:
    s0: float
    sigma: float
    T: float
    dt: float
    A: float
    k: float
    fill_rule: str = "linear"  # "linear" (paper, stage 1) or "exact" (stage 2+), see mms.sim.fills

    @property
    def n_steps(self) -> int:
        n = round(self.T / self.dt)
        if not np.isclose(n * self.dt, self.T):
            raise ValueError("T must be an integer multiple of dt")
        return n


@dataclass
class SimResult:
    """Per-path outcomes (arrays of shape (n_paths,)) plus optional full paths."""

    seeds: np.ndarray
    pnl: np.ndarray
    cash: np.ndarray
    q: np.ndarray
    max_abs_q: np.ndarray
    n_bid_fills: np.ndarray
    n_ask_fills: np.ndarray
    spread_captured_sum: np.ndarray  # sum over fills of delta (distance from mid at fill)
    mean_quoted_spread: np.ndarray   # time-average of ask - bid on each path
    n_steps: int
    n_clipped: int                   # (path, step, side) cases with lambda*dt > 1
    notional: np.ndarray = None      # sum over fills of the fill price (USDT traded, per path)
    fee_bps: float = 0.0             # maker fee used in the run
    paths: dict[str, np.ndarray] = field(default_factory=dict)

    def pnl_with_fee(self, fee_bps: float) -> np.ndarray:
        """P&L under another maker fee. Fees only change cash, by -fee * price per fill, so
        P&L(f) = P&L(f_run) - (f - f_run) * notional exactly (fills do not depend on the fee)."""
        return self.pnl - (fee_bps - self.fee_bps) * 1e-4 * self.notional

    @property
    def n_fills(self) -> np.ndarray:
        return self.n_bid_fills + self.n_ask_fills

    @property
    def fill_rate(self) -> np.ndarray:
        """Fraction of (step, side) quote opportunities that were filled."""
        return self.n_fills / (2 * self.n_steps)

    @property
    def avg_spread_captured(self) -> np.ndarray:
        """Average distance from the mid per fill, per path (NaN if a path had no fills)."""
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(self.n_fills > 0, self.spread_captured_sum / self.n_fills, np.nan)


def round_quotes(bid: np.ndarray, ask: np.ndarray, tick: float) -> tuple[np.ndarray, np.ndarray]:
    """Round bid down and ask up to the tick grid.

    The small epsilon stops floating-point noise (e.g. 100.1/0.1 = 1000.9999...)
    from pushing an on-grid price to the wrong tick.
    """
    eps = 1e-9
    return np.floor(bid / tick + eps) * tick, np.ceil(ask / tick - eps) * tick


def apply_fills(X, q, bid, ask, bid_hit, ask_hit, fee_rate: float):
    """Update cash and inventory for one step. fee_rate is a fraction (1 bp = 1e-4)."""
    X = X - bid_hit * bid * (1 + fee_rate) + ask_hit * ask * (1 - fee_rate)
    q = q + bid_hit.astype(int) - ask_hit.astype(int)
    return X, q


def simulate(
    strategy: Strategy,
    params: MarketParams,
    seeds: Iterable[int],
    tick_size: float | None = None,
    maker_fee_bps: float = 0.0,
    record_paths: bool = False,
) -> SimResult:
    """Run ``strategy`` on one path per seed. Same seeds => same prices and order-flow draws."""
    check_dt(params.A, params.dt, params.fill_rule)
    seeds = np.asarray(list(seeds))
    n, N, dt = len(seeds), params.n_steps, params.dt
    rnd = draw_shocks(seeds, N)
    S = mid_price_paths(params.s0, params.sigma, dt, rnd["z"])
    fee = maker_fee_bps * 1e-4

    X = np.zeros(n)
    q = np.zeros(n, dtype=int)
    max_abs_q = np.zeros(n, dtype=int)
    nb = np.zeros(n, dtype=int)
    na = np.zeros(n, dtype=int)
    captured = np.zeros(n)
    notional = np.zeros(n)
    spread_sum = np.zeros(n)
    n_clipped = 0
    if record_paths:
        rec = {key: np.empty((n, N)) for key in ("bid", "ask")}
        rec["q"] = np.empty((n, N + 1), dtype=int)
        rec["q"][:, 0] = 0

    for i in range(N):
        t = i * dt
        Si = S[:, i]
        bid, ask = strategy.quotes(t, Si, q)
        if tick_size is not None:
            bid, ask = round_quotes(bid, ask, tick_size)
        d_b, d_a = Si - bid, ask - Si
        bid_hit, c_b = bernoulli_fills(fill_probability(d_b, params.A, params.k, dt, params.fill_rule), rnd["u_bid"][:, i])
        ask_hit, c_a = bernoulli_fills(fill_probability(d_a, params.A, params.k, dt, params.fill_rule), rnd["u_ask"][:, i])
        n_clipped += c_b + c_a

        X, q = apply_fills(X, q, bid, ask, bid_hit, ask_hit, fee)
        nb += bid_hit
        na += ask_hit
        captured += bid_hit * d_b + ask_hit * d_a
        notional += bid_hit * bid + ask_hit * ask
        spread_sum += ask - bid
        np.maximum(max_abs_q, np.abs(q), out=max_abs_q)
        if record_paths:
            rec["bid"][:, i], rec["ask"][:, i], rec["q"][:, i + 1] = bid, ask, q

    if n_clipped:
        log.info("%s: %d quote-steps had lambda*dt > 1 (quote through the mid); clipped to 1",
                 getattr(strategy, "name", "strategy"), n_clipped)

    paths = {}
    if record_paths:
        paths = {"mid": S, **rec, "t": np.arange(N + 1) * dt}
    return SimResult(
        seeds=seeds, pnl=X + q * S[:, -1], cash=X, q=q, max_abs_q=max_abs_q,
        n_bid_fills=nb, n_ask_fills=na, spread_captured_sum=captured,
        mean_quoted_spread=spread_sum / N, n_steps=N, n_clipped=n_clipped, notional=notional, fee_bps=maker_fee_bps, paths=paths,
    )


def simulate_chunked(strategy: Strategy, params: MarketParams, seeds: Iterable[int], chunk: int = 200,
                     **kwargs) -> SimResult:
    """Run ``simulate`` on groups of ``chunk`` seeds and join the results.

    Each path's randomness depends only on its own seed, so the result is identical to one
    big run; chunking only caps memory (the pre-drawn shocks are n_paths x n_steps arrays).
    """
    if kwargs.get("record_paths"):
        raise ValueError("record_paths is not supported when chunking")
    seeds = list(seeds)
    parts = [simulate(strategy, params, seeds[i:i + chunk], **kwargs) for i in range(0, len(seeds), chunk)]
    cat = lambda name: np.concatenate([getattr(r, name) for r in parts])
    return SimResult(
        seeds=cat("seeds"), pnl=cat("pnl"), cash=cat("cash"), q=cat("q"), max_abs_q=cat("max_abs_q"),
        n_bid_fills=cat("n_bid_fills"), n_ask_fills=cat("n_ask_fills"),
        spread_captured_sum=cat("spread_captured_sum"), mean_quoted_spread=cat("mean_quoted_spread"),
        n_steps=parts[0].n_steps, n_clipped=sum(r.n_clipped for r in parts), notional=cat("notional"),
        fee_bps=parts[0].fee_bps,
    )
