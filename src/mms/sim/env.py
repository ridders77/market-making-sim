r"""Market-making environment with a reset/step API, a lot size and (optionally) informed flow.

This is the stage 3 engine. With ``flow=None`` and ``lot=1`` it reproduces the stage 1/2
``simulate`` loop exactly (same random numbers, same order of operations; tested). Stage 4's
RL agent uses the same API:

    env = MarketMakingEnv(config)
    obs = env.reset(seed)                       # one seed, or a list of seeds (vectorised paths)
    obs, reward, done, info = env.step((bid, ask))

Observation (arrays over paths): ``inventory`` (in lots), ``time_remaining`` (T - t, s),
``signal`` (the order-flow signal x_t; 0 when flow is off), ``sigma_hat`` (trailing volatility of
the simulated mid), plus ``mid`` and ``t``, which a quoting rule needs to turn distances into prices.
The action is the pair of quote prices (bid, ask). Reward = change in marked-to-market wealth
(cash + inventory at the mid, USDT) over the step, so the rewards sum to the final P&L.

One step from t_i to t_{i+1} = t_i + dt
--------------------------------------
1. Quotes (bid, ask). Stage 3 option ``min_distance``: a quote closer to the mid than this is moved out to it
   (counted). lambda(delta) and the markout curve are calibrated only beyond the touch (delta >= 0.1 USDT;
   the touch needs a queue model we do not have), and a maker order can never cross the mid.
   Then rounded outward to the tick (bid down, ask up) if a tick is set.
2. Distances from the mid delta_b = S_i - bid, delta_a = ask - S_i. Each side fills one lot with
   probability 1 - exp(-lambda dt) (or lambda dt, stage 1 rule), with
       lambda_ask = A exp(-k delta_a) * exp(+theta x_i - theta^2/2),
       lambda_bid = A exp(-k delta_b) * exp(-theta x_i - theta^2/2)      (theta = 0 when flow is off).
3. Accounting per lot of u BTC with maker fee c: bid hit: q += 1, cash -= u bid (1 + c);
   ask hit: q -= 1, cash += u ask (1 - c).
4. Mid: S_{i+1} = S_i + kappa x_i dt + sigma_eps sqrt(dt) Z_i - J_b [bid hit] + J_a [ask hit].

Informed flow (stage 3)
-----------------------
* Signal. x is an Ornstein-Uhlenbeck process with time constant tau and unit stationary variance,
  simulated exactly: x_{i+1} = rho x_i + sqrt(1 - rho^2) xi_i, rho = exp(-dt/tau), x_0 ~ N(0, 1).
  It stands for standardised order-flow imbalance: x > 0 means buyers have been aggressive.
* Drift (predictive power). The mid drifts by kappa x per second. Given x_t, the expected move over
  the next h seconds is
      E[S_{t+h} - S_t | x_t] = kappa int_0^h E[x_{t+s} | x_t] ds = kappa x_t int_0^h e^{-s/tau} ds
                             = kappa tau (1 - e^{-h/tau}) x_t,
  which is how kappa is calibrated from the measured OFI regression slope at horizon h.
  The drift adds variance: for h >> tau, Var(int kappa x) ~ 2 kappa^2 tau h, so the noise is
  sigma_eps^2 = sigma^2 - 2 kappa^2 tau to keep the long-run variance at the calibrated sigma^2.
* Fill tilt. When x > 0 buyers are more active, so our ask is hit more often and our bid less.
  The factor exp(theta x - theta^2/2) has mean 1 under x ~ N(0, 1), so the *average* fill rate is
  unchanged. A side effect used in calibration: the x seen at an ask fill has mean theta
  (exponential tilting shifts a standard normal's mean by theta), so fills carry on average
  theta units of signal against us.
* Markout jump. When our quote at distance delta fills, the mid jumps against us by
  J(delta) + noise (down after we buy, up after we sell). J is the calibrated markout curve
  (interpolated linearly in delta, flat beyond the grid); noise ~ N(0, jump_sd^2).

All randomness for a path comes from its seed: the stage 1/2 draws (price shocks, fill uniforms)
from ``default_rng(seed)``, and the flow draws (signal, jump noise) from a separate stream
``default_rng([seed, 1])``, so switching flow on does not change the baseline draws (paired runs).
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
from scipy.signal import lfilter

from mms.sim.fills import bernoulli_fills, check_dt, fill_probability
from mms.sim.price import draw_shocks, mid_price_paths
from mms.sim.simulator import MarketParams, SimResult, round_quotes

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class FlowParams:
    kappa: float = 0.0                    # drift per unit signal (USDT per second)
    tau: float = 30.0                     # signal time constant (s)
    theta: float = 0.0                    # fill-rate tilt per unit signal
    jump_delta: tuple = (0.0,)            # grid of quote distances (USDT) for the markout curve
    jump_mean: tuple = (0.0,)             # mean jump against us at those distances (USDT)
    jump_sd: float = 0.0                  # std of the jump noise (USDT)

    def jump(self, delta: np.ndarray) -> np.ndarray:
        """Mean markout jump J(delta), linear interpolation, flat outside the grid."""
        return np.interp(delta, self.jump_delta, self.jump_mean)

    def expected_move(self, x, h: float):
        """E[S_{t+h} - S_t | x_t = x] = kappa tau (1 - exp(-h/tau)) x (drift only, no jumps)."""
        return self.kappa * self.tau * (1.0 - np.exp(-h / self.tau)) * np.asarray(x, float)

    def noise_sigma(self, sigma: float) -> float:
        """sigma_eps = sqrt(sigma^2 - 2 kappa^2 tau): keeps the long-run variance at sigma^2."""
        v = sigma**2 - 2.0 * self.kappa**2 * self.tau
        if v < 0:
            raise ValueError(f"drift variance 2 kappa^2 tau = {2 * self.kappa**2 * self.tau:.4g} exceeds sigma^2")
        return float(np.sqrt(v))


@dataclass(frozen=True)
class EnvConfig:
    market: MarketParams
    lot: float = 1.0                      # BTC per fill
    tick_size: float | None = None
    maker_fee_bps: float = 0.0
    flow: FlowParams | None = None        # None: stage 1/2 dynamics
    sigma_span_s: float = 1800.0          # EWMA span of the observed sigma_hat (seconds)
    min_distance: float | None = None     # stage 3: quotes kept >= this far from the mid (calibrated range of lambda, J)


def _as_seeds(seeds) -> np.ndarray:
    if np.isscalar(seeds):
        return np.array([int(seeds)])
    return np.asarray(list(seeds))


class MarketMakingEnv:
    """Vectorised over paths: every array in obs/info has one entry per seed."""

    def __init__(self, config: EnvConfig):
        self.cfg = config
        p = config.market
        check_dt(p.A, p.dt, p.fill_rule)
        self.N = p.n_steps
        self._i = None

    # ------------------------------------------------------------------ reset
    def reset(self, seeds: int | Iterable[int], record_paths: bool = False) -> dict:
        cfg, p, f = self.cfg, self.cfg.market, self.cfg.flow
        self.seeds = _as_seeds(seeds)
        n, N = len(self.seeds), self.N
        rnd = draw_shocks(self.seeds, N)
        self._u_bid, self._u_ask = rnd["u_bid"], rnd["u_ask"]
        sig = p.sigma if f is None else f.noise_sigma(p.sigma)
        self._B = mid_price_paths(p.s0, sig, p.dt, rnd["z"])      # diffusion part of the mid
        self._D = np.zeros(n)                                     # drift + jumps added so far
        if f is not None:
            rho = np.exp(-p.dt / f.tau)
            e = np.empty((n, N + 1))               # e_0 = x_0, e_{i+1} = sqrt(1 - rho^2) xi_i
            self._jump_noise = np.empty((n, N, 2))
            for j, s in enumerate(self.seeds):
                rng = np.random.default_rng([int(s), 1])
                e[j, 0] = rng.standard_normal()
                e[j, 1:] = np.sqrt(1 - rho**2) * rng.standard_normal(N)
                self._jump_noise[j] = rng.standard_normal((N, 2))
            # exact OU recursion x_{i+1} = rho x_i + e_{i+1}, run as a linear filter (fast, same numbers)
            self._x = lfilter([1.0], [1.0, -rho], e, axis=1)
        else:
            self._x = np.zeros((n, N + 1))
        self._fee = cfg.maker_fee_bps * 1e-4
        self._X = np.zeros(n)                     # cash per unit lot (USDT per BTC of lot)
        self._q = np.zeros(n, dtype=int)          # inventory in lots
        self._max_abs_q = np.zeros(n, dtype=int)
        self._nb = np.zeros(n, dtype=int)
        self._na = np.zeros(n, dtype=int)
        self._captured = np.zeros(n)
        self._notional = np.zeros(n)              # per unit lot; x lot at the end
        self._spread_sum = np.zeros(n)
        self._n_clipped = 0
        self._n_min_distance = 0
        self._var = np.full(n, p.sigma**2)        # EWMA of (dS)^2/dt, started at the calibrated sigma^2
        self._alpha = 2.0 / (cfg.sigma_span_s / p.dt + 1.0)
        self._record = record_paths
        if record_paths:
            self._rec = {k: np.empty((n, N)) for k in ("bid", "ask")}
            self._rec["mid"] = np.empty((n, N + 1))
            self._rec["q"] = np.zeros((n, N + 1), dtype=int)
        self._i = 0
        self._S = self._mid(0)
        if record_paths:
            self._rec["mid"][:, 0] = self._S
        return self._obs()

    def _mid(self, i: int) -> np.ndarray:
        if self.cfg.flow is None:
            return self._B[:, i]
        return self._B[:, i] + self._D

    def _obs(self) -> dict:
        t = self._i * self.cfg.market.dt
        n = len(self.seeds)
        return {"inventory": self._q.copy(), "time_remaining": np.full(n, self.cfg.market.T - t),
                "signal": self._x[:, self._i].copy(), "sigma_hat": np.sqrt(self._var),
                "mid": self._S.copy(), "t": t}

    def signal_path(self) -> np.ndarray:
        """The full signal path x (n_paths, N + 1) of the current episode (for tests and charts)."""
        return self._x

    # ------------------------------------------------------------------ step
    def step(self, action) -> tuple[dict, np.ndarray, bool, dict]:
        if self._i is None or self._i >= self.N:
            raise RuntimeError("call reset() before step(), and not after the episode is done")
        cfg, p, f, i = self.cfg, self.cfg.market, self.cfg.flow, self._i
        bid, ask = (np.asarray(a, dtype=float) for a in action)
        if cfg.min_distance is not None:
            lo, hi = self._S - cfg.min_distance, self._S + cfg.min_distance
            self._n_min_distance += int(np.count_nonzero(bid > lo) + np.count_nonzero(ask < hi))
            bid, ask = np.minimum(bid, lo), np.maximum(ask, hi)
        if cfg.tick_size is not None:
            bid, ask = round_quotes(bid, ask, cfg.tick_size)
        Si = self._S
        d_b, d_a = Si - bid, ask - Si
        if f is None or f.theta == 0.0:
            A_b = A_a = p.A
        else:
            x = self._x[:, i]
            A_a = p.A * np.exp(f.theta * x - f.theta**2 / 2)
            A_b = p.A * np.exp(-f.theta * x - f.theta**2 / 2)
        bid_hit, c_b = bernoulli_fills(fill_probability(d_b, A_b, p.k, p.dt, p.fill_rule), self._u_bid[:, i])
        ask_hit, c_a = bernoulli_fills(fill_probability(d_a, A_a, p.k, p.dt, p.fill_rule), self._u_ask[:, i])
        self._n_clipped += c_b + c_a

        W0 = self._X + self._q * Si
        fee = self._fee
        self._X = self._X - bid_hit * bid * (1 + fee) + ask_hit * ask * (1 - fee)
        self._q = self._q + bid_hit.astype(int) - ask_hit.astype(int)
        self._nb += bid_hit
        self._na += ask_hit
        self._captured += bid_hit * d_b + ask_hit * d_a
        self._notional += bid_hit * bid + ask_hit * ask
        self._spread_sum += ask - bid
        np.maximum(self._max_abs_q, np.abs(self._q), out=self._max_abs_q)

        jump_b = jump_a = np.zeros(len(Si))
        if f is not None:
            jump_b = bid_hit * (f.jump(d_b) + f.jump_sd * self._jump_noise[:, i, 0])
            jump_a = ask_hit * (f.jump(d_a) + f.jump_sd * self._jump_noise[:, i, 1])
            self._D = self._D + f.kappa * self._x[:, i] * p.dt - jump_b + jump_a
        self._i = i + 1
        S_new = self._mid(self._i)
        self._var = (1 - self._alpha) * self._var + self._alpha * (S_new - Si) ** 2 / p.dt
        self._S = S_new
        reward = cfg.lot * (self._X + self._q * S_new - W0)
        if self._record:
            self._rec["bid"][:, i], self._rec["ask"][:, i] = bid, ask
            self._rec["mid"][:, i + 1], self._rec["q"][:, i + 1] = S_new, self._q
        done = self._i == self.N
        info = {"bid": bid, "ask": ask, "bid_hit": bid_hit, "ask_hit": ask_hit, "delta_bid": d_b,
                "delta_ask": d_a, "jump_bid": jump_b, "jump_ask": jump_a}
        return self._obs(), reward, done, info

    # ------------------------------------------------------------------ result
    def result(self) -> SimResult:
        """Per-path outcomes in USDT (cash, P&L, notional) with inventory in lots."""
        if self._i != self.N:
            raise RuntimeError("episode not finished")
        lot = self.cfg.lot
        paths = {}
        if self._record:
            paths = {**self._rec, "signal": self._x, "t": np.arange(self.N + 1) * self.cfg.market.dt}
        return SimResult(
            seeds=self.seeds, pnl=lot * (self._X + self._q * self._S), cash=lot * self._X, q=self._q,
            max_abs_q=self._max_abs_q, n_bid_fills=self._nb, n_ask_fills=self._na,
            spread_captured_sum=self._captured, mean_quoted_spread=self._spread_sum / self.N,
            n_steps=self.N, n_clipped=self._n_clipped, notional=lot * self._notional,
            fee_bps=self.cfg.maker_fee_bps, paths=paths, lot=lot, n_min_distance=self._n_min_distance,
        )


def act(strategy, obs: dict):
    """Quotes from a strategy: ``act(obs)`` if it has one, else the stage 1/2 ``quotes(t, S, q)``."""
    if hasattr(strategy, "act"):
        return strategy.act(obs)
    return strategy.quotes(obs["t"], obs["mid"], obs["inventory"])


def run_env(strategy, config: EnvConfig, seeds: Iterable[int], record_paths: bool = False,
            chunk: int | None = None) -> SimResult:
    """Run one episode per seed and return the outcomes. ``chunk`` caps memory (identical results)."""
    seeds = list(_as_seeds(seeds))
    if chunk is not None and len(seeds) > chunk:
        if record_paths:
            raise ValueError("record_paths is not supported when chunking")
        parts = [run_env(strategy, config, seeds[i:i + chunk]) for i in range(0, len(seeds), chunk)]
        return concat_results(parts)
    env = MarketMakingEnv(config)
    obs = env.reset(seeds, record_paths=record_paths)
    done = False
    while not done:
        obs, _, done, _ = env.step(act(strategy, obs))
    return env.result()


def concat_results(parts: list[SimResult]) -> SimResult:
    cat = lambda name: np.concatenate([getattr(r, name) for r in parts])
    return SimResult(
        seeds=cat("seeds"), pnl=cat("pnl"), cash=cat("cash"), q=cat("q"), max_abs_q=cat("max_abs_q"),
        n_bid_fills=cat("n_bid_fills"), n_ask_fills=cat("n_ask_fills"),
        spread_captured_sum=cat("spread_captured_sum"), mean_quoted_spread=cat("mean_quoted_spread"),
        n_steps=parts[0].n_steps, n_clipped=sum(r.n_clipped for r in parts), notional=cat("notional"),
        fee_bps=parts[0].fee_bps, lot=parts[0].lot, n_min_distance=sum(r.n_min_distance for r in parts),
    )
