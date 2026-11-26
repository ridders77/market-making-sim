"""Tests for the stage 3 environment (reset/step API, lot size, informed flow)."""

import dataclasses

import numpy as np
import pytest

from mms.sim.env import EnvConfig, FlowParams, MarketMakingEnv, run_env
from mms.sim.simulator import MarketParams, simulate
from mms.strategies.avellaneda_stoikov import AvellanedaStoikov, FeeAwareAvellanedaStoikov
from mms.strategies.naive import NaiveSymmetric

PAPER = MarketParams(s0=100.0, sigma=2.0, T=1.0, dt=0.005, A=140.0, k=1.5)
REAL = MarketParams(s0=30000.0, sigma=2.0, T=60.0, dt=0.05, A=1.5, k=1.0, fill_rule="exact")

FIELDS = ("pnl", "cash", "q", "max_abs_q", "n_bid_fills", "n_ask_fills", "spread_captured_sum",
          "mean_quoted_spread", "notional")


@pytest.mark.parametrize("params, tick, fee", [(PAPER, None, 0.0), (REAL, 0.1, 2.0), (REAL, 0.1, -0.5)])
@pytest.mark.parametrize("make", [
    lambda p: NaiveSymmetric(1.4),
    lambda p: AvellanedaStoikov(gamma=0.1 if p.s0 == 100 else 1e-3, sigma=p.sigma, k=p.k, T=p.T),
])
def test_old_strategies_identical_through_env(params, tick, fee, make):
    # The stage 1/2 simulator and the new reset/step environment must give bit-identical results
    # for the old strategies when informed flow is off and the lot is 1.
    strat = make(params)
    old = simulate(strat, params, range(25), tick_size=tick, maker_fee_bps=fee)
    new = run_env(strat, EnvConfig(params, tick_size=tick, maker_fee_bps=fee), range(25))
    for f in FIELDS:
        np.testing.assert_array_equal(getattr(new, f), getattr(old, f), err_msg=f)
    assert new.n_clipped == old.n_clipped and new.n_steps == old.n_steps


def test_rewards_sum_to_final_pnl_and_done_at_horizon():
    cfg = EnvConfig(REAL, tick_size=0.1, maker_fee_bps=1.0, lot=0.01)
    env = MarketMakingEnv(cfg)
    obs = env.reset(range(5))
    strat = AvellanedaStoikov(gamma=1e-3, sigma=REAL.sigma, k=REAL.k, T=REAL.T)
    total, done, n = np.zeros(5), False, 0
    while not done:
        obs, reward, done, info = env.step(strat.quotes(obs["t"], obs["mid"], obs["inventory"]))
        total += reward
        n += 1
    assert n == REAL.n_steps
    np.testing.assert_allclose(total, env.result().pnl, atol=1e-8)
    with pytest.raises(RuntimeError):
        env.step((obs["mid"] - 1, obs["mid"] + 1))


def test_reset_accepts_a_single_seed():
    env = MarketMakingEnv(EnvConfig(REAL, tick_size=0.1))
    obs = env.reset(7)
    assert obs["inventory"].shape == (1,)
    assert set(obs) >= {"inventory", "time_remaining", "signal", "sigma_hat", "mid", "t"}
    assert obs["time_remaining"][0] == pytest.approx(REAL.T)


def test_lot_scales_pnl_and_notional():
    # Strategies see inventory in lots, so the quotes and fills are the same whatever the lot;
    # cash, P&L and traded notional scale exactly with the lot (0.01 BTC lot -> 1/100 of the USDT P&L).
    strat = NaiveSymmetric(1.4)
    a = run_env(strat, EnvConfig(REAL, tick_size=0.1, maker_fee_bps=2.0, lot=1.0), range(10))
    b = run_env(strat, EnvConfig(REAL, tick_size=0.1, maker_fee_bps=2.0, lot=0.01), range(10))
    np.testing.assert_array_equal(a.q, b.q)
    np.testing.assert_allclose(b.pnl, 0.01 * a.pnl, rtol=1e-12, atol=1e-9)
    np.testing.assert_allclose(b.notional, 0.01 * a.notional, rtol=1e-12)
    assert b.lot == 0.01
    # P&L per unit notional (bps) does not depend on the lot
    np.testing.assert_allclose(b.pnl_bps(), a.pnl_bps())


def test_informed_off_equals_baseline():
    # Flow switched on but with zero drift, zero tilt and zero jumps: identical to the baseline.
    strat = AvellanedaStoikov(gamma=1e-3, sigma=REAL.sigma, k=REAL.k, T=REAL.T)
    base = run_env(strat, EnvConfig(REAL, tick_size=0.1), range(15))
    flow = run_env(strat, EnvConfig(REAL, tick_size=0.1, flow=FlowParams()), range(15))
    for f in FIELDS:
        np.testing.assert_array_equal(getattr(flow, f), getattr(base, f), err_msg=f)


def test_jump_against_us_after_fill_hand_worked():
    # sigma = 0 and no drift: the mid only moves through our own fills' markout jumps.
    # Jump curve: J(delta) interpolated between (0, 0.5) and (2, 1.5): J(1.0) = 1.0, J(0.5) = 0.75.
    params = MarketParams(s0=100.0, sigma=0.0, T=200.0, dt=0.05, A=1.9, k=0.0001, fill_rule="exact")
    flow = FlowParams(jump_delta=(0.0, 2.0), jump_mean=(0.5, 1.5))
    env = MarketMakingEnv(EnvConfig(params, flow=flow))
    obs = env.reset(range(200))
    seen_bid = seen_ask = False
    for _ in range(params.n_steps):
        S = obs["mid"]
        obs, _, done, info = env.step((S - 1.0, S + 0.5))   # bid 1.0 below, ask 0.5 above the mid
        # bought -> mid drops by J(1.0) = 1.0; sold -> mid rises by J(0.5) = 0.75
        expected = S - 1.0 * info["bid_hit"] + 0.75 * info["ask_hit"]
        np.testing.assert_allclose(obs["mid"], expected, atol=1e-9)
        seen_bid |= info["bid_hit"].any()
        seen_ask |= info["ask_hit"].any()
        if done:
            break
    assert seen_bid and seen_ask


def test_signal_is_stationary_ou():
    # x_{i+1} = rho x_i + sqrt(1 - rho^2) xi_i with rho = exp(-dt/tau): mean 0, variance 1, lag-1 corr rho.
    params = MarketParams(s0=100.0, sigma=1.0, T=600.0, dt=0.5, A=0.1, k=1.0, fill_rule="exact")
    env = MarketMakingEnv(EnvConfig(params, flow=FlowParams(tau=10.0)))
    env.reset(range(300))
    x = env.signal_path()
    rho = np.exp(-0.5 / 10.0)
    assert abs(x.mean()) < 0.03
    assert x.var() == pytest.approx(1.0, abs=0.03)
    assert np.corrcoef(x[:, :-1].ravel(), x[:, 1:].ravel())[0, 1] == pytest.approx(rho, abs=0.01)


def test_drift_predicts_the_next_move_as_derived():
    # E[S_{t+h} - S_t | x_t] = kappa * tau * (1 - exp(-h/tau)) * x_t. Regress simulated h-step moves on x_t.
    kappa, tau, h, dt = 0.05, 20.0, 30.0, 0.5
    params = MarketParams(s0=100.0, sigma=0.4, T=900.0, dt=dt, A=1e-6, k=1.0, fill_rule="exact")
    flow = FlowParams(kappa=kappa, tau=tau)
    env = MarketMakingEnv(EnvConfig(params, flow=flow))
    obs = env.reset(range(400))
    mids = [obs["mid"]]
    done = False
    while not done:   # quotes far away: no fills, so no jumps
        obs, _, done, _ = env.step((obs["mid"] - 1e6, obs["mid"] + 1e6))
        mids.append(obs["mid"])
    S, x = np.array(mids).T, env.signal_path()
    lag = int(h / dt)
    y, xx = (S[:, lag:] - S[:, :-lag]).ravel(), x[:, :-lag].ravel()
    slope = np.sum(xx * y) / np.sum(xx * xx)
    assert slope == pytest.approx(flow.expected_move(1.0, h), rel=0.05)
    assert flow.expected_move(1.0, h) == pytest.approx(kappa * tau * (1 - np.exp(-h / tau)))
    # sigma of the noise is reduced so the long-run variance stays sigma^2: sigma_eps^2 = sigma^2 - 2 kappa^2 tau
    assert flow.noise_sigma(0.4) == pytest.approx(np.sqrt(0.16 - 2 * kappa**2 * tau))


def test_fill_tilt_shifts_fills_towards_the_flow():
    # lambda_ask = lambda(delta) exp(theta x - theta^2/2): same average rate, but asks fill more when x > 0.
    # For x ~ N(0, 1), the x seen at an ask fill has mean theta (exponential tilting of a normal).
    theta = 0.6
    params = MarketParams(s0=100.0, sigma=0.0, T=2000.0, dt=0.5, A=0.02, k=0.01, fill_rule="exact")
    cfg = EnvConfig(params, flow=FlowParams(theta=theta, tau=5.0))
    env = MarketMakingEnv(cfg)
    obs = env.reset(range(100))
    xs_ask, xs_bid, done = [], [], False
    n_ask = 0
    while not done:
        x = obs["signal"].copy()
        obs, _, done, info = env.step((obs["mid"] - 1.0, obs["mid"] + 1.0))
        xs_ask.append(x[info["ask_hit"]]); xs_bid.append(x[info["bid_hit"]])
        n_ask += info["ask_hit"].sum()
    xa, xb = np.concatenate(xs_ask), np.concatenate(xs_bid)
    assert xa.mean() == pytest.approx(theta, abs=0.05)
    assert xb.mean() == pytest.approx(-theta, abs=0.05)
    # Average ask fill rate unchanged by the tilt: compare to 1 - exp(-lambda dt) per step
    p = -np.expm1(-0.02 * np.exp(-0.01) * 0.5)
    assert n_ask / (100 * params.n_steps) == pytest.approx(p, rel=0.05)


def test_fee_aware_strategy_runs_through_env_with_obs():
    # Strategies with an act(obs) method get the full observation; others get (t, mid, inventory).
    strat = FeeAwareAvellanedaStoikov(gamma=1e-3, sigma=REAL.sigma, k=REAL.k, T=REAL.T, fee_bps=2.0)
    r = run_env(strat, EnvConfig(REAL, tick_size=0.1, maker_fee_bps=2.0, lot=0.01), range(5))
    assert np.isfinite(r.pnl).all()
