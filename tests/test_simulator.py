"""Tests for the simulator loop and P&L accounting."""

import numpy as np
import pytest

from mms.sim.simulator import MarketParams, apply_fills, round_quotes, simulate
from mms.strategies.avellaneda_stoikov import AvellanedaStoikov
from mms.strategies.naive import NaiveSymmetric

PAPER = MarketParams(s0=100.0, sigma=2.0, T=1.0, dt=0.005, A=140.0, k=1.5)


def test_apply_fills_conserves_value():
    # Wealth W = X + q*S, marked at the mid S. Buying 1 at bid b = S - d_b
    # changes W by -b + S = d_b; selling 1 at ask a = S + d_a changes W by a - S = d_a.
    # So with no fee, a fill adds exactly the quoted half-spread to wealth:
    # no value is created or destroyed except the spread captured.
    S = np.array([100.0, 100.0, 100.0])
    X = np.array([0.0, 10.0, -5.0])
    q = np.array([0, 3, -2])
    bid = np.array([99.0, 99.5, 99.2])
    ask = np.array([101.0, 100.7, 100.1])
    bid_hit = np.array([True, False, True])
    ask_hit = np.array([True, True, False])
    W0 = X + q * S
    X1, q1 = apply_fills(X, q, bid, ask, bid_hit, ask_hit, fee_rate=0.0)
    W1 = X1 + q1 * S
    expected_gain = bid_hit * (S - bid) + ask_hit * (ask - S)  # [2.0, 0.7, 0.8]
    np.testing.assert_allclose(W1 - W0, expected_gain)
    np.testing.assert_array_equal(q1, [0, 2, -1])


def test_apply_fills_charges_fee_on_notional():
    # Fee 2 bps = 0.0002. Buying at 99 costs 99 + 0.0002*99 = 99.0198.
    X1, q1 = apply_fills(np.array([0.0]), np.array([0]), np.array([99.0]), np.array([101.0]),
                         np.array([True]), np.array([False]), fee_rate=2e-4)
    assert X1[0] == pytest.approx(-99.0198)
    assert q1[0] == 1


def test_round_quotes_to_tick():
    # Bid rounds down, ask rounds up (never quote tighter than the model wants).
    bid, ask = round_quotes(np.array([100.04, 100.10]), np.array([100.11, 100.20]), tick=0.1)
    np.testing.assert_allclose(bid, [100.0, 100.1])
    np.testing.assert_allclose(ask, [100.2, 100.2])


def test_zero_vol_symmetric_spread_capture():
    # sigma = 0 so the mid never moves. A symmetric quote with spread s earns
    # s/2 per fill and inventory is marked at the unchanged mid, so
    # final P&L = (s/2) * (number of fills) exactly.
    params = MarketParams(s0=100.0, sigma=0.0, T=1.0, dt=0.005, A=140.0, k=1.5)
    s = 1.2
    res = simulate(NaiveSymmetric(spread=s), params, seeds=range(200))
    n_fills = res.n_bid_fills + res.n_ask_fills
    np.testing.assert_allclose(res.pnl, (s / 2) * n_fills)
    # And on average fills match the analytic rate: 2 sides * N steps * A e^{-k s/2} dt
    p = 140.0 * np.exp(-1.5 * s / 2) * 0.005
    expected = 2 * params.n_steps * p
    se = np.sqrt(2 * params.n_steps * p * (1 - p) / len(res.pnl))
    assert abs(n_fills.mean() - expected) < 4 * se
    np.testing.assert_allclose(res.avg_spread_captured, s / 2)


def test_final_wealth_identity():
    # P&L reported must equal X_T + q_T * S_T (starting from X = q = 0).
    res = simulate(AvellanedaStoikov(0.1, 2.0, 1.5, 1.0), PAPER, seeds=range(50), record_paths=True)
    np.testing.assert_allclose(res.pnl, res.cash + res.q * res.paths["mid"][:, -1])
    np.testing.assert_array_equal(res.q, res.n_bid_fills - res.n_ask_fills)


def test_same_seeds_are_paired_and_reproducible():
    a = simulate(NaiveSymmetric(1.5), PAPER, seeds=range(30), record_paths=True)
    b = simulate(AvellanedaStoikov(0.1, 2.0, 1.5, 1.0), PAPER, seeds=range(30), record_paths=True)
    c = simulate(NaiveSymmetric(1.5), PAPER, seeds=range(30))
    np.testing.assert_array_equal(a.paths["mid"], b.paths["mid"])  # same price paths
    np.testing.assert_array_equal(a.pnl, c.pnl)                    # reproducible


def test_simulate_rejects_too_large_dt():
    bad = MarketParams(s0=100.0, sigma=2.0, T=1.0, dt=0.01, A=140.0, k=1.5)
    with pytest.raises(ValueError):
        simulate(NaiveSymmetric(1.5), bad, seeds=range(2))


def test_market_params_n_steps():
    assert PAPER.n_steps == 200


# ---------------------------------------------------------------- stage 2 additions
REAL = MarketParams(s0=30000.0, sigma=2.0, T=60.0, dt=0.05, A=1.5, k=4.0, fill_rule="exact")


def test_fee_equals_fee_free_pnl_minus_fee_times_notional():
    # Fees only move cash: P&L(f) = P&L(0) - f * sum(|price| of each fill). Check on a real-scale setup.
    strat = NaiveSymmetric(0.6)
    r0 = simulate(strat, REAL, range(20), tick_size=0.1, maker_fee_bps=0.0)
    r2 = simulate(strat, REAL, range(20), tick_size=0.1, maker_fee_bps=2.0)
    rr = simulate(strat, REAL, range(20), tick_size=0.1, maker_fee_bps=-0.5)
    np.testing.assert_allclose(r2.pnl, r0.pnl - 2e-4 * r0.notional, rtol=0, atol=1e-6)
    np.testing.assert_allclose(rr.pnl, r0.pnl + 0.5e-4 * r0.notional, rtol=0, atol=1e-6)
    np.testing.assert_allclose(r0.pnl_with_fee(2.0), r2.pnl, atol=1e-6)
    assert (r0.notional > 0).all()


def test_simulate_chunked_matches_unchunked():
    from mms.sim.simulator import simulate_chunked
    strat = AvellanedaStoikov(gamma=1e-3, sigma=REAL.sigma, k=REAL.k, T=REAL.T)
    a = simulate(strat, REAL, range(10), tick_size=0.1)
    b = simulate_chunked(strat, REAL, range(10), chunk=3, tick_size=0.1)
    np.testing.assert_array_equal(a.pnl, b.pnl)
    np.testing.assert_array_equal(a.q, b.q)
    np.testing.assert_array_equal(a.seeds, b.seeds)
    assert a.n_clipped == b.n_clipped and a.n_steps == b.n_steps


def test_exact_rule_dt_check_in_simulate():
    with pytest.raises(ValueError):
        simulate(NaiveSymmetric(1.0), MarketParams(30000.0, 2.0, 10.0, 0.1, 1.5, 4.0, fill_rule="exact"), range(2))
