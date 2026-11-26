"""Tests for the naive and Avellaneda-Stoikov quoting rules."""

import numpy as np
import pytest

from mms.strategies.avellaneda_stoikov import AvellanedaStoikov
from mms.strategies.naive import NaiveSymmetric


def test_avellaneda_stoikov_hand_worked_quote():
    # Inputs: S = 100, q = 2, gamma = 0.1, sigma = 2, T - t = 0.5, k = 1.5
    #
    # Reservation price r = S - q*gamma*sigma^2*(T - t)
    #                     = 100 - 2 * 0.1 * 4 * 0.5 = 100 - 0.4 = 99.6
    #
    # Total spread = gamma*sigma^2*(T - t) + (2/gamma)*ln(1 + gamma/k)
    #              = 0.1*4*0.5           + 20 * ln(1 + 0.1/1.5)
    #              = 0.2                 + 20 * ln(16/15)
    #              = 0.2                 + 20 * 0.0645385211
    #              = 0.2 + 1.2907704 = 1.4907704
    #
    # bid = r - spread/2 = 99.6 - 0.7453852 = 98.8546148
    # ask = r + spread/2 = 99.6 + 0.7453852 = 100.3453852
    #
    # Long 2 units, so both quotes are shifted down: the ask (where we sell)
    # is only 0.345 above the mid while the bid is 1.145 below it.
    strat = AvellanedaStoikov(gamma=0.1, sigma=2.0, k=1.5, T=1.0)
    t = 0.5
    assert strat.reservation_price(100.0, 2, t) == pytest.approx(99.6)
    assert strat.spread(t) == pytest.approx(1.4907704, abs=1e-7)
    bid, ask = strat.quotes(t, np.array([100.0]), np.array([2]))
    assert bid[0] == pytest.approx(98.8546148, abs=1e-7)
    assert ask[0] == pytest.approx(100.3453852, abs=1e-7)


def test_avellaneda_stoikov_flat_inventory_is_symmetric():
    strat = AvellanedaStoikov(gamma=0.1, sigma=2.0, k=1.5, T=1.0)
    bid, ask = strat.quotes(0.3, np.array([100.0]), np.array([0]))
    assert 100.0 - bid[0] == pytest.approx(ask[0] - 100.0)


def test_avellaneda_stoikov_spread_at_horizon_is_liquidity_term_only():
    # At t = T the inventory-risk term vanishes: spread = (2/gamma) ln(1 + gamma/k)
    strat = AvellanedaStoikov(gamma=0.5, sigma=2.0, k=1.5, T=1.0)
    assert strat.spread(1.0) == pytest.approx(4 * np.log(1 + 0.5 / 1.5))


def test_avellaneda_stoikov_small_gamma_limit():
    # As gamma -> 0, (2/gamma) ln(1 + gamma/k) -> 2/k and the skew vanishes.
    strat = AvellanedaStoikov(gamma=1e-8, sigma=2.0, k=1.5, T=1.0)
    assert strat.spread(0.0) == pytest.approx(2 / 1.5, rel=1e-6)


def test_avellaneda_stoikov_vectorised_over_sims():
    strat = AvellanedaStoikov(gamma=0.1, sigma=2.0, k=1.5, T=1.0)
    S = np.array([100.0, 100.0, 101.0])
    q = np.array([0, 2, -1])
    bid, ask = strat.quotes(0.5, S, q)
    assert bid.shape == (3,)
    # more long inventory => lower quotes
    assert bid[1] < bid[0] and ask[1] < ask[0]
    np.testing.assert_allclose(ask - bid, strat.spread(0.5))


def test_naive_symmetric():
    strat = NaiveSymmetric(spread=1.5)
    bid, ask = strat.quotes(0.2, np.array([100.0, 50.0]), np.array([5, -3]))
    np.testing.assert_allclose(bid, [99.25, 49.25])
    np.testing.assert_allclose(ask, [100.75, 50.75])


def test_naive_rejects_negative_spread():
    with pytest.raises(ValueError):
        NaiveSymmetric(spread=-1.0)


# ---------------------------------------------------------------- stage 3: fee-aware A-S
from mms.strategies.avellaneda_stoikov import FeeAwareAvellanedaStoikov


def test_fee_aware_as_hand_worked():
    # Same inputs as the hand-worked A-S quote (S = 100, q = 2, gamma = 0.1, sigma = 2, T - t = 0.5, k = 1.5):
    # plain A-S: bid 98.8546148, ask 100.3453852.
    # Maker fee c = 2 bp = 0.0002 of notional: each fill costs c * S = 0.0002 * 100 = 0.02,
    # so each half-spread grows by 0.02: bid 98.8346148, ask 100.3653852 (reservation price unchanged).
    strat = FeeAwareAvellanedaStoikov(gamma=0.1, sigma=2.0, k=1.5, T=1.0, fee_bps=2.0)
    bid, ask = strat.quotes(0.5, np.array([100.0]), np.array([2]))
    assert bid[0] == pytest.approx(98.8346148, abs=1e-7)
    assert ask[0] == pytest.approx(100.3653852, abs=1e-7)


def test_fee_aware_as_rebate_narrows_and_zero_fee_is_plain_as():
    # A -0.5 bp rebate at S = 30,000 pays 1.5 USDT per fill, so each side moves 1.5 USDT closer to the mid.
    kw = dict(gamma=1e-4, sigma=1.9, k=0.46, T=3600.0)
    plain = AvellanedaStoikov(**kw)
    S, q = np.array([30000.0]), np.array([3])
    b0, a0 = plain.quotes(100.0, S, q)
    b1, a1 = FeeAwareAvellanedaStoikov(**kw, fee_bps=-0.5).quotes(100.0, S, q)
    assert b1[0] - b0[0] == pytest.approx(1.5) and a0[0] - a1[0] == pytest.approx(1.5)
    b2, a2 = FeeAwareAvellanedaStoikov(**kw, fee_bps=0.0).quotes(100.0, S, q)
    np.testing.assert_array_equal(b2, b0)
    np.testing.assert_array_equal(a2, a0)
