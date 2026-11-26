"""Tests for the flow-aware (fee-aware) A-S strategy."""

import numpy as np
import pytest

from mms.strategies.avellaneda_stoikov import FeeAwareAvellanedaStoikov
from mms.strategies.flow_aware import FlowAwareAvellanedaStoikov, markout_allowance

BASE = dict(gamma=0.1, sigma=2.0, k=1.5, T=1.0, fee_bps=2.0)


def test_markout_allowance_flat_curve_equals_the_markout():
    # With a constant markout j, a fill at delta is worth delta - cS - j, so the risk-neutral best distance is
    # argmax e^{-k delta}(delta - cS - j) = 1/k + cS + j: the markout shifts the optimum by exactly j.
    j = markout_allowance(k=1.5, fee_per_fill=0.02, jump_delta=(0.0, 10.0), jump_mean=(0.3, 0.3))
    assert j == pytest.approx(0.3, abs=1e-6)


def test_markout_allowance_rising_curve_hand_worked():
    # J(delta) = 1 + 0.5 delta (linear), no fee, k = 1. Maximise f = e^{-delta}(delta - 1 - 0.5 delta) = e^{-delta}(0.5 delta - 1):
    # f' = e^{-delta}(0.5 - 0.5 delta + 1) = 0  ->  delta* = 3. Without the markout delta* = 1/k = 1, so the allowance is 2.
    # (Adding J at the quoting distance instead, delta = 1 + J(delta), would give delta = 4: too wide.)
    j = markout_allowance(k=1.0, fee_per_fill=0.0, jump_delta=(0.0, 100.0), jump_mean=(1.0, 51.0))
    assert j == pytest.approx(2.0, abs=1e-5)


def _strategy(**kw):
    args = dict(BASE, kappa=0.0, tau_x=2.0, jump_delta=(0.0, 10.0), jump_mean=(0.3, 0.3), s_ref=100.0)
    args.update(kw)
    return FlowAwareAvellanedaStoikov(**args)


def _obs(S=100.0, q=2, x=0.0, t=0.5):
    return {"mid": np.array([S]), "inventory": np.array([q]), "signal": np.array([x]), "t": t}


def test_no_signal_is_fee_aware_as_widened_by_the_markout():
    # Fee-aware A-S hand-worked quote (S = 100, q = 2, gamma = 0.1, sigma = 2, T - t = 0.5, k = 1.5, 2 bp):
    # bid 98.8346148, ask 100.3653852. A flat 0.3 markout widens each side by 0.3:
    # bid 98.5346148, ask 100.6653852.
    bid, ask = _strategy().act(_obs())
    assert bid[0] == pytest.approx(98.5346148, abs=1e-5)
    assert ask[0] == pytest.approx(100.6653852, abs=1e-5)
    fb, fa = FeeAwareAvellanedaStoikov(**BASE).quotes(0.5, np.array([100.0]), np.array([2]))
    assert fb[0] - bid[0] == pytest.approx(0.3, abs=1e-5) and ask[0] - fa[0] == pytest.approx(0.3, abs=1e-5)


def test_signal_shifts_reservation_price_by_expected_drift():
    # Flat inventory (q = 0) so the post-only clamp is not involved.
    # D = kappa tau_x (1 - exp(-(T - t)/tau_x)) x. kappa = 0.1, tau_x = 2, x = 1.5, T - t = 2 (T = 3, t = 1):
    # D = 0.1 * 2 * 1.5 * (1 - e^{-1}) = 0.3 * 0.6321206 = 0.1896362. Both quotes move up by D: the ask
    # (the side buyers are about to pick off) moves away from the mid, the bid moves towards it.
    s0 = _strategy(kappa=0.0, T=3.0)
    s1 = _strategy(kappa=0.1, T=3.0)
    b0, a0 = s0.act(_obs(x=1.5, t=1.0, q=0))
    b1, a1 = s1.act(_obs(x=1.5, t=1.0, q=0))
    assert b1[0] - b0[0] == pytest.approx(0.1896362, abs=1e-6)
    assert a1[0] - a0[0] == pytest.approx(0.1896362, abs=1e-6)
    assert s1.drift_shift(1.5, 1.0) == pytest.approx(0.1896362, abs=1e-6)
    # use_signal=False ignores the signal
    b2, a2 = _strategy(kappa=0.1, T=3.0, use_signal=False).act(_obs(x=1.5, t=1.0, q=0))
    np.testing.assert_allclose([b2[0], a2[0]], [b0[0], a0[0]])


def test_post_only_never_quotes_at_or_through_the_mid():
    # A huge buy signal would push the bid above the mid; it is held half a tick (0.05) below the mid.
    b, a = _strategy(kappa=50.0).act(_obs(x=3.0))
    assert b[0] == pytest.approx(100.0 - 0.05)
    assert a[0] > 100.0
