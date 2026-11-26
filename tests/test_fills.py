"""Tests for the exponential fill-intensity model lambda(delta) = A exp(-k delta)."""

import numpy as np
import pytest
from scipy import stats

from mms.sim.fills import bernoulli_fills, check_dt, fill_probability


def test_fill_probability_at_mid_is_A_dt():
    # delta = 0  =>  lambda = A, so P(fill in dt) = A*dt = 140*0.005 = 0.7
    assert fill_probability(0.0, A=140, k=1.5, dt=0.005) == pytest.approx(0.7)


def test_fill_probability_hand_worked():
    # delta = 1: lambda = 140*exp(-1.5) = 140*0.22313016 = 31.2382...
    # P = 31.2382 * 0.005 = 0.156191...
    assert fill_probability(1.0, 140, 1.5, 0.005) == pytest.approx(140 * np.exp(-1.5) * 0.005)
    assert fill_probability(1.0, 140, 1.5, 0.005) == pytest.approx(0.1561911, rel=1e-6)


def test_fill_probability_decreases_with_distance():
    d = np.array([0.0, 0.5, 1.0, 2.0])
    p = fill_probability(d, 140, 1.5, 0.005)
    assert np.all(np.diff(p) < 0)
    # ratio between quotes 1 apart is exp(-k)
    assert p[3] / p[2] == pytest.approx(np.exp(-1.5))


def test_check_dt_rejects_large_dt():
    check_dt(A=140, dt=0.005)  # A*dt = 0.7 < 1: fine
    with pytest.raises(ValueError):
        check_dt(A=140, dt=0.01)  # A*dt = 1.4: a quote at the mid would "fill" with prob > 1


def test_bernoulli_fills_and_clipping_count():
    p = np.array([0.2, 0.2, 1.3, 0.0])
    u = np.array([0.1, 0.3, 0.99, 0.0])
    filled, n_clipped = bernoulli_fills(p, u)
    # fill iff u < p ; p = 1.3 is clipped to 1 (always fills) and counted
    np.testing.assert_array_equal(filled, [True, False, True, False])
    assert n_clipped == 1


def test_simulated_fill_rate_matches_analytic():
    # Over n Bernoulli(p) trials the fill count is Binomial(n, p).
    # Check the simulated count falls inside the central 99.9% binomial interval.
    p_true = float(fill_probability(0.8, 140, 1.5, 0.005))
    n = 200_000
    u = np.random.default_rng(42).random(n)
    filled, _ = bernoulli_fills(np.full(n, p_true), u)
    lo, hi = stats.binom.interval(0.999, n, p_true)
    assert lo <= filled.sum() <= hi
