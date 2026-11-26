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


# ---------------------------------------------------------------- stage 2: exact Poisson rule
def test_exact_fill_probability_hand_worked():
    # lambda = 2 exp(-3 * 0.5) = 2 * 0.2231302 = 0.4462603 per s; dt = 0.1 s
    # P = 1 - exp(-0.04462603) = 0.0436450
    p = fill_probability(0.5, A=2.0, k=3.0, dt=0.1, rule="exact")
    assert p == pytest.approx(1 - np.exp(-0.2 * np.exp(-1.5)))
    assert p == pytest.approx(0.0436450, rel=1e-5)
    # always a probability, even for a quote through the mid:
    # delta = -1: lambda dt = 0.2 e^3 = 4.017 -> p = 1 - e^-4.017 = 0.982
    assert fill_probability(-1.0, A=2.0, k=3.0, dt=0.1, rule="exact") == pytest.approx(1 - np.exp(-0.2 * np.exp(3)))
    assert 0 <= fill_probability(-10.0, A=2.0, k=3.0, dt=0.1, rule="exact") <= 1.0


def test_exact_rule_close_to_linear_when_small():
    # 1 - e^{-x} = x - x^2/2 + ...: relative gap ~ x/2 = 0.5% at x = 0.01
    lin = fill_probability(0.0, 1.0, 1.0, 0.01)
    ex = fill_probability(0.0, 1.0, 1.0, 0.01, rule="exact")
    assert ex == pytest.approx(lin * (1 - 0.005), rel=1e-4)


def test_unknown_rule_rejected():
    with pytest.raises(ValueError):
        fill_probability(0.0, 1.0, 1.0, 0.01, rule="other")


def test_check_dt_exact_rule_requires_small_rate():
    check_dt(A=2.0, dt=0.04, rule="exact")  # A dt = 0.08 < 0.1
    with pytest.raises(ValueError):
        check_dt(A=2.0, dt=0.06, rule="exact")  # 0.12 >= 0.1
