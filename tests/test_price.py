"""Tests for the arithmetic Brownian motion mid-price."""

import numpy as np
import pytest

from mms.sim.price import draw_shocks, mid_price_paths


def test_zero_volatility_is_constant():
    z = np.random.default_rng(0).standard_normal((3, 50))
    S = mid_price_paths(s0=100.0, sigma=0.0, dt=0.01, z=z)
    assert S.shape == (3, 51)
    assert np.all(S == 100.0)


def test_hand_worked_two_steps():
    # S1 = S0 + sigma*sqrt(dt)*z1 = 100 + 2*0.1*1   = 100.2
    # S2 = S1 + sigma*sqrt(dt)*z2 = 100.2 + 2*0.1*(-2) = 99.8
    z = np.array([[1.0, -2.0]])
    S = mid_price_paths(s0=100.0, sigma=2.0, dt=0.01, z=z)
    np.testing.assert_allclose(S, [[100.0, 100.2, 99.8]])


def test_increment_variance_matches_sigma_squared_dt():
    # Var(dS) = sigma^2 dt. With n = 200,000 increments the sample variance has
    # relative standard error sqrt(2/n) ~ 0.32%, so 2% tolerance is > 6 standard errors.
    sigma, dt = 2.0, 0.005
    z = np.random.default_rng(1).standard_normal((1000, 200))
    S = mid_price_paths(100.0, sigma, dt, z)
    inc = np.diff(S, axis=1)
    assert inc.var() == pytest.approx(sigma**2 * dt, rel=0.02)
    assert abs(inc.mean()) < 5 * np.sqrt(sigma**2 * dt / inc.size)


def test_shocks_reproducible_per_seed():
    a = draw_shocks([7, 8, 9], n_steps=10)
    b = draw_shocks([7, 8, 9], n_steps=10)
    for key in ("z", "u_bid", "u_ask"):
        np.testing.assert_array_equal(a[key], b[key])
    # Path i depends only on its own seed: seed 8 alone reproduces row 1.
    c = draw_shocks([8], n_steps=10)
    np.testing.assert_array_equal(a["z"][1], c["z"][0])
    # Different seeds give different paths.
    assert not np.array_equal(a["z"][0], a["z"][1])
