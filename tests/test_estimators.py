"""Tests for the calibration estimators: sigma, touch fractions, intensities, exponential fit."""

import numpy as np
import pandas as pd
import pytest

from mms.calibration.estimators import (
    bootstrap_days,
    fit_exponential,
    intensity_from_fraction,
    realised_sigma,
    touch_curve,
    touch_excursions,
)


# ---------------------------------------------------------------- sigma
def test_realised_sigma_hand_worked():
    # mid 100, 101, 100, 102 sampled every 1 s: increments 1, -1, 2.
    # sigma^2 = (1 + 1 + 4) / (3 * 1 s) = 2  ->  sigma = sqrt(2) = 1.41421 per sqrt(s)
    assert realised_sigma(np.array([100, 101, 100, 102.0]), dt_s=1.0) == pytest.approx(np.sqrt(2))


def test_realised_sigma_subsampling_and_nan():
    # Every 2nd point, dt = 2 s: 100 -> 100 -> 102: increments 0, 2 -> sigma^2 = 4 / (2 * 2) = 1
    assert realised_sigma(np.array([100, 101, 100, 105, 102.0]), dt_s=1.0, step=2) == pytest.approx(1.0)
    # NaN increments are dropped, not treated as zero: 100, NaN, 101, 103 -> only 101->103 is valid
    assert realised_sigma(np.array([100, np.nan, 101, 103.0]), dt_s=1.0) == pytest.approx(2.0)


def test_realised_sigma_recovers_simulated_abm():
    rng = np.random.default_rng(1)
    sigma, dt = 2.5, 1.0
    mid = 30000 + np.cumsum(sigma * np.sqrt(dt) * rng.standard_normal(200_000))
    assert realised_sigma(mid, dt) == pytest.approx(sigma, rel=0.01)


# ---------------------------------------------------------------- touch excursions
def _snap_trades():
    # Snapshots at t = 0, 1, 2, 3 s with mids 100.05, 100.05, 100.25, 100.05
    snap = pd.DataFrame({"time_s": [0, 1, 2, 3], "mid": [100.05, 100.05, 100.25, 100.05]})
    trades = pd.DataFrame({
        "time": [100, 500, 900, 1200, 2100, 2200, 2999, 3500],
        "price": [100.1, 100.3, 100.0, 99.9, 100.3, 100.2, 100.6, 100.1],
        "is_buyer_maker": [False, False, True, True, False, True, False, False],
    })
    return snap, trades


def test_touch_excursions_one_second_windows():
    snap, trades = _snap_trades()
    w = touch_excursions(snap, trades, window_s=1)
    # window [0,1): buys 100.1, 100.3 -> ask exc = 100.3 - 100.05 = 0.25; sell 100.0 -> bid exc 0.05
    # window [1,2): only a sell at 99.9 -> ask exc = -inf (no buy), bid exc = 0.15
    # window [2,3): mid0 = 100.25; buys 100.3, 100.6 -> 0.35; sell 100.2 -> 0.05
    # window [3,4): buy 100.1 vs mid0 100.05 -> 0.05; no sell -> -inf
    np.testing.assert_allclose(w.ask_exc, [0.25, -np.inf, 0.35, 0.05])
    np.testing.assert_allclose(w.bid_exc, [0.05, 0.15, 0.05, -np.inf])


def test_touch_excursions_two_second_windows_use_mid_at_start():
    snap, trades = _snap_trades()
    w = touch_excursions(snap, trades, window_s=2)
    # window [0,2): mid0 100.05; max buy 100.3 -> 0.25; min sell 99.9 -> 0.15
    # window [2,4): mid0 100.25; max buy 100.6 -> 0.35; min sell 100.2 -> 0.05
    np.testing.assert_allclose(w.ask_exc, [0.25, 0.35])
    np.testing.assert_allclose(w.bid_exc, [0.15, 0.05])


def test_touch_excursions_drop_windows_without_mid():
    snap, trades = _snap_trades()
    snap.loc[0, "mid"] = np.nan
    w = touch_excursions(snap, trades, window_s=1)
    assert list(w.time_s) == [1, 2, 3]


# ---------------------------------------------------------------- fractions and intensities
def test_touch_curve_counts():
    exc = np.array([0.25, -np.inf, 0.35, 0.05])
    hits, n = touch_curve(exc, deltas=np.array([0.0, 0.1, 0.3, 0.4]))
    # >= 0: 3 windows; >= 0.1: 0.25, 0.35 -> 2; >= 0.3: 1; >= 0.4: 0
    np.testing.assert_array_equal(hits, [3, 2, 1, 0])
    assert n == 4


def test_touch_curve_boundary_is_inclusive_despite_float_noise():
    # 100.15 - 100.05 = 0.10000000000000853 in floating point, 100.25 - 100.15 = 0.0999999999999943
    exc = np.array([100.25 - 100.15])
    hits, _ = touch_curve(exc, np.array([0.1]))
    assert hits[0] == 1


def test_intensity_from_fraction_hand_worked():
    # p = 1 - e^{-1} = 0.6321206 over 2 s  ->  lambda = 1 / 2 = 0.5 per s
    assert intensity_from_fraction(1 - np.exp(-1), 2.0) == pytest.approx(0.5)
    # small p: lambda ~ p / dt.  p = 0.01, dt = 1: -ln(0.99) = 0.0100503
    assert intensity_from_fraction(0.01, 1.0) == pytest.approx(0.01005034, rel=1e-6)
    assert np.isinf(intensity_from_fraction(1.0, 1.0))


# ---------------------------------------------------------------- exponential fit
def test_fit_exponential_exact_data():
    d = np.array([0.0, 0.5, 1.0, 1.5])
    lam = 2.0 * np.exp(-3.0 * d)
    fit = fit_exponential(d, lam)
    assert fit["A"] == pytest.approx(2.0)
    assert fit["k"] == pytest.approx(3.0)
    assert fit["r2"] == pytest.approx(1.0)
    np.testing.assert_allclose(fit["residuals"], 0, atol=1e-12)


def test_fit_exponential_hand_worked_least_squares():
    # y = ln(lambda) = [0, -1, -1] at x = [0, 1, 2].  x_bar = 1, y_bar = -2/3.
    # slope = sum (x - x_bar)(y - y_bar) / sum (x - x_bar)^2 = [(-1)(2/3) + 0 + (1)(-1/3)] / 2 = -1/2
    # intercept = y_bar - slope x_bar = -2/3 + 1/2 = -1/6  ->  A = e^{-1/6}, k = 0.5
    # fitted = [-1/6, -2/3, -7/6], residuals = [1/6, -1/3, 1/6]; SSres = 1/6, SStot = 2/3 -> R^2 = 0.75
    fit = fit_exponential(np.array([0, 1, 2.0]), np.exp([0, -1, -1.0]))
    assert fit["k"] == pytest.approx(0.5)
    assert fit["A"] == pytest.approx(np.exp(-1 / 6))
    assert fit["r2"] == pytest.approx(0.75)
    np.testing.assert_allclose(fit["residuals"], [1 / 6, -1 / 3, 1 / 6])


def test_fit_exponential_rejects_nonpositive_or_infinite():
    with pytest.raises(ValueError):
        fit_exponential(np.array([0, 1.0]), np.array([1.0, 0.0]))
    with pytest.raises(ValueError):
        fit_exponential(np.array([0, 1.0]), np.array([np.inf, 1.0]))


def test_bootstrap_days_recovers_parameters_and_brackets():
    # 10 "days", each with hits generated from lambda = 2 exp(-3 delta) over 1 s windows.
    rng = np.random.default_rng(0)
    deltas = np.arange(0, 1.01, 0.1)
    p = 1 - np.exp(-2 * np.exp(-3 * deltas))
    days = {d: (rng.binomial(20_000, p), 20_000) for d in range(10)}
    out = bootstrap_days(days, deltas, window_s=1.0, n_boot=200, seed=0)
    assert out["A"][1] <= 2.0 <= out["A"][2]
    assert out["k"][1] <= 3.0 <= out["k"][2]
    assert out["A"][0] == pytest.approx(2.0, rel=0.02)


# ---------------------------------------------------------------- sigma-conditional model
def test_fit_loglog_hand_worked():
    # y = 3 x^2 exactly: ln y = ln 3 + 2 ln x
    from mms.calibration.estimators import fit_loglog
    x = np.array([1.0, 2.0, 4.0])
    f = fit_loglog(x, 3 * x**2)
    assert f["slope"] == pytest.approx(2.0)
    assert f["scale"] == pytest.approx(3.0)
    assert f["r2"] == pytest.approx(1.0)
    assert f["predict"](3.0) == pytest.approx(27.0)


def test_fit_loglog_least_squares_values():
    # ln x = [0, 1, 2], ln y = [0, -1, -1] -> same numbers as the linear hand-worked test:
    # slope -1/2, intercept -1/6, so y = e^{-1/6} x^{-1/2}, R^2 = 0.75
    from mms.calibration.estimators import fit_loglog
    f = fit_loglog(np.exp([0, 1, 2.0]), np.exp([0, -1, -1.0]))
    assert f["slope"] == pytest.approx(-0.5)
    assert f["scale"] == pytest.approx(np.exp(-1 / 6))
    assert f["r2"] == pytest.approx(0.75)
