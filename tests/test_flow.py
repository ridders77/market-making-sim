"""Tests for stage 3 adverse-selection measurements: markouts, OFI, prediction stats, drift decay, fill tilt."""

import numpy as np
import pandas as pd
import pytest

from mms.calibration.flow import (
    fit_drift_decay,
    fit_tilt,
    forward_move,
    markouts,
    ols,
    prediction_stats,
    rolling_ofi,
)


def test_markouts_hand_worked():
    # 1-second snapshots (mid in force just before each second): 100.05 for t < 3 s, then 101.05.
    snap = pd.DataFrame({"time_s": np.arange(70), "mid": np.r_[np.full(3, 100.05), np.full(67, 101.05)]})
    trades = pd.DataFrame({
        "time": [1500, 1500],                  # ms
        "price": [100.1, 100.0],
        "is_buyer_maker": [False, True],       # 1st: aggressive buy (maker sold); 2nd: aggressive sell (maker bought)
        "bid": [100.0, 100.0], "ask": [100.1, 100.1],
    })
    m = markouts(trades, snap, horizons=(1, 60))
    # delta = |price - mid_before| = 0.05 for both
    np.testing.assert_allclose(m.delta, [0.05, 0.05])
    # h = 1 s: mid at the first second >= 1.5 + 1 = 2.5 s -> snapshot 3 -> 101.05: up 1.0.
    # Maker who sold: the rise is against them, +1.0. Maker who bought: the rise is in their favour, -1.0.
    np.testing.assert_allclose(m.adverse_1, [1.0, -1.0])
    np.testing.assert_allclose(m.adverse_60, [1.0, -1.0])
    np.testing.assert_allclose(m.adverse_bps_1, [1e4 * 1.0 / 100.05, -1e4 * 1.0 / 100.05])
    # A horizon past the end of the day is NaN, not zero
    m2 = markouts(trades, snap, horizons=(70,))
    assert m2.adverse_70.isna().all()


def test_rolling_ofi_and_imbalance():
    # per-second buy / sell volume
    buy = np.array([1.0, 0.0, 2.0, 0.0, 0.0])
    sell = np.array([0.0, 1.0, 1.0, 0.0, 0.0])
    ofi, imb = rolling_ofi(buy, sell, n=2)
    # OFI at second s uses seconds s-2 and s-1 only (trades before s): s=2: (1-0)+(0-1) = 0; s=3: -1 + 1 = 0;
    # s=4: (2-1) + 0 = 1. Imbalance = OFI / volume: s=2: 0/2 = 0; s=3: 0/4; s=4: 1/3.
    np.testing.assert_allclose(ofi, [np.nan, np.nan, 0.0, 0.0, 1.0])
    np.testing.assert_allclose(imb, [np.nan, np.nan, 0.0, 0.0, 1 / 3])
    # no volume at all -> imbalance undefined (NaN), never zero-filled
    _, imb2 = rolling_ofi(np.zeros(4), np.zeros(4), n=1)
    assert np.isnan(imb2).all()


def test_forward_move_bps():
    mid = np.array([100.0, 101.0, 99.0, np.nan])
    np.testing.assert_allclose(forward_move(mid, 1), [100.0, -1e4 * 2 / 101, np.nan, np.nan])


def test_ols_and_prediction_stats_hand_worked():
    # y = 1 + 2x exactly: slope 2, intercept 1, R^2 = 1
    x = np.array([0.0, 1, 2, 3])
    f = ols(x, 1 + 2 * x)
    assert f["slope"] == pytest.approx(2) and f["intercept"] == pytest.approx(1) and f["r2"] == pytest.approx(1)
    # Out of sample with a given slope: prediction yhat = 0 + 1*x; y = [1, -1, 2, 0], x = [1, 1, 1, -1]
    # R^2 vs the zero forecast = 1 - sum (y - yhat)^2 / sum y^2 = 1 - (0 + 4 + 1 + 1) / (1 + 1 + 4 + 0) = 0
    # Hit rate: signs of yhat vs y where both non-zero: (+,+) (+,-) (+,+) -> y=0 excluded -> 2/3
    s = prediction_stats(np.array([1.0, 1, 1, -1]), np.array([1.0, -1, 2, 0]), slope=1.0, intercept=0.0)
    assert s["r2_oos"] == pytest.approx(0.0)
    assert s["hit_rate"] == pytest.approx(2 / 3)
    assert s["n"] == 4 and s["n_hit"] == 3


def test_fit_drift_decay_recovers_ou_parameters():
    # beta(h) = kappa tau (1 - exp(-h/tau)) with kappa = 0.1, tau = 4
    h = np.array([1, 2, 5, 10, 30, 60.0])
    beta = 0.1 * 4 * (1 - np.exp(-h / 4))
    f = fit_drift_decay(h, beta)
    assert f["tau"] == pytest.approx(4.0, rel=0.02)
    assert f["kappa"] == pytest.approx(0.1, rel=0.02)


def test_fit_tilt_recovers_theta():
    # Model: lambda_ask(x) / lambda_bid(x) = exp(2 theta x) -> log ratio slope in x is 2 theta.
    rng = np.random.default_rng(0)
    theta, n = 0.3, 400_000
    x = rng.standard_normal(n)
    p = 0.05
    ask = rng.random(n) < p * np.exp(theta * x - theta**2 / 2)
    bid = rng.random(n) < p * np.exp(-theta * x - theta**2 / 2)
    f = fit_tilt(x, ask, bid, n_bins=10)
    assert f["theta"] == pytest.approx(theta, abs=0.02)
