"""Tests for bootstrap confidence intervals and summary metrics."""

import numpy as np
import pytest

from mms.analysis.metrics import bootstrap_ci, sharpe_like


def test_bootstrap_ci_covers_mean_of_normal_sample():
    # For n = 2000 draws from N(5, 1), the 95% CI of the mean should be about
    # 5 +/- 1.96/sqrt(2000) = 5 +/- 0.044.
    x = np.random.default_rng(0).normal(5.0, 1.0, 2000)
    est, lo, hi = bootstrap_ci(x, np.mean, n_boot=2000, seed=1)
    assert est == pytest.approx(x.mean())
    assert lo < est < hi
    assert hi - lo == pytest.approx(2 * 1.96 / np.sqrt(2000), rel=0.15)


def test_bootstrap_ci_reproducible():
    x = np.arange(100.0)
    assert bootstrap_ci(x, np.std, seed=3) == bootstrap_ci(x, np.std, seed=3)


def test_sharpe_like():
    # mean 2, population std 1 (ddof=1 on [1,3]: std = sqrt(2))
    assert sharpe_like(np.array([1.0, 3.0])) == pytest.approx(2 / np.sqrt(2))


def test_ratio_of_means_ci_hand_worked():
    from mms.analysis.metrics import ratio_of_means_ci
    # P&L per notional in bps = 1e4 * sum(pnl) / sum(notional) (volume-weighted margin):
    # pnl = [1, 2, 3] USDT on notional [10000, 10000, 20000] -> 1e4 * 6 / 40000 = 1.5 bp.
    est, lo, hi = ratio_of_means_ci(np.array([1.0, 2, 3]), np.array([1e4, 1e4, 2e4]), scale=1e4, n_boot=500)
    assert est == pytest.approx(1.5)
    assert lo <= est <= hi
    # Constant ratio: every resample gives the same value, so the CI collapses.
    est, lo, hi = ratio_of_means_ci(np.array([2.0, 4.0]), np.array([1.0, 2.0]), n_boot=200)
    assert est == lo == hi == pytest.approx(2.0)
