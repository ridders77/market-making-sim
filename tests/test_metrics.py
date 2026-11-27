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


def test_zero_crossing_hand_worked():
    from mms.analysis.metrics import zero_crossing
    # P&L -10 at f = 1, -4 at 0.5, +2 at 0.25: crosses between 0.5 and 0.25, at 0.5 - 0.25 * 4/6
    assert zero_crossing([1, 0.5, 0.25, 0], [-10, -4, 2, 5]) == pytest.approx(0.5 - 0.25 * 4 / 6)
    assert np.isnan(zero_crossing([1, 0.5, 0.25, 0], [-10, -4, -2, -1]))      # never breaks even
    assert zero_crossing([0, 0.5, 1], [3, 1, -1]) == pytest.approx(0.75)        # order of the grid does not matter


def test_break_even_bracket_hand_worked():
    from mms.analysis.metrics import break_even_bracket
    # loses at f = 1, 0.5, 0.25, 0.1; profitable at 0.05 and 0 -> bracket (0.1, 0.05)
    assert break_even_bracket([1, 0.5, 0.25, 0.1, 0.05, 0], [-9, -6, -3, -1, 2, 50]) == (0.1, 0.05)
    lo, hi = break_even_bracket([1, 0.5, 0], [-9, -6, -1])
    assert np.isnan(lo) and np.isnan(hi)                    # never profitable
    lo, hi = break_even_bracket([0, 0.5, 1], [5, 3, 1])     # profitable even at the back
    assert np.isnan(lo) and hi == 1.0
