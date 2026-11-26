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
