"""Summary metrics with percentile-bootstrap 95% confidence intervals.

Bootstrap (percentile method): resample the n per-path outcomes with
replacement B times, compute the statistic on each resample, and take the
2.5% and 97.5% quantiles of those B values as the 95% CI. It needs no
normality assumption, which matters for skewed P&L distributions and for
statistics like the std or the Sharpe-like ratio whose standard errors are
awkward to write down.

Paired comparison: strategies run on the same seeds, so for a difference
(e.g. A-S P&L minus symmetric P&L) we resample *paths* and keep each path's
pair together. This removes the shared price-path noise from the comparison.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd

from mms.sim.simulator import SimResult


def sharpe_like(x: np.ndarray) -> float:
    """mean / std (ddof=1) of final P&L across paths. Not annualised."""
    return float(np.mean(x) / np.std(x, ddof=1))


def _std(x: np.ndarray, axis=None) -> np.ndarray:
    return np.std(x, ddof=1, axis=axis)


def bootstrap_ci(
    x: np.ndarray,
    stat: Callable[[np.ndarray], float],
    n_boot: int = 2000,
    seed: int = 0,
    level: float = 0.95,
) -> tuple[float, float, float]:
    """Return (estimate on full sample, lower, upper) percentile-bootstrap CI."""
    x = np.asarray(x, dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(n_boot, len(x)))
    boots = np.array([stat(x[row]) for row in idx])
    alpha = (1 - level) / 2
    lo, hi = np.quantile(boots, [alpha, 1 - alpha])
    return float(stat(x)), float(lo), float(hi)


def _metric_columns(res: SimResult) -> dict[str, tuple[np.ndarray, Callable]]:
    """Each reported metric = (per-path array, statistic over paths)."""
    return {
        "mean_pnl": (res.pnl, np.mean),
        "std_pnl": (res.pnl, _std),
        "sharpe_like": (res.pnl, sharpe_like),
        "mean_final_q": (res.q, np.mean),
        "std_final_q": (res.q, _std),
        "mean_max_abs_q": (res.max_abs_q, np.mean),
        "fill_rate": (res.fill_rate, np.mean),
        "avg_spread_captured": (res.avg_spread_captured, np.nanmean),
        "mean_quoted_spread": (res.mean_quoted_spread, np.mean),
    }


def summarise(res: SimResult, n_boot: int = 2000, seed: int = 0) -> pd.DataFrame:
    """One row per metric: estimate and 95% bootstrap CI."""
    rows = []
    for name, (x, stat) in _metric_columns(res).items():
        est, lo, hi = bootstrap_ci(x, stat, n_boot, seed)
        rows.append({"metric": name, "estimate": est, "ci_low": lo, "ci_high": hi})
    return pd.DataFrame(rows)


def paired_difference(a: np.ndarray, b: np.ndarray, stat: Callable = np.mean,
                      n_boot: int = 2000, seed: int = 0) -> tuple[float, float, float]:
    """CI for stat(a) - stat(b), resampling paths jointly (a, b from the same seeds)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.shape != b.shape:
        raise ValueError("paired arrays must have the same shape")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(a), size=(n_boot, len(a)))
    boots = np.array([stat(a[r]) - stat(b[r]) for r in idx])
    lo, hi = np.quantile(boots, [0.025, 0.975])
    return float(stat(a) - stat(b)), float(lo), float(hi)
