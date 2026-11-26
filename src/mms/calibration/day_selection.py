r"""Choose which days to calibrate and test on, using daily realised volatility.

Daily realised volatility
-------------------------
From 1-minute klines with closes C_0, ..., C_n on one UTC day, the log returns
are r_i = ln(C_i / C_{i-1}) and the realised volatility of the day is

    RV = sqrt( sum_i r_i^2 ).

Why: if log prices follow a Brownian motion with volatility s per sqrt(day),
E[sum r_i^2] = s^2 (sum of the n variances s^2 / n), so RV estimates the
day's volatility without needing a mean. Returns are taken only *within* a day
(the first bar of a day is not compared with the last bar of the day before),
so each day's value depends only on that day. Reported in percent.

Selection rule (deterministic, no tuning)
-----------------------------------------
1. Regimes: global terciles of RV over the whole available range
   (low < 1/3 quantile <= medium < 2/3 quantile <= high).
2. Out-of-time split: the first half of the days (by date) is the calibration
   pool, the second half the test pool. All calibration days precede all test days.
3. In each pool and each regime, sort the days by RV and take n days at the
   within-group quantiles q_i = (2i+1)/(2n), i = 0..n-1 (sorted position
   floor(q_i * m) of m days). This spreads the picks evenly over the regime
   instead of taking its extremes.
"""

from __future__ import annotations

import datetime as dt
import logging

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

REGIMES = ("low", "medium", "high")


def daily_realised_vol(klines: pd.DataFrame) -> pd.Series:
    """Realised volatility (%) per UTC day from 1-minute klines (columns open_time [ms], close)."""
    t = pd.to_datetime(klines["open_time"], unit="ms", utc=True)
    day = t.dt.date
    logc = np.log(klines["close"].astype(float))
    r = logc.groupby(day.values).diff()  # NaN for the first bar of each day
    rv = np.sqrt((r**2).groupby(day.values).sum()) * 100
    rv.index.name = "day"
    return rv.rename("rv_pct")


def pick_quantile_days(rv: pd.Series, n: int) -> list[dt.date]:
    """Pick n days spread evenly (by RV rank) over the days in ``rv``."""
    m = len(rv)
    if m < n:
        raise ValueError(f"need {n} days, group has only {m}")
    order = rv.sort_values(kind="stable").index
    positions = [((2 * i + 1) * m) // (2 * n) for i in range(n)]
    return sorted(order[p] for p in positions)


def select_days(rv: pd.Series, counts: dict[str, int]) -> pd.DataFrame:
    """Apply the rule in the module docstring. Returns columns day, rv, regime, split."""
    rv = rv.sort_index()
    lo, hi = rv.quantile([1 / 3, 2 / 3])
    regime = pd.Series(np.where(rv < lo, "low", np.where(rv < hi, "medium", "high")), index=rv.index)
    half = len(rv) // 2
    pools = {"calibration": rv.index[:half], "test": rv.index[half:]}
    rows = []
    for split, days in pools.items():
        for reg in REGIMES:
            group = rv.loc[[d for d in days if regime[d] == reg]]
            for d in pick_quantile_days(group, counts[reg]):
                rows.append({"day": d, "rv": float(rv[d]), "regime": reg, "split": split})
    log.info("RV terciles: low < %.3f%% <= medium < %.3f%% <= high; split at %s", lo, hi, rv.index[half])
    return pd.DataFrame(rows).sort_values("day").reset_index(drop=True)
