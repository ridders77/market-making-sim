"""Tests for daily realised volatility and the stratified day-selection rule."""

import datetime as dt

import numpy as np
import pandas as pd
import pytest

from mms.calibration.day_selection import daily_realised_vol, pick_quantile_days, select_days


def _klines(day: str, closes: list[float]) -> pd.DataFrame:
    t0 = int(pd.Timestamp(day, tz="UTC").timestamp() * 1000)
    return pd.DataFrame({"open_time": [t0 + 60_000 * i for i in range(len(closes))], "close": closes})


def test_daily_realised_vol_hand_worked():
    # Closes 100, 110, 99: log returns ln(1.1) = 0.0953102, ln(0.9) = -0.1053605.
    # RV = sqrt(0.0953102^2 + 0.1053605^2) = sqrt(0.0090840 + 0.0111008) = 0.1420732 -> 14.207 %
    k = _klines("2023-06-01", [100, 110, 99])
    rv = daily_realised_vol(k)
    assert rv.loc[dt.date(2023, 6, 1)] == pytest.approx(14.20732, rel=1e-5)


def test_daily_realised_vol_does_not_cross_midnight():
    # A jump between the last bar of day 1 and the first bar of day 2 must not count.
    k = pd.concat([_klines("2023-06-01", [100, 100]), _klines("2023-06-02", [200, 200])])
    rv = daily_realised_vol(k)
    assert rv.loc[dt.date(2023, 6, 1)] == 0
    assert rv.loc[dt.date(2023, 6, 2)] == 0


def test_pick_quantile_days_midpoints():
    # 6 days with RV 1..6 (m = 6); pick n = 3 at quantiles q_i = (2i+1)/(2n) = 1/6, 1/2, 5/6.
    # Sorted position floor(q_i * m) = floor(1), floor(3), floor(5) = 1, 3, 5 -> RV 2, 4, 6.
    s = pd.Series([1.0, 2, 3, 4, 5, 6], index=pd.date_range("2023-06-01", periods=6).date)
    picked = pick_quantile_days(s, 3)
    assert len(picked) == 3
    assert list(s.loc[picked]) == [2.0, 4.0, 6.0]


def test_select_days_split_and_regimes():
    rng = np.random.default_rng(0)
    days = pd.date_range("2023-05-16", periods=120).date
    rv = pd.Series(rng.lognormal(0.7, 0.4, len(days)), index=days)
    sel = select_days(rv, counts={"low": 3, "medium": 4, "high": 3})
    assert len(sel) == 20
    cal, test = sel[sel.split == "calibration"], sel[sel.split == "test"]
    assert len(cal) == len(test) == 10
    assert cal.day.max() < test.day.min()  # test is strictly out of time
    for part in (cal, test):
        assert part.regime.value_counts().to_dict() == {"medium": 4, "low": 3, "high": 3}
    # regimes are global terciles
    lo, hi = rv.quantile([1 / 3, 2 / 3])
    assert (sel[sel.regime == "low"].rv < lo).all()
    assert (sel[sel.regime == "high"].rv >= hi).all()
