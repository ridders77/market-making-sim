"""Tests for reducing raw bookTicker / aggTrades to 1-second snapshots and quote-tagged trades."""

import numpy as np
import pandas as pd
import pytest

from mms.calibration.reduce import attach_quotes, book_quality, snapshots_1s


def _book() -> pd.DataFrame:
    # Two updates share the timestamp 1000 ms; update_id decides their order.
    return pd.DataFrame({
        "time": [500, 1000, 1000, 2500],
        "update_id": [1, 3, 2, 4],          # deliberately out of order
        "bid": [10.0, 10.0, 10.1, 10.2],
        "bid_qty": [1.0, 2.0, 3.0, 4.0],
        "ask": [10.1, 10.2, 10.2, 10.3],
        "ask_qty": [5.0, 6.0, 7.0, 8.0],
    })


def test_snapshots_use_state_strictly_before_each_second():
    snap = snapshots_1s(_book(), start_ms=0, end_ms=3000)
    assert list(snap.time_s) == [0, 1, 2]
    # t = 0: no update before -> missing (NaN), never zero-filled
    assert np.isnan(snap.mid.iloc[0])
    # t = 1000: the update at 500 (the ones *at* 1000 are not yet in force)
    r = snap.iloc[1]
    assert (r.bid, r.ask) == (10.0, 10.1)
    assert r.mid == pytest.approx(10.05)
    assert r.spread == pytest.approx(0.1)
    assert r.quote_age_ms == 500
    assert r.n_updates == 1
    # t = 2000: last update at 1000 is update_id 3 (bid 10.0 / ask 10.2), age 1000 ms;
    # two updates happened in [1000, 2000)
    r = snap.iloc[2]
    assert (r.bid, r.ask, r.bid_qty) == (10.0, 10.2, 2.0)
    assert r.quote_age_ms == 1000
    assert r.n_updates == 2


def test_attach_quotes_backward_strict():
    trades = pd.DataFrame({"time": [400, 1000, 1001, 3000], "price": [10.1, 10.2, 10.2, 10.3],
                           "qty": [1.0, 1.0, 1.0, 1.0], "is_buyer_maker": [False] * 4})
    out = attach_quotes(trades, _book())
    # trade at 400: no quote before it -> NaN
    assert np.isnan(out.bid.iloc[0])
    # trade at 1000: quote just before is the one at 500, not the updates at 1000
    assert (out.bid.iloc[1], out.ask.iloc[1], out.quote_time.iloc[1]) == (10.0, 10.1, 500)
    # trade at 1001: the last update at 1000 by update_id (3): bid 10.0, ask 10.2
    assert (out.bid.iloc[2], out.ask.iloc[2], out.bid_qty.iloc[2]) == (10.0, 10.2, 2.0)
    # trade at 3000: update at 2500
    assert out.ask.iloc[3] == 10.3
    assert list(out.price) == [10.1, 10.2, 10.2, 10.3]  # trade columns kept, order kept


def test_book_quality_counts():
    b = _book()
    b.loc[3, "ask"] = 10.2  # crossed/locked: bid 10.2 >= ask 10.2
    q = book_quality(b, start_ms=0, end_ms=5000)
    assert q["n_updates"] == 4
    assert q["n_crossed"] == 1
    assert q["max_gap_ms"] == 2500  # between 2500 and end 5000


# ---------------------------------------------------------------- stage 4: 100 ms snapshots
from mms.calibration.reduce import snapshots_100ms, bucket_trades


def _book100() -> pd.DataFrame:
    # bucket 0: [0, 100), bucket 1: [100, 200), bucket 2: [200, 300), bucket 3: [300, 400)
    return pd.DataFrame({
        "time":      [10, 50, 50, 120, 150, 180, 330],
        "update_id": [1, 3, 2, 4, 5, 6, 7],
        "bid":     [10.0, 10.0, 10.0, 10.0, 9.9, 10.0, 10.0],
        "bid_qty": [5.0, 2.0, 3.0, 4.0, 7.0, 6.0, 1.0],
        "ask":     [10.1, 10.1, 10.1, 10.1, 10.1, 10.1, 10.2],
        "ask_qty": [1.0, 1.0, 1.0, 2.0, 3.0, 4.0, 9.0],
    })


def test_snapshots_100ms_last_state_and_min_max_at_final_price():
    s = snapshots_100ms(_book100(), start_ms=0, end_ms=400)
    assert list(s.t_ms) == [100, 200, 300, 400]           # state at the END of each bucket
    # bucket 0: updates by (time, update_id): qty 5 (t10), 3 (id 2), 2 (id 3) -> last 2, min 2, max 5
    r = s.iloc[0]
    assert (r.bid, r.bid_qty, r.bid_qty_min, r.bid_qty_max) == (10.0, 2.0, 2.0, 5.0)
    assert r.n_updates == 3
    # bucket 1: carry-in 10.0 @ 2; updates 10.0 @ 4, 9.9 @ 7, 10.0 @ 6. Final price 10.0:
    # observations at 10.0: carry 2, 4, 6 -> min 2, max 6; the 9.9 @ 7 update is a different price
    r = s.iloc[1]
    assert (r.bid, r.bid_qty, r.bid_qty_min, r.bid_qty_max) == (10.0, 6.0, 2.0, 6.0)
    assert (r.ask, r.ask_qty, r.ask_qty_min, r.ask_qty_max) == (10.1, 4.0, 1.0, 4.0)
    # bucket 2: no updates -> state carried (the book did not change), min = max = last
    r = s.iloc[2]
    assert (r.bid, r.bid_qty, r.bid_qty_min, r.bid_qty_max, r.n_updates) == (10.0, 6.0, 6.0, 6.0, 0)
    # bucket 3: ask moves to 10.2 @ 9: carry-in was at 10.1, so min = max = 9
    r = s.iloc[3]
    assert (r.ask, r.ask_qty, r.ask_qty_min, r.ask_qty_max) == (10.2, 9.0, 9.0, 9.0)
    assert (r.bid_qty, r.bid_qty_min, r.bid_qty_max) == (1.0, 1.0, 6.0)


def test_snapshots_100ms_nan_before_first_update():
    s = snapshots_100ms(_book100().iloc[3:], start_ms=0, end_ms=200)
    assert np.isnan(s.bid.iloc[0]) and np.isnan(s.bid_qty_min.iloc[0])
    assert s.bid.iloc[1] == 10.0


def test_bucket_trades_index_and_aggressor_side():
    trades = pd.DataFrame({"time": [5, 99, 100, 350], "price": [1.0, 2.0, 3.0, 4.0], "qty": [1.0] * 4,
                           "is_buyer_maker": [True, False, False, True]})
    out = bucket_trades(trades, start_ms=0)
    assert list(out.bucket) == [0, 0, 1, 3]
    assert list(out.side) == [-1, 1, 1, -1]     # +1 buyer-initiated (hits the ask), -1 seller-initiated
