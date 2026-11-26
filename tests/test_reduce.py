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
