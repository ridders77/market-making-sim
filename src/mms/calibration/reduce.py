"""Reduce one day of raw Binance data to small, analysis-ready tables.

Inputs (one UTC day, BTCUSDT USD-M futures, millisecond timestamps):

* bookTicker: every change of the best bid / best ask (~40 million rows a day).
  Columns used: time (= transaction_time, matching-engine time), update_id,
  bid, bid_qty, ask, ask_qty.
* aggTrades: every trade, aggregated by taker order and price (~1-2.5 million rows).
  ``is_buyer_maker`` True means the buyer's order was resting, i.e. the
  *seller* was the aggressor (a sell market order hit the bid).

Outputs:

1. 1-second snapshots. For each second t on the grid, the best bid/ask *in force
   just before* t: the last update with time < t (ties at the same millisecond
   broken by update_id). Plus mid = (bid + ask)/2, spread = ask - bid,
   quote_age_ms (how stale that quote is) and n_updates in [t - 1 s, t).
   Before the day's first update the snapshot is NaN, not zero-filled.
2. Trades with the prevailing quote: each trade gets the best bid/ask in force
   strictly before the trade's timestamp (pandas merge_asof, direction
   "backward", allow_exact_matches=False). "Strictly" because a book update
   stamped at the same millisecond as a trade is usually that trade's own
   effect on the book, so it is not the quote the aggressor saw.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

QUOTE_COLS = ["bid", "bid_qty", "ask", "ask_qty"]


def sort_book(book: pd.DataFrame) -> pd.DataFrame:
    """Order updates by time, then update_id (same-millisecond updates)."""
    return book.sort_values(["time", "update_id"], kind="stable").reset_index(drop=True)


def snapshots_1s(book: pd.DataFrame, start_ms: int, end_ms: int) -> pd.DataFrame:
    """Best quote in force just before each second t in [start_ms, end_ms)."""
    book = sort_book(book)
    times = book["time"].to_numpy(np.int64)
    grid = np.arange(start_ms, end_ms, 1000, dtype=np.int64)
    # index of the last update with time < t (side="left" gives the first index with time >= t)
    idx = np.searchsorted(times, grid, side="left") - 1
    ok = idx >= 0
    out = pd.DataFrame({"time_s": grid // 1000})
    for c in QUOTE_COLS:
        vals = np.full(len(grid), np.nan)
        vals[ok] = book[c].to_numpy(float)[idx[ok]]
        out[c] = vals
    out["mid"] = (out["bid"] + out["ask"]) / 2
    out["spread"] = out["ask"] - out["bid"]
    age = np.full(len(grid), np.nan)
    age[ok] = grid[ok] - times[idx[ok]]
    out["quote_age_ms"] = age
    n_before = np.searchsorted(times, grid, side="left")
    n_before_prev = np.searchsorted(times, grid - 1000, side="left")
    out["n_updates"] = (n_before - n_before_prev).astype(np.int32)
    n_missing = int((~ok).sum())
    if n_missing:
        log.info("snapshots: %d second(s) before the first book update are NaN", n_missing)
    return out


def attach_quotes(trades: pd.DataFrame, book: pd.DataFrame) -> pd.DataFrame:
    """Add the best bid/ask (and its timestamp) in force strictly before each trade."""
    book = sort_book(book)
    right = book[["time", *QUOTE_COLS]].rename(columns={"time": "quote_time"})
    right["time"] = right["quote_time"]
    left = trades.reset_index(drop=True).copy()
    left["_order"] = np.arange(len(left))
    left = left.sort_values("time", kind="stable")
    out = pd.merge_asof(left, right, on="time", direction="backward", allow_exact_matches=False)
    out = out.sort_values("_order").drop(columns="_order").reset_index(drop=True)
    n_missing = int(out["bid"].isna().sum())
    if n_missing:
        log.info("trades: %d trade(s) before the first book update have no quote (NaN)", n_missing)
    return out


def book_quality(book: pd.DataFrame, start_ms: int, end_ms: int) -> dict:
    """Simple checks: number of updates, crossed/locked quotes, longest gap without updates."""
    times = np.sort(book["time"].to_numpy(np.int64))
    edges = np.concatenate([[start_ms], times, [end_ms]])
    return {
        "n_updates": int(len(book)),
        "n_crossed": int((book["bid"] >= book["ask"]).sum()),
        "max_gap_ms": int(np.diff(edges).max()),
    }
