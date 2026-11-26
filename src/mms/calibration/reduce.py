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


# ---------------------------------------------------------------------------- stage 4: 100 ms book
BUCKET_MS = 100


def snapshots_100ms(book: pd.DataFrame, start_ms: int, end_ms: int, bucket_ms: int = BUCKET_MS) -> pd.DataFrame:
    """Best bid/ask state at the end of each 100 ms bucket, plus the min/max size seen during the bucket.

    Bucket b covers [start + b*100, start + (b+1)*100) ms; its row is stamped t_ms = the bucket's end and
    holds the last update in the bucket (or the carried-in state if there was none: the book did not
    change, so this is not a fill-in). NaN before the day's first update.

    ``bid_qty_min`` / ``bid_qty_max``: smallest / largest displayed size at the bucket's *final* bid price
    among the observations of that price during the bucket: the carried-in state (if it was at that price)
    and every update in the bucket at that price. Sizes at other prices are a different queue and are
    ignored. The replay uses the min as a hard cap on the queue ahead of us: orders ahead can only leave,
    so the queue ahead at the end of the bucket is at most the smallest size the level showed.
    """
    book = sort_book(book)
    t = book["time"].to_numpy(np.int64)
    if len(t) and (t.min() < start_ms or t.max() >= end_ms):
        book = book[(t >= start_ms) & (t < end_ms)].reset_index(drop=True)
        t = book["time"].to_numpy(np.int64)
    nb = (end_ms - start_ms) // bucket_ms
    b = (t - start_ms) // bucket_ms
    out = {"t_ms": start_ms + (np.arange(nb, dtype=np.int64) + 1) * bucket_ms}
    # index of the last update in each bucket (rows are sorted, so the last occurrence)
    has = np.zeros(nb, dtype=bool)
    last_idx = np.full(nb, -1, dtype=np.int64)
    last_idx[b] = np.arange(len(b))          # later rows overwrite earlier ones
    has[b] = True
    # carried state: index of the last update at or before the bucket
    carry = np.maximum.accumulate(np.where(has, last_idx, -1))
    ok = carry >= 0
    out["n_updates"] = np.bincount(b, minlength=nb).astype(np.int32)
    for side in ("bid", "ask"):
        px, qty = book[side].to_numpy(float), book[f"{side}_qty"].to_numpy(float)
        P, Q = np.full(nb, np.nan), np.full(nb, np.nan)
        P[ok], Q[ok] = px[carry[ok]], qty[carry[ok]]
        # rows at their bucket's final price
        at_final = px == P[b]
        qmin = np.full(nb, np.inf)
        qmax = np.full(nb, -np.inf)
        np.minimum.at(qmin, b[at_final], qty[at_final])
        np.maximum.at(qmax, b[at_final], qty[at_final])
        # carried-in observation (state at the end of the previous bucket) if at the same price
        Pprev = np.concatenate([[np.nan], P[:-1]])
        Qprev = np.concatenate([[np.nan], Q[:-1]])
        same = Pprev == P
        qmin[same] = np.minimum(qmin[same], Qprev[same])
        qmax[same] = np.maximum(qmax[same], Qprev[same])
        qmin[~ok], qmax[~ok] = np.nan, np.nan
        out[side], out[f"{side}_qty"], out[f"{side}_qty_min"], out[f"{side}_qty_max"] = P, Q, qmin, qmax
    s = pd.DataFrame(out)
    n_missing = int((~ok).sum())
    if n_missing:
        log.info("100 ms snapshots: %d bucket(s) before the first book update are NaN", n_missing)
    return s


def bucket_trades(trades: pd.DataFrame, start_ms: int, bucket_ms: int = BUCKET_MS) -> pd.DataFrame:
    """Add the 100 ms bucket index and the aggressor side (+1 buyer-initiated, i.e. it lifted the ask;
    -1 seller-initiated). ``is_buyer_maker`` True means the resting order was a bid, so the seller was
    the aggressor."""
    out = trades.copy()
    out["bucket"] = ((out["time"].to_numpy(np.int64) - start_ms) // bucket_ms).astype(np.int32)
    out["side"] = np.where(out["is_buyer_maker"].to_numpy(bool), -1, 1).astype(np.int8)
    return out
