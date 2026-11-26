r"""One day of replay data: the 100 ms book, the trades per bucket, and the features a strategy may observe.

Time convention. Bucket i covers [t_i, t_{i+1}) with t_i = day start + i * 100 ms. Row i of the 100 ms table
is the book at the *end* of bucket i, i.e. at t_{i+1}. At decision time t_i a strategy sees row i - 1 (the
book in force at t_i) and the trades of buckets < i; then bucket i is replayed against our orders.

Features at t_i (all use only data before t_i):

* ofi_z: 5 s order-flow imbalance (B - S)/(B + S) of aggressive buy/sell volume in buckets i-50 .. i-1,
  divided by its calibration-day standard deviation 0.624 (stage 3's main signal). NaN when nothing traded.
* sigma_hat: stage 3's trailing volatility (EWMA, span 30 min, of squared 60 s mid changes, USDT/sqrt(s)),
  from ``mms.calibration.estimators.trailing_sigma`` on the 1 s grid. NaN in the first 30 minutes.
* typical_size: trailing 30-minute median of the average touch size (bid + ask)/2, one sample per second
  before t_i. Used to express queue sizes as a fraction of "normal".
"""

from __future__ import annotations

import functools
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from mms.calibration.estimators import trailing_sigma

REPO = Path(__file__).resolve().parents[3]
STAGE4 = REPO / "data" / "processed" / "stage4"
TICK = 0.1
BUCKET_MS = 100
BUCKETS_PER_S = 10
OFI_BUCKETS = 50                  # 5 s
OFI_Z_SCALE = 0.6244381102283721  # stage 3 calibration std of the 5 s imbalance (results/stage3/flow_calibration.json)
TYPICAL_WINDOW_S = 1800


def to_ticks(price) -> np.ndarray:
    return np.rint(np.asarray(price, float) / TICK).astype(np.int64)


@dataclass
class ReplayDay:
    day: str
    t_ms: np.ndarray          # end time of each bucket (ms)
    bid: np.ndarray           # ticks, row i = end of bucket i
    ask: np.ndarray
    bid_qty: np.ndarray       # BTC
    ask_qty: np.ndarray
    bid_qty_min: np.ndarray
    ask_qty_min: np.ndarray
    mid: np.ndarray           # USDT
    tr_start: np.ndarray      # trades of bucket i are tr_*[tr_start[i]:tr_start[i+1]]
    tr_price: np.ndarray      # ticks
    tr_qty: np.ndarray
    tr_side: np.ndarray       # +1 buyer-initiated, -1 seller-initiated
    sell_min: np.ndarray      # lowest seller-initiated price in bucket i (ticks; huge if none)
    buy_max: np.ndarray       # highest buyer-initiated price in bucket i (ticks; -huge if none)
    ofi_z: np.ndarray         # indexed by decision time i (uses buckets < i)
    sigma_hat: np.ndarray     # indexed by decision time i
    typical_size: np.ndarray  # indexed by decision time i

    @property
    def n_buckets(self) -> int:
        return len(self.t_ms)


def build_day(book: pd.DataFrame, trades: pd.DataFrame, day: str) -> ReplayDay:
    """Arrays and features from the stage 4 parquet tables (``snapshots_100ms`` and ``bucket_trades`` output)."""
    nb = len(book)
    bid, ask = book["bid"].to_numpy(float), book["ask"].to_numpy(float)
    ok = np.isfinite(bid) & np.isfinite(ask)
    bid_t = np.where(ok, np.rint(np.nan_to_num(bid) / TICK), -1).astype(np.int64)   # -1 marks "no book yet"
    ask_t = np.where(ok, np.rint(np.nan_to_num(ask) / TICK), -1).astype(np.int64)
    tr = trades[trades["bucket"].between(0, nb - 1)]
    b = tr["bucket"].to_numpy(np.int64)
    if np.any(np.diff(b) < 0):
        order = np.argsort(b, kind="stable")
        tr, b = tr.iloc[order], b[order]
    tr_price = to_ticks(tr["price"].to_numpy(float))
    tr_qty = tr["qty"].to_numpy(float)
    tr_side = tr["side"].to_numpy(np.int8)
    tr_start = np.searchsorted(b, np.arange(nb + 1))
    big = np.iinfo(np.int64).max // 4
    sell_min = np.full(nb, big, dtype=np.int64)
    buy_max = np.full(nb, -big, dtype=np.int64)
    s = tr_side < 0
    np.minimum.at(sell_min, b[s], tr_price[s])
    np.maximum.at(buy_max, b[~s], tr_price[~s])
    buy_vol = np.bincount(b[~s], weights=tr_qty[~s], minlength=nb)
    sell_vol = np.bincount(b[s], weights=tr_qty[s], minlength=nb)

    # OFI at decision time i: buckets i-50 .. i-1
    cb = np.concatenate([[0.0], np.cumsum(buy_vol)])
    cs = np.concatenate([[0.0], np.cumsum(sell_vol)])
    i = np.arange(nb)
    lo = np.maximum(i - OFI_BUCKETS, 0)
    B, S = cb[i] - cb[lo], cs[i] - cs[lo]
    with np.errstate(invalid="ignore", divide="ignore"):
        imb = np.where((B + S > 0) & (i >= OFI_BUCKETS), (B - S) / (B + S), np.nan)
    ofi_z = imb / OFI_Z_SCALE

    # state at decision time i = row i - 1
    mid = (bid + ask) / 2
    state_mid = np.concatenate([[np.nan], mid[:-1]])
    n_sec = nb // BUCKETS_PER_S
    snap1s = pd.DataFrame({"time_s": np.arange(n_sec), "mid": state_mid[::BUCKETS_PER_S][:n_sec]})
    sig_s = trailing_sigma(snap1s)
    sigma_hat = sig_s[np.minimum(i // BUCKETS_PER_S, n_sec - 1)]

    size = (book["bid_qty"].to_numpy(float) + book["ask_qty"].to_numpy(float)) / 2
    state_size = np.concatenate([[np.nan], size[:-1]])[::BUCKETS_PER_S][:n_sec]      # size at each second
    med = pd.Series(state_size).rolling(TYPICAL_WINDOW_S, min_periods=TYPICAL_WINDOW_S).median().to_numpy()
    med = np.concatenate([[np.nan], med[:-1]])                                        # seconds strictly before
    typical = med[np.minimum(i // BUCKETS_PER_S, n_sec - 1)]

    return ReplayDay(day=day, t_ms=book["t_ms"].to_numpy(np.int64), bid=bid_t, ask=ask_t,
                     bid_qty=book["bid_qty"].to_numpy(float), ask_qty=book["ask_qty"].to_numpy(float),
                     bid_qty_min=book["bid_qty_min"].to_numpy(float), ask_qty_min=book["ask_qty_min"].to_numpy(float),
                     mid=mid, tr_start=tr_start, tr_price=tr_price, tr_qty=tr_qty, tr_side=tr_side,
                     sell_min=sell_min, buy_max=buy_max, ofi_z=ofi_z, sigma_hat=sigma_hat, typical_size=typical)


@functools.lru_cache(maxsize=3)
def load_day(day: str) -> ReplayDay:
    """Load a processed day (scripts/build_stage4_dataset.py)."""
    bp = STAGE4 / f"BTCUSDT-book100ms-{day}.parquet"
    tp = STAGE4 / f"BTCUSDT-trades4-{day}.parquet"
    if not bp.exists() or not tp.exists():
        raise FileNotFoundError(f"stage 4 data for {day} missing: run scripts/build_stage4_dataset.py")
    book = pd.read_parquet(bp)
    trades = pd.read_parquet(tp, columns=["bucket", "price", "qty", "side"])
    return build_day(book, trades, day)
