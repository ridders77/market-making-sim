"""Stage 4, Step 0: re-process the 20 selected days with queue information (100 ms book + trades).

For each day in results/stage2/selected_days.csv (resumable: a day whose stage 4 meta JSON exists is skipped):

1. download the bookTicker and aggTrades zips into data/raw/stage4_tmp/<day>/ and verify their SHA-256;
2. stream the CSVs out of the zips (never unzipped to disk);
3. reduce:
   * data/processed/stage4/BTCUSDT-book100ms-<day>.parquet: best bid/ask price and size at the end of
     every 100 ms bucket, plus the min/max size at that price during the bucket
     (``mms.calibration.reduce.snapshots_100ms``);
   * data/processed/stage4/BTCUSDT-trades4-<day>.parquet: every aggTrade (ms time, price, qty, aggressor side,
     100 ms bucket) with the best bid/ask and sizes in force strictly before it (same rule as stage 2);
4. write the meta JSON last (row counts, quality checks, sizes), then delete the day's raw zips.

Run:  .venv\\Scripts\\python scripts\\build_stage4_dataset.py [--max-days N]
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from build_dataset import BOOK_COLS, MARKET, SYMBOL, TRADE_COLS, read_zip_csv  # scripts/ on sys.path
from download_binance import daily_url, fetch_verified
from mms.calibration.reduce import attach_quotes, book_quality, bucket_trades, snapshots_100ms

REPO = Path(__file__).resolve().parents[1]
RAW = REPO / "data" / "raw" / "stage4_tmp"
OUT = REPO / "data" / "processed" / "stage4"
RESULTS = REPO / "results" / "stage4"
log = logging.getLogger("build_stage4")


def day_paths(day: str) -> dict[str, Path]:
    return {"book": OUT / f"{SYMBOL}-book100ms-{day}.parquet",
            "trades": OUT / f"{SYMBOL}-trades4-{day}.parquet",
            "meta": OUT / f"{SYMBOL}-{day}.stage4.meta.json"}


def process_day(session: requests.Session, day: dt.date) -> dict:
    d = day.isoformat()
    paths = day_paths(d)
    if paths["meta"].exists():
        log.info("%s already processed, skipping", d)
        return json.loads(paths["meta"].read_text())
    tmp = RAW / d
    t0 = time.time()
    zips = {}
    for dtype in ("bookTicker", "aggTrades"):
        zp = fetch_verified(session, daily_url(MARKET, dtype, SYMBOL, day), tmp / f"{SYMBOL}-{dtype}-{d}.zip", 900)
        if zp is None:
            raise RuntimeError(f"{dtype} not published for {d}")
        zips[dtype] = zp
    t_dl = time.time() - t0

    book = read_zip_csv(zips["bookTicker"], BOOK_COLS, ["update_id", "best_bid_price", "best_bid_qty",
                                                         "best_ask_price", "best_ask_qty", "transaction_time"])
    book = book.to_pandas().rename(columns={"transaction_time": "time", "best_bid_price": "bid", "best_bid_qty": "bid_qty",
                                            "best_ask_price": "ask", "best_ask_qty": "ask_qty"})
    trades = read_zip_csv(zips["aggTrades"], TRADE_COLS, ["price", "quantity", "transact_time",
                                                           "is_buyer_maker"]).to_pandas()
    trades = trades.rename(columns={"transact_time": "time", "quantity": "qty"})
    if trades["is_buyer_maker"].dtype != bool:
        trades["is_buyer_maker"] = trades["is_buyer_maker"].astype(str).str.lower().eq("true")

    start_ms = int(dt.datetime.combine(day, dt.time(), dt.UTC).timestamp() * 1000)
    end_ms = start_ms + 86_400_000
    quality = book_quality(book, start_ms, end_ms)
    snap = snapshots_100ms(book, start_ms, end_ms)
    tagged = bucket_trades(attach_quotes(trades, book), start_ms)
    n_outside = int(((tagged.time < start_ms) | (tagged.time >= end_ms)).sum())
    tagged = tagged[(tagged.time >= start_ms) & (tagged.time < end_ms)].reset_index(drop=True)
    del book

    OUT.mkdir(parents=True, exist_ok=True)
    for key, df in (("book", snap), ("trades", tagged)):
        part = paths[key].with_suffix(".parquet.part")
        df.to_parquet(part, index=False, compression="zstd")
        part.replace(paths[key])

    spread = (snap["ask"] - snap["bid"]).round(1)
    meta = {
        "day": d,
        "book_updates": quality["n_updates"],
        "book_crossed_or_locked": quality["n_crossed"],
        "book_max_gap_ms": quality["max_gap_ms"],
        "buckets": int(len(snap)),
        "buckets_missing": int(snap["bid"].isna().sum()),
        "buckets_with_updates": int((snap["n_updates"] > 0).sum()),
        "share_buckets_1tick_spread": float((spread == 0.1).mean()),
        "median_bid_qty": float(snap["bid_qty"].median()),
        "median_ask_qty": float(snap["ask_qty"].median()),
        "trades": int(len(tagged)),
        "trades_outside_day_dropped": n_outside,
        "trades_without_quote": int(tagged["bid"].isna().sum()),
        "zip_mb": {k: round(v.stat().st_size / 2**20, 1) for k, v in zips.items()},
        "processed_mb": {k: round(paths[k].stat().st_size / 2**20, 2) for k in ("book", "trades")},
        "download_s": round(t_dl, 1),
        "total_s": round(time.time() - t0, 1),
    }
    paths["meta"].write_text(json.dumps(meta, indent=2))
    shutil.rmtree(tmp)   # delete the day's raw files before the next day
    log.info("%s done: %s", d, meta)
    return meta


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-days", type=int, default=None)
    args = ap.parse_args()
    sel = pd.read_csv(REPO / "results" / "stage2" / "selected_days.csv")
    days = [dt.date.fromisoformat(s) for s in sel["day"]][: args.max_days]
    with requests.Session() as session:
        session.headers["User-Agent"] = "market-making-sim research downloader"
        metas = [process_day(session, d) for d in days]
    rows = [{**{k: v for k, v in m.items() if not isinstance(v, dict)},
             "book_mb": m["processed_mb"]["book"], "trades_mb": m["processed_mb"]["trades"],
             "zip_book_mb": m["zip_mb"]["bookTicker"], "zip_trades_mb": m["zip_mb"]["aggTrades"]} for m in metas]
    summary = pd.DataFrame(rows).merge(sel[["day", "split", "regime"]] if "split" in sel else sel, on="day", how="left")
    RESULTS.mkdir(parents=True, exist_ok=True)
    summary.to_csv(RESULTS / "processed_days.csv", index=False)
    log.info("processed %d days: %.1f MB book + %.1f MB trades on disk", len(summary),
             summary.book_mb.sum(), summary.trades_mb.sum())


if __name__ == "__main__":
    main()
