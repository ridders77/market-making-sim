"""Stage 2 data pipeline: choose 20 days, then download and reduce them one at a time.

1. Day selection. Download the monthly 1-minute klines (USD-M futures BTCUSDT,
   2023-05 .. 2024-03; a few MB, checksum-verified), compute daily realised
   volatility and apply the rule in ``mms.calibration.day_selection``:
   global RV terciles, first half of the range = calibration pool, second half =
   test pool, 3 low / 4 medium / 3 high days from each. Written to
   results/stage2/selected_days.csv.
2. For each selected day (resumable: a day whose ``.meta.json`` exists is skipped):
   a. download the bookTicker and aggTrades zips, verify SHA-256;
   b. stream the CSVs straight out of the zips (never unzipped to disk);
   c. reduce: 1-second snapshots + trades tagged with the quote just before them;
   d. save data/processed/BTCUSDT-snap1s-<day>.parquet and BTCUSDT-trades-<day>.parquet,
      then the meta JSON (row counts, quality checks, sizes) last;
   e. delete that day's zips before moving on.

Run:  .venv\\Scripts\\python scripts\\build_dataset.py [--only-select] [--max-days N]
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import shutil
import time
import zipfile
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.csv as pacsv
import requests

from download_binance import BASE_URL, daily_url, fetch_verified, monthly_url  # noqa: F401  (scripts/ on sys.path)
from mms.calibration.day_selection import daily_realised_vol, select_days
from mms.calibration.reduce import attach_quotes, book_quality, snapshots_1s

REPO = Path(__file__).resolve().parents[1]
RAW = REPO / "data" / "raw"
PROCESSED = REPO / "data" / "processed"
RESULTS = REPO / "results" / "stage2"
SYMBOL, MARKET = "BTCUSDT", "futures/um"
RANGE = (dt.date(2023, 5, 16), dt.date(2024, 3, 30))  # bookTicker availability (DECISIONS.md)
COUNTS = {"low": 3, "medium": 4, "high": 3}

BOOK_COLS = ["update_id", "best_bid_price", "best_bid_qty", "best_ask_price", "best_ask_qty",
             "transaction_time", "event_time"]
TRADE_COLS = ["agg_trade_id", "price", "quantity", "first_trade_id", "last_trade_id",
              "transact_time", "is_buyer_maker"]

log = logging.getLogger("build_dataset")


def months(start: dt.date, end: dt.date) -> list[str]:
    out, y, m = [], start.year, start.month
    while (y, m) <= (end.year, end.month):
        out.append(f"{y:04d}-{m:02d}")
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)
    return out


def load_klines(session: requests.Session) -> pd.DataFrame:
    kdir = RAW / "klines"
    frames = []
    for mth in months(*RANGE):
        zp = kdir / f"{SYMBOL}-1m-{mth}.zip"
        if not zp.exists():
            if fetch_verified(session, monthly_url(MARKET, "klines", SYMBOL, mth, "1m"), zp) is None:
                raise RuntimeError(f"klines for {mth} not published")
        frames.append(read_zip_csv(zp, None, None).to_pandas())
    k = pd.concat(frames, ignore_index=True)
    day = pd.to_datetime(k["open_time"], unit="ms", utc=True).dt.date
    return k[(day >= RANGE[0]) & (day <= RANGE[1])]


def read_zip_csv(zip_path: Path, names: list[str] | None, include: list[str] | None) -> pa.Table:
    """Read the single CSV inside a Binance zip, with or without a header row."""
    with zipfile.ZipFile(zip_path) as zf:
        members = [m for m in zf.namelist() if m.lower().endswith(".csv")]
        if len(members) != 1 or Path(members[0]).name != members[0]:
            raise ValueError(f"{zip_path.name}: unexpected members {zf.namelist()}")
        with zf.open(members[0]) as f:
            first = f.readline().decode()
        has_header = not first.split(",")[0].strip().lstrip("-").isdigit()
        ropts = pacsv.ReadOptions(block_size=64 << 20,
                                  column_names=None if has_header or names is None else names)
        if not has_header and names is None:
            raise ValueError(f"{zip_path.name}: no header and no column names given")
        copts = pacsv.ConvertOptions(include_columns=include) if include else None
        with zf.open(members[0]) as f:
            return pacsv.read_csv(f, read_options=ropts, convert_options=copts)


def day_paths(day: dt.date) -> dict[str, Path]:
    d = day.isoformat()
    return {"snap": PROCESSED / f"{SYMBOL}-snap1s-{d}.parquet",
            "trades": PROCESSED / f"{SYMBOL}-trades-{d}.parquet",
            "meta": PROCESSED / f"{SYMBOL}-{d}.meta.json"}


def process_day(session: requests.Session, day: dt.date) -> dict:
    paths = day_paths(day)
    if paths["meta"].exists():
        log.info("%s already processed, skipping", day)
        return json.loads(paths["meta"].read_text())
    tmp = RAW / "pipeline_tmp" / day.isoformat()
    t0 = time.time()
    zips = {}
    for dtype in ("bookTicker", "aggTrades"):
        zp = fetch_verified(session, daily_url(MARKET, dtype, SYMBOL, day), tmp / f"{SYMBOL}-{dtype}-{day}.zip", 600)
        if zp is None:
            raise RuntimeError(f"{dtype} not published for {day}: choose another day and log it")
        zips[dtype] = zp
    t_dl = time.time() - t0

    book = read_zip_csv(zips["bookTicker"], BOOK_COLS, ["update_id", "best_bid_price", "best_bid_qty",
                                                         "best_ask_price", "best_ask_qty", "transaction_time"])
    book = book.to_pandas().rename(columns={"transaction_time": "time", "best_bid_price": "bid", "best_bid_qty": "bid_qty",
                                            "best_ask_price": "ask", "best_ask_qty": "ask_qty"})
    trades = read_zip_csv(zips["aggTrades"], TRADE_COLS, ["price", "quantity", "first_trade_id", "last_trade_id",
                                                           "transact_time", "is_buyer_maker"]).to_pandas()
    trades = trades.rename(columns={"transact_time": "time", "quantity": "qty"})
    trades["n_trades"] = (trades.pop("last_trade_id") - trades.pop("first_trade_id") + 1).astype("int32")
    trades["is_buyer_maker"] = trades["is_buyer_maker"].astype(str).str.lower().eq("true") \
        if trades["is_buyer_maker"].dtype != bool else trades["is_buyer_maker"]

    start_ms = int(dt.datetime.combine(day, dt.time(), dt.UTC).timestamp() * 1000)
    end_ms = start_ms + 86_400_000
    quality = book_quality(book, start_ms, end_ms)
    n_outside = int(((book.time < start_ms) | (book.time >= end_ms)).sum())
    n_trades_outside = int(((trades.time < start_ms) | (trades.time >= end_ms)).sum())

    snap = snapshots_1s(book, start_ms, end_ms)
    tagged = attach_quotes(trades, book)
    del book

    PROCESSED.mkdir(parents=True, exist_ok=True)
    for key, df in (("snap", snap), ("trades", tagged)):
        part = paths[key].with_suffix(".parquet.part")
        df.to_parquet(part, index=False, compression="zstd")
        part.replace(paths[key])

    meta = {
        "day": day.isoformat(),
        "book_updates": quality["n_updates"],
        "book_crossed_or_locked": quality["n_crossed"],
        "book_max_gap_ms": quality["max_gap_ms"],
        "book_rows_outside_day": n_outside,
        "trades": int(len(tagged)),
        "trades_outside_day": n_trades_outside,
        "trades_without_quote": int(tagged["bid"].isna().sum()),
        "snapshots_missing": int(snap["mid"].isna().sum()),
        "median_spread": float(snap["spread"].median()),
        "share_seconds_1tick_spread": float((snap["spread"].round(1) == 0.1).mean()),
        "zip_mb": {k: round(v.stat().st_size / 2**20, 1) for k, v in zips.items()},
        "processed_mb": {k: round(paths[k].stat().st_size / 2**20, 2) for k in ("snap", "trades")},
        "download_s": round(t_dl, 1),
        "total_s": round(time.time() - t0, 1),
    }
    paths["meta"].write_text(json.dumps(meta, indent=2))
    shutil.rmtree(tmp)  # delete the day's raw files before the next day
    log.info("%s done: %s", day, meta)
    return meta


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--only-select", action="store_true")
    ap.add_argument("--max-days", type=int, default=None)
    args = ap.parse_args()
    RESULTS.mkdir(parents=True, exist_ok=True)

    with requests.Session() as session:
        session.headers["User-Agent"] = "market-making-sim research downloader"
        rv = daily_realised_vol(load_klines(session))
        if len(rv) != (RANGE[1] - RANGE[0]).days + 1:
            raise RuntimeError(f"klines cover {len(rv)} days, expected {(RANGE[1] - RANGE[0]).days + 1}")
        rv.to_csv(RESULTS / "daily_realised_vol.csv")
        sel = select_days(rv, COUNTS)
        sel.to_csv(RESULTS / "selected_days.csv", index=False)
        log.info("selected days:\n%s", sel.to_string())
        if args.only_select:
            return
        metas = []
        for day in sel["day"].iloc[: args.max_days]:
            metas.append(process_day(session, day))
    summary = pd.DataFrame([{**{k: v for k, v in m.items() if not isinstance(v, dict)},
                             "snap_mb": m["processed_mb"]["snap"], "trades_mb": m["processed_mb"]["trades"]}
                            for m in metas])
    summary = summary.merge(sel.assign(day=sel.day.astype(str)), on="day")
    summary.to_csv(RESULTS / "processed_days.csv", index=False)
    log.info("processed %d days, total %.1f MB", len(summary), (summary.snap_mb + summary.trades_mb).sum())


if __name__ == "__main__":
    main()
