"""Summarise downloaded Binance CSVs: columns, row count, timestamp unit and range, file size.

Streams each file with pyarrow (bookTicker days are several GB), so memory stays small.
Timestamp unit is inferred from magnitude: ~1e12 = milliseconds, ~1e15 = microseconds.

Run:  .venv\\Scripts\\python scripts\\inspect_binance.py [csv ...]
(no arguments = every CSV under data/raw)
"""

from __future__ import annotations

import argparse
import datetime as dt
import logging
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv as pacsv

REPO_ROOT = Path(__file__).resolve().parents[1]
log = logging.getLogger("inspect_binance")

# Header-less files (Binance spot dumps) get these documented column names.
SPOT_TRADES_COLS = ["id", "price", "qty", "quote_qty", "time", "is_buyer_maker", "is_best_match"]
TIME_COLS = ("transaction_time", "time", "event_time", "transact_time")


def _has_header(path: Path) -> bool:
    with open(path, encoding="utf-8") as f:
        first = f.readline()
    return not first.split(",")[0].strip().lstrip("-").isdigit()


def ts_unit(v: int) -> str:
    return "microseconds" if v > 1e14 else "milliseconds" if v > 1e11 else "seconds?"


def inspect(path: Path) -> dict:
    header = _has_header(path)
    read_opts = pacsv.ReadOptions(block_size=64 << 20,
                                  column_names=None if header else SPOT_TRADES_COLS)
    reader = pacsv.open_csv(path, read_options=read_opts)
    cols = reader.schema.names
    tcol = next(c for c in TIME_COLS if c in cols)
    n, tmin, tmax = 0, None, None
    for batch in reader:
        n += batch.num_rows
        col = batch.column(cols.index(tcol))
        lo, hi = pc.min(col).as_py(), pc.max(col).as_py()
        tmin = lo if tmin is None else min(tmin, lo)
        tmax = hi if tmax is None else max(tmax, hi)
    unit = ts_unit(tmax)
    div = 1e6 if unit == "microseconds" else 1e3
    to_dt = lambda v: dt.datetime.fromtimestamp(v / div, dt.UTC).isoformat(timespec="seconds")
    return {"file": path.name, "header": header, "columns": cols, "rows": n, "time_col": tcol,
            "ts_unit": unit, "first": to_dt(tmin), "last": to_dt(tmax),
            "size_mb": round(path.stat().st_size / 2**20, 1)}


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("files", nargs="*", type=Path)
    files = p.parse_args().files or sorted((REPO_ROOT / "data" / "raw").rglob("*.csv"))
    for f in files:
        info = inspect(f)
        log.info("%s", info)


if __name__ == "__main__":
    main()
