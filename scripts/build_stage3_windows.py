"""Stage 3: per-second window table for every processed day (cached under data/derived/, gitignored).

For each 1-second window of each of the 20 days (calibration and test), one row with:
  time_s, mid0                          window start and the mid in force when it opens
  sigma_hat                             trailing EWMA sigma (60 s changes, span 30 min; NaN in warm-up)
  ask_exc_any, bid_exc_any              stage 2 rule: distance of the deepest print (any size)
  ask_exc_l001, bid_exc_l001            lot rule, 0.01 BTC: deepest distance with >= 1 lot traded through
  ask_exc_l01,  bid_exc_l01             lot rule, 0.1 BTC
  ask_exc_l1,   bid_exc_l1              lot rule, 1 BTC (the stage 2 lot, for reference)
  buy_vol, sell_vol                     aggressor buy / sell volume (BTC) printed during the window
Everything later (lambda fits, model choice, OFI) is computed from these tables.

Run:  .venv\\Scripts\\python scripts\\build_stage3_windows.py
"""

from __future__ import annotations

import logging
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from mms.calibration.estimators import lot_excursions, touch_excursions, trailing_sigma

REPO = Path(__file__).resolve().parents[1]
PROC = REPO / "data" / "processed"
DERIVED = REPO / "data" / "derived"
LOTS = {"l001": 0.01, "l01": 0.1, "l1": 1.0}
log = logging.getLogger("stage3_windows")


def build_day(day: str) -> str:
    out = DERIVED / f"BTCUSDT-win1s-{day}.parquet"
    if out.exists():
        return f"{day}: cached"
    snap = pd.read_parquet(PROC / f"BTCUSDT-snap1s-{day}.parquet", columns=["time_s", "mid"])
    trades = pd.read_parquet(PROC / f"BTCUSDT-trades-{day}.parquet", columns=["time", "price", "qty", "is_buyer_maker"])
    sig = trailing_sigma(snap, span_min=30)
    base = touch_excursions(snap, trades, 1).rename(columns={"ask_exc": "ask_exc_any", "bid_exc": "bid_exc_any"})
    for name, lot in LOTS.items():
        w = lot_excursions(snap, trades, 1, lot)
        assert (w.time_s.to_numpy() == base.time_s.to_numpy()).all()
        base[f"ask_exc_{name}"], base[f"bid_exc_{name}"] = w.ask_exc.to_numpy(), w.bid_exc.to_numpy()
    # per-second aggressor volumes (bins aligned with the windows)
    t0 = int(snap.time_s.iloc[0])
    sec = trades.time.to_numpy(np.int64) // 1000 - t0
    buy = ~trades.is_buyer_maker.to_numpy(bool)
    n = len(snap)
    ok = (sec >= 0) & (sec < n)
    bv = np.bincount(sec[ok & buy], weights=trades.qty.to_numpy()[ok & buy], minlength=n)
    sv = np.bincount(sec[ok & ~buy], weights=trades.qty.to_numpy()[ok & ~buy], minlength=n)
    full = pd.DataFrame({"time_s": snap.time_s, "mid_snap": snap.mid, "sigma_hat": sig, "buy_vol": bv, "sell_vol": sv})
    full = full.merge(base, on="time_s", how="left")   # windows with NaN mid0 keep NaN excursions
    tmp = out.with_suffix(".tmp")
    full.to_parquet(tmp, index=False)
    tmp.replace(out)
    return f"{day}: {len(full)} windows, {int(full.mid0.isna().sum())} without a mid"


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    DERIVED.mkdir(parents=True, exist_ok=True)
    days = pd.read_csv(REPO / "results" / "stage2" / "selected_days.csv")["day"].astype(str).tolist()
    with ProcessPoolExecutor(max_workers=8) as ex:
        for msg in ex.map(build_day, days):
            log.info(msg)


if __name__ == "__main__":
    main()
