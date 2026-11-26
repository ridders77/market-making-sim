"""Load the stage 3 per-second window tables (scripts/build_stage3_windows.py) and count touch hits.

A "hit" at distance delta in a window means the ask (or bid) quote at mid0 +- delta would have been
filled under the chosen rule: ``any`` (stage 2: any print reaches it) or a lot rule
(``l001`` = 0.01 BTC, ``l01`` = 0.1 BTC, ``l1`` = 1 BTC traded at or through it).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from mms.calibration.estimators import touch_curve

REPO = Path(__file__).resolve().parents[3]
DERIVED = REPO / "data" / "derived"
RULE_LOT = {"any": None, "l001": 0.01, "l01": 0.1, "l1": 1.0}


def load_windows(day: str) -> pd.DataFrame:
    path = DERIVED / f"BTCUSDT-win1s-{day}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing: run scripts/build_stage3_windows.py")
    return pd.read_parquet(path)


def excursions(w: pd.DataFrame, rule: str, unit: str = "usdt") -> np.ndarray:
    """Ask and bid excursions of the windows with a mid, stacked (sides pooled). unit 'bps' divides by mid0."""
    w = w[np.isfinite(w["mid0"])]
    e = np.concatenate([w[f"ask_exc_{rule}"].to_numpy(), w[f"bid_exc_{rule}"].to_numpy()])
    if unit == "bps":
        m = np.concatenate([w["mid0"].to_numpy(), w["mid0"].to_numpy()])
        e = 1e4 * e / m
    elif unit != "usdt":
        raise ValueError(unit)
    return e


def day_hits(days, rule: str, deltas: np.ndarray, unit: str = "usdt", cache: dict | None = None) -> dict:
    """day -> (hits over deltas, number of (window, side) pairs), sides pooled."""
    out = {}
    for d in days:
        w = cache[d] if cache is not None else load_windows(d)
        out[d] = touch_curve(excursions(w, rule, unit), deltas)
    return out
