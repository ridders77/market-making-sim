"""Shared pieces of the stage 4 scripts: episodes, strategy construction, one (config, day) task, caching.

A task runs one strategy configuration over the 23 hour-long episodes of one day (01:00-02:00 ... 23:00-24:00
UTC; the first hour is the warm-up for the 30-minute sigma_hat and typical size). Per-episode summaries are
cached in data/derived/stage4_runs/, and the individual fills (with markouts) of the main-latency runs in
data/derived/stage4_fills/.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from mms.replay.data import BUCKETS_PER_S, load_day
from mms.replay.env import ReplayConfig, run_episode
from mms.strategies.touch import AlwaysTouch, ReplayMarkoutAwareAS, TouchOFIAware, TouchQueueAware, TouchSkew

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "results" / "stage4"
RUNS = REPO / "data" / "derived" / "stage4_runs"
FILLS = REPO / "data" / "derived" / "stage4_fills"
STAGE3 = REPO / "results" / "stage3"
SEL = pd.read_csv(REPO / "results" / "stage2" / "selected_days.csv")
CAL_DAYS = SEL.loc[SEL.split == "calibration", "day"].tolist()
TEST_DAYS = SEL.loc[SEL.split == "test", "day"].tolist()
HOURS = list(range(1, 24))
EP_STEPS = 3600 * BUCKETS_PER_S
GAMMA = 1e-4
HORIZONS = (1, 5, 30, 60)
MARKOUT_VALID_S = 3600 - 60          # markouts use fills in the first 59 minutes (so +60 s stays in the episode)
FEES = [-0.5, 0.0, 2.0]
RULES = ["pessimistic", "proportional", "optimistic"]
log = logging.getLogger("stage4")

_CAL = json.loads((STAGE3 / "calibration_stage3.json").read_text(encoding="utf-8"))
_FLOW = json.loads((STAGE3 / "flow_calibration.json").read_text(encoding="utf-8"))
MODEL_B = _CAL["model_b"]
_MED = _FLOW["regimes"]["medium"]
S0_MED = _CAL["regimes"]["medium"]["s0"]
JUMP_DELTA = _MED["jump_delta"]                                   # USDT (stage 3 bucket means)
JUMP_MEAN_BPS = [1e4 * j / S0_MED for j in _MED["jump_mean"]]     # bps of the mid


def make_policy(spec: dict):
    n = spec["name"]
    if n == "touch":
        return AlwaysTouch()
    if n == "touch_skew":
        return TouchSkew(max_lots=spec["max_lots"], gamma=spec["gamma"])
    if n == "touch_queue":
        return TouchQueueAware(theta=spec["theta"])
    if n == "touch_ofi":
        return TouchOFIAware(z_star=spec["z_star"])
    if n == "markout_as":
        return ReplayMarkoutAwareAS(GAMMA, MODEL_B, JUMP_DELTA, JUMP_MEAN_BPS, fee_bps=spec["fee"])
    raise ValueError(n)


def fee_dependent(spec: dict) -> bool:
    return spec["name"] == "markout_as"


def spec_key(spec: dict) -> str:
    """e.g. {"name": "touch_ofi", "z_star": 1.0} -> "touch_ofi_z_star1"."""
    parts = [spec["name"]] + [f"{k}{spec[k]:g}" for k in sorted(spec) if k != "name"]
    return "_".join(parts)


def task_key(spec: dict, rule: str, latency: int, lot: float, day: str, unseen: str = "back") -> str:
    k = f"{spec_key(spec)}__{rule}_lat{latency}_lot{lot:g}"
    if unseen != "back":
        k += f"_unseen{unseen}"
    return f"{k}__{day}"


def episode_markouts(r, d, start: int) -> dict:
    """qty-weighted sums of the h-second markout -s (m_{t+h} - m_0) over fills in the first 59 minutes, plus the
    spread captured s (m_0 - p) and per-fill arrays."""
    f = r.fills
    out = {}
    if r.n_fills == 0:
        empty = np.array([])
        return {"arrays": {"step": empty, "side": empty, "price": empty, "qty": empty, "mid0": empty, "how": empty,
                           **{f"mk{h}": empty for h in HORIZONS}},
                "sums": {**{f"mk{h}_sum": 0.0 for h in HORIZONS}, "mk_qty": 0.0, "captured_sum": 0.0,
                         **{f"{k}_{how}": 0.0 for how in ("queue", "through", "cross")
                            for k in ("qty", "mk60_sum", "mk_qty")}}}          # sums over no fills are 0
    step, side, qty, mid0 = f["step"], f["side"].astype(float), f["qty"], f["mid0"]
    price = f["price"] * 0.1
    valid = (step - start) < MARKOUT_VALID_S * BUCKETS_PER_S
    arrays = {"step": step, "side": side, "price": price, "qty": qty, "mid0": mid0, "how": f["how"]}
    for h in HORIZONS:
        j = step - 1 + h * BUCKETS_PER_S                 # mid in force at t_i + h
        m = np.full(len(step), np.nan)
        ok = j < d.n_buckets
        m[ok] = -side[ok] * (d.mid[j[ok]] - mid0[ok])
        arrays[f"mk{h}"] = m
        out[f"mk{h}_sum"] = float(np.sum((qty * m)[valid]))
    out["mk_qty"] = float(qty[valid].sum())
    cap = side * (mid0 - price)
    out["captured_sum"] = float(np.sum(qty * cap))
    for how in ("queue", "through", "cross"):
        k = f["how"] == how
        out[f"qty_{how}"] = float(qty[k].sum())
        out[f"mk60_sum_{how}"] = float(np.sum((qty * arrays["mk60"])[k & valid]))
        out[f"mk_qty_{how}"] = float(qty[k & valid].sum())
    return {"arrays": arrays, "sums": out}


def run_task(task: tuple) -> str:
    """task = (spec, rule, latency_ms, lot, day, unseen_depth, save_fills)."""
    spec, rule, latency, lot, day, unseen, save_fills = task
    key = task_key(spec, rule, latency, lot, day, unseen)
    path = RUNS / f"{key}.parquet"
    if path.exists():
        return key
    d = load_day(day)
    fee = spec.get("fee", 0.0)
    cfg = ReplayConfig(latency_ms=latency, cancel_rule=rule, lot=lot, fee_bps=fee, unseen_depth=unseen, gamma=GAMMA)
    rows, fills = [], []
    for h in HOURS:
        start = h * EP_STEPS
        pol = make_policy(spec)
        r = run_episode(pol, d, start, cfg)
        mk = episode_markouts(r, d, start)
        rows.append({"day": day, "hour": h, "start": start, "pnl": r.pnl, "fee_run": fee, "notional": r.notional,
                     "fill_qty": r.fill_qty, "n_fills": r.n_fills, "q": r.q, "max_abs_q": r.max_abs_q,
                     "penalty": r.penalty, "rej_post": r.n_rejected_post_only, "rej_limit": r.n_rejected_limit,
                     "quoted_bid": r.steps_quoted[0], "quoted_ask": r.steps_quoted[1],
                     "touch_bid": r.steps_at_touch[0], "touch_ask": r.steps_at_touch[1],
                     "sigma_hat0": float(d.sigma_hat[start]), "mid0": float(d.mid[start - 1]), **mk["sums"]})
        if save_fills and r.n_fills:
            a = pd.DataFrame(mk["arrays"])
            a["hour"] = h
            fills.append(a)
    RUNS.mkdir(parents=True, exist_ok=True)
    if save_fills:
        FILLS.mkdir(parents=True, exist_ok=True)
        fp = FILLS / f"{key}.parquet"
        (pd.concat(fills, ignore_index=True) if fills else pd.DataFrame()).to_parquet(fp, index=False)
    tmp = path.with_suffix(".part")
    pd.DataFrame(rows).to_parquet(tmp, index=False)
    tmp.replace(path)
    return key


SUM_COLS = [f"mk{h}_sum" for h in HORIZONS] + ["mk_qty", "captured_sum"] +     [f"{k}_{how}" for how in ("queue", "through", "cross") for k in ("qty", "mk60_sum", "mk_qty")]


def load_runs(spec: dict, rule: str, latency: int, lot: float, days, unseen: str = "back") -> pd.DataFrame:
    df = pd.concat([pd.read_parquet(RUNS / f"{task_key(spec, rule, latency, lot, d, unseen)}.parquet") for d in days],
                   ignore_index=True)
    # Episodes cached before the fix in episode_markouts stored no sums when they had no fills; the sum over no
    # fills is exactly 0 (only for n_fills == 0 rows; any other NaN is left alone and would surface).
    none = df["n_fills"] == 0
    for c in SUM_COLS:
        if c not in df:
            df[c] = np.nan
        df.loc[none, c] = df.loc[none, c].fillna(0.0)
    return df


def load_fills(spec: dict, rule: str, latency: int, lot: float, days) -> pd.DataFrame:
    parts = []
    for d in days:
        f = pd.read_parquet(FILLS / f"{task_key(spec, rule, latency, lot, d)}.parquet")
        if len(f):
            f["day"] = d
            parts.append(f)
    return pd.concat(parts, ignore_index=True)


def pnl_at_fee(df: pd.DataFrame, fee: float) -> np.ndarray:
    """Fees change only cash: P&L(f) = P&L(f_run) - (f - f_run) * notional (exact)."""
    return (df["pnl"] - (fee - df["fee_run"]) * 1e-4 * df["notional"]).to_numpy()
