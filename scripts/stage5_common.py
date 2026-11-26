"""Shared pieces of the stage 5 scripts: day splits, fee scenarios, day cache, memory-aware worker pool,
episode evaluation with markouts.

Splits (results/stage2/selected_days.csv): training = the first 8 calibration days by date, validation = the
last 2 calibration days (hyperparameters only), test = the 10 test days (evaluated once).
"""

from __future__ import annotations

import ctypes
import json
import logging
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path

import numpy as np
import pandas as pd

from mms.execution.env import BUY, ExecConfig, ExecutionEnv
from mms.execution.qlearning import StateSpec
from mms.replay import data as rdata

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "results" / "stage5"
DERIVED = REPO / "data" / "derived"
AGENTS = DERIVED / "stage5_agents"
EVALS = DERIVED / "stage5_eval"
SEL = pd.read_csv(REPO / "results" / "stage2" / "selected_days.csv")
CAL_DAYS = sorted(SEL.loc[SEL.split == "calibration", "day"])
TRAIN_DAYS, VAL_DAYS = CAL_DAYS[:8], CAL_DAYS[8:]
TEST_DAYS = sorted(SEL.loc[SEL.split == "test", "day"])
QTY = 0.5
DECISION_S = 10.0
H_MAIN, H_SENS = 60.0, (120.0, 300.0)
FEES = {"retail": (2.0, 5.0), "pro": (-0.5, 2.0)}      # (maker, taker) bps
RULES = ["pessimistic", "proportional", "optimistic"]
MAIN_RULE = "proportional"
LATENCY = 100
VARIANTS = [(), ("ofi",), ("queue",), ("sigma",), ("ofi", "queue", "sigma")]
SIDES = {"buy": 1, "sell": -1}
log = logging.getLogger("stage5")

_DAYS: dict = {}


def get_day(day: str):
    """Unbounded per-process cache (stage 4's load_day keeps only 3 days)."""
    d = _DAYS.get(day)
    if d is None:
        d = _DAYS[day] = rdata.load_day.__wrapped__(day)
    return d


def drop_days(keep=()) -> None:
    for k in list(_DAYS):
        if k not in keep:
            del _DAYS[k]


def make_cfg(H: float, side: int, fee: str = "retail", rule: str = MAIN_RULE, queue_frac: float = 1.0) -> ExecConfig:
    m, t = FEES[fee]
    return ExecConfig(side=side, qty=QTY, horizon_s=H, decision_s=DECISION_S, latency_ms=LATENCY, cancel_rule=rule,
                      maker_bps=m, taker_bps=t, queue_frac=queue_frac)


def load_cuts() -> dict:
    return json.loads((OUT / "state_cuts.json").read_text(encoding="utf-8"))


def make_spec(market: tuple, cuts: dict | None = None) -> StateSpec:
    c = load_cuts() if cuts is None else cuts
    return StateSpec(market=tuple(market), ofi_c=c["ofi_c"], queue_cuts=tuple(c["queue_cuts"]),
                     sigma_cuts=tuple(c["sigma_cuts"]))


def variant_name(market: tuple) -> str:
    return "private" if not market else "+".join(market)


# ---------------------------------------------------------------------------------------------- memory
class _MEMSTAT(ctypes.Structure):
    _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]


def free_gb() -> float:
    m = _MEMSTAT()
    m.dwLength = ctypes.sizeof(_MEMSTAT)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
    return m.ullAvailPhys / 2**30


def run_pool(fn, tasks: list, workers: int = 6, low_workers: int = 4, low_gb: float = 4.0) -> list:
    """Run fn over tasks in fresh worker processes (one task per process, so memory is returned after each).
    At most ``workers`` tasks run at once, ``low_workers`` once free memory has fallen below ``low_gb``
    (checked every few seconds; the drop is permanent for this pool and logged)."""
    results, cap, low = [None] * len(tasks), workers, False
    todo = list(enumerate(tasks))
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=workers, max_tasks_per_child=1) as ex:
        running = {}
        done_n = 0
        while todo or running:
            while todo and len(running) < cap:
                i, t = todo.pop(0)
                running[ex.submit(fn, t)] = i
            fin, _ = wait(list(running), timeout=5, return_when=FIRST_COMPLETED)
            for f in fin:
                results[running.pop(f)] = f.result()
                done_n += 1
                if done_n % 5 == 0 or done_n == len(tasks):
                    log.info("%d/%d tasks (%.0f s, %.1f GB free, %d workers)", done_n, len(tasks),
                             time.time() - t0, free_gb(), cap)
            g = free_gb()
            if not low and g < low_gb:
                low, cap = True, low_workers
                log.warning("free memory %.1f GB < %.0f GB: dropping to %d workers", g, low_gb, low_workers)
    return results


# ---------------------------------------------------------------------------------------------- evaluation
MK_H = (10, 60)


def episode_row(r, d, extra: dict) -> dict:
    """One episode's numbers. Markouts of passive fills (bps of the pre-fill mid, positive = adverse: the price
    kept moving our way after we were filled, e.g. it fell after a buy): -side (m_{t+h} - m_pre) / m_pre."""
    f = r.fills
    how = f["how"].astype(str) if len(f["how"]) else np.array([], dtype=str)
    passive = np.isin(how, ["queue", "through", "cross"]) if len(how) else np.zeros(0, bool)
    row = {"day": r.day, "start": r.start, "side": r.side, "m0": r.m0, "cost_bps": r.cost_bps,
           "price_usdt": r.price_usdt, "passive_qty": r.passive_qty, "passive_notional": r.passive_notional,
           "taker_qty": r.taker_qty, "taker_notional": r.taker_notional, "cleanup_qty": r.cleanup_qty,
           "beyond": r.beyond_display, "beyond_cleanup": bool(r.beyond_cleanup), "post_reject": r.n_post_reject,
           "steps_used": r.steps_used, "first_action": -1 if r.actions[0] is None else r.actions[0],
           "n_cross": sum(1 for a in r.actions if a == 0)}
    for h_ in ("queue", "through", "cross"):
        row[f"qty_{h_}"] = float(f["qty"][how == h_].sum()) if len(how) else 0.0
    if passive.any():
        j = f["bucket"][passive].astype(np.int64)
        q = f["qty"][passive].astype(float)
        m_pre = f["mid_pre"][passive].astype(float)
        for h in MK_H:
            jj = j - 1 + h * rdata.BUCKETS_PER_S
            if np.any(jj >= d.n_buckets):
                raise ValueError("markout horizon beyond the day: episode starts are limited to avoid this")
            mk = -r.side * (d.mid[jj] - m_pre) / m_pre * 1e4
            row[f"mk{h}_sum"] = float(np.sum(q * mk))
    else:
        for h in MK_H:
            row[f"mk{h}_sum"] = 0.0          # sum over no passive fills
    row.update(extra)
    return row


def evaluate_policy(policy_factory, cfg: ExecConfig, episodes, extra: dict) -> list[dict]:
    """Run a policy over (day, start) episodes, day by day (one day in memory at a time)."""
    rows = []
    by_day: dict = {}
    for d, st in episodes:
        by_day.setdefault(d, []).append(st)
    for d, starts in by_day.items():
        dd = get_day(d)
        env = ExecutionEnv(dd, cfg)
        for st in starts:
            pol = policy_factory()
            obs = env.reset(st)
            done = False
            while not done:
                if hasattr(pol, "act"):
                    a, x, passive = pol.act(obs)
                    obs, _, done, _ = env.step(a, take_qty=x, passive=passive)
                else:
                    obs, _, done, _ = env.step(pol(obs))
            rows.append(episode_row(env.result(), dd, extra))
        drop_days()
    return rows


def cost_at(df: pd.DataFrame, fee: str) -> np.ndarray:
    """Exact repricing of the shortfall at another fee scenario (fills do not depend on fees)."""
    m, t = FEES[fee]
    usdt = df["price_usdt"] + 1e-4 * (m * df["passive_notional"] + t * df["taker_notional"])
    return (1e4 * usdt / (QTY * df["m0"])).to_numpy()


def setup_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


__all__ = [n for n in dir() if not n.startswith("_")] + ["BUY"]
