"""Stage 5 Part B (WHAT-IF): how close to the front of the queue must a market maker be before quoting pays?

The only change from stage 4: when our order joins the touch, the queue ahead of it is f x the displayed size
(f = 1 back of the queue, as in stage 4; 0.5; 0.25; 0 = front). Everything else is stage 4's test-day setup:
always-at-touch and touch + inventory skew (stage 4's thresholds, chosen per fee on the calibration days at f = 1),
proportional cancel rule, 100 ms latency, 0.01 BTC lot, 230 test hours, fees -0.5 / 0 / +2 bp.
Not achievable for a retail participant: priority at the front of the queue is what professional makers buy
with speed and resting early, and our order is assumed not to change anyone else's behaviour.

Phases:  run     replay all (strategy, f, day) tasks (cache: data/derived/stage5_queue_runs/)
         report  results/stage5/queue_whatif.{csv,md}, break_even_queue.png
         mm      only if some f is clearly profitable: tabular Q-learning market maker at that f (see DECISIONS.md)
"""

from __future__ import annotations

import argparse
import json
import logging

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from stage5_common import DERIVED, OUT, run_pool, setup_logging
from stage4_common import (EP_STEPS, GAMMA, HOURS, OUT as OUT4, RUNS as RUNS4, TEST_DAYS, episode_markouts,
                           make_policy, pnl_at_fee, spec_key, task_key)
from mms.analysis.metrics import bootstrap_ci, break_even_bracket
from mms.replay.data import load_day
from mms.replay.env import ReplayConfig, run_episode

log = logging.getLogger("stage5")
RUNS = DERIVED / "stage5_queue_runs"
FRACS = [1.0, 0.5, 0.25, 0.15, 0.1, 0.05, 0.02, 0.0]
FEES4 = [-0.5, 0.0, 2.0]
RULE, LAT, LOT = "proportional", 100, 0.01
N_BOOT = 2000
LABEL = {"touch": "always at touch", "touch_skew": "touch + inventory skew"}


def specs_by_fee() -> dict:
    ch = json.loads((OUT4 / "chosen_thresholds.json").read_text(encoding="utf-8"))
    return {(name, fee): ch[f"{name}|{fee:g}"]["spec"] for name in ("touch", "touch_skew") for fee in FEES4}


def key(spec, f, day) -> str:
    return f"{spec_key(spec)}__f{f:g}__{day}"


def run_task(t) -> str:
    spec, f, day = t
    path = RUNS / f"{key(spec, f, day)}.parquet"
    if path.exists():
        return path.name
    d = load_day(day)
    cfg = ReplayConfig(latency_ms=LAT, cancel_rule=RULE, lot=LOT, fee_bps=0.0, gamma=GAMMA, queue_frac=f)
    rows = []
    for h in HOURS:
        start = h * EP_STEPS
        r = run_episode(make_policy(spec), d, start, cfg)
        mk = episode_markouts(r, d, start)
        rows.append({"day": day, "hour": h, "pnl": r.pnl, "fee_run": 0.0, "notional": r.notional,
                     "fill_qty": r.fill_qty, "n_fills": r.n_fills, "q": r.q, "max_abs_q": r.max_abs_q,
                     **mk["sums"]})
    RUNS.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".part")
    pd.DataFrame(rows).to_parquet(tmp, index=False)
    tmp.replace(path)
    return path.name


def unique_specs() -> list:
    out = []
    for s in specs_by_fee().values():
        if s not in out:
            out.append(s)
    return out


def run(workers: int) -> None:
    tasks = [(s, f, d) for s in unique_specs() for f in FRACS for d in TEST_DAYS
             if not (RUNS / f"{key(s, f, d)}.parquet").exists()]
    log.info("%d tasks", len(tasks))
    run_pool(run_task, tasks, workers)


def load(spec, f) -> pd.DataFrame:
    return pd.concat([pd.read_parquet(RUNS / f"{key(spec, f, d)}.parquet") for d in TEST_DAYS], ignore_index=True)


def report() -> None:
    sp = specs_by_fee()
    # sanity: f = 1 reproduces stage 4 exactly
    for (name, fee), s in sp.items():
        p4 = RUNS4 / f"{task_key(s, RULE, LAT, LOT, TEST_DAYS[0])}.parquet"
        if p4.exists():
            a = pd.read_parquet(p4)
            b = pd.read_parquet(RUNS / f"{key(s, 1.0, TEST_DAYS[0])}.parquet")
            diff = np.max(np.abs(pnl_at_fee(a, 0.0) - pnl_at_fee(b, 0.0)))
            if diff > 1e-9:
                raise AssertionError(f"f = 1 does not reproduce stage 4 for {name}: max diff {diff}")
            log.info("f = 1 reproduces stage 4 for %s (%s, fee %g)", name, TEST_DAYS[0], fee)
    rows, be = [], []
    rng_seed = 5
    for (name, fee), s in sp.items():
        dfs = {f: load(s, f) for f in FRACS}
        P = np.column_stack([pnl_at_fee(dfs[f], fee) for f in FRACS])          # hours x f (paired)
        means = P.mean(axis=0)
        for j, f in enumerate(FRACS):
            df = dfs[f]
            est = bootstrap_ci(P[:, j], np.mean, N_BOOT, seed=rng_seed)
            fq = df.fill_qty.sum()
            mkq = {h: (df[f"mk60_sum_{h}"].sum() / df[f"mk_qty_{h}"].sum()) if df[f"mk_qty_{h}"].sum() > 0 else np.nan
                   for h in ("queue", "through", "cross")}
            rows.append({"strategy": name, "fee_bps": fee, "f": f, "mean_pnl": est[0], "lo": est[1], "hi": est[2],
                         "btc_per_h": df.fill_qty.mean(), "share_queue": df.qty_queue.sum() / fq,
                         "share_sweep": df.qty_through.sum() / fq, "share_cross": df.qty_cross.sum() / fq,
                         "mk60_all": df.mk60_sum.sum() / df.mk_qty.sum(), "mk60_queue": mkq["queue"],
                         "mk60_sweep": mkq["through"], "bp_per_notional": 1e4 * P[:, j].sum() / df.notional.sum(),
                         "mean_max_abs_q": df.max_abs_q.mean() / LOT})
        # break-even as a bracket: last losing f and first profitable f (means), and whether each end is
        # significant (95% CI entirely below / above 0)
        lo_f, hi_f = break_even_bracket(FRACS, means)
        sig = {}
        for j, f in enumerate(FRACS):
            e = bootstrap_ci(P[:, j], np.mean, N_BOOT, seed=rng_seed)
            sig[f] = "loss" if e[2] < 0 else ("profit" if e[1] > 0 else "unclear")
        be.append({"strategy": name, "fee_bps": fee, "last_losing_f": lo_f, "first_profitable_f": hi_f,
                   "last_losing_sig": sig.get(lo_f, ""), "first_profitable_sig": sig.get(hi_f, ""),
                   "clearly_profitable_fs": ",".join(f"{f:g}" for f in FRACS if sig[f] == "profit")})
    t, b = pd.DataFrame(rows), pd.DataFrame(be)
    t.to_csv(OUT / "queue_whatif.csv", index=False)
    b.to_csv(OUT / "break_even_queue.csv", index=False)
    _markdown(t, b)
    _chart(t, b)


def _fmt(r) -> str:
    return f"{r.mean_pnl:.2f} [{r.lo:.2f}, {r.hi:.2f}]"


def _markdown(t: pd.DataFrame, b: pd.DataFrame) -> None:
    M = ["# Stage 5 Part B — WHAT-IF: queue priority for a market maker\n",
         "**What-if, not achievable for a retail participant.** The only change from stage 4: on joining the touch, the queue "
         "ahead of our order is f × the displayed size (f = 1: back of the queue, stage 4; f = 0: front). Test days, 230 paired "
         "hours, proportional cancel rule, 100 ms latency, 0.01 BTC lot, ±10-lot limit, stage 4's thresholds. "
         "Mean P&L per hour (USDT) [95% bootstrap CI over hours]. f = 1 reproduces stage 4 exactly (checked).\n",
         "| strategy | f | −0.5 bp (rebate) | 0 bp | +2 bp (retail) | BTC / h | queue / sweep share of volume | 60 s markout: queue / sweep fills (USDT/BTC) |",
         "|---|---|---|---|---|---|---|---|"]
    for name in ("touch", "touch_skew"):
        for f in FRACS:
            sub = t[(t.strategy == name) & (t.f == f)]
            cells = [_fmt(sub[sub.fee_bps == fee].iloc[0]) for fee in FEES4]
            r0 = sub[sub.fee_bps == 0.0].iloc[0]
            M.append(f"| {LABEL[name]} | {f:g} | " + " | ".join(cells) +
                     f" | {r0.btc_per_h:.1f} | {r0.share_queue:.0%} / {r0.share_sweep + r0.share_cross:.0%} | "
                     f"{r0.mk60_queue:.2f} / {r0.mk60_sweep:.2f} |")
    M += ["\nVolume shares and markouts at 0 bp; for touch + skew they are for its 0 bp configuration (the +2 bp configuration "
          "uses a different skew γ). Sweep = filled by a trade through our price or a book crossing.\n",
          "## Break-even queue position\n",
          "Reported as a bracket: the last f (scanning from the back of the queue, f = 1, to the front, f = 0) where the mean "
          "P&L is negative and the first where it is positive. Not interpolated: P&L is far from linear in f (most of the "
          "gain arrives in the last few % of the queue). (loss) / (profit) = the 95% CI at that f lies entirely below / "
          "above zero.\n",
          "| strategy | −0.5 bp | 0 bp | +2 bp |", "|---|---|---|---|"]
    for name in ("touch", "touch_skew"):
        cells = []
        for fee in FEES4:
            r = b[(b.strategy == name) & (b.fee_bps == fee)].iloc[0]
            if np.isfinite(r.first_profitable_f) and np.isfinite(r.last_losing_f):
                cells.append(f"between f = {r.last_losing_f:g} ({r.last_losing_sig}) and f = {r.first_profitable_f:g} "
                             f"({r.first_profitable_sig})")
            elif np.isfinite(r.first_profitable_f):
                cells.append("profitable at every f")
            else:
                cells.append("none: loses at every f, including the front")
        M.append(f"| {LABEL[name]} | " + " | ".join(cells) + " |")
    (OUT / "queue_whatif.md").write_text("\n".join(M) + "\n", encoding="utf-8")


def _chart(t: pd.DataFrame, b: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=False)
    colors = {"touch": "#7f7f7f", "touch_skew": "#1f77b4"}
    for ax, fee in zip(axes, FEES4):
        for name in ("touch", "touch_skew"):
            sub = t[(t.strategy == name) & (t.fee_bps == fee)].sort_values("f")
            ax.plot(sub.f, sub.mean_pnl, "o-", color=colors[name], label=LABEL[name])
            ax.fill_between(sub.f, sub.lo, sub.hi, color=colors[name], alpha=0.2)
        ax.axhline(0, color="k", lw=0.8)
        ax.set_xlabel("f = share of displayed queue ahead of us on joining\n(0 = front, 1 = back as in stage 4)")
        ax.set_title(f"maker fee {fee:+g} bp")
        ax.invert_xaxis()
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("mean P&L per hour (USDT), 95% CI")
    axes[0].legend(fontsize=8)
    fig.suptitle("WHAT-IF (not achievable for a retail participant): market making vs queue priority, BTCUSDT test days")
    fig.tight_layout()
    fig.savefig(OUT / "break_even_queue.png", dpi=130)
    plt.close(fig)


def main() -> None:
    setup_logging()
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["run", "report"])
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    if a.phase == "run":
        run(a.workers)
    else:
        report()


if __name__ == "__main__":
    main()
