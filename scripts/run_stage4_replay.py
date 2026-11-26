"""Stage 4, Steps 3 and 5: touch strategies in the replay backtester.

Phases (each resumable through the per-(config, day) cache, see stage4_common.py):

  tune    Calibration days only. Grid over each strategy's threshold, proportional cancel rule, 100 ms latency,
          0.01 BTC lot. The threshold with the highest mean P&L per hour is chosen separately for each fee
          scenario (-0.5, 0, +2 bp; fills do not depend on the fee, so P&L is repriced exactly). Writes
          results/stage4/tuning.csv and chosen_thresholds.json. Run before any test-day result exists.
  test    Test days: the chosen configurations under the 3 cancel rules x 3 fees at 100 ms; latency 0 and 300 ms;
          lot 0.1 BTC (proportional rule).
  report  Tables and charts in results/stage4/.

Run:  .venv\\Scripts\\python scripts\\run_stage4_replay.py tune|test|report [--workers N]
"""

from __future__ import annotations

import argparse
import json
import logging
from concurrent.futures import ProcessPoolExecutor

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from stage4_common import (CAL_DAYS, FEES, HORIZONS, OUT, RULES, TEST_DAYS, fee_dependent, load_fills, load_runs,
                           log, pnl_at_fee, run_task, spec_key)
from mms.analysis.metrics import bootstrap_ci, paired_difference

GRIDS = {
    "touch": [{"name": "touch"}],
    "touch_skew": [{"name": "touch_skew", "max_lots": m, "gamma": g} for m in (1.0, 2.0, 5.0, 10.0)
                   for g in (0.0, 1e-5, 1e-4)],
    "touch_queue": [{"name": "touch_queue", "theta": t} for t in (0.05, 0.1, 0.25, 0.5, 1.0, 2.0)],
    "touch_ofi": [{"name": "touch_ofi", "z_star": z} for z in (0.25, 0.5, 0.75, 1.0, 1.25, 1.5)],
}
TUNE_RULE, MAIN_LAT, LOT = "proportional", 100, 0.01
LATENCIES = [0, 100, 300]
STRATS = ["touch", "touch_skew", "touch_queue", "touch_ofi", "markout_as"]
LABEL = {"touch": "always at touch", "touch_skew": "touch + inventory skew", "touch_queue": "touch + queue-aware",
         "touch_ofi": "touch + OFI-aware", "markout_as": "markout-aware A–S (quote wide)"}
COLOR = {"touch": "#7f7f7f", "touch_skew": "#1f77b4", "touch_queue": "#2ca02c", "touch_ofi": "#d62728",
         "markout_as": "#ff7f0e"}
N_BOOT = 2000


def run_all(tasks: list, workers: int) -> None:
    tasks = list(dict.fromkeys(tasks))        # de-duplicate, keep order
    log.info("%d tasks", len(tasks))
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for i, k in enumerate(ex.map(run_task, tasks, chunksize=1), 1):
            if i % 25 == 0 or i == len(tasks):
                log.info("%d/%d (%s)", i, len(tasks), k)


# ---------------------------------------------------------------------------------------------- tune
def tune(workers: int) -> None:
    specs = [s for g in GRIDS.values() for s in g]
    run_all([(s, TUNE_RULE, MAIN_LAT, LOT, d, "back", False) for s in specs for d in CAL_DAYS], workers)
    rows = []
    for s in specs:
        df = load_runs(s, TUNE_RULE, MAIN_LAT, LOT, CAL_DAYS)
        for fee in FEES:
            p = pnl_at_fee(df, fee)
            est = bootstrap_ci(p, np.mean, N_BOOT, seed=1)
            rows.append({"strategy": s["name"], "config": spec_key(s), "spec": json.dumps(s), "fee_bps": fee,
                         "mean_pnl": est[0], "lo": est[1], "hi": est[2], "fill_btc_per_h": df.fill_qty.mean(),
                         "n_episodes": len(df)})
    t = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    t.to_csv(OUT / "tuning.csv", index=False)
    chosen = {}
    for name in GRIDS:
        for fee in FEES:
            sub = t[(t.strategy == name) & (t.fee_bps == fee)]
            best = sub.loc[sub.mean_pnl.idxmax()]
            chosen[f"{name}|{fee:g}"] = {"spec": json.loads(best.spec), "cal_mean_pnl": best.mean_pnl,
                                          "cal_ci": [best.lo, best.hi], "fill_btc_per_h": best.fill_btc_per_h}
    (OUT / "chosen_thresholds.json").write_text(json.dumps(chosen, indent=2), encoding="utf-8")
    M = ["# Stage 4: threshold tuning on the calibration days\n",
         f"Proportional cancel rule, {MAIN_LAT} ms latency, {LOT} BTC lot, 10 calibration days × 23 hours = 230 episodes. "
         "Mean P&L per hour (USDT) [95% bootstrap CI over episodes]. The chosen threshold (bold) maximises the mean, per fee.\n",
         "| strategy | config | BTC filled / h | P&L at −0.5 bp | P&L at 0 bp | P&L at +2 bp |", "|---|---|---|---|---|---|"]
    for cfg_, sub in t.groupby("config", sort=False):
        cells = []
        for fee in FEES:
            r = sub[sub.fee_bps == fee].iloc[0]
            is_best = chosen[f"{r.strategy}|{fee:g}"]["spec"] == json.loads(r.spec)
            c = f"{r.mean_pnl:.2f} [{r.lo:.2f}, {r.hi:.2f}]"
            cells.append(f"**{c}**" if is_best else c)
        M.append(f"| {LABEL[sub.strategy.iloc[0]]} | {cfg_} | {sub.fill_btc_per_h.iloc[0]:.2f} | " + " | ".join(cells) + " |")
    (OUT / "tuning.md").write_text("\n".join(M) + "\n", encoding="utf-8")
    log.info("chosen: %s", {k: spec_key(v["spec"]) for k, v in chosen.items()})


def chosen_specs() -> dict:
    """(strategy, fee) -> spec. markout_as is fee-dependent and has no threshold."""
    ch = json.loads((OUT / "chosen_thresholds.json").read_text(encoding="utf-8"))
    out = {(k.split("|")[0], float(k.split("|")[1])): v["spec"] for k, v in ch.items()}
    for fee in FEES:
        out[("markout_as", fee)] = {"name": "markout_as", "fee": fee}
    return out


# ---------------------------------------------------------------------------------------------- test
def test_tasks() -> list:
    ch = chosen_specs()
    tasks = []
    for (name, fee), spec in ch.items():
        for rule in RULES:
            tasks += [(spec, rule, MAIN_LAT, LOT, d, "back", True) for d in TEST_DAYS]
    # latency sensitivity (fee 0 configuration for the threshold strategies; all three rules)
    for name in STRATS:
        spec = ch[(name, 0.0)]
        for lat in (0, 300):
            for rule in RULES:
                tasks += [(spec, rule, lat, LOT, d, "back", False) for d in TEST_DAYS]
        tasks += [(spec, "proportional", MAIN_LAT, 0.1, d, "back", False) for d in TEST_DAYS]
    return tasks


def test(workers: int) -> None:
    run_all(test_tasks(), workers)


# ---------------------------------------------------------------------------------------------- report
def fmt(t, nd=2):
    return f"{t[0]:.{nd}f} [{t[1]:.{nd}f}, {t[2]:.{nd}f}]"


def day_block_ci(df: pd.DataFrame, values: np.ndarray, n_boot=N_BOOT, seed=3):
    """Mean with a bootstrap that resamples whole days (hours within a day share that day's conditions)."""
    days = df["day"].to_numpy()
    u = np.unique(days)
    sums = np.array([values[days == d].sum() for d in u])
    counts = np.array([(days == d).sum() for d in u])
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(u), (n_boot, len(u)))
    b = sums[idx].sum(1) / counts[idx].sum(1)
    return float(values.mean()), *np.quantile(b, [0.025, 0.975])


def ratio_ci(num: np.ndarray, den: np.ndarray, n_boot=N_BOOT, seed=4):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(num), (n_boot, len(num)))
    with np.errstate(invalid="ignore", divide="ignore"):
        b = num[idx].sum(1) / den[idx].sum(1)
    return float(num.sum() / den.sum()), *np.nanquantile(b, [0.025, 0.975])


def report() -> None:
    ch = chosen_specs()
    rows = []
    data = {}
    for fee in FEES:
        for rule in RULES:
            for name in STRATS:
                spec = ch[(name, fee)]
                df = load_runs(spec, rule, MAIN_LAT, LOT, TEST_DAYS)
                data[(name, fee, rule)] = df
                p = pnl_at_fee(df, fee)
                m = bootstrap_ci(p, np.mean, N_BOOT, seed=1)
                rows.append({"fee_bps": fee, "rule": rule, "strategy": name, "config": spec_key(spec),
                             "mean_pnl": m[0], "lo": m[1], "hi": m[2],
                             "day_lo": day_block_ci(df, p)[1], "day_hi": day_block_ci(df, p)[2],
                             "pnl_bps": ratio_ci(p, df.notional.to_numpy() / 1e4)[0],
                             "std_pnl": p.std(ddof=1), "sharpe_like": p.mean() / p.std(ddof=1),
                             "btc_per_h": df.fill_qty.mean(), "fills_per_h": df.n_fills.mean(),
                             "mk60_usdt_per_btc": df.mk60_sum.sum() / df.mk_qty.sum(),
                             "captured_usdt_per_btc": df.captured_sum.sum() / df.fill_qty.sum(),
                             "std_q_btc": df.q.std(ddof=1), "max_abs_q_btc": df.max_abs_q.mean(),
                             "share_through": df.qty_through.sum() / df.fill_qty.sum(),
                             "share_time_quoted": (df.quoted_bid + df.quoted_ask).sum() / (2 * 36000 * len(df)),
                             "pnl_minus_penalty": (p - df.penalty.to_numpy()).mean()})
    tab = pd.DataFrame(rows)
    tab.to_csv(OUT / "touch_strategy_table.csv", index=False)

    M = ["# Stage 4: P&L by strategy, fee and cancel rule (test days)\n",
         "10 test days × 23 hour-long episodes = 230 episodes, the same hours for every strategy (paired). "
         f"0.01 BTC lot, {MAIN_LAT} ms latency, inventory limit ±10 lots. Mean P&L per hour (USDT) "
         "[95% bootstrap CI over episodes]; the next column resamples whole days instead (hours within a day are correlated). "
         "Thresholds chosen on the calibration days, separately per fee (tuning.md).\n"]
    for fee in FEES:
        M += [f"\n## Maker fee {fee:+g} bp\n",
              "| cancel rule | strategy | config | mean P&L (USDT/h) | CI resampling days | P&L (bp of notional) | std P&L | BTC filled/h | 60 s markout (USDT/BTC) | spread captured (USDT/BTC) | share filled by sweeps through our price | time quoted |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for rule in RULES:
            for _, r in tab[(tab.fee_bps == fee) & (tab.rule == rule)].iterrows():
                M.append(f"| {rule} | {LABEL[r.strategy]} | {r.config} | {fmt((r.mean_pnl, r.lo, r.hi))} | "
                         f"[{r.day_lo:.2f}, {r.day_hi:.2f}] | {r.pnl_bps:.3f} | {r.std_pnl:.2f} | {r.btc_per_h:.3f} | "
                         f"{r.mk60_usdt_per_btc:.2f} | {r.captured_usdt_per_btc:.3f} | {100 * r.share_through:.0f}% | "
                         f"{100 * r.share_time_quoted:.0f}% |")
    # paired differences against always-at-touch and the OFI question
    M += ["\n## Paired differences (same 230 hours), mean P&L per hour [95% CI]\n",
          "| fee | cancel rule | queue-aware − touch | OFI-aware − touch | skew − touch | OFI-aware − queue-aware | best touch variant − markout-aware A–S |",
          "|---|---|---|---|---|---|---|"]
    diffs = []
    for fee in FEES:
        for rule in RULES:
            P = {n: pnl_at_fee(data[(n, fee, rule)], fee) for n in STRATS}
            best = max(["touch", "touch_skew", "touch_queue", "touch_ofi"], key=lambda n: P[n].mean())
            d = {k: paired_difference(P[a], P[b], np.mean, N_BOOT, seed=2) for k, (a, b) in
                 {"queue": ("touch_queue", "touch"), "ofi": ("touch_ofi", "touch"), "skew": ("touch_skew", "touch"),
                  "ofi_vs_queue": ("touch_ofi", "touch_queue"), "best_vs_as": (best, "markout_as")}.items()}
            diffs.append({"fee": fee, "rule": rule, "best": best, **{k: v for k, v in d.items()}})
            M.append(f"| {fee:+g} | {rule} | {fmt(d['queue'])} | {fmt(d['ofi'])} | {fmt(d['skew'])} | {fmt(d['ofi_vs_queue'])} | "
                     f"{LABEL[best]}: {fmt(d['best_vs_as'])} |")
    (OUT / "touch_strategy_table.md").write_text("\n".join(M) + "\n", encoding="utf-8")

    markout_report(ch)
    latency_report(ch, tab)
    lot_report(ch)
    charts(tab, ch)
    sample_hour(ch)
    log.info("report written to %s", OUT)


def markout_report(ch: dict) -> None:
    """Our fills' markouts by strategy and fill type (main config, fee 0, all rules), and by distance from the mid."""
    rows = []
    allf = []
    for rule in RULES:
        for name in STRATS:
            f = load_fills(ch[(name, 0.0)], rule, MAIN_LAT, LOT, TEST_DAYS)
            f = f[(f.step - (f.hour * 36000)) < (3600 - 60) * 10]
            f["strategy"], f["rule"] = name, rule
            f["delta"] = f.side * (f.mid0 - f.price)
            allf.append(f[["strategy", "rule", "how", "qty", "delta", "mid0", "day", "hour"] + [f"mk{h}" for h in HORIZONS]])
            for how, g in [("all", f)] + list(f.groupby("how")):
                w = g.qty.to_numpy()
                r = {"rule": rule, "strategy": name, "fills": how, "btc": w.sum(), "share": w.sum() / f.qty.sum(),
                     "captured": np.average(g.delta, weights=w)}
                for h in HORIZONS:
                    r[f"mk{h}"] = np.average(g[f"mk{h}"], weights=w)
                    r[f"mk{h}_bps"] = np.average(1e4 * g[f"mk{h}"] / g.mid0, weights=w)
                # CI for the 60 s markout: bootstrap over episodes (ratio of sums)
                e = g.assign(x=g.qty * g.mk60).groupby(["day", "hour"])[["x", "qty"]].sum()
                r["mk60_ci"] = ratio_ci(e.x.to_numpy(), e.qty.to_numpy())[1:]
                rows.append(r)
    t = pd.DataFrame(rows)
    t.to_csv(OUT / "markouts_ours.csv", index=False)
    allf = pd.concat(allf, ignore_index=True)
    # by distance from the mid (all strategies pooled, proportional rule)
    bins = [0, 0.075, 0.2, 1, 2, 3, 5, np.inf]
    labels = ["touch (≤0.05)", "0.1–0.2", "0.2–1", "1–2", "2–3", "3–5", "≥5"]
    g = allf[allf.rule == "proportional"].copy()
    g["bucket"] = pd.cut(g.delta, bins, labels=labels, right=False)
    bd = []
    for lab, x in g.groupby("bucket", observed=True):
        w = x.qty.to_numpy()
        e = x.assign(v=x.qty * x.mk60).groupby(["strategy", "day", "hour"])[["v", "qty"]].sum()
        bd.append({"bucket": lab, "btc": w.sum(), **{f"mk{h}": np.average(x[f"mk{h}"], weights=w) for h in HORIZONS},
                   "mk60_bps": np.average(1e4 * x.mk60 / x.mid0, weights=w), "mk60_ci": ratio_ci(e.v.to_numpy(), e.qty.to_numpy())[1:]})
    bd = pd.DataFrame(bd)
    s3 = pd.read_csv(OUT.parent / "stage3" / "markouts_by_day.csv") if (OUT.parent / "stage3" / "markouts_by_day.csv").exists() else None
    M = ["# Markouts: our fills at the touch vs further out (test days)\n",
         "Markout h = −s (m_{t+h} − m₀): the mid move against us h seconds after the fill (USDT per BTC filled; positive = adverse), "
         "m₀ = the mid at the decision time of the step in which we were filled. Quantity-weighted; fills in the first 59 minutes "
         "of each episode. 0.01 BTC lot, 100 ms latency, fee-0 configurations. 60 s CI: bootstrap over episodes.\n",
         "## By strategy and how the fill happened\n",
         "*queue*: traded volume at our price exceeded the queue ahead of us; *through*: an aggressor traded beyond our price "
         "(the whole level was swept, or our quote was stale above the market); *cross*: the opposite quote moved onto our price.\n",
         "| cancel rule | strategy | fills | BTC | share | spread captured | 1 s | 5 s | 30 s | 60 s [CI] | 60 s (bp) |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for _, r in t.iterrows():
        M.append(f"| {r.rule} | {LABEL[r.strategy]} | {r.fills} | {r.btc:.1f} | {100 * r.share:.0f}% | {r.captured:.3f} | {r.mk1:.2f} | "
                 f"{r.mk5:.2f} | {r.mk30:.2f} | {r.mk60:.2f} [{r.mk60_ci[0]:.2f}, {r.mk60_ci[1]:.2f}] | {r.mk60_bps:.3f} |")
    M += ["\n## By distance from the mid at the fill (all five strategies pooled, proportional rule)\n",
          "| distance (USDT) | BTC | 1 s | 5 s | 30 s | 60 s [CI] | 60 s (bp) |", "|---|---|---|---|---|---|---|"]
    for _, r in bd.iterrows():
        M.append(f"| {r.bucket} | {r.btc:.1f} | {r.mk1:.2f} | {r.mk5:.2f} | {r.mk30:.2f} | {r.mk60:.2f} [{r.mk60_ci[0]:.2f}, {r.mk60_ci[1]:.2f}] | {r.mk60_bps:.3f} |")
    M += ["\nFor comparison, stage 3 measured the market-wide 60 s markout of *all* test-day trades: 0.22 bp at the touch and "
          "0.98–1.16 bp beyond it (results/stage3/markouts.md)."]
    (OUT / "markouts_touch_vs_wide.md").write_text("\n".join(M) + "\n", encoding="utf-8")
    t.to_pickle(OUT.parent.parent / "data" / "derived" / "stage4_markouts_ours.pkl")
    bd.to_csv(OUT / "markouts_by_distance.csv", index=False)

    # chart
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8))
    ax = axes[0]
    sub = t[(t.rule == "proportional")]
    hs = np.array(HORIZONS)
    for name in STRATS:
        for how, ls in (("queue", "-"), ("through", "--")):
            r = sub[(sub.strategy == name) & (sub.fills == how)]
            if len(r):
                r = r.iloc[0]
                ax.plot(hs, [r[f"mk{h}"] for h in HORIZONS], ls, marker="o", ms=3, color=COLOR[name],
                        label=f"{LABEL[name]} – {how}")
    ax.axhline(0, color="black", lw=0.6)
    ax.set_xscale("log"); ax.set_xticks(hs, [f"{h} s" for h in HORIZONS])
    ax.set_ylabel("markout (USDT per BTC, + = adverse)"); ax.set_title("Our fills, proportional rule: queue fills (solid) vs sweeps through our price (dashed)", fontsize=9)
    ax.legend(fontsize=6.5, frameon=False, ncol=2)
    ax = axes[1]
    x = np.arange(len(bd))
    err = np.array([[r.mk60 - r.mk60_ci[0], r.mk60_ci[1] - r.mk60] for _, r in bd.iterrows()]).T
    ax.bar(x, bd.mk60, color="#9467bd", yerr=err, capsize=3, label="our fills (60 s markout)")
    ax.plot(x, bd.captured if "captured" in bd else np.nan, color="black")
    ax.set_xticks(x, bd.bucket, fontsize=8); ax.set_xlabel("distance of our quote from the mid at the fill (USDT)")
    ax.set_ylabel("60 s markout (USDT per BTC)")
    ax.set_title("60 s markout by distance from the mid (all strategies, proportional rule, 95% CI)", fontsize=9)
    fig.tight_layout(); fig.savefig(OUT / "markouts_touch_vs_wide.png", dpi=130); plt.close(fig)


def latency_report(ch: dict, tab: pd.DataFrame) -> None:
    rows = []
    for name in STRATS:
        spec = ch[(name, 0.0)]
        for rule in RULES:
            for lat in (0, 100, 300):
                df = load_runs(spec, rule, lat, LOT, TEST_DAYS)
                for fee in FEES:
                    if fee_dependent(spec) and fee != 0.0:
                        continue          # markout-aware A-S: latency runs only at its fee-0 quotes
                    p = pnl_at_fee(df, fee)
                    m = bootstrap_ci(p, np.mean, N_BOOT, seed=1)
                    rows.append({"strategy": name, "rule": rule, "latency_ms": lat, "fee_bps": fee, "mean_pnl": m[0],
                                 "lo": m[1], "hi": m[2], "btc_per_h": df.fill_qty.mean(),
                                 "mk60": df.mk60_sum.sum() / df.mk_qty.sum(),
                                 "share_through": df.qty_through.sum() / df.fill_qty.sum()})
    t = pd.DataFrame(rows)
    t.to_csv(OUT / "latency_sensitivity.csv", index=False)
    M = ["# Latency and cancel-rule sensitivity (test days, fee-0 configurations)\n",
         "Mean P&L per hour (USDT) [95% CI over 230 episodes]; BTC filled per hour; 60 s markout (USDT/BTC). Fees −0.5 and +2 bp are exact "
         "repricings of the same fills (not for the markout-aware A–S, whose quotes depend on the fee: fee 0 only).\n",
         "| strategy | cancel rule | fee | 0 ms | 100 ms | 300 ms | BTC/h (0 / 100 / 300) | markout 60 s (0 / 100 / 300) |", "|---|---|---|---|---|---|---|---|"]
    for (name, rule, fee), g in t.groupby(["strategy", "rule", "fee_bps"], sort=False):
        g = g.set_index("latency_ms")
        M.append(f"| {LABEL[name]} | {rule} | {fee:+g} | " + " | ".join(fmt((g.loc[l, 'mean_pnl'], g.loc[l, 'lo'], g.loc[l, 'hi'])) for l in (0, 100, 300))
                 + " | " + " / ".join(f"{g.loc[l, 'btc_per_h']:.2f}" for l in (0, 100, 300))
                 + " | " + " / ".join(f"{g.loc[l, 'mk60']:.2f}" for l in (0, 100, 300)) + " |")
    (OUT / "latency_sensitivity.md").write_text("\n".join(M) + "\n", encoding="utf-8")

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6), sharey=True)
    for ax, fee in zip(axes, FEES):
        for name in STRATS:
            for rule, ls in zip(RULES, (":", "-", "--")):
                g = t[(t.strategy == name) & (t.rule == rule) & (t.fee_bps == fee)].sort_values("latency_ms")
                if len(g):
                    ax.plot(g.latency_ms, g.mean_pnl, ls, marker="o", ms=3, color=COLOR[name],
                            label=f"{LABEL[name]}" if rule == "proportional" else None)
        ax.axhline(0, color="black", lw=0.6)
        ax.set_xticks([0, 100, 300]); ax.set_xlabel("latency (ms)")
        ax.set_title(f"fee {fee:+g} bp (dotted pessimistic, solid proportional, dashed optimistic)", fontsize=8.5)
    axes[0].set_ylabel("mean P&L (USDT per hour, 0.01 BTC lot)")
    axes[0].legend(fontsize=7, frameon=False)
    fig.tight_layout(); fig.savefig(OUT / "latency_sensitivity.png", dpi=130); plt.close(fig)


def lot_report(ch: dict) -> None:
    M = ["# Lot size sensitivity (test days, proportional rule, 100 ms, fee-0 configurations)\n",
         "Mean P&L per hour (USDT) and per BTC traded (bp of notional), 0.01 vs 0.1 BTC lot. The replay assumes our order does not "
         "change anyone else's behaviour; that assumption is weaker for 0.1 BTC.\n",
         "| strategy | lot | fee | mean P&L (USDT/h) | P&L (bp of notional) | BTC filled/h |", "|---|---|---|---|---|---|"]
    for name in STRATS:
        spec = ch[(name, 0.0)]
        for lot in (0.01, 0.1):
            df = load_runs(spec, "proportional", MAIN_LAT, lot, TEST_DAYS)
            for fee in ([0.0] if fee_dependent(spec) else FEES):
                p = pnl_at_fee(df, fee)
                M.append(f"| {LABEL[name]} | {lot:g} | {fee:+g} | {fmt(bootstrap_ci(p, np.mean, N_BOOT, seed=1))} | "
                         f"{ratio_ci(p, df.notional.to_numpy() / 1e4)[0]:.3f} | {df.fill_qty.mean():.3f} |")
    (OUT / "lot_sensitivity.md").write_text("\n".join(M) + "\n", encoding="utf-8")


def charts(tab: pd.DataFrame, ch: dict) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(17, 5), sharey=False)
    for ax, fee in zip(axes, FEES):
        w = 0.16
        for j, name in enumerate(STRATS):
            sub = tab[(tab.fee_bps == fee) & (tab.strategy == name)].set_index("rule").loc[RULES]
            x = np.arange(len(RULES)) + (j - 2) * w
            ax.bar(x, sub.mean_pnl, w, color=COLOR[name], label=LABEL[name],
                   yerr=[sub.mean_pnl - sub.lo, sub.hi - sub.mean_pnl], capsize=2)
        ax.axhline(0, color="black", lw=0.6)
        ax.set_xticks(range(len(RULES)), RULES)
        ax.set_title(f"maker fee {fee:+g} bp", fontsize=10)
    axes[0].set_ylabel("mean P&L (USDT per hour, 0.01 BTC lot)")
    axes[0].legend(fontsize=7.5, frameon=False)
    fig.suptitle("Replay on the test days: P&L by strategy, fee and cancel rule (230 paired hours, 100 ms latency, 95% CI)")
    fig.tight_layout(); fig.savefig(OUT / "pnl_by_strategy_fee_rule.png", dpi=130); plt.close(fig)


def sample_hour(ch: dict) -> None:
    """One test hour, always-at-touch (proportional rule): book, our quotes, queue ahead and fills."""
    from mms.replay.data import load_day
    from mms.replay.env import ReplayConfig, run_episode
    from stage4_common import make_policy
    day, hour = TEST_DAYS[1], 14
    d = load_day(day)
    start = hour * 36000
    r = run_episode(make_policy(ch[("touch_queue", 0.0)]), d, start,
                    ReplayConfig(latency_ms=100, cancel_rule="proportional", record=True))
    R = r.record
    n = len(R["mid"])
    t = np.arange(n) / 10.0
    i0 = 600 * 10                     # show 3 minutes starting 10 minutes in
    sl = slice(i0, i0 + 1800)
    f = r.fills
    fs = (f["step"] - start)
    fig, axes = plt.subplots(3, 1, figsize=(13, 9), sharex=True, gridspec_kw={"height_ratios": [2.3, 1.3, 1]})
    ax = axes[0]
    ax.step(t[sl], R["best_bid"][sl] * 0.1, where="post", color="#1f77b4", lw=0.8, label="best bid")
    ax.step(t[sl], R["best_ask"][sl] * 0.1, where="post", color="#d62728", lw=0.8, label="best ask")
    ax.step(t[sl], R["bid_px"][sl] * 0.1, where="post", color="#1f77b4", lw=2.2, alpha=0.45, label="our bid")
    ax.step(t[sl], R["ask_px"][sl] * 0.1, where="post", color="#d62728", lw=2.2, alpha=0.45, label="our ask")
    m = (fs >= sl.start) & (fs < sl.stop)
    for side, mk, c in ((1, "^", "#1f77b4"), (-1, "v", "#d62728")):
        k = m & (f["side"] == side)
        ax.scatter(fs[k] / 10.0, f["price"][k] * 0.1, marker=mk, s=40, color=c, edgecolor="black", zorder=5,
                   label="we buy" if side == 1 else "we sell")
    ax.set_ylabel("USDT"); ax.legend(fontsize=7.5, frameon=False, ncol=6)
    ax.set_title(f"{day}, {hour:02d}:10–{hour:02d}:13 UTC: touch + queue-aware (θ = {ch[('touch_queue', 0.0)]['theta']:g}), "
                 "proportional cancel rule, 100 ms latency", fontsize=10)
    ax = axes[1]
    ax.step(t[sl], R["Q_bid"][sl], where="post", color="#1f77b4", label="queue ahead of our bid (BTC)")
    ax.step(t[sl], R["Q_ask"][sl], where="post", color="#d62728", label="queue ahead of our ask (BTC)")
    ax.step(t[sl], R["bid_size"][sl], where="post", color="#1f77b4", lw=0.6, ls=":", label="displayed best-bid size")
    ax.step(t[sl], R["ask_size"][sl], where="post", color="#d62728", lw=0.6, ls=":", label="displayed best-ask size")
    ax.set_ylabel("BTC"); ax.set_yscale("symlog", linthresh=0.1); ax.legend(fontsize=7.5, frameon=False, ncol=2)
    ax = axes[2]
    ax.step(t[sl], R["q"][sl] / 0.01, where="post", color="black")
    ax.set_ylabel("inventory (lots)"); ax.set_xlabel("seconds into the hour")
    fig.tight_layout(); fig.savefig(OUT / "sample_hour_queue.png", dpi=130); plt.close(fig)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["tune", "test", "report"])
    ap.add_argument("--workers", type=int, default=14)
    a = ap.parse_args()
    {"tune": lambda: tune(a.workers), "test": lambda: test(a.workers), "report": report}[a.phase]()


if __name__ == "__main__":
    main()
