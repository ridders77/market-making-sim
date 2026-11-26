"""Stage 3, Step 5: strategy comparison with adverse selection.

Strategies: naive (A-S's average spread), A-S, fee-aware A-S, markout-aware A-S (ablation: flow-aware without
the signal) and flow-aware A-S (fee-aware). Settings: model (b) calibration per volatility regime (sigma_60s),
0.01 BTC lot, gamma = 1e-4 per lot, T = 1 h, dt = 0.2 s, tick 0.1, 1,000 paired paths (seeds 20,000-20,999).
Environments: no informed flow (stage 2 dynamics) and informed flow (markout jumps, OFI signal, fill tilt)
at predictive power x1; medium regime also swept over x0, x0.5, x1.5, x2.
Fees: -0.5, 0, 2 bp; fee-dependent strategies also on a grid (-2 ... 4 bp) for break-even fees.

Answers: (1) share of A-S profit that was adverse selection, (2) how much flow-awareness recovers,
(3) break-even fee per strategy, (4) quoted spread vs the real 1-tick spread.

Writes results/stage3/: strategy_table.{csv,md}, answers.md, pnl_by_strategy_fee.png, break_even_fee.png,
predictive_power_sweep.png, sample_path_flow.png. Raw per-path results are cached in data/derived/stage3_runs/.

Run:  .venv\\Scripts\\python scripts\\run_stage3_sim.py
"""

from __future__ import annotations

import json
import logging
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from mms.analysis.metrics import paired_difference, ratio_of_means_ci, summary_row
from mms.sim.env import EnvConfig, FlowParams, run_env
from mms.sim.simulator import MarketParams, SimResult
from mms.strategies.avellaneda_stoikov import AvellanedaStoikov, FeeAwareAvellanedaStoikov
from mms.strategies.flow_aware import FlowAwareAvellanedaStoikov
from mms.strategies.naive import NaiveSymmetric

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "results" / "stage3"
CACHE = REPO / "data" / "derived" / "stage3_runs"
T, DT, TICK, GAMMA, LOT = 3600.0, 0.2, 0.1, 1e-4, 0.01
SEEDS = range(20_000, 21_000)
FEES = [-0.5, 0.0, 2.0]
FEE_GRID = [-1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 3.0, 4.0]
MIN_DISTANCE = 0.1   # quotes kept >= 0.1 USDT from the mid: the calibrated range of lambda and J
SWEEP = [0.0, 0.5, 1.0, 1.5, 2.0]
REAL_SPREAD = 0.1
N_BOOT = 2000
STRATS = ["naive", "avellaneda_stoikov", "fee_aware_as", "markout_aware_as", "flow_aware_as"]
FEE_DEPENDENT = {"fee_aware_as", "markout_aware_as", "flow_aware_as"}
LABEL = {"naive": "naive", "avellaneda_stoikov": "A–S", "fee_aware_as": "fee-aware A–S",
         "markout_aware_as": "markout-aware A–S (no signal)", "flow_aware_as": "flow-aware A–S"}
COLOR = {"naive": "#999999", "avellaneda_stoikov": "#1f77b4", "fee_aware_as": "#2ca02c",
         "markout_aware_as": "#ff7f0e", "flow_aware_as": "#d62728"}
log = logging.getLogger("stage3_sim")

CAL = json.loads((OUT / "calibration_stage3.json").read_text(encoding="utf-8"))
FLOW = json.loads((OUT / "flow_calibration.json").read_text(encoding="utf-8"))
REGIMES = list(CAL["regimes"])


def market(reg: str) -> MarketParams:
    c = CAL["regimes"][reg]
    return MarketParams(s0=c["s0"], sigma=c["sigma"], T=T, dt=DT, A=c["A"], k=c["k"], fill_rule="exact")


def flow_params(reg: str, mult: float | None) -> FlowParams | None:
    if mult is None:
        return None
    f = FLOW["regimes"][reg]
    return FlowParams(kappa=mult * f["kappa_usdt_per_s"], tau=f["tau_s"], theta=f["theta"],
                      jump_delta=tuple(f["jump_delta"]), jump_mean=tuple(f["jump_mean"]), jump_sd=f["jump_sd"])


def strategy(name: str, reg: str, fee: float):
    p, f = market(reg), FLOW["regimes"][reg]
    a_s = AvellanedaStoikov(gamma=GAMMA, sigma=p.sigma, k=p.k, T=T)
    if name == "avellaneda_stoikov":
        return a_s
    if name == "naive":
        return NaiveSymmetric(float(np.mean([a_s.spread(i * DT) for i in range(p.n_steps)])), name="naive")
    if name == "fee_aware_as":
        return FeeAwareAvellanedaStoikov(gamma=GAMMA, sigma=p.sigma, k=p.k, T=T, fee_bps=fee)
    return FlowAwareAvellanedaStoikov(
        gamma=GAMMA, sigma=p.sigma, k=p.k, T=T, fee_bps=fee, kappa=f["kappa_usdt_per_s"], tau_x=f["tau_s"],
        jump_delta=tuple(f["jump_delta"]), jump_mean=tuple(f["jump_mean"]), s_ref=p.s0,
        use_signal=(name == "flow_aware_as"), name=name)


def key(reg, mult, name, fee) -> str:
    m = "none" if mult is None else f"{mult:g}"
    return f"{reg}_flow{m}_{name}_fee{fee:+g}"


def run_task(task) -> str:
    reg, mult, name, fee = task
    path = CACHE / f"{key(reg, mult, name, fee)}.npz"
    if path.exists():
        return path.name
    cfg = EnvConfig(market(reg), lot=LOT, tick_size=TICK, maker_fee_bps=fee, flow=flow_params(reg, mult),
                    min_distance=MIN_DISTANCE)
    r = run_env(strategy(name, reg, fee), cfg, SEEDS, chunk=250)
    tmp = path.with_suffix(".tmp.npz")
    np.savez(tmp, seeds=r.seeds, pnl=r.pnl, cash=r.cash, q=r.q, max_abs_q=r.max_abs_q, n_bid_fills=r.n_bid_fills,
             n_ask_fills=r.n_ask_fills, spread_captured_sum=r.spread_captured_sum,
             mean_quoted_spread=r.mean_quoted_spread, notional=r.notional,
             meta=np.array([r.n_steps, r.n_clipped, r.fee_bps, r.lot, r.n_min_distance]))
    tmp.replace(path)
    return path.name


def load(reg, mult, name, fee) -> SimResult:
    """Fee-independent strategies are simulated at fee 0 and repriced exactly (P&L(f) = P&L(0) - f * notional)."""
    sim_fee = fee if name in FEE_DEPENDENT else 0.0
    z = np.load(CACHE / f"{key(reg, mult, name, sim_fee)}.npz")
    n_steps, n_clipped, fee_run, lot, n_md = z["meta"]
    r = SimResult(seeds=z["seeds"], pnl=z["pnl"], cash=z["cash"], q=z["q"], max_abs_q=z["max_abs_q"],
                  n_bid_fills=z["n_bid_fills"], n_ask_fills=z["n_ask_fills"], spread_captured_sum=z["spread_captured_sum"],
                  mean_quoted_spread=z["mean_quoted_spread"], n_steps=int(n_steps), n_clipped=int(n_clipped),
                  notional=z["notional"], fee_bps=float(fee_run), lot=float(lot),
                  n_min_distance=int(n_md))
    if fee != r.fee_bps:
        r = SimResult(**{**r.__dict__, "pnl": r.pnl_with_fee(fee), "fee_bps": fee})
    return r


def tasks() -> list:
    out = set()
    for reg in REGIMES:
        for mult, fees in ((None, FEES), (1.0, FEE_GRID)):
            out |= {(reg, mult, n, 0.0) for n in ("naive", "avellaneda_stoikov")}
            out |= {(reg, mult, n, f) for n in FEE_DEPENDENT for f in fees}
    for mult in SWEEP:
        out |= {("medium", mult, n, 0.0) for n in ("naive", "avellaneda_stoikov")}
        out |= {("medium", mult, n, f) for n in FEE_DEPENDENT for f in FEES}
    return sorted(out, key=lambda t: (t[0], -1 if t[1] is None else t[1], t[2], t[3]))


def fmt(t, nd=2) -> str:
    return f"{t[0]:.{nd}f} [{t[1]:.{nd}f}, {t[2]:.{nd}f}]"


def env_label(mult) -> str:
    return "no informed flow" if mult is None else f"informed flow ×{mult:g}"


def break_even(reg, mult, name, n_boot=N_BOOT, seed=7) -> tuple[float, float, float] | str:
    """Fee (bp) at which the mean P&L is zero, with a paired bootstrap CI over paths.

    Fixed quotes: P&L(f) = P&L(0) - f N, so f* = P&L(0)/N in bps (a ratio of means).
    Fee-dependent quotes: mean P&L on the fee grid (same seeds at every fee), linear interpolation of the
    first sign change; if there is none the answer is reported as outside the grid."""
    if name not in FEE_DEPENDENT:
        r = load(reg, mult, name, 0.0)
        return ratio_of_means_ci(r.pnl, r.notional, 1e4, n_boot, seed)
    runs = [load(reg, mult, name, f) for f in FEE_GRID]
    active = np.array([r.n_fills.mean() >= 1.0 for r in runs])         # >= 1 fill per hour on average
    if not active.all():
        # Above some fee the strategy (rightly) stops quoting anything that fills; its P&L ~ 0 there is not a
        # break-even, so the inactive fees are left out and the cut-off is reported.
        first_off = FEE_GRID[int(np.argmin(active))]
        runs = [r for r, a in zip(runs, active) if a]
        m = np.array([r.pnl.mean() for r in runs])
        if (m > 0).all():
            return f"never loses; stops trading (<1 fill/h) from {first_off:+g} bp"
    P = np.array([r.pnl for r in runs])                                  # fees x paths
    grid = np.array(FEE_GRID)[active]

    def root(m):
        s = np.sign(m)
        idx = np.flatnonzero((s[:-1] > 0) & (s[1:] <= 0))
        if len(idx) == 0:
            return np.nan
        i = idx[0]
        return grid[i] + (grid[i + 1] - grid[i]) * m[i] / (m[i] - m[i + 1])

    est = root(P.mean(1))
    if np.isnan(est):
        m = P.mean(1)
        return f"> {grid[-1]:g} bp" if (m > 0).all() else (f"< {grid[0]:g} bp" if (m < 0).all() else "no single crossing")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, P.shape[1], (n_boot, P.shape[1]))
    b = np.array([root(P[:, i].mean(1)) for i in idx])
    b = b[np.isfinite(b)]
    return float(est), *map(float, np.quantile(b, [0.025, 0.975]))


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    CACHE.mkdir(parents=True, exist_ok=True)
    todo = tasks()
    log.info("%d runs (%d cached)", len(todo), sum((CACHE / f"{key(*t)}.npz").exists() for t in todo))
    with ProcessPoolExecutor(max_workers=8) as ex:
        for i, name in enumerate(ex.map(run_task, todo), 1):
            if i % 20 == 0:
                log.info("%d/%d done (%s)", i, len(todo), name)

    # ------------------------------------------------------------ main table
    rows = []
    for reg in REGIMES:
        for mult in (None, 1.0):
            for fee in FEES:
                res = {n: load(reg, mult, n, fee) for n in STRATS}
                for n in STRATS:
                    rows.append({"regime": reg, "env": env_label(mult), "flow_mult": mult, "fee_bps": fee, "strategy": n,
                                 **summary_row(res[n], n_boot=N_BOOT, seed=1)})
                for a, b in (("fee_aware_as", "avellaneda_stoikov"), ("markout_aware_as", "fee_aware_as"),
                             ("flow_aware_as", "markout_aware_as"), ("flow_aware_as", "avellaneda_stoikov")):
                    d = paired_difference(res[a].pnl, res[b].pnl, np.mean, N_BOOT, seed=2)
                    rows.append({"regime": reg, "env": env_label(mult), "flow_mult": mult, "fee_bps": fee,
                                 "strategy": f"{a} minus {b}", "mean_pnl": d[0], "mean_pnl_lo": d[1], "mean_pnl_hi": d[2]})
            log.info("table: %s done", reg)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "strategy_table.csv", index=False)

    M = ["# Stage 3: strategy comparison with adverse selection\n",
         f"Model (b) calibration per regime, σ_60s, 0.01 BTC lot, γ = {GAMMA:g} per lot, T = 1 h, dt = {DT} s, tick {TICK}, "
         f"1,000 paired paths (seeds {SEEDS.start}–{SEEDS.stop - 1}). P&L per hour in USDT and in bps of traded notional, "
         "[95% bootstrap CI]. Informed flow ×1 = markout jumps + OFI signal + fill tilt at the measured strength. "
         "Quotes are kept at least 0.1 USDT from the mid (the calibrated range); the last column is the share of quote-sides moved out to it.\n"]
    for reg in REGIMES:
        c = CAL["regimes"][reg]
        for mult in (None, 1.0):
            sub = df[(df.regime == reg) & (df.env == env_label(mult))]
            M += [f"\n## {reg} volatility, {env_label(mult)} (σ = {c['sigma']:.3f}, A = {c['A']:.4f}, k = {c['k']:.3f}, S₀ = {c['s0']:.0f})\n",
                  "| fee (bp) | strategy | quoted spread (× real) | mean P&L (USDT/h) | P&L (bps of notional) | std P&L | mean/std | std q_T (BTC) | fills/h | quotes held at 0.1 |",
                  "|---|---|---|---|---|---|---|---|---|---|"]
            for _, r in sub.iterrows():
                if " minus " in r.strategy:
                    a, b = r.strategy.split(" minus ")
                    M.append(f"| {r.fee_bps:+g} | *{LABEL[a]} − {LABEL[b]}* | | {fmt((r.mean_pnl, r.mean_pnl_lo, r.mean_pnl_hi))} | | | | | | |")
                else:
                    M.append(f"| {r.fee_bps:+g} | {LABEL[r.strategy]} | {r.mean_quoted_spread:.2f} ({r.mean_quoted_spread / REAL_SPREAD:.0f}×) | "
                             f"{fmt((r.mean_pnl, r.mean_pnl_lo, r.mean_pnl_hi))} | {fmt((r.pnl_bps, r.pnl_bps_lo, r.pnl_bps_hi), 3)} | "
                             f"{r.std_pnl:.2f} | {r.sharpe_like:.2f} | {r.std_final_q_btc:.3f} | {r.fills_per_episode:.0f} | "
                             f"{100 * r.share_min_distance:.2f}% |")
    (OUT / "strategy_table.md").write_text("\n".join(M) + "\n", encoding="utf-8")

    # ------------------------------------------------------------ answers
    A = ["# Stage 3: answers\n"]
    # Q1 adverse selection share of A-S profit
    A += ["## 1. How much of A–S's profit (no informed flow) was really adverse selection?\n",
          "Same seeds, same strategy; the only change is switching informed flow on (×1). "
          "Share lost = (P&L without − P&L with) / P&L without. 95% CI: paired bootstrap over paths.\n",
          "| regime | fee (bp) | strategy | P&L without (USDT/h) | P&L with informed flow | difference | share of profit lost |",
          "|---|---|---|---|---|---|---|"]
    q1 = []
    rng = np.random.default_rng(11)
    for reg in REGIMES:
        for fee in FEES:
            for n in ("avellaneda_stoikov", "naive"):
                a, b = load(reg, None, n, fee).pnl, load(reg, 1.0, n, fee).pnl
                d = paired_difference(a, b, np.mean, N_BOOT, seed=3)
                idx = rng.integers(0, len(a), (N_BOOT, len(a)))
                sh = (a[idx].mean(1) - b[idx].mean(1)) / a[idx].mean(1)
                share = ((a.mean() - b.mean()) / a.mean(), *np.quantile(sh, [0.025, 0.975])) if a.mean() > 0 else None
                q1.append({"regime": reg, "fee": fee, "strategy": n, "without": a.mean(), "with": b.mean(), "diff": d,
                           "share": share})
                A.append(f"| {reg} | {fee:+g} | {LABEL[n]} | {a.mean():.2f} | {b.mean():.2f} | {fmt(d)} | "
                         + (f"{100 * share[0]:.0f}% [{100 * share[1]:.0f}%, {100 * share[2]:.0f}%]" if share else "n/a (no profit to lose)") + " |")
    # Q2 recovery
    A += ["\n## 2. Does flow-awareness recover it?\n",
          "Adverse-selection cost = fee-aware A–S without informed flow − with it. Recovered = (strategy − fee-aware A–S), "
          "both with informed flow ×1, as a share of that cost. The markout-aware ablation (no signal) separates pricing "
          "the average markout from reading the order flow.\n",
          "| regime | fee (bp) | AS cost (USDT/h) | markout-aware − fee-aware | flow-aware − markout-aware (signal value) | flow-aware recovered share | flow-aware vs no-flow fee-aware |",
          "|---|---|---|---|---|---|---|"]
    q2 = []
    for reg in REGIMES:
        for fee in FEES:
            fa0, fa1 = load(reg, None, "fee_aware_as", fee).pnl, load(reg, 1.0, "fee_aware_as", fee).pnl
            ma1, fl1 = load(reg, 1.0, "markout_aware_as", fee).pnl, load(reg, 1.0, "flow_aware_as", fee).pnl
            cost = paired_difference(fa0, fa1, np.mean, N_BOOT, seed=4)
            d_ma = paired_difference(ma1, fa1, np.mean, N_BOOT, seed=4)
            d_sig = paired_difference(fl1, ma1, np.mean, N_BOOT, seed=4)
            idx = rng.integers(0, len(fa0), (N_BOOT, len(fa0)))
            rec_b = (fl1[idx].mean(1) - fa1[idx].mean(1)) / (fa0[idx].mean(1) - fa1[idx].mean(1))
            rec = ((fl1.mean() - fa1.mean()) / (fa0.mean() - fa1.mean()), *np.quantile(rec_b, [0.025, 0.975]))
            vs = paired_difference(fl1, fa0, np.mean, N_BOOT, seed=4)
            q2.append({"regime": reg, "fee": fee, "cost": cost, "d_ma": d_ma, "d_sig": d_sig, "rec": rec, "vs_noflow": vs})
            A.append(f"| {reg} | {fee:+g} | {fmt(cost)} | {fmt(d_ma)} | {fmt(d_sig, 3)} | {100 * rec[0]:.0f}% [{100 * rec[1]:.0f}%, {100 * rec[2]:.0f}%] | {fmt(vs)} |")
    # sweep
    A += ["\n### Predictive-power sweep (medium regime)\n",
          "The true drift per unit signal is scaled by ×0…×2 (×1 = measured); the flow-aware strategy always assumes ×1. "
          "Mean P&L (USDT/h) [95% CI].\n",
          "| fee (bp) | strategy | " + " | ".join(f"×{m:g}" for m in SWEEP) + " |", "|---|---" + "|---" * len(SWEEP) + "|"]
    for fee in FEES:
        for n in STRATS:
            cells = []
            for m in SWEEP:
                p = load("medium", m, n, fee).pnl
                cells.append(f"{p.mean():.2f}")
            A.append(f"| {fee:+g} | {LABEL[n]} | " + " | ".join(cells) + " |")
        cells = [fmt(paired_difference(load("medium", m, "flow_aware_as", fee).pnl, load("medium", m, "markout_aware_as", fee).pnl,
                                       np.mean, N_BOOT, seed=5), 3) for m in SWEEP]
        A.append(f"| {fee:+g} | *signal value: flow-aware − markout-aware* | " + " | ".join(cells) + " |")
    # Q3 break-even
    A += ["\n## 3. Break-even maker fee (bp) — mean P&L = 0\n",
          "Fixed quotes (naive, A–S): f* = P&L at zero fee / traded notional. Fee-dependent quotes: interpolated on the fee grid "
          f"{FEE_GRID} with the strategy re-optimised for each fee. Positive = the strategy can pay this fee; negative = it needs a rebate. "
          "95% CI: bootstrap over paths.\n",
          "| regime | environment | " + " | ".join(LABEL[n] for n in STRATS) + " |", "|---|---" + "|---" * len(STRATS) + "|"]
    be = {}
    for reg in REGIMES:
        for mult in (None, 1.0):
            cells = []
            for n in STRATS:
                if mult is None and n in FEE_DEPENDENT:
                    cells.append("—")
                    continue
                v = break_even(reg, mult, n)
                be[(reg, mult, n)] = v
                cells.append(v if isinstance(v, str) else fmt(v))
            A.append(f"| {reg} | {env_label(mult)} | " + " | ".join(cells) + " |")
    A += ["\nMean P&L (USDT/h) on the fee grid, informed flow ×1:\n", "| regime | strategy | " + " | ".join(f"{f:+g}" for f in FEE_GRID) + " |",
          "|---|---" + "|---" * len(FEE_GRID) + "|"]
    for reg in REGIMES:
        for n in STRATS:
            A.append(f"| {reg} | {LABEL[n]} | " + " | ".join(f"{load(reg, 1.0, n, f).pnl.mean():.2f}" for f in FEE_GRID) + " |")
    # Q4 reality check
    A += ["\n## 4. Reality check: quoted spread vs the real 1-tick (0.1 USDT) spread\n",
          "Time-averaged quoted spread (ask − bid, USDT) with informed flow ×1, and fills per hour.\n",
          "| regime | fee (bp) | " + " | ".join(LABEL[n] for n in STRATS) + " |", "|---|---" + "|---" * len(STRATS) + "|"]
    for reg in REGIMES:
        for fee in FEES:
            cells = []
            for n in STRATS:
                r = load(reg, 1.0, n, fee)
                s = r.mean_quoted_spread.mean()
                cells.append(f"{s:.1f} ({s / REAL_SPREAD:.0f}×, {r.n_fills.mean():.0f}/h)")
            A.append(f"| {reg} | {fee:+g} | " + " | ".join(cells) + " |")
    (OUT / "answers.md").write_text("\n".join(A) + "\n", encoding="utf-8")
    pd.DataFrame([{"regime": k[0], "env": env_label(k[1]), "strategy": k[2],
                   **({"break_even_bp": v[0], "lo": v[1], "hi": v[2]} if not isinstance(v, str) else {"note": v})}
                  for k, v in be.items()]).to_csv(OUT / "break_even_fees.csv", index=False)

    # ------------------------------------------------------------ charts
    fig, axes = plt.subplots(2, 3, figsize=(18, 8.5), sharey="row")
    for i, mult in enumerate((None, 1.0)):
        for j, reg in enumerate(REGIMES):
            ax = axes[i, j]
            width = 0.16
            for s_i, n in enumerate(STRATS):
                sub = df[(df.regime == reg) & (df.env == env_label(mult)) & (df.strategy == n)]
                x = np.arange(len(FEES)) + (s_i - 2) * width
                ax.bar(x, sub.mean_pnl, width, color=COLOR[n], label=LABEL[n],
                       yerr=[sub.mean_pnl - sub.mean_pnl_lo, sub.mean_pnl_hi - sub.mean_pnl], capsize=2)
            ax.axhline(0, color="black", lw=0.7)
            ax.set_xticks(range(len(FEES)), [f"{f:+g} bp" for f in FEES])
            ax.set_title(f"{reg} vol, {env_label(mult)}", fontsize=10)
        axes[i, 0].set_ylabel("mean P&L (USDT per hour, 0.01 BTC lot)")
    axes[0, 0].legend(fontsize=8, frameon=False)
    fig.suptitle("P&L by strategy and maker fee (1,000 paired paths, 95% CI)")
    fig.tight_layout(); fig.savefig(OUT / "pnl_by_strategy_fee.png", dpi=130); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8))
    ax = axes[0]
    for n in STRATS:
        m = [load("medium", 1.0, n, f).pnl.mean() for f in FEE_GRID]
        ax.plot(FEE_GRID, m, "o-", ms=3, color=COLOR[n], label=LABEL[n])
    ax.axhline(0, color="black", lw=0.7)
    ax.set_xlabel("maker fee (bp; negative = rebate)"); ax.set_ylabel("mean P&L (USDT/h)")
    ax.set_title("Medium vol, informed flow ×1: P&L vs fee (each fee-aware strategy re-optimised per fee)", fontsize=9)
    ax.legend(fontsize=8, frameon=False)
    ax = axes[1]
    width = 0.16
    for s_i, n in enumerate(STRATS):
        for j, reg in enumerate(REGIMES):
            v = be.get((reg, 1.0, n))
            x = j + (s_i - 2) * width
            if isinstance(v, tuple):
                ax.bar(x, v[0], width, color=COLOR[n], yerr=[[v[0] - v[1]], [v[2] - v[0]]], capsize=2,
                       label=LABEL[n] if j == 0 else None)
            else:
                ax.text(x, 0.02, "never loses" if "never" in v or ">" in v else v, rotation=90, fontsize=7, ha="center", va="bottom", color=COLOR[n])
    ax.axhline(0, color="black", lw=0.7)
    ax.set_xticks(range(len(REGIMES)), REGIMES); ax.set_ylabel("break-even maker fee (bp)")
    ax.set_title("Break-even fee by strategy, informed flow ×1 (text: no crossing on the grid)", fontsize=9)
    ax.legend(fontsize=7, frameon=False)
    fig.tight_layout(); fig.savefig(OUT / "break_even_fee.png", dpi=130); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for n in STRATS:
        m = [load("medium", mm, n, 0.0).pnl.mean() for mm in SWEEP]
        ax.plot(SWEEP, m, "o-", color=COLOR[n], label=LABEL[n])
    ax.axhline(0, color="black", lw=0.7)
    ax.set_xlabel("true predictive power of the flow signal (× measured)"); ax.set_ylabel("mean P&L (USDT/h)")
    ax.set_title("Medium vol, fee 0: value of the signal (flow-aware assumes ×1)", fontsize=10)
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout(); fig.savefig(OUT / "predictive_power_sweep.png", dpi=130); plt.close(fig)

    # sample path: one medium-regime path, informed flow x1, fee 0
    reg, fee = "medium", 0.0
    strat = strategy("flow_aware_as", reg, fee)
    cfg = EnvConfig(market(reg), lot=LOT, tick_size=TICK, maker_fee_bps=fee, flow=flow_params(reg, 1.0),
                    min_distance=MIN_DISTANCE)
    r = run_env(strat, cfg, [SEEDS.start], record_paths=True)
    P = r.paths
    t = P["t"][:-1]
    x = P["signal"][0, :-1]
    i0 = int(np.argmax(np.abs(x[: len(x) // 2]))) - 300          # a window around a strong flow episode
    i0 = max(i0, 0)
    sl = slice(i0, i0 + 900)                                     # 180 s
    D = strat.drift_shift(x, t)
    mid, bid, ask = P["mid"][0, :-1], P["bid"][0], P["ask"][0]
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True, gridspec_kw={"height_ratios": [2.2, 1.3, 1.3]})
    ax = axes[0]
    ax.plot(t[sl], mid[sl], color="black", lw=1, label="mid")
    ax.plot(t[sl], ask[sl], color="#d62728", lw=1, label="flow-aware ask")
    ax.plot(t[sl], bid[sl], color="#1f77b4", lw=1, label="flow-aware bid")
    ax.plot(t[sl], ask[sl] - D[sl], color="#d62728", lw=0.8, ls=":", label="same quotes without the signal")
    ax.plot(t[sl], bid[sl] - D[sl], color="#1f77b4", lw=0.8, ls=":")
    ax.set_ylabel("USDT"); ax.legend(fontsize=8, frameon=False, ncol=4)
    ax.set_title("Medium vol, informed flow ×1, one path: quotes lean ahead of the order flow")
    ax = axes[1]
    centre = (bid + ask) / 2 - mid
    ax.plot(t[sl], centre[sl], color="#9467bd", lw=1, label="quote centre − mid (USDT)")
    ax.plot(t[sl], D[sl], color="#ff7f0e", lw=1, ls="--", label="signal shift D = κτ(1−e^(−(T−t)/τ))·x")
    ax.axhline(0, color="grey", lw=0.6); ax.legend(fontsize=8, frameon=False); ax.set_ylabel("USDT")
    ax = axes[2]
    ax.plot(t[sl], x[sl], color="#2ca02c", lw=1, label="order-flow signal x (z)")
    fwd = np.full_like(mid, np.nan)
    h = int(5 / DT)
    fwd[:-h] = mid[h:] - mid[:-h]
    ax2 = ax.twinx()
    ax2.plot(t[sl], fwd[sl], color="black", lw=0.6, alpha=0.6, label="next 5 s mid move (USDT, right)")
    ax.axhline(0, color="grey", lw=0.6); ax.set_xlabel("time (s)"); ax.set_ylabel("x")
    ax.legend(fontsize=8, frameon=False, loc="upper left"); ax2.legend(fontsize=8, frameon=False, loc="upper right")
    fig.tight_layout(); fig.savefig(OUT / "sample_path_flow.png", dpi=130); plt.close(fig)
    log.info("wrote %s", OUT)


if __name__ == "__main__":
    main()
