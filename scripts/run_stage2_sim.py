"""Stage 2: naive vs Avellaneda-Stoikov with parameters calibrated to BTCUSDT.

For each volatility regime (low / medium / high calibration days: sigma, A, k, price level)
and each risk aversion gamma, run 1,000 paired paths (seeds 10,000-10,999, disjoint from
stage 1) of:
  * A-S (reservation price + optimal spread), quotes rounded to the 0.1 tick;
  * naive symmetric quotes whose (pre-rounding) spread equals A-S's average model spread,
    rounded the same way, so the two differ only in where the quotes are centred.

Horizon T = 3600 s, dt = 0.2 s, exact fill probability 1 - exp(-lambda dt) (asserted
A*dt < 0.1). One fill = 1 BTC; P&L in USDT, inventory marked at the final mid.
Fee scenarios (maker fee on notional): -0.5 (rebate), 0, 1, 2 bps. Fees only change cash,
so each run is simulated once and P&L under each fee is P&L(0) - fee * traded notional
(exact; tested in tests/test_simulator.py).

Writes results/stage2/: strategy_comparison.csv/.md, reality_check.md,
pnl_by_fee.png, pnl_by_fee_all_regimes.png

Run:  .venv\\Scripts\\python scripts\\run_stage2_sim.py
"""

from __future__ import annotations

import dataclasses
import json
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from mms.analysis.metrics import paired_difference, summarise
from mms.sim.simulator import MarketParams, simulate_chunked
from mms.strategies.avellaneda_stoikov import AvellanedaStoikov
from mms.strategies.naive import NaiveSymmetric

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "results" / "stage2"
T, DT, TICK = 3600.0, 0.2, 0.1
GAMMAS = [1e-5, 1e-4, 1e-3]
MAIN_GAMMA = 1e-4
FEES_BPS = [-0.5, 0.0, 1.0, 2.0]
SEEDS = range(10_000, 11_000)
N_BOOT = 2000
REAL_SPREAD = 0.1
log = logging.getLogger("stage2_sim")


def fmt(t, nd=1) -> str:
    return f"{t[0]:.{nd}f} [{t[1]:.{nd}f}, {t[2]:.{nd}f}]"


def run_pair(params: MarketParams, gamma: float):
    a_s = AvellanedaStoikov(gamma=gamma, sigma=params.sigma, k=params.k, T=params.T)
    model_spread = float(np.mean([a_s.spread(i * params.dt) for i in range(params.n_steps)]))
    r_as = simulate_chunked(a_s, params, SEEDS, chunk=250, tick_size=TICK)
    r_nv = simulate_chunked(NaiveSymmetric(model_spread, name="naive"), params, SEEDS, chunk=250, tick_size=TICK)
    return a_s, model_spread, r_as, r_nv


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    cal = json.loads((OUT / "calibration.json").read_text(encoding="utf-8"))
    rows, md, reality, store = [], [], [], {}

    for reg, rp in cal["regimes"].items():
        params = MarketParams(s0=round(rp["mean_mid"], 1), sigma=rp["sigma_1s"][0], T=T, dt=DT,
                              A=rp["A"][0], k=rp["k"][0], fill_rule="exact")
        log.info("%s: %s, A*dt = %.4f", reg, params, params.A * params.dt)
        for g in GAMMAS:
            a_s, model_spread, r_as, r_nv = run_pair(params, g)
            store[(reg, g)] = (r_as, r_nv)
            for fee in FEES_BPS:
                res = {"avellaneda_stoikov": dataclasses.replace(r_as, pnl=r_as.pnl_with_fee(fee), fee_bps=fee),
                       "naive": dataclasses.replace(r_nv, pnl=r_nv.pnl_with_fee(fee), fee_bps=fee)}
                for name, r in res.items():
                    s = summarise(r, N_BOOT, seed=1).set_index("metric")
                    row = {"regime": reg, "gamma": g, "fee_bps": fee, "strategy": name,
                           "sigma": params.sigma, "A": params.A, "k": params.k, "s0": params.s0,
                           "model_spread_avg": model_spread, "fills_per_hour": float(r.n_fills.mean()),
                           "n_clipped": r.n_clipped}
                    for m in s.index:
                        row[m], row[m + "_lo"], row[m + "_hi"] = s.loc[m, "estimate"], s.loc[m, "ci_low"], s.loc[m, "ci_high"]
                    rows.append(row)
                dm = paired_difference(res["avellaneda_stoikov"].pnl, res["naive"].pnl, np.mean, N_BOOT, seed=2)
                ds = paired_difference(res["avellaneda_stoikov"].pnl, res["naive"].pnl, lambda x: np.std(x, ddof=1), N_BOOT, seed=2)
                rows.append({"regime": reg, "gamma": g, "fee_bps": fee, "strategy": "A-S minus naive (paired)",
                             "mean_pnl": dm[0], "mean_pnl_lo": dm[1], "mean_pnl_hi": dm[2],
                             "std_pnl": ds[0], "std_pnl_lo": ds[1], "std_pnl_hi": ds[2]})
            half = model_spread / 2
            reality.append({
                "regime": reg, "gamma": g, "sigma": params.sigma, "k": params.k, "s0": params.s0,
                "liquidity_term_2_over_k": 2 / params.k,
                "as_spread_t0": a_s.spread(0.0), "as_spread_avg_model": model_spread,
                "as_spread_avg_quoted": float(r_as.mean_quoted_spread.mean()),
                "ratio_to_real_spread": float(r_as.mean_quoted_spread.mean()) / REAL_SPREAD,
                "reservation_shift_per_btc_t0": g * params.sigma**2 * T,
                "fills_per_hour_as": float(r_as.n_fills.mean()),
                "fee_per_fill_1bp": params.s0 * 1e-4, "fee_per_fill_2bp": params.s0 * 2e-4,
                "half_spread_avg": half,
                "fee_aware_half_spread_2bp": 1 / params.k + params.s0 * 2e-4,
            })
            log.info("%s gamma=%g done: A-S spread %.2f, fills/h %.0f", reg, g, model_spread, r_as.n_fills.mean())

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "strategy_comparison.csv", index=False)

    # Markdown: one table per regime, rows = gamma x fee x strategy
    md = ["# Stage 2: naive vs A–S with calibrated BTCUSDT parameters\n",
          f"T = {T:.0f} s, dt = {DT} s, exact fill rule, tick {TICK}, 1 fill = 1 BTC, {len(SEEDS)} paired paths "
          f"(seeds {SEEDS.start}–{SEEDS.stop - 1}). P&L in USDT per hour. Cells: estimate [95% bootstrap CI]. "
          "Naive uses A–S's average model spread; both rounded outward to the tick.\n"]
    for reg in cal["regimes"]:
        rp = cal["regimes"][reg]
        md.append(f"\n## {reg} volatility: σ = {rp['sigma_1s'][0]:.3f} USDT/√s, A = {rp['A'][0]:.4f}/s, "
                  f"k = {rp['k'][0]:.3f}/USDT, S0 = {rp['mean_mid']:.0f}\n")
        md.append("| γ | fee (bp) | strategy | avg quoted spread | mean P&L | std P&L | mean/std | std q_T | mean max|q| | fills/h |")
        md.append("|---|---|---|---|---|---|---|---|---|---|")
        for _, r in df[df.regime == reg].iterrows():
            if r.strategy.startswith("A-S minus"):
                md.append(f"| {r.gamma:g} | {r.fee_bps:+g} | **A–S − naive** | | {fmt((r.mean_pnl, r.mean_pnl_lo, r.mean_pnl_hi))} | "
                          f"{fmt((r.std_pnl, r.std_pnl_lo, r.std_pnl_hi))} | | | | |")
            else:
                md.append(f"| {r.gamma:g} | {r.fee_bps:+g} | {r.strategy} | {r.mean_quoted_spread:.2f} | "
                          f"{fmt((r.mean_pnl, r.mean_pnl_lo, r.mean_pnl_hi))} | {fmt((r.std_pnl, r.std_pnl_lo, r.std_pnl_hi))} | "
                          f"{fmt((r.sharpe_like, r.sharpe_like_lo, r.sharpe_like_hi), 2)} | {r.std_final_q:.1f} | "
                          f"{r.mean_max_abs_q:.1f} | {r.fills_per_hour:.0f} |")
    (OUT / "strategy_comparison.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    rc = pd.DataFrame(reality)
    rc.to_csv(OUT / "reality_check.csv", index=False)
    lines = ["# Reality check: A–S optimal spread vs the real BTCUSDT spread\n",
             f"Real spread: median 0.1 USDT (1 tick) on every processed day. 1 fill = 1 BTC.\n",
             "| regime | γ | 2/k | A–S spread at t=0 | A–S avg quoted spread | × real spread | reservation shift per BTC at t=0 | "
             "fills/h | fee per fill at 1 / 2 bp | fee-aware half-spread 1/k + fee (2 bp) |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for _, r in rc.iterrows():
        lines.append(f"| {r.regime} | {r.gamma:g} | {r.liquidity_term_2_over_k:.2f} | {r.as_spread_t0:.2f} | "
                     f"{r.as_spread_avg_quoted:.2f} | {r.ratio_to_real_spread:.0f}× | {r.reservation_shift_per_btc_t0:.2f} | "
                     f"{r.fills_per_hour_as:.0f} | {r.fee_per_fill_1bp:.2f} / {r.fee_per_fill_2bp:.2f} | {r.fee_aware_half_spread_2bp:.2f} |")
    (OUT / "reality_check.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # Charts: P&L distributions by fee scenario
    def hist_panel(ax, r_as, r_nv, fee, title):
        a, n = r_as.pnl_with_fee(fee), r_nv.pnl_with_fee(fee)
        bins = np.linspace(min(a.min(), n.min()), max(a.max(), n.max()), 50)
        ax.hist(n, bins=bins, alpha=0.55, color="#ff7f0e", label="naive (same avg spread)")
        ax.hist(a, bins=bins, alpha=0.55, color="#1f77b4", label="Avellaneda–Stoikov")
        ax.axvline(0, color="black", lw=0.7)
        ax.set_title(title, fontsize=9)

    fig, axes = plt.subplots(1, 4, figsize=(18, 4))
    r_as, r_nv = store[("medium", MAIN_GAMMA)]
    for ax, fee in zip(axes, FEES_BPS):
        hist_panel(ax, r_as, r_nv, fee, f"maker fee {fee:+g} bp")
        ax.set_xlabel("P&L over 1 h (USDT)")
    axes[0].set_ylabel("number of paths"); axes[0].legend(fontsize=8, frameon=False)
    fig.suptitle(f"Medium-volatility parameters, γ = {MAIN_GAMMA:g}: P&L distribution by fee scenario (1,000 paired paths)")
    fig.tight_layout(); fig.savefig(OUT / "pnl_by_fee.png", dpi=130); plt.close(fig)

    fig, axes = plt.subplots(3, 4, figsize=(18, 11))
    for i, reg in enumerate(cal["regimes"]):
        r_as, r_nv = store[(reg, MAIN_GAMMA)]
        for j, fee in enumerate(FEES_BPS):
            hist_panel(axes[i, j], r_as, r_nv, fee, f"{reg} vol, fee {fee:+g} bp")
    axes[0, 0].legend(fontsize=8, frameon=False)
    for ax in axes[-1]:
        ax.set_xlabel("P&L over 1 h (USDT)")
    fig.suptitle(f"P&L distributions, γ = {MAIN_GAMMA:g}, by volatility regime and fee")
    fig.tight_layout(); fig.savefig(OUT / "pnl_by_fee_all_regimes.png", dpi=130); plt.close(fig)
    log.info("wrote %s", OUT)


if __name__ == "__main__":
    main()
