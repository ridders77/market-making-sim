"""Replicate Avellaneda & Stoikov (2008), section 4 (numerical simulations).

Setup: S0 = 100, T = 1, sigma = 2, dt = 0.005, A = 140, k = 1.5,
gamma in {0.01, 0.1, 1}, 1,000 paths each (seeds 0..999, shared by both
strategies so the comparison is paired). No fees, no tick size (as in the paper).

Benchmark: a symmetric strategy whose fixed spread equals the A-S spread
averaged over the run. A second benchmark uses only the liquidity term
(2/gamma) ln(1 + gamma/k), to show how much the benchmark definition matters.
Fills use the paper's rule P(fill in dt) = lambda * dt (stage 1 only).

Writes to results/stage1/: replication.csv, replication.md, sample_path.png,
pnl_distributions.png, inventory_distributions.png.

Run:  .venv\\Scripts\\python scripts\\run_replication.py
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from mms.analysis.charts import plot_distributions, plot_sample_path
from mms.analysis.metrics import paired_difference, summarise
from mms.sim.simulator import MarketParams, simulate
from mms.strategies.avellaneda_stoikov import AvellanedaStoikov
from mms.strategies.naive import NaiveSymmetric

OUT = Path(__file__).resolve().parents[1] / "results" / "stage1"
PARAMS = MarketParams(s0=100.0, sigma=2.0, T=1.0, dt=0.005, A=140.0, k=1.5)
GAMMAS = [0.01, 0.1, 1.0]
SEEDS = range(1000)
N_BOOT = 2000

# Avellaneda & Stoikov (2008), Quantitative Finance 8(3), section 4, tables for
# gamma = 0.1, 0.01 and 1, checked against the paper by the owner (2026-11-26).
# Values: (average spread, mean profit, std profit, std final q).
# The paper reports one average spread per gamma, used by both strategies.
PAPER = {
    (0.1, "avellaneda_stoikov"): (1.49, 65.0, 6.6, 2.9),
    (0.1, "symmetric"): (1.49, 68.4, 12.7, 8.4),
    (0.01, "avellaneda_stoikov"): (1.35, 68.6, 8.7, 5.1),
    (0.01, "symmetric"): (1.35, 68.8, 12.8, 8.7),
    (1.0, "avellaneda_stoikov"): (3.02, 31.4, 5.0, 1.7),
    (1.0, "symmetric"): (3.02, 44.0, 11.0, 5.1),
}

log = logging.getLogger("replication")


def fmt(est: float, lo: float, hi: float, nd: int = 2) -> str:
    return f"{est:.{nd}f} [{lo:.{nd}f}, {hi:.{nd}f}]"


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    OUT.mkdir(parents=True, exist_ok=True)

    rows, md_rows, results = [], [], {}
    for g in GAMMAS:
        a_s = AvellanedaStoikov(gamma=g, sigma=PARAMS.sigma, k=PARAMS.k, T=PARAMS.T)
        r_as = simulate(a_s, PARAMS, SEEDS, record_paths=(g == 0.1))
        avg_spread = float(r_as.mean_quoted_spread.mean())
        liq_spread = (2 / g) * np.log1p(g / PARAMS.k)
        r_sym = simulate(NaiveSymmetric(avg_spread), PARAMS, SEEDS)
        r_sym_liq = simulate(NaiveSymmetric(liq_spread, name="symmetric_liquidity_spread"), PARAMS, SEEDS)
        results[g] = {"avellaneda_stoikov": r_as, "symmetric": r_sym}

        for name, res in [("avellaneda_stoikov", r_as), ("symmetric", r_sym),
                          ("symmetric_liquidity_spread", r_sym_liq)]:
            s = summarise(res, N_BOOT, seed=1).set_index("metric")
            row = {"gamma": g, "strategy": name, "n_paths": len(res.pnl), "n_clipped": res.n_clipped}
            for m in s.index:
                row[m] = s.loc[m, "estimate"]
                row[m + "_lo"] = s.loc[m, "ci_low"]
                row[m + "_hi"] = s.loc[m, "ci_high"]
            paper = PAPER.get((g, name))
            if paper:
                row.update(paper_avg_spread=paper[0], paper_mean_pnl=paper[1], paper_std_pnl=paper[2],
                           paper_std_q=paper[3])
            rows.append(row)
            md_rows.append(
                f"| {g} | {name} | {row['mean_quoted_spread']:.3f} | "
                f"{fmt(row['mean_pnl'], row['mean_pnl_lo'], row['mean_pnl_hi'])} | "
                f"{fmt(row['std_pnl'], row['std_pnl_lo'], row['std_pnl_hi'])} | "
                f"{fmt(row['sharpe_like'], row['sharpe_like_lo'], row['sharpe_like_hi'])} | "
                f"{fmt(row['mean_final_q'], row['mean_final_q_lo'], row['mean_final_q_hi'])} | "
                f"{fmt(row['std_final_q'], row['std_final_q_lo'], row['std_final_q_hi'])} | "
                f"{row['mean_max_abs_q']:.2f} | {row['fill_rate']:.3f} | {row['avg_spread_captured']:.3f} | "
                f"{res.n_clipped} | "
                + (f"{paper[0]} / {paper[1]} / {paper[2]} / {paper[3]}" if paper else "—") + " |"
            )

        # Paired differences A-S minus symmetric (same seeds)
        d_mean = paired_difference(r_as.pnl, r_sym.pnl, np.mean, N_BOOT, seed=2)
        d_std = paired_difference(r_as.pnl, r_sym.pnl, lambda x: np.std(x, ddof=1), N_BOOT, seed=2)
        log.info("gamma=%s  A-S minus symmetric: mean P&L %s, std P&L %s", g, fmt(*d_mean), fmt(*d_std))
        md_rows.append(f"| {g} | **A–S − symmetric (paired)** | | {fmt(*d_mean)} | {fmt(*d_std)} | | | | | | | | |")

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "replication.csv", index=False)

    header = (
        "# Stage 1: Avellaneda–Stoikov replication\n\n"
        f"S0=100, T=1, σ=2, dt=0.005, A=140, k=1.5; {len(SEEDS)} paired paths per γ (seeds 0–999); "
        "no fees, no tick. Cells show estimate [95% bootstrap CI]. "
        "`symmetric` uses the mean A–S quoted spread; `symmetric_liquidity_spread` uses only the liquidity term (2/γ)ln(1+γ/k). "
        "Paper figures: Avellaneda & Stoikov (2008), section 4 tables. `clipped` = quote-steps with λ·dt > 1.\n\n"
        "| γ | strategy | avg spread | mean P&L | std P&L | mean/std | mean q_T | std q_T | "
        "mean max|q| | fill rate | spread captured | clipped | paper avg spread / mean / std P&L / std q |\n"
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|\n"
    )
    (OUT / "replication.md").write_text(header + "\n".join(md_rows) + "\n", encoding="utf-8")

    # Charts
    r01 = results[0.1]["avellaneda_stoikov"]
    plot_sample_path(r01, path=0, title="A–S, γ = 0.1, seed 0: mid, quotes and inventory",
                     out=OUT / "sample_path.png")
    plot_distributions(results, "pnl", "final P&L", "Final P&L, 1,000 paired paths",
                       OUT / "pnl_distributions.png")
    plot_distributions(results, "q", "final inventory q_T", "Final inventory, 1,000 paired paths",
                       OUT / "inventory_distributions.png", discrete=True)
    log.info("wrote results to %s", OUT)


if __name__ == "__main__":
    main()
