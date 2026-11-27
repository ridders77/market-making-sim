"""Stage 6: one summary figure for the top of the README (results/summary.png), built only from cached results.

No simulation or replay is run here: every number is read from the CSVs written by the stage 1-5 scripts.

Panels (the story in order):
  1. Theory: the A-S replication matches the paper (std of P&L, ours vs paper, 3 risk aversions).
  2. Synthetic vs replay: the same wide-quoting strategy, predicted profitable by the calibrated synthetic
     model, loses money when replayed on the real book (adverse selection ~1.5x larger than modelled).
  3. Touch strategies in the replay: all lose, even with the -0.5 bp rebate (108 of 108 scenarios).
  4. What-if: P&L of always-at-touch against queue priority f (rebate case) - break-even between 0.15 and 0.1.
  5. Execution: cost of buying 0.5 BTC within 1 minute - RL matches rest-at-touch, ~2 bp below taking.

Run:  .venv\\Scripts\\python scripts\\make_summary_figure.py
"""

from __future__ import annotations

import ast
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
RES = REPO / "results"
log = logging.getLogger("summary")

GREY, BLUE, ORANGE, RED, GREEN = "#7f7f7f", "#1f77b4", "#ff7f0e", "#d62728", "#2ca02c"


def _tuple(s: str) -> tuple[float, float, float]:
    """The validation CSV stores (estimate, lo, hi) as text like '(1.2, np.float64(1.1), np.float64(1.3))'."""
    return tuple(float(x) for x in ast.literal_eval(s.replace("np.float64", "")))


def touch_scenarios_all_negative() -> tuple[int, int]:
    """(number of touch scenarios, number whose 95% CI upper bound is below 0): 4 strategies x 3 cancel rules x
    3 latencies x 3 fees, from results/stage4/latency_sensitivity.csv."""
    t = pd.read_csv(RES / "stage4" / "latency_sensitivity.csv")
    t = t[t.strategy.str.startswith("touch")]
    return len(t), int((t.hi < 0).sum())


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    fig = plt.figure(figsize=(17.5, 9.6))
    gs = fig.add_gridspec(2, 6, hspace=0.55, wspace=1.6)
    axes = [fig.add_subplot(gs[0, 0:2]), fig.add_subplot(gs[0, 2:4]), fig.add_subplot(gs[0, 4:6]),
            fig.add_subplot(gs[1, 0:3]), fig.add_subplot(gs[1, 3:6])]

    # 1. Replication ------------------------------------------------------------------------------------
    ax = axes[0]
    r = pd.read_csv(RES / "stage1" / "replication.csv")
    gam = [0.01, 0.1, 1.0]
    x = np.arange(len(gam))
    for j, (strat, col, lab) in enumerate([("avellaneda_stoikov", BLUE, "A–S (ours)"),
                                           ("symmetric", ORANGE, "symmetric, same avg spread (ours)")]):
        s = r[r.strategy == strat].set_index("gamma").loc[gam]
        xx = x + (j - 0.5) * 0.36
        ax.bar(xx, s.std_pnl, 0.34, color=col, label=lab,
               yerr=[s.std_pnl - s.std_pnl_lo, s.std_pnl_hi - s.std_pnl], capsize=3)
        ax.scatter(xx, s.paper_std_pnl, marker="_", s=260, color="black", zorder=5,
                   label="paper (Avellaneda & Stoikov 2008)" if j == 0 else None)
    ax.set_xticks(x, [f"γ = {g:g}" for g in gam])
    ax.set_ylabel("std of final P&L (paper units)")
    ax.set_title("1. Theory: A–S replication matches the paper\n(same mean P&L, about half the risk)", fontsize=10)
    ax.legend(fontsize=7.5, frameon=False, loc="upper right")
    ax.set_ylim(0, 22)

    # 2. Synthetic vs replay ----------------------------------------------------------------------------
    ax = axes[1]
    v = pd.read_csv(RES / "stage4" / "validation_replay_vs_synthetic.csv")
    v = v[(v.rule == "proportional") & (v.unseen_depth == "back") & (v.fee.isin([-0.5, 0.0]))].set_index("fee")
    fees = [-0.5, 0.0]
    x = np.arange(2)
    ax.bar(x - 0.18, v.loc[fees, "synthetic_pnl_h"], 0.34, color=GREY, label="calibrated synthetic model (stage 3)")
    ax.bar(x + 0.18, v.loc[fees, "replay_pnl_h"], 0.34, color=RED, label="replay on the real book (stage 4)")
    for i, f in enumerate(fees):
        mk = _tuple(v.loc[f, "mk60_ratio"])
        ax.text(i, max(v.loc[f, "synthetic_pnl_h"], 0) + 0.6, f"markout ×{mk[0]:.2f}\n[{mk[1]:.2f}, {mk[2]:.2f}]",
                ha="center", fontsize=7.5)
    ax.axhline(0, color="black", lw=0.7)
    ax.set_xticks(x, ["maker fee −0.5 bp\n(rebate)", "maker fee 0 bp"])
    ax.set_ylabel("mean P&L (USDT per hour, 0.01 BTC lot)")
    ax.set_title("2–3. Quoting wide: the model says profit,\nthe replay says loss (230 test hours)", fontsize=10)
    ax.legend(fontsize=7.5, frameon=False, loc="lower left")
    ax.set_ylim(-5, 11.5)

    # 3. Touch strategies --------------------------------------------------------------------------------
    ax = axes[2]
    t = pd.read_csv(RES / "stage4" / "touch_strategy_table.csv")
    t = t[(t.rule == "proportional") & (t.fee_bps == -0.5)].set_index("strategy")
    order = [("touch", "always at touch"), ("touch_skew", "+ inventory skew"), ("touch_queue", "+ queue-aware"),
             ("touch_ofi", "+ order-flow aware"), ("markout_as", "quote wide (A–S)")]
    y = np.arange(len(order))[::-1]
    m = t.loc[[k for k, _ in order]]
    ax.barh(y, m.mean_pnl, color=[GREY, BLUE, GREEN, RED, ORANGE],
            xerr=[m.mean_pnl - m.lo, m.hi - m.mean_pnl], capsize=3)
    ax.set_yticks(y, [lab for _, lab in order], fontsize=8.5)
    ax.axvline(0, color="black", lw=0.7)
    n, neg = touch_scenarios_all_negative()
    ax.set_xlabel("mean P&L (USDT per hour), 95% CI")
    ax.set_title(f"4. Replay with a queue model: nothing profits\nat the touch ({neg}/{n} scenarios lose, CI < 0)",
                 fontsize=10)
    ax.text(0.02, 0.02, "best case shown: −0.5 bp rebate,\n100 ms latency, proportional cancels",
            transform=ax.transAxes, fontsize=7.5, color="dimgray")

    # 4. Queue what-if -----------------------------------------------------------------------------------
    ax = axes[3]
    q = pd.read_csv(RES / "stage5" / "queue_whatif.csv")
    for strat, col, lab in [("touch", GREY, "always at touch"), ("touch_skew", BLUE, "touch + inventory skew")]:
        s = q[(q.strategy == strat) & (q.fee_bps == -0.5)].sort_values("f", ascending=False)
        xs = np.arange(len(s))
        ax.errorbar(xs, s.mean_pnl, yerr=[s.mean_pnl - s.lo, s.hi - s.mean_pnl], marker="o", color=col,
                    capsize=3, label=lab)
        ticks = [f"{f:g}" for f in s.f]
    ax.set_xticks(xs, ticks)
    ax.set_yscale("symlog", linthresh=10)
    ax.axhline(0, color="black", lw=0.7)
    i15, i10 = ticks.index("0.15"), ticks.index("0.1")
    ax.axvspan(i15, i10, color=GREEN, alpha=0.15, label="break-even bracket (0.15 → 0.1)")
    ax.set_xlabel("f = share of the displayed queue ahead of us when we join (1 = back, 0 = front)")
    ax.set_ylabel("mean P&L (USDT/h, symlog), 95% CI")
    ax.set_title("WHAT-IF (not achievable for a retail participant): queue priority needed to profit\n"
                 "with the −0.5 bp rebate, you need less than ~1/10 of the queue ahead of you", fontsize=10)
    ax.legend(fontsize=8, frameon=False, loc="upper left")

    # 5. Execution -------------------------------------------------------------------------------------
    ax = axes[4]
    e = pd.read_csv(RES / "stage5" / "execution_cost.csv")
    e = e[(e.H == 60.0) & (e.rule == "proportional") & (e.side == "buy")]
    strats = [("fixed_cross", "immediate taker"), ("twap", "TWAP taker slices"),
              ("fixed_touch", "rest at touch, cross at deadline"), ("rl_sigma", "RL agent (pre-registered headline)")]
    cols = [GREY, GREY, GREEN, BLUE]
    y = np.arange(len(strats))[::-1]
    for j, (fee, lab, hatch) in enumerate([("retail", "retail fees (maker 2 / taker 5 bp)", None),
                                           ("pro", "professional fees (maker −0.5 / taker 2 bp, assumed)", "//")]):
        s = e[e.fee == fee].set_index("strategy").loc[[k for k, _ in strats]]
        yy = y + (0.2 if j == 0 else -0.2)
        ax.barh(yy, s["mean"], 0.38, color=cols, hatch=hatch, edgecolor="white",
                xerr=[s["mean"] - s.lo, s.hi - s["mean"]], capsize=2, label=lab)
        for yi, val in zip(yy, s["mean"]):
            ax.text(val + 0.1, yi, f"{val:.2f}", va="center", fontsize=7.5)
    ax.set_yticks(y, [lab for _, lab in strats], fontsize=8.5)
    ax.set_xlabel("implementation shortfall (bps of notional, fees included), 95% CI")
    ax.set_title("5. RL for execution (buy 0.5 BTC within 1 min): RL matches the simple rule\n"
                 "and saves ~2 bp against taking liquidity (6,890 test episodes)", fontsize=10)
    ax.legend(fontsize=7.5, frameon=False, loc="lower right")
    ax.set_xlim(0, 6.3)

    fig.suptitle("Market making and execution on real BTCUSDT futures data (2023–24): the study at a glance",
                 fontsize=13, y=0.995)
    out = RES / "summary.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    log.info("wrote %s (touch scenarios: %d of %d lose with CI < 0)", out, neg, n)


if __name__ == "__main__":
    main()
