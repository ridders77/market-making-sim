"""Stage 3, Step 0: realistic lot sizes and fee-aware A-S; rerun of the stage 2 table.

1. Re-estimate lambda(delta) = A exp(-k delta) on the calibration days with the lot rule (a window
   is a fill at delta only if >= 1 lot of aggressor volume printed at or through delta), for
   lots of 0.01 BTC (main), 0.1 BTC (sensitivity) and 1 BTC (the stage 2 lot, for reference),
   next to the stage 2 "any print" rule. Pooled and per volatility regime, bootstrap CIs over days.
2. Rerun the stage 2 comparison with sigma_60s (owner decision), the lot and the lot-rule A, k:
   naive (A-S's average spread), A-S, fee-aware A-S (half-spread + c S), fees -0.5 / 0 / 1 / 2 bp.
3. Decompose how much P&L per hour falls from stage 2 to the realistic setup.

Seeds 10,000-10,999 (the stage 2 seeds, so the rerun is paired with the stage 2 table).
Writes results/stage3/step0_*.{csv,md}.

Run:  .venv\\Scripts\\python scripts\\run_stage3_step0.py
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from mms.analysis.metrics import paired_difference, summary_row
from mms.calibration.estimators import bootstrap_days, intensity_from_fraction
from mms.calibration.windows import day_hits, load_windows
from mms.sim.env import EnvConfig, run_env
from mms.sim.simulator import MarketParams
from mms.strategies.avellaneda_stoikov import AvellanedaStoikov, FeeAwareAvellanedaStoikov
from mms.strategies.naive import NaiveSymmetric

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "results" / "stage3"
DELTAS = np.round(np.arange(0, 81) * 0.1, 10)
FIT_LO, FIT_HI = 0.1, 5.0
RULES = {"any": "any print (stage 2)", "l001": "0.01 BTC lot", "l01": "0.1 BTC lot", "l1": "1 BTC lot"}
T, DT, TICK, GAMMA = 3600.0, 0.2, 0.1, 1e-4
FEES = [-0.5, 0.0, 1.0, 2.0]
SEEDS = range(10_000, 11_000)
N_BOOT = 2000
log = logging.getLogger("stage3_step0")


def fit_rule(cache, days, rule, n_boot=1000, seed=0) -> dict:
    per_day = {}
    for d, (h, n) in day_hits(days, rule, DELTAS, cache=cache).items():
        h = h.copy()
        h[(DELTAS < FIT_LO - 1e-9) | (DELTAS > FIT_HI + 1e-9)] = 0   # only the fit range enters the fit
        per_day[d] = (h, n)
    f = bootstrap_days(per_day, DELTAS, 1.0, n_boot=n_boot, seed=seed)
    H = sum(v[0] for v in day_hits(days, rule, DELTAS, cache=cache).values())
    N = sum(v[1] for v in day_hits(days, rule, DELTAS, cache=cache).values())
    return {"A": f["A"], "k": f["k"], "r2": f["r2"], "lambda_touch": float(intensity_from_fraction(H[0] / N, 1.0))}


def fmt(t, nd=2) -> str:
    return f"{t[0]:.{nd}f} [{t[1]:.{nd}f}, {t[2]:.{nd}f}]"


def run_setup(reg, s0, sigma, A, k, lot, fees, label) -> tuple[list[dict], dict]:
    """naive / A-S / fee-aware A-S for one parameter set; returns summary rows and the raw results."""
    p = MarketParams(s0=s0, sigma=sigma, T=T, dt=DT, A=A, k=k, fill_rule="exact")
    a_s = AvellanedaStoikov(gamma=GAMMA, sigma=sigma, k=k, T=T)
    model_spread = float(np.mean([a_s.spread(i * DT) for i in range(p.n_steps)]))
    cfg0 = EnvConfig(p, lot=lot, tick_size=TICK, maker_fee_bps=0.0)
    res = {"naive": run_env(NaiveSymmetric(model_spread, name="naive"), cfg0, SEEDS, chunk=500),
           "avellaneda_stoikov": run_env(a_s, cfg0, SEEDS, chunk=500)}
    rows = []
    for fee in fees:
        fa = FeeAwareAvellanedaStoikov(gamma=GAMMA, sigma=sigma, k=k, T=T, fee_bps=fee)
        res[("fee_aware_as", fee)] = run_env(fa, EnvConfig(p, lot=lot, tick_size=TICK, maker_fee_bps=fee), SEEDS, chunk=500)
        for name in ("naive", "avellaneda_stoikov", "fee_aware_as"):
            r = res[(name, fee)] if name == "fee_aware_as" else res[name]
            s = summary_row(r, fee_bps=fee, n_boot=N_BOOT, seed=1)
            rows.append({"setup": label, "regime": reg, "lot": lot, "fee_bps": fee, "strategy": name,
                         "sigma": sigma, "A": A, "k": k, "s0": s0, "model_spread_avg": model_spread, **s})
        pa = res[("fee_aware_as", fee)].pnl_with_fee(fee)
        pb = res["avellaneda_stoikov"].pnl_with_fee(fee)
        d = paired_difference(pa, pb, np.mean, N_BOOT, seed=2)
        rows.append({"setup": label, "regime": reg, "lot": lot, "fee_bps": fee, "strategy": "fee-aware minus A-S (paired)",
                     "mean_pnl": d[0], "mean_pnl_lo": d[1], "mean_pnl_hi": d[2]})
    return rows, res


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    OUT.mkdir(parents=True, exist_ok=True)
    cal2 = json.loads((REPO / "results" / "stage2" / "calibration.json").read_text(encoding="utf-8"))
    cal_days = cal2["calibration_days"]
    cache = {d: load_windows(d) for d in cal_days}

    # ---------------------------------------------------------------- 1. lambda by lot rule
    lam_rows, fits = [], {}
    for rule in RULES:
        groups = {"pooled": cal_days, **{r: v["days"] for r, v in cal2["regimes"].items()}}
        for g, days in groups.items():
            f = fit_rule(cache, days, rule)
            fits[(rule, g)] = f
            lam_rows.append({"rule": rule, "group": g, "A": f["A"][0], "A_lo": f["A"][1], "A_hi": f["A"][2],
                             "k": f["k"][0], "k_lo": f["k"][1], "k_hi": f["k"][2], "r2": f["r2"],
                             "lambda_touch": f["lambda_touch"]})
            log.info("%s %s: A=%.4f k=%.4f", rule, g, f["A"][0], f["k"][0])
    lam = pd.DataFrame(lam_rows)
    lam.to_csv(OUT / "step0_lambda_by_lot.csv", index=False)
    base = lam[lam.rule == "any"].set_index("group")
    L = ["# Step 0: fill intensity under the lot rule (10 calibration days, Δt = 1 s, δ ∈ [0.1, 5] USDT)\n",
         "A window counts as a fill at δ only if at least one lot of aggressor volume printed at or through δ. "
         "Change vs the stage 2 'any print' rule in brackets. CIs: bootstrap over days.\n",
         "| rule | group | A (/s) | k (/USDT) | R² | ΔA | Δk | λ at the touch (/s) |", "|---|---|---|---|---|---|---|---|"]
    for _, r in lam.iterrows():
        b = base.loc[r.group]
        L.append(f"| {RULES[r.rule]} | {r.group} | {fmt((r.A, r.A_lo, r.A_hi), 4)} | {fmt((r.k, r.k_lo, r.k_hi), 3)} | "
                 f"{r.r2:.4f} | {100 * (r.A / b.A - 1):+.1f}% | {100 * (r.k / b.k - 1):+.1f}% | {r.lambda_touch:.3f} |")
    (OUT / "step0_lambda_by_lot.md").write_text("\n".join(L) + "\n", encoding="utf-8")

    # ---------------------------------------------------------------- 2./3. simulations
    rows, keep = [], {}
    for reg, rp in cal2["regimes"].items():
        s0 = round(rp["mean_mid"], 1)
        s1, s60 = rp["sigma_1s"][0], rp["sigma_60s"][0]
        A = lambda rule: fits[(rule, reg)]["A"][0]
        k = lambda rule: fits[(rule, reg)]["k"][0]
        setups = [
            ("1. stage 2 as published: 1 BTC lot, σ_1s, any-print λ", s1, A("any"), k("any"), 1.0, [0.0]),
            ("2. σ_60s", s60, A("any"), k("any"), 1.0, [0.0]),
            ("3. 0.01 BTC lot, any-print λ", s60, A("any"), k("any"), 0.01, [0.0]),
            ("4. 0.01 BTC lot, 0.01-lot λ (main)", s60, A("l001"), k("l001"), 0.01, FEES),
            ("5. 0.1 BTC lot, 0.1-lot λ (sensitivity)", s60, A("l01"), k("l01"), 0.1, FEES),
            ("6. 1 BTC lot, 1-BTC-lot λ (stage 2 lot, honest fills)", s60, A("l1"), k("l1"), 1.0, [0.0]),
        ]
        for label, sig, a, kk, lot, fees in setups:
            r, res = run_setup(reg, s0, sig, a, kk, lot, fees, label)
            rows += r
            keep[(reg, label)] = res
            log.info("%s | %s done", reg, label)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "step0_strategy_table.csv", index=False)

    # Decomposition table (A-S and naive at 0 fee)
    D = ["# Step 0: how P&L per hour changes from stage 2 to realistic fills\n",
         f"A–S (γ = {GAMMA:g} per lot) and naive (A–S's average spread), fee 0, T = 1 h, 1,000 paired paths "
         f"(stage 2 seeds {SEEDS.start}–{SEEDS.stop - 1}). P&L in USDT per hour [95% CI], and per unit traded notional in bps.\n"]
    for reg in cal2["regimes"]:
        D += [f"\n## {reg} volatility\n", "| setup | strategy | A | k | σ | mean P&L (USDT/h) | P&L (bps of notional) | fills/h | notional/h (USDT) |",
              "|---|---|---|---|---|---|---|---|---|"]
        sub = df[(df.regime == reg) & (df.fee_bps == 0.0) & df.strategy.isin(["avellaneda_stoikov", "naive"])]
        for _, r in sub.iterrows():
            D.append(f"| {r.setup} | {r.strategy} | {r.A:.4f} | {r.k:.3f} | {r.sigma:.3f} | {fmt((r.mean_pnl, r.mean_pnl_lo, r.mean_pnl_hi))} | "
                     f"{fmt((r.pnl_bps, r.pnl_bps_lo, r.pnl_bps_hi), 3)} | {r.fills_per_episode:.0f} | {r.notional_per_episode:,.0f} |")
    (OUT / "step0_pnl_decomposition.md").write_text("\n".join(D) + "\n", encoding="utf-8")

    # Main table (lot 0.01 and 0.1, all fees)
    M = ["# Step 0: rerun of the stage 2 table with realistic lots, σ_60s and fee-aware A–S\n",
         f"γ = {GAMMA:g} per lot, T = 1 h, dt = {DT} s, tick {TICK}, 1,000 paired paths. P&L per hour in USDT and in bps of "
         "traded notional, [95% bootstrap CI]. Fee-aware A–S adds c·S to each half-spread.\n"]
    for label in ("4. 0.01 BTC lot, 0.01-lot λ (main)", "5. 0.1 BTC lot, 0.1-lot λ (sensitivity)"):
        for reg in cal2["regimes"]:
            sub = df[(df.setup == label) & (df.regime == reg)]
            r0 = sub.iloc[0]
            M += [f"\n## {label}, {reg} volatility (σ = {r0.sigma:.3f} USDT/√s, A = {r0.A:.4f}/s, k = {r0.k:.3f}/USDT, S₀ = {r0.s0:.0f})\n",
                  "| fee (bp) | strategy | avg quoted spread | mean P&L (USDT/h) | P&L (bps of notional) | std P&L | mean/std | std q_T (BTC) | fills/h |",
                  "|---|---|---|---|---|---|---|---|---|"]
            for _, r in sub.iterrows():
                if r.strategy.startswith("fee-aware minus"):
                    M.append(f"| {r.fee_bps:+g} | **fee-aware − A–S** | | {fmt((r.mean_pnl, r.mean_pnl_lo, r.mean_pnl_hi))} | | | | | |")
                else:
                    M.append(f"| {r.fee_bps:+g} | {r.strategy} | {r.mean_quoted_spread:.2f} | {fmt((r.mean_pnl, r.mean_pnl_lo, r.mean_pnl_hi))} | "
                             f"{fmt((r.pnl_bps, r.pnl_bps_lo, r.pnl_bps_hi), 3)} | {r.std_pnl:.2f} | {r.sharpe_like:.2f} | "
                             f"{r.std_final_q_btc:.3f} | {r.fills_per_episode:.0f} |")
    (OUT / "step0_strategy_table.md").write_text("\n".join(M) + "\n", encoding="utf-8")
    log.info("wrote %s", OUT)


if __name__ == "__main__":
    main()
