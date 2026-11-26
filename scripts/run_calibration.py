"""Stage 2 calibration: sigma, A, k from the 20 processed BTCUSDT days.

Reads data/processed/ (built by build_dataset.py) and results/stage2/selected_days.csv.
Main settings (see DECISIONS.md): touch windows Dt = 1 s (also 2 s and 5 s as a check),
delta grid 0..8 USDT in 0.1 ticks, exponential fit over delta in [0.1, 5.0] USDT
(one tick beyond the touch to 50 ticks). Calibration = the 10 earlier days,
evaluation = the 10 later days (out of time).

Writes to results/stage2/:
  day_stats.csv           sigma (1 s, 10 s, 60 s), price level, spread, ABM check, per-day A/k fits
  sigma_by_hour.csv       intraday sigma profile (per UTC hour, per day)
  touch_counts.csv        hits per (day, side, window, delta) -- everything the fits need
  calibration.json        pooled + per-regime parameters with bootstrap CIs (used by run_stage2_sim.py)
  calibration.md          human-readable tables
  fit_calibration.png, oos_fit.png, parameter_stability.png, window_sensitivity.png

Run:  .venv\\Scripts\\python scripts\\run_calibration.py
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from mms.calibration.estimators import (
    bootstrap_days,
    fit_counts,
    fit_loglog,
    intensity_from_fraction,
    realised_sigma,
    touch_curve,
    touch_excursions,
)

REPO = Path(__file__).resolve().parents[1]
PROC = REPO / "data" / "processed"
OUT = REPO / "results" / "stage2"
TICK = 0.1
DELTAS = np.round(np.arange(0, 80 + 1) * TICK, 10)          # 0 .. 8 USDT
FIT_LO, FIT_HI = 0.1, 5.0                                    # fit range (USDT)
WINDOWS = (1, 2, 5)                                          # Dt in seconds
MAIN_W = 1
HORIZON_S = 3600                                             # ABM check horizon (trading horizon T)
N_BOOT = 2000
log = logging.getLogger("calibration")


def fit_range_counts(hits: np.ndarray) -> np.ndarray:
    """Zero out hits outside the fit range so fit_counts' mask drops them (hits < min_hits)."""
    h = hits.copy()
    h[(DELTAS < FIT_LO - 1e-9) | (DELTAS > FIT_HI + 1e-9)] = 0
    return h


def day_analysis(day: str) -> tuple[dict, pd.DataFrame, list[dict]]:
    snap = pd.read_parquet(PROC / f"BTCUSDT-snap1s-{day}.parquet")
    trades = pd.read_parquet(PROC / f"BTCUSDT-trades-{day}.parquet", columns=["time", "price", "is_buyer_maker"])
    mid = snap["mid"].to_numpy()
    s1, s10, s60 = (realised_sigma(mid, 1.0, st) for st in (1, 10, 60))
    # ABM check: how far does the mid move, relative to its level, over the trading horizon?
    m = pd.Series(mid)
    rel = ((m.shift(-HORIZON_S) - m).abs() / m).dropna()
    stats = {
        "day": day, "mean_mid": float(np.nanmean(mid)),
        "sigma_1s": s1, "sigma_10s": s10, "sigma_60s": s60,
        "var_ratio_60_1": (s60 / s1) ** 2,
        "rel_move_1h_median_pct": 100 * float(rel.median()), "rel_move_1h_p99_pct": 100 * float(rel.quantile(0.99)),
        "rel_move_1h_max_pct": 100 * float(rel.max()),
        "sigma_sqrtT_over_S_pct": 100 * s1 * np.sqrt(HORIZON_S) / float(np.nanmean(mid)),
        "median_spread": float(snap["spread"].median()),
        "share_1tick_spread": float((snap["spread"].round(1) == TICK).mean()),
        "trades": len(trades),
    }
    hour = (snap["time_s"] // 3600) % 24
    by_hour = [{"day": day, "hour": int(h), "sigma_1s": realised_sigma(g["mid"].to_numpy(), 1.0)}
                for h, g in snap.groupby(hour)]
    counts = []
    for w in WINDOWS:
        exc = touch_excursions(snap, trades, w)
        for side in ("ask", "bid"):
            hits, n = touch_curve(exc[f"{side}_exc"].to_numpy(), DELTAS)
            counts.append({"day": day, "side": side, "window_s": w, "n_windows": n, "hits": hits})
    return stats, pd.DataFrame(by_hour), counts


def pooled_hits(counts: list[dict], days, window: int, sides=("ask", "bid")) -> dict:
    """day -> (hits summed over sides, windows summed over sides)."""
    out = {}
    for d in days:
        rows = [c for c in counts if c["day"] == d and c["window_s"] == window and c["side"] in sides]
        out[d] = (sum(c["hits"] for c in rows), sum(c["n_windows"] for c in rows))
    return out


def fit_days(counts, days, window=MAIN_W, sides=("ask", "bid"), n_boot=N_BOOT, seed=0) -> dict:
    per_day = {d: (fit_range_counts(h), n) for d, (h, n) in pooled_hits(counts, days, window, sides).items()}
    return bootstrap_days(per_day, DELTAS, window, n_boot=n_boot, seed=seed)


def observed_lambda(counts, days, window=MAIN_W) -> np.ndarray:
    h, n = 0, 0
    for _, (hh, nn) in pooled_hits(counts, days, window).items():
        h, n = h + hh, n + nn
    return intensity_from_fraction(h / n, window)


def ci(t, nd=3) -> str:
    return f"{t[0]:.{nd}f} [{t[1]:.{nd}f}, {t[2]:.{nd}f}]"


def boot_ci(x, n_boot=N_BOOT, seed=0) -> tuple[float, float, float]:
    """Mean over days with a percentile bootstrap CI (resampling days)."""
    x = np.asarray(x, float)
    rng = np.random.default_rng(seed)
    b = x[rng.integers(0, len(x), (n_boot, len(x)))].mean(1)
    return float(x.mean()), *map(float, np.quantile(b, [0.025, 0.975]))


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    sel = pd.read_csv(OUT / "selected_days.csv")
    sel["day"] = sel["day"].astype(str)
    missing = [d for d in sel.day if not (PROC / f"BTCUSDT-{d}.meta.json").exists()]
    if missing:
        raise RuntimeError(f"days not processed yet: {missing} (run build_dataset.py)")

    stats, hours, counts = [], [], []
    for d in sel.day:
        s, h, c = day_analysis(d)
        stats.append(s), hours.append(h), counts.extend(c)
        log.info("%s sigma_1s=%.3f sigma_60s=%.3f", d, s["sigma_1s"], s["sigma_60s"])
    st = pd.DataFrame(stats).merge(sel, on="day")
    hours = pd.concat(hours).merge(sel[["day", "regime", "split"]], on="day")

    # touch counts table (small; lets every fit be redone without the data)
    tc = pd.DataFrame([{"day": c["day"], "side": c["side"], "window_s": c["window_s"], "n_windows": c["n_windows"],
                        **{f"d{dlt:.1f}": int(h) for dlt, h in zip(DELTAS, c["hits"])}} for c in counts])
    tc.to_csv(OUT / "touch_counts.csv", index=False)

    cal_days = list(sel[sel.split == "calibration"].day)
    test_days = list(sel[sel.split == "test"].day)

    # --- per-day fits (stability) --------------------------------------------------------
    for i, row in st.iterrows():
        for side, sides in (("pooled", ("ask", "bid")), ("ask", ("ask",)), ("bid", ("bid",))):
            h, n = pooled_hits(counts, [row.day], MAIN_W, sides)[row.day]
            f = fit_counts(fit_range_counts(h), n, DELTAS, MAIN_W)
            st.loc[i, f"A_{side}"], st.loc[i, f"k_{side}"], st.loc[i, f"r2_{side}"] = f["A"], f["k"], f["r2"]
        hp, np_ = pooled_hits(counts, [row.day], MAIN_W)[row.day]
        st.loc[i, "lambda_touch"] = float(intensity_from_fraction(hp[0] / np_, MAIN_W))
    st.to_csv(OUT / "day_stats.csv", index=False)
    hours.to_csv(OUT / "sigma_by_hour.csv", index=False)

    # --- side comparison: paired per-day difference in k (ask - bid) ---------------------
    cal = st[st.split == "calibration"]
    dk = boot_ci(cal.k_ask - cal.k_bid)
    dA = boot_ci((cal.A_ask - cal.A_bid) / cal.A_bid)
    fit_ask = fit_days(counts, cal_days, sides=("ask",))
    fit_bid = fit_days(counts, cal_days, sides=("bid",))
    pool_ok = (dk[1] <= 0 <= dk[2]) or abs(dk[0]) < 0.1 * cal.k_pooled.mean()
    log.info("ask-bid: dk = %s, relative dA = %s -> pool sides: %s", ci(dk), ci(dA), pool_ok)
    if not pool_ok:
        log.warning("ask and bid differ; pooled fit reported but treat with care")

    # --- main pooled fit + regimes + window sensitivity ---------------------------------
    main_fit = fit_days(counts, cal_days)
    regimes = {}
    for reg in ("low", "medium", "high"):
        dd = list(sel[(sel.split == "calibration") & (sel.regime == reg)].day)
        f = fit_days(counts, dd, seed=1)
        sig = boot_ci(st.set_index("day").loc[dd, "sigma_1s"])
        regimes[reg] = {"days": dd, "A": f["A"], "k": f["k"], "r2": f["r2"], "sigma_1s": sig,
                        "sigma_60s": boot_ci(st.set_index("day").loc[dd, "sigma_60s"]),
                        "mean_mid": float(st.set_index("day").loc[dd, "mean_mid"].mean())}
    windows = {w: fit_days(counts, cal_days, window=w, n_boot=500) for w in WINDOWS}
    # Fit including delta = 0 (the touch) to show why it is excluded
    hsum = sum(pooled_hits(counts, cal_days, MAIN_W)[d][0] for d in cal_days)
    nsum = sum(pooled_hits(counts, cal_days, MAIN_W)[d][1] for d in cal_days)
    h0 = hsum.copy(); h0[DELTAS > FIT_HI + 1e-9] = 0
    with_touch = fit_counts(h0, nsum, DELTAS, MAIN_W)
    sigma_cal = boot_ci(cal.sigma_1s)
    sigma_cal60 = boot_ci(cal.sigma_60s)

    # --- out of sample -------------------------------------------------------------------
    rng_mask = (DELTAS >= FIT_LO - 1e-9) & (DELTAS <= FIT_HI + 1e-9)
    pred = main_fit["A"][0] * np.exp(-main_fit["k"][0] * DELTAS)
    obs_test = observed_lambda(counts, test_days)
    obs_cal = observed_lambda(counts, cal_days)
    lr = np.log(obs_test[rng_mask] / pred[rng_mask])
    oos = {"pooled": {"mean_abs_log_ratio": float(np.mean(np.abs(lr))), "max_abs_log_ratio": float(np.max(np.abs(lr))),
                      "mean_log_ratio": float(np.mean(lr))}}
    test_fit = fit_days(counts, test_days, n_boot=500, seed=3)
    for reg in ("low", "medium", "high"):
        dd = list(sel[(sel.split == "test") & (sel.regime == reg)].day)
        o = observed_lambda(counts, dd)
        for src, (A, k) in (("pooled_params", (main_fit["A"][0], main_fit["k"][0])),
                            ("regime_params", (regimes[reg]["A"][0], regimes[reg]["k"][0]))):
            p = A * np.exp(-k * DELTAS)
            l = np.log(o[rng_mask] / p[rng_mask])
            oos[f"{reg}_{src}"] = {"mean_abs_log_ratio": float(np.mean(np.abs(l))), "max_abs_log_ratio": float(np.max(np.abs(l))),
                                   "mean_log_ratio": float(np.mean(l))}

    # --- sigma-conditional model (fitted on calibration days only) ----------------------
    fA, fk = fit_loglog(cal.sigma_1s, cal.A_pooled), fit_loglog(cal.sigma_1s, cal.k_pooled)
    rng = np.random.default_rng(4)
    sb = []
    for _ in range(N_BOOT):
        b = cal.iloc[rng.integers(0, len(cal), len(cal))]
        if b.sigma_1s.nunique() < 2:
            continue
        sb.append((fit_loglog(b.sigma_1s, b.A_pooled)["slope"], fit_loglog(b.sigma_1s, b.A_pooled)["scale"],
                   fit_loglog(b.sigma_1s, b.k_pooled)["slope"], fit_loglog(b.sigma_1s, b.k_pooled)["scale"]))
    sb = np.array(sb)
    q = lambda j: tuple(map(float, np.quantile(sb[:, j], [0.025, 0.975])))
    sigma_model = {"A_scale": (fA["scale"], *q(1)), "A_exponent": (fA["slope"], *q(0)), "A_r2": fA["r2"],
                   "k_scale": (fk["scale"], *q(3)), "k_exponent": (fk["slope"], *q(2)), "k_r2": fk["r2"],
                   "calibration_sigma_range": (float(cal.sigma_1s.min()), float(cal.sigma_1s.max()))}

    # Per test day: mean |ln(observed / predicted)| over the fit range for three predictors
    day_obs = {d: observed_lambda(counts, [d]) for d in sel.day}
    oos_days = []
    for _, r in st[st.split == "test"].iterrows():
        preds = {"pooled": (main_fit["A"][0], main_fit["k"][0]),
                 "regime": (regimes[r.regime]["A"][0], regimes[r.regime]["k"][0]),
                 "sigma_model": (float(fA["predict"](r.sigma_1s)), float(fk["predict"](r.sigma_1s)))}
        row = {"day": r.day, "regime": r.regime, "sigma_1s": r.sigma_1s, "A_day": r.A_pooled, "k_day": r.k_pooled}
        for name, (A, k) in preds.items():
            l = np.log(day_obs[r.day][rng_mask] / (A * np.exp(-k * DELTAS[rng_mask])))
            row[f"{name}_A"], row[f"{name}_k"] = A, k
            row[f"{name}_mean_abs_log_ratio"] = float(np.mean(np.abs(l)))
            row[f"{name}_bias"] = float(np.mean(l))
        oos_days.append(row)
    oos_days = pd.DataFrame(oos_days)
    oos_days.to_csv(OUT / "oos_by_day.csv", index=False)
    oos_summary = {name: {"mean_over_days": boot_ci(oos_days[f"{name}_mean_abs_log_ratio"]),
                          "worst_day": float(oos_days[f"{name}_mean_abs_log_ratio"].max())}
                   for name in ("pooled", "regime", "sigma_model")}

    calib = {
        "units": "time in seconds, prices in USDT per BTC",
        "window_s": MAIN_W, "fit_range_usdt": [FIT_LO, FIT_HI], "tick": TICK,
        "calibration_days": cal_days, "test_days": test_days,
        "pooled": {"A": main_fit["A"], "k": main_fit["k"], "r2": main_fit["r2"], "sigma_1s": sigma_cal,
                   "sigma_60s": sigma_cal60, "mean_mid": float(cal.mean_mid.mean()),
                   "lambda_touch": boot_ci(cal.lambda_touch)},
        "ask": {"A": fit_ask["A"], "k": fit_ask["k"]}, "bid": {"A": fit_bid["A"], "k": fit_bid["k"]},
        "ask_minus_bid_k": dk, "ask_minus_bid_A_rel": dA, "pooled_sides": bool(pool_ok),
        "regimes": regimes,
        "windows": {str(w): {"A": f["A"], "k": f["k"], "r2": f["r2"]} for w, f in windows.items()},
        "fit_with_touch": {"A": with_touch["A"], "k": with_touch["k"], "r2": with_touch["r2"]},
        "test_days_fit": {"A": test_fit["A"], "k": test_fit["k"], "r2": test_fit["r2"]},
        "oos": oos,
        "oos_by_day_summary": oos_summary,
        "sigma_model": sigma_model,
        "residuals_main": dict(zip([f"{d:.1f}" for d in DELTAS[main_fit["fit"]["mask"]]],
                                   map(float, main_fit["fit"]["residuals"]))),
        "stability_cv": {c: float(cal[c].std(ddof=1) / cal[c].mean()) for c in ("sigma_1s", "A_pooled", "k_pooled")},
        "stability_cv_all20": {c: float(st[c].std(ddof=1) / st[c].mean()) for c in ("sigma_1s", "A_pooled", "k_pooled")},
        "corr_with_sigma_all20": {c: float(np.corrcoef(st.sigma_1s, st[c])[0, 1]) for c in ("A_pooled", "k_pooled")},
        "intraday_sigma": {
            "cal_mean_by_hour": hours[hours.split == "calibration"].groupby("hour").sigma_1s.mean().round(4).to_dict(),
        },
    }
    hb = hours[hours.split == "calibration"].groupby("hour").sigma_1s.mean()
    calib["intraday_sigma"]["max_over_min"] = float(hb.max() / hb.min())
    calib["intraday_sigma"]["cv_across_hours"] = float(hb.std(ddof=1) / hb.mean())
    (OUT / "calibration.json").write_text(json.dumps(calib, indent=2, default=float), encoding="utf-8")

    # ----------------------------------------------------------------- charts
    mask = main_fit["fit"]["mask"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    ax = axes[0]
    ax.plot(DELTAS, np.log(obs_cal), "o", ms=3, color="#1f77b4", label="observed, calibration days (pooled sides)")
    ax.plot(DELTAS[mask], np.log(pred[mask]), "-", color="#d62728",
            label=f"fit: A={main_fit['A'][0]:.3f}/s, k={main_fit['k'][0]:.3f}/USDT, R²={main_fit['r2']:.3f}")
    ax.plot(DELTAS, np.log(pred), ":", color="#d62728", lw=1)
    ax.axvspan(FIT_LO, FIT_HI, color="grey", alpha=0.08, label="fit range")
    ax.set_xlabel("distance from mid δ (USDT)"); ax.set_ylabel("ln λ(δ)  (λ per second)")
    ax.set_title(f"Touch intensity, Δt = {MAIN_W} s, 10 calibration days"); ax.legend(fontsize=8, frameon=False)
    ax = axes[1]
    ax.axhline(0, color="grey", lw=0.6)
    ax.plot(DELTAS[mask], main_fit["fit"]["residuals"], "o-", ms=3)
    ax.set_xlabel("δ (USDT)"); ax.set_ylabel("residual ln λ − fit"); ax.set_title("Residuals of the exponential fit")
    fig.tight_layout(); fig.savefig(OUT / "fit_calibration.png", dpi=130); plt.close(fig)

    fig, axes = plt.subplots(1, 4, figsize=(18, 4.3), sharey=True)
    panels = [("all test days", test_days, main_fit)] + [
        (f"{r} vol test days", list(sel[(sel.split == "test") & (sel.regime == r)].day), None) for r in ("low", "medium", "high")]
    for ax, (title, dd, _) in zip(axes, panels):
        o = observed_lambda(counts, dd)
        ax.semilogy(DELTAS, o, "o", ms=3, color="black", label="observed (test)")
        ax.semilogy(DELTAS, pred, "-", color="#d62728", label="predicted: pooled calibration fit")
        ps = np.mean([fA["predict"](sg) * np.exp(-fk["predict"](sg) * DELTAS)
                      for sg in st.set_index("day").loc[dd, "sigma_1s"]], axis=0)
        ax.semilogy(DELTAS, ps, "-.", color="#9467bd", label="predicted: σ-conditional model (day's own σ)")
        reg = title.split()[0]
        if reg in regimes:
            pr = regimes[reg]["A"][0] * np.exp(-regimes[reg]["k"][0] * DELTAS)
            ax.semilogy(DELTAS, pr, "--", color="#2ca02c", label=f"predicted: {reg}-vol calibration fit")
        ax.axvspan(FIT_LO, FIT_HI, color="grey", alpha=0.08)
        ax.set_title(title); ax.set_xlabel("δ (USDT)")
        ax.legend(fontsize=7, frameon=False)
    axes[0].set_ylabel("λ(δ) per second (log scale)")
    fig.suptitle("Out of sample: predicted vs observed fill intensity on the 10 later (test) days, Δt = 1 s")
    fig.tight_layout(); fig.savefig(OUT / "oos_fit.png", dpi=130); plt.close(fig)

    fig, axes = plt.subplots(1, 4, figsize=(18, 4))
    colours = {"low": "#2ca02c", "medium": "#ff7f0e", "high": "#d62728"}
    for ax, col, lab in zip(axes[:3], ("sigma_1s", "A_pooled", "k_pooled"),
                            ("σ (USDT/√s), 1 s mid changes", "A (fills/s)", "k (1/USDT)")):
        for reg, g in st.groupby("regime"):
            for split, mk in (("calibration", "o"), ("test", "s")):
                gg = g[g.split == split]
                ax.scatter(pd.to_datetime(gg.day), gg[col], color=colours[reg], marker=mk,
                           label=f"{reg} / {split}")
        ax.set_title(lab); ax.tick_params(axis="x", rotation=45)
    axes[0].legend(fontsize=7, frameon=False)
    ax = axes[3]
    for reg, g in st.groupby("regime"):
        ax.scatter(g.sigma_1s, g.k_pooled, color=colours[reg], label=reg)
    ax.set_xlabel("σ (USDT/√s)"); ax.set_ylabel("k (1/USDT)"); ax.set_title("k vs σ across the 20 days")
    fig.suptitle("Parameter stability across days (circles = calibration, squares = test)")
    fig.tight_layout(); fig.savefig(OUT / "parameter_stability.png", dpi=130); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.5, 4.3))
    for w in WINDOWS:
        o = observed_lambda(counts, cal_days, w)
        ax.semilogy(DELTAS, o, "o", ms=2.5, label=f"Δt = {w} s: A={windows[w]['A'][0]:.3f}, k={windows[w]['k'][0]:.3f}")
    ax.axvspan(FIT_LO, FIT_HI, color="grey", alpha=0.08)
    ax.set_xlabel("δ (USDT)"); ax.set_ylabel("λ(δ) per second"); ax.legend(fontsize=8, frameon=False)
    ax.set_title("Sensitivity of λ(δ) to the window length Δt")
    fig.tight_layout(); fig.savefig(OUT / "window_sensitivity.png", dpi=130); plt.close(fig)

    # ----------------------------------------------------------------- markdown
    L = ["# Stage 2: calibration of σ, A, k to BTCUSDT (USD-M futures)\n",
         f"Calibration days: {', '.join(cal_days)}. Test days: {', '.join(test_days)}.",
         f"Touch estimator: Δt = {MAIN_W} s windows, sides pooled, fit over δ ∈ [{FIT_LO}, {FIT_HI}] USDT. "
         "CIs: 95% percentile bootstrap over days.\n",
         "## Pooled calibration parameters\n",
         "| parameter | estimate [95% CI] |", "|---|---|",
         f"| σ, 1 s mid changes (USDT/√s) | {ci(sigma_cal)} |",
         f"| σ, 60 s mid changes (USDT/√s) | {ci(sigma_cal60)} |",
         f"| A (fills/s at δ = 0, extrapolated) | {ci(main_fit['A'], 4)} |",
         f"| k (1/USDT) | {ci(main_fit['k'])} |",
         f"| R² of ln λ fit | {main_fit['r2']:.4f} |",
         f"| λ at the touch (δ = 0, observed) | {ci(calib['pooled']['lambda_touch'])} |",
         f"| ask − bid k (paired by day) | {ci(dk)} |",
         f"| (A_ask − A_bid)/A_bid | {ci(dA)} |",
         f"| fit including δ = 0 | A = {with_touch['A']:.3f}, k = {with_touch['k']:.3f}, R² = {with_touch['r2']:.3f} |",
         "\n## By volatility regime (calibration days)\n",
         "| regime | days | σ 1 s | A | k | R² | mean mid |", "|---|---|---|---|---|---|---|"]
    for reg, r in regimes.items():
        L.append(f"| {reg} | {', '.join(r['days'])} | {ci(r['sigma_1s'])} | {ci(r['A'], 4)} | {ci(r['k'])} | "
                 f"{r['r2']:.4f} | {r['mean_mid']:.0f} |")
    L += ["\n## Window length Δt (calibration days)\n", "| Δt (s) | A | k | R² |", "|---|---|---|---|"]
    for w, f in windows.items():
        L.append(f"| {w} | {ci(f['A'], 4)} | {ci(f['k'])} | {f['r2']:.4f} |")
    L += ["\n## Out of sample (test days)\n",
          "Mean / max |ln(observed / predicted)| over the fit range (0.4 ≈ a factor 1.5).\n",
          "| test set | parameters | mean abs log ratio | max abs log ratio | mean log ratio (bias) |", "|---|---|---|---|---|"]
    for key, v in oos.items():
        L.append(f"| {key.split('_')[0]} | {'_'.join(key.split('_')[1:]) or 'pooled_params'} | "
                 f"{v['mean_abs_log_ratio']:.3f} | {v['max_abs_log_ratio']:.3f} | {v['mean_log_ratio']:+.3f} |")
    L += ["\n### Per test day: mean |ln(observed/predicted)| over the fit range\n",
          "Predictors: pooled calibration fit; fit on the calibration days of the same regime; "
          "σ-conditional model A = a_A σ^b_A, k = a_k σ^b_k fitted on the calibration days and evaluated at the test day's own σ.\n",
          "| day | regime | σ 1 s | pooled | regime | σ-model | day's own A, k | σ-model A, k |", "|---|---|---|---|---|---|---|---|"]
    for _, r in oos_days.iterrows():
        L.append(f"| {r.day} | {r.regime} | {r.sigma_1s:.2f} | {r.pooled_mean_abs_log_ratio:.2f} | {r.regime_mean_abs_log_ratio:.2f} | "
                 f"{r.sigma_model_mean_abs_log_ratio:.2f} | {r.A_day:.3f}, {r.k_day:.3f} | {r.sigma_model_A:.3f}, {r.sigma_model_k:.3f} |")
    for name, v in oos_summary.items():
        L.append(f"\n- {name}: mean over test days {ci(v['mean_over_days'])}, worst day {v['worst_day']:.2f}")
    sm = sigma_model
    L.append(f"\nσ-conditional model (calibration days, σ range {sm['calibration_sigma_range'][0]:.2f}–{sm['calibration_sigma_range'][1]:.2f}): "
             f"A = {sm['A_scale'][0]:.4f} σ^{sm['A_exponent'][0]:.3f} (exponent CI [{sm['A_exponent'][1]:.2f}, {sm['A_exponent'][2]:.2f}], R² {sm['A_r2']:.2f}); "
             f"k = {sm['k_scale'][0]:.3f} σ^{sm['k_exponent'][0]:.3f} (exponent CI [{sm['k_exponent'][1]:.2f}, {sm['k_exponent'][2]:.2f}], R² {sm['k_r2']:.2f}).")
    L.append(f"\nFit on the test days themselves: A = {ci(test_fit['A'], 4)}, k = {ci(test_fit['k'])}.\n")
    L += ["\n## Per day\n", "| day | regime | split | RV % | mean mid | σ 1 s | σ 60 s | A | k | R² | λ touch | 1-tick spread share | 1 h move p99 % |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for _, r in st.iterrows():
        L.append(f"| {r.day} | {r.regime} | {r.split} | {r.rv:.2f} | {r.mean_mid:.0f} | {r.sigma_1s:.3f} | {r.sigma_60s:.3f} | "
                 f"{r.A_pooled:.4f} | {r.k_pooled:.3f} | {r.r2_pooled:.3f} | {r.lambda_touch:.2f} | {r.share_1tick_spread:.4f} | "
                 f"{r.rel_move_1h_p99_pct:.2f} |")
    L.append(f"\nIntraday σ (calibration days, mean by UTC hour): max/min = {calib['intraday_sigma']['max_over_min']:.2f}, "
             f"CV across hours = {calib['intraday_sigma']['cv_across_hours']:.2f}.")
    L.append(f"Across-day CV (calibration days): " + ", ".join(f"{k} {v:.2f}" for k, v in calib["stability_cv"].items()) + ".")
    L.append(f"Correlation with σ across all 20 days: " + ", ".join(f"{k} {v:+.2f}" for k, v in calib["corr_with_sigma_all20"].items()) + ".")
    (OUT / "calibration.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    log.info("wrote %s", OUT)


if __name__ == "__main__":
    main()
