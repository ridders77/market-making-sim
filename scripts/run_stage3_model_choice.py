"""Stage 3, Step 1: pre-registered choice of the fill-intensity model (see DECISIONS.md, "Step 1 pre-registration").

Fit on the 10 calibration days only (0.01 BTC lot rule, Delta t = 1 s, sides pooled):
  (a) relative units: lambda = A_rel exp(-k_rel delta_bps), delta_bps = 1e4 delta / mid0,
      fit over delta_bps in [0.03, 1.80] on a 0.01 bp grid;
  (b) sigma-conditional: A = a sigma_hat^b, k = c sigma_hat^d with the trailing EWMA sigma_hat (60 s changes,
      span 30 min), from decile fits of A, k (USDT, delta in [0.1, 5]) and log-log OLS on decile medians.
Score each on the 10 test days: windows from 00:30 UTC with a valid mid0 and sigma_hat; for delta = 0.1 ... 5.0 USDT
compare observed and predicted fill shares (predicted = mean over windows of 1 - exp(-lambda dt), information at
window start only), as intensities; score_day = mean |ln(lambda_pred / lambda_obs)| over deltas with >= 30 hits;
model score = mean over test days. Rule: (a) if score <= 0.4, else (b) if <= 0.4, else paper parameters.

Writes results/stage3/model_choice.{md,json}, model_choice.png, calibration_stage3.json (parameters for the simulator).

Run:  .venv\\Scripts\\python scripts\\run_stage3_model_choice.py
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

from mms.calibration.estimators import EPS, fit_counts, fit_loglog, intensity_from_fraction, realised_sigma, touch_curve
from mms.calibration.windows import day_hits, excursions, load_windows

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "results" / "stage3"
PROC = REPO / "data" / "processed"
RULE = "l001"
DT_W = 1.0
D_USDT = np.round(np.arange(0, 81) * 0.1, 10)
FIT_USDT = (0.1, 5.0)
D_BPS = np.round(np.arange(0, 301) * 0.01, 10)
FIT_BPS = (0.03, 1.80)
EVAL_D = np.round(np.arange(1, 51) * 0.1, 10)        # 0.1 ... 5.0 USDT
WARMUP_S = 1800
MIN_HITS = 30
PASS = 0.4
N_BOOT = 2000
log = logging.getLogger("stage3_model_choice")


def masked(h, grid, lo, hi):
    h = h.copy()
    h[(grid < lo - 1e-9) | (grid > hi + 1e-9)] = 0
    return h


def fit_model_a(cache, days) -> dict:
    H, N = 0, 0
    for d, (h, n) in day_hits(days, RULE, D_BPS, unit="bps", cache=cache).items():
        H, N = H + h, N + n
    f = fit_counts(masked(H, D_BPS, *FIT_BPS), N, D_BPS, DT_W, MIN_HITS)
    return {"A_rel": f["A"], "k_rel_per_bp": f["k"], "r2": f["r2"], "n_points": f["n_used"]}


def eval_windows(w: pd.DataFrame) -> pd.DataFrame:
    """Windows used for scoring (and for model b's fit): valid mid0 and sigma_hat, from 00:30 UTC on."""
    t0 = int(w.time_s.iloc[0])
    keep = np.isfinite(w.mid0) & np.isfinite(w.sigma_hat) & (w.time_s - t0 >= WARMUP_S)
    return w[keep]


def fit_model_b(cache, days) -> dict:
    ws = pd.concat([eval_windows(cache[d]) for d in days], ignore_index=True)
    edges = np.quantile(ws.sigma_hat, np.linspace(0, 1, 11))
    dec = np.clip(np.searchsorted(edges, ws.sigma_hat, side="right") - 1, 0, 9)
    rows = []
    for j in range(10):
        g = ws[dec == j]
        h, n = touch_curve(excursions(g, RULE), D_USDT)
        f = fit_counts(masked(h, D_USDT, *FIT_USDT), n, D_USDT, DT_W, MIN_HITS)
        rows.append({"decile": j, "sigma_median": float(g.sigma_hat.median()), "n_windows": len(g),
                     "A": f["A"], "k": f["k"], "r2": f["r2"]})
    dec_df = pd.DataFrame(rows)
    fA, fk = fit_loglog(dec_df.sigma_median, dec_df.A), fit_loglog(dec_df.sigma_median, dec_df.k)
    return {"a": fA["scale"], "b": fA["slope"], "c": fk["scale"], "d": fk["slope"], "A_r2": fA["r2"], "k_r2": fk["r2"],
            "deciles": dec_df.to_dict(orient="records")}


def lam_pred(model: str, params: dict, w: pd.DataFrame, delta: float) -> np.ndarray:
    """Per-window intensity at USDT distance delta, from information available when the window opens."""
    if model == "a":
        return params["A_rel"] * np.exp(-params["k_rel_per_bp"] * 1e4 * delta / w.mid0.to_numpy())
    if model == "b":
        s = w.sigma_hat.to_numpy()
        return params["a"] * s ** params["b"] * np.exp(-params["c"] * s ** params["d"] * delta)
    if model == "usdt_fixed":
        return np.full(len(w), params["A"] * np.exp(-params["k"] * delta))
    raise ValueError(model)


def score_day(model: str, params: dict, w_all: pd.DataFrame) -> dict:
    w = eval_windows(w_all)
    exc = excursions(w, RULE)
    hits, n = touch_curve(exc, EVAL_D)
    p_obs = hits / n
    lr, lam_o, lam_p = [], [], []
    for j, dl in enumerate(EVAL_D):
        lp = intensity_from_fraction(np.mean(-np.expm1(-lam_pred(model, params, w, dl) * DT_W)), DT_W)
        lo = intensity_from_fraction(p_obs[j], DT_W)
        lam_o.append(float(lo)), lam_p.append(float(lp))
        if hits[j] >= MIN_HITS and hits[j] < n:
            lr.append(float(np.log(lp / lo)))
    lr = np.array(lr)
    return {"score": float(np.mean(np.abs(lr))), "bias": float(np.mean(lr)), "n_points": len(lr),
            "n_dropped": len(EVAL_D) - len(lr), "n_windows": len(w), "lam_obs": lam_o, "lam_pred": lam_p}


def boot_mean(x, seed=0):
    x = np.asarray(x, float)
    b = x[np.random.default_rng(seed).integers(0, len(x), (N_BOOT, len(x)))].mean(1)
    return float(x.mean()), *map(float, np.quantile(b, [0.025, 0.975]))


def sigma60_bps(day: str) -> float:
    """sigma of 60 s *relative* mid changes, in bps per sqrt(s) (zero-mean estimator as in stage 2)."""
    mid = pd.read_parquet(PROC / f"BTCUSDT-snap1s-{day}.parquet", columns=["mid"]).mid.to_numpy()
    return realised_sigma(1e4 * np.log(mid), 1.0, step=60)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    OUT.mkdir(parents=True, exist_ok=True)
    cal2 = json.loads((REPO / "results" / "stage2" / "calibration.json").read_text(encoding="utf-8"))
    cal_days, test_days = cal2["calibration_days"], cal2["test_days"]
    cache = {d: load_windows(d) for d in cal_days + test_days}

    # ------------------------------------------------ fit on calibration days only
    ma = fit_model_a(cache, cal_days)
    mb = fit_model_b(cache, cal_days)
    H, N = 0, 0
    for d, (h, n) in day_hits(cal_days, RULE, D_USDT, cache=cache).items():
        H, N = H + h, N + n
    fu = fit_counts(masked(H, D_USDT, *FIT_USDT), N, D_USDT, DT_W, MIN_HITS)
    mu = {"A": fu["A"], "k": fu["k"]}
    log.info("model a: %s", ma)
    log.info("model b: a=%.4f b=%.3f c=%.4f d=%.3f", mb["a"], mb["b"], mb["c"], mb["d"])

    # ------------------------------------------------ score on test days
    per_day = []
    curves = {}
    for d in test_days:
        row = {"day": d}
        for name, params in (("a", ma), ("b", mb), ("usdt_fixed", mu)):
            s = score_day(name, params, cache[d])
            row[f"{name}_score"], row[f"{name}_bias"], row[f"{name}_dropped"] = s["score"], s["bias"], s["n_dropped"]
            curves[(d, name)] = s
        per_day.append(row)
    pdf = pd.DataFrame(per_day)
    scores = {m: boot_mean(pdf[f"{m}_score"]) for m in ("a", "b", "usdt_fixed")}
    if scores["a"][0] <= PASS:
        chosen = "a"
    elif scores["b"][0] <= PASS:
        chosen = "b"
    else:
        chosen = "paper"
    log.info("scores: %s -> chosen %s", {k: round(v[0], 3) for k, v in scores.items()}, chosen)

    # ------------------------------------------------ simulator parameters per regime
    regimes = {}
    for reg, rp in cal2["regimes"].items():
        s60 = rp["sigma_60s"][0]
        s60_bps = float(np.mean([sigma60_bps(d) for d in rp["days"]]))
        s0 = round(rp["mean_mid"], 1)
        r = {"days": rp["days"], "s0": s0, "sigma_60s_usdt": s60, "sigma_60s_bps": s60_bps}
        if chosen == "a":
            r.update(sigma=s60_bps * s0 / 1e4, A=ma["A_rel"], k=ma["k_rel_per_bp"] * 1e4 / s0)
        elif chosen == "b":
            r.update(sigma=s60, A=mb["a"] * s60 ** mb["b"], k=mb["c"] * s60 ** mb["d"])
        regimes[reg] = r
    calib = {"rule": RULE, "chosen_model": chosen, "pass_mark": PASS, "scores": scores, "model_a": ma,
             "model_b": {k: v for k, v in mb.items()}, "usdt_fixed_reference": mu, "regimes": regimes,
             "calibration_days": cal_days, "test_days": test_days}
    (OUT / "calibration_stage3.json").write_text(json.dumps(calib, indent=2, default=float), encoding="utf-8")
    pdf.to_csv(OUT / "model_choice_by_day.csv", index=False)

    # ------------------------------------------------ report
    names = {"a": "(a) relative units, fixed A_rel, k_rel", "b": "(b) σ-conditional, trailing σ̂",
             "usdt_fixed": "reference: fixed A, k in USDT (not part of the rule)"}
    L = ["# Step 1: pre-registered choice of the fill-intensity model\n",
         "Rule fixed in DECISIONS.md before these scores were computed: choose (a) if its score ≤ 0.4, else (b) if ≤ 0.4, "
         "else the paper parameters. Score = mean over the 10 test days of the mean |ln(λ_pred/λ_obs)| over δ = 0.1…5.0 USDT "
         "(0.01 BTC lot rule, windows from 00:30 UTC). 95% CI: bootstrap over test days.\n",
         "| model | score [95% CI] | worst day | mean bias ln(pred/obs) | passes (≤ 0.4)? |", "|---|---|---|---|---|"]
    for m in ("a", "b", "usdt_fixed"):
        L.append(f"| {names[m]} | {scores[m][0]:.3f} [{scores[m][1]:.3f}, {scores[m][2]:.3f}] | {pdf[f'{m}_score'].max():.3f} | "
                 f"{pdf[f'{m}_bias'].mean():+.3f} | {'yes' if scores[m][0] <= PASS else 'no'} |")
    L.append(f"\n**Chosen: {names.get(chosen, 'paper parameters (fallback)')}.**\n")
    L += ["## Fitted parameters (calibration days)\n",
          f"- (a) A_rel = {ma['A_rel']:.4f} /s, k_rel = {ma['k_rel_per_bp']:.3f} /bp (R² {ma['r2']:.4f}, {ma['n_points']} grid points). "
          f"At S = 27,800 this is k = {ma['k_rel_per_bp'] * 1e4 / 27800:.3f} /USDT.",
          f"- (b) A = {mb['a']:.4f}·σ̂^{mb['b']:.3f} (R² {mb['A_r2']:.3f}), k = {mb['c']:.4f}·σ̂^{mb['d']:.3f} (R² {mb['k_r2']:.3f}); "
          f"σ̂ deciles span {mb['deciles'][0]['sigma_median']:.2f}–{mb['deciles'][-1]['sigma_median']:.2f} USDT/√s.",
          f"- reference: A = {mu['A']:.4f} /s, k = {mu['k']:.3f} /USDT.\n",
          "## Per test day\n", "| day | (a) score | (a) bias | (b) score | (b) bias | USDT-fixed score | δ points dropped (a/b) |",
          "|---|---|---|---|---|---|---|"]
    for _, r in pdf.iterrows():
        L.append(f"| {r.day} | {r.a_score:.3f} | {r.a_bias:+.3f} | {r.b_score:.3f} | {r.b_bias:+.3f} | {r.usdt_fixed_score:.3f} | "
                 f"{r.a_dropped}/{r.b_dropped} |")
    L += ["\n## Simulator parameters by regime (chosen model, regime calibration days)\n",
          "| regime | S₀ | σ_60s (USDT/√s) | σ_60s (bp/√s) | σ used | A | k (/USDT) |", "|---|---|---|---|---|---|---|"]
    for reg, r in regimes.items():
        if "A" in r:
            L.append(f"| {reg} | {r['s0']:.0f} | {r['sigma_60s_usdt']:.3f} | {r['sigma_60s_bps']:.3f} | {r['sigma']:.3f} | {r['A']:.4f} | {r['k']:.3f} |")
    (OUT / "model_choice.md").write_text("\n".join(L) + "\n", encoding="utf-8")

    # ------------------------------------------------ chart
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    ax = axes[0]
    x = np.arange(len(test_days))
    for off, m, col in ((-0.27, "a", "#1f77b4"), (0.0, "b", "#9467bd"), (0.27, "usdt_fixed", "#bbbbbb")):
        ax.bar(x + off, pdf[f"{m}_score"], width=0.27, color=col, label=names[m])
    ax.axhline(PASS, color="#d62728", ls="--", lw=1, label="pass mark 0.4")
    ax.set_xticks(x, [d[2:] for d in test_days], rotation=45, fontsize=8)
    ax.set_ylabel("mean |ln(λ_pred / λ_obs)|"); ax.set_title("Out-of-sample score per test day")
    ax.legend(fontsize=7, frameon=False)
    ax = axes[1]
    for d, col in zip((test_days[2], test_days[-1]), ("#2ca02c", "#d62728")):
        ax.semilogy(EVAL_D, curves[(d, "a")]["lam_obs"], "o", ms=3, color=col, label=f"observed {d}")
        ax.semilogy(EVAL_D, curves[(d, "a")]["lam_pred"], "-", color=col, label="(a) predicted")
        ax.semilogy(EVAL_D, curves[(d, "b")]["lam_pred"], "--", color=col, label="(b) predicted")
    ax.set_xlabel("δ (USDT)"); ax.set_ylabel("λ(δ) per second"); ax.set_title("Lowest- and highest-price test days")
    ax.legend(fontsize=7, frameon=False)
    fig.tight_layout(); fig.savefig(OUT / "model_choice.png", dpi=130); plt.close(fig)
    log.info("wrote %s", OUT)


if __name__ == "__main__":
    main()
