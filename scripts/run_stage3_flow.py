"""Stage 3, Step 2: adverse selection measured from the data (markouts and order-flow imbalance).

Markouts: for every aggTrade, the mid move against the passive (maker) side at 1, 5, 30, 60 s, by distance
delta of the print from the mid (buckets in USDT). Calibration days give the estimates; test days check them.
95% CIs by bootstrap over days (days are the independent units).

OFI: OFI_N = signed aggressor volume over the last N = 5, 30, 60 s, raw (BTC) and normalised
(imbalance = OFI / volume). Predict the next h = 5, 30 s mid move (bps): OLS on calibration days
(slope, R^2, hit rate), then the same line on test days (R^2 vs the zero forecast, hit rate).
Main signal (rule fixed before looking at test days): the (N, raw/imbalance) pair with the highest
calibration R^2 averaged over h = 5 and 30 s.

Simulator flow parameters (calibration days only): signal z = main signal / its calibration std;
beta(h) = slope per unit z at h = 1..120 s -> (kappa, tau) of the OU drift; fill tilt theta from
ask vs bid lot-fill shares by z decile; markout jump J(delta) = 60 s markout minus the part the
simulator's drift already produces, per volatility regime; jump noise from the 1 s markout spread.

Writes results/stage3/: markouts_by_day.csv, markouts.md, markout_curves.png, ofi.csv, ofi.md,
ofi_predictive_power.png, flow_calibration.json.

Run:  .venv\\Scripts\\python scripts\\run_stage3_flow.py
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

from mms.calibration.flow import fit_drift_decay, fit_tilt, forward_move, markouts, ols, prediction_stats, rolling_ofi
from mms.calibration.windows import load_windows

REPO = Path(__file__).resolve().parents[1]
PROC = REPO / "data" / "processed"
OUT = REPO / "results" / "stage3"
HORIZONS = (1, 5, 30, 60)
BUCKETS = [0.0, 0.1, 0.2, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0, np.inf]
BLABELS = ["touch (<0.1)", "0.1–0.2", "0.2–0.5", "0.5–1", "1–2", "2–3", "3–5", "5–10", "≥10"]
NOISE_BUCKETS = range(1, 8)   # δ in [0.1, 10): the jump-noise estimate leaves out the heavy-tailed ≥10 sweeps
OFI_N = (5, 30, 60)
OFI_H = (5, 30)
BETA_H = (1, 2, 3, 5, 10, 20, 30, 60, 120)
TILT_DELTAS = (0.1, 0.5, 1.0, 2.0, 3.0)
TILT_MAIN_DELTA = 2.0
JUMP_H = 60
N_BOOT = 2000
log = logging.getLogger("stage3_flow")


def signal_arrays(w: pd.DataFrame) -> dict:
    """Per second: OFI and imbalance for each N."""
    out = {}
    for n in OFI_N:
        ofi, imb = rolling_ofi(w.buy_vol.to_numpy(), w.sell_vol.to_numpy(), n)
        out[(n, "raw")], out[(n, "imb")] = ofi, imb
    return out


def day_markouts(day: str, main_key: tuple | None = None, z_scale: float | None = None) -> pd.DataFrame:
    """Per-bucket sums for one day (so pooled means and day-bootstrap CIs need no trade-level data)."""
    snap = pd.read_parquet(PROC / f"BTCUSDT-snap1s-{day}.parquet", columns=["time_s", "mid"])
    tr = pd.read_parquet(PROC / f"BTCUSDT-trades-{day}.parquet", columns=["time", "price", "is_buyer_maker", "bid", "ask"])
    n_no_quote = int(tr[["bid", "ask"]].isna().any(axis=1).sum())
    tr = tr.dropna(subset=["bid", "ask"])
    m = markouts(tr, snap, HORIZONS)
    m["bucket"] = pd.cut(m.delta, BUCKETS, right=False, labels=False)
    if main_key is not None:
        # signal before the trade's second, signed in the taker's direction (+ when the flow agreed with the taker)
        w = load_windows(day)
        sig = signal_arrays(w)[main_key] / z_scale
        sec = (tr.time.to_numpy(np.int64) // 1000) - int(snap.time_s.iloc[0])
        taker_sign = np.where(tr.is_buyer_maker.to_numpy(bool), -1.0, 1.0)
        m["x_taker"] = sig[np.clip(sec, 0, len(sig) - 1)] * taker_sign
    rows = []
    for b, g in m.groupby("bucket"):
        r = {"day": day, "bucket": int(b), "n": len(g), "delta_sum": g.delta.sum(), "n_no_quote_day": n_no_quote}
        for h in HORIZONS:
            for unit in ("", "bps_"):
                v = g[f"adverse_{unit}{h}"].dropna()
                r[f"sum_{unit}{h}"], r[f"n_{unit}{h}"] = v.sum(), len(v)
        v = g["adverse_1"].dropna()
        r["sumsq_1"] = float((v**2).sum())
        if "x_taker" in g:
            v = g.x_taker.dropna()
            r["sum_x_taker"], r["n_x_taker"] = v.sum(), len(v)
        rows.append(r)
    return pd.DataFrame(rows)


def pooled_mean(df: pd.DataFrame, days, col_sum, col_n, n_boot=N_BOOT, seed=0) -> dict:
    """bucket -> (pooled mean, lo, hi) with a bootstrap over days."""
    out = {}
    rng = np.random.default_rng(seed)
    days = list(days)
    for b, g in df[df.day.isin(days)].groupby("bucket"):
        g = g.set_index("day").reindex(days)
        s, n = g[col_sum].fillna(0).to_numpy(), g[col_n].fillna(0).to_numpy()
        idx = rng.integers(0, len(days), (n_boot, len(days)))
        with np.errstate(invalid="ignore"):
            bs = s[idx].sum(1) / n[idx].sum(1)
        out[int(b)] = (float(s.sum() / n.sum()), *map(float, np.nanquantile(bs, [0.025, 0.975])))
    return out


def fmt(t, nd=2):
    return f"{t[0]:.{nd}f} [{t[1]:.{nd}f}, {t[2]:.{nd}f}]"


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    OUT.mkdir(parents=True, exist_ok=True)
    cal2 = json.loads((REPO / "results" / "stage2" / "calibration.json").read_text(encoding="utf-8"))
    c3 = json.loads((OUT / "calibration_stage3.json").read_text(encoding="utf-8"))
    cal_days, test_days = cal2["calibration_days"], cal2["test_days"]
    days = cal_days + test_days
    W = {d: load_windows(d) for d in days}

    # ================================================================ OFI
    SIG = {d: signal_arrays(W[d]) for d in days}
    MID = {d: W[d].mid_snap.to_numpy() for d in days}
    pair_cache = {}

    def pair(d, n, kind, h):
        """Finite (x, y) pairs for one day: signal at s, next-h-second move from s."""
        key = (d, n, kind, h)
        if key not in pair_cache:
            x, y = SIG[d][(n, kind)], forward_move(MID[d], h)
            ok = np.isfinite(x) & np.isfinite(y)
            pair_cache[key] = (x[ok].astype(np.float32), y[ok].astype(np.float32))
        return pair_cache[key]

    def pooled(dd, n, kind, h):
        xs, ys = zip(*(pair(d, n, kind, h) for d in dd))
        return np.concatenate(xs).astype(float), np.concatenate(ys).astype(float)

    ofi_rows = []
    fits = {}
    rng = np.random.default_rng(5)
    boot_idx_cal = rng.integers(0, len(cal_days), (300, len(cal_days)))
    boot_idx_test = rng.integers(0, len(test_days), (300, len(test_days)))
    for n in OFI_N:
        for kind in ("raw", "imb"):
            for h in OFI_H:
                x, y = pooled(cal_days, n, kind, h)
                f = ols(x, y)
                fits[(n, kind, h)] = f
                ins = prediction_stats(x, y, f["slope"], f["intercept"])
                xt, yt = pooled(test_days, n, kind, h)
                oos = prediction_stats(xt, yt, f["slope"], f["intercept"])
                # day-bootstrap CIs: in-sample refit on resampled calibration days; OOS with the fixed line
                b_r2, b_oos, b_hit = [], [], []
                for bi in boot_idx_cal[:200]:
                    xb, yb = pooled([cal_days[i] for i in bi], n, kind, h)
                    b_r2.append(ols(xb, yb)["r2"])
                for bi in boot_idx_test[:200]:
                    xb, yb = pooled([test_days[i] for i in bi], n, kind, h)
                    s = prediction_stats(xb, yb, f["slope"], f["intercept"])
                    b_oos.append(s["r2_oos"]), b_hit.append(s["hit_rate"])
                q = lambda v: tuple(map(float, np.quantile(v, [0.025, 0.975])))
                ofi_rows.append({"N": n, "signal": kind, "h": h, "slope_bp_per_sd": f["slope"] * x.std(), "x_sd_cal": float(x.std()),
                                 "r2_cal": f["r2"], "r2_cal_lo": q(b_r2)[0], "r2_cal_hi": q(b_r2)[1], "hit_cal": ins["hit_rate"],
                                 "r2_test": oos["r2_oos"], "r2_test_lo": q(b_oos)[0], "r2_test_hi": q(b_oos)[1],
                                 "hit_test": oos["hit_rate"], "hit_test_lo": q(b_hit)[0], "hit_test_hi": q(b_hit)[1],
                                 "slope_test_refit_bp_per_sd": ols(xt, yt)["slope"] * x.std(), "n_cal": len(x), "n_test": len(xt)})
                log.info("OFI N=%d %s h=%d: R2 cal %.4f test %.4f hit %.3f/%.3f", n, kind, h, f["r2"], oos["r2_oos"],
                         ins["hit_rate"], oos["hit_rate"])
                pair_cache.clear()   # keep memory small: one (N, signal, h) at a time
    ofi = pd.DataFrame(ofi_rows)
    ofi.to_csv(OUT / "ofi.csv", index=False)
    score = ofi.groupby(["N", "signal"]).r2_cal.mean()
    main_key = tuple(score.idxmax())
    main_key = (int(main_key[0]), main_key[1])
    log.info("main signal (highest mean calibration R2): %s", main_key)
    xc, _ = pooled(cal_days, *main_key, 5)
    z_scale = float(xc.std())

    # beta(h) per unit z on calibration and test days, and the OU fit (calibration only)
    beta_cal, beta_test = [], []
    for h in BETA_H:
        x, y = pooled(cal_days, *main_key, h)
        beta_cal.append(ols(x / z_scale, y)["slope"])
        x, y = pooled(test_days, *main_key, h)
        beta_test.append(ols(x / z_scale, y)["slope"])
    decay = fit_drift_decay(np.array(BETA_H, float), np.array(beta_cal))
    log.info("drift decay (bps): kappa=%.4f bp/s per z, tau=%.2f s", decay["kappa"], decay["tau"])

    # ================================================================ fill tilt (lot rule, calibration days)
    tilt = {}
    xs_all, hits = [], {dl: ([], []) for dl in TILT_DELTAS}
    for d in cal_days:
        w = W[d]
        z = signal_arrays(w)[main_key] / z_scale
        ok = np.isfinite(z) & np.isfinite(w.mid0.to_numpy())
        xs_all.append(z[ok])
        for dl in TILT_DELTAS:
            hits[dl][0].append(w.ask_exc_l001.to_numpy()[ok] >= dl - 1e-6)
            hits[dl][1].append(w.bid_exc_l001.to_numpy()[ok] >= dl - 1e-6)
    zc = np.concatenate(xs_all)
    for dl in TILT_DELTAS:
        f = fit_tilt(zc, np.concatenate(hits[dl][0]), np.concatenate(hits[dl][1]))
        tilt[dl] = f
        log.info("tilt at delta=%.1f: theta=%.3f (R2 %.3f)", dl, f["theta"], f["r2"])
    theta = tilt[TILT_MAIN_DELTA]["theta"]

    # ================================================================ markouts
    with ProcessPoolExecutor(max_workers=8) as ex:
        parts = list(ex.map(day_markouts, days, [main_key] * len(days), [z_scale] * len(days)))
    mk = pd.concat(parts, ignore_index=True)
    mk.to_csv(OUT / "markouts_by_day.csv", index=False)
    no_quote = mk.groupby("day").n_no_quote_day.first()
    if no_quote.sum():
        log.info("trades without a prior quote (dropped, not filled): %s", no_quote[no_quote > 0].to_dict())
    curves = {}
    for split, dd in (("calibration", cal_days), ("test", test_days)):
        for unit in ("", "bps_"):
            for h in HORIZONS:
                curves[(split, unit, h)] = pooled_mean(mk, dd, f"sum_{unit}{h}", f"n_{unit}{h}")
    mean_delta = mk[mk.day.isin(cal_days)].groupby("bucket").apply(lambda g: g.delta_sum.sum() / g.n.sum())
    x_taker = pooled_mean(mk, cal_days, "sum_x_taker", "n_x_taker")
    x_taker_test = pooled_mean(mk, test_days, "sum_x_taker", "n_x_taker")

    # ================================================================ simulator flow parameters per regime
    flow_reg = {}
    for reg, rp in c3["regimes"].items():
        s0, sigma = rp["s0"], rp["sigma"]
        kappa_usdt = decay["kappa"] * s0 / 1e4          # bps/s per z -> USDT/s per z
        tau = decay["tau"]
        drift_part = kappa_usdt * tau * (1 - np.exp(-JUMP_H / tau)) * theta   # E[drift over 60 s | fill], USDT
        # Pooled calibration-day markout curve in bps (stable out of sample, unlike USDT), converted at the regime's S0.
        # Per-regime USDT curves rest on 3 days each and have unstable tails (DECISIONS.md).
        mr = {b: tuple(v * s0 / 1e4 for v in t) for b, t in curves[("calibration", "bps_", JUMP_H)].items()}
        md = mean_delta
        J = {int(b): mr[b][0] - drift_part for b in mr}
        neg = [b for b, v in J.items() if v < 0]
        if neg:
            log.warning("%s: J < 0 in buckets %s, clipped to 0", reg, neg)
        g = mk[mk.day.isin(rp["days"]) & mk.bucket.isin(NOISE_BUCKETS)]
        # spread of the 1 s markout around its bucket means, net of 1.5 s of diffusion
        within = (g.sumsq_1 - g.sum_1**2 / g.n_1).sum() / g.n_1.sum()
        jump_sd = float(np.sqrt(max(0.0, within - sigma**2 * 1.5)))
        flow_reg[reg] = {
            "kappa_usdt_per_s": kappa_usdt, "tau_s": tau, "theta": theta,
            "jump_delta": [float(md[b]) for b in sorted(J)], "jump_mean": [max(0.0, J[b]) for b in sorted(J)],
            "markout60_usdt": [mr[b][0] for b in sorted(J)], "drift_part_usdt": float(drift_part),
            "jump_sd": jump_sd, "noise_sigma": float(np.sqrt(sigma**2 - 2 * kappa_usdt**2 * tau)),
            "drift_variance_share": float(2 * kappa_usdt**2 * tau / sigma**2),
        }
    flow = {"main_signal": {"N": main_key[0], "kind": main_key[1], "z_scale": z_scale},
            "beta_h": list(BETA_H), "beta_cal_bp_per_z": beta_cal, "beta_test_bp_per_z": beta_test,
            "kappa_bp_per_s": decay["kappa"], "tau_s": decay["tau"],
            "theta": theta, "tilt_by_delta": {str(k): {"theta": v["theta"], "r2": v["r2"]} for k, v in tilt.items()},
            "x_taker_at_fill_cal": {BLABELS[b]: v for b, v in x_taker.items()},
            "x_taker_at_fill_test": {BLABELS[b]: v for b, v in x_taker_test.items()},
            "regimes": flow_reg, "jump_horizon_s": JUMP_H}
    (OUT / "flow_calibration.json").write_text(json.dumps(flow, indent=2, default=float), encoding="utf-8")

    # ================================================================ markdown
    L = ["# Step 2: adverse selection measured from the data\n",
         "## Markouts: mid move against the passive side after a print at distance δ from the mid\n",
         "Positive = against the maker. Pooled over trades; 95% CI = bootstrap over days. Calibration = 10 earlier days, "
         "test = 10 later days (out of sample). h = 1 s means the book 1–2 s after the trade (1-second snapshots).\n"]
    for unit, uname, nd in (("", "USDT", 2), ("bps_", "bps of mid", 3)):
        L += [f"\n### In {uname}\n", "| δ bucket (USDT) | trades (cal / test) | " + " | ".join(f"{h} s cal" for h in HORIZONS) + " | "
              + " | ".join(f"{h} s test" for h in HORIZONS) + " |", "|---" * (2 + 2 * len(HORIZONS)) + "|"]
        ncal = mk[mk.day.isin(cal_days)].groupby("bucket").n.sum()
        ntest = mk[mk.day.isin(test_days)].groupby("bucket").n.sum()
        for b, lab in enumerate(BLABELS):
            cells = [fmt(curves[("calibration", unit, h)][b], nd) for h in HORIZONS] + \
                    [fmt(curves[("test", unit, h)][b], nd) for h in HORIZONS]
            L.append(f"| {lab} | {ncal[b]:,} / {ntest[b]:,} | " + " | ".join(cells) + " |")
    L += ["\nSigned signal at the fill (z of the main OFI signal just before the trade, + = flow was on the taker's side), "
          "calibration / test:\n", "| δ bucket | calibration | test |", "|---|---|---|"]
    for b, lab in enumerate(BLABELS):
        L.append(f"| {lab} | {fmt(x_taker[b], 3)} | {fmt(x_taker_test[b], 3)} |")
    L += ["\n## Simulator markout jump J(δ) per regime (pooled calibration-day bps curve × S₀)\n",
          f"J = 60 s markout − the part the simulator's drift already produces after a fill "
          f"(κ τ (1 − e^(−60/τ)) θ). Jump noise sd from the 1 s markout spread (δ ∈ [0.1, 10)) net of 1.5 s of diffusion.\n",
          "| regime | " + " | ".join(BLABELS) + " | drift part | jump sd |", "|---" * (len(BLABELS) + 3) + "|"]
    for reg, f in flow_reg.items():
        L.append(f"| {reg} | " + " | ".join(f"{m:.2f} → {j:.2f}" for m, j in zip(f["markout60_usdt"], f["jump_mean"]))
                 + f" | {f['drift_part_usdt']:.3f} | {f['jump_sd']:.2f} |")
    L.append("\n(cells: 60 s markout → J, USDT; bucket mean δ: " +
             ", ".join(f"{mean_delta[b]:.2f}" for b in range(len(BLABELS))) + ")")
    (OUT / "markouts.md").write_text("\n".join(L) + "\n", encoding="utf-8")

    O = ["# Step 2: order-flow imbalance (OFI) as a predictor of the next mid move\n",
         "x = signed aggressor volume over the last N s (raw, BTC) or divided by total volume (imbalance). y = next h s mid move (bps). "
         "OLS fitted on the 10 calibration days; on the 10 test days the same line is applied (R² vs predicting no move). "
         "Hit rate = share of seconds where the predicted and actual moves have the same sign (non-zero moves). 95% CIs: bootstrap over days.\n",
         "| N (s) | signal | h (s) | slope (bp per sd of x) | R² calibration | R² test (out of sample) | hit rate cal | hit rate test | slope refitted on test |",
         "|---|---|---|---|---|---|---|---|---|"]
    for _, r in ofi.iterrows():
        O.append(f"| {r.N} | {r.signal} | {r.h} | {r.slope_bp_per_sd:+.4f} | {100 * r.r2_cal:.3f}% [{100 * r.r2_cal_lo:.3f}, {100 * r.r2_cal_hi:.3f}] | "
                 f"{100 * r.r2_test:.3f}% [{100 * r.r2_test_lo:.3f}, {100 * r.r2_test_hi:.3f}] | {r.hit_cal:.3f} | "
                 f"{r.hit_test:.3f} [{r.hit_test_lo:.3f}, {r.hit_test_hi:.3f}] | {r.slope_test_refit_bp_per_sd:+.4f} |")
    O += [f"\n**Main signal** (highest calibration R² averaged over h = 5 and 30 s): N = {main_key[0]} s, {main_key[1]}; "
          f"z = signal / {z_scale:.4f} (its calibration std).\n",
          "## Drift decay: predicted move per unit z at horizon h (bps)\n", "| h (s) | " + " | ".join(map(str, BETA_H)) + " |",
          "|---" * (len(BETA_H) + 1) + "|",
          "| calibration | " + " | ".join(f"{b:.4f}" for b in beta_cal) + " |",
          "| fitted κτ(1−e^(−h/τ)) | " + " | ".join(f"{b:.4f}" for b in decay["fitted"]) + " |",
          "| test | " + " | ".join(f"{b:.4f}" for b in beta_test) + " |",
          f"\nκ = {decay['kappa']:.4f} bp/s per unit z, τ = {decay['tau']:.2f} s (fitted on calibration days).\n",
          "## Fill tilt (0.01 BTC lot rule, calibration days)\n", "| δ (USDT) | θ | R² over z deciles |", "|---|---|---|"]
    for dl, f in tilt.items():
        O.append(f"| {dl} | {f['theta']:.3f} | {f['r2']:.3f} |")
    O.append(f"\nθ used in the simulator: {theta:.3f} (δ = {TILT_MAIN_DELTA} USDT, about the A–S half-spread). "
             f"A 1-sd buying imbalance makes our ask {np.exp(2 * theta):.2f}× as likely to fill as our bid.")
    (OUT / "ofi.md").write_text("\n".join(O) + "\n", encoding="utf-8")

    # ================================================================ charts
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    cmap = plt.get_cmap("viridis")
    for ax, unit, ylab in ((axes[0], "", "adverse move (USDT)"), (axes[1], "bps_", "adverse move (bps of mid)")):
        for b, lab in enumerate(BLABELS):
            col = cmap(b / (len(BLABELS) - 1))
            for split, ls, mkr in (("calibration", "-", "o"), ("test", "--", "s")):
                v = np.array([curves[(split, unit, h)][b] for h in HORIZONS])
                ax.plot(HORIZONS, v[:, 0], ls, marker=mkr, ms=3, color=col, label=f"δ {lab}" if split == "calibration" else None)
                ax.fill_between(HORIZONS, v[:, 1], v[:, 2], color=col, alpha=0.12)
        ax.set_xscale("log"); ax.set_xticks(HORIZONS, [str(h) for h in HORIZONS])
        ax.set_xlabel("seconds after the trade"); ax.set_ylabel(ylab); ax.axhline(0, color="grey", lw=0.6)
    axes[0].legend(fontsize=7, frameon=False, title="solid = calibration, dashed = test", title_fontsize=7)
    fig.suptitle("Markout curves: how far the mid moves against the passive side after a print at distance δ (95% CI over days)")
    fig.tight_layout(); fig.savefig(OUT / "markout_curves.png", dpi=130); plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(17, 4.6))
    ax = axes[0]
    lab = [f"N={r.N} {r.signal}\nh={r.h}" for _, r in ofi.iterrows()]
    xpos = np.arange(len(ofi))
    ax.bar(xpos - 0.2, 100 * ofi.r2_cal, 0.4, color="#1f77b4", label="calibration (in sample)")
    ax.bar(xpos + 0.2, 100 * ofi.r2_test, 0.4, color="#ff7f0e", label="test (out of sample)",
           yerr=[100 * (ofi.r2_test - ofi.r2_test_lo), 100 * (ofi.r2_test_hi - ofi.r2_test)], capsize=2)
    ax.axhline(0, color="grey", lw=0.6); ax.set_xticks(xpos, lab, fontsize=6); ax.set_ylabel("R² (%)")
    ax.set_title("OFI → next mid move: R²"); ax.legend(fontsize=7, frameon=False)
    ax = axes[1]
    ax.bar(xpos - 0.2, ofi.hit_cal, 0.4, color="#1f77b4", label="calibration")
    ax.bar(xpos + 0.2, ofi.hit_test, 0.4, color="#ff7f0e", label="test",
           yerr=[ofi.hit_test - ofi.hit_test_lo, ofi.hit_test_hi - ofi.hit_test], capsize=2)
    ax.axhline(0.5, color="grey", lw=0.6); ax.set_ylim(0.45, 0.62); ax.set_xticks(xpos, lab, fontsize=6)
    ax.set_title("hit rate (sign agreement)"); ax.legend(fontsize=7, frameon=False)
    ax = axes[2]
    ax.plot(BETA_H, beta_cal, "o", color="#1f77b4", label="calibration")
    ax.plot(BETA_H, beta_test, "s", color="#ff7f0e", label="test")
    hh = np.linspace(0.5, 120, 200)
    ax.plot(hh, decay["kappa"] * decay["tau"] * (1 - np.exp(-hh / decay["tau"])), "-", color="#1f77b4",
            label=f"OU fit: κ={decay['kappa']:.3f} bp/s, τ={decay['tau']:.1f} s")
    ax.set_xscale("log"); ax.set_xlabel("horizon h (s)"); ax.set_ylabel("predicted move per unit z (bp)")
    ax.set_title(f"main signal (N={main_key[0]} s {main_key[1]}): slope by horizon"); ax.legend(fontsize=7, frameon=False)
    fig.tight_layout(); fig.savefig(OUT / "ofi_predictive_power.png", dpi=130); plt.close(fig)
    log.info("wrote %s", OUT)


if __name__ == "__main__":
    main()
