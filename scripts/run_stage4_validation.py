"""Stage 4, Step 2: validate the stage 3 synthetic simulator against the replay, hour by hour on the test days.

Strategy: stage 3's markout-aware A-S (quotes >= 0.1 USDT from the mid), 0.01 BTC lot, gamma = 1e-4 per lot.

For every test-day hour (10 days x 23 hours) both models get the same inputs, all known at the start of the hour:
sigma = trailing sigma_hat, S0 = the mid, A and k from stage 3's model (b) at sigma_hat, and the stage 3 flow
calibration (OFI drift kappa and tau, fill tilt theta, markout jumps J in bps of the mid; jump noise interpolated
between the regimes' values by sigma).

* Replay: the hour replayed against the recorded book and trades (scripts/run_stage4_replay.py tasks), under
  each cancel rule, and with the "front" convention for unseen depth as an upper bound on fills behind the touch.
* Synthetic prediction: the stage 3 environment with informed flow x1, 200 paths (seeds 40,000-40,199,
  the same for every hour), dt = 0.2 s or smaller so that A dt < 0.1.

Quantities compared: BTC filled per hour, 60 s markout per BTC (fills in the first 59 minutes), P&L per hour.
Ratio = replay / synthetic as a ratio of sums over hours; 95% CI by bootstrapping hours.

Run:  .venv\\Scripts\\python scripts\\run_stage4_validation.py [--workers N]
"""

from __future__ import annotations

import argparse
import logging
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from stage4_common import (FEES, HORIZONS, JUMP_DELTA, JUMP_MEAN_BPS, MODEL_B, OUT, REPO, RULES, TEST_DAYS, _CAL, _FLOW,
                           GAMMA, load_runs, run_task)
from mms.sim.env import EnvConfig, FlowParams, MarketMakingEnv
from mms.sim.simulator import MarketParams
from mms.strategies.flow_aware import FlowAwareAvellanedaStoikov

SYN = REPO / "data" / "derived" / "stage4_synth"
SEEDS = list(range(40_000, 40_200))
LOT, TICK, T = 0.01, 0.1, 3600.0
N_BOOT = 2000
log = logging.getLogger("stage4_validation")

_REG = _FLOW["regimes"]
_REG_SIGMA = np.array([_CAL["regimes"][r]["sigma"] for r in ("low", "medium", "high")])
_REG_JSD = np.array([_REG[r]["jump_sd"] for r in ("low", "medium", "high")])


def hour_params(sigma: float, s0: float) -> dict:
    A = MODEL_B["a"] * sigma ** MODEL_B["b"]
    k = MODEL_B["c"] * sigma ** MODEL_B["d"]
    dt = 0.2
    while A * dt >= 0.1:
        dt /= 2
    kappa = _FLOW["kappa_bp_per_s"] * s0 / 1e4
    return {"A": A, "k": k, "dt": dt, "kappa": kappa, "tau": _FLOW["tau_s"], "theta": _FLOW["theta"],
            "jump_mean": tuple(np.array(JUMP_MEAN_BPS) * s0 / 1e4), "jump_sd": float(np.interp(sigma, _REG_SIGMA, _REG_JSD))}


def synth_hour(sigma: float, s0: float, fee: float) -> dict:
    hp = hour_params(sigma, s0)
    flow = FlowParams(kappa=hp["kappa"], tau=hp["tau"], theta=hp["theta"], jump_delta=tuple(JUMP_DELTA),
                      jump_mean=hp["jump_mean"], jump_sd=hp["jump_sd"])
    if sigma**2 <= 2 * hp["kappa"] ** 2 * hp["tau"]:
        raise ValueError(f"sigma {sigma:.3f} too small for the drift variance at S0 {s0:.0f}")
    mp = MarketParams(s0=s0, sigma=sigma, T=T, dt=hp["dt"], A=hp["A"], k=hp["k"], fill_rule="exact")
    cfg = EnvConfig(mp, lot=LOT, tick_size=TICK, maker_fee_bps=fee, flow=flow, min_distance=0.1)
    strat = FlowAwareAvellanedaStoikov(gamma=GAMMA, sigma=sigma, k=hp["k"], T=T, fee_bps=fee, kappa=hp["kappa"],
                                       tau_x=hp["tau"], jump_delta=tuple(JUMP_DELTA), jump_mean=hp["jump_mean"],
                                       s_ref=s0, use_signal=False, name="markout_aware_as")
    env = MarketMakingEnv(cfg)
    obs = env.reset(SEEDS)
    N, n = mp.n_steps, len(SEEDS)
    mids = np.empty((n, N + 1))
    mids[:, 0] = obs["mid"]
    nb = np.zeros((n, N), dtype=bool)
    na = np.zeros((n, N), dtype=bool)
    done, i = False, 0
    while not done:
        obs, _, done, info = env.step(strat.act(obs))
        nb[:, i], na[:, i] = info["bid_hit"], info["ask_hit"]
        mids[:, i + 1] = obs["mid"]
        i += 1
    res = env.result()
    hsteps = int(round(60 / hp["dt"]))
    valid = np.arange(N) < N - hsteps
    mk_b = np.zeros((n, N))                               # we bought at step i: adverse if the mid falls
    mk_b[:, :N - hsteps] = -(mids[:, hsteps:N] - mids[:, :N - hsteps])
    mk_sum = (nb * valid * mk_b).sum() - (na * valid * mk_b).sum()      # ask: -(+) sign flip
    mk_qty = (nb * valid).sum() + (na * valid).sum()
    return {"syn_fill_btc": LOT * res.n_fills.mean(), "syn_pnl": res.pnl.mean(), "syn_pnl_sd": res.pnl.std(ddof=1),
            "syn_mk60_sum": LOT * mk_sum / n, "syn_mk_qty": LOT * mk_qty / n, "dt": hp["dt"], "A": hp["A"], "k": hp["k"]}


def synth_task(task) -> str:
    day, fee = task
    path = SYN / f"{day}_fee{fee:+g}.parquet"
    if path.exists():
        return path.name
    rep = load_runs({"name": "markout_as", "fee": fee}, "proportional", 100, LOT, [day])
    rows = [{"day": day, "hour": int(r.hour), "fee": fee, "sigma_hat0": r.sigma_hat0, "mid0": r.mid0,
             **synth_hour(r.sigma_hat0, r.mid0, fee)} for r in rep.itertuples()]
    SYN.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".part")
    pd.DataFrame(rows).to_parquet(tmp, index=False)
    tmp.replace(path)
    return path.name


def boot_ratio(num, den, seed=0):
    num, den = np.asarray(num, float), np.asarray(den, float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(num), (N_BOOT, len(num)))
    b = num[idx].sum(1) / den[idx].sum(1)
    return float(num.sum() / den.sum()), *np.quantile(b, [0.025, 0.975])


def boot_markout_ratio(rs, rq, ss, sq, seed=0):
    rs, rq, ss, sq = (np.asarray(x, float) for x in (rs, rq, ss, sq))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(rs), (N_BOOT, len(rs)))
    b = (rs[idx].sum(1) / rq[idx].sum(1)) / (ss[idx].sum(1) / sq[idx].sum(1))
    return float((rs.sum() / rq.sum()) / (ss.sum() / sq.sum())), *np.quantile(b, [0.025, 0.975])


def boot_diff(a, b, seed=0):
    d = np.asarray(a, float) - np.asarray(b, float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), (N_BOOT, len(d)))
    return float(d.mean()), *np.quantile(d[idx].mean(1), [0.025, 0.975])


def fmt(t, nd=2):
    return f"{t[0]:.{nd}f} [{t[1]:.{nd}f}, {t[2]:.{nd}f}]"


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=14)
    a = ap.parse_args()
    # replay runs needed (most are shared with run_stage4_replay.py test): every rule x fee, plus unseen = front
    tasks = [({"name": "markout_as", "fee": f}, r, 100, LOT, d, "back", True) for f in FEES for r in RULES for d in TEST_DAYS]
    tasks += [({"name": "markout_as", "fee": 0.0}, r, 100, LOT, d, "front", False) for r in RULES for d in TEST_DAYS]
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        list(ex.map(run_task, tasks, chunksize=1))
        log.info("replay runs ready; synthetic predictions...")
        for i, name in enumerate(ex.map(synth_task, [(d, f) for f in FEES for d in TEST_DAYS], chunksize=1), 1):
            log.info("synthetic %d/%d %s", i, len(FEES) * len(TEST_DAYS), name)

    syn = pd.concat([pd.read_parquet(SYN / f"{d}_fee{f:+g}.parquet") for f in FEES for d in TEST_DAYS], ignore_index=True)
    rows = []
    for fee in FEES:
        s = syn[syn.fee == fee].set_index(["day", "hour"]).sort_index()
        for rule in RULES:
            for unseen in (["back", "front"] if fee == 0.0 else ["back"]):
                rep = load_runs({"name": "markout_as", "fee": fee}, rule, 100, LOT, TEST_DAYS, unseen)
                rep = rep.set_index(["day", "hour"]).sort_index()
                assert rep.index.equals(s.index)
                rows.append({
                    "fee": fee, "rule": rule, "unseen_depth": unseen, "n_hours": len(rep),
                    "replay_fill_btc_h": rep.fill_qty.mean(), "synthetic_fill_btc_h": s.syn_fill_btc.mean(),
                    "fill_ratio": boot_ratio(rep.fill_qty, s.syn_fill_btc, 1),
                    "replay_mk60": rep.mk60_sum.sum() / rep.mk_qty.sum(), "synthetic_mk60": s.syn_mk60_sum.sum() / s.syn_mk_qty.sum(),
                    "mk60_ratio": boot_markout_ratio(rep.mk60_sum, rep.mk_qty, s.syn_mk60_sum, s.syn_mk_qty, 2),
                    "replay_pnl_h": rep.pnl.mean(), "synthetic_pnl_h": s.syn_pnl.mean(),
                    "pnl_ratio": boot_ratio(rep.pnl, s.syn_pnl, 3), "pnl_diff": boot_diff(rep.pnl, s.syn_pnl, 4),
                    "replay_share_through": rep.qty_through.sum() / rep.fill_qty.sum(),
                    "corr_fill_by_hour": float(np.corrcoef(rep.fill_qty, s.syn_fill_btc)[0, 1]),
                })
    t = pd.DataFrame(rows)
    t.to_csv(OUT / "validation_replay_vs_synthetic.csv", index=False)
    s0 = syn[syn.fee == 0.0]
    M = ["# Step 2: replay vs the stage 3 synthetic simulator (markout-aware A–S, test days)\n",
         "Same strategy, same hours (10 test days × 23 hours), same inputs known at the start of each hour (trailing σ̂, mid, "
         "model (b) A and k at σ̂, stage 3 flow calibration). Synthetic = mean over 200 simulated paths per hour. "
         "Ratio = replay / synthetic (ratio of sums over hours), 95% bootstrap CI over hours. 60 s markout in USDT per BTC, "
         "fills in the first 59 minutes. *unseen depth = front* treats every print at our price behind the touch as filling us "
         "(upper bound); *back* (main) assumes we are behind everything until the level becomes best.\n",
         f"Synthetic dt: {sorted(s0.dt.unique())} s (smaller where A·dt would reach 0.1); A per hour {s0.A.min():.3f}–{s0.A.max():.3f} /s, "
         f"k {s0.k.min():.3f}–{s0.k.max():.3f} /USDT.\n",
         "| fee | cancel rule | unseen depth | BTC filled/h replay | synthetic | ratio [CI] | markout 60 s replay | synthetic | ratio [CI] | P&L/h replay | synthetic | ratio [CI] | difference [CI] | replay fills by sweeps through |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for _, r in t.iterrows():
        M.append(f"| {r.fee:+g} | {r.rule} | {r.unseen_depth} | {r.replay_fill_btc_h:.4f} | {r.synthetic_fill_btc_h:.4f} | {fmt(r.fill_ratio)} | "
                 f"{r.replay_mk60:.2f} | {r.synthetic_mk60:.2f} | {fmt(r.mk60_ratio)} | {r.replay_pnl_h:.3f} | {r.synthetic_pnl_h:.3f} | "
                 f"{fmt(r.pnl_ratio)} | {fmt(r.pnl_diff, 3)} | {100 * r.replay_share_through:.0f}% |")
    (OUT / "validation_replay_vs_synthetic.md").write_text("\n".join(M) + "\n", encoding="utf-8")

    # chart: per-hour fills and P&L, replay (proportional) vs synthetic, fee 0
    rep = load_runs({"name": "markout_as", "fee": 0.0}, "proportional", 100, LOT, TEST_DAYS).set_index(["day", "hour"]).sort_index()
    s = s0.set_index(["day", "hour"]).sort_index()
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    ax = axes[0]
    ax.scatter(s.syn_fill_btc, rep.fill_qty, s=10, alpha=0.6)
    lim = max(s.syn_fill_btc.max(), rep.fill_qty.max()) * 1.05
    ax.plot([0, lim], [0, lim], color="black", lw=0.7)
    ax.set_xlabel("synthetic prediction (BTC filled per hour)"); ax.set_ylabel("replay (BTC filled per hour)")
    ax.set_title("Fills per hour (each dot = one test hour)", fontsize=10)
    ax = axes[1]
    ax.scatter(s.syn_pnl, rep.pnl, s=10, alpha=0.6, color="#ff7f0e")
    lo_, hi_ = min(s.syn_pnl.min(), rep.pnl.min()), max(s.syn_pnl.max(), rep.pnl.max())
    ax.plot([lo_, hi_], [lo_, hi_], color="black", lw=0.7)
    ax.set_xlabel("synthetic prediction (mean P&L, USDT/h)"); ax.set_ylabel("replay P&L (USDT/h)")
    ax.set_title("P&L per hour", fontsize=10)
    ax = axes[2]
    sub = t[(t.fee == 0.0)]
    labels = [f"{r.rule[:4]}\n{r.unseen_depth}" for _, r in sub.iterrows()]
    x = np.arange(len(sub))
    for j, (col, name, c) in enumerate((("fill_ratio", "fills", "#1f77b4"), ("mk60_ratio", "60 s markout", "#d62728"))):
        v = np.array([r[col] for _, r in sub.iterrows()])
        ax.bar(x + (j - 0.5) * 0.38, v[:, 0], 0.38, yerr=[v[:, 0] - v[:, 1], v[:, 2] - v[:, 0]], capsize=2, color=c, label=name)
    ax.axhline(1, color="black", lw=0.7)
    ax.set_xticks(x, labels, fontsize=8); ax.set_ylabel("replay / synthetic"); ax.legend(fontsize=8, frameon=False)
    ax.set_title("Ratios with 95% CI (fee 0)", fontsize=10)
    fig.suptitle("Markout-aware A–S on the test days: replay vs the stage 3 synthetic simulator", fontsize=11)
    fig.tight_layout(); fig.savefig(OUT / "validation_replay_vs_synthetic.png", dpi=130); plt.close(fig)
    log.info("written %s", OUT / "validation_replay_vs_synthetic.md")


if __name__ == "__main__":
    main()
