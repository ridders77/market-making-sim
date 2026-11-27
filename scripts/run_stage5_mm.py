"""Stage 5 Part B (WHAT-IF), conditional step: a small tabular Q-learning market maker at a queue share f where
quoting at the touch is clearly profitable (pre-registered condition in DECISIONS.md).

Setup (stage 4's replay, the only change being queue_frac = f): 0.01 BTC lot, +-10-lot limit, proportional
cancel rule, 100 ms latency, 1-hour episodes marked to the mid at the end, maker fee FEE.

Agent. Decision every 1 s (the chosen quotes are held for 10 buckets of 100 ms; an unchanged price keeps its
queue place). Actions: stage 4's 16 joint actions {touch, 1 behind, 3 behind, off} per side. Actions that could
breach the limit are masked: at q >= +10 lots no bid, at q <= -10 lots no ask. State (15 cells): inventory bucket
(<= -6, -5..-2, -1..1, 2..5, >= 6 lots) x 5 s OFI bucket (z < -c, |z| <= c, z > c; c from state_cuts.json).
Reward per decision: the replay's rewards over the 10 buckets = change in mark-to-market wealth minus the running
inventory penalty (gamma/2) lot q_lots^2 sigma_hat^2 dt, with gamma set so that holding the limit for an hour
costs the 60 s markout of the fills that got there (measured at this f on the calibration days).
Q-learning, discount 0.99 per decision (about a 100 s horizon: time is not in the state, the task is treated as
continuing), alpha = 0.05, epsilon linear 1 -> 0.05 over the first half of N_EPISODES, Q initialised at 0.
Training: random 1-hour episodes on the 8 training days (seeds 1-5); test: stage 4's 230 test hours.

Phases: gamma | train | test | report
"""

from __future__ import annotations

import argparse
import json
import logging

import numpy as np
import pandas as pd

from stage5_common import DERIVED, OUT, TRAIN_DAYS, load_cuts, run_pool, setup_logging
from stage4_common import CAL_DAYS, EP_STEPS, HOURS, TEST_DAYS, episode_markouts, make_policy, pnl_at_fee
from mms.analysis.metrics import bootstrap_ci, paired_difference
from mms.replay.data import load_day
from mms.replay.env import ReplayConfig, ReplayEnv, run_episode
from mms.replay.rl_env import N_ACTIONS, action_to_prices, decode_action

log = logging.getLogger("stage5")
MM = DERIVED / "stage5_mm"
RULE, LAT, LOT, LIMIT = "proportional", 100, 0.01, 10
HOLD = 10                           # buckets per decision (1 s)
DISCOUNT, ALPHA, N_EPISODES = 0.99, 0.05, 300
SEEDS = [1, 2, 3, 4, 5]
N_BOOT = 2000


def setting() -> dict:
    return json.loads((OUT / "mm_setting.json").read_text(encoding="utf-8"))


def cfg(f: float, gamma: float, fee: float) -> ReplayConfig:
    return ReplayConfig(latency_ms=LAT, cancel_rule=RULE, lot=LOT, fee_bps=fee, gamma=gamma, queue_frac=f,
                        inv_limit_lots=LIMIT)


def inv_bucket(q_lots: float) -> int:
    q = round(q_lots)
    return 0 if q <= -6 else 1 if q <= -2 else 2 if q <= 1 else 3 if q <= 5 else 4


def ofi_bucket(z: float, c: float) -> int:
    if not np.isfinite(z):
        return 1
    return 0 if z < -c else 2 if z > c else 1


def allowed(q_lots: float) -> np.ndarray:
    ok = np.ones(N_ACTIONS, bool)
    for a in range(N_ACTIONS):
        b, s = decode_action(a)
        if q_lots >= LIMIT - 1e-9 and b is not None:
            ok[a] = False
        if q_lots <= -LIMIT + 1e-9 and s is not None:
            ok[a] = False
    return ok


def run_mm_episode(env: ReplayEnv, start: int, Q: np.ndarray | None, c: float, rng=None, eps: float = 0.0,
                   learn: bool = False):
    obs = env.reset(start)
    done = False
    s, a, acc = None, None, 0.0
    k = 0
    while not done:
        if k % HOLD == 0:
            s_new = (inv_bucket(obs["inventory"]), ofi_bucket(obs["ofi_z"], c))
            ok = allowed(obs["inventory"])
            if learn and s is not None:
                Q[s + (a,)] += ALPHA * (acc + DISCOUNT * np.max(np.where(ok, Q[s_new], -np.inf)) - Q[s + (a,)])
            if Q is None:
                a_new = 0
            elif rng is not None and rng.random() < eps:
                a_new = int(rng.choice(np.flatnonzero(ok)))
            else:
                a_new = int(np.argmax(np.where(ok, Q[s_new], -np.inf)))
            s, a, acc = s_new, a_new, 0.0
            prices = action_to_prices(a, obs)
        obs, r, done, _ = env.step(prices)
        acc += r
        k += 1
        if not done and k % HOLD != 0:
            prices = _hold(prices, obs, a)
    if learn and s is not None:
        Q[s + (a,)] += ALPHA * (acc - Q[s + (a,)])
    return env.result()


def _hold(prices, obs, a):
    """Within a decision the action is held: the same *prices* are re-sent (keeping queue places); a side whose
    order was filled re-joins at the same price, as the touch strategies do on their next decision."""
    return prices


# ---------------------------------------------------------------------------------------------- phases
def _touch_cal_task(t):
    f, day = t
    d = load_day(day)
    out = []
    for h in HOURS:
        r = run_episode(make_policy({"name": "touch"}), d, h * EP_STEPS, cfg(f, 0.0, 0.0))
        out.append({**episode_markouts(r, d, h * EP_STEPS)["sums"], "sigma_hat0": float(d.sigma_hat[h * EP_STEPS])})
    return out


def gamma_phase(f: float, fee: float, workers: int) -> None:
    rows = [r for rs in run_pool(_touch_cal_task, [(f, d) for d in CAL_DAYS], workers) for r in rs]
    df = pd.DataFrame(rows)
    m = df.mk60_sum.sum() / df.mk_qty.sum()
    s2 = float((df.sigma_hat0 ** 2).mean())
    gamma = LIMIT * LOT * m / (0.5 * LOT * LIMIT ** 2 * s2 * 3600)
    out = {"f": f, "fee_bps": fee, "markout60_usdt_per_btc": m, "mean_sigma2": s2, "gamma_per_lot": gamma,
           "penalty_hour_at_limit_usdt": 0.5 * gamma * LOT * LIMIT ** 2 * s2 * 3600,
           "markout_cost_of_10_lots_usdt": LIMIT * LOT * m}
    (OUT / "mm_setting.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    log.info("MM setting: %s", out)


def _train_task(seed):
    st = setting()
    c = load_cuts()["ofi_c"]
    rng = np.random.default_rng(seed)
    Q = np.zeros((5, 3, N_ACTIONS))
    curve = []
    days = {d: load_day.__wrapped__(d) for d in TRAIN_DAYS}
    envs = {d: ReplayEnv(dd, cfg(st["f"], st["gamma_per_lot"], st["fee_bps"])) for d, dd in days.items()}
    for ep in range(N_EPISODES):
        eps = max(0.05, 1 - 0.95 * ep / (N_EPISODES / 2))
        d = TRAIN_DAYS[int(rng.integers(len(TRAIN_DAYS)))]
        start = int(rng.integers(3600 * 10, 23 * 3600 * 10 + 1))
        r = run_mm_episode(envs[d], start, Q, c, rng, eps, learn=True)
        curve.append({"seed": seed, "episode": ep, "pnl": r.pnl, "epsilon": eps})
    MM.mkdir(parents=True, exist_ok=True)
    np.save(MM / f"Q_seed{seed}.npy", Q)
    return curve


def _test_task(t):
    kind, day = t
    st = setting()
    c = load_cuts()["ofi_c"]
    d = load_day(day)
    rows = []
    for h in HOURS:
        start = h * EP_STEPS
        if kind == "rl":
            for sd in SEEDS:
                Q = np.load(MM / f"Q_seed{sd}.npy")
                r = run_mm_episode(ReplayEnv(d, cfg(st["f"], st["gamma_per_lot"], st["fee_bps"])), start, Q, c)
                rows.append({"strategy": f"rl_seed{sd}", "day": day, "hour": h, "pnl": r.pnl, "fee_run": st["fee_bps"],
                             "notional": r.notional, "fill_qty": r.fill_qty, "max_abs_q": r.max_abs_q,
                             "penalty": r.penalty})
    return rows


def train_phase(workers: int) -> None:
    curves = [r for rs in run_pool(_train_task, SEEDS, workers) for r in rs]
    pd.DataFrame(curves).to_csv(OUT / "mm_training_curve.csv", index=False)


def test_phase(workers: int) -> None:
    rows = [r for rs in run_pool(_test_task, [("rl", d) for d in TEST_DAYS], workers) for r in rs]
    pd.DataFrame(rows).to_parquet(MM / "test_rl.parquet", index=False)


def report_phase() -> None:
    from run_stage5_queue import load, specs_by_fee
    st = setting()
    f, fee = st["f"], st["fee_bps"]
    rl = pd.read_parquet(MM / "test_rl.parquet").sort_values(["day", "hour"])
    per = np.column_stack([rl[rl.strategy == f"rl_seed{s}"].pnl.to_numpy() for s in SEEDS])
    rl_mean = per.mean(axis=1)
    sp = specs_by_fee()
    rows = [("RL market maker (mean of 5 seeds)", rl_mean, rl[rl.strategy == "rl_seed1"])]
    base = {}
    for name in ("touch", "touch_skew"):
        df = load(sp[(name, fee)], f).sort_values(["day", "hour"])
        base[name] = pnl_at_fee(df, fee)
        rows.append(({"touch": "always at touch", "touch_skew": "touch + inventory skew"}[name], base[name], df))
    M = [f"# Stage 5 Part B — WHAT-IF: tabular Q-learning market maker at f = {f:g}, maker fee {fee:+g} bp\n",
         "**What-if, not achievable for a retail participant** (front-of-queue priority). Test days, 230 paired hours, "
         f"proportional rule, 100 ms latency, 0.01 BTC lot, ±10 lots (masked). Inventory penalty γ = {st['gamma_per_lot']:.2e} "
         f"per lot (holding 10 lots for an hour costs {st['penalty_hour_at_limit_usdt']:.3f} USDT = the 60 s markout of 10 lots "
         f"of always-at-touch fills at this f on the calibration days, {st['markout60_usdt_per_btc']:.2f} USDT/BTC). "
         "P&L per hour (USDT) excludes the penalty; mean [95% CI over hours].\n",
         "| strategy | mean P&L / h | BTC / h | mean max |q| (lots) |", "|---|---|---|---|"]
    for lab, x, df in rows:
        est = bootstrap_ci(x, np.mean, N_BOOT, seed=8)
        mx = df.max_abs_q.mean() / LOT
        M.append(f"| {lab} | {est[0]:.2f} [{est[1]:.2f}, {est[2]:.2f}] | {df.fill_qty.mean():.1f} | {mx:.1f} |")
    M.append("\nPaired differences (RL − baseline, USDT/h):\n")
    for name, b in base.items():
        d = paired_difference(rl_mean, b, np.mean, N_BOOT, seed=9)
        M.append(f"- vs {name}: {d[0]:.2f} [{d[1]:.2f}, {d[2]:.2f}]")
    seed_means = per.mean(axis=0)
    M.append(f"\nPer-seed mean P&L / h: " + ", ".join(f"{m:.1f}" for m in seed_means))
    pol = []
    for sd in SEEDS:
        Q = np.load(MM / f"Q_seed{sd}.npy")
        pol.append([[int(np.argmax(np.where(allowed({0: -10, 1: -3, 2: 0, 3: 3, 4: 10}[i]), Q[i, j], -np.inf)))
                     for j in range(3)] for i in range(5)])
    from mms.replay.rl_env import LEVELS
    M.append("\nGreedy action (bid level / ask level) by inventory bucket × OFI bucket, seed 1 … 5:\n")
    M.append("| inventory | sell flow (z < −c) | neutral | buy flow (z > c) |")
    M.append("|---|---|---|---|")
    for i, lab in enumerate(["≤ −6", "−5…−2", "−1…1", "2…5", "≥ 6"]):
        cells = []
        for j in range(3):
            acts = [pol[s][i][j] for s in range(len(SEEDS))]
            cells.append("; ".join(f"{LEVELS[a // 4]}/{LEVELS[a % 4]}" for a in acts))
        M.append(f"| {lab} | " + " | ".join(cells) + " |")
    (OUT / "mm_whatif.md").write_text("\n".join(M) + "\n", encoding="utf-8")


def main() -> None:
    setup_logging()
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["gamma", "train", "test", "report"])
    ap.add_argument("--f", type=float)
    ap.add_argument("--fee", type=float, default=-0.5)
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    if a.phase == "gamma":
        gamma_phase(a.f, a.fee, a.workers)
    elif a.phase == "train":
        train_phase(a.workers)
    elif a.phase == "test":
        test_phase(a.workers)
    else:
        report_phase()


if __name__ == "__main__":
    main()
