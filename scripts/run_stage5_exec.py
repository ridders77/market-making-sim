"""Stage 5 Part A: reinforcement learning for optimal execution (Nevmyvaka, Feng & Kearns 2006) in the replay.

Phases (run in order; each writes its outputs and is safe to re-run):

  cuts       state cut points (OFI, queue, sigma terciles) from the training days -> results/stage5/state_cuts.json
  fixed      fixed actions and TWAP on training-day episodes; the best fixed action per (H, side, fee)
             -> results/stage5/best_fixed.json
  tune       hyperparameter grid (learning rate x exploration decay, checkpoints = number of episodes), scored on
             the validation days only -> results/stage5/tuning.csv, chosen_hyper.json
  train      final agents: 5 seeds x state variants x sides x fees (x horizons) -> data/derived/stage5_agents/
  test       every strategy on the same test-day episodes, 3 cancel rules -> data/derived/stage5_eval/
  report     tables and charts -> results/stage5/

Run:  .venv\\Scripts\\python scripts\\run_stage5_exec.py <phase> [--workers 6]
"""

from __future__ import annotations

import argparse
import itertools
import json
import logging

import numpy as np
import pandas as pd

from stage5_common import (AGENTS, EVALS, FEES, H_MAIN, H_SENS, MAIN_RULE, OUT, QTY, RULES, SIDES, TEST_DAYS,
                           TRAIN_DAYS, VAL_DAYS, VARIANTS, cost_at, drop_days, evaluate_policy, get_day, make_cfg,
                           make_spec, run_pool, setup_logging, variant_name)
from mms.execution.baselines import TWAP, FixedAction
from mms.execution.env import ACTIONS, N_ACTIONS
from mms.execution.qlearning import GreedyPolicy, TabularQ
from mms.execution.training import run_greedy, stratified_starts, train

log = logging.getLogger("stage5")
BUDGET = 160_000
CHECKPOINTS = [2_500, 5_000, 10_000, 20_000, 40_000, 80_000, 160_000]
ALPHAS = [0.01, 0.03, 0.1]
DECAYS = [0.25, 0.5]
TUNE_SEED = 100
SEEDS = [1, 2, 3, 4, 5]
EP_SEED = {"train_eval": 11, "val": 12, "test": 13}
N_TRAIN_EVAL = 1000
SENS_VARIANTS = [(), ("ofi", "queue", "sigma")]


def hkey(H: float) -> str:
    return f"H{int(H)}"


def episodes(split: str, H: float) -> list:
    cfg = make_cfg(H, 1)
    if split == "test":
        return stratified_starts(TEST_DAYS, cfg, EP_SEED["test"])
    if split == "val":
        return stratified_starts(VAL_DAYS, cfg, EP_SEED["val"])
    eps = stratified_starts(TRAIN_DAYS, cfg, EP_SEED["train_eval"])
    rng = np.random.default_rng(EP_SEED["train_eval"])
    idx = np.sort(rng.choice(len(eps), size=min(N_TRAIN_EVAL, len(eps)), replace=False))
    return [eps[i] for i in idx]


# ---------------------------------------------------------------------------------------------- cuts
def cuts() -> None:
    ofi, size_ratio, sig = [], [], []
    for d in TRAIN_DAYS:
        dd = get_day(d)
        i = np.arange(3600 * 10, 86400 * 10 - 1, 100)                 # every 10 s from 01:00
        ofi.append(dd.ofi_z[i])
        size_ratio += [dd.bid_qty[i - 1] / dd.typical_size[i], dd.ask_qty[i - 1] / dd.typical_size[i]]
        sig.append(dd.sigma_hat[i])
        drop_days()
    ofi = np.concatenate(ofi)
    n_nan = int(np.isnan(ofi).sum())
    a = np.abs(ofi[np.isfinite(ofi)])
    sr, sg = np.concatenate(size_ratio), np.concatenate(sig)
    for name, x in (("size ratio", sr), ("sigma", sg)):
        if not np.all(np.isfinite(x)):
            raise ValueError(f"non-finite {name} on the training decision grid")
    out = {"ofi_c": float(np.quantile(a, 1 / 3)), "queue_cuts": [float(q) for q in np.quantile(sr, [1 / 3, 2 / 3])],
           "sigma_cuts": [float(q) for q in np.quantile(sg, [1 / 3, 2 / 3])],
           "n_decision_points": int(len(ofi)), "ofi_nan_share": n_nan / len(ofi), "days": TRAIN_DAYS,
           "note": "ofi_c = 1/3 quantile of |ofi_z| (each tail ~1/3); queue = displayed touch size / 30-min typical "
                   "size, both sides pooled; sigma = sigma_hat (USDT/sqrt(s)). Grid: every 10 s, 01:00-24:00."}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "state_cuts.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    log.info("cuts: %s", out)


# ---------------------------------------------------------------------------------------------- baselines
def baseline_factories(H: float) -> dict:
    n = make_cfg(H, 1).n_steps
    f = {f"fixed_{ACTIONS[a].replace(' ', '')}": (lambda a=a: FixedAction(a)) for a in range(N_ACTIONS)}
    f["twap"] = lambda: TWAP(QTY, n)
    return f


def _fixed_task(t):
    H, side, split, rule = t
    cfg = make_cfg(H, side, "retail", rule)
    rows = []
    for name, fac in baseline_factories(H).items():
        rows += evaluate_policy(fac, cfg, episodes(split, H), {"strategy": name, "H": H, "rule": rule, "fee_run": "retail"})
    return rows


def fixed(workers: int) -> None:
    tasks = [(H, s, "train_eval", MAIN_RULE) for H in (H_MAIN,) + H_SENS for s in SIDES.values()]
    df = pd.DataFrame([r for rows in run_pool(_fixed_task, tasks, workers) for r in rows])
    best, rows = {}, []
    for (H, side), g in df.groupby(["H", "side"]):
        for fee in FEES:
            means = {s: cost_at(gg, fee).mean() for s, gg in g.groupby("strategy") if s.startswith("fixed_")}
            for s, gg in g.groupby("strategy"):
                rows.append({"H": H, "side": side, "fee": fee, "strategy": s, "mean_cost_bps": cost_at(gg, fee).mean(),
                             "passive_share": gg.passive_qty.sum() / (QTY * len(gg)), "n": len(gg)})
            b = min(means, key=means.get)
            best[f"{hkey(H)}|{side}|{fee}"] = {"strategy": b, "mean_cost_bps": means[b]}
    pd.DataFrame(rows).to_csv(OUT / "fixed_actions_train.csv", index=False)
    (OUT / "best_fixed.json").write_text(json.dumps(best, indent=2), encoding="utf-8")
    log.info("best fixed: %s", best)


# ---------------------------------------------------------------------------------------------- tune / train
def _make_eval(spec, cfg, eval_sets):
    def ev(agent):
        out = {}
        for name, eps in eval_sets.items():
            out[f"{name}_cost"] = float(run_greedy(agent, spec, cfg, get_day, eps).mean())
        return out
    return ev


def _train_task(t):
    """t = dict(market, side, fee, alpha, decay, seed, H, stop_at, save)."""
    from stage5_common import setup_logging as _sl
    _sl()
    market, side, fee, H = tuple(t["market"]), t["side"], t["fee"], t["H"]
    spec, cfg = make_spec(market), make_cfg(H, side, fee)
    sets = {"val": episodes("val", H), "train": episodes("train_eval", H)}
    res = train(spec, cfg, get_day, TRAIN_DAYS, BUDGET, t["alpha"], t["decay"], t["seed"], CHECKPOINTS,
                evaluate=_make_eval(spec, cfg, sets), stop_at=t.get("stop_at"))
    if t.get("save"):
        AGENTS.mkdir(parents=True, exist_ok=True)
        ag = res["agent"]
        np.savez(AGENTS / f"{agent_key(market, side, fee, H, t['seed'])}.npz", Q=ag.Q, N=ag.N)
    return [{**{k: v for k, v in t.items() if k != "market"}, "variant": variant_name(market), **row}
            for row in res["curve"]]


def agent_key(market, side, fee, H, seed) -> str:
    return f"{variant_name(market)}_{'buy' if side > 0 else 'sell'}_{fee}_{hkey(H)}_seed{seed}"


def tune(workers: int) -> None:
    tasks = [dict(market=m, side=s, fee=f, alpha=a, decay=dc, seed=TUNE_SEED, H=H_MAIN)
             for m in SENS_VARIANTS for s in SIDES.values() for f in FEES for a in ALPHAS for dc in DECAYS]
    df = pd.DataFrame([r for rows in run_pool(_train_task, tasks, workers) for r in rows])
    df.to_csv(OUT / "tuning.csv", index=False)
    score = df.groupby(["alpha", "decay", "episodes"]).val_cost.mean().rename("mean_val_cost").reset_index()
    best = score.loc[score.mean_val_cost.idxmin()]
    chosen = {"alpha": float(best.alpha), "decay": float(best.decay), "episodes": int(best.episodes),
              "budget": BUDGET, "mean_val_cost": float(best.mean_val_cost),
              "rule": "minimise the mean validation cost over private and all-variable agents x buy/sell x retail/pro "
                      "(H = 1 min, seed 100); the same setting is used for every agent and horizon"}
    (OUT / "chosen_hyper.json").write_text(json.dumps(chosen, indent=2), encoding="utf-8")
    piv = score.pivot_table(index=["alpha", "decay"], columns="episodes", values="mean_val_cost")
    lines = ["# Stage 5: hyperparameter tuning (validation days only)\n",
             f"Validation days {', '.join(VAL_DAYS)}; {len(episodes('val', H_MAIN))} paired episodes per side, H = 1 min, "
             "Q = 0.5 BTC. Cell = mean validation shortfall (bps, fees included) of the greedy policy, averaged over "
             "{private, all variables} × {buy, sell} × {retail, professional}. Learning rate α constant; ε decays "
             f"linearly from 1 to 0.05 over the given share of the {BUDGET:,}-episode budget.\n",
             "| α | ε decay | " + " | ".join(f"{c:,} ep" for c in piv.columns) + " |",
             "|---|---|" + "---|" * len(piv.columns)]
    for (a, dc), r in piv.iterrows():
        cells = [(f"**{v:.3f}**" if (a == chosen["alpha"] and dc == chosen["decay"] and c == chosen["episodes"]) else f"{v:.3f}")
                 for c, v in r.items()]
        lines.append(f"| {a:g} | {dc:g} | " + " | ".join(cells) + " |")
    lines.append(f"\nChosen: α = {chosen['alpha']:g}, ε decay over {chosen['decay']:g} of the budget, "
                 f"stop after {chosen['episodes']:,} episodes.")
    (OUT / "tuning.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    log.info("chosen %s", chosen)


def chosen_hyper() -> dict:
    return json.loads((OUT / "chosen_hyper.json").read_text(encoding="utf-8"))


def train_tasks() -> list:
    ch = chosen_hyper()
    base = dict(alpha=ch["alpha"], decay=ch["decay"], stop_at=ch["episodes"], save=True)
    tasks = [dict(market=m, side=s, fee=f, seed=sd, H=H_MAIN, **base)
             for m in VARIANTS for s in SIDES.values() for f in FEES for sd in SEEDS]
    tasks += [dict(market=m, side=s, fee=f, seed=sd, H=H, **base)
              for H in H_SENS for m in SENS_VARIANTS for s in SIDES.values() for f in FEES for sd in SEEDS]
    return tasks


def train_all(workers: int) -> None:
    tasks = [t for t in train_tasks()
             if not (AGENTS / f"{agent_key(tuple(t['market']), t['side'], t['fee'], t['H'], t['seed'])}.npz").exists()]
    log.info("%d agents to train", len(tasks))
    rows = [r for rows in run_pool(_train_task, tasks, workers) for r in rows]
    path = OUT / "learning_curves.csv"
    old = pd.read_csv(path) if path.exists() else pd.DataFrame()
    pd.concat([old, pd.DataFrame(rows)], ignore_index=True).to_csv(path, index=False)


# ---------------------------------------------------------------------------------------------- test
def load_agent(market, side, fee, H, seed) -> TabularQ:
    z = np.load(AGENTS / f"{agent_key(market, side, fee, H, seed)}.npz")
    return TabularQ(tuple(z["Q"].shape[:-1]), Q=z["Q"].copy(), N=z["N"].copy())


def _test_task(t):
    kind, H, side, rule = t["kind"], t["H"], t["side"], t["rule"]
    eps = episodes("test", H)
    rows = []
    if kind == "baselines":
        cfg = make_cfg(H, side, "retail", rule)
        for name, fac in baseline_factories(H).items():
            rows += evaluate_policy(fac, cfg, eps, {"strategy": name, "H": H, "rule": rule, "fee_run": "retail",
                                                    "variant": "", "seed": -1, "fallbacks": 0, "decisions": 0})
    else:
        market, fee = tuple(t["market"]), t["fee"]
        cfg, spec = make_cfg(H, side, fee, rule), make_spec(market)
        for sd in SEEDS:
            ag = load_agent(market, side, fee, H, sd)
            pol = GreedyPolicy(ag, spec)
            r = evaluate_policy(lambda: pol, cfg, eps, {"strategy": "rl", "H": H, "rule": rule, "fee_run": fee,
                                                        "variant": variant_name(market), "seed": sd})
            for x in r:
                x["fallbacks"], x["decisions"] = ag.fallbacks, ag.decisions   # per-policy totals (same in each row)
            rows += r
    return pd.DataFrame(rows)


def test_key(t) -> str:
    side = "buy" if t["side"] > 0 else "sell"
    if t["kind"] == "baselines":
        return f"baselines_{side}_{hkey(t['H'])}_{t['rule']}"
    return f"rl_{variant_name(tuple(t['market']))}_{side}_{t['fee']}_{hkey(t['H'])}_{t['rule']}"


def test_tasks() -> list:
    tasks = []
    for H in (H_MAIN,) + H_SENS:
        variants = VARIANTS if H == H_MAIN else SENS_VARIANTS
        for s in SIDES.values():
            for rule in RULES:
                tasks.append(dict(kind="baselines", H=H, side=s, rule=rule))
                tasks += [dict(kind="rl", H=H, side=s, rule=rule, market=m, fee=f) for m in variants for f in FEES]
    return tasks


def _test_and_save(t):
    df = _test_task(t)
    EVALS.mkdir(parents=True, exist_ok=True)
    p = EVALS / f"{test_key(t)}.parquet"
    tmp = p.with_suffix(".part")
    df.to_parquet(tmp, index=False)
    tmp.replace(p)
    return test_key(t)


def test(workers: int) -> None:
    tasks = [t for t in test_tasks() if not (EVALS / f"{test_key(t)}.parquet").exists()]
    log.info("%d test tasks", len(tasks))
    run_pool(_test_and_save, tasks, workers)


def main() -> None:
    setup_logging()
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["cuts", "fixed", "tune", "train", "test", "report"])
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    if a.phase == "cuts":
        cuts()
    elif a.phase == "fixed":
        fixed(a.workers)
    elif a.phase == "tune":
        tune(a.workers)
    elif a.phase == "train":
        train_all(a.workers)
    elif a.phase == "test":
        test(a.workers)
    else:
        from stage5_report import report
        report()


if __name__ == "__main__":
    main()
