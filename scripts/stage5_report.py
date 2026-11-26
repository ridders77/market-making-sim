"""Stage 5 Part A report: cost tables, value of each state variable, policy heatmaps, learning curves, markouts.

Reads data/derived/stage5_eval/ (test-day runs) and results/stage5/learning_curves.csv; writes results/stage5/.
Headline RL variant: pre-registered rule (DECISIONS.md) = the state variant with the lowest mean *validation*
cost at the chosen checkpoint, averaged over seeds, sides and fee scenarios (H = 1 min).
"""

from __future__ import annotations

import json
import logging

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap

from stage5_common import (EVALS, FEES, H_MAIN, H_SENS, MAIN_RULE, OUT, QTY, RULES, SIDES, VARIANTS, cost_at,
                           variant_name)
from mms.analysis.metrics import bootstrap_ci, paired_difference
from mms.execution.env import ACTIONS

log = logging.getLogger("stage5")
N_BOOT = 2000
SEEDS = [1, 2, 3, 4, 5]
BASELINES = {"fixed_cross": "immediate taker", "twap": "TWAP taker slices", "fixed_touch": "rest at touch, then cross",
             "fixed_1behind": "fixed: 1 tick behind", "fixed_3behind": "fixed: 3 ticks behind"}
FEE_LABEL = {"retail": "retail (maker 2 / taker 5 bp)", "pro": "professional (maker −0.5 / taker 2 bp)"}
SIDE_LABEL = {1: "buy", -1: "sell"}
# validated default categorical order (dataviz palette): blue, orange, aqua, yellow, magenta, green, violet
CAT = ["#257de0", "#f47230", "#00b894", "#f2a600", "#ee7fac", "#008a17", "#6a4ce2", "#ec4647"]
ACTION_COLORS = CAT[:4]
UNVISITED = "#d9d9d9"
TEXT2 = "#666666"


def hkey(H) -> str:
    return f"H{int(H)}"


def fmt(t, nd=2) -> str:
    return f"{t[0]:.{nd}f} [{t[1]:.{nd}f}, {t[2]:.{nd}f}]"


# ---------------------------------------------------------------------------------------------- loading
def load_block(H, side, rule, fee, variants) -> dict:
    """Per-episode costs (bps) for every strategy on the same episodes, keyed by strategy name.
    RL: 'rl_<variant>' = mean over the 5 seeds per episode; 'rl_<variant>_seed<k>' per seed."""
    s = SIDE_LABEL[side]
    b = pd.read_parquet(EVALS / f"baselines_{s}_{hkey(H)}_{rule}.parquet")
    out, meta = {}, {}
    ref = None
    for name, g in b.groupby("strategy"):
        g = g.sort_values(["day", "start"])
        if ref is None:
            ref = g[["day", "start"]].to_numpy()
        elif not (g[["day", "start"]].to_numpy() == ref).all():
            raise AssertionError("baseline episodes are not paired")
        out[name] = cost_at(g, fee)
        meta[name] = g
    for m in variants:
        v = variant_name(m)
        r = pd.read_parquet(EVALS / f"rl_{v}_{s}_{fee}_{hkey(H)}_{rule}.parquet")
        per = []
        for sd in SEEDS:
            g = r[r.seed == sd].sort_values(["day", "start"])
            if not (g[["day", "start"]].to_numpy() == ref).all():
                raise AssertionError("RL episodes are not paired with the baselines")
            c = cost_at(g, fee)
            if not np.allclose(c, g.cost_bps.to_numpy()):
                raise AssertionError("repricing check failed")
            out[f"rl_{v}_seed{sd}"] = c
            meta[f"rl_{v}_seed{sd}"] = g
            per.append(c)
        out[f"rl_{v}"] = np.mean(per, axis=0)
        meta[f"rl_{v}"] = pd.concat([meta[f"rl_{v}_seed{sd}"] for sd in SEEDS], ignore_index=True)
    return {"cost": out, "meta": meta, "episodes": ref}


def headline_variant() -> str:
    lc = pd.read_csv(OUT / "learning_curves.csv")
    ch = json.loads((OUT / "chosen_hyper.json").read_text(encoding="utf-8"))
    last = lc[(lc.H == H_MAIN) & (lc.episodes == ch["episodes"])]
    return last.groupby("variant").val_cost.mean().idxmin()


def day_block_ci(days: np.ndarray, x: np.ndarray, seed=3):
    u = np.unique(days)
    sums = np.array([x[days == d].sum() for d in u])
    cnt = np.array([(days == d).sum() for d in u])
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(u), (N_BOOT, len(u)))
    b = sums[idx].sum(1) / cnt[idx].sum(1)
    return float(x.mean()), *[float(v) for v in np.quantile(b, [0.025, 0.975])]


def best_fixed(H, side, fee) -> str:
    bf = json.loads((OUT / "best_fixed.json").read_text(encoding="utf-8"))
    return bf[f"{hkey(H)}|{int(side)}|{fee}"]["strategy"]


# ---------------------------------------------------------------------------------------------- tables
def strategy_label(k: str) -> str:
    if k in BASELINES:
        return BASELINES[k]
    return "RL: " + k[3:].replace("private", "private (time × inventory)")


def cost_tables(head: str) -> pd.DataFrame:
    rows, diffs = [], []
    for H in (H_MAIN,) + H_SENS:
        variants = VARIANTS if H == H_MAIN else [(), ("ofi", "queue", "sigma")]
        for side in SIDES.values():
            for fee in FEES:
                bfix = best_fixed(H, side, fee)
                for rule in RULES:
                    blk = load_block(H, side, rule, fee, variants)
                    C, meta = blk["cost"], blk["meta"]
                    keys = list(BASELINES) + [f"rl_{variant_name(m)}" for m in variants]
                    for k in keys:
                        g = meta[k]
                        est = bootstrap_ci(C[k], np.mean, N_BOOT, seed=1)
                        n_ep = len(g)
                        rows.append({"H": H, "side": SIDE_LABEL[side], "fee": fee, "rule": rule, "strategy": k,
                                     "mean": est[0], "lo": est[1], "hi": est[2], "std": float(np.std(C[k], ddof=1)),
                                     "passive_share": g.passive_qty.sum() / (QTY * n_ep),
                                     "cleanup_episodes": float((g.cleanup_qty > 1e-12).mean()),
                                     "cleanup_beyond_display": float(g.beyond_cleanup.mean()),
                                     "any_beyond_display": float((g.beyond > 0).mean()),
                                     "best_fixed": k == bfix, "n": len(C[k])})
                    for v in [f"rl_{variant_name(m)}" for m in variants]:
                        for bk in list(BASELINES):
                            d = paired_difference(C[v], C[bk], np.mean, N_BOOT, seed=2)
                            diffs.append({"H": H, "side": SIDE_LABEL[side], "fee": fee, "rule": rule, "rl": v,
                                          "baseline": bk, "diff": d[0], "lo": d[1], "hi": d[2],
                                          "best_fixed": bk == bfix})
    t, d = pd.DataFrame(rows), pd.DataFrame(diffs)
    t.to_csv(OUT / "execution_cost.csv", index=False)
    d.to_csv(OUT / "execution_paired_diffs.csv", index=False)
    return t, d


def md_cost(t: pd.DataFrame, d: pd.DataFrame, head: str) -> list[str]:
    M = ["# Stage 5 Part A: execution cost by strategy (test days)\n",
         f"Buy or sell Q = {QTY} BTC within H, decisions every 10 s, 100 ms latency, BTCUSDT USD-M test days. "
         "Implementation shortfall against the arrival mid, bps of the order's notional, fees included; mean [95% bootstrap CI "
         "over episodes]. RL = per-episode mean of the 5 seeds' agents (trained on the proportional rule). Every strategy runs "
         "on the same episodes (paired).\n"]
    for H in (H_MAIN,) + H_SENS:
        sub = t[t.H == H]
        n = int(sub.n.iloc[0])
        M.append(f"\n## H = {int(H) // 60} min ({n:,} paired episodes per side)\n")
        for rule in ([MAIN_RULE] + [r for r in RULES if r != MAIN_RULE]):
            s2 = sub[sub.rule == rule]
            M.append(f"\n### Cancel rule: {rule}{' (main)' if rule == MAIN_RULE else ' (bound)'}\n")
            M.append("| strategy | buy, retail | sell, retail | buy, pro | sell, pro | passive share (buy, retail) |")
            M.append("|---|---|---|---|---|---|")
            for k in s2.strategy.unique():
                cells = []
                for fee in FEES:
                    for side in ("buy", "sell"):
                        r = s2[(s2.strategy == k) & (s2.side == side) & (s2.fee == fee)]
                        c = fmt((r["mean"].iloc[0], r.lo.iloc[0], r.hi.iloc[0]))
                        cells.append(f"**{c}**" if r.best_fixed.iloc[0] else c)
                pr = s2[(s2.strategy == k) & (s2.side == "buy") & (s2.fee == "retail")].passive_share.iloc[0]
                lab = strategy_label(k) + (" ← headline" if k == f"rl_{head}" else "")
                M.append(f"| {lab} | {cells[0]} | {cells[1]} | {cells[2]} | {cells[3]} | {pr:.0%} |")
        M.append("\nBold = the best fixed action chosen on the training days for that side and fee.")
        # paired differences of the headline (or 'all' for sensitivity) vs each baseline, main rule
        hv = f"rl_{head}" if H == H_MAIN else "rl_ofi+queue+sigma"
        if hv not in set(d[d.H == H].rl):
            hv = "rl_ofi+queue+sigma"
        dd = d[(d.H == H) & (d.rule == MAIN_RULE) & (d.rl == hv)]
        M.append(f"\nPaired difference {strategy_label(hv)} − baseline (bps; negative = RL cheaper), {MAIN_RULE} rule:\n")
        M.append("| baseline | buy, retail | sell, retail | buy, pro | sell, pro |")
        M.append("|---|---|---|---|---|")
        for bk in BASELINES:
            cells = []
            for fee in FEES:
                for side in ("buy", "sell"):
                    r = dd[(dd.baseline == bk) & (dd.side == side) & (dd.fee == fee)].iloc[0]
                    cells.append(fmt((r["diff"], r.lo, r.hi), 3))
            M.append(f"| {BASELINES[bk]} | " + " | ".join(cells) + " |")
    return M


def state_value(head: str) -> pd.DataFrame:
    """Each variant minus private-only, paired, seed-averaged, main rule, H = 1 min."""
    rows = []
    for side in SIDES.values():
        for fee in FEES:
            blk = load_block(H_MAIN, side, MAIN_RULE, fee, VARIANTS)
            C = blk["cost"]
            for m in VARIANTS[1:]:
                v = variant_name(m)
                dd = paired_difference(C[f"rl_{v}"], C["rl_private"], np.mean, N_BOOT, seed=4)
                per_seed = [C[f"rl_{v}_seed{s}"].mean() - C[f"rl_private_seed{s}"].mean() for s in SEEDS]
                rows.append({"side": SIDE_LABEL[side], "fee": fee, "variant": v, "diff": dd[0], "lo": dd[1],
                             "hi": dd[2], "seed_min": min(per_seed), "seed_max": max(per_seed)})
            dd = paired_difference(C["rl_private"], C["fixed_touch"], np.mean, N_BOOT, seed=4)
            rows.append({"side": SIDE_LABEL[side], "fee": fee, "variant": "private vs rest-at-touch", "diff": dd[0],
                         "lo": dd[1], "hi": dd[2], "seed_min": np.nan, "seed_max": np.nan})
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "state_value.csv", index=False)
    return df


def md_state_value(sv: pd.DataFrame) -> list[str]:
    M = ["# Stage 5 Part A: value of each state variable (test days, H = 1 min, proportional rule)\n",
         "Paired difference in mean shortfall (bps) of the agent with the extra market variable(s) minus the private-only "
         "(time × inventory) agent, both averaged over 5 seeds; negative = the variable reduces cost out of sample. "
         "[95% bootstrap CI over episodes]; 'seeds' = range of the same difference seed by seed. The last row compares the "
         "private agent with resting at the touch.\n",
         "| added to time × inventory | buy, retail | sell, retail | buy, pro | sell, pro |", "|---|---|---|---|---|"]
    for v in sv.variant.unique():
        cells = []
        for fee in FEES:
            for side in ("buy", "sell"):
                r = sv[(sv.variant == v) & (sv.side == side) & (sv.fee == fee)].iloc[0]
                c = fmt((r["diff"], r.lo, r.hi), 3)
                if np.isfinite(r.seed_min):
                    c += f"; seeds {r.seed_min:+.3f}…{r.seed_max:+.3f}"
                cells.append(c)
        M.append(f"| {v} | " + " | ".join(cells) + " |")
    return M


def fills_table(head: str) -> pd.DataFrame:
    rows = []
    for side in SIDES.values():
        for fee in FEES:
            blk = load_block(H_MAIN, side, MAIN_RULE, fee, [tuple(head.split("+")) if head != "private" else ()]
                             if head != "private" else [()])
            for k in ["fixed_touch", "fixed_1behind", "fixed_3behind", f"rl_{head}"]:
                g = blk["meta"][k]
                pq = g.passive_qty.sum()
                rows.append({"side": SIDE_LABEL[side], "fee": fee, "strategy": k,
                             "passive_share": pq / (QTY * len(g)),
                             "taker_share": g.taker_qty.sum() / (QTY * len(g)),
                             "of_passive_queue": g.qty_queue.sum() / pq, "of_passive_sweep": g.qty_through.sum() / pq,
                             "of_passive_cross": g.qty_cross.sum() / pq,
                             "mk10_bps": g.mk10_sum.sum() / pq, "mk60_bps": g.mk60_sum.sum() / pq,
                             "mk60_ci": _ratio_ci(g.mk60_sum.to_numpy(), g.passive_qty.to_numpy())})
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "passive_fills_markouts.csv", index=False)
    return df


def _ratio_ci(num, den, seed=6):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(num), (N_BOOT, len(num)))
    b = num[idx].sum(1) / den[idx].sum(1)
    return tuple(float(v) for v in np.quantile(b, [0.025, 0.975]))


def md_fills(f: pd.DataFrame) -> list[str]:
    M = ["# Stage 5 Part A: passive vs aggressive fills and markouts of passive fills (test days, H = 1 min)\n",
         "Share of the 0.5 BTC filled passively (our resting order) vs as a taker (cross, TWAP slice or deadline clean-up); "
         "passive volume split by how it filled (queue = traded volume reached us in the queue; sweep = a trade through our "
         "price; cross = the opposite quote moved onto our price). Markout of passive fills = −side·(m_{t+h} − m_pre)/m_pre "
         "in bps, quantity-weighted; positive = adverse (the price kept moving in our favour after we filled, so waiting would "
         "have been cheaper). Proportional rule. RL = all 5 seeds pooled.\n",
         "| side | fee | strategy | passive | taker | passive via queue / sweep / cross | markout 10 s | markout 60 s [CI] |",
         "|---|---|---|---|---|---|---|---|"]
    for _, r in f.iterrows():
        M.append(f"| {r.side} | {r.fee} | {strategy_label(r.strategy)} | {r.passive_share:.1%} | {r.taker_share:.1%} | "
                 f"{r.of_passive_queue:.0%} / {r.of_passive_sweep:.0%} / {r.of_passive_cross:.0%} | {r.mk10_bps:.2f} | "
                 f"{r.mk60_bps:.2f} [{r.mk60_ci[0]:.2f}, {r.mk60_ci[1]:.2f}] |")
    return M


# ---------------------------------------------------------------------------------------------- policies
def load_tables(variant: str, side: int, fee: str, H=H_MAIN) -> list[tuple[np.ndarray, np.ndarray]]:
    from stage5_common import AGENTS
    out = []
    for sd in SEEDS:
        z = np.load(AGENTS / f"{variant}_{SIDE_LABEL[side]}_{fee}_{hkey(H)}_seed{sd}.npz")
        out.append((z["Q"], z["N"]))
    return out


def greedy_tables(tabs) -> np.ndarray:
    """seed x state -> greedy action over tried actions, -1 if unvisited."""
    res = []
    for Q, N in tabs:
        q = np.where(N > 0, Q, np.inf)
        a = np.argmin(q, axis=-1)
        res.append(np.where((N > 0).any(axis=-1), a, -1))
    return np.array(res)


def majority(acts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """acts: seeds x ...; returns (majority action or -1, number of seeds agreeing)."""
    flat = acts.reshape(acts.shape[0], -1)
    maj, agree = np.full(flat.shape[1], -1), np.zeros(flat.shape[1], int)
    for j in range(flat.shape[1]):
        v = flat[:, j][flat[:, j] >= 0]
        if len(v):
            c = np.bincount(v, minlength=4)
            maj[j], agree[j] = int(np.argmax(c)), int(c.max())
    return maj.reshape(acts.shape[1:]), agree.reshape(acts.shape[1:])


def _heat(ax, maj, agree, visits, title, min_visits=50):
    cmap = ListedColormap([UNVISITED] + ACTION_COLORS)
    show = np.where(visits >= min_visits, maj, -1)
    # rows: inventory bucket (top = most left), columns: time bucket (left = start)
    img = show.T[::-1, ::-1]
    ag = agree.T[::-1, ::-1]
    ax.imshow(img + 1, cmap=cmap, vmin=0, vmax=4, aspect="equal")
    short = {0: "cross", 1: "touch", 2: "−1", 3: "−3"}
    for r in range(img.shape[0]):
        for c in range(img.shape[1]):
            if img[r, c] >= 0:
                ax.text(c, r, f"{short[img[r, c]]}\n{ag[r, c]}/5", ha="center", va="center", fontsize=7,
                        color="#191919" if img[r, c] in (1, 2, 3) else "white")
    n_t, n_i = img.shape[1], img.shape[0]
    ax.set_xticks(range(n_t))
    ax.set_xticklabels(["start", "", "", "", "last"][:n_t] if n_t == 5 else range(n_t), fontsize=7)
    ax.set_yticks(range(n_i))
    ax.set_yticklabels([">80%", "60–80", "40–60", "20–40", "≤20%"], fontsize=7)
    ax.set_title(title, fontsize=8)
    for s in ax.spines.values():
        s.set_visible(False)


def policy_charts() -> dict:
    """Private-policy heatmaps per side x fee; market-variable shifts from the single-variable agents."""
    summary = {}
    fig, axes = plt.subplots(2, 2, figsize=(8, 8))
    for (ax, (side, fee)) in zip(axes.ravel(), [(1, "retail"), (-1, "retail"), (1, "pro"), (-1, "pro")]):
        tabs = load_tables("private", side, fee)
        maj, agree = majority(greedy_tables(tabs))
        visits = np.sum([N.sum(axis=-1) for _, N in tabs], axis=0) / len(tabs)
        _heat(ax, maj, agree, visits, f"{SIDE_LABEL[side]}, {FEE_LABEL[fee]}")
        summary[f"private|{SIDE_LABEL[side]}|{fee}"] = maj.tolist()
    for ax in axes[:, 0]:
        ax.set_ylabel("inventory left")
    for ax in axes[1]:
        ax.set_xlabel("time left (10 s decisions, H = 1 min)")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in ACTION_COLORS + [UNVISITED]]
    fig.legend(handles, list(ACTIONS) + ["visited < 50 times"], loc="lower center", ncol=5, fontsize=8, frameon=False)
    fig.suptitle("Learned execution policy, private state (time × inventory): majority action of 5 seeds (k/5 agree)",
                 fontsize=9)
    fig.tight_layout(rect=(0, 0.05, 1, 0.97))
    fig.savefig(OUT / "policy_heatmap_private.png", dpi=140)
    plt.close(fig)

    names = {"ofi": ["favourable flow", "neutral flow", "adverse flow"],
             "queue": ["short queue", "medium queue", "long queue"],
             "sigma": ["calm (low σ̂)", "medium σ̂", "volatile (high σ̂)"]}
    for fee in FEES:
        fig, axes = plt.subplots(3, 3, figsize=(9, 9.5))
        for row, var in enumerate(("ofi", "queue", "sigma")):
            tabs_b = load_tables(var, 1, fee)
            acts = greedy_tables(tabs_b)
            for b in range(3):
                maj, agree = majority(acts[..., b])
                visits = np.sum([N[..., b, :].sum(axis=-1) for _, N in tabs_b], axis=0) / len(tabs_b)
                _heat(axes[row, b], maj, agree, visits, f"+{var}: {names[var][b]}")
                summary[f"{var}|buy|{fee}|{b}"] = maj.tolist()
        for ax in axes[:, 0]:
            ax.set_ylabel("inventory left")
        for ax in axes[2]:
            ax.set_xlabel("time left")
        handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in ACTION_COLORS + [UNVISITED]]
        fig.legend(handles, list(ACTIONS) + ["visited < 50 times"], loc="lower center", ncol=5, fontsize=8, frameon=False)
        fig.suptitle(f"How each market variable shifts the policy (buy, {FEE_LABEL[fee]}): one-variable agents, "
                     "majority of 5 seeds", fontsize=9)
        fig.tight_layout(rect=(0, 0.04, 1, 0.97))
        fig.savefig(OUT / f"policy_heatmap_market_{fee}.png", dpi=140)
        plt.close(fig)
    return summary


def seed_agreement() -> pd.DataFrame:
    rows = []
    for m in VARIANTS:
        v = variant_name(m)
        for side in SIDES.values():
            for fee in FEES:
                tabs = load_tables(v, side, fee)
                acts = greedy_tables(tabs)
                visits = np.mean([N.sum(axis=-1) for _, N in tabs], axis=0)
                maj, agree = majority(acts)
                ok = visits >= 50
                rows.append({"variant": v, "side": SIDE_LABEL[side], "fee": fee, "states": int(ok.sum()),
                             "all_5_agree": float((agree[ok] == 5).mean()), "4plus_agree": float((agree[ok] >= 4).mean())})
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "seed_agreement.csv", index=False)
    return df


def action_shares(head: str) -> pd.DataFrame:
    """Decisions taken on the test days by the one-variable agents, by market bucket: how often each action."""
    rows = []
    for side in SIDES.values():
        for fee in FEES:
            for v in ("private", "ofi", "queue", "sigma", "ofi+queue+sigma"):
                r = pd.read_parquet(EVALS / f"rl_{v}_{SIDE_LABEL[side]}_{fee}_{hkey(H_MAIN)}_{MAIN_RULE}.parquet")
                rows.append({"side": SIDE_LABEL[side], "fee": fee, "variant": v,
                             "first_cross": float((r.first_action == 0).mean()),
                             "first_touch": float((r.first_action == 1).mean()),
                             "first_behind": float(r.first_action.isin([2, 3]).mean()),
                             "any_cross": float((r.n_cross > 0).mean()),
                             "fallback_share": float(r.groupby("seed").fallbacks.first().sum() /
                                                     max(1, r.groupby("seed").decisions.first().sum()))})
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "rl_action_shares.csv", index=False)
    return df


# ---------------------------------------------------------------------------------------------- curves
def learning_chart() -> None:
    lc = pd.read_csv(OUT / "learning_curves.csv")
    lc = lc[lc.H == H_MAIN]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, fee in zip(axes, FEES):
        for j, v in enumerate(["private", "ofi", "queue", "sigma", "ofi+queue+sigma"]):
            sub = lc[(lc.variant == v) & (lc.fee == fee)]
            for col, ls, lab in (("train_cost", ":", "training days"), ("val_cost", "-", "validation days")):
                g = sub.groupby("episodes")[col]
                m, lo, hi = g.mean(), g.min(), g.max()
                ax.plot(m.index, m.values, ls, color=CAT[j], lw=2, marker="o", ms=4,
                        label=f"{v} ({lab})" if True else None)
                if col == "val_cost":
                    ax.fill_between(m.index, lo.values, hi.values, color=CAT[j], alpha=0.12)
        ax.set_xscale("log")
        ax.set_xlabel("training episodes")
        ax.set_title(FEE_LABEL[fee], fontsize=9)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("mean shortfall of the greedy policy (bps), buy and sell averaged")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="center right", fontsize=7, frameon=False)
    fig.suptitle("Learning curves (H = 1 min): solid = validation days, dotted = training-day sample; band = range over 5 "
                 "seeds × 2 sides", fontsize=9)
    fig.tight_layout(rect=(0, 0, 0.8, 0.95))
    fig.savefig(OUT / "learning_curves.png", dpi=140)
    plt.close(fig)


def cost_chart(t: pd.DataFrame, head: str) -> None:
    sub = t[(t.H == H_MAIN) & (t.rule == MAIN_RULE)]
    keys = ["fixed_cross", "twap", "fixed_3behind", "fixed_1behind", "fixed_touch", "rl_private", f"rl_{head}"]
    keys = list(dict.fromkeys(keys))
    fig, axes = plt.subplots(1, 4, figsize=(14, 4), sharey=True)
    for ax, (fee, side) in zip(axes, [(f, s) for f in FEES for s in ("buy", "sell")]):
        for j, k in enumerate(keys):
            r = sub[(sub.strategy == k) & (sub.fee == fee) & (sub.side == side)].iloc[0]
            col = CAT[0] if k.startswith("rl_") else "#7f7f7f"
            ax.errorbar(r["mean"], j, xerr=[[r["mean"] - r.lo], [r.hi - r["mean"]]], fmt="o", color=col, ms=6,
                        capsize=3)
            ax.text(r.hi + 0.08, j, f"{r['mean']:.2f}", va="center", fontsize=7, color=TEXT2)
        ax.set_title(f"{side}, {FEE_LABEL[fee]}", fontsize=8)
        ax.axvline(0, color="k", lw=0.6)
        ax.grid(axis="x", alpha=0.3)
        ax.set_xlabel("shortfall (bps), 95% CI")
    axes[0].set_yticks(range(len(keys)))
    axes[0].set_yticklabels([strategy_label(k) for k in keys], fontsize=8)
    fig.suptitle("Execution cost on the test days: buy/sell 0.5 BTC within 1 minute (blue = learned policies)", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "execution_cost.png", dpi=140)
    plt.close(fig)


def state_value_chart(sv: pd.DataFrame) -> None:
    vs = ["ofi", "queue", "sigma", "ofi+queue+sigma"]
    fig, ax = plt.subplots(figsize=(8, 4))
    combos = [(s, f) for f in FEES for s in ("buy", "sell")]
    w = 0.18
    for j, (side, fee) in enumerate(combos):
        for i, v in enumerate(vs):
            r = sv[(sv.variant == v) & (sv.side == side) & (sv.fee == fee)].iloc[0]
            x = i + (j - 1.5) * w
            ax.errorbar(x, r["diff"], yerr=[[r["diff"] - r.lo], [r.hi - r["diff"]]], fmt="o", color=CAT[j], ms=6,
                        capsize=2, label=f"{side}, {fee}" if i == 0 else None)
    ax.axhline(0, color="k", lw=0.7)
    ax.set_xticks(range(len(vs)))
    ax.set_xticklabels(["+ OFI", "+ queue", "+ σ̂", "+ all three"])
    ax.set_ylabel("cost change vs time × inventory only (bps)\nnegative = variable helps")
    ax.legend(fontsize=8, frameon=False)
    ax.grid(axis="y", alpha=0.3)
    ax.set_title("Value of each state variable out of sample (test days, H = 1 min, 95% CI)", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "state_value.png", dpi=140)
    plt.close(fig)


# ---------------------------------------------------------------------------------------------- main
def report() -> None:
    head = headline_variant()
    log.info("headline variant (validation rule): %s", head)
    t, d = cost_tables(head)
    sv = state_value(head)
    f = fills_table(head)
    sa = seed_agreement()
    ash = action_shares(head)
    pol = policy_charts()
    learning_chart()
    cost_chart(t, head)
    state_value_chart(sv)
    (OUT / "policy_tables.json").write_text(json.dumps(pol), encoding="utf-8")
    M = md_cost(t, d, head)
    (OUT / "execution_cost.md").write_text("\n".join(M) + "\n", encoding="utf-8")
    (OUT / "state_value.md").write_text("\n".join(md_state_value(sv)) + "\n", encoding="utf-8")
    (OUT / "passive_fills_markouts.md").write_text("\n".join(md_fills(f)) + "\n", encoding="utf-8")
    S = ["# Seed stability and action use\n", "Share of states (visited ≥ 50 times on average) where the 5 seeds' greedy "
         "actions agree.\n", sa.to_markdown(index=False, floatfmt=".2f") if hasattr(sa, "to_markdown") else sa.to_string(),
         "\nTest-day decisions: share of episodes whose first action is cross / touch / behind, share with any cross, and the "
         "share of decisions that used the unvisited-state fallback.\n", ash.to_string(index=False)]
    (OUT / "seed_stability.md").write_text("\n".join(S) + "\n", encoding="utf-8")
    # headline numbers
    hl = {"headline_variant": head}
    for side in SIDES.values():
        for fee in FEES:
            blk = load_block(H_MAIN, side, MAIN_RULE, fee, [tuple(head.split("+"))] if head != "private" else [()])
            C = blk["cost"]
            means = {k: C[k].mean() for k in BASELINES}
            best = min(means, key=means.get)
            dd = paired_difference(C[f"rl_{head}"], C[best], np.mean, N_BOOT, seed=7)
            days = blk["episodes"][:, 0]
            db = day_block_ci(days, C[f"rl_{head}"] - C[best])
            hl[f"{SIDE_LABEL[side]}|{fee}"] = {"best_baseline_on_test": best, "rl_mean": float(C[f'rl_{head}'].mean()),
                                               "best_mean": float(means[best]), "diff": dd, "diff_day_block": db}
    (OUT / "headline.json").write_text(json.dumps(hl, indent=2), encoding="utf-8")
    log.info("headline: %s", hl)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    report()
