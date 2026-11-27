"""Streamlit demo of the market-making / execution study (stage 6). Runs locally, no training, no downloads.

    .venv\\Scripts\\streamlit run app\\demo.py

Tabs
1. A–S explainer: the Avellaneda–Stoikov formulas with sliders, plus one synthetic path (stage 1 code; instant).
2. Replay an episode: one test-day hour of a touch strategy replayed on the recorded BTCUSDT book with the
   stage 4 queue model (needs data/processed/stage4/, rebuilt by scripts/build_stage4_dataset.py).
3. Execution policy: the learned policy heatmap (time left x inventory left) and one replayed execution
   episode, RL agent vs the baselines (needs data/processed/stage4/ and data/derived/stage5_agents/).

Everything shown is recomputed from cached data with the same code as the stage scripts; where a cached result
of the same run exists (data/derived/), the demo shows it next to the replayed number as a cross-check.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from mms.sim.simulator import MarketParams, simulate  # noqa: E402
from mms.strategies.avellaneda_stoikov import AvellanedaStoikov  # noqa: E402
from mms.strategies.naive import NaiveSymmetric  # noqa: E402

RESULTS = REPO / "results"
STAGE4_DATA = REPO / "data" / "processed" / "stage4"
STAGE4_RUNS = REPO / "data" / "derived" / "stage4_runs"
AGENTS = REPO / "data" / "derived" / "stage5_agents"
EVALS = REPO / "data" / "derived" / "stage5_eval"
TEST_DAYS = sorted(pd.read_csv(RESULTS / "stage2" / "selected_days.csv").query("split == 'test'")["day"])
MISSING_MSG = ("The processed order-book data for **{day}** is not on this machine (`data/` is not in git: it is "
               "about 0.5 GB per 20 days and is rebuilt from Binance's public files). Run\n\n"
               "`.venv\\Scripts\\python scripts\\build_stage4_dataset.py`\n\n"
               "to download and process it (~1 h, ~4 GB temporary disk), then reload this page. The charts in "
               "`results/stage4/` and `results/stage5/` show the same results without the data.")

st.set_page_config(page_title="Market making on BTCUSDT", layout="wide")
st.title("Market making and execution on real BTCUSDT data")
st.caption("Avellaneda–Stoikov theory → calibration → adverse selection → replay with a queue model → RL for "
           "execution. Full write-up: README.md; every judgement call: DECISIONS.md.")


def day_available(day: str) -> bool:
    return (STAGE4_DATA / f"BTCUSDT-book100ms-{day}.parquet").exists() and \
        (STAGE4_DATA / f"BTCUSDT-trades4-{day}.parquet").exists()


@st.cache_resource(max_entries=2, show_spinner="Loading one day of the 100 ms book and trades...")
def get_day(day: str):
    from mms.replay.data import load_day
    return load_day.__wrapped__(day)


tab1, tab2, tab3 = st.tabs(["1. A–S explainer", "2. Replay a market-making hour", "3. Execution policy (RL)"])

# ==================================================================================================== tab 1
with tab1:
    st.markdown(
        "A market maker holding inventory **q** quotes around a **reservation price** instead of the mid, and "
        "uses the optimal total **spread** (Avellaneda & Stoikov 2008):\n\n"
        "- r = S − q·γ·σ²·(T − t)  (long inventory → value the asset below the mid → sell more eagerly)\n"
        "- spread = γ·σ²·(T − t) + (2/γ)·ln(1 + γ/k)  (inventory-risk term + 'monopoly' liquidity term)\n"
        "- bid = r − spread/2, ask = r + spread/2;  fills arrive at rate λ(δ) = A·exp(−k·δ) at distance δ.\n\n"
        "Units are the paper's (S₀ = 100, T = 1, A = 140, dt = 0.005).")
    c1, c2 = st.columns([1, 2])
    with c1:
        gamma = st.select_slider("risk aversion γ", options=[0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0], value=0.1)
        sigma = st.slider("volatility σ (price units / √time)", 0.5, 4.0, 2.0, 0.1)
        k = st.slider("fill-intensity decay k (1 / price unit)", 0.3, 3.0, 1.5, 0.1)
        q = st.slider("inventory q (units)", -10, 10, 3)
        t = st.slider("time t (horizon T = 1)", 0.0, 1.0, 0.0, 0.05)
    a_s = AvellanedaStoikov(gamma=gamma, sigma=sigma, k=k, T=1.0)
    S = 100.0
    r = float(a_s.reservation_price(S, q, t))
    spr = a_s.spread(t)
    bid, ask = r - spr / 2, r + spr / 2
    inv_term, liq_term = gamma * sigma**2 * (1 - t), (2 / gamma) * np.log1p(gamma / k)
    with c1:
        st.markdown(
            f"**Worked numbers.** r = 100 − {q}·{gamma:g}·{sigma:g}²·{1 - t:.2f} = **{r:.3f}**  \n"
            f"spread = {inv_term:.3f} (inventory) + {liq_term:.3f} (liquidity) = **{spr:.3f}**  \n"
            f"bid = **{bid:.3f}**, ask = **{ask:.3f}** (distance from mid: δ_b = {S - bid:.3f}, δ_a = {ask - S:.3f})  \n"
            f"fill rates: bid {140 * np.exp(-k * max(S - bid, 0)):.1f}/unit time, "
            f"ask {140 * np.exp(-k * max(ask - S, 0)):.1f}/unit time")
    with c2:
        fig, ax = plt.subplots(figsize=(7.5, 2.6))
        lo, hi = min(bid, S) - 1, max(ask, S) + 1
        ax.axvline(S, color="black", lw=2, label=f"mid S = {S:g}")
        ax.axvline(r, color="purple", ls="--", lw=2, label=f"reservation price r = {r:.2f}")
        ax.axvline(bid, color="#2ca02c", lw=3, label=f"bid = {bid:.2f}")
        ax.axvline(ask, color="#d62728", lw=3, label=f"ask = {ask:.2f}")
        ax.axvspan(bid, ask, color="grey", alpha=0.1)
        ax.set_xlim(lo, hi)
        ax.set_yticks([])
        ax.set_xlabel("price (paper units)")
        ax.set_title("Quotes for the chosen γ, σ, k, q and t")
        ax.legend(fontsize=8, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.35), frameon=False)
        st.pyplot(fig, clear_figure=True)

        seed = st.number_input("synthetic path seed", 0, 999, 0)
        params = MarketParams(s0=100.0, sigma=sigma, T=1.0, dt=0.005, A=140.0, k=k)
        res = simulate(a_s, params, [seed], record_paths=True)
        sym = simulate(NaiveSymmetric(float(res.mean_quoted_spread[0])), params, [seed])
        p = res.paths
        fig, (a1, a2) = plt.subplots(2, 1, figsize=(7.5, 4.6), sharex=True, height_ratios=[2, 1])
        a1.plot(p["t"], p["mid"][0], color="black", lw=1, label="mid")
        a1.plot(p["t"][:-1], p["ask"][0], color="#d62728", lw=0.7, label="A–S ask")
        a1.plot(p["t"][:-1], p["bid"][0], color="#2ca02c", lw=0.7, label="A–S bid")
        a1.set_ylabel("price (paper units)")
        a1.legend(fontsize=8, frameon=False)
        a1.set_title(f"One synthetic path (seed {seed}): A–S P&L {res.pnl[0]:.1f}, final q {res.q[0]:+d}; "
                     f"symmetric benchmark P&L {sym.pnl[0]:.1f}, final q {sym.q[0]:+d}", fontsize=9)
        a2.step(p["t"], p["q"][0], where="post")
        a2.axhline(0, color="grey", lw=0.5)
        a2.set_ylabel("inventory q (units)")
        a2.set_xlabel("time t (fraction of horizon T = 1)")
        st.pyplot(fig, clear_figure=True)
        st.caption("Stage 1 result over 1,000 paired paths: A–S earns about the same mean P&L as a symmetric "
                   "quote with the same average spread, with roughly half the P&L std and a third of the "
                   "inventory std (results/stage1/replication.md).")

# ==================================================================================================== tab 2
STRATS = {"always at touch": "touch", "touch + inventory skew": "touch_skew", "touch + queue-aware": "touch_queue",
          "touch + order-flow aware": "touch_ofi", "markout-aware A–S (quote wide)": "markout_as"}


@st.cache_data(max_entries=8, show_spinner="Replaying one hour (36,000 steps of 100 ms)...")
def replay_hour(day: str, hour: int, strat: str, fee: float, rule: str, queue_frac: float):
    import json

    import stage4_common as s4
    from mms.replay.env import ReplayConfig, run_episode
    if strat == "markout_as":
        spec = {"name": "markout_as", "fee": fee}
    else:
        ch = json.loads((RESULTS / "stage4" / "chosen_thresholds.json").read_text(encoding="utf-8"))
        spec = ch[f"{strat}|{fee:g}"]["spec"]
    d = get_day(day)
    start = hour * s4.EP_STEPS
    cfg = ReplayConfig(latency_ms=100, cancel_rule=rule, lot=0.01, fee_bps=fee, gamma=s4.GAMMA, record=True,
                       queue_frac=queue_frac)
    res = run_episode(s4.make_policy(spec), d, start, cfg)
    cached = None
    if queue_frac == 1.0:
        f = STAGE4_RUNS / f"{s4.task_key({k: v for k, v in spec.items()}, rule, 100, 0.01, day)}.parquet"
        if f.exists():
            c = pd.read_parquet(f).set_index("hour")
            cached = float(c.loc[hour, "pnl"] - (fee - c.loc[hour, "fee_run"]) * 1e-4 * c.loc[hour, "notional"])
    return res, spec, cached


with tab2:
    st.markdown(
        "Our 0.01 BTC orders are placed into the **recorded** Binance USD-M BTCUSDT book (100 ms snapshots of "
        "the best bid/ask and sizes, plus every trade). An order joins the **back of the queue**; trades at our "
        "price eat the queue ahead of us first; a trade *through* our price (a **sweep**) fills us at once. "
        "100 ms latency; cancellations split ahead/behind us by the chosen rule. Test days only.")
    c1, c2, c3, c4, c5 = st.columns(5)
    day = c1.selectbox("test day", TEST_DAYS, index=4)
    hour = c2.slider("hour (UTC)", 1, 23, 14)
    sname = c3.selectbox("strategy", list(STRATS))
    fee = c4.selectbox("maker fee (bp)", [-0.5, 0.0, 2.0], index=1,
                       format_func=lambda f: {-0.5: "−0.5 (rebate)", 0.0: "0", 2.0: "+2 (retail)"}[f])
    rule = c5.selectbox("cancel rule", ["proportional", "pessimistic", "optimistic"])
    qf = 1.0
    if STRATS[sname] in ("touch", "touch_skew"):
        qf = st.select_slider("WHAT-IF queue priority f (share of the displayed queue ahead of us on joining; "
                              "1 = back of the queue, the realistic case)", [1.0, 0.5, 0.25, 0.15, 0.1, 0.05, 0.0],
                              value=1.0)
    if not day_available(day):
        st.info(MISSING_MSG.format(day=day))
    else:
        res, spec, cached = replay_hour(day, hour, STRATS[sname], fee, rule, qf)
        rec = res.record
        n = len(rec["mid"])
        tsec = np.arange(n) / 10.0
        cash = np.zeros(n)
        f = res.fills
        for st_, sd, px, qty in zip(f["step"], f["side"], f["price"], f["qty"]):
            p_ = px * 0.1
            cash[st_ - res.start] += -sd * qty * p_ - fee * 1e-4 * qty * p_
        pnl = np.cumsum(cash) + rec["q"] * rec["mid"]
        how = np.asarray(f["how"]).astype(str) if res.n_fills else np.array([], str)
        qty_by = {h: float(f["qty"][how == h].sum()) if res.n_fills else 0.0 for h in ("queue", "through", "cross")}
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("P&L this hour (USDT)", f"{res.pnl:+.2f}",
                  help="Fees included, final inventory marked at the mid.")
        m2.metric("BTC filled", f"{res.fill_qty:.2f}")
        tot = max(res.fill_qty, 1e-12)
        m3.metric("filled from the queue / by sweeps", f"{qty_by['queue'] / tot:.0%} / "
                  f"{(qty_by['through'] + qty_by['cross']) / tot:.0%}")
        m4.metric("max |inventory| (lots)", f"{res.max_abs_q / 0.01:.0f}")
        if cached is not None:
            st.caption(f"Cross-check: the stage 4 run cached for this hour gives {cached:+.2f} USDT "
                       f"({'identical' if abs(cached - res.pnl) < 1e-6 else 'DIFFERENT - please report'}). "
                       f"Strategy settings (chosen on calibration days): {spec}.")
        if qf < 1:
            st.warning("WHAT-IF: queue priority f < 1 is not achievable for a new retail participant; it shows how "
                       "much priority would be needed (README, Part B).")

        w0 = st.slider("zoom: minute of the hour", 0, 57, 10)
        wl = st.slider("zoom window (minutes)", 1, 10, 3)
        sl = slice(w0 * 600, min(n, (w0 + wl) * 600))
        fig, axs = plt.subplots(3, 1, figsize=(11, 8.5), sharex=True, height_ratios=[2.2, 1.2, 1])
        ax = axs[0]
        ax.step(tsec[sl], rec["best_bid"][sl] * 0.1, where="post", color="#2ca02c", lw=0.8, alpha=0.6, label="best bid")
        ax.step(tsec[sl], rec["best_ask"][sl] * 0.1, where="post", color="#d62728", lw=0.8, alpha=0.6, label="best ask")
        ax.step(tsec[sl], rec["bid_px"][sl] * 0.1, where="post", color="#2ca02c", lw=2.2, label="our bid")
        ax.step(tsec[sl], rec["ask_px"][sl] * 0.1, where="post", color="#d62728", lw=2.2, label="our ask")
        if res.n_fills:
            ft = (f["step"] - res.start) / 10.0
            inw = (ft >= tsec[sl][0]) & (ft <= tsec[sl][-1])
            for h, mk, lab in (("queue", "o", "fill from the queue"), ("through", "X", "fill by a sweep through our price"),
                               ("cross", "D", "fill: book crossed onto us")):
                m = inw & (how == h)
                if m.any():
                    ax.scatter(ft[m], f["price"][m] * 0.1, marker=mk, s=60, edgecolor="black", zorder=5,
                               color=np.where(f["side"][m] > 0, "#2ca02c", "#d62728"), label=lab)
        ax.set_ylabel("price (USDT)")
        ax.legend(fontsize=7.5, ncol=4, frameon=False, loc="upper left")
        ax.set_title(f"{day} {hour:02d}:{w0:02d}–{hour:02d}:{w0 + wl:02d} UTC, {sname}, fee {fee:+g} bp, {rule} "
                     f"cancels, 100 ms latency (green = buy, red = sell)", fontsize=9)
        ax = axs[1]
        ax.plot(tsec[sl], rec["bid_size"][sl], color="#2ca02c", lw=0.6, alpha=0.5, label="displayed best-bid size")
        ax.plot(tsec[sl], rec["Q_bid"][sl], color="#2ca02c", lw=1.8, label="queue ahead of our bid")
        ax.plot(tsec[sl], rec["ask_size"][sl], color="#d62728", lw=0.6, alpha=0.5, label="displayed best-ask size")
        ax.plot(tsec[sl], rec["Q_ask"][sl], color="#d62728", lw=1.8, label="queue ahead of our ask")
        ax.set_yscale("symlog", linthresh=0.1)
        ax.set_ylabel("BTC (symlog)")
        ax.legend(fontsize=7.5, ncol=2, frameon=False)
        ax = axs[2]
        ax.step(tsec[sl], rec["q"][sl] / 0.01, where="post", color="#1f77b4")
        ax.axhline(0, color="grey", lw=0.5)
        ax.set_ylabel("inventory (lots)")
        ax.set_xlabel("seconds into the hour")
        fig.tight_layout()
        st.pyplot(fig, clear_figure=True)

        fig, ax = plt.subplots(figsize=(11, 2.8))
        ax.plot(tsec / 60, pnl, color="black", lw=1)
        ax.axhline(0, color="grey", lw=0.5)
        ax.axvspan(w0, w0 + wl, color="orange", alpha=0.15, label="zoom window")
        ax.set_xlabel("minutes into the hour")
        ax.set_ylabel("cumulative P&L (USDT)")
        ax.set_title("Mark-to-market P&L over the hour (cash + inventory × mid, fees included)", fontsize=9)
        ax.legend(fontsize=8, frameon=False)
        st.pyplot(fig, clear_figure=True)
        st.caption("Across all 230 test hours, every touch strategy loses money at every fee, cancel rule and "
                   "latency (108 of 108 scenarios, 95% CI below zero). Sweeps are the problem: after a sweep fill "
                   "the price moves ~4.9 USDT/BTC against us within 60 s, against ~1.5 for queue fills.")

# ==================================================================================================== tab 3
H60 = 60.0


@st.cache_data(show_spinner=False)
def majority_policy(side: str, fee: str):
    """Majority greedy action over the 5 seeds of the private-state (time x inventory) agent; -1 = unvisited."""
    from mms.execution.qlearning import TabularQ
    tabs = []
    for s in range(1, 6):
        p = AGENTS / f"private_{side}_{fee}_H60_seed{s}.npz"
        if not p.exists():
            return None
        z = np.load(p)
        tabs.append(TabularQ(tuple(z["Q"].shape[:-1]), Q=z["Q"], N=z["N"]).policy_table())
    a = np.stack(tabs)
    maj = np.full(a.shape[1:], -1)
    agree = np.zeros(a.shape[1:], int)
    for idx in np.ndindex(a.shape[1:]):
        v = a[(slice(None),) + idx]
        v = v[v >= 0]
        if len(v):
            vals, cnt = np.unique(v, return_counts=True)
            maj[idx], agree[idx] = vals[cnt.argmax()], cnt.max()
    return maj, agree


@st.cache_data(max_entries=16, show_spinner="Replaying the execution episode...")
def run_exec(day: str, start: int, side: str, fee: str, seed: int):
    import stage5_common as s5
    from mms.execution.baselines import TWAP, FixedAction
    from mms.execution.env import CROSS, TOUCH, ExecutionEnv
    from mms.execution.qlearning import GreedyPolicy, TabularQ
    sd = s5.SIDES[side]
    cfg = s5.make_cfg(H60, sd, fee)
    d = get_day(day)
    z = np.load(AGENTS / f"sigma_{side}_{fee}_H60_seed{seed}.npz")
    agent = TabularQ(tuple(z["Q"].shape[:-1]), Q=z["Q"], N=z["N"])
    pols = {"immediate taker": lambda: FixedAction(CROSS), "TWAP taker slices": lambda: TWAP(cfg.qty, cfg.n_steps),
            "rest at touch, cross at deadline": lambda: FixedAction(TOUCH),
            f"RL agent (+σ̂, seed {seed})": lambda: GreedyPolicy(agent, s5.make_spec(("sigma",)))}
    out = {}
    env = ExecutionEnv(d, cfg)
    for name, mk in pols.items():
        pol = mk()
        obs = env.reset(start)
        done = False
        while not done:
            if hasattr(pol, "act"):
                a, x, passive = pol.act(obs)
                obs, _, done, _ = env.step(a, take_qty=x, passive=passive)
            else:
                obs, _, done, _ = env.step(pol(obs))
        out[name] = env.result()
    return out


with tab3:
    st.markdown(
        "**Task:** buy (or sell) **0.5 BTC within 1 minute**, deciding every 10 s where the order rests: cross now "
        "(taker), at the touch, 1 tick behind, or 3 ticks behind; anything left at the deadline is crossed. Cost = "
        "implementation shortfall against the arrival mid, in bps, fees included. Tabular Q-learning "
        "(Nevmyvaka, Feng & Kearns 2006) on the replay with queue keeping; trained on 8 days, tuned on 2, tested once "
        "on 10.\n\n**Finding:** the learned policy rediscovers *rest at the touch, cross the remainder at the "
        "deadline*. It saves ~2 bp against taking liquidity but does not beat that simple rule (it is 0.02–0.09 bp "
        "worse); market signals (order flow, queue size, volatility) add nothing out of sample.")
    c1, c2 = st.columns([1, 1.25])
    with c1:
        side = st.radio("side", ["buy", "sell"], horizontal=True)
        fee = st.radio("fees", ["retail", "pro"], horizontal=True,
                       format_func=lambda f: {"retail": "retail (maker 2 / taker 5 bp)",
                                              "pro": "professional (maker −0.5 / taker 2 bp, assumed)"}[f])
        pol = majority_policy(side, fee) if AGENTS.exists() else None
        names = ["cross", "touch", "1 behind", "3 behind"]
        if pol is None:
            st.info("Trained agents (data/derived/stage5_agents/) are not on this machine; showing the saved chart.")
            st.image(str(RESULTS / "stage5" / "policy_heatmap_private.png"))
        else:
            maj, agree = pol
            from matplotlib.colors import ListedColormap
            cmap = ListedColormap(["#d62728", "#2ca02c", "#ffbf00", "#9467bd"])
            fig, ax = plt.subplots(figsize=(5.2, 4.4))
            ax.imshow(np.ma.masked_less(maj, 0).T[::-1], cmap=cmap, vmin=-0.5, vmax=3.5, aspect="auto")
            for i in range(maj.shape[0]):
                for j in range(maj.shape[1]):
                    lab = "–" if maj[i, j] < 0 else f"{names[maj[i, j]]}\n{agree[i, j]}/5"
                    ax.text(i, maj.shape[1] - 1 - j, lab, ha="center", va="center", fontsize=7)
            ax.set_xticks(range(5), ["≤10 s", "20 s", "30 s", "40 s", "50–60 s"])
            ax.set_yticks(range(5), ["80–100%", "60–80%", "40–60%", "20–40%", "0–20%"])
            ax.set_xlabel("time left in the minute")
            ax.set_ylabel("inventory left (% of 0.5 BTC)")
            ax.set_title(f"Learned policy, {side}, {fee}: majority action of 5 seeds\n(time × inventory state; "
                         "k/5 = seeds agreeing)", fontsize=9)
            st.pyplot(fig, clear_figure=True)
    with c2:
        if not AGENTS.exists() or not day_available(TEST_DAYS[0]):
            st.info("Replaying an episode needs data/processed/stage4/ and data/derived/stage5_agents/. "
                    + MISSING_MSG.format(day="the test days"))
        else:
            from mms.execution.env import ExecConfig
            from mms.execution.training import stratified_starts
            eday = st.selectbox("test day ", TEST_DAYS, index=0)
            if not day_available(eday):
                st.info(MISSING_MSG.format(day=eday))
            else:
                starts = [s for d_, s in stratified_starts(TEST_DAYS, ExecConfig(horizon_s=H60, decision_s=10.0), 13)
                          if d_ == eday]
                k_ep = st.slider(f"test episode on {eday} (one of the {len(starts)} pre-drawn, non-overlapping "
                                 "1-minute episodes)", 0, len(starts) - 1, len(starts) // 2)
                seed = st.selectbox("RL training seed", [1, 2, 3, 4, 5])
                start = starts[k_ep]
                out = run_exec(eday, start, side, fee, seed)
                d = get_day(eday)
                sd = 1 if side == "buy" else -1
                rows = []
                for name, r in out.items():
                    acts = ", ".join(names[a] if a is not None else "slice" for a in r.actions)
                    rows.append({"strategy": name, "cost (bps)": round(r.cost_bps, 3),
                                 "passive share": f"{r.passive_qty / r.qty:.0%}",
                                 "crossed at deadline (BTC)": round(r.cleanup_qty, 3), "actions (every 10 s)": acts})
                st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
                ev = EVALS / f"rl_sigma_{side}_{fee}_H60_proportional.parquet"
                if ev.exists():
                    e = pd.read_parquet(ev)
                    e = e[(e.day == eday) & (e.start == start) & (e.seed == seed)]
                    if len(e):
                        rl = out[f"RL agent (+σ̂, seed {seed})"]
                        st.caption(f"Cross-check: stage 5's cached test evaluation of this agent on this episode: "
                                   f"{float(e.cost_bps.iloc[0]):.3f} bps (replayed here: {rl.cost_bps:.3f}).")
                i0, i1 = start - 50, start + 600 + 50
                tt = (np.arange(i0, i1) - start) / 10.0
                fig, ax = plt.subplots(figsize=(7.5, 4.2))
                ax.step(tt, d.bid[i0 - 1:i1 - 1] * 0.1, where="post", color="#2ca02c", lw=1, label="best bid")
                ax.step(tt, d.ask[i0 - 1:i1 - 1] * 0.1, where="post", color="#d62728", lw=1, label="best ask")
                ax.axhline(out["immediate taker"].m0, color="black", ls=":", lw=0.8, label="arrival mid")
                mk = {"rest at touch, cross at deadline": ("o", "#2ca02c"), f"RL agent (+σ̂, seed {seed})": ("^", "#1f77b4"),
                      "immediate taker": ("s", "grey")}
                for name, (m, c) in mk.items():
                    r = out[name]
                    if len(r.fills["qty"]):
                        ft = (np.asarray(r.fills["bucket"]) - start) / 10.0
                        ax.scatter(ft, np.asarray(r.fills["price"]) * 0.1, s=20 + 300 * np.asarray(r.fills["qty"]),
                                   marker=m, color=c, edgecolor="black", alpha=0.75, zorder=5,
                                   label=f"{name} fills ({r.cost_bps:.2f} bp)")
                for x in range(0, 61, 10):
                    ax.axvline(x, color="grey", lw=0.4, alpha=0.5)
                ax.set_xlabel("seconds since arrival (vertical lines = decisions; 60 = deadline)")
                ax.set_ylabel("price (USDT)")
                ax.set_title(f"{eday}: {side} 0.5 BTC within 1 minute (marker size ∝ fill size)", fontsize=9)
                ax.legend(fontsize=7, frameon=False, loc="best")
                st.pyplot(fig, clear_figure=True)
    st.caption("Test-day averages (6,890 episodes per side): immediate taker 5.06 bp, rest at touch 2.86–2.93 bp, "
               "RL 2.95–2.98 bp (retail); 2.06 / 0.35–0.42 / 0.40–0.45 bp (professional). See "
               "results/stage5/execution_cost.md.")
