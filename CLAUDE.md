# Market-Making Simulator — project guide for Claude Code

## What this project is

A Python research project on optimal market making. A simulated market maker quotes a bid and an ask around a mid-price, and random traders arrive and hit those quotes. Compare:

- **Naive**: symmetric quotes around the mid with a fixed spread.
- **Avellaneda–Stoikov** (A–S, 2008): quotes around an inventory-adjusted reservation price, using the optimal spread.
- **Flow-aware A–S** (stage 3): A–S that widens or skews its quotes when order flow looks informed.
- **RL quoter** (stage 4): tabular Q-learning that picks adjustments on top of the A–S quotes.

The simulator is calibrated to real BTCUSDT data from Binance's free public data (data.binance.vision). It also includes informed traders, who cause adverse selection.

The owner is a first-year maths student building this for quant trading internship applications. He must be able to explain every formula and design choice in an interview. Maths code must be clear and commented, with derivations in docstrings. Prefer simple, explainable methods over clever ones.

## Ground rules (never break these)

- Tests first for every maths function (pytest), including hand-worked examples.
- Reproducible: every simulation takes a seed. Results are reported over many seeds, with 95% bootstrap confidence intervals, never from a single run.
- Train/test separation for anything fitted or learned: calibrate on one date range and evaluate on another; train RL on one set of seeds and test on a different set.
- No zero-filling or silent fallbacks. If data is missing or a fit fails, say so and log it in DECISIONS.md.
- Never commit raw market data. `data/` is gitignored, and download scripts recreate it. Commit code, small aggregated results and charts only.
- Honest costs: include exchange fees (maker fee as a parameter, with sensitivity at 0, 1 and 2 bps) and the tick size (0.1 USDT for BTCUSDT futures).
- Log every judgement call in DECISIONS.md (date, decision, one-line reason).
- Don't delete files or rewrite git history. Don't make the repo public.

## The maths (A–S)

- Mid-price: arithmetic Brownian motion, dS = σ dW (discretised with step dt).
- Fill intensity at distance δ from mid: λ(δ) = A·exp(−k·δ). In a step of length dt, a quote at distance δ fills with probability λ(δ)·dt (dt must be small enough that this stays well below 1; assert it).
- Inventory q, cash X, wealth = X + q·S. Risk aversion γ, horizon T.
- Reservation price: r = S − q·γ·σ²·(T − t)
- Total spread: δ_a + δ_b = γ·σ²·(T − t) + (2/γ)·ln(1 + γ/k)
- Quotes: ask = r + spread/2, bid = r − spread/2, rounded to the tick size when a tick size is set.
- Paper setup for replication: S0 = 100, T = 1, σ = 2, dt = 0.005, A = 140, k = 1.5, γ ∈ {0.1, 0.01, 1}, 1,000 simulations, fills with probability λ(δ)·dt (the paper's rule; stage 1 only). The benchmark is a symmetric strategy using the same average spread. Paper figures (Avellaneda & Stoikov 2008, Quantitative Finance, section 4), mean profit / std profit / std final q:
  - γ = 0.1: inventory strategy 65.0 / 6.6 / 2.9; symmetric 68.4 / 12.7 / 8.4 (average spread 1.49)
  - γ = 0.01: inventory strategy 68.6 / 8.7 / 5.1; symmetric 68.8 / 12.8 / 8.7 (average spread 1.35)
  - γ = 1: inventory strategy 31.4 / 5.0 / 1.7; symmetric 44.0 / 11.0 / 5.1 (average spread 3.02)
  Expected pattern: A–S has a similar or lower mean profit but a much lower std of profit and of final inventory. Treat these as sanity targets, not exact values.

## Layout

```
src/mms/
  sim/          # price process, order arrivals, matching, P&L accounting
  strategies/   # naive.py, avellaneda_stoikov.py, flow_aware.py, rl_agent.py
  calibration/  # estimate sigma, A, k from Binance data (stage 2)
  analysis/     # metrics, bootstrap CIs, charts
scripts/        # download_binance.py, run_replication.py, ...
tests/
results/        # small CSVs/JSON + PNG charts (committed)
data/           # gitignored
```

## Metrics (for every strategy comparison)

Mean and std of final P&L, a Sharpe-like ratio (mean/std), mean and std of final inventory, max absolute inventory, fill rate and average spread captured. Use 95% bootstrap CIs, and run the strategies on the same seeds so the comparison is paired.

## Conventions

- Python 3.13, `.venv`, pinned `requirements.txt`. numpy, pandas, scipy, matplotlib, pytest, pyarrow (streamlit in stage 5; torch only for the stage 4 stretch).
- Vectorise across simulations with numpy where it keeps the code readable.
- Type hints and short docstrings. `logging`, not `print`, in library code.
- Windows: use pathlib, no hard-coded `/` paths.
- Small commits with clear messages after each working step; push to the private repo.

## How the owner works

He has little time and runs each stage unattended in auto mode. Don't ask him questions. Decide, log the decision in DECISIONS.md and carry on. At the end of each stage give him:

1. What was built
2. The headline result in 2–3 sentences
3. One or two commands to check it
4. Anything that needs his decision

Then stop.

## Stages

1. Setup + core simulator + naive vs A–S + replication of the paper
2. Calibration to BTCUSDT (σ, A, k, fees) and rerun with real parameters
3. Adverse selection: informed traders, order-flow imbalance, flow-aware A–S
4. RL quoter: tabular Q-learning on top of A–S (DQN stretch)
5. Streamlit demo, README, charts, CV bullet
