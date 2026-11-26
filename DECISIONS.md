# Decisions log

Format: date — decision — one-line reason.

## Stage 1

- 2026-11-26 — Package installed editable from `src/` via `pyproject.toml`; pytest uses `pythonpath = ["src"]`. — Scripts and tests import `mms` the same way without path hacks.
- 2026-11-26 — Added `requests` to the pinned requirements. — Used by the Binance downloader; simpler and clearer than urllib.
- 2026-11-26 — Each simulated path has its own integer seed; all of its randomness (price shocks, bid/ask fill uniforms) is drawn from that seed before any strategy runs. — Gives paired comparisons (common random numbers) and lets any single path be reproduced; RL train/test can later use disjoint seed ranges.
- 2026-11-26 — Stage 1 replication uses seeds 0–999. — Simple and documented; later stages will use separate, non-overlapping ranges.
- 2026-11-26 — Price shocks are Gaussian (exact ABM step), not the paper's ±σ√dt binomial steps. — Same mean and variance per step; Gaussian is the textbook ABM and exact for any dt.
- 2026-11-26 — Fills: at most one unit per side per step, Bernoulli with p = A·exp(−kδ)·dt, δ measured from the mid at quote time; `check_dt` raises if A·dt ≥ 1. — Follows the paper's discretisation; A·dt = 0.7 for the paper setup.
- 2026-11-26 — When an A–S quote is through the mid (δ < 0) and λ·dt > 1, the probability is clipped to 1 and the number of clipped quote-steps is counted and reported in the results table (33 at γ=0.1, 5,168 at γ=0.5 out of 400,000). — A crossing quote fills for certain; counting it keeps this visible instead of silent.
- 2026-11-26 — Tick rounding: bid rounded down, ask rounded up. — Never quote tighter than the model asks for, and keeps bid < ask.
- 2026-11-26 — Fee: maker fee charged on notional, cash −= bid·(1+f) on a buy, += ask·(1−f) on a sell. Replication uses f = 0 and no tick (the paper has neither). — Matches how exchanges charge maker fees; costs come in for real data in stage 2.
- 2026-11-26 — Final P&L = X_T + q_T·S_T (inventory marked at the final mid, no liquidation cost). — The paper's convention; a liquidation penalty can be added in stage 2.
- 2026-11-26 — Fill rate = fills / (2 × steps), i.e. the share of quote opportunities (per side per step) that filled. Spread captured = mean distance from the mid per fill. — Clear and comparable across strategies.
- 2026-11-26 — Bootstrap: percentile method, 2,000 resamples; paired differences resample paths jointly. — No normality assumption; the pairing removes shared price noise.
- 2026-11-26 — Symmetric benchmark spread = mean A–S quoted spread over the run (1.349 / 1.492 / 2.156 for γ = 0.01 / 0.1 / 0.5). A second benchmark uses (2/γ)ln(1+γ/k) (1.329 / 1.291 / 1.151). — The paper's tables print the second number as "spread", which omits the γσ²(T−t) term; reporting both shows the definition barely matters for the symmetric P&L (~68 either way).
- 2026-11-26 — Paper sanity targets in scripts/run_replication.py were transcribed by Claude from Avellaneda & Stoikov (2008), Tables 1–3, not copied from the PDF in this session. — Owner should check them against the paper.

### Replication vs paper (stage 1)
- γ = 0.01 and 0.1 match the expected pattern and are close to the paper: e.g. γ = 0.1 A–S 64.9 ± 6.9 (paper 62.9 ± 5.9), symmetric 68.3 ± 13.3 (paper 67.2 ± 13.4); std of final inventory 2.9 vs 8.4 (paper 2.8 vs 8.7).
- γ = 0.5: A–S mean P&L 48.5 vs the paper's 33.9 (std 6.3 vs 4.7; inventory std 2.0 vs 1.9, which matches). Investigated without tuning (scratch run, 1,000 seeds): exact Poisson probability 1−exp(−λdt) → 42.8; never quoting through the mid → 50.5; spread = liquidity term only (what the paper's table prints) → 23.8. None reproduces 33.9 exactly; the paper does not fully specify its discretisation, so the main result keeps the CLAUDE.md model. The qualitative finding (A–S lower mean, much lower P&L and inventory std) holds at every γ.
- Noted for stage 2: at γ = 0.1 the λ·dt vs 1−exp(−λdt) choice moves A–S mean P&L by ~8 (64.9 → 57.2) because A·dt = 0.7 is not small. With real data, dt must be chosen so that A·dt is well below 1 (e.g. < 0.1).

### Binance data check (step 3)
- 2026-11-26 — data.binance.vision is reachable; Bybit fallback not needed and not tested.
- 2026-11-26 — USD-M futures BTCUSDT bookTicker is published only for **2023-05-16 → 2024-03-30** (320 consecutive days, no gaps; probed daily 2023-03-01 → 2024-06-01 and monthly back to 2020; 404 for 2024-06 onward and for 2026-11-24). aggTrades exist on every one of those days, so that is the joint range. — Stage 2 must calibrate and evaluate within this window (e.g. train 2023-05-16 → 2023-12-31, test 2024-01-01 → 2024-03-30).
- Sample day sizes (unzipped CSV):
  - futures/um bookTicker 2024-03-01: 38,973,486 rows, 3,499 MB; header `update_id,best_bid_price,best_bid_qty,best_ask_price,best_ask_qty,transaction_time,event_time`; ms timestamps.
  - futures/um aggTrades 2024-03-01: 1,434,196 rows, 91 MB; 2026-11-24: 2,462,147 rows, 156 MB; header `agg_trade_id,price,quantity,first_trade_id,last_trade_id,transact_time,is_buyer_maker`; ms timestamps.
  - spot trades 2026-11-24: 4,852,266 rows, 351 MB; **no header** (`id,price,qty,quote_qty,time,is_buyer_maker,is_best_match`); **microsecond** timestamps (spot switched from ms to µs in 2025).
- 2026-11-26 — bookTicker is ~3.5 GB/day unzipped (~1.1 TB for the full 320-day range). Stage 2 plan: download a limited set of days and reduce each to parquet (e.g. 100 ms or 1 s snapshots) on load. — Raw full-range storage is impractical.
