# Stage 4 report: replay backtester with a queue model

All numbers: BTCUSDT USD-M futures, 0.01 BTC lot, inventory limit ±10 lots, 1-hour episodes marked to the mid at the end,
95% bootstrap CIs over hour-long episodes (230 per split: 10 days × 23 hours, 01:00–24:00 UTC). Test-day results use the same
230 hours for every strategy (paired). Details: `tuning.md`, `touch_strategy_table.md`, `markouts_touch_vs_wide.md`,
`latency_sensitivity.md`, `lot_sensitivity.md`, `validation_replay_vs_synthetic.md`; design choices in DECISIONS.md (stage 4).

## What the replay does
- **Data (step 0).** The 20 stage 2 days were re-downloaded (SHA-256 verified) and reduced to 100 ms best bid/ask with sizes plus
  the min/max size within each bucket (252 MB), and every aggTrade with its aggressor side and the quote in force before it (249 MB);
  3.5 GB of raw zips were deleted day by day. 0 crossed books; 1-tick spread in 98.7–99.98% of buckets; median touch size 2.3–12.6 BTC by day.
- **Queue model (step 1).** Our order joins the back of its level (queue ahead = displayed size). Volume traded at our price eats
  the queue ahead first, then fills us; a trade *through* our price, or the opposite quote moving onto it, fills us in full.
  Unexplained size decreases (cancellations) are split ahead/behind by three rules: **pessimistic** (none ahead), **proportional**,
  **optimistic** (ahead first). Under every rule the queue ahead can never exceed the smallest size the level showed (orders ahead only
  leave). A level that stops being best keeps our place unless we cancelled. Latency (0 / **100** / 300 ms) delays new orders and cancels;
  post-only orders that would cross are rejected. Our order is assumed not to change anyone else's behaviour (limitation below).
- Tests: a fill happens exactly when traded volume exceeds the queue ahead; the three rules give optimistic ≥ proportional ≥ pessimistic
  on a hand-built book; latency delays activation by exactly 0 / 1 / 3 steps; rewards sum to P&L = cash + q·mid = Σ fills, and repricing
  the fee is exact. 132 tests pass.

## Step 2: does the synthetic simulator match the replay?
Markout-aware A–S (stage 3's strategy, quoting ≥ 0.1 USDT from the mid) on the 230 test hours, replayed vs the stage 3 synthetic
simulator's prediction for the *same hours* (same σ̂, mid, model (b) A and k, flow calibration; 200 simulated paths per hour).
Ratio = replay / synthetic, ratio of sums over hours, 95% CI bootstrapping hours.

| fee | BTC filled / h: replay vs synthetic | ratio | 60 s markout (USDT/BTC) | ratio | P&L / h: replay vs synthetic | difference |
|---|---|---|---|---|---|---|
| −0.5 bp | 2.81 vs 2.88 | **0.98 [0.86, 1.10]** | 6.53 vs 4.17 | **1.57 [1.46, 1.68]** | −2.37 vs +8.17 | −10.5 [−11.7, −9.4] |
| 0 bp | 0.82 vs 0.78 | **1.06 [0.91, 1.21]** | 7.29 vs 4.89 | **1.49 [1.39, 1.60]** | −1.34 vs +2.04 | −3.4 [−4.0, −2.8] |
| +2 bp | 0.025 vs 0.0002 | (both barely trade) | 21.3 vs 15.3 | 1.40 [0.97, 1.91] | −0.06 vs 0.00 | −0.06 [−0.44, 0.21] |

The P&L ratio is not shown because the signs differ (it is in the detailed table). The three cancel rules give identical results for this strategy:
92–96% of its fills are sweeps *through* its price, and it rarely rests at the best level, where the rules matter. Treating every print at our
price behind the touch as a fill (the "front" upper bound) changes fills by only +3% (ratio 1.09 [0.94, 1.25]).

**Verdict: the synthetic model predicts fill volume well but underestimates adverse selection by ~50%, and that turns a predicted profit into a
real loss.** (1) Fill rate: ratio 0.98–1.06, CI covering 1. Stage 3's worry that model (b) over-predicts fills by ~36% does not show up for this
strategy: queue position hardly matters for quotes 1–5 USDT out, because almost all their fills are sweeps through the price. (2) Markout: the
replay's 60 s markout is 1.5–1.6× the synthetic one. The synthetic jump J(δ) is the market-wide average markout of trades at distance δ, but *our*
fills are selected: they happen only when a sweep reaches our price, and those sweeps move the price more than the average trade there.
(3) So the synthetic model's prediction that this strategy is profitable on these hours (+2.0 USDT/h at 0 bp, +8.2 at −0.5 bp; stage 3's
medium-regime table had +2.5 and +5.8 for the near-identical flow-aware version) does not survive the replay, which loses −1.3 and −2.4 USDT/h.
Stage 3's synthetic P&L for quoting wide should be read as optimistic by about 3–10 USDT/h.

## Step 3: touch strategies (test days, 100 ms latency, thresholds chosen on calibration days)

Thresholds were chosen per fee on the calibration days (proportional rule). Every touch configuration lost money there too, and the best
threshold of each strategy was at the **low-activity edge** of its pre-registered grid (skew: max 1 lot; queue-aware: join only if the
queue is < 5% of typical; OFI-aware: pull at |z| > 0.25). On the owner's instruction the grids were not widened.

Mean P&L per hour (USDT) [95% CI], proportional cancel rule (pessimistic / optimistic in `touch_strategy_table.md`):

| strategy | −0.5 bp (rebate) | 0 bp | +2 bp (retail) | bp of notional at 0 bp | BTC filled / h | 60 s markout (USDT/BTC) |
|---|---|---|---|---|---|---|
| always at touch | −27.9 [−32.7, −23.0] | −74.4 [−84.6, −64.7] | −260.8 [−293.8, −230.0] | −0.80 | 20.9 | 3.56 |
| touch + inventory skew | −17.7 [−21.1, −14.5] | −45.6 [−53.2, −38.6] | −143.2 [−162.8, −125.0] | −0.82 | 12.3 | 3.83 |
| touch + queue-aware | −8.8 [−11.5, −6.1] | −20.8 [−23.9, −17.9] | −69.0 [−74.5, −63.7] | −0.86 | 5.7 | 3.51 |
| touch + OFI-aware | −18.2 [−22.1, −14.5] | −39.4 [−46.2, −33.0] | −124.3 [−143.4, −106.1] | −0.93 | 9.3 | 4.14 |
| markout-aware A–S (quote wide) | −2.37 [−3.21, −1.59] | −1.34 [−1.76, −0.89] | −0.06 [−0.43, 0.22] (0.025 BTC/h: barely trades) | −0.41 | 0.82 | 7.29 |

Resampling whole days instead of hours widens the intervals about 2.5–4× but changes no sign (`touch_strategy_table.md`).

### 1. Can anything earn money at the touch, at 0 bp or with the −0.5 bp rebate?
**No, not in any scenario, including the best case.** Over 4 touch strategies × 3 cancel rules × 3 latencies (0, 100, 300 ms) × 3 fees
= 108 cases, every mean is negative and every 95% CI lies entirely below zero. The best case you asked to keep (0 ms latency + optimistic
cancels) is not the best scenario: with the −0.5 bp rebate it gives always-at-touch −28.4 [−35.5, −21.9], skew −11.0 [−14.9, −7.5],
queue-aware −17.1 [−20.0, −14.0], OFI-aware −16.6 [−22.1, −11.6] USDT/h. Lower latency means *more* fills, and fills lose on average.
The least-bad touch result anywhere is queue-aware at 300 ms with the rebate, −5.6 [−7.7, −3.7] USDT/h. The rebate cuts losses by ~60%
but never turns them positive. The "quote wide" markout-aware A–S also loses (−1.3 USDT/h at 0 bp, −2.4 at −0.5 bp) and stops trading at +2 bp.

### 2. Adverse selection at the touch vs 1–5 USDT out
| our fills (proportional rule) | 60 s markout, USDT/BTC [CI] | bp | spread captured (USDT/BTC) |
|---|---|---|---|
| at the touch, all | 3.64 [3.50, 3.77] | 0.81 | ~0.05–0.14 |
| at the touch, filled from the queue (39% of always-at-touch volume) | 1.46 [1.28, 1.64] | 0.35 | 0.05 |
| at the touch, filled by a sweep through our price (58%) | 4.94 [4.62, 5.24] | 1.09 | 0.14 |
| 1–2 / 2–3 / 3–5 USDT from the mid | 6.1 / 5.9 / 8.9 | 1.2–1.8 | 1–5 |
| markout-aware A–S fills (96% by sweeps) | 7.29 [6.83, 7.82] | 1.81 | 5.46 |

For comparison, stage 3 measured the market-wide markout of all trades at the touch as 0.22 bp. Our fills are 4× worse because a back-of-queue order
is mostly reached when a sweep takes out the whole level, just before the price moves. Further out the markout in USDT is about twice the
touch's, but the quote there earns ~5 USDT of spread instead of ~0.1, which is why quoting wide loses far less per BTC (−0.41 vs −0.80 bp).
The striking number is the queue fills: 0.35 bp of markout against a 0.5 bp rebate. **Queue fills alone would be profitable with the rebate;
the sweeps are what destroy the touch.**

### 3. Does the OFI signal add value at the touch?
**No.** OFI-aware beats always-at-touch by 35.1 [31.0, 39.4] USDT/h at 0 bp, but only because it quotes half the time and trades 55% less.
Per BTC traded it is *worse* (−0.93 vs −0.80 bp; markout 4.14 vs 3.56 USDT/BTC), and it loses to queue-aware by 18.6 [13.5, 24.3] USDT/h.
Pulling the side the flow is about to sweep leaves the other side, which is then hit by counter-flow sweeps that are just as informed. The 56% hit
rate does not survive 100 ms latency and back-of-queue priority. The same caveat applies to queue-awareness: its gain is from trading less
(5.7 BTC/h, quoted 10% of the time), not from better fills (−0.86 bp/BTC, markout 3.51). The short-queue rationale did not show up in the data.

### 4. Sensitivity to the cancel rule and latency
- **Cancel rule:** it changes how much we trade, not what each fill is worth. Optimistic gives always-at-touch 24% more volume than proportional
  (25.9 vs 20.9 BTC/h) with better markouts (2.97 vs 3.56), but P&L at 0 bp is −76.6 vs −74.4 vs −71.9 (optimistic / proportional / pessimistic).
  At −0.5 bp the extra volume earns more rebate, so the ordering reverses (−19.3 / −27.9 / −30.3). For queue-aware the rules are within 0.3 USDT/h.
  No rule changes any conclusion.
- **Latency:** 0 → 100 → 300 ms reduces volume (always-at-touch 30.7 → 20.9 → 16.6 BTC/h) and slightly worsens the markout (3.49 → 3.56 → 3.61),
  so losses in USDT/h *shrink* with latency (−107 → −74 → −58 at 0 bp). Per BTC everything is ~−0.8 bp at every latency.
- **Lot 0.1 BTC:** P&L scales ~10× and per notional is unchanged (always-at-touch −0.82 vs −0.80 bp), within the no-impact assumption.
- **Inventory:** every touch variant except skew hits the ±10-lot limit in every episode, so their results include a one-sided position;
  skew keeps max |q| at 2–3 lots and is the only one with controlled risk (std P&L 57 vs 78 USDT/h).

## Honest answer: can a small participant make money here?
**Not by quoting passively, in any configuration we tested.** A 0.01 BTC order at the back of the queue with 0–300 ms latency is filled
mostly when someone sweeps the whole level, and the price then moves ~5 USDT/BTC against it while it earned ~0.1. Even the best rebate tier
(−0.5 bp ≈ 2 USDT/BTC) does not cover that, and at the 2 bp retail fee losses are ~3 bp of everything traded. Quoting wide (stage 3's
markout-aware A–S) loses least because it trades least, but still loses at every fee where it trades. The profitable makers on this book
(stage 2 measured 1.9 fills/s at the touch) must have what we lack: queue priority (being early in the queue at a level, not late), cancels faster
than a sweep, and rebates. Our queue-fill markout (0.35 bp < 0.5 bp rebate) says the money is there for whoever is filled from the queue rather
than by sweeps; a 100 ms participant arriving at the back is not that participant.

## Step 4: RL environment for stage 5
`mms.replay.rl_env.ReplayRLEnv(split="calibration"|"test")`: `reset(seed)` draws a day of the split and a 1-hour start on the 100 ms grid
between 01:00 and 23:00; `step(action)` takes one of 16 joint actions ({touch, 1 behind, 3 behind, off} per side) or a (bid, ask) price pair.
Observation: inventory (lots), time remaining, queue ahead per side / typical size (inf when behind the touch), spread in ticks, 5 s OFI z, trailing σ̂.
Reward: mark-to-market P&L change − (γ/2)·lot·q²·σ̂²·dt with γ = 1e-4 per lot (the running version of A–S's inventory term; ~0.5 USDT over an
hour at 10 lots, small next to touch P&L). Tests confirm always-at-touch, queue-aware, OFI-aware (as discrete actions) and skew (as prices)
give bit-identical P&L and fills through the API.

## Limitations
- Our order is assumed not to change anyone's behaviour; strategies see the recorded book, which does not contain our own orders.
- Only the best bid/ask is recorded: queues behind the touch are unseen and assumed to be ahead of us until the level becomes best
  (bracketed by the "front" convention in step 2). The order of events within 100 ms is unknown (trades applied first).
- 230 test hours from 10 days; hours within a day are correlated (day-block CIs reported alongside).
- Thresholds sit at the edge of their grids; the true optimum in this class is "trade even less".

## Charts
`pnl_by_strategy_fee_rule.png`, `markouts_touch_vs_wide.png`, `latency_sensitivity.png`, `validation_replay_vs_synthetic.png`, `sample_hour_queue.png`
(three minutes of queue-aware quoting: our quotes, queue ahead vs displayed size, fills and inventory pinned near +10 lots).
