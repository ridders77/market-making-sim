# Stage 5 report: reinforcement learning

Part A (main): RL for optimal execution, a replication of Nevmyvaka, Feng & Kearns (2006) on BTCUSDT with stage 4's
queue-aware replay. Part B (**what-if**): how much queue priority a market maker needs before quoting pays.
All test numbers come from the 10 test days, evaluated once, after training on 8 calibration days and choosing
hyperparameters on the other 2. Design choices are logged in DECISIONS.md (stage 5).
Details: `execution_cost.md`, `state_value.md`, `passive_fills_markouts.md`, `seed_stability.md`, `tuning.md`,
`queue_whatif.md`, `mm_whatif.md`.

## Part A: optimal execution

### Task
Buy (or, separately, sell) **Q = 0.5 BTC within H = 1 minute**, deciding every 10 s (6 decisions). At each decision,
choose where the order for the whole remaining quantity rests: {cross now (taker), at the touch, 1 tick behind, 3 ticks behind}.
At the deadline any remainder is crossed as a taker. Orders take 100 ms to arrive; resting orders evolve in stage 4's
queue model (proportional cancel rule as the main case; pessimistic and optimistic as bounds). Cost = implementation
shortfall against the arrival mid in bps of the order's notional, fees included, for two fee tiers: retail (maker 2, taker 5 bp)
and professional (maker −0.5, taker 2 bp; an assumption, see DECISIONS.md).

- **Why 1 minute and not the suggested 5:** on the training days, resting at the touch for 5 minutes filled the *whole*
  order passively in 100% of episodes for every Q from 0.1 to 2 BTC. A touch queue of ~5 BTC is eaten within seconds,
  and re-joining the touch chases the price, so there is no trade-off left. At 1 minute the order fills completely
  ~93% of the time on the training days (97.8% on the test days). H = 2 and 5 minutes are reported as sensitivities.
- **Our improvement over Nevmyvaka et al.:** they re-placed the whole order every step, so it went to the back of the queue
  each time. Here an order keeps its place in the queue when the action does not ask for a different price, and a
  "1 behind" order that the market falls onto keeps its earlier arrival. The replay also caps a large order's fills by the
  volume that could actually have reached it (new `size_cap` in the queue model; stage 4 results unchanged).
- **State:** time left (5 buckets) × inventory left (5 buckets) = Nevmyvaka's private state, then plus 5 s OFI signed
  towards our side, queue size at our touch, and trailing σ̂ (3 buckets each, cut points from the training days),
  one at a time and all together (675 states).
- **Learning:** tabular Q-learning with a shaped per-step cost that sums exactly to the shortfall. Q-learning rather than
  backward induction because the queue place makes the problem path-dependent: the outcome of "touch" at step t depends
  on how long the order has already rested, which a fresh-order backward induction would ignore. 160,000 episodes per
  agent (half with random "exploring" starts in time and inventory), α = 0.01, ε from 1 to 0.05 over the first half.
  These were chosen on the 2 validation days, and all six settings fell within 0.13 bp of each other. Each agent is
  trained with 5 seeds.

### Headline
| H = 1 min, proportional rule | buy, retail | sell, retail | buy, pro | sell, pro |
|---|---|---|---|---|
| immediate taker | 5.06 [5.05, 5.08] | 5.06 [5.05, 5.08] | 2.06 [2.05, 2.08] | 2.06 [2.05, 2.08] |
| TWAP taker slices | 5.05 [4.97, 5.12] | 4.99 [4.92, 5.07] | 2.05 [1.97, 2.12] | 1.99 [1.92, 2.07] |
| **rest at touch, then cross** (= best fixed action on the training days) | **2.93 [2.88, 2.99]** | **2.86 [2.80, 2.92]** | **0.42 [0.37, 0.47]** | **0.35 [0.29, 0.41]** |
| RL, time × inventory only | 2.96 [2.91, 3.01] | 2.95 [2.89, 3.01] | 0.48 [0.43, 0.53] | 0.36 [0.30, 0.42] |
| RL, + σ̂ (headline, chosen on validation) | 2.98 [2.92, 3.03] | 2.95 [2.89, 3.01] | 0.45 [0.39, 0.50] | 0.40 [0.34, 0.46] |
| RL − rest at touch (paired) | **+0.044 [+0.037, +0.052]** | **+0.086 [+0.075, +0.097]** | **+0.024 [+0.020, +0.029]** | **+0.046 [+0.040, +0.053]** |
| RL − immediate taker (paired) | −2.09 [−2.14, −2.03] | −2.12 [−2.18, −2.05] | −1.62 [−1.67, −1.56] | −1.66 [−1.73, −1.60] |

Mean shortfall in bps [95% bootstrap CI over 6,890 paired test episodes per side]; RL = per-episode mean over the 5 seeds' agents.
Day-block CIs (resampling the 10 test days) of RL − rest at touch: +0.044 [+0.031, +0.056], +0.086 [+0.066, +0.103],
+0.024 [+0.017, +0.032], +0.046 [+0.034, +0.059].

**The learned policy saves 1.6–2.1 bp against crossing immediately or slicing with takers, but it does not beat the
simple rule "rest at the touch, cross what is left at the deadline": it costs 0.02–0.09 bp more**, with every CI above
zero. Every one of the 20 seed × case combinations is worse than rest-at-touch (by 0.008 to 0.175 bp). The same holds
for every horizon (H = 2 and 5 min: the all-variables agent costs 0.09–0.18 bp more than rest-at-touch) and for the
pessimistic and optimistic cancel rules (`execution_cost.md`). The saving against taking liquidity is real and large.
It comes almost entirely from the fee gap (3 bp retail, 2.5 bp pro) earned on ~97% passive fills, minus about 0.9 bp
of adverse price moves while waiting, and the simple rule captures all of it.

### Which state variables matter
None of them, out of sample (`state_value.md`, `state_value.png`). Each market variable's paired effect on cost, against
time × inventory alone, ranges from −0.03 to +0.08 bp and switches sign between sides and fee tiers. The seed-to-seed
spread of the same difference (−0.21 to +0.16 bp) is larger than any effect. Using all three together is consistently
worse (+0.08 to +0.16 bp): its 675-state table is learned from the same data and fits noise. Its 5 seeds agree on the
greedy action in only 1–11% of states, against 52–56% for the 25-state private agent (`seed_stability.md`).
Why the signals don't help:
(1) passive fills are nearly certain within a minute (97–98%), so there is little fill risk for a signal to manage;
(2) crossing costs 3 bp (retail) or 2.5 bp (pro) more in fees, while the 10 s price move a 5 s OFI or σ̂ bucket can
predict is a fraction of a bp, so the signals never justify crossing (the agents crossed in < 3.5% of test episodes);
(3) resting behind the touch gains only 0.025 bp per tick but is filled mainly by sweeps (93% of its passive volume),
which carry more adverse selection (60 s markout 1.23–1.26 vs 0.92–0.94 bp at the touch).

### What the learned policy does (plain English)
"Join the best bid (to buy) or best ask (to sell) straight away and stay there; if the price moves away, move with it;
never cross early; whatever is left at the end of the minute, buy or sell at the market." The private-state heatmap
(`policy_heatmap_private.png`) shows "touch" as the majority action in 98 of the 100 cells (4 side × fee panels × 25),
and "cross" in none, not even with 10 s and more than 80% left. The few "1 behind" / "3 behind" cells are where only
2–3 of the 5 seeds agree. In the one-variable agents (`policy_heatmap_market_*.png`), the market variables move a
handful of cells to "behind" with no consistent direction: adverse OFI, long queues and high σ̂ each produce a few
such cells, and different seeds pick different ones. These deviations are why RL loses slightly to rest-at-touch: with
returns this noisy (shortfall std ≈ 3 bp per episode) and true action gaps of tenths of a bp, a constant-step-size
Q-learner keeps some wrong greedy actions. The learning curves (`learning_curves.png`) are still falling slowly at
160,000 episodes: validation cost goes from ~3.2 to ~2.85 bp (retail) and ~0.45 to ~0.35 (pro), approaching but not
reaching rest-at-touch.

### Passive vs aggressive and markouts (`passive_fills_markouts.md`)
Rest-at-touch fills 97.8–98.0% passively. Of that passive volume, 36% is filled from the queue, 62% by sweeps
through our price and 2% by the book crossing onto it. The 60 s markout of its passive fills is 0.92–0.94 bp: the price
keeps moving our way after a fill, so waiting longer would on average have paid. Orders 1 or 3 ticks behind fill
92–93% passively, 93% of it by sweeps, with markouts of 1.23–1.26 bp. They are about 0.3 bp more expensive than the touch.
The deadline clean-up went beyond the displayed best size in at most 0.9% of episodes (0.2–0.4% for rest-at-touch and the headline agent).
The thinner-depth slippage rule therefore barely matters.

## Part B: market making with queue priority — WHAT-IF, not achievable for a retail participant

The only change from stage 4: on joining the touch, our order has f × the displayed queue ahead of it (f = 1: back of the
queue, as in stage 4; f = 0: front). Test days, 230 paired hours, proportional rule, 100 ms latency, stage 4's thresholds.

| mean P&L/h (USDT) | f = 1 (stage 4) | 0.25 | 0.15 | 0.1 | 0.05 | 0 (front) |
|---|---|---|---|---|---|---|
| always at touch, −0.5 bp | −27.9 | −13.7 | −5.2 [−8.4, −2.5] | **+2.9 [0.2, 5.5]** | +20.3 | +155.4 |
| always at touch, 0 bp | −74.4 | −79.9 | −81.3 | −82.0 | −81.4 | −58.4 |
| always at touch, +2 bp | −260.8 | −344.7 | −385.5 | −421.2 | −488.1 | −913.8 |
| touch + skew, −0.5 bp | −17.7 | −7.7 | −0.4 [−2.1, 1.1] | **+6.8 [5.6, 8.0]** | +21.9 | +142.9 |

**Break-even queue position: with the −0.5 bp rebate, between f = 0.15 (still losing) and f = 0.1 (profitable)**, for
both strategies. You must join with less than about a tenth of the displayed queue ahead of you. Without a rebate (0 bp)
or at the retail fee (+2 bp), quoting loses at every f, including the very front. At +2 bp more priority makes it *worse*,
because it brings more fills and each one loses. More priority turns fills from sweeps into queue fills (sweeps are
61% of volume at f = 1, 12% at f = 0), and queue fills are benign: 60 s markout 1.46 USDT/BTC at f = 1, 0.06 at f = 0, versus
~4.9 for sweeps at every f. The rebate (~2 USDT/BTC) pays for queue fills but never for sweeps.
(An earlier version interpolated a break-even of 0.23 from a coarser grid; that was an artefact of interpolating across a
jump, and is withdrawn.)

### What-if RL market maker (`mm_whatif.md`)
The pre-registered condition (some f clearly profitable) holds only with the −0.5 bp rebate, so a small tabular
Q-learning market maker was trained there, at the largest clearly profitable f, **f = 0.1**. Setup: 1 s decisions, stage 4's 16
actions, actions masked at ±10 lots, 15 states (inventory × OFI), 300 one-hour training episodes, 5 seeds. The inventory
penalty follows your rule: holding the limit for an hour costs the 60 s markout of the 10 lots of fills that got there
(1.23 USDT/BTC at f = 0.1 on the calibration days), so γ = 1.7e-5 per lot = 0.12 USDT per hour at the limit. **That is
formally matched but not material:** it is small next to a strategy earning ±3–7 USDT/h.

| WHAT-IF, f = 0.1, −0.5 bp | mean P&L / h (USDT) | BTC / h | mean max \|q\| (lots) |
|---|---|---|---|
| RL market maker (mean of 5 seeds) | **−11.95 [−13.80, −10.27]** | 6.2 | 9.6 |
| always at touch | +2.87 [0.08, 5.52] | 38.1 | 10.0 |
| touch + inventory skew | +6.83 [5.61, 8.00] | 25.9 | 2.9 |

The RL market maker loses money and is beaten by both hand-written rules (paired: −14.8 [−16.9, −12.8] and
−18.8 [−20.5, −17.2] USDT/h); every seed loses (−7.9 to −15.1). Its 5 seeds pick different actions in almost every state
and it drifts to the inventory limit. With hour-long, very noisy rewards, 300 episodes are far too few for even a 15 × 16 table,
and the one profitable edge here (+3 USDT/h for always-at-touch) is thin. We did not tune it further: it is a labelled what-if
at a queue priority a retail participant cannot get.

## Limitations
- The replay assumes our orders change nobody's behaviour. 0.5 BTC is 4–20% of a typical touch, so this is plausible for
  joining a queue but optimistic for the deadline clean-up. Depth beyond the best quote is not recorded and is assumed
  thinner than the touch.
- Latency is fixed at 100 ms; the order of events inside 100 ms is unknown (trades applied first, as in stage 4).
- The professional fee tier is an assumption, not a quoted schedule.
- RL agents were trained only under the proportional cancel rule. The test CIs resample episodes and, for the headline, days;
  they do not include training-seed uncertainty, which is shown separately as the per-seed range and is larger than the
  variable effects.
- Hyperparameters were chosen at the edge of the episode grid (160,000), and validation cost was still falling slowly. More
  training would likely close part of the gap to rest-at-touch, but nothing suggests it would beat it.
- Part B priority is a what-if knob, not something we modelled how to obtain.

## Charts
`execution_cost.png`, `policy_heatmap_private.png`, `policy_heatmap_market_retail.png`, `policy_heatmap_market_pro.png`,
`state_value.png`, `learning_curves.png`, `break_even_queue.png`.

## For stage 6 (README, demo, CV bullet): needs the owner's decision
1. **Headline framing.** Recommended: "RL execution agent saves ~2 bp (≈40%) vs taking liquidity, matching but not
   beating a simple rest-at-touch rule; market signals added nothing out of sample." Leading with "beats the baseline"
   would be false.
2. **Main horizon.** The report uses H = 1 min (H = 5 min made passive fills certain). Keep 1 min in the README, or show
   all three?
3. **Part B in the README:** include the break-even queue position (between 10% and 15% of the queue, rebate only) as a
   clearly labelled what-if, or leave it to this report?
4. **Demo scope:** recommended: replay one execution episode (rest-at-touch vs RL vs taker) plus the policy heatmap;
   no live RL training in Streamlit.
5. **Professional fee tier:** keep it as a stated assumption, or replace it with a sourced fee schedule before
   publishing.
