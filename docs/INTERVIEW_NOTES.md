# Interview notes

For each stage: the question, the method (with the key formula and one worked example), the result, and the three
questions I expect, with short answers. Every number comes from `results/` or DECISIONS.md.

---

## Stage 1 — Theory: does my Avellaneda–Stoikov implementation reproduce the paper?

**Question.** Can I implement A–S correctly, and does inventory-aware quoting reduce risk the way the paper says?

**Method.** The mid is arithmetic Brownian motion, dS = σ dW. A quote at distance δ from the mid fills with
intensity λ(δ) = A·e^(−kδ); in each step of length dt it fills with probability λ·dt (the paper's rule). With CARA
utility (risk aversion γ), the maker quotes around a reservation price and uses a spread that does not depend on q:

    r = S − q·γ·σ²·(T − t)
    spread = γ·σ²·(T − t) + (2/γ)·ln(1 + γ/k)
    bid = r − spread/2,  ask = r + spread/2

*Worked example* (paper setup, S = 100, σ = 2, k = 1.5, γ = 0.1, t = 0, T = 1, long q = 3):
r = 100 − 3·0.1·4·1 = **98.8**. spread = 0.1·4·1 + (2/0.1)·ln(1 + 0.1/1.5) = 0.4 + 20·0.0645 = **1.691**.
bid = 98.8 − 0.846 = **97.954**, ask = 98.8 + 0.846 = **99.646**. The ask is only 0.354 above the mid and the bid
2.046 below it, so the maker is much more likely to sell, which pulls q back to 0.

**Result.** 1,000 paired paths per γ match the paper within about one CI width. At γ = 0.1, A–S has mean / std profit
64.9 / 6.9 against the paper's 65.0 / 6.6, and the symmetric benchmark has 68.3 / 13.3 against 68.4 / 12.7. A–S
halves the P&L std (paired difference −6.4 [−7.3, −5.6]) for a slightly lower mean.

**Likely questions.**
1. *Where does r come from?* Holding q until T, wealth is normal with variance q²σ²(T − t). The CARA certainty
   equivalent is mean − (γ/2)·variance. The price at which I am indifferent to buying or selling one more unit,
   averaged over the two, is S − qγσ²(T − t).
2. *Why does the spread not depend on q?* In the paper's approximation, inventory only shifts the centre of the
   quotes. The width trades off the profit per fill against fill probability (the (2/γ)ln(1 + γ/k) term, which tends
   to 2/k when γ → 0) plus an inventory-risk premium.
3. *Why compare to a symmetric quote with the same average spread?* That isolates the effect of the skew. A
   different spread would change both the fill rate and the margin, and confound the comparison. Same seeds means
   the comparison is paired.

---

## Stage 2 — Calibration: what are σ, A and k for BTCUSDT, and do they hold out of sample?

**Question.** What parameters does the real market imply, and do they predict fills on later days?

**Method.** Data: Binance USD-M BTCUSDT best bid/ask and trades on 10 calibration days and 10 later test days, chosen
by a fixed volatility-tercile rule. σ is the realised volatility of mid changes. For λ(δ), take every 1 s window and
every distance δ on a 0.1 USDT grid. The window counts as a "fill at δ" if an aggressive buy printed at or above
mid₀ + δ (asks; bids are the mirror image). Then λ = −ln(1 − p)/Δt, and ln λ is fitted on δ by OLS over 0.1–5 USDT.

*Worked example.* The fit gives A = 0.154 /s and k = 0.461 /USDT. A quote 2 USDT from the mid fills at
0.154·e^(−0.922) = **0.061 /s**, about 220 times an hour. The risk-neutral best half-spread is 1/k = **2.17 USDT**.
A 2 bp maker fee at S = 27,500 costs **5.5 USDT** per fill, more than the 2 USDT the quote earns.

**Result.** The fit is excellent (R² = 0.999). Out of sample, fixed parameters fail the pre-registered test (error
0.75 against a pass mark of 0.4): BTC doubled, and λ rose with volatility. A σ-conditional model, chosen by a rule
pre-registered in stage 3, passes at 0.32 [0.20, 0.45]. With fees, every A–S configuration loses at 1–2 bp. The
optimal A–S spread is 33–165× the real one-tick spread.

**Likely questions.**
1. *Why exclude δ = 0 from the fit?* At the touch, fills depend on queue position, which this model does not have.
   λ jumps ~15× between δ = 0 and one tick. Excluding it amounts to assuming we are at the back of the queue, the
   honest assumption for a newcomer (stage 4 then models the queue explicitly).
2. *Why does the A–S spread come out so wide?* The model has one maker facing the whole flow, so it prices like a
   monopolist (≈ 1/k). The real book is competitive, with many makers at one tick competing on queue position and
   rebates.
3. *Why 1-second σ, and is it biased?* It is the most local estimate, but the 60 s variance is ~1.3× larger
   (positively autocorrelated mid changes), so σ_1s understates hourly inventory risk. From stage 3 on we use σ_60s.

---

## Stage 3 — Adverse selection: how much of the spread goes to informed traders, and can order flow help?

**Question.** How much does trading against informed flow cost, and can an order-flow signal avoid it?

**Method.** The markout of a trade is how far the mid moves against the passive side within h seconds:
markout = −s·(m_{t+h} − m₀), with s = +1 if the maker bought. Order-flow imbalance (OFI) over 5 s is
(B − S)/(B + S) of aggressive buy and sell volume. In the simulator, the signal drives a drift of the mid
(κ = 0.045 bp/s per unit of z, decaying with τ = 2.5 s), and each fill carries the measured markout jump J(δ). The
flow-aware strategy shifts r by the expected move D = κτ(1 − e^(−(T−t)/τ))·z and prices the markout like a fee.

*Worked example.* With z = +1 (buyers dominating) and an hour left, D = 0.045·2.5 ≈ **0.11 bp**, about 0.31 USDT at
27,400. Both quotes move up by 0.31 USDT, so the ask the buyers are about to hit is 0.31 further away. In comparison,
the average 60 s markout beyond the touch is ~1 bp ≈ 2.7 USDT.

**Result.** Markouts are ~1 bp at 60 s, stable from calibration to test days. OFI predicts the next 5 s weakly out of
sample (R² 0.26% [0.13, 0.53], hit rate 56%). In simulation, adverse selection takes 87% [84, 89] of A–S's profit at
0 bp. Pricing the markout recovers 11–26%, and the signal itself adds nothing measurable. **Superseded:** the stage 4
replay showed that the markout on our own fills is 1.5× this, so the synthetic profits were optimistic.

**Likely questions.**
1. *Why is a 56% hit rate worthless?* The predicted move is ~0.1 bp per unit of z, a few cents per BTC, against a
   ~2.7 USDT markout and latency. Being right slightly more often about a tiny move does not cover the cost.
2. *Why not add J(δ) to each half-spread directly?* The optimum is δ* = argmax e^(−kδ)(δ − cS − J(δ)). For a flat J
   this is the fee argument (shift by J). When J rises with δ, the right shift is smaller. I derived and tested it.
3. *How do you know the simulator is right?* I didn't, which is why stage 4 checked it on the real book: fill
   volume was right, adverse selection was 1.5× too small.

---

## Stage 4 — Replay with a queue model: can a small participant make money passively?

**Question.** Put virtual orders into the recorded order book with a realistic queue. Does any passive strategy profit?

**Method.** 100 ms snapshots of the best bid/ask and their sizes, plus every trade. Our 0.01 BTC order joins at the
back of the queue (queue ahead Q = displayed size). Each 100 ms: (1) a trade through our price fills us; (2) volume
V traded at our price first reduces Q, and only the excess fills us; (3) an unexplained size decrease
C = max(0, size_prev − size_min − V) is a cancellation, split by a rule; (4) Q is capped at the smallest displayed size.
Cancellations ahead of us: pessimistic 0, proportional C·Q/(size_prev − V), optimistic min(C, Q).

*Worked example.* We join a bid showing 5 BTC (Q = 5), and 1 BTC more joins behind us, so the level shows 6.
In the next 100 ms V = 3 BTC trade at our price (Q = 2 ahead, 1 behind), and the level's smallest size is 2.5 BTC.
So C = 6 − 2.5 − 3 = 0.5 BTC was cancelled. Proportional: 0.5·2/3 = 0.33 of it was ahead of us, so Q = 1.67.
Pessimistic: Q = 2. Optimistic: Q = 1.5. The cap Q ≤ 2.5 does not bind. A later sweep through our price fills us
under every rule.

**Result.** All 108 scenarios lose with CI below zero. At 0 bp, always-at-touch loses −74 USDT/h; with the best rebate
the least-bad case is −5.6 [−7.7, −3.7]. Sweeps make up 58% of our touch volume, with a 60 s markout of 4.94 USDT/BTC
against 1.46 for queue fills. Per BTC every touch variant loses about the same ~0.8 bp: the "improvements" come from
trading less, not from better fills.

**Likely questions.**
1. *Isn't this just a pessimistic queue model?* The three cancel rules bracket it, and even the optimistic rule with
   0 ms latency loses. The cap on Q is a physical constraint (orders ahead only leave), not a pessimistic assumption.
2. *What is the main weakness?* Our orders are invisible: nobody reacts to them, and we see only the top of the book.
   For 0.01 BTC against a ~5 BTC queue that is reasonable, and I state it as a limitation.
3. *Why does lower latency make it worse?* More fills, and each fill loses on average. Speed only helps if it gets
   you *queue priority* or lets you cancel before a sweep, not just more of the same fills.

---

## Stage 5A — RL for optimal execution: can a learned policy beat simple rules?

**Question.** Buying or selling 0.5 BTC within 1 minute, can tabular RL beat taking liquidity, TWAP or resting at
the touch?

**Method.** Following Nevmyvaka, Feng & Kearns (2006): decide every 10 s (6 decisions) between cross, touch, 1 behind
and 3 behind for the whole remainder, and cross whatever is left at the deadline. The state is time left × inventory
left (5×5), optionally plus OFI, queue size or σ̂ (3 buckets each). The order keeps its queue place unless the action
asks for a new price. Cost = implementation shortfall in bps, fees included:

    IS = 1e4 / (Q·m₀) · Σ_j [ side·x_j·(p_j − m₀) + f_j·x_j·p_j ]

Training uses Q-learning (not backward induction, because the queue place makes outcomes path-dependent) with
160,000 episodes per agent and 5 seeds. Hyperparameters were chosen on 2 validation days, and the headline agent was
pre-registered.

*Worked example* (retail fees, m₀ = 40,000, 1-tick spread). Immediate taker: buy at 40,000.05, so the price part is
0.5·0.05 = 0.025 USDT and the fee is 5 bp·20,000 = 10 USDT. IS = 1e4·10.025/20,000 = **5.01 bp** (measured: 5.06).
Rest at the bid and fill there: the price part is −0.025 and the fee 2 bp·20,000 = 4 USDT, so IS = **1.99 bp**. The
measured 2.9 bp adds ~0.9 bp of adverse price moves while waiting and the occasional taker clean-up.

**Result.** RL saves 1.6–2.1 bp against crossing (buy, retail −2.09 [−2.14, −2.03]). It costs 0.02–0.09 bp more than
rest-at-touch, with every CI above zero and all 20 seed × case combinations worse. It learns "touch" in 98 of 100
cells. No market variable helps out of sample, and all three together overfit (the 5 seeds agree on 1–11% of states).

**Likely questions.**
1. *So RL failed?* It matched the best simple rule and found it without being told. The finding is that under these
   fees the problem is easy: the 3 bp maker/taker fee gap dwarfs anything a signal can predict in 10 s, and passive
   fills are 97–98% certain within a minute.
2. *Why is it slightly worse than the rule?* The action gaps are tenths of a bp, the per-episode noise is ~3 bp, and
   a constant-step-size learner keeps a few wrong greedy actions. More training would shrink the gap but not reverse it.
3. *Why 1 minute and not 5?* At 5 minutes the passive order filled completely in 100% of training episodes, so there
   was no trade-off to learn. 2 and 5 minutes are shown as sensitivity checks.

---

## Stage 5B — What-if: how much queue priority does a market maker need?

**Question.** If we could join the queue nearer the front, at what point would touch market making pay?

**Method.** On joining, the queue ahead is f × the displayed size, for f ∈ {1, 0.5, 0.25, 0.15, 0.1, 0.05, 0.02, 0}.
Everything else is as in stage 4, with the same thresholds and the same 230 test hours. The break-even is reported as
a bracket (last losing f, first profitable f), without interpolation.

*Worked example.* With a 5 BTC queue, f = 0.25 still leaves 1.25 BTC ahead of us. That is more than most single
trades, so we are mostly reached by sweeps. At f = 0.1 it is 0.5 BTC, which normal trades eat, so queue fills dominate.

**Result.** With the −0.5 bp rebate, f = 0.15 loses (−5.2 [−8.4, −2.5]) and f = 0.1 profits (+2.9 [0.2, 5.5]). At 0 or
+2 bp it loses at every f. A small RL market maker trained at f = 0.1 lost −12.0 USDT/h and both rules beat it.

**Likely questions.**
1. *Why not interpolate the break-even?* P&L is not linear in f (it jumps when the queue ahead drops below typical
   trade sizes). An earlier linear interpolation gave 0.23 with a degenerate CI. I withdrew it and reported the bracket.
2. *How would anyone get f = 0.1?* By being early at a new price level, which needs speed and anticipation. That is
   the business of professional market makers, and why this is labelled a what-if.
3. *Why did the RL market maker fail?* 300 hour-long episodes are far too few for noisy hourly rewards, and the edge
   (+3 USDT/h) is thin. I didn't tune it further, because it is an illustration, not a result.

---

## Decisions I'd defend (from DECISIONS.md)

- **Pre-registration.** The rules were committed before the test data was scored: the fill-model choice (stage 3,
  commit c241c67), the strategy thresholds per fee (stage 4), the headline RL agent and its comparison (stage 5), and
  the rule for training a what-if market maker (Part B). When the pre-registered fixed-parameter model failed in
  stage 2, I said so. The σ-conditional fix was flagged as chosen after seeing the result, then re-tested under a
  pre-registered rule in stage 3.
- **Train / validation / test split by date.** Calibration days come strictly before test days, with each volatility
  regime in both halves. RL trains on 8 days, tunes on 2 and is tested once on 10. Bootstrap CIs over paired hours or
  episodes, plus day-block CIs because hours within a day are correlated.
- **Keeping the fixed grid.** In stage 4 every touch threshold landed at the low-activity edge of its grid. I did not
  widen the grid after seeing that: extending it only converges on "don't quote", and widening after the fact is a
  look at the data. In stage 5 the episode budget also sat at the grid edge, and that is stated as a limitation.
- **Refusing to interpolate the break-even.** The coarse grid gave an interpolated 0.23 that was an artefact of a
  jump. It was withdrawn, the grid was refined, and the result is reported as a bracket with significance at each end.
- **Bugs caught and fixed before reporting.**
  - Paper targets typed from memory included a γ = 0.5 row that does not exist. Corrected against the paper, and the
    investigation built on it was removed.
  - With large rebates, quotes went through the mid and booked rebates no exchange would pay (a crossing order is a
    taker). Fixed with a minimum distance from the mid.
  - A markout aggregation stored NaN instead of 0 for zero-fill hours. Fixed exactly, only where n_fills = 0.
  - The memory-limited parallel run was restarted with fewer workers, and caching meant no results were lost.
- **Honest costs.** Fees run everywhere (−0.5 / 0 / 1 / 2 bp), quotes respect the 0.1 tick, and orders are post-only.
  The synthetic results were marked superseded once the replay contradicted them.
