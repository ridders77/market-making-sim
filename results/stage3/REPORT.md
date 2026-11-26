> **Superseded by the stage 4 replay validation (added 2026-11-26, stage 5).** This report's P&L numbers come from the
> synthetic simulator. Replaying the same strategy on the real book and trades (results/stage4/REPORT.md, step 2) showed that
> the synthetic model predicts fill volume well (ratio 0.98–1.06) but **underestimates the markout of our fills by about 1.5×**
> (60 s markout ratio 1.49 [1.39, 1.60] at 0 bp). As a result, quoting wide (markout-aware A–S) at 0 bp **loses about −1.3 USDT/h**
> (−1.34 [−1.76, −0.89]) on the test hours instead of the **+2.0 USDT/h** the synthetic model predicted for the same hours.
> Adverse selection is even larger than reported below; read the stage 3 headline as optimistic. The text below is kept unchanged.

# Stage 3 report: realistic fills, adverse selection, flow-aware quoting

All numbers: 0.01 BTC lot, γ = 1e-4 per lot, T = 1 h, tick 0.1, 1,000 paired paths, 95% bootstrap CIs.
Details: `step0_*.md`, `model_choice.md`, `markouts.md`, `ofi.md`, `strategy_table.md`, `answers.md`.

## Step 0: lot size and fee-aware A–S
- The lot rule hardly changes λ beyond the touch: 0.01 BTC lot A −2.0%, k −0.3%; 0.1 BTC A −11%, k −3%
  (1 BTC: A −40%, k −13%). Prints beyond the touch are nearly all parts of large sweeps.
- P&L per hour (medium vol, A–S, fee 0): stage 2 782 USDT → σ_60s 774 → 0.01 BTC lot 7.74 → lot-rule λ 7.62 USDT.
  ~99% of the fall is the lot size; fill realism costs 1.6%. Per unit notional it is unchanged (0.75 bp).
  At its own 1 BTC lot, stage 2 overstated P&L by ~32% (782 vs 535 USDT/h with the 1-BTC lot rule).
- Fee-aware A–S (half-spread + c·S) at +1 bp: +2.04 vs A–S −2.49 USDT/h (medium, no informed flow).

## Step 1: pre-registered model choice (rule committed in c241c67 before scoring)
| model | score (mean abs log ratio, test days) | passes ≤ 0.4 |
|---|---|---|
| (a) relative units | 0.504 [0.272, 0.761] | no |
| (b) σ-conditional, trailing σ̂ | **0.322 [0.203, 0.452]** | **yes → chosen** |
Caveats: (b)'s CI crosses 0.4, and it over-predicts λ by ~36% on average (bias +0.31).

## Step 2: adverse selection in the data (calibration → test)
- 60 s markout beyond the touch: 0.96–1.16 bp calibration vs 0.98–1.16 bp test (stable in bps; in USDT ~2.7–3.3 → ~4.2–4.8
  as the price rose). At the touch: 0.14 → 0.22 bp. A quote 1–5 USDT from the mid is filled only by sweeps that move the mid
  ~3 USDT against it, more than the ~2 USDT the quote earns.
- OFI: best signal = 5 s imbalance (B−S)/(B+S). Next 5 s move: R² 0.37% [0.27, 0.58] in sample, **0.26% [0.13, 0.53] out of sample**;
  hit rate 58.5% → **56.2% [54.3, 58.3]**. Next 30 s: R² 0.09% → 0.06%. Raw OFI over 30–60 s has no out-of-sample power.
- The predictable part of a fill's markout is ~0.08 USDT of ~3 USDT.

## Step 5: comparison (informed flow ×1; P&L USDT/h, bps of notional in brackets)
| regime | fee | naive | A–S | fee-aware A–S | flow-aware A–S |
|---|---|---|---|---|---|
| medium | −0.5 | 1.48 (0.13) | **7.01 (0.60)** | 2.02 (0.09) | 5.77 (1.10) |
| medium | 0 | −4.00 (−0.37) | 1.14 (0.10) | 1.14 (0.10) | **2.50 (0.96)** |
| medium | +2 | −25.94 | −22.34 | −0.38 | 0.02 (stops trading: 0.4 fills/h) |
| high | 0 | −0.11 | 4.66 | 4.66 | **6.25** |
| low | 0 | −3.15 | −0.34 | −0.34 | **0.76** |

1. **Adverse selection share of A–S's profit:** 87% [84, 89] at 0 bp, 52% at −0.5 bp (medium);
   75–109% at 0 bp across regimes. Naive loses all of it and more.
2. **Flow-awareness recovers 11–26% at 0 bp (37% low vol at −0.5 bp), all of it from pricing the markout. The signal itself adds
   nothing measurable:** flow-aware − markout-aware = −0.004 [−0.06, 0.05] USDT/h (medium, 0 bp), and it stays
   within ±0.1 even at 2× the measured predictive power. At −0.5 bp plain A–S is best: its inventory skew already
   collects the rebate, and the per-fill markout logic ignores that a jump also revalues the remaining inventory.
3. **Break-even maker fee:** naive −0.50 / −0.37 / −0.01 bp; A–S −0.05 / +0.10 / +0.22 bp; fee-aware A–S −0.62 / +0.65 / +0.96 bp
   (low / medium / high). Markout- and flow-aware never lose on the grid; they stop trading from +2 bp (+3 high).
   Without informed flow, A–S breaks even at 0.60 / 0.73 / 0.86 bp. Nobody makes money at the 2 bp retail fee.
4. **Spread vs the real 1 tick:** naive / A–S 36–64×, fee-aware 10–177× (fee-dependent), flow-aware 67–124× at −0.5/0 bp,
   ~320× at 2 bp. The model still prices like a monopolist; the real 0.1 spread is a queue game it cannot represent.
