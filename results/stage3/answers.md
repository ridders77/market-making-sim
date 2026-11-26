# Stage 3: answers

## 1. How much of A–S's profit (no informed flow) was really adverse selection?

Same seeds, same strategy; the only change is switching informed flow on (×1). Share lost = (P&L without − P&L with) / P&L without. 95% CI: paired bootstrap over paths.

| regime | fee (bp) | strategy | P&L without (USDT/h) | P&L with informed flow | difference | share of profit lost |
|---|---|---|---|---|---|---|
| low | -0.5 | A–S | 7.21 | 2.92 | 4.29 [4.19, 4.39] | 60% [58%, 61%] |
| low | -0.5 | naive | 7.16 | 0.01 | 7.15 [6.67, 7.66] | 100% [91%, 109%] |
| low | +0 | A–S | 3.95 | -0.34 | 4.29 [4.19, 4.39] | 109% [105%, 112%] |
| low | +0 | naive | 4.00 | -3.15 | 7.15 [6.67, 7.66] | 179% [159%, 201%] |
| low | +2 | A–S | -9.10 | -13.39 | 4.28 [4.18, 4.39] | n/a (no profit to lose) |
| low | +2 | naive | -8.64 | -15.79 | 7.16 [6.67, 7.66] | n/a (no profit to lose) |
| medium | -0.5 | A–S | 14.51 | 7.01 | 7.50 [7.32, 7.68] | 52% [50%, 53%] |
| medium | -0.5 | naive | 14.36 | 1.48 | 12.88 [11.49, 14.23] | 90% [79%, 101%] |
| medium | +0 | A–S | 8.62 | 1.14 | 7.48 [7.30, 7.66] | 87% [84%, 89%] |
| medium | +0 | naive | 8.87 | -4.00 | 12.87 [11.49, 14.22] | 145% [126%, 167%] |
| medium | +2 | A–S | -14.94 | -22.34 | 7.41 [7.23, 7.59] | n/a (no profit to lose) |
| medium | +2 | naive | -13.07 | -25.94 | 12.87 [11.48, 14.24] | n/a (no profit to lose) |
| high | -0.5 | A–S | 29.63 | 15.44 | 14.19 [13.90, 14.48] | 48% [47%, 49%] |
| high | -0.5 | naive | 29.54 | 9.17 | 20.38 [17.13, 23.55] | 69% [58%, 81%] |
| high | +0 | A–S | 18.77 | 4.66 | 14.10 [13.81, 14.40] | 75% [74%, 77%] |
| high | +0 | naive | 20.27 | -0.11 | 20.38 [17.14, 23.55] | 101% [84%, 121%] |
| high | +2 | A–S | -24.68 | -38.44 | 13.75 [13.46, 14.04] | n/a (no profit to lose) |
| high | +2 | naive | -16.85 | -37.24 | 20.39 [17.15, 23.54] | n/a (no profit to lose) |

## 2. Does flow-awareness recover it?

Adverse-selection cost = fee-aware A–S without informed flow − with it. Recovered = (strategy − fee-aware A–S), both with informed flow ×1, as a share of that cost. The markout-aware ablation (no signal) separates pricing the average markout from reading the order flow.

| regime | fee (bp) | AS cost (USDT/h) | markout-aware − fee-aware | flow-aware − markout-aware (signal value) | flow-aware recovered share | flow-aware vs no-flow fee-aware |
|---|---|---|---|---|---|---|
| low | -0.5 | 10.94 [10.76, 11.11] | 4.03 [3.83, 4.22] | -0.006 [-0.053, 0.042] | 37% [35%, 38%] | -6.91 [-7.05, -6.78] |
| low | +0 | 4.29 [4.19, 4.39] | 1.11 [0.97, 1.24] | -0.003 [-0.041, 0.036] | 26% [23%, 29%] | -3.18 [-3.30, -3.07] |
| low | +2 | 0.28 [0.24, 0.31] | 0.18 [0.13, 0.24] | 0.002 [-0.000, 0.006] | 68% [47%, 88%] | -0.09 [-0.15, -0.03] |
| medium | -0.5 | 15.44 [15.18, 15.70] | 3.73 [3.45, 4.02] | 0.023 [-0.048, 0.093] | 24% [23%, 26%] | -11.69 [-11.87, -11.51] |
| medium | +0 | 7.48 [7.31, 7.66] | 1.36 [1.15, 1.56] | -0.004 [-0.061, 0.051] | 18% [16%, 21%] | -6.12 [-6.27, -5.97] |
| medium | +2 | 0.89 [0.83, 0.95] | 0.39 [0.28, 0.49] | 0.002 [-0.007, 0.011] | 44% [32%, 54%] | -0.50 [-0.60, -0.41] |
| high | -0.5 | 23.84 [23.45, 24.25] | 2.37 [1.95, 2.77] | -0.056 [-0.167, 0.052] | 10% [8%, 11%] | -21.52 [-21.75, -21.29] |
| high | +0 | 14.10 [13.83, 14.39] | 1.58 [1.29, 1.87] | 0.014 [-0.069, 0.098] | 11% [9%, 13%] | -12.51 [-12.71, -12.31] |
| high | +2 | 2.41 [2.32, 2.51] | 0.79 [0.64, 0.96] | -0.001 [-0.023, 0.021] | 33% [27%, 39%] | -1.62 [-1.75, -1.48] |

### Predictive-power sweep (medium regime)

The true drift per unit signal is scaled by ×0…×2 (×1 = measured); the flow-aware strategy always assumes ×1. Mean P&L (USDT/h) [95% CI].

| fee (bp) | strategy | ×0 | ×0.5 | ×1 | ×1.5 | ×2 |
|---|---|---|---|---|---|---|
| -0.5 | naive | 1.83 | 1.60 | 1.48 | 1.55 | 1.01 |
| -0.5 | A–S | 7.31 | 7.14 | 7.01 | 6.85 | 6.68 |
| -0.5 | fee-aware A–S | 2.65 | 2.34 | 2.02 | 1.77 | 1.42 |
| -0.5 | markout-aware A–S (no signal) | 5.92 | 5.80 | 5.75 | 5.69 | 5.61 |
| -0.5 | flow-aware A–S | 5.81 | 5.76 | 5.77 | 5.72 | 5.70 |
| -0.5 | *signal value: flow-aware − markout-aware* | -0.109 [-0.184, -0.037] | -0.041 [-0.112, 0.032] | 0.023 [-0.046, 0.094] | 0.038 [-0.030, 0.110] | 0.091 [0.021, 0.163] |
| +0 | naive | -3.66 | -3.88 | -4.00 | -3.93 | -4.48 |
| +0 | A–S | 1.44 | 1.27 | 1.14 | 0.98 | 0.81 |
| +0 | fee-aware A–S | 1.44 | 1.27 | 1.14 | 0.98 | 0.81 |
| +0 | markout-aware A–S (no signal) | 2.59 | 2.55 | 2.50 | 2.46 | 2.43 |
| +0 | flow-aware A–S | 2.54 | 2.52 | 2.50 | 2.48 | 2.47 |
| +0 | *signal value: flow-aware − markout-aware* | -0.054 [-0.111, 0.002] | -0.039 [-0.094, 0.021] | -0.004 [-0.062, 0.053] | 0.027 [-0.031, 0.086] | 0.041 [-0.018, 0.100] |
| +2 | naive | -25.59 | -25.82 | -25.94 | -25.87 | -26.42 |
| +2 | A–S | -22.04 | -22.21 | -22.34 | -22.50 | -22.67 |
| +2 | fee-aware A–S | -0.33 | -0.36 | -0.38 | -0.39 | -0.41 |
| +2 | markout-aware A–S (no signal) | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 |
| +2 | flow-aware A–S | 0.02 | 0.01 | 0.02 | 0.02 | 0.02 |
| +2 | *signal value: flow-aware − markout-aware* | 0.005 [-0.003, 0.013] | 0.002 [-0.007, 0.010] | 0.002 [-0.007, 0.011] | 0.005 [-0.004, 0.013] | 0.003 [-0.005, 0.011] |

## 3. Break-even maker fee (bp) — mean P&L = 0

Fixed quotes (naive, A–S): f* = P&L at zero fee / traded notional. Fee-dependent quotes: interpolated on the fee grid [-1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 3.0, 4.0] with the strategy re-optimised for each fee. Positive = the strategy can pay this fee; negative = it needs a rebate. 95% CI: bootstrap over paths.

| regime | environment | naive | A–S | fee-aware A–S | markout-aware A–S (no signal) | flow-aware A–S |
|---|---|---|---|---|---|---|
| low | no informed flow | 0.63 [0.57, 0.70] | 0.60 [0.59, 0.62] | — | — | — |
| low | informed flow ×1 | -0.50 [-0.60, -0.40] | -0.05 [-0.07, -0.03] | -0.62 [-0.64, -0.61] | never loses; stops trading (<1 fill/h) from +2 bp | never loses; stops trading (<1 fill/h) from +2 bp |
| medium | no informed flow | 0.81 [0.72, 0.89] | 0.73 [0.72, 0.74] | — | — | — |
| medium | informed flow ×1 | -0.37 [-0.51, -0.22] | 0.10 [0.08, 0.11] | 0.65 [0.49, 0.79] | never loses; stops trading (<1 fill/h) from +2 bp | never loses; stops trading (<1 fill/h) from +2 bp |
| high | no informed flow | 1.09 [0.99, 1.20] | 0.86 [0.86, 0.87] | — | — | — |
| high | informed flow ×1 | -0.01 [-0.21, 0.19] | 0.22 [0.20, 0.23] | 0.96 [0.90, 1.13] | never loses; stops trading (<1 fill/h) from +3 bp | never loses; stops trading (<1 fill/h) from +3 bp |

Mean P&L (USDT/h) on the fee grid, informed flow ×1:

| regime | strategy | -1 | -0.5 | +0 | +0.5 | +1 | +2 | +3 | +4 |
|---|---|---|---|---|---|---|---|---|---|
| low | naive | 3.17 | 0.01 | -3.15 | -6.31 | -9.47 | -15.79 | -22.11 | -28.44 |
| low | A–S | 6.18 | 2.92 | -0.34 | -3.60 | -6.87 | -13.39 | -19.91 | -26.44 |
| low | fee-aware A–S | 5.09 | -1.69 | -0.34 | -0.34 | -0.32 | -0.18 | -0.07 | -0.01 |
| low | markout-aware A–S (no signal) | 5.07 | 2.34 | 0.77 | 0.16 | 0.04 | 0.00 | 0.00 | 0.00 |
| low | flow-aware A–S | 6.00 | 2.33 | 0.76 | 0.16 | 0.03 | 0.00 | 0.00 | 0.00 |
| medium | naive | 6.96 | 1.48 | -4.00 | -9.49 | -14.97 | -25.94 | -36.90 | -47.87 |
| medium | A–S | 12.88 | 7.01 | 1.14 | -4.73 | -10.60 | -22.34 | -34.09 | -45.83 |
| medium | fee-aware A–S | 11.15 | 2.02 | 1.14 | 0.14 | -0.32 | -0.38 | -0.20 | -0.08 |
| medium | markout-aware A–S (no signal) | 11.66 | 5.75 | 2.50 | 0.84 | 0.21 | 0.01 | 0.00 | 0.00 |
| medium | flow-aware A–S | 12.74 | 5.77 | 2.50 | 0.86 | 0.22 | 0.02 | 0.00 | 0.00 |
| high | naive | 18.45 | 9.17 | -0.11 | -9.39 | -18.68 | -37.24 | -55.80 | -74.36 |
| high | A–S | 26.21 | 15.44 | 4.66 | -6.11 | -16.89 | -38.44 | -59.99 | -81.54 |
| high | fee-aware A–S | 27.45 | 10.75 | 4.66 | 1.34 | -0.10 | -0.76 | -0.46 | -0.22 |
| high | markout-aware A–S (no signal) | 27.64 | 13.12 | 6.24 | 2.74 | 0.86 | 0.03 | 0.02 | 0.01 |
| high | flow-aware A–S | 28.78 | 13.06 | 6.25 | 2.81 | 0.85 | 0.03 | 0.01 | 0.01 |

## 4. Reality check: quoted spread vs the real 1-tick (0.1 USDT) spread

Time-averaged quoted spread (ask − bid, USDT) with informed flow ×1, and fills per hour.

| regime | fee (bp) | naive | A–S | fee-aware A–S | markout-aware A–S (no signal) | flow-aware A–S |
|---|---|---|---|---|---|---|
| low | -0.5 | 3.6 (36×, 227/h) | 3.6 (36×, 234/h) | 1.0 (10×, 513/h) | 6.7 (67×, 90/h) | 6.7 (67×, 87/h) |
| low | +0 | 3.6 (36×, 227/h) | 3.6 (36×, 234/h) | 3.6 (36×, 234/h) | 9.5 (95×, 38/h) | 9.5 (95×, 37/h) |
| low | +2 | 3.6 (36×, 227/h) | 3.6 (36×, 234/h) | 14.7 (147×, 7/h) | 32.0 (320×, 0/h) | 32.0 (320×, 0/h) |
| medium | -0.5 | 4.6 (46×, 400/h) | 4.6 (46×, 428/h) | 2.0 (20×, 820/h) | 7.6 (76×, 197/h) | 7.6 (76×, 192/h) |
| medium | +0 | 4.6 (46×, 400/h) | 4.6 (46×, 428/h) | 4.6 (46×, 428/h) | 10.3 (103×, 98/h) | 10.3 (103×, 95/h) |
| medium | +2 | 4.6 (46×, 400/h) | 4.6 (46×, 428/h) | 15.5 (155×, 25/h) | 31.8 (318×, 0/h) | 31.8 (318×, 0/h) |
| high | -0.5 | 6.3 (63×, 656/h) | 6.4 (64×, 762/h) | 3.8 (38×, 1303/h) | 9.3 (93×, 412/h) | 9.3 (93×, 403/h) |
| high | +0 | 6.3 (63×, 656/h) | 6.4 (64×, 762/h) | 6.4 (64×, 762/h) | 12.4 (124×, 216/h) | 12.4 (124×, 211/h) |
| high | +2 | 6.3 (63×, 656/h) | 6.4 (64×, 762/h) | 17.7 (177×, 70/h) | 35.8 (358×, 1/h) | 35.8 (358×, 1/h) |
