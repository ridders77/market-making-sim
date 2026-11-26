# Markouts: our fills at the touch vs further out (test days)

Markout h = −s (m_{t+h} − m₀): the mid move against us h seconds after the fill (USDT per BTC filled; positive = adverse), m₀ = the mid at the decision time of the step in which we were filled. Quantity-weighted; fills in the first 59 minutes of each episode. 0.01 BTC lot, 100 ms latency, fee-0 configurations. 60 s CI: bootstrap over episodes.

## By strategy and how the fill happened

*queue*: traded volume at our price exceeded the queue ahead of us; *through*: an aggressor traded beyond our price (the whole level was swept, or our quote was stale above the market); *cross*: the opposite quote moved onto our price.

| cancel rule | strategy | fills | BTC | share | spread captured | 1 s | 5 s | 30 s | 60 s [CI] | 60 s (bp) |
|---|---|---|---|---|---|---|---|---|---|---|
| pessimistic | always at touch | all | 4214.1 | 100% | 0.111 | 3.22 | 3.79 | 3.91 | 3.86 [3.62, 4.11] | 0.861 |
| pessimistic | always at touch | cross | 155.6 | 4% | 0.102 | 2.40 | 3.27 | 3.80 | 4.14 [3.47, 4.80] | 0.935 |
| pessimistic | always at touch | queue | 1287.8 | 31% | 0.051 | 1.05 | 1.48 | 1.55 | 1.50 [1.25, 1.76] | 0.363 |
| pessimistic | always at touch | through | 2770.7 | 66% | 0.139 | 4.28 | 4.89 | 5.02 | 4.95 [4.63, 5.26] | 1.088 |
| pessimistic | touch + inventory skew | all | 2363.8 | 100% | 0.139 | 3.65 | 4.18 | 4.24 | 4.23 [3.92, 4.53] | 0.914 |
| pessimistic | touch + inventory skew | cross | 74.8 | 3% | 0.141 | 2.90 | 3.68 | 4.31 | 4.92 [3.90, 6.01] | 1.066 |
| pessimistic | touch + inventory skew | queue | 636.8 | 27% | 0.052 | 1.25 | 1.64 | 1.60 | 1.53 [1.21, 1.86] | 0.370 |
| pessimistic | touch + inventory skew | through | 1652.1 | 70% | 0.172 | 4.61 | 5.18 | 5.26 | 5.24 [4.81, 5.66] | 1.117 |
| pessimistic | touch + queue-aware | all | 1285.6 | 100% | 0.079 | 2.48 | 3.22 | 3.50 | 3.55 [3.27, 3.84] | 0.855 |
| pessimistic | touch + queue-aware | cross | 70.6 | 5% | 0.059 | 1.81 | 2.89 | 3.29 | 4.19 [3.42, 5.02] | 1.015 |
| pessimistic | touch + queue-aware | queue | 365.0 | 28% | 0.050 | 0.81 | 1.36 | 1.57 | 1.60 [1.24, 1.98] | 0.404 |
| pessimistic | touch + queue-aware | through | 850.1 | 66% | 0.093 | 3.25 | 4.05 | 4.34 | 4.33 [4.01, 4.67] | 1.035 |
| pessimistic | touch + OFI-aware | all | 1906.7 | 100% | 0.111 | 3.72 | 4.33 | 4.40 | 4.44 [4.06, 4.84] | 0.959 |
| pessimistic | touch + OFI-aware | cross | 69.7 | 4% | 0.117 | 2.73 | 3.57 | 3.74 | 3.97 [2.89, 5.05] | 0.914 |
| pessimistic | touch + OFI-aware | queue | 473.8 | 25% | 0.052 | 1.37 | 1.77 | 1.76 | 1.79 [1.37, 2.28] | 0.421 |
| pessimistic | touch + OFI-aware | through | 1363.2 | 71% | 0.132 | 4.59 | 5.25 | 5.36 | 5.38 [4.91, 5.85] | 1.148 |
| pessimistic | markout-aware A–S (quote wide) | all | 186.6 | 100% | 5.448 | 7.61 | 7.91 | 7.56 | 7.29 [6.83, 7.82] | 1.806 |
| pessimistic | markout-aware A–S (quote wide) | cross | 7.5 | 4% | 2.965 | 4.14 | 4.25 | 3.32 | 3.38 [1.39, 5.37] | 0.784 |
| pessimistic | markout-aware A–S (quote wide) | queue | 0.4 | 0% | 0.059 | -0.53 | -0.58 | 0.33 | 1.96 [-6.90, 11.19] | 0.584 |
| pessimistic | markout-aware A–S (quote wide) | through | 178.7 | 96% | 5.564 | 7.77 | 8.08 | 7.76 | 7.47 [6.99, 8.04] | 1.851 |
| proportional | always at touch | all | 4728.7 | 100% | 0.104 | 2.92 | 3.48 | 3.62 | 3.56 [3.33, 3.80] | 0.795 |
| proportional | always at touch | cross | 142.0 | 3% | 0.107 | 2.41 | 3.28 | 3.88 | 4.22 [3.56, 4.93] | 0.949 |
| proportional | always at touch | queue | 1839.6 | 39% | 0.051 | 0.92 | 1.38 | 1.48 | 1.46 [1.28, 1.64] | 0.351 |
| proportional | always at touch | through | 2747.1 | 58% | 0.140 | 4.28 | 4.90 | 5.03 | 4.94 [4.62, 5.24] | 1.085 |
| proportional | touch + inventory skew | all | 2782.2 | 100% | 0.126 | 3.25 | 3.76 | 3.84 | 3.83 [3.55, 4.12] | 0.830 |
| proportional | touch + inventory skew | cross | 70.2 | 3% | 0.149 | 2.85 | 3.65 | 4.51 | 5.18 [4.13, 6.30] | 1.103 |
| proportional | touch + inventory skew | queue | 1017.6 | 37% | 0.052 | 1.02 | 1.45 | 1.48 | 1.46 [1.25, 1.67] | 0.349 |
| proportional | touch + inventory skew | through | 1694.4 | 61% | 0.170 | 4.60 | 5.16 | 5.24 | 5.20 [4.78, 5.62] | 1.108 |
| proportional | touch + queue-aware | all | 1304.1 | 100% | 0.078 | 2.44 | 3.18 | 3.47 | 3.51 [3.24, 3.81] | 0.846 |
| proportional | touch + queue-aware | cross | 69.0 | 5% | 0.059 | 1.82 | 2.89 | 3.32 | 4.20 [3.39, 5.04] | 1.019 |
| proportional | touch + queue-aware | queue | 393.9 | 30% | 0.050 | 0.81 | 1.40 | 1.62 | 1.67 [1.32, 2.03] | 0.415 |
| proportional | touch + queue-aware | through | 841.2 | 65% | 0.093 | 3.26 | 4.04 | 4.35 | 4.32 [4.00, 4.67] | 1.034 |
| proportional | touch + OFI-aware | all | 2118.2 | 100% | 0.105 | 3.40 | 4.01 | 4.11 | 4.14 [3.79, 4.52] | 0.898 |
| proportional | touch + OFI-aware | cross | 63.3 | 3% | 0.124 | 2.72 | 3.55 | 4.03 | 4.50 [3.41, 5.63] | 1.013 |
| proportional | touch + OFI-aware | queue | 699.2 | 33% | 0.051 | 1.17 | 1.63 | 1.67 | 1.68 [1.32, 2.09] | 0.394 |
| proportional | touch + OFI-aware | through | 1355.7 | 64% | 0.131 | 4.59 | 5.26 | 5.38 | 5.39 [4.93, 5.86] | 1.153 |
| proportional | markout-aware A–S (quote wide) | all | 186.6 | 100% | 5.448 | 7.61 | 7.91 | 7.56 | 7.29 [6.83, 7.82] | 1.806 |
| proportional | markout-aware A–S (quote wide) | cross | 7.5 | 4% | 2.965 | 4.14 | 4.25 | 3.32 | 3.38 [1.39, 5.37] | 0.784 |
| proportional | markout-aware A–S (quote wide) | queue | 0.4 | 0% | 0.059 | -0.53 | -0.58 | 0.33 | 1.96 [-6.90, 11.19] | 0.584 |
| proportional | markout-aware A–S (quote wide) | through | 178.7 | 96% | 5.564 | 7.77 | 8.08 | 7.76 | 7.47 [6.99, 8.04] | 1.851 |
| optimistic | always at touch | all | 5867.1 | 100% | 0.094 | 2.39 | 2.89 | 3.01 | 2.97 [2.77, 3.19] | 0.664 |
| optimistic | always at touch | cross | 140.7 | 2% | 0.109 | 2.39 | 3.27 | 3.83 | 4.21 [3.51, 4.95] | 0.946 |
| optimistic | always at touch | queue | 2981.9 | 51% | 0.051 | 0.64 | 1.03 | 1.12 | 1.12 [1.01, 1.24] | 0.267 |
| optimistic | always at touch | through | 2744.5 | 47% | 0.140 | 4.28 | 4.89 | 5.02 | 4.92 [4.61, 5.23] | 1.081 |
| optimistic | touch + inventory skew | all | 3604.1 | 100% | 0.109 | 2.53 | 2.97 | 3.04 | 3.03 [2.78, 3.29] | 0.662 |
| optimistic | touch + inventory skew | cross | 69.9 | 2% | 0.149 | 2.80 | 3.67 | 4.40 | 5.16 [4.06, 6.39] | 1.099 |
| optimistic | touch + inventory skew | queue | 1842.8 | 51% | 0.051 | 0.64 | 0.96 | 0.99 | 1.00 [0.87, 1.14] | 0.238 |
| optimistic | touch + inventory skew | through | 1691.4 | 47% | 0.170 | 4.57 | 5.13 | 5.21 | 5.16 [4.75, 5.55] | 1.106 |
| optimistic | touch + queue-aware | all | 1324.7 | 100% | 0.078 | 2.39 | 3.14 | 3.42 | 3.48 [3.22, 3.77] | 0.837 |
| optimistic | touch + queue-aware | cross | 68.9 | 5% | 0.059 | 1.82 | 2.90 | 3.28 | 4.17 [3.36, 5.01] | 1.015 |
| optimistic | touch + queue-aware | queue | 422.8 | 32% | 0.050 | 0.78 | 1.39 | 1.63 | 1.68 [1.34, 2.04] | 0.415 |
| optimistic | touch + queue-aware | through | 833.0 | 63% | 0.094 | 3.26 | 4.05 | 4.34 | 4.33 [4.02, 4.67] | 1.037 |
| optimistic | touch + OFI-aware | all | 2638.3 | 100% | 0.093 | 2.74 | 3.27 | 3.43 | 3.46 [3.14, 3.79] | 0.751 |
| optimistic | touch + OFI-aware | cross | 62.0 | 2% | 0.124 | 2.72 | 3.59 | 3.86 | 4.36 [3.33, 5.46] | 0.973 |
| optimistic | touch + OFI-aware | queue | 1240.3 | 47% | 0.051 | 0.77 | 1.13 | 1.28 | 1.27 [1.02, 1.53] | 0.294 |
| optimistic | touch + OFI-aware | through | 1335.9 | 51% | 0.131 | 4.57 | 5.25 | 5.40 | 5.44 [4.99, 5.90] | 1.164 |
| optimistic | markout-aware A–S (quote wide) | all | 186.6 | 100% | 5.448 | 7.61 | 7.91 | 7.56 | 7.29 [6.83, 7.82] | 1.806 |
| optimistic | markout-aware A–S (quote wide) | cross | 7.5 | 4% | 2.965 | 4.14 | 4.25 | 3.32 | 3.38 [1.39, 5.37] | 0.784 |
| optimistic | markout-aware A–S (quote wide) | queue | 0.4 | 0% | 0.059 | -0.53 | -0.58 | 0.33 | 1.96 [-6.90, 11.19] | 0.584 |
| optimistic | markout-aware A–S (quote wide) | through | 178.7 | 96% | 5.564 | 7.77 | 8.08 | 7.76 | 7.47 [6.99, 8.04] | 1.851 |

## By distance from the mid at the fill (all five strategies pooled, proportional rule)

| distance (USDT) | BTC | 1 s | 5 s | 30 s | 60 s [CI] | 60 s (bp) |
|---|---|---|---|---|---|---|
| touch (≤0.05) | 10663.5 | 2.96 | 3.54 | 3.68 | 3.64 [3.50, 3.77] | 0.814 |
| 0.1–0.2 | 26.7 | 4.58 | 4.91 | 8.78 | 9.64 [6.87, 12.63] | 2.039 |
| 0.2–1 | 79.9 | 4.08 | 4.85 | 5.71 | 6.19 [4.68, 7.71] | 1.263 |
| 1–2 | 68.1 | 5.35 | 5.77 | 5.80 | 6.05 [4.37, 7.62] | 1.401 |
| 2–3 | 51.5 | 6.14 | 6.60 | 4.50 | 5.85 [3.77, 8.04] | 1.176 |
| 3–5 | 74.2 | 6.97 | 8.15 | 8.31 | 8.90 [6.71, 11.04] | 1.827 |
| ≥5 | 133.3 | 10.00 | 10.24 | 9.09 | 8.97 [7.99, 9.98] | 2.019 |

For comparison, stage 3 measured the market-wide 60 s markout of *all* test-day trades: 0.22 bp at the touch and 0.98–1.16 bp beyond it (results/stage3/markouts.md).
