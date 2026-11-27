# Stage 5 Part B — WHAT-IF: queue priority for a market maker

**What-if, not achievable for a retail participant.** The only change from stage 4: on joining the touch, the queue ahead of our order is f × the displayed size (f = 1: back of the queue, stage 4; f = 0: front). Test days, 230 paired hours, proportional cancel rule, 100 ms latency, 0.01 BTC lot, ±10-lot limit, stage 4's thresholds. Mean P&L per hour (USDT) [95% bootstrap CI over hours]. f = 1 reproduces stage 4 exactly (checked).

| strategy | f | −0.5 bp (rebate) | 0 bp | +2 bp (retail) | BTC / h | queue / sweep share of volume | 60 s markout: queue / sweep fills (USDT/BTC) |
|---|---|---|---|---|---|---|---|
| always at touch | 1 | -27.85 [-33.19, -23.14] | -74.44 [-85.26, -64.65] | -260.77 [-295.48, -229.94] | 20.9 | 39% / 61% | 1.46 / 4.94 |
| always at touch | 0.5 | -22.70 [-27.49, -18.49] | -78.00 [-89.06, -67.84] | -299.22 [-338.73, -264.33] | 24.8 | 50% / 50% | 1.40 / 4.99 |
| always at touch | 0.25 | -13.68 [-17.64, -10.27] | -79.88 [-91.16, -69.84] | -344.65 [-389.28, -305.10] | 29.7 | 60% / 40% | 1.21 / 4.96 |
| always at touch | 0.15 | -5.22 [-8.36, -2.51] | -81.28 [-92.64, -71.11] | -385.51 [-434.74, -341.95] | 34.1 | 65% / 35% | 1.08 / 4.94 |
| always at touch | 0.1 | 2.87 [0.20, 5.45] | -81.95 [-93.19, -71.67] | -421.20 [-474.72, -374.12] | 38.1 | 69% / 31% | 0.97 / 4.94 |
| always at touch | 0.05 | 20.27 [17.39, 23.24] | -81.40 [-92.48, -71.21] | -488.08 [-548.70, -434.66] | 45.7 | 74% / 26% | 0.77 / 4.90 |
| always at touch | 0.02 | 47.96 [43.30, 53.00] | -78.75 [-89.43, -69.07] | -585.56 [-655.69, -524.72] | 57.2 | 80% / 20% | 0.55 / 4.86 |
| always at touch | 0 | 155.43 [144.36, 167.59] | -58.41 [-66.89, -50.66] | -913.77 [-1001.55, -835.96] | 98.2 | 88% / 12% | 0.06 / 4.85 |
| touch + inventory skew | 1 | -17.70 [-21.32, -14.51] | -45.62 [-53.39, -38.53] | -143.20 [-163.02, -125.17] | 12.3 | 37% / 63% | 1.46 / 5.20 |
| touch + inventory skew | 0.5 | -14.33 [-17.55, -11.48] | -48.23 [-56.31, -40.89] | -161.90 [-184.19, -141.82] | 15.0 | 49% / 51% | 1.35 / 5.23 |
| touch + inventory skew | 0.25 | -7.67 [-10.05, -5.56] | -50.13 [-58.50, -42.60] | -186.06 [-211.16, -163.55] | 18.8 | 60% / 40% | 1.10 / 5.20 |
| touch + inventory skew | 0.15 | -0.36 [-2.05, 1.11] | -51.03 [-59.45, -43.40] | -209.21 [-236.93, -184.41] | 22.5 | 66% / 34% | 0.90 / 5.16 |
| touch + inventory skew | 0.1 | 6.83 [5.58, 8.01] | -51.45 [-59.97, -43.81] | -230.47 [-260.54, -203.83] | 25.9 | 70% / 30% | 0.76 / 5.13 |
| touch + inventory skew | 0.05 | 21.89 [20.06, 23.75] | -51.54 [-59.92, -43.97] | -273.06 [-307.15, -242.75] | 32.8 | 76% / 24% | 0.57 / 5.03 |
| touch + inventory skew | 0.02 | 46.64 [42.74, 50.79] | -50.28 [-58.40, -43.02] | -338.85 [-378.00, -303.43] | 43.5 | 81% / 19% | 0.34 / 5.02 |
| touch + inventory skew | 0 | 142.87 [132.89, 153.81] | -38.48 [-45.01, -32.67] | -589.19 [-640.72, -542.92] | 83.2 | 88% / 12% | -0.05 / 4.91 |

Volume shares and markouts at 0 bp; for touch + skew they are for its 0 bp configuration (the +2 bp configuration uses a different skew γ). Sweep = filled by a trade through our price or a book crossing.

## Break-even queue position

Reported as a bracket: the last f (scanning from the back of the queue, f = 1, to the front, f = 0) where the mean P&L is negative and the first where it is positive. Not interpolated: P&L is far from linear in f (most of the gain arrives in the last few % of the queue). (loss) / (profit) = the 95% CI at that f lies entirely below / above zero.

| strategy | −0.5 bp | 0 bp | +2 bp |
|---|---|---|---|
| always at touch | between f = 0.15 (loss) and f = 0.1 (profit) | none: loses at every f, including the front | none: loses at every f, including the front |
| touch + inventory skew | between f = 0.15 (unclear) and f = 0.1 (profit) | none: loses at every f, including the front | none: loses at every f, including the front |
