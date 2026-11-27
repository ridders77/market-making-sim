# Stage 5 Part B — WHAT-IF: queue priority for a market maker

**What-if, not achievable for a retail participant.** The only change from stage 4: on joining the touch, the queue ahead of our order is f × the displayed size (f = 1: back of the queue, stage 4; f = 0: front). Test days, 230 paired hours, proportional cancel rule, 100 ms latency, 0.01 BTC lot, ±10-lot limit, stage 4's thresholds. Mean P&L per hour (USDT) [95% bootstrap CI over hours]. f = 1 reproduces stage 4 exactly (checked).

| strategy | f | −0.5 bp (rebate) | 0 bp | +2 bp (retail) | BTC / h | queue / sweep share of volume | 60 s markout: queue / sweep fills (USDT/BTC) |
|---|---|---|---|---|---|---|---|
| always at touch | 1 | -27.85 [-33.19, -23.14] | -74.44 [-85.26, -64.65] | -260.77 [-295.48, -229.94] | 20.9 | 39% / 61% | 1.46 / 4.94 |
| always at touch | 0.5 | -22.70 [-27.49, -18.49] | -78.00 [-89.06, -67.84] | -299.22 [-338.73, -264.33] | 24.8 | 50% / 50% | 1.40 / 4.99 |
| always at touch | 0.25 | -13.68 [-17.64, -10.27] | -79.88 [-91.16, -69.84] | -344.65 [-389.28, -305.10] | 29.7 | 60% / 40% | 1.21 / 4.96 |
| always at touch | 0 | 155.43 [144.36, 167.59] | -58.41 [-66.89, -50.66] | -913.77 [-1001.55, -835.96] | 98.2 | 88% / 12% | 0.06 / 4.85 |
| touch + inventory skew | 1 | -17.70 [-21.32, -14.51] | -45.62 [-53.39, -38.53] | -143.20 [-163.02, -125.17] | 12.3 | 37% / 63% | 1.46 / 5.20 |
| touch + inventory skew | 0.5 | -14.33 [-17.55, -11.48] | -48.23 [-56.31, -40.89] | -161.90 [-184.19, -141.82] | 15.0 | 49% / 51% | 1.35 / 5.23 |
| touch + inventory skew | 0.25 | -7.67 [-10.05, -5.56] | -50.13 [-58.50, -42.60] | -186.06 [-211.16, -163.55] | 18.8 | 60% / 40% | 1.10 / 5.20 |
| touch + inventory skew | 0 | 142.87 [132.89, 153.81] | -38.48 [-45.01, -32.67] | -589.19 [-640.72, -542.92] | 83.2 | 88% / 12% | -0.05 / 4.91 |

Volume shares and markouts at 0 bp; for touch + skew they are for its 0 bp configuration (the +2 bp configuration uses a different skew γ). Sweep = filled by a trade through our price or a book crossing.

## Break-even queue position

f at which the mean P&L crosses zero (linear interpolation between the grid points; 95% CI from a paired bootstrap over hours, shown when at least half of the resamples cross). 'none' = loses at every f, including the front.

| strategy | −0.5 bp | 0 bp | +2 bp |
|---|---|---|---|
| always at touch | 0.23 [0.23, 0.23] | none | none |
| touch + inventory skew | 0.24 [0.23, 0.24] | none | none |
