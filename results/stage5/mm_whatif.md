# Stage 5 Part B — WHAT-IF: tabular Q-learning market maker at f = 0.1, maker fee -0.5 bp

**What-if, not achievable for a retail participant** (front-of-queue priority). Test days, 230 paired hours, proportional rule, 100 ms latency, 0.01 BTC lot, ±10 lots (masked). Inventory penalty γ = 1.68e-05 per lot (holding 10 lots for an hour costs 0.123 USDT = the 60 s markout of 10 lots of always-at-touch fills at this f on the calibration days, 1.23 USDT/BTC). P&L per hour (USDT) excludes the penalty; mean [95% CI over hours].

| strategy | mean P&L / h | BTC / h | mean max |q| (lots) |
|---|---|---|---|
| RL market maker (mean of 5 seeds) | -11.95 [-13.80, -10.27] | 6.2 | 9.6 |
| always at touch | 2.87 [0.08, 5.52] | 38.1 | 10.0 |
| touch + inventory skew | 6.83 [5.61, 8.00] | 25.9 | 2.9 |

Paired differences (RL − baseline, USDT/h):

- vs touch: -14.82 [-16.92, -12.76]
- vs touch_skew: -18.78 [-20.45, -17.24]

Per-seed mean P&L / h: -9.4, -15.1, -15.1, -12.3, -7.9

Greedy action (bid level / ask level) by inventory bucket × OFI bucket, seed 1 … 5:

| inventory | sell flow (z < −c) | neutral | buy flow (z > c) |
|---|---|---|---|
| ≤ −6 | 1 behind/off; touch/off; 1 behind/off; off/off; 1 behind/off | 3 behind/off; off/off; touch/off; off/off; 3 behind/off | 3 behind/off; off/off; off/off; 3 behind/off; touch/off |
| −5…−2 | touch/1 behind; 1 behind/3 behind; touch/off; 1 behind/1 behind; 1 behind/off | touch/3 behind; off/1 behind; 1 behind/off; touch/3 behind; touch/touch | 1 behind/3 behind; 1 behind/1 behind; touch/3 behind; off/touch; 3 behind/off |
| −1…1 | off/off; off/touch; touch/touch; touch/1 behind; off/off | 3 behind/3 behind; 1 behind/3 behind; 3 behind/1 behind; touch/touch; touch/off | 1 behind/off; touch/1 behind; 3 behind/1 behind; 1 behind/1 behind; touch/off |
| 2…5 | off/touch; off/touch; touch/touch; touch/off; 3 behind/touch | 3 behind/3 behind; off/1 behind; 3 behind/3 behind; 1 behind/touch; 1 behind/touch | 1 behind/touch; 1 behind/off; touch/off; 3 behind/3 behind; touch/3 behind |
| ≥ 6 | off/off; off/touch; off/off; off/3 behind; off/touch | off/touch; off/touch; off/1 behind; off/1 behind; off/off | off/3 behind; off/1 behind; off/touch; off/1 behind; off/touch |
