# Stage 4: P&L by strategy, fee and cancel rule (test days)

10 test days × 23 hour-long episodes = 230 episodes, the same hours for every strategy (paired). 0.01 BTC lot, 100 ms latency, inventory limit ±10 lots. Mean P&L per hour (USDT) [95% bootstrap CI over episodes]; the next column resamples whole days instead (hours within a day are correlated). Thresholds chosen on the calibration days, separately per fee (tuning.md).


## Maker fee -0.5 bp

| cancel rule | strategy | config | mean P&L (USDT/h) | CI resampling days | P&L (bp of notional) | std P&L | BTC filled/h | 60 s markout (USDT/BTC) | spread captured (USDT/BTC) | share filled by sweeps through our price | time quoted |
|---|---|---|---|---|---|---|---|---|---|---|---|
| pessimistic | always at touch | touch | -30.25 [-35.27, -25.34] | [-51.21, -16.42] | -0.363 | 38.23 | 18.584 | 3.86 | 0.110 | 66% | 84% |
| pessimistic | touch + inventory skew | touch_skew_gamma0_max_lots1 | -18.95 [-22.38, -15.76] | [-33.88, -9.47] | -0.397 | 26.09 | 10.422 | 4.23 | 0.138 | 70% | 59% |
| pessimistic | touch + queue-aware | touch_queue_theta0.05 | -8.72 [-11.43, -6.13] | [-14.08, -4.65] | -0.367 | 19.99 | 5.664 | 3.55 | 0.078 | 66% | 11% |
| pessimistic | touch + OFI-aware | touch_ofi_z_star0.25 | -18.85 [-22.88, -14.93] | [-34.93, -8.61] | -0.491 | 31.35 | 8.405 | 4.44 | 0.111 | 71% | 50% |
| pessimistic | markout-aware A–S (quote wide) | markout_as_fee-0.5 | -2.37 [-3.21, -1.59] | [-4.52, -0.84] | -0.192 | 6.39 | 2.809 | 6.53 | 3.521 | 92% | 100% |
| proportional | always at touch | touch | -27.85 [-32.73, -23.03] | [-48.47, -14.24] | -0.299 | 37.98 | 20.862 | 3.56 | 0.104 | 58% | 85% |
| proportional | touch + inventory skew | touch_skew_gamma0_max_lots1 | -17.70 [-21.11, -14.53] | [-32.53, -8.28] | -0.317 | 26.10 | 12.278 | 3.83 | 0.126 | 61% | 62% |
| proportional | touch + queue-aware | touch_queue_theta0.05 | -8.76 [-11.47, -6.11] | [-14.17, -4.66] | -0.363 | 20.08 | 5.746 | 3.51 | 0.078 | 64% | 10% |
| proportional | touch + OFI-aware | touch_ofi_z_star0.25 | -18.16 [-22.12, -14.48] | [-33.61, -8.35] | -0.428 | 30.28 | 9.340 | 4.14 | 0.104 | 64% | 50% |
| proportional | markout-aware A–S (quote wide) | markout_as_fee-0.5 | -2.37 [-3.21, -1.59] | [-4.52, -0.84] | -0.192 | 6.39 | 2.809 | 6.53 | 3.521 | 92% | 100% |
| optimistic | always at touch | touch | -19.33 [-23.97, -14.82] | [-38.24, -6.81] | -0.169 | 35.74 | 25.899 | 2.97 | 0.093 | 47% | 86% |
| optimistic | touch + inventory skew | touch_skew_gamma0_max_lots1 | -11.26 [-14.51, -8.24] | [-24.86, -2.47] | -0.158 | 24.34 | 15.913 | 3.03 | 0.108 | 47% | 65% |
| optimistic | touch + queue-aware | touch_queue_theta0.05 | -8.57 [-11.29, -5.97] | [-13.83, -4.59] | -0.350 | 20.30 | 5.836 | 3.48 | 0.078 | 63% | 10% |
| optimistic | touch + OFI-aware | touch_ofi_z_star0.25 | -14.24 [-17.93, -10.73] | [-28.49, -5.00] | -0.272 | 28.89 | 11.641 | 3.46 | 0.093 | 51% | 49% |
| optimistic | markout-aware A–S (quote wide) | markout_as_fee-0.5 | -2.37 [-3.21, -1.59] | [-4.52, -0.84] | -0.192 | 6.39 | 2.809 | 6.53 | 3.521 | 92% | 100% |

## Maker fee +0 bp

| cancel rule | strategy | config | mean P&L (USDT/h) | CI resampling days | P&L (bp of notional) | std P&L | BTC filled/h | 60 s markout (USDT/BTC) | spread captured (USDT/BTC) | share filled by sweeps through our price | time quoted |
|---|---|---|---|---|---|---|---|---|---|---|---|
| pessimistic | always at touch | touch | -71.94 [-82.02, -62.27] | [-117.18, -42.42] | -0.863 | 77.17 | 18.584 | 3.86 | 0.110 | 66% | 84% |
| pessimistic | touch + inventory skew | touch_skew_gamma0_max_lots1 | -42.83 [-50.42, -36.06] | [-74.62, -22.68] | -0.897 | 55.64 | 10.422 | 4.23 | 0.138 | 70% | 59% |
| pessimistic | touch + queue-aware | touch_queue_theta0.05 | -20.60 [-23.63, -17.65] | [-29.33, -14.06] | -0.867 | 22.74 | 5.664 | 3.55 | 0.078 | 66% | 11% |
| pessimistic | touch + OFI-aware | touch_ofi_z_star0.25 | -38.05 [-44.96, -31.51] | [-67.32, -19.38] | -0.991 | 52.63 | 8.405 | 4.44 | 0.111 | 71% | 50% |
| pessimistic | markout-aware A–S (quote wide) | markout_as_fee0 | -1.34 [-1.76, -0.89] | [-2.27, -0.40] | -0.410 | 3.43 | 0.821 | 7.29 | 5.460 | 96% | 100% |
| proportional | always at touch | touch | -74.44 [-84.58, -64.70] | [-120.48, -44.48] | -0.799 | 78.21 | 20.862 | 3.56 | 0.104 | 58% | 85% |
| proportional | touch + inventory skew | touch_skew_gamma0_max_lots1 | -45.62 [-53.24, -38.63] | [-78.52, -24.65] | -0.817 | 57.27 | 12.278 | 3.83 | 0.126 | 61% | 62% |
| proportional | touch + queue-aware | touch_queue_theta0.05 | -20.81 [-23.86, -17.87] | [-29.56, -14.22] | -0.863 | 22.82 | 5.746 | 3.51 | 0.078 | 64% | 10% |
| proportional | touch + OFI-aware | touch_ofi_z_star0.25 | -39.39 [-46.21, -33.01] | [-68.52, -20.69] | -0.928 | 52.19 | 9.340 | 4.14 | 0.104 | 64% | 50% |
| proportional | markout-aware A–S (quote wide) | markout_as_fee0 | -1.34 [-1.76, -0.89] | [-2.27, -0.40] | -0.410 | 3.43 | 0.821 | 7.29 | 5.460 | 96% | 100% |
| optimistic | always at touch | touch | -76.62 [-86.77, -66.97] | [-123.23, -46.40] | -0.669 | 78.53 | 25.899 | 2.97 | 0.093 | 47% | 86% |
| optimistic | touch + inventory skew | touch_skew_gamma0_max_lots1 | -46.79 [-54.26, -39.84] | [-79.38, -26.00] | -0.658 | 56.59 | 15.913 | 3.03 | 0.108 | 47% | 65% |
| optimistic | touch + queue-aware | touch_queue_theta0.05 | -20.81 [-23.85, -17.89] | [-29.44, -14.36] | -0.850 | 22.95 | 5.836 | 3.48 | 0.078 | 63% | 10% |
| optimistic | touch + OFI-aware | touch_ofi_z_star0.25 | -40.39 [-47.34, -34.04] | [-69.31, -21.57] | -0.772 | 51.83 | 11.641 | 3.46 | 0.093 | 51% | 49% |
| optimistic | markout-aware A–S (quote wide) | markout_as_fee0 | -1.34 [-1.76, -0.89] | [-2.27, -0.40] | -0.410 | 3.43 | 0.821 | 7.29 | 5.460 | 96% | 100% |

## Maker fee +2 bp

| cancel rule | strategy | config | mean P&L (USDT/h) | CI resampling days | P&L (bp of notional) | std P&L | BTC filled/h | 60 s markout (USDT/BTC) | spread captured (USDT/BTC) | share filled by sweeps through our price | time quoted |
|---|---|---|---|---|---|---|---|---|---|---|---|
| pessimistic | always at touch | touch | -238.67 [-270.32, -209.04] | [-381.48, -145.87] | -2.863 | 241.22 | 18.584 | 3.86 | 0.110 | 66% | 84% |
| pessimistic | touch + inventory skew | touch_skew_gamma0.0001_max_lots1 | -132.59 [-151.89, -114.59] | [-214.89, -78.34] | -2.909 | 144.29 | 10.144 | 4.46 | 0.384 | 75% | 67% |
| pessimistic | touch + queue-aware | touch_queue_theta0.05 | -68.14 [-73.58, -62.77] | [-90.86, -50.35] | -2.867 | 42.38 | 5.664 | 3.55 | 0.078 | 66% | 11% |
| pessimistic | touch + OFI-aware | touch_ofi_z_star0.25 | -114.84 [-133.66, -97.07] | [-196.58, -62.35] | -2.991 | 142.98 | 8.405 | 4.44 | 0.111 | 71% | 50% |
| pessimistic | markout-aware A–S (quote wide) | markout_as_fee2 | -0.06 [-0.43, 0.22] | [-0.40, 0.18] | -0.488 | 2.56 | 0.025 | 21.31 | 25.825 | 100% | 100% |
| proportional | always at touch | touch | -260.77 [-293.83, -230.02] | [-408.80, -165.67] | -2.799 | 248.32 | 20.862 | 3.56 | 0.104 | 58% | 85% |
| proportional | touch + inventory skew | touch_skew_gamma0.0001_max_lots1 | -143.20 [-162.78, -124.95] | [-227.10, -88.38] | -2.853 | 146.46 | 11.241 | 4.18 | 0.366 | 68% | 70% |
| proportional | touch + queue-aware | touch_queue_theta0.05 | -69.02 [-74.50, -63.66] | [-91.70, -51.35] | -2.863 | 42.46 | 5.746 | 3.51 | 0.078 | 64% | 10% |
| proportional | touch + OFI-aware | touch_ofi_z_star0.25 | -124.30 [-143.41, -106.05] | [-208.35, -70.22] | -2.928 | 145.78 | 9.340 | 4.14 | 0.104 | 64% | 50% |
| proportional | markout-aware A–S (quote wide) | markout_as_fee2 | -0.06 [-0.43, 0.22] | [-0.40, 0.18] | -0.488 | 2.56 | 0.025 | 21.31 | 25.825 | 100% | 100% |
| optimistic | always at touch | touch | -305.79 [-340.16, -273.68] | [-461.79, -204.48] | -2.669 | 260.22 | 25.899 | 2.97 | 0.093 | 47% | 86% |
| optimistic | touch + inventory skew | touch_skew_gamma0.0001_max_lots1 | -166.93 [-186.94, -148.31] | [-254.03, -109.40] | -2.719 | 150.62 | 13.895 | 3.50 | 0.336 | 57% | 75% |
| optimistic | touch + queue-aware | touch_queue_theta0.05 | -69.77 [-75.31, -64.35] | [-92.30, -52.21] | -2.850 | 42.49 | 5.836 | 3.48 | 0.078 | 63% | 10% |
| optimistic | touch + OFI-aware | touch_ofi_z_star0.25 | -144.95 [-164.69, -125.96] | [-232.67, -87.84] | -2.772 | 151.05 | 11.641 | 3.46 | 0.093 | 51% | 49% |
| optimistic | markout-aware A–S (quote wide) | markout_as_fee2 | -0.06 [-0.43, 0.22] | [-0.40, 0.18] | -0.488 | 2.56 | 0.025 | 21.31 | 25.825 | 100% | 100% |

## Paired differences (same 230 hours), mean P&L per hour [95% CI]

| fee | cancel rule | queue-aware − touch | OFI-aware − touch | skew − touch | OFI-aware − queue-aware | best touch variant − markout-aware A–S |
|---|---|---|---|---|---|---|
| -0.5 | pessimistic | 21.53 [17.78, 25.42] | 11.40 [8.89, 13.98] | 11.31 [9.07, 13.61] | -10.13 [-13.76, -6.81] | touch + queue-aware: -6.35 [-8.65, -4.09] |
| -0.5 | proportional | 19.10 [15.45, 22.86] | 9.69 [7.18, 12.31] | 10.15 [7.91, 12.42] | -9.41 [-13.02, -6.18] | touch + queue-aware: -6.38 [-8.70, -4.09] |
| -0.5 | optimistic | 10.76 [7.21, 14.51] | 5.09 [2.36, 7.88] | 8.07 [5.84, 10.33] | -5.67 [-9.31, -2.32] | touch + queue-aware: -6.20 [-8.53, -3.94] |
| +0 | pessimistic | 51.33 [43.52, 59.99] | 33.89 [29.96, 38.17] | 29.11 [25.99, 32.59] | -17.44 [-23.22, -12.39] | touch + queue-aware: -19.27 [-22.35, -16.40] |
| +0 | proportional | 53.63 [45.61, 62.32] | 35.05 [31.02, 39.36] | 28.82 [25.70, 32.16] | -18.58 [-24.26, -13.48] | touch + queue-aware: -19.47 [-22.58, -16.57] |
| +0 | optimistic | 55.81 [47.61, 64.68] | 36.24 [31.89, 40.74] | 29.83 [26.59, 33.20] | -19.57 [-25.53, -14.46] | touch + queue-aware: -19.48 [-22.67, -16.49] |
| +2 | pessimistic | 170.53 [144.83, 199.37] | 123.83 [111.03, 137.82] | 106.08 [93.60, 119.77] | -46.70 [-61.98, -33.17] | touch + queue-aware: -68.09 [-73.85, -62.71] |
| +2 | proportional | 191.75 [165.10, 221.70] | 136.48 [122.96, 151.31] | 117.57 [104.43, 132.07] | -55.28 [-70.95, -41.55] | touch + queue-aware: -68.96 [-74.73, -63.63] |
| +2 | optimistic | 236.02 [207.96, 267.96] | 160.84 [146.44, 176.53] | 138.86 [124.62, 154.18] | -75.18 [-91.77, -60.46] | touch + queue-aware: -69.72 [-75.41, -64.42] |
