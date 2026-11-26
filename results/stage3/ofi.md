# Step 2: order-flow imbalance (OFI) as a predictor of the next mid move

x = signed aggressor volume over the last N s (raw, BTC) or divided by total volume (imbalance). y = next h s mid move (bps). OLS fitted on the 10 calibration days; on the 10 test days the same line is applied (R² vs predicting no move). Hit rate = share of seconds where the predicted and actual moves have the same sign (non-zero moves). 95% CIs: bootstrap over days.

| N (s) | signal | h (s) | slope (bp per sd of x) | R² calibration | R² test (out of sample) | hit rate cal | hit rate test | slope refitted on test |
|---|---|---|---|---|---|---|---|---|
| 5 | raw | 5 | +0.0681 | 0.185% [0.000, 1.187] | -0.043% [-0.192, 0.147] | 0.565 | 0.544 [0.531, 0.557] | +0.0071 |
| 5 | raw | 30 | +0.0681 | 0.028% [0.000, 0.264] | -0.024% [-0.094, 0.006] | 0.499 | 0.506 [0.501, 0.512] | -0.0602 |
| 5 | imb | 5 | +0.0958 | 0.365% [0.274, 0.578] | 0.255% [0.129, 0.530] | 0.585 | 0.562 [0.543, 0.583] | +0.0894 |
| 5 | imb | 30 | +0.1236 | 0.091% [0.037, 0.212] | 0.058% [0.022, 0.123] | 0.527 | 0.518 [0.512, 0.524] | +0.1078 |
| 30 | raw | 5 | +0.0210 | 0.018% [0.000, 0.160] | -0.014% [-0.048, 0.001] | 0.517 | 0.516 [0.512, 0.521] | -0.0210 |
| 30 | raw | 30 | -0.1987 | 0.236% [0.024, 0.404] | 0.025% [-0.053, 0.152] | 0.494 | 0.498 [0.494, 0.503] | -0.1374 |
| 30 | imb | 5 | +0.0393 | 0.061% [0.025, 0.128] | 0.016% [0.001, 0.037] | 0.546 | 0.532 [0.522, 0.543] | +0.0265 |
| 30 | imb | 30 | +0.0285 | 0.005% [0.000, 0.042] | 0.007% [-0.006, 0.022] | 0.514 | 0.513 [0.506, 0.519] | +0.0396 |
| 60 | raw | 5 | -0.0150 | 0.009% [0.000, 0.091] | 0.009% [0.003, 0.014] | 0.486 | 0.497 [0.494, 0.500] | -0.0362 |
| 60 | raw | 30 | -0.1506 | 0.136% [0.012, 0.546] | 0.070% [-0.030, 0.139] | 0.498 | 0.498 [0.493, 0.505] | -0.2162 |
| 60 | imb | 5 | +0.0192 | 0.015% [0.002, 0.040] | 0.011% [0.001, 0.026] | 0.529 | 0.525 [0.517, 0.533] | +0.0193 |
| 60 | imb | 30 | +0.0013 | 0.000% [0.000, 0.015] | 0.001% [-0.013, 0.009] | 0.496 | 0.509 [0.502, 0.516] | +0.0391 |

**Main signal** (highest calibration R² averaged over h = 5 and 30 s): N = 5 s, imb; z = signal / 0.6244 (its calibration std).

## Drift decay: predicted move per unit z at horizon h (bps)

| h (s) | 1 | 2 | 3 | 5 | 10 | 20 | 30 | 60 | 120 |
|---|---|---|---|---|---|---|---|---|---|
| calibration | 0.0399 | 0.0619 | 0.0765 | 0.0958 | 0.1213 | 0.1297 | 0.1236 | 0.1043 | 0.0846 |
| fitted κτ(1−e^(−h/τ)) | 0.0373 | 0.0623 | 0.0789 | 0.0975 | 0.1105 | 0.1125 | 0.1125 | 0.1125 | 0.1125 |
| test | 0.0418 | 0.0618 | 0.0737 | 0.0894 | 0.1087 | 0.0996 | 0.1078 | 0.1286 | 0.1034 |

κ = 0.0453 bp/s per unit z, τ = 2.48 s (fitted on calibration days).

## Fill tilt (0.01 BTC lot rule, calibration days)

| δ (USDT) | θ | R² over z deciles |
|---|---|---|
| 0.1 | 0.383 | 0.960 |
| 0.5 | 0.335 | 0.955 |
| 1.0 | 0.292 | 0.948 |
| 2.0 | 0.257 | 0.937 |
| 3.0 | 0.255 | 0.927 |

θ used in the simulator: 0.257 (δ = 2.0 USDT, about the A–S half-spread). A 1-sd buying imbalance makes our ask 1.67× as likely to fill as our bid.
