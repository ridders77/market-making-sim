# Step 1: pre-registered choice of the fill-intensity model

Rule fixed in DECISIONS.md before these scores were computed: choose (a) if its score ≤ 0.4, else (b) if ≤ 0.4, else the paper parameters. Score = mean over the 10 test days of the mean |ln(λ_pred/λ_obs)| over δ = 0.1…5.0 USDT (0.01 BTC lot rule, windows from 00:30 UTC). 95% CI: bootstrap over test days.

| model | score [95% CI] | worst day | mean bias ln(pred/obs) | passes (≤ 0.4)? |
|---|---|---|---|---|
| (a) relative units, fixed A_rel, k_rel | 0.504 [0.272, 0.761] | 1.293 | -0.126 | no |
| (b) σ-conditional, trailing σ̂ | 0.322 [0.203, 0.452] | 0.782 | +0.311 | yes |
| reference: fixed A, k in USDT (not part of the rule) | 0.683 [0.419, 1.037] | 1.967 | -0.491 | no |

**Chosen: (b) σ-conditional, trailing σ̂.**

## Fitted parameters (calibration days)

- (a) A_rel = 0.1515 /s, k_rel = 1.288 /bp (R² 0.9991, 178 grid points). At S = 27,800 this is k = 0.463 /USDT.
- (b) A = 0.0827·σ̂^1.355 (R² 0.996), k = 0.6392·σ̂^-0.377 (R² 0.939); σ̂ deciles span 0.52–3.71 USDT/√s.
- reference: A = 0.1508 /s, k = 0.459 /USDT.

## Per test day

| day | (a) score | (a) bias | (b) score | (b) bias | USDT-fixed score | δ points dropped (a/b) |
|---|---|---|---|---|---|---|
| 2023-10-25 | 0.963 | -0.963 | 0.064 | +0.048 | 1.183 | 0/0 |
| 2023-10-27 | 0.238 | -0.238 | 0.146 | +0.146 | 0.448 | 0/0 |
| 2023-11-06 | 0.126 | +0.126 | 0.079 | -0.014 | 0.113 | 0/0 |
| 2023-11-12 | 0.888 | +0.888 | 0.285 | +0.285 | 0.597 | 0/0 |
| 2023-12-15 | 0.199 | -0.199 | 0.286 | +0.286 | 0.602 | 0/0 |
| 2023-12-17 | 0.059 | -0.026 | 0.308 | +0.308 | 0.420 | 0/0 |
| 2024-01-13 | 0.417 | -0.417 | 0.472 | +0.472 | 0.827 | 0/0 |
| 2024-01-28 | 0.088 | +0.088 | 0.358 | +0.358 | 0.311 | 0/0 |
| 2024-02-04 | 0.773 | +0.773 | 0.444 | +0.444 | 0.363 | 0/0 |
| 2024-03-04 | 1.293 | -1.293 | 0.782 | +0.782 | 1.967 | 0/0 |

## Simulator parameters by regime (chosen model, regime calibration days)

| regime | S₀ | σ_60s (USDT/√s) | σ_60s (bp/√s) | σ used | A | k (/USDT) |
|---|---|---|---|---|---|---|
| low | 27882 | 1.107 | 0.397 | 1.107 | 0.0949 | 0.615 |
| medium | 27407 | 1.782 | 0.650 | 1.782 | 0.1808 | 0.514 |
| high | 28276 | 2.938 | 1.043 | 2.938 | 0.3560 | 0.426 |
