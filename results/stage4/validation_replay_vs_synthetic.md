# Step 2: replay vs the stage 3 synthetic simulator (markout-aware A–S, test days)

Same strategy, same hours (10 test days × 23 hours), same inputs known at the start of each hour (trailing σ̂, mid, model (b) A and k at σ̂, stage 3 flow calibration). Synthetic = mean over 200 simulated paths per hour. Ratio = replay / synthetic (ratio of sums over hours), 95% bootstrap CI over hours. 60 s markout in USDT per BTC, fills in the first 59 minutes. *unseen depth = front* treats every print at our price behind the touch as filling us (upper bound); *back* (main) assumes we are behind everything until the level becomes best.

Synthetic dt: [np.float64(0.00625), np.float64(0.025), np.float64(0.05), np.float64(0.1), np.float64(0.2)] s (smaller where A·dt would reach 0.1); A per hour 0.044–10.147 /s, k 0.168–0.763 /USDT.

| fee | cancel rule | unseen depth | BTC filled/h replay | synthetic | ratio [CI] | markout 60 s replay | synthetic | ratio [CI] | P&L/h replay | synthetic | ratio [CI] | difference [CI] | replay fills by sweeps through |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| -0.5 | pessimistic | back | 2.8093 | 2.8758 | 0.98 [0.86, 1.10] | 6.53 | 4.17 | 1.57 [1.46, 1.68] | -2.371 | 8.167 | -0.29 [-0.40, -0.20] | -10.538 [-11.714, -9.438] | 92% |
| -0.5 | proportional | back | 2.8093 | 2.8758 | 0.98 [0.86, 1.10] | 6.53 | 4.17 | 1.57 [1.46, 1.68] | -2.371 | 8.167 | -0.29 [-0.40, -0.20] | -10.538 [-11.714, -9.438] | 92% |
| -0.5 | optimistic | back | 2.8093 | 2.8758 | 0.98 [0.86, 1.10] | 6.53 | 4.17 | 1.57 [1.46, 1.68] | -2.371 | 8.167 | -0.29 [-0.40, -0.20] | -10.538 [-11.714, -9.438] | 92% |
| +0 | pessimistic | back | 0.8210 | 0.7777 | 1.06 [0.91, 1.21] | 7.29 | 4.89 | 1.49 [1.39, 1.60] | -1.336 | 2.040 | -0.66 [-0.87, -0.45] | -3.376 [-3.975, -2.795] | 96% |
| +0 | pessimistic | front | 0.8458 | 0.7777 | 1.09 [0.94, 1.25] | 7.26 | 4.89 | 1.49 [1.39, 1.59] | -1.370 | 2.040 | -0.67 [-0.89, -0.46] | -3.409 [-4.001, -2.825] | 93% |
| +0 | proportional | back | 0.8210 | 0.7777 | 1.06 [0.91, 1.21] | 7.29 | 4.89 | 1.49 [1.39, 1.60] | -1.336 | 2.040 | -0.66 [-0.87, -0.45] | -3.376 [-3.975, -2.795] | 96% |
| +0 | proportional | front | 0.8458 | 0.7777 | 1.09 [0.94, 1.25] | 7.26 | 4.89 | 1.49 [1.39, 1.59] | -1.370 | 2.040 | -0.67 [-0.89, -0.46] | -3.409 [-4.001, -2.825] | 93% |
| +0 | optimistic | back | 0.8210 | 0.7777 | 1.06 [0.91, 1.21] | 7.29 | 4.89 | 1.49 [1.39, 1.60] | -1.336 | 2.040 | -0.66 [-0.87, -0.45] | -3.376 [-3.975, -2.795] | 96% |
| +0 | optimistic | front | 0.8458 | 0.7777 | 1.09 [0.94, 1.25] | 7.26 | 4.89 | 1.49 [1.39, 1.59] | -1.370 | 2.040 | -0.67 [-0.89, -0.46] | -3.409 [-4.001, -2.825] | 93% |
| +2 | pessimistic | back | 0.0248 | 0.0002 | 106.04 [62.02, 191.05] | 21.31 | 15.25 | 1.40 [0.97, 1.91] | -0.058 | 0.002 | -25.91 [-331.96, 177.36] | -0.060 [-0.441, 0.212] | 100% |
| +2 | proportional | back | 0.0248 | 0.0002 | 106.04 [62.02, 191.05] | 21.31 | 15.25 | 1.40 [0.97, 1.91] | -0.058 | 0.002 | -25.91 [-331.96, 177.36] | -0.060 [-0.441, 0.212] | 100% |
| +2 | optimistic | back | 0.0248 | 0.0002 | 106.04 [62.02, 191.05] | 21.31 | 15.25 | 1.40 [0.97, 1.91] | -0.058 | 0.002 | -25.91 [-331.96, 177.36] | -0.060 [-0.441, 0.212] | 100% |
