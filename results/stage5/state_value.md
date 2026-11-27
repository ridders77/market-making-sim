# Stage 5 Part A: value of each state variable (test days, H = 1 min, proportional rule)

Paired difference in mean shortfall (bps) of the agent with the extra market variable(s) minus the private-only (time × inventory) agent, both averaged over 5 seeds; negative = the variable reduces cost out of sample. [95% bootstrap CI over episodes]; 'seeds' = range of the same difference seed by seed. The last row compares the private agent with resting at the touch.

| added to time × inventory | buy, retail | sell, retail | buy, pro | sell, pro |
|---|---|---|---|---|
| ofi | 0.032 [0.024, 0.039]; seeds -0.067…+0.090 | -0.032 [-0.040, -0.024]; seeds -0.183…+0.095 | -0.001 [-0.006, 0.005]; seeds -0.147…+0.086 | 0.058 [0.050, 0.066]; seeds +0.000…+0.148 |
| queue | 0.069 [0.058, 0.080]; seeds +0.026…+0.113 | 0.015 [0.006, 0.023]; seeds -0.153…+0.145 | 0.022 [0.016, 0.028]; seeds -0.204…+0.164 | 0.077 [0.066, 0.088]; seeds +0.048…+0.105 |
| sigma | 0.015 [0.010, 0.021]; seeds -0.051…+0.082 | -0.006 [-0.011, -0.000]; seeds -0.188…+0.137 | -0.034 [-0.040, -0.028]; seeds -0.211…+0.061 | 0.040 [0.034, 0.047]; seeds +0.003…+0.146 |
| ofi+queue+sigma | 0.161 [0.146, 0.177]; seeds +0.067…+0.232 | 0.098 [0.086, 0.110]; seeds -0.027…+0.251 | 0.075 [0.064, 0.086]; seeds -0.102…+0.180 | 0.108 [0.096, 0.122]; seeds +0.067…+0.189 |
| private vs rest-at-touch | 0.029 [0.024, 0.034] | 0.091 [0.081, 0.102] | 0.058 [0.051, 0.066] | 0.006 [0.005, 0.008] |
