# Stage 5 Part A: passive vs aggressive fills and markouts of passive fills (test days, H = 1 min)

Share of the 0.5 BTC filled passively (our resting order) vs as a taker (cross, TWAP slice or deadline clean-up); passive volume split by how it filled (queue = traded volume reached us in the queue; sweep = a trade through our price; cross = the opposite quote moved onto our price). Markout of passive fills = −side·(m_{t+h} − m_pre)/m_pre in bps, quantity-weighted; positive = adverse (the price kept moving in our favour after we filled, so waiting would have been cheaper). Proportional rule. RL = all 5 seeds pooled.

| side | fee | strategy | passive | taker | passive via queue / sweep / cross | markout 10 s | markout 60 s [CI] |
|---|---|---|---|---|---|---|---|
| buy | retail | rest at touch, then cross | 97.8% | 2.2% | 36% / 62% / 2% | 0.86 | 0.92 [0.78, 1.06] |
| buy | retail | fixed: 1 tick behind | 92.7% | 7.3% | 2% / 93% / 5% | 1.18 | 1.23 [1.07, 1.38] |
| buy | retail | fixed: 3 ticks behind | 92.4% | 7.6% | 1% / 94% / 5% | 1.20 | 1.26 [1.10, 1.40] |
| buy | retail | RL: sigma | 97.0% | 3.0% | 32% / 66% / 2% | 0.90 | 0.96 [0.89, 1.02] |
| buy | pro | rest at touch, then cross | 97.8% | 2.2% | 36% / 62% / 2% | 0.86 | 0.92 [0.78, 1.06] |
| buy | pro | fixed: 1 tick behind | 92.7% | 7.3% | 2% / 93% / 5% | 1.18 | 1.23 [1.07, 1.38] |
| buy | pro | fixed: 3 ticks behind | 92.4% | 7.6% | 1% / 94% / 5% | 1.20 | 1.26 [1.10, 1.40] |
| buy | pro | RL: sigma | 97.4% | 2.6% | 34% / 64% / 2% | 0.89 | 0.95 [0.88, 1.01] |
| sell | retail | rest at touch, then cross | 98.0% | 2.0% | 36% / 62% / 2% | 0.92 | 0.94 [0.79, 1.09] |
| sell | retail | fixed: 1 tick behind | 93.3% | 6.7% | 2% / 93% / 5% | 1.22 | 1.25 [1.10, 1.41] |
| sell | retail | fixed: 3 ticks behind | 92.9% | 7.1% | 1% / 93% / 5% | 1.24 | 1.26 [1.10, 1.42] |
| sell | retail | RL: sigma | 97.2% | 2.8% | 27% / 70% / 3% | 1.01 | 1.04 [0.97, 1.10] |
| sell | pro | rest at touch, then cross | 98.0% | 2.0% | 36% / 62% / 2% | 0.92 | 0.94 [0.79, 1.09] |
| sell | pro | fixed: 1 tick behind | 93.3% | 6.7% | 2% / 93% / 5% | 1.22 | 1.25 [1.10, 1.41] |
| sell | pro | fixed: 3 ticks behind | 92.9% | 7.1% | 1% / 93% / 5% | 1.24 | 1.26 [1.10, 1.42] |
| sell | pro | RL: sigma | 97.4% | 2.6% | 31% / 66% / 3% | 0.96 | 0.99 [0.93, 1.05] |
