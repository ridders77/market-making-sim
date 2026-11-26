# Lot size sensitivity (test days, proportional rule, 100 ms, fee-0 configurations)

Mean P&L per hour (USDT) and per BTC traded (bp of notional), 0.01 vs 0.1 BTC lot. The replay assumes our order does not change anyone else's behaviour; that assumption is weaker for 0.1 BTC.

| strategy | lot | fee | mean P&L (USDT/h) | P&L (bp of notional) | BTC filled/h |
|---|---|---|---|---|---|
| always at touch | 0.01 | -0.5 | -27.85 [-32.73, -23.03] | -0.299 | 20.862 |
| always at touch | 0.01 | +0 | -74.44 [-84.58, -64.70] | -0.799 | 20.862 |
| always at touch | 0.01 | +2 | -260.77 [-293.83, -230.02] | -2.799 | 20.862 |
| always at touch | 0.1 | -0.5 | -278.56 [-327.19, -231.08] | -0.316 | 197.191 |
| always at touch | 0.1 | +0 | -718.96 [-818.77, -624.23] | -0.816 | 197.191 |
| always at touch | 0.1 | +2 | -2480.54 [-2797.28, -2183.75] | -2.816 | 197.191 |
| touch + inventory skew | 0.01 | -0.5 | -17.70 [-21.11, -14.53] | -0.317 | 12.278 |
| touch + inventory skew | 0.01 | +0 | -45.62 [-53.24, -38.63] | -0.817 | 12.278 |
| touch + inventory skew | 0.01 | +2 | -157.29 [-181.63, -135.08] | -2.817 | 12.278 |
| touch + inventory skew | 0.1 | -0.5 | -185.44 [-220.53, -153.49] | -0.331 | 123.978 |
| touch + inventory skew | 0.1 | +0 | -465.62 [-540.93, -396.88] | -0.831 | 123.978 |
| touch + inventory skew | 0.1 | +2 | -1586.37 [-1820.57, -1368.91] | -2.831 | 123.978 |
| touch + queue-aware | 0.01 | -0.5 | -8.76 [-11.47, -6.11] | -0.363 | 5.746 |
| touch + queue-aware | 0.01 | +0 | -20.81 [-23.86, -17.87] | -0.863 | 5.746 |
| touch + queue-aware | 0.01 | +2 | -69.02 [-74.50, -63.66] | -2.863 | 5.746 |
| touch + queue-aware | 0.1 | -0.5 | -87.44 [-114.63, -60.51] | -0.384 | 54.230 |
| touch + queue-aware | 0.1 | +0 | -201.32 [-231.82, -171.61] | -0.884 | 54.230 |
| touch + queue-aware | 0.1 | +2 | -656.85 [-711.25, -603.58] | -2.884 | 54.230 |
| touch + OFI-aware | 0.01 | -0.5 | -18.16 [-22.12, -14.48] | -0.428 | 9.340 |
| touch + OFI-aware | 0.01 | +0 | -39.39 [-46.21, -33.01] | -0.928 | 9.340 |
| touch + OFI-aware | 0.01 | +2 | -124.30 [-143.41, -106.05] | -2.928 | 9.340 |
| touch + OFI-aware | 0.1 | -0.5 | -181.73 [-220.53, -144.55] | -0.449 | 89.023 |
| touch + OFI-aware | 0.1 | +0 | -384.02 [-450.71, -320.09] | -0.949 | 89.023 |
| touch + OFI-aware | 0.1 | +2 | -1193.17 [-1376.81, -1016.26] | -2.949 | 89.023 |
| markout-aware A–S (quote wide) | 0.01 | +0 | -1.34 [-1.76, -0.89] | -0.410 | 0.821 |
| markout-aware A–S (quote wide) | 0.1 | +0 | -13.48 [-17.79, -8.97] | -0.413 | 8.212 |
