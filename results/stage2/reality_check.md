# Reality check: A–S optimal spread vs the real BTCUSDT spread

Real spread: median 0.1 USDT (1 tick) on every processed day. 1 fill = 1 BTC.

| regime | γ | 2/k | A–S spread at t=0 | A–S avg quoted spread | × real spread | reservation shift per BTC at t=0 | fills/h | fee per fill at 1 / 2 bp | fee-aware half-spread 1/k + fee (2 bp) |
|---|---|---|---|---|---|---|---|---|---|
| low | 1e-05 | 3.19 | 3.22 | 3.31 | 33× | 0.03 | 234 | 2.79 / 5.58 | 7.17 |
| low | 0.0001 | 3.19 | 3.49 | 3.44 | 34× | 0.30 | 229 | 2.79 / 5.58 | 7.17 |
| low | 0.001 | 3.19 | 6.17 | 4.78 | 48× | 2.98 | 186 | 2.79 / 5.58 | 7.17 |
| medium | 1e-05 | 4.02 | 4.11 | 4.16 | 42× | 0.09 | 401 | 2.74 / 5.48 | 7.49 |
| medium | 0.0001 | 4.02 | 4.96 | 4.59 | 46× | 0.94 | 381 | 2.74 / 5.48 | 7.49 |
| medium | 0.001 | 4.02 | 13.46 | 8.84 | 88× | 9.45 | 219 | 2.74 / 5.48 | 7.49 |
| high | 1e-05 | 5.03 | 5.25 | 5.24 | 52× | 0.23 | 571 | 2.83 / 5.66 | 8.17 |
| high | 0.0001 | 5.03 | 7.30 | 6.26 | 63× | 2.28 | 517 | 2.83 / 5.66 | 8.17 |
| high | 0.001 | 5.03 | 27.78 | 16.50 | 165× | 22.76 | 184 | 2.83 / 5.66 | 8.17 |
