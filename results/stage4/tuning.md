# Stage 4: threshold tuning on the calibration days

Proportional cancel rule, 100 ms latency, 0.01 BTC lot, 10 calibration days × 23 hours = 230 episodes. Mean P&L per hour (USDT) [95% bootstrap CI over episodes]. The chosen threshold (bold) maximises the mean, per fee.

| strategy | config | BTC filled / h | P&L at −0.5 bp | P&L at 0 bp | P&L at +2 bp |
|---|---|---|---|---|---|
| always at touch | touch | 18.04 | **-9.94 [-11.77, -8.16]** | **-35.12 [-38.88, -31.36]** | **-135.83 [-148.63, -123.14]** |
| touch + inventory skew | touch_skew_gamma0_max_lots1 | 9.76 | **-6.07 [-7.10, -5.11]** | **-19.72 [-22.21, -17.30]** | -74.31 [-83.40, -65.79] |
| touch + inventory skew | touch_skew_gamma1e-05_max_lots1 | 9.88 | -6.54 [-7.60, -5.51] | -20.36 [-22.90, -17.86] | -75.64 [-84.64, -67.15] |
| touch + inventory skew | touch_skew_gamma0.0001_max_lots1 | 9.46 | -7.01 [-7.97, -6.03] | -20.23 [-22.52, -17.94] | **-73.13 [-81.17, -65.47]** |
| touch + inventory skew | touch_skew_gamma0_max_lots2 | 12.56 | -7.64 [-8.85, -6.50] | -25.19 [-28.10, -22.27] | -95.41 [-105.97, -85.37] |
| touch + inventory skew | touch_skew_gamma1e-05_max_lots2 | 12.15 | -8.50 [-9.76, -7.24] | -25.46 [-28.27, -22.62] | -93.32 [-103.30, -83.80] |
| touch + inventory skew | touch_skew_gamma0.0001_max_lots2 | 10.87 | -8.36 [-9.43, -7.30] | -23.54 [-25.96, -21.12] | -84.25 [-92.71, -75.87] |
| touch + inventory skew | touch_skew_gamma0_max_lots5 | 15.99 | -9.40 [-10.90, -7.92] | -31.73 [-35.06, -28.36] | -121.03 [-133.19, -109.10] |
| touch + inventory skew | touch_skew_gamma1e-05_max_lots5 | 14.63 | -10.84 [-12.38, -9.35] | -31.25 [-34.43, -28.09] | -112.92 [-123.61, -102.24] |
| touch + inventory skew | touch_skew_gamma0.0001_max_lots5 | 12.05 | -9.22 [-10.36, -8.08] | -26.03 [-28.50, -23.53] | -93.26 [-101.70, -84.76] |
| touch + inventory skew | touch_skew_gamma0_max_lots10 | 18.03 | -9.97 [-11.79, -8.19] | -35.14 [-38.90, -31.37] | -135.84 [-148.67, -123.13] |
| touch + inventory skew | touch_skew_gamma1e-05_max_lots10 | 15.71 | -11.72 [-13.54, -10.03] | -33.64 [-37.09, -30.24] | -121.31 [-132.26, -110.22] |
| touch + inventory skew | touch_skew_gamma0.0001_max_lots10 | 12.33 | -9.25 [-10.43, -8.11] | -26.45 [-28.91, -23.94] | -95.24 [-103.58, -86.78] |
| touch + queue-aware | touch_queue_theta0.05 | 4.53 | **-3.39 [-4.45, -2.37]** | **-9.69 [-11.04, -8.45]** | **-34.89 [-37.80, -31.93]** |
| touch + queue-aware | touch_queue_theta0.1 | 6.70 | -4.98 [-6.17, -3.83] | -14.31 [-16.01, -12.74] | -51.63 [-55.90, -47.36] |
| touch + queue-aware | touch_queue_theta0.25 | 10.44 | -6.63 [-7.93, -5.39] | -21.18 [-23.35, -19.08] | -79.39 [-85.98, -72.68] |
| touch + queue-aware | touch_queue_theta0.5 | 13.48 | -7.60 [-9.15, -6.22] | -26.40 [-29.03, -23.82] | -101.60 [-110.11, -92.82] |
| touch + queue-aware | touch_queue_theta1 | 16.07 | -7.99 [-9.59, -6.48] | -30.42 [-33.46, -27.42] | -120.12 [-130.68, -109.27] |
| touch + queue-aware | touch_queue_theta2 | 17.58 | -9.20 [-10.94, -7.55] | -33.74 [-37.17, -30.22] | -131.89 [-143.92, -119.72] |
| touch + OFI-aware | touch_ofi_z_star0.25 | 7.70 | **-6.73 [-8.25, -5.03]** | **-17.49 [-19.79, -15.21]** | **-60.55 [-67.41, -53.71]** |
| touch + OFI-aware | touch_ofi_z_star0.5 | 9.66 | -8.08 [-9.79, -6.31] | -21.57 [-24.42, -18.78] | -75.56 [-84.63, -66.74] |
| touch + OFI-aware | touch_ofi_z_star0.75 | 11.68 | -8.69 [-10.41, -7.01] | -25.01 [-28.33, -21.90] | -90.29 [-101.35, -79.56] |
| touch + OFI-aware | touch_ofi_z_star1 | 13.68 | -9.04 [-10.61, -7.55] | -28.14 [-31.60, -24.79] | -104.58 [-117.00, -92.63] |
| touch + OFI-aware | touch_ofi_z_star1.25 | 15.57 | -9.52 [-11.14, -7.94] | -31.27 [-34.85, -27.68] | -118.27 [-131.47, -105.43] |
| touch + OFI-aware | touch_ofi_z_star1.5 | 17.34 | -9.91 [-11.76, -8.18] | -34.13 [-37.85, -30.33] | -130.99 [-144.14, -118.05] |
