# Stage 5: hyperparameter tuning (validation days only)

Validation days 2023-10-02, 2023-10-10; 1378 paired episodes per side, H = 1 min, Q = 0.5 BTC. Cell = mean validation shortfall (bps, fees included) of the greedy policy, averaged over {private, all variables} × {buy, sell} × {retail, professional}. Learning rate α constant; ε decays linearly from 1 to 0.05 over the given share of the 160,000-episode budget.

| α | ε decay | 2,500 ep | 5,000 ep | 10,000 ep | 20,000 ep | 40,000 ep | 80,000 ep | 160,000 ep |
|---|---|---|---|---|---|---|---|---|
| 0.01 | 0.25 | 1.860 | 1.864 | 1.729 | 1.712 | 1.722 | 1.743 | 1.666 |
| 0.01 | 0.5 | 1.853 | 1.846 | 1.712 | 1.740 | 1.707 | 1.678 | **1.644** |
| 0.03 | 0.25 | 1.833 | 1.825 | 1.756 | 1.758 | 1.723 | 1.703 | 1.677 |
| 0.03 | 0.5 | 1.903 | 1.792 | 1.773 | 1.847 | 1.739 | 1.708 | 1.669 |
| 0.1 | 0.25 | 1.863 | 1.848 | 1.789 | 1.753 | 1.818 | 1.809 | 1.757 |
| 0.1 | 0.5 | 1.849 | 1.810 | 1.825 | 1.835 | 1.827 | 1.750 | 1.774 |

Chosen: α = 0.01, ε decay over 0.5 of the budget, stop after 160,000 episodes.
