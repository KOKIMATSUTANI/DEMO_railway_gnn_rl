## evaluation set (50 scenarios, optimum computed for 50)

| dispatcher | mean W (train-min) | 95% bootstrap CI | SD | gap vs opt. | diff vs FCFS [95% CI] | Wilcoxon p | d_z | mean delay | deadlocks | truncations |
|---|---|---|---|---|---|---|---|---|---|---|
| fcfs | 13.84 | [11.84, 15.98] | 7.51 | 12.2 % | - | - | - | 8.04 | 0 | 0 |
| fixed_priority | 13.78 | [11.84, 15.88] | 7.24 | 11.7 % | -0.06 [-0.40, 0.26] | 0.7582 | -0.05 | 8.38 | 0 | 0 |
| longest_waiting | 13.88 | [11.86, 16.04] | 7.51 | 12.5 % | 0.04 [-0.12, 0.24] | 0.6547 | 0.06 | 8.02 | 0 | 0 |
| random | 18.80 | [16.14, 21.58] | 9.88 | 52.4 % | 4.96 [3.48, 6.48] | 0.0000 | 0.91 | 12.22 | 0 | 0 |
| mlp | 13.76 | [11.75, 15.93] | 7.48 | 11.5 % | -0.08 [-0.56, 0.41] | 0.8347 | -0.04 | 8.35 | 0 | 0 |
| gnn | 13.71 | [11.70, 15.88] | 7.46 | 11.1 % | -0.13 [-0.63, 0.36] | 0.7452 | -0.07 | 8.34 | 0 | 0 |
| optimum | 12.34 | [10.52, 14.32] | 6.82 | 0.0 % | -1.50 [-2.00, -1.02] | 0.0000 | -0.82 | n/a | 0 | 0 |

Learned policies, seed level (mean over scenarios per seed; 95% t-interval over seeds):

- **gnn**: per seed [13.46, 13.80, 13.68, 13.94, 13.66] -> mean 13.71 [13.49, 13.93] (n = 5)
- **mlp**: per seed [13.72, 13.94, 13.84, 13.54, 13.78] -> mean 13.76 [13.58, 13.95] (n = 5)

Ablation GNN − MLP: per-scenario paired mean diff -0.06 [-0.22, 0.10] (bootstrap), Wilcoxon p = 0.3957, d_z = -0.09; seed level -0.06 [-0.38, 0.27] (t-interval over paired seeds)

## heldout set (20 scenarios, optimum computed for 20)

| dispatcher | mean W (train-min) | 95% bootstrap CI | SD | gap vs opt. | diff vs FCFS [95% CI] | Wilcoxon p | d_z | mean delay | deadlocks | truncations |
|---|---|---|---|---|---|---|---|---|---|---|
| fcfs | 18.75 | [15.60, 22.15] | 7.58 | 16.8 % | - | - | - | 12.10 | 0 | 0 |
| fixed_priority | 18.20 | [14.95, 21.70] | 7.81 | 13.4 % | -0.55 [-1.40, 0.10] | 0.1441 | -0.31 | 11.60 | 0 | 0 |
| longest_waiting | 18.95 | [15.80, 22.35] | 7.56 | 18.1 % | 0.20 [0.00, 0.50] | 0.1573 | 0.32 | 12.10 | 0 | 0 |
| random | 22.75 | [19.20, 26.55] | 8.56 | 41.7 % | 4.00 [1.95, 6.05] | 0.0022 | 0.85 | 15.75 | 0 | 0 |
| mlp | 18.06 | [14.93, 21.54] | 7.56 | 12.5 % | -0.69 [-1.48, -0.07] | 0.0796 | -0.41 | 11.81 | 0 | 0 |
| gnn | 17.96 | [14.90, 21.36] | 7.43 | 11.9 % | -0.79 [-1.62, -0.10] | 0.0625 | -0.44 | 11.84 | 0 | 0 |
| optimum | 16.05 | [13.10, 19.45] | 7.30 | 0.0 % | -2.70 [-3.65, -1.75] | 0.0009 | -1.19 | n/a | 0 | 0 |

Learned policies, seed level (mean over scenarios per seed; 95% t-interval over seeds):

- **gnn**: per seed [18.00, 17.90, 17.90, 18.10, 17.90] -> mean 17.96 [17.85, 18.07] (n = 5)
- **mlp**: per seed [18.20, 18.10, 18.10, 18.00, 17.90] -> mean 18.06 [17.92, 18.20] (n = 5)

Ablation GNN − MLP: per-scenario paired mean diff -0.10 [-0.32, 0.08] (bootstrap), Wilcoxon p = 0.2850, d_z = -0.21; seed level -0.10 [-0.28, 0.08] (t-interval over paired seeds)
