# Phase 3 — Aggregated Stats (mean ± 95% CI across seeds)

| Dataset | Model | precision | recall | f1 | pr_auc | false_alarms_per_10k_events | n_seeds |
|---|---|---|---|---|---|---|---|
| hdfs | lstm | 0.240 ± 0.095 | 0.296 ± 0.072 | 0.265 ± 0.084 | 0.146 ± 0.067 | 277.800 ± 104.755 | 3 |
| hdfs | transformer | 0.258 ± 0.048 | 0.285 ± 0.064 | 0.271 ± 0.055 | 0.147 ± 0.037 | 239.853 ± 8.524 | 3 |
| hdfs | lstm_combined | 0.266 | 0.285 | 0.275 | 0.148 | 230.579 | 1 |
| hdfs | isolation_forest | 0.543 | 0.918 | 0.682 | 0.709 | 226.058 | 1 (no seed variation) |
| hdfs | random_forest | 0.999 | 0.999 | 0.999 | 1.000 | 0.348 | 1 (no seed variation) |
| bgl | lstm | 0.952 ± 0.040 | 0.906 ± 0.009 | 0.928 ± 0.014 | 0.971 ± 0.001 | 47.128 ± 41.609 | 3 |
| bgl | transformer | 0.945 ± 0.013 | 0.908 ± 0.010 | 0.926 ± 0.003 | 0.970 ± 0.003 | 54.197 ± 14.195 | 3 |
| bgl | lstm_combined | 0.945 | 0.909 | 0.927 | 0.970 | 53.725 | 1 |
| bgl | isolation_forest | 0.197 | 0.934 | 0.325 | 0.252 | 3904.991 | 1 (no seed variation) |
| bgl | random_forest | 0.986 | 0.978 | 0.982 | 0.997 | 14.138 | 1 (no seed variation) |
| thunderbird | lstm | 0.754 ± 0.046 | 0.906 ± 0.029 | 0.823 ± 0.033 | 0.784 ± 0.041 | 1136.045 ± 274.814 | 3 |
| thunderbird | transformer | 0.750 ± 0.073 | 0.894 ± 0.019 | 0.815 ± 0.045 | 0.769 ± 0.085 | 1150.070 ± 433.953 | 3 |
| thunderbird | lstm_combined | 0.709 | 0.909 | 0.796 | 0.740 | 1432.579 | 1 |
| thunderbird | isolation_forest | 0.397 | 0.988 | 0.566 | 0.288 | 5755.026 | 1 (no seed variation) |
| thunderbird | random_forest | 1.000 | 0.999 | 0.999 | 1.000 | 0.668 | 1 (no seed variation) |