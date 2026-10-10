# Phase 2 — Classical Baselines (locked)

| Dataset | Model | Precision | Recall | F1 | PR-AUC | FPR | FA/10k seq | Train (s) |
|---|---|---|---|---|---|---|---|---|
| hdfs | isolation_forest | 0.543 | 0.918 | 0.682 | 0.709 | 0.023 | 226.1 | 4.7 |
| hdfs | random_forest | 0.999 | 0.999 | 0.999 | 1.000 | 0.000 | 0.3 | 13.3 |
| bgl | isolation_forest | 0.197 | 0.934 | 0.325 | 0.252 | 0.435 | 3905.0 | 0.7 |
| bgl | random_forest | 0.986 | 0.978 | 0.982 | 0.997 | 0.002 | 14.1 | 1.1 |
| thunderbird | isolation_forest | 0.397 | 0.988 | 0.566 | 0.288 | 0.934 | 5755.0 | 1.5 |
| thunderbird | random_forest | 1.000 | 0.999 | 0.999 | 1.000 | 0.000 | 0.7 | 6.3 |