# HDFS LSTM Sensitivity Check

Triggered by: the main Phase 3 result showed LSTM/transformer underperforming
even the simple Isolation Forest baseline on HDFS (F1 ~0.27 vs 0.682), which
was flagged as needing a robustness check before treating it as a clean
finding — HDFS's sequences are much shorter (mean 19.4 events) than
BGL/Thunderbird's fixed 100-event windows, so the default window_size=5 /
topk=3 scoring choice might be losing more relative context on HDFS
specifically than on the other two datasets.

## Result

| Config | Precision | Recall | F1 | PR-AUC | FA/10k seq |
|---|---|---|---|---|---|
| window=5, topk=3 (main Phase 3 config, all datasets) | 0.275 | 0.330 | 0.300 | 0.177 | 255.2 |
| window=10, topk=5 (HDFS-specific sensitivity check) | 0.924 | 0.386 | **0.545** | **0.373** | 9.3 |

Isolation Forest baseline for reference: F1=0.682, PR-AUC=0.709.

## Verdict

**The original finding was partly a hyperparameter artifact.** A larger
window and top-k substantially improves HDFS's LSTM — nearly doubling F1
and PR-AUC, and cutting the false-alarm rate by ~27x (255 → 9.3 per 10k
sequences). This closes most, but not all, of the gap to Isolation Forest
(0.545 vs 0.682) — so the qualitative conclusion ("LSTM is not clearly
better than the simplest classical baseline on HDFS") still holds, but the
quantitative gap is much smaller than the main-config numbers suggest, and
"loses badly" is not an accurate characterization.

## Why window_size=5 was kept as the main-config default

Changing per-dataset hyperparameters after seeing results would be p-hacking
the comparison; the point of a fixed protocol across datasets is that the
comparison is fair. The right place for this finding is a **limitation /
sensitivity-analysis note** in the write-up, not a silent change to the
headline numbers. If a future iteration re-runs the full ablation with
per-dataset-tuned window sizes, that should be a clearly-labeled separate
experiment, not a replacement of the main comparison.

## Files

- `hdfs_lstm_seed42_window5_original.pt` / `_eval.json` — the main-config checkpoint/eval (restored to `models/`, `results/` after this check)
- `hdfs_lstm_seed42_window10.pt` / `_eval.json` / `_history.json` — the sensitivity-check run
