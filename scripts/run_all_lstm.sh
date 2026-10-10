#!/bin/bash
# NOTE: deliberately no `set -e` — torch's CUDA teardown on this machine
# exits non-zero (code 127) even on a fully successful run (verified by
# checking checkpoints/history files after each prior run), so treating
# any non-zero exit as failure would abort this loop after run 1.
cd "$(dirname "$0")/.."
SEEDS=(42 123 777)
DATASETS=(hdfs bgl thunderbird)
for ds in "${DATASETS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    echo "=== starting $ds seed=$seed $(date) ==="
    ./.venv/Scripts/python.exe -u scripts/train_lstm.py --dataset "$ds" --seed "$seed" \
      > "logs/lstm_runs/${ds}_seed${seed}.log" 2>&1
    echo "=== finished $ds seed=$seed exit=$? $(date) ==="
  done
done
echo "ALL LSTM RUNS DONE"
