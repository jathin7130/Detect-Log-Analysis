#!/bin/bash
# Same benign-exit-127 caveat as run_all_lstm.sh — no `set -e`.
cd "$(dirname "$0")/.."
SEEDS=(42 123 777)
DATASETS=(hdfs bgl thunderbird)
mkdir -p logs/transformer_runs
for ds in "${DATASETS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    echo "=== starting $ds seed=$seed $(date) ==="
    ./.venv/Scripts/python.exe -u scripts/train_transformer.py --dataset "$ds" --seed "$seed" \
      > "logs/transformer_runs/${ds}_seed${seed}.log" 2>&1
    echo "=== finished $ds seed=$seed exit=$? $(date) ==="
  done
done
echo "ALL TRANSFORMER RUNS DONE"
