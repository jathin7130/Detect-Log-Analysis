#!/bin/bash
# Scores every checkpoint that exists under models/ with evaluate_sequence_model.py.
# Safe to re-run — skips (model, dataset, seed) combos whose checkpoint is missing.
cd "$(dirname "$0")/.."
SEEDS=(42 123 777)
DATASETS=(hdfs bgl thunderbird)
mkdir -p logs/eval_runs

for model_type in lstm transformer; do
  for ds in "${DATASETS[@]}"; do
    for seed in "${SEEDS[@]}"; do
      ckpt="models/${ds}_${model_type}_seed${seed}.pt"
      if [ -f "$ckpt" ]; then
        echo "=== evaluating $model_type $ds seed=$seed $(date) ==="
        ./.venv/Scripts/python.exe -u scripts/evaluate_sequence_model.py \
          --model "$model_type" --dataset "$ds" --seed "$seed" \
          > "logs/eval_runs/${model_type}_${ds}_seed${seed}.log" 2>&1
      fi
    done
  done
done
echo "ALL EVAL RUNS DONE"
