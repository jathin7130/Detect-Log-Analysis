#!/bin/bash
# Ablation: template-only (plain LSTM, already trained in run_all_lstm.sh,
# seed 42) vs combined (template-id + semantic template embedding).
# Scoped to 1 seed per dataset, matching the reduced 2-variant ablation
# plan agreed earlier (raw-text-only variant dropped).
cd "$(dirname "$0")/.."
DATASETS=(hdfs bgl thunderbird)
mkdir -p logs/ablation

for ds in "${DATASETS[@]}"; do
  echo "=== building template embeddings for $ds $(date) ==="
  ./.venv/Scripts/python.exe -u scripts/build_template_embeddings.py --dataset "$ds" \
    > "logs/ablation/${ds}_embeddings.log" 2>&1

  echo "=== training combined LSTM for $ds $(date) ==="
  ./.venv/Scripts/python.exe -u scripts/train_lstm_combined.py --dataset "$ds" --seed 42 \
    > "logs/ablation/${ds}_combined_train.log" 2>&1

  echo "=== evaluating combined LSTM for $ds $(date) ==="
  ./.venv/Scripts/python.exe -u scripts/evaluate_sequence_model.py \
    --model lstm_combined --dataset "$ds" --seed 42 \
    > "logs/ablation/${ds}_combined_eval.log" 2>&1
done
echo "ABLATION DONE"
