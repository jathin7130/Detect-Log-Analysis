#!/bin/bash
cd "$(dirname "$0")/.."
PAIRS=("hdfs bgl" "hdfs thunderbird" "bgl hdfs" "bgl thunderbird" "thunderbird hdfs" "thunderbird bgl")
for pair in "${PAIRS[@]}"; do
  read -r src tgt <<< "$pair"
  echo "=== building ${src}_to_${tgt} $(date) ==="
  ./.venv/Scripts/python.exe scripts/cross_domain_build.py --source "$src" --target "$tgt" \
    > "logs/cross_domain_${src}_to_${tgt}.log" 2>&1
  echo "=== done ${src}_to_${tgt} exit=$? $(date) ==="
done
echo "ALL CROSS-DOMAIN BUILDS DONE"
