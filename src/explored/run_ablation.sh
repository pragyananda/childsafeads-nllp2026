#!/usr/bin/env bash
# Accuracy at each data-access level -- the ablation the task is actually about.
# Runs sequentially: one 16GB GPU, one encoder at a time.
set -u
cd "$(dirname "$0")"
mkdir -p ../work/ablation
for L in 1 2 3 4; do
  echo "=== level $L ==="
  python3 encoder.py --model "${MODEL:-answerdotai/ModernBERT-large}" \
      --level "$L" --target dev --maxlen "${MAXLEN:-1024}" \
      --out "../work/ablation/enc_dev_L${L}.jsonl" \
      2>&1 | tee "../work/ablation/L${L}.log" | grep -E "epoch |L${L} |wrote |Traceback|Error"
done
echo "=== summary ==="
grep -h "st1_all" ../work/ablation/L*.log
