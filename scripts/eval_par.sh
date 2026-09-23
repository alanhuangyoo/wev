#!/usr/bin/env bash
# eval_par.sh <model> <out> [gpus]: eval_all.sh with one benchmark per GPU, e.g. eval_par.sh runs/x evals/x 0,1,2,3
# SPLIT=test SETS="m2w-v2 general/kev-v7" to choose the split and benchmarks
set -u
cd "$(dirname "$0")/.."
MODEL=$1; OUT=$2; IFS=, read -ra GPUS <<< "${3:-0,1,2,3,4,5,6,7}"; mkdir -p "$OUT"
i=0
SETS=${SETS:-"m2w-v2 nnetnav-v3-clean teacher-v1 general/kev-v7 general/kev-transfer-v4 general/typed-decisions-train ext/tasksource-jev ext/jev-distill ext/td-synth"}
for ds in $SETS; do
  [ -f "data/$ds/${SPLIT:-dev}.jsonl" ] || continue
  CUDA_VISIBLE_DEVICES=${GPUS[$((i % ${#GPUS[@]}))]} .venv/bin/wev evaluate --model "$MODEL" --data "data/$ds" --split "${SPLIT:-dev}" \
    --out "$OUT/$(basename $ds).json" > "$OUT/$(basename $ds).log" 2>&1 || echo "failed $ds" &
  i=$((i + 1))
done
wait
.venv/bin/python scripts/print_results.py "$OUT"
