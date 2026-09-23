#!/usr/bin/env bash
# eval_all.sh <model> <out>: dev splits of every benchmark; typed-decisions via the held-out 20% of its train split
set -u
cd "$(dirname "$0")/.."
MODEL=$1; OUT=$2; mkdir -p "$OUT"
for ds in m2w-v2 nnetnav-v3-clean teacher-v1 general/kev-v7 general/kev-transfer-v4 general/typed-decisions-train ext/tasksource-jev ext/jev-distill ext/td-synth; do
  .venv/bin/wev evaluate --model "$MODEL" --data "data/$ds" --split dev --out "$OUT/$(basename $ds).json" > /dev/null 2>&1 || echo "failed $ds"
done
