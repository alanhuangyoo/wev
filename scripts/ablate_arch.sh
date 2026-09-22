#!/usr/bin/env bash
# Architecture ablation on Qwen3-0.6B: same 4,000-record subset, one epoch, only the head and depth change.
set -u
cd "$(dirname "$0")/.."
COMMON="--data data/m2w-v2,data/nnetnav-v2 --base Qwen/Qwen3-0.6B-Base --epochs 1 --lr 1e-4 --subsample 4000 --eval_n 600 --log_every 50"
for v in "pointer 0" "set 0" "pointer 18" "set 18"; do
  set -- $v
  name="abl-06b-$1-L${2}"
  .venv/bin/python -m wev.train $COMMON --head "$1" --keep_layers "$2" --out "runs/$name" > "logs/$name.log" 2>&1
done
echo ABLATION DONE
