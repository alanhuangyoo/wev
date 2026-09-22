#!/usr/bin/env bash
# Head-to-head on the Mind2Web dev split: every model gets the same requests, one model on the GPU at a time.
set -u
KEV_DIR=${KEV_DIR:-$HOME/kev-ref}          # a checkout of github.com/jaredpalmer/kev with its venv
LAYA_PY=${LAYA_PY:-$HOME/laya-env/.venv/bin/python}   # a python with `pip install laya`
cd "$(dirname "$0")/.."
mkdir -p results logs
DATA=${DATA:-data/m2w-v2/dev.jsonl}

wait_up() { for _ in $(seq 1 300); do curl -sf "http://127.0.0.1:$1/v1/models" >/dev/null && return 0; sleep 2; done; return 1; }

serve_and_compare() {   # name port server-command...
  local name=$1 port=$2; shift 2
  "$@" > "logs/serve-$name.log" 2>&1 &
  local pid=$!
  if wait_up "$port"; then
    .venv/bin/python scripts/compare.py --backend "http://127.0.0.1:$port/v1/systemone" --data "$DATA" \
      --name "$name" --out "results/$name.json" > "logs/compare-$name.log" 2>&1
  else
    echo "$name: server did not come up" | tee -a "logs/compare-$name.log"
  fi
  kill "$pid"; wait "$pid" 2>/dev/null
}

serve_and_compare wev-4b   8009 .venv/bin/python -m wev.serve --run runs/wd-4b-v2   --port 8009
serve_and_compare wev-1.7b 8009 .venv/bin/python -m wev.serve --run runs/wd-1.7b-v2 --port 8009
serve_and_compare kev-4b 8010 env -C "$KEV_DIR" KEV_DTYPE=bf16 .venv/bin/python -m kev.serve --run jaredpalmer/kev-4b --port 8010
serve_and_compare kev-8b 8010 env -C "$KEV_DIR" KEV_DTYPE=bf16 .venv/bin/python -m kev.serve --run jaredpalmer/kev-8b --port 8010

LAYA="$LAYA_PY scripts/compare.py --data $DATA"
$LAYA --backend laya:convaiinnovations/laya --name laya --out results/laya.json > logs/compare-laya.log 2>&1
$LAYA --backend laya:convaiinnovations/laya:4096:1536 --name laya-long --out results/laya-long.json > logs/compare-laya-long.log 2>&1
$LAYA --backend laya:convaiinnovations/laya-typed-decisions --name laya-typed-decisions --out results/laya-typed-decisions.json > logs/compare-laya-typed-decisions.log 2>&1
echo ALL DONE
