#!/usr/bin/env bash
# kev comparisons, each started once its pinned base model has finished downloading (fetch_kev_bases.py log).
set -u
KEV_DIR=${KEV_DIR:-$HOME/kev-ref}          # a checkout of github.com/jaredpalmer/kev with its venv
LAYA_PY=${LAYA_PY:-$HOME/laya-env/.venv/bin/python}   # a python with `pip install laya`
cd "$(dirname "$0")/.."
DATA=data/m2w-v2/dev.jsonl
wait_up() { for _ in $(seq 1 600); do curl -sf "http://127.0.0.1:$1/v1/models" >/dev/null && return 0; sleep 2; done; return 1; }
run() {   # name repo base-marker
  until grep -q "$3 ok" logs/fetch_kev_bases.log; do sleep 30; done
  env -C "$KEV_DIR" KEV_DTYPE=bf16 HF_HUB_OFFLINE=1 .venv/bin/python -m kev.serve --run "$2" --port 8010 > "logs/serve-$1.log" 2>&1 &
  local pid=$!
  if wait_up 8010; then
    .venv/bin/python scripts/compare.py --backend http://127.0.0.1:8010/v1/systemone --data "$DATA" --name "$1" \
      --out "results/$1.json" > "logs/compare-$1.log" 2>&1
  else
    echo "$1: server did not come up" | tee -a "logs/compare-$1.log"
  fi
  kill "$pid"; wait "$pid" 2>/dev/null
}
run kev-4b jaredpalmer/kev-4b "Qwen/Qwen3.5-4B-Base"
run kev-8b jaredpalmer/kev-8b "Qwen/Qwen3-8B-Base"
echo KEV DONE
