#!/usr/bin/env bash
# Baselines on the general typed-decision benchmarks (test splits), scored by compare.py like wev.evaluate.
set -u
KEV_DIR=${KEV_DIR:-$HOME/kev-ref}
LAYA_PY=${LAYA_PY:-$HOME/laya-env/.venv/bin/python}
SETS=${SETS:-"general/kev-v7 general/kev-transfer-v4 general/typed-decisions"}
MODELS=${MODELS:-"kev-4b kev-8b laya laya-typed-decisions"}
cd "$(dirname "$0")/.."
mkdir -p results/general logs
wait_up() { for _ in $(seq 1 600); do curl -sf "http://127.0.0.1:$1/v1/models" >/dev/null && return 0; sleep 2; done; return 1; }
compare() {   # name backend python
  for ds in $SETS; do
    $3 scripts/compare.py --backend "$2" --data "data/$ds/test.jsonl" --name "$1" \
      --out "results/general/$1-$(basename $ds).json" > "logs/general-$1-$(basename $ds).log" 2>&1
  done
}
for m in $MODELS; do
  case $m in kev-*) ;; *) continue ;; esac
  env -C "$KEV_DIR" KEV_DTYPE=bf16 HF_HUB_OFFLINE=1 .venv/bin/python -m kev.serve --run jaredpalmer/$m --port 8013 > logs/serve-$m.log 2>&1 &
  pid=$!
  wait_up 8013 && compare $m http://127.0.0.1:8013/v1/systemone .venv/bin/python || echo "$m: server did not come up"
  kill $pid; wait $pid 2>/dev/null
done
for m in $MODELS; do
  case $m in laya*) compare $m laya:convaiinnovations/$m "$LAYA_PY" ;; esac
done
echo GENERAL DONE
