#!/usr/bin/env bash
# Dev-split evaluation of one model on every benchmark: eval_suite.sh <model> <out dir>
set -u
cd "$(dirname "$0")/.."
MODEL=$1; OUT=$2; mkdir -p "$OUT"
for ds in m2w-v2 nnetnav-v3-clean general/kev-v7 general/kev-transfer-v4 general/typed-decisions; do
  name=$(basename $ds)
  .venv/bin/wev evaluate --model "$MODEL" --data "data/$ds" --split dev --out "$OUT/$name.json" > /dev/null 2>&1 || echo "failed: $ds"
done
.venv/bin/python - "$OUT" <<'PY'
import json, sys, pathlib
for p in sorted(pathlib.Path(sys.argv[1]).glob("*.json")):
    m = json.loads(p.read_text())
    done = m.get("operation_by_gold", {}).get("DONE", {}).get("recall")
    print(f"{p.stem:<18} all_questions {m['all_questions']['accuracy']:.3f}  step {m['step_success']:.3f}"
          + (f"  DONE recall {done:.3f}  premature {m.get('premature_done_rate', 0):.3f}" if done is not None else "")
          + f"  n={m['n']}")
PY
