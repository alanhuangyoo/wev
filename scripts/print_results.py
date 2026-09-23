import json, pathlib, sys
for p in sorted(pathlib.Path(sys.argv[1]).glob("*.json")):
    m = json.loads(p.read_text())
    g = m.get("operation_by_gold", {}).get("DONE", {})
    extra = f"  DONE recall {g['recall']:.3f} premature {m.get('premature_done_rate', 0):.3f}" if g else ""
    print(f"{p.stem:<18} all_questions {m['all_questions']['accuracy']:.3f}  step {m['step_success']:.3f}  n={m['n']}{extra}")
