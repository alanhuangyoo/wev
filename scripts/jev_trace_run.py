"""Run one jev-ultrafast task and print every decision (copy into the jev-ultrafast checkout as trace_run.py; collect.py runs it there)."""
import argparse
import json
import time

from jev_ultrafast import Agent
from jev_ultrafast.browser import StalePage

ap = argparse.ArgumentParser()
ap.add_argument("--url", required=True)
ap.add_argument("--goal", action="append", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--stale_retries", type=int, default=10, help="re-observe a page that is still navigating")
a = ap.parse_args()


def settle(fn):
    """Call fn, retrying while the page is mid-navigation (StalePage), which Agent does not retry by itself."""
    for i in range(a.stale_retries):
        try:
            return fn()
        except StalePage:
            time.sleep(1)
    return fn()


steps = []
with settle(lambda: Agent(a.url, a.goal)) as agent:
    state = agent.state
    while agent.state["status"] not in {"done", "blocked"}:
        try:
            state = agent.command("tick")
        except StalePage:   # the tick's own recovery observes again and can hit a navigating page
            settle(lambda: agent.state.update(page=agent.state["browser"].observe(screenshot=agent.screenshots)))
            continue
        d = state.get("decision") or {}
        ops = sorted((d.get("operation_probabilities") or {}).items(), key=lambda x: -x[1])[:3]
        tp = sorted((d.get("target_probabilities") or {}).items(), key=lambda x: -x[1])[:3]
        req = d.get("request") or {}
        elems = {e["index"]: e["label"] for e in (req.get("state") or {}).get("elements", [])}
        tgt = d.get("target")
        last = state["history"][-1] if state["history"] else {}
        line = {"ms": state["elapsed_ms"], "status": state["status"], "url": state["page"]["url"][:80],
                "op": d.get("operation"), "op_top3": [(k, round(v, 2)) for k, v in ops],
                "target": tgt, "target_label": elems.get(str(tgt).split(":")[0]) if tgt else None,
                "target_top3": [(k, elems.get(str(k).split(":")[0], "")[:30], round(v, 2)) for k, v in tp],
                "n_elements": len(elems), "last_action": {k: last.get(k) for k in ("action", "kind", "text")}}
        steps.append(line)
        print(json.dumps(line, ensure_ascii=False))
steps.append({"final_text": state["page"]["text"][:1500]})
json.dump(steps, open(a.out, "w"), indent=1, ensure_ascii=False)
