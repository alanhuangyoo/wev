"""DONE analysis on a labelled split: premature-DONE cases for manual review, and the threshold trade-off.

An agent can accept DONE only when p(DONE) >= t and otherwise take the best other operation. For each t this reports
DONE recall (stops when it should) and the premature-DONE rate (stops on a step whose label is not DONE).
"""
import argparse
import json

import wev
from wev.model import ContextTooLong


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True, help="jsonl of premature-DONE cases (model DONE, label not DONE)")
    a = ap.parse_args()
    m = wev.load(a.model)
    rows = [json.loads(l) for l in open(a.data)]
    scored = []
    with open(a.out, "w") as f:
        for r in rows:
            req = r["request"]
            try:
                ans = m.predict(req["state"], req["questions"])["answers"]["operation"]
            except ContextTooLong:
                continue
            p = ans["probabilities"].get("DONE", 0.0)
            gold = r["labels"]["operation"]
            scored.append((p, gold))
            if ans["choice"] == "DONE" and gold != "DONE":
                st = req["state"]
                f.write(json.dumps({"p_done": p, "gold": gold, "goal": req["questions"]["operation"]["instructions"]["goal"],
                                    "url": st["page"]["url"][:200], "title": st["page"]["title"],
                                    "recent_actions": st["recent_actions"], "page_text": st["page"]["text"][:1500],
                                    "id": r["_meta"].get("id")}, ensure_ascii=False) + "\n")
    done = [p for p, g in scored if g == "DONE"]
    other = [p for p, g in scored if g != "DONE"]
    print(f"{len(scored)} steps: {len(done)} DONE, {len(other)} other")
    print(f"{'threshold':>9} {'DONE recall':>11} {'premature':>9}")
    for t in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95):
        print(f"{t:>9.2f} {sum(p >= t for p in done) / len(done):>11.3f} {sum(p >= t for p in other) / len(other):>9.3f}")


if __name__ == "__main__":
    main()
