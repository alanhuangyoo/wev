"""DONE metrics on steps where the dataset label and an LLM judge agree (see judge_done.py).

  truly-not-done: label is not DONE and the judge says the goal is not yet achieved -> premature-DONE rate
  truly-done:     label is DONE and the judge agrees the goal is achieved              -> DONE recall
Also reports how often label and judge disagree, which is the label noise the raw metrics carry.
"""
import argparse
import json
from collections import Counter

import wev
from wev.model import ContextTooLong


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--judge", required=True, help="judge_done.py output for the same rows")
    a = ap.parse_args()
    judge = {j["line"]: j["done"] for j in map(json.loads, open(a.judge))}   # keyed by line: task ids repeat
    m = wev.load(a.model)
    agree, rows = Counter(), []
    for rid, line in enumerate(open(a.data)):
        r = json.loads(line)
        if rid not in judge:
            continue
        gold_done = r["labels"]["operation"] == "DONE"
        agree[(gold_done, judge[rid])] += 1
        try:
            ans = m.predict(r["request"]["state"], r["request"]["questions"])["answers"]["operation"]
        except ContextTooLong:
            continue
        rows.append((gold_done, judge[rid], ans["choice"] == "DONE", ans["probabilities"].get("DONE", 0.0)))

    n = sum(agree.values())
    print(f"{n} judged steps. label vs judge:")
    print(f"  label DONE,     judge done:     {agree[(True, True)]:>5}")
    print(f"  label DONE,     judge not done: {agree[(True, False)]:>5}")
    print(f"  label not DONE, judge done:     {agree[(False, True)]:>5}   <- goal already met, label says continue")
    print(f"  label not DONE, judge not done: {agree[(False, False)]:>5}")
    truly_done = [(c, p) for g, j, c, p in rows if g and j]
    truly_not = [(c, p) for g, j, c, p in rows if not g and not j]
    raw_not = [(c, p) for g, j, c, p in rows if not g]
    raw_done = [(c, p) for g, j, c, p in rows if g]

    def rate(xs, t=None):
        return sum((p >= t) if t is not None else c for c, p in xs) / max(len(xs), 1)

    print(f"\n{'':<34}{'raw labels':>12}{'label+judge agree':>20}")
    print(f"{'DONE recall (argmax)':<34}{rate(raw_done):>12.3f}{rate(truly_done):>20.3f}   n={len(raw_done)}/{len(truly_done)}")
    print(f"{'premature DONE (argmax)':<34}{rate(raw_not):>12.3f}{rate(truly_not):>20.3f}   n={len(raw_not)}/{len(truly_not)}")
    print(f"\n{'threshold':>9}{'recall (clean)':>16}{'premature (clean)':>19}")
    for t in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8):
        print(f"{t:>9.2f}{rate(truly_done, t):>16.3f}{rate(truly_not, t):>19.3f}")


if __name__ == "__main__":
    main()
