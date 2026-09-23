"""Summarise teacher decisions per episode: outcome, number of steps, the decision sequence."""
import json
import sys
from collections import defaultdict

log, episodes = sys.argv[1], sys.argv[2]
outcome = {e["episode"]: e for e in map(json.loads, open(episodes))}
by = defaultdict(list)
for r in map(json.loads, open(log)):
    by[r["session"]].append(r)
for s, rows in by.items():
    e = outcome.get(s, {})
    seq = []
    for r in rows:
        tgt = [v for k, v in r["labels"].items() if k != "operation"]
        seq.append(r["labels"]["operation"] + (f"[{tgt[0]}]" if tgt else ""))
    print(f"{s}: outcome={e.get('outcome')} {e.get('seconds')}s decisions={len(rows)} final={e.get('final_url', '')[:60]}")
    print("   " + " → ".join(seq)[:400])
    print("   last reason:", rows[-1]["reason"][:200])
