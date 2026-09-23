"""End-to-end success = the agent said DONE *and* an LLM judge, reading the final page, agrees the goal is achieved.

Reads <run>/episodes.jsonl and <run>/traces/*.json (from collect.py), writes <run>/verified.jsonl and prints success
rates overall and per task source.
"""
import argparse
import json
import os
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from judge_done import SYSTEM, call, parse  # noqa: E402


def judge(ep, run, base, key, model):
    trace = run / "traces" / f"{ep['episode']}.json"
    if ep["outcome"] != "done" or not trace.exists():
        return {**ep, "verified": False}
    steps = json.load(open(trace))
    final = steps[-1].get("final_text", "") if steps else ""
    msg = (f"GOAL: {ep['goal']}\n\nURL: {ep['final_url'][:300]}\n\nVISIBLE PAGE TEXT:\n{final}\n\n"
           "The agent has stopped here and claims the goal is achieved.")
    try:
        v = parse(call(base, key, model, [{"role": "system", "content": SYSTEM},
                                          {"role": "user", "content": msg}])["choices"][0]["message"]["content"])
    except Exception as e:  # counted as unverified
        v = None
    return {**ep, "verified": bool(v and v["done"]), "judge_reason": (v or {}).get("reason", "")}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run", nargs="+", help="collect.py output directories")
    a = ap.parse_args()
    base, key = os.environ["WEV_JUDGE_BASE_URL"], os.environ["WEV_JUDGE_API_KEY"]
    model = os.environ.get("WEV_JUDGE_MODEL", "deepseek-v4.1-flash")
    for r in a.run:
        run = Path(r)
        eps = [json.loads(l) for l in open(run / "episodes.jsonl")]
        with ThreadPoolExecutor(16) as pool:
            res = list(pool.map(lambda e: judge(e, run, base, key, model), eps))
        with open(run / "verified.jsonl", "w") as f:
            for x in res:
                f.write(json.dumps(x, ensure_ascii=False) + "\n")
        by = Counter(x["source"] for x in res)
        ok = Counter(x["source"] for x in res if x["verified"])
        claimed = sum(x["outcome"] == "done" for x in res)
        print(f"{run.name}: verified success {sum(ok.values())}/{len(res)} = {sum(ok.values()) / len(res):.1%} "
              f"(claimed DONE {claimed}); outcomes {dict(Counter(x['outcome'] for x in res))}")
        for s in sorted(by):
            print(f"   {s:<10} {ok[s]}/{by[s]} = {ok[s] / by[s]:.1%}")


if __name__ == "__main__":
    main()
