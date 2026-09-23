"""Head-to-head on labelled System One requests: send the SAME requests to each model, score the returned answers.

Backends
  http://HOST:PORT/v1/systemone   any System One server (wev.serve, kev.serve, ...)
  laya:<repo>[:<max_len>:<head_max_len>]   Laya via its python package (agent.predict), optional longer context

Scores
  operation accuracy, per-target accuracy, target_k>1 (drops single-candidate targets), step success
  (every labelled question right). A missing or malformed answer counts as wrong and is counted separately.

Standard library only (plus `laya` for that backend) so it runs in any environment.
"""
import argparse
import json
import statistics
import sys
import time
import urllib.request
from collections import defaultdict


def http_backend(url):
    def ask(body):
        req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"content-type": "application/json"})
        return json.load(urllib.request.urlopen(req, timeout=300))
    return ask


def laya_backend(spec):
    import laya
    parts = spec.split(":")
    repo = parts[1]
    try:
        agent = laya.load(repo, device="cuda")
    except TypeError:
        agent = laya.load(repo)
    if len(parts) >= 4:
        agent.cfg["max_len"], agent.cfg["head_max_len"] = int(parts[2]), int(parts[3])
    print(f"laya cfg: max_len={agent.cfg.get('max_len')} head_max_len={agent.cfg.get('head_max_len')}", file=sys.stderr)

    def ask(body):
        return agent.predict(body["state"], body["questions"])
    return ask


def predicted(ans, kind):
    """The answer's top option, scored like wev.evaluate: argmax over the options of the question."""
    if kind == "noul":
        return None if ans.get("noul") is None else ans["noul"] >= 0.5
    if kind == "score":
        if ans.get("probabilities"):
            return int(max(ans["probabilities"], key=ans["probabilities"].get))
        return None if ans.get("score") is None else round(ans["score"])
    return ans.get("choice")


def normalise(gold, kind):
    return bool(gold) if kind == "noul" else int(gold) if kind == "score" else gold


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backend", required=True)
    ap.add_argument("--data", required=True, help="labelled jsonl (rows with request, labels)")
    ap.add_argument("--name", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    ask = laya_backend(a.backend) if a.backend.startswith("laya:") else http_backend(a.backend)
    rows = [json.loads(l) for l in open(a.data) if l.strip()]
    rows = [r for r in rows if all(len(q.get("criteria") or {}) <= 255 for q in r["request"]["questions"].values())]
    if a.limit:
        rows = rows[: a.limit]

    hits, steps, latency, failures = defaultdict(list), [], [], defaultdict(int)
    for i, row in enumerate(rows):
        t = time.perf_counter()
        try:
            answers = ask(row["request"])["answers"]
        except Exception as e:   # counted as wrong on every question
            answers = {}
            failures[f"request:{type(e).__name__}"] += 1
        latency.append((time.perf_counter() - t) * 1000)
        ok = True
        browser = "operation" in row["labels"]
        for qid, gold in row["labels"].items():
            q = row["request"]["questions"][qid]
            got = predicted(answers.get(qid) or {}, q.get("type", "choice"))
            if got is None:
                failures[f"missing:{qid.split('_')[0]}"] += 1
            right = got is not None and got == normalise(gold, q.get("type", "choice"))
            keys = ["all_questions"]
            if browser:
                k = len(q["criteria"])
                keys += ["operation"] if qid == "operation" else ["target", qid] + (["target_k>1"] if k > 1 else [])
            for key in keys:
                hits[key].append(right)
            ok &= right
        steps.append(ok)
        if (i + 1) % 100 == 0:
            print(f"{a.name}: {i + 1}/{len(rows)} step_success so far {sum(steps) / len(steps):.3f}", file=sys.stderr)

    ops = [r["labels"]["operation"] for r in rows if "operation" in r["labels"]] or ["-"]
    majority = max(set(ops), key=ops.count)
    result = {"name": a.name, "backend": a.backend, "n": len(rows),
              "step_success": round(sum(steps) / len(steps), 4),
              **{k: {"n": len(v), "accuracy": round(sum(v) / len(v), 4)} for k, v in sorted(hits.items())},
              "latency_ms_median": round(statistics.median(latency), 1),
              "latency_ms_p90": round(sorted(latency)[int(0.9 * (len(latency) - 1))], 1),
              "failures": dict(failures),
              "baselines": {"operation_majority": f"{majority} {ops.count(majority) / len(ops):.4f}"}}
    json.dump(result, open(a.out, "w"), indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
