"""External typed-decision corpora -> labelled rows (with soft labels where the corpus has them).

  tasksource-jev   tasksource/tasksource-jev: hundreds of human-labelled classification / multiple-choice tasks recast
                   as runtime-defined decisions. Sampled evenly across source tasks.
  jev-distill      SargeDev/jev-distill-corpus-v3 (Apache-2.0): synthetic operational scenarios over 53 domains,
                   memory-relevance judgements from a 32B teacher, and Open-Jev (CC0) rows; soft labels.
                   Sampled evenly across domains.
  td-synth         n4ze3m/typed-decisions-synth (MIT): LLM-written multi-question cases over 149 domains.
Any row whose state equals a LocalLLaMA/typed-decisions test state is dropped.
"""
import argparse
import hashlib
import json
import random
from collections import Counter
from pathlib import Path


def _norm(s):
    return " ".join(str(s).split()).lower()


def single(r: dict, source: str):
    """One-question corpus row {state, kind, options, target, question} -> labelled row, or None."""
    kind, opts, target = r["kind"], [str(o) for o in r["options"]], [float(x) for x in r["target"]]
    if len(opts) != len(target) or not opts or not r.get("state") or not r.get("question"):
        return None
    best = max(range(len(target)), key=target.__getitem__)
    if kind == "choice":
        if len(set(opts)) != len(opts) or not 2 <= len(opts) <= 255:
            return None
        q = {"type": "choice", "instructions": r["question"], "criteria": {o: None for o in opts}}
        label, soft = opts[best], dict(zip(opts, target))
    elif kind == "noul":
        m = {o.lower(): p for o, p in zip(opts, target)}
        p_true, p_false = m.get("true", m.get("yes")), m.get("false", m.get("no"))
        if p_true is None or p_false is None:
            return None
        q = {"type": "noul", "instructions": r["question"]}
        label, soft = p_true >= p_false, {"false": p_false, "true": p_true}
    elif kind == "score":
        if not 2 <= len(opts) <= 255:
            return None
        q = {"type": "score", "instructions": r["question"], "criteria": opts}
        label, soft = best, target
    else:
        return None
    return {"request": {"model": "wev-latest", "state": str(r["state"]), "questions": {"q": q}},
            "labels": {"q": label}, "soft_labels": {"q": soft}, "_meta": {"source": source, "id": r.get("id")}}


def stratified(groups: list, n: int, rng: random.Random) -> list:
    """Indices of about n rows spread as evenly as possible across groups."""
    by = {}
    for i, g in enumerate(groups):
        by.setdefault(g, []).append(i)
    for v in by.values():
        rng.shuffle(v)
    out, keys = [], sorted(by)
    while len(out) < n and keys:
        nxt = []
        for k in keys:
            if by[k]:
                out.append(by[k].pop())
                nxt.append(k)
            if len(out) >= n:
                break
        keys = [k for k in nxt if by[k]]
    return out


def write(d: Path, split: str, rows: list):
    d.mkdir(parents=True, exist_ok=True)
    with open(d / f"{split}.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return len(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tasksource_n", type=int, default=60000)
    ap.add_argument("--distill_n", type=int, default=50000)
    ap.add_argument("--dev_n", type=int, default=800)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    from datasets import load_dataset
    rng, out, report = random.Random(a.seed), Path(a.out), {}
    td_test = {hashlib.sha256(_norm(r["state"]).encode()).hexdigest()
               for r in load_dataset("LocalLLaMA/typed-decisions", "all", split="test")}
    clean = lambda rows: [r for r in rows if r and hashlib.sha256(_norm(r["request"]["state"]).encode()).hexdigest() not in td_test]

    ts = load_dataset("tasksource/tasksource-jev", split="train")
    split_col = ts["split"]
    tr_idx = [i for i, s in enumerate(split_col) if s == "train"]
    ev_idx = [i for i, s in enumerate(split_col) if s != "train"] or tr_idx[-5000:]
    src = ts["source"]
    pick = [tr_idx[i] for i in stratified([src[i] for i in tr_idx], a.tasksource_n, rng)]
    dev_pick = [ev_idx[i] for i in stratified([src[i] for i in ev_idx], a.dev_n, rng)]
    report["tasksource-jev"] = {
        "train": write(out / "tasksource-jev", "train", clean([single(ts[i], "tasksource-jev") for i in pick])),
        "dev": write(out / "tasksource-jev", "dev", clean([single(ts[i], "tasksource-jev") for i in dev_pick])),
        "source_tasks": len({src[i] for i in pick})}

    jd = load_dataset("SargeDev/jev-distill-corpus-v3", split="train")
    groups = [f"{s}/{d}" for s, d in zip(jd["source"], jd["domain"])]
    pick = stratified(groups, a.distill_n, rng)
    jv = load_dataset("SargeDev/jev-distill-corpus-v3", split="validation")
    vpick = stratified([f"{s}/{d}" for s, d in zip(jv["source"], jv["domain"])], a.dev_n, rng)
    report["jev-distill"] = {
        "train": write(out / "jev-distill", "train", clean([single(jd[i], "jev-distill") for i in pick])),
        "dev": write(out / "jev-distill", "dev", clean([single(jv[i], "jev-distill") for i in vpick])),
        "groups": len({groups[i] for i in pick})}

    def case(r):
        qs, gold = json.loads(r["questions"]), json.loads(r["gold"])
        if set(qs) != set(gold):
            return None
        return {"request": {"model": "wev-latest", "state": r["state"], "questions": qs}, "labels": gold,
                "_meta": {"source": "td-synth", "id": r["state_id"], "domain": r["domain"]}}
    report["td-synth"] = {s: write(out / "td-synth", ours, clean([case(r) for r in load_dataset("n4ze3m/typed-decisions-synth", split=s)]))
                          for s, ours in (("train", "train"), ("validation", "dev"))}
    (out / "manifest.json").write_text(json.dumps({"doc": __doc__, "report": report, "args": vars(a)}, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
