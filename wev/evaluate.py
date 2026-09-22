"""Score a run on a labelled split: operation accuracy, target (element) accuracy, step success, calibration, latency.

step success = every labelled question of the step is right (operation and, when labelled, its target).
target is also reported per operation (click_target, ...) and as target_k>1, which excludes single-candidate questions
(often TYPE steps with one editable field) that are trivially right. chance = mean 1/K, for context.
"""
import argparse
import json
import statistics
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch


def ece(conf, hit, bins=10) -> float:
    conf, hit = np.asarray(conf, dtype=float), np.asarray(hit, dtype=float)
    edges, e = np.linspace(0, 1, bins + 1), 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf >= lo) & ((conf < hi) if hi < 1 else (conf <= hi))
        if m.any():
            e += m.mean() * abs(hit[m].mean() - conf[m].mean())
    return float(e)


def kind_of(qid: str) -> str:
    return "target" if qid.endswith("_target") else qid


@torch.no_grad()
def evaluate_items(model, items) -> dict:
    was_training = model.training
    model.eval()
    per = defaultdict(lambda: {"hit": [], "conf": [], "chance": []})
    steps, latency = [], []
    confusion = defaultdict(lambda: defaultdict(int))   # gold operation -> predicted operation -> count
    cuda = str(model.device).startswith("cuda")
    for it in items:
        if cuda:
            torch.cuda.synchronize()
        t = time.perf_counter()
        probs = model.probs([it["enc"]])[0]
        if cuda:
            torch.cuda.synchronize()
        latency.append((time.perf_counter() - t) * 1000)
        ok = True
        for p, m, y in zip(probs, it["meta"], it["enc"]["labels"]):
            pred = max(range(len(p)), key=p.__getitem__)
            kinds = [kind_of(m["id"])]
            if kinds[0] == "target":   # per operation, and without trivial single-candidate questions
                kinds.append(m["id"])
                if len(p) > 1:
                    kinds.append("target_k>1")
            if m["id"] == "operation":
                confusion[m["keys"][y]][m["keys"][pred]] += 1
            for k in kinds:
                r = per[k]
                r["hit"].append(pred == y)
                r["conf"].append(p[pred])
                r["chance"].append(1 / len(p))
            ok &= pred == y
        steps.append(ok)
    if was_training:
        model.train()
    out = {"n": len(items), "step_success": round(float(np.mean(steps)), 4) if steps else None,
           "latency_ms_median": round(statistics.median(latency), 1) if latency else None,
           "latency_ms_p90": round(float(np.percentile(latency, 90)), 1) if latency else None}
    ops = {g: dict(c) for g, c in confusion.items()}
    out["operation_by_gold"] = {g: {"n": sum(c.values()), "recall": round(c.get(g, 0) / sum(c.values()), 4)}
                                for g, c in sorted(ops.items())}
    not_done = sum(sum(c.values()) for g, c in ops.items() if g != "DONE")
    if not_done:   # an agent that says DONE too early abandons the task: the costliest operation error
        out["premature_done_rate"] = round(sum(c.get("DONE", 0) for g, c in ops.items() if g != "DONE") / not_done, 4)
    out["operation_confusion"] = ops
    for k, r in per.items():
        out[k] = {"n": len(r["hit"]), "accuracy": round(float(np.mean(r["hit"])), 4),
                  "ece": round(ece(r["conf"], r["hit"]), 4), "mean_confidence": round(float(np.mean(r["conf"])), 4),
                  "chance": round(float(np.mean(r["chance"])), 4)}
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", "--model", dest="run", required=True, help="exported model dir, Hub repo, or run dir")
    ap.add_argument("--data", required=True, help="directory with {train,dev,test}.jsonl")
    ap.add_argument("--split", default="dev", choices=["train", "dev", "test"])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--dtype", choices=["bf16", "fp32"], default="bf16")
    ap.add_argument("--out", help="write metrics JSON here")
    a = ap.parse_args()
    from .data import load_items
    from .inference import load
    m = load(a.run, dtype=torch.bfloat16 if a.dtype == "bf16" else torch.float32)
    items, dropped = load_items(m.tokenizer, Path(a.data) / f"{a.split}.jsonl", m.max_state, m.max_branch,
                                limit=a.limit or None)
    metrics = {"run": a.run, "split": a.split, "dropped": dropped, **evaluate_items(m.model, items)}
    print(json.dumps(metrics, indent=2))
    if a.out:
        Path(a.out).write_text(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
