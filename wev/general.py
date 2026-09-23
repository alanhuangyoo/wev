"""General-purpose typed-decision data -> labelled rows (same shape as the browser converters).

  kev-v7                kev's decision-v7 suite (github.com/jaredpalmer/kev): ten public classification / QA sources plus
                        programmatic policy and rule-composition records. train from the kev-suites mirror; dev/test are
                        the suite's development/test partitions from the kev repository.
  kev-transfer-v4       kev's out-of-domain suite, dev/test only: compare with kev's published transfer numbers.
  typed-decisions       LocalLLaMA/typed-decisions (Apache-2.0), four agent/ops workflows, 5 questions per case.
                        dev = its train split, test = its test split: a model that never trains on it is scored as a
                        generalist, like Jev in the dataset card. typed-decisions-train holds 80% of its train split for
                        training a specialist and 20% as that specialist's dev.
"""
import argparse
import hashlib
import json
import random
from pathlib import Path

KEV_SUITES = "jaredpalmer/kev-suites"
KEV_SUITES_REVISION = "a3318ddc1f630c5673232efacd8123a84de3f480"


def from_kev(row: dict, source: str) -> dict:
    """kev suite record (labels inside the questions) -> labelled row."""
    questions, labels = {}, {}
    for qid, q in row["questions"].items():
        labels[qid] = q["label"]
        questions[qid] = {k: v for k, v in q.items() if k in ("type", "instructions", "criteria")}
    meta = {"source": source, "kev_source": row.get("_meta", {}).get("source"), "id": row.get("_meta", {}).get("id")}
    return {"request": {"model": "wev-latest", "state": row["state"], "questions": questions}, "labels": labels,
            "_meta": meta}


def from_typed_decisions(row: dict) -> dict:
    questions, gold = json.loads(row["questions"]), json.loads(row["gold"])
    labels = {}
    for qid, q in questions.items():
        g = gold[qid]["label"]
        labels[qid] = {"noul": lambda x: str(x).lower() == "true", "score": int}.get(q["type"], str)(g)
    return {"request": {"model": "wev-latest", "state": row["state"], "questions": questions}, "labels": labels,
            "_meta": {"source": "typed-decisions", "id": row["id"], "workflow": row["workflow"]}}


def write(directory: Path, split: str, rows: list):
    directory.mkdir(parents=True, exist_ok=True)
    with open(directory / f"{split}.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return {"rows": len(rows), "sha256": hashlib.sha256((directory / f"{split}.jsonl").read_bytes()).hexdigest()}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, help="parent directory; one subdirectory per dataset")
    ap.add_argument("--kev_repo", required=True, help="a checkout of github.com/jaredpalmer/kev (for its dev/test suites)")
    a = ap.parse_args()
    out, kev = Path(a.out), Path(a.kev_repo)
    from datasets import load_dataset
    from huggingface_hub import hf_hub_download
    report = {}

    path = hf_hub_download(KEV_SUITES, "v7/decision-v7/train.jsonl", repo_type="dataset", revision=KEV_SUITES_REVISION)
    d = out / "kev-v7"
    report["kev-v7"] = {"train": write(d, "train", [from_kev(json.loads(l), "kev-v7") for l in open(path)])}
    for ours, theirs in (("dev", "development"), ("test", "test")):
        rows = [from_kev(json.loads(l), "kev-v7") for l in open(kev / "evals/v7/decision-v7" / f"{theirs}.jsonl")]
        report["kev-v7"][ours] = write(d, ours, rows)
    d = out / "kev-transfer-v4"
    report["kev-transfer-v4"] = {}
    for ours, theirs in (("dev", "development"), ("test", "test")):
        rows = [from_kev(json.loads(l), "kev-transfer-v4") for l in open(kev / "evals/v4/transfer-v4" / f"{theirs}.jsonl")]
        report["kev-transfer-v4"][ours] = write(d, ours, rows)

    td_train = [from_typed_decisions(r) for r in load_dataset("LocalLLaMA/typed-decisions", "all", split="train")]
    td_test = [from_typed_decisions(r) for r in load_dataset("LocalLLaMA/typed-decisions", "all", split="test")]
    report["typed-decisions"] = {"dev": write(out / "typed-decisions", "dev", td_train),
                                 "test": write(out / "typed-decisions", "test", td_test)}
    shuffled = td_train[:]
    random.Random(0).shuffle(shuffled)
    cut = int(0.8 * len(shuffled))
    report["typed-decisions-train"] = {"train": write(out / "typed-decisions-train", "train", shuffled[:cut]),
                                       "dev": write(out / "typed-decisions-train", "dev", shuffled[cut:])}
    for name, splits in report.items():
        (out / name / "manifest.json").write_text(json.dumps({"splits": splits, "doc": __doc__}, indent=2))
    print(json.dumps({k: {s: v["rows"] for s, v in sp.items()} for k, sp in report.items()}, indent=2))


if __name__ == "__main__":
    main()
