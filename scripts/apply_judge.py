"""Apply judge_done.py verdicts to a labelled dataset directory -> a cleaned copy.

  label not DONE, judge done      -> relabelled DONE (the goal is already met; stopping is the right step)
  label DONE,     judge not done  -> dropped (the recorded stop is wrong, and the right next step is unknown)
  label and judge agree           -> kept as is
  no verdict                      -> kept as is (counted)

Relabelled rows keep only the operation question: a DONE step has no target.
"""
import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="dataset directory with {train,dev,test}.jsonl")
    ap.add_argument("--judge", required=True, help="judge file pattern with {split}, e.g. judge-nnetnav-{split}.jsonl")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    if out.exists():
        ap.error(f"refusing to overwrite {out}")
    out.mkdir(parents=True)
    report = {}
    for split in ("train", "dev", "test"):
        src = Path(a.data) / f"{split}.jsonl"
        jpath = Path(a.judge.format(split=split))
        if not src.exists():
            continue
        verdict = {j["line"]: j["done"] for j in map(json.loads, open(jpath))} if jpath.exists() else {}
        c = Counter()
        with open(out / f"{split}.jsonl", "w") as f:
            for i, line in enumerate(open(src)):
                row = json.loads(line)
                gold_done = row["labels"]["operation"] == "DONE"
                v = verdict.get(i)
                if v is None:
                    c["no_verdict"] += 1
                elif gold_done and not v:
                    c["dropped_false_done"] += 1
                    continue
                elif not gold_done and v:
                    row["request"]["questions"] = {"operation": row["request"]["questions"]["operation"]}
                    row["_meta"]["relabel"] = {"from": row["labels"]["operation"], "by": "judge"}
                    row["labels"] = {"operation": "DONE"}
                    c["relabelled_to_done"] += 1
                else:
                    c["kept"] += 1
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                c["rows_out"] += 1
                c[f"op:{row['labels']['operation']}"] += 1
        report[split] = dict(sorted(c.items()))
    manifest = {"source": str(a.data), "judge": a.judge, "rules": __doc__.strip().splitlines()[2:6], "splits": report,
                "sha256": {s: hashlib.sha256((out / f"{s}.jsonl").read_bytes()).hexdigest() for s in report}}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
