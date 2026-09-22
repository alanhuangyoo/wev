"""Export round trip: a tiny run is trained for a few steps, exported, reloaded, and must answer like the run.
Needs CUDA and Qwen/Qwen3-0.6B-Base."""
import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
ROOT = Path(__file__).resolve().parents[1]


def _row(i):
    opts = {"CLICK": "Click", "TYPE_TEXT": "Type", "DONE": "Done"}
    gold = ["CLICK", "TYPE_TEXT", "DONE"][i % 3]
    return {"request": {"state": {"page": {"url": "", "title": "t", "text": f"page {i} " * 50}},
                        "questions": {"operation": {"type": "choice", "criteria": opts,
                                                    "instructions": {"goal": f"goal {i}"}},
                                      "ok": {"type": "noul", "instructions": "Is it ok?"}}},
            "labels": {"operation": gold, "ok": i % 2 == 0}, "_meta": {}}


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    d = tmp_path_factory.mktemp("wd")
    data = d / "data"
    data.mkdir()
    for split in ("train", "dev"):
        (data / f"{split}.jsonl").write_text("\n".join(json.dumps(_row(i)) for i in range(24)))
    out = d / "run"
    subprocess.run([sys.executable, "-m", "wev.train", "--data", str(data), "--base", "Qwen/Qwen3-0.6B-Base",
                    "--epochs", "1", "--accum", "2", "--eval_n", "4", "--head", "set", "--keep_layers", "12",
                    "--out", str(out)], check=True, cwd=ROOT)
    return d, out


def test_export_answers_like_the_run(run):
    import wev
    d, out = run
    exp = d / "export"
    subprocess.run([sys.executable, "-m", "wev.export", "--run", str(out), "--out", str(exp), "--dtype", "fp32"],
                   check=True, cwd=ROOT)
    meta = json.loads((exp / "wev.json").read_text())
    assert meta["num_layers"] == 12 and meta["head_type"] == "set"
    assert not (exp / "adapter_config.json").exists()      # self-contained: adapter merged, no base needed
    a = wev.load(str(out), dtype=torch.float32)
    b = wev.load(str(exp), dtype=torch.float32)
    for i in range(6):
        r = _row(i)["request"]
        pa = a.predict(r["state"], r["questions"])["answers"]
        pb = b.predict(r["state"], r["questions"])["answers"]
        assert pa["operation"]["choice"] == pb["operation"]["choice"]
        for k, v in pa["operation"]["probabilities"].items():
            assert abs(v - pb["operation"]["probabilities"][k]) < 2e-3
        assert abs(pa["ok"]["noul"] - pb["ok"]["noul"]) < 2e-3
