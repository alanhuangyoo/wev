"""Run browser tasks in parallel through jev-ultrafast against any System One endpoint (the teacher, or wev itself).

Each episode gets a fresh headless Chromium with a throwaway profile. Its bearer token is the episode id, so the teacher
server can tag the decisions it logs. Outcomes go to <out>/episodes.jsonl; per-episode traces to <out>/traces/.

    python scripts/collect.py --tasks tasks.jsonl --split train --system_one http://127.0.0.1:8010 --out runs/collect-1
"""
import argparse
import glob
import json
import os
import shutil
import socket
import subprocess
import tempfile
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

JEV = Path.home() / "jev-ultrafast"
CHROME = sorted(glob.glob(str(Path.home() / ".cache/ms-playwright/chromium-*/chrome-linux64/chrome")))[-1]
LOCK = threading.Lock()


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def run_episode(task, worker, a, out):
    ep = f"{a.tag}-{task['id']}"
    port, profile = free_port(), tempfile.mkdtemp(prefix="wev-collect-")
    chrome = subprocess.Popen([CHROME, "--headless=new", "--no-sandbox", f"--remote-debugging-port={port}",
                               f"--user-data-dir={profile}", "--window-size=1280,1000", "--lang=en-US", "--no-first-run",
                               "about:blank"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    t0, outcome, err = time.time(), None, ""
    try:
        for _ in range(60):
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{port}/json/version", timeout=2)
                break
            except OSError:
                time.sleep(0.5)
        env = {**os.environ, "BU_CDP_URL": f"http://127.0.0.1:{port}", "BU_NAME": f"w{worker}-{ep}",
               "BH_TELEMETRY": "0", "BH_UPDATE_CHECK": "0", "TYPESAFE_BASE_URL": a.system_one, "TYPESAFE_API_KEY": ep,
               "TYPESAFE_MODEL": a.model_name, "TEXT_MODEL_BASE_URL": os.environ["WEV_JUDGE_BASE_URL"],
               "TEXT_MODEL_API_KEY": os.environ["WEV_JUDGE_API_KEY"], "TEXT_MODEL": a.text_model}
        trace = out / "traces" / f"{ep}.json"
        p = subprocess.run([str(JEV / ".venv/bin/python"), "trace_run.py", "--url", task["url"], "--goal", task["goal"],
                            "--out", str(trace)], cwd=JEV, env=env, capture_output=True, text=True, timeout=a.timeout)
        lines = [json.loads(l) for l in p.stdout.splitlines() if l.startswith("{")]
        outcome = lines[-1]["status"] if lines else "no_state"
        if p.returncode != 0:
            tail = p.stderr.strip().splitlines()[-1] if p.stderr.strip() else ""
            outcome, err = ("budget" if "action demo budget" in tail else "error"), tail[:300]
        final_url = lines[-1]["url"] if lines else ""
    except subprocess.TimeoutExpired:
        outcome, final_url = "timeout", ""
    finally:
        chrome.kill()
        shutil.rmtree(profile, ignore_errors=True)
    rec = {"episode": ep, "task": task["id"], "source": task["source"], "goal": task["goal"], "outcome": outcome,
           "final_url": final_url, "seconds": round(time.time() - t0, 1), "error": err}
    with LOCK, open(out / "episodes.jsonl", "a") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return rec


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--split", default="train")
    ap.add_argument("--sources", default="", help="comma-separated task sources to include (default all)")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--system_one", required=True, help="base URL of the System One server")
    ap.add_argument("--model_name", default="teacher")
    ap.add_argument("--text_model", default="qwen-flash")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--timeout", type=int, default=300)
    ap.add_argument("--tag", default="ep")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out).resolve()   # traces are written by a subprocess that runs in the jev-ultrafast checkout
    (out / "traces").mkdir(parents=True, exist_ok=True)
    done = {json.loads(l)["task"] for l in open(out / "episodes.jsonl")} if (out / "episodes.jsonl").exists() else set()
    tasks = [t for t in map(json.loads, open(a.tasks)) if t["split"] == a.split and t["id"] not in done
             and (not a.sources or t["source"] in a.sources.split(","))]
    if a.limit:
        tasks = tasks[: a.limit]
    print(f"{len(tasks)} tasks to run ({len(done)} already done), {a.workers} workers", flush=True)
    counts, n = {}, [0]

    def job(i_task):
        i, task = i_task
        rec = run_episode(task, i % a.workers, a, out)
        with LOCK:
            counts[rec["outcome"]] = counts.get(rec["outcome"], 0) + 1
            n[0] += 1
            if n[0] % 10 == 0 or n[0] == len(tasks):
                print(f"{n[0]}/{len(tasks)} {counts}", flush=True)

    with ThreadPoolExecutor(a.workers) as pool:
        list(pool.map(job, enumerate(tasks)))
    print("finished", counts, flush=True)


if __name__ == "__main__":
    main()
