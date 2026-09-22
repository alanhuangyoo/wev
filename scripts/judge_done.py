"""Re-label "is the goal already achieved?" on labelled browser steps with an LLM judge (OpenAI-compatible API).

NNetNav objectives were written after the fact and are broad, and trajectories often continue after the objective is
already met, so a step labelled CLICK can be a step where the task is in fact done. The judge sees what the decision
model sees (goal, URL, title, page text, elements, recent actions) and answers {"done": bool, "reason": str}.

Credentials come from the environment only and are never printed:
  WEV_JUDGE_BASE_URL, WEV_JUDGE_API_KEY, WEV_JUDGE_MODEL

Rows are keyed by their line number in --data (NNetNav's own "id" is per task, shared by all its steps).
Resumable: rows already in --out are skipped.
"""
import argparse
import json
import os
import re
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

SYSTEM = """You judge browser-agent progress. Given a user's goal and the CURRENT state of the browser (URL, title,
visible page text, interactive elements, and the actions taken so far), decide whether the goal has ALREADY been fully
achieved, so that the correct next step is to stop and report completion.

Rules:
- "done" is true only if the current page and the actions taken already satisfy every part of the goal.
- For information-seeking goals ("find", "look up", "research", "check"), done is true when the requested
  information is visible on the current page or was clearly reached by the actions taken.
- For goals that require an action (save, book, filter, submit, star...), done is true only if that action has
  visibly been completed.
- If a required part is missing, or the page does not show the information yet, done is false.
Reply with a JSON object only: {"done": true or false, "reason": "<one short sentence>"}"""


def user_message(req: dict) -> str:
    st = req["state"]
    goal = req["questions"]["operation"]["instructions"]["goal"]
    elements = "\n".join(f"- [{e.get('index')}] {e.get('role')}: {e.get('label')}" for e in st.get("elements", [])[:40])
    history = "\n".join(
        f"- {h.get('kind')}: {h.get('action')}" + (f" = {h['text']!r}" if h.get("text") else "")
        for h in st.get("recent_actions", [])) or "- (none)"
    page = st.get("page", {})
    return (f"GOAL: {goal}\n\nURL: {page.get('url', '')[:300]}\nTITLE: {page.get('title', '')}\n\n"
            f"PAGE TEXT:\n{page.get('text', '')[:5000]}\n\nINTERACTIVE ELEMENTS (sample):\n{elements}\n\n"
            f"ACTIONS SO FAR:\n{history}")


def call(base, key, model, messages, retries=6):
    # max_tokens covers the model's hidden reasoning too: at 200, a fifth of the replies were cut off empty
    body = json.dumps({"model": model, "messages": messages, "temperature": 0, "max_tokens": 2048}).encode()
    for attempt in range(retries):
        req = urllib.request.Request(base.rstrip("/") + "/chat/completions", data=body,
                                     headers={"content-type": "application/json", "authorization": f"Bearer {key}"})
        try:
            return json.load(urllib.request.urlopen(req, timeout=120))
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 502, 503, 504) and attempt < retries - 1:
                time.sleep(2 ** attempt)
                continue
            raise RuntimeError(f"HTTP {e.code}: {e.read()[:300]!r}") from None
        except (urllib.error.URLError, TimeoutError):
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
                continue
            raise


def parse(text: str):
    m = re.search(r"\{.*\}", text or "", re.S)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj.get("done"), bool) else None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="labelled jsonl (wev converter output)")
    ap.add_argument("--out", required=True, help="jsonl of verdicts, appended to")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    base, key, model = (os.environ.get(k) for k in ("WEV_JUDGE_BASE_URL", "WEV_JUDGE_API_KEY", "WEV_JUDGE_MODEL"))
    if not (base and key and model):
        sys.exit("set WEV_JUDGE_BASE_URL, WEV_JUDGE_API_KEY and WEV_JUDGE_MODEL (e.g. source ~/.wev_judge_env)")

    rows = [dict(json.loads(l), _line=i) for i, l in enumerate(open(a.data))]
    if a.limit:
        rows = rows[: a.limit]
    done_ids = set()
    if os.path.exists(a.out):
        done_ids = {json.loads(l)["line"] for l in open(a.out)}
    todo = [r for r in rows if r["_line"] not in done_ids]
    print(f"{len(rows)} rows, {len(done_ids)} already judged, {len(todo)} to go; model {model}", flush=True)

    lock, usage, n, failures = threading.Lock(), {"prompt_tokens": 0, "completion_tokens": 0}, [0], [0]
    out = open(a.out, "a")

    def work(r):
        msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user_message(r["request"])}]
        try:
            resp = call(base, key, model, msgs)
            text = resp["choices"][0]["message"]["content"]
            verdict = parse(text)
        except Exception as e:   # recorded, not fatal: the row is retried on the next run
            with lock:
                failures[0] += 1
                if failures[0] <= 3:
                    print(f"  failed line {r['_line']}: {e}", flush=True)
            return
        with lock:
            for k in usage:
                usage[k] += (resp.get("usage") or {}).get(k, 0)
            if verdict is None:
                failures[0] += 1
                return
            out.write(json.dumps({"line": r["_line"], "id": r["_meta"].get("id"), "gold": r["labels"]["operation"],
                                  "done": verdict["done"],
                                  "reason": str(verdict.get("reason", ""))[:300], "model": model}) + "\n")
            out.flush()
            n[0] += 1
            if n[0] % 100 == 0:
                print(f"  {n[0]}/{len(todo)} judged, tokens {usage}", flush=True)

    with ThreadPoolExecutor(a.workers) as pool:
        list(pool.map(work, todo))
    out.close()
    print(f"judged {n[0]}, failed {failures[0]} (rerun to retry), tokens {usage}", flush=True)


if __name__ == "__main__":
    main()
