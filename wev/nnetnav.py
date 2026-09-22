"""NNetNav -> labelled System One requests in the same jev-ultrafast shape as wev.mind2web.

NNetNav (stanfordnlp/nnetnav-live, Apache-2.0) is one agent step per row: an accessibility tree, the URL, the objective,
previous actions, and the next action in a ``` block. It supplies what Mind2Web lacks: DONE (stop [answer]),
BLOCKED (stop [N/A]) and SCROLL steps.

  click [id] / hover [id]      -> CLICK      + click_target
  type [id] [text] [0|1]       -> TYPE_TEXT  + type_text_target
  scroll [down|up]             -> SCROLL_DOWN / SCROLL_UP
  stop [N/A]                   -> BLOCKED
  stop [answer]                -> DONE
  goto / go_back / press / tabs, or no parseable action -> dropped (no jev-ultrafast equivalent)

NNetNav trajectories are LLM-explored and relabelled, not human-annotated; stop steps are reliable by construction
(the objective is written from the trajectory that ends there).
"""
import argparse
import hashlib
import json
import random
import re
from collections import Counter
from pathlib import Path

from .mind2web import _clip, _norm, _weighted_sample
from .prompts import (CONTROL_LABELS, HISTORY_STEPS, NEXT_ACTION, OPERATION_LABELS, PAGE_TEXT_CHARS, TARGET,
                      TERMINAL_LABELS)

NODE = re.compile(r"^(?P<indent>\t*)(?:\[(?P<id>\d+)\] )?(?P<role>[\w-]+) '(?P<rest>.*)$")
INTERACTIVE = {"button", "link", "textbox", "searchbox", "combobox", "checkbox", "radio", "menuitem", "option", "tab",
               "switch", "listbox", "menuitemcheckbox", "menuitemradio", "slider", "spinbutton", "treeitem"}
EDITABLE = {"textbox", "searchbox", "combobox", "spinbutton"}
ACTIONS = [
    (re.compile(r"^(?:click|hover) \[(\d+)\]", re.S), "CLICK"),
    (re.compile(r"^type \[(\d+)\] \[(.*?)\](?: \[[01]\])?\s*$", re.S), "TYPE_TEXT"),
    (re.compile(r"^scroll \[(down|up)\]", re.S), "SCROLL"),
    (re.compile(r"^stop \[(.*)\]\s*$", re.S), "STOP"),
]
HISTORY = re.compile(r"^\d+:\s*(?P<act>.*?)(?: where \[\d+\] is (?P<label>.*))?$")


class Node:
    __slots__ = ("id", "role", "name", "props", "parent", "children")

    def __init__(self, nid, role, name, props, parent):
        self.id, self.role, self.name, self.props, self.parent, self.children = nid, role, name, props, parent, []


def parse_tree(text: str):
    """Accessibility tree -> (root title, root url, nodes by id, page text lines in order)."""
    nodes, stack, lines, title, url = {}, [], [], "", ""
    for raw in text.split("\n"):
        m = NODE.match(raw)
        if not m:
            continue
        rest = m["rest"]
        cut = rest.find("', ")
        name, props = (rest[:cut], rest[cut + 3:]) if cut >= 0 else (rest.rstrip("'"), "")
        depth = len(m["indent"])
        if m["role"] == "RootWebArea":
            title = _norm(name)
            u = re.search(r"url='([^']*)'", props)
            url = u.group(1) if u else ""
            stack = []
            continue
        while stack and stack[-1][0] >= depth:
            stack.pop()
        parent = stack[-1][1] if stack else None
        node = Node(m["id"], m["role"], _norm(name), props, parent)
        if parent is not None:
            parent.children.append(node)
        if node.id:
            nodes[node.id] = node
        stack.append((depth, node))
        if m["role"] in ("StaticText", "heading") and node.name:
            lines.append(node.name)
    return title, url, nodes, lines


def _label(node) -> str:
    if node.name:
        return _clip(node.name)
    texts = []
    todo = list(node.children)
    while todo and len(" ".join(texts)) < 200:
        c = todo.pop(0)
        if c.name:
            texts.append(c.name)
        todo = c.children + todo
    return _clip(" ".join(texts)) or node.role


def _relatives(node):
    out, p = set(), node.parent
    while p is not None:
        out.add(p)
        p = p.parent
    todo = list(node.children)
    while todo:
        c = todo.pop()
        out.add(c)
        todo.extend(c.children)
    return out


def parse_prompt(prompt: str):
    obs = prompt[prompt.rfind("OBSERVATION:") + len("OBSERVATION:"):]
    tree, _, tail = obs.partition("\nURL:")
    url, _, tail = tail.partition("\nOBJECTIVE:")
    objective, _, tail = tail.partition("\nPREVIOUS ACTIONS:")
    history = tail.split("<|eot_id|>")[0]
    return tree, url.strip(), objective.strip(), [l.strip() for l in history.strip().split("\n") if l.strip()]


def parse_action(output: str):
    blocks = re.findall(r"```(.*?)```", output, re.S)
    if not blocks:
        return None
    act = blocks[-1].strip()
    for rx, kind in ACTIONS:
        m = rx.match(act)
        if m:
            return kind, m.groups()
    return None


def recent_actions(lines):
    out = []
    for line in lines:
        m = HISTORY.match(line)
        if not m or m["act"] in ("None", ""):
            continue
        act, label = m["act"], _clip(m["label"] or "")
        if act.startswith(("click", "hover")):
            out.append({"action": label or "element", "kind": "click", "text": None, "page_changed": None})
        elif act.startswith("type"):
            t = re.match(r"type \[\d+\] \[(.*?)\]", act)
            out.append({"action": label or "field", "kind": "fill", "text": _norm(t.group(1)) if t else None,
                        "page_changed": None})
        elif act.startswith("scroll"):
            out.append({"action": "Scroll up" if "up" in act else "Scroll down", "kind": "scroll", "text": None,
                        "page_changed": None})
    return out[-HISTORY_STEPS:]


def convert_row(row, rng, k_min, k_max, text_min, text_max):
    """-> (labelled row, None) or (None, skip reason)."""
    action = parse_action(row["output"])
    if action is None:
        return None, "no_action"
    kind, args = action
    tree, url, objective, history_lines = parse_prompt(row["prompt"])
    if not objective:
        return None, "no_objective"
    title, root_url, nodes, text_lines = parse_tree(tree)
    if kind == "STOP":
        op = "BLOCKED" if _norm(args[0]).upper() in ("N/A", "NA", "") else "DONE"
    elif kind == "SCROLL":
        op = "SCROLL_UP" if args[0] == "up" else "SCROLL_DOWN"
    else:
        op = kind
    gold = nodes.get(args[0]) if op in ("CLICK", "TYPE_TEXT") else None
    if op in ("CLICK", "TYPE_TEXT") and gold is None:
        return None, "target_not_in_tree"

    related = _relatives(gold) if gold is not None else set()
    seen = {_label(gold).casefold()} if gold is not None else set()
    candidates = []
    for n in nodes.values():
        if n is gold or n in related:
            continue
        if n.role not in INTERACTIVE and "clickable" not in n.props:
            continue
        key = _label(n).casefold()
        if key in seen:
            continue
        seen.add(key)
        candidates.append(n)
    k = rng.randint(k_min, k_max)
    weights = [3.0 if n.role in INTERACTIVE else 1.0 for n in candidates]
    chosen = _weighted_sample(candidates, weights, k - (1 if gold is not None else 0), rng)
    elements = ([gold] if gold is not None else []) + chosen
    rng.shuffle(elements)
    if not elements:
        return None, "no_elements"

    targets = {"CLICK": {}, "TYPE_TEXT": {}}
    state_elements, gold_target = [], None
    for idx, n in enumerate(elements, start=1):
        index = str(idx)
        ops = ["TYPE_TEXT", "CLICK"] if n.role in EDITABLE else ["CLICK"]
        if n is gold and op not in ops:
            ops = [op] + ops
        label = _label(n)
        state_elements.append({"role": n.role, "index": index, "label": label, "operations": ops})
        for o in ("TYPE_TEXT", "CLICK"):
            if o in ops:
                shown = f"Open {label}" if o == "CLICK" and "TYPE_TEXT" in ops else label
                targets[o][index] = {"element": f"[{index}] {shown}", "current_value": "", "role": n.role}
                if n is gold and op == o:
                    gold_target = index

    history = recent_actions(history_lines)
    present = [o for o in ("CLICK", "TYPE_TEXT") if targets[o]]
    rng.shuffle(present)
    operations = {o: OPERATION_LABELS[o] for o in present}
    # jev-ultrafast offers scroll up only once the page has been scrolled
    operations["SCROLL_DOWN"] = CONTROL_LABELS["SCROLL_DOWN"]
    if op == "SCROLL_UP" or any(h["action"] == "Scroll down" for h in history):
        operations["SCROLL_UP"] = CONTROL_LABELS["SCROLL_UP"]
    operations["WAIT"] = CONTROL_LABELS["WAIT"]
    operations.update(TERMINAL_LABELS)
    if op not in operations:
        return None, f"op_not_offered:{op}"

    questions = {"operation": {"type": "choice", "criteria": operations,
                               "instructions": {"goal": objective, "rules": NEXT_ACTION}}}
    labels = {"operation": op}
    if gold_target is not None:
        qid = op.lower() + "_target"
        questions[qid] = {"type": "choice", "criteria": targets[op],
                          "instructions": {"goal": objective, "operation": op, "rules": [NEXT_ACTION, TARGET]}}
        labels[qid] = gold_target
    page_text = "\n".join(text_lines)[: rng.randint(text_min, min(text_max, PAGE_TEXT_CHARS))]
    state = {"page": {"url": url or root_url, "title": title, "text": page_text},
             "elements": state_elements, "recent_actions": history}
    meta = {"source": "nnetnav-live", "id": row["id"], "task": row["task_name"], "op": op, "k": len(elements),
            "has_target": gold_target is not None}
    return {"request": {"model": "wev-latest", "state": state, "questions": questions},
            "labels": labels, "_meta": meta}, None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--k_min", type=int, default=8)
    ap.add_argument("--k_max", type=int, default=40)
    ap.add_argument("--text_min", type=int, default=1500)
    ap.add_argument("--text_max", type=int, default=PAGE_TEXT_CHARS)
    ap.add_argument("--max_train", type=int, default=12000,
                    help="cap on training rows, sampled at the natural operation mix (all SCROLL_UP rows kept)")
    ap.add_argument("--max_eval", type=int, default=1200, help="cap on dev and test rows each")
    a = ap.parse_args()
    out = Path(a.out)
    if out.exists():
        ap.error(f"refusing to overwrite {out}")
    out.mkdir(parents=True)
    from datasets import load_dataset

    counts, skips, ops = Counter(), Counter(), Counter()
    for src_split in ("train", "test"):
        ds = load_dataset("stanfordnlp/nnetnav-live", split=src_split)
        rows, rare = [], []
        for i in range(len(ds)):
            r = ds[i]
            rng = random.Random(f"{a.seed}:{r['id']}")
            row, why = convert_row(r, rng, a.k_min, a.k_max, a.text_min, a.text_max)
            if row is None:
                skips[why] += 1
                continue
            (rare if row["labels"]["operation"] not in ("CLICK", "TYPE_TEXT") else rows).append(row)
            if (i + 1) % 5000 == 0:
                print(f"{src_split}: {i + 1}/{len(ds)} kept {len(rows) + len(rare)} skipped {sum(skips.values())}",
                      flush=True)
        if src_split == "train":
            # Sample at the natural operation mix: keeping every DONE row made DONE half the training set, which
            # teaches a browser agent to stop early. Only SCROLL_UP, the rarest operation, is always kept.
            pool = rows + [r for r in rare if r["labels"]["operation"] != "SCROLL_UP"]
            scroll_up = [r for r in rare if r["labels"]["operation"] == "SCROLL_UP"]
            random.Random(a.seed).shuffle(pool)
            splits = {"train": scroll_up + pool[: max(0, a.max_train - len(scroll_up))]}
        else:   # NNetNav's own test split, divided by task into our dev and locked test
            dev, test = [], []
            for row in rare + rows:
                h = int(hashlib.sha256(f"{a.seed}:{row['_meta']['task']}".encode()).hexdigest(), 16) % 2
                (dev if h == 0 else test).append(row)
            random.Random(a.seed).shuffle(dev)
            random.Random(a.seed + 1).shuffle(test)
            splits = {"dev": dev[: a.max_eval], "test": test[: a.max_eval]}
        for name, part in splits.items():
            with open(out / f"{name}.jsonl", "w") as f:
                for row in part:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
                    ops[(name, row["labels"]["operation"])] += 1
            counts[name] = len(part)
    manifest = {"source": "stanfordnlp/nnetnav-live", "args": vars(a), "rows": dict(counts),
                "ops": {f"{s}:{o}": n for (s, o), n in sorted(ops.items())}, "skipped": dict(skips),
                "sha256": {s: hashlib.sha256((out / f"{s}.jsonl").read_bytes()).hexdigest() for s in counts}}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps({k: manifest[k] for k in ("rows", "ops", "skipped")}, indent=2))


if __name__ == "__main__":
    main()
