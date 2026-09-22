"""LoRA + pointer-head training on labelled System One requests.

Recipe defaults follow kev's findings: LoRA r=16 on all attention + MLP projections, cross-entropy on the option
distribution, one-cycle schedule, and a low learning rate (5e-5 for 4B+; kev measured that 2e-4 erodes base
capability). Base weights stay frozen in bf16; adapter and head train in fp32.
"""
import argparse
import json
import math
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from .data import load_items
from .evaluate import evaluate_items
from .model import MAX_BRANCH, MAX_STATE, DecisionModel


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", required=True,
                    help="comma-separated directories with train.jsonl and dev.jsonl; training mixes all train files, "
                         "each dev file is scored separately")
    ap.add_argument("--base", default="Qwen/Qwen3-0.6B-Base")
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--head_lr", type=float, default=0.0, help="0 = same as --lr")
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--lora", type=int, default=16)
    ap.add_argument("--head_dim", type=int, default=256)
    ap.add_argument("--head", choices=["pointer", "set"], default="pointer",
                    help="pointer: score each option alone; set: options attend to each other before scoring")
    ap.add_argument("--keep_layers", type=int, default=0, help="keep only the first N backbone layers (0 = all)")
    ap.add_argument("--subsample", type=int, default=0, help="train on a fixed random subset of N records")
    ap.add_argument("--batch", type=int, default=1, help="records per forward pass")
    ap.add_argument("--accum", type=int, default=8, help="micro-batches per optimizer step")
    ap.add_argument("--dtype", choices=["bf16", "fp32"], default="bf16", help="frozen base weight dtype")
    ap.add_argument("--checkpointing", type=int, choices=[0, 1], default=1)
    ap.add_argument("--max_state", type=int, default=MAX_STATE)
    ap.add_argument("--max_branch", type=int, default=MAX_BRANCH)
    ap.add_argument("--max_tokens", type=int, default=6144, help="drop packed records longer than this")
    ap.add_argument("--limit", type=int, default=0, help="use only the first N training records (smoke runs)")
    ap.add_argument("--eval_n", type=int, default=500, help="dev records scored after each epoch")
    ap.add_argument("--log_every", type=int, default=10, help="optimizer steps between log lines")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    out = Path(a.out)
    if out.exists():
        ap.error(f"refusing to overwrite {out}")
    out.mkdir(parents=True)
    torch.manual_seed(a.seed)
    rng = random.Random(a.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if dev == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    tok = AutoTokenizer.from_pretrained(a.base)
    train, dropped, devs = [], 0, {}
    for d in [Path(x) for x in a.data.split(",")]:
        items, n_drop = load_items(tok, d / "train.jsonl", a.max_state, a.max_branch, a.max_tokens, a.limit or None)
        train += items
        dropped += n_drop
        devs[d.name], _ = load_items(tok, d / "dev.jsonl", a.max_state, a.max_branch, a.max_tokens, a.eval_n)
        print(f"{d.name}: train {len(items)} (dropped {n_drop}), dev {len(devs[d.name])}", flush=True)
    if a.subsample and a.subsample < len(train):
        train = random.Random(a.seed).sample(train, a.subsample)
    lengths = sorted(len(it["enc"]["ids"]) for it in train)
    print(f"train {len(train)} records; packed tokens median {lengths[len(lengths) // 2]} max {lengths[-1]}", flush=True)
    if not train:
        raise SystemExit("empty training set")

    dtype = torch.bfloat16 if a.dtype == "bf16" else torch.float32
    model = DecisionModel(a.base, tok, dev, lora=a.lora, dtype=dtype, head_dim=a.head_dim, head=a.head,
                          keep_layers=a.keep_layers or None)
    if a.checkpointing:
        model.lm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    head_ids = {id(p) for p in model.head.parameters()}
    groups = [{"params": [p for p in model.trainable_parameters() if id(p) not in head_ids], "lr": a.lr},
              {"params": list(model.head.parameters()), "lr": a.head_lr or a.lr}]
    n_trainable = sum(p.numel() for g in groups for p in g["params"])
    print(f"base {a.base} ({a.dtype}), layers {len(model.lm.base_model.model.layers) if a.lora else len(model.lm.layers)}, "
          f"head {a.head}, trainable {n_trainable / 1e6:.1f}M params", flush=True)
    opt = torch.optim.AdamW(groups, weight_decay=a.weight_decay)
    micro = math.ceil(len(train) / a.batch)
    steps_per_epoch = math.ceil(micro / a.accum)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[a.lr, a.head_lr or a.lr],
                                                total_steps=a.epochs * steps_per_epoch, pct_start=0.1)
    config = {"args": vars(a), "train_records": len(train), "dropped": dropped,
              "dev_records": {k: len(v) for k, v in devs.items()}}
    (out / "training_config.json").write_text(json.dumps(config, indent=2))

    history, step, t0 = [], 0, time.time()
    model.train()
    for ep in range(a.epochs):
        order = list(range(len(train)))
        rng.shuffle(order)
        run_loss, run_n, seen, t_ep = 0.0, 0, 0, time.time()
        for mb in range(micro):
            chunk = [train[i] for i in order[mb * a.batch: (mb + 1) * a.batch]]
            logits = model.forward_batch([it["enc"] for it in chunk])
            losses = [F.cross_entropy(z[None], torch.tensor([y], device=dev))
                      for rec, it in zip(logits, chunk) for z, y in zip(rec, it["enc"]["labels"])]
            loss = torch.stack(losses).mean()
            (loss / a.accum).backward()
            run_loss += loss.item()
            run_n += 1
            seen += len(chunk)
            if (mb + 1) % a.accum == 0 or mb == micro - 1:
                torch.nn.utils.clip_grad_norm_(model.trainable_parameters(), 1.0)
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
                step += 1
                if step % a.log_every == 0 or mb == micro - 1:
                    el = time.time() - t0
                    mem = torch.cuda.max_memory_allocated() / 2 ** 30 if dev == "cuda" else 0
                    print(f"ep {ep} step {step}/{a.epochs * steps_per_epoch} loss {run_loss / run_n:.4f} "
                          f"lr {sched.get_last_lr()[0]:.2e} {seen / max(time.time() - t_ep, 1e-9):.2f} rec/s peak {mem:.1f}GB "
                          f"elapsed {el / 60:.1f}m", flush=True)
                    run_loss, run_n = 0.0, 0
        entry = {"epoch": ep, "step": step}
        for name, items in devs.items():
            entry[name] = evaluate_items(model, items)
            print(f"== epoch {ep} dev[{name}]: {json.dumps(entry[name])}", flush=True)
        history.append(entry)
        (out / "metrics.json").write_text(json.dumps(history, indent=2))
    model.save(out, max_state=a.max_state, max_branch=a.max_branch, train_args=vars(a))
    print(f"saved {out} in {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
