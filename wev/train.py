"""LoRA + pointer-head training on labelled System One requests.

Recipe defaults follow kev's findings: LoRA r=16 on all attention + MLP projections, cross-entropy on the option
distribution, one-cycle schedule, and a low learning rate (5e-5 for 4B+; kev measured that 2e-4 erodes base
capability). Base weights stay frozen in bf16; adapter and head train in fp32.
"""
import argparse
import datetime
import json
import math
import os
import random
import time
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
from transformers import AutoTokenizer

from .data import load_items
from .evaluate import evaluate_items
from .model import MAX_BRANCH, MAX_STATE, DecisionModel


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", required=True,
                    help="comma-separated directories with train.jsonl and dev.jsonl; training mixes all train files, "
                         "each dev file is scored separately. dir:K repeats that directory's train file K times")
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
    ap.add_argument("--batch_tokens", type=int, default=0,
                    help="token budget per micro-batch (padded); records of similar length are batched together. "
                         "0 = fixed --batch records per micro-batch in random order")
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

    # multi-GPU: launch with torchrun; each rank trains on its share of the batches, gradients are averaged
    rank, world = int(os.environ.get("RANK", 0)), int(os.environ.get("WORLD_SIZE", 1))
    out = Path(a.out)
    if rank == 0:
        if out.exists():
            ap.error(f"refusing to overwrite {out}")
        out.mkdir(parents=True)
    torch.manual_seed(a.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if world > 1:
        dev = f"cuda:{int(os.environ['LOCAL_RANK'])}"
        torch.cuda.set_device(dev)
        dist.init_process_group("nccl", timeout=datetime.timedelta(hours=3))
    log = print if rank == 0 else (lambda *x, **k: None)
    if dev == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    tok = AutoTokenizer.from_pretrained(a.base)
    train, dropped, devs = [], 0, {}
    for spec in a.data.split(","):
        path, _, weight = spec.partition(":")
        d, weight = Path(path), int(weight or 1)
        items, n_drop = load_items(tok, d / "train.jsonl", a.max_state, a.max_branch, a.max_tokens, a.limit or None)
        train += items * weight
        dropped += n_drop
        devs[d.name], _ = load_items(tok, d / "dev.jsonl", a.max_state, a.max_branch, a.max_tokens, a.eval_n)
        log(f"{d.name}: train {len(items)} x{weight} (dropped {n_drop}), dev {len(devs[d.name])}", flush=True)
    if a.subsample and a.subsample < len(train):
        train = random.Random(a.seed).sample(train, a.subsample)
    lengths = sorted(len(it["enc"]["ids"]) for it in train)
    log(f"train {len(train)} records; packed tokens median {lengths[len(lengths) // 2]} max {lengths[-1]}", flush=True)
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
    log(f"base {a.base} ({a.dtype}), layers {len(model.lm.base_model.model.layers) if a.lora else len(model.lm.layers)}, "
          f"head {a.head}, trainable {n_trainable / 1e6:.1f}M params", flush=True)
    opt = torch.optim.AdamW(groups, weight_decay=a.weight_decay)
    def batches(epoch_rng):
        order = list(range(len(train)))
        epoch_rng.shuffle(order)
        if not a.batch_tokens:
            return [order[i: i + a.batch] for i in range(0, len(order), a.batch)]
        out = []
        for start in range(0, len(order), 4096):   # sort within large windows: similar lengths, still shuffled
            window = sorted(order[start: start + 4096], key=lambda i: len(train[i]["enc"]["ids"]))
            cur, cur_max = [], 0
            for i in window:
                n = len(train[i]["enc"]["ids"])
                if cur and max(cur_max, n) * (len(cur) + 1) > a.batch_tokens:
                    out.append(cur)
                    cur, cur_max = [], 0
                cur.append(i)
                cur_max = max(cur_max, n)
            if cur:
                out.append(cur)
        epoch_rng.shuffle(out)
        return out

    plans = [batches(random.Random(a.seed + ep)) for ep in range(a.epochs)]   # fixed up front: exact step count
    if world > 1:   # same plan on every rank; rank r takes every world-th batch, equal counts keep ranks in step
        plans = [p[rank: len(p) - len(p) % world: world] for p in plans]
    total_steps = sum(math.ceil(len(p) / a.accum) for p in plans)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[a.lr, a.head_lr or a.lr],
                                                total_steps=total_steps, pct_start=0.1)
    config = {"args": vars(a), "train_records": len(train), "dropped": dropped,
              "dev_records": {k: len(v) for k, v in devs.items()}}
    if rank == 0:
        (out / "training_config.json").write_text(json.dumps(config, indent=2))
    params = model.trainable_parameters()

    history, step, t0 = [], 0, time.time()
    model.train()
    for ep in range(a.epochs):
        plan = plans[ep]
        micro = len(plan)
        run_loss, run_n, seen, t_ep = 0.0, 0, 0, time.time()
        for mb in range(micro):
            chunk = [train[i] for i in plan[mb]]
            logits = model.forward_batch([it["enc"] for it in chunk])
            losses = []
            for rec, it in zip(logits, chunk):
                for z, y, soft in zip(rec, it["enc"]["labels"], it["enc"].get("soft") or [None] * len(rec)):
                    target = torch.tensor([soft], device=dev) if soft is not None else torch.tensor([y], device=dev)
                    losses.append(F.cross_entropy(z[None], target))
            loss = torch.stack(losses).mean()
            (loss / a.accum).backward()
            run_loss += loss.item()
            run_n += 1
            seen += len(chunk)
            if (mb + 1) % a.accum == 0 or mb == micro - 1:
                if world > 1:
                    grads = [p.grad if p.grad is not None else torch.zeros_like(p) for p in params]
                    flat = torch.cat([g.flatten() for g in grads])
                    dist.all_reduce(flat)
                    flat /= world
                    for p, g in zip(params, flat.split([g.numel() for g in grads])):
                        p.grad = g.view_as(p)
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
                step += 1
                if step % a.log_every == 0 or mb == micro - 1:
                    el = time.time() - t0
                    mem = torch.cuda.max_memory_allocated(dev) / 2 ** 30 if dev.startswith("cuda") else 0
                    log(f"ep {ep} step {step}/{total_steps} loss {run_loss / run_n:.4f} "
                          f"lr {sched.get_last_lr()[0]:.2e} {world * seen / max(time.time() - t_ep, 1e-9):.2f} rec/s peak {mem:.1f}GB "
                          f"elapsed {el / 60:.1f}m", flush=True)
                    run_loss, run_n = 0.0, 0
        if rank == 0:
            entry = {"epoch": ep, "step": step}
            for name, items in devs.items():
                entry[name] = evaluate_items(model, items)
                print(f"== epoch {ep} dev[{name}]: {json.dumps(entry[name])}", flush=True)
            history.append(entry)
            (out / "metrics.json").write_text(json.dumps(history, indent=2))
        if world > 1:
            dist.barrier()
    if rank == 0:
        model.save(out, max_state=a.max_state, max_branch=a.max_branch, train_args=vars(a), world_size=world)
        print(f"saved {out} in {(time.time() - t0) / 60:.1f} min", flush=True)
    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
