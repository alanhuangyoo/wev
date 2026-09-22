# wev

**A local, open-weights decision model for browser agents.** At every step a browser agent asks *which operation?*
(click, type, select, scroll, done…) and *which element?* `wev` answers those typed questions with calibrated
probabilities in one forward pass, without generating text, on your own GPU.

It speaks the `POST /v1/systemone` request and response shapes, so an agent written for TypeSafe's Jev — such as
[jev-ultrafast](https://github.com/browser-use/jev-ultrafast) — can point at `http://127.0.0.1:8009/v1/systemone`
instead of the hosted API: no API key, no queue, no per-call cost.

> **Status: research preview.** Numbers below are on development splits; locked test numbers will be published with
> the released checkpoints. `wev` is an independent project and is not affiliated with TypeSafe AI.

## Why a browser-specific model

General-purpose open decision models do not transfer to browser steps as shipped. Same 586 requests, recorded on
websites none of the models trained on (Mind2Web development split, jev-ultrafast request format):

| model | step success | operation | click target | latency |
|---|---|---|---|---|
| **wev 4B** | **77.3%** | **93.3%** | **80.0%** | 239 ms |
| **wev 1.7B** | **68.3%** | 89.8% | 74.1% | 103 ms |
| kev-8B | 28.8% | 81.9% | 35.2% | 323 ms |
| kev-4B | 27.7% | 42.7% | 51.4% | 296 ms |
| Laya (typed-decisions) | 3.4% | 24.9% | 8.9% | 21 ms |
| Laya | 0.5% | 6.7% | 8.0% | 14 ms |
| *always CLICK / random element* | — | *82.9%* | *≈5%* | — |

*Step success: operation and target both right. Click target: among 8–40 candidate elements. Latency: median on one
RTX 5090 (wev, kev) as served by each project; Laya ran through its Python package. kev and Laya were run as
published, without training on web data; their context budgets (384 and ~320 state tokens) cut most of a page.
Reproduce with `scripts/run_compare.sh`.*

## Quickstart

```bash
pip install "wev-ai[serve] @ git+https://github.com/alanhuangyoo/wev"
```

```python
import wev
m = wev.load("path/to/export")                # exported model directory (Hub repos: coming with the release)
out = m.predict(state, questions)             # {"answers": {...}, "usage": {...}, "latency_ms": ...}
```

```bash
wev serve --model path/to/export --port 8009  # drop-in POST /v1/systemone
```

## How it works

```
<state> page text · elements · recent actions
<q> which operation? <opt> CLICK </opt> <opt> TYPE_TEXT </opt> … <opt> DONE </opt> <decide>
<q> which element to CLICK? <opt> [1] Search </opt> <opt> [2] Sign in </opt> … <decide>
```

- **Backbone**: a Qwen3 base model with its vocabulary head removed (it cannot generate), adapted with LoRA and
  optionally truncated to its first N layers.
- **Mask**: each question sees the state and itself, never another question; each branch restarts its positions
  after the state. One forward pass answers every question, and packing never changes an answer (tested).
- **Readout**: a pointer head scores each option's `</opt>` state against the `<decide>` state, optionally refined by
  a small permutation-equivariant transformer over the option set (`--head set`).

Packing, mask, pointer head and request rendering are adapted from [kev](https://github.com/jaredpalmer/kev)
(Apache-2.0); see [NOTICE](NOTICE).

## Data

Both converters produce requests shaped exactly like jev-ultrafast's per-step request (same state fields, operation
options, instruction text, `[index] label` targets).

| source | license | contributes | split |
|---|---|---|---|
| [Mind2Web](https://huggingface.co/datasets/osunlp/Mind2Web) | CC BY 4.0 | human-recorded CLICK / TYPE / SELECT steps | by website |
| [NNetNav-live](https://huggingface.co/datasets/stanfordnlp/nnetnav-live) | Apache-2.0 | DONE, BLOCKED and SCROLL steps, more CLICK / TYPE | by task |

```bash
python -m wev.mind2web --out data/m2w-v2 --dev_frac 0.15 --test_frac 0.1
python -m wev.nnetnav  --out data/nnetnav-v2
wev train --data data/m2w-v2,data/nnetnav-v2 --base Qwen/Qwen3-1.7B-Base --epochs 1 --lr 1e-4 --out runs/wev-1.7b
wev evaluate --model runs/wev-1.7b --data data/nnetnav-v2 --split dev
wev export --run runs/wev-1.7b --out exports/wev-1.7b --check data/nnetnav-v2/dev.jsonl
pytest -q tests
```

## Limitations

- **Stopping.** DONE is learned from NNetNav; on its development split the model says DONE on about 6% of steps where
  the task was not finished. Long tasks feel this; gate DONE on its probability if early stops are costly.
- **No WAIT data**, and few BLOCKED / SCROLL examples: the model rarely chooses them.
- **Candidates are sampled.** Element accuracy is over 8–40 candidates per step, not every element on the page, so it
  is not comparable to the Mind2Web leaderboard.
- English pages only. TYPE values come from a separate text model, as in jev-ultrafast.
- Not compared with Jev itself (no API access).

## License

Apache-2.0. Includes code adapted from kev (Apache-2.0) and instruction text from jev-ultrafast (MIT); training data
from Mind2Web (CC BY 4.0) and NNetNav (Apache-2.0). See [NOTICE](NOTICE).
