# wev

**A local decision model. Typed questions in, calibrated probabilities out, in one forward pass.**

`wev` answers the `POST /v1/systemone` request shape: a free-form state plus any number of questions, each a choice,
a yes/no (`noul`) or a score. It covers general decisions (triage, routing, policy checks, agent monitoring) and
browser-agent steps (*which operation? which element?*). It runs on your own GPU. You need no API key, pay nothing per
call, and it generates no text.

| model | size | for |
|---|---|---|
| [wev-1.7b](https://huggingface.co/alanhuangya/wev-1.7b) | 3.5 GB | laptops and small GPUs; the fastest |
| [**wev-4b**](https://huggingface.co/alanhuangya/wev-4b) | 8.1 GB | the default; one consumer GPU |
| [wev-8b](https://huggingface.co/alanhuangya/wev-8b) | 15.2 GB | the most accurate, especially out of domain |

> `wev` is an independent project. It is not affiliated with TypeSafe AI and does not use Jev.

## Quickstart

```bash
pip install "wev-ai[serve]"
```

```python
import wev

m = wev.load("alanhuangya/wev-4b")   # downloads once from the Hugging Face Hub
out = m.predict(
    state="Refund request: order #4411 arrived damaged, customer attached photos, first refund this year.",
    questions={
        "action": {"type": "choice", "instructions": "What should support do?",
                   "criteria": {"refund": "Refund the order.", "replace": "Ship a replacement.",
                                "escalate": "Send to a human agent."}},
        "fraud_risk": {"type": "noul", "instructions": "This request looks fraudulent.",
                       "criteria": {"true": "Likely fraud.", "false": "No sign of fraud."}},
    },
)
out["answers"]
# {'action': {'type': 'choice', 'choice': 'replace', 'probabilities': {'refund': 0.42, 'replace': 0.53, ...}},
#  'fraud_risk': {'type': 'noul', 'noul': 0.04}}
```

Serve it over HTTP, with the same request and response shapes as a System One API:

```bash
wev serve --model alanhuangya/wev-4b --port 8009
curl -s localhost:8009/v1/systemone -H 'content-type: application/json' -d '{"state": "...", "questions": {...}}'
```

A browser agent written for a System One API, such as
[jev-ultrafast](https://github.com/browser-use/jev-ultrafast), can call `http://127.0.0.1:8009/v1/systemone`
instead of the hosted endpoint. jev-ultrafast hard-codes that URL in `jev_ultrafast/model.py` (`SYSTEM_ONE_URL`), so
change that one line.

## Results

These numbers come from one read of each locked test split. Every other model was run on the same requests and scored
the same way: per-question accuracy, with each model's top option taken as its answer. Raw results are in
[`results/`](results); `scripts/compare.py` runs any System One server or Laya against a split.

**General typed decisions**

| model | kev decision-v7 | kev transfer-v4 | typed-decisions |
|---|---|---|---|
| **wev-8b** | 82.4 | 77.2 | 79.1 |
| **wev-4b** | 80.6 | 73.3 | **79.6** |
| **wev-1.7b** | 81.1 | 65.5 | 79.5 |
| Kev-4B | **88.2** | **82.1** | 65.1 |
| Kev-8B | 88.1 | 76.8 | 62.7 |
| Laya (typed-decisions) | 65.7 | 62.8 | 76.8 |
| Laya | 64.3 | 63.7 | 36.2 |

- *kev decision-v7* is [Kev](https://github.com/jaredpalmer/kev)'s own training suite. `wev` trains on its train
  split too, but it shares its capacity with browser data, and Kev leads here by 6–8 points.
- *kev transfer-v4* is out-of-domain for every model in the table. wev-8b is level with Kev-8B there, and Kev-4B leads.
- *[typed-decisions](https://huggingface.co/datasets/LocalLLaMA/typed-decisions)* covers four agent and ops workflows
  with five questions per case. `wev` trains on 80% of its train split, as the Laya (typed-decisions) specialist does.
  Kev and plain Laya do not, so they are generalists on this column.

**Browser steps.** These are single steps from the Mind2Web test split: 873 requests on websites unseen in training,
in jev-ultrafast's request format. A step succeeds when both the operation and the target element are right.

| model | step success | operation |
|---|---|---|
| **wev-8b** | **75.5** | **90.3** |
| **wev-4b** | 75.4 | 90.0 |
| **wev-1.7b** | 68.2 | 88.1 |
| Kev-4B | 21.2 | 35.7 |
| Kev-8B | 19.0 | 73.3 |
| Laya (typed-decisions) | 0.7 | 13.1 |
| Laya | 0.0 | 2.5 |

`wev` is evaluated with a 4096-token state, and 11 of the requests exceed it. Those 11 count as wrong for `wev`. Kev
and Laya were run as published, without web training data, and their context budgets cut most of a page.

**End to end on live websites.** jev-ultrafast ran 153 held-out tasks with each model as its System One. A task counts
as a success when the agent says DONE and an LLM judge, reading the final page, agrees.

| System One | tasks completed |
|---|---|
| **wev-8b** | **28 / 153 (18.3%)** |
| wev-4b | 21 / 153 (13.7%) |
| qwen3-max, prompted (the teacher) | 27 / 153 (17.6%) |

Live sites differ from run to run, so treat gaps of a few tasks as noise. Google Flights often answers automated
browsing with a CAPTCHA, and every model fails most of those tasks.

**Speed.** These are median in-process latencies on one RTX 5090 in bf16, with a warm model, one request at a time.
Every question in a request is answered in the same pass.

| model | kev decision-v7 (~180 tokens, 1 question) | typed-decisions (~380 tokens, 5 questions) | browser step (~3,200 tokens, 2 questions) |
|---|---|---|---|
| wev-1.7b | 10 ms | 13 ms | 91 ms |
| wev-4b | 15 ms | 25 ms | 217 ms |
| wev-8b | 21 ms | 37 ms | 322 ms |

Reproduce with `scripts/bench_latency.py --model <export> --data <jsonl>...`.

## How it works

```
<state> page text · elements · recent actions
<q> which operation? <opt> CLICK </opt> <opt> TYPE_TEXT </opt> … <opt> DONE </opt> <decide>
<q> which element to CLICK? <opt> [1] Search </opt> <opt> [2] Sign in </opt> … <decide>
```

- **Backbone.** A Qwen3 base model with its vocabulary head removed, so it cannot generate text. It is adapted with
  LoRA (r=16, every attention and MLP projection), and the adapter is merged into the released weights.
- **Mask.** Each question sees the state and itself, never another question, and each question's positions restart
  after the state. One pass answers every question in a request, and question order never changes an answer (tested).
- **Readout.** A pointer head scores each option's `</opt>` hidden state against the question's `<decide>` state and
  normalises the scores into probabilities.
- **Options in the code, not in the release.** `--head set` adds a small permutation-equivariant transformer over the
  option set, and `--keep_layers N` truncates the backbone. On 1.7B, keeping 18 of 28 layers ran 23% faster and cost
  about 3 points of accuracy. The set head did not help at full depth, so the released models use the pointer head
  and every layer.

The packing, mask, pointer head and request rendering are adapted from [kev](https://github.com/jaredpalmer/kev)
(Apache-2.0). See [NOTICE](NOTICE).

## Training data

| source | license | what it adds |
|---|---|---|
| [Mind2Web](https://huggingface.co/datasets/osunlp/Mind2Web) | CC BY 4.0 | human browser steps (click, type, select), split by website |
| [NNetNav-live](https://huggingface.co/datasets/stanfordnlp/nnetnav-live) | Apache-2.0 | live-web steps; DONE relabelled by an LLM judge (about a third of the original DONE labels were wrong) |
| teacher episodes | outputs of qwen3-max | jev-ultrafast on live sites with qwen3-max as System One; only judge-verified successes are kept |
| [kev decision-v7](https://github.com/jaredpalmer/kev) | per source | ten public classification and QA sources plus rule-composition records |
| [typed-decisions](https://huggingface.co/datasets/LocalLLaMA/typed-decisions) | Apache-2.0 | agent and ops workflows (80% of its train split) |
| [tasksource-jev](https://huggingface.co/datasets/tasksource/tasksource-jev) | mixed, per source task (some research-only) | hundreds of classification tasks recast as decisions |
| [jev-distill-corpus-v3](https://huggingface.co/datasets/SargeDev/jev-distill-corpus-v3) | Apache-2.0 | synthetic operational scenarios with soft labels |
| [typed-decisions-synth](https://huggingface.co/datasets/n4ze3m/typed-decisions-synth) | MIT | multi-question cases over 149 domains |

Several sources come with their own terms: research-only tasks in tasksource-jev, and the model-output terms of the
teacher. Check them before you use the models commercially.

## Train your own

```bash
pip install -e ".[train,serve,dev]"

# data
python -m wev.mind2web --out data/m2w-v2 --dev_frac 0.15 --test_frac 0.1
python -m wev.nnetnav  --out data/nnetnav-v2
python -m wev.general  --out data/general --kev_repo path/to/kev
python -m wev.external --out data/ext
# DONE relabelling of NNetNav by an LLM judge (OpenAI-compatible endpoint from WEV_JUDGE_BASE_URL / _API_KEY / _MODEL)
python scripts/judge_done.py --data data/nnetnav-v2/train.jsonl --out judge-train.jsonl
python scripts/apply_judge.py --data data/nnetnav-v2 --judge 'judge-{split}.jsonl' --out data/nnetnav-v3-clean

# the wev-4b recipe; dir:K repeats a training file K times. One GPU works too: drop torchrun.
torchrun --nproc_per_node 8 -m wev.train --base Qwen/Qwen3-4B-Base --head pointer --lr 1e-4 --epochs 1 \
  --batch_tokens 6000 --accum 1 --checkpointing 0 --out runs/wev-4b \
  --data data/m2w-v2,data/nnetnav-v3-clean,data/teacher-v1:3,data/general/kev-v7:2,data/general/typed-decisions-train:8,data/ext/tasksource-jev,data/ext/jev-distill,data/ext/td-synth

wev evaluate --model runs/wev-4b --data data/general/kev-transfer-v4 --split dev
wev export --run runs/wev-4b --out exports/wev-4b --check data/general/typed-decisions/dev.jsonl
pytest -q tests
```

To collect teacher episodes, use `scripts/teacher_server.py`, `make_tasks.py`, `collect.py` and
`build_teacher_data.py`: an LLM serves as a System One server behind jev-ultrafast on live sites.

## Limitations

- English only. `wev` makes decisions and does not write text; in jev-ultrafast, typed values come from its separate
  text model.
- Browser targets are scored among the candidates the agent lists (8–40 per step), not against every element on the
  page. These numbers are not comparable to the Mind2Web leaderboard.
- DONE and BLOCKED are the hardest browser operations. On NNetNav's test split, wev-4b says DONE too early on 8% of
  steps. If an early stop is costly, act on DONE only above a probability threshold.
- On Kev's own suite, Kev is more accurate.
- `wev` has not been compared with Jev itself, because we have no API access.

## License

Apache-2.0. The repository includes code adapted from kev (Apache-2.0) and instruction text from jev-ultrafast (MIT).
The models build on Qwen3 base models (Apache-2.0) and use the datasets listed above. See [NOTICE](NOTICE).
