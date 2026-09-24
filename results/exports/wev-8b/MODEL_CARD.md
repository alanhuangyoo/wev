---
license: apache-2.0
base_model: Qwen/Qwen3-8B-Base
language:
- en
tags:
- decision-model
- browser-agent
- system-one
- lora
datasets:
- osunlp/Mind2Web
- stanfordnlp/nnetnav-live
- LocalLLaMA/typed-decisions
- tasksource/tasksource-jev
- SargeDev/jev-distill-corpus-v3
- n4ze3m/typed-decisions-synth
---

# wev-8b

**A local decision model: typed questions in, calibrated probabilities out, in one forward pass.** `wev-8b` answers
the `POST /v1/systemone` request shape (choice, yes/no and score questions over a free-form state), for general
decisions and for browser-agent steps (*which operation? which element?*). It runs on your own GPU: no API key,
no per-call cost, nothing generated.

Code, training and evaluation: [alanhuangyoo/wev](https://github.com/alanhuangyoo/wev). Paper:
[doi:10.5281/zenodo.22941164](https://zenodo.org/records/22941164). Independent project; not affiliated
with TypeSafe AI.

## Quickstart

```bash
pip install "wev-ai[serve]"
```

```python
import wev
m = wev.load("alanhuangya/wev-8b")
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
print(out["answers"])
```

```bash
wev serve --model alanhuangya/wev-8b --port 8009   # drop-in POST /v1/systemone, e.g. for jev-ultrafast
```

## Results

Test splits, held out from training; every other model was run on the same requests and scored the same way
(per-question accuracy; `scripts/compare.py`). For wev-4b and wev-8b, two candidates each were read on test (see the
repository README).

**General typed decisions**

| model | kev decision-v7 | kev transfer-v4 | typed-decisions |
|---|---|---|---|
| **wev-8b** | 82.4 | 77.2 | **79.1** |
| Kev-4B | **88.2** | **82.1** | 65.1 |
| Kev-8B | 88.1 | 76.8 | 62.7 |
| Laya (typed-decisions) | 65.7 | 62.8 | 76.8 |
| Laya | 64.3 | 63.7 | 36.2 |

`wev-8b` trains on 80% of the typed-decisions train split, like the Laya (typed-decisions) specialist; Kev and Laya
do not, so on that column they are generalists. kev decision-v7 is Kev's own training suite (`wev-8b` also trains on
its train split); transfer-v4 is out-of-domain for every model here.

**Browser steps** (Mind2Web test split: websites unseen in training, jev-ultrafast request format; step success =
operation and target element both right)

| model | step success | operation |
|---|---|---|
| **wev-8b** | **75.5** | **90.3** |
| Kev-4B | 21.2 | 35.7 |
| Kev-8B | 19.0 | 73.3 |
| Laya (typed-decisions) | 0.7 | 13.1 |
| Laya | 0.0 | 2.5 |

873 requests; 11 exceed the context `wev-8b` is evaluated with and count as wrong for it.

NNetNav test split (live-web steps, DONE judged by an LLM): step success 60.8, DONE recall
79.6, premature DONE 8.2.

**End to end** (153 held-out tasks on live websites, run by [jev-ultrafast](https://github.com/browser-use/jev-ultrafast)
with `wev-8b` as its System One; success = the agent says DONE and an LLM judge reading the final page agrees):
28/153 (18.3%) tasks, vs 27/153 (17.6%) for the qwen3-max teacher behind the same agent.
Live sites differ from run to run; treat gaps of a few tasks as noise.

## Model

- Backbone: `Qwen/Qwen3-8B-Base` without its vocabulary head, LoRA r=16 on every attention and MLP projection, merged
  into the weights of this export; 36 layers, bf16.
- Readout: a pointer head scores each option's `</opt>` state against the question's `<decide>` state.
- Each question sees the state and itself only (block-causal branches, positions restart after the state), so a
  request with many questions costs one pass and answers never depend on question order.
- Context: state up to 4096 tokens, each question up to 8192 tokens (trained with
  2048); longer page states are shrunk before encoding.

## Training

1 epoch, lr 0.0001, one-cycle schedule, soft-label cross-entropy where the source has soft labels.
Recipe and data builders: [alanhuangyoo/wev](https://github.com/alanhuangyoo/wev).

| source | license | what it adds |
|---|---|---|
| [Mind2Web](https://huggingface.co/datasets/osunlp/Mind2Web) | CC BY 4.0 | human browser steps: click, type, select |
| [NNetNav-live](https://huggingface.co/datasets/stanfordnlp/nnetnav-live) | Apache-2.0 | live-web steps; DONE relabelled by an LLM judge |
| teacher episodes | outputs of qwen3-max | jev-ultrafast on live sites with qwen3-max as System One, success judge-verified |
| [kev decision-v7](https://github.com/jaredpalmer/kev) | per source | ten public classification / QA sources plus rule records |
| [typed-decisions](https://huggingface.co/datasets/LocalLLaMA/typed-decisions) | Apache-2.0 | agent / ops workflows, 5 questions per case (80% of train) |
| [tasksource-jev](https://huggingface.co/datasets/tasksource/tasksource-jev) | mixed (per source task; some research-only) | hundreds of classification tasks as decisions |
| [jev-distill-corpus-v3](https://huggingface.co/datasets/SargeDev/jev-distill-corpus-v3) | Apache-2.0 | synthetic operational scenarios, soft labels |
| [typed-decisions-synth](https://huggingface.co/datasets/n4ze3m/typed-decisions-synth) | MIT | multi-question cases over 149 domains |

**Use terms.** Some training data carries its own terms: several tasksource-jev source tasks are research-only, and the
teacher episodes are qwen3-max outputs subject to its provider's terms. Treat this model as a research artifact and
check those terms before any commercial use.

## Limitations

- English only. Decisions, not text: TYPE values come from a separate text model, as in jev-ultrafast.
- Browser targets are scored among the candidates the agent lists (8–40 per step), not every element on the page.
- DONE and BLOCKED are the hardest operations; gate DONE on its probability when early stops are costly.
- Not compared with Jev itself (no API access).

## Citation

Jun Huang and Xin Ren contributed equally (University of Electronic Science and Technology of China).

```bibtex
@misc{huang2026wev,
  title     = {wev: Distilling LLM Browser Agents into Open, Local System-One Decision Models},
  author    = {Huang, Jun and Ren, Xin},
  year      = {2026},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.22941164},
  url       = {https://doi.org/10.5281/zenodo.22941164}
}
```

## License

Apache-2.0, like the base model. Architecture code adapted from [kev](https://github.com/jaredpalmer/kev) (Apache-2.0).
