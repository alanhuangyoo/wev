---
license: other
license_name: mixed-see-card
language:
- en
pretty_name: wev data
tags:
- browser-agent
- web-navigation
- decision-model
- system-one
- distillation
size_categories:
- 10K<n<100K
configs:
- config_name: mind2web
  data_files:
  - {split: train, path: mind2web/train.jsonl}
  - {split: validation, path: mind2web/validation.jsonl}
  - {split: test, path: mind2web/test.jsonl}
- config_name: nnetnav_audited
  data_files:
  - {split: train, path: nnetnav_audited/train.jsonl}
  - {split: validation, path: nnetnav_audited/validation.jsonl}
  - {split: test, path: nnetnav_audited/test.jsonl}
- config_name: teacher_first
  data_files:
  - {split: train, path: teacher_first/train.jsonl}
  - {split: validation, path: teacher_first/validation.jsonl}
- config_name: teacher_second
  data_files:
  - {split: train, path: teacher_second/train.jsonl}
  - {split: validation, path: teacher_second/validation.jsonl}
- config_name: live_tasks
  data_files:
  - {split: train, path: live_tasks/train.jsonl}
  - {split: test, path: live_tasks/test.jsonl}
---

# wev data

This dataset holds the browser-step data behind the [wev](https://github.com/alanhuangyoo/wev) decision models:
[wev-1.7b](https://huggingface.co/alanhuangya/wev-1.7b), [wev-4b](https://huggingface.co/alanhuangya/wev-4b) and
[wev-8b](https://huggingface.co/alanhuangya/wev-8b). Every row is one browser step, written as a
`POST /v1/systemone` request (a state plus typed questions) with its labelled answers. The requests use exactly the
format the open browser agent [jev-ultrafast](https://github.com/browser-use/jev-ultrafast) sends to its System One,
so a decision model trained here can be served behind that agent unchanged.

| Subset | Rows (train / validation / test) | Source | License |
|---|---|---|---|
| `mind2web` | 5,863 / 586 / 875 | [Mind2Web](https://huggingface.co/datasets/osunlp/Mind2Web), converted | CC BY 4.0 |
| `nnetnav_audited` | 11,408 / 1,140 / 1,150 | [NNetNav-live](https://huggingface.co/datasets/stanfordnlp/nnetnav-live), converted, stopping labels audited | Apache-2.0 |
| `teacher_first` | 2,548 / 199 / – | Episodes of a prompted LLM (qwen3-max) acting as System One on live websites | see *Terms* |
| `teacher_second` | 2,274 / 248 / – | A second collection on the tasks the first had not solved | see *Terms* |
| `live_tasks` | 1,402 / – / 153 | Goals and start URLs for live-website runs; `test` is the held-out end-to-end suite | see *Terms* |

The general typed-decision corpora used in training (Kev decision-v7, typed-decisions, tasksource-jev,
jev-distill-corpus-v3, typed-decisions-synth) are not redistributed here; the repository's builders download and
convert them from their sources.

## Format

```json
{
  "request": {
    "model": "wev-latest",
    "state": {
      "page": {"url": "...", "title": "...", "text": "visible page text"},
      "elements": [{"index": "1", "role": "button", "label": "Search", "operations": ["CLICK"]}],
      "recent_actions": [{"action": "...", "kind": "click", "text": null, "page_changed": true}]
    },
    "questions": {
      "operation": {"type": "choice",
                    "instructions": {"goal": "...", "rules": ["..."]},
                    "criteria": {"CLICK": "...", "TYPE_TEXT": "...", "DONE": "...", "BLOCKED": "..."}},
      "click_target": {"type": "choice",
                       "instructions": {"goal": "...", "operation": "CLICK", "rules": ["..."]},
                       "criteria": {"1": {"element": "[1] Search", "current_value": "", "role": "button"}}}
    }
  },
  "labels": {"operation": "CLICK", "click_target": "1"},
  "_meta": {"source": "..."}
}
```

A step asks for the operation and, for each operation that needs one, a target (`click_target`, `type_text_target`
or `select_target`). `labels` holds the reference option key for each labelled question. A step is answered correctly when the operation
and, if the operation takes one, its target both match. Targets are chosen among the 8–40 candidate elements listed in
the state, not among every element on the page. In teacher rows, `_meta` records the episode, the task and the
teacher's one-sentence reason for its choice.

```python
from datasets import load_dataset
steps = load_dataset("alanhuangya/wev-data", "nnetnav_audited", split="test")
```

To train with the wev package, download the files and pass the subset folders to `wev train --data`;
`validation.jsonl` is read as the development split.

## How the subsets were built

**mind2web.** Each recorded step becomes one request containing the page's visible text, a sample of 8–40
candidate elements that includes the gold element, the task and the preceding actions. Splits are by website, so every
test website is unseen in training.

**nnetnav_audited.** NNetNav supplies the stopping (DONE), giving-up (BLOCKED) and scrolling steps that Mind2Web
lacks. Its trajectories come from unsupervised exploration, so its stopping labels are noisy. An LLM judge
(DeepSeek-V4.1-Flash) read the goal and the state at every step and decided whether the goal was already achieved.
On the training split it rejected 592 of 1,898 DONE labels (31%) and found the goal already achieved at 2,281 of the
10,102 other steps (23%). Those steps were relabelled DONE and the rejected DONE steps were dropped. The validation
and test splits were audited the same way.

**teacher_first, teacher_second.** A prompted LLM (qwen3-max) served as the System One of jev-ultrafast on live
websites. Its prompt forbade signing in, registering, buying, booking, posting, messaging and submitting personal
information, and asked it to answer BLOCKED on CAPTCHAs, unusual-traffic pages and login walls. Every request and the
teacher's choice were logged. An LLM judge then read the final page of each episode, and the kept rows are:

- every decision of an episode whose DONE the judge confirmed;
- every decision up to a BLOCKED on a page that blocked the agent;
- every decision but the last of an episode cut short by the browser harness.

Episodes that looped or exhausted their budget were dropped, as were repeated states. The judge rejected 155 of the
teacher's 431 DONE claims (36%). Train and validation are split by task.

**live_tasks.** Goals paired with start URLs: NNetNav goals on their live sites, Wikipedia look-ups and Google
Flights searches. The 153 `test` tasks are the end-to-end suite used to evaluate the wev models; none of them appears
in the teacher subsets.

## Terms

The Mind2Web-derived subset follows CC BY 4.0 and the NNetNav-derived subset follows Apache-2.0. Attribute the
original datasets (see *Citation*). The teacher subsets and the live tasks contain text captured from public
websites, which remains subject to those sites' terms, and the teacher's choices are outputs of qwen3-max, which are
subject to its provider's terms. Treat these subsets as research data, and check the applicable terms before any
commercial use. The data describes browser steps only and contains no credentials; the teacher was instructed never to
sign in to any site.

## Citation

Jun Huang and Xin Ren contributed equally (University of Electronic Science and Technology of China).

```bibtex
@misc{huang2026wev,
  title  = {wev: Distilling LLM Browser Agents into Open, Local System-One Decision Models},
  author = {Huang, Jun and Ren, Xin},
  year   = {2026},
  url    = {https://github.com/alanhuangyoo/wev}
}
```

If you use the converted subsets, please also cite their sources:

```bibtex
@inproceedings{deng2023mind2web,
  title     = {Mind2Web: Towards a Generalist Agent for the Web},
  author    = {Xiang Deng and Yu Gu and Boyuan Zheng and Shijie Chen and Samuel Stevens and Boshi Wang and Huan Sun and Yu Su},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2023}
}
@misc{murty2024nnetnav,
  title         = {NNetNav: Unsupervised Learning of Browser Agents Through Environment Interaction in the Wild},
  author        = {Shikhar Murty and Hao Zhu and Dzmitry Bahdanau and Christopher D. Manning},
  year          = {2024},
  eprint        = {2410.02907},
  archivePrefix = {arXiv}
}
```
