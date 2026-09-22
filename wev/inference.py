"""Load a model and answer System One requests in-process.

    import wev
    m = wev.load("alanhuangyoo/wev-1.7b")          # exported model dir, Hub repo, or a training run dir
    out = m.predict(state, questions)                    # {"answers": ..., "usage": ..., "latency_ms": ...}

Two on-disk formats are understood:
  exported  wev.json + a merged, truncated backbone (config.json, model.safetensors) + head.safetensors
            + tokenizer: self-contained, no base-model download
  run       head.pt + a LoRA adapter over `base` (what wev.train writes)
"""
import copy
import json
import time
from pathlib import Path

import torch

from .api import SystemOneRequest, output_tokens, to_answers, to_record
from .model import MAX_BRANCH, MAX_STATE, ContextTooLong, DecisionModel, encode, load_run

FORMAT = "wev"


class _DropFalseRegexWarning:
    """transformers flags Qwen tokenizers saved to disk with a Mistral regex warning; the saved tokenizer is verified
    identical to the original (same pre-tokenizer, same ids on 715k tokens of real requests), so it is noise."""

    def filter(self, record):
        return "incorrect regex pattern" not in record.getMessage()


def _quiet_tokenizer_warning():
    import logging
    for name in ("transformers.tokenization_utils_base", "transformers.tokenization_utils_fast"):
        logging.getLogger(name).addFilter(_DropFalseRegexWarning())


def default_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def default_dtype(device: str):
    return torch.bfloat16 if device.startswith("cuda") else torch.float32


def _resolve(path_or_repo: str) -> Path:
    p = Path(path_or_repo)
    if p.is_dir():
        return p
    from huggingface_hub import snapshot_download
    return Path(snapshot_download(path_or_repo, allow_patterns=["*.json", "*.safetensors", "*.pt", "*.txt", "*.jinja",
                                                                  "*.model", "*.md"]))


def _load_exported(root: Path, device: str, dtype):
    from safetensors.torch import load_file
    from transformers import AutoModel, AutoTokenizer
    meta = json.loads((root / "wev.json" if (root / "wev.json").exists() else root / "webdecide.json").read_text())
    _quiet_tokenizer_warning()
    attn = "sdpa" if device.startswith("cuda") else "eager"
    tok = AutoTokenizer.from_pretrained(root)
    backbone = AutoModel.from_pretrained(root, dtype=dtype, attn_implementation=attn)
    model = DecisionModel(meta["base"], tok, device, lora=0, dtype=dtype, head_dim=meta["head_dim"],
                          head=meta["head_type"], keep_layers=meta.get("keep_layers"), backbone=backbone)
    model.head.load_state_dict(load_file(str(root / "head.safetensors")))
    model.eval()
    return tok, model, meta


class WebDecide:
    def __init__(self, tok, model, meta, name: str):
        self.tokenizer, self.model, self.meta, self.name = tok, model, meta, name
        self.max_state = meta.get("max_state", MAX_STATE)
        self.max_branch = meta.get("max_branch", MAX_BRANCH)

    def encode(self, req: SystemOneRequest):
        """Encode, shortening state.page.text by 25% per try when the state is too long for the context.
        Raises ContextTooLong when a question itself does not fit or the state has no page text to shorten."""
        req = copy.deepcopy(req)
        for _ in range(12):
            rec, meta = to_record(req)
            try:
                return encode(self.tokenizer, rec, self.max_state, self.max_branch, strict=True), meta
            except ContextTooLong as e:
                page = req.state.get("page") if isinstance(req.state, dict) else None
                if e.part != "state" or not isinstance(page, dict) or not page.get("text"):
                    raise
                page["text"] = page["text"][: int(len(page["text"]) * 0.75)]
        raise ContextTooLong("state", -1, self.max_state)

    def predict(self, state, questions, model: str | None = None) -> dict:
        """Same response shape as POST /v1/systemone."""
        req = SystemOneRequest(state=state, questions=questions, model=model or self.name)
        return self.predict_request(req)

    def predict_request(self, req: SystemOneRequest) -> dict:
        t = time.perf_counter()
        enc, meta = self.encode(req)
        probs = self.model.probs([enc])[0]
        answers = to_answers(probs, meta)
        return {"model": req.model, "answers": answers,
                "usage": {"input_tokens": len(enc["ids"]), "output_tokens": output_tokens(self.tokenizer, answers)},
                "latency_ms": round((time.perf_counter() - t) * 1000)}


def load(path_or_repo: str, device: str | None = None, dtype=None) -> WebDecide:
    device = device or default_device()
    dtype = dtype or default_dtype(device)
    root = _resolve(path_or_repo)
    if (root / "wev.json").exists() or (root / "webdecide.json").exists():   # webdecide.json: pre-rename exports
        tok, model, meta = _load_exported(root, device, dtype)
    elif (root / "head.pt").exists():
        tok, model, meta = load_run(root, device, dtype=dtype)
    else:
        raise FileNotFoundError(f"{path_or_repo}: neither wev.json (exported) nor head.pt (training run)")
    return WebDecide(tok, model, meta, name=meta.get("name", Path(str(path_or_repo)).name))
