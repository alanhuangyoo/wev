"""System One request/response shapes (POST /v1/systemone) mapped onto one pointer primitive.

Adapted from kev (https://github.com/jaredpalmer/kev, Apache-2.0), kev/api.py.

Noul   -> 2 options [no, yes];                answer = p(yes)
Choice -> options 'name' or 'name: desc';      answer = argmax, probabilities by name, confidence
Score  -> options = ordered level descriptions; answer = expected level, legend, probabilities by index

Training data and live requests both go through to_record(), so the model never sees a rendering at inference
that it did not see in training.
"""
import json
from typing import Any, Literal, Union

from pydantic import BaseModel, Field, model_validator

JSONContent = Union[str, dict, list, int, float, bool, None]
MAX_OPTIONS = 255


class Noul(BaseModel):
    type: Literal["noul"]
    instructions: JSONContent
    criteria: dict[str, JSONContent] | None = None


class Choice(BaseModel):
    type: Literal["choice"]
    instructions: JSONContent
    criteria: dict[str, JSONContent]

    @model_validator(mode="after")
    def _check(self):
        if not 1 <= len(self.criteria) <= MAX_OPTIONS:
            raise ValueError(f"criteria must have 1..{MAX_OPTIONS} options")
        return self


class Score(BaseModel):
    type: Literal["score"]
    instructions: JSONContent
    criteria: list[JSONContent] = Field(min_length=2, max_length=MAX_OPTIONS)


Question = Union[Noul, Choice, Score]


class SystemOneRequest(BaseModel):
    state: JSONContent
    model: str = "wev-latest"
    questions: dict[str, Question] = Field(min_length=1)


def render(v: JSONContent, indent: int = 0) -> str:
    """Flatten str | object | array into the text the model sees. Field names are kept as labels."""
    pad = "  " * indent
    if v is None:
        return ""
    if isinstance(v, (str, int, float, bool)):
        return str(v)
    if isinstance(v, list):
        return "\n".join(f"{pad}- {render(x, indent + 1).lstrip()}" for x in v)
    return "\n".join(
        f"{pad}{k}:\n{render(x, indent + 1)}" if isinstance(x, (dict, list)) else f"{pad}{k}: {render(x)}"
        for k, x in v.items()
    )


def option_text(name: str, desc: JSONContent) -> str:
    return name if desc is None or desc == "" else f"{name}: {render(desc)}"


def to_record(req: SystemOneRequest):
    """-> (record for encode(), per-question metadata to map probabilities back to keys)."""
    qs, meta = [], []
    for qid, q in req.questions.items():
        instr = render(q.instructions)
        if q.type == "noul":
            c = q.criteria or {}
            opts = [option_text("no", c.get("false")), option_text("yes", c.get("true"))]
            meta.append({"id": qid, "type": "noul"})
        elif q.type == "choice":
            opts = [option_text(k, v) for k, v in q.criteria.items()]
            meta.append({"id": qid, "type": "choice", "keys": list(q.criteria.keys())})
        else:
            opts = [render(x) for x in q.criteria]
            meta.append({"id": qid, "type": "score", "legend": {str(i): render(x) for i, x in enumerate(q.criteria)}})
        qs.append({"instr": instr, "options": opts})
    return {"state": render(req.state), "questions": qs}, meta


def choice_confidence(p: list[float]) -> float:
    k = len(p)
    return 1.0 if k == 1 else (max(p) - 1 / k) / (1 - 1 / k)


def score_confidence(p: list[float]) -> float:
    """1 - E|level - mode| / (L - 1); TypeSafe's exact formula is unpublished."""
    n = len(p)
    mode = max(range(n), key=lambda i: p[i])
    return 1.0 - sum(pi * abs(i - mode) for i, pi in enumerate(p)) / (n - 1)


def _r(x: float, digits: int) -> float:
    return round(float(x), digits)


def to_answers(probs: list[list[float]], meta: list[dict], digits: int = 4) -> dict[str, Any]:
    """Probabilities are rounded to `digits` (4, not 2): jev-ultrafast rejects a Choice whose probabilities do not sum
    to 1 within 0.02, and 2-digit rounding over 40+ options can miss that bound."""
    out = {}
    for p, m in zip(probs, meta):
        if m["type"] == "noul":
            out[m["id"]] = {"type": "noul", "noul": _r(p[1], digits)}
        elif m["type"] == "choice":
            best = max(range(len(p)), key=lambda i: p[i])
            dist = {k: _r(v, digits) for k, v in zip(m["keys"], p)}
            out[m["id"]] = {"type": "choice", "choice": m["keys"][best],
                            "confidence": _r(choice_confidence(p), digits), "probabilities": dist}
        else:
            score = sum(i * pi for i, pi in enumerate(p))
            out[m["id"]] = {"type": "score", "score": _r(score, digits), "legend": m["legend"],
                            "probabilities": {str(i): _r(v, digits) for i, v in enumerate(p)},
                            "confidence": _r(score_confidence(p), digits)}
    return out


def output_tokens(tok, answers: dict) -> int:
    """Billing-style figure: tokens of the serialised answers. Nothing is generated."""
    return len(tok(json.dumps(answers), add_special_tokens=False).input_ids)
