"""Local POST /v1/systemone server (TypeSafe-compatible request/response shapes).

    wev serve --model user/wev-1.7b --port 8009

Point a client at http://127.0.0.1:8009/v1/systemone instead of https://api.typesafe.ai/v1/systemone.
Over-long page text is shortened (never the element list) until the request fits the model's context.
"""
import argparse

import torch
from fastapi import FastAPI, HTTPException

from .api import SystemOneRequest
from .inference import load
from .model import ContextTooLong

app = FastAPI(title="wev")
S: dict = {}


@app.get("/v1/models")
def models():
    m = S["m"]
    return {"data": [{"id": m.name, "base": m.meta.get("base"), "layers": m.meta.get("num_layers"),
                      "head": m.meta.get("head_type")}]}


@app.post("/v1/systemone")
def systemone(req: SystemOneRequest):
    try:
        return S["m"].predict_request(req)
    except ContextTooLong as e:
        raise HTTPException(422, f"request too long: {e}") from None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", "--run", dest="model", required=True, help="exported model dir, Hub repo, or run dir")
    ap.add_argument("--port", type=int, default=8009)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--device", default=None, help="cuda | mps | cpu (default: best available)")
    ap.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default=None)
    a = ap.parse_args()
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}.get(a.dtype)
    S["m"] = load(a.model, device=a.device, dtype=dtype)
    import uvicorn
    uvicorn.run(app, host=a.host, port=a.port)


if __name__ == "__main__":
    main()
