"""Local POST /v1/systemone server (TypeSafe-compatible request/response shapes).

    wev serve --model user/wev-1.7b --port 8009

Point a client at http://127.0.0.1:8009/v1/systemone instead of https://api.typesafe.ai/v1/systemone.
Over-long page text is shortened (never the element list) until the request fits the model's context.
"""
import argparse
import logging

import torch
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from .api import SystemOneRequest
from .inference import load
from .model import ContextTooLong

app = FastAPI(title="wev")
S: dict = {}
log = logging.getLogger("wev.serve")


@app.exception_handler(RequestValidationError)
async def invalid_request(request: Request, exc: RequestValidationError):
    """422 with the reason, and the reason logged: a rejected step otherwise looks like a silent model failure."""
    errors = [{"loc": list(e.get("loc", [])), "msg": e.get("msg")} for e in exc.errors()][:5]
    log.warning("rejected request: %s", errors)
    return JSONResponse(status_code=422, content={"detail": errors})


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
        log.warning("request too long: %s", e)
        raise HTTPException(422, f"request too long: {e}") from None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", "--run", dest="model", required=True, help="exported model dir, Hub repo, or run dir")
    ap.add_argument("--port", type=int, default=8009)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--device", default=None, help="cuda | mps | cpu (default: best available)")
    ap.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default=None)
    ap.add_argument("--max_state", type=int, default=0, help="override the model's state token limit")
    ap.add_argument("--max_branch", type=int, default=8192,
                    help="per-question token limit. Training saw up to 2048, but a page with hundreds of targets (an open "
                         "date picker) needs more; answering beyond the training range beats rejecting the step")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}.get(a.dtype)
    S["m"] = load(a.model, device=a.device, dtype=dtype)
    if a.max_state:
        S["m"].max_state = a.max_state
    if a.max_branch:
        S["m"].max_branch = a.max_branch
    import uvicorn
    uvicorn.run(app, host=a.host, port=a.port)


if __name__ == "__main__":
    main()
