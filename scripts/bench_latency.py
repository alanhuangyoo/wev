"""In-process latency on real requests: wev.load(model).predict(...), warm-up excluded, CUDA synchronised."""
import argparse
import json
import statistics
import time

import torch

import wev
from wev.model import ContextTooLong


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True, nargs="+")
    ap.add_argument("--n", type=int, default=200, help="requests per data file")
    a = ap.parse_args()
    m = wev.load(a.model)
    cuda = str(m.model.device).startswith("cuda")
    out = {"model": a.model, "device": str(m.model.device),
           "gpu": torch.cuda.get_device_name(0) if cuda else None, "layers": m.meta.get("num_layers")}
    for path in a.data:
        rows = [json.loads(l) for _, l in zip(range(a.n + 5), open(path))]
        times, tokens, too_long = [], [], 0
        for i, r in enumerate(rows):
            if cuda:
                torch.cuda.synchronize()
            t = time.perf_counter()
            try:
                res = m.predict(r["request"]["state"], r["request"]["questions"])
            except ContextTooLong:   # the server answers these with 422
                too_long += 1
                continue
            if cuda:
                torch.cuda.synchronize()
            if i >= 5:
                times.append((time.perf_counter() - t) * 1000)
                tokens.append(res["usage"]["input_tokens"])
        out[path] = {"n": len(times), "median_ms": round(statistics.median(times), 1),
                     "p90_ms": round(sorted(times)[int(0.9 * (len(times) - 1))], 1),
                     "median_input_tokens": int(statistics.median(tokens)), "too_long": too_long}
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
