"""Generate the Figure 1 overview illustration with an image API (OpenAI-compatible /v1/images/generations).
Credentials come from ~/.wev_img_env (MICU_API_KEY, MICU_BASE_URL); nothing is printed or stored in the repository.

    python paper/icml/figures/gen_overview.py gpt-image-2.5-flare out.png
"""
import base64, json, os, sys, urllib.request
from pathlib import Path

env = dict(l.strip().split("=", 1) for l in open(Path.home() / ".wev_img_env") if "=" in l)
PROMPT = """A clean, publication-quality technical diagram for a machine-learning paper (ICML style), flat vector
illustration, white background, thin dark-gray outlines, rounded rectangles, soft pastel fills (light blue, light
orange, light green), crisp sans-serif labels, generous whitespace, no shadows, no 3D, no people, no logos, no extra
text. Wide landscape layout with two rows, arrows flowing left to right.

TOP ROW, titled exactly "Interface distillation on live websites":
1. A browser window icon labeled exactly "Live website".
2. Arrow to a box labeled exactly "Browser agent".
3. Arrow labeled exactly "typed questions" to a light-orange box labeled exactly "Teacher LLM", with a return arrow
   labeled exactly "probabilities".
4. Arrow down from the teacher to a document-stack icon labeled exactly "Decision log".
5. Arrow to a magnifying-glass / gavel icon labeled exactly "LLM judge", with a small green check mark and red cross.

BOTTOM ROW, titled exactly "Training the student":
6. Three light-blue data boxes in one row, in this exact left-to-right order: "General decisions",
   "Web demonstrations", "Verified episodes". The "Verified episodes" box is the RIGHTMOST box and sits directly
   below the "LLM judge". A short solid arrow goes straight down from "LLM judge" into "Verified episodes", labeled
   exactly "keep verified". No other arrow enters any of the three data boxes.
7. Each of the three data boxes has one arrow down into a single larger light-green box below them, labeled exactly
   "wev decision model", with a small chip icon and the caption inside it exactly "runs locally, no text generation".
8. One dashed arrow leaves the LEFT side of the "wev decision model" box and goes up along the left side of the
   figure into the bottom of the "Browser agent" box, labeled exactly "same interface". It does not touch any data box.

Use only the quoted labels above as text, spelled exactly. Balanced, symmetric, elegant, minimal."""


def main(model, out):
    """Calls the API with curl (the gateway rejects Python's default HTTP client); the key goes through a header file
    readable only by the user, never through the command line."""
    import subprocess, tempfile
    body = {"model": model, "prompt": PROMPT, "size": "1536x1024", "n": 1, "quality": "low" if "2.5" in model else "high"}
    with tempfile.TemporaryDirectory() as tmp:
        hdr, req = Path(tmp) / "h", Path(tmp) / "b.json"
        hdr.write_text("Authorization: Bearer " + env["MICU_API_KEY"] + "\nContent-Type: application/json\n")
        os.chmod(hdr, 0o600)
        req.write_text(json.dumps(body))
        raw = subprocess.run(["curl", "-s", "-m", "600", "-H", f"@{hdr}", "--data-binary", f"@{req}",
                              env["MICU_BASE_URL"].rstrip("/") + "/v1/images/generations"],
                             capture_output=True, check=True).stdout
    d = json.loads(raw)
    if "data" not in d:
        raise SystemExit(f"{model}: {str(d)[:300]}")
    d = d["data"][0]
    if d.get("b64_json"):
        Path(out).write_bytes(base64.b64decode(d["b64_json"]))
    else:
        Path(out).write_bytes(subprocess.run(["curl", "-s", "-m", "300", d["url"]], capture_output=True, check=True).stdout)
    print("wrote", out)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
