"""Export a Tinker LoRA checkpoint for vLLM: PEFT adapter, or merged bf16 weights. Runs on Modal.

Sampler checkpoints (tinker://.../sampler_weights/...) cannot be downloaded; use the *state_path*
(tinker://.../weights/...) from runs/checkpoints.jsonl.

uv run --env-file .env python -m scripts.export_lora --state tinker://RUN/weights/r1-ep0 --name r1
uv run --env-file .env python -m scripts.export_lora --state tinker://RUN/weights/r1-ep0 --name r1 --merge

Needs `modal deploy modal_sampler.py` first. Writes qwen-weights:/adapters/<name> (PEFT) or /merged/<name>.
Serve: ADAPTERS="r1=/vol/adapters/r1" modal deploy modal_sampler.py   (request model="r1")
   or: MERGED_MODEL=/vol/merged/r1 modal deploy modal_sampler.py
(Local download+build was tried: the Tinker download is OK but the 18 GB base-model fetch for the build and a
380 MB upload from a laptop are slow, so the whole thing runs in Modal.)
"""
import argparse
from dotenv import load_dotenv
load_dotenv(".env")
import modal, os

ap = argparse.ArgumentParser()
ap.add_argument("--state", required=True)
ap.add_argument("--name", required=True)
ap.add_argument("--merge", action="store_true", help="merge into full bf16 weights instead of PEFT adapter")
a = ap.parse_args()
modal.Function.from_name(os.environ.get("APP_NAME", "qwen-sampler"), "build").remote(a.name, a.state, a.merge)
print("exported to qwen-weights:/" + ("merged" if a.merge else "adapters") + "/" + a.name)
