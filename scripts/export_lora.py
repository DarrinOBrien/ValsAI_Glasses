"""Export a Tinker LoRA checkpoint for vLLM: PEFT adapter, or merged bf16 weights.

Sampler checkpoints (tinker://.../sampler_weights/...) cannot be downloaded; use the *state_path*
(tinker://.../weights/...) from runs/checkpoints.jsonl.

uv run --env-file .env python -m scripts.export_lora --state tinker://RUN/weights/r1-ep0 --name r1
uv run --env-file .env python -m scripts.export_lora --state tinker://RUN/weights/r1-ep0 --name r1 --merge

Steps: (1) download the raw Tinker adapter (~380 MB at rank 32) to exports/<name>/raw, (2) upload it to the
qwen-weights volume at /raw/<name>, (3) run modal_sampler.build next to the base weights, which writes
/adapters/<name> (PEFT) or /merged/<name>. (Building locally re-downloads the 18 GB base model from the hub.)
Serve: ADAPTERS="r1=/vol/adapters/r1" modal deploy modal_sampler.py   (request model="r1")
   or: MERGED_MODEL=/vol/merged/r1 modal deploy modal_sampler.py
"""
import argparse, os
from dotenv import load_dotenv
load_dotenv(".env")

ap = argparse.ArgumentParser()
ap.add_argument("--state", required=True)
ap.add_argument("--name", required=True)
ap.add_argument("--merge", action="store_true", help="merge into full bf16 weights instead of PEFT adapter")
a = ap.parse_args()

import modal
from tinker_cookbook import weights

raw = f"exports/{a.name}/raw"
if not os.path.exists(f"{raw}/adapter_model.safetensors"):
    os.makedirs(raw, exist_ok=True)
    weights.download(tinker_path=a.state, output_dir=raw)
vol = modal.Volume.from_name("qwen-weights", create_if_missing=True)
with vol.batch_upload(force=True) as b:
    b.put_directory(raw, f"/raw/{a.name}")
fn = modal.Function.from_name("qwen-sampler", "build")
fn.remote(a.name, a.merge)
print("exported to qwen-weights:/" + ("merged" if a.merge else "adapters") + "/" + a.name)
