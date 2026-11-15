"""Export a Tinker LoRA checkpoint for vLLM (PEFT adapter) or merge it into full HF weights.

Sampler checkpoints (tinker://.../sampler_weights/...) cannot be downloaded; use the *state_path*
(tinker://.../weights/...) from runs/checkpoints.jsonl.

uv run python -m scripts.export_lora --state tinker://RUN/weights/r1-ep0 --name r1            # PEFT adapter
uv run python -m scripts.export_lora --state tinker://RUN/weights/r1-ep0 --name r1 --merge    # merged model
Outputs under exports/<name>/{raw,adapter|merged}. Upload to the Modal volume:
  uv run modal volume put qwen-weights exports/r1/adapter /adapters/r1
  uv run modal volume put qwen-weights exports/r1/merged /merged/r1
Serve: ADAPTERS="r1=/vol/adapters/r1" modal deploy modal_sampler.py   (then model="r1" in requests)
   or: MERGED_MODEL=/vol/merged/r1 modal deploy modal_sampler.py
"""
import argparse, os, shutil
from dotenv import load_dotenv
load_dotenv(".env")
from tinker_cookbook import weights

BASE = "Qwen/Qwen3.5-9B"

ap = argparse.ArgumentParser()
ap.add_argument("--state", required=True)
ap.add_argument("--name", required=True)
ap.add_argument("--merge", action="store_true", help="merge into full bf16 weights instead of PEFT adapter")
ap.add_argument("--keep-raw", action="store_true")
a = ap.parse_args()

root = f"exports/{a.name}"
raw = f"{root}/raw"
if not os.path.exists(raw):
    os.makedirs(root, exist_ok=True)
    weights.download(tinker_path=a.state, output_dir=raw)
out = f"{root}/{'merged' if a.merge else 'adapter'}"
if os.path.exists(out):
    shutil.rmtree(out)
if a.merge:
    weights.build_hf_model(base_model=BASE, adapter_path=raw, output_path=out)
else:
    weights.build_lora_adapter(base_model=BASE, adapter_path=raw, output_path=out)
print("wrote", out)
