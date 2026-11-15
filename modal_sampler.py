"""Qwen3.5-9B vLLM server on Modal (OpenAI-compatible), fanned out over N H100 replicas.

uv run modal run modal_sampler.py::download            # weights -> volume qwen-weights (CPU only)
uv run modal deploy modal_sampler.py                   # server (scales 0 -> N_REPLICAS on demand)
uv run modal run modal_sampler.py::bench --n 64        # measure real tokens/s
uv run modal app stop qwen-sampler                     # STOP PAYING for GPUs

Env at deploy time: N_REPLICAS (default 3), ADAPTERS="name=/vol/adapters/r1,name2=..." (vLLM LoRA modules),
MERGED_MODEL=/vol/merged/r1 (serve a merged checkpoint instead of base).
Native chat template, thinking on (default for Qwen3.5), reasoning parser splits reasoning_content.
Auth: Bearer VLLM_KEY.
"""
import hashlib, os
import modal

APP_NAME = os.environ.get("APP_NAME", "qwen-sampler")  # use another name for experiments so the main deployment is untouched
app = modal.App(APP_NAME)
HF_MODEL = "Qwen/Qwen3.5-9B"
VOL_MODEL = "/vol/qwen3.5-9b"
MODEL = os.environ.get("MERGED_MODEL") or VOL_MODEL
N_REPLICAS = int(os.environ.get("N_REPLICAS", "3"))
MAX_LEN = int(os.environ.get("MAX_LEN", "65536"))
ADAPTERS = os.environ.get("ADAPTERS", "")  # "name=/vol/adapters/x,..."
vol = modal.Volume.from_name("qwen-weights", create_if_missing=True)
key = modal.Secret.from_dict({"VLLM_KEY": os.environ.get("VLLM_KEY") or hashlib.sha256(
    (os.environ.get("MODAL_TOKEN_SECRET", "") + "vllm").encode()).hexdigest()[:32]})

image = (
    modal.Image.from_registry("nvidia/cuda:12.8.0-devel-ubuntu22.04", add_python="3.12")
    .entrypoint([])
    .uv_pip_install("vllm", "huggingface-hub", "hf_transfer", "httpx")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "ADAPTERS": ADAPTERS, "MERGED_MODEL": os.environ.get("MERGED_MODEL", ""),
          "MAX_LEN": str(MAX_LEN)})  # deploy-time settings must be baked in; the container does not see local env
)


@app.function(image=image, volumes={"/vol": vol}, timeout=3600, cpu=8,
              secrets=[modal.Secret.from_dict({"HF_TOKEN": os.environ.get("HF_ACCESS_TOKEN", "")})])
def download():
    import os.path
    from huggingface_hub import snapshot_download
    if os.path.exists(VOL_MODEL + "/config.json"):
        print("already present")
        return
    snapshot_download(HF_MODEL, local_dir=VOL_MODEL)
    vol.commit()


@app.function(image=image, gpu="H100", volumes={"/vol": vol}, secrets=[key], timeout=12 * 3600,
              scaledown_window=300, max_containers=N_REPLICAS)
@modal.concurrent(max_inputs=96)
@modal.web_server(port=8000, startup_timeout=20 * 60)
def serve():
    import subprocess
    vol.reload()
    cmd = ["vllm", "serve", MODEL, "--served-model-name", "qwen", "--host", "0.0.0.0", "--port", "8000",
           "--max-model-len", str(MAX_LEN), "--gpu-memory-utilization", "0.92",
           "--reasoning-parser", "qwen3", "--enable-auto-tool-choice", "--tool-call-parser", "qwen3_coder",
           "--enable-prefix-caching", "--max-num-seqs", "96", "--api-key", os.environ["VLLM_KEY"]]
    if ADAPTERS:
        cmd += ["--enable-lora", "--max-lora-rank", "32", "--lora-modules", *ADAPTERS.split(",")]
    subprocess.Popen(cmd)


@app.function(image=image, secrets=[key], timeout=1800, cpu=2)
def bench(n: int = 64, max_tokens: int = 4096):
    """Fire n concurrent long generations at the deployed server; report aggregate tokens/s."""
    import asyncio, time, httpx
    url = serve.get_web_url()
    prompts = [f"Solve Codeforces-style problem #{i}: given n up to 2e5 and an array, count pairs (i<j) with "
               f"a_i xor a_j > k. Think carefully, then write C++20." for i in range(n)]

    async def one(c, p):
        r = await c.post(url + "/v1/chat/completions", json={
            "model": "qwen", "messages": [{"role": "user", "content": p}], "max_tokens": max_tokens,
            "temperature": 0.8, "top_p": 0.95})
        if r.status_code != 200:
            print("ERR", r.status_code, r.text[:200])
            return 0
        return r.json()["usage"]["completion_tokens"]

    async def main():
        async with httpx.AsyncClient(timeout=1800, headers={"Authorization": f"Bearer {os.environ['VLLM_KEY']}"}) as c:
            for _ in range(120):  # wait for a replica to boot (cold start ~3-6 min)
                try:
                    if (await c.get(url + "/v1/models")).status_code == 200:
                        break
                except Exception:
                    pass
                await asyncio.sleep(10)
            t = time.time()
            toks = await asyncio.gather(*[one(c, p) for p in prompts])
            dt = time.time() - t
            print(f"{sum(toks)} tokens in {dt:.1f}s = {sum(toks)/dt:.0f} tok/s aggregate (n={n}, replicas<= {N_REPLICAS})")
    asyncio.run(main())


@app.function(image=image, secrets=[key], timeout=600, cpu=1)
def url() -> str:
    return serve.get_web_url()


export_image = (modal.Image.debian_slim(python_version="3.12")
                .pip_install("tinker-cookbook", "transformers", "torch", "safetensors", "huggingface_hub"))


@app.function(image=export_image, volumes={"/vol": vol}, cpu=8, memory=65536, timeout=3600,
              secrets=[modal.Secret.from_dict({"TINKER_API_KEY": os.environ.get("TINKER_API_KEY", "")})])
def build(name: str, state: str, merge: bool = False):
    """Tinker state checkpoint -> /vol/adapters/<name> (PEFT) or /vol/merged/<name> (merged bf16).
    Everything runs in Modal (fast links, base weights local); from a laptop the 380 MB upload and the
    18 GB base download both take tens of minutes."""
    import glob, shutil
    from tinker_cookbook import weights
    vol.reload()
    shutil.rmtree(f"/vol/raw/{name}", ignore_errors=True)
    weights.download(tinker_path=state, output_dir=f"/vol/raw/{name}")
    out = f"/vol/{'merged' if merge else 'adapters'}/{name}"
    shutil.rmtree(out, ignore_errors=True)
    kw = dict(base_model=VOL_MODEL, adapter_path=f"/vol/raw/{name}", output_path=out)
    (weights.build_hf_model if merge else weights.build_lora_adapter)(**kw)
    if merge:  # tokenizer/config extras vLLM needs next to the merged weights
        for f in glob.glob(VOL_MODEL + "/*.json") + glob.glob(VOL_MODEL + "/*.jinja") + glob.glob(VOL_MODEL + "/*.txt"):
            if not os.path.exists(f"{out}/{os.path.basename(f)}"):
                shutil.copy(f, out)
    vol.commit()
    print("wrote", out)
