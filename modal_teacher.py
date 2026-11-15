"""Teacher server: Qwen3.5-122B-A10B (FP8) on vLLM, 2xH100, OpenAI-compatible.

uv run modal deploy modal_teacher.py          # serve (stays up until `modal app stop tinker-cpp-teacher`)
uv run modal run modal_teacher.py::test       # smoke-test the deployed endpoint
Endpoint: https://<workspace>--tinker-cpp-teacher-serve.modal.run/v1  (api_key: $TEACHER_API_KEY or "EMPTY" if unset)
"""
import os
import modal

MODEL = "Qwen/Qwen3.5-122B-A10B-FP8"
SERVED_NAME = "qwen3.5-122b"
GPU = "H100:2"
PORT = 8000
MINUTES = 60

app = modal.App("tinker-cpp-teacher")

hf_cache = modal.Volume.from_name("teacher-hf-cache", create_if_missing=True)
vllm_cache = modal.Volume.from_name("teacher-vllm-cache", create_if_missing=True)

image = (
    modal.Image.from_registry("nvidia/cuda:12.9.1-devel-ubuntu24.04", add_python="3.12")
    .entrypoint([])
    .pip_install("vllm==0.32.0", "huggingface_hub[hf_transfer]")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "HF_HOME": "/root/.cache/huggingface",
          "VLLM_ENGINE_ITERATION_TIMEOUT_S": "600"})
)


@app.function(image=image, volumes={"/root/.cache/huggingface": hf_cache}, timeout=2 * 3600)
def download():
    from huggingface_hub import snapshot_download
    snapshot_download(MODEL)
    hf_cache.commit()


@app.function(
    image=image,
    gpu=GPU,
    volumes={"/root/.cache/huggingface": hf_cache, "/root/.cache/vllm": vllm_cache},
    scaledown_window=60 * MINUTES,
    timeout=24 * 60 * MINUTES,
    secrets=[modal.Secret.from_dict({"TEACHER_API_KEY": os.environ.get("TEACHER_API_KEY", "")})],
)
@modal.concurrent(max_inputs=64)
@modal.web_server(port=PORT, startup_timeout=30 * MINUTES)
def serve():
    import subprocess
    cmd = [
        "vllm", "serve", MODEL,
        "--served-model-name", SERVED_NAME,
        "--host", "0.0.0.0", "--port", str(PORT),
        "--tensor-parallel-size", "2",
        "--max-model-len", "65536",
        "--gpu-memory-utilization", "0.90",
        "--max-num-seqs", "64",
        "--reasoning-parser", "qwen3",
        "--enable-auto-tool-choice", "--tool-call-parser", "qwen3_coder",
        "--language-model-only",
    ]
    if os.environ.get("TEACHER_API_KEY"):
        cmd += ["--api-key", os.environ["TEACHER_API_KEY"]]
    subprocess.Popen(cmd)


@app.local_entrypoint()
def test():
    import json, urllib.request
    url = serve.get_web_url() + "/v1/chat/completions"
    key = os.environ.get("TEACHER_API_KEY") or "EMPTY"
    body = {"model": SERVED_NAME, "max_tokens": 512,
            "messages": [{"role": "user", "content": "Write a C++20 function that returns the n-th Fibonacci number mod 1e9+7 in O(log n). Be brief."}]}
    req = urllib.request.Request(url, json.dumps(body).encode(),
                                 {"Content-Type": "application/json", "Authorization": f"Bearer {key}"})
    with urllib.request.urlopen(req, timeout=30 * MINUTES) as r:
        out = json.load(r)
    m = out["choices"][0]["message"]
    print("URL:", url)
    print("reasoning:", (m.get("reasoning_content") or m.get("reasoning") or "")[:300])
    print("content:", m["content"])
    print("usage:", out["usage"])
