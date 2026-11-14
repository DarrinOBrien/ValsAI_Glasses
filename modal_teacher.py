"""P2: Qwen3.5-122B-A10B (FP8) teacher on Modal, run through our harness copy.

uv run modal run modal_teacher.py::download                 # weights -> volume (CPU only, cheap)
uv run modal deploy modal_teacher.py                        # deploy server + runners
uv run modal run modal_teacher.py::gen --n 20 --tag smoke   # generate trajectories
uv run modal volume get cpp-data pools/p2_smoke.jsonl data/pools/
uv run modal app stop tinker-cpp-teacher                    # stop paying for GPUs
"""
import hashlib, os
import modal

app = modal.App("tinker-cpp-teacher")
MODEL = "Qwen/Qwen3.5-122B-A10B-FP8"
N_GPU = 2
MAX_LEN = 40000
hf = modal.Volume.from_name("hf-cache", create_if_missing=True)
vol = modal.Volume.from_name("cpp-data", create_if_missing=True)
key = modal.Secret.from_dict({"VLLM_KEY": os.environ.get("VLLM_KEY") or hashlib.sha256(
    (os.environ.get("MODAL_TOKEN_SECRET", "") + "vllm").encode()).hexdigest()[:32]})

vllm_image = (
    modal.Image.from_registry("nvidia/cuda:12.8.0-devel-ubuntu22.04", add_python="3.12")
    .entrypoint([])
    .uv_pip_install("vllm", "huggingface-hub", "hf_transfer")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)
runner_image = (
    modal.Image.from_registry("gcc:14", add_python="3.12")
    .pip_install("tinker", "tinker-cookbook", "transformers", "httpx", "pyarrow", "huggingface_hub", "hf_transfer")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "CXX": "g++"})
    .workdir("/root")
    .add_local_python_source("cpp", "scripts")
)


@app.function(image=vllm_image, volumes={"/root/.cache/huggingface": hf}, timeout=3 * 3600, cpu=8)
def download():
    from huggingface_hub import snapshot_download
    snapshot_download(MODEL)
    hf.commit()


@app.function(image=vllm_image, gpu=f"H100:{N_GPU}", volumes={"/root/.cache/huggingface": hf},
              secrets=[key], timeout=12 * 3600, scaledown_window=600, max_containers=1)
@modal.concurrent(max_inputs=200)
@modal.web_server(port=8000, startup_timeout=45 * 60)
def serve():
    import subprocess
    subprocess.Popen([
        "vllm", "serve", MODEL, "--served-model-name", "teacher", "--host", "0.0.0.0", "--port", "8000",
        "--tensor-parallel-size", str(N_GPU), "--max-model-len", str(MAX_LEN),
        "--gpu-memory-utilization", "0.95", "--api-key", os.environ["VLLM_KEY"]])


def _dump(o):
    return o.model_dump() if hasattr(o, "model_dump") else str(o)


@app.function(image=runner_image, secrets=[key], cpu=2, memory=4096, timeout=3 * 3600, max_containers=100,
              retries=0)
async def run_problem(problem: dict, url: str, max_tokens: int = 16000, max_turns: int = 12,
                      temperature: float = 0.6) -> dict:
    import asyncio, json, httpx
    from cpp import harness, policy
    r = policy.get_renderer()
    sysm = policy.system_messages(r)
    tok = r.tokenizer
    im_end = tok.encode("<|im_end|>", add_special_tokens=False)
    stops = r.get_stop_sequences()
    stop_ids = [s for s in stops if isinstance(s, int)] or im_end
    client = httpx.AsyncClient(timeout=1800, headers={"Authorization": f"Bearer {os.environ['VLLM_KEY']}"})

    async def pol(messages):
        ids = r.build_generation_prompt(messages).to_ints()
        room = MAX_LEN - len(ids) - 16
        if room < 1000:
            return {"role": "assistant", "content": "EXIT"}, True, 0
        body = {"model": "teacher", "prompt": ids, "max_tokens": min(max_tokens, room),
                "temperature": temperature, "top_p": 0.95, "stop_token_ids": stop_ids,
                "skip_special_tokens": False}
        for attempt in range(4):
            try:
                resp = await client.post(url + "/v1/completions", json=body)
                resp.raise_for_status()
                break
            except Exception:
                if attempt == 3:
                    raise
                await asyncio.sleep(5 * (attempt + 1))
        c = resp.json()["choices"][0]
        toks = tok.encode(c["text"], add_special_tokens=False)
        clean = c["finish_reason"] == "stop"
        if clean:
            toks = toks + im_end[:1]
        msg, term = r.parse_response(toks)
        return msg, bool(term.is_clean), len(toks)

    try:
        ep = await harness.run_episode(problem, pol, sysm, max_turns=max_turns)
        await asyncio.to_thread(harness.grade, ep, problem)
    except Exception as e:
        return {"problem_id": problem["id"], "error": repr(e)[:300]}
    d = json.loads(json.dumps(ep.__dict__, default=_dump))
    d["messages"] = d["messages"][len(sysm):]
    d.update(rating=problem["rating"], n_tests=len(problem["tests"]), tests_complete=problem["tests_complete"],
             source="Qwen3.5-122B-A10B-FP8 on open-r1/codeforces")
    return d


@app.function(image=runner_image, volumes={"/root/data": vol}, cpu=8, memory=65536, timeout=12 * 3600)
def generate(n: int, tag: str, exclude_ids: list[str], hard: float = 0.6, medium: float = 0.3,
             attempts: int = 1, seed: int = 0):
    import json, os, random, time
    from cpp import problems
    random.seed(seed)
    out = f"data/pools/p2_{tag}.jsonl"
    os.makedirs("data/pools", exist_ok=True)
    ex = set(exclude_ids)
    if os.path.exists(out):  # resume: skip problems already attempted under this tag
        ex |= {json.loads(l)["problem_id"] for l in open(out)}
    ps = [p for p in problems.load_split("train") if p["id"] not in ex and p["rating"]
          and len(p["tests"]) >= 3 and sum(len(t["input"]) for t in p["tests"]) < 8_000_000]
    random.shuffle(ps)
    H = [p for p in ps if p["rating"] >= 1900]
    M = [p for p in ps if 1300 <= p["rating"] < 1900]
    E = [p for p in ps if p["rating"] < 1300]
    pick = H[: int(n * hard)] + M[: int(n * medium)] + E[: n - int(n * hard) - int(n * medium)]
    random.shuffle(pick)
    print(f"pool hard={len(H)} medium={len(M)} easy={len(E)}; running {len(pick)} x{attempts}", flush=True)
    url = serve.get_web_url()
    t0, k, solved, toks = time.time(), 0, 0, 0
    with open(out, "a") as f:
        for d in run_problem.map(pick * attempts, kwargs={"url": url}, order_outputs=False,
                                 return_exceptions=True):
            k += 1
            if isinstance(d, Exception) or "error" in d:
                print("ERR", str(d)[:200], flush=True)
                continue
            solved += bool(d["score"]["solved"])
            toks += d["output_tokens"]
            f.write(json.dumps(d) + "\n")
            f.flush()
            if k % 10 == 0:
                vol.commit()
                print(f"{k}/{len(pick) * attempts} solved={solved} out_tokens={toks/1e6:.2f}M "
                      f"elapsed={(time.time()-t0)/60:.1f}min", flush=True)
    vol.commit()
    print(f"done: {k} episodes, {solved} solved, {toks/1e6:.2f}M output tokens, {(time.time()-t0)/60:.1f} min")


@app.local_entrypoint()
def gen(n: int = 20, tag: str = "smoke", attempts: int = 1, hard: float = 0.6, medium: float = 0.3):
    import json
    ex = [json.loads(l)["id"] for l in open("data/holdout.jsonl")]
    if os.path.exists("data/pools/p1.jsonl"):
        ex += [json.loads(l)["id"] for l in open("data/pools/p1.jsonl")]
    generate.remote(n=n, tag=tag, exclude_ids=ex, hard=hard, medium=medium, attempts=attempts)
