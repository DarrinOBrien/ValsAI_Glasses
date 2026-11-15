"""Modal jobs: data builds and the C++ sandbox run where the network and CPUs are fast.

uv run modal run modal_app.py::build_p1 --target 1500
uv run modal volume get cpp-data pools/p1.jsonl data/pools/p1.jsonl
"""
import modal

app = modal.App("tinker-cpp")
vol = modal.Volume.from_name("cpp-data", create_if_missing=True)
image = (
    modal.Image.from_registry("gcc:14", add_python="3.12")
    .pip_install("pyarrow", "huggingface_hub", "hf_transfer")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "CXX": "g++"})
    .workdir("/root")
    .add_local_python_source("cpp", "scripts")
)


@app.function(image=image, volumes={"/root/data": vol}, cpu=32, memory=65536, timeout=4 * 3600)
def build_p1(hard: int = 900, medium: int = 700, easy: int = 150, max_tokens: int = 11000, tries: int = 5, oversample: int = 3,
             holdout_jsonl: str = "", out: str = "p1.jsonl"):
    import argparse, os
    from scripts import build_p1 as b
    os.makedirs("data/pools", exist_ok=True)
    with open("data/holdout.jsonl", "w") as f:
        f.write(holdout_jsonl)
    b.main(argparse.Namespace(hard=hard, medium=medium, easy=easy, max_tokens=max_tokens, min_tests=3, tries=tries,
                              oversample=oversample, retry_prob=0.8, workers=30,
                              out=f"data/pools/{out}"))
    vol.commit()


@app.local_entrypoint()
def p1(hard: int = 900, medium: int = 700, easy: int = 150, tries: int = 5, oversample: int = 3, out: str = "p1.jsonl"):
    build_p1.remote(out=out, hard=hard, medium=medium, easy=easy, tries=tries, oversample=oversample,
                    holdout_jsonl=open("data/holdout.jsonl").read())


# ---- Rollouts with any Tinker-served model (student self-samples for R3, or a Tinker-hosted teacher) ----
import os as _os

tinker_secret = modal.Secret.from_dict({"TINKER_API_KEY": _os.environ.get("TINKER_API_KEY", "")})
rollout_image = (
    modal.Image.from_registry("gcc:14", add_python="3.12")
    .pip_install("tinker", "tinker-cookbook", "transformers", "pyarrow", "huggingface_hub", "hf_transfer")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "CXX": "g++"})
    .workdir("/root")
    .add_local_python_source("cpp", "scripts")
)


@app.function(image=rollout_image, secrets=[tinker_secret], cpu=2, memory=4096, timeout=2 * 3600,
              max_containers=60)
async def rollout(problem: dict, model_path: str = "", base_model: str = "Qwen/Qwen3.5-9B",
                  max_tokens: int = 12000, max_turns: int = 12, temperature: float = 0.7) -> dict:
    import asyncio, json
    from tinker_cookbook import model_info
    from cpp import harness, policy
    student = policy.get_renderer()  # tool declarations always use the student's format
    name = model_info.get_recommended_renderer_name(base_model)
    r = student if base_model == policy.BASE_MODEL else policy.get_renderer(base_model, name)
    sysm = r.create_conversation_prefix_with_tools(policy.tools.tool_specs(), policy.tools.SYSTEM_PROMPT)
    pol = policy.tinker_policy(policy.make_sampling_client(model_path or None, base_model), r, max_tokens,
                               temperature)
    try:
        ep = await harness.run_episode(problem, pol, sysm, max_turns=max_turns)
        await asyncio.to_thread(harness.grade, ep, problem)
    except Exception as e:
        return {"problem_id": problem["id"], "error": repr(e)[:300]}
    d = json.loads(json.dumps(ep.__dict__, default=lambda o: o.model_dump() if hasattr(o, "model_dump") else str(o)))
    d["messages"] = d["messages"][len(sysm):]
    d.update(rating=problem["rating"], n_tests=len(problem["tests"]), tests_complete=problem["tests_complete"],
             source=f"{model_path or base_model} on open-r1/codeforces")
    return d


@app.function(image=rollout_image, volumes={"/root/data": vol}, cpu=8, memory=65536, timeout=10 * 3600)
def sample_pool(n: int, tag: str, exclude_ids: list[str], model_path: str = "",
                base_model: str = "Qwen/Qwen3.5-9B", lo: int = 800, hi: int = 3500, attempts: int = 1,
                max_tokens: int = 12000, temperature: float = 0.7, seed: int = 0):
    """Run `attempts` rollouts on `n` train problems rated in [lo, hi]; append episodes to pools/{tag}.jsonl."""
    import json, os, random, time
    from cpp import problems
    random.seed(seed)
    out = f"data/pools/{tag}.jsonl"
    os.makedirs("data/pools", exist_ok=True)
    ex = set(exclude_ids)
    ps = [p for p in problems.load_split("train") if p["id"] not in ex and p["rating"]
          and lo <= p["rating"] <= hi and len(p["tests"]) >= 3
          and sum(len(t["input"]) for t in p["tests"]) < 8_000_000]
    random.shuffle(ps)
    pick = ps[:n]
    print(f"{len(ps)} candidates, running {len(pick)} x{attempts}", flush=True)
    t0, k, solved, toks = time.time(), 0, 0, 0
    kw = dict(model_path=model_path, base_model=base_model, max_tokens=max_tokens, temperature=temperature)
    with open(out, "a") as f:
        for d in rollout.map(pick * attempts, kwargs=kw, order_outputs=False, return_exceptions=True):
            k += 1
            if isinstance(d, Exception) or "error" in d:
                print("ERR", str(d)[:200], flush=True)
                continue
            solved += bool(d["score"]["solved"])
            toks += d["output_tokens"]
            f.write(json.dumps(d) + "\n")
            f.flush()
            if k % 20 == 0:
                vol.commit()
                print(f"{k}/{len(pick) * attempts} solved={solved} out_tokens={toks/1e6:.2f}M "
                      f"{(time.time()-t0)/60:.1f}min", flush=True)
    vol.commit()
    print(f"done: {k} episodes, {solved} solved, {toks/1e6:.2f}M output tokens")


@app.local_entrypoint()
def sample(n: int = 20, tag: str = "p3_smoke", model_path: str = "", base_model: str = "Qwen/Qwen3.5-9B",
           lo: int = 800, hi: int = 3500, attempts: int = 1, temperature: float = 0.7, max_tokens: int = 12000):
    import glob, json
    ex = [json.loads(l)["id"] for l in open("data/holdout.jsonl")]
    sample_pool.remote(n=n, tag=tag, exclude_ids=ex, model_path=model_path, base_model=base_model, lo=lo, hi=hi,
                       attempts=attempts, temperature=temperature, max_tokens=max_tokens)
