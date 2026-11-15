"""Expert-iteration sampling: N attempts per problem through the harness, trajectories -> Modal volume.

Deploy the server first (modal deploy modal_sampler.py), then:
  uv run modal run scripts/sample.py --tag r1 --attempts 4 --limit 20           # pool: /vol/pool.jsonl
  uv run modal run scripts/sample.py --tag r2 --model r1 --pool /vol/pool_r2.jsonl   # LoRA adapter name
  uv run modal run scripts/sample.py --tag smoke --limit 3 --stub               # no harness / no GPU needed
  uv run modal volume get qwen-weights trajectories/r1.jsonl data/
  uv run modal app stop qwen-sampler                                              # STOP GPUs afterwards

Output rows (one JSON per attempt), normalized by `normalize()`:
  problem_id, rating, attempt, model, messages (OpenAI chat format incl. reasoning_content),
  turns, output_tokens, malformed, ended_by, submissions[{compiled, full_passed, passed, total}], score{solved}
plus "system" and "tools" if the harness reports them (train_sft uses these to rebuild the prompt prefix).

Harness interface (Track A, cpp/harness.py), coded against the documented signature:
  async run_episode(problem: dict, client: openai.AsyncOpenAI, **kw) -> Trajectory
Trajectory may be a dataclass, pydantic model or dict; field names are mapped leniently in normalize().
"""
import dataclasses, json, os
import modal

app = modal.App("qwen-sampling-runs")
vol = modal.Volume.from_name("qwen-weights", create_if_missing=True)
key = modal.Secret.from_dict({"VLLM_KEY": os.environ.get("VLLM_KEY") or __import__("hashlib").sha256(
    (os.environ.get("MODAL_TOKEN_SECRET", "") + "vllm").encode()).hexdigest()[:32]})
image = (
    modal.Image.from_registry("gcc:14", add_python="3.12")
    .pip_install("openai", "httpx", "pyarrow", "pydantic")
    .env({"CXX": "g++"})
    .workdir("/root")
    .add_local_python_source("cpp", "scripts")
)


def to_dict(t) -> dict:
    if isinstance(t, dict):
        return t
    if dataclasses.is_dataclass(t):
        return dataclasses.asdict(t)
    if hasattr(t, "model_dump"):
        return t.model_dump()
    return dict(vars(t))


def normalize(t, problem: dict, attempt: int, model: str) -> dict:
    d = json.loads(json.dumps(to_dict(t), default=str))
    g = lambda *ks, default=None: next((d[k] for k in ks if k in d and d[k] is not None), default)
    msgs = g("messages", default=[])
    out = {
        "problem_id": g("problem_id", default=problem.get("id")), "rating": problem.get("rating"),
        "attempt": attempt, "model": model, "messages": msgs,
        "turns": g("turns", "n_turns", default=sum(m["role"] == "assistant" for m in msgs)),
        "output_tokens": g("output_tokens", "completion_tokens", default=0),
        "malformed": g("malformed", "malformed_calls", "n_malformed", default=0),
        "ended_by": g("ended_by", "end_reason", default=""),
        "truncated": g("truncated", default=0),
        "submissions": g("submissions", "submission_log", default=None),
        "score": g("score", "result", default=None),
    }
    for k in ("system", "tools"):
        if k in d:
            out[k] = d[k]
    return out


async def _stub_episode(problem, client, **kw):
    """Placeholder until Track A's run_episode lands: one chat turn, no tools. Exercises the plumbing only."""
    r = await client.chat.completions.create(
        model=kw.get("model", "qwen"), max_tokens=kw.get("max_tokens", 256),
        messages=[{"role": "user", "content": str(problem.get("statement", problem.get("id")))[:2000]}])
    m = r.choices[0].message
    rc = getattr(m, "reasoning_content", None) or getattr(m, "reasoning", None)
    return {"problem_id": problem.get("id"), "turns": 1, "output_tokens": r.usage.completion_tokens,
            "malformed": 0, "ended_by": "stub", "score": {"solved": False},
            "messages": [{"role": "user", "content": "stub"},
                         {"role": "assistant", "content": m.content or "", "reasoning_content": rc}]}


@app.function(image=image, secrets=[key], cpu=2, memory=4096, timeout=3 * 3600, max_containers=200)
async def run_attempt(problem: dict, attempt: int, url: str, model: str = "qwen", stub: bool = False,
                      temperature: float = 0.8, max_tokens: int = 16384, max_turns: int = 30) -> dict:
    from openai import AsyncOpenAI
    client = AsyncOpenAI(base_url=url.rstrip("/") + "/v1", api_key=os.environ["VLLM_KEY"], timeout=1800,
                         max_retries=4)
    try:
        if stub:
            t = await _stub_episode(problem, client, model=model)
        else:
            from cpp import harness
            t = await harness.run_episode(problem, client, model=model, temperature=temperature,
                                          max_tokens=max_tokens, max_turns=max_turns)
        return normalize(t, problem, attempt, model)
    except Exception as e:
        return {"problem_id": problem.get("id"), "attempt": attempt, "error": repr(e)[:300]}


@app.function(image=image, volumes={"/vol": vol}, timeout=12 * 3600, cpu=2)
def run(tag: str, attempts: int, limit: int, pool: str, url: str, model: str, stub: bool, temperature: float,
        max_tokens: int):
    import time
    problems = [json.loads(l) for l in open(pool)]
    if limit:
        problems = problems[:limit]
    os.makedirs("/vol/trajectories", exist_ok=True)
    out = f"/vol/trajectories/{tag}.jsonl"
    done = set()
    if os.path.exists(out):  # resume
        for l in open(out):
            r = json.loads(l)
            done.add((r["problem_id"], r["attempt"]))
    jobs = [(p, a) for p in problems for a in range(attempts) if (p["id"], a) not in done]
    print(f"{len(problems)} problems x {attempts}; {len(jobs)} attempts to run", flush=True)
    t0, k, solved, toks, errs = time.time(), 0, 0, 0, 0
    with open(out, "a") as f:
        for r in run_attempt.starmap([(p, a, url, model, stub, temperature, max_tokens) for p, a in jobs],
                                     order_outputs=False, return_exceptions=True):
            k += 1
            if isinstance(r, Exception) or "error" in r:
                errs += 1
                print("ERR", str(r)[:200], flush=True)
                continue
            solved += bool((r.get("score") or {}).get("solved"))
            toks += r["output_tokens"] or 0
            f.write(json.dumps(r) + "\n")
            f.flush()
            if k % 20 == 0:
                vol.commit()
                print(f"{k}/{len(jobs)} solved={solved} errs={errs} out_tokens={toks/1e6:.2f}M "
                      f"{(time.time()-t0)/60:.1f}min", flush=True)
    vol.commit()
    print(f"done {k} attempts, solved {solved}, errors {errs}, {toks/1e6:.2f}M output tokens -> {out}")


@app.local_entrypoint()
def main(tag: str, attempts: int = 4, limit: int = 0, pool: str = "/vol/pool.jsonl", model: str = "qwen",
         stub: bool = False, temperature: float = 0.8, max_tokens: int = 16384):
    url = modal.Function.from_name("qwen-sampler", "serve").get_web_url()
    run.remote(tag, attempts, limit, pool, url, model, stub, temperature, max_tokens)
