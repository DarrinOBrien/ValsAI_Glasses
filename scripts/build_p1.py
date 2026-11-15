"""P1: turn verified R1 C++ traces (open-r1/codeforces-cots) into harness-format trajectories.

Each kept trace's final code passes the official tests in our sandbox. Tool outputs in the
trajectory are real: we run the code and record what the tools return.

uv run python -m scripts.build_p1 --target 1500
"""
import argparse, glob, json, os, random, re
from concurrent.futures import ProcessPoolExecutor, as_completed
import pyarrow.parquet as pq
from cpp import judge, problems, tools

COTS = "data/hf/open-r1__codeforces-cots/solutions_decontaminated"
CODE_RE = re.compile(r"```(?:cpp|c\+\+|C\+\+)?\s*\n(.*?)```", re.DOTALL)
# Fixed connective lines between tool calls. Deliberately minimal: all reasoning is the teacher's.
SUBMIT_NOTE = "The output matches the expected sample output. Submitting."
RETRY_NOTE = "That attempt is wrong. Let me rethink the problem from scratch."
DONE_NOTE = "All sample tests passed. The solution is submitted, so I am done."


def band(rating):
    if rating is None:
        return None
    return "hard" if rating >= 1900 else "medium" if rating >= 1300 else "easy"


def parse_trace(r):
    g = r["generation"]
    if r["finish_reason"] != "stop" or "</think>" not in g:
        return None
    think, answer = g.split("</think>", 1)
    think = think.replace("<think>", "").strip()
    blocks = CODE_RE.findall(answer)
    if not blocks or not think:
        return None
    return {"id": r["id"], "thinking": think, "code": blocks[-1].strip(),
            "tokens": r["api_metadata"]["completion_tokens"]}


def load_traces(max_tokens):
    out = []
    for f in sorted(glob.glob(f"{COTS}/*.parquet")):
        pf = pq.ParquetFile(f)
        for b in pf.iter_batches(batch_size=256, columns=["id", "generation", "finish_reason", "api_metadata"]):
            for r in b.to_pylist():
                if r["api_metadata"]["completion_tokens"] <= max_tokens:
                    t = parse_trace(r)
                    if t:
                        out.append(t)
    return out


def call(name, **args):
    return {"type": "function", "function": {"name": name, "arguments": json.dumps(args)}}


def asst(thinking, text, tool_call=None):
    content = [{"type": "thinking", "thinking": thinking}] if thinking else []
    if text:
        content.append({"type": "text", "text": text})
    m = {"role": "assistant", "content": content or text}
    if tool_call:
        m["tool_calls"] = [tool_call]
    return m


def run_turn(p, code, stdin):
    r = judge.run_code(code, stdin, p["time_limit"], p["memory_limit"])
    return r, {"role": "tool", "name": tools.RUN_TOOL, "content": tools.format_run_result(r)}


def sample_ok(r, sample):
    return r["compiled"] and r["status"] == "OK" and judge.outputs_match(r["stdout"], sample["output"])


def build(job):
    """job: (problem, traces, retry_prob). Returns a trajectory dict or {"ok": False}.

    Every trace is first run on sample 1. A trace that passes the full tests becomes the solve. Traces
    that fail the sample can become real failed first attempts, so the model sees a bad tool result
    followed by another try and not only run, submit, EXIT. Failed attempts are context only
    ("train": False): we never train the model to write a wrong solution.
    """
    p, traces, retry_prob, seed = job
    rng = random.Random(seed)
    s = p["samples"][0]
    good, bads = None, []
    for t in traces:
        r, tm = run_turn(p, t["code"], s["input"])
        if not sample_ok(r, s):
            bads.append((t, tm))
            continue
        if good is None:
            j = judge.judge(t["code"], p["tests"], p["time_limit"], p["memory_limit"])
            if j["compiled"] and j["passed"] == j["total"] and j["max_time"] <= 0.7 * p["time_limit"]:
                good = (t, tm, j)
    if good is None:
        return {"id": p["id"], "ok": False}
    g, gtm, j = good
    sj = judge.judge(g["code"], p["samples"], p["time_limit"], p["memory_limit"], False)
    if sj["passed"] != sj["total"]:
        return {"id": p["id"], "ok": False}
    msgs = [{"role": "user", "content": tools.task_prompt(p)}]
    use = []
    if bads and rng.random() < retry_prob:
        use = bads[: (2 if len(bads) > 1 and rng.random() < 0.3 else 1)]
    for k, (t, tm) in enumerate(use):
        m = asst((RETRY_NOTE + "\n\n" if k else "") + t["thinking"], "",
                 call(tools.RUN_TOOL, code=t["code"], stdin=s["input"]))
        m["train"] = False
        msgs += [m, tm]
    # Every assistant turn carries non-empty thinking: Qwen3.5 samples after a "<think>\n" prefill.
    msgs += [asst((RETRY_NOTE + "\n\n" if use else "") + g["thinking"], "",
                  call(tools.RUN_TOOL, code=g["code"], stdin=s["input"])), gtm,
             asst(SUBMIT_NOTE, "", call(tools.SUBMIT_TOOL, code=g["code"])),
             {"role": "tool", "name": tools.SUBMIT_TOOL, "content": tools.format_submit_result(sj, 1)},
             asst(DONE_NOTE, "EXIT")]
    return {"id": p["id"], "ok": True, "kind": "retry" if use else "happy", "n_failed_first": len(use),
            "rating": p["rating"], "band": band(p["rating"]),
            "source": "open-r1/codeforces-cots (DeepSeek-R1)", "n_tests": len(p["tests"]),
            "tests_complete": p["tests_complete"], "max_time": j["max_time"], "messages": msgs}


def main(a):
    random.seed(0)
    hold = {json.loads(l)["id"] for l in open("data/holdout.jsonl")}
    from huggingface_hub import snapshot_download
    snapshot_download("open-r1/codeforces-cots", repo_type="dataset", local_dir=os.path.dirname(COTS),
                      allow_patterns=["solutions_decontaminated/*.parquet"])
    traces = load_traces(a.max_tokens)
    by_id = {}
    for t in traces:
        by_id.setdefault(t["id"], []).append(t)
    print(f"{len(traces)} parsed traces under {a.max_tokens} tokens, {len(by_id)} problems", flush=True)
    probs = [p for p in problems.load_split("train", set(by_id) - hold)
             if p["id"] not in hold and p["rating"]
             and len(p["tests"]) >= a.min_tests and sum(len(t["input"]) for t in p["tests"]) < 20_000_000]
    # hard first: verified hard traces are the scarce resource
    order = {"hard": 0, "medium": 1, "easy": 2}
    random.shuffle(probs)
    probs.sort(key=lambda p: order[band(p["rating"])])
    quota = {"hard": a.hard, "medium": a.medium, "easy": a.easy}
    cand = {"hard": [], "medium": [], "easy": []}
    for p in probs:
        cand[band(p["rating"])].append(p)
    print({k: len(v) for k, v in cand.items()}, quota, flush=True)
    jobs = []
    for b, ps in cand.items():
        for k, p in enumerate(ps[: int(quota[b] * a.oversample)]):
            ts = by_id[p["id"]][:]
            random.shuffle(ts)
            jobs.append((p, ts[: a.tries], a.retry_prob, k))
    print(f"{len(jobs)} verification jobs", flush=True)
    done, got = set(), {"hard": 0, "medium": 0, "easy": 0}
    os.makedirs("data/pools", exist_ok=True)
    n = 0
    with open(a.out, "w") as f, ProcessPoolExecutor(a.workers) as ex:
        futs = [ex.submit(build, j) for j in jobs]
        for fu in as_completed(futs):
            n += 1
            try:
                r = fu.result()
            except Exception as e:
                continue
            if r and r["ok"] and r["id"] not in done and got[r["band"]] < quota[r["band"]]:
                done.add(r["id"])
                got[r["band"]] += 1
                f.write(json.dumps(r) + "\n")
                f.flush()
            if n % 200 == 0:
                print(n, got, flush=True)
    print("final", got, flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--hard", type=int, default=900)
    ap.add_argument("--medium", type=int, default=700)
    ap.add_argument("--easy", type=int, default=150)
    ap.add_argument("--max-tokens", type=int, default=8000)
    ap.add_argument("--min-tests", type=int, default=3)
    ap.add_argument("--tries", type=int, default=5)
    ap.add_argument("--oversample", type=int, default=3)
    ap.add_argument("--retry-prob", type=float, default=0.8)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", default="data/pools/p1.jsonl")
    main(ap.parse_args())
