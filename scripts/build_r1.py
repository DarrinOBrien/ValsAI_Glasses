"""R1 hard-problem traces -> harness-format trajectories, verified in our sandbox.

For each hard problem (default rating >= 2200) that has an R1 trace (open-r1/codeforces-cots), we rewrite the
trace as a short harness episode with REAL tool outputs and drop anything that does not verify:

    user: task prompt
    assistant: <R1 thinking> + run_cpp(code, stdin=sample 1)      -> real run output
    assistant: submit(code)                                       -> real sample verdicts
    assistant: EXIT

Kept only if the extracted C++ compiles and passes ALL full tests (and the problem's checker, if any), is not
slow (max test time <= --max-time-frac of the limit), its sample-1 run matches the expected output (exact-match
problems), and all samples pass on submit. Problems in --exclude (the eval split) are never used.
Output rows: {id, ok, kind: "r1", rating, source, max_time, messages} - the same shape scripts.train_sft reads
(system prompt is added at train time). Rows with ok=False are written only with --keep-failures (for stats).

    uv run python -m scripts.build_r1 --problems data/pool.jsonl --traces data/cots/*.parquet \
        --exclude data/eval150.jsonl --out data/pools/r1_hard.jsonl [--tests-root data/tests]

--problems: JSONL in the harness schema. full_tests may be a list, or the pool builder's pointer
{"path": ".../<id>.json", "n": ..} to a JSON file {"tests": [...]}; with --tests-root only the file's basename
is used, so a pulled copy of a Modal volume works. --traces: JSONL or parquet with `id` and `generation`
(optional `finish_reason`, `api_metadata.completion_tokens`).
"""
from __future__ import annotations

import argparse, glob, json, os, re
from concurrent.futures import ThreadPoolExecutor

from cpp import judge, tools

CODE_RE = re.compile(r"```(?:cpp|c\+\+|C\+\+)?\s*\n(.*?)```", re.DOTALL)
SUBMIT_NOTE = "The output matches the expected sample output. Submitting."
DONE_NOTE = "All sample tests passed. The solution is submitted, so I am done."


def parse_trace(r: dict) -> dict | None:
    g = r.get("generation") or ""
    if r.get("finish_reason", "stop") != "stop" or "</think>" not in g:
        return None
    think, answer = g.split("</think>", 1)
    think = think.replace("<think>", "").strip()
    blocks = CODE_RE.findall(answer)
    if not blocks or not think:
        return None
    toks = ((r.get("api_metadata") or {}).get("completion_tokens")) or len(g) // 3
    return {"id": r["id"], "thinking": think, "code": blocks[-1].strip(), "tokens": toks}


def load_traces(paths: list[str], max_tokens: int) -> dict[str, list[dict]]:
    by_id: dict[str, list[dict]] = {}
    for p in paths:
        if p.endswith(".parquet"):
            import pyarrow.parquet as pq
            rows = (r for b in pq.ParquetFile(p).iter_batches(batch_size=256) for r in b.to_pylist())
        else:
            rows = (json.loads(l) for l in open(p) if l.strip())
        for r in rows:
            t = parse_trace(r)
            if t and t["tokens"] <= max_tokens:
                by_id.setdefault(t["id"], []).append(t)
    return by_id


def resolve_tests(p: dict, tests_root: str | None) -> list[dict]:
    ft = p["full_tests"]
    if isinstance(ft, list):
        return ft
    path = os.path.join(tests_root, os.path.basename(ft["path"])) if tests_root else ft["path"]
    return json.load(open(path))["tests"]


def _call(name, **args):
    return {"type": "function", "function": {"name": name, "arguments": json.dumps(args)}}


def _asst(thinking, text, tool_call=None):
    # non-empty thinking on every turn: Qwen3.5 samples after a "<think>\n" prefill
    m = {"role": "assistant", "content": [{"type": "thinking", "thinking": thinking}] +
         ([{"type": "text", "text": text}] if text else [])}
    if tool_call:
        m["tool_calls"] = [tool_call]
    return m


def build(p: dict, trace: dict, tests: list[dict], max_time_frac: float = 0.6) -> dict:
    """Verify one (problem, trace) and rewrite it. Returns {id, ok, reason?, ...}."""
    pid, tl, ml, chk = p["id"], p["time_limit"], p["memory_limit"], p.get("checker")
    fail = lambda why: {"id": pid, "ok": False, "reason": why}
    if chk and "#include" not in chk:
        return fail("unsupported_checker")  # judge expects testlib C++ source
    code, s = trace["code"], p["sample_tests"][0]
    j = judge.judge(code, tests, tl, ml, chk)
    if not j["compiled"]:
        return fail("compile_error")
    if j["passed"] != j["total"] or not j["total"]:
        return fail("wrong_answer")
    if j["max_time"] > max_time_frac * tl:
        return fail("slow")
    r = judge.run_code(code, s["input"], tl, ml)
    if not r["compiled"] or r["status"] != "OK":
        return fail("sample_run_failed")
    if not chk and not judge.outputs_match(r["stdout"], s["output"]):
        return fail("sample_run_mismatch")
    sj = judge.judge(code, p["sample_tests"], tl, ml, chk, False)
    if sj["passed"] != sj["total"]:
        return fail("samples_failed")
    msgs = [{"role": "user", "content": tools.task_prompt(p)},
            _asst(trace["thinking"], "", _call(tools.RUN_TOOL, code=code, stdin=s["input"])),
            {"role": "tool", "name": tools.RUN_TOOL, "content": tools.format_run_result(r)},
            _asst(SUBMIT_NOTE, "", _call(tools.SUBMIT_TOOL, code=code)),
            {"role": "tool", "name": tools.SUBMIT_TOOL, "content": tools.format_submit_result(sj, 1)},
            _asst(DONE_NOTE, "EXIT")]
    return {"id": pid, "ok": True, "kind": "r1", "rating": p.get("rating"), "source": "open-r1/codeforces-cots (DeepSeek-R1)",
            "n_tests": len(tests), "max_time": j["max_time"], "trace_tokens": trace["tokens"], "messages": msgs}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--problems", required=True)
    ap.add_argument("--traces", nargs="+", required=True, help="JSONL/parquet files or globs")
    ap.add_argument("--out", required=True)
    ap.add_argument("--exclude", nargs="*", default=[], help="problem JSONLs whose ids must never be used (eval)")
    ap.add_argument("--tests-root", default=None)
    ap.add_argument("--min-rating", type=int, default=2200)
    ap.add_argument("--max-problems", type=int, default=200)
    ap.add_argument("--tries", type=int, default=3, help="traces to try per problem")
    ap.add_argument("--max-tokens", type=int, default=8000, help="skip traces longer than this (train max length)")
    ap.add_argument("--max-time-frac", type=float, default=0.6, help="drop solutions slower than this fraction of the limit (see docs/judge_noise.md)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--keep-failures", action="store_true")
    a = ap.parse_args(argv)

    banned = {json.loads(l)["id"] for f in a.exclude for l in open(f) if l.strip()}
    paths = [x for g in a.traces for x in sorted(glob.glob(g))]
    by_id = load_traces(paths, a.max_tokens)
    probs = [p for p in (json.loads(l) for l in open(a.problems) if l.strip())
             if (p.get("rating") or 0) >= a.min_rating and p["id"] not in banned and p["id"] in by_id]
    print(f"{len(by_id)} problems with parsed traces; {len(probs)} eligible (rating>={a.min_rating}, not excluded)")

    def work(p):
        tests = resolve_tests(p, a.tests_root)
        last = None
        for t in by_id[p["id"]][: a.tries]:
            last = build(p, t, tests, a.max_time_frac)
            if last["ok"]:
                break
        return last

    kept, reasons = 0, {}
    with open(a.out, "w") as f, ThreadPoolExecutor(a.workers) as ex:
        for r in ex.map(lambda p: _safe(work, p), probs):
            if r["ok"]:
                if kept >= a.max_problems:
                    continue
                kept += 1
            else:
                reasons[r["reason"]] = reasons.get(r["reason"], 0) + 1
            if r["ok"] or a.keep_failures:
                f.write(json.dumps(r) + "\n")
    print(f"kept {kept}/{len(probs)} -> {a.out}; dropped: {reasons}")


def _safe(fn, p):
    try:
        return fn(p)
    except Exception as e:
        return {"id": p["id"], "ok": False, "reason": f"error:{type(e).__name__}"}


if __name__ == "__main__":
    main()
