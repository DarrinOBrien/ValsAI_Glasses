"""R1 hard-problem traces -> harness-format trajectories, verified in our sandbox.

For each hard problem (default rating >= 2200) that has an R1 trace (open-r1/codeforces-cots), we rewrite the
trace as a short harness episode with REAL tool outputs and drop anything that does not verify:

    user: task prompt
    assistant: <R1 thinking> + run_cpp(code, stdin=sample 1)      -> real run output
    assistant: submit(code)                                       -> real sample verdicts
    assistant: EXIT

Kept only if the extracted C++ compiles and passes ALL full tests (and the problem's checker, python or testlib), is not
slow (max test time <= --max-time-frac of the limit), its sample-1 run matches the expected output (exact-match
problems), and all samples pass on submit. Problems in --exclude (the eval split) are never used.
Output rows: {id, ok, kind: "r1", rating, source, max_time, messages} - the same shape scripts.train_sft reads
(system prompt is added at train time). Rows with ok=False are written only with --keep-failures (for stats).

    uv run python -m scripts.build_r1 --problems data/pool.jsonl --traces data/cots/*.parquet \
        --exclude data/eval150.jsonl --out data/pools/r1_hard.jsonl [--tests-root data/tests]

--problems: pool JSONL. full_tests may be a list, or the pool builder's pointer {"path": "/vol/pool/tests/<id>.json", ...}.
--judge modal (default for pointers without --tests-root): full-test verification runs on the deployed `tinker-cpp` Modal
judge, which reads the tests from the cpp-data volume (they are 28 GB: never download them). --judge local reads the
tests from --tests-root (or the pointer path) and judges with the local sandbox.
--traces: JSONL or parquet, either raw cots rows (`id`, `generation`, optional `finish_reason`,
`api_metadata.completion_tokens`) or pre-parsed rows (`id`, `thinking`, `code`, `tokens`) as in r1_hard*.jsonl.
"""
from __future__ import annotations

import argparse, glob, json, os, re
from concurrent.futures import ThreadPoolExecutor

from cpp import judge, tools

CODE_RE = re.compile(r"```(?:cpp|c\+\+|C\+\+)?\s*\n(.*?)```", re.DOTALL)
SUBMIT_NOTE = "The output matches the expected sample output. Submitting."
DONE_NOTE = "All sample tests passed. The solution is submitted, so I am done."


def parse_trace(r: dict) -> dict | None:
    if r.get("thinking") and r.get("code") and "generation" not in r:   # pre-parsed (build_cots output)
        return {"id": r["id"], "thinking": r["thinking"].strip(), "code": r["code"].strip(),
                "tokens": r.get("tokens") or len(r["thinking"]) // 3}
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


def load_traces(paths: list[str], max_tokens: int | None = None) -> dict[str, list[dict]]:
    by_id: dict[str, list[dict]] = {}
    for p in paths:
        if p.endswith(".parquet"):
            import pyarrow.parquet as pq
            rows = (r for b in pq.ParquetFile(p).iter_batches(batch_size=256) for r in b.to_pylist())
        else:
            rows = (json.loads(l) for l in open(p) if l.strip())
        for r in rows:
            t = parse_trace(r)
            if t and (max_tokens is None or t["tokens"] <= max_tokens):
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


def modal_full_judge(retries: int = 2):
    """Sync judge fn backed by the deployed `tinker-cpp` Modal function (tests are read from the cpp-data volume)."""
    import modal
    fn = modal.Function.from_name("tinker-cpp", "judge_remote")

    def j(code, tests, tl, ml, chk, stop_on_fail=True, chk_lang=None):
        for i in range(retries + 1):
            try:
                return fn.remote(code, tests, tl, ml, chk, chk_lang)
            except Exception:
                if i == retries:
                    raise
    return j


def local_full_judge(code, tests, tl, ml, chk, stop_on_fail=True, chk_lang=None):
    return judge.judge(code, tests, tl, ml, chk, stop_on_fail, chk_lang)


def build(p: dict, trace: dict, tests, max_time_frac: float = 0.6, judge_fn=local_full_judge) -> dict:
    """Verify one (problem, trace) and rewrite it. Returns {id, ok, reason?, ...}.
    tests: list of tests, or the {path,..} pointer when judge_fn resolves it remotely (Modal)."""
    pid, tl, ml, chk = p["id"], p["time_limit"], p["memory_limit"], p.get("checker")
    cl = p.get("checker_lang")
    fail = lambda why: {"id": pid, "ok": False, "reason": why}
    code, s = trace["code"], p["sample_tests"][0]
    j = judge_fn(code, tests, tl, ml, chk, True, cl)
    if not j["compiled"]:
        return fail("compile_error")
    if j["passed"] != j["total"] or not j["total"]:
        return fail("wrong_answer")
    if j["max_time"] > max_time_frac * tl:
        return fail("slow")
    r = judge.run_code(code, s["input"], max(tl, 10.0), ml)   # local: only the output matters here (tool-result text)
    if not r["compiled"] or r["status"] != "OK":
        return fail("sample_run_failed")
    if not chk and not judge.outputs_match(r["stdout"], s["output"]):
        return fail("sample_run_mismatch")
    sj = judge_fn(code, p["sample_tests"], max(tl, 10.0) if judge_fn is local_full_judge else tl, ml, chk, False, cl)
    if sj["passed"] != sj["total"]:
        return fail("samples_failed")
    msgs = [{"role": "user", "content": tools.task_prompt(p)},
            _asst(trace["thinking"], "", _call(tools.RUN_TOOL, code=code, stdin=s["input"])),
            {"role": "tool", "name": tools.RUN_TOOL, "content": tools.format_run_result(r)},
            _asst(SUBMIT_NOTE, "", _call(tools.SUBMIT_TOOL, code=code)),
            {"role": "tool", "name": tools.SUBMIT_TOOL, "content": tools.format_submit_result(sj, 1)},
            _asst(DONE_NOTE, "EXIT")]
    return {"id": pid, "ok": True, "kind": "r1", "rating": p.get("rating"), "source": "open-r1/codeforces-cots (DeepSeek-R1)",
            "n_tests": len(tests) if isinstance(tests, list) else (tests.get("n") if isinstance(tests, dict) else None),
            "max_time": j["max_time"], "trace_tokens": trace["tokens"], "messages": msgs}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--problems", required=True)
    ap.add_argument("--traces", nargs="+", required=True, help="JSONL/parquet files or globs")
    ap.add_argument("--out", required=True)
    ap.add_argument("--exclude", nargs="*", default=[], help="problem JSONLs whose ids must never be used (eval)")
    ap.add_argument("--tests-root", default=None, help="local dir holding <id>.json test files (--judge local)")
    ap.add_argument("--judge", choices=["modal", "local"], default=None,
                    help="where full tests are judged; default: local if --tests-root is given, else modal")
    ap.add_argument("--min-rating", type=int, default=2200)
    ap.add_argument("--max-problems", type=int, default=200, help="stop keeping after this many verified problems")
    ap.add_argument("--limit", type=int, default=0, help="attempt only the first N eligible problems (trial runs)")
    ap.add_argument("--tries", type=int, default=3, help="traces to try per problem")
    ap.add_argument("--max-tokens", type=int, default=14000, help="skip traces longer than this (SFT max length is 16k incl. prompt)")
    ap.add_argument("--max-time-frac", type=float, default=0.6, help="drop solutions slower than this fraction of the limit (see docs/judge_noise.md)")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--keep-failures", action="store_true")
    a = ap.parse_args(argv)

    banned = {json.loads(l)["id"] for f in a.exclude for l in open(f) if l.strip()}
    paths = [x for g in a.traces for x in sorted(glob.glob(g))]
    by_id = load_traces(paths)   # length filter is applied per problem below so drops are counted
    probs = [p for p in (json.loads(l) for l in open(a.problems) if l.strip())
             if (p.get("rating") or 0) >= a.min_rating and p["id"] not in banned and p["id"] in by_id]
    if a.limit:
        probs = probs[:a.limit]
    print(f"{len(by_id)} problems with parsed traces; {len(probs)} eligible (rating>={a.min_rating}, not excluded)")

    mode = a.judge or ("local" if a.tests_root else "modal")
    jfn = modal_full_judge() if mode == "modal" else local_full_judge

    def work(p):
        tests = p["full_tests"] if mode == "modal" else resolve_tests(p, a.tests_root)
        last = None
        ts = [t for t in by_id[p["id"]] if t["tokens"] <= a.max_tokens]
        if not ts:
            return {"id": p["id"], "ok": False, "reason": "too_long"}
        for t in ts[: a.tries]:
            last = build(p, t, tests, a.max_time_frac, jfn)
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
