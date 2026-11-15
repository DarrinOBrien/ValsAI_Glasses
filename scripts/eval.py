"""Local eval: run a model (any OpenAI-compatible endpoint, e.g. vLLM) through the harness copy.

uv run --with openai python -m scripts.eval --name baseline --problems data/holdout.jsonl \
    --base-url http://localhost:8000/v1 --model Qwen/Qwen3.5-9B --n 150 --judge modal

Problems: JSONL in the harness schema {id, statement, rating, time_limit, memory_limit, sample_tests,
full_tests, checker}. Writes runs/eval/<name>.jsonl (trajectories) and appends to runs/eval/summary.jsonl.
"""
from __future__ import annotations

import argparse, asyncio, json, os, statistics as st

from cpp import harness
from cpp.judge import n_tests as judge_n_tests


def summarize(trajs: list[harness.Trajectory], problems: list[dict] | None = None) -> dict:
    n = len(trajs)
    turns = sum(t.turns for t in trajs) or 1
    s = {
        "n": n,
        "pass_rate": sum(t.solved for t in trajs) / n,
        "malformed_rate_per_turn": sum(t.malformed for t in trajs) / turns,
        "episodes_ended_malformed": sum(t.ended_by == "malformed" for t in trajs) / n,
        "truncation_rate_per_turn": sum(t.truncated for t in trajs) / turns,
        "episodes_with_truncation": sum(t.truncated > 0 for t in trajs) / n,
        "avg_turns": sum(t.turns for t in trajs) / n,
        "avg_output_tokens": st.mean(t.output_tokens for t in trajs),
        "exit_rate": sum(t.ended_with_exit for t in trajs) / n,
        "no_submission_rate": sum(t.final_code is None for t in trajs) / n,
        "hit_max_turns": sum(t.ended_by == "max_turns" for t in trajs) / n,
    }
    solved = [t for t in trajs if t.solved]
    s["avg_turns_when_solved"] = st.mean(t.turns for t in solved) if solved else None
    if problems:
        rating = {p["id"]: p.get("rating") or 0 for p in problems}
        for name, lo, hi in (("<1600", 0, 1600), ("1600-2199", 1600, 2200), ("2200+", 2200, 10**9)):
            b = [t for t in trajs if lo <= rating.get(t.problem_id, 0) < hi]
            if b:
                s[f"pass_rate[{name}]"] = {"n": len(b), "rate": sum(t.solved for t in b) / len(b)}
    return s


async def evaluate(problems, client, model, k=1, concurrency=16, judge_fn=harness.local_judge,
                   config=harness.DEFAULT_CONFIG, max_turns=None, max_tokens=16000, temperature=0.8,
                   top_p=None, log=print) -> list[harness.Trajectory]:
    sem = asyncio.Semaphore(concurrency)

    async def one(p):
        async with sem:
            try:
                t = await harness.run_episode(p, client, max_turns, max_tokens, temperature, model=model,
                                              config=config, judge_fn=judge_fn, top_p=top_p)
            except Exception as e:  # infrastructure error: report, count as a failed episode
                log(f"{p['id']:>10} ERROR {type(e).__name__}: {e}")
                t = harness.Trajectory(problem_id=p["id"], ended_by="error")
                t.final = {"solved": False, "compiled": False, "passed": 0, "total": judge_n_tests(p["full_tests"])}
                return t
            log(f"{p['id']:>10} r={p.get('rating')} solved={t.solved} turns={t.turns} subs={len(t.submissions)} "
                f"tok={t.output_tokens} malformed={t.malformed} trunc={t.truncated} end={t.ended_by}")
            return t

    return await asyncio.gather(*[one(p) for p in problems for _ in range(k)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--problems", required=True)
    ap.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL", "http://localhost:8000/v1"))
    ap.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", "EMPTY"))
    ap.add_argument("--model", default="Qwen/Qwen3.5-9B")
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--k", type=int, default=1, help="attempts per problem")
    ap.add_argument("--max-turns", type=int, default=None)
    ap.add_argument("--max-tokens", type=int, default=16000)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-p", type=float, default=None)
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--judge", choices=["local", "modal"], default="local")
    a = ap.parse_args()

    from openai import AsyncOpenAI  # pip install openai (uv run --with openai)
    client = AsyncOpenAI(base_url=a.base_url, api_key=a.api_key, timeout=3600, max_retries=2)
    with open(a.problems) as f:
        problems = [json.loads(l) for l in f if l.strip()][: a.n]
    judge_fn = harness.modal_judge() if a.judge == "modal" else harness.local_judge
    trajs = asyncio.run(evaluate(problems, client, a.model, a.k, a.concurrency, judge_fn,
                                 max_turns=a.max_turns, max_tokens=a.max_tokens, temperature=a.temperature,
                                 top_p=a.top_p))
    os.makedirs("runs/eval", exist_ok=True)
    with open(f"runs/eval/{a.name}.jsonl", "w") as f:
        for t in trajs:
            f.write(json.dumps(t.to_dict()) + "\n")
    summ = {"name": a.name, "model": a.model, "temperature": a.temperature, "max_tokens": a.max_tokens,
            **summarize(trajs, problems)}
    print(json.dumps(summ, indent=1))
    with open("runs/eval/summary.jsonl", "a") as f:
        f.write(json.dumps(summ) + "\n")


if __name__ == "__main__":
    main()
