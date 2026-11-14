"""Run a checkpoint (or the untouched base model) through the harness copy on the holdout.

uv run python -m scripts.eval --name baseline            # untouched Qwen3.5-9B
uv run python -m scripts.eval --name r0 --model-path tinker://...
"""
import argparse, asyncio, json, os, statistics as st
from dotenv import load_dotenv
load_dotenv(".env")
from cpp import harness, policy, problems


def dump(o):
    return o.model_dump() if hasattr(o, "model_dump") else str(o)


async def main(a):
    ps = problems.holdout(a.n)[: a.n]
    r = policy.get_renderer()
    sysm = policy.system_messages(r)
    pol = policy.tinker_policy(policy.make_sampling_client(a.model_path), r, a.max_tokens, a.temperature)
    sem = asyncio.Semaphore(a.concurrency)

    async def one(p):
        async with sem:
            ep = await harness.run_episode(p, pol, sysm, max_turns=a.max_turns)
            await asyncio.to_thread(harness.grade, ep, p)
            print(f"{p['id']:>9} r={p['rating']} solved={ep.score['solved']} turns={ep.turns} "
                  f"subs={ep.submissions} malformed={ep.malformed} trunc={ep.truncated} end={ep.ended_by}", flush=True)
            return ep

    eps = await asyncio.gather(*[one(p) for p in ps])
    os.makedirs("runs/eval", exist_ok=True)
    with open(f"runs/eval/{a.name}.jsonl", "w") as f:
        for ep in eps:
            f.write(json.dumps(ep.__dict__, default=dump) + "\n")
    n, turns = len(eps), sum(e.turns for e in eps)
    summary = {
        "name": a.name, "model_path": a.model_path, "n": n,
        "solved": sum(e.score["solved"] for e in eps) / n,
        "samples_pass_at_end": sum(e.last_sample_pass for e in eps) / n,
        "no_submission": sum(e.final_code is None for e in eps) / n,
        "malformed_per_turn": sum(e.malformed for e in eps) / turns,
        "episodes_ended_malformed": sum(e.ended_by == "malformed" for e in eps) / n,
        "truncated_per_turn": sum(e.truncated for e in eps) / turns,
        "exit_rate": sum(e.ended_by == "exit" for e in eps) / n,
        "avg_turns": turns / n, "avg_output_tokens": st.mean(e.output_tokens for e in eps),
    }
    print(json.dumps(summary, indent=1))
    with open("runs/eval/summary.jsonl", "a") as f:
        f.write(json.dumps(summary) + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--model-path", default=None)
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--max-turns", type=int, default=20)
    ap.add_argument("--max-tokens", type=int, default=12000)
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--concurrency", type=int, default=8)
    asyncio.run(main(a := ap.parse_args()))
