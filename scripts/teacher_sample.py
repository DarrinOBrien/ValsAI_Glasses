"""Sample trajectories from a larger Qwen teacher on Tinker through the harness (distillation data).

    uv run python -m scripts.teacher_sample --teacher Qwen/Qwen3.5-397B-A17B --pool data/pool_r2.jsonl --n 15 --k 2 \
        --price-sample 7.5 --price-prefill 1.65 --cap 7 --out data/trajectories/teach_pilot_397b.jsonl

Rows are normalized like scripts/sample.py (so select_traj / thinkfix work unchanged). Spend is estimated from token
counts (prefill counted uncached = upper bound) and new episodes stop once --cap dollars is reached.
"""
from __future__ import annotations

import argparse, asyncio, json, os, sys

from cpp import harness, policy
from scripts.grpo import TinkerChatClient


class Budget:
    def __init__(self, cap, p_sample, p_prefill):
        self.cap, self.ps, self.pp = cap, p_sample, p_prefill
        self.out_tok = self.in_tok = 0

    @property
    def spend(self):
        return (self.out_tok * self.ps + self.in_tok * self.pp) / 1e6


class CappedClient(TinkerChatClient):
    def __init__(self, *a, budget: Budget, **kw):
        super().__init__(*a, **kw)
        self.budget = budget

    async def _create(self, **kw):
        if self.budget.spend >= self.budget.cap:
            raise RuntimeError("budget cap reached")
        res = await super()._create(**kw)
        self.budget.out_tok += res.usage.completion_tokens
        self.budget.in_tok += res.usage.prompt_tokens
        return res
    # TinkerChatClient binds chat.completions.create in __init__ to self._create, which resolves to this override


async def main_async(a):
    from transformers import AutoTokenizer
    from tinker_cookbook import renderers
    from scripts.sample import normalize
    tok = AutoTokenizer.from_pretrained(a.teacher)
    r = renderers.get_renderer(a.renderer, tok)
    sc = policy.make_sampling_client(None, base_model=a.teacher)
    prefix = policy.system_messages(r)
    bud = Budget(a.cap, a.price_sample, a.price_prefill)
    probs = [json.loads(l) for l in open(a.pool) if l.strip()][a.skip: a.skip + a.n]
    sem = asyncio.Semaphore(a.concurrency)
    f = open(a.out, "a")
    stats = {"done": 0, "solved": 0, "err": 0}

    async def one(p, att):
        async with sem:
            if bud.spend >= 0.8 * bud.cap:   # leave headroom for in-flight episodes
                return
            c = CappedClient(sc, r, prefix, budget=bud)
            try:
                t = await harness.run_episode(p, c, max_turns=a.max_turns, max_tokens=a.max_tokens,
                                              temperature=a.temperature,
                                              judge_fn=harness.modal_judge())
                row = normalize(t, p, att + a.attempt_offset, a.teacher)
            except Exception as e:
                stats["err"] += 1
                print("ERR", p.get("id"), repr(e)[:200], flush=True)
                return
            f.write(json.dumps(row) + "\n"); f.flush()
            stats["done"] += 1
            stats["solved"] += bool((row.get("score") or {}).get("solved"))
            print(f"{stats['done']} solved={stats['solved']} err={stats['err']} spend~${bud.spend:.2f} "
                  f"turns={row['turns']} mal={row['malformed']} end={row['ended_by']}", flush=True)

    await asyncio.gather(*[one(p, k) for p in probs for k in range(a.k)])
    print(f"FINAL teacher={a.teacher} episodes={stats['done']} solved={stats['solved']} err={stats['err']} "
          f"out_tok={bud.out_tok/1e6:.2f}M in_tok={bud.in_tok/1e6:.2f}M spend~${bud.spend:.2f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--teacher", required=True)
    ap.add_argument("--renderer", default="qwen3_5")
    ap.add_argument("--pool", default="data/pool_r2.jsonl")
    ap.add_argument("--skip", type=int, default=0)
    ap.add_argument("--n", type=int, default=15)
    ap.add_argument("--k", type=int, default=2)
    ap.add_argument("--attempt-offset", type=int, default=0)
    ap.add_argument("--max-turns", type=int, default=40)
    ap.add_argument("--max-tokens", type=int, default=24000)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--price-sample", type=float, required=True)
    ap.add_argument("--price-prefill", type=float, required=True)
    ap.add_argument("--cap", type=float, required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from dotenv import load_dotenv
    load_dotenv(".env")
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    sys.exit(asyncio.run(main_async(a)))


if __name__ == "__main__":
    main()
