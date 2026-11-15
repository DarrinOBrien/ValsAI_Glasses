"""Real-model harness smoke test: Qwen3.5-9B (or a tinker:// checkpoint) through harness.run_episode via Tinker.

    uv run python -m scripts.smoke_real --problems data/syn_pool.jsonl --k 2 [--model-path tinker://...]
    uv run python -m scripts.smoke_real --template-only      # no sampling: renderer vs HF chat template

Checks (prints PASS/FAIL per item, exit 1 on failure):
  1. the model emits valid tool calls (malformed rate per turn is low)
  2. reasoning is captured (assistant reasoning_content non-empty; no stray <think> in content)
  3. episodes finish (EXIT / solved), truncation is rare
  4. TEMPLATE: the prompt tokens the training renderer builds equal what the HF chat template (what vLLM serves)
     produces for the same conversation, at every assistant turn of the sampled episodes.
"""
from __future__ import annotations

import argparse, asyncio, json, sys

from cpp import harness, policy, tools


def hf_messages(msgs: list[dict]) -> list[dict]:
    """Harness chat -> what vLLM's chat template receives (OpenAI format: arguments may be JSON strings)."""
    out = []
    for m in msgs:
        m = {k: v for k, v in m.items() if k in ("role", "content", "tool_calls", "name", "tool_call_id")}
        if m.get("tool_calls"):
            m["tool_calls"] = [{**tc, "function": {**tc["function"], "arguments": json.loads(tc["function"]["arguments"])
                                                  if isinstance(tc["function"]["arguments"], str) else tc["function"]["arguments"]}}
                               for tc in m["tool_calls"]]
        out.append(m)
    return out


def template_diffs(trajs, renderer, tokenizer, tool_defs, enable_thinking=True) -> list[dict]:
    """For each assistant turn compare renderer prompt text with HF apply_chat_template(add_generation_prompt)."""
    prefix = policy.system_messages(renderer)
    bad = []
    for tr in trajs:
        msgs = tr.messages
        for i, m in enumerate(msgs):
            if m["role"] != "assistant":
                continue
            hist = [{k: v for k, v in x.items() if k != "reasoning_content"} for x in msgs[:i]]  # reasoning not resent
            mine = renderer.build_generation_prompt(prefix + policy.hydrate([x for x in hist if x["role"] != "system"]))
            mine_txt = tokenizer.decode(mine.to_ints())
            ref = tokenizer.apply_chat_template(hf_messages(hist), tools=tool_defs, add_generation_prompt=True,
                                                tokenize=False, enable_thinking=enable_thinking)
            if mine_txt != ref:
                bad.append({"problem": tr.problem_id, "turn": i, "mine": mine_txt, "ref": ref})
    return bad


def first_diff(a: str, b: str) -> str:
    n = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))
    return f"@{n}: mine={a[max(0, n - 40):n + 60]!r}\n      ref ={b[max(0, n - 40):n + 60]!r}"


async def main_async(a):
    from transformers import AutoTokenizer
    from scripts.grpo import TinkerChatClient
    tok = AutoTokenizer.from_pretrained(policy.BASE_MODEL)
    r = policy.get_renderer()
    ok_all = True

    def report(name, ok, detail=""):
        nonlocal ok_all
        ok_all &= bool(ok)
        print(f"{'PASS' if ok else 'FAIL'}  {name}  {detail}", flush=True)

    probs = [json.loads(l) for l in open(a.problems) if l.strip()][: a.n]
    sc = policy.make_sampling_client(a.model_path)
    prefix = policy.system_messages(r)
    sem = asyncio.Semaphore(a.concurrency)

    clients = []

    async def one(p):
        async with sem:
            c = TinkerChatClient(sc, r, prefix)
            clients.append(c)
            return await harness.run_episode(p, c, max_turns=a.max_turns, max_tokens=a.max_tokens,
                                             temperature=a.temperature,
                                             judge_fn=harness.modal_judge() if a.judge == "modal" else harness.local_judge)

    trajs = await asyncio.gather(*[one(p) for p in probs for _ in range(a.k)])
    turns = sum(t.turns for t in trajs) or 1
    for t in trajs:
        print(f"  {t.problem_id:>12} solved={t.solved} turns={t.turns} tok={t.output_tokens} malformed={t.malformed} "
              f"trunc={t.truncated} end={t.ended_by}")
    mal = sum(t.malformed for t in trajs) / turns
    report("valid tool calls", mal <= a.max_malformed, f"malformed/turn={mal:.1%} over {turns} turns")
    asst = [m for t in trajs for m in t.messages if m["role"] == "assistant"]
    with_reason = sum(bool(m.get("reasoning_content")) for m in asst)
    stray = sum("<think>" in (m.get("content") or "") or "</think>" in (m.get("content") or "") for m in asst)
    report("reasoning captured", with_reason > 0 and stray == 0,
           f"{with_reason}/{len(asst)} assistant msgs have reasoning; {stray} with stray think tags in content")
    trunc = sum(t.truncated for t in trajs) / turns
    report("episodes finish", sum(t.ended_with_exit for t in trajs) >= 0.5 * len(trajs) and trunc < 0.05,
           f"exit={sum(t.ended_with_exit for t in trajs)}/{len(trajs)} solved={sum(t.solved for t in trajs)} trunc/turn={trunc:.1%}")
    # informational: tool call emitted inside an unclosed think block (reasoning_content ends up empty here;
    # a server-side reasoning parser may treat such turns differently - check against the real vLLM run)
    raw = [tok.decode(t.tokens) for c in clients for t in c.turns]
    inside = sum("<tool_call>" in x and "</think>" not in x for x in raw)
    print(f"INFO  tool call inside unclosed <think>: {inside}/{len(raw)} turns ({inside / max(len(raw), 1):.0%})")
    diffs = template_diffs(trajs, r, tok, tools.TOOLS)
    report("train renderer == HF chat template", not diffs, f"{len(diffs)} mismatching prompts")
    for d in diffs[:3]:
        print("   ", d["problem"], "turn", d["turn"], first_diff(d["mine"], d["ref"]))
    if a.save:
        with open(a.save, "w") as f:
            for t in trajs:
                f.write(json.dumps(t.to_dict()) + "\n")
    return 0 if ok_all else 1


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--problems", default="data/syn_pool.jsonl")
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--k", type=int, default=2)
    ap.add_argument("--model-path", default=None, help="tinker:// sampler path; default base Qwen3.5-9B")
    ap.add_argument("--max-turns", type=int, default=12)
    ap.add_argument("--max-tokens", type=int, default=8000)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--max-malformed", type=float, default=0.05)
    ap.add_argument("--save", default=None, help="write trajectories JSONL here")
    ap.add_argument("--judge", choices=["local", "modal"], default="local")
    a = ap.parse_args(argv)
    from dotenv import load_dotenv
    load_dotenv(".env")
    sys.exit(asyncio.run(main_async(a)))


if __name__ == "__main__":
    main()
