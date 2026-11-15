"""LoRA SFT on Tinker. Loss on assistant tokens only.

Rendered with Qwen3.5's native chat template via the tinker `qwen3_5` renderer (thinking kept in history
for turns after the last user message, exactly as the HF template does). Defaults: rank 32, lr 1e-4
(variant: --lr 5e-5), batch 32, max length 32k (over-long examples are dropped, never truncated), 1 epoch.

uv run python -m scripts.train_sft --name r1 --data data/selected_r1.jsonl
uv run python -m scripts.train_sft --name r1v --data data/selected_r1.jsonl --lr 5e-5
uv run python -m scripts.train_sft --name r2 --data ... --init tinker://.../weights/...   # continue from state
Input rows: {"messages": [...]} in OpenAI chat format (assistant: content, reasoning_content, tool_calls),
without the system/tool-spec prefix (rebuilt from row["system"]/row["tools"] if present, else cpp.tools). Output: runs/<name>/, runs/checkpoints.jsonl
with the tinker:// sampler path (use for Tinker sampling) and state path (use for export_lora.py).
"""
import argparse, json, math, os, random, time
from dotenv import load_dotenv
load_dotenv(".env")
import tinker
from tinker_cookbook import renderers
from tinker_cookbook.supervised.common import compute_mean_nll
from tinker_cookbook.supervised.data import conversation_to_datum
from tinker_cookbook.tokenizer_utils import get_tokenizer

BASE_MODEL = "Qwen/Qwen3.5-9B"


def get_renderer():
    return renderers.get_renderer("qwen3_5_preserve_thinking", get_tokenizer(BASE_MODEL))


def prefix_for(row, r):
    """System + tool-spec prefix. Prefer what the trajectory was sampled with (row["system"], row["tools"]);
    fall back to the harness copy's cpp.tools."""
    if row.get("tools") is not None and row.get("system") is not None:
        specs = row["tools"]  # full OpenAI dicts: HF template dumps them as-is
        return r.create_conversation_prefix_with_tools(specs, row["system"])
    from cpp import tools
    return r.create_conversation_prefix_with_tools(tools.TOOLS, tools.SYSTEM_PROMPT)


def hydrate(messages):
    """JSON tool calls (dicts) -> renderers.ToolCall objects."""
    out = []
    for m in messages:
        if m.get("tool_calls"):
            m = {**m, "tool_calls": [renderers.ToolCall(function=renderers.ToolCall.FunctionBody(
                name=t["function"]["name"], arguments=t["function"]["arguments"])) if isinstance(t, dict) else t
                for t in m["tool_calls"]]}
        out.append(m)
    return out


def to_renderer_messages(messages: list[dict]) -> list[dict]:
    """OpenAI-style chat messages (vLLM: reasoning_content / reasoning) -> tinker renderer messages."""
    out = []
    for m in messages:
        if m["role"] == "system":
            continue
        m = dict(m)
        if m["role"] == "assistant" and isinstance(m.get("content") or "", str):
            think = m.pop("reasoning_content", None) or m.pop("reasoning", None)
            m.pop("reasoning", None)
            m.pop("reasoning_content", None)
            parts = []
            if think:
                parts.append({"type": "thinking", "thinking": think.strip()})
            parts.append({"type": "text", "text": m.get("content") or ""})
            m["content"] = parts
        if m["role"] == "tool":
            m = {k: v for k, v in m.items() if k in ("role", "content", "name", "tool_call_id")}
        if m.get("tool_calls"):  # arguments must be a JSON string
            m["tool_calls"] = [{**t, "function": {**t["function"], "arguments": t["function"]["arguments"]
                                if isinstance(t["function"]["arguments"], str)
                                else json.dumps(t["function"]["arguments"])}} for t in m["tool_calls"]]
        out.append(m)
    return hydrate(out)


def main(a):
    random.seed(a.seed)
    r = get_renderer()
    rows = []
    for path in a.data:
        rows += [json.loads(l) for l in open(path)]
    random.shuffle(rows)
    if a.max_examples:
        rows = rows[: a.max_examples]
    data, skipped, ntok = [], 0, 0
    for row in rows:
        # One datum per assistant turn, trained on that turn only. Qwen3.5 strips earlier thinking from
        # history, so this is the only way each target sees the same prefix it will see when sampling.
        msgs = prefix_for(row, r) + to_renderer_messages(row["messages"])
        asst = [i for i, m in enumerate(msgs) if m["role"] == "assistant"]
        if not asst:
            skipped += 1
            continue
        # The Qwen3.5 template keeps thinking for every assistant turn after the LAST real user message.
        # If the only user message is the task prompt, all turns see the same history they will when
        # sampled, so one datum per trajectory (all assistant tokens, thinking included) is exact.
        # A later real user message strips earlier thinking, so fall back to one datum per turn.
        # DEFAULT (--mode turn, lead's decision): always one datum per assistant turn, so each target sees exactly
        # the prompt it saw at sampling time (the harness copy does not resend earlier reasoning). Costs more tokens.
        if a.mode == "trajectory" and sum(m["role"] == "user" for m in msgs) == 1:
            ds = [conversation_to_datum(msgs, r, None, renderers.TrainOnWhat.ALL_ASSISTANT_MESSAGES)]
        else:
            ds = [conversation_to_datum(msgs[: i + 1], r, None, renderers.TrainOnWhat.LAST_ASSISTANT_MESSAGE)
                  for i in asst]
        if max(d.model_input.length for d in ds) > a.max_length:  # drop, never truncate
            skipped += 1
            continue
        data += ds
        ntok += sum(d.model_input.length for d in ds)
    steps_per_epoch = len(data) // a.batch_size
    total = steps_per_epoch * a.epochs
    cost = ntok * a.epochs * 1.463 / 1e6
    print(f"{len(data)} training examples ({skipped} trajectories over {a.max_length} tokens dropped), {ntok/1e6:.2f}M tokens/epoch, "
          f"{total} steps, est. cost ${cost:.2f}", flush=True)
    if a.dry_run:
        return
    out = f"runs/{a.name}"
    os.makedirs(out, exist_ok=True)
    sc = tinker.ServiceClient()
    if a.init:
        tc = sc.create_training_client_from_state(a.init).result()
    else:
        tc = sc.create_lora_training_client(base_model=BASE_MODEL, rank=a.rank).result()
    log = open(f"{out}/metrics.jsonl", "a")
    step, warm = 0, max(1, int(0.05 * total))
    for ep in range(a.epochs):
        random.shuffle(data)
        for i in range(steps_per_epoch):
            batch = data[i * a.batch_size:(i + 1) * a.batch_size]
            lr = a.lr * min(1.0, (step + 1) / warm) * (0.5 * (1 + math.cos(math.pi * step / total)) * 0.9 + 0.1)
            t0 = time.time()
            fb = tc.forward_backward(batch, loss_fn="cross_entropy")
            op = tc.optim_step(tinker.AdamParams(learning_rate=lr, beta1=0.9, beta2=0.95, eps=1e-8))
            res = fb.result(); op.result()
            nll = compute_mean_nll([x["logprobs"] for x in res.loss_fn_outputs],
                                   [d.loss_fn_inputs["weights"] for d in batch])
            m = {"step": step, "epoch": ep, "lr": lr, "nll": nll, "tokens": sum(d.model_input.length for d in batch),
                 "sec": time.time() - t0}
            log.write(json.dumps(m) + "\n"); log.flush()
            if step % 5 == 0:
                print(m, flush=True)
            step += 1
        path = tc.save_weights_for_sampler(name=f"{a.name}-ep{ep}").result().path
        state = tc.save_state(name=f"{a.name}-ep{ep}").result().path
        rec = {"name": a.name, "epoch": ep, "sampler_path": path, "state_path": state, "lr": a.lr,
               "rank": a.rank, "examples": len(data), "data": a.data, "init": a.init}
        print(rec, flush=True)
        with open("runs/checkpoints.jsonl", "a") as f:
            f.write(json.dumps(rec) + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--data", nargs="+", required=True)
    ap.add_argument("--init", default=None)
    ap.add_argument("--max-examples", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--max-length", type=int, default=32000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mode", choices=["turn", "trajectory"], default="turn",
                    help="turn: one datum per assistant turn (default); trajectory: one datum per trajectory when exact")
    ap.add_argument("--dry-run", action="store_true")
    main(ap.parse_args())
