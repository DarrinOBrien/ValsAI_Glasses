"""Direct LoRA SFT on verified reasoning traces, with loss on the reasoning only.

Each example is: problem statement -> the trace's reasoning. There is no tool wrapper, no tool names and
no EXIT in the trained tokens, so this cannot teach the model a tool format that differs from the real
harness. It only pushes the model's C++ problem-solving reasoning toward traces whose code we verified.

uv run python -m scripts.train_think --name t1 --data data/pools/p1v2.jsonl --max-examples 600
"""
import argparse, json, math, os, random, time
from dotenv import load_dotenv
load_dotenv(".env")
import tinker, torch
from tinker_cookbook import renderers
from tinker_cookbook.supervised.common import compute_mean_nll, datum_from_model_input_weights
from cpp import policy
from scripts.build_p1 import RETRY_NOTE


def solve_turn(row):
    """The verified solve: the trained assistant turn that runs the final code."""
    for m in row["messages"]:
        if m["role"] == "assistant" and m.get("train", True) and m.get("tool_calls"):
            think = next(p["thinking"] for p in m["content"] if p["type"] == "thinking")
            code = json.loads(m["tool_calls"][0]["function"]["arguments"])["code"]
            return think.removeprefix(RETRY_NOTE).strip(), code
    return None


def main(a):
    random.seed(a.seed)
    r = policy.get_renderer()
    end_think = r.tokenizer.encode("</think>", add_special_tokens=False)[0]
    rows = []
    for path in a.data:
        rows += [json.loads(l) for l in open(path)]
    random.shuffle(rows)
    data, ntok, skipped = [], 0, 0
    for row in rows:
        if a.max_examples and len(data) >= a.max_examples:
            break
        st = solve_turn(row)
        if not st:
            continue
        think, code = st
        msgs = [row["messages"][0],
                {"role": "assistant", "content": [{"type": "thinking", "thinking": think},
                                                  {"type": "text", "text": f"```cpp\n{code}\n```"}]}]
        mi, w = r.build_supervised_example(msgs, train_on_what=renderers.TrainOnWhat.LAST_ASSISTANT_MESSAGE)
        ids = mi.to_ints()
        if len(ids) > a.max_length or end_think not in ids:
            skipped += 1
            continue
        w = torch.as_tensor(w, dtype=torch.float32).clone()
        if not a.train_code:
            w[ids.index(end_think) + 1:] = 0  # loss stops at </think>: reasoning only
        data.append(datum_from_model_input_weights(mi, w, None, reduction="mean"))
        ntok += len(ids)
    steps_per_epoch = len(data) // a.batch_size
    total = steps_per_epoch * a.epochs
    print(f"{len(data)} examples ({skipped} skipped), {ntok/1e6:.2f}M tokens/epoch, {total} steps, "
          f"est. cost ${ntok * a.epochs * 1.463 / 1e6:.2f}", flush=True)
    if a.dry_run:
        return
    os.makedirs(f"runs/{a.name}", exist_ok=True)
    tc = tinker.ServiceClient().create_lora_training_client(base_model=policy.BASE_MODEL, rank=a.rank).result()
    log = open(f"runs/{a.name}/metrics.jsonl", "a")
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
            m = {"step": step, "epoch": ep, "lr": lr, "nll": nll, "sec": time.time() - t0}
            log.write(json.dumps(m) + "\n"); log.flush()
            if step % 10 == 0:
                print(m, flush=True)
            step += 1
            if a.save_at and step in a.save_at:
                p = tc.save_weights_for_sampler(name=f"{a.name}-s{step}").result().path
                rec = {"name": f"{a.name}-s{step}", "sampler_path": p, "lr": a.lr, "rank": a.rank, "steps": step}
                print(rec, flush=True)
                open("runs/checkpoints.jsonl", "a").write(json.dumps(rec) + "\n")
        p = tc.save_weights_for_sampler(name=f"{a.name}-ep{ep}").result().path
        rec = {"name": f"{a.name}-ep{ep}", "sampler_path": p, "lr": a.lr, "rank": a.rank, "steps": step,
               "examples": len(data), "mode": "think+code" if a.train_code else "think"}
        print(rec, flush=True)
        open("runs/checkpoints.jsonl", "a").write(json.dumps(rec) + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--data", nargs="+", required=True)
    ap.add_argument("--max-examples", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--max-length", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save-at", type=int, nargs="*", default=[])
    ap.add_argument("--train-code", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    main(ap.parse_args())
