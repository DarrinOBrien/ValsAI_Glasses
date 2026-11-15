"""LoRA SFT on Tinker. Loss on assistant tokens only.

uv run python -m scripts.train_sft --name r0 --data data/pools/p1.jsonl --max-examples 200 --lr 1e-4
uv run python -m scripts.train_sft --name r2 --data ... --init tinker://.../weights/...   # branch
"""
import argparse, json, math, os, random, time
from dotenv import load_dotenv
load_dotenv(".env")
import tinker
from tinker_cookbook import renderers
from tinker_cookbook.supervised.common import compute_mean_nll
from tinker_cookbook.supervised.data import conversation_to_datum
from cpp import policy


def main(a):
    random.seed(a.seed)
    r = policy.get_renderer()
    sysm = policy.system_messages(r)
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
        msgs = sysm + policy.hydrate(row["messages"])
        ds = [conversation_to_datum(msgs[: i + 1], r, None, renderers.TrainOnWhat.LAST_ASSISTANT_MESSAGE)
              for i, m in enumerate(msgs) if m["role"] == "assistant"]
        if max(d.model_input.length for d in ds) > a.max_length:  # drop, never truncate
            skipped += 1
            continue
        data += ds
        ntok += sum(d.model_input.length for d in ds)
    steps_per_epoch = len(data) // a.batch_size
    total = steps_per_epoch * a.epochs
    cost = ntok * a.epochs * 1.463 / 1e6
    print(f"{len(data)} turn examples ({skipped} trajectories over {a.max_length} tokens dropped), {ntok/1e6:.2f}M tokens/epoch, "
          f"{total} steps, est. cost ${cost:.2f}", flush=True)
    if a.dry_run:
        return
    out = f"runs/{a.name}"
    os.makedirs(out, exist_ok=True)
    sc = tinker.ServiceClient()
    if a.init:
        tc = sc.create_training_client_from_state(a.init).result()
    else:
        tc = sc.create_lora_training_client(base_model=policy.BASE_MODEL, rank=a.rank).result()
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
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--max-length", type=int, default=16000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true")
    main(ap.parse_args())
