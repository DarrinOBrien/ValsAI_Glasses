"""One resumable command per expert-iteration round: sample -> select -> train -> export -> eval -> report.

    uv run python -m scripts.round plan  --tag r1                       # print the stages and commands, run nothing
    uv run python -m scripts.round run   --tag r1 --yes                 # round 1 from base Qwen3.5-9B
    uv run python -m scripts.round run   --tag r2 --yes --model r1 --init tinker://.../weights/r1-ep0 \
        --pool /vol/pool_r2.jsonl --eval-url https://<workspace>--qwen-sampler-serve.modal.run/v1
    uv run python -m scripts.round status --tag r1
    ... --from-stage train      # restart at a stage      ... --only report     # run a single stage
    ... --force                 # ignore "done" marks     ... --skip eval,report

State is kept in runs/rounds/<tag>/state.json; a stage is skipped when marked done AND its outputs still exist,
so re-running the same command resumes after a crash. `run` without --yes only prints the plan (stages spend
Modal/Tinker money). Stages and the CLIs they call (documented in docs/training.md and each script's docstring):

  deploy_sample  modal deploy modal_sampler.py   (ADAPTERS=<--model>=/vol/adapters/<--model> when sampling a LoRA)
  sample         modal run scripts/sample.py --tag T --attempts 4 [--model M] [--pool P]   -> /vol/trajectories/T.jsonl
  fetch          modal volume get qwen-weights trajectories/T.jsonl data/trajectories/
  stop_sample    modal app stop qwen-sampler --yes
  select         python -m scripts.select_traj data/trajectories/T.jsonl --out data/selected_T.jsonl [+ --extra files]
  train          python -m scripts.train_sft --name T --data data/selected_T.jsonl [--init S] [--lr X]
  export         python -m scripts.export_lora --state <state_path from runs/checkpoints.jsonl> --name T ; modal volume put
  deploy_eval    ADAPTERS=T=/vol/adapters/T modal deploy modal_sampler.py
  eval           python -m scripts.eval --name T --problems data/eval150.jsonl --base-url <--eval-url> --model T --judge modal
  stop_eval      modal app stop qwen-sampler --yes
  report         python -m scripts.analyze <baseline> runs/eval/T.jsonl --problems data/eval150.jsonl > runs/rounds/T/report.md

STUBS / manual: the Modal web-endpoint URL is not discoverable from code - pass --eval-url (printed by `modal
deploy`). Round >= 2 needs --init (the best round-1 `state_path`). Merging (--merge) is the fallback if vLLM
LoRA serving fails for this architecture. For a GPU-free check of the plumbing use `--dry-run`.
"""
from __future__ import annotations

import argparse, json, os, shlex, subprocess, sys
from dataclasses import dataclass, field
from typing import Callable

STAGES = ["deploy_sample", "sample", "fetch", "stop_sample", "select", "train", "export",
          "deploy_eval", "eval", "stop_eval", "report"]


@dataclass
class Cmd:
    argv: list[str]
    env: dict = field(default_factory=dict)

    def show(self) -> str:
        e = " ".join(f"{k}={shlex.quote(v)}" for k, v in self.env.items())
        return (e + " " if e else "") + " ".join(shlex.quote(x) for x in self.argv)


@dataclass
class Ctx:
    tag: str
    attempts: int = 4
    model: str | None = None        # LoRA adapter name to sample from (round >= 2); None = base
    pool: str | None = None
    init: str | None = None         # tinker:// state to continue from (round >= 2)
    lr: float | None = None
    extra_data: list[str] = field(default_factory=list)   # e.g. R1 traces from build_r1
    eval_problems: str = "data/eval150.jsonl"
    eval_url: str | None = None
    baseline: str = "runs/eval/baseline.jsonl"
    replicas: str = "3"
    merge: bool = False
    train_args: list[str] = field(default_factory=list)
    root: str = "."

    @property
    def dir(self): return os.path.join(self.root, "runs", "rounds", self.tag)
    @property
    def traj(self): return os.path.join(self.root, "data", "trajectories", f"{self.tag}.jsonl")
    @property
    def selected(self): return os.path.join(self.root, "data", f"selected_{self.tag}.jsonl")
    @property
    def eval_out(self): return os.path.join(self.root, "runs", "eval", f"{self.tag}.jsonl")
    @property
    def report(self): return os.path.join(self.dir, "report.md")
    @property
    def export_dir(self): return os.path.join(self.root, "exports", self.tag, "merged" if self.merge else "adapter")


def last_checkpoint(ctx: Ctx) -> dict | None:
    """Most recent record for this run name in runs/checkpoints.jsonl (written by train_sft)."""
    p = os.path.join(ctx.root, "runs", "checkpoints.jsonl")
    if not os.path.exists(p):
        return None
    recs = [json.loads(l) for l in open(p) if l.strip()]
    recs = [r for r in recs if r.get("name") == ctx.tag]
    return recs[-1] if recs else None


def py(*a): return ["uv", "run", "python", "-m", *a]
def modal(*a): return ["uv", "run", "--env-file", ".env", "modal", *a]


def commands(stage: str, ctx: Ctx) -> list[Cmd]:
    t = ctx.tag
    if stage == "deploy_sample":
        env = {"N_REPLICAS": ctx.replicas}
        if ctx.model:
            env["ADAPTERS"] = f"{ctx.model}=/vol/adapters/{ctx.model}"
        return [Cmd(modal("deploy", "modal_sampler.py"), env)]
    if stage == "sample":
        c = modal("run", "scripts/sample.py", "--tag", t, "--attempts", str(ctx.attempts))
        if ctx.model: c += ["--model", ctx.model]
        if ctx.pool: c += ["--pool", ctx.pool]
        return [Cmd(c)]
    if stage == "fetch":
        return [Cmd(modal("volume", "get", "--force", "qwen-weights", f"trajectories/{t}.jsonl", "data/trajectories/"))]
    if stage in ("stop_sample", "stop_eval"):
        return [Cmd(modal("app", "stop", "qwen-sampler", "--yes"))]
    if stage == "select":
        return [Cmd(py("scripts.select_traj", ctx.traj, "--out", ctx.selected))]
    if stage == "train":
        c = py("scripts.train_sft", "--name", t, "--data", ctx.selected, *ctx.extra_data)
        if ctx.init: c += ["--init", ctx.init]
        if ctx.lr: c += ["--lr", str(ctx.lr)]
        return [Cmd(c + ctx.train_args)]
    if stage == "export":
        ck = last_checkpoint(ctx)
        state = ck["state_path"] if ck else "<state_path from runs/checkpoints.jsonl>"
        c = py("scripts.export_lora", "--state", state, "--name", t) + (["--merge"] if ctx.merge else [])
        dst = f"/merged/{t}" if ctx.merge else f"/adapters/{t}"
        return [Cmd(c), Cmd(modal("volume", "put", "--force", "qwen-weights", ctx.export_dir, dst))]
    if stage == "deploy_eval":
        env = {"N_REPLICAS": "1"}
        env["MERGED_MODEL" if ctx.merge else "ADAPTERS"] = f"/vol/merged/{t}" if ctx.merge else f"{t}=/vol/adapters/{t}"
        return [Cmd(modal("deploy", "modal_sampler.py"), env)]
    if stage == "eval":
        if not ctx.eval_url:
            raise SystemExit("eval needs --eval-url (the /v1 URL printed by `modal deploy`); set VLLM_KEY as OPENAI_API_KEY")
        return [Cmd(py("scripts.eval", "--name", t, "--problems", ctx.eval_problems, "--base-url", ctx.eval_url,
                       "--model", "qwen" if ctx.merge else t, "--judge", "modal"),
                    {"OPENAI_API_KEY": os.environ["VLLM_KEY"]} if os.environ.get("VLLM_KEY") else {})]
    if stage == "report":
        runs = [ctx.baseline] if os.path.exists(os.path.join(ctx.root, ctx.baseline)) else []
        return [Cmd(py("scripts.analyze", *runs, ctx.eval_out, "--problems", ctx.eval_problems, "--json",
                       os.path.join(ctx.dir, "report.json")))]
    raise KeyError(stage)


def outputs(stage: str, ctx: Ctx) -> list[str]:
    return {"fetch": [ctx.traj], "select": [ctx.selected], "export": [ctx.export_dir],
            "eval": [ctx.eval_out], "report": [ctx.report]}.get(stage, [])


# ------------------------------------------------------------------ state + runner
def load_state(ctx: Ctx) -> dict:
    p = os.path.join(ctx.dir, "state.json")
    return json.load(open(p)) if os.path.exists(p) else {"done": {}}


def save_state(ctx: Ctx, st: dict):
    os.makedirs(ctx.dir, exist_ok=True)
    json.dump(st, open(os.path.join(ctx.dir, "state.json"), "w"), indent=1)


def is_done(stage: str, ctx: Ctx, st: dict) -> bool:
    return stage in st["done"] and all(os.path.exists(o) for o in outputs(stage, ctx))


def default_runner(cmd: Cmd, ctx: Ctx, stdout_path: str | None = None) -> int:
    env = {**os.environ, **cmd.env}
    if stdout_path:
        os.makedirs(os.path.dirname(stdout_path), exist_ok=True)
        with open(stdout_path, "w") as f:
            r = subprocess.run(cmd.argv, env=env, cwd=ctx.root, stdout=f)
        print(open(stdout_path).read())
        return r.returncode
    return subprocess.run(cmd.argv, env=env, cwd=ctx.root).returncode


def run_stages(ctx: Ctx, stages: list[str], execute: bool, force: bool = False,
               runner: Callable = default_runner, log=print) -> bool:
    st = load_state(ctx)
    for s in stages:
        if not force and is_done(s, ctx, st):
            log(f"[{s}] done, skipping")
            continue
        cmds = commands(s, ctx)
        for c in cmds:
            log(f"[{s}] {'$' if execute else '(plan)'} {c.show()}")
        if not execute:
            continue
        for c in cmds:
            if runner(c, ctx, ctx.report if s == "report" else None) != 0:
                log(f"[{s}] FAILED; fix and re-run the same command to resume here")
                return False
        if s == "train" and not last_checkpoint(ctx):
            log("[train] no checkpoint recorded in runs/checkpoints.jsonl")
            return False
        st["done"][s] = True
        save_state(ctx, st)
    return True


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("action", choices=["plan", "run", "status"])
    ap.add_argument("--tag", required=True, help="round name, e.g. r1, r2, variant")
    ap.add_argument("--attempts", type=int, default=4)
    ap.add_argument("--model", help="LoRA adapter to sample from (round 2 samples from the round-1 adapter)")
    ap.add_argument("--pool"); ap.add_argument("--init"); ap.add_argument("--lr", type=float)
    ap.add_argument("--extra-data", nargs="*", default=[], help="more SFT JSONL, e.g. data/pools/r1_hard.jsonl")
    ap.add_argument("--eval-problems", default="data/eval150.jsonl")
    ap.add_argument("--eval-url"); ap.add_argument("--baseline", default="runs/eval/baseline.jsonl")
    ap.add_argument("--replicas", default="3"); ap.add_argument("--merge", action="store_true")
    ap.add_argument("--train-args", default="", help='extra train_sft flags, e.g. "--epochs 1 --batch-size 32"')
    ap.add_argument("--from-stage", choices=STAGES); ap.add_argument("--only", choices=STAGES)
    ap.add_argument("--skip", default="", help="comma-separated stages to skip")
    ap.add_argument("--force", action="store_true"); ap.add_argument("--yes", action="store_true",
                                                                       help="actually execute (spends money)")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    ctx = Ctx(tag=a.tag, attempts=a.attempts, model=a.model, pool=a.pool, init=a.init, lr=a.lr,
              extra_data=a.extra_data, eval_problems=a.eval_problems, eval_url=a.eval_url, baseline=a.baseline,
              replicas=a.replicas, merge=a.merge, train_args=shlex.split(a.train_args))
    stages = [a.only] if a.only else STAGES[STAGES.index(a.from_stage):] if a.from_stage else list(STAGES)
    stages = [s for s in stages if s not in a.skip.split(",")]
    if a.action == "status":
        st = load_state(ctx)
        for s in STAGES:
            print(f"{'done   ' if is_done(s, ctx, st) else 'pending'} {s}")
        return 0
    ok = run_stages(ctx, stages, execute=a.action == "run" and a.yes and not a.dry_run, force=a.force)
    if a.action == "run" and not a.yes:
        print("\n(plan only: pass --yes to execute)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
