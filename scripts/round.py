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

  deploy_sample  [MERGED_MODEL=/vol/merged/<--model>] N_REPLICAS=3 modal deploy modal_sampler.py   (base weights when no --model)
  sample         modal run scripts/sample.py --tag T --attempts 4 [--pool P]   -> /vol/trajectories/T.jsonl
                 (requests go to model "qwen"; round 2 samples from the merged round-1 weights chosen at deploy time)
  fetch          modal volume get qwen-weights trajectories/T.jsonl data/trajectories/
  stop_sample    modal app stop qwen-sampler --yes
  select         python -m scripts.select_traj data/trajectories/T.jsonl --out data/selected_T.jsonl
  normalize      python -m scripts.thinkfix normalize data/selected_T.jsonl [--extra-data ..] --out data/selected_T.norm.jsonl
                 (closes </think> before tool calls so training turns parse under any reasoning parser)
  train          python -m scripts.train_sft --name T --data data/selected_T.norm.jsonl [--init S] [--lr X]
  deploy_eval    MERGED_MODEL=/vol/merged/T N_REPLICAS=1 modal deploy modal_sampler.py
                 (deployed BEFORE export on purpose: export calls the app's `build` function, so the app must exist; the
                 vLLM server only reads /vol/merged/T when its first request starts a container, i.e. after export)
  export         python -m scripts.export_lora --state <state_path from runs/checkpoints.jsonl> --name T --merge
                 (uploads the raw Tinker adapter, builds the MERGED bf16 model at qwen-weights:/merged/T on Modal)
  eval           python -m scripts.eval --name T --problems data/pool/eval150.jsonl --base-url <url>/v1 --model qwen --judge modal
  stop_eval      modal app stop qwen-sampler --yes
  report         python -m scripts.analyze <baseline> runs/eval/T.jsonl --problems data/pool/eval150.jsonl > runs/rounds/T/report.md

MERGED PATH (default). vLLM rejects the Tinker PEFT adapter (split linear_attn in_proj q/k/v vs fused modules), so every
checkpoint is served merged: model name is always "qwen", the checkpoint is chosen with MERGED_MODEL at deploy time.
--adapter switches to the legacy PEFT/ADAPTERS path (kept only for experiments).

VARIANT (same data, lower lr):  python -m scripts.round run --tag v1 --lr 5e-5 --reuse-data r1 --yes   (train onward)

ROUND 2:  --model r1 = the round-1 checkpoint to SAMPLE FROM (its merged weights must already be at /vol/merged/r1, i.e.
round 1's export stage ran). --init = the Tinker `state_path` to CONTINUE TRAINING from; if omitted it is looked up from
runs/checkpoints.jsonl (last record named like --model). The new run T is exported and evaluated as its own merged model.
    uv run python -m scripts.round run --tag r2 --yes --model r1 --pool /cpp/pool/pool_r2.jsonl
Entering at `--from-stage export` needs the qwen-sampler app deployed (run `--only deploy_eval` first).

The Modal URL is auto-discovered (modal.Function.from_name("qwen-sampler","serve").get_web_url()) when executing; pass
--eval-url to override. VLLM_KEY is read from .env and used as the eval client's API key. `plan` runs nothing.
"""
from __future__ import annotations

import argparse, json, os, shlex, subprocess, sys
from dataclasses import dataclass, field
from typing import Callable

STAGES = ["deploy_sample", "sample", "fetch", "stop_sample", "select", "normalize", "train", "deploy_eval", "export",
          "eval", "stop_eval", "report"]


@dataclass
class Cmd:
    argv: list[str]
    env: dict = field(default_factory=dict)

    def show(self) -> str:
        e = " ".join(f"{k}={'***' if 'KEY' in k else shlex.quote(v)}" for k, v in self.env.items())  # never print secrets
        return (e + " " if e else "") + " ".join(shlex.quote(x) for x in self.argv)


@dataclass
class Ctx:
    tag: str
    attempts: int = 4
    model: str | None = None        # checkpoint name to sample from (round >= 2: merged at /vol/merged/<model>); None = base
    pool: str | None = None
    init: str | None = None         # tinker:// state to continue from (round >= 2)
    lr: float | None = None
    extra_data: list[str] = field(default_factory=list)   # e.g. R1 traces from build_r1
    eval_problems: str = "data/pool/eval150.jsonl"
    eval_url: str | None = None
    baseline: str = "runs/eval/baseline.jsonl"
    replicas: str = "3"
    merge: bool = True              # serve merged weights (default); False = legacy PEFT adapter path
    max_empty_frac: float = 1.0
    train_args: list[str] = field(default_factory=list)
    root: str = "."
    data_tag: str | None = None     # reuse another round's selected/normalized data (variant runs: same data, other lr)

    @property
    def dir(self): return os.path.join(self.root, "runs", "rounds", self.tag)
    @property
    def traj(self): return os.path.join(self.root, "data", "trajectories", f"{self.tag}.jsonl")
    @property
    def selected(self): return os.path.join(self.root, "data", f"selected_{self.data_tag or self.tag}.jsonl")
    @property
    def normalized(self): return os.path.join(self.root, "data", f"selected_{self.data_tag or self.tag}.norm.jsonl")
    @property
    def eval_out(self): return os.path.join(self.root, "runs", "eval", f"{self.tag}.jsonl")
    @property
    def report(self): return os.path.join(self.dir, "report.md")
    @property
    def export_marker(self):  # local proof the raw adapter was downloaded; the merged model itself lives on the Modal volume
        return os.path.join(self.root, "exports", self.tag, "raw", "adapter_model.safetensors")

    def init_state(self) -> str | None:
        """tinker:// state to continue training from: --init, else the last checkpoint recorded for --model."""
        if self.init:
            return self.init
        if self.model:
            ck = last_checkpoint(self, self.model)
            return ck["state_path"] if ck else f"<state_path of {self.model} from runs/checkpoints.jsonl>"
        return None


def last_checkpoint(ctx: Ctx, name: str | None = None) -> dict | None:
    """Most recent record for a run name (default: this round's) in runs/checkpoints.jsonl (written by train_sft)."""
    p = os.path.join(ctx.root, "runs", "checkpoints.jsonl")
    if not os.path.exists(p):
        return None
    recs = [json.loads(l) for l in open(p) if l.strip()]
    recs = [r for r in recs if r.get("name") == (name or ctx.tag)]
    return recs[-1] if recs else None


def py(*a): return ["uv", "run", "python", "-m", *a]
def modal(*a): return ["uv", "run", "--env-file", ".env", "modal", *a]


def served_env(ctx: Ctx, name: str | None, replicas: str) -> dict:
    """Env for `modal deploy modal_sampler.py`: merged checkpoint (default), legacy adapter, or base weights."""
    env = {"N_REPLICAS": replicas}
    if name:
        if ctx.merge:
            env["MERGED_MODEL"] = f"/vol/merged/{name}"
        else:
            env["ADAPTERS"] = f"{name}=/vol/adapters/{name}"
    return env


def serve_model_name(ctx: Ctx, name: str | None) -> str:
    return name if (name and not ctx.merge) else "qwen"   # merged / base servers always answer as "qwen"


def commands(stage: str, ctx: Ctx) -> list[Cmd]:
    t = ctx.tag
    if stage == "deploy_sample":
        return [Cmd(modal("deploy", "modal_sampler.py"), served_env(ctx, ctx.model, ctx.replicas))]
    if stage == "sample":
        c = modal("run", "scripts/sample.py", "--tag", t, "--attempts", str(ctx.attempts))
        if serve_model_name(ctx, ctx.model) != "qwen": c += ["--model", serve_model_name(ctx, ctx.model)]
        if ctx.pool: c += ["--pool", ctx.pool]
        return [Cmd(c)]
    if stage == "fetch":
        return [Cmd(modal("volume", "get", "--force", "qwen-weights", f"trajectories/{t}.jsonl", "data/trajectories/"))]
    if stage in ("stop_sample", "stop_eval"):
        return [Cmd(modal("app", "stop", "qwen-sampler", "--yes"))]
    if stage == "select":
        return [Cmd(py("scripts.select_traj", ctx.traj, "--out", ctx.selected))]
    if stage == "normalize":
        c = py("scripts.thinkfix", "normalize", ctx.selected, *ctx.extra_data, "--out", ctx.normalized)
        if ctx.max_empty_frac < 1.0: c += ["--max-empty-frac", str(ctx.max_empty_frac)]
        return [Cmd(c)]
    if stage == "train":
        c = py("scripts.train_sft", "--name", t, "--data", ctx.normalized)
        if ctx.init_state(): c += ["--init", ctx.init_state()]
        if ctx.lr: c += ["--lr", str(ctx.lr)]
        return [Cmd(c + ctx.train_args)]
    if stage == "deploy_eval":   # before export: export needs the app's `build` function; weights are read lazily
        return [Cmd(modal("deploy", "modal_sampler.py"), served_env(ctx, t, "1"))]
    if stage == "export":
        ck = last_checkpoint(ctx)
        state = ck["state_path"] if ck else "<state_path from runs/checkpoints.jsonl>"
        return [Cmd(py("scripts.export_lora", "--state", state, "--name", t) + (["--merge"] if ctx.merge else []))]
    if stage == "eval":
        url = ctx.eval_url or "<serve-url>/v1"
        return [Cmd(py("scripts.eval", "--name", t, "--problems", ctx.eval_problems, "--base-url", url,
                       "--model", serve_model_name(ctx, t), "--judge", "modal"),
                    {"OPENAI_API_KEY": os.environ["VLLM_KEY"]} if os.environ.get("VLLM_KEY") else {})]
    if stage == "report":
        runs = [ctx.baseline] if os.path.exists(os.path.join(ctx.root, ctx.baseline)) else []
        return [Cmd(py("scripts.analyze", *runs, ctx.eval_out, "--problems", ctx.eval_problems, "--json",
                       os.path.join(ctx.dir, "report.json")))]
    raise KeyError(stage)


def outputs(stage: str, ctx: Ctx) -> list[str]:
    return {"fetch": [ctx.traj], "select": [ctx.selected], "normalize": [ctx.normalized], "export": [ctx.export_marker],
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


def discover_url() -> str:
    import modal
    return modal.Function.from_name("qwen-sampler", "serve").get_web_url().rstrip("/") + "/v1"


def run_stages(ctx: Ctx, stages: list[str], execute: bool, force: bool = False,
               runner: Callable = default_runner, log=print) -> bool:
    st = load_state(ctx)
    for s in stages:
        if not force and is_done(s, ctx, st):
            log(f"[{s}] done, skipping")
            continue
        if execute and s == "eval" and not ctx.eval_url:
            ctx.eval_url = discover_url()
            log(f"[eval] serve url: {ctx.eval_url}")
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
    ap.add_argument("--model", help="round-1 checkpoint name to sample from (merged at /vol/merged/<name>); omit for base")
    ap.add_argument("--pool"); ap.add_argument("--lr", type=float)
    ap.add_argument("--init", help="tinker:// state to continue training from (default: last checkpoint of --model)")
    ap.add_argument("--reuse-data", metavar="TAG", help="variant run: train on TAG's already-normalized data (skips "
                    "sample/select/normalize), e.g. --tag v1 --lr 5e-5 --reuse-data r1")
    ap.add_argument("--extra-data", nargs="*", default=[], help="more SFT JSONL, e.g. data/pools/r1_hard.jsonl")
    ap.add_argument("--eval-problems", default="data/pool/eval150.jsonl")
    ap.add_argument("--eval-url"); ap.add_argument("--baseline", default="runs/eval/baseline.jsonl")
    ap.add_argument("--max-empty-frac", type=float, default=1.0,
                    help="drop selected trajectories whose tool-call turns are mostly reasoning-free")
    ap.add_argument("--replicas", default="3"); ap.add_argument("--adapter", action="store_true", help="legacy: PEFT adapter + ADAPTERS (vLLM rejects the Tinker adapter)")
    ap.add_argument("--train-args", default="", help='extra train_sft flags, e.g. "--epochs 1 --batch-size 32"')
    ap.add_argument("--from-stage", choices=STAGES); ap.add_argument("--only", choices=STAGES)
    ap.add_argument("--skip", default="", help="comma-separated stages to skip")
    ap.add_argument("--force", action="store_true"); ap.add_argument("--yes", action="store_true",
                                                                       help="actually execute (spends money)")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    try:
        from dotenv import load_dotenv
        load_dotenv(".env")   # VLLM_KEY for the eval client
    except ImportError:
        pass
    ctx = Ctx(tag=a.tag, attempts=a.attempts, model=a.model, pool=a.pool, init=a.init, lr=a.lr,
              extra_data=a.extra_data, eval_problems=a.eval_problems, eval_url=a.eval_url, baseline=a.baseline,
              replicas=a.replicas, merge=not a.adapter, data_tag=a.reuse_data, max_empty_frac=a.max_empty_frac, train_args=shlex.split(a.train_args))
    stages = [a.only] if a.only else STAGES[STAGES.index(a.from_stage):] if a.from_stage else list(STAGES)
    if a.reuse_data and not a.only and not a.from_stage:
        stages = stages[STAGES.index("train"):]
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
