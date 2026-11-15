# Qwen3.5-9B for C++: expert iteration + LoRA

Plan: `Plan Qwen3.5-9B for C++ (expert iteration + LoRA).md` (scope, budget, targets, steps). Submission log: `submissions.md`.
We start from the post-trained `Qwen/Qwen3.5-9B` (thinking on), sample it in our copy of the harness, keep its own shortest
fully-passing trajectories, and LoRA-fine-tune on them for 2 rounds (+ a variant run, optional GRPO). Sampling and evals run on
Modal (vLLM), training on Tinker. Rules: at most 9B active params, no OpenAI/Anthropic teacher traces, never train on the scoring
problems (the open-r1 `test` split is eval-only).

## Setup

- `uv sync`. Secrets live in `.env` (gitignored; copy it by hand into each checkout/worktree): `TINKER_API_KEY`,
  `MODAL_TOKEN_ID`, `MODAL_TOKEN_SECRET`, `MODAL_PROFILE`, `HF_ACCESS_TOKEN`; optional `VLLM_KEY` (else derived from the Modal secret).
- Run Modal as `uv run --env-file .env modal ...` (no global `modal`). Modal GPUs need a payment method on file.
- g++ must be on PATH for local judging (`cpp/judge.py`; the Modal judge runs the same code on CPU containers).
- Tests: `uv run --with pytest python -m pytest tests -q` (sandbox tests skip without g++).

## Layout

| Path | What |
| --- | --- |
| `cpp/tools.py` | **Everything the scoring harness controls**: tool schema, system/task prompt, tool-result text. Edit here if staff give the real values, then regenerate data |
| `cpp/harness.py` | `run_episode(problem, client, ...)` -> `Trajectory`; OpenAI-compatible client, `HarnessConfig`, local/Modal judge |
| `cpp/judge.py`, `modal_app.py` | g++ sandbox, Python (open-r1) and testlib-C++ checkers; Modal CPU `judge_remote` |
| `cpp/policy.py` | Renderer + system prefix for Tinker. Must render prompts identically to the HF chat template (checked by `smoke_real`) |
| `scripts/build_pool.py` | Problem pool, eval150 holdout and R1 hard set on Modal volumes |
| `modal_sampler.py` | Qwen3.5-9B vLLM server on Modal (N H100 replicas, optional LoRA / merged model) |
| `scripts/sample.py`, `select_traj.py` | Expert-iteration sampling (4 attempts/problem) and shortest-passing selection |
| `scripts/train_sft.py`, `export_lora.py` | LoRA SFT on Tinker; export a checkpoint for vLLM |
| `scripts/eval.py` | Local eval through the harness -> `runs/eval/<name>.jsonl` |
| `scripts/analyze.py` | Pass rate by rating bucket, failure categories, turn/token distributions, checkpoint comparison |
| `scripts/round.py` | One resumable command per round (sample -> select -> train -> export -> eval -> report) |
| `scripts/build_r1.py` | R1 (codeforces-cots) hard-problem traces rewritten into harness format, verified in the sandbox |
| `scripts/grpo.py` | Optional GRPO round on Tinker + medium-difficulty filter |
| `scripts/thinkfix.py` | Unclosed-`<think>` stats + SFT normalizer/filter (run by `round.py` between select and train) |
| `scripts/judge_noise.py`, `docs/judge_noise.md` | Judge false-TLE benchmark on Modal and its results |
| `scripts/smoke_real.py` | Real-model harness smoke test + training-vs-serving template check |
| `docs/training.md` | Track C notes (sampler throughput, renderer choice, export) |

## Modal volumes

| Volume | Contents |
| --- | --- |
| `cpp-data` (`/vol`) | `cf_raw/` raw open-r1 parquet, `cf_tests/` generated tests for kept contests, `pool/{pool.jsonl,eval150.jsonl,stats.json,tests/<id>.json}` |
| `qwen-weights` (`/vol`) | `qwen3.5-9b/` base weights, `adapters/<name>` and `merged/<name>` exports, `trajectories/<tag>.jsonl` sampling output |

`sample.py` reads the pool from `/cpp/pool/pool.jsonl` (the `cpp-data` volume) by default; pass `--pool` for anything else. ALWAYS stop GPUs after a run: `uv run --env-file .env modal app stop qwen-sampler --yes`.

## Commands per stage

```bash
# 0. data + weights (CPU / one-off)
uv run --env-file .env modal run scripts/build_pool.py::download
uv run --env-file .env modal run scripts/build_pool.py::weights        # or: modal run modal_sampler.py::download
uv run --env-file .env modal run scripts/build_pool.py::pool           # pool.jsonl, eval150.jsonl, tests
uv run --env-file .env modal run scripts/build_pool.py::pull           # -> ./data/

# 1. sanity check the harness against the real model (cents, via Tinker sampling)
uv run python -m scripts.smoke_real --problems data/syn_pool.jsonl --n 3 --k 2

# 2. baseline local eval (deploy sampler, eval, stop)
N_REPLICAS=1 uv run --env-file .env modal deploy modal_sampler.py
uv run python -m scripts.eval --name baseline --problems data/eval150.jsonl --base-url <URL>/v1 --model qwen --judge modal
uv run --env-file .env modal app stop qwen-sampler --yes

# 3. a whole round in one command (plan first; --yes spends money; re-run to resume)
uv run python -m scripts.round plan --tag r1
uv run python -m scripts.round run  --tag r1 --yes --eval-url <URL>/v1
uv run python -m scripts.round run  --tag r2 --yes --model r1 --init tinker://.../weights/r1-ep0 --pool <round-2 pool> --eval-url <URL>/v1
#    variant: --tag v1 --lr 5e-5 [--extra-data data/pools/r1_hard.jsonl]

# 4. R1 hard-problem traces (after pool is built; verified, real tool outputs)
#    full tests stay on the cpp-data volume: verification runs on the deployed `tinker-cpp` Modal judge (default; `--judge local --tests-root` for local files)
uv run python -m scripts.build_r1 --problems data/pool/pool.jsonl --traces data/pool/r1_hard_all.jsonl \
    --exclude data/pool/eval150.jsonl --max-tokens 30000 --keep-failures --out data/pools/r1_all_full.jsonl
#    (--limit 10 for a trial; drop --keep-failures for a train-ready file; --max-tokens must fit the SFT max length)

# 5. compare checkpoints / decide what to submit
uv run python -m scripts.analyze baseline=runs/eval/baseline.jsonl r1=runs/eval/r1.jsonl v1=runs/eval/v1.jsonl \
    --problems data/eval150.jsonl --json runs/compare.json

# 6. optional GRPO (only if budget remains)
uv run python -m scripts.grpo select --trajs data/trajectories/r1.jsonl --problems data/pool/pool.jsonl --out data/grpo_pool.jsonl
uv run python -m scripts.grpo train --name grpo --init <round-2 state path> --problems data/grpo_pool.jsonl
```

Each stage's flags, outputs and failure behaviour are in the script's docstring (`python -m scripts.<x> --help`).

## Rounds in practice

- **Round 1** samples base Qwen3.5-9B (1,500-problem pool, 4 attempts), selects, trains (LoRA r32, lr 1e-4, 1 epoch), exports, evals on eval150. **Variant**: same data at lr 5e-5, or plus R1 traces; drop R1 traces if local eval gets worse.
- **Round 2** samples from the best of round 1/variant (`--model <adapter>`; if vLLM cannot serve the adapter, `--merge`) and trains from that state (`--init`).
- Record every checkpoint: `train_sft` appends `sampler_path` and `state_path` to `runs/checkpoints.jsonl` (state path is what export and `--init` need).

## Evaluate and submit

1. We do not submit the untouched Qwen3.5-9B. Its local eval on eval150 is the bar; the staff-reported leaderboard baseline is ~20%.
2. Each checkpoint: local eval on eval150, then `scripts.analyze` against the baseline. Targets: pass above baseline, malformed ~0%, truncation < 5%, turns falling round over round. Drop a run that is not above baseline.
3. Submissions: 4 per hour, one processing at a time, edits count. Queue by local eval score; log checkpoint path, settings and score in `submissions.md` (`analyze` prints the "pass / malformed / trunc / turns" cell).
4. Once the best checkpoint is clear, sweep sampling settings (temperature 0.6 vs 0.8, max output tokens, thinking budget) with the remaining slots.
5. Final submission by 1pm Sunday (deadline 2pm).

## Known gotchas

- **Prompt template parity.** Training prompts are built by the Tinker renderer, serving prompts by vLLM's HF template. `smoke_real` compares them token-for-token; tool specs must be the full `{"type":"function","function":...}` objects (fixed in `cpp/policy.py`).
- **Tool calls inside an unclosed `<think>` block**: about 60% of baseline tool-call turns (26/43 in the smoke test) never write `</think>`; the Tinker renderer recovers the call, but the real harness's reasoning parser may not (-> malformed calls). Training rows are therefore normalized by `scripts.thinkfix normalize` (pre-call text moves into the reasoning field, so targets read `<think>..</think><tool_call>`), `round.py` runs it before training, and `analyze` reports an "unclosed think" column - it should fall to ~0 after round 1. `--max-empty-frac` drops trajectories that mostly skip thinking.
- **Checkers**: open-r1 checkers are Python (`checker_lang: "python"`, run as `python checker.py in ans out`, accept = exit 0 and score >= `checker_max_score`); testlib C++ checkers are also supported. A checker timeout is a `JE` verdict, never a pass.
- **Judge timing**: on Modal the judge is quiet (no false TLEs in 18k runs; see `docs/judge_noise.md`), so keep `cpu=2, max_inputs=2` and select with a 0.6 x limit time margin. Local macOS timing is noisy; use Modal for real runs.
- **Cold start**: first vLLM start on a replica takes minutes (weights, compile, CUDA graphs); Modal answers `303` while it starts, so use `httpx`/`curl -L` and poll `/health`.
- Cost guards: all `round.py` spending stages need `--yes`; the `qwen-sampler` GPU app bills by the hour until stopped.
