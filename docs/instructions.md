# Handoff instructions

Branch: `LoRA` (origin `ValsAI_Glasses.git`). Plan: `Plan Qwen3.5-9B for C++ (expert iteration + LoRA).md`. Reference: `README.md`, `docs/data.md`, `docs/training.md`, `docs/judge_noise.md`. Submission log: `submissions.md`.

Everything follows the NEW plan only: Qwen3.5-9B, thinking on, expert iteration on its own trajectories, LoRA SFT on Tinker, sampling/evals on Modal vLLM. No teacher model, no P1/P2 pools.

## Rules for both

- Hard rules: never train on the open-r1 `test` split (`eval150.jsonl` is eval-only), no OpenAI/Anthropic-generated traces, at most 9B params.
- Budget: Tinker $100, Modal $200. Spend only what the step says. Report spend after each GPU job.
- **ALWAYS stop GPUs when done:** `uv run --env-file .env modal app stop qwen-sampler --yes`.
- Secrets live in `.env` (gitignored, copy by hand): `TINKER_API_KEY`, `MODAL_TOKEN_ID`, `MODAL_TOKEN_SECRET`, `MODAL_PROFILE`, `HF_ACCESS_TOKEN`. **Set the same `VLLM_KEY=<any string>` on both machines**, so the server's key and the eval client's key match. Without it the key is derived from `MODAL_TOKEN_SECRET`.
- Run Modal as `uv run --env-file .env modal ...`. Python scripts as `uv run python -m scripts.<name>`.
- Never `git push --force`. Never commit `.env`, `runs/`, `data/` or `exports/`. Commit small, push to `LoRA` (or a branch off it and tell the lead).
- Only the other computer launches GPU jobs on the `qwen-sampler` Modal app. Agent C may also deploy that app for its vLLM LoRA test. Run `uv run --env-file .env modal app list` first and ask the lead if it is busy.

---

## Part 1: The other computer (GPU work)

Goal: get the baseline score of the untouched Qwen3.5-9B through our harness, then pilot sampling.

### 1.1 Setup
```bash
git clone https://github.com/DarrinOBrien/ValsAI_Glasses.git && cd ValsAI_Glasses
git checkout LoRA
# copy .env in by hand, add VLLM_KEY=...
uv sync
uv run --with pytest python -m pytest tests -q     # expect all pass or skip; sandbox tests skip without g++
uv run --env-file .env modal run scripts/build_pool.py::pull   # copies pool.jsonl, eval150.jsonl to ./data/pool/
ls data/pool                                                    # pool.jsonl eval150.jsonl ...
```
The Modal judge `tinker-cpp` is already deployed and mounts the test data from the `cpp-data` volume, so `full_tests` never needs to be downloaded.

### 1.2 Baseline eval (do this first)
```bash
set -a; . ./.env; set +a
N_REPLICAS=1 uv run modal deploy modal_sampler.py     # note the printed serve URL
URL=$(uv run python -c "import modal;print(modal.Function.from_name('qwen-sampler','serve').get_web_url())")
curl -sL "$URL/health"                                 # poll until 200; first start takes several minutes (303 meanwhile)
uv run python -m scripts.eval --name baseline --problems data/pool/eval150.jsonl \
    --base-url "$URL/v1" --api-key "$VLLM_KEY" --model qwen --judge modal --n 150 --k 1 --concurrency 16
uv run --env-file .env modal app stop qwen-sampler --yes
```
Output: `runs/eval/baseline.jsonl` and a line in `runs/eval/summary.jsonl`. Expected pass rate around 20% (staff figure). Start with `--n 10` if you want a cheap trial first.

Report back:
- pass rate, malformed-call rate, truncation rate, avg turns, avg tokens
- **malformed rate vs ~2% seen in the Tinker smoke test.** A much higher rate means vLLM's reasoning/tool parser mishandles tool calls inside an unclosed `<think>` block. That is the key open risk.
- actual tokens/s and cost

### 1.3 Sampling pilot (only if 1.2 looks sane)
```bash
N_REPLICAS=2 uv run modal deploy modal_sampler.py
uv run --env-file .env modal run scripts/sample.py --tag pilot --attempts 4 --limit 20
uv run --env-file .env modal volume get qwen-weights trajectories/pilot.jsonl data/
uv run python -m scripts.select_traj data/pilot.jsonl --out data/pilot_selected.jsonl
uv run python -m scripts.thinkfix stats data/pilot.jsonl
uv run --env-file .env modal app stop qwen-sampler --yes
```
`sample.py` has never run live against the real `run_episode`; expect to find and fix integration bugs here. Fix them on a branch off `LoRA` and tell the lead.

Report: attempts/solved, tokens/s, how many problems got a selected trajectory, unclosed-think rate, any error patterns.

### 1.4 Do NOT run yet
- Full round 1 (`scripts.round run --tag r1 --yes`, about $16-20 of sampling). Wait for the lead after the baseline and pilot.
- Any Tinker training. Wait for the lead.
- Do not copy `runs/eval/baseline.jsonl` into git. Send it to the lead out of band, or push it to a `runs-data` branch if the lead says so.

---

## Part 2: The other window on the main machine (CPU work and analysis)

Goal: verify the R1 hard-problem traces on real data, and analyze baseline results when they arrive. No GPU jobs.

### 2.1 Setup
```bash
cd <your worktree>
git fetch origin && git rebase origin/LoRA     # or: git checkout -b analysis-2 origin/LoRA
uv sync
```

### 2.2 R1 hard-problem traces on real data
Inputs already built by Track B: `cpp-data:/pool/r1_hard.jsonl` (200 problems, DeepSeek-R1 C++ traces, all in the pool) and `r1_hard_all.jsonl`. Pull them, then rewrite and verify with the real sandbox:
```bash
uv run --env-file .env modal run scripts/build_pool.py::pull    # puts jsonl files in ./data/pool/
uv run python -m scripts.build_r1 --problems data/pool/pool.jsonl --traces data/pool/r1_hard_all.jsonl \
    --exclude data/pool/eval150.jsonl --out data/pools/r1_hard.jsonl --workers 8
```
Check `--help` and `scripts/build_r1.py` for how `--tests-root` and Modal judging interact. Full tests are on the volume (`pool/tests/<id>.json`), so the verification step needs the Modal judge or a mounted volume rather than local files. Adapt the script if needed (for example a `--judge modal` path).

Report: how many traces survive (compile, pass full tests, under 0.6x time limit), reasons for drops, the rating mix of survivors, mean token length. Never use ids from `eval150.jsonl`.

### 2.3 Analysis of the baseline (when the other computer sends `runs/eval/baseline.jsonl`)
```bash
uv run python -m scripts.analyze baseline=runs/eval/baseline.jsonl --problems data/pool/eval150.jsonl --json runs/baseline_report.json
uv run python -m scripts.thinkfix stats runs/eval/baseline.jsonl
```
Report: pass rate by rating bucket, top failure categories, unclosed-think rate, and a recommended `--max-empty-frac` value for `thinkfix normalize`.

### 2.4 Small fixes you may do
- The README "Checkers" gotcha is stale: checkers are now Python (`checker_lang`), already supported by `cpp/judge.py`.
- `README.md` Modal volume note about the pool path: `sample.py` now reads `/cpp/pool/pool.jsonl` from the `cpp-data` volume.

### 2.5 Do NOT
- Launch any Modal GPU app or Tinker training. GRPO waits for a round-2 checkpoint.
- Edit C's files (`modal_sampler.py`, `scripts/sample.py`, `select_traj.py`, `train_sft.py`, `export_lora.py`) without telling the lead.

---

## Still in flight (lead's side)

- Agent C: LoRA export plus the vLLM LoRA load test and the unclosed-think check against vLLM's reasoning parser. Its new commits will be merged and pushed to `LoRA`; `git pull` before starting new work.
- Next after baseline plus pilot: full round 1 sampling, SFT on Tinker, local eval, submit. Record each submission in `submissions.md`.
