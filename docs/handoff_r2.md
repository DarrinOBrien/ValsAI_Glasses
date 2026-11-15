# Round 2 handoff (2026-11-15, ~02:50 PST)

You are running round 2 (r2) of expert iteration, sampled from the r1c checkpoint. Approved by the human (Hussein).
Another session (`valsai-glasses-7b`) owns the leaderboard: it submits settings runs and will submit r2 for you.
**Do not touch hackathon.vals.ai.** When r2 passes its check, send its sampler_path to `valsai-glasses-7b` with SendMessage.

## Budget and rules (hard limits)

- **Modal: ~$25 for r2 sampling.** 4 x H100 is ~$16/h, so stop sampling at **90 minutes** wall-clock even if unfinished
  (partial trajectories are kept on the volume). Modal balance was ~$95.
- **Tinker: total cap $60** (raised by the human). Already spent ~$36 (r1/v1/r1b/r1c training + checks). r2 training plus
  its check must stay under **$24**. Run `--dry-run` first; if the estimate is over $18, cut data (`--max-length 20000`)
  or stop and ask the human.
- After EVERY GPU step: `uv run --env-file .env modal app list` and stop anything idle. Only `tinker-cpp` (the judge,
  CPU) may stay deployed. A forgotten app costs ~$4/h per GPU.
- No GRPO, no extra variants, no leaderboard submissions, no commits of data/runs. Never print `.env` or keys.
- Deadline: final leaderboard pick by **1 pm PT**; r2 should be checked by ~8 am so it can be scored.
- If anything unexpected happens (errors, costs off, results odd), stop and ask the human.

## Known facts

- r1c: sampler `tinker://3f4f016f-8fef-5c48-a355-55a23c736581:train:0/sampler_weights/r1c-ep0`,
  state `tinker://3f4f016f-8fef-5c48-a355-55a23c736581:train:0/weights/r1c-ep0` (also in `runs/checkpoints.jsonl`).
  Leaderboard 19.51 (base ~20). Format fixed; solving ability ~ base.
- r1c data recipe (commit 14d7410): `select_traj` default `--exit-mode cut`, `thinkfix normalize`, `train_sft --max-length 24000`.
  93 trajectories in `data/selected_r1c.norm.jsonl`.
- Direct-Tinker check on r1c (first 12 of eval150, 15 turns): 12k tok/t0.8 5/12; **24k/t0.8 6/12**; 24k/t0.6 4/12.
  More output tokens halves truncation; lower temperature hurts. So sample r2 at 24k tokens, temperature 0.8.
- `pool_r1.jsonl` (qwen-weights volume) has 550 problems; r1 sampled only the first 248 (`data/r1_full.jsonl`).
  The other **302 were never sampled** -> r2 pool. None overlap eval150.
- Merged r1c does not exist yet (`merged/` has only r1, v1). Merged weights behave like Tinker (verified by the lead on r1).

## Steps

Run from the repo root. Prefix Modal commands with `uv run --env-file .env`.

1. **Pool** (local, no cost):
   ```
   uv run --env-file .env modal volume get --force qwen-weights pool_r1.jsonl data/pool_r1.jsonl
   python3 -c "
   import json
   seen={json.loads(l)['problem_id'] for l in open('data/r1_full.jsonl')}
   rows=[l for l in open('data/pool_r1.jsonl') if json.loads(l)['id'] not in seen]
   open('data/pool_r2.jsonl','w').writelines(rows); print(len(rows),'problems')"
   uv run --env-file .env modal volume put --force qwen-weights data/pool_r2.jsonl /pool_r2.jsonl
   ```
   Expect 302.
2. **Export r1c merged** (CPU on Modal, ~15 min). The `build` function lives in the `qwen-sampler` app, so deploy it
   first (base weights, scales from 0, no GPU until a request):
   ```
   N_REPLICAS=1 uv run --env-file .env modal deploy modal_sampler.py
   uv run --env-file .env python -m scripts.export_lora --state tinker://3f4f016f-8fef-5c48-a355-55a23c736581:train:0/weights/r1c-ep0 --name r1c --merge
   ```
   Check: `modal volume ls qwen-weights merged/r1c` shows 4 safetensors shards + config.json.
3. **Serve merged r1c** on 4 replicas:
   ```
   MERGED_MODEL=/vol/merged/r1c N_REPLICAS=4 uv run --env-file .env modal deploy modal_sampler.py
   ```
4. **Sample** (note the start time; hard stop at 90 min):
   ```
   uv run --env-file .env modal run scripts/sample.py --tag r2 --attempts 4 --pool /vol/pool_r2.jsonl --max-tokens 24000 > runs/logs/r2.log 2>&1
   ```
   Run it in the background and watch `runs/logs/r2.log` (lines like `120/1208 solved=.. 26.2min`). At 90 min, or when done:
   ```
   uv run --env-file .env modal app stop qwen-sampler --yes
   uv run --env-file .env modal app stop qwen-sampling-runs --yes
   uv run --env-file .env modal app list
   ```
   Kill the local `modal run` process too if it is still running.
5. **Fetch, select, normalize, merge with r1c data:**
   ```
   mkdir -p data/trajectories
   uv run --env-file .env modal volume get --force qwen-weights trajectories/r2.jsonl data/trajectories/
   uv run python -m scripts.select_traj data/trajectories/r2.jsonl --out data/selected_r2.jsonl
   uv run python -m scripts.thinkfix normalize data/selected_r2.jsonl --out data/selected_r2.norm.jsonl
   cat data/selected_r1c.norm.jsonl data/selected_r2.norm.jsonl > data/selected_r2mix.norm.jsonl
   ```
   Train from the base model on the union (same recipe as r1c, which is validated) rather than stacking on r1c.
6. **Train** (Tinker):
   ```
   set -a; . ./.env; set +a
   uv run python -m scripts.train_sft --name r2 --data data/selected_r2mix.norm.jsonl --max-length 24000 --dry-run
   uv run python -m scripts.train_sft --name r2 --data data/selected_r2mix.norm.jsonl --max-length 24000
   ```
   Respect the $18 dry-run limit above. The sampler_path and state_path land in `runs/checkpoints.jsonl`.
7. **Direct check** (Tinker, no GPUs), same flags as r1c's best local run so it is comparable:
   ```
   uv run python -m scripts.smoke_real --problems data/pool/eval150.jsonl --n 12 --k 1 --model-path <r2 sampler_path> --max-turns 15 --max-tokens 24000 --temperature 0.8 --concurrency 12 --judge modal --save runs/direct_r2.jsonl
   ```
   Bar: r1c got **6/12** here (base 7/12 at 12k). If r2 solves **>= 6/12** with malformed/turn not worse than ~13%,
   send `valsai-glasses-7b` the sampler_path plus the summary (solved, malformed/turn, trunc/turn, exit count).
   If it solves fewer, tell `valsai-glasses-7b` and the human; do not retrain.
8. Final `modal app list` (only tinker-cpp), then report spend: Modal sampling minutes x replicas, Tinker dry-run figure.
