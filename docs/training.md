# Track C: sampling and training

Pipeline per round: `modal_sampler.py` (vLLM) -> `scripts/sample.py` -> `scripts/select_traj.py` -> `scripts/train_sft.py` (Tinker) -> `scripts/export_lora.py` -> back to the sampler for the next round / evals.

## Sampler (modal_sampler.py)
- `uv run --env-file .env modal run modal_sampler.py::download` (weights to volume `qwen-weights:/qwen3.5-9b`, skipped if present).
- `N_REPLICAS=3 uv run --env-file .env modal deploy modal_sampler.py` starts nothing until a request arrives; each replica is one H100 (vLLM, native chat template, `--reasoning-parser qwen3`, tool parser `qwen3_coder`, prefix caching). Modal load-balances one URL across replicas.
- Server model name is `qwen`. LoRA: `ADAPTERS="r1=/vol/adapters/r1"` adds `--enable-lora`; request `model="r1"`. Merged: `MERGED_MODEL=/vol/merged/r1`.
- Auth: `Authorization: Bearer $VLLM_KEY` (derived from the Modal token secret unless VLLM_KEY set).
- Throughput benchmark (random long generations, 2 H100 replicas, 384 concurrent, 4k max tokens): ~7.6k tok/s aggregate, i.e. ~3.8k tok/s per H100 (128 concurrent: 5.2k). Plan assumed 5k/GPU; budget ~25% more GPU time (72M tokens/round is about 5 GPU-hours).
- ALWAYS `uv run --env-file .env modal app stop qwen-sampler --yes` after a run.

## Sampling
`uv run --env-file .env modal run scripts/sample.py --tag r1 --attempts 4` reads `/vol/pool.jsonl`, writes `/vol/trajectories/<tag>.jsonl` (resumable). Needs Track A's `cpp.harness.run_episode(problem, client, ...)`; `--stub` exercises the plumbing without it.

## Selection
`python -m scripts.select_traj data/trajectories/r1.jsonl --out data/selected_r1.jsonl` (rules in the file docstring; also writes pass-rate stats for rebalancing).

## Training (Tinker)
`python -m scripts.train_sft --name r1 --data data/selected_r1.jsonl` (variant `--lr 5e-5`). Rank 32, batch 32, max 32k, 1 epoch, cosine lr with warmup.
- Rendering: tinker `qwen3_5_preserve_thinking` renderer. Verified token-identical to `tokenizer.apply_chat_template` (thinking on, tools) for a tool-using trajectory (only a trailing newline differs). The plain `qwen3_5` renderer drops thinking on earlier assistant turns when rendering a whole conversation, so it must not be used for one-datum-per-trajectory training.
- Loss: assistant tokens only (thinking, tool calls, EXIT). One datum per trajectory when the only user message is the task prompt; otherwise one datum per assistant turn.
- Examples over `--max-length` are dropped, not truncated. Each checkpoint is saved with `save_weights_for_sampler` (tinker:// sampler path, for Tinker sampling) and `save_state` (needed for export); both are logged to `runs/checkpoints.jsonl`.
- Billing of masked tokens: Tinker exposes no usage API, so this could not be measured from code. Treat all tokens in the sequence (prompt plus masked) as billed; the cost line printed at start does this.

## Export (vLLM)
`APP_NAME=<app> python -m scripts.export_lora --state <tinker://.../weights/NAME> --name r1 --merge` (state path, not sampler path; sampler checkpoints cannot be downloaded). Runs entirely on Modal (`modal_sampler.build`): downloads the checkpoint, builds in place next to the base weights. A laptop route was tried and is slow (380 MB upload and an 18 GB base-model download).
- PEFT adapter route DOES NOT WORK on Qwen3.5-9B with current vLLM: startup fails with "expected target modules {in_proj_qkv, q_proj, ...} but received ...linear_attn.in_proj_q/k/v" (Tinker trains split q/k/v; the cookbook README claimed it worked for 4B). Use merged weights.
- Merged route verified: `MERGED_MODEL=/vol/merged/smoke` serves under the same flags and returns proper tool_calls (12/12).
- Deploy-time env (MERGED_MODEL, ADAPTERS, MAX_LEN) is baked into the image env; set `APP_NAME=...` to deploy an experiment without touching the main `qwen-sampler`.
- Unclosed-think check: on a trivial prompt the reasoning parser plus qwen3_coder tool parser returned structured tool_calls for 12/12 turns. Re-check the malformed rate on real episodes.
