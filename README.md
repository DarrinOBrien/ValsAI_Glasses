# Tinker C++ post-training

Plan: `Tinker C++ Post-Training Plan.pdf`. Code: `cpp/` (harness, policy, judge, problems), `scripts/` (P1 build, SFT, eval), `modal_app.py` (CPU jobs), `modal_teacher.py` (teacher GPU server + P2 generation).

## Setup

- `uv sync`, then load `.env` (gitignored, copy it by hand): `set -a; . ./.env; set +a`.
- Needs `TINKER_API_KEY`, `MODAL_TOKEN_ID`, `MODAL_TOKEN_SECRET`, `MODAL_PROFILE`. Modal workspace: `darrin-obrien`.
- Run Modal commands as `uv run modal ...`. There is no global `modal` binary.

## Teacher (P2): Qwen3.5-122B-A10B FP8 on Modal

vLLM 2xH100 (tensor-parallel 2), OpenAI-compatible, app name `tinker-cpp-teacher`.

```
uv run modal deploy modal_teacher.py                        # deploy; GPUs only spin up on the first request
uv run modal run modal_teacher.py::gen --n 20 --tag smoke   # generate trajectories through the harness
uv run modal volume get cpp-data pools/p2_smoke.jsonl data/pools/
uv run modal app stop tinker-cpp-teacher                    # stop paying (~$8/hr while a container is up)
```

Notes learned the hard way:

- **GPUs need a card on file.** Modal rejects every GPU type (even a T4) on a workspace without a payment method, regardless of credit balance. This was resolved on 2026-11-14; if you see `Please add a payment method to use ... GPU functions`, that is the cause.
- **First start takes ~20 min** (weight load, torch.compile, FlashInfer JIT, CUDA graphs). Later starts reuse the volume caches. The container scales down 1 hour after the last request.
- **Cold-start 303s.** While the container starts, Modal answers with `303 See Other`. `urllib` cannot follow that for a POST; use `httpx`/`requests`/`curl -L`, or poll `GET /health` until 200 first.
- **Thinking model.** The reasoning parser (`qwen3`) puts chain of thought in `reasoning`/`reasoning_content`; `content` is `None` until thinking ends. Use a large `max_tokens` (thousands) or you will get no answer.
- **Smoke test** (works once healthy):
  ```
  curl -sL "$URL/v1/chat/completions" -H 'Content-Type: application/json' \
    -d '{"model":"qwen3.5-122b","max_tokens":4000,"messages":[{"role":"user","content":"Fibonacci mod 1e9+7 in C++20, O(log n). Brief."}]}'
  ```
- **Check the file vs. what is deployed.** The endpoint deployed and verified on 2026-11-14 was the earlier version of `modal_teacher.py`: URL `https://darrin-obrien--tinker-cpp-teacher-serve.us-east.modal.run`, model name `qwen3.5-122b`, no API key, 64k context. The current file was since rewritten (key derived from `MODAL_TOKEN_SECRET` unless `VLLM_KEY` is set, served model name `teacher`, `MAX_LEN = 40000`, volumes `hf-cache`/`cpp-data`). A redeploy picks up the current file and has not been smoke-tested. Run `gen --n 20 --tag smoke` after redeploying.

## Fallbacks

If the 122B passes under ~15% of hard problems after its first 2 hours, switch teachers per the plan: GLM-5.3-Flash, DeepSeek-V4.1-Flash or Qwen3.5-397B on Tinker (never OpenAI or Anthropic models). A cheaper Modal option is Qwen3.5-35B-A3B on one H100.
