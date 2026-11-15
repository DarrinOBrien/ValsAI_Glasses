# Plan: Qwen3.5-9B for C++ (expert iteration + LoRA)

Nov 14, 2026 · @Darrin

## Scope

We start from the post-trained **Qwen3.5-9B** (thinking mode) and improve it with expert iteration: sample it in our copy of the harness, keep its own shortest passing attempts, and LoRA-fine-tune on them, for 2 rounds. The model already chats, thinks and calls tools, so the work is adapting it to this harness and these problems, not teaching those skills.

|  | In | Out |
| --- | --- | --- |
| Start | `Qwen/Qwen3.5-9B` (hybrid, thinking on) | Qwen3.5-9B-Base |
| Data | The model's own verified trajectories on open-r1/codeforces problems; a few hundred R1 traces for hard problems | Large-scale R1 distillation, our own teacher model |
| Training | 2 rounds of LoRA SFT, plus 1 variant run; short GRPO only if budget remains | Full fine-tuning |
| Compute | Modal GPUs for sampling and evals (vLLM); Tinker for training | Sampling on Tinker (costs more) |

The main risk is making the model worse than it started, so we compare every run against a local baseline eval of the untouched model (we do not submit the untouched model), and keep every SFT round small and gentle.

**Rules to keep in view:** at most 9B active parameters, no OpenAI or Anthropic teacher traces, never train on the scoring problems. Staff have confirmed that starting from Qwen3.5-9B is allowed.

## Download

The problem pool now matters more than the traces: open-r1/codeforces supplies the problems, tests and ratings we sample on, and codeforces-cots is a small extra for hard problems. Neither requires the 110 GB of generated tests; we fetch only the contests we use.

| Dataset | Config | Rows | Role | Size |
| --- | --- | --- | --- | --- |
| [open-r1/codeforces](https://huggingface.co/datasets/open-r1/codeforces) | `verifiable` (train split) | \~8.8k problems | Main pool: statements, ratings, official tests, checkers | Small (official tests only) |
| [open-r1/codeforces](https://huggingface.co/datasets/open-r1/codeforces) | `verifiable` (test split) | 468 problems | Eval only; the card says not to train on it | Small |
| open-r1/codeforces generated tests | `generated_tests/test_cases_<contest>.parquet` | Per contest | Extra tests where official ones are fewer than 5 | Only what we keep |
| [open-r1/codeforces-cots](https://huggingface.co/datasets/open-r1/codeforces-cots) | `solutions_decontaminated` | 40.7k | R1 traces with C++ solutions, used only for hard problems the model never solves | \~1–2 GB (estimate) |
| `Qwen/Qwen3.5-9B` weights | — | — | Served with vLLM on Modal for sampling | \~18 GB (bf16) |

**Commands** (Modal CPU container, saving to a Modal Volume)

```python
from datasets import load_dataset
probs = load_dataset("open-r1/codeforces", "verifiable", split="train")
holdout = load_dataset("open-r1/codeforces", "verifiable", split="test")  # eval only
cots = load_dataset("open-r1/codeforces-cots", "solutions_decontaminated", split="train")  # optional
```

```bash
# After picking the pool: generated tests for the contests we kept
huggingface-cli download open-r1/codeforces --repo-type=dataset \
  --include='generated_tests/test_cases_1950.parquet' \
  --max-workers=8 --local-dir /vol/cf_tests
huggingface-cli download Qwen/Qwen3.5-9B --local-dir /vol/qwen3.5-9b
```

## Problem pool

We pick about 1,500 problems to sample on in round 1, weighted toward the difficulty the model sometimes solves. Problems it always or never solves teach nothing in expert iteration.

**Filters, in order**

1. **Start** from the `verifiable` train split, about 8.8k problems.
2. **Hold out** the `test` split, and deduplicate by problem ID.
3. **Drop interactive problems**, since the harness only runs a program on an input.
4. **Require at least 5 tests** per problem, fetching that contest's generated tests where official ones are fewer. Fewer tests let wrong solutions through.
5. **Check the checker:** keep problems with an exact-match answer or a working checker.
6. **Sample by difficulty** using the `rating` field to hit the mix below.

| Codeforces rating | Share | Problems | Why |
| --- | --- | --- | --- |
| Under 1600 | \~15% | \~225 | Cheap, short trajectories that reinforce clean format and the submit-then-EXIT habit |
| 1600–2200 | \~55% | \~825 | Where the model likely passes some attempts, so expert iteration has signal |
| 2200+ | \~30% | \~450 | Harder techniques; R1 traces fill in where the model never solves them |

After round 1, rebalance using real pass rates: keep problems the model solved 1–3 times out of 4, and drop ones it solved every time.

## Sampling (expert iteration)

Each round, we run the current model in our copy of the harness, 4 attempts per problem, and keep its shortest fully passing trajectory. Because the data is the model's own output, it matches its style and format, and is easy for it to learn without forgetting.

**Harness details:** we don't need the harness's exact system prompt or tool schema, because we hand over the model and the harness drives it. Our copy uses sensible defaults (native Qwen3.5 chat template and tool format) that stay configurable.

**Setup**

- Serve `Qwen/Qwen3.5-9B` with vLLM on Modal H100s, fanned out over 2–4 GPUs, using the model's native chat template.
- Harness copy: a run tool (`g++ -std=c++20 -O2` under the problem's limits), a submit tool reporting sample results, up to 50 submissions, the last compiling one graded on full tests.
- Sampling: thinking on, temperature \~0.8 for variety, max output tokens equal to what we'll submit with.

**Keep a trajectory only if**

- its final submission passes every full test (not just the samples),
- it has no malformed tool calls and ends with EXIT,
- no later submission replaced a passing one with worse code.

Then keep the shortest by turns, then tokens, one per problem (two for 2200+).

**Estimated cost per round** (estimates, not measured): \~1,500 problems × 4 attempts × \~12k tokens is about 72M generated tokens. At roughly 5k tokens/s per H100 that's \~4 GPU-hours, about $16–20 on Modal. Measure real throughput in the first hour and adjust.

**Round 2** samples from the round-1 checkpoint. Export the LoRA from Tinker and load it in vLLM; if vLLM's LoRA support doesn't cover Qwen3.5's architecture, merge the adapter into the weights first.

**R1 traces for hard problems:** for up to \~200 problems rated 2200+ that the model never solves, add a verified R1 trace from codeforces-cots, rewritten into the harness format with real tool outputs. Drop them if local eval gets worse, since R1 reasons in a different style.

## Train

We train small LoRA runs on Tinker, about $40 for the planned runs, plus an optional short GRPO run. Sampling and evals run on Modal, which keeps Tinker credit for training. Every checkpoint is submitted, since submissions are free.

**Tinker ($100)**

| Run | Starts from | Data | Train tokens | Cost |
| --- | --- | --- | --- | --- |
| Round 1 | Qwen3.5-9B | \~1,000 self-generated trajectories | \~10M | \~$15 |
| Variant | Qwen3.5-9B | Same data, lower learning rate, or plus R1 traces for hard problems | \~10M | \~$15 |
| Round 2 | Best of round 1 / variant | \~600 new trajectories from that checkpoint | \~6M | \~$9 |
| GRPO (optional) | Round 2 | Medium problems, pass/fail reward minus a small per-turn penalty | \~10M sampled + trained | \~$25 |
| Reserve |  |  |  | \~$36 |

**Modal ($200, estimates)**

| Item | Cost |
| --- | --- |
| Sampling, 2 rounds (\~4 H100-hours each) | \~$35–40 |
| Local evals (baseline + 4 checkpoints) | \~$10 |
| CPU sandbox and test runner | \~$15–20 |
| Reserve | \~$130 |

**LoRA config**

| Setting | Value | Why |
| --- | --- | --- |
| Model | `Qwen/Qwen3.5-9B` | Post-trained, thinking on |
| LoRA rank | 32 | Small change to an already capable model |
| Learning rate | \~1e-4 (variant: 5e-5) | Gentler than from base, to avoid forgetting |
| Batch size | 32 sequences |  |
| Max sequence length | Same as the submission's max output, 16–32k | Training must cover full thinking traces |
| Loss | Assistant tokens only, thinking included |  |
| Epochs | 1 |  |

**Per-turn training (decided):** `train_sft` defaults to one datum per assistant turn (`--mode turn`), so each target sees the same prompt it saw at sampling time (earlier reasoning is not resent). Earlier turns repeat as context, so training tokens run above the plan's 10M estimate; run `train_sft --dry-run` for the real count and cost before launching.

Costs use $1.463 per million train tokens and $0.807 per million sampled, budgeting as if masked tokens are billed. Save each checkpoint with `save_weights_for_sampler` and record its `tinker://` path.

## Evaluate and submit

The untouched Qwen3.5-9B is submitted first and is the score every run must beat. Each later checkpoint gets a local eval on Modal, then a submission.

**Local eval** (\~150 problems from the open-r1/codeforces `test` split, in our harness copy)

| Metric | Target | If it's off |
| --- | --- | --- |
| Pass rate on full tests | Above the baseline | Drop the run; try the lower learning rate |
| Malformed tool-call rate | \~0% | Check the chat template and tool schema match the harness |
| Truncation rate | < 5% | Raise max output tokens, or prefer shorter trajectories |
| Average turns per task | Falling round over round | Weight selection harder toward fewer turns |

**Submitting**

- We do not submit the untouched model. The local eval of Qwen3.5-9B on eval150 is the bar; the staff-reported leaderboard baseline is ~20%.
- We get 4 attempts per hour, one processing at a time, and edits count too. Queue checkpoints in order of local eval score.
- Once the best checkpoint is clear, try sampling settings: temperature (0.6 vs 0.8), max output tokens, thinking budget.
- Log each submission's checkpoint path, settings and score.
- Make the final submission by about 1pm Sunday, an hour before the 2pm deadline.

## Steps

The pipeline takes roughly 12–16 hours of wall time. The times are estimates, so track actual progress here.

- [x] Confirm with staff that starting from Qwen3.5-9B is allowed (done)
- [x] ~~Submit untouched Qwen3.5-9B as the baseline~~ Dropped: we only submit checkpoints we trained
- [x] Harness tool schema, chat template, turn limit, thinking setting: not needed, we hand over the model in the submission and the harness drives it. Our harness copy uses sensible defaults.
- [ ] Build the C++ sandbox and harness copy on Modal (\~1–2 hrs)
- [ ] Download `codeforces` (`verifiable`) and the model weights to a Modal Volume (\~30 min)
- [ ] Build the \~1,500-problem pool: filters, generated tests where needed, difficulty mix (\~1–2 hrs)
- [ ] Serve Qwen3.5-9B with vLLM, run the local eval for the baseline (\~1 hr)
- [ ] Round 1 sampling: 4 attempts per problem, keep shortest passing (\~1–2 hrs on 2–4 GPUs)
- [ ] Train round 1 and the variant on Tinker, run local evals, submit both (\~2–3 hrs)
- [ ] Round 2 sampling from the best checkpoint (\~1–2 hrs)
- [ ] Train round 2, run a local eval, submit (\~1–2 hrs)
- [ ] Optional: short GRPO from round 2 if budget and time allow
- [ ] Sweep sampling settings on the best checkpoint with the remaining submissions
- [ ] Final submission by 1pm Sunday

## Sources

- [open-r1/codeforces dataset card](https://huggingface.co/datasets/open-r1/codeforces)
- [open-r1/codeforces-cots subsets](https://huggingface.co/datasets/open-r1/codeforces-cots/viewer/solutions/train?p=10)
- [Open R1 Update #3 (CodeForces-CoTs)](https://huggingface.co/blog/open-r1/update-3)
- [Tinker Cookbook](https://github.com/thinking-machines-lab/tinker-cookbook)
- [Tinker docs](https://tinker-docs.thinkingmachines.ai/)
