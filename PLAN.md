# Tinker C++ Post-Training Plan

2026-11-14

## Goal and constraints

We post-train Qwen3.5-9B on Tinker with LoRA to solve hard C++20 competitive programming problems, then submit our best checkpoint by Sunday 2pm. We get the highest score on the hidden set by solving more problems, and we break ties by using fewer turns.

**Task format**

- Each task is one problem: a text statement, the sample tests as files, and time and memory limits (e.g. 2 s, 1 GiB).
- The model has two tools. One compiles and runs C++ on any input. The other submits a solution and reports results on the sample tests.
- The model can submit up to 50 times. The last submission that compiles is graded on the hidden tests: small tests are worth 40 points and large tests 60.
- A task ends when the model answers EXIT or runs out of turns. Repeated malformed tool calls also end it.

**Rules we must follow**

- Never read, store, share or train on the scoring problems or their solutions, and never train the model to leak them.
- Public datasets and distillation from larger open-source models are allowed. Distilling from Anthropic or OpenAI models is not, so gpt-oss can't be a teacher, and we check each dataset card for which model wrote the traces.
- Submit only checkpoints we trained ourselves, with at most 9B total parameters. Staff confirmed the cap is on total size, that an already post-trained model like Qwen3.5-9B is a fine starting point, and that they encourage LoRA checkpoints.
- Staff review scoring traces, token counts and turn counts. If we're unsure about a method, we ask staff first.

**Resources**

| Resource | Amount | Notes |
| --- | --- | --- |
| Tinker credits | $100 | Training, sampling and local evals. Submissions are not charged. |
| Modal credits | $200 | Teacher inference, C++ sandbox, optional student sampling |
| Submissions | 4 per hour | One processing at a time; edits also use an attempt |
| Deadline | Sunday 2pm | Judged on our best submission |

## Model choice

We start from **Qwen3.5-9B** (Tinker ID `Qwen/Qwen3.5-9B`), the post-trained hybrid version, and train a LoRA on top of it. Staff confirmed three things: the 9B cap counts total parameters, post-trained models are allowed, and LoRA checkpoints are encouraged. That makes Qwen3.5-9B the strongest model on Tinker we can use. It's dense, so it's 9B total and 9B active, and it costs the same as the Base version.

| Model | Type | Prefill ($/M) | Sample ($/M) | Train ($/M) | Status |
| --- | --- | --- | --- | --- | --- |
| Qwen3.5-9B | Hybrid, post-trained | 0.30 (0.06 cached) | 0.807 | 1.463 | **Chosen** |
| Qwen3.5-9B-Base | Base, dense | 0.30 (0.06 cached) | 0.807 | 1.463 | Fallback only (no chat or tool use yet) |
| Qwen3-8B | Hybrid, post-trained | 0.195 (0.039 cached) | 0.60 | 0.44 | Allowed, but older, weaker and 32K context |

**What this means for the plan**

- **We no longer teach chat, thinking or tool use from scratch.** Qwen3.5-9B already does all three. SFT now only has to match the harness's exact tool-call format and make the model better at hard C++, so we need less data and fewer training tokens.
- **Measure the starting point first.** Run the untouched Qwen3.5-9B through our harness copy on the holdout before any training. We can't submit it as is, since it isn't a checkpoint we trained, but it tells us what each run actually adds.
- **The new main risk is making it worse.** A post-trained model can forget what it knows if we train too hard. We lower the LoRA rank and learning rate, and we compare every run against the baseline.
- **Keep its native format.** Render all training data with Qwen3.5's own chat template and thinking tags. If the harness uses the same template, R0 may need very little format training. Qwen3.5-122B-A10B is from the same family, so its traces already look like what the student writes.
- **Expert iteration and RL get more useful.** The model should solve some problems from the start, so R3 has more passing runs to learn from, and R4 is more likely to be worth running.
- **Some budget frees up.** A smaller R1 leaves Tinker credit for R3 and R4.

We ruled out every model over 9B total, including MoE models with few active parameters such as Qwen3.6-35B-A3B, Qwen3.5-35B-A3B-Base and gpt-oss-20b. We also ruled out third-party fine-tunes such as Ornith-1.0-9B and OmniCoder-9B, because those aren't checkpoints we trained.

## Pipeline

The pipeline is distillation through SFT, then expert iteration, with GRPO only if time and money allow. Each stage starts from the previous checkpoint, so every run can be submitted.

1. **Distillation SFT (R1).** Train a LoRA on Qwen3.5-9B using verified C++ solutions from larger open models: R1-generated public traces plus our own Qwen3.5-122B-A10B trajectories. Here, distillation just means the SFT data comes from a teacher.
2. **Targeted SFT (R2).** Continue from R1 on the shortest agentic trajectories, to sharpen tool use and cut turns.
3. **Expert iteration (R3).** The student samples its own attempts, the tests act as the reward, and we SFT on the shortest passing runs. This is a simple, on-policy form of RL that is hard to break.
4. **Optional GRPO (R4).** A few dozen steps on medium-difficulty problems from R3, with a small penalty per turn. Run it only if R3 finishes by 8am and we have more than $30 of Tinker credit left.

We skip full RL as the main stage because a useful GRPO run would cost $30-60 or more in sampling, and RL on a freshly SFT'd 9B model in under a day is easy to get wrong.

## Data

We build three training pools once and reuse them across runs. Every hard problem with a verified solution goes into training, not into a holdout. A local eval on ~100 problems is too noisy to rank checkpoints, so we rank them on the leaderboard and keep only a small holdout for format checks. With about 30M training tokens to spend, we rank sources by quality and drop the rest.

| Pool | Source | Size | Used in | Cost |
| --- | --- | --- | --- | --- |
| Format-check holdout | Easy and medium Codeforces-style stdin problems, from a different source than the training pools | 30-50 problems | Malformed-call and truncation checks, R0 learning rate sweep | Free |
| P1: public traces, made agentic | Verified C++ traces from R1-generated sets ([open-r1/codeforces](https://huggingface.co/datasets/open-r1/codeforces), OpenCodeReasoning). Keep traces under ~8k tokens, and wrap each in the harness format: think, run the samples, submit, EXIT. | ~1,500 trajectories | R0, R1 | Free, plus a little Modal CPU |
| P2: teacher trajectories | Qwen3.5-122B-A10B running in our harness copy on CodeContests+, CodeContests-O and open-r1 problems. Include some runs where it fixes a compile error or wrong answer. | ~1,000 trajectories | R0, R1, R2 | Modal GPU |
| P3: student self-samples | The best SFT checkpoint on problems it solves 20-80% of the time | ~150-250 trajectories | R3 | Tinker or Modal |

**Difficulty mix**

The scoring problems are hard, so the training mix leans hard: roughly 60% hard, 30% medium and 10% easy.

- Hard solutions reuse the techniques from easier problems, so hard traces teach both.
- We keep some medium and easy problems so the model doesn't overthink simple ones, which wastes turns and risks truncation.
- The teacher fails most hard problems, so the hard share is capped by how many verified traces we actually get. If we fall short of 60%, we take what we have and don't pad with unverified traces.
- R3 still samples on problems the student solves 20-80% of the time. Hard problems move into that band as the model improves.
- Small tests are worth 40 points, so a correct brute force still scores. Include trajectories where the model writes a brute force first, checks the fast solution against it, and falls back to the brute force if the fast one fails.

**Fallback: distill from a stronger open model**

If the plan above doesn't work, we switch teachers and distill from whichever open-source model is best at hard C++. We switch if the 122B teacher passes fewer than about 15% of hard problems after its first 2 hours, or if R1 doesn't beat the untrained Qwen3.5-9B on the leaderboard.

- Candidates already served on Tinker, so there's no setup: GLM-5.3-Flash ($2.11/M sampled), DeepSeek-V4.1-Flash ($3.60/M) and Qwen3.5-397B-A17B ($7.50/M). Kimi-K3.1-Pro is the largest but costs $25/M, so we'd use it only on a small set of the hardest problems.
- Never gpt-oss or anything else from OpenAI or Anthropic.
- Pick the teacher with a quick test on 20 hard problems, then use the same harness, verification and filters as P2.
- Rough cost for 300 verified hard trajectories at a 25% pass rate (about 12M sampled tokens): $25 on GLM-5.3-Flash, $43 on DeepSeek-V4.1-Flash, $90 on Qwen3.5-397B. This comes out of the Tinker reserve and the R4 money, so we start with the cheaper teachers.

**Filtering rules**

- Keep a trajectory only if its final solution passes the full tests in our sandbox, built with `g++ -std=c++20 -O2` under the problem's time and memory limits.
- Prefer problems that ship validators and checkers (CodeContests+, CodeContests-O). Synthesized tests can pass wrong solutions.
- Drop slow solutions, since large hidden tests are worth 60 of the 100 points.
- Cut wasted turns, not debugging. Drop trajectories that repeat a failing approach or idle, but keep the ones where the model hits a wrong answer or a timeout and fixes it. Turns only break ties, so trading a solve for a shorter trace is a bad deal.
- To convert P1, run the trace's final code in the sandbox and insert the real tool outputs, so every tool result in the training data is real. Don't wrap every trace as think, run, submit, EXIT. That only shows the happy path and teaches the model to EXIT no matter what the tools say. Some converted traces should show a failed run and a second attempt.

**Hygiene**

- Split by problem, never by solution, so no eval problem appears in training.
- Use LiveCodeBench Pro, ICPC-Eval and Frontier-CS only for eval.
- Check each dataset card for which model generated the traces. Exclude anything from OpenAI or Anthropic models, which may include parts of Nemotron-Competitive-Programming-v1.
- Use Codehacks to stress-test our sandbox and checkers, not as training data.

## Training runs

We plan four runs plus an optional fifth. Each run branches from the previous checkpoint, so later runs cost less than a fresh run, and every run gets submitted.

| Run | Starts from | Data | Train tokens | Purpose |
| --- | --- | --- | --- | --- |
| R0 smoke | Qwen3.5-9B | 200 examples from P1 and P2 | ~1M | Check the format, the learning rate and the submission pipeline, and compare against the untrained baseline |
| R1 main SFT | Qwen3.5-9B | P1 + P2, 1 epoch | ~15M | Teach reasoning, C++ and tool use |
| R2 branch | R1 | P2 only, clean trajectories with no wasted turns | ~5M | Sharpen tool calls. Skip this run if R1 already has a low malformed-call rate |
| R3 expert iteration | Best of R1/R2 | P3, shortest passing runs | ~2M (from ~8M sampled, about 1,000 attempts) | On-policy gains on medium problems |
| R4 GRPO (optional) | R3 | Medium problems, pass/fail reward minus a per-turn penalty | Few dozen steps | Extra gain if time allows |

Sampling settings (temperature, max output tokens, thinking) need no training, so we try them on the best checkpoint as separate submissions.

**Starting hyperparameters**

| Setting | Value | Note |
| --- | --- | --- |
| Method | LoRA | Tinker default, and staff encourage LoRA checkpoints |
| LoRA rank | 32 | The model already chats and calls tools, so a lower rank is enough. Try 64 if R1 underfits |
| Learning rate | ~1e-4 | Lower than for a base model, so we don't erase what it already knows. Sweep 5e-5, 1e-4 and 2e-4 in R0 |
| Batch size | 8 sequences | With ~2,500 examples, a batch of 32-64 gives only 40-80 optimizer steps, and R0 would get 3-6, too few to tell learning rates apart |
| Max sequence length | 16k tokens | |
| Loss | Assistant tokens only | Never on tool outputs or prompts |
| Epochs | 2-3 | The dataset is small, so one epoch underfits. Watch for the model copying teacher phrasing instead of reasoning |

We save each checkpoint with `save_weights_for_sampler` and submit its `tinker://` path.

## Budget

The planned runs use about $85 of the $100 Tinker credit and about $100 of the $200 Modal credit, leaving room for one extra run on each. Submissions are free, so all Tinker spend goes to training and local evals. Tinker costs use Qwen3.5-9B prices, which match the Base model: $1.463/M train, $0.807/M sample, $0.30/M prefill ($0.06/M cached). Starting from a post-trained model means R1 can likely be smaller than planned, which frees credit for R3 and R4.

**Tinker ($100)**

| Item | Tokens | Cost |
| --- | --- | --- |
| R0 smoke SFT | 1M train | ~$1.50 |
| R1 main SFT | 15M train | ~$22 |
| R2 branch SFT | 5M train | ~$7.30 |
| R3 expert iteration | 8M sampled + ~10M prefill + 2M train | ~$12.50 |
| Local evals | 5 x 40 problems x ~15k tokens | ~$6 |
| R4 GRPO (optional) | Rollouts + training | ~$25 |
| Reserve | Covers the fallback teacher if we need it | ~$24 |

To save Tinker credit, export R1 or R2's LoRA and run R3's sampling with vLLM on Modal instead. If R4 doesn't run, its $25 can go to a second R1 variant, such as a different data mix or learning rate.

**Modal ($200, estimates)**

| Item | Setup | Cost |
| --- | --- | --- |
| Teacher for P2 | Qwen3.5-122B-A10B with FP8 weights on 2xH100 (~$8/hr), ~9 hrs, for ~1k accepted trajectories at a 30-40% pass rate | ~$70-80 |
| C++ sandbox and test runner | CPU containers with time and memory limits | ~$10-15 |
| Student sampling for R3 (optional) | 1xH100 with the LoRA loaded, ~3 hrs | ~$12 |
| Reserve | | ~$95 |

If the 122B teacher is slow or won't fit, switch to Qwen3.5-35B-A3B on one H100 at about half the cost, and expect fewer hard solves. The GPU prices here are estimates, so check Modal's pricing page before launching.

**Cost controls**

- Check whether Tinker bills masked prompt tokens. Budget as if it does.
- Watch the usage meter during R1 and stop early if it runs hot.
- Cap trajectory length in every pool, since tokens are what we pay for.

## Timeline

Saturday 2pm to Sunday 2pm, by hour. Final checkpoint submitted by 1pm, an hour before the deadline.

| Stage | Window |
| --- | --- |
| Harness + data setup | 2pm to 5pm |
| Teacher P2 on Modal | 4pm to 1am |
| Convert P1 traces | 5pm to 1am |
| R0 smoke + submit | 5pm to 7pm |
| R1 main SFT | 1am to 3am |
| R2 branch SFT | 3am to 6am |
| Sample P3 | 3am to 6am |
| R3 expert iteration | 6am to 9am |
| R4 GRPO (optional) | 9am to 11am |
| Settings sweep | 9am to 1pm |
| Final submission | 1pm |
| Deadline | 2pm |

Teacher generation runs on Modal overnight, while R0 proves the format early. Nothing real trains until 1am under this schedule, which is 11 hours in, so start an SFT on P1 alone as soon as the harness copy works and add P2 as it arrives. Each Tinker run is submitted as soon as it finishes.

## Submission strategy and risks

Submit every run as soon as it finishes. Submissions are free and only our best score counts, so a weak submission costs nothing. Make the final submission by about 1pm Sunday, in case it fails the basic checks.

**Submission rules of thumb**

- Submit R0 within the first few hours, to prove the format and pipeline end to end.
- Only one submission can process at a time, and edits also use one of the 4 attempts per hour. Queue submissions in order of expected score.
- Once the best checkpoint is known, spend the remaining attempts on sampling settings: temperature, max output tokens and thinking budget.
- Keep a log of each submission's checkpoint path, settings and score.

**Behaviors to teach in every run**

- Submit once the samples pass, then answer EXIT. Fewer turns win ties.
- Never follow a good submission with a worse one, since the last submission that compiles is graded.
- Never emit a malformed tool call. Track the malformed-call rate in every local eval.
- Write efficient code that fits the time limit on large inputs.
- Test edge cases and large inputs with the run tool before submitting.
- Generate a maximum-size input with the run tool and time the solution on it before submitting. Large tests are 60 of the 100 points.
- Check a fast solution against a brute force on small random inputs, and submit the brute force if the fast one is wrong.

**Local eval metrics** (on the format-check holdout, after every run): malformed-call rate, truncation rate, average turns, average output tokens. Pass rate is too noisy on this few problems, so we rank checkpoints by leaderboard score.

**Risks**

| Risk | Effect | Mitigation |
| --- | --- | --- |
| Training format doesn't match the harness | Malformed calls end tasks | Get the exact chat template and tool schema first, and confirm with R0 |
| Reasoning runs too long | Truncation, wasted turns | Cap trace length and tune max tokens at submission |
| Teacher data takes too long | No time for R2 or R3 | Cap teacher generation at ~9 hrs, and train R1 on whatever P1 + P2 exists |
| Weak synthesized tests | Wrong solutions in training | Prefer datasets with validators and checkers, and require multiple tests |
| Tinker overspend | No budget for R3 | Watch the meter and sample on Modal |
| Teacher too weak on hard problems, or R1 doesn't beat the baseline | Training adds nothing | Fall back to distilling from a stronger open model on Tinker (see Data) |

**Open questions**

- [ ] Get the harness's exact tool-call schema and chat template from staff or the docs.
- [ ] Check the dataset cards to confirm no OpenAI or Anthropic teacher traces.
- [ ] Check whether Tinker bills masked tokens in SFT.
- [ ] Confirm OpenCodeReasoning has a C++ split. v1 may be Python only, in which case use OpenCodeReasoning-2.
