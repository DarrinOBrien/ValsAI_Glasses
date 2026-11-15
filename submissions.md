# Submission log

Limits: 4 submissions/hour, one processing at a time, edits count as an attempt. Deadline Sunday 2pm; final submission by 1pm.
Staff-reported baseline of the untouched Qwen3.5-9B: ~20%. We do NOT submit the untouched model; every checkpoint must beat its local eval.

## Log

| # | Date/time | Checkpoint (`tinker://` path or base) | Run | Settings (temp, max tokens, thinking) | Local eval (pass / malformed / trunc / turns) | Leaderboard score | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| - |  | `Qwen/Qwen3.5-9B` (untouched) | Baseline (local eval only, not submitted) |  |  | ~20% (staff) |  |
| - |  | r1, v1, r1b | Round 1 attempts | - | direct Tinker 12: r1 0/12, r1b 1/12 (base 7/12) | not submitted | broken SFT targets (empty think; memorized EXIT line); fixed in d05e97a / 14d7410 |
| 1 | 2026-11-15 ~02:00 | `tinker://3f4f016f-8fef-5c48-a355-55a23c736581:train:0/sampler_weights/r1c-ep0` | r1c (cut at first passing submit, no synthetic EXIT) | leaderboard defaults (none set) | direct Tinker 12 (12k tok, 15 turns): 5/12, 16/102 turns truncated | **19.51** | ~ base; format fixed, no solving gain |
| 2 | 2026-11-15 02:27 | r1c (same path) | r1c settings sweep | temperature 0.6, rest default | direct 12 at 24k/t0.6: 4/12 (worse than t0.8) | pending | leaderboard "Glasses #4" |

## Settings sweep (best checkpoint only)

| # | Checkpoint | Temperature | Max output tokens | Thinking budget | Score | Notes |
| --- | --- | --- | --- | --- | --- | --- |
|  |  |  |  |  |  |  |

## Queue (r1c settings sweep, one knob at a time vs the 19.51 default run)

Local direct-Tinker check, r1c, first 12 of eval150, 15 turns (n=12, noisy):
12k tok / t0.8: 5/12, trunc 15.7%/turn, malformed 16.7% | 24k / t0.8: **6/12**, trunc 8.7%, malformed 13.0% | 24k / t0.6: 4/12, trunc 24.1%, malformed 26.6%
-> more output tokens helps (truncation halves); lower temperature hurts (longer loops, more cut-offs). Skip t0.6.

1. r1c, max output tokens raised (e.g. 2x the default, >= 24k), temperature 0.8
2. r1c, same + thinking budget raised
3. r1c, all defaults again (repeat: measures leaderboard noise)
4. r1c, best of 1-2 with temperature 1.0
5. r1c, best combination so far
6. Repeat the best setting once more to confirm before the final pick

## Final submission

- Checkpoint:
- Settings:
- Submitted at (target before 1pm Sunday):
