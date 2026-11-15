# Submission log

Limits: 4 submissions/hour, one processing at a time, edits count as an attempt. Deadline Sunday 2pm; final submission by 1pm.
Staff-reported baseline of the untouched Qwen3.5-9B: ~20%. We do NOT submit the untouched model; every checkpoint must beat its local eval.

## Log

| # | Date/time | Checkpoint (`tinker://` path or base) | Run | Settings (temp, max tokens, thinking) | Local eval (pass / malformed / trunc / turns) | Leaderboard score | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| - |  | `Qwen/Qwen3.5-9B` (untouched) | Baseline (local eval only, not submitted) |  |  | ~20% (staff) |  |
| - |  | r1, v1, r1b | Round 1 attempts | - | direct Tinker 12: r1 0/12, r1b 1/12 (base 7/12) | not submitted | broken SFT targets (empty think; memorized EXIT line); fixed in d05e97a / 14d7410 |
| 1 | 2026-11-15 ~02:00 | `tinker://3f4f016f-8fef-5c48-a355-55a23c736581:train:0/sampler_weights/r1c-ep0` | r1c (cut at first passing submit, no synthetic EXIT) | leaderboard defaults (none set) | direct Tinker 12 (12k tok, 15 turns): 5/12, 16/102 turns truncated | **19.51** | ~ base; format fixed, no solving gain |
| 2 | 2026-11-15 02:27 | r1c (same path) | r1c settings sweep | temperature 0.6, rest default | direct 12 at 24k/t0.6: 4/12 (local said worse) | **29.86** | leaderboard "Glasses #4"; team best; local n=12 check was misleading |
| 3 | 2026-11-15 03:31 | r1c (same path) | r1c settings sweep | temperature 0.6, max output tokens 32768, thinking on, rest default | - | **19.19** | leaderboard "Glasses #5"; 32k tokens did not help (or #4 was lucky) |
| 4 | 2026-11-15 04:54 | r1c (same path) | repeat of #4 (noise check) | temperature 0.6, rest default | - | **22.60** | leaderboard "Glasses #6"; same settings as #4 (29.86) -> leaderboard noise is several points |
| 5 | 2026-11-15 07:23 | `tinker://47532075-7d8a-5a87-bfee-159cccb6faa1:train:0/sampler_weights/r2-ep0` | r2 (r1c samples on 302 new problems + r1c data, 1124 examples, from base) | temperature 0.6, rest default | direct Tinker 12 (24k, t0.8, 15 turns): 6/12 (r1c 6/12), malformed 15.7%/turn, trunc 11.2%, 8 exits (2 unsolved), 89 turns (r1c 115) | pending | leaderboard "Glasses #7" |
| - |  | `Glasses #2` (0.00, 12:56 AM) | submitted by the earlier lead; checkpoint not recorded here | - | - | 0.00 | r1 or v1 (both broken) |
| 6 | 2026-11-15 ~09:25 | `tinker://a729186b-7972-5ac8-8759-2966ed8f930f:train:0/sampler_weights/r4-ep0` | r4 (r1c data + 81 teacher-397B trajectories) | temperature 0.6, rest default | direct 12x3 (24k, t0.8): 18/36, 23/36 clean exits, malformed 10.4%/turn | **26.43** | submitted by human; took ~1h+ to score; ~ r1c t0.6 mean (26.2), no clear gain |
| 7 | (not yet submitted) | `tinker://79984457-30c6-50f5-af5b-9b32a252774e:train:0/sampler_weights/r6-ep0` | r6 (r1c data + 184 teacher-397B trajectories) | planned: temperature 0.6, rest default | direct 12x3: 18/36, 26/36 clean exits, malformed 7.0%/turn | - | same check: r1c 16/36, r3 17/36, r5 15/36. Note: a separate lead-side r2 (`7b5e8719...`, 5/12, malformed 32%) failed the check and was NOT submitted; the r2 in row 5 is the GPU Claude's different checkpoint |

Form fields (hackathon.vals.ai/submit, "Advanced settings"; empty = default): Thinking On/Off (default On), Temperature
(Tinker default), Top p (Tinker default), Max output tokens (default 16384, step 256), Max turns (default 100, max 100).
There is no separate thinking-budget field. The team's API key is saved on the site (ends AAAA); leave the key field empty.

## Settings sweep (best checkpoint only)

| # | Checkpoint | Temperature | Max output tokens | Thinking budget | Score | Notes |
| --- | --- | --- | --- | --- | --- | --- |
|  |  |  |  |  |  |  |

## Queue (r1c settings sweep, one knob at a time vs the 19.51 default run)

Local direct-Tinker check, r1c, first 12 of eval150, 15 turns (n=12, noisy):
12k tok / t0.8: 5/12, trunc 15.7%/turn, malformed 16.7% | 24k / t0.8: **6/12**, trunc 8.7%, malformed 13.0% | 24k / t0.6: 4/12, trunc 24.1%, malformed 26.6%
-> the leaderboard disagreed: t0.6 scored 29.86 vs 19.51 at default temperature. Trust the leaderboard; n=12 is noise.

1. (submitted, #5) r1c, t0.6, 32768 max output tokens
2. r1c, t0.4, max tokens = whichever of 16384/32768 scored better (is lower temperature still better?)
3. r1c, best setting so far repeated (measures leaderboard noise)
4. r2 at the best setting, once the r2 session reports a passing check
5. Repeat the overall best once more to confirm before the final pick

## Final submission

- Checkpoint:
- Settings:
- Submitted at (target before 1pm Sunday):
