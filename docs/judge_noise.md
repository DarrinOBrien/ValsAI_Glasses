# Judge noise check (Modal CPU judge)

Tool: `uv run --env-file .env modal run scripts/judge_noise.py --n 300` (results in `runs/judge_noise.json`, gitignored).
Method: for each (cpu, max_inputs) config, 300 judge calls fired at once (c2m2 = today's `judge_remote`), each compiling one
known-good program and running 5 tests through `cpp.judge.run_exe` (light ~0.05 s, 2x medium ~0.3 s, 2x heavy ~0.55 s idle, limit 2.0 s).
Every program is correct, so any TLE would be a false TLE. Two full sweeps (13.5k runs per config pair, 9k runs per sweep).

| Config | False TLE @ 2.0 s limit | heavy slowdown p99 / max vs idle | within-container p99 / max | compile p99 |
| --- | --- | --- | --- | --- |
| c1m1 | 0 | 1.09 / 1.19 | 1.11 / 1.26 | 3.6 s |
| c2m1 | 0 | 1.09 / 1.11 | 1.08 / 1.19 | 3.2 s |
| c2m2 (current) | 0 | 1.12 / 1.20 | 1.09 / 1.26 | 3.9 s |
| c2m4 | 0 | 1.06 / 1.09 | 1.11 / 1.22 | 3.3 s |
| c4m2 | 0 | 1.13 / 1.16 | 1.20 / 1.31 | 3.3 s |
| c4m4 | 0 | 1.12 / 1.19 | 1.12 / 1.22 | 4.2 s |

(The first sweep, before the light-test fix to the false-TLE statistic, showed a worst single run of 1.55x idle on c2m2; no sweep
produced a TLE, RE, wrong output or compile failure.)

Findings
- Contention is small: same-container slowdown p99 is 1.1-1.2x for every config, and host-to-host speed differences (~1.25x p99) matter as
  much as concurrency. The config choice is within noise for compute-bound solutions.
- Recommendation: **keep `cpu=2, max_inputs=2`** (one core per concurrent judge). `cpu=2, max_inputs=4` showed no penalty either and would halve
  container count, but it was only tested on single-threaded compute-bound code; try it only after a run with I/O-heavy / memory-bound programs.
  Do not pay for `cpu=4`.
- The real protection is a time margin, not hardware: a solution that runs in more than ~1/1.5 of the limit can flip to TLE (worst observed 1.55x).
  Selection should drop trajectories whose full-test `max_time` > 0.6 x limit (`build_r1 --max-time-frac` now defaults to 0.6).
- Local macOS judging is much noisier than Modal: a trivial A+B program reported ~1.3 s under parallel load (first-exec latency of a fresh binary),
  which made `tests/test_harness.py::test_happy_path` flaky with a 2 s limit. The test problem now uses a 10 s limit.
- Caveat: the benchmark program is single-threaded and compute-bound. Programs with big input parsing, memory-bandwidth-bound loops or many
  threads were not tested.
