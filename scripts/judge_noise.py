"""Judge noise check: false-TLE rate of the Modal judge under the concurrency we will actually use.

    uv run --env-file .env modal run scripts/judge_noise.py --n 300                       # all configs
    uv run --env-file .env modal run scripts/judge_noise.py --n 300 --configs c2m2,c4m2   # subset
Config names (c1m1 c2m1 c2m2 c2m4 c4m2 c4m4) are c<cpu>m<max_inputs> (Modal `cpu=` and `@modal.concurrent(max_inputs=)`); c2m2 is what
modal_app.py::judge_remote uses today. For each config we fire --n judge calls AT ONCE (the sampling fan-out
bursts like that), each compiling one known-good program and running 5 tests (1 light, 2 medium, 2 heavy; heavy ~0.5 s
on an idle core) through cpp.judge.run_exe, i.e. the exact timing path the real judge uses (wall clock, TLE when
elapsed > limit). All programs are correct by construction, so any TLE is a false TLE; a WA would be a false WA.

Reported per config: runs, TLE/WA counts at the real limit (--tl, default 2.0 s), slowdown vs the idle baseline
(percentiles of time / idle median, per test size), what the false-TLE rate WOULD be for tighter limits (1.0x..4.0x the idle
time - the problems where the original solution runs close to its limit), and compile-time percentiles.
A baseline is measured first with a single warm container and sequential calls. Results: runs/judge_noise.json.
"""
from __future__ import annotations

import json, os, socket, statistics as st, time

import modal

app = modal.App("judge-noise")
image = (
    modal.Image.from_registry("gcc:14", add_python="3.12")
    .env({"CXX": "g++"})
    .workdir("/root")
    .add_local_python_source("cpp")
)

PROG = r"""#include <bits/stdc++.h>
int main(){ long long n; std::cin>>n; unsigned long long x=1,s=0;
  for(long long i=0;i<n;i++){ x=x*6364136223846793005ULL+1442695040888963407ULL; s+=x>>33; }
  std::cout<<s<<"\n"; }
"""
SIZES = {"light": 20_000_000, "medium": 200_000_000, "heavy": 400_000_000}
TESTS = ["light", "medium", "medium", "heavy", "heavy"]


def _run(tl: float) -> dict:
    import tempfile, shutil
    from cpp import judge
    d = tempfile.mkdtemp(prefix="jn_")
    try:
        t0 = time.time()
        ok, err, exe = judge.compile_cpp(PROG, d)
        ct = time.time() - t0
        if not ok:
            return {"compiled": False, "compile_s": ct, "err": err[:300], "runs": []}
        runs = []
        for name in TESTS:
            r = judge.run_exe(exe, f"{SIZES[name]}\n", tl, 256)
            runs.append({"size": name, "status": r["status"], "time": r["time"], "out": r["stdout"].strip()})
        return {"compiled": True, "compile_s": ct, "runs": runs, "host": os.environ.get("MODAL_TASK_ID") or socket.gethostname()}
    finally:
        shutil.rmtree(d, ignore_errors=True)


def _fn(cpu, mi):
    kw = dict(image=image, cpu=cpu, memory=4096, timeout=900, max_containers=100, name=f"judge_c{cpu}m{mi}")
    def deco(f):
        return app.function(**kw)(modal.concurrent(max_inputs=mi)(f))
    return deco


@_fn(1, 1)
def c1m1(tl: float): return _run(tl)
@_fn(2, 1)
def c2m1(tl: float): return _run(tl)
@_fn(2, 2)
def c2m2(tl: float): return _run(tl)
@_fn(2, 4)
def c2m4(tl: float): return _run(tl)
@_fn(4, 2)
def c4m2(tl: float): return _run(tl)
@_fn(4, 4)
def c4m4(tl: float): return _run(tl)


FNS = {"c2m4": c2m4, "c1m1": c1m1, "c2m1": c2m1, "c2m2": c2m2, "c4m2": c4m2, "c4m4": c4m4}
CPUS = {"c2m4": 2, "c1m1": 1, "c2m1": 2, "c2m2": 2, "c4m2": 4, "c4m4": 4}


def pct(xs, q):
    s = sorted(xs)
    return s[min(len(s) - 1, int(q * len(s)))] if s else float("nan")


def analyze(results: list[dict], idle: dict[str, float], tl: float) -> dict:
    runs = [r for res in results for r in res["runs"]]
    out = {"calls": len(results), "compile_failures": sum(not r["compiled"] for r in results), "runs": len(runs),
           "TLE": sum(r["status"] == "TLE" for r in runs), "RE": sum(r["status"] == "RE" for r in runs),
           "compile_s": {"p50": pct([r["compile_s"] for r in results], .5), "p99": pct([r["compile_s"] for r in results], .99)},
           "by_size": {}, "false_tle_at": {}}
    ref_out = {}
    for r in runs:
        ref_out.setdefault(r["size"], r["out"])
    out["wrong_output"] = sum(r["out"] != ref_out[r["size"]] for r in runs if r["status"] == "OK")
    for sz in SIZES:
        ts = [r["time"] for r in runs if r["size"] == sz]
        sl = [t / idle[sz] for t in ts]
        out["by_size"][sz] = {"n": len(ts), "idle_s": idle[sz], "p50_s": pct(ts, .5), "p99_s": pct(ts, .99), "max_s": max(ts),
                              "slowdown_p50": pct(sl, .5), "slowdown_p99": pct(sl, .99), "slowdown_max": max(sl)}
    # contention only: time vs the fastest same-size run seen on the SAME container (removes host-speed differences)
    best = {}
    for res in results:
        for r in res["runs"]:
            key = (res["host"], r["size"])
            best[key] = min(best.get(key, 9e9), r["time"])
    own = [r["time"] / best[(res["host"], r["size"])] for res in results for r in res["runs"] if r["size"] != "light"]
    out["contention_slowdown"] = {"p50": pct(own, .5), "p99": pct(own, .99), "max": max(own)}
    out["hosts"] = len({res["host"] for res in results})
    for k in (1.25, 1.5, 2.0, 3.0, 4.0):   # a solution whose idle time is tl/k: how often would it be called TLE?
        heavy = [r for r in runs if r["size"] != "light"]   # light (~50 ms) tests are pure jitter, not a limit risk
        out["false_tle_at"][f"limit={k}x idle"] = sum(r["time"] > idle[r["size"]] * k for r in heavy) / len(heavy)
    return out


def recommend(stats: dict[str, dict], cost_cpu: dict[str, int]) -> str:
    """Cheapest config whose slowdown p99 <= 1.25 and max <= 1.5 (heavy tests), else the least noisy."""
    def worst(s, key): return max(v[key] for v in s["by_size"].values() if v["idle_s"] > 0.1)
    ok = [c for c, s in stats.items() if s["TLE"] == 0 and worst(s, "slowdown_p99") <= 1.25 and worst(s, "slowdown_max") <= 1.5]
    if ok:
        best = min(ok, key=lambda c: (cost_cpu[c] / (int(c.split("m")[1])), cost_cpu[c]))  # cpu-seconds per concurrent slot
        return f"{best} (no false TLEs; slowdown p99<=1.25x, max<=1.5x; cheapest per concurrent judge among those)"
    best = min(stats, key=lambda c: worst(stats[c], "slowdown_p99"))
    return f"{best} is the least noisy, but none met slowdown p99<=1.25x and max<=1.5x - leave a time-limit margin (see false_tle_at)"


@app.local_entrypoint()
def main(n: int = 300, configs: str = "c1m1,c2m1,c2m2,c2m4,c4m2,c4m4", tl: float = 2.0, idle_calls: int = 8):
    # idle baseline: one warm container, sequential, plenty of cpu
    base = [FNS["c4m2"].remote(tl) for _ in range(idle_calls)]
    base = base[2:]  # drop the first calls (cold container)
    idle = {sz: st.median(r["time"] for res in base for r in res["runs"] if r["size"] == sz) for sz in SIZES}
    print("idle median seconds:", {k: round(v, 3) for k, v in idle.items()})
    stats = {}
    for c in configs.split(","):
        t0 = time.time()
        results = list(FNS[c].starmap([(tl,)] * n))
        s = analyze(results, idle, tl)
        s["wall_s"] = time.time() - t0
        stats[c] = s
        print(f"\n== {c}: {s['calls']} calls, {s['runs']} runs in {s['wall_s']:.0f}s; TLE={s['TLE']} RE={s['RE']} "
              f"wrong_output={s['wrong_output']} compile_fail={s['compile_failures']}; compile p50/p99 "
              f"{s['compile_s']['p50']:.1f}/{s['compile_s']['p99']:.1f}s")
        for sz, v in s["by_size"].items():
            print(f"   {sz:>6}: idle {v['idle_s']:.2f}s  p50 {v['p50_s']:.2f}  p99 {v['p99_s']:.2f}  max {v['max_s']:.2f}  "
                  f"slowdown p50 {v['slowdown_p50']:.2f}x p99 {v['slowdown_p99']:.2f}x max {v['slowdown_max']:.2f}x")
        c = s["contention_slowdown"]
        print(f"   within-container slowdown (medium+heavy, {s['hosts']} containers): p50 {c['p50']:.2f}x p99 {c['p99']:.2f}x max {c['max']:.2f}x")
        print("   false-TLE rate if limit were k x idle time:", {k: f"{100 * v:.2f}%" for k, v in s["false_tle_at"].items()})
    rec = recommend(stats, CPUS)
    print("\nRECOMMENDATION:", rec)
    os.makedirs("runs", exist_ok=True)
    json.dump({"idle": idle, "tl": tl, "n": n, "stats": stats, "recommendation": rec}, open("runs/judge_noise.json", "w"), indent=1)
