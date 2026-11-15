"""Eval analysis: trajectories (+ problems for ratings) -> pass rate by rating bucket, failure
categories, turn/token distributions, and a checkpoint comparison table.

    uv run python -m scripts.analyze runs/eval/baseline.jsonl runs/eval/round1.jsonl \
        --problems data/holdout.jsonl [--fine] [--json out.json] [--no-detail]
    # run args are PATH or NAME=PATH; the first run is the baseline for paired comparisons.

Input is the JSONL written by scripts.eval (one Trajectory.to_dict() per line; several lines per
problem id are fine = k attempts). Stdlib only; reads dicts, so it does not import cpp.harness.

Failure categories (mutually exclusive, applied to unsolved episodes in this order):
    infra_error      episode crashed (ended_by == "error")
    truncation       no compiling submission and some turn hit the token limit (finish_reason=length)
    malformed_call   no compiling submission and a malformed tool call / unparseable turn
    never_submitted  no submit call at all (ran out of turns/context while only using `run`, or EXIT early)
    never_compiled   submitted, but nothing ever compiled
    TLE              final submission failed the full tests with >=1 time-limit verdict
    runtime_error    ... else >=1 runtime-error verdict
    wrong_answer     ... else (WA, or too few tests passed)
"""
from __future__ import annotations

import argparse, json, math, os, statistics as st
from collections import Counter, defaultdict

BUCKETS = [("<1600", 0, 1600), ("1600-2199", 1600, 2200), ("2200+", 2200, 10**9)]
FAIL_ORDER = ["infra_error", "truncation", "malformed_call", "never_submitted", "never_compiled",
              "TLE", "runtime_error", "wrong_answer"]


# ------------------------------------------------------------------ loading
def load_jsonl(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def parse_run_arg(s: str) -> tuple[str, str]:
    if "=" in s and not os.path.exists(s):
        n, p = s.split("=", 1)
        return n, p
    return os.path.splitext(os.path.basename(s))[0], s


def rating_map(problems: list[dict] | None) -> dict[str, int]:
    return {p["id"]: int(p.get("rating") or 0) for p in problems or []}


# ------------------------------------------------------------------ per-episode
def solved(t: dict) -> bool:
    return bool((t.get("final") or {}).get("solved"))


def failure_category(t: dict) -> str | None:
    """None if solved, else one of FAIL_ORDER."""
    if solved(t):
        return None
    if t.get("ended_by") == "error":
        return "infra_error"
    final = t.get("final") or {}
    if not t.get("final_code"):
        if t.get("truncated", 0) > 0:
            return "truncation"
        if t.get("malformed", 0) > 0 or t.get("ended_by") == "malformed":
            return "malformed_call"
        return "never_compiled" if t.get("submissions") else "never_submitted"
    verdicts = final.get("verdicts") or []
    if "TLE" in verdicts:
        return "TLE"
    if "RE" in verdicts:
        return "runtime_error"
    return "wrong_answer"


def bucket_of(rating: int, fine: bool) -> str:
    if not rating:
        return "unrated"
    if fine:
        lo = rating // 200 * 200
        return f"{lo}-{lo + 199}"
    for name, lo, hi in BUCKETS:
        if lo <= rating < hi:
            return name
    return "unrated"


def bucket_sort_key(b: str) -> int:
    if b == "unrated":
        return -1
    return 0 if b.startswith("<") else int(b.strip("+").split("-")[0])


# ------------------------------------------------------------------ stats helpers
def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def quantile(xs: list[float], q: float) -> float:
    s = sorted(xs)
    if not s:
        return float("nan")
    i = (len(s) - 1) * q
    lo, hi = math.floor(i), math.ceil(i)
    return s[lo] + (s[hi] - s[lo]) * (i - lo)


def dist(xs: list[float]) -> dict:
    if not xs:
        return {"n": 0}
    return {"n": len(xs), "mean": st.mean(xs), "p10": quantile(xs, .1), "median": quantile(xs, .5),
            "p90": quantile(xs, .9), "max": max(xs)}


def histogram(xs: list[float], edges: list[float]) -> list[tuple[str, int]]:
    out, n = [], len(edges)
    for i in range(n + 1):
        lo = edges[i - 1] if i else None
        hi = edges[i] if i < n else None
        label = f"<{hi:g}" if lo is None else (f">={lo:g}" if hi is None else f"{lo:g}-{hi:g}")
        out.append((label, sum((lo is None or x >= lo) and (hi is None or x < hi) for x in xs)))
    return out


# ------------------------------------------------------------------ run summary
def analyze_run(trajs: list[dict], ratings: dict[str, int] | None = None, fine: bool = False) -> dict:
    ratings = ratings or {}
    n = len(trajs)
    if n == 0:
        raise ValueError("empty run")
    turns_total = sum(t.get("turns", 0) for t in trajs) or 1
    ok = [t for t in trajs if solved(t)]
    k_solved = len(ok)
    lo, hi = wilson(k_solved, n)

    by_bucket: dict[str, list[dict]] = defaultdict(list)
    for t in trajs:
        r = ratings.get(t["problem_id"]) or t.get("rating") or 0
        by_bucket[bucket_of(int(r), fine)].append(t)
    buckets = {}
    for b in sorted(by_bucket, key=bucket_sort_key):
        ts = by_bucket[b]
        ks = sum(solved(t) for t in ts)
        blo, bhi = wilson(ks, len(ts))
        buckets[b] = {"n": len(ts), "solved": ks, "rate": ks / len(ts), "ci": [blo, bhi]}

    fails = Counter(c for c in map(failure_category, trajs) if c)
    # per-problem view (k attempts): mean solve rate and pass@any
    per_problem: dict[str, list[bool]] = defaultdict(list)
    for t in trajs:
        per_problem[t["problem_id"]].append(solved(t))

    return {
        "n_episodes": n,
        "n_problems": len(per_problem),
        "pass_rate": k_solved / n,
        "pass_ci": [lo, hi],
        "pass_at_any": sum(any(v) for v in per_problem.values()) / len(per_problem),
        "malformed_rate_per_turn": sum(t.get("malformed", 0) for t in trajs) / turns_total,
        "episodes_with_malformed": sum(t.get("malformed", 0) > 0 for t in trajs) / n,
        "truncation_rate_per_turn": sum(t.get("truncated", 0) for t in trajs) / turns_total,
        "episodes_with_truncation": sum(t.get("truncated", 0) > 0 for t in trajs) / n,
        "exit_rate": sum(bool(t.get("ended_with_exit")) for t in trajs) / n,
        "no_submission_rate": sum(not t.get("final_code") for t in trajs) / n,
        "regressed_rate": sum(bool(t.get("regressed")) for t in trajs) / n,
        "ended_by": dict(Counter(t.get("ended_by") or "?" for t in trajs)),
        "avg_submissions": st.mean(len(t.get("submissions") or []) for t in trajs),
        "turns": dist([t.get("turns", 0) for t in trajs]),
        "turns_solved": dist([t.get("turns", 0) for t in ok]),
        "tokens": dist([t.get("output_tokens", 0) for t in trajs]),
        "tokens_solved": dist([t.get("output_tokens", 0) for t in ok]),
        "turn_hist": histogram([t.get("turns", 0) for t in trajs], [2, 4, 6, 9, 13, 20, 30]),
        "token_hist": histogram([t.get("output_tokens", 0) for t in trajs],
                                [2000, 5000, 10000, 16000, 24000, 32000]),
        "buckets": buckets,
        "failures": {c: fails[c] for c in FAIL_ORDER if fails[c]},
        "n_failed": n - k_solved,
    }


def paired(base: list[dict], other: list[dict]) -> dict | None:
    """Per-problem solve rates on the problems both runs cover."""
    def rates(ts):
        d = defaultdict(list)
        for t in ts:
            d[t["problem_id"]].append(solved(t))
        return {k: sum(v) / len(v) for k, v in d.items()}
    a, b = rates(base), rates(other)
    common = sorted(set(a) & set(b))
    if not common:
        return None
    gain = sum(b[p] > a[p] for p in common)
    loss = sum(b[p] < a[p] for p in common)
    return {"common": len(common), "better": gain, "worse": loss,
            "delta_pass": st.mean(b[p] - a[p] for p in common),
            "sign_test_p": sign_test_p(gain, loss)}


def sign_test_p(wins: int, losses: int) -> float:
    """Two-sided exact sign test (ties dropped)."""
    n = wins + losses
    if n == 0:
        return 1.0
    k = min(wins, losses)
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * p)


# ------------------------------------------------------------------ rendering
def md_table(header: list[str], rows: list[list[str]]) -> str:
    out = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(out)


def f1(x) -> str:
    return "-" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.1f}"


def render_comparison(names: list[str], res: dict[str, dict], pairs: dict[str, dict | None]) -> str:
    rows = []
    for n in names:
        r = res[n]
        p = pairs.get(n)
        d = "base" if n == names[0] else (
            f"{100 * p['delta_pass']:+.1f}pp ({p['better']}W/{p['worse']}L, p={p['sign_test_p']:.2f})" if p else "-")
        rows.append([n, str(r["n_episodes"]), f"{pct(r['pass_rate'])} [{pct(r['pass_ci'][0])}-{pct(r['pass_ci'][1])}]",
                     d, pct(r["malformed_rate_per_turn"]), pct(r["episodes_with_truncation"]),
                     f1(r["turns"]["mean"]), f1(r["tokens"]["mean"]), pct(r["exit_rate"]),
                     pct(r["no_submission_rate"])])
    t = md_table(["run", "episodes", "pass (95% CI)", "vs " + names[0], "malformed/turn", "trunc eps",
                  "avg turns", "avg tokens", "EXIT", "no submit"], rows)

    bnames = sorted({b for n in names for b in res[n]["buckets"]}, key=bucket_sort_key)
    brow = []
    for n in names:
        cells = []
        for b in bnames:
            x = res[n]["buckets"].get(b)
            cells.append(f"{pct(x['rate'])} ({x['solved']}/{x['n']})" if x else "-")
        brow.append([n] + cells)
    bt = md_table(["pass by rating"] + bnames, brow)

    frow = []
    for n in names:
        f = res[n]["failures"]
        frow.append([n] + [str(f.get(c, 0)) for c in FAIL_ORDER])
    ft = md_table(["failures (count)"] + FAIL_ORDER, frow)
    return f"## Comparison\n\n{t}\n\n### Pass rate by rating bucket\n\n{bt}\n\n### Failure categories\n\n{ft}\n"


def render_detail(name: str, r: dict) -> str:
    def drow(label, d):
        return [label, str(d["n"])] + ([f1(d[k]) for k in ("mean", "p10", "median", "p90", "max")]
                                       if d["n"] else ["-"] * 5)
    dt = md_table(["", "n", "mean", "p10", "median", "p90", "max"],
                  [drow("turns (all)", r["turns"]), drow("turns (solved)", r["turns_solved"]),
                   drow("output tokens (all)", r["tokens"]), drow("output tokens (solved)", r["tokens_solved"])])
    nf = max(r["n_failed"], 1)
    ft = md_table(["failure", "count", "% of failures"],
                  [[c, str(k), pct(k / nf)] for c, k in r["failures"].items()] or [["(none)", "0", "-"]])
    th = " · ".join(f"{l}: {c}" for l, c in r["turn_hist"])
    kh = " · ".join(f"{l}: {c}" for l, c in r["token_hist"])
    eb = ", ".join(f"{k}={v}" for k, v in sorted(r["ended_by"].items()))
    return (f"## {name}\n\npass {pct(r['pass_rate'])} on {r['n_episodes']} episodes / {r['n_problems']} problems "
            f"(pass@any {pct(r['pass_at_any'])}) · avg submissions {r['avg_submissions']:.1f} · "
            f"regressed {pct(r['regressed_rate'])}\n\nended by: {eb}\n\n{dt}\n\nturn histogram: {th}\n\n"
            f"token histogram: {kh}\n\n{ft}\n")


def weakest(name: str, r: dict, min_n: int = 5) -> str:
    cand = [(b, x) for b, x in r["buckets"].items() if x["n"] >= min_n and b != "unrated"]
    out = []
    if cand:
        b, x = min(cand, key=lambda kv: kv[1]["rate"])
        out.append(f"weakest rating bucket: {b} ({pct(x['rate'])}, n={x['n']})")
    if r["failures"]:
        c, k = max(r["failures"].items(), key=lambda kv: kv[1])
        out.append(f"top failure: {c} ({k}/{r['n_failed']} of failures)")
    return f"- **{name}**: " + "; ".join(out) if out else f"- **{name}**: (not enough data)"


def submission_row(name: str, r: dict) -> str:
    """Local-eval cell for submissions.md: pass / malformed / trunc / turns."""
    return (f"{pct(r['pass_rate'])} / {pct(r['malformed_rate_per_turn'])} / "
            f"{pct(r['episodes_with_truncation'])} / {r['turns']['mean']:.1f}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", help="trajectory JSONL files, PATH or NAME=PATH; first is the baseline")
    ap.add_argument("--problems", help="problems JSONL (id, rating) for the rating buckets")
    ap.add_argument("--fine", action="store_true", help="200-point rating buckets instead of the plan's 3")
    ap.add_argument("--no-detail", action="store_true", help="comparison tables only")
    ap.add_argument("--json", help="also write machine-readable results here")
    a = ap.parse_args(argv)

    ratings = rating_map(load_jsonl(a.problems)) if a.problems else {}
    names, raw, res = [], {}, {}
    for s in a.runs:
        n, p = parse_run_arg(s)
        while n in raw:
            n += "'"
        names.append(n)
        raw[n] = load_jsonl(p)
        res[n] = analyze_run(raw[n], ratings, a.fine)
    pairs = {n: paired(raw[names[0]], raw[n]) for n in names[1:]}

    out = [render_comparison(names, res, pairs), "### Where each run is weakest\n\n" +
           "\n".join(weakest(n, res[n]) for n in names) +
           "\n\n### submissions.md cell (pass / malformed per turn / trunc eps / avg turns)\n\n" +
           "\n".join(f"- {n}: `{submission_row(n, res[n])}`" for n in names) + "\n"]
    if not a.no_detail:
        out += [render_detail(n, res[n]) for n in names]
    if not ratings:
        out.append("_No --problems given: ratings unknown, everything is in the 'unrated' bucket._")
    print("\n".join(out))
    if a.json:
        with open(a.json, "w") as f:
            json.dump({"runs": res, "paired_vs_first": pairs}, f, indent=1)


if __name__ == "__main__":
    main()
