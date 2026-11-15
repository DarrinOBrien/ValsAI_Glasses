"""Select SFT trajectories from sampled attempts (expert iteration).

uv run python -m scripts.select_traj data/trajectories/r1.jsonl --out data/selected_r1.jsonl

Keep an attempt only if
  1. the final submission passes ALL full tests (score.solved),
  2. no malformed tool calls and it ends with EXIT (ended_by == "exit", no truncated turns),
  3. no later submission replaced a passing one with worse code (when per-submission results exist: the
     last compiling submission passes, and no submission after the first full pass failed).
Then per problem keep the shortest by (turns, output_tokens): one, or two for rating >= 2200.
Also writes pass-rate stats per problem (for rebalancing the round-2 pool) to <out>.stats.json.
"""
import argparse, collections, json, re

_ALL_PASS = re.compile(r"(\d+)/(\d+) sample tests passed")


def _submit_code(m: dict):
    for c in m.get("tool_calls") or []:
        f = c.get("function") or {}
        if f.get("name") == "submit":
            try:
                a = f.get("arguments")
                return (json.loads(a) if isinstance(a, str) else a).get("code")
            except Exception:
                return None
    return None


def _samples_all_pass(tool_msg: dict) -> bool:
    m = _ALL_PASS.search(str(tool_msg.get("content") or ""))
    return bool(m) and m.group(1) == m.group(2) and int(m.group(2)) > 0


def synth_exit(t: dict, add_exit: bool = True) -> dict | None:
    """Solved attempts that never wrote EXIT (the model keeps resubmitting passing code until the turn cap):
    cut right after the FIRST submission whose code equals the final graded code and passes all samples, then append
    an `EXIT` turn. The cut keeps exactly the code that passed the full tests, so the label stays correct, and it
    teaches the stop-after-passing habit (fewer turns win ties). Returns a new row, or None if it can't be done."""
    if t.get("error") or not (t.get("score") or {}).get("solved") or t.get("malformed") or t.get("truncated"):
        return None
    msgs = t.get("messages") or []
    subs = [(i, _submit_code(m)) for i, m in enumerate(msgs) if m["role"] == "assistant" and _submit_code(m)]
    if not subs:
        return None
    final_code = subs[-1][1]
    for i, code in subs:
        if code == final_code and i + 1 < len(msgs) and msgs[i + 1]["role"] == "tool" and _samples_all_pass(msgs[i + 1]):
            kept = msgs[: i + 2]
            frac = max(1, sum(len(str(m.get("content") or "")) + len(str(m.get("reasoning_content") or "")) for m in kept)) / \
                max(1, sum(len(str(m.get("content") or "")) + len(str(m.get("reasoning_content") or "")) for m in msgs))
            new = dict(t)
            if add_exit:
                new["messages"] = kept + [{"role": "assistant", "content": "EXIT",
                                           "reasoning_content": "The solution passes all the sample tests, so I am finished."}]
            else:  # cut only: the last training target is the passing submit; no EXIT is taught (see r1b post-mortem)
                new["messages"] = kept
                new["synth_cut"] = True
            new["turns"] = sum(m["role"] == "assistant" for m in new["messages"])
            new["output_tokens"] = int((t.get("output_tokens") or 0) * frac)
            new["ended_by"], new["regressed"], new["synth_exit"] = "exit", False, True
            new["submissions"] = []
            return new
    return None


def ok(t: dict) -> bool:
    if t.get("error") or not (t.get("score") or {}).get("solved"):
        return False
    if t.get("malformed") or t.get("truncated") or str(t.get("ended_by", "")).lower() != "exit":
        return False
    msgs = t.get("messages") or []
    last = next((m for m in reversed(msgs) if m["role"] == "assistant"), None)
    if last is None or (not t.get("synth_cut") and "EXIT" not in (last.get("content") or "")):
        return False
    if t.get("regressed"):  # harness flag: a later submission replaced a passing one with worse code
        return False
    subs = [s for s in (t.get("submissions") or []) if s.get("compiled", True)]
    if subs and any("full_passed" in s for s in subs):  # harness rows only carry sample results per submission
        if not subs[-1].get("full_passed"):
            return False
        first = next(i for i, s in enumerate(subs) if s.get("full_passed"))
        if any(not s.get("full_passed") for s in subs[first:]):
            return False
    return True


def main(a):
    rows = [json.loads(l) for f in a.files for l in open(f)]
    n_synth = 0
    if a.exit_mode != "off":
        out = []
        for r in rows:
            if str(r.get("ended_by", "")).lower() != "exit":
                new = synth_exit(r, add_exit=a.exit_mode == "exit")
                if new is not None:
                    n_synth += 1
                    r = new
            out.append(r)
        rows = out
    by = collections.defaultdict(list)
    for r in rows:
        by[r["problem_id"]].append(r)
    keep, stats = [], {}
    for pid, ts in by.items():
        good = sorted((t for t in ts if ok(t)), key=lambda t: (t["turns"], t["output_tokens"]))
        rating = next((t["rating"] for t in ts if t.get("rating")), None)
        stats[pid] = {"rating": rating, "attempts": len(ts), "passed": len(good),
                      "solved": sum(bool((t.get("score") or {}).get("solved")) for t in ts)}
        keep += good[: 2 if (rating or 0) >= 2200 else 1]
    with open(a.out, "w") as f:
        for t in keep:
            f.write(json.dumps({k: t[k] for k in ("problem_id", "rating", "messages", "turns", "output_tokens",
                                                   "system", "tools") if k in t}) + "\n")
    json.dump(stats, open(a.out + ".stats.json", "w"))
    n = len(by)
    sol = sum(s["passed"] > 0 for s in stats.values())
    print(f"rescued {n_synth} solved attempts that never exited (exit-mode={a.exit_mode})")
    print(f"{len(rows)} attempts over {n} problems; {sol} problems with a clean pass; kept {len(keep)} trajectories "
          f"(avg turns {sum(t['turns'] for t in keep)/max(1,len(keep)):.1f}, avg out tokens "
          f"{sum(t['output_tokens'] for t in keep)/max(1,len(keep)):.0f}) -> {a.out}")
    hist = collections.Counter(s["passed"] for s in stats.values())
    print("clean passes per problem histogram:", dict(sorted(hist.items())))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--exit-mode", choices=["cut", "exit", "off"], default="cut",
                    help="solved attempts that never wrote EXIT: cut = stop after the first passing submission, no EXIT target "
                         "(default; a synthetic EXIT collapsed r1b into exit-after-run_cpp); exit = also append a synthetic EXIT; off = drop them")
    main(ap.parse_args())
