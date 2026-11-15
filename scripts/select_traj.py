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
import argparse, collections, json


def ok(t: dict) -> bool:
    if t.get("error") or not (t.get("score") or {}).get("solved"):
        return False
    if t.get("malformed") or t.get("truncated") or str(t.get("ended_by", "")).lower() != "exit":
        return False
    msgs = t.get("messages") or []
    last = next((m for m in reversed(msgs) if m["role"] == "assistant"), None)
    if last is None or "EXIT" not in (last.get("content") or ""):
        return False
    subs = [s for s in (t.get("submissions") or []) if s.get("compiled", True)]
    if subs:
        if not subs[-1].get("full_passed"):
            return False
        first = next(i for i, s in enumerate(subs) if s.get("full_passed"))
        if any(not s.get("full_passed") for s in subs[first:]):
            return False
    return True


def main(a):
    rows = [json.loads(l) for f in a.files for l in open(f)]
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
    print(f"{len(rows)} attempts over {n} problems; {sol} problems with a clean pass; kept {len(keep)} trajectories "
          f"(avg turns {sum(t['turns'] for t in keep)/max(1,len(keep)):.1f}, avg out tokens "
          f"{sum(t['output_tokens'] for t in keep)/max(1,len(keep)):.0f}) -> {a.out}")
    hist = collections.Counter(s["passed"] for s in stats.values())
    print("clean passes per problem histogram:", dict(sorted(hist.items())))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--out", required=True)
    main(ap.parse_args())
