"""Turn raw harness episodes (teacher P2 or student P3) into a training pool.

Keeps an episode only if its final submission passed every test, it ended with EXIT, and it never
produced a malformed or truncated turn. Per problem, keeps the shortest few. Debugging turns stay.

uv run python -m scripts.filter_traj data/pools/p2_raw.jsonl --out data/pools/p2.jsonl --per-problem 1
"""
import argparse, collections, json, re


def wasted(ep) -> bool:
    """Idle or repeated turns: same code submitted or run twice in a row, or submit after a passing submit."""
    prev, passed = None, False
    for m in ep["messages"]:
        if m["role"] == "tool" and m.get("name") == "submit":
            g = re.search(r"(\d+)/(\d+) sample tests passed", m["content"])
            passed = bool(g) and g.group(1) == g.group(2)
        for tc in m.get("tool_calls") or []:
            key = (tc["function"]["name"], tc["function"]["arguments"])
            if key == prev or (passed and tc["function"]["name"] == "submit"):
                return True
            prev = key
    return False


def main(a):
    by = collections.defaultdict(list)
    n = 0
    c = collections.Counter()
    for path in a.inp:
        for l in open(path):
            ep = json.loads(l)
            n += 1
            if not (ep.get("score") or {}).get("solved"):
                c["unsolved"] += 1
            elif ep["ended_by"] != "exit":
                c["no_exit"] += 1
            elif ep["malformed"] or ep["truncated"]:
                c["malformed_or_truncated"] += 1
            elif ep["turns"] > a.max_turns or wasted(ep):
                c["wasted_turns"] += 1
            else:
                c["kept"] += 1
                by[ep["problem_id"]].append(ep)
    out = 0
    with open(a.out, "w") as f:
        for pid, eps in by.items():
            eps.sort(key=lambda e: (e["turns"], e["output_tokens"]))
            for ep in eps[: a.per_problem]:
                r = ep.get("rating") or 0
                f.write(json.dumps({"id": pid, "ok": True, "kind": "debug" if ep["submissions"] > 1 or ep["turns"] > 3 else "happy",
                                    "rating": r, "band": "hard" if r >= 1900 else "medium" if r >= 1300 else "easy",
                                    "source": ep.get("source"), "turns": ep["turns"], "messages": ep["messages"]}) + "\n")
                out += 1
    print(n, "episodes;", dict(c), "; problems solved:", len(by), "; written:", out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("inp", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--per-problem", type=int, default=1)
    ap.add_argument("--max-turns", type=int, default=10)
    main(ap.parse_args())
