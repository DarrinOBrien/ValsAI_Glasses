"""Unclosed-<think> handling: measure it, normalize SFT rows so every tool-call turn closes </think>, filter.

Qwen3.5 is prefilled with "<think>\\n" and often emits its <tool_call> without ever writing </think> (about half of
baseline turns in our smoke test). Our Tinker renderer recovers such calls, but a server-side reasoning parser may not,
and then the call is malformed. We want training turns to look like  <think>\\nREASONING\\n</think>\\n\\n<tool_call>.

Detection works on chat messages (the only thing every trajectory file has): an assistant turn WITH tool_calls and
NO reasoning (reasoning_content / thinking part empty) is an "unclosed" turn - the model never produced a closed think
block before the call. Two kinds:
  unclosed_text   visible text before the call (really thinking that was never closed) -> normalizer MOVES it into
                  the reasoning field, so the rendered turn becomes <think>text</think><tool_call>
  unclosed_empty  no text at all (model jumped straight to the call) -> re-renders as an empty closed think block;
                  kept as is, but --max-empty-frac can drop trajectories that are mostly like this (they teach
                  "skip thinking")

    uv run python -m scripts.thinkfix stats  runs/smoke_real.jsonl data/trajectories/r1.jsonl
    uv run python -m scripts.thinkfix normalize data/selected_r1.jsonl --out data/selected_r1.norm.jsonl \
        [--max-empty-frac 0.5]
Rows keep their representation: messages carrying reasoning_content stay that way; content-part lists keep
{"type": "thinking"} parts. Stdlib only.
"""
from __future__ import annotations

import argparse, copy, json


def _parts(m: dict) -> list[dict]:
    c = m.get("content")
    return c if isinstance(c, list) else []


def reasoning_of(m: dict) -> str:
    rc = m.get("reasoning_content") or ""
    return rc if rc.strip() else "".join(p.get("thinking", "") for p in _parts(m) if p.get("type") == "thinking")


def text_of(m: dict) -> str:
    c = m.get("content")
    if isinstance(c, str):
        return c
    return "".join(p.get("text", "") for p in _parts(m) if p.get("type") == "text")


def classify(m: dict) -> str | None:
    """None for non-tool-call turns; else closed | unclosed_text | unclosed_empty."""
    if m.get("role") != "assistant" or not m.get("tool_calls"):
        return None
    if reasoning_of(m).strip():
        return "closed"
    return "unclosed_text" if text_of(m).strip() else "unclosed_empty"


def turn_counts(messages: list[dict]) -> dict:
    out = {"closed": 0, "unclosed_text": 0, "unclosed_empty": 0}
    for m in messages:
        k = classify(m)
        if k:
            out[k] += 1
    out["tool_turns"] = sum(out[k] for k in ("closed", "unclosed_text", "unclosed_empty"))
    return out


def normalize_messages(messages: list[dict]) -> list[dict]:
    """Move pre-call text of unclosed turns into the reasoning field (returns a deep copy)."""
    out = copy.deepcopy(messages)
    for m in out:
        if classify(m) != "unclosed_text":
            continue
        text = text_of(m).strip()
        if isinstance(m.get("content"), list):
            m["content"] = [{"type": "thinking", "thinking": text}]
        else:
            m["reasoning_content"], m["content"] = text, ""
    return out


def normalize_row(row: dict, max_empty_frac: float = 1.0) -> tuple[dict | None, dict]:
    """-> (row or None if filtered, counts of the ORIGINAL turns)."""
    c = turn_counts(row.get("messages") or [])
    if c["tool_turns"] and c["unclosed_empty"] / c["tool_turns"] > max_empty_frac:
        return None, c
    return {**row, "messages": normalize_messages(row["messages"])}, c


def load(paths):
    for p in paths:
        for l in open(p):
            if l.strip():
                yield json.loads(l)


def summarize(rows) -> dict:
    tot = {"closed": 0, "unclosed_text": 0, "unclosed_empty": 0, "tool_turns": 0}
    eps = eps_un = 0
    for r in rows:
        c = turn_counts(r.get("messages") or [])
        for k in tot:
            tot[k] += c[k]
        eps += 1
        eps_un += (c["unclosed_text"] + c["unclosed_empty"]) > 0
    n = max(tot["tool_turns"], 1)
    return {**tot, "episodes": eps, "episodes_with_unclosed": eps_un,
            "unclosed_rate": (tot["unclosed_text"] + tot["unclosed_empty"]) / n}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("stats")
    s.add_argument("files", nargs="+")
    n = sub.add_parser("normalize")
    n.add_argument("files", nargs="+")
    n.add_argument("--out", required=True)
    n.add_argument("--max-empty-frac", type=float, default=1.0,
                   help="drop trajectories where more than this fraction of tool-call turns have no reasoning at all")
    a = ap.parse_args(argv)
    if a.cmd == "stats":
        for f in a.files:
            print(f, json.dumps(summarize(load([f]))))
        return
    kept = dropped = moved = 0
    with open(a.out, "w") as f:
        for r in load(a.files):
            nr, c = normalize_row(r, a.max_empty_frac)
            if nr is None:
                dropped += 1
                continue
            kept += 1
            moved += c["unclosed_text"]
            f.write(json.dumps(nr) + "\n")
    print(f"kept {kept}, dropped {dropped} (mostly-empty reasoning), moved text into reasoning on {moved} turns -> {a.out}")


if __name__ == "__main__":
    main()
