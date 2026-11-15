"""Rewrite trajectories sampled in our harness format into the real Vals harness format.

    uv run python -m scripts.to_vals_format data/selected_r6.norm.jsonl --out data/selected_r6.vals.jsonl

Renames tools (run_cpp->cpp_executor, submit->submission), the `code` argument (->cpp_code), turns tool-result text
into the JSON the real tools return, moves the instructions into the user message and drops the system message.
Rows get the new `tools` / `system` ("" = no system prompt) so train_sft rebuilds the right prefix.
"""
from __future__ import annotations

import argparse, json, re

from cpp import tools as T

VERDICT = {"passed": "passed", "wrong answer": "wrong answer, got:\n[output not shown]",
           "time limit exceeded": "time limit exceeded", "runtime error": "runtime error (exit code 1): ",
           "judge error": "runtime error (exit code 1): "}
RENAME = {"run_cpp": "cpp_executor", "submit": "submission"}


def conv_run(text: str) -> str:
    if text.startswith("Compilation error:\n"):
        return "Compilation failed:\n" + text[len("Compilation error:\n"):]
    if text.startswith("Time limit exceeded"):
        return "Execution timed out after 10 seconds"
    m = re.match(r"(Exit code (\d+)|Runtime error \(exit code (-?\d+)\))\. Time: [\d.]+ s\nstdout:\n", text, re.S)
    if not m:
        return text
    code = int(m.group(2) or m.group(3))
    rest = text[m.end():]
    out, err = rest, ""
    if "\nstderr:\n" in rest:
        out, err = rest.rsplit("\nstderr:\n", 1)
    return json.dumps({"exit_code": code, "stdout": out, "stderr": err})


def conv_submit(text: str) -> str:
    if text.startswith("Compilation error:\n"):
        n = re.search(r"Submissions used: (\d+)/", text)
        return text.replace("Compilation error:\n", "Compilation failed:\n", 1)  # no counter in our compile-error text
    samples, used = {}, None
    for line in text.splitlines():
        m = re.match(r"Sample (\d+): (.+)$", line)
        if m:
            samples[m.group(1)] = VERDICT.get(m.group(2), m.group(2))
        u = re.search(r"Submissions used: (\d+)/(\d+)", line)
        if u:
            used = int(u.group(1))
    if not samples:
        return text
    return json.dumps({"saved": True, "samples": samples,
                       "submissions_remaining": T.MAX_SUBMISSIONS - (used or 0)})


def convert(row: dict) -> dict:
    msgs, tool_names = [], {}
    for m in row["messages"]:
        m = dict(m)
        if m["role"] == "system":
            continue
        if m["role"] == "user" and not msgs:
            m["content"] = T.VALS_INSTRUCTIONS.format(question=m["content"])
        if m["role"] == "assistant" and m.get("tool_calls"):
            new = []
            for tc in m["tool_calls"]:
                tc = json.loads(json.dumps(tc))
                f = tc["function"]
                tool_names[tc.get("id")] = f["name"]
                f["name"] = RENAME.get(f["name"], f["name"])
                a = f["arguments"]
                a = json.loads(a) if isinstance(a, str) else a
                if "code" in a:
                    a["cpp_code"] = a.pop("code")
                f["arguments"] = json.dumps(a) if isinstance(tc["function"]["arguments"], str) else a
                new.append(tc)
            m["tool_calls"] = new
        if m["role"] == "tool":
            orig = tool_names.get(m.get("tool_call_id")) or m.get("name")
            m["name"] = RENAME.get(m.get("name"), m.get("name"))
            if orig == "run_cpp":
                m["content"] = conv_run(m["content"])
            elif orig == "submit":
                m["content"] = conv_submit(m["content"])
        msgs.append(m)
    out = {**row, "messages": msgs, "tools": T.VALS_TOOLS, "system": ""}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data", nargs="+")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    n = 0
    with open(a.out, "w") as f:
        for p in a.data:
            for l in open(p):
                f.write(json.dumps(convert(json.loads(l))) + "\n")
                n += 1
    print(f"converted {n} rows -> {a.out}")


if __name__ == "__main__":
    main()
