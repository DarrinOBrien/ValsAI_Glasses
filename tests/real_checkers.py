"""Validate the judge against REAL open-r1 python checkers from Track B's pool.

No reference solutions exist, so each test uses an 'oracle' C++ program that looks the input up in a table of
the problem's own tests and prints the jury answer (-> must be AC on every test), plus a 'garbage' program
that prints "0" (-> should be WA where the checker is sensitive). Uses the full pipeline (compile, run,
python checker), with tests resolved from {path} like the harness does.

Modal:  uv run modal run modal_app.py::checkers --pool path/to/eval150.jsonl --n 8
"""
from cpp import judge

ORACLE_HEAD = '#include <bits/stdc++.h>\nint main(){ std::string in((std::istreambuf_iterator<char>(std::cin)), {});\n'
GARBAGE = '#include <cstdio>\nint main(){ puts("0"); }'


def oracle_src(tests: list[dict]) -> str:
    body = ORACLE_HEAD
    for t in tests:
        body += f' if (in == R"~~~({t["input"]})~~~") {{ fputs(R"~~~({t["output"]})~~~", stdout); return 0; }}\n'
    return body + " return 3; }\n"


def check_problem(p: dict, max_tests: int = 6, max_bytes: int = 150_000) -> dict:
    tests = judge.resolve_tests(p["full_tests"])
    pick, size = [], 0
    for t in tests:
        sz = len(t["input"]) + len(t["output"])
        if sz + size <= max_bytes and sz < 60_000:
            pick.append(t); size += sz
        if len(pick) >= max_tests:
            break
    if not pick:
        return {"id": p["id"], "skipped": "tests too large"}
    ck, lang = p.get("checker"), p.get("checker_lang")
    good = judge.judge(oracle_src(pick), pick, p["time_limit"], max(p["memory_limit"], 256), ck, False, lang)
    bad = judge.judge(GARBAGE, pick, p["time_limit"], max(p["memory_limit"], 256), ck, False, lang)
    return {"id": p["id"], "n": len(pick), "compiled": good["compiled"], "oracle": good["verdicts"],
            "garbage": bad["verdicts"], "ok": good["verdicts"] == ["AC"] * len(pick) and "AC" not in bad["verdicts"][:1]}
