import json, shutil, pytest

from scripts import build_r1 as B

pytestmark = pytest.mark.skipif(not shutil.which("g++"), reason="needs g++")
SUM = '#include <bits/stdc++.h>\nint main(){long long a,b;std::cin>>a>>b;std::cout<<a+b<<"\\n";}'


def gen(code, think="Add them."):
    return f"<think>\n{think}\n</think>\n\nSolution:\n```cpp\n{code}\n```"


def prob(pid, rating=2300, **kw):
    tests = [{"input": "1 2\n", "output": "3\n"}, {"input": "7 8\n", "output": "15\n"}]
    return {"id": pid, "statement": "Add.", "rating": rating, "time_limit": 2.0, "memory_limit": 256.0,
            "sample_tests": tests[:1], "full_tests": tests, "checker": None, **kw}


def test_parse_trace():
    assert B.parse_trace({"id": "a", "generation": gen(SUM)})["code"] == SUM
    assert B.parse_trace({"id": "a", "generation": "no think ```cpp\nx\n```"}) is None
    assert B.parse_trace({"id": "a", "generation": gen(SUM), "finish_reason": "length"}) is None


def test_build_good_and_bad():
    p = prob("a")
    ok = B.build(p, B.parse_trace({"id": "a", "generation": gen(SUM)}), p["full_tests"])
    assert ok["ok"] and [m["role"] for m in ok["messages"]] == ["user", "assistant", "tool", "assistant", "tool", "assistant"]
    assert "3" in ok["messages"][2]["content"] and "passed" in ok["messages"][4]["content"]
    assert ok["messages"][-1]["content"][-1]["text"] == "EXIT"
    bad = B.build(p, B.parse_trace({"id": "a", "generation": gen(SUM.replace("a+b", "a-b"))}), p["full_tests"])
    assert not bad["ok"] and bad["reason"] == "wrong_answer"
    assert B.build(p, B.parse_trace({"id": "a", "generation": gen("int main(){")}), p["full_tests"])["reason"] == "compile_error"


PYCHK = "import sys\nx=int(open(sys.argv[3]).read().split()[0]); a,b=map(int,open(sys.argv[1]).read().split())\nprint(1 if x==a+b else 0)\n"


def test_python_checker_supported():
    # python checkers (checker_lang) are judged, not dropped; the checker ignores a wrong jury answer
    p = prob("a", checker=PYCHK, checker_lang="python")
    p["full_tests"][1]["output"] = "999\n"            # token diff would say WA; checker only looks at input
    tr = B.parse_trace({"id": "a", "generation": gen(SUM)})
    r = B.build(p, tr, p["full_tests"])
    assert r["ok"], r
    bad = B.build(p, B.parse_trace({"id": "a", "generation": gen(SUM.replace("a+b", "a-b"))}), p["full_tests"])
    assert not bad["ok"] and bad["reason"] == "wrong_answer"


def test_preparsed_rows_and_modal_style_judge():
    row = {"id": "a", "rating": 2500, "tokens": 123, "thinking": "reason", "code": SUM, "in_pool": True}
    tr = B.parse_trace(row)
    assert tr == {"id": "a", "thinking": "reason", "code": SUM, "tokens": 123}
    p = prob("a")
    ptr = {"path": "/vol/pool/tests/a.json", "n": 2}
    calls = []

    def fake_modal(code, tests, tl, ml, chk, stop_on_fail=True, chk_lang=None):
        calls.append(tests)     # the pointer must reach the judge unresolved (Modal reads it from the volume)
        from cpp import judge
        t = p["full_tests"] if tests is ptr else tests
        return judge.judge(code, t, tl, ml, chk, stop_on_fail, chk_lang)

    r = B.build(p, tr, ptr, judge_fn=fake_modal)
    assert r["ok"] and r["n_tests"] == 2 and calls[0] is ptr
    slow = B.build(p, tr, ptr, max_time_frac=0.0, judge_fn=fake_modal)
    assert slow["reason"] == "slow"


def test_cli_pointer_tests_and_exclusion(tmp_path):
    p1, p2, p3 = prob("a"), prob("held"), prob("easy", rating=1000)
    tests = p1["full_tests"]
    (tmp_path / "t").mkdir()
    (tmp_path / "t" / "a.json").write_text(json.dumps({"tests": tests}))
    p1["full_tests"] = {"path": "/vol/pool/tests/a.json", "n": 2}
    pf, tf, ef = tmp_path / "p.jsonl", tmp_path / "tr.jsonl", tmp_path / "e.jsonl"
    pf.write_text("\n".join(map(json.dumps, [p1, p2, p3])))
    tf.write_text("\n".join(json.dumps({"id": i, "generation": gen(SUM)}) for i in ["a", "held", "easy"]))
    ef.write_text(json.dumps({"id": "held"}))
    out = tmp_path / "o.jsonl"
    args = ["--problems", str(pf), "--traces", str(tf), "--exclude", str(ef), "--tests-root", str(tmp_path / "t"),
            "--judge", "local", "--out", str(out)]
    B.main(args)
    rows = [json.loads(l) for l in out.read_text().splitlines()]
    assert [r["id"] for r in rows] == ["a"] and rows[0]["ok"]
    # too-long traces are counted as a drop reason, not silently skipped
    B.main(args + ["--max-tokens", "1", "--keep-failures"])
    rows = [json.loads(l) for l in out.read_text().splitlines()]
    assert rows[0]["reason"] == "too_long"
