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
    assert B.build(prob("a", checker="python x"), B.parse_trace({"id": "a", "generation": gen(SUM)}),
                   p["full_tests"])["reason"] == "unsupported_checker"


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
    B.main(["--problems", str(pf), "--traces", str(tf), "--exclude", str(ef), "--tests-root", str(tmp_path / "t"),
            "--out", str(out)])
    rows = [json.loads(l) for l in out.read_text().splitlines()]
    assert [r["id"] for r in rows] == ["a"] and rows[0]["ok"]
