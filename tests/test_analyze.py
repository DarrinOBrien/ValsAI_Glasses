import json
from scripts import analyze as A


def T(pid, solved=False, code="x", verdicts=None, **kw):
    d = dict(problem_id=pid, turns=3, output_tokens=1000, malformed=0, truncated=0, ended_with_exit=solved,
             ended_by="exit" if solved else "max_turns", regressed=False, submissions=[{"n": 1}] if code else [],
             final_code=code, final={"solved": solved, "verdicts": verdicts or []})
    d.update(kw)
    return d


CASES = {
    "infra_error": T("a", ended_by="error", code=None),
    "truncation": T("a", code=None, truncated=1, malformed=1, ended_by="malformed"),
    "malformed_call": T("a", code=None, malformed=3, ended_by="malformed"),
    "never_submitted": T("a", code=None, submissions=[]),
    "never_compiled": T("a", code=None, submissions=[{"n": 1}]),
    "TLE": T("a", verdicts=["AC", "TLE", "WA"]),
    "runtime_error": T("a", verdicts=["AC", "RE"]),
    "wrong_answer": T("a", verdicts=["AC", "WA"]),
}


def test_failure_categories():
    for want, t in CASES.items():
        assert A.failure_category(t) == want, want
    assert A.failure_category(T("a", solved=True)) is None


def test_analyze_buckets_and_compare():
    ratings = {"e": 900, "m": 1800, "h": 2400}
    base = [T("e", True), T("m"), T("h", code=None, truncated=1), T("h", verdicts=["TLE"])]
    better = [T("e", True), T("m", True, turns=2), T("h", True, turns=5), T("h", verdicts=["WA"])]
    rb, rn = A.analyze_run(base, ratings), A.analyze_run(better, ratings)
    assert rb["pass_rate"] == 0.25 and rn["pass_rate"] == 0.75
    assert rb["buckets"]["2200+"]["n"] == 2 and rb["buckets"]["<1600"]["rate"] == 1.0
    assert rb["failures"] == {"truncation": 1, "TLE": 1, "wrong_answer": 1}
    assert A.analyze_run(base, ratings, fine=True)["buckets"]["2400-2599"]["n"] == 2
    p = A.paired(base, better)
    assert p["common"] == 3 and p["better"] == 2 and p["worse"] == 0   # h: 0/2 -> 1/2
    assert 0 < A.wilson(1, 4)[0] < 0.25 < A.wilson(1, 4)[1] < 1
    assert A.sign_test_p(5, 0) == 0.0625 and A.sign_test_p(0, 0) == 1.0


def test_cli(tmp_path, capsys):
    pr = tmp_path / "p.jsonl"
    pr.write_text("\n".join(json.dumps({"id": i, "rating": r}) for i, r in [("e", 900), ("m", 1800), ("h", 2400)]))
    for n, ts in (("base", [T("e", True), T("m"), T("h")]), ("new", [T("e", True), T("m", True), T("h")])):
        (tmp_path / f"{n}.jsonl").write_text("\n".join(json.dumps(t) for t in ts))
    A.main([str(tmp_path / "base.jsonl"), "x=" + str(tmp_path / "new.jsonl"), "--problems", str(pr),
            "--json", str(tmp_path / "o.json")])
    out = capsys.readouterr().out
    assert "## Comparison" in out and "| x |" in out and "1600-2199" in out and "weakest" in out
    assert json.load(open(tmp_path / "o.json"))["runs"]["x"]["pass_rate"] > 0.6


def test_sample_py_row_format():
    row = {"problem_id": "a", "turns": 3, "output_tokens": 10, "malformed": 0, "ended_by": "exit",
           "submissions": [{"compiled": True}], "score": {"solved": True}}
    assert A.solved(row) and A.failure_category(row) is None
    row["score"] = {"solved": False}
    assert A.failure_category(row) == "wrong_answer"
    from scripts import grpo
    assert grpo.pass_counts([{"problem_id": "a", "score": {"solved": True}}]) == {"a": (1, 1)}
