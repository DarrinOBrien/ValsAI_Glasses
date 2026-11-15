import json
from scripts.select_traj import synth_exit, ok


def _sub(code):
    return {"role": "assistant", "content": "", "tool_calls": [{"function": {"name": "submit", "arguments": json.dumps({"code": code})}}]}


def _res(p, n=3):
    return {"role": "tool", "content": f"{p}/{n} sample tests passed. Submissions used: 1/50."}


def row(msgs, **kw):
    r = {"problem_id": "x", "rating": 1800, "messages": msgs, "turns": 9, "output_tokens": 900, "malformed": 0,
         "truncated": False, "ended_by": "max_turns", "score": {"solved": True}, "submissions": [], "regressed": False}
    r.update(kw)
    return r


def test_cut_after_first_passing_submission_of_final_code():
    msgs = [{"role": "user", "content": "p"}, _sub("A"), _res(1), _sub("B"), _res(3), _sub("B"), _res(3), _sub("B"), _res(3)]
    new = synth_exit(row(msgs))
    assert new["messages"][-1] == {"role": "assistant", "content": "EXIT", "reasoning_content": ""}
    assert len(new["messages"]) == 6 and new["messages"][-2]["content"].startswith("3/3")  # user, A, res, B, res, EXIT
    assert new["turns"] == 3 and new["ended_by"] == "exit" and ok(new)


def test_not_rescued_when_unsolved_or_malformed_or_never_passing_samples():
    msgs = [{"role": "user", "content": "p"}, _sub("A"), _res(1)]
    assert synth_exit(row(msgs)) is None
    msgs = [{"role": "user", "content": "p"}, _sub("A"), _res(3)]
    assert synth_exit(row(msgs, score={"solved": False})) is None
    assert synth_exit(row(msgs, malformed=1)) is None
