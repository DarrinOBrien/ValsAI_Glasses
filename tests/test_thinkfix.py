import json

from scripts import thinkfix as F, analyze as A

CALL = [{"id": "c", "type": "function", "function": {"name": "submit", "arguments": "{}"}}]


def asst(reasoning="", text="", call=True, parts=False):
    m = {"role": "assistant", "tool_calls": CALL} if call else {"role": "assistant"}
    if parts:
        m["content"] = ([{"type": "thinking", "thinking": reasoning}] if reasoning else []) + ([{"type": "text", "text": text}] if text else [])
    else:
        m["content"] = text
        if reasoning:
            m["reasoning_content"] = reasoning
    return m


def test_classify():
    assert F.classify(asst("why", "")) == "closed"
    assert F.classify(asst("", "thinking aloud")) == "unclosed_text"
    assert F.classify(asst("", "")) == "unclosed_empty"
    assert F.classify(asst("", "EXIT", call=False)) is None
    assert F.classify(asst("why", "", parts=True)) == "closed"
    assert F.classify({"role": "user", "content": "x"}) is None


def test_normalize_moves_text_both_representations():
    msgs = [{"role": "user", "content": "q"}, asst("", "I should submit.\n\n"), asst("", "I should submit.", parts=True),
            asst("", ""), asst("r", "", call=True), asst("", "EXIT", call=False)]
    out = F.normalize_messages(msgs)
    assert out[1]["reasoning_content"] == "I should submit." and out[1]["content"] == ""
    assert out[2]["content"] == [{"type": "thinking", "thinking": "I should submit."}]
    assert out[3] == msgs[3] and out[4] == msgs[4] and out[5] == msgs[5]   # empty / closed / EXIT untouched
    assert msgs[1]["content"] == "I should submit.\n\n"                    # input not mutated
    assert F.turn_counts(out)["unclosed_text"] == 0 and F.turn_counts(out)["unclosed_empty"] == 1


def test_filter_and_cli(tmp_path):
    good = {"problem_id": "a", "messages": [asst("r"), asst("", "t")]}
    mostly_empty = {"problem_id": "b", "messages": [asst(""), asst(""), asst("r")]}
    nr, c = F.normalize_row(mostly_empty, 0.5)
    assert nr is None and c["unclosed_empty"] == 2
    assert F.normalize_row(mostly_empty)[0] is not None and F.normalize_row(good, 0.5)[0] is not None
    src, dst = tmp_path / "s.jsonl", tmp_path / "d.jsonl"
    src.write_text("\n".join(map(json.dumps, [good, mostly_empty])))
    F.main(["normalize", str(src), "--out", str(dst), "--max-empty-frac", "0.5"])
    rows = [json.loads(l) for l in dst.read_text().splitlines()]
    assert [r["problem_id"] for r in rows] == ["a"] and F.turn_counts(rows[0]["messages"])["unclosed_text"] == 0
    s = F.summarize([good, mostly_empty])
    assert s["tool_turns"] == 5 and s["unclosed_rate"] == 3 / 5


def test_analyze_metric():
    t = {"problem_id": "a", "turns": 2, "final": {"solved": True}, "ended_by": "exit", "final_code": "x",
         "messages": [asst("r"), asst("", "txt"), asst("", "")]}
    r = A.analyze_run([t])
    assert abs(r["unclosed_think_rate"] - 2 / 3) < 1e-9 and r["episodes_with_unclosed_think"] == 1
    assert abs(r["unclosed_empty_rate"] - 1 / 3) < 1e-9
    assert A.analyze_run([{**t, "messages": []}])["unclosed_think_rate"] is None
