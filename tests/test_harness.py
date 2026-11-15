"""Harness loop tests with a scripted fake OpenAI-compatible client."""
import asyncio, json
from types import SimpleNamespace as NS

from cpp.harness import run_episode, HarnessConfig

SUM = '#include <bits/stdc++.h>\nint main(){long long a,b;std::cin>>a>>b;std::cout<<a+b<<"\\n";}'
WRONG = SUM.replace("a+b", "a-b")
PROB = {"id": "t1", "statement": "Add two numbers.", "rating": 800, "time_limit": 10.0,  # generous: macOS first-exec latency can exceed 1 s under load
         "memory_limit": 256.0,
        "sample_tests": [{"input": "1 2\n", "output": "3\n"}],
        "full_tests": [{"input": "1 2\n", "output": "3\n"}, {"input": "7 8\n", "output": "15\n"}],
        "checker": None}


def call(name, args, i=[0]):
    i[0] += 1
    a = args if isinstance(args, str) else json.dumps(args)
    return NS(id=f"c{i[0]}", type="function", function=NS(name=name, arguments=a))


def msg(content="", calls=None, reasoning=None, finish="stop", toks=10, prompt_toks=100):
    m = NS(content=content, tool_calls=calls or None, reasoning_content=reasoning)
    return NS(choices=[NS(message=m, finish_reason=finish)],
              usage=NS(completion_tokens=toks, prompt_tokens=prompt_toks))


class Fake:
    def __init__(self, script):
        self.script, self.seen = list(script), []
        self.chat = NS(completions=NS(create=self.create))

    async def create(self, **kw):
        self.seen.append(kw)
        return self.script.pop(0)


def run(script, **kw):
    c = Fake(script)
    return asyncio.run(run_episode(PROB, c, **kw)), c


def test_happy_path():
    tr, c = run([msg("", [call("run_cpp", {"code": SUM, "stdin": "3 4\n"})], reasoning="think"),
                 msg("", [call("submit", {"code": SUM})]),
                 msg("EXIT")], max_turns=10)
    assert tr.final["solved"] and tr.ended_with_exit and tr.malformed == 0 and tr.turns == 3
    assert tr.output_tokens == 30 and tr.keep_ok() and len(tr.submissions) == 1
    assert tr.messages[2]["reasoning_content"] == "think"
    assert "7" in tr.messages[3]["content"]
    # reasoning is not echoed back to the model by default
    assert all("reasoning_content" not in m for m in c.seen[1]["messages"])


def test_wrong_final_not_solved():
    tr, _ = run([msg("", [call("submit", {"code": WRONG})]), msg("EXIT")])
    assert not tr.final["solved"] and tr.final["compiled"] and not tr.keep_ok()


def test_last_compiling_submission_is_graded():
    tr, _ = run([msg("", [call("submit", {"code": SUM})]),
                 msg("", [call("submit", {"code": "int main(){ nope }"})]),
                 msg("EXIT")])
    assert tr.final["solved"] and tr.submissions[1]["compiled"] is False


def test_regression_flagged():
    tr, _ = run([msg("", [call("submit", {"code": SUM})]), msg("", [call("submit", {"code": WRONG})]), msg("EXIT")])
    assert tr.regressed and not tr.keep_ok()


def test_malformed_ends_episode():
    bad = lambda: msg("", [call("run_cpp", "{not json")])
    tr, _ = run([bad(), bad(), bad(), bad()])
    assert tr.malformed == 3 and tr.ended_by == "malformed" and not tr.ended_with_exit


def test_malformed_streak_resets():
    tr, _ = run([msg("", [call("run_cpp", "{bad")]), msg("", [call("submit", {"code": SUM})]),
                 msg("", [call("nope", {"code": "x"})]), msg("EXIT")])
    assert tr.malformed == 2 and tr.ended_with_exit and not tr.keep_ok()


def test_truncation_counted():
    tr, _ = run([msg("long...", finish="length"), msg("", [call("submit", {"code": SUM})]), msg("EXIT")])
    assert tr.truncated == 1 and tr.malformed == 1 and tr.final["solved"]


def test_submission_cap():
    cfg = HarnessConfig(max_submissions=2)
    tr, _ = run([msg("", [call("submit", {"code": SUM})]) for _ in range(3)] + [msg("EXIT")], config=cfg)
    assert len(tr.submissions) == 2 and "limit" in tr.messages[-2]["content"].lower()


def test_max_turns():
    tr, _ = run([msg("", [call("run_cpp", {"code": SUM, "stdin": "1 1"})]) for _ in range(5)], max_turns=3)
    assert tr.ended_by == "max_turns" and tr.turns == 3


def test_think_tag_in_content_and_no_submission():
    tr, _ = run([msg("<think>hmm</think>EXIT")])
    assert tr.messages[2]["reasoning_content"] == "hmm" and tr.ended_with_exit
    assert tr.final["solved"] is False and tr.final["total"] == 2


def test_tool_call_markup_in_text_is_malformed():
    tr, _ = run([msg("<tool_call>{...}</tool_call>"), msg("EXIT")])
    assert tr.malformed == 1
