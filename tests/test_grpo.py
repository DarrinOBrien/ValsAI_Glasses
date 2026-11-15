import asyncio, json
from types import SimpleNamespace as NS

from cpp.harness import run_episode
from scripts import grpo as G
from tests.test_harness import PROB, SUM, WRONG


class FakeRenderer:
    """Scripted 'model': each sample is (text-ish message dict, tokens)."""
    def __init__(self, script):
        self.script = list(script)
        self.cur = None
        self.prompts = []

    def build_generation_prompt(self, messages):
        self.prompts.append(messages)
        n = 5 + 3 * len(messages)
        return NS(to_ints=lambda: list(range(100, 100 + n)), length=n)

    def get_stop_sequences(self):
        return [0]

    def parse_response(self, toks):
        return self.cur, NS(is_clean=True)


class FakeSampler:
    def __init__(self, r):
        self.r = r

    async def sample_async(self, prompt, num_samples, sampling_params):
        msg, toks = self.r.script.pop(0)
        self.r.cur = msg
        seq = NS(tokens=toks, logprobs=[-0.5] * len(toks), stop_reason="stop")
        return NS(sequences=[seq])


def tc(code):
    fn = NS(name="submit", arguments=json.dumps({"code": code}))
    return NS(function=fn)


def episode(code, prefix_msgs=None):
    r = FakeRenderer([
        ({"role": "assistant", "content": [{"type": "thinking", "thinking": "hm"}, {"type": "text", "text": ""}],
          "tool_calls": [tc(code)]}, [1, 2, 3]),
        ({"role": "assistant", "content": "EXIT"}, [4, 5]),
    ])
    c = G.TinkerChatClient(FakeSampler(r), r, prefix=[{"role": "system", "content": "sys"}])
    tr = asyncio.run(run_episode(PROB, c, max_turns=5, max_tokens=100))
    return tr, c, r


def test_client_records_turns_and_harness_sees_tool_call():
    tr, c, r = episode(SUM)
    assert tr.solved and tr.turns == 2 and tr.malformed == 0 and tr.ended_with_exit
    assert [len(t.tokens) for t in c.turns] == [3, 2]
    assert r.prompts[0][0]["role"] == "system" and r.prompts[0][1]["role"] == "user"
    # second prompt carries the assistant tool call (hydrated) and the tool result, no reasoning
    roles = [m["role"] for m in r.prompts[1]]
    assert roles == ["system", "user", "assistant", "tool"]
    assert "reasoning_content" not in r.prompts[1][2]
    assert tr.messages[2]["reasoning_content"] == "hm"


def test_datum_alignment():
    t = G.Turn(prompt=[10, 11, 12, 13], tokens=[20, 21], logprobs=[-1.0, -2.0])
    d = G.turn_to_datum(t, 0.7)
    assert d.model_input.length == 5
    tgt = d.loss_fn_inputs["target_tokens"].to_torch().tolist()
    assert tgt == [11, 12, 13, 20, 21]
    assert d.loss_fn_inputs["logprobs"].to_torch().tolist() == [0, 0, 0, -1.0, -2.0]
    adv = d.loss_fn_inputs["advantages"].to_torch().tolist()
    assert adv[:3] == [0, 0, 0] and all(abs(x - 0.7) < 1e-6 for x in adv[3:])


def test_reward_group_and_build_data():
    good, gc, _ = episode(SUM)
    bad, bc, _ = episode(WRONG)
    assert not bad.solved
    assert G.reward(good) > 0.9 > G.reward(bad) + 0.5 and G.reward(good) == 1 - 0.005 * 2
    adv = G.group_advantages([1.0, 0.0])
    assert adv == [0.5, -0.5]
    data, st = G.build_data([("p", [(good, gc.turns), (bad, bc.turns)]),
                             ("flat", [(good, gc.turns), (good, gc.turns)])], 0.005, 100)
    assert st["groups"] == 2 and st["skipped_flat"] == 1 and len(data) == 4  # 2 turns x 2 episodes
    sign = [d.loss_fn_inputs["advantages"].to_torch()[-1].item() for d in data]
    assert sign[0] > 0 and sign[2] < 0
    _, st2 = G.build_data([("p", [(good, gc.turns), (bad, bc.turns)])], 0.005, 10)
    assert st2["dropped_long"] > 0


def test_medium_filter(tmp_path):
    def t(pid, ok): return {"problem_id": pid, "final": {"solved": ok}}
    trajs = [t("easy", 1)] * 4 + [t("never", 0)] * 4 + [t("mid", 1)] + [t("mid", 0)] * 3 + [t("mid2", 1)] * 3 + [t("mid2", 0)]
    assert G.medium_ids(trajs) == {"mid", "mid2"}
    tf, pf, of = tmp_path / "t.jsonl", tmp_path / "p.jsonl", tmp_path / "o.jsonl"
    tf.write_text("\n".join(map(json.dumps, trajs)))
    pf.write_text("\n".join(json.dumps({"id": i}) for i in ["easy", "never", "mid", "mid2", "other"]))
    G.main(["select", "--trajs", str(tf), "--problems", str(pf), "--out", str(of)])
    assert [json.loads(l)["id"] for l in of.read_text().splitlines()] == ["mid", "mid2"]
