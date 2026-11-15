import asyncio
from scripts.eval import evaluate, summarize
from tests.test_harness import Fake, msg, call, PROB, SUM


def test_evaluate_and_summarize():
    script = [msg("", [call("submit", {"code": SUM})], toks=50), msg("EXIT", toks=5),
              msg("long", finish="length", toks=100)] + [msg("EXIT")] * 5
    c = Fake(script)
    trajs = asyncio.run(evaluate([PROB], c, "m", k=2, concurrency=1))
    s = summarize(trajs, [PROB])
    assert s["n"] == 2 and s["pass_rate"] == 0.5 and s["avg_turns"] > 0
    assert s["truncation_rate_per_turn"] > 0 and s["pass_rate[<1600]"]["n"] == 2
