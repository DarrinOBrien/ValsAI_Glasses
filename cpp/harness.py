"""Agent loop copy: one problem, two tools, EXIT to finish. Policy-agnostic."""
import asyncio, json
from dataclasses import dataclass, field
from typing import Awaitable, Callable

from cpp import judge, tools

# policy(messages) -> (assistant_message, clean_stop: bool, n_output_tokens)
Policy = Callable[[list[dict]], Awaitable[tuple[dict, bool, int]]]


@dataclass
class Episode:
    problem_id: str
    messages: list[dict] = field(default_factory=list)
    final_code: str | None = None  # last submission that compiled
    submissions: int = 0
    turns: int = 0
    malformed: int = 0
    truncated: int = 0
    output_tokens: int = 0
    ended_by: str = ""
    last_sample_pass: bool = False
    score: dict | None = None


def text_of(msg: dict) -> str:
    c = msg.get("content", "")
    if isinstance(c, str):
        return c
    return "".join(p["text"] for p in c if p["type"] == "text")


def _args(tc) -> dict:
    fn = tc.function if hasattr(tc, "function") else tc["function"]
    a = fn.arguments if hasattr(fn, "arguments") else fn["arguments"]
    return json.loads(a) if isinstance(a, str) else a


def _name(tc) -> str:
    fn = tc.function if hasattr(tc, "function") else tc["function"]
    return fn.name if hasattr(fn, "name") else fn["name"]


async def run_episode(problem: dict, policy: Policy, system_messages: list[dict],
                      max_turns: int = 30, max_malformed: int = 3) -> Episode:
    ep = Episode(problem_id=problem["id"])
    tl, ml = problem["time_limit"], problem["memory_limit"]
    msgs = [*system_messages, {"role": "user", "content": tools.task_prompt(problem)}]
    bad_streak = 0
    while ep.turns < max_turns:
        msg, clean, ntok = await policy(msgs)
        ep.turns += 1
        ep.output_tokens += ntok
        msgs.append(msg)
        if not clean:
            ep.truncated += 1
        calls = msg.get("tool_calls") or []
        bad = bool(msg.get("unparsed_tool_calls")) or (not clean and not calls)
        results = []
        for tc in calls:
            try:
                name, a = _name(tc), _args(tc)
                if name == tools.RUN_TOOL and isinstance(a.get("code"), str):
                    r = await asyncio.to_thread(judge.run_code, a["code"], str(a.get("stdin", "")), tl, ml)
                    results.append((name, tools.format_run_result(r)))
                elif name == tools.SUBMIT_TOOL and isinstance(a.get("code"), str):
                    if ep.submissions >= tools.MAX_SUBMISSIONS:
                        results.append((name, "Submission limit reached."))
                        continue
                    ep.submissions += 1
                    j = await asyncio.to_thread(judge.judge, a["code"], problem["samples"], tl, ml, False)
                    if j["compiled"]:
                        ep.final_code = a["code"]
                        ep.last_sample_pass = j["passed"] == j["total"]
                    results.append((name, tools.format_submit_result(j, ep.submissions)))
                else:
                    bad = True
                    results.append((name, f"Error: unknown tool or bad arguments for '{name}'."))
            except Exception as e:  # bad JSON args etc.
                bad = True
                results.append(("error", f"Error: malformed tool call ({e})."))
        if bad:
            ep.malformed += 1
            bad_streak += 1
            if bad_streak >= max_malformed:
                ep.ended_by = "malformed"
                break
            if not results:
                msgs.append({"role": "user", "content": "Your last message was not a valid tool call. "
                             "Call a tool in the required format, or reply EXIT."})
                continue
        else:
            bad_streak = 0
        if results:
            for name, content in results:
                msgs.append({"role": "tool", "name": name, "content": content})
            continue
        if "EXIT" in text_of(msg):
            ep.ended_by = "exit"
            break
        msgs.append({"role": "user", "content": "Continue. Call a tool, or reply EXIT when finished."})
    else:
        ep.ended_by = "max_turns"
    ep.messages = msgs
    return ep


def grade(ep: Episode, problem: dict) -> dict:
    """Full hidden-test verdict for the last compiling submission."""
    if not ep.final_code:
        ep.score = {"solved": False, "passed": 0, "total": len(problem["tests"])}
    else:
        j = judge.judge(ep.final_code, problem["tests"], problem["time_limit"], problem["memory_limit"])
        ep.score = {"solved": j["passed"] == j["total"], "passed": j["passed"], "total": j["total"],
                    "max_time": j["max_time"]}
    return ep.score
