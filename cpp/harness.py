"""Harness copy: the agent loop for one problem. Works with any OpenAI-compatible client.

INTERFACE (stable; other tracks depend on it)
---------------------------------------------
Problem (dict):
    {id, statement, rating, time_limit (s), memory_limit (MB),
     sample_tests: [{input, output}], full_tests: [{input, output}],
     checker: None (token-wise exact match) | str source, with checker_lang "python" (open-r1 score
              checker: `python checker.py input correct contestant`, accepted iff exit 0 and score > 0)
              or "cpp" (testlib: `checker in contestant ans`, exit 0 = AC); checker_lang absent ->
              guessed from the source
     full_tests may instead be {path, n, ...} (or a path): JSON {"tests":[...]} on volume cpp-data
              (/vol/pool/tests/<id>.json); resolved inside the judge (set VOL_ROOT to remap /vol/ locally)}
    optional: title.

    traj = await run_episode(problem, client, max_turns=30, max_tokens=16000, temperature=0.8,
                             model="Qwen/Qwen3.5-9B", config=DEFAULT_CONFIG, judge_fn=local_judge)

    client: anything exposing `await client.chat.completions.create(model=, messages=, tools=,
            max_tokens=, temperature=, [top_p=], [extra_body=])` (openai.AsyncOpenAI pointed at vLLM).
            Thinking is read from `message.reasoning_content` / `message.reasoning` if present, else
            from a `<think>...</think>` prefix in content.
    judge_fn(code, tests, time_limit, mem_limit, checker, checker_lang) -> verdict dict (async or sync).
            `local_judge` runs cpp.judge locally; `modal_judge()` calls Modal CPU containers
            (modal_app.py::judge_remote). Verdict: {compiled, compile_error, passed, total,
            verdicts[AC|WA|TLE|RE|JE(judge error)], max_time}.

Trajectory (dataclass, .to_dict() is JSON-safe):
    problem_id, tools, messages : full chat (system, user, assistant{content, reasoning_content,
                             tool_calls}, tool{tool_call_id, name, content}, nudge user msgs)
    submissions            : [{n, compiled, sample_passed, sample_total, verdicts}] per submit call
    final                  : full-test verdict of the LAST COMPILING submission
                             {solved, compiled, passed, total, verdicts, max_time} (solved False if none)
    final_code, turns, output_tokens (sum of completion tokens), malformed, truncated
    (finish_reason=length count), ended_with_exit, ended_by in {exit, malformed, max_turns, context},
    regressed (a submission passing all samples was later replaced by one that did not)
    .keep_ok(): expert-iteration filter = solved & ended_with_exit & malformed==0 & not regressed

Everything the real scoring harness may dictate (system prompt, tool JSON schema, tool names,
turn limit, submission cap, EXIT token, tool-result text, chat-template kwargs) lives in
HarnessConfig; swap it wholesale once the real values are known.
"""
from __future__ import annotations

import asyncio, inspect, json, re
from dataclasses import dataclass, field, asdict
from typing import Any, Callable

from cpp import judge as _judge, tools as T


@dataclass
class HarnessConfig:
    system_prompt: str = T.SYSTEM_PROMPT
    tools: list[dict] = field(default_factory=lambda: T.TOOLS)  # OpenAI tool JSON schema
    run_tool: str = T.RUN_TOOL
    submit_tool: str = T.SUBMIT_TOOL
    exit_token: str = "EXIT"
    max_turns: int = 100           # used when run_episode(max_turns=None)
    max_submissions: int = T.MAX_SUBMISSIONS
    max_malformed: int = 3         # consecutive malformed assistant messages that end the episode
    max_context_tokens: int = 100000  # stop if prompt_tokens exceeds this
    context_window: int = 65536      # server max context (MAX_LEN of modal_sampler); keep in sync. 0 = unlimited
    context_margin: int = 2000       # tokens assumed added to the prompt by the next turn (visible reply + tool result)
    min_gen_tokens: int = 2048       # end the episode ('context') if less than this can still be generated
    no_action_is_malformed: bool = True  # assistant message with no tool call and no EXIT
    send_reasoning_back: bool = False    # include reasoning_content in history sent to the model
    task_prompt: Callable[[dict], str] = T.task_prompt
    format_run: Callable[[dict], str] = T.format_run_result
    format_submit: Callable[[dict, int], str] = T.format_submit_result
    extra_body: dict = field(default_factory=dict)  # e.g. {"chat_template_kwargs": {"enable_thinking": True}}


DEFAULT_CONFIG = HarnessConfig()


@dataclass
class Trajectory:
    problem_id: str
    messages: list[dict] = field(default_factory=list)
    tools: list[dict] = field(default_factory=list)
    submissions: list[dict] = field(default_factory=list)
    final: dict = field(default_factory=lambda: {"solved": False, "compiled": False, "passed": 0, "total": 0})
    final_code: str | None = None
    turns: int = 0
    output_tokens: int = 0
    malformed: int = 0
    truncated: int = 0
    ended_with_exit: bool = False
    ended_by: str = ""
    regressed: bool = False

    @property
    def solved(self) -> bool:
        return bool(self.final.get("solved"))

    def keep_ok(self) -> bool:
        return self.solved and self.ended_with_exit and self.malformed == 0 and not self.regressed

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------- judges
async def local_judge(code, tests, time_limit, mem_limit, checker=None, checker_lang=None) -> dict:
    return await asyncio.to_thread(_judge.judge, code, tests, time_limit, mem_limit, checker, True, checker_lang)


def modal_judge(app_name: str = "tinker-cpp", fn_name: str = "judge_remote"):
    """judge_fn backed by the deployed Modal function (`uv run modal deploy modal_app.py`)."""
    import modal
    fn = modal.Function.from_name(app_name, fn_name)

    async def j(code, tests, time_limit, mem_limit, checker=None, checker_lang=None):
        # tests may be an inline list or {path,...}; a dict is resolved inside the container from the volume
        return await fn.remote.aio(code, tests, time_limit, mem_limit, checker, checker_lang)
    return j


async def _maybe_await(x):
    return await x if inspect.isawaitable(x) else x


# ---------------------------------------------------------------- parsing helpers
_THINK = re.compile(r"^\s*<think>(.*?)</think>\s*", re.S)


def _split_message(m: Any) -> tuple[str, str, list[dict]]:
    """-> (content, reasoning, tool_calls as plain dicts) from an SDK message or dict."""
    g = (lambda k: m.get(k)) if isinstance(m, dict) else (lambda k: getattr(m, k, None))
    content = g("content") or ""
    reasoning = g("reasoning_content") or g("reasoning") or ""
    if not reasoning:
        mt = _THINK.match(content)
        if mt:
            reasoning, content = mt.group(1).strip(), content[mt.end():]
    elif "</think>" in content:
        content = content.split("</think>", 1)[1].lstrip()
    calls = []
    for tc in g("tool_calls") or []:
        fn = tc["function"] if isinstance(tc, dict) else tc.function
        name = fn["name"] if isinstance(fn, dict) else fn.name
        args = fn["arguments"] if isinstance(fn, dict) else fn.arguments
        tid = tc["id"] if isinstance(tc, dict) else tc.id
        calls.append({"id": tid, "type": "function", "function": {"name": name, "arguments": args}})
    return content, reasoning, calls


def _history(msgs: list[dict], cfg: HarnessConfig) -> list[dict]:
    out = []
    for m in msgs:
        if m["role"] == "assistant" and not cfg.send_reasoning_back:
            m = {k: v for k, v in m.items() if k != "reasoning_content"}
        out.append(m)
    return out


# ---------------------------------------------------------------- the loop
async def run_episode(problem: dict, client, max_turns: int | None = None, max_tokens: int = 16000,
                      temperature: float = 0.8, model: str = "Qwen/Qwen3.5-9B",
                      config: HarnessConfig = DEFAULT_CONFIG, judge_fn=local_judge,
                      top_p: float | None = None) -> Trajectory:
    cfg = config
    max_turns = cfg.max_turns if max_turns is None else max_turns
    tl, ml, checker = problem["time_limit"], problem["memory_limit"], problem.get("checker")
    clang = problem.get("checker_lang")
    tr = Trajectory(problem_id=problem["id"], tools=cfg.tools)
    msgs: list[dict] = [{"role": "system", "content": cfg.system_prompt},
                        {"role": "user", "content": cfg.task_prompt(problem)}]
    tr.messages = msgs
    bad_streak = 0
    best_sample_pass = False
    tr.ended_by = "max_turns"

    last_prompt = 0
    while tr.turns < max_turns:
        gen = max_tokens
        if cfg.context_window:
            room = cfg.context_window - last_prompt - (cfg.context_margin if last_prompt else 0) - 64
            if room < cfg.min_gen_tokens:
                tr.ended_by = "context"
                break
            gen = min(max_tokens, room)
        kw = dict(model=model, messages=_history(msgs, cfg), tools=cfg.tools, max_tokens=gen,
                  temperature=temperature)
        if top_p is not None:
            kw["top_p"] = top_p
        if cfg.extra_body:
            kw["extra_body"] = cfg.extra_body
        try:
            resp = await client.chat.completions.create(**kw)
        except Exception as e:  # prompt exceeded the server context: end cleanly instead of crashing the episode
            if getattr(e, "status_code", None) == 400 and "context" in str(e).lower():
                tr.ended_by = "context"
                break
            raise
        choice = resp.choices[0]
        tr.turns += 1
        usage = getattr(resp, "usage", None)
        last_prompt = (getattr(usage, "prompt_tokens", 0) or 0) if usage is not None else last_prompt
        tr.output_tokens += getattr(usage, "completion_tokens", 0) or 0
        truncated = choice.finish_reason == "length"
        tr.truncated += truncated
        content, reasoning, calls = _split_message(choice.message)
        am: dict = {"role": "assistant", "content": content}
        if reasoning:
            am["reasoning_content"] = reasoning
        if calls:
            am["tool_calls"] = calls
        msgs.append(am)

        bad = False
        results: list[dict] = []
        for tc in calls:
            name, raw = tc["function"]["name"], tc["function"]["arguments"]
            try:
                a = json.loads(raw) if isinstance(raw, str) else raw
                code = a.get("code")
                if name not in (cfg.run_tool, cfg.submit_tool) or not isinstance(code, str):
                    raise ValueError(f"unknown tool or missing 'code' for '{name}'")
                if name == cfg.run_tool:
                    r = await asyncio.to_thread(_judge.run_code, code, str(a.get("stdin", "")), tl, ml)
                    out = cfg.format_run(r)
                elif len(tr.submissions) >= cfg.max_submissions:
                    out = "Submission limit reached."
                else:
                    j = await _maybe_await(judge_fn(code, problem["sample_tests"], tl, ml, checker, clang))
                    n = len(tr.submissions) + 1
                    sp = bool(j["compiled"]) and j["passed"] == j["total"]
                    tr.submissions.append({"n": n, "compiled": j["compiled"], "sample_passed": j["passed"],
                                           "sample_total": j["total"], "verdicts": j["verdicts"]})
                    if j["compiled"]:
                        if best_sample_pass and not sp:
                            tr.regressed = True
                        best_sample_pass = sp
                        tr.final_code = code
                    out = cfg.format_submit(j, n)
            except Exception as e:
                bad = True
                out = f"Error: malformed tool call ({e})."
            results.append({"role": "tool", "tool_call_id": tc["id"], "name": name, "content": out})
        msgs.extend(results)

        if not calls:
            if "<tool_call>" in content or truncated:
                bad = True
            elif cfg.exit_token in content:
                tr.ended_with_exit, tr.ended_by = True, "exit"
                break
            elif cfg.no_action_is_malformed:
                bad = True
            msgs.append({"role": "user", "content": "Your last message was not a valid tool call. "
                         f"Call a tool in the required format, or reply {cfg.exit_token} when finished."})
        if bad:
            tr.malformed += 1
            bad_streak += 1
            if bad_streak >= cfg.max_malformed:
                tr.ended_by = "malformed"
                break
        else:
            bad_streak = 0
        if usage is not None and (getattr(usage, "prompt_tokens", 0) or 0) > cfg.max_context_tokens:
            tr.ended_by = "context"
            break
    # full-test verdict of the last compiling submission
    if tr.final_code:
        j = await _maybe_await(judge_fn(tr.final_code, problem["full_tests"], tl, ml, checker, clang))
        tr.final = {"solved": j["passed"] == j["total"] and j["total"] > 0, "compiled": True,
                    "passed": j["passed"], "total": j["total"], "verdicts": j["verdicts"],
                    "max_time": j["max_time"]}
    else:
        tr.final = {"solved": False, "compiled": False, "passed": 0, "total": _judge.n_tests(problem["full_tests"])}
    return tr
