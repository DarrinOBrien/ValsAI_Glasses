"""Our copy of the task format: tool schema, prompts and tool-result text.

EVERYTHING the scoring harness controls lives in this file. It is our best guess until staff
give us the real schema; when they do, change it here and regenerate the training data.
"""
import json, os

CODE_ARG = "code"
RUN_TOOL = "run_cpp"
SUBMIT_TOOL = "submit"
MAX_SUBMISSIONS = 50
MAX_TOOL_OUTPUT_CHARS = 4000

TOOLS = [
    {"type": "function", "function": {
        "name": RUN_TOOL,
        "description": "Compile a C++20 program (g++ -std=c++20 -O2) and run it on the given stdin. "
                       "Returns compiler errors, or stdout, stderr, exit status and run time.",
        "parameters": {"type": "object", "properties": {
            "code": {"type": "string", "description": "Complete C++ source code."},
            "stdin": {"type": "string", "description": "Input passed to the program on standard input."}},
            "required": ["code", "stdin"]}}},
    {"type": "function", "function": {
        "name": SUBMIT_TOOL,
        "description": "Submit a C++20 solution. It is run on the sample tests and the result of each is "
                       "reported. The last submission that compiles is graded on hidden tests.",
        "parameters": {"type": "object", "properties": {
            "code": {"type": "string", "description": "Complete C++ source code."}},
            "required": ["code"]}}},
]

SYSTEM_PROMPT = (
    "You are an expert competitive programmer. Solve the problem in C++20. Your program reads from "
    "standard input and writes to standard output.\n"
    f"Use {RUN_TOOL} to compile and test code on any input, and {SUBMIT_TOOL} to submit a solution "
    f"(at most {MAX_SUBMISSIONS} submissions). The last submission that compiles is graded on hidden "
    "tests, including large ones, under the time and memory limits.\n"
    "When you are finished, reply with exactly EXIT."
)


def task_prompt(p: dict) -> str:
    parts = [f"# {p.get('title') or 'Problem'}", "",
             f"Time limit: {p['time_limit']:g} s. Memory limit: {p['memory_limit']:g} MB.", "",
             p["statement"].strip(), "", "## Sample tests", ""]
    for i, s in enumerate(p["sample_tests"], 1):
        parts += [f"samples/{i}.in:", "```", s["input"].strip(), "```",
                  f"samples/{i}.out:", "```", s["output"].strip(), "```", ""]
    return "\n".join(parts).strip()


def _clip(s: str, n: int = MAX_TOOL_OUTPUT_CHARS) -> str:
    return s if len(s) <= n else s[: n // 2] + f"\n... [{len(s) - n} chars truncated] ...\n" + s[-n // 2:]


def format_run_result(r: dict) -> str:
    if not r["compiled"]:
        return "Compilation error:\n" + _clip(r["compile_error"])
    head = {"OK": "Exit code 0", "TLE": "Time limit exceeded",
            "RE": f"Runtime error (exit code {r.get('code')})"}[r["status"]]
    out = f"{head}. Time: {r['time']:.2f} s\nstdout:\n{_clip(r['stdout'])}"
    if r.get("stderr"):
        out += f"\nstderr:\n{_clip(r['stderr'], 1000)}"
    return out


def format_submit_result(j: dict, n_used: int) -> str:
    if not j["compiled"]:
        return "Compilation error:\n" + _clip(j["compile_error"])
    names = {"AC": "passed", "WA": "wrong answer", "TLE": "time limit exceeded", "RE": "runtime error", "JE": "judge error"}
    lines = [f"Sample {i}: {names[v]}" for i, v in enumerate(j["verdicts"], 1)]
    lines.append(f"{j['passed']}/{j['total']} sample tests passed. Submissions used: {n_used}/{MAX_SUBMISSIONS}.")
    return "\n".join(lines)


_orig_task_prompt = task_prompt


def tool_specs() -> list[dict]:
    return [t["function"] for t in TOOLS]


# ---------------------------------------------------------------------------------------------------------
# Real Vals scoring harness (vals-ai/hackathon-post-training-agent). Opt in with HARNESS_FORMAT=vals; every
# consumer (harness, policy, train_sft fallback, smoke_real) reads names/schema/prompt from this module.
# ---------------------------------------------------------------------------------------------------------
VALS_INSTRUCTIONS = """
You will solve a competitive programming problem.

Write your solution as a complete C++20 program that reads standard input and writes standard output. Enclose code in three backticks, for example:

```
[your code here]
```

You have two tools. cpp_executor compiles a program and runs it on stdin you choose, for experiments, brute forces and stress tests. submission saves a solution as your answer and runs it on the sample tests from the statement. Your last submission that compiles is graded on hidden tests, under the time limit, with partial credit for each test it passes.

You may make at most 50 submissions in at most 100 turns. When you are done and do not want to submit again, respond with 'EXIT'.

The problem statement follows:
{question}
"""

VALS_TOOLS = [
    {"type": "function", "function": {
        "name": "cpp_executor",
        "description": "Compile a C++20 program (g++ -O2) and run it once with the given stdin. "
                       "Returns stdout, stderr and the exit code. Runs are killed after 10 seconds. "
                       "Use it for experiments, brute-force checkers and generators.",
        "parameters": {"type": "object", "properties": {
            "cpp_code": {"type": "string", "description": "The C++20 program, optionally inside a code block."},
            "stdin": {"type": "string", "description": "Text passed to the program on standard input."}},
            "required": ["cpp_code"]}}},
    {"type": "function", "function": {
        "name": "submission",
        "description": "Submit a complete C++20 solution. It is compiled, run on the sample tests from the statement, and "
                       "saved as your final answer; your last submission that compiles is the one graded on the hidden "
                       "tests. Only sample results are reported, never hidden test results.",
        "parameters": {"type": "object", "properties": {
            "cpp_code": {"type": "string", "description": "The full C++20 solution, optionally inside a code block."}},
            "required": ["cpp_code"]}}},
]


def vals_task_prompt(p: dict) -> str:
    return VALS_INSTRUCTIONS.format(question=_orig_task_prompt(p))


def vals_format_run_result(r: dict) -> str:
    if not r["compiled"]:
        return "Compilation failed:\n" + _clip(r["compile_error"])
    if r["status"] == "TLE":
        return "Execution timed out after 10 seconds"
    code = 0 if r["status"] == "OK" else (r.get("code") or 1)
    return json.dumps({"exit_code": code, "stdout": _clip(r["stdout"]), "stderr": _clip(r.get("stderr") or "")})


_VALS_VERDICT = {"AC": "passed", "WA": "wrong answer, got:\n[output not shown]", "TLE": "time limit exceeded",
                 "RE": "runtime error (exit code 1): ", "JE": "runtime error (exit code 1): "}


def vals_format_submit_result(j: dict, n_used: int) -> str:
    remaining = MAX_SUBMISSIONS - n_used
    if not j["compiled"]:
        return (f"Compilation failed:\n{_clip(j['compile_error'])}\nNot saved; your previous submission stands. "
                f"Submissions remaining: {remaining}")
    samples = {str(i): _VALS_VERDICT[v] for i, v in enumerate(j["verdicts"], 1)}
    return json.dumps({"saved": True, "samples": samples, "submissions_remaining": remaining})


if os.environ.get("HARNESS_FORMAT") == "vals":
    _orig_task_prompt = task_prompt
    CODE_ARG = "cpp_code"
    RUN_TOOL, SUBMIT_TOOL = "cpp_executor", "submission"
    TOOLS = VALS_TOOLS
    SYSTEM_PROMPT = ""
    task_prompt = vals_task_prompt
    format_run_result = vals_format_run_result
    format_submit_result = vals_format_submit_result


if __name__ == "__main__":
    print(json.dumps(TOOLS, indent=1))
