"""Our copy of the task format: tool schema, prompts and tool-result text.

EVERYTHING the scoring harness controls lives in this file. It is our best guess until staff
give us the real schema; when they do, change it here and regenerate the training data.
"""
import json

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
    for i, s in enumerate(p["samples"], 1):
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
    names = {"AC": "passed", "WA": "wrong answer", "TLE": "time limit exceeded", "RE": "runtime error"}
    lines = [f"Sample {i}: {names[v]}" for i, v in enumerate(j["verdicts"], 1)]
    lines.append(f"{j['passed']}/{j['total']} sample tests passed. Submissions used: {n_used}/{MAX_SUBMISSIONS}.")
    return "\n".join(lines)


def tool_specs() -> list[dict]:
    return [t["function"] for t in TOOLS]


if __name__ == "__main__":
    print(json.dumps(TOOLS, indent=1))
