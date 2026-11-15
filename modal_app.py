"""Modal jobs: the C++ sandbox/judge (judge_remote) and its self-tests.

Deploy: uv run modal deploy modal_app.py
"""
import modal

app = modal.App("tinker-cpp")
vol = modal.Volume.from_name("cpp-data", create_if_missing=True)
image = (
    modal.Image.from_registry("gcc:14", add_python="3.12")
    .pip_install("pyarrow", "huggingface_hub", "hf_transfer", "pytest")
    .run_commands("mkdir -p /opt/testlib && curl -sL https://raw.githubusercontent.com/MikeMirzayanov/testlib/master/testlib.h -o /opt/testlib/testlib.h")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "CXX": "g++", "TESTLIB_DIR": "/opt/testlib"})
    .workdir("/root")
    .add_local_python_source("cpp", "scripts", "tests")
)


@app.function(image=image, volumes={"/vol": vol}, cpu=2, memory=4096, timeout=900, max_containers=100)
@modal.concurrent(max_inputs=2)
def judge_remote(code: str, tests, time_limit: float, mem_limit: float, checker: str | None = None,
                 checker_lang: str | None = None) -> dict:
    """Compile (g++ -std=c++20 -O2) and judge on Modal CPU. Deploy: `uv run modal deploy modal_app.py`.
    Call: modal.Function.from_name("tinker-cpp", "judge_remote").remote(...) or harness.modal_judge()."""
    from cpp import judge
    try:
        return judge.judge(code, tests, time_limit, mem_limit, checker, True, checker_lang)
    except FileNotFoundError:  # tests written to the volume after this container started
        vol.reload()
        return judge.judge(code, tests, time_limit, mem_limit, checker, True, checker_lang)


@app.function(image=image, cpu=2, memory=4096, timeout=1200)
def sandbox_tests() -> str:
    """Run the judge tests on Linux (memory/stack limits only enforced there): uv run modal run modal_app.py::selftest"""
    import subprocess
    p = subprocess.run(["python", "-m", "pytest", "tests/test_judge.py", "-q", "-rs"], capture_output=True, text=True)
    return p.stdout + p.stderr


@app.function(image=image, volumes={"/vol": vol}, cpu=4, memory=8192, timeout=1800)
def checker_selftest(problems: list, max_tests: int = 6) -> list:
    """Real-checker validation (tests/real_checkers.py) with tests read from the cpp-data volume."""
    from tests.real_checkers import check_problem
    return [check_problem(p, max_tests) for p in problems]


@app.local_entrypoint()
def checkers(pool: str, n: int = 8, max_tests: int = 6):
    """uv run modal run modal_app.py::checkers --pool data/pool/eval150.jsonl --n 8"""
    import json
    ps = [json.loads(l) for l in open(pool)]
    ps = [{k: v for k, v in p.items() if k != "statement"} for p in ps if p.get("checker")][:n]
    for r in checker_selftest.remote(ps, max_tests):
        print(r)


@app.local_entrypoint()
def selftest():
    print(sandbox_tests.remote())
