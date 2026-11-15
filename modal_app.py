"""Modal jobs: data builds and the C++ sandbox run where the network and CPUs are fast.

uv run modal run modal_app.py::build_p1 --target 1500
uv run modal volume get cpp-data pools/p1.jsonl data/pools/p1.jsonl
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


@app.function(image=image, volumes={"/root/data": vol}, cpu=32, memory=65536, timeout=4 * 3600)
def build_p1(hard: int = 900, medium: int = 700, easy: int = 150, max_tokens: int = 11000, tries: int = 3, oversample: int = 3,
             holdout_jsonl: str = "", out: str = "p1.jsonl"):
    import argparse, os
    from scripts import build_p1 as b
    os.makedirs("data/pools", exist_ok=True)
    with open("data/holdout.jsonl", "w") as f:
        f.write(holdout_jsonl)
    b.main(argparse.Namespace(hard=hard, medium=medium, easy=easy, max_tokens=max_tokens, min_tests=3, tries=tries,
                              oversample=oversample, retry_frac=0.15, workers=30,
                              out=f"data/pools/{out}"))
    vol.commit()


@app.local_entrypoint()
def p1(hard: int = 900, medium: int = 700, easy: int = 150, tries: int = 3, oversample: int = 3, out: str = "p1.jsonl"):
    build_p1.remote(out=out, hard=hard, medium=medium, easy=easy, tries=tries, oversample=oversample,
                    holdout_jsonl=open("data/holdout.jsonl").read())


@app.function(image=image, cpu=2, memory=4096, timeout=900, max_containers=100)
@modal.concurrent(max_inputs=2)
def judge_remote(code: str, tests: list, time_limit: float, mem_limit: float, checker: str | None = None) -> dict:
    """Compile (g++ -std=c++20 -O2) and judge on Modal CPU. Deploy: `uv run modal deploy modal_app.py`.
    Call: modal.Function.from_name("tinker-cpp", "judge_remote").remote(...) or harness.modal_judge()."""
    from cpp import judge
    return judge.judge(code, tests, time_limit, mem_limit, checker)


@app.function(image=image, cpu=2, memory=4096, timeout=1200)
def sandbox_tests() -> str:
    """Run the judge tests on Linux (memory/stack limits only enforced there): uv run modal run modal_app.py::selftest"""
    import subprocess
    p = subprocess.run(["python", "-m", "pytest", "tests/test_judge.py", "-q", "-rs"], capture_output=True, text=True)
    return p.stdout + p.stderr


@app.local_entrypoint()
def selftest():
    print(sandbox_tests.remote())
