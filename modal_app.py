"""Modal jobs: data builds and the C++ sandbox run where the network and CPUs are fast.

uv run modal run modal_app.py::build_p1 --target 1500
uv run modal volume get cpp-data pools/p1.jsonl data/pools/p1.jsonl
"""
import modal

app = modal.App("tinker-cpp")
vol = modal.Volume.from_name("cpp-data", create_if_missing=True)
image = (
    modal.Image.from_registry("gcc:14", add_python="3.12")
    .pip_install("pyarrow", "huggingface_hub", "hf_transfer")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "CXX": "g++"})
    .workdir("/root")
    .add_local_python_source("cpp", "scripts")
)


@app.function(image=image, volumes={"/root/data": vol}, cpu=32, memory=65536, timeout=4 * 3600)
def build_p1(target: int = 1500, max_tokens: int = 8000, tries: int = 2, oversample: int = 3,
             holdout_jsonl: str = ""):
    import argparse, os
    from scripts import build_p1 as b
    os.makedirs("data/pools", exist_ok=True)
    with open("data/holdout.jsonl", "w") as f:
        f.write(holdout_jsonl)
    b.main(argparse.Namespace(target=target, max_tokens=max_tokens, min_tests=3, tries=tries,
                              oversample=oversample, retry_frac=0.15, workers=30,
                              out="data/pools/p1.jsonl"))
    vol.commit()


@app.local_entrypoint()
def p1(target: int = 1500, tries: int = 2, oversample: int = 3):
    build_p1.remote(target=target, tries=tries, oversample=oversample,
                    holdout_jsonl=open("data/holdout.jsonl").read())
