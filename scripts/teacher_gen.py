"""Call the deployed teacher app (keeps the GPU server warm between runs).

uv run modal deploy modal_teacher.py
uv run python -m scripts.teacher_gen --n 20 --tag smoke
"""
import argparse, json, os
from dotenv import load_dotenv
load_dotenv(".env")
import modal

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=20)
ap.add_argument("--tag", default="smoke")
ap.add_argument("--attempts", type=int, default=1)
ap.add_argument("--hard", type=float, default=0.6)
ap.add_argument("--medium", type=float, default=0.3)
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()
ex = [json.loads(l)["id"] for l in open("data/holdout.jsonl")]
gen = modal.Function.from_name("tinker-cpp-teacher", "generate")
with modal.enable_output():
    gen.remote(n=a.n, tag=a.tag, exclude_ids=ex, hard=a.hard, medium=a.medium, attempts=a.attempts, seed=a.seed)
