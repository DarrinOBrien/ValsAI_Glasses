"""Problem records built from open-r1/codeforces (tests, ratings) for eval and verification."""
import glob, json, os
from huggingface_hub import snapshot_download
import pyarrow.parquet as pq

REPO = "open-r1/codeforces"
LOCAL = "data/hf/open-r1__codeforces"


def _norm(s: str) -> str:
    return s.replace("\r\n", "\n")


def to_problem(r: dict) -> dict | None:
    """None if the problem can't be judged by plain stdin/stdout diff."""
    if r.get("input_mode") != "stdio" or r.get("interaction_format") or r.get("generated_checker"):
        return None
    if not r.get("executable") or not r.get("official_tests") or not r.get("examples") or not r.get("description"):
        return None
    st = [r["description"]]
    for head, key in (("Input", "input_format"), ("Output", "output_format"), ("Note", "note")):
        if r.get(key):
            st += [f"## {head}", r[key]]
    return {
        "id": r["id"], "title": r["title"], "statement": "\n\n".join(st),
        "time_limit": float(r["time_limit"] or 2.0), "memory_limit": float(r["memory_limit"] or 256.0),
        "samples": [{"input": _norm(e["input"]), "output": _norm(e["output"])} for e in r["examples"]],
        "tests": [{"input": _norm(t["input"]), "output": _norm(t["output"])} for t in r["official_tests"]],
        "tests_complete": bool(r.get("official_tests_complete")),
        "rating": r.get("rating"), "tags": r.get("tags") or [], "year": r.get("contest_start_year"),
    }


def load_split(split: str) -> list[dict]:
    """split: 'train' or 'test'. Downloads the parquet shards on first use."""
    snapshot_download(REPO, repo_type="dataset", local_dir=LOCAL, allow_patterns=[f"data/{split}-*.parquet"])
    out = []
    for f in sorted(glob.glob(f"{LOCAL}/data/{split}-*.parquet")):
        for batch in pq.ParquetFile(f).iter_batches(batch_size=64):
            for r in batch.to_pylist():
                p = to_problem(r)
                if p:
                    out.append(p)
    return out


def holdout(n: int = 40, max_rating: int = 1700, path: str = "data/holdout.jsonl") -> list[dict]:
    """Format-check holdout: easy/medium problems from the 2025 test split, never trained on."""
    if os.path.exists(path):
        return [json.loads(l) for l in open(path)]
    ps = [p for p in load_split("test") if p["rating"] and p["rating"] <= max_rating
          and sum(len(t["input"]) for t in p["tests"]) < 3_000_000]
    ps.sort(key=lambda p: p["id"])
    ps = ps[:: max(1, len(ps) // n)][:n]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for p in ps:
            f.write(json.dumps(p) + "\n")
    return ps
