"""Problem pool loaders. Pool/eval files are built by scripts/build_pool.py (Track B) and live on the Modal
volume `cpp-data` under /pool; `modal run scripts/build_pool.py::pull` copies the jsonl files to data/pool/."""
import json, os

POOL_DIR = "data/pool"


def load_pool(name: str = "pool.jsonl", pool_dir: str = POOL_DIR) -> list[dict]:
    """name: 'pool.jsonl' (training pool) or 'eval150.jsonl' (test-split holdout; never train on it).
    Record: {id, title, statement, rating, tags, time_limit (s), memory_limit (MB), sample_tests:[{input,output}],
    full_tests:{path,n,n_official,n_generated}, checker_lang ('python' or None), checker (python source or None)}."""
    out = [json.loads(l) for l in open(os.path.join(pool_dir, name))]
    for p in out:
        p.setdefault("checker_lang", "python" if p.get("checker") else None)
    return out


def full_tests(p: dict, vol_root: str | None = None) -> list[dict]:
    """[{'input','output'}] for a pool/eval record. Test JSON is at p['full_tests']['path'] (/vol/pool/tests/<id>.json
    when cpp-data is mounted at /vol); pass vol_root if the volume is mounted elsewhere."""
    path = p["full_tests"]["path"]
    if vol_root:
        path = path.replace("/vol/", vol_root.rstrip("/") + "/", 1)
    return json.load(open(path))["tests"]
