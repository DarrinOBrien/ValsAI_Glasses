"""Track B: build the problem pool, eval holdout and R1 hard set on Modal volumes.

    uv run --env-file .env modal run scripts/build_pool.py::download    # raw HF data -> cpp-data:/cf_raw
    uv run --env-file .env modal run scripts/build_pool.py::weights     # Qwen3.5-9B -> qwen-weights:/qwen3.5-9b
    uv run --env-file .env modal run scripts/build_pool.py::pool        # pool.jsonl, eval150.jsonl, tests
    uv run --env-file .env modal run scripts/build_pool.py::pull        # copy jsonl files to ./data/

Volume `cpp-data` (mounted at /vol) layout:
    cf_raw/             raw parquet from open-r1/codeforces (verifiable) and codeforces-cots
    cf_tests/           generated_tests/test_cases_<contest>.parquet, only contests we kept
    pool/pool.jsonl, pool/eval150.jsonl, pool/stats.json
    pool/tests/<id>.json   full tests {"tests":[{"input","output"}...]} for each pool/eval problem
Volume `qwen-weights` (mounted at /vol): qwen3.5-9b/
"""
import hashlib, json, os, random, subprocess, sys, tempfile
from concurrent.futures import ThreadPoolExecutor

import modal

app = modal.App("cf-pool")
data_vol = modal.Volume.from_name("cpp-data", create_if_missing=True)
weights_vol = modal.Volume.from_name("qwen-weights", create_if_missing=True)
secret = modal.Secret.from_dict({"HF_TOKEN": os.environ.get("HF_ACCESS_TOKEN", os.environ.get("HF_TOKEN", ""))})
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("pyarrow", "huggingface_hub", "hf_transfer")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)
CF = "open-r1/codeforces"
COTS = "open-r1/codeforces-cots"
RAW = "/vol/cf_raw"
TESTS = "/vol/cf_tests"
OUT = "/vol/pool"
MAX_GEN = 30          # generated tests kept per problem
MAX_TEST_BYTES = 8_000_000
BANDS = [("easy", 0, 1600, 0.15), ("mid", 1600, 2200, 0.55), ("hard", 2200, 10_000, 0.30)]


def nl(s):
    return (s or "").replace("\r\n", "\n")


def band_of(r):
    for name, lo, hi, _ in BANDS:
        if lo <= r < hi:
            return name


def statement(r):
    st = [nl(r["description"])]
    for head, key in (("Input", "input_format"), ("Output", "output_format"), ("Note", "note")):
        if r.get(key):
            st += [f"## {head}", nl(r[key])]
    return "\n\n".join(st)


def norm_desc_hash(r):
    return hashlib.md5(" ".join((r.get("title") or "").split() + (r.get("description") or "").split()).encode()).hexdigest()


def safe(id_):
    return id_.replace("/", "_")


# ---------------------------------------------------------------- downloads
@app.function(image=image, secrets=[secret], volumes={"/vol": data_vol}, timeout=3 * 3600, cpu=8)
def download():
    from huggingface_hub import snapshot_download
    snapshot_download(CF, repo_type="dataset", local_dir=RAW + "/codeforces",
                      allow_patterns=["verifiable/*.parquet"], max_workers=16)
    snapshot_download(COTS, repo_type="dataset", local_dir=RAW + "/cots",
                      allow_patterns=["solutions_decontaminated/*.parquet", "README.md"], max_workers=16)
    data_vol.commit()
    print(subprocess.run(["du", "-sh", RAW + "/codeforces", RAW + "/cots"], capture_output=True, text=True).stdout)


@app.function(image=image, secrets=[secret], volumes={"/vol": weights_vol}, timeout=3600, cpu=8)
def download_weights():
    from huggingface_hub import snapshot_download
    snapshot_download("Qwen/Qwen3.5-9B", local_dir="/vol/qwen3.5-9b", max_workers=8)
    weights_vol.commit()
    print(subprocess.run(["du", "-sh", "/vol/qwen3.5-9b"], capture_output=True, text=True).stdout)


# ---------------------------------------------------------------- pool
def read_split(split):
    import glob
    import pyarrow.parquet as pq
    cols = ["id", "aliases", "contest_id", "index", "time_limit", "memory_limit", "title", "description",
            "input_format", "output_format", "interaction_format", "note", "examples", "rating", "tags",
            "official_tests", "official_tests_complete", "input_mode", "generated_checker", "executable",
            "generated_tests", "contest_start_year"]
    for f in sorted(glob.glob(f"{RAW}/codeforces/verifiable/{split}-*.parquet")):
        for batch in pq.ParquetFile(f).iter_batches(batch_size=32, columns=cols):
            yield from batch.to_pylist()


def ids_of(r):
    return {r["id"], *(r.get("aliases") or [])}


def prefilter(rows, stats, banned_ids=frozenset(), banned_hash=frozenset()):
    """Filters 2-5 that need no extra downloads. Returns candidate rows (with `_ntests_possible`)."""
    seen_ids, seen_h, out = set(), set(), []
    for r in rows:
        stats["start"] = stats.get("start", 0) + 1
        h = norm_desc_hash(r)
        if r["id"] in seen_ids or h in seen_h:
            stats["dup"] = stats.get("dup", 0) + 1; continue
        if ids_of(r) & banned_ids or h in banned_hash:
            stats["holdout_overlap"] = stats.get("holdout_overlap", 0) + 1; continue
        seen_ids.add(r["id"]); seen_h.add(h)
        if r.get("interaction_format"):
            stats["interactive"] = stats.get("interactive", 0) + 1; continue
        if r.get("input_mode") != "stdio":
            stats["file_io"] = stats.get("file_io", 0) + 1; continue
        if not r.get("executable") or not r.get("description") or not r.get("examples") or r.get("rating") is None:
            stats["not_executable_or_incomplete"] = stats.get("not_executable_or_incomplete", 0) + 1; continue
        n_off = len(r.get("official_tests") or [])
        if n_off + (r.get("generated_tests") or 0) < 5:
            stats["lt5_tests"] = stats.get("lt5_tests", 0) + 1; continue
        out.append(r)
    stats["after_prefilter"] = len(out)
    return out


def fetch_generated(contests):
    """Download generated-test parquet for these contests only. Returns {contest: path or None}."""
    from huggingface_hub import hf_hub_download
    def one(c):
        try:
            return c, hf_hub_download(CF, f"generated_tests/test_cases_{c}.parquet", repo_type="dataset", local_dir=TESTS)
        except Exception as e:
            print("no generated tests for", c, type(e).__name__)
            return c, None
    with ThreadPoolExecutor(8) as ex:
        return dict(ex.map(one, sorted(contests)))


def pick_generated(gen, rng):
    gen = [g for g in gen if len(g["input"]) <= MAX_TEST_BYTES and len(g["output"]) <= MAX_TEST_BYTES]
    if len(gen) <= MAX_GEN:
        return gen
    by_size = sorted(gen, key=lambda g: -len(g["input"]))
    big, rest = by_size[:MAX_GEN // 2], by_size[MAX_GEN // 2:]
    return big + rng.sample(rest, MAX_GEN - len(big))


def run_checker(src, tests):
    """Sanity check: the checker must accept the jury answer. Returns its (max) score, or None if rejected."""
    t = tests[0]
    with tempfile.TemporaryDirectory() as d:
        for n, s in (("checker.py", src), ("input.txt", t["input"]), ("correct.txt", t["output"]), ("out.txt", t["output"])):
            open(f"{d}/{n}", "w").write(s)
        try:
            p = subprocess.run([sys.executable, f"{d}/checker.py", f"{d}/input.txt", f"{d}/correct.txt", f"{d}/out.txt"],
                               capture_output=True, text=True, timeout=10, cwd=d)
            score = float(p.stdout.strip().split()[-1])
            return score if p.returncode == 0 and score > 0 else None
        except Exception:
            return None


def materialize(cands, gen_paths, rng, stats):
    """Attach full tests (official + generated), enforce >=5 tests and working checker; write test files."""
    import pyarrow.parquet as pq
    by_contest = {}
    for r in cands:
        by_contest.setdefault(r["contest_id"], []).append(r)
    good = []
    for c, rs in by_contest.items():
        gen_by_pid = {}
        if gen_paths.get(c):
            ids = [r["id"] for r in rs]
            tbl = pq.read_table(gen_paths[c], columns=["problem_id", "input", "output", "test_i"],
                                filters=[("problem_id", "in", ids)])
            for row in tbl.to_pylist():
                gen_by_pid.setdefault(row["problem_id"], []).append(row)
        for r in rs:
            off = [{"input": nl(t["input"]), "output": nl(t["output"])} for t in r["official_tests"]]
            gen = [{"input": nl(g["input"]), "output": nl(g["output"])} for g in pick_generated(gen_by_pid.get(r["id"], []), rng)]
            tests = off + gen
            if len(tests) < 5:
                stats["lt5_after_fetch"] = stats.get("lt5_after_fetch", 0) + 1; continue
            r["_cmax"] = None
            if r.get("generated_checker"):
                r["_cmax"] = run_checker(r["generated_checker"], tests)
                if r["_cmax"] is None:
                    stats["checker_failed"] = stats.get("checker_failed", 0) + 1; continue
            r["_tests"], r["_n_off"], r["_n_gen"] = tests, len(off), len(gen)
            good.append(r)
    return good


def to_record(r):
    os.makedirs(OUT + "/tests", exist_ok=True)
    path = f"{OUT}/tests/{safe(r['id'])}.json"
    json.dump({"tests": r["_tests"]}, open(path, "w"))
    return {
        "id": r["id"], "title": r["title"], "statement": statement(r), "rating": r["rating"], "tags": r["tags"] or [],
        "time_limit": float(r["time_limit"] or 2.0), "memory_limit": float(r["memory_limit"] or 256.0),
        "sample_tests": [{"input": nl(e["input"]), "output": nl(e["output"])} for e in r["examples"]],
        "full_tests": {"path": path, "n": len(r["_tests"]), "n_official": r["_n_off"], "n_generated": r["_n_gen"]},
        "checker_lang": "python" if r.get("generated_checker") else None,
        "checker_max_score": r.get("_cmax"),  # score printed on the jury answer (1 or 100); require score >= this
        "checker": r.get("generated_checker"),  # python checker.py input correct_output contestant_output -> score; None = exact match (token/whitespace tolerant)
        "year": r.get("contest_start_year"), "contest_id": r["contest_id"],
    }


def mix(recs):
    out = {n: 0 for n, *_ in BANDS}
    for r in recs:
        out[band_of(r["rating"])] += 1
    return out


@app.function(image=image, secrets=[secret], volumes={"/vol": data_vol}, timeout=4 * 3600, cpu=8, memory=32768)
def build_pool(n_pool: int = 1500, n_eval: int = 150, seed: int = 0):
    rng = random.Random(seed)
    stats = {"train": {}, "test": {}}
    test_rows = list(read_split("test"))
    banned_ids = set().union(*(ids_of(r) for r in test_rows))
    banned_hash = {norm_desc_hash(r) for r in test_rows}

    # ---- eval (test split, never trained on)
    ecands = prefilter(test_rows, stats["test"])
    rng.shuffle(ecands)
    gp = fetch_generated({r["contest_id"] for r in ecands})
    egood = materialize(ecands, gp, rng, stats["test"])
    egood = egood[:n_eval]
    eval_recs = [to_record(r) for r in egood]
    del test_rows

    # ---- pool (train split)
    cands = prefilter(read_split("train"), stats["train"], banned_ids, banned_hash)
    by_band = {n: [] for n, *_ in BANDS}
    for r in cands:
        by_band[band_of(r["rating"])].append(r)
    stats["train"]["available_by_band"] = {k: len(v) for k, v in by_band.items()}
    quotas = {n: round(n_pool * share) for n, _, _, share in BANDS}
    # Prefer problems with plenty of ground truth: that only matters ties; random order otherwise.
    chosen = []
    for n, *_ in BANDS:
        rng.shuffle(by_band[n])
        chosen += by_band[n][: int(quotas[n] * 1.15) + 5]   # oversample: some drop at the fetch/checker steps
    gp = fetch_generated({r["contest_id"] for r in chosen})
    good = materialize(chosen, gp, rng, stats["train"])
    final = []
    for n, *_ in BANDS:
        final += [r for r in good if band_of(r["rating"]) == n][: quotas[n]]
    stats["train"]["quotas"] = quotas
    pool_recs = [to_record(r) for r in final]
    for r in pool_recs:  # sanity: no overlap with eval
        assert r["id"] not in {e["id"] for e in eval_recs}
    os.makedirs(OUT, exist_ok=True)
    with open(OUT + "/pool.jsonl", "w") as f:
        for r in pool_recs: f.write(json.dumps(r) + "\n")
    with open(OUT + "/eval150.jsonl", "w") as f:
        for r in eval_recs: f.write(json.dumps(r) + "\n")
    stats["pool"] = {"n": len(pool_recs), "mix": mix(pool_recs), "with_checker": sum(1 for r in pool_recs if r["checker"]),
                     "mean_tests": sum(r["full_tests"]["n"] for r in pool_recs) / max(1, len(pool_recs)),
                     "with_generated": sum(1 for r in pool_recs if r["full_tests"]["n_generated"])}
    stats["eval"] = {"n": len(eval_recs), "mix": mix(eval_recs), "with_checker": sum(1 for r in eval_recs if r["checker"])}
    json.dump(stats, open(OUT + "/stats.json", "w"), indent=1)
    data_vol.commit()
    print(json.dumps(stats, indent=1))
    print(subprocess.run(["du", "-sh", OUT + "/tests", TESTS], capture_output=True, text=True).stdout)


@app.local_entrypoint()
def main(step: str = "pool"):
    {"download": download, "weights": download_weights, "pool": build_pool}[step].remote()


@app.local_entrypoint()
def pull():
    os.makedirs("data/pool", exist_ok=True)
    for f in ("pool.jsonl", "eval150.jsonl", "stats.json", "r1_hard.jsonl", "r1_hard_all.jsonl"):
        subprocess.run(["modal", "volume", "get", "--force", "cpp-data", f"pool/{f}", f"data/pool/{f}"])
