"""Optional hard-problem R1 set: candidate DeepSeek-R1 traces (open-r1/codeforces-cots, solutions_decontaminated)
for problems rated 2200+ with a C++ solution. Runs on Modal after `build_pool.py::main --step pool`.

    uv run --env-file .env modal run scripts/build_cots.py

Writes cpp-data:/pool/r1_hard.jsonl (<=200, one shortest trace per problem, pool problems first) and
r1_hard_all.jsonl (every eligible 2200+ trace). Records:
    {id, rating, tokens, thinking, code, in_pool}
Traces are NOT verified here. Verify the code against pool tests in the sandbox before use, and rewrite into harness
format (see scripts/build_p1.py for the older conversion). Problems in eval150 (or aliases) are excluded.
"""
import glob, json, os, re, sys
import modal

sys.path.insert(0, os.path.dirname(__file__))
from build_pool import app, image, secret, data_vol, read_split, ids_of, RAW, OUT  # noqa: E402

CODE_RE = re.compile(r"```(?:cpp|c\+\+|C\+\+)?\s*\n(.*?)```", re.DOTALL)
MIN_RATING, TARGET = 2200, 200


@app.function(image=image, secrets=[secret], volumes={"/vol": data_vol}, timeout=3600, cpu=4, memory=16384)
def build_cots():
    import pyarrow.parquet as pq
    eval_ids = set()
    for l in open(OUT + "/eval150.jsonl"):
        eval_ids.add(json.loads(l)["id"])
    pool_ids = {json.loads(l)["id"] for l in open(OUT + "/pool.jsonl")}
    # alias-expanded exclusion of the whole test split (never train on it) + rating lookup from the train split
    banned = set()
    for r in read_split("test"):
        banned |= ids_of(r)
    rating = {r["id"]: r["rating"] for r in read_split("train") if r.get("rating") is not None}
    best, stats = {}, {"rows": 0, "rating<2200_or_unknown": 0, "test_split": 0, "not_finished": 0, "no_code": 0}
    for f in sorted(glob.glob(f"{RAW}/cots/solutions_decontaminated/*.parquet")):
        for batch in pq.ParquetFile(f).iter_batches(batch_size=64, columns=["id", "aliases", "generation", "finish_reason", "api_metadata", "interaction_format"]):
            for r in batch.to_pylist():
                stats["rows"] += 1
                if r["id"] in banned or set(r.get("aliases") or []) & banned or r["id"] in eval_ids:
                    stats["test_split"] += 1; continue
                if rating.get(r["id"], 0) < MIN_RATING or r.get("interaction_format"):
                    stats["rating<2200_or_unknown"] += 1; continue
                g = r["generation"] or ""
                if r["finish_reason"] != "stop" or "</think>" not in g:
                    stats["not_finished"] += 1; continue
                think, answer = g.split("</think>", 1)
                think = think.replace("<think>", "").strip()
                blocks = CODE_RE.findall(answer)
                if not blocks or not think:
                    stats["no_code"] += 1; continue
                rec = {"id": r["id"], "rating": rating[r["id"]], "tokens": r["api_metadata"]["completion_tokens"],
                       "thinking": think, "code": blocks[-1].strip(), "in_pool": r["id"] in pool_ids}
                if r["id"] not in best or rec["tokens"] < best[r["id"]]["tokens"]:
                    best[r["id"]] = rec
    allr = sorted(best.values(), key=lambda x: (not x["in_pool"], x["tokens"]))
    sel = allr[:TARGET]
    for name, rs in (("r1_hard_all.jsonl", allr), ("r1_hard.jsonl", sel)):
        with open(f"{OUT}/{name}", "w") as fh:
            for r in rs: fh.write(json.dumps(r) + "\n")
    stats.update(eligible_problems=len(allr), eligible_in_pool=sum(r["in_pool"] for r in allr), selected=len(sel),
                 selected_in_pool=sum(r["in_pool"] for r in sel),
                 selected_rating_mean=sum(r["rating"] for r in sel) / max(1, len(sel)),
                 selected_mean_tokens=sum(r["tokens"] for r in sel) / max(1, len(sel)))
    json.dump(stats, open(OUT + "/r1_stats.json", "w"), indent=1)
    data_vol.commit()
    print(json.dumps(stats, indent=1))


@app.local_entrypoint()
def main():
    build_cots.remote()
