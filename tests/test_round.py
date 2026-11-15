import json, os

from scripts import round as R


def ctx(tmp_path, **kw):
    return R.Ctx(tag="r1", root=str(tmp_path), eval_url="http://x/v1", **kw)


def test_round1_merge_path_by_default(tmp_path):
    c = ctx(tmp_path, pool="/vol/p.jsonl")
    d = R.commands("deploy_sample", c)[0]
    assert d.env == {"N_REPLICAS": "3"}                       # round 1: base weights, no ADAPTERS, no MERGED_MODEL
    s = R.commands("sample", c)[0].argv
    assert s[-6:] == ["--tag", "r1", "--attempts", "4", "--pool", "/vol/p.jsonl"] and "--model" not in s
    de = R.commands("deploy_eval", c)[0]
    assert de.env == {"N_REPLICAS": "1", "MERGED_MODEL": "/vol/merged/r1"} and "ADAPTERS" not in de.env
    ex = R.commands("export", c)[0].argv
    assert ex[-4:] == ["--name", "r1", "--merge"][-4:] or ex[-3:] == ["--name", "r1", "--merge"]
    assert R.commands("eval", c)[0].argv[-4:-2] == ["--model", "qwen"]     # merged server always answers as "qwen"
    assert all("volume" not in " ".join(x.argv) or "get" in x.argv for st in R.STAGES for x in R.commands(st, c))  # no `volume put`


def test_stage_order_deploys_before_export():
    # export_lora calls the deployed app's `build` function, so the app must be deployed first
    assert R.STAGES.index("deploy_eval") < R.STAGES.index("export") < R.STAGES.index("eval")
    assert R.STAGES.index("stop_sample") < R.STAGES.index("deploy_eval")


def test_round2_plumbing(tmp_path):
    os.makedirs(tmp_path / "runs")
    (tmp_path / "runs" / "checkpoints.jsonl").write_text(
        "\n".join(json.dumps(x) for x in [{"name": "r1", "state_path": "tinker://a/weights/r1-ep0"},
                                          {"name": "other", "state_path": "tinker://zzz"}]) + "\n")
    c = R.Ctx(tag="r2", root=str(tmp_path), model="r1", lr=5e-5, extra_data=["data/r1.jsonl"])
    assert R.commands("deploy_sample", c)[0].env["MERGED_MODEL"] == "/vol/merged/r1"        # sample from round-1 weights
    assert "--model" not in R.commands("sample", c)[0].argv
    t = R.commands("train", c)[0].argv
    assert t[t.index("--init") + 1] == "tinker://a/weights/r1-ep0"                         # looked up from checkpoints.jsonl
    assert t[t.index("--data") + 1].endswith("selected_r2.norm.jsonl") and "--lr" in t
    assert "data/r1.jsonl" in R.commands("normalize", c)[0].argv
    assert R.commands("deploy_eval", c)[0].env["MERGED_MODEL"] == "/vol/merged/r2"          # new checkpoint evaluated as r2
    c2 = R.Ctx(tag="r2", root=str(tmp_path), model="r1", init="tinker://explicit")
    assert R.commands("train", c2)[0].argv[-2:] == ["--init", "tinker://explicit"]           # explicit --init wins


def test_legacy_adapter_path(tmp_path):
    c = R.Ctx(tag="r2", root=str(tmp_path), model="r1", merge=False)
    assert R.commands("deploy_sample", c)[0].env["ADAPTERS"] == "r1=/vol/adapters/r1"
    assert R.commands("sample", c)[0].argv[-2:] == ["--model", "r1"]
    assert "--merge" not in R.commands("export", c)[0].argv


def test_secrets_never_printed(tmp_path, monkeypatch):
    monkeypatch.setenv("VLLM_KEY", "supersecret")
    c = ctx(tmp_path)
    cmd = R.commands("eval", c)[0]
    assert cmd.env == {"OPENAI_API_KEY": "supersecret"} and "supersecret" not in cmd.show()


def test_variant_reuses_data(tmp_path, capsys):
    assert R.main(["plan", "--tag", "zz_v", "--lr", "5e-5", "--reuse-data", "zz_r"]) == 0
    out = capsys.readouterr().out
    assert "[sample]" not in out and "[select]" not in out and "selected_zz_r.norm.jsonl" in out and "[train]" in out


def test_resume_and_state(tmp_path):
    c = ctx(tmp_path)
    ran = []

    def runner(cmd, ctx_, out=None):
        ran.append(cmd.argv[3] if cmd.argv[2] == "-m" else " ".join(cmd.argv[4:6]))
        for o in R.outputs(stage[0], ctx_):
            os.makedirs(os.path.dirname(o), exist_ok=True)
            open(o, "w").write("x")
        if stage[0] == "train":
            os.makedirs(tmp_path / "runs", exist_ok=True)
            (tmp_path / "runs" / "checkpoints.jsonl").write_text(json.dumps({"name": "r1", "state_path": "tinker://st"}) + "\n")
        return 0

    stage = [None]
    wrapped = lambda s: (lambda cmd, ctx_, out=None: (stage.__setitem__(0, s), runner(cmd, ctx_, out))[1])
    # run select+train+export one stage at a time with the stage-aware runner
    for s in ["select", "train", "export"]:
        assert R.run_stages(c, [s], True, runner=wrapped(s), log=lambda *_: None)
    assert R.last_checkpoint(c)["state_path"] == "tinker://st"
    assert "tinker://st" in R.commands("export", c)[0].show()
    st = R.load_state(c)
    assert R.is_done("select", c, st) and R.is_done("train", c, st)
    n = len(ran)
    assert R.run_stages(c, ["select", "train"], True, runner=wrapped("x"), log=lambda *_: None) and len(ran) == n  # skipped
    os.remove(c.selected)  # output vanished -> stage re-runs
    assert not R.is_done("select", c, R.load_state(c))


def test_failure_stops_and_plan_runs_nothing(tmp_path):
    c = ctx(tmp_path)
    calls = []
    assert not R.run_stages(c, ["select", "train"], True, runner=lambda *a: 1 if calls.append(1) is None else 0,
                            log=lambda *_: None) and len(calls) == 1
    assert R.run_stages(c, R.STAGES, False, runner=lambda *a: calls.append(1), log=lambda *_: None) and len(calls) == 1


def test_cli_status_and_plan(tmp_path, capsys):
    assert R.main(["plan", "--tag", "zz_test", "--eval-url", "http://x"]) == 0
    out = capsys.readouterr().out
    assert "(plan)" in out and "scripts.select_traj" in out and "scripts.analyze" in out
