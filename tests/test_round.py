import json, os

from scripts import round as R


def ctx(tmp_path, **kw):
    return R.Ctx(tag="r1", root=str(tmp_path), eval_url="http://x/v1", **kw)


def test_plan_commands(tmp_path):
    c = ctx(tmp_path, model="r0", init="tinker://s", lr=5e-5, pool="/vol/p.jsonl", extra_data=["data/r1.jsonl"])
    assert "ADAPTERS=r0=/vol/adapters/r0" in R.commands("deploy_sample", c)[0].show()
    s = R.commands("sample", c)[0].argv
    assert s[-6:] == ["--attempts", "4", "--model", "r0", "--pool", "/vol/p.jsonl"]
    t = R.commands("train", c)[0].argv
    assert "--init" in t and "tinker://s" in t and "--lr" in t and "data/r1.jsonl" in t
    assert R.commands("eval", c)[0].argv[-4:-2] == ["--model", "r1"]


def test_resume_and_state(tmp_path):
    c = ctx(tmp_path)
    ran = []

    def runner(cmd, ctx_, out=None):
        ran.append(cmd.argv[3] if cmd.argv[2] == "-m" else " ".join(cmd.argv[4:6]))
        for o in R.outputs(stage[0], ctx_):
            os.makedirs(os.path.dirname(o), exist_ok=True)
            os.makedirs(o, exist_ok=True) if o.endswith("adapter") else open(o, "w").write("x")
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
