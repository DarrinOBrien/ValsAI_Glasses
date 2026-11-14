"""Compile and run C++20 under time/memory limits. Works on macOS (g++-16) and Linux (g++)."""
import os, resource, shutil, subprocess, sys, tempfile, time

CXX = os.environ.get("CXX") or shutil.which("g++-16") or shutil.which("g++-14") or "g++"
FLAGS = ["-std=c++20", "-O2", "-pipe"]
if sys.platform == "darwin":  # Homebrew GCC needs the current SDK spelled out
    try:
        FLAGS += ["-isysroot", subprocess.check_output(["xcrun", "--show-sdk-path"], text=True).strip()]
    except Exception:
        pass
MAX_OUT = 64 * 1024 * 1024


def compile_cpp(code: str, workdir: str, timeout: float = 60) -> tuple[bool, str, str]:
    src, exe = os.path.join(workdir, "a.cpp"), os.path.join(workdir, "a.out")
    with open(src, "w") as f:
        f.write(code)
    try:
        p = subprocess.run([CXX, *FLAGS, src, "-o", exe], capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return False, "Compilation timed out", exe
    return p.returncode == 0, p.stderr[-4000:], exe


def _limits(ml_mb: float, tl: float):
    def fn():
        cpu = int(tl) + 2
        resource.setrlimit(resource.RLIMIT_CPU, (cpu, cpu))
        if sys.platform.startswith("linux"):
            b = int(ml_mb * 1024 * 1024)
            resource.setrlimit(resource.RLIMIT_AS, (b, b))
            resource.setrlimit(resource.RLIMIT_STACK, (resource.RLIM_INFINITY, resource.RLIM_INFINITY))
        else:
            try:
                hard = resource.getrlimit(resource.RLIMIT_STACK)[1]
                resource.setrlimit(resource.RLIMIT_STACK, (hard, hard))
            except Exception:
                pass
    return fn


def run_exe(exe: str, stdin: str, tl: float = 2.0, ml_mb: float = 1024) -> dict:
    """status is one of OK, TLE, RE (runtime error; includes memory limit)."""
    t0 = time.time()
    try:
        p = subprocess.run([exe], input=stdin.encode(), capture_output=True, timeout=tl + 0.5,
                           preexec_fn=_limits(ml_mb, tl))
    except subprocess.TimeoutExpired as e:
        return {"status": "TLE", "stdout": (e.stdout or b"")[:MAX_OUT].decode(errors="replace"),
                "stderr": "", "time": time.time() - t0, "code": None}
    dt = time.time() - t0
    status = "OK" if p.returncode == 0 else ("TLE" if dt > tl else "RE")
    if dt > tl:
        status = "TLE"
    return {"status": status, "stdout": p.stdout[:MAX_OUT].decode(errors="replace"),
            "stderr": p.stderr[-2000:].decode(errors="replace"), "time": dt, "code": p.returncode}


def outputs_match(got: str, want: str) -> bool:
    """Token-wise compare, like the standard Codeforces diff checker."""
    return got.split() == want.split()


def judge(code: str, tests: list[dict], tl: float = 2.0, ml_mb: float = 1024,
          stop_on_fail: bool = True) -> dict:
    """tests: [{'input':..., 'output':...}]. Returns compile status and per-test verdicts."""
    d = tempfile.mkdtemp(prefix="judge_")
    try:
        ok, err, exe = compile_cpp(code, d)
        if not ok:
            return {"compiled": False, "compile_error": err, "passed": 0, "total": len(tests),
                    "verdicts": [], "max_time": 0.0}
        verdicts, max_t = [], 0.0
        for t in tests:
            r = run_exe(exe, t["input"], tl, ml_mb)
            v = r["status"] if r["status"] != "OK" else ("AC" if outputs_match(r["stdout"], t["output"]) else "WA")
            verdicts.append(v)
            max_t = max(max_t, r["time"])
            if v != "AC" and stop_on_fail:
                break
        return {"compiled": True, "compile_error": "", "passed": verdicts.count("AC"),
                "total": len(tests), "verdicts": verdicts, "max_time": max_t}
    finally:
        shutil.rmtree(d, ignore_errors=True)


def run_code(code: str, stdin: str, tl: float = 2.0, ml_mb: float = 1024) -> dict:
    """Compile + run once on arbitrary stdin (the harness 'run' tool)."""
    d = tempfile.mkdtemp(prefix="run_")
    try:
        ok, err, exe = compile_cpp(code, d)
        if not ok:
            return {"compiled": False, "compile_error": err}
        return {"compiled": True, **run_exe(exe, stdin, tl, ml_mb)}
    finally:
        shutil.rmtree(d, ignore_errors=True)
