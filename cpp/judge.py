"""Compile and run C++20 under time/memory limits. Works on macOS (g++-16) and Linux (g++)."""
import json, os, resource, shutil, subprocess, sys, tempfile, time

CXX = os.environ.get("CXX") or shutil.which("g++-16") or shutil.which("g++-14") or "g++"
FLAGS = ["-std=c++20", "-O2", "-pipe"]
if sys.platform == "darwin":  # Homebrew GCC needs the current SDK spelled out
    try:
        FLAGS += ["-isysroot", subprocess.check_output(["xcrun", "--show-sdk-path"], text=True).strip()]
    except Exception:
        pass
MAX_OUT = 64 * 1024 * 1024


STACK_MB = 256  # Codeforces gives a 256 MB stack; Modal's gVisor caps the main-thread stack at ~8 MB
# Linked after the user's TU: a constructor switches to an mmap'd 256 MB stack and runs the unmodified
# main() there (keeps implicit `return 0`), then exit()s. Running last in init order means the user's
# global constructors have already run.
WRAPPER = r"""
#include <sys/mman.h>
#include <ucontext.h>
#include <cstdlib>
int main(int, char**);
static ucontext_t g_main_ctx, g_run_ctx;
static void run() { char* av[] = {(char*)"a.out", nullptr}; exit(main(1, av)); }
__attribute__((constructor)) static void switch_stack() {
    size_t sz = (size_t)%d << 20;
    void* p = mmap(nullptr, sz, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
    if (p == MAP_FAILED) return;  // fall back to the normal stack
    getcontext(&g_run_ctx);
    g_run_ctx.uc_stack.ss_sp = p; g_run_ctx.uc_stack.ss_size = sz; g_run_ctx.uc_link = &g_main_ctx;
    makecontext(&g_run_ctx, run, 0);
    swapcontext(&g_main_ctx, &g_run_ctx);
}
""" % STACK_MB


def compile_cpp(code: str, workdir: str, timeout: float = 60) -> tuple[bool, str, str]:
    src, exe = os.path.join(workdir, "a.cpp"), os.path.join(workdir, "a.out")
    files = [src]
    with open(src, "w") as f:
        f.write(code)
    if sys.platform.startswith("linux"):  # macOS (dev only) already allows a raised stack rlimit
        wrap = os.path.join(workdir, "wrap.cpp")
        with open(wrap, "w") as f:
            f.write(WRAPPER)
        files.append(wrap)
    try:
        p = subprocess.run([CXX, *FLAGS, *files, "-o", exe],
                           capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return False, "Compilation timed out", exe
    return p.returncode == 0, p.stderr[-4000:], exe


def _limits(ml_mb: float, tl: float):
    def fn():
        cpu = int(tl) + 2
        resource.setrlimit(resource.RLIMIT_CPU, (cpu, cpu))
        if sys.platform.startswith("linux"):
            b = int((ml_mb + STACK_MB) * 1024 * 1024)  # the big stack thread's mapping counts toward AS
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


TESTLIB_URL = "https://raw.githubusercontent.com/MikeMirzayanov/testlib/master/testlib.h"


def testlib_dir() -> str | None:
    """Directory holding testlib.h (env TESTLIB_DIR, baked into the Modal image, or fetched once)."""
    for d in (os.environ.get("TESTLIB_DIR"), "/opt/testlib", os.path.join(os.path.dirname(__file__), "..", "data", "testlib")):
        if d and os.path.exists(os.path.join(d, "testlib.h")):
            return d
    d = os.path.join(os.path.dirname(__file__), "..", "data", "testlib")
    try:
        import urllib.request
        os.makedirs(d, exist_ok=True)
        urllib.request.urlretrieve(TESTLIB_URL, os.path.join(d, "testlib.h"))
        return d
    except Exception:
        return None


def compile_checker(src: str, workdir: str) -> tuple[bool, str, str]:
    path, exe = os.path.join(workdir, "chk.cpp"), os.path.join(workdir, "chk.out")
    with open(path, "w") as f:
        f.write(src)
    inc = testlib_dir()
    cmd = [CXX, *FLAGS, *(["-I", inc] if inc else []), path, "-o", exe]
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    except subprocess.TimeoutExpired:
        return False, "checker compile timeout", exe
    return p.returncode == 0, p.stderr[-2000:], exe


def run_checker(chk_exe: str, workdir: str, inp: str, got: str, want: str) -> bool:
    """testlib convention: checker <input> <contestant output> <jury answer>; exit 0 means accepted."""
    paths = []
    for name, data in (("in", inp), ("out", got), ("ans", want)):
        p = os.path.join(workdir, f"t.{name}")
        with open(p, "w") as f:
            f.write(data)
        paths.append(p)
    try:
        r = subprocess.run([chk_exe, *paths], capture_output=True, timeout=20)
    except subprocess.TimeoutExpired:
        return False
    return r.returncode == 0


def run_py_checker(chk_path: str, workdir: str, inp: str, got: str, want: str) -> str:
    """open-r1 generated_checker: `python checker.py input correct_output contestant_output` prints a score.
    Returns 'AC' (exit 0 and last stdout token is a number > 0), 'WA', or 'JE' (judge error: the checker
    timed out, or it crashes even when the contestant output is the jury answer)."""
    def once(contestant: str) -> str | None:
        paths = []
        for name, data in (("in", inp), ("ans", want), ("out", contestant)):
            p = os.path.join(workdir, f"p.{name}")
            with open(p, "w") as f:
                f.write(data)
            paths.append(p)
        try:
            r = subprocess.run([sys.executable, chk_path, *paths], capture_output=True, text=True,
                               timeout=PY_CHECKER_TIMEOUT, cwd=workdir)
            if r.returncode != 0:
                return "crash"
            return "AC" if float(r.stdout.strip().split()[-1]) > 0 else "WA"
        except subprocess.TimeoutExpired:
            return "timeout"
        except (ValueError, IndexError):  # no numeric score printed
            return "crash"
    v = once(got)
    if v in ("AC", "WA"):
        return v
    if v == "timeout":
        return "JE"
    # crashed on the contestant output: WA if the checker is healthy on the jury answer, else judge error
    return "WA" if once(want) in ("AC", "WA") else "JE"


PY_CHECKER_TIMEOUT = 30
_TEST_CACHE: dict[str, list] = {}


def resolve_tests(tests) -> list[dict]:
    """tests may be an inline list, a dict {path,...} or a path to a JSON file ({"tests":[...]} or a list).
    Pool paths look like /vol/pool/tests/<id>.json (Modal volume cpp-data); locally set VOL_ROOT to remap /vol/."""
    if isinstance(tests, list):
        return tests
    path = tests["path"] if isinstance(tests, dict) else tests
    if path in _TEST_CACHE:
        return _TEST_CACHE[path]
    cands = [path]
    if os.environ.get("VOL_ROOT") and path.startswith("/vol/"):
        cands.insert(0, os.environ["VOL_ROOT"].rstrip("/") + "/" + path[5:])
    for c in cands:
        if os.path.exists(c):
            data = json.load(open(c))
            data = data["tests"] if isinstance(data, dict) else data
            if len(_TEST_CACHE) > 64:
                _TEST_CACHE.clear()
            _TEST_CACHE[path] = data
            return data
    raise FileNotFoundError(f"tests file not found: {cands}")


def n_tests(tests) -> int:
    if isinstance(tests, dict) and "n" in tests:
        return tests["n"]
    return len(resolve_tests(tests))


def judge(code: str, tests, tl: float = 2.0, ml_mb: float = 1024,
          checker: str | None = None, stop_on_fail: bool = True, checker_lang: str | None = None) -> dict:
    """tests: [{'input','output'}] or {path,...}/path (see resolve_tests). checker: None (token diff) or source,
    with checker_lang 'python' (open-r1 score checker) or 'cpp' (testlib); if None it is guessed ('#include' -> cpp).
    Returns {compiled, compile_error, passed, total, verdicts[AC|WA|TLE|RE|JE], max_time}."""
    tests = resolve_tests(tests)
    if checker and not checker_lang:
        checker_lang = "cpp" if "#include" in checker else "python"
    d = tempfile.mkdtemp(prefix="judge_")
    try:
        ok, err, exe = compile_cpp(code, d)
        if not ok:
            return {"compiled": False, "compile_error": err, "passed": 0, "total": len(tests),
                    "verdicts": [], "max_time": 0.0}
        chk_exe = None
        if checker and checker_lang == "cpp":
            cok, cerr, chk_exe = compile_checker(checker, d)
            if not cok:
                raise RuntimeError(f"checker failed to compile: {cerr}")
        elif checker:
            chk_py = os.path.join(d, "checker.py")
            with open(chk_py, "w") as f:
                f.write(checker)
        verdicts, max_t = [], 0.0
        for t in tests:
            r = run_exe(exe, t["input"], tl, ml_mb)
            if r["status"] != "OK":
                v = r["status"]
            elif chk_exe:
                v = "AC" if run_checker(chk_exe, d, t["input"], r["stdout"], t["output"]) else "WA"
            elif checker:
                v = run_py_checker(chk_py, d, t["input"], r["stdout"], t["output"])
            else:
                v = "AC" if outputs_match(r["stdout"], t["output"]) else "WA"
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
