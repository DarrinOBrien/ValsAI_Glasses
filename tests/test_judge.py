"""Sandbox validation against known-good / known-bad C++ solutions.
Run: uv run --with pytest pytest tests   (memory/stack-limit tests only run on Linux; use Modal for those)
"""
import sys
import pytest
from cpp import judge

linux_only = pytest.mark.skipif(not sys.platform.startswith("linux"), reason="rlimit AS/stack semantics are Linux")

SUM = '#include <bits/stdc++.h>\nint main(){long long a,b;std::cin>>a>>b;std::cout<<a+b<<"\\n";}'
WRONG = SUM.replace("a+b", "a-b")
LOOP = "int main(){volatile long long x=0; for(;;) x++;}"
CRASH = "#include <cstdlib>\nint main(){int*p=nullptr; return *p;}"
EXIT1 = "int main(){return 1;}"
BIGMEM = ("#include <vector>\nint main(){std::vector<char> v(2000000000); "
          "for(size_t i=0;i<v.size();i+=4096) v[i]=1; return v[4096];}")
DEEP = ('#include <cstdio>\nint f(int n){ volatile int a[64]; a[0]=n; return n? f(n-1)+a[0]:0;}\n'
        'int main(){printf("%d\\n", f(1000000)>0);}')
SLOW = '#include <cstdio>\nint main(){volatile long long s=0; for(long long i=0;i<30000000000LL;i++) s+=i; puts("0");}'
CHECKER = """#include "testlib.h"
int main(int argc,char*argv[]){registerTestlibCmd(argc,argv);
 long long a=inf.readLong(), b=inf.readLong(); long long x=ouf.readLong();
 if(x==a+b) quitf(_ok,"ok"); quitf(_wa,"bad");}"""
TESTS = [{"input": "1 2\n", "output": "3\n"}, {"input": "-5 5\n", "output": "0\n"}]


def v(code, tl=2.0, ml=256, **kw):
    return judge.judge(code, TESTS, tl, ml, **kw)


def test_ac():
    r = v(SUM)
    assert r["compiled"] and r["verdicts"] == ["AC", "AC"] and r["passed"] == 2


def test_wa_stops_and_counts():
    r = v(WRONG)
    assert r["verdicts"][0] == "WA" and r["passed"] < 2
    assert v(WRONG, stop_on_fail=False)["verdicts"] == ["WA", "WA"]


def test_compile_error():
    r = v("int main(){ this is not c++ }")
    assert not r["compiled"] and r["compile_error"]


def test_cpp20_features():
    code = ("#include <bits/stdc++.h>\nint main(){ auto f=[]<class T>(T x){return std::popcount((unsigned)x);};"
            " long long x,y; std::cin>>x>>y; std::cout<<x+y+(f(1)-1)<<\"\\n\"; }")
    assert v(code)["passed"] == 2


def test_tle_infinite_loop():
    assert v(LOOP, tl=1.0)["verdicts"] == ["TLE"]


def test_tle_slow():
    assert v(SLOW, tl=1.0)["verdicts"] == ["TLE"]


def test_runtime_error():
    assert v(CRASH)["verdicts"] == ["RE"]
    assert v(EXIT1)["verdicts"] == ["RE"]


@linux_only
def test_memory_limit():
    assert v(BIGMEM, ml=256)["verdicts"] == ["RE"]


@linux_only
def test_deep_recursion_needs_big_stack():
    assert judge.judge(DEEP, [{"input": "", "output": "1"}], 2.0, 1024)["verdicts"] == ["AC"]


def test_whitespace_insensitive():
    assert judge.judge(SUM, [{"input": "1 2", "output": "  3  \n\n"}], 2.0, 256)["verdicts"] == ["AC"]


def test_checker():
    assert judge.judge(SUM, TESTS, 2.0, 256, CHECKER)["verdicts"] == ["AC", "AC"]
    assert judge.judge(WRONG, TESTS, 2.0, 256, CHECKER)["verdicts"][0] == "WA"


def test_checker_ignores_jury_answer_diff():
    t = [{"input": "1 2\n", "output": "999\n"}]  # token diff would say WA; the checker only looks at input
    assert judge.judge(SUM, t, 2.0, 256, None)["verdicts"] == ["WA"]
    assert judge.judge(SUM, t, 2.0, 256, CHECKER)["verdicts"] == ["AC"]


def test_implicit_return_globals_and_args_main():
    code = ('#include <bits/stdc++.h>\nstd::vector<int> g = {1,2,3}; std::map<int,int> m; struct S{S(){m[1]=5;}} s;\n'
            'int main(int argc, char** argv){ std::cout << g.size()+m[1] << "\\n"; }')
    assert judge.judge(code, [{"input": "", "output": "8"}], 2.0, 256)["verdicts"] == ["AC"]


def test_exit_codes():
    t = [{"input": "", "output": "1"}]
    assert judge.judge('#include <cstdlib>\nint main(){ exit(3); }', t, 2.0, 256)["verdicts"] == ["RE"]
    assert judge.judge('#include <cstdlib>\n#include <cstdio>\nint main(){ puts("1"); exit(0); }',
                       t, 2.0, 256)["verdicts"] == ["AC"]


def test_run_code():
    r = judge.run_code(SUM, "2 3\n", 2.0, 256)
    assert r["compiled"] and r["status"] == "OK" and r["stdout"].strip() == "5"
    assert judge.run_code("garbage", "", 2.0, 256)["compiled"] is False
