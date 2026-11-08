"""Runs INSIDE the sandbox process. Protocol on stdin: one JSON header line, then script bytes, then npz bytes.
Prints one JSON line. Applies resource limits first, restricts builtins and imports, then calls decide() once per
bar with ONLY the bars up to that point. Never imports ooma itself (so it can live in /opt, owned by root)."""
import builtins
import io
import json
import os
import resource
import sys

_hdr_line = sys.stdin.buffer.readline()
hdr = json.loads(_hdr_line)
import signal  # noqa: E402

for lim, val in ((resource.RLIMIT_CPU, int(hdr.get("cpu_s", 150))), (resource.RLIMIT_AS, int(hdr.get("as_bytes", 3 * 1024 ** 3))),
                 (resource.RLIMIT_FSIZE, 0), (resource.RLIMIT_CORE, 0), (resource.RLIMIT_NOFILE, 64)):
    try:
        resource.setrlimit(lim, (val, val))
    except (ValueError, OSError):
        pass
signal.alarm(int(hdr.get("wall_s", 200)))        # the runner kills ITSELF: no orphan can outlive its wall-clock budget

import numpy as np  # noqa: E402  (imported before restricting __import__)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ooma_lib  # noqa: E402,F401


ALLOWED = {"numpy", "math", "statistics", "ooma_lib"}
_real_import = builtins.__import__


def _guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
    if level or name.split(".")[0] not in ALLOWED:
        raise ImportError(f"import blocked: {name}")
    return _real_import(name, globals, locals, fromlist, level)


SAFE = {k: getattr(builtins, k) for k in (
    "abs", "all", "any", "bool", "dict", "enumerate", "filter", "float", "int", "isinstance", "len", "list", "map",
    "max", "min", "range", "reversed", "round", "set", "sorted", "sum", "tuple", "zip", "True", "False", "None",
    "ValueError", "ZeroDivisionError", "Exception", "IndexError", "KeyError", "TypeError", "divmod", "pow", "slice",
    "FloatingPointError", "ArithmeticError", "OverflowError")}
SAFE["__import__"] = _guarded_import
SAFE["__build_class__"] = None


def fail(msg, code=2):
    sys.stdout.write(json.dumps({"ok": False, "error": msg}) + "\n")
    sys.stdout.flush()
    os._exit(code)


def main():
    out_stream = sys.stdout
    script = sys.stdin.buffer.read(hdr["script_len"]).decode()
    npz = np.load(io.BytesIO(sys.stdin.buffer.read(hdr["npz_len"])), allow_pickle=False)
    data = {k: npz[k] for k in npz.files}
    n = len(next(iter(data.values())))
    start, L, mode = hdr["start"], hdr["lookback"], hdr["mode"]
    for _name in ("save", "savez", "savez_compressed", "savetxt", "load", "loadtxt", "genfromtxt", "fromfile", "memmap", "fromregex"):
        if hasattr(np, _name):
            setattr(np, _name, None)              # input is loaded; from here on the script gets no file I/O through numpy
    ns = {"__builtins__": SAFE, "__name__": "strategy"}
    try:
        exec(compile(script, "<strategy>", "exec"), ns)
        decide = ns["decide"]
    except Exception as e:                                                  # noqa: BLE001
        fail(f"load error: {type(e).__name__}: {e}")
    idx = [n - 1] if mode == "last" else range(start, n)
    dirs, acts, errors, first = [], [], 0, None
    for i in idx:
        lo = max(0, i - L + 1)
        view = {k: v[lo:i + 1] for k, v in data.items()}
        try:
            d, a = decide(view)
            if isinstance(d, float) and d != d or isinstance(a, float) and a != a:
                raise ValueError("NaN returned")
            dirs.append("1" if bool(d) else "0")
            acts.append("1" if bool(a) else "0")
        except Exception as e:                                              # noqa: BLE001
            errors += 1
            first = first or f"bar {i}: {type(e).__name__}: {e}"
            dirs.append("0"); acts.append("0")
    total = max(1, len(dirs))
    res = {"ok": errors / total <= 0.05, "dir": "".join(dirs), "act": "".join(acts), "start": 0 if mode == "last" else start,
           "n": n, "errors": errors, "first_error": first}
    if not res["ok"]:
        res["error"] = f"too many runtime errors ({errors}/{total}); first: {first}"
    out_stream.write(json.dumps(res) + "\n")
    out_stream.flush()


if __name__ == "__main__":
    main()
