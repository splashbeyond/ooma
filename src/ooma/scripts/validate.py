"""Static checks on LLM-written scripts BEFORE they ever run. Defense in depth: the sandbox (separate OS user,
no network, rlimits) is the real wall; this rejects obviously unsafe or non-conforming code early and cheaply.

Contract: module-level constants HORIZON_BARS (int), TIMEFRAME ('1h'|'4h'|'1d'), MARKETS (list or 'ALL');
function decide(bars) -> (direction, act), both booleans. direction True = price will be higher after HORIZON_BARS,
False = lower. act True = take the trade, False = skip. `bars` is a dict of numpy arrays holding ONLY past and
current closed bars (last element = latest closed bar).
"""
from __future__ import annotations

import ast
from dataclasses import dataclass

ALLOWED_IMPORTS = {"numpy", "math", "statistics", "ooma_lib"}
FORBIDDEN_NAMES = {"exec", "eval", "compile", "open", "__import__", "input", "breakpoint", "globals", "locals",
                   "vars", "getattr", "setattr", "delattr", "dir", "type", "memoryview", "help", "exit", "quit",
                   "os", "sys", "subprocess", "socket", "builtins", "importlib", "ctypes", "pickle", "marshal"}
FORBIDDEN_ATTRS = {"load", "save", "savez", "savez_compressed", "savetxt", "loadtxt", "genfromtxt", "fromfile", "tofile",
                   "memmap", "ctypeslib", "f2py", "distutils", "testing", "system", "popen", "DataSource", "show_config",
                   "get_include", "fromregex", "frombuffer_ex"}
TIMEFRAMES = {"1h", "4h", "1d"}
MARKETS = {"BTC", "ETH", "SOL", "HYPE"}
MAX_BYTES = 30_000


class ScriptInvalid(ValueError):
    pass


@dataclass
class ScriptMeta:
    horizon_bars: int
    timeframe: str
    markets: list


def _const(node):
    try:
        return ast.literal_eval(node)
    except Exception as e:                                   # noqa: BLE001
        raise ScriptInvalid(f"module-level constants must be plain literals (line {getattr(node, 'lineno', '?')})") from e


def validate(source: str) -> ScriptMeta:
    if len(source.encode()) > MAX_BYTES:
        raise ScriptInvalid(f"script larger than {MAX_BYTES} bytes")
    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        raise ScriptInvalid(f"syntax error line {e.lineno}: {e.msg}") from e

    consts, has_decide = {}, False
    for node in tree.body:                                   # module level: imports, literal constants, defs, docstring only
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            continue
        if isinstance(node, ast.FunctionDef):
            if node.name == "decide":
                has_decide = True
                if len(node.args.args) != 1 or node.args.vararg or node.args.kwarg or node.args.kwonlyargs:
                    raise ScriptInvalid("decide must take exactly one argument: decide(bars)")
            continue
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant):
            continue                                         # docstring
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            consts[node.targets[0].id] = _const(node.value)
            continue
        raise ScriptInvalid(f"only imports, constants and function definitions are allowed at module level (line {node.lineno})")
    if not has_decide:
        raise ScriptInvalid("missing function decide(bars)")

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                if a.name.split(".")[0] not in ALLOWED_IMPORTS:
                    raise ScriptInvalid(f"import not allowed: {a.name} (allowed: {sorted(ALLOWED_IMPORTS)})")
        elif isinstance(node, ast.ImportFrom):
            if node.level or (node.module or "").split(".")[0] not in ALLOWED_IMPORTS:
                raise ScriptInvalid(f"import not allowed: {node.module}")
            if any(a.name == "*" for a in node.names):
                raise ScriptInvalid("star imports are not allowed")
        elif isinstance(node, ast.Name) and node.id in FORBIDDEN_NAMES:
            raise ScriptInvalid(f"name not allowed: {node.id}")
        elif isinstance(node, ast.Attribute):
            if node.attr.startswith("__") or node.attr in FORBIDDEN_ATTRS:
                raise ScriptInvalid(f"attribute not allowed: .{node.attr}")
        elif isinstance(node, (ast.Global, ast.AsyncFunctionDef, ast.Await, ast.Yield, ast.YieldFrom, ast.ClassDef)):
            raise ScriptInvalid(f"{type(node).__name__} is not allowed")
        elif isinstance(node, ast.Constant) and isinstance(node.value, str) and "__" in node.value:
            raise ScriptInvalid("dunder strings are not allowed")

    h, tf, mk = consts.get("HORIZON_BARS"), consts.get("TIMEFRAME"), consts.get("MARKETS", "ALL")
    if not isinstance(h, int) or isinstance(h, bool) or not 1 <= h <= 200:
        raise ScriptInvalid("HORIZON_BARS must be an int in 1..200")
    if tf not in TIMEFRAMES:
        raise ScriptInvalid(f"TIMEFRAME must be one of {sorted(TIMEFRAMES)}")
    markets = sorted(MARKETS) if mk == "ALL" else mk
    if not isinstance(markets, list) or not markets or any(m not in MARKETS for m in markets):
        raise ScriptInvalid(f"MARKETS must be 'ALL' or a list drawn from {sorted(MARKETS)}")
    return ScriptMeta(h, tf, list(markets))
