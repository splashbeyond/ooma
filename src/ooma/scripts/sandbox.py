"""Spawn scripts in an isolated subprocess.

Production (server): runs as a separate unprivileged OS user (`sudo -n -u ooma-sbx`), whose network egress is blocked by
an iptables owner rule and which cannot read /home/ooma (keys, database). Local/dev: same user, rlimits only.
Either way: restricted builtins and imports, no file writes (RLIMIT_FSIZE=0), CPU/memory limits, wall-clock timeout.
Data goes in via stdin and results come back via stdout: the script never touches the filesystem or the network.
"""
from __future__ import annotations

import io
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
LOOKBACK = 500


@dataclass
class SandboxResult:
    ok: bool
    direction: np.ndarray | None = None       # bool per decided bar
    act: np.ndarray | None = None
    start: int = 0
    error: str | None = None


class SandboxRunner:
    def __init__(self, isolated_user: str | None = None, runner_path: str | None = None, python: str | None = None,
                 timeout_s: int = 240, cpu_s: int = 150):
        self.user, self.timeout, self.cpu = isolated_user, timeout_s, cpu_s
        if isolated_user:                                    # root-owned copies, readable by the sandbox user
            self.runner = runner_path or "/opt/ooma-sandbox/sandbox_main.py"
            self.python = python or "/usr/bin/python3"
        else:
            self.runner = runner_path or str(HERE / "sandbox_main.py")
            self.python = python or sys.executable

    def run(self, source: str, arrays: dict, *, start: int = LOOKBACK, mode: str = "all") -> SandboxResult:
        buf = io.BytesIO()
        np.savez(buf, **{k: np.asarray(v, dtype=float) for k, v in arrays.items()})
        npz, script = buf.getvalue(), source.encode()
        header = json.dumps({"script_len": len(script), "npz_len": len(npz), "start": start,
                             "lookback": LOOKBACK, "mode": mode,
                             "cpu_s": self.cpu, "wall_s": max(10, self.timeout - 5)}).encode() + b"\n"
        cmd = [self.python, "-I", self.runner]
        if self.user:
            cmd = ["sudo", "-n", "-u", self.user, *cmd]
        env = {"PATH": "/usr/bin:/bin", "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1", "PYTHONHASHSEED": "0"}
        try:
            p = subprocess.run(cmd, input=header + script + npz, capture_output=True, timeout=self.timeout, env=env)
        except subprocess.TimeoutExpired:
            return SandboxResult(False, error=f"timeout after {self.timeout}s")
        out = p.stdout.decode(errors="replace").strip().splitlines()
        if not out:
            tail = p.stderr.decode(errors="replace")[-300:]
            return SandboxResult(False, error=f"no output (exit {p.returncode}): {tail}")
        try:
            r = json.loads(out[-1])
        except json.JSONDecodeError:
            return SandboxResult(False, error="unparseable sandbox output")
        if not r.get("ok"):
            return SandboxResult(False, error=r.get("error", "failed"))
        return SandboxResult(True, np.array([c == "1" for c in r["dir"]]), np.array([c == "1" for c in r["act"]]), r["start"])
