#!/usr/bin/env python3
"""agent_rl 完成后自动跑评测（思考开/关两组）。"""
from __future__ import annotations

import datetime
import os
import subprocess
import time
from pathlib import Path

B = Path("/home/sodastar/project/LocalSight")
LOG = B / "artifacts" / "eval_chain.log"
ENV = {**os.environ, "PYTHONPATH": f"{B}/src"}


def log(msg: str) -> None:
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(f"[{datetime.datetime.now():%H:%M:%S}] {msg}\n")


def main() -> None:
    target = B / "artifacts" / "agent_rl" / "model.pt"
    deadline = datetime.datetime(2026, 8, 15, 12, 0)
    log("等待 agent_rl")
    while datetime.datetime.now() < deadline and not target.exists():
        time.sleep(60)
    if not target.exists():
        log("超时，终止")
        return
    time.sleep(5)
    for thinking in (False, True):
        flag = "--open-thinking" if thinking else ""
        cmd = [
            str(B / ".venv/bin/python"), str(B / "scripts/run_evals.py"),
            "--checkpoint", str(B / "artifacts/agent_rl"),
            "--data-dir", str(B / "data/eval"),
            "--bench", "mmlu,ceval,gsm8k",
            "--tokenizer", str(B / "data/tokenizer"),
            "--limit", "100",
        ]
        if thinking:
            cmd.append("--open-thinking")
        log(f"评测 thinking={thinking}")
        result = subprocess.run(cmd, cwd=B, env=ENV, start_new_session=True)
        log(f"评测 thinking={thinking} exit={result.returncode}")
    log("评测完成")


if __name__ == "__main__":
    main()
