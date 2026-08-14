#!/usr/bin/env python3
"""等待 agent_rl 完成后自动跑发布管线。"""
from __future__ import annotations

import datetime
import os
import subprocess
import time
from pathlib import Path

B = Path("/home/sodastar/project/LocalSight")
LOG = B / "artifacts" / "release_chain.log"


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
    result = subprocess.run(
        [str(B / ".venv/bin/python"), str(B / "scripts/release_pipeline.py")],
        cwd=B, env={**os.environ, "PYTHONPATH": f"{B}/src"},
        start_new_session=True,
    )
    log(f"发布管线结束 exit={result.returncode}")


if __name__ == "__main__":
    main()
