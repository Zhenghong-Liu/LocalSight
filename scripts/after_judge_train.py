#!/usr/bin/env python3
"""等待 judge 打分完成（scored.jsonl 行数达标）后自动运行 RLAIF SimPO 训练。"""
from __future__ import annotations

import datetime
import os
import subprocess
import time
from pathlib import Path

B = Path("/home/sodastar/project/LocalSight")
LOG = B / "artifacts" / "after_judge.log"
ENV = {
    **os.environ,
    "PYTHONPATH": f"{B}/src",
    "NCCL_P2P_DISABLE": "1",
    "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
}


def log(msg: str) -> None:
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(f"[{datetime.datetime.now():%H:%M:%S}] {msg}\n")


def main() -> None:
    target = B / "artifacts" / "rlaif" / "scored.jsonl"
    deadline = datetime.datetime(2026, 8, 14, 23, 0)
    log("等待 judge 完成")
    while datetime.datetime.now() < deadline:
        try:
            lines = sum(1 for _ in open(target, encoding="utf-8"))
        except FileNotFoundError:
            lines = 0
        if lines >= 16000:
            break
        time.sleep(60)
    time.sleep(5)
    log(f"scored 行数={lines}，启动 RLAIF 训练")
    result = subprocess.run(
        [
            str(B / ".venv/bin/torchrun"), "--nproc_per_node=2",
            str(B / "scripts/rlaif_train.py"),
            "--scored", str(target),
            "--prompts-dir", str(B / "data/processed/rlaif_prompts"),
            "--start-checkpoint", str(B / "artifacts/dpo"),
            "--output-dir", str(B / "artifacts/rlaif_round1"),
            "--epochs", "1",
        ],
        cwd=B,
        env=ENV,
        start_new_session=True,
    )
    log(f"RLAIF 训练结束 exit={result.returncode}")


if __name__ == "__main__":
    main()
