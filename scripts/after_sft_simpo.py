#!/usr/bin/env python3
"""SFT 完成后自动启动 SimPO（看门狗，setsid 运行）。"""
from __future__ import annotations

import datetime
import os
import subprocess
import time
from pathlib import Path

B = Path("/home/sodastar/project/LocalSight")
LOG = B / "artifacts" / "chain.log"
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
    log("衔接看门狗启动，等待 SFT 完成")
    final = B / "artifacts" / "sft" / "final" / "model.pt"
    deadline = datetime.datetime(2026, 8, 14, 14, 0)
    while datetime.datetime.now() < deadline:
        if final.exists():
            break
        time.sleep(60)
    if not final.exists():
        log("超时：SFT final 未出现，终止")
        return
    time.sleep(10)
    log("SFT 完成，启动 SimPO")
    subprocess.run(
        [
            str(B / ".venv/bin/torchrun"),
            "--nproc_per_node=2",
            str(B / "src/localsight/rl/simpo.py"),
            "--config", str(B / "configs/dpo_simpo.yaml"),
            "--data-dir", str(B / "data/processed/dpo"),
            "--output-dir", str(B / "artifacts/dpo"),
            "--start-checkpoint", str(B / "artifacts/sft/final"),
        ],
        cwd=B,
        env=ENV,
        start_new_session=True,
    )
    log("SimPO 启动完成")


if __name__ == "__main__":
    main()
