#!/usr/bin/env python3
"""等待 RLAIF round2 完成后自动启动 Agent RL（GRPO+DAPO）。"""
from __future__ import annotations

import datetime
import os
import subprocess
import time
from pathlib import Path

B = Path("/home/sodastar/project/LocalSight")
LOG = B / "artifacts" / "agent_chain.log"
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
    target = B / "artifacts" / "rlaif_round2" / "model.pt"
    deadline = datetime.datetime(2026, 8, 15, 6, 0)
    log("等待 rlaif_round2")
    while datetime.datetime.now() < deadline and not target.exists():
        time.sleep(60)
    if not target.exists():
        log("超时：rlaif_round2 未出现，终止")
        return
    time.sleep(5)
    log("启动 Agent RL")
    result = subprocess.run(
        [
            str(B / ".venv/bin/torchrun"), "--nproc_per_node=2",
            str(B / "src/localsight/rl/agent_grpo.py"),
            "--config", str(B / "configs/agent_grpo.yaml"),
            "--data-dir", str(B / "data/processed/agent_prompts"),
            "--output-dir", str(B / "artifacts/agent_rl"),
            "--tokenizer", str(B / "data/tokenizer"),
            "--start-checkpoint", str(B / "artifacts/rlaif_round2"),
            "--limit", "4000",
        ],
        cwd=B,
        env=ENV,
        start_new_session=True,
    )
    log(f"Agent RL 结束 exit={result.returncode}")


if __name__ == "__main__":
    main()
