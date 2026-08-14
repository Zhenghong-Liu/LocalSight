#!/usr/bin/env python3
"""RLAIF 第二轮编排：从 round1 权重采样 → judge → SimPO → round2。"""
from __future__ import annotations

import datetime
import os
import subprocess
from pathlib import Path

B = Path("/home/sodastar/project/LocalSight")
LOG = B / "artifacts" / "rlaif_round2.log"
ENV = {
    **os.environ,
    "PYTHONPATH": f"{B}/src",
    "NCCL_P2P_DISABLE": "1",
    "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
}
JUDGE = str(B / "models/judge/Qwen--Qwen2.5-7B-Instruct")


def log(msg: str) -> None:
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(f"[{datetime.datetime.now():%H:%M:%S}] {msg}\n")


def run(cmd: list[str], name: str) -> None:
    log(f"=== 开始 {name} ===")
    result = subprocess.run(cmd, cwd=B, env=ENV, start_new_session=True)
    log(f"=== 结束 {name}: exit={result.returncode} ===")


def main() -> None:
    log("RLAIF 第二轮开始")
    run(
        [str(B / ".venv/bin/torchrun"), "--nproc_per_node=2",
         str(B / "scripts/rlaif_sample.py"),
         "--checkpoint", str(B / "artifacts/rlaif_round1"),
         "--data-dir", str(B / "data/processed/rlaif_prompts"),
         "--out", str(B / "artifacts/rlaif2"),
         "--tokenizer", str(B / "data/tokenizer"),
         "--k", "2", "--max-new", "256", "--limit", "8000"],
        "第二轮采样",
    )
    run(
        [str(B / ".venv/bin/python"), str(B / "scripts/rlaif_judge.py"),
         "--samples", str(B / "artifacts/rlaif2"),
         "--questions", str(B / "data/processed/rlaif_prompts/questions.jsonl"),
         "--judge-model", JUDGE,
         "--out", str(B / "artifacts/rlaif2/scored.jsonl")],
        "第二轮 judge",
    )
    run(
        [str(B / ".venv/bin/torchrun"), "--nproc_per_node=2",
         str(B / "scripts/rlaif_train.py"),
         "--scored", str(B / "artifacts/rlaif2/scored.jsonl"),
         "--prompts-dir", str(B / "data/processed/rlaif_prompts"),
         "--start-checkpoint", str(B / "artifacts/rlaif_round1"),
         "--output-dir", str(B / "artifacts/rlaif_round2"),
         "--epochs", "1"],
        "第二轮 SimPO",
    )
    log("RLAIF 第二轮完成")


if __name__ == "__main__":
    main()
