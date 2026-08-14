#!/usr/bin/env python3
"""RLAIF 一轮编排：采样 → judge 打分 → SimPO 训练（顺序执行，setsid 运行）。"""
from __future__ import annotations

import datetime
import os
import subprocess
from pathlib import Path

B = Path("/home/sodastar/project/LocalSight")
LOG = B / "artifacts" / "rlaif_pipeline.log"
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
    log("RLAIF 一轮开始")
    run(
        [str(B / ".venv/bin/torchrun"), "--nproc_per_node=2",
         str(B / "scripts/rlaif_sample.py"),
         "--checkpoint", str(B / "artifacts/dpo"),
         "--data-dir", str(B / "data/processed/rlaif_prompts"),
         "--out", str(B / "artifacts/rlaif"),
         "--tokenizer", str(B / "data/tokenizer"),
         "--k", "4"],
        "采样",
    )
    run(
        [str(B / ".venv/bin/python"), str(B / "scripts/rlaif_judge.py"),
         "--samples", str(B / "artifacts/rlaif"),
         "--questions", str(B / "data/processed/rlaif_prompts/questions.jsonl"),
         "--judge-model", JUDGE,
         "--out", str(B / "artifacts/rlaif/scored.jsonl")],
        "judge 打分",
    )
    run(
        [str(B / ".venv/bin/torchrun"), "--nproc_per_node=2",
         str(B / "scripts/rlaif_train.py"),
         "--scored", str(B / "artifacts/rlaif/scored.jsonl"),
         "--prompts-dir", str(B / "data/processed/rlaif_prompts"),
         "--start-checkpoint", str(B / "artifacts/dpo"),
         "--output-dir", str(B / "artifacts/rlaif_round1"),
         "--epochs", "1"],
        "SimPO 训练",
    )
    log("RLAIF 一轮完成")


if __name__ == "__main__":
    main()
