#!/usr/bin/env python3
"""按用户截止要求提前收尾：epoch1 后停 pretrain → model soup → SFT → SimPO。

看门狗在服务器上以 setsid nohup 运行，不依赖 SSH 会话存活。
"""
from __future__ import annotations

import datetime
import os
import subprocess
import time
from pathlib import Path

B = Path("/home/sodastar/project/LocalSight")
LOG = B / "artifacts" / "early_finish.log"
ENV = {
    **os.environ,
    "PYTHONPATH": f"{B}/src",
    "NCCL_P2P_DISABLE": "1",
    "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
}


def log(msg: str) -> None:
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(f"[{datetime.datetime.now():%H:%M:%S}] {msg}\n")


def run(cmd: list[str], name: str) -> None:
    log(f"=== 开始 {name}: {' '.join(cmd[:4])}... ===")
    result = subprocess.run(cmd, cwd=B, env=ENV, start_new_session=True)
    log(f"=== 结束 {name}: exit={result.returncode} ===")


def main() -> None:
    deadline = datetime.datetime(2026, 8, 14, 4, 0)  # 服务器本地时间（CST）
    log("看门狗启动，等待 04:00")
    while datetime.datetime.now() < deadline:
        time.sleep(20)

    # 1) 停止 pretrain（此时已过 epoch1，checkpoint 1000/2000 已保存）
    subprocess.run(["pkill", "-TERM", "-f", "training/pretrain.py"])
    time.sleep(10)

    # 2) model soup（最后 3 个 checkpoint）
    run(
        [
            str(B / ".venv/bin/python"),
            "-c",
            "from pathlib import Path; from localsight.training.pretrain import soup_checkpoints; "
            "soup_checkpoints(Path('artifacts/pretrain'), Path('artifacts/pretrain/soup'), keep=3); "
            "print('SOUP_OK')",
        ],
        "model soup",
    )

    # 3) 保存训练指标摘要
    try:
        lines = (B / "artifacts/pretrain.log").read_text(errors="replace").splitlines()
        steps = [line for line in lines if "step=" in line][-8:]
        (B / "artifacts/pretrain/soup/summary.txt").write_text("\n".join(steps), encoding="utf-8")
    except Exception as exc:  # noqa: BLE001
        log(f"summary 失败: {exc}")

    # 4) SFT（2 epochs，bf16，NEFTune，packing）
    run(
        [
            str(B / ".venv/bin/torchrun"),
            "--nproc_per_node=2",
            "src/localsight/training/sft.py",
            "--config", "configs/sft.yaml",
            "--data-dir", "data/processed/sft",
            "--output-dir", "artifacts/sft",
            "--start-checkpoint", "artifacts/pretrain/soup",
        ],
        "SFT",
    )

    # 5) SimPO
    run(
        [
            str(B / ".venv/bin/torchrun"),
            "--nproc_per_node=2",
            "src/localsight/rl/simpo.py",
            "--config", "configs/dpo_simpo.yaml",
            "--data-dir", "data/processed/dpo",
            "--output-dir", "artifacts/dpo",
            "--start-checkpoint", "artifacts/sft/final",
        ],
        "SimPO",
    )
    log("=== ALL_DONE ===")


if __name__ == "__main__":
    main()
