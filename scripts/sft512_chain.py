#!/usr/bin/env python3
"""无人值守 SFT 接力（2026-08-16 用户指令）：round1 完成后无条件接 sft_512 训练。

流程：
  1. 等待 artifacts/sft_v2/final 出现（round1 由正在运行的进程完成，本脚本只等）；
  2. 跑一次 round1 质量报告（信息性，不设门槛）；
  3. 构建 data/processed/sft512（若不存在）→ SFT 训练 1 epoch（--resume 可断点续训）；
  4. round2 完成后跑质量报告 → 写 SFT_CHAIN_DONE → 停止等用户指令。

用 setsid/nohup 后台运行，SSH 断连不影响。
"""
from __future__ import annotations

import datetime
import json
import os
import subprocess
import time
from pathlib import Path

B = Path("/home/sodastar/project/LocalSight")
LOG = B / "artifacts" / "sft512_chain.log"
DONE = B / "artifacts" / "SFT_CHAIN_DONE"
ENV = {
    **os.environ,
    "PYTHONPATH": f"{B}/src",
    "NCCL_P2P_DISABLE": "1",
    "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
}

SFT512_SRC = "/media/liuzh/data/DLData/minimind/sft_512.jsonl"
ROUND1_FINAL = B / "artifacts/sft_v2/final/pytorch_model.bin"
ROUND2_DIR = B / "artifacts/sft_v2_512"
ROUND2_FINAL = ROUND2_DIR / "final/pytorch_model.bin"
SFT512_DATA = B / "data/processed/sft512"


def log(msg: str) -> None:
    line = f"[{datetime.datetime.now():%H:%M:%S}] {msg}"
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")
    print(line, flush=True)


def run(cmd: list[str], log_path: Path) -> int:
    log(f"run: {' '.join(str(x) for x in cmd[:6])}... -> {log_path.name}")
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(f"\n===== {datetime.datetime.now():%F %T} =====\n")
        f.flush()
        proc = subprocess.run(cmd, cwd=B, env=ENV, stdout=f, stderr=subprocess.STDOUT)
    log(f"run 结束 rc={proc.returncode}")
    return proc.returncode


def wait_round1(timeout_hours: float = 8.0) -> bool:
    deadline = datetime.datetime.now() + datetime.timedelta(hours=timeout_hours)
    while datetime.datetime.now() < deadline:
        if ROUND1_FINAL.exists():
            log("round1 final 已出现")
            return True
        time.sleep(60)
    log("等待 round1 超时")
    return ROUND1_FINAL.exists()


def quality(checkpoint: Path, out_dir: Path) -> None:
    log(f"质量报告（信息性）：{checkpoint}")
    run(
        [
            str(B / ".venv/bin/python"),
            str(B / "scripts/sft_quality_check.py"),
            "--checkpoint", str(checkpoint),
            "--judge-model", str(B / "models/judge/Qwen--Qwen2.5-7B-Instruct"),
            "--out-dir", str(out_dir),
        ],
        B / "artifacts" / "sft_quality.log",
    )


def run_round2() -> bool:
    if not (SFT512_DATA / "manifest.json").exists():
        log(f"构建 sft512 数据（{SFT512_SRC}）")
        run(
            [
                str(B / ".venv/bin/python"),
                str(B / "scripts/build_chat_data.py"),
                "--src", SFT512_SRC,
                "--out", str(SFT512_DATA),
                "--tokenizer", str(B / "data/tokenizer"),
            ],
            B / "artifacts" / "build_sft512.log",
        )
    else:
        log("sft512 数据已存在，跳过构建")

    for attempt in range(3):
        log(f"启动 round2（sft512，1 epoch，第 {attempt + 1}/3 次）")
        run(
            [
                str(B / ".venv/bin/torchrun"), "--nproc_per_node=2",
                str(B / "src/localsight/training/sft.py"),
                "--config", str(B / "configs/sft.yaml"),
                "--data-dir", str(SFT512_DATA),
                "--output-dir", str(ROUND2_DIR),
                "--start-checkpoint", str(B / "artifacts/sft_v2/final"),
                "--epochs", "1",
                "--sample-interval", "100",
                "--sample-prompts", str(B / "data/eval/thinking_prompts.txt"),
                "--sample-out", str(B / "artifacts/sft_samples_round2"),
                "--sample-limit", "25",
                "--resume",
            ],
            B / "artifacts" / "sft512.log",
        )
        if ROUND2_FINAL.exists():
            log("round2 完成")
            return True
        log(f"round2 第 {attempt + 1} 次未完成，300s 后 --resume 重试")
        time.sleep(300)
    return False


def main() -> None:
    log("sft512 接力链启动（无人值守，断网不影响）")
    if not wait_round1():
        DONE.write_text(json.dumps({"time": time.strftime("%F %T"), "error": "round1 未完成"}, ensure_ascii=False, indent=2))
        log("round1 未完成，终止")
        return
    quality(B / "artifacts/sft_v2/final", B / "artifacts/sft_v2/quality")

    ok = run_round2()
    if ok:
        quality(ROUND2_FINAL.parent, ROUND2_DIR / "quality")
    summary = {
        "time": datetime.datetime.now().isoformat(timespec="seconds"),
        "rounds": ["round1_sft_mini", "round2_sft512"] if ok else ["round1_sft_mini"],
        "round2_done": ok,
        "waiting_for_user": True,
    }
    DONE.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    log("接力结束，等待用户指令（详见 SFT_CHAIN_DONE 与各 quality/gate.json）")


if __name__ == "__main__":
    main()
