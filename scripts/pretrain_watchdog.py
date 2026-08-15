#!/usr/bin/env python3
"""pretrain 看门狗：训练异常退出后自动用最新 checkpoint 续训（默认最多 5 次）。

用法（在服务器项目根目录）：
    .venv/bin/python scripts/pretrain_watchdog.py \
        --torchrun .venv/bin/torchrun --config configs/pretrain.yaml \
        --data-dir data/processed/pretrain-full \
        --base artifacts/pretrain/step-1000 --max-retries 5 \
        --log artifacts/pretrain_resume.log
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def latest_step(ckpt_root: Path, min_step: int) -> Path | None:
    best: Path | None = None
    best_step = min_step
    for state_file in ckpt_root.glob("step-*/state.json"):
        try:
            step = int(json.loads(state_file.read_text())["step"])
        except (ValueError, KeyError):
            continue
        if step >= best_step and (state_file.parent / "model.pt").exists():
            best = state_file.parent
            best_step = step
    return best


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--torchrun", default=".venv/bin/torchrun")
    parser.add_argument("--nproc", type=int, default=2)
    parser.add_argument("--config", default="configs/pretrain.yaml")
    parser.add_argument("--data-dir", default="data/processed/pretrain-full")
    parser.add_argument("--base", default="artifacts/pretrain/step-1000")
    parser.add_argument("--max-total-tokens", type=int, default=5_500_000_000)
    parser.add_argument("--max-retries", type=int, default=5)
    parser.add_argument("--log", default="artifacts/pretrain_resume.log")
    parser.add_argument("--watchdog-log", default="artifacts/watchdog.log")
    args = parser.parse_args()

    env = os.environ.copy()
    env["NCCL_P2P_DISABLE"] = "1"
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    env["PYTHONPATH"] = "src"

    retries = 0
    while True:
        ckpt = latest_step(Path(args.base).parent, int(Path(args.base).name.split("-")[1]))
        resume = str(ckpt) if ckpt is not None else args.base
        cmd = [
            args.torchrun, f"--nproc_per_node={args.nproc}",
            "src/localsight/training/pretrain.py",
            "--config", args.config,
            "--data-dir", args.data_dir,
            "--resume", resume,
            "--max-total-tokens", str(args.max_total_tokens),
        ]
        with open(args.log, "a", encoding="utf-8") as log_file:
            log_file.write(f"\n===== watchdog launch {time.strftime('%F %T')} resume={resume} =====\n")
            log_file.flush()
            proc = subprocess.run(cmd, env=env, stdout=log_file, stderr=subprocess.STDOUT)
        segment = Path(args.log).read_text(encoding="utf-8")
        done = "PRETRAIN_DONE" in segment
        with open(args.watchdog_log, "a", encoding="utf-8") as wf:
            wf.write(json.dumps(
                {"time": time.strftime("%F %T"), "resume": resume, "returncode": proc.returncode,
                 "done": done, "retries": retries}, ensure_ascii=False,
            ) + "\n")
        if done:
            print(f"watchdog: PRETRAIN_DONE (resume={resume})", flush=True)
            final_eval = Path("scripts/pretrain_final_eval.sh")
            if final_eval.exists():
                print("watchdog: 运行收尾评测（MMLU/C-Eval/GSM8K/NIAH/IFEval）", flush=True)
                subprocess.run(["bash", str(final_eval)], env=env, check=False)
            return
        retries += 1
        if retries > args.max_retries:
            print(f"watchdog: 达到重试上限 {args.max_retries}，停止", flush=True)
            sys.exit(1)
        print(f"watchdog: 第 {retries} 次退出（rc={proc.returncode}），30s 后用 {resume} 重试", flush=True)
        time.sleep(30)


if __name__ == "__main__":
    main()
