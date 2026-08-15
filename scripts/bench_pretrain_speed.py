#!/usr/bin/env python3
"""轻量提速套餐基准：compile 开/关各 20 步，对比 tok/s；≥1.2× 才建议启用。

用法（服务器项目根目录）：
    .venv/bin/python scripts/bench_pretrain_speed.py
输出末尾打印对照表与结论。
"""
from __future__ import annotations

import os
import re
import subprocess
import sys


def run_variant(compile_on: bool) -> list[float]:
    cmd = [
        ".venv/bin/torchrun", "--nproc_per_node=2",
        "src/localsight/training/pretrain.py",
        "--config", "configs/pretrain.yaml",
        "--data-dir", "data/processed/pretrain-mini",
        "--max-steps", "20",
        "--eval-interval-sec", "0",
        "--no-milestone-bench",
    ]
    if compile_on:
        cmd.append("--compile")
    env = os.environ.copy()
    env["NCCL_P2P_DISABLE"] = "1"
    env["PYTHONPATH"] = "src"
    out = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=1800)
    rates = [float(m) for m in re.findall(r"tok/s=([0-9.]+)", out.stdout)]
    if out.returncode != 0 or not rates:
        print(f"variant compile={compile_on} FAILED rc={out.returncode}")
        print(out.stdout[-3000:])
        print(out.stderr[-2000:])
        sys.exit(1)
    return rates[-5:]


def main() -> None:
    base = run_variant(compile_on=False)
    comp = run_variant(compile_on=True)
    base_mean = sum(base) / len(base)
    comp_mean = sum(comp) / len(comp)
    ratio = comp_mean / base_mean
    print("=" * 60)
    print(f"baseline (no compile)      : {base_mean:7.0f} tok/s  (last5 {[round(x) for x in base]})")
    print(f"compile (no cudagraphs)    : {comp_mean:7.0f} tok/s  (last5 {[round(x) for x in comp]})")
    print(f"speedup                    : {ratio:.2f}x")
    print("建议:", "启用 compile" if ratio >= 1.2 else "回退基线（不启用 compile）")
    print("=" * 60)


if __name__ == "__main__":
    main()
