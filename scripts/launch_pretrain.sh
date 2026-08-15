#!/usr/bin/env bash
# 启动 pretrain 看门狗：从 step-1000 续训至 5.5B tokens（2×4090）。
# 训练异常退出会自动用最新 checkpoint 重启，最多 5 次。
set -euo pipefail
cd ~/project/LocalSight
mkdir -p artifacts
nohup .venv/bin/python scripts/pretrain_watchdog.py \
  --torchrun .venv/bin/torchrun \
  --config configs/pretrain.yaml \
  --data-dir data/processed/pretrain-full \
  --base artifacts/pretrain/step-1000 \
  --max-total-tokens 5500000000 \
  --max-retries 5 \
  --log artifacts/pretrain_resume.log \
  >> artifacts/launch.out 2>&1 &
echo "watchdog 已启动，日志：artifacts/pretrain_resume.log / artifacts/watchdog.log"
