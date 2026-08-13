#!/usr/bin/env bash
# 启动全量 5-epoch pretrain（2×4090，重计算 + compile，无 NVLink）。
set -euo pipefail
cd ~/project/LocalSight
export NCCL_P2P_DISABLE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
exec env PYTHONPATH=src .venv/bin/torchrun --nproc_per_node=2 \
  src/localsight/training/pretrain.py \
  --config configs/pretrain.yaml \
  --data-dir data/processed/pretrain-full \
  --compile
