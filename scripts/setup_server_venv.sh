#!/usr/bin/env bash
# LocalSight 训练环境安装脚本（venv 方案，服务器上运行，幂等）。
#
# 背景：conda 镜像（TUNA pkgs/main）当前带宽过低且经常超时，改用已有的
# kdl python 3.11 创建项目级 venv，包走阿里云镜像。功能与 conda env 等价。
set -euo pipefail

cd ~/project/LocalSight
PY311="$HOME/miniconda3/envs/kdl/bin/python"
VENV="$HOME/project/LocalSight/.venv"
PIP="$VENV/bin/pip"
PYPI="https://mirrors.aliyun.com/pypi/simple/"
mkdir -p artifacts
exec > >(tee -a artifacts/env_setup.log) 2>&1

echo "=== [1/4] 创建 venv（基于 kdl 的 python 3.11） ==="
if [ ! -x "$VENV/bin/python" ]; then
    "$PY311" -m venv "$VENV"
fi

echo "=== [2/4] 安装 PyTorch cu128（阿里云 pytorch-wheels + pypi 依赖） ==="
"$PIP" install --upgrade pip wheel setuptools ninja -i "$PYPI"
"$PIP" install torch \
    --index-url https://mirrors.aliyun.com/pytorch-wheels/cu128/ \
    --extra-index-url "$PYPI"

echo "=== [3/4] 安装训练生态依赖 ==="
"$PIP" install \
    transformers tokenizers datasets accelerate trl \
    safetensors pyyaml numpy wandb pytest ruff -i "$PYPI"

echo "=== [4/4] 安装 flash-attn（失败则回退 SDPA） ==="
if "$PIP" install flash-attn --no-build-isolation -i "$PYPI"; then
    echo "flash-attn OK"
else
    echo "flash-attn 安装失败：回退到 SDPA 后端（功能等价）"
fi

echo "=== 环境验证 ==="
"$PIP" freeze > artifacts/env.lock.txt
"$VENV/bin/python" -c '
import torch, transformers
print("torch:", torch.__version__, "| cuda:", torch.version.cuda,
      "| available:", torch.cuda.is_available(),
      "| gpus:", torch.cuda.device_count())
print("transformers:", transformers.__version__)
try:
    import flash_attn
    print("flash_attn:", flash_attn.__version__)
except Exception:
    print("flash_attn: 未安装（SDPA 后端）")
'
echo "=== DONE ==="
