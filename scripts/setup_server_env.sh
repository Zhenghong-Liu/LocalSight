#!/usr/bin/env bash
# LocalSight 训练环境安装脚本（在服务器上运行，幂等）。
# 用法：bash scripts/setup_server_env.sh（建议放进 tmux，日志见 artifacts/env_setup.log）
set -euo pipefail

cd ~/project/LocalSight
CONDA="$HOME/miniconda3/bin/conda"
ENV_PY="$HOME/miniconda3/envs/localsight/bin/python"
mkdir -p artifacts
exec > >(tee -a artifacts/env_setup.log) 2>&1

echo "=== [1/6] 创建 conda 环境 localsight (python 3.11) ==="
"$CONDA" create -n localsight python=3.11 -y

echo "=== [2/6] 升级 pip/setuptools ==="
"$ENV_PY" -m pip install --upgrade pip wheel setuptools ninja

echo "=== [3/6] 安装 PyTorch (cu128) ==="
if ! "$ENV_PY" -m pip install torch --index-url https://download.pytorch.org/whl/cu128; then
    echo "官方 PyTorch 源失败，尝试阿里云镜像"
    "$ENV_PY" -m pip install torch --index-url https://mirrors.aliyun.com/pytorch-wheels/cu128/
fi

echo "=== [4/6] 安装训练生态依赖（走 TUNA 镜像） ==="
"$ENV_PY" -m pip install \
    transformers tokenizers datasets accelerate trl \
    safetensors pyyaml numpy wandb pytest ruff

echo "=== [5/6] 安装 flash-attn（失败则用 SDPA 后端，不阻断） ==="
if "$ENV_PY" -m pip install flash-attn --no-build-isolation; then
    echo "flash-attn OK"
else
    echo "flash-attn 安装失败：回退到 SDPA 后端（功能等价，后续可再尝试）"
fi

echo "=== [6/6] 环境验证 ==="
"$ENV_PY" -m pip freeze > artifacts/env.lock.txt
"$ENV_PY" -c '
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
