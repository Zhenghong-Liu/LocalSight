# 06 · 基础设施：服务器、git 与同步

## 1. 服务器现状（2026-08-13 实测）

| 项 | 值 |
| --- | --- |
| 主机 | `ETDataServer10`，Ubuntu 22.04，kernel 6.8 |
| GPU | 2× RTX 4090（24GB/卡），驱动 580.173.02 |
| CUDA | 系统 nvcc 11.5（不用它）；PyTorch 自带 cu12x 运行时 |
| 磁盘 | 根盘 1.8T（剩 314G）；数据盘 15T 挂 `/media/liuzh/data`（剩 4.5T） |
| 内存 | 系统内存充足（/dev/shm 31G） |
| conda | miniconda3，已有 `kdl`（torch 2.5.1+cu124）——**不动它** |
| 网络 | 可访问 GitHub |
| 数据 | `/media/liuzh/data/DLData/LocalSight`（只读） |

## 2. 服务器目标布局

```
/home/sodastar/project/LocalSight/
├── (git 仓库内容)
├── data/
│   ├── tokenizer/          # 从数据目录复制的快照
│   └── processed/          # 派生数据
└── artifacts/              # checkpoints/log（gitignore）
```

数据不用复制：训练直接读 `/media/liuzh/data/DLData/LocalSight`；只把 tokenizer 快照复制进项目，保证仓库与数据解耦。

## 3. 服务器环境搭建

```bash
cd ~/project/LocalSight
conda create -n localsight python=3.11 -y
conda activate localsight
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install transformers tokenizers datasets accelerate trl safetensors pyyaml numpy wandb
pip install flash-attn --no-build-isolation   # 按 cu128 选 wheel；失败则先用 SDPA 后端
```

验证：

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.device_count())"
torchrun --nproc_per_node=2 scripts/smoke_test.py
```

注：若 PyTorch 官方源下载慢，可用清华镜像的 torch 源或 conda 源；flash-attn 编译失败不阻塞——SDPA 后端功能等价（只是速度略低）。

## 4. git 工作流（本地 mac ↔ GitHub ↔ 服务器）

- 远程：`https://github.com/Zhenghong-Liu/LocalSight.git`（main 分支）。
- 本地 mac 已 `git remote add origin`；日常：在 `codex/*` 分支开发 → 合并 main → push。
- 服务器只做「拉取 + 训练」，**不在服务器上改代码提交**（紧急修复走 PR 回流，避免双写冲突）：

```bash
cd ~/project/LocalSight && git pull --ff-only
```

- 首次服务器克隆：

```bash
cd ~/project && git clone https://github.com/Zhenghong-Liu/LocalSight.git
```

## 5. 本地 ↔ 服务器同步（非 git 文件）

数据、checkpoint、GGUF 不进 git。需要时用 rsync：

```bash
# 本地 → 服务器（例如转换脚本）
rsync -av --progress scripts/ sodastar@119.78.227.152:~/project/LocalSight/scripts/
# 服务器 → 本地（例如下载 GGUF）
rsync -av sodastar@119.78.227.152:~/project/LocalSight/artifacts/release/ ./artifacts/release/
```

规则：只同步脚本与最终产物；中间 checkpoint 不下载到本地。

## 6. 训练运行方式

- 用 `tmux` 挂后台（断连不中断）：

```bash
tmux new -s pretrain
conda activate localsight
torchrun --nproc_per_node=2 src/localsight/training/pretrain.py --config configs/pretrain.yaml
```

- 检查：`tmux attach -t pretrain`；`nvidia-smi`；`watch -n 2 nvidia-smi`。
- 与他人共用机器时，开跑前先看 `nvidia-smi` 显存占用，避免抢占。

## 7. 磁盘预算

| 内容 | 位置 | 预估 |
| --- | --- | --- |
| 仓库 + 派生数据 | ~/project/LocalSight | < 5 GB |
| checkpoint（每阶段 ×3） | artifacts | ~2 GB |
| RLAIF 采样/打分缓存 | artifacts/rlaif | 5–20 GB |
| GGUF 发布 | artifacts/release | < 2 GB |

根盘剩 314G，完全够用；不要写数据盘（无写权限，也避免污染）。

## 8. 安全约定

- **密码/密钥绝不入库**；SSH 密码只用于交互登录。
- 建议尽快把服务器换成 SSH 公钥登录（`ssh-copy-id`），并在服务器 `/etc/ssh/sshd_config` 关密码登录前先确认公钥可用（需 root，可稍后处理）。
- git 推送使用个人 token/SSH 凭据，不在仓库里存 token。
- 训练脚本不访问与项目无关的服务器数据。
