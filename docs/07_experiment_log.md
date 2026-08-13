# 07 · 实验记录模板

> 每次正式训练新建一条记录，拷贝本表并填写。开发期冒烟实验可简写。

## M1 · 服务器环境（2026-08-13）

- 环境：项目级 venv `.venv`（基于 kdl 的 python 3.11.15；conda 镜像带宽不足时的等价方案）。
- 关键版本：torch 2.13.0+cu130、transformers 5.15.0、datasets 5.0.1、trl 1.9.2、tokenizers 0.22。
- flash-attn 无 cu130 预编译轮子 → **SDPA 后端**（Stage 0 已验证可用）。
- `scripts/smoke_test.py` 通过：world=2、NCCL allreduce OK、2×RTX 4090 可见。
- 环境锁：服务器 `artifacts/env.lock.txt`。

## M2 · 模型核心代码与测试（2026-08-13）

- 单元测试：**47 个全部通过**（RMSNorm、RoPE+YaRN、GQA/SDPA、document mask、MoE 偏置负载均衡、KV cache/spawn、采样器、HF 包装）。
- 参数账实测：总参 198,416,640 / 激活 63,936,768（QK-Norm 是 96 维，文档已修正）。
- Stage 0 小批（8 步随机数据，2 卡）：loss 6.10、专家负载 ≈ [0.23, 0.29, 0.28, 0.21]、显存 4.47GiB、P2P=False、compile forward eager 10.7ms→9.8ms。
- Stage 0 生产 batch（32×4096×2，激活重计算）：吞吐 **135,773 tok/s**、MFU 24.5%（未编译训练环）、显存 17.03GiB。
- 结论：5 epochs ≈ 20.8B tokens，未编译约 42h；接入 torch.compile 训练环后预期 ~21–28h（后续验证）。

## M3 · 数据管线（2026-08-13，进行中）

- 已实现并单测通过：精确去重 + minhash LSH（numpy splitmix）、sequence packing（pretrain/SFT 两种）。
- `pretrain_t2t_mini.jsonl` 派生数据构建中（服务器 `data/processed/pretrain-mini/`）。

## Run 记录

- 阶段：
- 日期 / 服务器 git commit：
- 配置：`configs/xxx.yaml`（附运行时合并后的 config 路径）
- 数据：manifest hash：
- seed：
- 起点 checkpoint：
- 终点 checkpoint：

## 超参快照

| 项 | 值 |
| --- | --- |
| lr / wd / warmup / 调度 | |
| batch（tokens 或 seq） | |
| seq len / epochs | |
| 其他（β/γ/G/K/temp 等） | |

## 结果

| 指标 | 值 |
| --- | --- |
| 最终 loss / grad norm | |
| 吞吐 / MFU | |
| 专家负载（4 专家占比） | |
| 评测分数 | |
| 思考开/关增益 | |

## 结论与下一步

- 观察：
- 问题：
- 下一步：
