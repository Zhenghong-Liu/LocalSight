# 08 · 最终评测与发布结果

> 模型：`localsight-198m-moe`（198,416,640 总参 / 63,936,768 激活参，4 专家 top-1、无共享专家）。
> 发布权重：`artifacts/agent_rl/model.pt`（bf16）。日期：2026-08-15。

## 一、训练过程摘要

| 阶段 | 数据规模 | 配置要点 | 产物 |
| --- | --- | --- | --- |
| Pretrain | 大语料 2.13B tokens（原计划 5 epochs） | Muon+AdamW、lr=3e-3、wd=0.1、seq 4096、有效 batch 1M tokens | `artifacts/pretrain/step-1000`（val_loss=1.6933） |
| SFT | 905,718 条 → 54,873 序列（449.5M tokens） | AdamW lr=1.5e-4、2 epochs、seq 8192、packing+document mask、NEFTune α=5 | `artifacts/sft/final` |
| SimPO | 17,166 对 | β=2.0、γ=1.2、lr=5e-6、1 epoch | `artifacts/dpo/model.pt` |
| RLAIF | 8k prompts × K=2 × 2 轮 | on-policy 采样 + 7B judge + SimPO（lr=3e-6） | `artifacts/rlaif_round1` / `rlaif_round2` |
| Agent RL | 4k prompts（含黄金 rollout） | GRPO+DAPO（ε_high=0.28、G=6、temp=0.9、lr=3e-6） | `artifacts/agent_rl/model.pt` |

> 与原规划差异（用户决定）：pretrain 由 5 epochs 提前至约 0.5 epoch 结束（NCCL 于 step ~1060 自行退出后，
> 用户 2026-08-14 07:40 确认收尾，保留 step-1000）。因此模型绝对能力明显低于完整训练预期。

## 二、最终评测表（agent_rl 权重）

| 基准 | 分数 | 规模 / 说明 |
| --- | --- | --- |
| MMLU | 17% | 100 样本（随机 4 选 1 ≈ 25%） |
| C-Eval | 19% | 100 样本 |
| GSM8K | 1% | 100 样本 |
| IFEval | 100/100 | 宽松规则版（开头/结尾/关键词/是-否），非严格 IFEval |
| NIAH @ 4k / 8k / 16k / 32k | 5/5、5/5、5/5、5/5 | 每长度 5 seed |
| Agent held-out | mean_reward=0.367、gt_hit=0.015 | 自建 400 条工具任务 |
| 思考开/关对比 | 得分相同 | MMLU/C-Eval/GSM8K 上思考开关无差异 |
| MoE 负载 | [0.269, 0.225, 0.240, 0.265] | pretrain 末期无坍塌 |

- Agent RL 训练期：mean_reward 0.21 → 0.60，gt_hit 0.167~0.208（组内含 1/6 黄金样本）。
- 未跑项：HumanEval、BFCL、CMMLU（下载或环境受限）。BFCL 以自建 held-out + gt 命中率替代，CMMLU 以 C-Eval 覆盖。

## 三、解读与已知局限

- 预训练仅覆盖约一半语料（step 1060/约 2131），绝对分偏低（MMLU/C-Eval 低于随机基线、GSM8K≈0）是预期结果，不是 bug。
- 思考能力没有带来可测增益：SFT 思考占比 34% 但预训练底座太弱，思考链本身难以承载有效推理。
- Agent 能力：格式奖励（工具调用格式合法）学到约 0.55~0.6，但真实任务完成率（gt_hit）接近 0。
- NIAH 全满与 IFEval 规则通过说明长上下文机制（RoPE/YaRN/QK-Norm）与格式跟随是健康的。
- 改进路径：完整跑满 5 epochs 预训练（约 31h）是性价比最高的一步，之后各阶段分数预期显著回升。

## 四、发布产物

| 产物 | 路径 / 名称 |
| --- | --- |
| GGUF f16（qwen3moe） | `artifacts/release/localsight.f16.gguf` |
| GGUF Q8_0 | `artifacts/release/localsight.Q8_0.gguf` |
| GGUF Q4_K_M | `artifacts/release/localsight.Q4_K_M.gguf` |
| Ollama 模型 | `localsight-198m`（用户态 0.32.13，`OLLAMA_HOST=127.0.0.1:11435`） |

Ollama 使用（服务器上）：

```bash
export OLLAMA_HOST=127.0.0.1:11435
~/ollama/bin/ollama run localsight-198m
```

> 系统自带的 Ollama 0.23.2（默认端口 11434）的 qwen2moe runner 与 QK-Norm/无共享专家结构不兼容，
> 必须使用上述用户态二进制与端口。

思考开关（官方 qwen3.5 renderer，已实测生效）：

- `think:false`：直接输出答案（无 thinking 字段）；
- `think:true`：先输出思考链（API 回复带 `thinking` 字段，CLI 用 `/think` 切换）。

## 五、数据构成与许可

- 语料：`/media/liuzh/data/DLData/LocalSight` 下的 pretrain/sft/dpo/rlaif/agent_rl 数据集（本仓库不含原始数据）。
- 词表：数据集自带 6400 词表 ByteLevel BPE（含 think/tool/vision/audio/TTS 特殊 token）。
- 用途：研究/本地使用；数据许可沿用原始数据集约束，发布前需按数据来源确认再许可。
