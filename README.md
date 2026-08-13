# LocalSight

从零训练一个**轻量、带思考能力的 MoE 语言模型**：`localsight-198m-moe`（总参数 198M，激活参数 64M），训练完成后转成 GGUF，通过 **Ollama** 本地运行。架构现在以文本为主，但按多模态预留扩展接口（词表已内置视觉/音频/TTS 特殊 token）。

> 当前状态：**规划阶段**。模型与训练规则已在本仓库冻结成文档与参考配置，代码实现与训练尚未开始。

## 一句话方案

`MiniMind-3-MoE 结构（Qwen3-MoE 命名兼容）+ Muon 预训练 + 混合思考 SFT + SimPO + 迭代式 RLAIF + GRPO/DAPO 工具智能体 RL`，训练顺序严格为：**pretrain → sft → dpo → rlaif → agent_rl**。

## 关键实测数字（2026-08-13 实测）

| 数据集 | 大小 | 记录数 | 结构 | 估计 tokens |
| --- | --- | --- | --- | --- |
| pretrain_t2t.jsonl | 7.8 GB | 8,468,827 | `{text}` | 2.13B（实测） |
| pretrain_t2t_mini.jsonl | 1.2 GB | 1,270,238 | `{text}` | 0.33B（实测） |
| sft_t2t_mini.jsonl | 1.7 GB | 905,718 | `{conversations}` | ~0.33B |
| dpo.jsonl | 52 MB | 17,166 | `{chosen, rejected}` | ~25M |
| rlaif.jsonl | 23 MB | 19,502 | `{conversations}`（末轮留空） | ~5M |
| agent_rl.jsonl | 79 MB | 39,988 | `{conversations, gt}`（末轮留空） | ~9M |

- Tokenizer：6400 词表 ByteLevel BPE，内置 `<think>`、`<tool_call>`、vision/audio/TTS 特殊 token。
- 预训练使用大语料（实测 2.13B tokens），按用户要求跑 **5 epochs**（约 10.7B tokens）；小语料只用于开发/冒烟与 LR 扫描。

## 文档导航

| 文档 | 内容 |
| --- | --- |
| [docs/00_project_overview.md](docs/00_project_overview.md) | 目标、原则、路线图、决策记录 |
| [docs/01_model_architecture.md](docs/01_model_architecture.md) | 模型架构规格（已按实测修正） |
| [docs/02_data_spec.md](docs/02_data_spec.md) | 数据实测结构、处理管线与质量规则 |
| [docs/03_training_plan.md](docs/03_training_plan.md) | 分阶段训练方案、超参与计算预算 |
| [docs/04_engineering_standards.md](docs/04_engineering_standards.md) | 编码规范、技术栈、目录结构 |
| [docs/05_evaluation_and_release.md](docs/05_evaluation_and_release.md) | 评测体系与发布/Ollama 流程 |
| [docs/06_infrastructure.md](docs/06_infrastructure.md) | 服务器、git 与本地↔服务器工作流 |
| [docs/07_experiment_log.md](docs/07_experiment_log.md) | 实验记录模板 |

## 目录结构（骨架）

```
LocalSight/
├── docs/            # 全部规划文档
├── configs/         # 模型与各阶段参考配置（YAML）
├── src/localsight/  # 模型、数据、训练、RL、评测代码（待实现）
├── scripts/         # 数据探查、同步、训练启动脚本
├── tests/           # 单元测试（待实现）
├── deploy/ollama/   # Ollama Modelfile 模板
└── artifacts/       # 训练产物（本地，gitignore）
```

## 环境

- 编码机：macOS（本仓库）。
- 训练机：`sodastar@119.78.227.152`，2× RTX 4090（24GB×2），Ubuntu 22.04，驱动 580.173。
- 数据（只读）：训练机 `/media/liuzh/data/DLData/LocalSight`。

## 开始

详见 [docs/06_infrastructure.md](docs/06_infrastructure.md)。规划阶段的下一步是：冻结规则 → 搭服务器 conda 环境 → 写模型核心代码与冒烟测试（Stage 0）。

如需把整个项目交给「目标模式」持续执行，使用 [GOAL_PROMPT.md](GOAL_PROMPT.md) 作为初始 prompt。
