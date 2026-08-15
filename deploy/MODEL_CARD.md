# LocalSight-198M-MoE 模型卡

## 基本信息

- 架构：MiniMind-3-MoE 风格 decoder-only，198.4M 总参数 / 63.9M 激活，4 专家 top-1 路由。
- 词表：6400 ByteLevel BPE（内置 think / tool_call / 多模态特殊 token）。
- 训练硬件：2× RTX 4090。
- 训练阶段：pretrain（大语料原计划 5 epochs，2026-08-14 用户决定提前至约 0.5 epoch 收尾）→ SFT（思考/非思考/工具混合）→ SimPO → RLAIF×2 → Agent RL（GRPO+DAPO）。

## 能力与已知局限

- 中文为主的基础问答与轻量推理；支持 `<think>` 开关式思考与 6 种内置工具调用。
- 局限：参数量小、知识有限；预训练语料以合成/指令型文本为主；思考可能不稳定或过短/过长。

## 数据与许可

- 训练数据路径与统计见 docs/02 与各阶段 manifest；原始数据未随模型发布。
- 使用前请自行确认训练数据来源与许可；本模型仅限研究/原型用途。

## 评测

见 [docs/08_final_metrics.md](docs/08_final_metrics.md)：

- MMLU 17% / C-Eval 19% / GSM8K 1%（各 100 样本，思考开/关无差异）；
- NIAH 4k/8k/16k/32k 全 5/5；IFEval 宽松规则版 100/100；
- Agent held-out（400 条）：mean_reward 0.367、gt_hit 0.015。

> 注意：当前权重是「约 0.5 epoch 预训练」的提前收尾版本，绝对能力明显低于完整训练预期。
