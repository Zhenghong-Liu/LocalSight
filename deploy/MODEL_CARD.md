# LocalSight-198M-MoE 模型卡

## 基本信息

- 架构：MiniMind-3-MoE 风格 decoder-only，198.4M 总参数 / 63.9M 激活，4 专家 top-1 路由。
- 词表：6400 ByteLevel BPE（内置 think / tool_call / 多模态特殊 token）。
- 训练硬件：2× RTX 4090。
- 训练阶段：pretrain（大语料 5 epochs）→ SFT（思考/非思考/工具混合）→ SimPO → RLAIF×2 → Agent RL（GRPO+DAPO）。

## 能力与已知局限

- 中文为主的基础问答与轻量推理；支持 `<think>` 开关式思考与 6 种内置工具调用。
- 局限：参数量小、知识有限；预训练语料以合成/指令型文本为主；思考可能不稳定或过短/过长。

## 数据与许可

- 训练数据路径与统计见 docs/02 与各阶段 manifest；原始数据未随模型发布。
- 使用前请自行确认训练数据来源与许可；本模型仅限研究/原型用途。

## 评测

见 docs/05 与 docs/07 实验日志（最终评测表发布时更新）。
