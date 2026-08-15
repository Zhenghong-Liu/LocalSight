# 05 · 评测与发布

## 1. 评测矩阵

| 维度 | 数据集/任务 | 说明 |
| --- | --- | --- |
| 中文知识 | C-Eval、CMMLU | 0-shot/5-shot，注意许可与题型 |
| 英文知识 | MMLU | 同上 |
| 数学推理 | GSM8K | 分「思考开/关」两组跑 |
| 代码 | HumanEval | pass@1 |
| 指令跟随 | IFEval | 严格 prompt/指令级 |
| 智能体/工具 | BFCL + 自建 held-out（agent_rl 切分） | 工具格式合法率 + gt 命中率 |
| 长文本 | Needle-in-a-Haystack @32k | 思考开/关 |
| MoE 健康 | 负载均衡、路由熵 | 工程指标 |

harness：`lighteval` 或 `opencompass` 跑标准基准；对话/思考/工具用自研 chat harness（保证与 Ollama 模板一致）。

## 2. 思考能力专项

- **收益**：数学/逻辑题「思考开 vs 关」准确率差值，目标为正且显著；
- **不过度思考**：闲聊集上思考长度分布，不应整段长篇推理；
- **质量抽检**：50–100 条思考链人工/LLM 复核（步骤合理性、无跳跃、结论由思考导出）；
- **开关一致性**：同一 prompt 在 `open_thinking=true/false` 下答案合理。

## 3. 各阶段闸门（软目标）

| 阶段后 | 检查 |
| --- | --- |
| Pretrain | 验证 loss 达标；专家负载健康；小样本续写通顺 |
| SFT | 指令跟随可对话；思考开/关增益为正；不过度思考 |
| SimPO | chosen 质量 > rejected（胜率 > 60%） |
| RLAIF | 思考质量评分较上一轮提升；长度不失控 |
| Agent RL | 工具格式率 > 95%；gt 命中率显著高于 SFT 基线 |

这些是相对目标，不设绝对分数线（198M 模型的绝对分有限）。

## 4. 发布流水线

### 4.1 训练产物

最终权重：Stage 5 checkpoint（safetensors bf16）+ `config.json` + tokenizer 快照。先内部验证，再打 tag。

### 4.2 GGUF 转换（llama.cpp）

1. 写权重映射脚本（Qwen3-MoE 命名 → llama.cpp MoE 张量命名）；
2. 转换 + 量化：`Q8_0`（默认推荐）与 `Q4_K_M`（低内存端）；
3. 冒烟：perplexity 与几个样例输出对比 bf16 原版。

### 4.3 Ollama

`deploy/ollama/Modelfile.template` 已提供：使用官方 `RENDERER qwen3.5`（与 Qwen3 的
`<|im_start|>` + `<think>` 格式一致），原生支持思考开关——API 传 `think: true/false`，
CLI 用 `/think` 切换。发布命令：

```bash
OLLAMA_HOST=127.0.0.1:11435 ollama create localsight-198m -f deploy/ollama/Modelfile
OLLAMA_HOST=127.0.0.1:11435 ollama run localsight-198m
```

> 注意：服务器系统自带的 Ollama 0.23.2（默认端口 11434）的 qwen2moe runner 与
> QK-Norm/无共享专家结构不兼容，必须使用 ≥0.32 的用户态二进制（`~/ollama/bin/ollama`）
> 和 11435 端口；实测 `think:false` 直接输出答案、`think:true` 返回 thinking 字段。

### 4.4 Model Card

发布时写清：架构、训练阶段与数据构成、评测结果、已知局限（小模型知识有限、语料为合成为主、思考可能不稳定）、使用许可。

## 5. 安全与合规清单

- 无个人信息/恶意内容训练数据残留的声明与抽查；
- 不发布原始数据；
- 明确「研究用途、风险自负」边界；
- Ollama 本地运行场景的性能报告（Q4_K_M 在 4090/CPU 上的 tok/s）。
