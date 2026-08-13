# 02 · 数据规格（基于 2026-08-13 实测）

## 1. 位置与权限

- 训练机只读目录：`/media/liuzh/data/DLData/LocalSight`（15T 数据盘，当前账号 sodastar 可读）。
- **规则**：原始数据一律只读、不移动、不修改。清洗/打包后的派生数据写入 `~/project/LocalSight/data/processed/<stage>/`。

## 2. 实测统计

采样方法：全量 `wc -l` + 随机字节偏移采样（pretrain 每文件 20k 条；其余文件全量解析）。

| 文件 | 大小 | 行数 | 顶层字段 | 平均字符 | 实测 tokens/char | 估计 tokens |
| --- | --- | --- | --- | --- | --- | --- |
| pretrain_t2t.jsonl | 7.8 GB | 8,468,827 | `{text}` | 719.4（字节加权） | 0.6819 | **2,131,183,459（实测）** |
| pretrain_t2t_mini.jsonl | 1.2 GB | 1,270,238 | `{text}` | 713.8（字节加权） | 0.6766 | **328,903,639（实测）** |
| sft_t2t_mini.jsonl | 1.7 GB | 905,718 | `{conversations}` | 540.0/条 | — | ~0.33B |
| dpo.jsonl | 52 MB | 17,166 | `{chosen, rejected}` | 2084.9/条 | — | ~25M |
| rlaif.jsonl | 23 MB | 19,502 | `{conversations}` | 402.4/条 | — | ~5M |
| agent_rl.jsonl | 79 MB | 39,988 | `{conversations, gt}` | 340.8/条 | — | ~9M |

关键结论：

- 全量 pretrain 实测 **2.13B tokens**，是**唯一主训语料**（按用户要求跑 5 epochs，约 10.7B tokens）；mini 实测 **0.33B**，只用于开发、冒烟测试与 LR 扫描。
- 注意：早期文档里的 4.15B/0.61B 是「字节加权采样」的估计值，被偏置放大了；以派生数据 manifest 的实测值为准。
- pretrain 文本以中文为主、且明显包含合成指令型文本（不是纯网页语料），需要按此设定预期并做质量过滤。

## 3. Tokenizer（沿用，不重训）

- 6400 词表 ByteLevel BPE（`merges=6108`，`added_tokens=36`）。
- 已内置特殊 token：`<|im_start|>`、`<|im_end|>`、`<|endoftext|>`、`<think>`、`</think>`、`<tool_call>`、`</tool_call>`、`<tool_response>`、`</tool_response>`、`<|object_ref_*|>`、`<|box_*|>`、`<|quad_*|>`、`<|vision_*|>`、`<|image_pad|>`、`<|video_pad|>`、`<|audio_*|>`、`<tts_*>`、`<|buffer1..9|>`。
- `tokenizer_config.json` 自带 Qwen 风格 `chat_template`：支持 `tools`、`reasoning_content`、`tool_calls`、`tool` 角色，assistant 输出**始终**被 `<think>...</think>` 包裹（无思考时为空 think 块），生成时可用 `open_thinking` 控制是否敞开思考。
- 规则：所有对话类数据**统一通过 `apply_chat_template` 序列化**，禁止手写 prompt 模板，保证训练与推理分布一致。

## 4. 各数据集 schema 与用途（按实测修正）

### 4.1 pretrain_t2t.jsonl / pretrain_t2t_mini.jsonl

```json
{"text": "..."}
```

- 用途：语言建模自回归预训练（`labels = input_ids`，全 token 算 loss）。
- 主语料：`pretrain_t2t.jsonl`；开发/冒烟：`pretrain_t2t_mini.jsonl`。
- 清洗规则：长度阈值（token 后 32–4096）、控制字符/乱码过滤、URL 规范化、minhash 去重（64 签名，阈值 0.8）。

### 4.2 sft_t2t_mini.jsonl

```json
{"conversations": [
  {"role": "system", "content": "", "tools": "[...OpenAI function JSON...]"},
  {"role": "user", "content": "..."},
  {"role": "assistant", "content": "...", "reasoning_content": "..."},
  {"role": "assistant", "content": "...", "tool_calls": "[...]"},
  {"role": "tool", "content": "..."}
]}
```

实测构成（905,718 条）：

- 34.0%（308,390 条）至少一条 assistant 带非空 `reasoning_content`；
- 9.4%（84,836 条）涉及工具：`tools` 8,258 条、`tool_calls` 76,574 条、`function_call` 3 条；
- **思考样本与工具样本不重叠**（同时含 reasoning 与 tool_calls 的为 0）；
- 角色分布：assistant 1,249,691 / user 1,172,003 / system 84,832 / tool 77,784。

用途：注入「思考 + 非思考 + 工具」三种行为。思考样本经模板变成 `<think>...</think>`；非思考样本保留空 think 块，保证开关语义。

### 4.3 dpo.jsonl

```json
{"chosen": [{"role":"user","content":"continue"},{"role":"assistant","content":"..."}],
 "rejected": [{"role":"user","content":"continue"},{"role":"assistant","content":"..."}]}
```

实测：17,166 对，全部是单轮续写偏好对，**chosen/rejected 均不含 `<think>`**。

用途修正：定位为「通用回答质量偏好」，而不是「思考质量偏好」。训练时同样过 chat_template（自然带空 think 块）；思考质量偏好由第 5 阶段 RLAIF 负责。

### 4.4 rlaif.jsonl

```json
{"conversations": [
  {"role":"user","content":"..."},
  {"role":"assistant","content":"..."},
  ...,
  {"role":"assistant","content":""}   // 末轮必为空
]}
```

实测：19,502 条，平均 4.6 轮（2–22 轮），**100% 末轮 assistant 为空串**。

用途修正：这是「补全式 prompt 集」，不是纯 prompt。RL 采样时 prompt = 去掉末轮空 assistant 后的前缀 + `open_thinking=true`。

### 4.5 agent_rl.jsonl

```json
{"conversations": [
  {"role":"system","content":"","tools":"[{\"type\":\"function\",\"function\":{...}}]"},
  {"role":"user","content":"..."},
  ...,
  {"role":"assistant","content":""}   // 末轮必为空
 ],
 "gt": ["14302730"]}                  // 20k 条非空；另 20k 条为空
```

实测：39,988 条，**100% 末轮 assistant 为空**；其中 20,000 条带 `tools` + 非空 `gt`，另 19,988 条无工具且 `gt` 为空。

工具只有 6 种（全部可在本地实现）：

| 工具 | 出现次数 | 实现 |
| --- | --- | --- |
| calculate_math | 9,947 | AST 安全求值 |
| get_exchange_rate | 9,850 | 静态汇率表 |
| get_current_weather | 9,847 | 城市→天气 stub |
| get_current_time | 9,804 | 系统时钟 |
| translate_text | 9,792 | 内置小词典/stub |
| unit_converter | 9,712 | 单位换算公式 |

用途修正：**没有可交互环境**。RL 阶段由我们实现上述 6 个工具执行器，把 `<tool_call>` 转成 `<tool_response>` 接回对话，用 `gt` 做结果奖励。v1 只用带 gt 的 2 万条做结果型 RL；无 gt 的 2 万条开放题暂不进入结果奖励（可作为 held-out 或后续 RLAIF 扩展）。

## 5. 处理管线规则

对每个阶段统一走「清洗 → 去重 → tokenize → 打包 → 缓存」：

1. **读取**：用 HuggingFace `datasets` 流式读取 JSONL（`load_dataset(..., streaming=True)`）；需要源文件 sha256 的严格复现场景可切回手工 jsonl 直读（`--backend jsonl`）。
2. **清洗**：控制字符、超长截断、空样本丢弃；保留统计到派生 manifest，便于溯源。
3. **去重**：精确（规范化文本 sha256）+ minhash LSH（64 签名，Jaccard ≥0.8）；指令数据用「规范化后字符串」精确去重 + prompt 级近似去重。
4. **tokenize**：用 `tokenizers`（Rust 后端）批量编码；一律用仓库内 tokenizer 快照（从数据目录复制进 `data/tokenizer/`），避免上游文件被改影响复现。
5. **打包**：
   - pretrain：sequence packing + document-aware attention mask，`<|endoftext|>` 分隔，目标 4096；
   - SFT/DPO：v1 不打包，按 batch 内最大长度 padding + loss mask；
   - RL：prompt 不打包。
6. **缓存**：处理后存 `data/processed/<stage>/*.bin`（`mmap` 索引格式），带 manifest（源文件、hash、过滤统计、seed）。

## 6. 派生数据目录

```
data/
├── tokenizer/            # tokenizer 快照（json + config + sha256）
├── processed/
│   ├── pretrain/         # .bin + manifest.json
│   ├── sft/
│   ├── dpo/
│   ├── rlaif/            # 采样的 prompt 前缀
│   └── agent_rl/         # 含工具 schema + gt
└── eval/                 # held-out 集
```

## 7. 随机性与版本化

- 每阶段使用**不同 seed**（pretrain 42 / sft 1337 / dpo 2025 / rlaif 2026 / agent_rl 31415），数据乱序 seed 也随阶段变化。
- 每个派生文件记录：源文件 sha256、行数、过滤后行数、去重移除数、生成命令。
- 评测集（held-out）在 M3 阶段一次性切分并冻结，之后任何阶段不得修改。

## 8. 数据伦理与许可

- 语料含合成指令文本与少量可能来自网络的内容；发布前需要自查来源与许可。
- 含个人信息/敏感内容的部分在清洗阶段按规则剔除；在 model card 中如实声明数据构成与局限。
