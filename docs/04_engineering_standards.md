# 04 · 工程与编码规范

## 1. 自研 vs 调包（分工原则）

**自己写**（核心、可学习、需要细粒度控制）：

| 模块 | 备注 |
| --- | --- |
| RMSNorm / QK-Norm | 10 行级别，参考 HF 做数值对齐测试 |
| RoPE（含 YaRN 因子） | 预计算 cos/sin 缓存，支持 rotate_half |
| Attention 封装 + 因果/文档掩码 | 内部走 SDPA/FA2 后端 |
| KV Cache / Prefix Cache | 供 RL 采样复用 |
| MoE 路由 + 专家 FFN | 分桶批量 GEMM |
| 负载均衡（偏置更新 + z-loss） | DeepSeek-V3 风格 |
| Muon 优化器 | 按 Moonlight 公式；NS 用 torch 编译内核 |
| 损失（LM loss / SimPO / GRPO+DAPO） | |
| 采样器（temperature/top-p/重复惩罚） | |
| 工具执行器（6 种） | |
| 打包与 mmap 数据加载器 | |

**调包**（外围、成熟、没必要重造）：

| 模块 | 包 |
| --- | --- |
| Tokenizer | `tokenizers` / `transformers.AutoTokenizer` |
| 配置解析 | `yaml` + `dataclasses`（或 `omegaconf`） |
| 数据集流式加载 | `datasets` / `pyarrow` |
| SFT/SimPO 训练循环 | 可复用 `transformers.Trainer`（模型按 `PreTrainedModel` 包装后） |
| 分布式 | `torch.distributed` + `accelerate` |
| 日志 | `wandb`（或 `swanlab`/本地 jsonl 兜底） |
| 权重保存 | `safetensors` |
| 评测 | `lighteval`/`opencompass` + 自研 chat harness |
| RL 参考实现 | `trl`（只作交叉验证，不绑死） |

原则：**核心路径不隐藏魔法**；能一行解释的行为不引第三方库。

## 2. 技术栈（锁定版本范围）

- Python 3.11
- PyTorch ≥ 2.8（cu128 官方 wheel；服务器驱动 580 支持 CUDA 12.8+）
- flash-attn 2.x（对应 cu128 的 wheel 或源码构建）
- transformers ≥ 5.0、tokenizers、datasets、accelerate、trl（与 transformers 匹配的最新版）
- safetensors、pyyaml、numpy、tqdm、wandb
- 开发工具：ruff、pytest

服务器建独立 conda env：`localsight`（不动现有 `kdl`）。

## 3. 目录结构（完整骨架）

```
LocalSight/
├── README.md
├── LICENSE
├── pyproject.toml
├── .gitignore
├── docs/                      # 01–07 规划文档
├── configs/
│   ├── model/minimind3_moe_198m.yaml
│   ├── base.yaml              # 全局默认
│   ├── pretrain.yaml
│   ├── sft.yaml
│   ├── dpo_simpo.yaml
│   ├── rlaif.yaml
│   └── agent_grpo.yaml
├── src/localsight/
│   ├── __init__.py
│   ├── model/                 # norm/rope/attention/moe/transformer/kv_cache
│   │   └── multimodal/        # 预留：encoder/projector 接口
│   ├── tokenizer/             # tokenizer 快照加载与包装
│   ├── data/                  # 清洗/tokenize/packing/mmap 加载器
│   ├── training/              # 训练循环、优化器、loss、日志
│   ├── rl/                    # SimPO/GRPO/DAPO/采样器/奖励/judge
│   ├── generation/            # 采样与 KV cache
│   ├── eval/                  # 评测 harness
│   └── utils/                 # 配置、种子、设备、日志
├── scripts/                   # profile_*.py、启动脚本、转换脚本
├── tools/                     # 工具执行器、judge 服务
├── tests/
├── deploy/
│   └── ollama/Modelfile.template
├── data/                      # tokenizer 快照 + processed（gitignore）
└── artifacts/                 # checkpoint/log（gitignore，仅服务器）
```

## 4. 配置系统

- 配置优先：任何训练启动都从 `configs/*.yaml` 读，命令行只允许覆盖叶子键。
- 结构：`base.yaml` 放通用项（seed、精度、裁剪、日志），阶段配置 merge 后生效。
- 每次运行把**完整生效配置 + git commit hash + 数据 manifest hash** 一并写入 `artifacts/<stage>/run-<ts>/config.yaml`，保证可复现。
- 配置即文档：YAML 中每个非平凡字段写注释。

## 5. 代码规范

- Python 3.11 语法；ruff（line-length 100）格式化；类型标注关键公共接口。
- 标识符用英文；模块 docstring 中文说明「是什么/为什么」。
- 不写魔法数：模型维度、损失系数全部来自配置。
- 每个训练脚本入口统一 `torchrun` 调用；不支持 `python -m` 单卡特判逻辑污染主流程（除 Stage 0 冒烟脚本）。
- 错误要显式：数据字段缺失/未知 → 抛错并给出文件名与行号，不静默丢弃。

## 6. 数值与精度规则

- 计算 bf16；`autocast` 范围明确，不做全局无脑 cast。
- embedding/RMSNorm/router bias/损失统计在 fp32。
- 避免 `torch.compile` 与动态 padding 混用带来的重编译：训练期 batch 内 max-length padding 固定后，编译一次。
- 梯度裁剪前记录 grad norm；NaN/Inf 立即 abort 并保存现场。

## 7. 日志与监控

- 必录指标（每 N 步）：loss 及分解、grad norm、lr、吞吐、MFU、专家负载、路由熵、attention 熵、生成长度、KL、思考触发率、格式奖励。
- 通道：wandb（主）+ `metrics.jsonl`（兜底、可离线）。
- 每个 checkpoint 目录带 `metrics.jsonl` 与 `config.yaml`。

## 8. 测试

- 数值对齐测试：RMSNorm/RoPE/attention 输出与 HF 参考实现误差 < 1e-3（bf16 下容差可放宽到 5e-3）；KV cache 分步 decode 与一次性 prefill 对齐。
- 结构测试：参数账（198.5M/63.9M）、router 输出 shape、打包掩码正确性。
- 负载均衡测试：构造倾斜输入，验证 bias 向均衡方向更新。
- 数据测试：6 个 jsonl 的 schema 校验器（字段、末轮空约束、gt 约束）。
- 每次 git push 前本地跑 `pytest`（无需 GPU 的部分）；GPU 测试在服务器 Stage 0 执行。

## 9. Git 工作流

- 分支前缀 `codex/`（如 `codex/model-core`）；常规提交信息用 Conventional Commits（中文说明）。
- main 只接受可用状态；文档与代码分 commit，便于回滚。
- 不提交：`data/`、`artifacts/`、任何权重/GGUF/日志。
- 大文件（>1MB）默认不放 git；确需共享走服务器目录或另行决策。
- 推送前自检：`ruff check`、`pytest`、文档链接有效。

## 10. 复现性

- 固定 seed 表：pretrain 42 / sft 1337 / dpo 2025 / rlaif 2026 / agent_rl 31415。
- 记录环境：`conda env export > artifacts/env.lock.yml`（或 pip freeze）。
- 数据 manifest（sha256 + 过滤统计）随 checkpoint 保存。
