# LocalSight 目标模式 Prompt

> 使用方式：在开启目标模式（goal mode）时，把本文件全文作为目标描述/初始 prompt。文件里的路径、账号与决策均为 2026-08-13 已核实的现状；仓库内的 `docs/` 与 `configs/` 是权威细节，两者冲突时以仓库最新 commit 为准并先更新文档。

## 1. 总目标

在两块 RTX 4090 的服务器上，完成 **LocalSight 198M-A64M 思考型 MoE 语言模型**从零到发布的全部工作：

服务器环境搭建 → 模型核心代码与测试 → 数据管线与冒烟测试 → 五阶段训练（pretrain → sft → dpo → rlaif → agent_rl，顺序不可调换）→ 评测 → GGUF 转换 → Ollama 本地运行，并把关键过程与结果持续写回仓库文档和实验日志。

全程以「最高训练效率」为原则：4090（Ada）不用 FlashAttention-3，改用 FA2/SDPA + torch.compile + Muon + sequence packing 等最优组合。

## 2. 完成定义（Done）

满足全部条件才算完成：

1. 服务器 `localsight` conda 环境可用，`scripts/smoke_test.py` 通过；
2. 模型代码完成且单元测试通过：RMSNorm、RoPE、GQA、MoE、KV cache 的数值对齐测试；参数账 = 198.4M 总参（精确 198,416,640）/ 63.9M 激活（精确 63,936,768）；
3. 数据管线产出全部阶段的派生数据与 manifest（sha256 + 过滤统计）；
4. 五个训练阶段按顺序跑完，每阶段 checkpoint 与 metrics 齐全，`docs/07_experiment_log.md` 有完整记录；
5. 评测完成：通用基准 + 思考开/关对比 + 工具 gt 命中率 + 32k NIAH + MoE 负载健康；
6. 产出 `Q8_0` / `Q4_K_M` 两个 GGUF，Ollama `localsight-198m` 可正常对话（思考开关生效）；
7. README、docs 与最终代码/配置一致，git 历史干净，敏感信息零入库。

## 3. 背景与已冻结的决策

这些是规划阶段实测 + 查证后冻结的结论，无充分证据不得推翻；确需改变时先改文档再改代码：

- **架构**：MiniMind-3-MoE 骨架，768 维 / 8 层 / 8 Q 头 / 4 KV 头（GQA），4 专家 top-1、无共享专家；`moe_intermediate_size=2432`（不是 2048，参数账决定）；词表 6400（数据集自带 BPE，含 think/tool/vision/audio/TTS 特殊 token）；tied embedding、无 bias、RMSNorm、QK-Norm、SwiGLU、RoPE theta=1e6、max_pos=32768。
- **多模态**：当前只做文本；代码预留 `ModalityEncoder` / `Projector` 接口与数据字段，不提前实现视觉塔。
- **训练顺序**：pretrain → sft → dpo → rlaif → agent_rl，不可调换。
- **pretrain**：主语料只用大文件 `pretrain_t2t.jsonl`（7.8GB，实测 2.13B tokens）。原计划 5 epochs（约 10.7B tokens），**2026-08-14 用户要求提前结束**：epoch 1（约 2.13B tokens）后即停训，model soup 保存，随后自动衔接 SFT→SimPO（`scripts/early_finish_pretrain.py` 看门狗）。`pretrain_t2t_mini.jsonl`（实测 0.33B）仅用于开发、冒烟与 LR 扫描。优化器 Muon（Moonlight 公式，NS=5、momentum=0.95）管理矩阵参数，AdamW 管 embedding/norm/router；lr=3e-3、wd=0.1（mini 扫描最优）。
- **MoE 路由**：DeepSeek-V3 偏置式负载均衡（偏置不参与梯度，步长 ±1e-2）+ router z-loss（α=1e-3）+ 常开轻量 balance aux loss（α=1e-2）。注：top-1 小模型实测会坍缩，因此从 aux-loss-free 调整为常开轻量 aux，与 MiniMind 原版策略一致。
- **SFT**：905,718 条，约 34% 带 reasoning_content、约 9.4% 带工具；统一用数据自带的 chat_template（assistant 永远包 `<think>...</think>`）；AdamW lr=1.5e-4，2 epochs，seq 8192，packing + document-aware mask，NEFTune α=5。
- **dpo**：17,166 对无思考格式 → SimPO 做通用质量偏好（β=2.0、γ=1.2 起步），lr=5e-6。
- **rlaif**：19,502 条末轮留空 → 补全式 prompt；on-policy 采样 K=6 + judge 打分 + SimPO 更新，2 轮；judge 用本地 7B/8B（vLLM），成本敏感可换 RM。
- **agent_rl**：39,988 条末轮留空，其中 2 万条带 tools+gt；实现 6 种本地工具执行器（calculate_math/get_exchange_rate/get_current_weather/get_current_time/translate_text/unit_converter），GRPO+DAPO（ε_low=0.2、ε_high=0.28、dynamic sampling、overlong shaping、默认 KL=0），G=6、temp=0.9，奖励 = 格式 0.3 + 工具调用 0.2 + gt 结果 0.5；无 gt 的 2 万条不进结果奖励。
- **硬件效率**：4090 只支持 FA2/SDPA；Stage 0 对两者 micro-benchmark 二选一；`torch.compile(max-autotune)`；Muon 省约一半 FLOPs；全部 packing 零填充；MFU 目标 ≥45%（<40% 必须 profiling）。

权威文档：[docs/01_model_architecture.md](docs/01_model_architecture.md)、[docs/02_data_spec.md](docs/02_data_spec.md)、[docs/03_training_plan.md](docs/03_training_plan.md)、[docs/04_engineering_standards.md](docs/04_engineering_standards.md)、[docs/05_evaluation_and_release.md](docs/05_evaluation_and_release.md)、[docs/06_infrastructure.md](docs/06_infrastructure.md)。

## 4. 环境事实

- **本地编码机**（macOS）：仓库 `/Users/liuzh/Documents/LocalSight`，git remote = `https://github.com/Zhenghong-Liu/LocalSight.git`（main）。本地直连 GitHub 偶尔超时，推送失败时：

```bash
export https_proxy=http://127.0.0.1:7897 http_proxy=http://127.0.0.1:7897 all_proxy=socks5://127.0.0.1:7897
git push origin main
```

- **训练服务器**：`sodastar@119.78.227.152`（Ubuntu 22.04），2× RTX 4090（24GB/卡，驱动 580.173）。SSH 公钥已配置，**本地 `ssh localsight` 免密可用**；不再需要密码。项目在 `~/project/LocalSight`。
- **数据**（只读）：服务器 `/media/liuzh/data/DLData/LocalSight`（tokenizer 快照已复制到项目 `data/tokenizer/`）。
- **conda**：已有 `kdl` 环境，**不要动它**；新建独立环境（首选 conda `localsight`，镜像不可用时改用 `scripts/setup_server_venv.sh` 的项目级 `.venv`；两者功能等价，Python 3.11、torch cu128、flash-attn/transformers/trl 等，见 docs/06）。
- **磁盘**：根盘剩约 314G，数据盘剩 4.5T；checkpoint/派生数据写项目目录，不写数据盘。

## 5. 硬约束

- 数据只读，不改不动原始文件；派生数据放 `data/processed/`。
- `artifacts/`、`data/`、权重、GGUF、日志一律不入 git。
- 每阶段不同 seed：pretrain 42 / sft 1337 / dpo 2025 / rlaif 2026 / agent_rl 31415。
- 任何架构或优化器变更后，先重跑 Stage 0 冒烟测试。
- 每阶段结束跑对应评测并填写 `docs/07_experiment_log.md`；不达标先诊断，只允许一次定向重训，不允许静默跳过闸门。
- 服务器只做「拉取 + 训练」，代码改动在本地完成、提交、push，服务器 `git pull --ff-only`。
- 敏感信息零入库：任何密码、私钥、token 不得出现在仓库文件或提交内容里。

## 6. 执行流程

### M1 服务器环境

1. 在服务器建独立 Python 环境（conda `localsight` 或项目 `.venv`，二选一）并安装依赖（官方源失败换镜像）；
2. `torchrun --nproc_per_node=2 scripts/smoke_test.py` 通过；
3. 记录 `artifacts/env.lock.yml`。

### M2 模型核心代码

按依赖顺序实现并配单元测试（放在 `src/localsight/model/`）：

1. RMSNorm / QK-Norm；
2. RoPE（含 YaRN 因子接口）；
3. GQA attention + SDPA/FA2 后端 + document-aware mask；
4. MoE：router（偏置式负载均衡 + z-loss）+ 专家分桶批量 GEMM；
5. Transformer 主干 + 初始化 + 参数账断言（198.5M / 63.9M）；
6. KV cache 与 prefix cache（供 RL 采样）。

每个组件与参考实现数值对齐（误差 <5e-3），测试无 GPU 也能跑的部分在本地跑，GPU 部分在服务器跑。

### M3 数据管线 + Stage 0

1. 清洗、minhash 去重、tokenize、packing、mmap 缓存 + manifest；
2. 切分并冻结 held-out 评测集；
3. Stage 0：mini 语料 8 步数值测试、DDP、专家负载、显存曲线、FA2 vs SDPA 基准、NCCL P2P 测试、compile 稳定性，MFU ≥45%。

### M4 Pretrain

mini 语料 LR 扫描 [1e-3, 2e-3, 3e-3] × wd {0.05, 0.1} → 全量大语料 **5 epochs**（有效 batch 1M tokens，seq 4096）→ 最后 3 个 ckpt model soup → 验证 loss 与专家负载。

### M5 SFT

AdamW lr=1.5e-4，2 epochs，seq 8192，packing + loss mask，NEFTune α=5；监控思考占比/长度漂移/工具格式率。

### M6 SimPO

β=2.0、γ=1.2，lr=5e-6，1 epoch；监控 margin 与 logp 分离度。

### M7 RLAIF

2 轮 on-policy：采样 K=6（temp 0.8）→ judge 打分 → SimPO 更新（lr=3e-6）；judge 瓶颈按预算调整（K=4 / 4B judge / RM 头）。

### M8 Agent RL

2 万条工具任务，6 种工具执行器；GRPO+DAPO（ε_high=0.28），G=6，temp=0.9，lr=3e-6；监控格式率、工具合法性、gt 命中率、KL。

### M9 评测与发布

1. 评测：C-Eval/CMMLU/MMLU/GSM8K/HumanEval/IFEval、思考开/关对比、BFCL + held-out 工具集、NIAH@32k、MoE 健康度；
2. 转 llama.cpp GGUF：`Q8_0`、`Q4_K_M`，与 bf16 原版做 perplexity/样例对齐；
3. 用 `deploy/ollama/Modelfile.template` 创建并验证 Ollama 模型；
4. 写 model card（数据构成、局限、许可）与最终评测表。

## 7. 每阶段完成后的固定动作

1. 跑该阶段闸门评测；
2. 更新 `docs/07_experiment_log.md`；
3. 更新受影响文档/配置（如有变更）；
4. 本地 commit（Conventional Commits，中文说明）；
5. 走代理 push 到 GitHub；
6. 服务器 `git pull --ff-only`，必要时 rsync 派生数据/脚本；
7. 在最终回复中简要汇报结果与下一步。

## 8. 遇到阻塞时的规则

- 先读 `docs/` 找答案；技术细节可查最新官方资料（论文原文/官方文档），并把结论写回文档；
- 数据、网络、权限、硬件被他人占用等外部阻塞：尝试所有安全范围内的替代方案（如镜像源、换后端、错峰运行），再向用户汇报；
- 不要为了「看起来完成」而降低闸门标准；不要伪造评测结果。

## 9. 沟通要求

- 每完成一个里程碑或做出影响方案的决定，在最终回复里说明：做了什么、依据、结果、下一步；
- 偏离本文件的任何决策都要明确标注「与原规划不同 + 理由」；
- 长任务中途用简洁更新同步进度，避免用户长时间无感知。
