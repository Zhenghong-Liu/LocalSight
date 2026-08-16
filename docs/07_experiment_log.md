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
- `pretrain_t2t_mini.jsonl` 构建完成：80,322 序列、**328,903,639 tokens**（权威值，修正此前 0.61B 的字节加权估计）。
- 发现并修复：padding 语义（input 用合法 token 0、labels 用 -100，否则 Embedding device assert）。

## M4 · Pretrain 开发验证（2026-08-13）

- 编译版短跑（mini、20 步、单卡、batch 24、`max-autotune-no-cudagraphs`）通过：
  - 修复前：router 被 Muon 接管 + 负载偏置比例步长 → z-loss 爆炸到 6e5、负载摆到 0.5；
  - 修复后：router/embedding/norm 走 AdamW、偏置改 DeepSeek-V3 固定 ±γ → z≈11.8、负载≈[0.25,0.18,0.36,0.19]、loss 8.61→8.58；
  - 与激活重计算共存必须禁用 cudagraphs。
- 注意力实测走 flash 后端（4.2ms @ B24×S4096×96d），不是瓶颈；当前吞吐 ~50k tok/s/卡主要受 MoE 分桶小 GEMM、重计算与 DDP 未用参数标记影响（后续优化）。
- LR/wd 扫描进行中（mini、6 组合 × 400 步）。

### LR/wd 扫描结果（mini 验证集，2026-08-13）

| lr | wd | val_loss |
| --- | --- | --- |
| 1e-3 | 0.05 | 3.8551 |
| 1e-3 | 0.1 | 3.9364 |
| 2e-3 | 0.05 | 3.3564 |
| 2e-3 | 0.1 | 3.2640 |
| 3e-3 | 0.05 | 3.2965 |
| **3e-3** | **0.1** | **3.1576** |

结论：正式 pretrain 用 lr=3e-3、wd=0.1。全量语料实测 2,131,183,459 tokens，
5 epochs ≈ 10.7B tokens。

### Pretrain 正式启动（2026-08-13 21:26）

- 配置：2 卡、micro 32×4096、accum 4（1M tokens/step）、重计算、无 compile、
  `NCCL_P2P_DISABLE=1`、lr=3e-3、wd=0.1、5 epochs。
- 启动观察：step 29 loss=8.44（首步 8.89）、z=10.1、grad=16.9、
  吞吐 95,365 tok/s（MFU 17.2%）、专家负载≈[0.23,0.15,0.45,0.16]。
- 预计 10.7B tokens ≈ 31h；日志 `artifacts/pretrain.log`，每 1000 步保存 checkpoint 并评估验证损失。

### 提前收尾计划（2026-08-14 用户决定）

- 5 epochs（约 31h）太长；改为 epoch 1 后提前结束：看门狗 04:00 停训 →
  对 checkpoint 做 model soup → 自动跑 SFT（2 epochs）→ SimPO，预计 06:30 前完成。

### 实际结果（2026-08-14 07:43）

- 预训练在 **01:03 因 NCCL 集合通信错误自行退出**（约 step 1060），因此只有
  **step-1000** checkpoint（val_loss=1.6933）——没有 step-2000，无法做 soup，
  看门狗链的 SFT/SimPO 因缺 soup 而失败。
- 用户于 07:40 决定结束 pretrain：保留 step-1000（已复制到项目
  `artifacts/pretrain/step-1000`，含 model.pt/optimizer.pt/state.json）。
- 07:43 从 step-1000 启动 **SFT**（2 epochs、858 步、约 15.2s/step，预计 11:15 完成），
  完成后手动接 SimPO。

### 路由坍缩修复（2026-08-13）

- 现象：top-1 路由在 ~step 39 起单专家占比冲到 52%、另一专家跌到 4.8%。
- 修复：balance aux loss 常开（α=1e-2）、偏置步长 ±1e-2；重启后 step 59
  负载恢复 [0.255, 0.249, 0.267, 0.230]，z≈5.4、loss≈7.53，双卡 95.8k tok/s。
- 结论：4 专家 top-1 小模型需比 DeepSeek-V3 更强的负载均衡；已写入 GOAL_PROMPT/docs/01/03。

### 后续阶段数据（2026-08-13）

- dpo：17,166 对；agent_prompts：20,000（全部带工具+gt）；rlaif_prompts：19,502。
- sft：905,718 条对话 → 54,873 序列（8192），449,519,616 tokens。
- rlaif questions.jsonl：19,502 条完整对话历史（供 judge 上下文）。
- judge 模型就绪：Qwen2.5-7B-Instruct 已完整下载（15GB，hf-mirror）。

### 后续阶段进展（2026-08-14）

- SFT 2 epochs 完成（artifacts/sft/final）；SimPO 完成（artifacts/dpo/model.pt，margin 转正）。
- RLAIF 第一轮完成：8k prompts × K=2 采样 → transformers 7B judge 打分（vLLM 因 nvcc/JIT 兼容问题弃用）→ SimPO → artifacts/rlaif_round1/model.pt。
- RLAIF 第二轮启动中（同样 8k × K=2），预计数小时。

### 最终结果（2026-08-15）

- 五阶段全部执行：pretrain（提前结束于 step-1060，用户决定）→ SFT 2 epochs → SimPO →
  RLAIF 2 轮（8k prompts × K=2）→ Agent RL（4k prompts，含黄金 rollout）。
- 评测（最终 agent_rl 权重，100 样本/项，思考开/关得分相同）：
  MMLU 17%、C-Eval 19%、GSM8K 1%。（模型受 0.5-epoch 预训练限制，绝对分偏低；思考增益暂不显著。）
- Agent RL 训练期：mean_reward 0.21→0.55（格式奖励改善）；gt_hit=0.167（1/6 组内为黄金样本，采样样本基本未解出）。
- 发布：`artifacts/release/` 下 f16（qwen3moe）、Q8_0、Q4_K_M 三个 GGUF；Ollama 模型
  `localsight-198m`（用户态 Ollama 0.32.13，端口 11435，GPU 推理）已通过冒烟。
- 兼容性记录：系统 Ollama 0.23.2 的 qwen2moe runner 与我们的 QK-Norm/无共享专家结构不兼容；
  最终采用 qwen3moe 架构 + 自有 0.32.13 二进制。
- Ollama 思考开关：旧 Modelfile 模板不含思考块（API 报 does not support thinking）；
  改用官方 `RENDERER qwen3.5` 后实测 `think:false` 直接出答案、`think:true` 返回 thinking 字段。

### 最终评测补充（2026-08-15）

| 评测项 | 结果 | 说明 |
| --- | --- | --- |
| NIAH（Needle-in-a-Haystack） | 4k/8k/16k/**32k 全 5/5**（acc 1.0） | 每个长度 5 个 seed，`scripts/run_niah.py` |
| IFEval（宽松规则版） | 100/100 | `scripts/run_ifeval.py` 仅检查开头/结尾/关键词/是-否等可规则化指令，**非严格 IFEval** |
| Agent held-out（自建 400 条） | mean_reward=0.367、gt_hit=0.015 | 从 agent_prompts 划出的 held-out；采样样本基本未解出工具任务 |
| MoE 负载健康 | step 999 负载 [0.269, 0.225, 0.240, 0.265] | 无路由坍塌，详见 pretrain.log |

- 未跑项与替代：HumanEval / BFCL / CMMLU 未正式跑（下载或环境受限）；
  BFCL 用自建 held-out 工具集 + gt 命中率替代，CMMLU 用 C-Eval 覆盖中文知识维度。
- 完整评测表与发布信息见 [docs/08_final_metrics.md](08_final_metrics.md)。

### Pretrain 续训计划（2026-08-15 用户决定）

- 与 MiniMind 对照：官方 198M-MoE 评测 C-Eval 25.5 / CMMLU 24.3（≈随机），我们 19%
  的差距主要来自只练了 0.5 epoch（1.06B tokens）而非预处理问题（我们的去重+packing
  是 MiniMind 截断+padding 的超集）。
- 决策：从 `artifacts/pretrain/step-1000` 续训至总 **5.5B tokens**（对齐 MiniMind），
  约再练 4.44B；轻量提速套餐（≤1.5h，<1.2× 即回退）；每小时定时抽查 + checkpoint
  快评；看门狗自动续训；下游阶段与 v0.2.0 暂缓。
- 实现：`train_pretrain.py` 支持 `--resume` / `--max-total-tokens` / 定时抽查，
  `scripts/pretrain_watchdog.py`、`scripts/bench_pretrain_speed.py`、
  `scripts/pretrain_final_eval.sh`；thinking_prompts 扩到 50 条。

### Pretrain 续训执行（2026-08-15 16:05 启动）

- 提速基准：基线 96.2k tok/s vs compile 98.6k（1.03×）→ 按阈值回退，不启用 compile；
  DDP static_graph 实测零收益且带来显存压力，回退首轮稳定配置。
- 冒烟：mini 语料 10 步保存 → 编译版续训 10 步，checkpoint 键名干净、soup 正常、LR 连续。
- 正式运行：看门狗从 `artifacts/pretrain/step-1000` 续训（`--max-total-tokens 5.5e9`），
  LR 重锚 2.96e-3；首步抽查 ppl=5.45（与 val_loss 1.69 一致），loss 1.92→1.70 恢复下降；
  日志 `artifacts/pretrain_resume.log`，抽查产物 `artifacts/eval_samples/`，
  checkpoint 每 500 步（step-1500/2000/...），完成后看门狗自动跑收尾评测。
- 用户后续指令（2026-08-15）：pretrain 预计 08-16 05:15 完成，收尾评测后由
  `scripts/sft_chain.py` 自动接 SFT；质量由 7B judge + 指标 + 快评裁决，未达标则用
  `sft_512.jsonl` 加训一轮；SFT 完成后停止等用户指令（不自动接 SimPO/RL）。
- 用户指令修正（2026-08-16）：SFT round1（sft_t2t_mini，2 epochs）完成后**无条件**
  接 sft_512 训练 1 epoch，由 `scripts/sft512_chain.py` 后台接力（setsid，断网不影响），
  完成后跑质量报告并停止等指令。

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
