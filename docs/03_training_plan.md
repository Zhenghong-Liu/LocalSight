# 03 · 训练方案（五阶段，顺序不可调换）

## 1. 全局硬约定

- 精度：**bf16** autocast；embedding/norm/router bias 等 1D 参数用 fp32 主权重。
- `torch.compile`：模型 forward 编译，MoE/动态 shape 出问题就局部关闭。
- 梯度裁剪 1.0；重计算：pretrain 全层重算，其余阶段选择性重算。
- 注意力后端：`sdpa` 或 `fa2`（4090 不支持 FA3，见 01 文档）。
- 每阶段不同 seed；checkpoint 保存优化器状态，支持断点续训。
- 记录：loss、grad norm、lr、吞吐（tok/s/GPU）、MFU、专家负载/路由熵、attention 熵、生成长度、KL、思考触发率。
- **性能目标**：MFU ≥ 45%（2×4090 合计约 660 TFLOPS bf16）；低于 40% 必须用 `torch.profiler` 定位瓶颈后再继续。
- **融合内核**：QKV+QK-Norm+RoPE 融合、SwiGLU 的 gate/up 合并 GEMM、fused cross-entropy。规则是先写参考实现并对齐数值，再上 Triton。
- **MoE 前向**：按专家分桶批量 GEMM，router 计算在 fp32，禁止逐 token 的 Python 循环。

## 2. 计算预算（2×4090）

单 token 训练 FLOPs ≈ 6N（N≈198.4M）≈ 1.19e9。

| 阶段 | tokens | 时间估算（40–50% MFU） |
| --- | --- | --- |
| Pretrain 5 epochs（~20.8B） | 20.8B | ~21–28 h |
| SFT 2 epochs | ~0.66B | ~1–2 h |
| SimPO 1 epoch | ~25M | ~0.5–1 h |
| RLAIF（采样+judge+更新）×2 | 采样 ~120M gen tokens | **12–30 h（judge 主导）** |
| Agent RL 1 epoch | 采样 ~150M gen tokens | ~2–4 h |

纯训练约 1.5–3 天（pretrain 5 epochs 是最大头）；加上开发调试，现实排期 1–2 周。

## 3. Stage 0 · 冒烟测试（先于任何正式训练）

用 `pretrain_t2t_mini.jsonl` 跑：

1. 前向/反向数值测试：单卡 8 步，检查 loss 下降、无 NaN、grad norm 正常；
2. Muon 与 AdamW 参数分组正确性（矩阵 vs 1D 参数）；
3. 专家负载打印：4 专家 token 占比；
4. DDP 双卡一致性与吞吐；
5. 显存占用曲线；
6. 注意力后端 micro-benchmark：FA2 vs SDPA(mem_efficient)+compile，固定 batch 比吞吐；
7. NCCL P2P 带宽测试：4090 无 NVLink，确认 P2P over PCIe 是否可用，不可用则设 `NCCL_P2P_DISABLE=1`；
8. `torch.compile(max-autotune)` 稳定性与重编译次数检查。

通过标准：loss 平滑下降、无 NaN、双卡吞吐接近单卡 2 倍、MFU ≥ 45%。**任何架构/优化器变更后先重跑 Stage 0。**

## 4. Stage 1 · Pretrain

**数据**：`pretrain_t2t.jsonl`（全量），packing 到 4096，document-aware mask。

**优化器（Muon + AdamW 混合，按 Moonlight 公式）**

- 2D 矩阵（attention 投影、各专家 FFN、head）→ Muon：
  - `W ← W − η·(0.2·O·√max(A,B) + λ·W)`，`O = Newton-Schulz(momentum)`，NS 步数 5；
  - momentum 0.95（Nesterov 形式），动量存 bf16。
- 1D/标量（embedding、RMSNorm、router、routing bias、各 bias）→ AdamW `β=(0.9,0.95)`。
- 学习率与 wd 在两类优化器间**共享**（Muon 更新已缩放到 AdamW 的 RMS 范围）。

**超参**

| 项 | 值 |
| --- | --- |
| 峰值 LR | **开发期在 mini 语料上扫 [1e-3, 2e-3, 3e-3]**，取验证 loss 最优者用于全量（初值 2e-3） |
| wd | 0.1（备选 0.05，一并扫） |
| 调度 | warmup 2% 步数 → cosine 衰减到峰值的 10% |
| 有效 batch | 1,048,576 tokens = 2 卡 × 32 seq × 4096 × 累积 4 |
| epochs | **5（用户锁定）**；小语料只用于开发/冒烟与 LR 扫描 |
| seq len | 4096 |

**MoE 监控与干预**

- 每步记录每层 4 专家 token 占比；偏离 25% 平衡点 >20pp 告警。
- 偏置式负载均衡（γ=1e-3）+ z-loss（α=1e-3）。
- 若前 10% 坍塌：叠加 balance aux loss 1e-3，恢复后移除。

**收尾**：取最后 3 个 checkpoint 做 model soup（权重平均）作为 SFT 起点。

**长上下文（可选子阶段）**：语料本身没有长文档；在进入 SFT 前，可先用 packing 到 16384 的序列 + YaRN 继续训 1–3B tokens（约 1–2 h），让 32k 推理有真实训练支撑。若时间紧可跳过，改为在 SFT 中逐步拉长。

## 5. Stage 2 · SFT（注入思考能力）

**数据**：`sft_t2t_mini.jsonl`，统一 `apply_chat_template`。思考样本走 `<think>`；非思考样本保留空 think 块；工具样本保留 tools/tool_calls/tool 轮次。

| 项 | 值 |
| --- | --- |
| 起点 | Stage 1 model soup |
| 优化器 | AdamW（本阶段不上 Muon），wd 0.1（可降 0.05） |
| LR | 1.5e-4，cosine，warmup 3% |
| 序列长度 | 8192（思考链天然更长；预留 16384） |
| batch | 128 seq/step（约 1M tokens） |
| epochs | 2 |
| 打包 | 开启 sequence packing + document-aware mask（同一 batch 混装多条对话，loss mask 区分，减少 padding 浪费） |
| loss mask | system/user/tool 轮与 role 头不参与 loss；assistant 的 think+answer 全部参与 |
| NEFTune | α=5（对 embedding 输入加噪声） |

**监控**：思考样本占比应与数据分布（约 34%）一致；回答长度分布不漂移；工具格式合法率抽查。

**闸门**：开启思考 vs 关闭思考在数学/推理 eval 上有正增益；「你好」类问题不输出长篇思考。

## 6. Stage 3 · 偏好对齐（SimPO）

**数据**：`dpo.jsonl`（17,166 对，无思考格式 → 通用质量偏好）。模板化后 chosen/rejected 都带空 think 块。

| 项 | 值 |
| --- | --- |
| 算法 | SimPO（无 ref、长度归一化、target margin） |
| β / γ | 起始 β=2.0，γ=1.2；开发期在 [β∈{2.0,2.5}] × [γ∈{0.8,1.2}] 小扫 |
| LR | 5e-6，cosine |
| batch | 48 pairs/step |
| epochs | 1 |
| 序列长度 | 4096 |

**监控**：chosen/rejected 的平均 logp 与 margin、长度差。若 margin 长期不分离，先降 γ 再检查模板。

## 7. Stage 4 · RLAIF（迭代式 on-policy，2 轮）

**prompt 构造**：`rlaif.jsonl` 去掉末轮空 assistant，`apply_chat_template(add_generation_prompt=True, open_thinking=True)`。

每轮：

1. 当前策略采样 **K=6**（temp 0.8，top_p 0.95），前缀缓存复用；
2. **judge 打分**（0–10，rubric 必须含：思考步骤是否合理/有无跳跃/结论是否由思考导出/回答正确性与简洁性）；
3. 同 prompt 内组对（best vs worst + 相邻对）→ SimPO 更新。

| 项 | 值 |
| --- | --- |
| 轮数 | 2 |
| β / γ / LR | 沿用 Stage 3 结论，LR 3e-6 |
| judge | 本地 Qwen2.5-7B-Instruct 或 Qwen3-8B，vLLM 批处理；备选：用 dpo 数据训轻量 RM 头 |
| 序列长度 | 4096（生成 ≤1024） |

**成本提示**：judge 是瓶颈（19.5k prompt × 6 采样 ≈ 11.7 万次打分/轮）。8B judge 单卡约 200 tok/s 时一轮要 10–20 h；可接受地降本方案：K=4、judge 换成 4B、只对 8k 条打分，或两卡都给 vLLM。此阶段安排夜间/后台跑。

## 8. Stage 5 · Agent RL（GRPO + DAPO）

**数据**：`agent_rl.jsonl` 中带 tools+gt 的 20,000 条。prompt = 去掉末轮空 assistant + 保留 system tools。

**工具执行循环**：模型生成 → 解析 `<tool_call>` JSON → 执行器运行 → 以 `<tool_response>` 接回 → 继续生成；最多 3 次工具调用。6 种工具全部本地实现。

**奖励（归一化到 [0,1]）**

- 格式 0.3：think 标签完整 + tool_call JSON 合法；
- 工具调用 0.2：名字/参数合法且被成功执行；
- 结果 0.5：最终答案与 `gt` 匹配（数值做归一化比较，文本做包含/相似匹配）；
- overlong shaping：超长按 DAPO 规则扣分，截断不惩罚。

| 项 | 值 |
| --- | --- |
| 算法 | GRPO + DAPO：clip ε_low=0.2 / **ε_high=0.28（clip-higher）**、dynamic sampling、overlong shaping |
| token-level loss | v1 关闭，实验对比后决定 |
| KL | 默认 0（DAPO）；出现熵坍缩再开 0.005（GRPO 风格） |
| G / temp | G=6，temp 0.9，top_p 0.95 |
| LR | 3e-6，constant，warmup 0（DAPO 的 cold start 结论） |
| batch | 24 prompts/step |
| 序列长度 | 8192（数据实测很短；32k 留到长文评估用，不用于本轮训练） |
| epochs | 1 |

**注意**：无 gt 的 19,988 条开放题**不进入**本阶段结果奖励（无环境、无裁判会奖励攻击）；可作为 held-out 评测或后续扩展。

## 9. 指标与闸门总表

| 阶段 | 必须监控 | 不达标动作 |
| --- | --- | --- |
| Pretrain | loss、grad norm、专家负载、吞吐/MFU | 坍塌→开 aux loss；loss 不降→回 Stage 0 查数值 |
| SFT | 思考占比、长度漂移、工具格式率 | 思考占比失控→调采样比例；格式差→回数据模板 |
| SimPO | margin、chosen/rejected logp | margin 不分离→降 γ 再跑 |
| RLAIF | judge 分数分布、reward margin | judge 打不出区分度→换 rubric/RM |
| Agent RL | reward 构成、格式率、gt 命中率、KL | 命中率不升→调奖励权重/采样温度 |

每阶段结束跑第 05 号文档的评测集，记录到实验日志。

## 10. 失败恢复与回退

- 所有阶段 checkpoint 保留最近 3 个，可续训；
- Muon 出问题（NaN/不收敛）→ 整体退回 AdamW（超参保留）；
- 偏置式路由不收敛 → 退回 MiniMind 原生 aux loss；
- RLAIF judge 成本失控 → RM 头方案；
- 任何阶段重训前，先在 Stage 0 复现问题。
