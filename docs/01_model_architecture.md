# 01 · 模型架构规格（localsight-198m-moe）

> 版本锁定：规划阶段 v0。所有「与最初想法不同的地方」都标注了原因。

## 1. 总体设计

- Decoder-only，Pre-Norm（RMSNorm），带 final RMSNorm，全模型 **无 bias**。
- 激活函数 SwiGLU（SwiGLU 中 gate 用 SiLU）。
- **tied embedding**（`lm_head.weight = embed_tokens.weight`）。
- 词表沿用数据集提供的 6400 ByteLevel BPE。
- 与 Qwen3-MoE 权重命名对齐，方便转换（映射表见 §8）。

## 2. 超参数（锁定）

| 参数 | 值 | 说明 |
| --- | --- | --- |
| `vocab_size` | 6400 | 数据集 tokenizer 实测 |
| `hidden_size` | 768 | |
| `num_hidden_layers` | 8 | |
| `num_attention_heads` | 8 | Q |
| `num_key_value_heads` | 4 | KV（GQA，2 组共享） |
| `head_dim` | 96 | 768/8 |
| `moe_intermediate_size` | **2432** | 每个专家 FFN 维度（修正：不是 2048） |
| `intermediate_size` | 2048 | 稠密 FFN 兜底配置，MoE 模式下不使用 |
| `num_experts` | 4 | |
| `num_experts_per_tok` | 1 | top-1 |
| `use_shared_expert` | false | 对齐 MiniMind-3-MoE |
| `max_position_embeddings` | 32768 | |
| `rope_theta` | 1_000_000 | 长上下文取向 |
| `use_qk_norm` | true | Q/K 各一个 RMSNorm |
| `norm_eps` | 1e-6 | |
| `tie_word_embeddings` | true | |
| `use_bias` | false | 全模型 |
| 精度 | bf16 计算 + fp32 主权重（1D/标量参数） | |
| 梯度裁剪 | 1.0 | |

### 参数账（为什么是 2432）

`moe_intermediate_size = 2432 = ceil(768·π/64)·64`，是 MiniMind-3 的实际取值：

| 组件 | 计算 | 参数 |
| --- | --- | --- |
| Embedding（tied） | 6400×768 | 4.92M |
| Attention ×8 | (768+384+384+768)×768 ×8 | 14.16M |
| MoE 专家 ×8 | 4 × 3×768×2432 ×8 | 179.40M |
| Router/Norms | 极小 | ~0.05M |
| **合计** | | **~198.4M（精确 198,416,640）** |
| 激活参数 | 4.92M + 14.16M + 8×3×768×2432 | **~63.9M** |

如果沿用 2048，总参数只有约 170M，与「198M-A64M」标签不符。

## 3. 组件规格

### 3.1 RMSNorm（自研）

- `y = x / sqrt(mean(x²) + eps) * weight`，`eps=1e-6`，weight 初始化为 1。
- 用于：每层 attention/MLP 输入、QK-Norm、final norm。

### 3.2 RoPE + YaRN（自研）

- 旋转对半形式（HF/Qwen 风格 `rotate_half`），应用在 QK-Norm 之后。
- `rope_theta=1e6`；`max_position_embeddings=32768`。
- YaRN 扩展因子在「长上下文续训」子阶段才启用，其余阶段使用原生 RoPE。不要对 32k 目标一上来就强行开大扩展因子。

### 3.3 GQA + FlashAttention 后端（自研包装）

- Q=8 头、KV=4 头，`head_dim=96`；KV 头做 `repeat_interleave` 到 8 头。
- 因果掩码 + 可选 document-aware 掩码（packing 时）走自定义 `attention_mask` 逻辑。
- **后端抽象层**：`AttentionBackend` 接口，按硬件自动选择：
  - 4090（Ada, sm_89）：`torch.nn.functional.scaled_dot_product_attention` 或 FlashAttention-2；
  - Hopper（sm_90）：可选 FlashAttention-3；
  - 训练配置里写 `attn_impl: sdpa|fa2|fa3`。
- **性能选型**：4090 上优先 FA2 内核；Stage 0 用同一 batch 对「FA2」与「SDPA(mem_efficient)+compile」做 micro-benchmark，取吞吐高者并写进实验日志。FA3 只作为未来 Hopper 硬件的扩展项。
- 注意：FlashAttention-3 内核只编译支持 Hopper；在 4090 上写 FA3 无法工作，不要在生产代码里硬编码。
- `torch.compile` 用 `max-autotune` + 静态 batch shape，autotune 缓存放 `artifacts/triton_cache` 避免重复编译；不稳定就局部关闭。

### 3.4 MoE FFN（自研）

- 每层 4 个独立专家，每个专家是 SwiGLU MLP：`gate_proj(768→2432)`、`up_proj(768→2432)`、`down_proj(2432→768)`。
- 训练 forward 按专家分桶 + `einsum/bmm` 批量 GEMM；不要求手写 Triton 内核（4090 上标准 GEMM 足够）。
- 权重命名：`layers.{i}.mlp.experts.{j}.gate_proj/up_proj/down_proj`，与 Qwen3-MoE 一致。

### 3.5 路由：aux-loss-free 负载均衡（关键修正）

原始方案里「z-loss 或 sinkhorn 替代 aux loss」的说法需要修正：**z-loss 不能做负载均衡**，它只是约束 router logits 幅度的稳定性项。负载均衡用 DeepSeek-V3 的偏置法：

- router：`logits = x W_r + bias_e`（`bias_e` 是每个专家的可维护标量，**不参与梯度**）。
- 每步统计每个专家的 token 数，`overloaded → bias_e -= γ`，`underloaded → bias_e += γ`，`γ=1e-3`。
- 同层同时加 router z-loss：`α · mean(logsumexp(logits)²)`，`α=1e-3`。
- 监控：每步记录各专家 token 占比；任意专家占比偏离均衡值（25%）超过 20 个百分点时告警。
- **兜底**：若 pretrain 前 10% 出现坍塌，临时叠加一个很小的 balance aux loss（系数 1e-3），恢复后移除。

> 实测修正（2026-08-13）：4 专家 top-1 小模型在前 70 步就出现路由坍缩（单专家占比
> 冲到 52%）。因此 pretrain 阶段**常开**轻量 balance aux loss（系数 1e-2），并把
> 偏置更新步长放大到 ±1e-2（机制仍与 DeepSeek-V3 相同）。这两个值在 configs/pretrain.yaml。

### 3.6 KV Cache（自研）

- 推理侧连续缓存，支持：prefill 一次性投影、增量 decode 追加、**共享 prompt 的 prefix cache**（RL 采样时 G 个 rollout 复用同一 prompt 前缀）。
- 缓存存 K/V 的 `bf16`，key 缓存旋转后的结果（RoPE 后缓存，避免 decode 重复旋转）。

## 4. 初始化规则

| 参数 | 初始化 |
| --- | --- |
| Embedding | N(0, 0.02) |
| 线性层 | N(0, 0.02)，`fan_in` 为 768 |
| 输出投影（attention out / expert down / final head） | N(0, 0.02 / sqrt(2·n_layers)) ≈ 0.005 |
| RMSNorm weight | 1.0 |
| Router weight | N(0, 0.01)；routing bias 初始 0 |

## 5. 显存预算（DDP，每卡）

以 bf16 权重 + bf16 动量 + fp32 主权重估计，序列 4096：

| 项 | 量级 |
| --- | --- |
| 权重（198M） | ~0.4 GB |
| 梯度 | ~0.4 GB |
| Muon 动量 + fp32 主权重 | ~1.2 GB |
| 激活/中间量（重计算开启） | 约 2–4 GB |
| **合计** | **< 6 GB/卡**，24GB 卡余量充足 |

结论：2×4090 跑 DDP 毫无压力，梯度累积可开很大；瓶颈在算力（MFU）而非显存。

## 6. 多模态扩展设计（现在只留接口）

词表已经内置 `<|vision_start|> <|vision_end|> <|vision_pad|> <|image_pad|>`、`<|audio_*|>`、`<tts_*>`、box/quad 等 token。代码层面预留：

1. `ModalityEncoder` 协议：任意编码器输出 `(hidden_states, position_ids, attention_mask)`；
2. `Projector` 协议：把视觉特征投影到 768 维（先按 2 层 MLP 设计，后续可换）；
3. 主 Transformer 的输入接口只认 `hidden_states + position_ids`，不关心模态；
4. 训练数据格式预留 `"images": [...]` / `"audios": [...]` 字段，数据加载器遇到未知字段报错而非静默丢弃。

后续路线：v2 视觉（SigLIP2 风格小 ViT + MLP 投影）→ v3 音频/语音 → v4 TTS。现在不写任何视觉塔代码。

## 7. 权重命名与导出

模块命名按 Qwen3-MoE 风格：

```
model.embed_tokens.weight
model.layers.{i}.self_attn.q_proj / k_proj / v_proj / o_proj
model.layers.{i}.self_attn.q_norm / k_norm
model.layers.{i}.input_layernorm / post_attention_layernorm
model.layers.{i}.mlp.gate.weight            # router
model.layers.{i}.mlp.experts.{j}.gate_proj / up_proj / down_proj
model.norm.weight
lm_head.weight（与 embed 共享）
```

- 用 `transformers.PretrainedConfig` 子类 + `PreTrainedModel` 包装 `LocalsightForCausalLM`，`save_pretrained` 直接产出 HF 格式（safetensors）。
- 转 llama.cpp 时写一个映射脚本；MoE 专家权重按 llama.cpp 的 `ffn_gate_inp/ffn_up/ffn_down` 命名导出。
- 每层额外保存 `mlp.gate.bias`（routing bias）供推理使用；它不属于梯度参数。

## 8. 与最初方案的差异清单

| 原方案 | 修正后 | 原因 |
| --- | --- | --- |
| ffn_hidden=2048 | **2432** | 参数账对不上 198M |
| FA2/FA3 并列 | 4090 只用 FA2/SDPA，FA3 留给 Hopper | FA3 硬件限制 |
| z-loss 或 sinkhorn 做负载均衡 | 偏置式 aux-loss-free + z-loss 稳定性 | z-loss 不负责均衡 |
| 输出投影 std × 1/√(2·n_layers) | 保留，另按 Moonlight 加 Muon 更新缩放 0.2·√max(A,B) | 两者是不同环节（初始化 vs 优化器更新） |
| — | 补充 QK-Norm 后的 RoPE 细节、KV cache 设计、多模态接口 | 让「可扩展」落到实处 |
