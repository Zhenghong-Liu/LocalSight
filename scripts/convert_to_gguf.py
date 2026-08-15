#!/usr/bin/env python3
"""LocalSight → GGUF 转换（llama.cpp Qwen2MoE 命名）。

注意：llama.cpp 加载自定义 6400 词表 + QK-Norm + 无共享专家的模型需要对应
arch 支持；本脚本按 Qwen2MoE 张量布局输出，M9 阶段与 llama.cpp 实际验证，
不匹配处在此处修正。用法：
    PYTHONPATH=src python scripts/convert_to_gguf.py \
        --checkpoint artifacts/agent_rl/final --out artifacts/release/localsight.Q8_0.gguf \
        --quant q8_0 --tokenizer data/tokenizer
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from localsight.model import LocalsightConfig
from localsight.tokenizer import LocalSightTokenizer


def tensor_map(config: LocalsightConfig) -> dict[str, tuple[str, tuple[int, ...]]]:
    """our state_dict key → (gguf name, 期望形状)。"""
    mapping = {
        "model.embed_tokens.weight": ("token_embd.weight", (config.vocab_size, config.hidden_size)),
        "model.norm.weight": ("output_norm.weight", (config.hidden_size,)),
        "lm_head.weight": ("output.weight", (config.vocab_size, config.hidden_size)),
    }
    for i in range(config.num_hidden_layers):
        prefix = f"model.layers.{i}."
        mapping[f"{prefix}input_layernorm.weight"] = (f"blk.{i}.attn_norm.weight", (config.hidden_size,))
        mapping[f"{prefix}post_attention_layernorm.weight"] = (f"blk.{i}.ffn_norm.weight", (config.hidden_size,))
        mapping[f"{prefix}self_attn.q_proj.weight"] = (f"blk.{i}.attn_q.weight", (config.hidden_size, config.hidden_size))
        mapping[f"{prefix}self_attn.k_proj.weight"] = (f"blk.{i}.attn_k.weight", (config.num_key_value_heads * config.head_dim, config.hidden_size))
        mapping[f"{prefix}self_attn.v_proj.weight"] = (f"blk.{i}.attn_v.weight", (config.num_key_value_heads * config.head_dim, config.hidden_size))
        mapping[f"{prefix}self_attn.o_proj.weight"] = (f"blk.{i}.attn_output.weight", (config.hidden_size, config.hidden_size))
        mapping[f"{prefix}self_attn.q_norm.weight"] = (f"blk.{i}.attn_q_norm.weight", (config.head_dim,))
        mapping[f"{prefix}self_attn.k_norm.weight"] = (f"blk.{i}.attn_k_norm.weight", (config.head_dim,))
        mapping[f"{prefix}mlp.gate.weight"] = (f"blk.{i}.ffn_gate_inp.weight", (config.num_experts, config.hidden_size))
        for e in range(config.num_experts):
            mapping[f"{prefix}mlp.experts.{e}.gate_proj.weight"] = (
                f"blk.{i}.ffn_gate_exps.weight", (config.num_experts, config.moe_intermediate_size, config.hidden_size)
            )
            mapping[f"{prefix}mlp.experts.{e}.up_proj.weight"] = (
                f"blk.{i}.ffn_up_exps.weight", (config.num_experts, config.moe_intermediate_size, config.hidden_size)
            )
            mapping[f"{prefix}mlp.experts.{e}.down_proj.weight"] = (
                f"blk.{i}.ffn_down_exps.weight", (config.num_experts, config.hidden_size, config.moe_intermediate_size)
            )
    return mapping


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--quant", default="q8_0", choices=["f16", "q8_0", "q4_k_m"])
    parser.add_argument("--tokenizer", default="data/tokenizer")
    args = parser.parse_args()

    config = LocalsightConfig()
    state = torch.load(Path(args.checkpoint) / "model.pt", map_location="cpu")
    mapping = tensor_map(config)
    tokenizer = LocalSightTokenizer(args.tokenizer)

    try:
        from gguf import GGUFWriter
    except ImportError:
        raise SystemExit("需要 pip install gguf")

    writer = GGUFWriter(args.out, "qwen2moe")
    writer.add_name("LocalSight-198M-MoE")
    writer.add_context_length(config.max_position_embeddings)
    writer.add_embedding_length(config.hidden_size)
    writer.add_block_count(config.num_hidden_layers)
    writer.add_head_count(config.num_attention_heads)
    writer.add_head_count_kv(config.num_key_value_heads)
    writer.add_expert_count(config.num_experts)
    writer.add_expert_used_count(config.num_experts_per_tok)
    writer.add_rope_dimension_count(config.head_dim)
    writer.add_rope_freq_base(config.rope_theta)

    # 词表：tokenizer 顺序 → 字符串；BPE merges 写出（llama.cpp 需要）
    vocab = tokenizer.tok.get_vocab()
    tokens = [""] * len(vocab)
    for token, idx in vocab.items():
        tokens[idx] = token
    writer.add_tokenizer_model("gpt2")
    writer.add_token_list(tokens)
    with open(Path(args.tokenizer) / "tokenizer.json", encoding="utf-8") as f:
        merges = list(json.load(f)["model"].get("merges", []))
    if merges:
        writer.add_array("tokenizer.ggml.merges", merges)
    writer.add_array("tokenizer.ggml.scores", [0.0] * len(tokens))

    # 非专家张量
    for our_name, (gguf_name, _) in mapping.items():
        if "_exps" in gguf_name:
            continue
        writer.add_tensor(gguf_name, state[our_name].float().numpy())

    # 专家张量按 (E, ...) 堆叠：gate/up (E, F, D)，down (E, D, F)
    e, d, f = config.num_experts, config.hidden_size, config.moe_intermediate_size
    for i in range(config.num_hidden_layers):
        prefix = f"model.layers.{i}.mlp.experts."
        for kind, gguf_name in (("gate_proj", f"blk.{i}.ffn_gate_exps.weight"),
                                ("up_proj", f"blk.{i}.ffn_up_exps.weight"),
                                ("down_proj", f"blk.{i}.ffn_down_exps.weight")):
            stacked = np.stack(
                [state[f"{prefix}{expert}.{kind}.weight"].float().numpy() for expert in range(e)],
                axis=0,
            )
            writer.add_tensor(gguf_name, stacked)

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    print("written:", args.out)


if __name__ == "__main__":
    main()
