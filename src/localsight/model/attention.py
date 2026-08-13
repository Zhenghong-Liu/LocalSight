"""GQA 注意力：QK-Norm + RoPE + SDPA，支持 KV cache 与 document-aware 掩码。

4090（Ada）上用 SDPA；`attn_impl: fa2` 预留接口（后续按 Stage 0 基准再接入
flash_attn 内核）。KV 在 RoPE 之后、按 kv 头（未 repeat）写入缓存。
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import LocalsightConfig
from .kv_cache import KVCache
from .norms import RMSNorm
from .rope import apply_rotary_emb, build_rope_for_positions, yarn_attention_scale


def build_document_causal_mask(
    document_ids: torch.Tensor,
    dtype: torch.dtype = torch.bool,
) -> torch.Tensor:
    """由每个 token 的文档 id 构造 bool 掩码 (B, 1, S, S)：True=允许注意力。"""
    b, s = document_ids.shape
    doc = document_ids[:, None, :] == document_ids[:, :, None]  # (B,S,S)
    pos = torch.arange(s, device=document_ids.device)
    causal = pos[None, :] <= pos[:, None]  # (S,S)：query i 可看到 key j<=i
    allowed = doc & causal[None, :, :]
    return allowed.to(dtype).view(b, 1, s, s)


class Attention(nn.Module):
    def __init__(self, config: LocalsightConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        h, n_q, n_kv, d = (
            config.hidden_size,
            config.num_attention_heads,
            config.num_key_value_heads,
            config.head_dim,
        )
        self.n_q = n_q
        self.n_kv = n_kv
        self.head_dim = d
        self.n_rep = n_q // n_kv

        self.q_proj = nn.Linear(h, n_q * d, bias=config.use_bias)
        self.k_proj = nn.Linear(h, n_kv * d, bias=config.use_bias)
        self.v_proj = nn.Linear(h, n_kv * d, bias=config.use_bias)
        self.o_proj = nn.Linear(n_q * d, h, bias=config.use_bias)

        if config.use_qk_norm:
            self.q_norm = RMSNorm(d, eps=config.norm_eps)
            self.k_norm = RMSNorm(d, eps=config.norm_eps)
        else:
            self.q_norm = None
            self.k_norm = None

        self.rope_theta = config.rope_theta
        self.rope_scaling = config.rope_scaling
        self.max_pos = config.max_position_embeddings
        self.attn_scale = yarn_attention_scale(config.rope_scaling)
    def forward(
        self,
        hidden: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        cache: Optional[KVCache] = None,
    ) -> torch.Tensor:
        b, s, _ = hidden.shape
        q = self.q_proj(hidden).view(b, s, self.n_q, self.head_dim)
        k = self.k_proj(hidden).view(b, s, self.n_kv, self.head_dim)
        v = self.v_proj(hidden).view(b, s, self.n_kv, self.head_dim)

        if self.q_norm is not None:
            q = self.q_norm(q)
            k = self.k_norm(k)

        offset = 0 if cache is None else cache.length
        if position_ids is None:
            positions = torch.arange(offset, offset + s, device=hidden.device)
        else:
            positions = position_ids.to(hidden.device)

        cos, sin = build_rope_for_positions(
            positions,
            self.head_dim,
            base=self.rope_theta,
            rope_scaling=self.rope_scaling,
            dtype=torch.float32,
        )
        cos = cos[None, :, None].to(q.dtype)  # (1,S,1,d/2) 对齐 (B,S,H,d)
        sin = sin[None, :, None].to(q.dtype)
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)

        if cache is not None:
            cache.append(self.layer_idx, k, v)
            k_full = cache.k(self.layer_idx)  # (B,L,n_kv,d)
            v_full = cache.v(self.layer_idx)
            q = q.transpose(1, 2)  # (B,n_q,S,d)
            k_full = k_full.repeat_interleave(self.n_rep, dim=2).transpose(1, 2)
            v_full = v_full.repeat_interleave(self.n_rep, dim=2).transpose(1, 2)
            is_causal = offset == 0 and attention_mask is None
        else:
            k_full = k.repeat_interleave(self.n_rep, dim=2).transpose(1, 2)
            v_full = v.repeat_interleave(self.n_rep, dim=2).transpose(1, 2)
            q = q.transpose(1, 2)
            is_causal = attention_mask is None

        out = F.scaled_dot_product_attention(
            q,
            k_full,
            v_full,
            attn_mask=attention_mask,
            is_causal=is_causal,
            scale=self.attn_scale,
            dropout_p=0.0,
        )
        out = out.transpose(1, 2).contiguous().view(b, s, -1)
        return self.o_proj(out)
