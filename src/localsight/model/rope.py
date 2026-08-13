"""RoPE（含 YaRN 接口）。

应用形式为 rotate-half（HF/Qwen 风格）。频率缓存为 fp32，应用时按输入 dtype 下采样。
"""
from __future__ import annotations

import math
from typing import Optional

import torch


def build_rope_cache(
    seq_len: int,
    head_dim: int,
    base: float = 1_000_000.0,
    rope_scaling: Optional[dict] = None,
    dtype: torch.dtype = torch.float32,
    device: Optional[torch.device] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """返回 (cos, sin)，形状均为 (seq_len, head_dim // 2)，fp32。"""
    positions = torch.arange(seq_len, dtype=torch.float32, device=device)
    return build_rope_for_positions(
        positions, head_dim, base=base, rope_scaling=rope_scaling, dtype=dtype
    )


def build_rope_for_positions(
    positions: torch.Tensor,
    head_dim: int,
    base: float = 1_000_000.0,
    rope_scaling: Optional[dict] = None,
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """按实际位置计算 (cos, sin)，形状 (S, head_dim//2)，避免预缓存整个 32k 频率表。"""
    if head_dim % 2 != 0:
        raise ValueError("head_dim 必须为偶数")

    dim = head_dim // 2
    theta = base
    if rope_scaling is not None:
        kind = rope_scaling.get("type", "linear")
        factor = float(rope_scaling.get("factor", 1.0))
        if factor != 1.0 and kind == "linear":
            theta *= factor
        elif factor != 1.0 and kind == "yarn":
            # YaRN：整体抬升 theta，配合后续注意力缩放（mscale）
            theta *= factor ** (head_dim / (head_dim - 2))

    inv_freq = 1.0 / (theta ** (torch.arange(0, dim, dtype=torch.float32, device=positions.device) / dim))
    freqs = positions.float()[:, None] * inv_freq[None, :]  # (S, dim)
    cos = freqs.cos()
    sin = freqs.sin()
    return cos.to(dtype), sin.to(dtype)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """x: (..., S, head_dim)；cos/sin: (S, head_dim//2)。"""
    d = x.shape[-1] // 2
    x1 = x[..., :d]
    x2 = x[..., d:]
    return torch.cat((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1)


def yarn_attention_scale(rope_scaling: Optional[dict]) -> float:
    """YaRN 的注意力缩放（mscale）：默认 1 + 0.1·ln(factor)。"""
    if rope_scaling is None or rope_scaling.get("type") != "yarn":
        return 1.0
    factor = float(rope_scaling.get("factor", 1.0))
    if factor <= 1.0:
        return 1.0
    return float(rope_scaling.get("mscale", 1.0 + 0.1 * math.log(factor)))
