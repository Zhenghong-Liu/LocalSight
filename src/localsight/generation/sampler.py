"""自回归采样：temperature/top-p + KV cache。"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F

from localsight.model import KVCache


@torch.no_grad()
def generate(
    model,
    input_ids: torch.Tensor,
    max_new_tokens: int = 128,
    temperature: float = 1.0,
    top_p: float = 1.0,
    eos_id: Optional[int] = None,
    cache: Optional[KVCache] = None,
) -> torch.Tensor:
    """自回归采样，返回完整序列 (1, L)。temperature<=0 时退化为贪心。"""
    if input_ids.ndim != 2:
        raise ValueError("input_ids 需要 (1, S) 形状")
    seq = input_ids
    for _ in range(max_new_tokens):
        logits, _, _ = model(seq[:, -1:] if cache is not None and cache.length > 0 else seq, cache=cache)
        if cache is not None:
            cache.commit()
        next_logits = logits[:, -1, :]
        if temperature <= 0:
            token = next_logits.argmax(dim=-1, keepdim=True)
        else:
            probs = F.softmax(next_logits / temperature, dim=-1)
            token = _sample_top_p(probs, top_p)
        seq = torch.cat([seq, token], dim=1)
        if eos_id is not None and token.item() == eos_id:
            break
    return seq


def _sample_top_p(probs: torch.Tensor, top_p: float) -> torch.Tensor:
    if top_p >= 1.0:
        return torch.multinomial(probs, num_samples=1)
    sorted_probs, sorted_idx = probs.sort(dim=-1, descending=True)
    cumulative = sorted_probs.cumsum(dim=-1)
    cutoff = cumulative > top_p
    cutoff[..., 1:] = cutoff[..., :-1].clone()
    cutoff[..., 0] = False
    sorted_probs = sorted_probs.masked_fill(cutoff, 0.0)
    sorted_probs /= sorted_probs.sum(dim=-1, keepdim=True).clamp(min=1e-9)
    sampled = torch.multinomial(sorted_probs, num_samples=1)
    return sorted_idx.gather(-1, sampled)
