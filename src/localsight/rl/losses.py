"""偏好对齐与 RL 损失：SimPO、GRPO(+DAPO 变体)。"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def simpo_loss(
    chosen_logps: torch.Tensor,
    rejected_logps: torch.Tensor,
    beta: float = 2.0,
    gamma: float = 1.2,
) -> torch.Tensor:
    """SimPO：无参考模型、按序列长度归一化后的 margin 损失。

    输入为 per-token logp，内部取每个响应的均值（长度归一化）。
    """
    pi_chosen = chosen_logps.mean(dim=-1)
    pi_rejected = rejected_logps.mean(dim=-1)
    return -F.logsigmoid(beta * (pi_chosen - pi_rejected) - gamma).mean()


def grpo_dapo_loss(
    old_logps: torch.Tensor,
    logps: torch.Tensor,
    advantages: torch.Tensor,
    mask: torch.Tensor | None = None,
    clip_low: float = 0.2,
    clip_high: float = 0.28,
    token_level: bool = False,
) -> torch.Tensor:
    """GRPO + DAPO(clip-higher) 的策略梯度损失。

    old_logps/logps: (G, L) per-token；advantages: (G,) 或 (G, 1)。
    DAPO 语义：ratio 同时做下界 1-clip_low 与上界 1+clip_high 裁剪。
    """
    ratio = torch.exp(logps - old_logps)
    adv = advantages.reshape(ratio.shape[0], 1)
    ratio_clipped = torch.clamp(ratio, 1 - clip_low, 1 + clip_high)
    if token_level and mask is not None:
        per_token = torch.minimum(ratio * adv, ratio_clipped * adv)
        denom = mask.sum()
        return -(per_token * mask).sum() / denom.clamp(min=1)
    seq_ratio = ratio.mean(dim=-1)
    seq_clipped = ratio_clipped.mean(dim=-1)
    return -torch.minimum(seq_ratio * advantages, seq_clipped * advantages).mean()


def group_advantages(rewards: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """组内归一化优势：(r - mean) / (std + eps)，按 G 分组。

    rewards 形状 (num_prompts, G) → 返回同形状 advantages。
    """
    mean = rewards.mean(dim=-1, keepdim=True)
    std = rewards.std(dim=-1, keepdim=True)
    return (rewards - mean) / (std + eps)
