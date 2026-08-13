"""MoE：top-1 路由 + 专家分桶前向 + aux-loss-free 偏置负载均衡 + z-loss。

- routing bias 是 buffer，不参与梯度；训练循环每步调 `update_balance_bias`。
- 每步返回 aux：z_loss、balance_loss、expert_counts，由上层收集监控/加权。
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import LocalsightConfig


class ExpertMLP(nn.Module):
    """单个 SwiGLU 专家：gate/up 合并算力由训练期的融合内核负责，此处保持结构清晰。"""

    def __init__(self, config: LocalsightConfig):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.moe_intermediate_size,
                                   bias=config.use_bias)
        self.up_proj = nn.Linear(config.hidden_size, config.moe_intermediate_size,
                                 bias=config.use_bias)
        self.down_proj = nn.Linear(config.moe_intermediate_size, config.hidden_size,
                                   bias=config.use_bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class MoEGate(nn.Module):
    def __init__(self, config: LocalsightConfig):
        super().__init__()
        self.n_experts = config.num_experts
        self.top_k = config.num_experts_per_tok
        self.weight = nn.Parameter(torch.empty(self.n_experts, config.hidden_size))
        nn.init.normal_(self.weight, mean=0.0, std=config.init_router_std)
        self.register_buffer("bias", torch.zeros(self.n_experts))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, dict]:
        logits = F.linear(x, self.weight, self.bias)  # bias 仅参与路由，不接收梯度
        probs = F.softmax(logits, dim=-1)
        topk_idx = probs.topk(self.top_k, dim=-1).indices  # (N,K)
        weights = probs.gather(-1, topk_idx)

        z_loss = torch.logsumexp(logits, dim=-1).pow(2).mean()
        counts = torch.zeros(self.n_experts, device=x.device)
        for e in range(self.n_experts):
            counts[e] = (topk_idx == e).sum()
        return topk_idx, weights, {
            "z_loss": z_loss,
            "expert_counts": counts,
            "probs": probs.detach(),
            "topk_idx": topk_idx.detach(),
        }

    @torch.no_grad()
    def update_balance_bias(self, counts: torch.Tensor, gamma: float = 1e-3) -> None:
        """DeepSeek-V3 偏置式负载均衡：超载专家降偏置，欠载专家升偏置。"""
        target = counts.mean()
        self.bias.sub_(gamma * (counts - target))


def load_balance_loss(probs: torch.Tensor, topk_idx: torch.Tensor, n_experts: int) -> torch.Tensor:
    """经典 balance aux loss（坍塌兜底用，正常训练权重为 0）。"""
    mask = F.one_hot(topk_idx, num_classes=n_experts).float()  # (N,K,E)
    f_e = mask.mean(dim=(0, 1))  # E
    p_e = probs.mean(dim=0)  # E
    return n_experts * (p_e * f_e).sum()


class MoEFeedForward(nn.Module):
    def __init__(self, config: LocalsightConfig):
        super().__init__()
        self.config = config
        self.n_experts = config.num_experts
        self.experts = nn.ModuleList([ExpertMLP(config) for _ in range(self.n_experts)])
        self.gate = MoEGate(config)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict]:
        b, s, d = x.shape
        xf = x.view(-1, d)
        topk_idx, weights, aux = self.gate(xf)
        out = torch.zeros_like(xf)

        if self.config.num_experts_per_tok == 1:
            for e in range(self.n_experts):
                mask = topk_idx[:, 0] == e
                if mask.any():
                    out[mask] = self.experts[e](xf[mask]) * weights[mask, 0, None]
        else:
            for e in range(self.n_experts):
                hits = (topk_idx == e).any(dim=-1)
                if hits.any():
                    w = torch.where(topk_idx[hits] == e, weights[hits], 0.0)
                    out[hits] += self.experts[e](xf[hits]) * w.sum(-1, keepdim=True)

        aux["balance_loss"] = load_balance_loss(aux["probs"], aux["topk_idx"], self.n_experts)
        return out.view(b, s, d), aux


class DenseMLP(nn.Module):
    """非 MoE 兜底 FFN（SwiGLU）。"""

    def __init__(self, config: LocalsightConfig):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size,
                                   bias=config.use_bias)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size,
                                 bias=config.use_bias)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size,
                                   bias=config.use_bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))
