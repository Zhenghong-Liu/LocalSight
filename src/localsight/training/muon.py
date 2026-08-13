"""Muon 优化器（Moonlight 公式）与 AdamW 的混合实现。

规则：ndim>=2 的参数走 Muon（momentum + Newton-Schulz 正交化 + 形状缩放），
其余（embedding/norm/router 等）走 AdamW；两者共享 lr 与 wd。
"""
from __future__ import annotations

import math

import torch
from torch.optim import Optimizer

NS_COEFFS = (3.4445, -4.7750, 2.0315)


def zeropower_via_newtonschulz5(
    g: torch.Tensor,
    steps: int = 5,
    eps: float = 1e-7,
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """对 (A, B) 矩阵梯度做 Newton-Schulz 正交化，返回 O。"""
    if g.ndim < 2:
        raise ValueError("Muon 只处理 2D 矩阵参数")
    a, b, c = NS_COEFFS
    x = g.to(dtype=dtype)
    x = x / (x.norm() + eps)
    if x.size(0) > x.size(1):
        x = x.T
    for _ in range(steps):
        xx = x @ x.T
        x = a * x + (b * xx + c * xx @ xx) @ x
    if g.size(0) > g.size(1):
        x = x.T
    return x


class Muon(Optimizer):
    """Moonlight Muon + AdamW。矩阵参数 ndim>=2 → Muon；其余 → AdamW。"""

    def __init__(
        self,
        params,
        lr: float = 2e-3,
        momentum: float = 0.95,
        ns_steps: int = 5,
        muon_scale: float = 0.2,
        wd: float = 0.1,
        betas: tuple[float, float] = (0.9, 0.95),
        eps: float = 1e-8,
    ):
        defaults = dict(
            lr=lr,
            momentum=momentum,
            ns_steps=ns_steps,
            muon_scale=muon_scale,
            wd=wd,
            betas=betas,
            eps=eps,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):  # noqa: D102
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr = group["lr"]
            wd = group["wd"]
            momentum = group["momentum"]
            ns_steps = group["ns_steps"]
            muon_scale = group["muon_scale"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                state = self.state[p]

                if grad.ndim >= 2:
                    if "momentum" not in state:
                        state["momentum"] = torch.zeros_like(p, dtype=torch.bfloat16)
                    m = state["momentum"]
                    m.mul_(momentum).add_(grad.to(torch.bfloat16))
                    x = momentum * m + grad.to(torch.bfloat16)  # Nesterov
                    o = zeropower_via_newtonschulz5(x, steps=ns_steps)
                    scale = muon_scale * math.sqrt(max(grad.shape))
                    p.add_(o.to(p.dtype), alpha=-lr * scale)
                    if wd != 0:
                        p.mul_(1 - lr * wd)
                else:
                    if "exp_avg" not in state:
                        state["step"] = 0
                        state["exp_avg"] = torch.zeros_like(p, dtype=torch.float32)
                        state["exp_avg_sq"] = torch.zeros_like(p, dtype=torch.float32)
                    state["step"] += 1
                    exp_avg = state["exp_avg"]
                    exp_avg_sq = state["exp_avg_sq"]
                    g = grad.to(torch.float32)
                    exp_avg.mul_(beta1).add_(g, alpha=1 - beta1)
                    exp_avg_sq.mul_(beta2).addcmul_(g, g, value=1 - beta2)
                    bias1 = 1 - beta1 ** state["step"]
                    bias2 = 1 - beta2 ** state["step"]
                    denom = (exp_avg_sq / bias2).sqrt().add_(eps)
                    p.addcdiv_(exp_avg / bias1, denom, value=-lr)
                    if wd != 0:
                        p.mul_(1 - lr * wd)
        return loss
