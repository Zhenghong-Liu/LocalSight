"""Localsight 模型配置。

字段与 configs/model/minimind3_moe_198m.yaml 一一对应；默认值即架构锁定值。
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class LocalsightConfig:
    # 基础结构
    model_type: str = "minimind"
    vocab_size: int = 6400
    hidden_size: int = 768
    num_hidden_layers: int = 8
    num_attention_heads: int = 8
    num_key_value_heads: int = 4
    head_dim: int = 96

    # FFN / MoE
    use_moe: bool = True
    intermediate_size: int = 2048            # 稠密 FFN 兜底
    moe_intermediate_size: int = 2432        # 每个专家 FFN：2432 → 198M/64M
    num_experts: int = 4
    num_experts_per_tok: int = 1             # top-1
    use_shared_expert: bool = False

    # 权重共享与偏置
    tie_word_embeddings: bool = True
    use_bias: bool = False

    # 位置编码
    max_position_embeddings: int = 32768
    rope_theta: float = 1_000_000.0
    rope_scaling: Optional[dict[str, Any]] = None

    # 归一化
    use_qk_norm: bool = True
    norm_eps: float = 1e-6
    hidden_act: str = "silu"

    # 初始化
    init_std: float = 0.02
    init_router_std: float = 0.01
    output_proj_scale: Optional[float] = None  # None → init_std / sqrt(2 * n_layers)

    # 注意力后端（当前实现 SDPA；fa2 后端接口预留）
    attn_impl: str = "sdpa"

    def __post_init__(self) -> None:
        self.validate()

    @property
    def scaled_init_std(self) -> float:
        if self.output_proj_scale is not None:
            return self.output_proj_scale
        return self.init_std / math.sqrt(2.0 * self.num_hidden_layers)

    def validate(self) -> None:
        if self.head_dim * self.num_attention_heads != self.hidden_size:
            raise ValueError("head_dim * num_attention_heads 必须等于 hidden_size")
        if self.num_attention_heads % self.num_key_value_heads != 0:
            raise ValueError("num_attention_heads 必须能被 num_key_value_heads 整除")
        if self.num_experts_per_tok > self.num_experts:
            raise ValueError("num_experts_per_tok 不能大于 num_experts")
        if self.use_shared_expert:
            raise NotImplementedError("当前架构不使用 shared expert")
        if self.attn_impl not in ("sdpa", "fa2", "fa3"):
            raise ValueError(f"不支持的 attn_impl: {self.attn_impl}")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "LocalsightConfig":
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        unknown = set(data) - known
        if unknown:
            raise ValueError(f"未知配置键: {sorted(unknown)}")
        return cls(**data)

    def to_dict(self) -> dict[str, Any]:
        return {
            f: getattr(self, f)
            for f in self.__dataclass_fields__  # type: ignore[attr-defined]
        }
