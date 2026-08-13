"""LocalSight 主干与 CausalLM 包装：198M-A64M MoE 解码器。"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .attention import Attention, build_document_causal_mask
from .config import LocalsightConfig
from .kv_cache import KVCache
from .moe import DenseMLP, MoEFeedForward
from .norms import RMSNorm


class LocalsightBlock(nn.Module):
    def __init__(self, config: LocalsightConfig, layer_idx: int):
        super().__init__()
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.norm_eps)
        self.self_attn = Attention(config, layer_idx)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.norm_eps)
        self.mlp = MoEFeedForward(config) if config.use_moe else DenseMLP(config)
        self.use_moe = config.use_moe

    def forward(
        self,
        hidden: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        document_ids: Optional[torch.Tensor] = None,
        cache: Optional[KVCache] = None,
    ) -> tuple[torch.Tensor, dict]:
        # 掩码在 checkpoint 内构建：重算而非作为每层输入保存（省显存）
        if attention_mask is None and document_ids is not None:
            attention_mask = build_document_causal_mask(document_ids, dtype=torch.bool)
        residual = hidden
        hidden = self.self_attn(
            self.input_layernorm(hidden),
            position_ids=position_ids,
            attention_mask=attention_mask,
            cache=cache,
        )
        hidden = residual + hidden

        residual = hidden
        if self.use_moe:
            hidden, aux = self.mlp(self.post_attention_layernorm(hidden))
        else:
            hidden = self.mlp(self.post_attention_layernorm(hidden))
            aux = {}
        return residual + hidden, aux


class LocalsightModel(nn.Module):
    def __init__(self, config: LocalsightConfig):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [LocalsightBlock(config, i) for i in range(config.num_hidden_layers)]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.norm_eps)
        self.gradient_checkpointing = False
        self.apply(self._init_weights)
        self._apply_scaled_init()

    def _init_weights(self, module: nn.Module) -> None:
        std = self.config.init_std
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=std)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=std)

    def _apply_scaled_init(self) -> None:
        cfg = self.config
        std_o = cfg.scaled_init_std
        for layer in self.layers:
            nn.init.normal_(layer.self_attn.o_proj.weight, mean=0.0, std=std_o)
            if cfg.use_moe:
                for expert in layer.mlp.experts:
                    nn.init.normal_(expert.down_proj.weight, mean=0.0, std=std_o)
                nn.init.normal_(layer.mlp.gate.weight, mean=0.0, std=cfg.init_router_std)
            else:
                nn.init.normal_(layer.mlp.down_proj.weight, mean=0.0, std=std_o)

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        document_ids: Optional[torch.Tensor] = None,
        cache: Optional[KVCache] = None,
    ) -> tuple[torch.Tensor, dict]:
        hidden = self.embed_tokens(input_ids)
        z_loss = torch.zeros((), device=hidden.device)
        balance_loss = torch.zeros((), device=hidden.device)
        counts = []
        for layer in self.layers:
            if self.gradient_checkpointing and self.training:
                hidden, aux = checkpoint(
                    layer,
                    hidden,
                    position_ids,
                    attention_mask,
                    document_ids,
                    cache,
                    use_reentrant=False,
                )
            else:
                hidden, aux = layer(
                    hidden,
                    position_ids=position_ids,
                    attention_mask=attention_mask,
                    document_ids=document_ids,
                    cache=cache,
                )
            z_loss = z_loss + aux.get("z_loss", 0.0)
            balance_loss = balance_loss + aux.get("balance_loss", 0.0)
            counts.append(aux.get("expert_counts"))
        hidden = self.norm(hidden)
        aux = {
            "z_loss": z_loss,
            "balance_loss": balance_loss,
            "expert_counts": torch.stack(counts) if counts and counts[0] is not None else None,
        }
        return hidden, aux


class LocalsightForCausalLM(nn.Module):
    def __init__(self, config: LocalsightConfig):
        super().__init__()
        self.config = config
        self.model = LocalsightModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=config.use_bias)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        document_ids: Optional[torch.Tensor] = None,
        cache: Optional[KVCache] = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], dict]:
        hidden, aux = self.model(
            input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            document_ids=document_ids,
            cache=cache,
        )
        logits = self.lm_head(hidden)
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )
        return logits, loss, aux


def count_parameters(model: nn.Module) -> tuple[int, int]:
    """返回 (总参数, 激活参数)。激活参数 = 非专家参数 + 每个 MoE 层的 0 号专家。"""
    total = 0
    active = 0
    for name, param in model.named_parameters():
        n = param.numel()
        total += n
        if ".experts." in name:
            if ".experts.0." in name:
                active += n
        else:
            active += n
    return total, active
