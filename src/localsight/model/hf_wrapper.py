"""transformers 兼容包装：让 LocalSight 模型能被 HF Trainer / save_pretrained 使用。"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import torch

from .config import LocalsightConfig
from .transformer import LocalsightForCausalLM

try:
    from transformers import PretrainedConfig, PreTrainedModel
    from transformers.modeling_outputs import CausalLMOutputWithPast
except ImportError:  # 本地无 transformers 时仍可导入本模块的类型定义
    PretrainedConfig = object  # type: ignore[assignment,misc]
    PreTrainedModel = object  # type: ignore[assignment,misc]
    CausalLMOutputWithPast = None  # type: ignore[assignment,misc]


class LocalsightHFConfig(PretrainedConfig):
    model_type = "minimind"

    vocab_size = 6400
    hidden_size = 768
    num_hidden_layers = 8
    num_attention_heads = 8
    num_key_value_heads = 4
    head_dim = 96
    use_moe = True
    intermediate_size = 2048
    moe_intermediate_size = 2432
    num_experts = 4
    num_experts_per_tok = 1
    use_shared_expert = False
    tie_word_embeddings = True
    use_bias = False
    max_position_embeddings = 32768
    rope_theta = 1_000_000.0
    rope_scaling = None
    use_qk_norm = True
    norm_eps = 1e-6
    hidden_act = "silu"
    init_std = 0.02
    init_router_std = 0.01
    output_proj_scale = None
    attn_impl = "sdpa"

    def to_localsight(self) -> LocalsightConfig:
        fields = LocalsightConfig.__dataclass_fields__.keys()
        return LocalsightConfig.from_dict({k: getattr(self, k) for k in fields})


class LocalsightHFForCausalLM(PreTrainedModel):
    config_class = LocalsightHFConfig

    def __init__(self, config: LocalsightHFConfig):
        super().__init__(config)
        self.core = LocalsightForCausalLM(config.to_localsight())

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        document_ids: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        logits, loss, aux = self.core(
            input_ids,
            labels=labels,
            position_ids=position_ids,
            attention_mask=attention_mask,
            document_ids=document_ids,
        )
        return CausalLMOutputWithPast(loss=loss, logits=logits, past_key_values=None)

    def get_input_embeddings(self):
        return self.core.model.embed_tokens

    def set_input_embeddings(self, value):
        self.core.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.core.lm_head

    def save_pretrained(self, save_directory: str | Path, **kwargs):
        save_directory = Path(save_directory)
        save_directory.mkdir(parents=True, exist_ok=True)
        torch.save(self.core.state_dict(), save_directory / "pytorch_model.bin")
        self.config.save_pretrained(save_directory)

    @classmethod
    def from_pretrained(cls, model_dir: str | Path, **kwargs):
        model_dir = Path(model_dir)
        config = LocalsightHFConfig.from_pretrained(model_dir)
        model = cls(config)
        state = torch.load(model_dir / "pytorch_model.bin", map_location="cpu")
        model.core.load_state_dict(state)
        return model
