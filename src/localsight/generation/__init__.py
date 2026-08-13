"""推理采样：temperature/top-p、KV cache、prefix cache。"""

from .sampler import generate

__all__ = ["generate"]
