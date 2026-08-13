"""KV cache 与共享 prompt 前缀缓存。

设计约定：
- 每层缓存形状 (batch, max_len, n_kv_heads, head_dim)，存 RoPE 之后的 K/V；
- 一次 forward 内 `append` 写入本步窗口，`commit` 才推进 `length`；
- `spawn` 把已缓存的 prompt 前缀复制成 G 份独立缓存，供 RL 采样复用。
"""
from __future__ import annotations

import torch


class KVCache:
    def __init__(
        self,
        num_layers: int,
        batch_size: int,
        num_kv_heads: int,
        head_dim: int,
        max_len: int,
        dtype: torch.dtype = torch.bfloat16,
        device: torch.device = torch.device("cuda"),
    ):
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.max_len = max_len
        self.dtype = dtype
        self.device = device
        shape = (batch_size, max_len, num_kv_heads, head_dim)
        self._k = [torch.zeros(*shape, dtype=dtype, device=device) for _ in range(num_layers)]
        self._v = [torch.zeros(*shape, dtype=dtype, device=device) for _ in range(num_layers)]
        self.length = 0
        self._pending: int | None = None

    def append(self, layer_idx: int, k: torch.Tensor, v: torch.Tensor) -> None:
        s = k.shape[1]
        if self._pending is None:
            self._pending = s
        if s != self._pending:
            raise ValueError("同一次 forward 中所有层的步长必须一致")
        if self.length + s > self.max_len:
            raise RuntimeError(f"KV cache 溢出：{self.length + s} > {self.max_len}")
        self._k[layer_idx][:, self.length:self.length + s].copy_(k)
        self._v[layer_idx][:, self.length:self.length + s].copy_(v)

    def k(self, layer_idx: int) -> torch.Tensor:
        upto = self.length + (self._pending or 0)
        return self._k[layer_idx][:, :upto]

    def v(self, layer_idx: int) -> torch.Tensor:
        upto = self.length + (self._pending or 0)
        return self._v[layer_idx][:, :upto]

    def commit(self) -> None:
        if self._pending is None:
            raise RuntimeError("没有待提交的步长")
        self.length += self._pending
        self._pending = None

    def spawn(self, group_size: int) -> list["KVCache"]:
        """把当前已提交的前缀复制成 group_size 份独立缓存（batch=1）。"""
        if self.batch_size != 1:
            raise ValueError("只有 batch=1 的前缀缓存才能 spawn")
        caches = []
        for _ in range(group_size):
            child = KVCache(
                self.num_layers,
                1,
                self.num_kv_heads,
                self.head_dim,
                self.max_len,
                dtype=self.dtype,
                device=self.device,
            )
            for i in range(self.num_layers):
                child._k[i][:, :self.length].copy_(self._k[i][:, :self.length])
                child._v[i][:, :self.length].copy_(self._v[i][:, :self.length])
            child.length = self.length
            caches.append(child)
        return caches
