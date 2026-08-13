#!/usr/bin/env python3
"""GPU 环境冒烟测试（Stage 0 前置）。

用法（服务器）：
    torchrun --nproc_per_node=2 scripts/smoke_test.py
"""
from __future__ import annotations

import os

import torch
import torch.distributed as dist


def main() -> None:
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world = dist.get_world_size()

    x = torch.full((8, 8), float(rank + 1), device=f"cuda:{local_rank}")
    dist.all_reduce(x)
    expected = world * (world + 1) / 2
    ok = bool((x == expected).all().item())

    if rank == 0:
        print(
            f"world_size={world} allreduce_ok={ok} "
            f"torch={torch.__version__} cuda={torch.version.cuda} "
            f"gpus={torch.cuda.device_count()}"
        )

    dist.destroy_process_group()
    assert ok, "NCCL all-reduce 校验失败"


if __name__ == "__main__":
    main()
