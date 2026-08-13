#!/usr/bin/env python3
"""Stage 0 冒烟测试（正式训练前必跑）。

用法：
    torchrun --nproc_per_node=2 scripts/stage0_smoke.py --compile

覆盖：数值训练步、DDP、专家负载、显存峰值、NCCL P2P、compile 稳定性、MFU。
"""
from __future__ import annotations

import argparse
import os
import time

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from localsight.model import LocalsightConfig, LocalsightForCausalLM

N_PARAMS = 198_416_640
PEAK_BF16_TFLOPS = 2 * 330.0  # 2x RTX 4090


def setup(rank: int, world_size: int) -> None:
    torch.cuda.set_device(rank)
    dist.init_process_group(backend="nccl", init_method="env://")
    if rank == 0:
        print(f"[stage0] world={world_size} torch={torch.__version__} cuda={torch.version.cuda}")


def grad_norm(model: torch.nn.Module) -> float:
    total = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total += p.grad.detach().float().pow(2).sum().item()
    return total**0.5


def test_p2p(rank: int) -> None:
    if rank != 0:
        return
    if torch.cuda.device_count() < 2:
        print("[stage0] P2P: 单卡跳过")
        return
    peer_ok = torch.cuda.can_device_access_peer(0, 1)
    print(f"[stage0] P2P device access: {peer_ok}")
    if not peer_ok:
        print("[stage0] P2P 不可用：训练脚本建议 NCCL_P2P_DISABLE=1")
        return
    size_mb = 256
    src = torch.randn(size_mb * 1024 * 1024 // 4, device="cuda:0")
    dst = torch.empty_like(src, device="cuda:1")
    for _ in range(3):
        dst.copy_(src, non_blocking=True)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(10):
        dst.copy_(src, non_blocking=True)
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    gb_s = (size_mb * 10) / (dt * 1000)
    print(f"[stage0] P2P copy bandwidth: {gb_s:.1f} GB/s")


def bench_compile(model: torch.nn.Module, rank: int, batch: int, seq: int) -> None:
    if rank != 0:
        return
    model = model.module
    ids = torch.randint(0, 6400, (batch, seq), device="cuda:0")

    def run(m, iters: int) -> float:
        for _ in range(3):
            m(ids)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            m(ids)
        torch.cuda.synchronize()
        return (time.perf_counter() - t0) / iters

    model.eval()
    with torch.no_grad():
        eager = run(model, 10)
        compiled = None
        try:
            c_model = torch.compile(model, mode="max-autotune")
            compiled = run(c_model, 10)
        except Exception as exc:  # noqa: BLE001
            print(f"[stage0] torch.compile 失败: {exc}")
        if compiled is not None:
            print(f"[stage0] forward: eager={eager*1000:.1f}ms compile={compiled*1000:.1f}ms")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--compile", action="store_true", help="额外跑 compile 稳定性测试")
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--seq", type=int, default=128)
    args = parser.parse_args()

    rank = int(os.environ["LOCAL_RANK"])
    world = int(os.environ["WORLD_SIZE"])
    setup(rank, world)
    device = torch.device(f"cuda:{rank}")

    cfg = LocalsightConfig()
    model = LocalsightForCausalLM(cfg).to(device=device)
    ddp = DDP(model, device_ids=[rank])
    optimizer = torch.optim.AdamW(ddp.parameters(), lr=1e-3, betas=(0.9, 0.95))

    torch.cuda.reset_peak_memory_stats(rank)
    ids = torch.randint(0, cfg.vocab_size, (args.batch, args.seq), device=device)

    t0 = time.perf_counter()
    for step in range(args.steps):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits, loss, aux = ddp(ids, labels=ids)
        total = loss + 1e-3 * aux["z_loss"]
        total.backward()
        gn = grad_norm(ddp)
        torch.nn.utils.clip_grad_norm_(ddp.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()
        if rank == 0:
            counts = aux["expert_counts"].mean(0)  # (E,)
            frac = (counts / counts.sum()).tolist()
            print(
                f"[stage0] step={step} loss={loss.item():.4f} z={aux['z_loss'].item():.4f} "
                f"grad={gn:.3f} load={[round(f, 3) for f in frac]}"
            )
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    tokens = args.batch * args.seq * world * args.steps
    flops = 6 * N_PARAMS * tokens
    mfu = flops / (PEAK_BF16_TFLOPS * 1e12 * elapsed)
    if rank == 0:
        print(f"[stage0] 吞吐: {tokens / elapsed:.0f} tok/s | 单卡: {tokens / elapsed / world:.0f} tok/s")
        print(f"[stage0] MFU: {mfu*100:.1f}%（目标 ≥45%，<40% 需 profiling）")
        print(f"[stage0] 显存峰值: {torch.cuda.max_memory_allocated(rank)/1024**3:.2f} GiB")

    test_p2p(rank)
    if args.compile:
        bench_compile(ddp, rank, args.batch, args.seq)

    dist.barrier()
    if rank == 0:
        print("[stage0] PASS")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
