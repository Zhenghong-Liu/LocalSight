"""Pretrain 主循环：DDP + Muon/AdamW + bf16 + MoE 偏置负载均衡 + model soup。"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

from localsight.data import PretrainDataset
from localsight.model import LocalsightForCausalLM
from localsight.training.muon import Muon
from localsight.utils.config import resolve_stage_config

N_PARAMS = 198_416_640


def setup_distributed() -> tuple[int, int]:
    rank = int(os.environ["LOCAL_RANK"])
    world = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(rank)
    dist.init_process_group(backend="nccl")
    return rank, world


def cosine_schedule(step: int, warmup: int, total: int, lr: float, lr_min_ratio: float) -> float:
    if step < warmup:
        return lr * (step + 1) / max(1, warmup)
    if step >= total:
        return lr * lr_min_ratio
    progress = (step - warmup) / max(1, total - warmup)
    return lr * (lr_min_ratio + 0.5 * (1 - lr_min_ratio) * (1 + math.cos(math.pi * progress)))


def save_checkpoint(
    model: DDP,
    optimizer: Muon,
    path: Path,
    step: int,
    cfg: dict,
    rank: int,
) -> None:
    if rank != 0:
        return
    path.mkdir(parents=True, exist_ok=True)
    torch.save(model.module.state_dict(), path / "model.pt")
    torch.save(optimizer.state_dict(), path / "optimizer.pt")
    (path / "state.json").write_text(json.dumps({"step": step, "cfg": cfg}, ensure_ascii=False, indent=2))


def soup_checkpoints(ckpt_dir: Path, out: Path, keep: int = 3) -> None:
    ckpts = sorted(ckpt_dir.glob("step-*"), key=lambda p: int(p.name.split("-")[1]))
    if len(ckpts) < 2:
        return
    selected = ckpts[-keep:]
    states = [torch.load(p / "model.pt", map_location="cpu") for p in selected]
    soup = {}
    for key in states[0]:
        soup[key] = sum(s[key].float() for s in states) / len(states)
    out.mkdir(parents=True, exist_ok=True)
    torch.save(soup, out / "model.pt")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-dir", required=True, help="已构建的 processed/pretrain 目录")
    parser.add_argument("--max-steps", type=int, default=None, help="限制优化器步数（开发/冒烟用）")
    parser.add_argument("--micro-batch", type=int, default=None, help="覆盖每卡微批次")
    parser.add_argument("--grad-accum", type=int, default=None, help="覆盖梯度累积步数")
    parser.add_argument("--lr", type=float, default=None, help="覆盖学习率")
    parser.add_argument("--wd", type=float, default=None, help="覆盖权重衰减")
    parser.add_argument("--val-sequences", type=int, default=200, help="数据集尾部用于验证")
    parser.add_argument("--compile", action="store_true", help="torch.compile 训练环（max-autotune）")
    args = parser.parse_args()

    rank, world = setup_distributed()
    cfg, model_cfg = resolve_stage_config(Path(args.config))
    if args.micro_batch:
        cfg["micro_batch_size"] = args.micro_batch
    if args.grad_accum:
        cfg["grad_accum"] = args.grad_accum
    if args.lr is not None:
        cfg["lr"] = args.lr
    if args.wd is not None:
        cfg["wd"] = args.wd
    torch.manual_seed(cfg.get("seed", 42) + rank)

    model = LocalsightForCausalLM(model_cfg).to(f"cuda:{rank}")  # 主权重 fp32，计算走 bf16 autocast
    if cfg.get("activation_recompute") == "full":
        model.model.gradient_checkpointing = True

    matrix_params, other_params = [], []
    for name, param in model.named_parameters():
        if param.ndim >= 2 and "mlp.gate" not in name:
            matrix_params.append(param)
        else:
            other_params.append(param)
    optimizer = Muon(
        [
            {"params": matrix_params, "use_muon": True},
            {"params": other_params, "use_muon": False},  # embedding/norm/router/1D → AdamW
        ],
        lr=cfg["lr"],
        momentum=cfg["optimizer"]["muon_momentum"],
        ns_steps=cfg["optimizer"]["muon_ns_steps"],
        muon_scale=cfg["optimizer"]["muon_scale"],
        wd=cfg["wd"],
        betas=tuple(cfg["optimizer"]["adam_betas"]),
    )

    if args.compile:
        # 与激活重计算共存时禁用 cudagraphs（否则梯度张量被后续运行覆盖）
        model = torch.compile(model, mode="max-autotune-no-cudagraphs")
    model = DDP(model, device_ids=[rank], find_unused_parameters=True)

    dataset = PretrainDataset(Path(args.data_dir))
    val_indices = list(range(max(0, len(dataset) - args.val_sequences), len(dataset)))
    train_indices = list(range(max(0, len(dataset) - args.val_sequences)))
    train_subset = torch.utils.data.Subset(dataset, train_indices)
    sampler = DistributedSampler(
        train_subset,
        num_replicas=world,
        rank=rank,
        shuffle=True,
        seed=cfg.get("seed", 42),
    )
    loader = DataLoader(train_subset, batch_size=cfg["micro_batch_size"], sampler=sampler, pin_memory=True)

    total_steps = len(loader) * cfg["epochs"] // cfg["grad_accum"]
    warmup = int(total_steps * cfg["warmup_ratio"])
    routing = cfg["routing"]
    step = 0
    start = time.time()
    window_tokens = 0
    window_start = start
    acc_counts = torch.zeros(
        model_cfg.num_hidden_layers,
        model_cfg.num_experts,
        device=f"cuda:{rank}",
    )

    for epoch in range(cfg["epochs"]):
        sampler.set_epoch(epoch)
        for batch in loader:
            input_ids = batch["input_ids"].cuda(rank, non_blocking=True)
            labels = batch["labels"].cuda(rank, non_blocking=True)
            document_ids = batch["document_ids"].cuda(rank, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                _, loss, aux = model(input_ids, labels=labels, document_ids=document_ids)
                total_loss = loss + routing["z_loss_alpha"] * aux["z_loss"]
            acc_counts += aux["expert_counts"]
            window_tokens += input_ids.numel() * world
            total_loss = total_loss / cfg["grad_accum"]
            total_loss.backward()

            if (step + 1) % cfg["grad_accum"] == 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"]).item()
                lr = cosine_schedule(step // cfg["grad_accum"], warmup, total_steps, cfg["lr"], cfg["lr_min_ratio"])
                for group in optimizer.param_groups:
                    group["lr"] = lr
                optimizer.step()
                optimizer.zero_grad()
                optimizer_step = step // cfg["grad_accum"]

                counts = acc_counts.clone()
                for i, layer in enumerate(model.module.model.layers):
                    layer.mlp.gate.update_balance_bias(counts[i], gamma=routing["balance_gamma"])
                acc_counts.zero_()

                if rank == 0 and (optimizer_step + 1) % cfg["log_interval"] == 0:
                    dt = time.time() - window_start
                    tokens_per_sec = window_tokens / max(dt, 1e-6)
                    mfu = (6 * N_PARAMS * tokens_per_sec) / (2 * 330e12)
                    frac = (counts.sum(0) / counts.sum()).tolist()
                    print(
                        f"step={optimizer_step} loss={loss.item():.4f} z={aux['z_loss'].item():.4f} "
                        f"grad={grad_norm:.2f} lr={lr:.2e} tok/s={tokens_per_sec:.0f} "
                        f"mfu={mfu*100:.1f}% load={[round(f, 3) for f in frac]}"
                    )
                    window_tokens = 0
                    window_start = time.time()

                if (optimizer_step + 1) % cfg["save_interval"] == 0 and args.val_sequences > 0:
                    if rank == 0:
                        val_loader = DataLoader(
                            torch.utils.data.Subset(dataset, val_indices),
                            batch_size=4,
                        )
                        model.eval()
                        total, n = 0.0, 0
                        with torch.no_grad():
                            for vb in val_loader:
                                with torch.autocast("cuda", dtype=torch.bfloat16):
                                    _, vloss, _ = model(
                                        vb["input_ids"].cuda(rank),
                                        labels=vb["labels"].cuda(rank),
                                        document_ids=vb["document_ids"].cuda(rank),
                                    )
                                total += vloss.item() * vb["input_ids"].size(0)
                                n += vb["input_ids"].size(0)
                        model.train()
                        print(f"[val] step={optimizer_step + 1} val_loss={total / max(n, 1):.4f}")

                if (optimizer_step + 1) % cfg["save_interval"] == 0:
                    ckpt = Path(cfg["artifacts_dir"]) / "pretrain" / f"step-{optimizer_step + 1}"
                    save_checkpoint(model, optimizer, ckpt, optimizer_step + 1, cfg, rank)
                    dist.barrier()
                if args.max_steps is not None and optimizer_step + 1 >= args.max_steps:
                    break
            step += 1
        if args.max_steps is not None and step // cfg["grad_accum"] >= args.max_steps:
            break

    dist.barrier()
    soup_checkpoints(
        Path(cfg["artifacts_dir"]) / "pretrain",
        Path(cfg["artifacts_dir"]) / "pretrain" / "soup",
        keep=cfg["soup_last_n"],
    )
    if rank == 0:
        print(f"pretrain 完成，总耗时 {(time.time() - start)/3600:.2f} h")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
