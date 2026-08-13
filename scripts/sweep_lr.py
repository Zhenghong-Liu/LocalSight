#!/usr/bin/env python3
"""Pretrain LR/wd 扫描（在 mini 语料上，单卡）。

用法：
    PYTHONPATH=src python scripts/sweep_lr.py --data-dir data/processed/pretrain-mini \
        --lrs 1e-3,2e-3,3e-3 --wds 0.05,0.1 --steps 500
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from localsight.data import PretrainDataset
from localsight.model import LocalsightForCausalLM, LocalsightConfig
from localsight.training.muon import Muon


@torch.no_grad()
def eval_loss(model: LocalsightForCausalLM, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    total = 0.0
    n = 0
    for batch in loader:
        ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        docs = batch["document_ids"].to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            _, loss, _ = model(ids, labels=labels, document_ids=docs)
        total += loss.item() * ids.size(0)
        n += ids.size(0)
    return total / max(n, 1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--lrs", default="1e-3,2e-3,3e-3")
    parser.add_argument("--wds", default="0.05,0.1")
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--val-sequences", type=int, default=200)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = PretrainDataset(Path(args.data_dir))
    val_indices = list(range(max(0, len(dataset) - args.val_sequences), len(dataset)))
    train_indices = list(range(max(0, len(dataset) - args.val_sequences)))
    val_loader = DataLoader(torch.utils.data.Subset(dataset, val_indices), batch_size=4)

    results = []
    for lr, wd in itertools.product([float(x) for x in args.lrs.split(",")],
                                    [float(x) for x in args.wds.split(",")]):
        torch.manual_seed(42)
        model = LocalsightForCausalLM(LocalsightConfig()).to(device)
        model.model.gradient_checkpointing = True  # batch 32×4096 需要激活重计算
        optimizer = Muon(model.parameters(), lr=lr, wd=wd)
        g = torch.Generator().manual_seed(42)
        sampler = torch.utils.data.RandomSampler(train_indices, generator=g)
        loader = DataLoader(torch.utils.data.Subset(dataset, train_indices),
                            batch_size=args.batch, sampler=sampler, drop_last=True)
        it = iter(loader)
        model.train()
        for step in range(args.steps):
            try:
                batch = next(it)
            except StopIteration:
                it = iter(loader)
                batch = next(it)
            ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            docs = batch["document_ids"].to(device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                _, loss, aux = model(ids, labels=labels, document_ids=docs)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()
            if step % 100 == 0:
                print(f"lr={lr} wd={wd} step={step} loss={loss.item():.4f}")
        val = eval_loss(model, val_loader, device)
        results.append({"lr": lr, "wd": wd, "train_loss": loss.item(), "val_loss": val})
        print(f"RESULT lr={lr} wd={wd} val_loss={val:.4f}")

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
