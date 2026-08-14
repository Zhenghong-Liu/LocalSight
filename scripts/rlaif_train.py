#!/usr/bin/env python3
"""RLAIF 阶段 C：按 score 组对（best vs worst）做一轮 SimPO。"""
from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

from localsight.model import LocalsightConfig, LocalsightForCausalLM
from localsight.rl.losses import simpo_loss
from localsight.tokenizer import LocalSightTokenizer
from localsight.utils.config import resolve_stage_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scored", required=True)
    parser.add_argument("--prompts-dir", required=True)
    parser.add_argument("--config", default="configs/rlaif.yaml")
    parser.add_argument("--start-checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--tokenizer", default="data/tokenizer")
    parser.add_argument("--epochs", type=int, default=1)
    args = parser.parse_args()

    rank = int(os.environ["LOCAL_RANK"])
    world = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(rank)
    dist.init_process_group(backend="nccl")
    device = torch.device(f"cuda:{rank}")

    cfg, model_cfg = resolve_stage_config(Path(args.config))
    tokenizer = LocalSightTokenizer(args.tokenizer)
    model = LocalsightForCausalLM(model_cfg).to(device)
    model.load_state_dict(torch.load(Path(args.start_checkpoint) / "model.pt", map_location=device))
    model = DDP(model, device_ids=[rank], find_unused_parameters=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=0.0)

    prompts = np.memmap(Path(args.prompts_dir) / "prompts.bin", dtype=np.int32, mode="r")
    prompt_lens = np.memmap(Path(args.prompts_dir) / "prompt_len.bin", dtype=np.int32, mode="r")
    max_len = json.loads((Path(args.prompts_dir) / "manifest.json").read_text())["max_len"]

    groups: dict[int, list[dict]] = defaultdict(list)
    for line in Path(args.scored).read_text(encoding="utf-8").splitlines():
        if line.strip():
            rec = json.loads(line)
            groups[rec["idx"]].append(rec)

    pairs = []
    for idx, recs in groups.items():
        recs.sort(key=lambda r: r.get("score", 0.0), reverse=True)
        best, worst = recs[0], recs[-1]
        plen = int(prompt_lens[idx])
        prefix = prompts[idx * max_len:idx * max_len + plen].astype(np.int64).tolist()
        pairs.append((prefix, tokenizer.encode(best["text"]), tokenizer.encode(worst["text"])))

    step = 0
    model.train()
    for epoch in range(args.epochs):
        for i in range(rank, len(pairs), world):
            prefix, chosen, rejected = pairs[i]
            ids_c = torch.tensor([prefix + chosen], device=device)
            ids_r = torch.tensor([prefix + rejected], device=device)
            pl = torch.tensor([len(prefix)], device=device)
            max_len_pair = max(ids_c.shape[1], ids_r.shape[1])
            ids_c = torch.nn.functional.pad(ids_c, (0, max_len_pair - ids_c.shape[1]), value=0)
            ids_r = torch.nn.functional.pad(ids_r, (0, max_len_pair - ids_r.shape[1]), value=0)
            resp_len_c = torch.tensor([len(chosen)], device=device)
            resp_len_r = torch.tensor([len(rejected)], device=device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                ids = torch.cat([ids_c, ids_r], dim=0)
                logits = model(ids)[0]
                shift = F.log_softmax(logits, dim=-1)[:, :-1].gather(-1, ids[:, 1:, None]).squeeze(-1)
                positions = torch.arange(shift.shape[1], device=device)[None]
                start = (torch.cat([pl, pl]) - 1)[:, None]
                end = start + torch.cat([resp_len_c, resp_len_r])[:, None]
                mask = (positions >= start) & (positions < end)
                resp_len = mask.sum(-1).clamp(min=1)
                pi = (shift * mask).sum(-1) / resp_len
                pi_c, pi_r = pi[:1], pi[1:]
                loss = simpo_loss(pi_c[:, None], pi_r[:, None], beta=cfg["beta"], gamma=cfg["gamma"])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()
            if rank == 0 and step % 20 == 0:
                print(f"step={step} loss={loss.item():.4f} margin={(pi_c - pi_r).item():.4f}", flush=True)
            step += 1

    dist.barrier()
    if rank == 0:
        out = Path(args.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        torch.save(model.module.state_dict(), out / "model.pt")
        print("rlaif train done:", out / "model.pt")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
