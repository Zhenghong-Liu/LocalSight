#!/usr/bin/env python3
"""RLAIF 阶段 A：用当前策略对全部 prompt 采样 K 条回答，保存为 jsonl。"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from localsight.model import KVCache, LocalsightConfig, LocalsightForCausalLM
from localsight.rl.agent_grpo import decode
from localsight.tokenizer import LocalSightTokenizer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--tokenizer", default="data/tokenizer")
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--max-new", type=int, default=512)
    args = parser.parse_args()

    rank = int(os.environ["LOCAL_RANK"])
    world = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(rank)
    dist.init_process_group(backend="nccl")
    device = torch.device(f"cuda:{rank}")

    model = LocalsightForCausalLM(LocalsightConfig()).to(device)
    model.load_state_dict(torch.load(Path(args.checkpoint) / "model.pt", map_location=device))
    model = DDP(model, device_ids=[rank])
    model.eval()
    tokenizer = LocalSightTokenizer(args.tokenizer)

    data_dir = Path(args.data_dir)
    prompts = np.memmap(data_dir / "prompts.bin", dtype=np.int32, mode="r")
    prompt_lens = np.memmap(data_dir / "prompt_len.bin", dtype=np.int32, mode="r")
    manifest = json.loads((data_dir / "manifest.json").read_text())
    n = manifest["prompts"]
    max_len = manifest["max_len"]
    out_file = Path(args.out) / f"samples_rank{rank}.jsonl"
    out_file.parent.mkdir(parents=True, exist_ok=True)

    cfg = LocalsightConfig()
    with open(out_file, "w", encoding="utf-8") as f:
        for pi in range(rank, n, world):
            plen = int(prompt_lens[pi])
            prompt_ids = torch.tensor(
                [prompts[pi * max_len:pi * max_len + plen].astype(np.int64).tolist()],
                device=device,
            )
            cache = KVCache(
                cfg.num_hidden_layers, 1, cfg.num_key_value_heads, cfg.head_dim,
                args.max_new + plen + 8, dtype=torch.bfloat16, device=device,
            )
            with torch.autocast("cuda", dtype=torch.bfloat16):
                model(prompt_ids, cache=cache)
            cache.commit()
            children = cache.spawn(args.k)
            for child in children:
                ids, text = decode(model, child, prompt_ids[:, -1:], tokenizer,
                                   args.max_new, args.temperature, 0.95, {tokenizer.im_end_id})
                f.write(json.dumps({"idx": pi, "text": text}, ensure_ascii=False) + "\n")
            if pi % 1000 == 0 and rank == 0:
                print(f"rank0 sampled {pi}/{n}", flush=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
