#!/usr/bin/env python3
"""工具任务 held-out 评测（agent_prompts 中未参与 RL 的尾部 100 条）。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from localsight.model import KVCache, LocalsightConfig, LocalsightForCausalLM
from localsight.rl.agent_grpo import decode_batch
from localsight.rl.rewards import composite_reward
from localsight.tokenizer import LocalSightTokenizer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--tokenizer", default="data/tokenizer")
    parser.add_argument("--offset", type=int, default=4000)
    parser.add_argument("--n", type=int, default=100)
    args = parser.parse_args()

    device = torch.device("cuda:0")
    model = LocalsightForCausalLM(LocalsightConfig()).to(device)
    model.load_state_dict(torch.load(Path(args.checkpoint) / "model.pt", map_location=device))
    model.eval()
    tokenizer = LocalSightTokenizer(args.tokenizer)

    data_dir = Path(args.data_dir)
    prompts = np.memmap(data_dir / "prompts.bin", dtype=np.int32, mode="r")
    lens = np.memmap(data_dir / "prompt_len.bin", dtype=np.int32, mode="r")
    manifest = json.loads((data_dir / "manifest.json").read_text())
    max_len = manifest["max_len"]
    gt_rows = [json.loads(line) for line in open(data_dir / "gt.jsonl", encoding="utf-8")][args.offset:args.offset + args.n]

    total_reward = 0.0
    hits = 0
    for i, gt in enumerate(gt_rows):
        pi = args.offset + i
        plen = int(lens[pi])
        ids = torch.tensor([prompts[pi * max_len:pi * max_len + plen].astype(np.int64).tolist()], device=device)[:, :2048]
        ids = ids.repeat(4, 1)
        cache = KVCache(8, 4, 4, 96, 2048 + 512 + 8, dtype=torch.bfloat16, device=device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            model(ids, cache=cache)
        cache.commit()
        _, texts = decode_batch(model, cache, ids[:, -1:], tokenizer, 512, 0.8, 0.95, tokenizer.im_end_id)
        for text in texts:
            reward = composite_reward("<think>\n" + text, gt["expect_tool"], gt["gt"])
            total_reward += reward
            hits += reward > 0.9
    print(json.dumps({
        "n": args.n * 4,
        "mean_reward": total_reward / (args.n * 4),
        "gt_hit": hits / (args.n * 4),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
