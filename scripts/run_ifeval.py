#!/usr/bin/env python3
"""IFEval 轻量版：关键词/开头/结尾等可规则化指令的宽松判定。"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import torch

from localsight.generation import generate
from localsight.model import LocalsightConfig, LocalsightForCausalLM
from localsight.tokenizer import LocalSightTokenizer


def check_instruction(prompt: str, output: str) -> bool:
    out = output.strip().lower()
    ok = True
    if "以“" in prompt:
        start = re.search(r"以“(.+?)”开头", prompt)
        if start and not out.startswith(start.group(1).lower()):
            ok = False
    if "以“" in prompt and "结尾" in prompt:
        end = re.search(r"以“(.+?)”结尾", prompt)
        if end and not out.endswith(end.group(1).lower()):
            ok = False
    if "包含关键词" in prompt:
        kw = re.search(r"包含关键词“(.+?)”", prompt)
        if kw and kw.group(1).lower() not in out:
            ok = False
    if "回答“是”或“否”" in prompt or "回答是或否" in prompt:
        if not re.search(r"^(是|否)[。.，, ]", out):
            ok = False
    return ok


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-dir", default="data/eval/ifeval")
    parser.add_argument("--tokenizer", default="data/tokenizer")
    parser.add_argument("--limit", type=int, default=100)
    args = parser.parse_args()

    device = torch.device("cuda:0")
    model = LocalsightForCausalLM(LocalsightConfig()).to(device)
    model.load_state_dict(torch.load(Path(args.checkpoint) / "model.pt", map_location=device))
    model.eval()
    tokenizer = LocalSightTokenizer(args.tokenizer)

    from datasets import load_from_disk

    ds = load_from_disk(args.data_dir)
    correct = total = 0
    for i, row in enumerate(ds):
        if i >= args.limit:
            break
        prompt = row["prompt"]
        ids = torch.tensor([tokenizer.encode("<|im_start|>user\n" + prompt + "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n")], device=device)
        gen = generate(model, ids, max_new_tokens=256, temperature=0, eos_id=tokenizer.im_end_id)
        out = tokenizer.decode(gen[0].tolist())
        correct += check_instruction(prompt, out)
        total += 1
    print(json.dumps({"acc": correct / max(total, 1), "n": total}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
