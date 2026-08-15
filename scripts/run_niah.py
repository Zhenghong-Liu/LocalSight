#!/usr/bin/env python3
"""32k Needle-in-a-Haystack 评测。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from localsight.eval.niah import make_niah_prompt, niah_hit
from localsight.generation import generate
from localsight.model import LocalsightConfig, LocalsightForCausalLM
from localsight.tokenizer import LocalSightTokenizer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--tokenizer", default="data/tokenizer")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LocalsightForCausalLM(LocalsightConfig()).to(device)
    model.load_state_dict(torch.load(Path(args.checkpoint) / "model.pt", map_location=device))
    model.eval()
    tokenizer = LocalSightTokenizer(args.tokenizer)

    lengths = [4096, 8192, 16384, 32768]
    results = {}
    for length in lengths:
        hits = 0
        for seed in range(5):
            prompt, needle = make_niah_prompt(length, seed)
            ids = torch.tensor([tokenizer.encode("<|im_start|>user\n" + prompt + "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n")], device=device)
            gen = generate(model, ids, max_new_tokens=256, temperature=0, eos_id=tokenizer.im_end_id)
            out = tokenizer.decode(gen[0].tolist())
            hits += niah_hit(out, needle)
        results[str(length)] = {"hits": hits, "n": 5, "acc": hits / 5}
        print(length, results[str(length)])
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
