#!/usr/bin/env python3
"""思考开关评测 CLI：对一组 prompt 分别开/关思考生成并输出统计。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from localsight.eval.chat_harness import ThinkingEvalResult, aggregate, extract_think
from localsight.generation import generate
from localsight.model import KVCache, LocalsightConfig, LocalsightForCausalLM
from localsight.tokenizer import LocalSightTokenizer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--prompts", required=True)
    parser.add_argument("--tokenizer", default="data/tokenizer")
    parser.add_argument("--max-new", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--open-thinking", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = LocalsightConfig()
    model = LocalsightForCausalLM(config).to(device)
    model.load_state_dict(torch.load(Path(args.checkpoint) / "model.pt", map_location=device))
    model.eval()
    tokenizer = LocalSightTokenizer(args.tokenizer)

    prompts = Path(args.prompts).read_text(encoding="utf-8").splitlines()
    results = []
    for prompt in prompts:
        if not prompt.strip():
            continue
        think = "<think>\n" if args.open_thinking else "<think>\n\n</think>\n\n"
        text = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n{think}"
        ids = torch.tensor([tokenizer.encode(text)], device=device)
        gen = generate(model, ids, max_new_tokens=args.max_new, temperature=args.temperature,
                       top_p=0.9, eos_id=tokenizer.im_end_id)
        output = tokenizer.decode(gen[0].tolist())
        results.append({
            "prompt": prompt,
            "output": output,
            "think_chars": len(extract_think(output)),
            "len": len(output),
        })
    stats = {"n": len(results), "mean_len": sum(r["len"] for r in results) / max(len(results), 1),
             "think_trigger_rate": sum(bool(r["think_chars"]) for r in results) / max(len(results), 1)}
    print(json.dumps({"stats": stats, "samples": results[:5]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
