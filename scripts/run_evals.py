#!/usr/bin/env python3
"""通用评测 CLI：多选（MMLU/CMMLU/C-Eval）与 GSM8K，支持思考开关。

用法：
    PYTHONPATH=src python scripts/run_evals.py \
        --checkpoint artifacts/pretrain/step-1000 --data-dir data/eval \
        --bench mmlu,gsm8k --open-thinking --limit 100
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from localsight.eval.benchmarks import load_benchmark, run_gsm8k, run_mc
from localsight.generation import generate
from localsight.model import LocalsightConfig, LocalsightForCausalLM
from localsight.tokenizer import LocalSightTokenizer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-dir", default="data/eval")
    parser.add_argument("--bench", default="mmlu,gsm8k")
    parser.add_argument("--tokenizer", default="data/tokenizer")
    parser.add_argument("--open-thinking", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=0.0)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LocalsightForCausalLM(LocalsightConfig()).to(device)
    model.load_state_dict(torch.load(Path(args.checkpoint) / "model.pt", map_location=device))
    model.eval()
    tokenizer = LocalSightTokenizer(args.tokenizer)

    def chat(text: str) -> str:
        think = "<think>\n" if args.open_thinking else "<think>\n\n</think>\n\n"
        prompt = f"<|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n{think}"
        ids = torch.tensor([tokenizer.encode(prompt)], device=device)
        gen = generate(model, ids, max_new_tokens=512, temperature=args.temperature,
                       top_p=0.9, eos_id=tokenizer.im_end_id)
        return tokenizer.decode(gen[0].tolist())

    results = {}
    for name in args.bench.split(","):
        name = name.strip()
        ds = load_benchmark(args.data_dir, name)
        if name == "gsm8k":
            results[name] = run_gsm8k(chat, ds, limit=args.limit)
        else:
            results[name] = run_mc(chat, name, ds, limit=args.limit)
        print(name, results[name])
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
