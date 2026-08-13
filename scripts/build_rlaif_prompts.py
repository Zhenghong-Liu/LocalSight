#!/usr/bin/env python3
"""构建 RLAIF rollout prompt 集（去掉末轮空 assistant，强制思考）。

用法（服务器）：
    PYTHONPATH=src python scripts/build_rlaif_prompts.py \
        --src /media/liuzh/data/DLData/LocalSight/rlaif.jsonl \
        --out data/processed/rlaif_prompts --tokenizer data/tokenizer
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer

from localsight.data.chat import drop_final_empty_assistant, format_chat


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--tokenizer", default="data/tokenizer")
    parser.add_argument("--max-len", type=int, default=4096)
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    ds = load_dataset("json", data_files=args.src, split="train", streaming=True)

    prompts: list[list[int]] = []
    prompt_lens: list[int] = []
    questions: list[str] = []
    for row in ds:
        conv = row.get("conversations")
        if not isinstance(conv, list):
            continue
        original = [dict(m) for m in conv]
        context = "\n".join(
            f"{m.get('role', '')}: {m.get('content', '')}" for m in original[:-1]
        )
        messages = drop_final_empty_assistant(original)
        text = format_chat(tokenizer, messages, add_generation_prompt=True, open_thinking=True)
        ids = tokenizer.encode(text)
        if len(ids) > args.max_len:
            continue
        prompts.append(ids)
        prompt_lens.append(len(ids))
        questions.append(context.strip())

    arr = np.full((len(prompts), args.max_len), 0, dtype=np.int32)
    for i, row in enumerate(prompts):
        arr[i, :len(row)] = row
    arr.tofile(out_dir / "prompts.bin")
    np.asarray(prompt_lens, dtype=np.int32).tofile(out_dir / "prompt_len.bin")
    with open(out_dir / "questions.jsonl", "w", encoding="utf-8") as f:
        for q in questions:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")
    manifest = {"source": args.src, "max_len": args.max_len, "prompts": len(prompts)}
    (out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
