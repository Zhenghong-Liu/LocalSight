#!/usr/bin/env python3
"""构建 Agent RL 的 rollout prompt 集（只用带 tools+gt 的 2 万条）。

用法（服务器）：
    PYTHONPATH=src python scripts/build_agent_prompts.py \
        --src /media/liuzh/data/DLData/LocalSight/agent_rl.jsonl \
        --out data/processed/agent_prompts --tokenizer data/tokenizer
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer

from localsight.data.chat import drop_final_empty_assistant, extract_tools, format_chat


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--tokenizer", default="data/tokenizer")
    parser.add_argument("--max-len", type=int, default=8192)
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    ds = load_dataset("json", data_files=args.src, split="train", streaming=True)

    prompts: list[list[int]] = []
    prompt_lens: list[int] = []
    gt_rows: list[list[str]] = []
    expect_tool: list[bool] = []
    for row in ds:
        conv = row.get("conversations")
        gt = row.get("gt")
        if not isinstance(conv, list) or not isinstance(gt, list) or not gt:
            continue
        messages = drop_final_empty_assistant([dict(m) for m in conv])
        messages, tools = extract_tools(messages)
        text = format_chat(
            tokenizer,
            messages,
            tools=tools,
            add_generation_prompt=True,
            open_thinking=True,
        )
        ids = tokenizer.encode(text)
        if len(ids) > args.max_len:
            continue
        prompts.append(ids)
        prompt_lens.append(len(ids))
        gt_rows.append([str(x) for x in gt])
        expect_tool.append(tools is not None)

    arr = np.full((len(prompts), args.max_len), 0, dtype=np.int32)
    for i, row in enumerate(prompts):
        arr[i, :len(row)] = row
    arr.tofile(out_dir / "prompts.bin")
    np.asarray(prompt_lens, dtype=np.int32).tofile(out_dir / "prompt_len.bin")
    with open(out_dir / "gt.jsonl", "w", encoding="utf-8") as f:
        for gt, tool in zip(gt_rows, expect_tool):
            f.write(json.dumps({"gt": gt, "expect_tool": tool}, ensure_ascii=False) + "\n")
    manifest = {"source": args.src, "max_len": args.max_len, "prompts": len(prompts),
                "with_tools": sum(expect_tool)}
    (out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
