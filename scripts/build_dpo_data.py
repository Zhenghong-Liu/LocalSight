#!/usr/bin/env python3
"""构建 DPO/SimPO 派生数据：prompt/chosen/rejected 分词 + 响应掩码。

用法（服务器）：
    PYTHONPATH=src python scripts/build_dpo_data.py \
        --src /media/liuzh/data/DLData/LocalSight/dpo.jsonl \
        --out data/processed/dpo --tokenizer data/tokenizer
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer

from localsight.data.chat import format_chat


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

    prompts, chosen, rejected, prompt_lens = [], [], [], []
    skipped = 0
    for row in ds:
        ch, rj = row.get("chosen"), row.get("rejected")
        if not isinstance(ch, list) or not isinstance(rj, list) or len(ch) < 2 or len(rj) < 2:
            skipped += 1
            continue
        prompt_msgs = [dict(m) for m in ch[:-1]]
        prompt_text = format_chat(tokenizer, prompt_msgs, add_generation_prompt=True)
        prompt_ids = tokenizer.encode(prompt_text)
        ch_ids = tokenizer.encode(format_chat(tokenizer, [dict(m) for m in ch]))
        rj_ids = tokenizer.encode(format_chat(tokenizer, [dict(m) for m in rj]))
        if len(ch_ids) > args.max_len or len(rj_ids) > args.max_len:
            skipped += 1
            continue
        prompts.append(prompt_ids)
        chosen.append(ch_ids)
        rejected.append(rj_ids)
        prompt_lens.append(len(prompt_ids))

    def save(name: str, rows: list[list[int]], pad: int) -> None:
        arr = np.full((len(rows), args.max_len), pad, dtype=np.int32)
        for i, row in enumerate(rows):
            arr[i, :len(row)] = row
        arr.tofile(out_dir / name)

    save("prompt.bin", prompts, 0)
    save("chosen.bin", chosen, 0)
    save("rejected.bin", rejected, 0)
    np.asarray(prompt_lens, dtype=np.int32).tofile(out_dir / "prompt_len.bin")
    manifest = {"source": args.src, "max_len": args.max_len, "pairs": len(prompts), "skipped": skipped}
    (out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
