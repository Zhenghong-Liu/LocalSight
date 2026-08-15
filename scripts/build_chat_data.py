#!/usr/bin/env python3
"""构建 SFT 派生数据：conversations → chat_template → 带 loss mask 的 packing。

用法（服务器，需 transformers）：
    PYTHONPATH=src python scripts/build_chat_data.py \
        --src /media/liuzh/data/DLData/LocalSight/sft_t2t_mini.jsonl \
        --out data/processed/sft --tokenizer data/tokenizer
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer

from localsight.data.chat import extract_tools, tokenize_chat_with_labels
from localsight.data.packing import pack_labelled_sequences


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--tokenizer", default="data/tokenizer")
    parser.add_argument("--max-len", type=int, default=8192)
    parser.add_argument("--chunk", type=int, default=50_000)
    parser.add_argument("--max-rows", type=int, default=None, help="只处理前 N 行（冒烟用）")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    eos_id = tokenizer.convert_tokens_to_ids("<|endoftext|>")
    ds = load_dataset("json", data_files=args.src, split="train", streaming=True)

    stats = {"rows": 0, "skipped": 0, "samples": 0, "sequences": 0, "tokens": 0}
    batch: list[tuple[list[int], list[int]]] = []

    with open(out_dir / "tokens.bin", "wb") as tf, \
            open(out_dir / "labels.bin", "wb") as lf, \
            open(out_dir / "doc_ids.bin", "wb") as df:
        for row in ds:
            if args.max_rows is not None and stats["rows"] >= args.max_rows:
                break
            stats["rows"] += 1
            conversations = row.get("conversations")
            if not isinstance(conversations, list) or not conversations:
                stats["skipped"] += 1
                continue
            messages, tools = extract_tools(conversations)
            enc = tokenize_chat_with_labels(
                tokenizer, messages, args.max_len, tools=tools,
                add_generation_prompt=False,
            )
            ids = enc["input_ids"][0].tolist()
            labels = enc["labels"][0].tolist()
            if not any(l != -100 for l in labels):
                stats["skipped"] += 1
                continue
            batch.append((ids, labels))
            stats["samples"] += 1

            if len(batch) >= args.chunk:
                stats = _flush(batch, args.max_len, eos_id, tf, lf, df, stats)
                batch = []
        if batch:
            stats = _flush(batch, args.max_len, eos_id, tf, lf, df, stats)

    manifest = {
        "source": args.src,
        "max_len": args.max_len,
        "dtype": "int32",
        "stats": stats,
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


def _flush(
    batch: list[tuple[list[int], list[int]]],
    max_len: int,
    eos_id: int,
    tf,
    lf,
    df,
    stats: dict,
) -> dict:
    ids_rows, labels_rows, doc_rows = [], [], []
    n_tokens = 0
    for ids, labels, doc_ids in pack_labelled_sequences(batch, max_len, eos_id, pad_id=-1):
        ids = ids[0].clone()
        ids[ids == -1] = 0  # input 必须是合法 token；labels/doc_ids 才用 -1/-100
        ids_rows.append(ids.tolist())
        labels_rows.append(labels[0].tolist())
        doc_rows.append(doc_ids[0].tolist())
        n_tokens += sum(1 for t in ids_rows[-1] if t != -1)
        stats["sequences"] += 1
    tf.write(np.asarray(ids_rows, dtype=np.int32).tobytes())
    lf.write(np.asarray(labels_rows, dtype=np.int32).tobytes())
    df.write(np.asarray(doc_rows, dtype=np.int32).tobytes())
    stats["tokens"] += n_tokens
    return stats


if __name__ == "__main__":
    main()
