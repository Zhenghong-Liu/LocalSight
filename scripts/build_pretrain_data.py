#!/usr/bin/env python3
"""构建预训练派生数据（清洗 → 去重 → tokenize → packing → 二进制缓存 + manifest）。

用法（服务器）：
    PYTHONPATH=src python scripts/build_pretrain_data.py \
        --src /media/liuzh/data/DLData/LocalSight/pretrain_t2t_mini.jsonl \
        --out data/processed/pretrain-mini --tokenizer data/tokenizer
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from localsight.data.pretrain import PretrainDataBuilder
from localsight.tokenizer import LocalSightTokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--tokenizer", default="data/tokenizer")
    parser.add_argument("--max-len", type=int, default=4096)
    parser.add_argument("--min-chars", type=int, default=32)
    parser.add_argument("--max-chars", type=int, default=100_000)
    parser.add_argument("--threshold", type=float, default=0.8)
    parser.add_argument("--no-dedup", action="store_true")
    parser.add_argument("--chunk", type=int, default=200_000)
    args = parser.parse_args()

    tokenizer = LocalSightTokenizer(args.tokenizer)
    builder = PretrainDataBuilder(
        tokenizer,
        max_len=args.max_len,
        min_chars=args.min_chars,
        max_chars=args.max_chars,
        dedup=not args.no_dedup,
        dedup_threshold=args.threshold,
    )
    manifest = builder.build(Path(args.src), Path(args.out), chunk=args.chunk)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
