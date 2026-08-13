#!/usr/bin/env python3
"""下载 RLAIF judge 模型（默认经 hf-mirror 国内镜像）。

用法：
    HF_ENDPOINT=https://hf-mirror.com python scripts/download_judge.py \
        --model Qwen/Qwen2.5-7B-Instruct --out models/judge
"""
from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    path = snapshot_download(args.model, local_dir=str(out / args.model.replace("/", "--")))
    print("downloaded:", path)


if __name__ == "__main__":
    main()
