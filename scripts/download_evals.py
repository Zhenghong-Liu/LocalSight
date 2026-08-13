#!/usr/bin/env python3
"""下载评测数据集（HF datasets，默认走 hf-mirror）。

用法：
    HF_ENDPOINT=https://hf-mirror.com python scripts/download_evals.py --out data/eval

数据集：MMLU(hendrycks_test)、CMMLU、C-Eval、GSM8K(test)、IFEval。
HumanEval 用 HF json 单独下载。均为只读评测集，不参与训练。
"""
from __future__ import annotations

import argparse
from pathlib import Path

from datasets import load_dataset

DATASETS = {
    "mmlu": ("cais/mmlu", "test"),
    "cmmlu": ("haonan-li/cmmlu", "test"),
    "ceval": ("ceval/ceval-exam", "test"),
    "gsm8k": ("openai/gsm8k", "test"),
    "ifeval": ("google/IFEval", "train"),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/eval")
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for name, (repo, split) in DATASETS.items():
        try:
            kwargs = {"name": "main"} if name == "gsm8k" else {}
            ds = load_dataset(repo, split=split, **kwargs)
            ds.save_to_disk(str(out / name))
            print(name, "ok", len(ds))
        except Exception as exc:  # noqa: BLE001
            print(name, "FAILED:", exc)


if __name__ == "__main__":
    main()
