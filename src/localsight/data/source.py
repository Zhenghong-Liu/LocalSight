"""数据源读取：优先 HuggingFace datasets 流式读取（调包），jsonl 直读作兜底。"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Callable, Iterator, Optional


def iter_jsonl_texts(src: Path, field: str = "text") -> tuple[Iterator[str], Callable[[], str]]:
    """流式读 jsonl 并逐行取字段；返回 (文本迭代器, sha256 计算器)。"""
    hasher = hashlib.sha256()

    def gen() -> Iterator[str]:
        with open(src, "rb") as f:
            for raw in f:
                hasher.update(raw)
                try:
                    obj = json.loads(raw.decode("utf-8", errors="replace"))
                except json.JSONDecodeError:
                    continue
                text = obj.get(field)
                if isinstance(text, str):
                    yield text

    return gen(), lambda: hasher.hexdigest()


def iter_datasets_texts(
    src: Path,
    field: str = "text",
) -> tuple[Iterator[str], Optional[Callable[[], str]]]:
    """用 HuggingFace datasets 流式读取（无需手工 JSON 解析）。"""
    from datasets import load_dataset

    ds = load_dataset("json", data_files=str(src), split="train", streaming=True)

    def gen() -> Iterator[str]:
        for row in ds:
            text = row.get(field)
            if isinstance(text, str):
                yield text

    return gen(), None
