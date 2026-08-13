"""MinHash 近似去重：64 个 sketch，Jaccard 阈值默认 0.8。仅依赖标准库。"""
from __future__ import annotations

import hashlib
import re
from typing import Iterable, Iterator


def _hash64(seed: int, token: str) -> int:
    digest = hashlib.blake2b(f"{seed}\x00{token}".encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big")


def shingles(text: str, n: int = 5) -> Iterator[str]:
    compact = re.sub(r"\s+", " ", text.strip())
    if len(compact) < n:
        yield compact
        return
    for i in range(len(compact) - n + 1):
        yield compact[i:i + n]


class MinHashSketch:
    """一个文本的 64 维 minhash 签名（uint64）。"""

    NUM_HASHES = 64
    MAX = 2**64 - 1

    def __init__(self, text: str, ngram: int = 5):
        self.values: list[int] = [self.MAX] * self.NUM_HASHES
        for token in shingles(text, ngram):
            for seed in range(self.NUM_HASHES):
                h = _hash64(seed, token)
                if h < self.values[seed]:
                    self.values[seed] = h

    def jaccard(self, other: "MinHashSketch") -> float:
        matches = sum(a == b for a, b in zip(self.values, other.values))
        return matches / self.NUM_HASHES


def deduplicate(
    texts: Iterable[tuple[int, str]],
    threshold: float = 0.8,
    ngram: int = 5,
) -> list[int]:
    """返回保留下来的行号列表。

    贪心：按输入顺序遍历，与「已保留集合」任一 sketch 相似度超阈值则丢弃。
    语料规模大时，生产管线应换 banding（LSH）进一步加速；本实现用于小型去重与测试。
    """
    kept_sketches: list[MinHashSketch] = []
    kept_rows: list[int] = []
    for row_id, text in texts:
        sketch = MinHashSketch(text, ngram)
        duplicate = any(sketch.jaccard(other) >= threshold for other in kept_sketches)
        if not duplicate:
            kept_sketches.append(sketch)
            kept_rows.append(row_id)
    return kept_rows
