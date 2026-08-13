"""MinHash 近似去重：64 个 sketch，Jaccard 阈值默认 0.8。

性能说明：每个 shingle 只做一次 Python 内置 hash，再用 numpy 向量化的
splitmix64 展开成 64 个签名位；长文本按步长采样，单文档 shingle 上限 4096。
"""
from __future__ import annotations

import math
import re
from typing import Iterable, Iterator

import numpy as np

MASK64 = np.uint64((1 << 64) - 1)
GOLDEN = np.uint64(0x9E3779B97F4A7C15)


def _splitmix64(x: np.ndarray) -> np.ndarray:
    z = (x + GOLDEN) & MASK64
    z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    z = z ^ (z >> np.uint64(31))
    return z & MASK64


def shingles(text: str, n: int = 5) -> Iterator[str]:
    compact = re.sub(r"\s+", " ", text.strip())
    if len(compact) < n:
        yield compact
        return
    for i in range(len(compact) - n + 1):
        yield compact[i:i + n]


class MinHashSketch:
    """一个文本的 64 维 minhash 签名（python int 列表）。"""

    NUM_HASHES = 64
    MAX = 2**64 - 1
    MAX_SHINGLES = 4096

    def __init__(self, text: str, ngram: int = 5):
        self.values: list[int] = self._compute(text, ngram)

    @classmethod
    def _compute(cls, text: str, ngram: int) -> list[int]:
        compact = re.sub(r"\s+", " ", text.strip())
        if not compact:
            compact = " "
        total = max(1, len(compact) - ngram + 1)
        stride = max(1, math.ceil(total / cls.MAX_SHINGLES))
        tokens = []
        for i in range(0, total, stride):
            tokens.append(hash(compact[i:i + ngram]) & cls.MAX)
        if not tokens:
            tokens = [hash(compact) & cls.MAX]
        arr = np.asarray(tokens, dtype=np.uint64)
        seeds = np.arange(cls.NUM_HASHES, dtype=np.uint64) * GOLDEN
        values = _splitmix64(arr[:, None] ^ seeds[None, :])
        return [int(v) for v in values.min(axis=0)]

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


def dedupe_sketches(
    sketches: np.ndarray,
    num_bands: int = 16,
    threshold: float = 0.8,
) -> np.ndarray:
    """对 (N, 64) uint64 签名数组做分桶 LSH 去重，返回 keep 布尔掩码。

    每个 band 用字节视图做稳定排序，比较「同 band key 的相邻行」的 Jaccard；
    Jaccard ≥ threshold 时保留较早行。内存与 N 线性、无 Python 对象膨胀。
    """
    n, m = sketches.shape
    if m % num_bands != 0:
        raise ValueError("签名维度必须能被 band 数整除")
    band_size = m // num_bands
    void_dtype = f"V{band_size * 8}"
    keep = np.ones(n, dtype=bool)

    for band in range(num_bands):
        block = np.ascontiguousarray(sketches[:, band * band_size:(band + 1) * band_size])
        keys = block.view(void_dtype).ravel()
        order = np.argsort(keys, kind="stable")
        sorted_keys = keys[order]
        equal = sorted_keys[1:] == sorted_keys[:-1]
        idx = np.flatnonzero(equal)
        if idx.size == 0:
            continue
        a = order[idx]
        b = order[idx + 1]
        jaccard = (sketches[a] == sketches[b]).sum(axis=1) / m
        dup = jaccard >= threshold
        if dup.any():
            keep[b[dup]] = False
    return keep
