"""预训练数据构建：清洗 → 去重（精确 + minhash LSH）→ tokenize → packing → mmap 缓存。"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import numpy as np

from localsight.data.minhash import MinHashSketch
from localsight.data.packing import pack_sequences
from localsight.tokenizer.loader import LocalSightTokenizer

_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def clean_text(text: str) -> str:
    text = _CONTROL_RE.sub(" ", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


class MinHashIndex:
    """带分桶 LSH 的 minhash 去重索引：每个 band 只保留首个代表 sketch，内存有界。"""

    def __init__(self, num_bands: int = 16, threshold: float = 0.8):
        self.num_bands = num_bands
        self.band_size = MinHashSketch.NUM_HASHES // num_bands
        self.threshold = threshold
        self.tables: list[dict[tuple, MinHashSketch]] = [{} for _ in range(num_bands)]

    def _band_key(self, sketch: MinHashSketch, band: int) -> tuple:
        start = band * self.band_size
        return tuple(sketch.values[start:start + self.band_size])

    def is_duplicate(self, sketch: MinHashSketch) -> bool:
        for band in range(self.num_bands):
            rep = self.tables[band].get(self._band_key(sketch, band))
            if rep is not None and sketch.jaccard(rep) >= self.threshold:
                return True
        return False

    def add(self, sketch: MinHashSketch) -> None:
        for band in range(self.num_bands):
            self.tables[band].setdefault(self._band_key(sketch, band), sketch)


class PretrainDataBuilder:
    def __init__(
        self,
        tokenizer: LocalSightTokenizer,
        max_len: int = 4096,
        min_chars: int = 32,
        max_chars: int = 100_000,
        dedup: bool = True,
        dedup_threshold: float = 0.8,
    ):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.min_chars = min_chars
        self.max_chars = max_chars
        self.dedup = dedup
        self.dedup_threshold = dedup_threshold

    def build(self, src: Path, out_dir: Path, chunk: int = 200_000) -> dict:
        out_dir.mkdir(parents=True, exist_ok=True)
        tokens_path = out_dir / "tokens.bin"
        docids_path = out_dir / "doc_ids.bin"
        stats = {"rows": 0, "kept_rows": 0, "removed_too_short": 0, "removed_too_long": 0,
                 "removed_exact_dup": 0, "removed_minhash": 0, "sequences": 0, "tokens": 0}
        index = MinHashIndex(threshold=self.dedup_threshold) if self.dedup else None
        exact_seen: set[bytes] = set()
        batch: list[list[int]] = []
        total_tokens = 0
        src_hasher = hashlib.sha256()

        with open(tokens_path, "wb") as tokens_file, \
                open(docids_path, "wb") as docids_file, \
                open(src, "rb") as f:
            for row, raw in enumerate(f):
                src_hasher.update(raw)
                line = raw.decode("utf-8", errors="replace")
                stats["rows"] += 1
                try:
                    text = json.loads(line).get("text")
                except Exception:  # noqa: BLE001
                    continue
                if not isinstance(text, str):
                    continue
                text = clean_text(text)
                if len(text) < self.min_chars:
                    stats["removed_too_short"] += 1
                    continue
                if len(text) > self.max_chars:
                    stats["removed_too_long"] += 1
                    continue
                if self.dedup:
                    digest = hashlib.sha256(text.encode("utf-8")).digest()
                    if digest in exact_seen:
                        stats["removed_exact_dup"] += 1
                        continue
                    exact_seen.add(digest)
                    sketch = MinHashSketch(text)
                    if index.is_duplicate(sketch):
                        stats["removed_minhash"] += 1
                        continue
                    index.add(sketch)
                stats["kept_rows"] += 1
                batch.append(self.tokenizer.encode(text))

                if len(batch) >= chunk:
                    total_tokens += self._flush(batch, tokens_file, docids_file, stats)
                    batch = []
        if batch:
            total_tokens += self._flush(batch, tokens_file, docids_file, stats)

        stats["tokens"] = total_tokens
        manifest = {
            "source": str(src),
            "source_sha256": src_hasher.hexdigest(),
            "max_len": self.max_len,
            "dtype": "int32",
            "tokenizer_vocab": self.tokenizer.vocab_size,
            "stats": stats,
        }
        (out_dir / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return manifest

    def _flush(
        self,
        batch: list[list[int]],
        tokens_file,
        docids_file,
        stats: dict,
    ) -> int:
        total = 0
        new_tokens: list[list[int]] = []
        new_docids: list[list[int]] = []
        for input_ids, doc_ids in pack_sequences(
            batch, self.max_len, self.tokenizer.eos_id, pad_id=-1
        ):
            flat_ids = input_ids[0].tolist()
            flat_docs = doc_ids[0].tolist()
            total += sum(1 for t in flat_ids if t != -1)
            new_tokens.append(flat_ids)
            new_docids.append(flat_docs)
        if new_tokens:
            arr_ids = np.asarray(new_tokens, dtype=np.int32).reshape(-1)
            arr_docs = np.asarray(new_docids, dtype=np.int32).reshape(-1)
            tokens_file.write(arr_ids.tobytes())
            docids_file.write(arr_docs.tobytes())
            stats["sequences"] += len(new_tokens)
        return total
