"""预训练数据构建：清洗 → 去重（精确 + minhash LSH）→ tokenize → packing → mmap 缓存。"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import numpy as np

from localsight.data.minhash import MinHashSketch
from localsight.data.packing import pack_sequences
from localsight.data.source import iter_datasets_texts, iter_jsonl_texts
from localsight.tokenizer.loader import LocalSightTokenizer
from localsight.data.minhash import dedupe_sketches

_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def clean_text(text: str) -> str:
    text = _CONTROL_RE.sub(" ", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


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
        return self.build_from_texts(src, out_dir, chunk=chunk, backend="datasets")

    def build_from_texts(
        self,
        src: Path,
        out_dir: Path,
        chunk: int = 200_000,
        backend: str = "datasets",
    ) -> dict:
        """三阶段构建（backend: datasets | jsonl）。

        阶段 1：清洗/长度过滤/精确去重 + 写 minhash 签名（磁盘 uint64）；
        阶段 2：分桶排序 LSH 去重 → keep 掩码；
        阶段 3：按 keep 掩码重新流式读取 → tokenize → packing → bins。
        避免在内存中保存 Python 对象索引（大语料会 OOM）。
        """
        out_dir.mkdir(parents=True, exist_ok=True)
        tokens_path = out_dir / "tokens.bin"
        docids_path = out_dir / "doc_ids.bin"
        stats = {"rows": 0, "kept_rows": 0, "removed_too_short": 0, "removed_too_long": 0,
                 "removed_exact_dup": 0, "removed_minhash": 0, "sequences": 0, "tokens": 0}
        exact_seen: set[bytes] = set()

        if backend == "datasets":
            text_iter, hasher = iter_datasets_texts(src)
        elif backend == "jsonl":
            text_iter, hasher = iter_jsonl_texts(src)
        else:
            raise ValueError(f"未知 backend: {backend}")

        # ---- 阶段 1：清洗 + 精确去重 + 签名 ----
        sketches_path = out_dir / "sketches.bin"
        rowids_path = out_dir / "row_ids.bin"
        with open(sketches_path, "wb") as sk_file, open(rowids_path, "wb") as rid_file:
            sketch_batch: list[np.ndarray] = []
            row_batch: list[int] = []
            drop_rows: list[int] = []

            def flush_sketches() -> None:
                if not sketch_batch:
                    return
                sk_file.write(np.stack(sketch_batch).astype(np.uint64).tobytes())
                rid_file.write(np.asarray(row_batch, dtype=np.int32).tobytes())
                sketch_batch.clear()
                row_batch.clear()

            for text in text_iter:
                stats["rows"] += 1
                text = clean_text(text)
                if len(text) < self.min_chars:
                    stats["removed_too_short"] += 1
                    drop_rows.append(stats["rows"] - 1)
                    continue
                if len(text) > self.max_chars:
                    stats["removed_too_long"] += 1
                    drop_rows.append(stats["rows"] - 1)
                    continue
                if self.dedup:
                    digest = hashlib.sha256(text.encode("utf-8")).digest()
                    if digest in exact_seen:
                        stats["removed_exact_dup"] += 1
                        drop_rows.append(stats["rows"] - 1)
                        continue
                    exact_seen.add(digest)
                stats["kept_rows"] += 1
                if self.dedup:
                    sketch = MinHashSketch(text)
                    sketch_batch.append(np.asarray(sketch.values, dtype=np.uint64))
                    row_batch.append(stats["rows"] - 1)
                if len(sketch_batch) >= chunk:
                    flush_sketches()
            flush_sketches()

        total_rows = stats["rows"]
        n_sketched = stats["kept_rows"] if self.dedup else 0
        keep_by_row = np.ones(total_rows, dtype=bool)
        if drop_rows:
            keep_by_row[np.asarray(drop_rows)] = False
        if self.dedup and n_sketched:
            sketches = np.memmap(sketches_path, dtype=np.uint64, mode="r")
            row_ids = np.memmap(rowids_path, dtype=np.int32, mode="r")
            keep = dedupe_sketches(sketches.reshape(-1, MinHashSketch.NUM_HASHES))
            keep_by_row[row_ids[~keep]] = False
            stats["removed_minhash"] = int((~keep).sum())

        # ---- 阶段 3：按掩码重读 → tokenize → pack ----
        if backend == "datasets":
            text_iter, _ = iter_datasets_texts(src)
        else:
            text_iter, _ = iter_jsonl_texts(src)
        batch: list[list[int]] = []
        pending_texts: list[str] = []
        total_tokens = 0
        with open(tokens_path, "wb") as tokens_file, open(docids_path, "wb") as docids_file:
            for row, text in enumerate(text_iter):
                if row >= total_rows or not keep_by_row[row]:
                    continue
                pending_texts.append(clean_text(text))
                if len(pending_texts) >= chunk:
                    batch.extend(self._tokenize(pending_texts))
                    pending_texts = []
                if len(batch) >= chunk:
                    total_tokens += self._flush(batch, tokens_file, docids_file, stats)
                    batch = []
            if pending_texts:
                batch.extend(self._tokenize(pending_texts))
            if batch:
                total_tokens += self._flush(batch, tokens_file, docids_file, stats)

        stats["tokens"] = total_tokens
        manifest = {
            "source": str(src),
            "source_sha256": hasher() if hasher else None,
            "backend": backend,
            "max_len": self.max_len,
            "dtype": "int32",
            "tokenizer_vocab": self.tokenizer.vocab_size,
            "stats": stats,
        }
        (out_dir / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return manifest

    def _tokenize(self, texts: list[str], sub_batch: int = 4096) -> list[list[int]]:
        """批量 tokenize（Rust tokenizers），sub_batch 限制峰值内存。"""
        out: list[list[int]] = []
        for i in range(0, len(texts), sub_batch):
            out.extend(self.tokenizer.encode_batch(texts[i:i + sub_batch]))
        return out

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
