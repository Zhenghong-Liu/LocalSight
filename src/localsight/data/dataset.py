"""mmap 预训练数据集与 DDP 采样封装。"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class PretrainDataset(Dataset):
    def __init__(self, data_dir: Path):
        manifest = json.loads((data_dir / "manifest.json").read_text(encoding="utf-8"))
        self.max_len = manifest["max_len"]
        self.tokens = np.memmap(data_dir / "tokens.bin", dtype=np.int32, mode="r")
        self.doc_ids = np.memmap(data_dir / "doc_ids.bin", dtype=np.int32, mode="r")
        if self.tokens.shape[0] % self.max_len != 0:
            raise RuntimeError("tokens.bin 长度不是 max_len 的整数倍")
        self.num_sequences = self.tokens.shape[0] // self.max_len
        self.manifest = manifest

    def __len__(self) -> int:
        return self.num_sequences

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        start = idx * self.max_len
        ids = torch.from_numpy(np.asarray(self.tokens[start:start + self.max_len], dtype=np.int64))
        docs = torch.from_numpy(np.asarray(self.doc_ids[start:start + self.max_len], dtype=np.int64))
        labels = ids.clone()
        labels[ids == -1] = -100
        return {"input_ids": ids, "labels": labels, "document_ids": docs}
