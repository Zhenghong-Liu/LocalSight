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
        pad = ids == -1
        ids[pad] = 0  # input_ids 必须落在词表内；labels 才是 -100
        labels[pad] = -100
        return {"input_ids": ids, "labels": labels, "document_ids": docs}


class SFTDataset(Dataset):
    """SFT 打包数据集：input_ids / labels / document_ids 三份 int32 二进制。"""

    def __init__(self, data_dir: Path):
        manifest = json.loads((data_dir / "manifest.json").read_text(encoding="utf-8"))
        self.max_len = manifest["max_len"]
        self.tokens = np.memmap(data_dir / "tokens.bin", dtype=np.int32, mode="r")
        self.labels = np.memmap(data_dir / "labels.bin", dtype=np.int32, mode="r")
        self.doc_ids = np.memmap(data_dir / "doc_ids.bin", dtype=np.int32, mode="r")
        for arr, name in ((self.tokens, "tokens"), (self.labels, "labels"), (self.doc_ids, "doc_ids")):
            if arr.shape[0] % self.max_len != 0:
                raise RuntimeError(f"{name}.bin 长度不是 max_len 的整数倍")
        self.num_sequences = self.tokens.shape[0] // self.max_len
        self.manifest = manifest

    def __len__(self) -> int:
        return self.num_sequences

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        start = idx * self.max_len
        ids = torch.from_numpy(np.asarray(self.tokens[start:start + self.max_len], dtype=np.int64))
        labels = torch.from_numpy(np.asarray(self.labels[start:start + self.max_len], dtype=np.int64))
        docs = torch.from_numpy(np.asarray(self.doc_ids[start:start + self.max_len], dtype=np.int64))
        return {"input_ids": ids, "labels": labels, "document_ids": docs}
