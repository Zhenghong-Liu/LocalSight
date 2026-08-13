"""Sequence packing：把多篇文档拼进定长序列，输出 document_ids 供掩码使用。"""
from __future__ import annotations

from typing import Iterator

import torch


def pack_sequences(
    docs: list[list[int]],
    max_len: int,
    eos_id: int,
    pad_id: int | None = None,
) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
    """流式产出 (input_ids, document_ids)，形状 (1, max_len)。

    规则：文档之间插入 eos_id 分隔；文档与它的 eos 不跨序列拆分；超长文档按
    剩余空间切块（最后一块才带 eos）。document_ids 每个源文档一个自增 id，
    padding 位置为 -1。
    """
    if max_len < 2:
        raise ValueError("max_len 至少为 2（需要容纳文档与分隔符）")

    def emit() -> tuple[torch.Tensor, torch.Tensor]:
        ids = list(buffer)
        dids = list(doc_ids)
        ids.extend([pad_id] * (max_len - len(ids)))
        dids.extend([-1] * (max_len - len(dids)))
        return torch.tensor([ids], dtype=torch.long), torch.tensor([dids], dtype=torch.long)

    buffer: list[int] = []
    doc_ids: list[int] = []
    for doc_id, doc in enumerate(docs):
        remaining = list(doc)
        while remaining:
            free = max_len - len(buffer)
            if free <= 1:  # 已满或只剩 eos 位：先封当前序列
                yield emit()
                buffer, doc_ids = [], []
                continue
            if len(remaining) <= free - 1:
                buffer.extend(remaining)
                buffer.append(eos_id)
                doc_ids.extend([doc_id] * (len(remaining) + 1))
                remaining = []
            else:
                take = free - 1
                buffer.extend(remaining[:take])
                doc_ids.extend([doc_id] * take)
                remaining = remaining[take:]

    if buffer:
        yield emit()


def pack_labelled_sequences(
    samples: list[tuple[list[int], list[int]]],
    max_len: int,
    eos_id: int,
    pad_id: int = -1,
) -> Iterator[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """SFT 打包：每个 (ids, labels) 视为一篇文档，产出 (input_ids, labels, document_ids)。

    padding 位置：input=pad_id、labels=-100、doc_ids=-1；文档间以 eos 分隔（不计 loss）。
    """
    if max_len < 2:
        raise ValueError("max_len 至少为 2")

    buffer: list[int] = []
    label_buffer: list[int] = []
    doc_ids: list[int] = []

    def emit() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        ids = buffer + [pad_id] * (max_len - len(buffer))
        labs = label_buffer + [-100] * (max_len - len(label_buffer))
        dids = doc_ids + [-1] * (max_len - len(doc_ids))
        return (
            torch.tensor([ids], dtype=torch.long),
            torch.tensor([labs], dtype=torch.long),
            torch.tensor([dids], dtype=torch.long),
        )

    for doc_id, (ids, labels) in enumerate(samples):
        assert len(ids) == len(labels)
        remaining_ids = list(ids)
        remaining_labels = list(labels)
        while remaining_ids:
            free = max_len - len(buffer)
            if free <= 1:
                yield emit()
                buffer, label_buffer, doc_ids = [], [], []
                continue
            if len(remaining_ids) <= free - 1:
                buffer.extend(remaining_ids)
                buffer.append(eos_id)
                label_buffer.extend(remaining_labels)
                label_buffer.append(-100)
                doc_ids.extend([doc_id] * (len(remaining_ids) + 1))
                remaining_ids, remaining_labels = [], []
            else:
                take = free - 1
                buffer.extend(remaining_ids[:take])
                label_buffer.extend(remaining_labels[:take])
                doc_ids.extend([doc_id] * take)
                remaining_ids = remaining_ids[take:]
                remaining_labels = remaining_labels[take:]

    if buffer:
        yield emit()
