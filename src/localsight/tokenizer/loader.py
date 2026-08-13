"""Tokenizer 快照加载与包装（复用 HuggingFace tokenizers，不自研分词算法）。"""
from __future__ import annotations

from pathlib import Path
from typing import Union

from tokenizers import Tokenizer as HFTokenizer

SPECIAL_TOKENS = {
    "eos": "<|endoftext|>",
    "im_start": "<|im_start|>",
    "im_end": "<|im_end|>",
    "think_start": "<think>",
    "think_end": "</think>",
    "tool_call": "<tool_call>",
    "tool_call_end": "</tool_call>",
    "tool_response": "<tool_response>",
    "tool_response_end": "</tool_response>",
}


class LocalSightTokenizer:
    def __init__(self, path: Union[str, Path]):
        tokenizer_file = Path(path) / "tokenizer.json"
        if not tokenizer_file.exists():
            raise FileNotFoundError(f"未找到 tokenizer.json: {tokenizer_file}")
        self.tok = HFTokenizer.from_file(str(tokenizer_file))
        self.vocab_size = self.tok.get_vocab_size()
        self.ids: dict[str, int] = {}
        missing = []
        for name, token in SPECIAL_TOKENS.items():
            token_id = self.tok.token_to_id(token)
            if token_id is None:
                missing.append(token)
            else:
                self.ids[name] = token_id
        if missing:
            raise ValueError(f"tokenizer 缺少特殊 token: {missing}")

    @property
    def eos_id(self) -> int:
        return self.ids["eos"]

    @property
    def im_start_id(self) -> int:
        return self.ids["im_start"]

    @property
    def im_end_id(self) -> int:
        return self.ids["im_end"]

    @property
    def think_start_id(self) -> int:
        return self.ids["think_start"]

    @property
    def think_end_id(self) -> int:
        return self.ids["think_end"]

    def encode(self, text: str) -> list[int]:
        return self.tok.encode(text).ids

    def encode_batch(self, texts: list[str]) -> list[list[int]]:
        return [e.ids for e in self.tok.encode_batch(texts)]

    def decode(self, ids: list[int]) -> str:
        return self.tok.decode(ids)
