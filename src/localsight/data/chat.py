"""对话数据格式化：统一走 chat_template，并生成「只算 assistant 轮」的 loss mask。"""
from __future__ import annotations

import json
import re
from typing import Any, Optional

import torch

_ASSISTANT_RE = re.compile(r"<\|im_start\|>assistant\n(.*?)<\|im_end\|>", re.DOTALL)


def extract_tools(messages: list[dict]) -> tuple[list[dict], Optional[list]]:
    """归一化消息：system.tools（字符串→list，并拆到顶层）与 assistant.tool_calls（字符串→list）。"""
    tools = None
    cleaned = []
    for msg in messages:
        msg = dict(msg)
        if msg.get("role") == "system" and "tools" in msg:
            raw = msg.pop("tools")
            if isinstance(raw, str) and raw.strip():
                try:
                    tools = json.loads(raw)
                except json.JSONDecodeError:
                    tools = None
            elif isinstance(raw, list):
                tools = raw
        if msg.get("role") == "assistant" and isinstance(msg.get("tool_calls"), str):
            raw = msg["tool_calls"]
            if raw.strip():
                try:
                    msg["tool_calls"] = json.loads(raw)
                except json.JSONDecodeError:
                    msg.pop("tool_calls")
            else:
                msg.pop("tool_calls")
        cleaned.append(msg)
    return cleaned, tools


def format_chat(
    tokenizer,
    messages: list[dict],
    tools: Optional[list] = None,
    add_generation_prompt: bool = False,
    open_thinking: Optional[bool] = None,
) -> str:
    kwargs: dict[str, Any] = {
        "tokenize": False,
        "add_generation_prompt": add_generation_prompt,
    }
    if tools is not None:
        kwargs["tools"] = tools
    if open_thinking is not None:
        kwargs["open_thinking"] = open_thinking
    return tokenizer.apply_chat_template(messages, **kwargs)


def tokenize_chat_with_labels(
    tokenizer,
    messages: list[dict],
    max_len: int,
    tools: Optional[list] = None,
    add_generation_prompt: bool = False,
    open_thinking: Optional[bool] = None,
) -> dict[str, torch.Tensor]:
    """返回 {input_ids, labels, attention_mask}，长度 ≤ max_len（右截断）。"""
    text = format_chat(tokenizer, messages, tools, add_generation_prompt, open_thinking)
    enc = tokenizer(text, return_offsets_mapping=True)
    ids = enc["input_ids"][:max_len]
    offsets = enc["offset_mapping"][:max_len]
    labels = [-100] * len(ids)

    for match in _ASSISTANT_RE.finditer(text):
        start, end = match.span(1)
        for i, (a, b) in enumerate(offsets):
            if a is not None and b is not None and a >= start and b <= end:
                labels[i] = ids[i]

    attention_mask = [1] * len(ids)
    return {
        "input_ids": torch.tensor([ids], dtype=torch.long),
        "labels": torch.tensor([labels], dtype=torch.long),
        "attention_mask": torch.tensor([attention_mask], dtype=torch.long),
    }


def drop_final_empty_assistant(messages: list[dict]) -> list[dict]:
    """RLAIF/agent 数据：去掉末轮空 assistant（补全式 prompt 的前缀）。"""
    if not messages:
        return messages
    last = messages[-1]
    if last.get("role") == "assistant" and not str(last.get("content", "")).strip():
        return messages[:-1]
    return messages
