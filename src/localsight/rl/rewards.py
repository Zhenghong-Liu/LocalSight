"""Agent RL 奖励函数：格式 / 工具合法性 / gt 结果匹配。"""
from __future__ import annotations

import json
import re

from localsight.tools.executor import TOOL_REGISTRY, execute_tool, parse_tool_calls

_THINK_OPEN_RE = re.compile(r"<think>")
_THINK_CLOSE_RE = re.compile(r"</think>")


def format_reward(text: str, expect_tool_call: bool) -> float:
    """think 标签完整 + （期望工具时）tool_call JSON 合法。"""
    score = 0.0
    if _THINK_OPEN_RE.search(text) and _THINK_CLOSE_RE.search(text):
        score += 0.5
    calls = parse_tool_calls(text)
    if expect_tool_call:
        score += 0.5 if calls else 0.0
    elif calls:
        score += 0.0  # 不需要工具时不奖励工具调用（可选：0.25 以鼓励克制）
    return score


def tool_call_reward(text: str) -> tuple[float, int, int]:
    """返回 (成功率, 成功数, 总数)。解析并尝试执行每个 tool_call。"""
    calls = parse_tool_calls(text)
    if not calls:
        return 0.0, 0, 0
    ok = 0
    for name, args in calls:
        try:
            execute_tool(name, args)
            ok += 1
        except Exception:  # noqa: BLE001
            pass
    return ok / len(calls), ok, len(calls)


def normalize_answer(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"[,\s]+", "", text)
    text = re.sub(r"^[\[\(\{]+|[\]\)\}]+$", "", text)
    return text


def numeric_equal(a: str, b: str) -> bool:
    try:
        return abs(float(a) - float(b)) < 1e-6
    except ValueError:
        return False


def gt_match(final_text: str, gt: list[str]) -> float:
    """最终答案与任一 gt 的匹配：数值精确 / 字符串包含（归一化后）。"""
    if not gt:
        return 0.0
    final = normalize_answer(final_text)
    for target in gt:
        target = normalize_answer(str(target))
        if not target:
            continue
        if numeric_equal(final, target) or target in final or final in target:
            return 1.0
    return 0.0


def composite_reward(
    text: str,
    expect_tool_call: bool,
    gt: list[str],
    weights: tuple[float, float, float] = (0.3, 0.2, 0.5),
) -> float:
    """归一化到 [0,1]：格式 0.3 + 工具 0.2 + 结果 0.5。"""
    fmt = format_reward(text, expect_tool_call)
    tool_ok, _, _ = tool_call_reward(text)
    result = gt_match(text, gt)
    return weights[0] * fmt + weights[1] * tool_ok + weights[2] * result
