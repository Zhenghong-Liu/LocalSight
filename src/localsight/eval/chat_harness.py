"""思考开关评测：同一 prompt 在 open_thinking 开/关下生成并统计。"""
from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class ThinkingEvalResult:
    prompt: str
    think_on: str
    think_off: str
    on_len: int
    off_len: int
    on_think_chars: int
    off_think_chars: int

    @property
    def overthink_ratio(self) -> float:
        return (self.on_len - self.off_len) / max(1, self.off_len)


def extract_think(text: str) -> str:
    match = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
    return match.group(1) if match else ""


def aggregate(results: list[ThinkingEvalResult]) -> dict[str, float]:
    """统计思考开关差异：长度比、思考占比、触发率。"""
    n = max(len(results), 1)
    triggered = sum(bool(r.on_think_chars) for r in results)
    on_chars = sum(r.on_think_chars for r in results)
    on_total = sum(r.on_len for r in results)
    return {
        "n": len(results),
        "think_trigger_rate": triggered / n,
        "think_share_of_on": on_chars / max(1, on_total),
        "mean_length_ratio": sum(r.overthink_ratio for r in results) / n,
    }
