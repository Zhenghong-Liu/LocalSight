"""RLAIF judge：评分 rubric + 结果解析 + 轻量客户端接口。"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass

_SCORE_RE = re.compile(
    r"[\"']?(?:score|评分|分数)[\"']?\s*[:：=]\s*(-?\d+(?:\.\d+)?)", re.IGNORECASE
)
_LAST_NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")


JUDGE_RUBRIC = """你是一名严格的回答质量裁判。请按以下维度给候选回答打分（0-10 的整数）：

1. 思考质量（4 分）：<think> 内的推理步骤是否合理、完整、无跳跃，结论是否由思考导出；
2. 正确性（3 分）：最终回答是否准确回应了问题，无明显事实错误；
3. 简洁性（3 分）：思考与回答是否与问题难度匹配，不冗长、不空洞。

只输出一个 JSON：{{"score": <0-10 的整数>}}

问题：
{question}

候选回答：
{answer}
"""


def build_judge_prompt(question: str, answer: str) -> str:
    return JUDGE_RUBRIC.format(question=question.strip(), answer=answer.strip())


def parse_judge_score(text: str) -> float | None:
    """从 judge 输出解析分数：优先找 score 字段，否则取末尾数字。"""
    match = _SCORE_RE.search(text)
    if match:
        score = float(match.group(1))
    else:
        nums = _LAST_NUM_RE.findall(text)
        if not nums:
            return None
        score = float(nums[-1])
    return max(0.0, min(10.0, score))


@dataclass
class JudgeResult:
    score: float
    raw: str


class JudgeClient:
    """judge 后端接口：vllm / transformers / api 各自实现。"""

    def score(self, question: str, answer: str) -> JudgeResult:
        raise NotImplementedError


class RuleJudge(JudgeClient):
    """无 LLM 的兜底：按启发式打分（仅用于链路测试，不建议作为最终裁判）。"""

    def score(self, question: str, answer: str) -> JudgeResult:
        score = 0.0
        if "<think>" in answer and "</think>" in answer:
            score += 2.0
        if len(answer) > len(question) * 0.3:
            score += 3.0
        if len(answer) < 3000:
            score += 3.0
        score += 2.0 if any(kw in answer for kw in ("因此", "所以", "综上", "答案")) else 0.0
        return JudgeResult(score=min(score, 10.0), raw=json.dumps({"score": score}))
