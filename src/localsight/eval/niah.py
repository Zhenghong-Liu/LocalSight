"""Needle-in-a-Haystack：在长上下文中检索一句话。"""
from __future__ import annotations

import random

FILLERS = [
    "这一年夏天天气格外炎热，街上的行人都步履匆匆。",
    "图书馆里安静得能听见翻书的声音。",
    "远处的山峦在夕阳下呈现出一片金黄。",
    "他每天早上都会去公园慢跑三十分钟。",
    "会议纪要显示，项目将在下个月启动。",
    "火车缓缓驶出站台，窗外是连绵的田野。",
    "实验室的数据表明，实验结果符合预期。",
]
NEEDLE = "秘密口令是 7F3K9Q。"


def make_niah_prompt(context_len: int, seed: int) -> tuple[str, str]:
    """返回 (prompt, needle)。context_len 为字符长度近似。"""
    rng = random.Random(seed)
    parts = []
    while sum(len(p) for p in parts) < context_len:
        parts.append(rng.choice(FILLERS))
    haystack = "".join(parts)
    insert = rng.randrange(0, max(1, len(haystack) - 20))
    haystack = haystack[:insert] + NEEDLE + haystack[insert + len(NEEDLE):]
    prompt = (
        "下面是上下文，请从中找出隐藏的秘密口令并原样回答。\n"
        f"上下文：\n{haystack}\n\n问题：秘密口令是什么？"
    )
    return prompt, NEEDLE


def niah_hit(output: str, needle: str) -> bool:
    return "7F3K9Q" in output
