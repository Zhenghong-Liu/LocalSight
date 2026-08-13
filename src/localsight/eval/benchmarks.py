"""标准基准的加载与打分：多选（MMLU/CMMLU/C-Eval）与 GSM8K。"""
from __future__ import annotations

import re
from typing import Callable, Iterator

from datasets import load_from_disk


def mc_examples(dataset_name: str, ds) -> Iterator[tuple[str, list[str], int]]:
    """产出 (question, choices, gold_index)。"""
    if dataset_name == "mmlu":
        for row in ds:
            yield row["question"], list(row["choices"]), int(row["answer"])
    elif dataset_name == "cmmlu":
        for row in ds:
            yield row["Question"], [row[k] for k in ("A", "B", "C", "D")], ord(row["Answer"]) - ord("A")
    elif dataset_name == "ceval":
        for row in ds:
            yield row["question"], [row[k] for k in ("A", "B", "C", "D")], ord(row["answer"]) - ord("A")
    else:
        raise ValueError(f"未知多选数据集: {dataset_name}")


def build_mc_prompt(question: str, choices: list[str]) -> str:
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    lines = [f"{letters[i]}. {choice}" for i, choice in enumerate(choices)]
    return f"{question}\n" + "\n".join(lines) + "\n请只回答选项字母（如 A）。"


def extract_mc_answer(text: str) -> str | None:
    match = re.search(r"\b([A-H])\b", text.strip(), re.IGNORECASE)
    return match.group(1).upper() if match else None


def extract_last_number(text: str) -> float | None:
    numbers = re.findall(r"-?\d+(?:[.,]\d+)?", text.replace(",", ""))
    if not numbers:
        return None
    last = numbers[-1].replace(",", ".")
    try:
        return float(last)
    except ValueError:
        return None


def run_mc(
    generate: Callable[[str], str],
    dataset_name: str,
    ds,
    limit: int | None = None,
) -> dict[str, float]:
    correct = total = 0
    for i, (question, choices, gold) in enumerate(mc_examples(dataset_name, ds)):
        if limit is not None and i >= limit:
            break
        answer = extract_mc_answer(generate(build_mc_prompt(question, choices)))
        correct += answer is not None and ord(answer) - ord("A") == gold
        total += 1
    return {"acc": correct / max(total, 1), "n": total}


def run_gsm8k(
    generate: Callable[[str], str],
    ds,
    limit: int | None = None,
) -> dict[str, float]:
    correct = total = 0
    for i, row in enumerate(ds):
        if limit is not None and i >= limit:
            break
        target = extract_last_number(row["answer"])
        got = extract_last_number(generate(row["question"]))
        correct += target is not None and got is not None and abs(target - got) < 1e-6
        total += 1
    return {"acc": correct / max(total, 1), "n": total}


def load_benchmark(base_dir: str, name: str):
    return load_from_disk(f"{base_dir}/{name}")
