#!/usr/bin/env python3
"""Profile the LocalSight JSONL corpus: schema, sizes and text-length stats.

Standard library only. Designed to run directly on the training server:

    python3 /tmp/profile_data.py /media/liuzh/data/DLData/LocalSight

For huge pretrain files it counts lines with `wc -l` and takes a uniform
random byte-offset sample instead of reading the whole file line by line.
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import random
import statistics
import subprocess
import sys


def wc_lines(path: str) -> int:
    out = subprocess.run(
        ["wc", "-l", path], capture_output=True, text=True, check=True
    )
    return int(out.stdout.split()[0])


def stats(values: list[int]) -> dict:
    if not values:
        return {"n": 0}
    values = sorted(values)
    q = lambda p: values[min(len(values) - 1, int(p * len(values)))]  # noqa: E731
    return {
        "n": len(values),
        "total_chars": sum(values),
        "mean": round(statistics.mean(values), 1),
        "median": q(0.50),
        "p90": q(0.90),
        "p95": q(0.95),
        "p99": q(0.99),
        "min": values[0],
        "max": values[-1],
    }


def sample_pretrain(path: str, k: int, seed: int = 0) -> dict:
    total = wc_lines(path)
    size = os.path.getsize(path)
    rng = random.Random(seed)
    keys = collections.Counter()
    lens = []
    attempts = 0
    max_attempts = k * 40
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        while len(lens) < k and attempts < max_attempts:
            attempts += 1
            pos = rng.randrange(max(1, size - 2))
            f.seek(pos)
            f.readline()  # discard partial line
            line = f.readline()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:  # noqa: BLE001
                continue
            keys.update(obj.keys())
            text = obj.get("text")
            if isinstance(text, str):
                lens.append(len(text))
    return {
        "lines": total,
        "bytes": size,
        "sampled_valid": len(lens),
        "top_keys": keys.most_common(10),
        "char_len": stats(lens),
    }


def _describe(value, depth: int = 0) -> str:
    if depth > 4:
        return "..."
    if isinstance(value, dict):
        inner = ",".join(f"{k}:{_describe(v, depth + 1)}" for k, v in value.items())
        return "{" + inner + "}"
    if isinstance(value, list):
        inner = "|".join(sorted({_describe(v, depth + 1) for v in value}))
        return "[" + inner + "]"
    return type(value).__name__


TOOL_MARKERS = [
    "tool_call",
    "function_call",
    '"tools"',
    '"observation"',
    '"action"',
    "<tool_call>",
    "«tool_call»",
]


def profile_jsonl(path: str) -> dict:
    total = wc_lines(path)
    keys = collections.Counter()
    shapes = collections.Counter()
    roles = collections.Counter()
    lens = []
    n_reasoning = 0
    n_tool = 0
    tool_markers = collections.Counter()
    errors = 0

    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for raw in f:
            raw = raw.strip()
            if not raw:
                continue
            try:
                obj = json.loads(raw)
            except Exception:  # noqa: BLE001
                errors += 1
                continue
            keys.update(obj.keys())
            shapes[_describe(obj)] += 1

            for marker in TOOL_MARKERS:
                if marker in raw:
                    n_tool += 1
                    tool_markers[marker] += 1
                    break

            contents: list[str] = []

            def walk(node) -> None:
                nonlocal n_reasoning
                if isinstance(node, dict):
                    role = node.get("role")
                    if isinstance(role, str):
                        roles[role] += 1
                    reasoning = node.get("reasoning_content")
                    if isinstance(reasoning, str) and reasoning.strip():
                        n_reasoning += 1
                    content = node.get("content")
                    if isinstance(content, str):
                        contents.append(content)
                    for value in node.values():
                        walk(value)
                elif isinstance(node, list):
                    for value in node:
                        walk(value)

            walk(obj)
            lens.append(sum(len(c) for c in contents))

    return {
        "lines": total,
        "bytes": os.path.getsize(path),
        "parse_errors": errors,
        "top_keys": keys.most_common(20),
        "shapes": shapes.most_common(15),
        "roles": roles.most_common(15),
        "records_with_reasoning_content": n_reasoning,
        "records_with_tool_markers": n_tool,
        "tool_markers_found": tool_markers.most_common(),
        "char_len": stats(lens),
    }


def profile_tokenizer(data_dir: str) -> dict:
    report: dict = {}
    with open(os.path.join(data_dir, "tokenizer.json"), encoding="utf-8") as f:
        tok = json.load(f)
    model = tok.get("model", {})
    report["vocab_size"] = len(model.get("vocab", {}))
    report["merges"] = len(model.get("merges", []))
    report["model_type"] = model.get("type")
    report["added_tokens"] = tok.get("added_tokens")
    report["pre_tokenizer"] = tok.get("pre_tokenizer")
    report["post_processor"] = tok.get("post_processor")
    report["normalizer"] = tok.get("normalizer")
    report["decoder"] = tok.get("decoder")

    with open(os.path.join(data_dir, "tokenizer_config.json"), encoding="utf-8") as f:
        cfg = json.load(f)
    report["config_keys"] = sorted(cfg.keys())
    report["added_tokens_decoder"] = cfg.get("added_tokens_decoder")
    report["special_tokens_map"] = cfg.get("special_tokens_map")
    report["chat_template"] = cfg.get("chat_template")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_dir")
    parser.add_argument("--sample", type=int, default=20000)
    args = parser.parse_args()

    report = {"data_dir": args.data_dir, "files": {}}
    files = {
        "pretrain_t2t.jsonl": None,
        "pretrain_t2t_mini.jsonl": None,
        "sft_t2t_mini.jsonl": None,
        "dpo.jsonl": None,
        "rlaif.jsonl": None,
        "agent_rl.jsonl": None,
    }
    for name in files:
        path = os.path.join(args.data_dir, name)
        if not os.path.exists(path):
            report["files"][name] = "MISSING"
            continue
        if name.startswith("pretrain"):
            report["files"][name] = sample_pretrain(path, args.sample)
        else:
            report["files"][name] = profile_jsonl(path)

    report["tokenizer"] = profile_tokenizer(args.data_dir)
    json.dump(report, sys.stdout, ensure_ascii=False, indent=2)
    print()


if __name__ == "__main__":
    main()
