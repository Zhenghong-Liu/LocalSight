#!/usr/bin/env python3
"""Second-pass profiling: token estimates and record-level schema details.

Requires the `tokenizers` package (available in the kdl conda env on the
training server):

    ~/miniconda3/envs/kdl/bin/python /tmp/profile_data2.py /media/liuzh/data/DLData/LocalSight
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys

from tokenizers import Tokenizer


def clip(value, n: int = 1800) -> str:
    text = json.dumps(value, ensure_ascii=False, indent=2)
    return text if len(text) <= n else text[:n] + f"\n...[truncated {len(text)} chars]"


def seek_sample(path: str, k: int, seed: int = 0) -> list[str]:
    size = os.path.getsize(path)
    rng = random.Random(seed)
    out: list[str] = []
    with open(path, encoding="utf-8", errors="replace") as f:
        tries = 0
        while len(out) < k and tries < k * 40:
            tries += 1
            f.seek(rng.randrange(max(1, size - 2)))
            f.readline()
            line = f.readline()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:  # noqa: BLE001
                continue
            text = obj.get("text")
            if isinstance(text, str):
                out.append(text)
    return out


def iter_records(path: str):
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    yield json.loads(line)
                except Exception:  # noqa: BLE001
                    continue


def token_estimate(data_dir: str) -> dict:
    tok = Tokenizer.from_file(os.path.join(data_dir, "tokenizer.json"))
    report = {}
    for name in ("pretrain_t2t.jsonl", "pretrain_t2t_mini.jsonl"):
        texts = seek_sample(os.path.join(data_dir, name), 1500)
        encodings = tok.encode_batch(texts)
        n_tok = [len(e.ids) for e in encodings]
        n_chr = [len(t) for t in texts]
        t_sum = sum(n_tok)
        c_sum = sum(n_chr)
        report[name] = {
            "sample_docs": len(texts),
            "tokens_per_char": round(t_sum / c_sum, 4),
            "chars_per_token": round(c_sum / t_sum, 4),
            "tokens_per_doc": round(t_sum / len(texts), 1),
        }
    return report


def profile_dpo(path: str) -> dict:
    chosen_think = rejected_think = both = none = 0
    sample_chosen = sample_rejected = None
    total = 0
    for rec in iter_records(path):
        total += 1
        c = json.dumps(rec.get("chosen"), ensure_ascii=False)
        r = json.dumps(rec.get("rejected"), ensure_ascii=False)
        c_has = "<think>" in c
        r_has = "<think>" in r
        chosen_think += c_has
        rejected_think += r_has
        both += c_has and r_has
        none += (not c_has) and (not r_has)
        if sample_chosen is None and c_has:
            sample_chosen = rec
        if sample_rejected is None and not r_has:
            sample_rejected = rec
    return {
        "total": total,
        "chosen_has_think": chosen_think,
        "rejected_has_think": rejected_think,
        "both_think": both,
        "neither_think": none,
        "sample_chosen_with_think": clip(sample_chosen, 1200),
        "sample_rejected_without_think": clip(sample_rejected, 1200),
    }


def profile_rlaif(path: str) -> dict:
    last_role = {}
    n_turns = []
    total = 0
    first = None
    for rec in iter_records(path):
        total += 1
        conv = rec.get("conversations", [])
        n_turns.append(len(conv))
        role = conv[-1].get("role") if conv else "empty"
        last_role[role] = last_role.get(role, 0) + 1
        if first is None:
            first = rec
    return {
        "total": total,
        "last_message_role": last_role,
        "turns_per_record": {
            "min": min(n_turns),
            "max": max(n_turns),
            "mean": round(sum(n_turns) / len(n_turns), 1),
        },
        "sample_record": clip(first, 1200),
    }


def profile_agent(path: str) -> dict:
    with_tools = with_gt = 0
    gt_lens = []
    first_tool_gt = None
    total = 0
    for rec in iter_records(path):
        total += 1
        conv = rec.get("conversations", [])
        has_tools = any(isinstance(m.get("tools"), (str, list, dict)) for m in conv)
        gt = rec.get("gt")
        has_gt = isinstance(gt, list) and len(gt) > 0
        with_tools += has_tools
        with_gt += has_gt
        if has_gt:
            gt_lens.append(len(json.dumps(gt, ensure_ascii=False)))
        if first_tool_gt is None and has_tools and has_gt:
            first_tool_gt = rec
    return {
        "total": total,
        "with_tools": with_tools,
        "with_nonempty_gt": with_gt,
        "gt_chars_stats": {
            "min": min(gt_lens) if gt_lens else 0,
            "max": max(gt_lens) if gt_lens else 0,
            "mean": round(sum(gt_lens) / len(gt_lens), 1) if gt_lens else 0,
        },
        "sample_tool_gt_record": clip(first_tool_gt, 2500),
    }


def profile_sft(path: str) -> dict:
    tool_call_sample = reasoning_sample = None
    n_reasoning_with_tool = 0
    total = 0
    for rec in iter_records(path):
        total += 1
        conv = rec.get("conversations", [])
        has_tc = any("tool_calls" in m for m in conv)
        has_rc = any(isinstance(m.get("reasoning_content"), str) and m["reasoning_content"] for m in conv)
        n_reasoning_with_tool += has_tc and has_rc
        if tool_call_sample is None and has_tc:
            tool_call_sample = rec
        if reasoning_sample is None and has_rc and not has_tc:
            reasoning_sample = rec
    return {
        "total": total,
        "reasoning_with_tool_calls": n_reasoning_with_tool,
        "sample_tool_call_record": clip(tool_call_sample, 2200),
        "sample_reasoning_record": clip(reasoning_sample, 1500),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_dir")
    args = parser.parse_args()
    report = {
        "token_estimate": token_estimate(args.data_dir),
        "dpo": profile_dpo(os.path.join(args.data_dir, "dpo.jsonl")),
        "rlaif": profile_rlaif(os.path.join(args.data_dir, "rlaif.jsonl")),
        "agent_rl": profile_agent(os.path.join(args.data_dir, "agent_rl.jsonl")),
        "sft": profile_sft(os.path.join(args.data_dir, "sft_t2t_mini.jsonl")),
    }
    json.dump(report, sys.stdout, ensure_ascii=False, indent=2)
    print()


if __name__ == "__main__":
    main()
