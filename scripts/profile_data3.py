#!/usr/bin/env python3
"""Third-pass profiling: prompt-readiness of rlaif/agent_rl and tool inventory."""
from __future__ import annotations

import collections
import json
import sys


def iter_records(path: str):
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def clip(value, n: int = 1500) -> str:
    text = json.dumps(value, ensure_ascii=False, indent=2)
    return text if len(text) <= n else text[:n] + f"\n...[truncated {len(text)} chars]"


def main() -> None:
    data_dir = "/media/liuzh/data/DLData/LocalSight"
    report = {}

    rlaif_total = 0
    empty = 0
    nonempty_sample = None
    for rec in iter_records(data_dir + "/rlaif.jsonl"):
        rlaif_total += 1
        conv = rec["conversations"]
        last = conv[-1]
        if last.get("role") == "assistant" and last.get("content", "").strip() == "":
            empty += 1
        elif nonempty_sample is None:
            nonempty_sample = rec
    report["rlaif"] = {
        "total": rlaif_total,
        "final_assistant_empty": empty,
        "sample_nonempty_final": clip(nonempty_sample),
    }

    agent_total = empty = with_tools = with_gt = 0
    last_role = collections.Counter()
    tool_names = collections.Counter()
    no_tools_gt_sample = None
    for rec in iter_records(data_dir + "/agent_rl.jsonl"):
        agent_total += 1
        conv = rec["conversations"]
        last = conv[-1]
        last_role[last.get("role")] += 1
        if last.get("role") == "assistant" and last.get("content", "").strip() == "":
            empty += 1
        gt = rec.get("gt") or []
        if isinstance(gt, list) and gt:
            with_gt += 1
        has_tools = False
        for msg in conv:
            tools_raw = msg.get("tools")
            if tools_raw:
                has_tools = True
                try:
                    for item in json.loads(tools_raw):
                        name = item.get("function", {}).get("name")
                        tool_names[name] += 1
                except Exception:  # noqa: BLE001
                    tool_names["<parse_error>"] += 1
        with_tools += has_tools
        if no_tools_gt_sample is None and (not has_tools) and gt:
            no_tools_gt_sample = rec
    report["agent_rl"] = {
        "total": agent_total,
        "final_assistant_empty": empty,
        "last_message_role": dict(last_role),
        "with_tools": with_tools,
        "with_gt": with_gt,
        "tool_names": tool_names.most_common(80),
        "sample_no_tools_with_gt": clip(no_tools_gt_sample),
    }

    json.dump(report, sys.stdout, ensure_ascii=False, indent=2)
    print()


if __name__ == "__main__":
    main()
