#!/usr/bin/env python3
"""RLAIF 阶段 B：vLLM judge 打分，输出 scored jsonl（每个样本加 score 字段）。"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

_site = Path(sys.prefix) / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages"
os.environ["PATH"] = (
    str(Path(sys.prefix) / "bin")
    + os.pathsep
    + str(_site / "nvidia" / "cuda_nvcc" / "bin")
    + os.pathsep
    + os.environ.get("PATH", "")
)

from vllm import LLM, SamplingParams

from localsight.rl.judge import build_judge_prompt, parse_judge_score


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", required=True, help="rlaif_sample 输出目录")
    parser.add_argument("--questions", required=True, help="questions.jsonl")
    parser.add_argument("--judge-model", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--batch", type=int, default=256)
    args = parser.parse_args()

    questions = [json.loads(line) for line in Path(args.questions).read_text(encoding="utf-8").splitlines()]
    samples_dir = Path(args.samples)
    records = []
    for f in sorted(samples_dir.glob("samples_rank*.jsonl")):
        for line in f.read_text(encoding="utf-8").splitlines():
            if line.strip():
                records.append(json.loads(line))

    llm = LLM(model=args.judge_model, tensor_parallel_size=1, dtype="bfloat16",
              max_model_len=8192, gpu_memory_utilization=0.95)
    sampling = SamplingParams(temperature=0.0, max_tokens=128)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as f:
        for start in range(0, len(records), args.batch):
            chunk = records[start:start + args.batch]
            judge_prompts = [
                build_judge_prompt(questions[r["idx"]], r["text"]) for r in chunk
            ]
            outputs = llm.generate(judge_prompts, sampling)
            for rec, out in zip(chunk, outputs):
                raw = out.outputs[0].text
                score = parse_judge_score(raw)
                rec["score"] = score if score is not None else 0.0
                rec["judge_raw"] = raw
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            print(f"judged {min(start + args.batch, len(records))}/{len(records)}", flush=True)
    print("done:", out_path)


if __name__ == "__main__":
    main()
