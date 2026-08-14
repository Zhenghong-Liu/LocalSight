#!/usr/bin/env python3
"""RLAIF 阶段 B：transformers judge 打分，输出 scored jsonl（每个样本加 score 字段）。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from localsight.rl.judge import build_judge_prompt, parse_judge_score


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", required=True, help="rlaif_sample 输出目录")
    parser.add_argument("--questions", required=True, help="questions.jsonl")
    parser.add_argument("--judge-model", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--batch", type=int, default=16)
    args = parser.parse_args()

    questions = [json.loads(line) for line in Path(args.questions).read_text(encoding="utf-8").splitlines()]
    samples_dir = Path(args.samples)
    records = []
    for f in sorted(samples_dir.glob("samples_rank*.jsonl")):
        for line in f.read_text(encoding="utf-8").splitlines():
            if line.strip():
                records.append(json.loads(line))

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.judge_model)
    judge = AutoModelForCausalLM.from_pretrained(
        args.judge_model, torch_dtype=torch.bfloat16, device_map="cuda:0"
    ).eval()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as f:
        for start in range(0, len(records), args.batch):
            chunk = records[start:start + args.batch]
            judge_prompts = [
                build_judge_prompt(questions[r["idx"]], r["text"]) for r in chunk
            ]
            inputs = tokenizer(
                judge_prompts, return_tensors="pt", padding=True,
                truncation=True, max_length=4096,
            ).to("cuda:0")
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                outputs = judge.generate(
                    **inputs, max_new_tokens=128, do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )
            new_tokens = outputs[:, inputs["input_ids"].shape[1]:]
            for rec, ids in zip(chunk, new_tokens):
                raw = tokenizer.decode(ids, skip_special_tokens=True)
                score = parse_judge_score(raw)
                rec["score"] = score if score is not None else 0.0
                rec["judge_raw"] = raw
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            print(f"judged {min(start + args.batch, len(records))}/{len(records)}", flush=True)
    print("done:", out_path)


if __name__ == "__main__":
    main()
