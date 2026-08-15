#!/usr/bin/env python3
"""SFT 质量裁判：抽样指标 + MMLU/C-Eval 快评 + 7B judge 打分 → gate.json + 报告。

用法（服务器项目根目录）：
    .venv/bin/python scripts/sft_quality_check.py \
        --checkpoint artifacts/sft_v2/final \
        --judge-model models/judge/Qwen--Qwen2.5-7B-Instruct \
        --out-dir artifacts/sft_v2/quality
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from localsight.eval.periodic import run_quick_bench, run_sample_eval
from localsight.model import LocalsightConfig, LocalsightForCausalLM
from localsight.tokenizer import LocalSightTokenizer


def load_model(checkpoint: Path, device: torch.device):
    cfg = LocalsightConfig()
    if (checkpoint / "config.json").exists():
        from localsight.model import LocalsightHFForCausalLM

        model = LocalsightHFForCausalLM.from_pretrained(checkpoint)
        return model.core.to(device), cfg
    model = LocalsightForCausalLM(cfg).to(device)
    model.load_state_dict(torch.load(checkpoint / "model.pt", map_location=device))
    return model, cfg


def judge_scores(judge_model: str, pairs: list[tuple[str, str]], batch: int = 8) -> list[float]:
    if not pairs:
        return []
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from localsight.rl.judge import build_judge_prompt, parse_judge_score

    tokenizer = AutoTokenizer.from_pretrained(judge_model)
    model = AutoModelForCausalLM.from_pretrained(
        judge_model, torch_dtype=torch.bfloat16, device_map="cuda:0"
    ).eval()
    scores: list[float] = []
    for start in range(0, len(pairs), batch):
        chunk = pairs[start : start + batch]
        inputs = tokenizer(
            [build_judge_prompt(q, a) for q, a in chunk],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=4096,
        ).to("cuda:0")
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            outputs = model.generate(
                **inputs, max_new_tokens=128, do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        new_tokens = outputs[:, inputs["input_ids"].shape[1] :]
        for ids in new_tokens:
            raw = tokenizer.decode(ids, skip_special_tokens=True)
            score = parse_judge_score(raw)
            scores.append(float(score) if score is not None else 0.0)
        print(f"judged {min(start + batch, len(pairs))}/{len(pairs)}", flush=True)
    return scores


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--tokenizer", default="data/tokenizer")
    parser.add_argument("--prompts", default="data/eval/thinking_prompts.txt")
    parser.add_argument("--eval-data-dir", default="data/eval")
    parser.add_argument("--judge-model", default="models/judge/Qwen--Qwen2.5-7B-Instruct")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--bench-limit", type=int, default=100)
    parser.add_argument("--judge-limit", type=int, default=20)
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model, model_cfg = load_model(Path(args.checkpoint), device)
    model.eval()
    tokenizer = LocalSightTokenizer(args.tokenizer)
    prompts = [
        line.strip() for line in Path(args.prompts).read_text(encoding="utf-8").splitlines() if line.strip()
    ]

    out_dir = Path(args.out_dir)
    sample_dir = out_dir / "samples"
    metrics = run_sample_eval(
        model, tokenizer, prompts, None, sample_dir, 0,
        model_cfg=model_cfg, tag="sft", device=device,
    )
    bench = run_quick_bench(
        model, tokenizer, args.eval_data_dir, model_cfg=model_cfg, limit=args.bench_limit, device=device
    )

    pairs: list[tuple[str, str]] = []
    sample_file = sample_dir / "sft-step-0000.jsonl"
    if sample_file.exists():
        for line in sample_file.read_text(encoding="utf-8").splitlines():
            if not line.strip() or len(pairs) >= args.judge_limit:
                break
            rec = json.loads(line)
            if rec.get("kind") == "thinking_on":
                pairs.append((rec["prompt"], rec["output"]))
    scores = judge_scores(args.judge_model, pairs)
    judge_mean = sum(scores) / len(scores) if scores else 0.0

    gate = {
        "think_trigger_rate": metrics["think_trigger_rate"],
        "rep4_mean": metrics["rep4_mean"],
        "mmlu": bench["mmlu"]["acc"],
        "ceval": bench["ceval"]["acc"],
        "judge_mean": round(judge_mean, 2),
        "judge_n": len(scores),
        "pass_think": metrics["think_trigger_rate"] >= 0.3,
        "pass_rep": metrics["rep4_mean"] <= 0.55,
        "pass_judge": judge_mean >= 4.0,
    }
    gate["pass"] = gate["pass_think"] and gate["pass_rep"] and gate["pass_judge"]
    (out_dir / "gate.json").write_text(json.dumps(gate, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "report.md").write_text(
        "# SFT 质量报告\n\n"
        f"- 思考触发率：{metrics['think_trigger_rate']}\n"
        f"- rep-4 重复度：{metrics['rep4_mean']}\n"
        f"- MMLU / C-Eval（{args.bench_limit} 样本）：{bench['mmlu']['acc']} / {bench['ceval']['acc']}\n"
        f"- 7B judge 均分（{len(scores)} 条）：{judge_mean:.2f} / 10\n"
        f"- 判定：**{'通过' if gate['pass'] else '未通过'}**（think≥0.3、rep4≤0.55、judge≥4.0）\n\n"
        "抽样详见 `samples/sft-step-0000.txt`。\n",
        encoding="utf-8",
    )
    print("GATE " + json.dumps(gate, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
