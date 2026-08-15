"""训练期定时抽查：think 开/关批量生成、语料续写、快速 PPL 与基准快评。

设计要点：
- 生成走批量 decode + padded attention mask（复用 KVCache），一次前向完成整批；
- 只做评估，不改训练状态；调用方负责 model.eval()/train() 切换与 barrier；
- 产物：<step>.jsonl（结构化记录）、<step>.txt（人读）、metrics.jsonl（时间序列）。
"""
from __future__ import annotations

import json
import math
import time
from collections import Counter
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F

from localsight.eval.chat_harness import extract_think
from localsight.model import KVCache

THINK_ON_PREFIX = "<think>\n"
THINK_OFF_PREFIX = "<think>\n\n</think>\n\n"


def rep_n(text: str, n: int = 4) -> float:
    """重复 n-gram 覆盖比例（seq-rep-n 的简化版），返回 [0,1]。"""
    text = text.strip()
    if len(text) < n:
        return 0.0
    counts = Counter(text[i : i + n] for i in range(len(text) - n + 1))
    repeated = {gram for gram, count in counts.items() if count > 1}
    if not repeated:
        return 0.0
    covered = sum(1 for i in range(len(text) - n + 1) if text[i : i + n] in repeated)
    return covered / (len(text) - n + 1)


def build_chat_ids(tokenizer, user_text: str, think_on: bool) -> list[int]:
    think = THINK_ON_PREFIX if think_on else THINK_OFF_PREFIX
    text = f"<|im_start|>user\n{user_text}<|im_end|>\n<|im_start|>assistant\n{think}"
    return tokenizer.encode(text)


@torch.no_grad()
def generate_padded_batch(
    model,
    tokenizer,
    prompts: list[list[int]],
    max_new: int,
    temperature: float,
    top_p: float,
    stop_id: int,
    *,
    model_cfg,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> list[str]:
    """不等长 prompt 的批量解码：pad 后带掩码预填 + 逐步生成。返回每个样本的解码文本。"""
    b = len(prompts)
    device = device or next(model.parameters()).device
    if dtype is None:
        # GPU 上前向走 bf16 autocast；CPU 上与模型参数 dtype 一致（避免 SDPA dtype 不匹配）
        dtype = torch.bfloat16 if device.type == "cuda" else next(model.parameters()).dtype
    max_len = max(len(p) for p in prompts)
    ids = torch.zeros((b, max_len), dtype=torch.long, device=device)
    valid = torch.zeros((b, max_len), dtype=torch.bool, device=device)
    for i, p in enumerate(prompts):
        ids[i, : len(p)] = torch.tensor(p, dtype=torch.long, device=device)
        valid[i, : len(p)] = True

    causal = torch.tril(torch.ones(max_len, max_len, dtype=torch.bool, device=device))
    prefill_mask = (valid[:, None, :, None] & valid[:, None, None, :] & causal[None, None])
    cache = KVCache(
        model_cfg.num_hidden_layers,
        b,
        model_cfg.num_key_value_heads,
        model_cfg.head_dim,
        max_len + max_new,
        dtype=dtype,
        device=device,
    )
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
        model(ids, attention_mask=prefill_mask, cache=cache)
    cache.commit()

    generated = torch.full((b, max_new), 0, dtype=torch.long, device=device)
    active = torch.ones(b, dtype=torch.bool, device=device)
    stop_step = torch.full((b,), max_new, dtype=torch.long, device=device)
    token = ids[:, -1:]
    key_valid = torch.cat(
        [valid, torch.ones((b, max_new), dtype=torch.bool, device=device)], dim=1
    )
    last_step = max_new - 1
    for step in range(max_new):
        total = cache.length + 1
        step_mask = key_valid[:, None, None, :total]
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            logits = model(token, attention_mask=step_mask, cache=cache)[0]
        cache.commit()
        next_logits = logits[:, -1, :]
        if temperature <= 0:
            next_token = next_logits.argmax(dim=-1)
        else:
            probs = F.softmax(next_logits / temperature, dim=-1)
            sorted_p, sorted_idx = probs.sort(dim=-1, descending=True)
            cum = sorted_p.cumsum(dim=-1)
            cutoff = cum > top_p
            cutoff[..., 1:] = cutoff[..., :-1].clone()
            cutoff[..., 0] = False
            sorted_p = sorted_p.masked_fill(cutoff, 0.0)
            sorted_p /= sorted_p.sum(dim=-1, keepdim=True).clamp(min=1e-9)
            next_token = sorted_idx.gather(-1, torch.multinomial(sorted_p, 1)).squeeze(-1)
        generated[:, step] = next_token
        token = next_token.unsqueeze(1)
        token[~active] = 0
        just_stopped = active & (next_token == stop_id) & (stop_step == max_new)
        stop_step[just_stopped] = step
        active &= next_token != stop_id
        if not active.any():
            last_step = step
            break
    texts = []
    for i in range(b):
        end = min(int(stop_step[i].item()) + 1, last_step + 1)
        texts.append(tokenizer.decode(generated[i, :end].tolist()))
    return texts


@torch.no_grad()
def quick_ppl(model, val_loader, device: torch.device) -> tuple[float, int]:
    """在 val 集上算 perplexity，按有效 label 数加权。"""
    total = 0.0
    n = 0
    for batch in val_loader:
        ids = batch["input_ids"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)
        docs = batch["document_ids"].to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            _, loss, _ = model(ids, labels=labels, document_ids=docs)
        valid = (labels != -100).sum().item()
        total += loss.item() * valid
        n += valid
    ppl = math.exp(total / max(n, 1))
    return ppl, n


@torch.no_grad()
def continuation_prompts(val_loader, count: int = 10, prefix_len: int = 128) -> list[list[int]]:
    """从 val 集抽 count 条序列，取前 prefix_len 个 token 作为续写 prompt。"""
    prompts: list[list[int]] = []
    for batch in val_loader:
        ids = batch["input_ids"]
        labels = batch["labels"]
        for i in range(ids.shape[0]):
            valid_len = int((labels[i] != -100).sum().item())
            if valid_len < prefix_len + 32:
                continue
            prompts.append(ids[i, :prefix_len].tolist())
            if len(prompts) >= count:
                return prompts
    return prompts


def run_sample_eval(
    model,
    tokenizer,
    prompts: list[str],
    val_loader,
    out_dir: Path,
    step: int,
    *,
    model_cfg,
    tag: str = "pretrain",
    max_new: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.9,
    device: Optional[torch.device] = None,
) -> dict:
    """执行一次定时抽查：think 开/关 + 续写 + PPL，写文件并返回指标。"""
    device = device or next(model.parameters()).device
    stop_id = tokenizer.im_end_id
    records: list[dict] = []

    def gen(kind: str, prompt_ids: list[list[int]], prompts_text: list[str]) -> None:
        texts = generate_padded_batch(
            model, tokenizer, prompt_ids, max_new, temperature, top_p, stop_id,
            model_cfg=model_cfg, device=device,
        )
        for text, prompt in zip(texts, prompts_text):
            records.append(
                {
                    "kind": kind,
                    "prompt": prompt[:120],
                    "output": text,
                    "n_chars": len(text),
                    "think_chars": len(extract_think(text)),
                    "rep4": round(rep_n(text, 4), 4),
                }
            )

    on_ids = [build_chat_ids(tokenizer, p, think_on=True) for p in prompts]
    off_ids = [build_chat_ids(tokenizer, p, think_on=False) for p in prompts]
    gen("thinking_on", on_ids, prompts)
    gen("thinking_off", off_ids, prompts)

    cont_prompts = continuation_prompts(val_loader, count=10) if val_loader is not None else []
    if cont_prompts:
        gen("continuation", cont_prompts, [tokenizer.decode(p) for p in cont_prompts])

    ppl, n_val = quick_ppl(model, val_loader, device) if val_loader is not None else (float("nan"), 0)

    on_recs = [r for r in records if r["kind"] == "thinking_on"]
    off_recs = [r for r in records if r["kind"] == "thinking_off"]
    on_len = [r["n_chars"] for r in on_recs]
    off_len = [r["n_chars"] for r in off_recs]
    metrics = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "step": step,
        "think_trigger_rate": round(sum(r["think_chars"] > 0 for r in on_recs) / max(len(on_recs), 1), 4),
        "mean_length_ratio": round(
            sum(o / max(f, 1) for o, f in zip(on_len, off_len)) / max(len(on_recs), 1), 4
        ),
        "rep4_mean": round(sum(r["rep4"] for r in records) / max(len(records), 1), 4),
        "ppl": None if math.isnan(ppl) else round(ppl, 4),
        "n_val_tokens": n_val,
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = f"step-{step:04d}"
    with open(out_dir / f"{tag}-{stamp}.jsonl", "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps({**{"step": step}, **r}, ensure_ascii=False) + "\n")
    with open(out_dir / f"{tag}-{stamp}.txt", "w", encoding="utf-8") as f:
        f.write(f"# 定时抽查 step={step}  {metrics['ts']}\n")
        f.write(f"# think_trigger_rate={metrics['think_trigger_rate']} "
                f"length_ratio={metrics['mean_length_ratio']} "
                f"rep4={metrics['rep4_mean']} ppl={metrics['ppl']}\n\n")
        for r in records:
            f.write(f"## [{r['kind']}] {r['prompt']}\n{r['output']}\n\n")
    with open(out_dir / "metrics.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(metrics, ensure_ascii=False) + "\n")
    return metrics


def run_quick_bench(
    model,
    tokenizer,
    data_dir: str,
    *,
    model_cfg,
    limit: int = 100,
    device: Optional[torch.device] = None,
) -> dict[str, dict]:
    """MMLU / C-Eval 各 limit 条的贪心快评（与 run_evals.py 同口径）。"""
    from localsight.eval.benchmarks import load_benchmark, run_mc

    device = device or next(model.parameters()).device
    stop_id = tokenizer.im_end_id

    def chat(text: str) -> str:
        ids = build_chat_ids(tokenizer, text, think_on=False)
        return generate_padded_batch(
            model, tokenizer, [ids], 64, 0.0, 1.0, stop_id,
            model_cfg=model_cfg, device=device,
        )[0]

    results = {}
    for name in ("mmlu", "ceval"):
        ds = load_benchmark(data_dir, name)
        results[name] = run_mc(chat, name, ds, limit=limit)
    return results
