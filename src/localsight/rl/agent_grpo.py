"""Agent RL（GRPO + DAPO）：prefix cache rollout + 工具执行循环 + 组内优势。"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

from localsight.model import KVCache, LocalsightConfig, LocalsightForCausalLM
from localsight.rl.losses import grpo_dapo_loss, group_advantages
from localsight.rl.rewards import composite_reward
from localsight.tokenizer import LocalSightTokenizer
from localsight.tools.executor import execute_tool, parse_tool_calls
from localsight.utils.config import resolve_stage_config


@torch.no_grad()
def decode(
    model,
    cache: KVCache,
    current_ids: torch.Tensor,
    tokenizer,
    max_new: int,
    temperature: float,
    top_p: float,
    stop_ids: set[int],
) -> tuple[torch.Tensor, str]:
    """从 current_ids 继续解码直到 stop token；返回 (完整新 tokens, 解码文本)。"""
    generated: list[int] = []
    for _ in range(max_new):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(current_ids, cache=cache)[0]
        cache.commit()
        next_logits = logits[:, -1, :]
        if temperature <= 0:
            token = next_logits.argmax(dim=-1, keepdim=True)
        else:
            probs = F.softmax(next_logits / temperature, dim=-1)
            token = torch.multinomial(probs, num_samples=1)
        tid = int(token.item())
        generated.append(tid)
        current_ids = token
        if tid in stop_ids:
            break
    text = tokenizer.decode(generated)
    return torch.tensor([generated], dtype=torch.long, device=cache.device), text


def rollout_one(
    model,
    prompt_ids: torch.Tensor,
    tokenizer,
    cfg: dict,
) -> tuple[torch.Tensor, str, torch.Tensor]:
    """单条 rollout：prompt 预填 + 至多 max_tool_rounds 轮工具交互。"""
    device = prompt_ids.device
    max_new = cfg["max_new_tokens"]
    temperature = cfg["sampling"]["temperature"]
    top_p = cfg["sampling"]["top_p"]
    stop_ids = {tokenizer.im_end_id}

    cache = KVCache(
        cfg["model_cfg"].num_hidden_layers, 1, cfg["model_cfg"].num_key_value_heads,
        cfg["model_cfg"].head_dim, cfg["seq_len"], dtype=torch.bfloat16, device=device,
    )
    with torch.autocast("cuda", dtype=torch.bfloat16):
        model(prompt_ids, cache=cache)
    cache.commit()
    all_ids = prompt_ids.clone()
    gen_mask = torch.zeros(all_ids.shape[1], dtype=torch.bool, device=device)
    tail = prompt_ids[:, -1:]
    texts: list[str] = []
    pos = prompt_ids.shape[1]
    for _ in range(cfg["max_tool_rounds"]):
        turn_ids, text = decode(model, cache, tail, tokenizer, max_new, temperature, top_p, stop_ids)
        texts.append(text)
        all_ids = torch.cat([all_ids, turn_ids], dim=1)
        gen_mask = torch.cat([gen_mask, torch.ones(turn_ids.shape[1], dtype=torch.bool, device=device)])
        pos += turn_ids.shape[1]
        calls = parse_tool_calls(text)
        if not calls:
            break
        responses = []
        for name, args in calls:
            try:
                responses.append(execute_tool(name, args))
            except Exception:  # noqa: BLE001
                responses.append('{"error": "execution failed"}')
        tool_text = "<|im_start|>user\n<tool_response>\n" + "\n".join(responses) + \
            "\n</tool_response><|im_end|>\n<|im_start|>assistant\n<think>\n"
        tail_ids = tokenizer.encode(tool_text)
        tail = torch.tensor([tail_ids], dtype=torch.long, device=device)
        all_ids = torch.cat([all_ids, tail], dim=1)
        gen_mask = torch.cat([gen_mask, torch.zeros(len(tail_ids), dtype=torch.bool, device=device)])
        pos += len(tail_ids)
        tail = tail[:, -1:]
    return all_ids, "".join(texts), gen_mask


def rollout_logps(model, ids: torch.Tensor, gen_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """生成部分的 per-token logp；返回 (masked_logp, 响应长度)。调用方决定是否 no_grad。"""
    logits = model(ids)[0]
    shift = F.log_softmax(logits, dim=-1)[:, :-1].gather(-1, ids[:, 1:, None]).squeeze(-1)
    mask = gen_mask[1:].unsqueeze(0)
    return shift * mask, mask.sum(dim=-1).clamp(min=1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--tokenizer", default="data/tokenizer")
    parser.add_argument("--start-checkpoint", required=True)
    args = parser.parse_args()

    rank = int(os.environ["LOCAL_RANK"])
    world = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(rank)
    dist.init_process_group(backend="nccl")

    cfg, model_cfg = resolve_stage_config(Path(args.config))
    cfg["model_cfg"] = model_cfg
    tokenizer = LocalSightTokenizer(args.tokenizer)

    model = LocalsightForCausalLM(model_cfg).to(f"cuda:{rank}")
    state = torch.load(Path(args.start_checkpoint) / "model.pt", map_location="cpu")
    model.load_state_dict(state)
    model = DDP(model, device_ids=[rank])
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=0.0)

    prompts = np.memmap(Path(args.data_dir) / "prompts.bin", dtype=np.int32, mode="r")
    prompt_lens = np.memmap(Path(args.data_dir) / "prompt_len.bin", dtype=np.int32, mode="r")
    max_len = json.loads((Path(args.data_dir) / "manifest.json").read_text())["max_len"]
    n_prompts = prompts.shape[0] // max_len
    gt_rows = [json.loads(line) for line in open(Path(args.data_dir) / "gt.jsonl", encoding="utf-8")]

    model.eval()
    g = cfg["sampling"]["group_size"]
    prompts_per_step = cfg["micro_batch_size"]
    start = time.time()
    for step_start in range(rank, n_prompts, world * prompts_per_step):
        step_prompts = list(range(step_start, min(step_start + prompts_per_step, n_prompts)))
        if not step_prompts:
            continue
        rollout_data: list[tuple[torch.Tensor, torch.Tensor, float]] = []
        rewards_list = []
        for pi in step_prompts:
            plen = int(prompt_lens[pi])
            prompt_ids = torch.tensor(
                [prompts[pi * max_len:pi * max_len + plen].astype(np.int64).tolist()],
                device=f"cuda:{rank}",
            )
            group_rewards = []
            for _ in range(g):
                ids, text, gen_mask = rollout_one(model, prompt_ids, tokenizer, cfg)
                with torch.no_grad():
                    masked_old, resp_len = rollout_logps(model, ids, gen_mask)
                reward = composite_reward(text, gt_rows[pi]["expect_tool"], gt_rows[pi]["gt"])
                old_mean = (masked_old.sum() / resp_len).item()
                rollout_data.append((ids, gen_mask, old_mean))
                group_rewards.append(reward)
            rewards_list.append(group_rewards)

        rewards = torch.tensor(rewards_list, device=f"cuda:{rank}")
        advantages = group_advantages(rewards)
        model.train()
        cur_means = []
        for ids, gen_mask, _ in rollout_data:
            masked_cur, resp_len = rollout_logps(model, ids, gen_mask)  # 保留梯度
            cur_means.append((masked_cur.sum() / resp_len).unsqueeze(0))
        cur = torch.cat(cur_means)
        old = torch.tensor([x[2] for x in rollout_data], device=f"cuda:{rank}")
        loss = grpo_dapo_loss(
            old.unsqueeze(1), cur.unsqueeze(1), advantages.reshape(-1),
            clip_low=cfg["loss"]["clip_low"], clip_high=cfg["loss"]["clip_high"],
            token_level=cfg["loss"]["token_level"],
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()
        model.eval()
        if rank == 0 and (step_start // prompts_per_step) % 10 == 0:
            print(
                f"step={step_start} loss={loss.item():.4f} "
                f"mean_reward={rewards.mean().item():.3f} "
                f"gt_hit={(rewards > 0.9).float().mean().item():.3f}"
            )

    dist.barrier()
    if rank == 0:
        out = Path(args.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        torch.save(model.module.state_dict(), out / "model.pt")
        print(f"agent_rl 完成，耗时 {(time.time() - start) / 60:.1f} min")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
