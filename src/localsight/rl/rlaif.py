"""RLAIF 一轮：on-policy 采样 K 条 → judge 打分 → 组内组对 → SimPO 更新。"""
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
from localsight.rl.agent_grpo import decode, rollout_logps
from localsight.rl.judge import JudgeClient, RuleJudge
from localsight.rl.losses import simpo_loss
from localsight.tokenizer import LocalSightTokenizer
from localsight.utils.config import resolve_stage_config


def sample_k(
    model,
    prompt_ids: torch.Tensor,
    tokenizer,
    max_new: int,
    temperature: float,
    top_p: float,
    k: int,
) -> list[tuple[torch.Tensor, str]]:
    cfg_cache = KVCache(
        model.module.config.num_hidden_layers,
        1,
        model.module.config.num_key_value_heads,
        model.module.config.head_dim,
        max_new + prompt_ids.shape[1] + 8,
        dtype=torch.bfloat16,
        device=prompt_ids.device,
    )
    model(prompt_ids, cache=cfg_cache)
    cfg_cache.commit()
    children = cfg_cache.spawn(k)
    out = []
    for child in children:
        ids, text = decode(model, child, prompt_ids[:, -1:], tokenizer, max_new,
                           temperature, top_p, {tokenizer.im_end_id})
        out.append((torch.cat([prompt_ids, ids], dim=1), text))
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--tokenizer", default="data/tokenizer")
    parser.add_argument("--start-checkpoint", required=True)
    parser.add_argument("--question-file", default=None, help="问题文本（每行一题，用于 judge）")
    args = parser.parse_args()

    rank = int(os.environ["LOCAL_RANK"])
    world = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(rank)
    dist.init_process_group(backend="nccl")

    cfg, model_cfg = resolve_stage_config(Path(args.config))
    tokenizer = LocalSightTokenizer(args.tokenizer)
    model = LocalsightForCausalLM(model_cfg).to(f"cuda:{rank}")
    state = torch.load(Path(args.start_checkpoint) / "model.pt", map_location="cpu")
    model.load_state_dict(state)
    model = DDP(model, device_ids=[rank])
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=0.0)

    prompts = np.memmap(Path(args.data_dir) / "prompts.bin", dtype=np.int32, mode="r")
    prompt_lens = np.memmap(Path(args.data_dir) / "prompt_len.bin", dtype=np.int32, mode="r")
    manifest = json.loads((Path(args.data_dir) / "manifest.json").read_text())
    max_len = manifest["max_len"]
    n_prompts = manifest["prompts"]
    questions = []
    if args.question_file:
        questions = Path(args.question_file).read_text(encoding="utf-8").splitlines()

    judge: JudgeClient = RuleJudge()  # TODO(M7): 换 vLLM 版 LLM judge
    k = cfg["sampling"]["k"]
    batch = cfg["micro_batch_size"]
    model.eval()
    start = time.time()

    for step_start in range(rank, n_prompts, world * batch):
        step_prompts = list(range(step_start, min(step_start + batch, n_prompts)))
        if not step_prompts:
            continue
        pairs_chosen, pairs_rejected, chosen_scores = [], [], []
        for pi in step_prompts:
            plen = int(prompt_lens[pi])
            prompt_ids = torch.tensor(
                [prompts[pi * max_len:pi * max_len + plen].astype(np.int64).tolist()],
                device=f"cuda:{rank}",
            )
            samples = sample_k(model, prompt_ids, tokenizer, cfg["max_new_tokens"],
                               cfg["sampling"]["temperature"], cfg["sampling"]["top_p"], k)
            question = questions[pi] if pi < len(questions) else f"prompt_{pi}"
            scored = []
            for ids, text in samples:
                result = judge.score(question, text)
                scored.append((result.score, ids, text))
            scored.sort(key=lambda x: x[0], reverse=True)
            best = scored[0]
            worst = scored[-1]
            pairs_chosen.append((best[1], best[0]))
            pairs_rejected.append((worst[1], worst[0]))
            chosen_scores.append(best[0])

        model.train()
        losses = []
        for (ids_c, _), (ids_r, _) in zip(pairs_chosen, pairs_rejected):
            mc, lc = rollout_logps(model, ids_c, torch.ones(ids_c.shape[1] - 1, device=ids_c.device).bool().unsqueeze(0))
            mr, lr = rollout_logps(model, ids_r, torch.ones(ids_r.shape[1] - 1, device=ids_r.device).bool().unsqueeze(0))
            pi_c = (mc.sum() / lc).unsqueeze(0).unsqueeze(0)
            pi_r = (mr.sum() / lr).unsqueeze(0).unsqueeze(0)
            losses.append(simpo_loss(pi_c, pi_r, beta=cfg["beta"], gamma=cfg["gamma"]))
        loss = torch.stack(losses).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()
        model.eval()
        if rank == 0 and (step_start // batch) % 5 == 0:
            print(f"step={step_start} loss={loss.item():.4f} "
                  f"best_score={sum(chosen_scores)/max(len(chosen_scores),1):.2f}")

    dist.barrier()
    if rank == 0:
        out = Path(args.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        torch.save(model.module.state_dict(), out / "model.pt")
        print(f"rlaif 一轮完成，耗时 {(time.time() - start) / 60:.1f} min")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
