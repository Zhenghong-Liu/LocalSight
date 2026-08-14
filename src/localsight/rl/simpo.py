"""SimPO 训练：无参考、长度归一化、target margin。"""
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
from torch.utils.data import DataLoader, Dataset, DistributedSampler

from localsight.model import LocalsightConfig, LocalsightForCausalLM
from localsight.rl.losses import simpo_loss
from localsight.utils.config import resolve_stage_config


class DPODataset(Dataset):
    def __init__(self, data_dir: Path):
        manifest = json.loads((data_dir / "manifest.json").read_text(encoding="utf-8"))
        self.max_len = manifest["max_len"]
        self.prompt = np.memmap(data_dir / "prompt.bin", dtype=np.int32, mode="r")
        self.chosen = np.memmap(data_dir / "chosen.bin", dtype=np.int32, mode="r").reshape(-1, self.max_len)
        self.rejected = np.memmap(data_dir / "rejected.bin", dtype=np.int32, mode="r").reshape(-1, self.max_len)
        self.prompt_len = np.memmap(data_dir / "prompt_len.bin", dtype=np.int32, mode="r")

    def __len__(self) -> int:
        return len(self.prompt_len)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        pl = int(self.prompt_len[idx])
        return {
            "prompt_len": torch.tensor(pl, dtype=torch.long),
            "chosen_ids": torch.from_numpy(self.chosen[idx].copy().astype(np.int64)),
            "rejected_ids": torch.from_numpy(self.rejected[idx].copy().astype(np.int64)),
        }


def response_logps(model, ids: torch.Tensor, prompt_lens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """返回 (响应 token 的 masked logp, 响应长度)，用于 SimPO 长度归一化。"""
    logits = model(ids)[0]
    shift_logp = F.log_softmax(logits, dim=-1)[:, :-1].gather(-1, ids[:, 1:, None]).squeeze(-1)
    mask = torch.arange(shift_logp.shape[1], device=ids.device)[None] >= (prompt_lens - 1)[:, None]
    return shift_logp * mask, mask.sum(dim=-1).clamp(min=1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--start-checkpoint", default=None)
    args = parser.parse_args()

    rank = int(os.environ["LOCAL_RANK"])
    world = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(rank)
    dist.init_process_group(backend="nccl")

    cfg, model_cfg = resolve_stage_config(Path(args.config))
    model = LocalsightForCausalLM(model_cfg).to(f"cuda:{rank}")
    if args.start_checkpoint:
        model.load_state_dict(torch.load(Path(args.start_checkpoint) / "model.pt", map_location="cpu"))
    model = DDP(model, device_ids=[rank], find_unused_parameters=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=0.0)

    dataset = DPODataset(Path(args.data_dir))
    sampler = DistributedSampler(dataset, num_replicas=world, rank=rank, shuffle=True, seed=cfg.get("seed", 42))
    loader = DataLoader(dataset, batch_size=cfg["micro_batch_size"], sampler=sampler, drop_last=True)

    model.train()
    start = time.time()
    for step, batch in enumerate(loader):
        ids_c = batch["chosen_ids"].cuda(rank)
        ids_r = batch["rejected_ids"].cuda(rank)
        pl = batch["prompt_len"].cuda(rank)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            # 单次前向同时算 chosen/rejected，保证 DDP 计算图每步一致
            ids = torch.cat([ids_c, ids_r], dim=0)
            pl_all = torch.cat([pl, pl], dim=0)
            masked, resp_len = response_logps(model, ids, pl_all)
            lp_c, lp_r = masked.chunk(2, dim=0)
            len_c, len_r = resp_len.chunk(2, dim=0)
            pi_c = lp_c.sum(dim=-1) / len_c
            pi_r = lp_r.sum(dim=-1) / len_r
            loss = simpo_loss(pi_c[:, None], pi_r[:, None], beta=cfg["beta"], gamma=cfg["gamma"])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()
        if rank == 0 and step % cfg.get("log_interval", 10) == 0:
            with torch.no_grad():
                margin = pi_c.mean() - pi_r.mean()
            print(f"step={step} loss={loss.item():.4f} margin={margin.item():.4f}")
        if (step + 1) % cfg["save_interval"] == 0:
            dist.barrier()
            if rank == 0:
                out = Path(args.output_dir)
                out.mkdir(parents=True, exist_ok=True)
                torch.save(model.module.state_dict(), out / "model.pt")

    dist.barrier()
    if rank == 0:
        out = Path(args.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        torch.save(model.module.state_dict(), out / "model.pt")
        print(f"simpo 完成，耗时 {(time.time() - start) / 60:.1f} min")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
