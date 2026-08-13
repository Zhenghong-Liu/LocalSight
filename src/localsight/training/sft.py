"""SFT 训练：HF Trainer + NEFTune + packing（document mask 走自定义 collator）。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import Trainer, TrainingArguments

from localsight.data import SFTDataset
from localsight.model import LocalsightHFConfig, LocalsightHFForCausalLM
from localsight.utils.config import resolve_stage_config


def collate(batch: list[dict]) -> dict[str, torch.Tensor]:
    input_ids = torch.stack([x["input_ids"] for x in batch])
    labels = torch.stack([x["labels"] for x in batch])
    document_ids = torch.stack([x["document_ids"] for x in batch])
    return {"input_ids": input_ids, "labels": labels, "document_ids": document_ids}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--start-checkpoint", default=None)
    parser.add_argument("--max-steps", type=int, default=None, help="冒烟测试时限制步数")
    parser.add_argument("--micro-batch", type=int, default=None)
    parser.add_argument("--grad-accum", type=int, default=None)
    parser.add_argument("--no-bf16", action="store_true", help="CPU 冒烟时关闭 bf16")
    args = parser.parse_args()

    cfg, model_cfg = resolve_stage_config(Path(args.config))
    hf_cfg = LocalsightHFConfig(**model_cfg.to_dict())
    if args.start_checkpoint:
        model = LocalsightHFForCausalLM(hf_cfg)
        ckpt = Path(args.start_checkpoint)
        if (ckpt / "config.json").exists():
            model = LocalsightHFForCausalLM.from_pretrained(ckpt)
        else:
            state = torch.load(ckpt / "model.pt", map_location="cpu")
            model.core.load_state_dict(state)
    else:
        model = LocalsightHFForCausalLM(hf_cfg)

    dataset = SFTDataset(Path(args.data_dir))
    batch = args.micro_batch or cfg["micro_batch_size"]
    accum = args.grad_accum or cfg["grad_accum"]
    steps_per_epoch = max(1, len(dataset) // (batch * accum))
    warmup_steps = int(steps_per_epoch * cfg["epochs"] * cfg["warmup_ratio"])
    train_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=batch,
        gradient_accumulation_steps=accum,
        num_train_epochs=cfg["epochs"],
        learning_rate=cfg["lr"],
        weight_decay=cfg["wd"],
        warmup_steps=warmup_steps,
        bf16=not args.no_bf16,
        logging_steps=cfg.get("log_interval", 10),
        save_steps=cfg.get("save_interval", 1000),
        save_total_limit=3,
        seed=cfg.get("seed", 42),
        neftune_noise_alpha=cfg.get("neftune_alpha"),
        report_to=[],
        ddp_find_unused_parameters=True,
    )
    if args.max_steps:
        train_args.max_steps = args.max_steps
        train_args.num_train_epochs = 1.0
    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=dataset,
        data_collator=collate,
    )
    trainer.train()
    model.save_pretrained(Path(args.output_dir) / "final")
    (Path(args.output_dir) / "final" / "manifest.json").write_text(
        json.dumps(dataset.manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
