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
    train_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=cfg["micro_batch_size"],
        gradient_accumulation_steps=cfg["grad_accum"],
        num_train_epochs=cfg["epochs"],
        learning_rate=cfg["lr"],
        weight_decay=cfg["wd"],
        warmup_ratio=cfg["warmup_ratio"],
        bf16=True,
        logging_steps=cfg.get("log_interval", 10),
        save_steps=cfg.get("save_interval", 1000),
        save_total_limit=3,
        seed=cfg.get("seed", 42),
        neftune_noise_alpha=cfg.get("neftune_alpha"),
        report_to=[],
        ddp_find_unused_parameters=True,
    )
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
