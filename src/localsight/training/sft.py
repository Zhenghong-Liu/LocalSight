"""SFT 训练：HF Trainer + NEFTune + packing（document mask 走自定义 collator）。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import Trainer, TrainerCallback, TrainingArguments

from localsight.data import SFTDataset
from localsight.eval.periodic import run_sample_eval
from localsight.model import LocalsightHFConfig, LocalsightHFForCausalLM
from localsight.tokenizer import LocalSightTokenizer
from localsight.utils.config import resolve_stage_config


def collate(batch: list[dict]) -> dict[str, torch.Tensor]:
    input_ids = torch.stack([x["input_ids"] for x in batch])
    labels = torch.stack([x["labels"] for x in batch])
    document_ids = torch.stack([x["document_ids"] for x in batch])
    return {"input_ids": input_ids, "labels": labels, "document_ids": document_ids}


class SampleCallback(TrainerCallback):
    """每 sample_interval 步用 thinking prompts 抽样生成，写 artifacts（仅主进程）。"""

    def __init__(
        self,
        raw_model,
        model_cfg,
        tokenizer,
        prompts: list[str],
        out_dir: Path,
        interval: int,
        limit: int = 25,
        tag: str = "sft",
    ):
        self.raw_model = raw_model
        self.model_cfg = model_cfg
        self.tokenizer = tokenizer
        self.prompts = prompts
        self.out_dir = out_dir
        self.interval = interval
        self.limit = limit
        self.tag = tag

    def on_step_end(self, args, state, control, **kwargs):  # noqa: ARG002
        if (
            self.interval <= 0
            or not state.is_world_process_zero
            or state.global_step % self.interval != 0
        ):
            return control
        core = self.raw_model.core
        core.eval()
        try:
            metrics = run_sample_eval(
                core,
                self.tokenizer,
                self.prompts[: self.limit],
                None,
                self.out_dir,
                state.global_step,
                model_cfg=self.model_cfg,
                tag=self.tag,
            )
            print(
                f"[sft_sample] step={state.global_step} think_trigger={metrics['think_trigger_rate']} "
                f"len_ratio={metrics['mean_length_ratio']} rep4={metrics['rep4_mean']}",
                flush=True,
            )
        finally:
            core.train()
        return control


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--start-checkpoint", default=None)
    parser.add_argument("--max-steps", type=int, default=None, help="冒烟测试时限制步数")
    parser.add_argument("--micro-batch", type=int, default=None)
    parser.add_argument("--grad-accum", type=int, default=None)
    parser.add_argument("--epochs", type=float, default=None, help="覆盖训练轮数")
    parser.add_argument("--no-bf16", action="store_true", help="CPU 冒烟时关闭 bf16")
    parser.add_argument("--no-cuda", action="store_true", help="CPU 冒烟时强制不用 GPU")
    parser.add_argument("--sample-interval", type=int, default=100, help="抽样步数间隔，<=0 关闭")
    parser.add_argument("--sample-prompts", type=Path, default=Path("data/eval/thinking_prompts.txt"))
    parser.add_argument("--sample-out", type=Path, default=Path("artifacts/sft_samples"))
    parser.add_argument("--sample-limit", type=int, default=25, help="每次抽样用前 N 条提示词")
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
    epochs = args.epochs if args.epochs is not None else cfg["epochs"]
    warmup_steps = int(steps_per_epoch * epochs * cfg["warmup_ratio"])
    train_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=batch,
        gradient_accumulation_steps=accum,
        num_train_epochs=epochs,
        learning_rate=cfg["lr"],
        weight_decay=cfg["wd"],
        warmup_steps=warmup_steps,
        bf16=not args.no_bf16,
        no_cuda=args.no_cuda,
        logging_steps=cfg.get("log_interval", 10),
        save_steps=cfg.get("save_interval", 1000),
        save_total_limit=3,
        seed=cfg.get("seed", 42),
        neftune_noise_alpha=cfg.get("neftune_alpha"),
        report_to=[],
        ddp_find_unused_parameters=True,
        gradient_checkpointing=True,
    )
    if args.max_steps:
        train_args.max_steps = args.max_steps
        train_args.num_train_epochs = 1.0
    prompts = None
    if args.sample_interval > 0:
        prompts = [
            line.strip()
            for line in args.sample_prompts.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=dataset,
        data_collator=collate,
        callbacks=[
            SampleCallback(
                model,
                model_cfg,
                LocalSightTokenizer(cfg["tokenizer_dir"]),
                prompts or [],
                args.sample_out,
                args.sample_interval,
                limit=args.sample_limit,
            )
        ],
    )
    trainer.train()
    model.save_pretrained(Path(args.output_dir) / "final")
    (Path(args.output_dir) / "final" / "manifest.json").write_text(
        json.dumps(dataset.manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
