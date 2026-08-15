#!/usr/bin/env python3
"""过夜自动衔接：pretrain 收尾评测 → SFT round1 → 质量裁判 →（未达标）sft512 再训 1 epoch → 停止。

用户指令：SFT 后暂停等下一步指令；SFT 质量由 7B judge + 指标 + 快评共同裁决，
未达标时自动用 /media/liuzh/data/DLData/minimind/sft_512.jsonl 再训一轮。
"""
from __future__ import annotations

import datetime
import json
import os
import subprocess
import time
from pathlib import Path

B = Path("/home/sodastar/project/LocalSight")
LOG = B / "artifacts" / "sft_chain.log"
DONE = B / "artifacts" / "SFT_CHAIN_DONE"
ENV = {
    **os.environ,
    "PYTHONPATH": f"{B}/src",
    "NCCL_P2P_DISABLE": "1",
    "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
}

SFT512_SRC = "/media/liuzh/data/DLData/minimind/sft_512.jsonl"


def log(msg: str) -> None:
    line = f"[{datetime.datetime.now():%H:%M:%S}] {msg}"
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")
    print(line, flush=True)


def run(cmd: list[str], log_path: Path) -> int:
    log(f"run: {' '.join(cmd[:6])}... -> {log_path.name}")
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(f"\n===== {datetime.datetime.now():%F %T} =====\n")
        f.flush()
        proc = subprocess.run(cmd, cwd=B, env=ENV, stdout=f, stderr=subprocess.STDOUT)
    log(f"run 结束 rc={proc.returncode}")
    return proc.returncode


def wait_final_eval() -> None:
    marker = B / "artifacts" / "pretrain_final_evals.log"
    deadline = datetime.datetime.now() + datetime.timedelta(hours=6)
    while datetime.datetime.now() < deadline:
        if marker.exists() and "FINAL_EVAL_DONE" in marker.read_text(encoding="utf-8", errors="replace"):
            log("pretrain 收尾评测已完成")
            return
        time.sleep(120)
    log("等待收尾评测超时，自行执行")
    run(["bash", str(B / "scripts/pretrain_final_eval.sh")], B / "artifacts" / "pretrain_final_evals.log")


def run_sft(data_dir: str, output_dir: str, start_ckpt: str, epochs: float, log_path: Path, sample_out: str) -> int:
    return run(
        [
            str(B / ".venv/bin/torchrun"), "--nproc_per_node=2",
            str(B / "src/localsight/training/sft.py"),
            "--config", str(B / "configs/sft.yaml"),
            "--data-dir", str(B / data_dir),
            "--output-dir", str(B / output_dir),
            "--start-checkpoint", str(B / start_ckpt),
            "--epochs", str(epochs),
            "--sample-interval", "100",
            "--sample-prompts", str(B / "data/eval/thinking_prompts.txt"),
            "--sample-out", str(B / sample_out),
            "--sample-limit", "25",
        ],
        log_path,
    )


def quality(checkpoint: str, out_dir: str) -> dict:
    proc = subprocess.run(
        [
            str(B / ".venv/bin/python"),
            str(B / "scripts/sft_quality_check.py"),
            "--checkpoint", str(B / checkpoint),
            "--judge-model", str(B / "models/judge/Qwen--Qwen2.5-7B-Instruct"),
            "--out-dir", str(B / out_dir),
        ],
        cwd=B, env=ENV, capture_output=True, text=True,
    )
    if proc.stdout.strip():
        log("quality check 输出尾部: " + proc.stdout.strip().splitlines()[-1])
    else:
        log("quality check 无输出（可能失败），stderr: " + proc.stderr.strip()[-500:])
    gate_path = B / out_dir / "gate.json"
    if gate_path.exists():
        return json.loads(gate_path.read_text(encoding="utf-8"))
    return {"pass": False, "error": "gate.json 缺失（裁判失败）", "judge_mean": 0.0}


def main() -> None:
    log("SFT 衔接链启动，等待 pretrain 收尾评测")
    wait_final_eval()

    log("开始 SFT round 1（sft_t2t_mini，2 epochs，起点 pretrain/soup）")
    run_sft(
        "data/processed/sft", "artifacts/sft_v2", "artifacts/pretrain/soup", 2,
        B / "artifacts" / "sft_v2.log", "artifacts/sft_samples_round1",
    )
    log("SFT round 1 完成，开始质量裁判")
    gate1 = quality("artifacts/sft_v2/final", "artifacts/sft_v2/quality")
    log(f"round1 gate: {json.dumps(gate1, ensure_ascii=False)}")

    rounds = ["round1"]
    if gate1.get("error"):
        log("质量裁判失败，暂停等待人工介入（不自动进入 round2）")
        DONE.write_text(
            json.dumps(
                {
                    "time": datetime.datetime.now().isoformat(timespec="seconds"),
                    "rounds": rounds,
                    "gate1": gate1,
                    "error": "quality check failed",
                    "waiting_for_user": True,
                },
                ensure_ascii=False, indent=2,
            ),
            encoding="utf-8",
        )
        return
    if gate1.get("pass"):
        log("round1 达标，按用户指令停止")
    else:
        log(f"round1 未达标，构建 sft512 数据（{SFT512_SRC}）")
        run(
            [
                str(B / ".venv/bin/python"),
                str(B / "scripts/build_chat_data.py"),
                "--src", SFT512_SRC,
                "--out", str(B / "data/processed/sft512"),
                "--tokenizer", str(B / "data/tokenizer"),
            ],
            B / "artifacts" / "build_sft512.log",
        )
        log("开始 SFT round 2（sft512，1 epoch）")
        run_sft(
            "data/processed/sft512", "artifacts/sft_v2_512", "artifacts/sft_v2/final", 1,
            B / "artifacts" / "sft512.log", "artifacts/sft_samples_round2",
        )
        gate2 = quality("artifacts/sft_v2_512/final", "artifacts/sft_v2_512/quality")
        log(f"round2 gate: {json.dumps(gate2, ensure_ascii=False)}")
        rounds.append("round2")

    DONE.write_text(
        json.dumps(
            {
                "time": datetime.datetime.now().isoformat(timespec="seconds"),
                "rounds": rounds,
                "gate1": gate1,
                "waiting_for_user": True,
            },
            ensure_ascii=False, indent=2,
        ),
        encoding="utf-8",
    )
    log("SFT 阶段结束，等待用户指令（详见 SFT_CHAIN_DONE 与各 quality/gate.json）")


if __name__ == "__main__":
    main()
