#!/usr/bin/env python3
"""Probe a Python environment for LocalSight dependencies and GPU info.

Run on the training server with the target interpreter, e.g.:
    ~/miniconda3/envs/kdl/bin/python /tmp/probe_env.py
"""
from __future__ import annotations

import importlib.util
import platform
import subprocess
import sys

print("python:", sys.version)
print("platform:", platform.platform())
print("machine:", platform.machine())

MODULES = [
    "torch",
    "torchvision",
    "transformers",
    "tokenizers",
    "datasets",
    "accelerate",
    "trl",
    "peft",
    "flash_attn",
    "vllm",
    "wandb",
    "triton",
    "einops",
    "numpy",
    "safetensors",
    "huggingface_hub",
    "omegaconf",
    "yaml",
    "pyarrow",
    "requests",
]

print("--- module availability ---")
for name in MODULES:
    try:
        spec = importlib.util.find_spec(name)
    except Exception as exc:  # noqa: BLE001
        print(f"{name:18s} ERROR {exc}")
        continue
    if spec is not None:
        try:
            mod = importlib.import_module(name)
            version = getattr(mod, "__version__", "?")
        except Exception as exc:  # noqa: BLE001
            version = f"import error: {exc}"
        print(f"{name:18s} YES  {version}")
    else:
        print(f"{name:18s} no")

print("--- torch / cuda ---")
try:
    import torch

    print("torch_version:", torch.__version__)
    print("torch_cuda_version:", torch.version.cuda)
    print("cuda_available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("gpu_count:", torch.cuda.device_count())
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(
                f"  gpu{i}:",
                props.name,
                f"{props.total_memory / 1024**3:.1f}GiB",
                f"sm={props.major}.{props.minor}",
            )
except Exception as exc:  # noqa: BLE001
    print("torch import error:", exc)

print("--- nvidia-smi ---")
subprocess.run(
    [
        "nvidia-smi",
        "--query-gpu=index,name,memory.total,memory.used,driver_version",
        "--format=csv,noheader",
    ]
)
