"""YAML 配置加载：extends 合并 + 模型配置解析。"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from localsight.model import LocalsightConfig


def load_yaml(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"配置根节点必须是映射: {path}")
    return data


def _merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _merge(out[key], value)
        else:
            out[key] = value
    return out


def resolve_stage_config(stage_path: Path) -> tuple[dict[str, Any], LocalsightConfig]:
    """加载阶段配置（处理 extends），并解析其引用的模型配置。"""
    stage_path = Path(stage_path)
    stage = load_yaml(stage_path)
    if "extends" in stage:
        base_path = (stage_path.parent / stage.pop("extends")).resolve()
        base = load_yaml(base_path)
    else:
        base = {}
    merged = _merge(base, stage)

    model_cfg_path = Path(merged.get("model_config", "configs/model/minimind3_moe_198m.yaml"))
    if not model_cfg_path.is_absolute():
        model_cfg_path = (stage_path.parent.parent / model_cfg_path).resolve()
    model_config = LocalsightConfig.from_dict(load_yaml(model_cfg_path))
    return merged, model_config
