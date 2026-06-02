# Weight loading for safetensors models — zero mlx_lm dependency.
"""Load model weights from safetensors files.

Reads config.json for architecture, then loads all safetensors
weight files and maps them into the MLX model via model.update().
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.nn as nn

from .kv_cache import make_kv_cache
from .model import Qwen3Config, Qwen3ForCausalLM


def _load_safetensors(model_dir: str) -> dict[str, mx.array]:
    """Load all weights from safetensors files, converting to MLX arrays."""
    import safetensors
    import torch

    weights: dict[str, mx.array] = {}
    model_path = Path(model_dir)

    st_files = sorted(model_path.glob("*.safetensors"))
    if not st_files:
        st_files = sorted(model_path.glob("model*.safetensors"))

    for st_file in st_files:
        with safetensors.safe_open(str(st_file), framework="pt") as f:
            for key in f.keys():
                t = f.get_tensor(key)
                if t.dtype == torch.bfloat16:
                    t = t.to(torch.float32)
                weights[key] = mx.array(t.numpy())

    return weights


def _build_update_tree(weights: dict[str, mx.array]) -> dict:
    """Build a nested dict tree suitable for model.update().

    Converts string-indexed dicts (for layers) into lists.
    HF name "model.layers.0.self_attn.q_proj.weight"
    → tree["model"]["layers"][0]["self_attn"]["q_proj"]["weight"]
    """
    tree: dict[str, Any] = {}
    for key, array in weights.items():
        parts = key.split(".")
        node = tree
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = array

    def _dicts_to_lists(d: Any) -> Any:
        """Recursively convert integer-keyed dicts to lists."""
        if isinstance(d, dict):
            # Check if all keys are integer strings
            if d and all(k.isdigit() for k in d):
                max_idx = max(int(k) for k in d)
                lst = [{}] * (max_idx + 1)
                for k, v in d.items():
                    lst[int(k)] = _dicts_to_lists(v)
                return lst
            return {k: _dicts_to_lists(v) for k, v in d.items()}
        return d

    return _dicts_to_lists(tree)


def load_qwen3_model(model_dir: str) -> tuple[Qwen3ForCausalLM, Any]:
    """Load Qwen3 model from directory with config.json + safetensors.

    Returns (model, config).
    """
    model_path = Path(model_dir)
    config_file = model_path / "config.json"

    if not config_file.exists():
        msg = f"config.json not found in {model_dir}"
        raise FileNotFoundError(msg)

    with open(config_file, encoding="utf-8") as f:
        config_dict = json.load(f)

    config = Qwen3Config.from_dict(config_dict)
    model = Qwen3ForCausalLM(config)
    weights = _load_safetensors(str(model_path))
    tree = _build_update_tree(weights)

    model.update(tree)
    mx.eval(model.parameters())

    print(f"  Loaded {len(weights)} parameters")
    return model, config
