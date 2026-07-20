"""Test helpers for port-model."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional


def make_requirements(
    task_id: str = "pm-1",
    *,
    form: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    base_form = {
        "model_dir": "/tmp/fake/model",
        "source_framework_dir": "/tmp/fake/source_fw",
        "target_framework_dir": "/tmp/fake/target_fw",
        "target_framework_type": "vLLM",
        "target_hardware": "NVIDIA H100",
        "test_prompts": "",
    }
    if form:
        base_form.update(form)
    return {
        "task_id": task_id,
        "task_type": "port-model",
        "created_at": 0.0,
        "form": base_form,
    }


def make_minimal_config() -> Dict[str, Any]:
    """A minimal HuggingFace-style config.json fixture."""
    return {
        "architectures": ["TestModelForCausalLM"],
        "model_type": "test_model",
        "hidden_size": 4096,
        "intermediate_size": 11008,
        "num_hidden_layers": 32,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "hidden_act": "silu",
        "rms_norm_eps": 1e-6,
        "rope_theta": 10000.0,
        "max_position_embeddings": 8192,
        "vocab_size": 32000,
        "torch_dtype": "float16",
        "tie_word_embeddings": False,
    }
