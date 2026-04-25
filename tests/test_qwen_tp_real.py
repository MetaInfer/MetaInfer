from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch
import torch.distributed as dist

from llm_engine import LLMEngine
from engine.models.qwen import can_load_qwen_weights
from engine.tp_layers.distributed import get_tp_rank, get_tp_size

QWEN_06B = Path("/data/xinference/cache/Qwen3-0.6B")
QWEN_8B = Path("/data/xinference/cache/Qwen3-8B")
PROMPTS_ZH = [
    "用一两句话介绍苏州园林的特点。",
    "什么是张量并行？用中文简要说明。",
    "写一句关于夏天傍晚的描写。",
    "什么是大语言模型？",
    "怎么做面包？",
]


@pytest.fixture(scope="module", autouse=True)
def _cleanup_distributed_group():
    yield
    if dist.is_available() and dist.is_initialized():
        try:
            dist.barrier()
        finally:
            dist.destroy_process_group()


def _skip_reason(model_dir: Path) -> str | None:
    if int(os.environ.get("WORLD_SIZE", "1")) != 4:
        return "该测试需 torchrun --nproc_per_node=4"
    if not torch.cuda.is_available():
        return "需要 CUDA"
    ok, reason = can_load_qwen_weights(model_dir)
    if not ok:
        return reason
    return None


# @pytest.mark.parametrize("model_dir", [QWEN_06B, QWEN_8B])
@pytest.mark.parametrize("model_dir", [QWEN_8B])
def test_qwen_tp_forward_and_generate(model_dir: Path) -> None:
    reason = _skip_reason(model_dir)
    if reason:
        pytest.skip(reason)
    os.environ["META_INFER_LOG_RANK0_ONLY"] = "1"
    eng = LLMEngine(model_dir=model_dir, inference_backend="qwen_tp", max_num_seqs=8, max_num_batched_tokens=4096)
    outs = eng.generate(PROMPTS_ZH, max_new_tokens=24, temperature=0.0, top_p=None)
    assert isinstance(outs, list) and len(outs) == len(PROMPTS_ZH)
    assert all(isinstance(t, str) for t in outs)
    if get_tp_rank() == 0:
        print(f"[qwen_tp] world={get_tp_size()} model={model_dir}")
        for i, text in enumerate(outs):
            print(f"[qwen_tp] output[{i}]={text[:200]!r}")
