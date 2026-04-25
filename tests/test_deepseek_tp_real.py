from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest
import torch
import torch.distributed as dist

from llm_engine import LLMEngine
from engine.models.deepseek_v2 import can_load_deepseek_weights
from engine.tp_layers.distributed import get_tp_rank, get_tp_size

# 项目根 meta-infer；完整 torchrun+pytest 输出请: bash run_test_deepseek_tp.sh
# 或: torchrun ... 2>&1 | tee "$PWD/torchrun_test_deepseek_tp.log"
_META_INFER_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = Path("/data/xinference/cache/deepseek-v2-chat-pytorch-16b")
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


def _skip_reason() -> str | None:
    if int(os.environ.get("WORLD_SIZE", "1")) != 4:
        return "该测试需 torchrun --nproc_per_node=4"
    if not torch.cuda.is_available():
        return "需要 CUDA"
    ok, reason = can_load_deepseek_weights(MODEL_DIR)
    if not ok:
        return reason
    return None


def test_deepseek_tp_engine_lazystep_and_outputs() -> None:
    reason = _skip_reason()
    if reason:
        pytest.skip(reason)
    os.environ["META_INFER_LOG_RANK0_ONLY"] = "1"
    # rank0 将本轮摘要追加到项目根（便于在容器/远程查看；非 /tmp）
    def _log(msg: str) -> None:
        if get_tp_rank() != 0:
            return
        line = f"{msg}\n"
        try:
            with open(_META_INFER_ROOT / "test_deepseek_tp_last_run.log", "a", encoding="utf-8") as f:
                f.write(line)
        except OSError as e:
            print(f"[deepseek_tp] could not write project log: {e}", file=sys.stderr)

    _log("--- new run ---")
    outs: list[str] = []
    try:
        eng = LLMEngine(
            model_dir=MODEL_DIR,
            inference_backend="tp",
            max_num_seqs=8,
            max_num_batched_tokens=4096,
        )
        eng.begin_generation(
            PROMPTS_ZH, max_new_tokens=24, temperature=0.0, top_p=None
        )
        while eng.has_unfinished_requests():
            eng.step(0.0, None)
        outs = eng.get_generation_outputs()
    except Exception as e:
        _log(f"ERROR: {type(e).__name__}: {e}")
        raise
    assert isinstance(outs, list) and len(outs) == len(PROMPTS_ZH)
    assert all(isinstance(t, str) for t in outs)
    if get_tp_rank() == 0:
        print(f"[deepseek_tp] world={get_tp_size()} model={MODEL_DIR}")
        for i, text in enumerate(outs):
            print(f"[deepseek_tp] output[{i}]={text[:200]!r}")
            _log(f"output[{i}]={text[:200]!r}")
