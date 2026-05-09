"""
张量并行 TDD：``FakeCausalLMTP`` 与 ``reference_forward_toy`` 的 logits 应对齐。

- 单进程：不依赖 init_process_group 的对照测试
- 多进程：``torchrun --nproc_per_node=2`` 等执行本文件（或 ``-m pytest`` 在 skip 的辅助下只跑单进程用例）
"""

from __future__ import annotations

import os

import pytest
import torch
import torch.distributed as dist

from engine.model_runner import FakeCausalLMTP, ModelRunner, reference_forward_toy
from engine.tp_distributed import get_tp_rank

# 3*H、4*H、V 均需被 TP 整除，故 H=16, V=128, tp=2 可用
H = 16
V = 128
SEED = 7


@pytest.mark.skipif(
    int(os.environ.get("WORLD_SIZE", 1)) != 1,
    reason="单进程用例，避免在 torchrun 多进程下重复与误用",
)
def test_toy_full_tp_rank1_matches_reference():
    """WORLD_SIZE=1 时 FakeCausalLMTP 退化为全权重，与 reference 一致（无 dist）。"""
    x = torch.tensor([1, 2, 3, 4], dtype=torch.long)
    ref = reference_forward_toy(
        x, vocab_size=V, hidden_size=H, ffn_mult=4, seed=SEED
    )
    m = FakeCausalLMTP(V, H, ffn_mult=4, seed=SEED)
    y = m(x)
    assert y.shape == (4, V)
    assert torch.allclose(ref, y, rtol=1e-4, atol=1e-5)


def _assert_tp_matches_ref(device: str) -> None:
    try:
        if device == "cuda":
            torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        x = torch.tensor([0, 5, 9, 12], dtype=torch.long, device=device)
        ref = reference_forward_toy(
            x.cpu(), vocab_size=V, hidden_size=H, ffn_mult=4, seed=SEED
        )
        r = ModelRunner(
            V,
            H,
            device=device,
            dtype=torch.float32,
            seed=SEED,
            use_tp=True,
        )
        y = r.model(x)
        assert y.shape == (4, V)
        assert torch.allclose(ref, y.cpu(), rtol=1e-4, atol=1e-5)
        if get_tp_rank() == 0:
            print("[test_tp] distributed logits allclose OK (rank0 check)")
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


@pytest.mark.skipif(
    int(os.environ.get("WORLD_SIZE", 1)) < 2,
    reason="需 torchrun 多进程，例如: torchrun --nproc_per_node=2 tests/test_tp.py",
)
def test_model_runner_tensor_parallel_matches_reference():
    """多进程下 ModelRunner+FakeCausalLMTP 与整权重前向一致。"""
    if (
        torch.cuda.is_available()
        and "LOCAL_RANK" in os.environ
        and os.environ.get("META_INFER_TP_BACKEND", "").lower() != "gloo"
    ):
        _assert_tp_matches_ref("cuda")
    else:
        _assert_tp_matches_ref("cpu")


def _main() -> None:
    ws = int(os.environ.get("WORLD_SIZE", 1))
    if ws > 1:
        if (
            torch.cuda.is_available()
            and "LOCAL_RANK" in os.environ
            and os.environ.get("META_INFER_TP_BACKEND", "").lower() != "gloo"
        ):
            _assert_tp_matches_ref("cuda")
        else:
            _assert_tp_matches_ref("cpu")
    else:
        test_toy_full_tp_rank1_matches_reference()
        print("[test_tp] single-process test ok")


if __name__ == "__main__":
    _main()
