"""Stage 0 资产: 从 vLLM 源码提取的纯净 Kernel Wrapper.

铁律: 所有函数均为不可修改的标品黑盒, 仅负责对齐输入/输出格式。
"""
import torch
import vllm._C  # trigger torch.ops._C registration
from vllm._custom_ops import fused_add_rms_norm as _vllm_fused_add_rms_norm
from vllm._custom_ops import rms_norm as _vllm_rms_norm


# === Snippet A: rms_norm ===
def rms_norm(
    out: torch.Tensor,
    input: torch.Tensor,
    weight: torch.Tensor,
    epsilon: float,
) -> None:
    """vLLM rms_norm CUDA kernel.

    Contract:
        out:     [*, H]  same dtype as input, contiguous, pre-allocated
        input:   [*, H]  bf16/fp16/fp32, contiguous
        weight:  [H]     same dtype as input, contiguous
        epsilon: float
    Operation: out = rms_norm(input) * weight  (internal fp32 compute)
    """
    _vllm_rms_norm(out, input, weight, epsilon)


# === Snippet B: fused_add_rms_norm ===
def fused_add_rms_norm(
    input: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    epsilon: float,
) -> None:
    """vLLM fused_add_rms_norm CUDA kernel.

    Contract:
        input:    [*, H]  bf16, contiguous — sublayer output
        residual: [*, H]  bf16, contiguous — running hidden state
        weight:   [H]     bf16, contiguous — RMSNorm weight
        epsilon:  float
    Two in-place operations:
        1. residual = residual + input
        2. input = rms_norm(residual) * weight
    """
    _vllm_fused_add_rms_norm(input, residual, weight, epsilon)


# === Snippet C: silu_and_mul ===
def silu_and_mul(out: torch.Tensor, input: torch.Tensor) -> None:
    """vLLM silu_and_mul CUDA kernel.

    Contract:
        input: [*, 2*d]  bf16, contiguous — merged gate+up projection output
        out:   [*, d]    bf16, contiguous, pre-allocated
    Operation: out = silu(input[..., :d]) * input[..., d:]
        where d = input.shape[-1] // 2
    """
    torch.ops._C.silu_and_mul(out, input)
