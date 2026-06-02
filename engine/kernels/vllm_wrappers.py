# engine/kernels/vllm_wrappers.py
# Phase 1: 数值基元 — 7 个 vLLM / flash_attn 标品 kernel 薄封装。
#
# 标品黑盒原则：
#   1. 从 vLLM 源码提取纯净 Python 调用接口
#   2. 对齐输入/输出 Tensor Shape 与 Dtype
#   3. 严禁修改 vLLM kernel 内部逻辑
#
# Ref:
#   kernel_replacement_plan.md §九（完整 kernel 调用契约表 + Snippet A-F）
#   vllm/_custom_ops.py:420-423 (rms_norm, fused_add_rms_norm)
#   vllm/_custom_ops.py:400-410 (rotary_embedding)
#   vllm/model_executor/layers/activation.py::SiluAndMul.forward_cuda

import torch
import torch.nn as nn

# === vLLM _custom_ops kernel imports ===
from vllm._custom_ops import rms_norm as _vllm_rms_norm
from vllm._custom_ops import fused_add_rms_norm as _vllm_fused_add_rms_norm
from vllm._custom_ops import rotary_embedding as _vllm_rotary_embedding

# === vllm._C 触发 torch.ops._C.silu_and_mul 注册 ===
import vllm._C  # noqa: F401  # 触发 torch.ops._C 算子注册

# === flash_attn 直接 re-export（nocompile 无需 custom_op 注册） ===
from flash_attn import flash_attn_varlen_func  # noqa: F401
from flash_attn.flash_attn_interface import flash_attn_with_kvcache  # noqa: F401

__all__ = [
    "rms_norm",
    "fused_add_rms_norm",
    "silu_and_mul",
    "rotary_embedding",
    "_get_cos_sin_cache",
    "make_cos_sin_cache",
    "flash_attn_varlen_func",
    "flash_attn_with_kvcache",
]


# ================================================================
# KERNEL 1: rms_norm
# ================================================================

def rms_norm(
    out: torch.Tensor,
    input: torch.Tensor,
    weight: torch.Tensor,
    epsilon: float,
) -> None:
    """
    标品黑盒 — vLLM rms_norm CUDA kernel.

    数据契约:
        out:     [*, H]  bf16/fp16/fp32, contiguous, 预分配 (empty_like(input))
        input:   [*, H]  bf16/fp16/fp32, contiguous
        weight:  [H]     bf16/fp16/fp32, contiguous
        epsilon: float (典型值 1e-6)

    操作: out = rms_norm(input) * weight  (内部升 fp32 计算)

    Ref: vllm/_custom_ops.py:420-423, kernel rms_norm_kernel<c10::BFloat16, 8, 3>
         kernel_replacement_plan.md §九 Snippet A
    """
    _vllm_rms_norm(out, input, weight, epsilon)


# ================================================================
# KERNEL 2: fused_add_rms_norm
# ================================================================

def fused_add_rms_norm(
    input: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    epsilon: float,
) -> None:
    """
    标品黑盒 — vLLM fused_add_rms_norm CUDA kernel.

    数据契约:
        input:    [*, H]  bf16, contiguous — 子层输出 (如 attention output)，原地被修改为归一化结果
        residual: [*, H]  bf16, contiguous — 残差状态，原地被修改为 residual + input
        weight:   [H]     bf16, contiguous — RMSNorm weight
        epsilon:  float

    两步 in-place 操作:
        1. residual = residual + input        (残差融合)
        2. input    = rms_norm(residual) * weight  (归一化，供下一子层使用)

    典型调用 (Qwen3 DecoderLayer 模式):
        # post-attention:
        fused_add_rms_norm(attn_output, residual, self.post_attention_layernorm.weight, eps)

        # post-mlp:
        fused_add_rms_norm(mlp_output, residual, self.input_layernorm.weight, eps)

    所有调用均使用本层的 self.xxx.weight。

    Ref: vllm/_custom_ops.py:420-423, kernel fused_add_rms_norm_kernel<c10::BFloat16, 8>
         kernel_replacement_plan.md §九 Snippet B
    """
    _vllm_fused_add_rms_norm(input, residual, weight, epsilon)


# ================================================================
# KERNEL 3: silu_and_mul
# ================================================================

def silu_and_mul(
    out: torch.Tensor,
    input: torch.Tensor,
) -> None:
    """
    标品黑盒 — vLLM silu_and_mul CUDA kernel.

    数据契约:
        input: [*, 2*d]  bf16, contiguous — MergedColumnParallelLinear 输出，前半 gate 后半 up
        out:   [*, d]    bf16, contiguous, 预分配 empty()

    操作: out = silu(input[..., :d]) * input[..., d:]
    其中 d = input.shape[-1] // 2

    前置要求: import vllm._C 已触发 torch.ops._C.silu_and_mul 注册（本文件开头已执行）。

    Ref: vllm/model_executor/layers/activation.py::SiluAndMul.forward_cuda
         kernel_replacement_plan.md §九 Snippet C
    """
    torch.ops._C.silu_and_mul(out, input)


# ================================================================
# KERNEL 4: rotary_embedding
# ================================================================

def rotary_embedding(
    positions: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor | None,
    head_size: int,
    cos_sin_cache: torch.Tensor,
    is_neox: bool,
) -> None:
    """
    标品黑盒 — vLLM rotary_embedding CUDA kernel.

    数据契约:
        positions:      [num_tokens]        int64 (torch.long), 1D
        query:          [num_tokens, N, D]  bf16, contiguous, in-place 修改
        key:            [num_tokens, Nkv, D] bf16, contiguous, in-place 修改 (可为 None)
        head_size:      int                 每头维度 (Qwen3=128)
        cos_sin_cache:  [max_pos, head_size] 预计算缓存 (前 head_size//2 cos, 后 head_size//2 sin)
        is_neox:        bool                Qwen3 严格 True (GPT-NeoX 风格)

    注意:
        - q/k 必须是 3D [tokens, heads, dim]，非 4D [B, S, H, D]。
          调用前先从 4D reshape/flatten 为 3D，调用后再 view 回 4D。
        - cos_sin_cache 格式为 [max_position, head_size] (非 [max_pos, 2*head_size])
        - vLLM kernel 内部自行处理 NeoX cos/sin 重复

    Ref: vllm/_custom_ops.py:400-410, kernel rotary_embedding_kernel<c10::BFloat16, true>
         kernel_replacement_plan.md §九 Snippet D
    """
    _vllm_rotary_embedding(positions, query, key, head_size, cos_sin_cache, is_neox)


# ================================================================
# KERNEL 5: cos_sin_cache 工厂 + 模块级 registry
# ================================================================

# 模块级 registry — 所有 DecoderLayer 共享同一 cache tensor，避免 36×8MB=288MB 显存浪费
_cos_sin_cache_registry: dict[tuple, torch.Tensor] = {}


def _get_cos_sin_cache(
    max_pos: int,
    head_dim: int,
    rope_theta: float = 1000000.0,
) -> torch.Tensor:
    """
    模块级共享 cos_sin_cache registry。

    使用 registry key=(max_pos, head_dim, rope_theta) 确保同参数模型共享 cache。
    返回的 tensor 在 CPU 上创建，调用方负责首次 forward 时 .to(device) (lazy GPU transfer)。

    Args:
        max_pos:    最大位置 (Qwen3-8B: 40960, from config.max_position_embeddings)
        head_dim:   头维度 (Qwen3-8B: 128)
        rope_theta: RoPE base (Qwen3-8B: 1000000.0)

    Returns:
        cos_sin_cache [max_pos, head_dim] bf16 on CPU
    """
    key = (max_pos, head_dim, rope_theta)
    if key not in _cos_sin_cache_registry:
        _cos_sin_cache_registry[key] = make_cos_sin_cache(max_pos, head_dim, rope_theta)
    return _cos_sin_cache_registry[key]


def make_cos_sin_cache(
    max_position: int,
    head_size: int,
    rope_theta: float = 1000000.0,
    dtype: torch.dtype = torch.bfloat16,
    device: torch.device | None = None,
) -> torch.Tensor:
    """
    构造 vLLM rotary_embedding 所需的 cos_sin_cache。

    格式: [max_position, head_size]  (NOT 2*head_size!)
      cache[pos, :head_size//2] = cos 值
      cache[pos, head_size//2:] = sin 值
    vLLM kernel 内部自行处理 NeoX 风格的 cos/sin 重复。

    Qwen3-8B 参数:
        max_position = 40960
        head_size = 128
        rope_theta = 1000000.0

    已通过数值验证: 与 vLLM RotaryEmbeddingBase._compute_cos_sin_cache 逻辑一致。

    Ref: vllm/model_executor/layers/rotary_embedding/base.py:76-84
         kernel_replacement_plan.md §九 Snippet E
    """
    inv_freq = 1.0 / (rope_theta ** (
        torch.arange(0, head_size, 2, dtype=torch.float32, device=device) / head_size
    ))
    t = torch.arange(max_position, dtype=torch.float32, device=device)
    freqs = torch.einsum("i,j -> ij", t, inv_freq)   # [max_pos, head_size//2]
    cos = freqs.cos().to(dtype=dtype)                 # [max_pos, head_size//2]
    sin = freqs.sin().to(dtype=dtype)                 # [max_pos, head_size//2]
    return torch.cat((cos, sin), dim=-1)               # [max_pos, head_size]
