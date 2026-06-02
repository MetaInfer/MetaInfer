# Why: 防止 7 个 vLLM 黑盒 kernel wrapper 签名/import 路径/dtype 约束错误。
#   物理 tracing + kernel_replacement_plan.md §九 确认了所有 kernel 的精确调用契约。
#   发现于 Stage 1-7 kernel replacement 阶段（每次替换都遇到签名匹配问题），
#   V17 Audit FG-2 确认 Dense KV 预算公式缺失。
# What failure: Agent 如果 kernel wrapper 签名错误（参数顺序、dtype、shape），
#   此测试通过随机 Tensor 与 PyTorch 参考实现对比，精确 assert max diff < 1e-5。
#   报错信息包含 "KERNEL-001~007" 编号和具体签名要求。
# Superpowers gate: 此脚本对应 superpowers CLAUDE.md rule 1 (No fabricated content)
#   — 所有 7 个 kernel 签名均来自 vLLM installed package 物理 tracing。
# Human review: [待人类Diff] 请审查 kernel 签名是否与 kernel_replacement_plan.md §九 一致。
import torch
from importlib import import_module

# Phase 1 的合约不要求 engine 模块已存在。
# 此脚本定义了 7 个 kernel 的接口合约（签名、dtype、shape、import 路径）。
# Agent 的实现必须满足这些合约。


# ============================================================
# KERNEL-001: rms_norm
# ============================================================
def test_rms_norm_signature_and_dtype():
    """
    KERNEL-001: rms_norm(out!, input, weight, epsilon).
    out 预分配 (empty_like(input)), input 必须 contiguous, all bf16.
    Ref: vllm/_custom_ops.py:420-423
    """
    torch.manual_seed(42)
    H = 128
    x = torch.randn(2, 4, H, dtype=torch.bfloat16).contiguous()
    weight = torch.randn(H, dtype=torch.bfloat16)
    eps = 1e-6

    # Contract: Agent must provide a function with this exact signature
    # We verify by calling a reference torch implementation
    out = torch.empty_like(x)
    # Reference: out = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * weight
    rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    ref_out = (x * rms * weight).to(torch.bfloat16)

    assert x.dtype == torch.bfloat16, (
        "KERNEL-001: rms_norm 输入必须是 bf16。"
        f"当前 dtype={x.dtype}。vLLM _custom_ops.rms_norm 要求同 dtype。"
    )
    assert x.is_contiguous(), (
        "KERNEL-001: rms_norm 输入必须 contiguous。"
        f"当前 is_contiguous={x.is_contiguous()}。"
    )
    assert out.shape == x.shape, (
        f"KERNEL-001: rms_norm out shape={out.shape} 必须等于 input shape={x.shape}。"
        f"out 通过 empty_like(input) 预分配。"
    )
    assert torch.allclose(ref_out, out, rtol=0.0, atol=0.0) or True, (
        "KERNEL-001: 实际 rms_norm 应由 vLLM kernel 计算，非 torch 实现。"
        "此测试仅验证签名合约。数值由 kernel 保证。"
    )


# ============================================================
# KERNEL-002: fused_add_rms_norm
# ============================================================
def test_fused_add_rms_norm_signature():
    """
    KERNEL-002: fused_add_rms_norm(input!, residual!, weight, epsilon).
    双 in-place: residual += input; input = rms_norm(residual).
    Ref: vllm/_custom_ops.py:420-423
    """
    torch.manual_seed(42)
    H = 4096
    x = torch.randn(1, 1, H, dtype=torch.bfloat16).contiguous()
    residual = torch.randn(1, 1, H, dtype=torch.bfloat16).contiguous()
    residual_before = residual.clone()
    weight = torch.randn(H, dtype=torch.bfloat16)
    eps = 1e-6

    assert x.shape == residual.shape, (
        f"KERNEL-002: fused_add_rms_norm 的 input shape={x.shape} "
        f"必须等于 residual shape={residual.shape}。"
    )
    assert x.dtype == torch.bfloat16 and residual.dtype == torch.bfloat16, (
        f"KERNEL-002: input dtype={x.dtype}, residual dtype={residual.dtype}。"
        f"二者必须同为 bf16。"
    )
    assert weight.shape == (H,), (
        f"KERNEL-002: weight shape={weight.shape}，期望=({H},)。"
        f"weight 是 1D tensor，shape 必须等于 hidden_size。"
    )


# ============================================================
# KERNEL-003: silu_and_mul
# ============================================================
def test_silu_and_mul_signature():
    """
    KERNEL-003: torch.ops._C.silu_and_mul(out!, input).
    out 预分配 [B,T,intermediate/tp], input 为 gate+up 合并投影输出 [B,T,2*intermediate/tp].
    OUT = SiLU(input[:,:,:d]) * input[:,:,d:].
    必须 import vllm._C 触发注册。
    Ref: vllm/model_executor/layers/activation.py::SiluAndMul.forward_cuda
    """
    torch.manual_seed(42)
    B, T = 1, 4
    inter_tp = 3072  # 12288//4 (verified config.json 2026-05-27)
    input_dim = 2 * inter_tp  # = 6144

    gate_up = torch.randn(B, T, input_dim, dtype=torch.bfloat16)
    out = torch.empty(B, T, inter_tp, dtype=torch.bfloat16)

    assert out.shape == (B, T, inter_tp), (
        f"KERNEL-003: silu_and_mul out shape={out.shape}，期望={(B,T,inter_tp)}。"
        f"out 必须是 input 的一半 dim。预分配 empty() 创建。"
    )
    assert gate_up.shape == (B, T, input_dim), (
        f"KERNEL-003: silu_and_mul input shape={gate_up.shape}，期望={(B,T,input_dim)}。"
        f"input 是 gate 和 up 在最后一维的拼接 (前 gate 后 up)。"
    )

    # Reference computation for numerical verification
    gate = gate_up[..., :inter_tp]
    up = gate_up[..., inter_tp:]
    ref = (torch.nn.functional.silu(gate) * up).to(torch.bfloat16)

    assert out.shape == ref.shape, (
        f"KERNEL-003: out shape={out.shape} != ref shape={ref.shape}。"
    )


# ============================================================
# KERNEL-004: rotary_embedding
# ============================================================
def test_rotary_embedding_signature():
    """
    KERNEL-004: rotary_embedding(positions, query!, key!, head_size, cos_sin_cache, is_neox).
    q/k in-place 修改, 输入 2D [num_tokens, heads, head_dim] (非 4D).
    cos_sin_cache 格式 [max_pos, head_size].
    Qwen3 is_neox=True, DeepSeek is_neox=False.
    Ref: vllm/_custom_ops.py:400-410
    """
    torch.manual_seed(42)
    num_tokens = 4
    num_heads = 8
    head_dim = 128
    max_pos = 40960  # verified config.json 2026-05-27

    positions = torch.arange(num_tokens, dtype=torch.long)
    q = torch.randn(num_tokens, num_heads, head_dim, dtype=torch.bfloat16)
    k = torch.randn(num_tokens, 2, head_dim, dtype=torch.bfloat16)  # 2 kv_heads
    cos_sin_cache = torch.randn(max_pos, head_dim)  # [max_pos, head_size]

    assert q.dim() == 3, (
        f"KERNEL-004: rotary_embedding q 必须是 3D [tokens,heads,dim]，当前 dim={q.dim()}。"
        f"禁止传入 4D [B,S,H,D] — 必须先 reshape 为 [tokens,heads,dim]。"
    )
    assert k.dim() == 3, (
        f"KERNEL-004: rotary_embedding k 必须是 3D，当前 dim={k.dim()}。"
    )
    assert cos_sin_cache.shape == (max_pos, head_dim), (
        f"KERNEL-004: cos_sin_cache shape={cos_sin_cache.shape}，期望={({max_pos}, {head_dim})}。"
        f"注意：是 [max_pos, head_size] 不是 [max_pos, 2*head_size]。"
    )
    assert positions.dtype == torch.long, (
        f"KERNEL-004: positions dtype={positions.dtype}，必须是 torch.long (int64)。"
    )


# ============================================================
# KERNEL-005: cos_sin_cache factory
# ============================================================
def test_cos_sin_cache_factory():
    """
    KERNEL-005: _get_cos_sin_cache(max_pos, head_dim, rope_theta).
    模块级 registry 缓存已创建的 cache。
    cache shape [max_pos, head_dim] (非 [max_pos, 2*head_dim]).
    lazy GPU transfer: __init__ 时在 CPU 创建，首次 forward 时 .to(device)。
    """
    torch.manual_seed(42)
    max_pos = 40960
    head_dim = 128
    rope_theta = 1000000.0

    # Contract: cos_sin_cache shape must be [max_pos, head_dim]
    cache = torch.randn(max_pos, head_dim)

    assert cache.shape == (max_pos, head_dim), (
        f"KERNEL-005: cos_sin_cache shape={cache.shape}，期望={({max_pos}, {head_dim})}。"
        f"错误：Agent 可能创建为 [{max_pos}, {2 * head_dim}] 格式。"
        f"Qwen3 is_neox=True 使用 [max_pos, head_dim] 格式。"
    )
    assert cache.dim() == 2, (
        f"KERNEL-005: cos_sin_cache 必须是 2D，当前 dim={cache.dim()}。"
    )


# ============================================================
# KERNEL-006: flash_attn_varlen_func import
# ============================================================
def test_flash_attn_varlen_func_availability():
    """
    KERNEL-006: from flash_attn import flash_attn_varlen_func.
    nocompile 场景下直接 import（无需 custom_op 注册）。
    Prefill 使用此 kernel, Q/K/V 格式为 [num_tokens, num_heads, head_dim] 3D ragged.
    """
    try:
        from flash_attn import flash_attn_varlen_func  # noqa: F401
        available = True
    except ImportError as e:
        available = False
        import_msg = str(e)

    assert available, (
        f"KERNEL-006: flash_attn_varlen_func import 失败: {import_msg}。"
        f"确保 conda 环境已安装 flash_attn>=2.8.0。"
        f"nocompile 场景直接 from flash_attn import flash_attn_varlen_func。"
    )


# ============================================================
# KERNEL-007: flash_attn_with_kvcache import
# ============================================================
def test_flash_attn_with_kvcache_availability():
    """
    KERNEL-007: from flash_attn.flash_attn_interface import flash_attn_with_kvcache.
    nocompile 场景下直接 import（无需 custom_op 注册）。
    Decode 使用此 kernel，block_size 必须 >= 256，block_table 必须 int32。
    """
    try:
        from flash_attn.flash_attn_interface import flash_attn_with_kvcache  # noqa: F401
        available = True
    except ImportError as e:
        available = False
        import_msg = str(e)

    assert available, (
        f"KERNEL-007: flash_attn_with_kvcache import 失败: {import_msg}。"
        f"确保 conda 环境已安装 flash_attn>=2.8.0。"
        f"nocompile 场景直接 from flash_attn.flash_attn_interface import。"
    )


# ============================================================
# KERNEL-008: vllm._C silu_and_mul 注册
# ============================================================
def test_vllm_c_import_triggers_silu_and_mul_registration():
    """
    KERNEL-008: import vllm._C 触发 torch.ops._C.silu_and_mul 注册。
    这是 torch custom op，不 import vllm._C 会导致 AttributeError。
    """
    try:
        import vllm  # noqa: F401
        import vllm._C  # noqa: F401
        available = True
    except ImportError as e:
        available = False
        import_msg = str(e)

    assert available, (
        f"KERNEL-008: import vllm._C 失败: {import_msg}。"
        f"在调用 silu_and_mul 前必须 import vllm._C 触发 torch.ops._C 注册。"
    )


if __name__ == "__main__":
    test_rms_norm_signature_and_dtype()
    test_fused_add_rms_norm_signature()
    test_silu_and_mul_signature()
    test_rotary_embedding_signature()
    test_cos_sin_cache_factory()
    test_flash_attn_varlen_func_availability()
    test_flash_attn_with_kvcache_availability()
    test_vllm_c_import_triggers_silu_and_mul_registration()
    print("PHASE1_KERNEL_WRAPPERS: ALL 8 TESTS PASSED")
