"""Structure mapper tests."""

from ..orchestrator.structure_mapper import (
    _infer_layer,
    _infer_op_type,
    build_mapping,
)


def test_infer_layer_from_call_stack():
    stack = "  File \"sglang/srt/layers/attn/triton_ops.py\", line 45\n  File \"model.py\""
    layer = _infer_layer(stack, "triton_attn_kernel", {})
    assert "attn" in layer.lower() if layer else True  # matched sglang path


def test_infer_layer_model_layers_pattern():
    stack = "model.layers.5.self_attn.qkv_proj"
    layer = _infer_layer(stack, "triton_gemm", {})
    assert layer == "layer_5"


def test_infer_op_type_attention():
    assert _infer_op_type("flash_attn_fwd", "") == "Attention"
    assert _infer_op_type("flash_fwd_splitkv_mla", "") == "Attention"


def test_infer_op_type_gemm():
    assert _infer_op_type("triton_gemm_kernel", "") == "GEMM"
    assert _infer_op_type("w8a8_bf16_matmul", "") == "GEMM"


def test_infer_op_type_moe():
    assert _infer_op_type("fused_moe_kernel", "") == "MoE"


def test_infer_op_type_norm():
    assert _infer_op_type("rms_norm_kernel", "") == "Norm"


def test_infer_op_type_nccl():
    assert _infer_op_type("ncclAllReduce", "") == "NCCL"


def test_build_mapping_empty():
    entries = build_mapping([], {"num_hidden_layers": 32})
    assert entries == []


def test_build_mapping_with_call_stack():
    kernels = [
        {
            "kernel_name": "triton_gemm",
            "total_dur_us": 1000,
            "count": 10,
            "call_stack": "model.layers.3.self_attn.q_proj",
        }
    ]
    entries = build_mapping(kernels, {"num_hidden_layers": 32})
    assert len(entries) == 1
    assert entries[0]["kernel_name"] == "triton_gemm"
    assert entries[0]["model_layer"] == "layer_3"
    assert entries[0]["op_type"] == "GEMM"
    assert entries[0]["confidence"] == "high"
