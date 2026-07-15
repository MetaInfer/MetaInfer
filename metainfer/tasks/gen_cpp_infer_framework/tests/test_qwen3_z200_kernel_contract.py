"""Static contract tests for the standalone Qwen3 Z200 HIP reference kernels."""

from __future__ import annotations

from pathlib import Path


KERNEL_SOURCE = (
    Path(__file__).parents[1]
    / "notebooks"
    / "qwen3_z200_kernels.hip.cpp"
)
MODEL_CONTRACT = (
    Path(__file__).parents[1]
    / "notebooks"
    / "03_qwen3_8b_cpp_contract.md"
)
OPERATOR_CONTRACT = (
    Path(__file__).parents[1]
    / "notebooks"
    / "04_qwen3_z200_operator_contract.md"
)


def test_q8_0_layout_and_required_entrypoints_are_present():
    source = KERNEL_SOURCE.read_text(encoding="utf-8")

    assert "int8_t qs[kQ8BlockSize]" in source
    assert 'static_assert(sizeof(BlockQ8_0) == 34' in source
    assert "qwen3_z200_launch_dequant_q8_0_to_fp16" in source
    assert "qwen3_z200_launch_cast_fp32_to_fp16" in source
    assert "qwen3_z200_launch_embedding_lookup_q8_0" in source
    assert "qwen3_z200_q8_linear_fp32" in source


def test_q8_linear_uses_fp16_inputs_and_fp32_accumulation():
    source = KERNEL_SOURCE.read_text(encoding="utf-8")
    linear = source.split(
        'extern "C" hipblasStatus_t qwen3_z200_q8_linear_fp32', 1
    )[1].split(
        'extern "C" hipError_t qwen3_z200_launch_embedding_lookup', 1
    )[0]

    assert linear.count("HIP_R_16F") >= 2
    assert "HIP_R_32F" in linear
    assert "HIPBLAS_COMPUTE_32F" in linear
    assert "HIPBLAS_OP_T" in linear
    assert "HIPBLAS_OP_N" in linear


def test_single_gpu_contract_documents_the_runnable_q8_flow():
    model_contract = MODEL_CONTRACT.read_text(encoding="utf-8")
    operator_contract = OPERATOR_CONTRACT.read_text(encoding="utf-8")

    assert "04_qwen3_z200_operator_contract.md" in model_contract
    assert "qwen3_z200_kernels.hip.cpp" in operator_contract
    assert "## 0. 核心约定：矩阵乘统一调用 hipBLAS" in operator_contract
    assert "所有带权重的矩阵乘都调用 hipBLAS" in operator_contract
    assert "hipblasGemmEx(FP16, FP16, FP32 compute)" in operator_contract
    assert "qwen3_z200_launch_embedding_lookup_q8_0" in operator_contract
    assert "qwen3_z200_q8_linear_fp32" in operator_contract
    assert "q8_linear(last_hidden, lm_head, M=1,N=151936,K=4096)" in operator_contract
    assert "1.159 GiB" in operator_contract
    assert "Fused QKV 必须 split/pack" in operator_contract
    assert "Fused gate-up 必须 pack 或使用 strided SwiGLU" in operator_contract
    assert "逐层 forward 和逐 token decode 中禁止 `hipMalloc/hipFree`" in operator_contract
