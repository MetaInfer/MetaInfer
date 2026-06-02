# engine/kernels/__init__.py
# 标品黑盒 kernel wrapper 包。
# 所有 wrapper 只做薄封装，不修改 vLLM kernel 内部逻辑。
# Phase 1: 7 个数值基元 kernel wrappers.

from engine.kernels.vllm_wrappers import (       # noqa: F401
    rms_norm,
    fused_add_rms_norm,
    silu_and_mul,
    rotary_embedding,
    _get_cos_sin_cache,
    make_cos_sin_cache,
    flash_attn_varlen_func,
    flash_attn_with_kvcache,
)

# 注意: flash_attn_with_kvcache 在 nocompile 场景下直接 import，
# 无需 custom_op 注册 (custom_op+register_fake 仅在 torch.compile tracing 时需要)。
