# engine/tp_layers/__init__.py
# Phase 2: TP 通信 — all_reduce_sum, all_gather_last_dim, CustomAR init.

from engine.tp_layers.distributed import (  # noqa: F401
    all_reduce_sum,
    all_gather_last_dim,
    init_tp_distributed,
    init_custom_ar,
    is_tp_enabled,
    get_tp_size,
    get_tp_rank,
    get_custom_ar_handle,
    CustomAllReduceHandle,
)
