"""
Phase 9 — KVMemoryPool: KV cache memory budget estimation.

Blueprint contracts:
  - framework_layer.components[1] KVMemoryPool
  - framework_layer.components[6] LLMEngine._estimate_kv_blocks.dense_pseudocode

TP path note:
  KVMemoryPool is only used for estimate_num_blocks (budget logging) in TP path.
  Actual KV cache (_key_cache/_value_cache) is created by QwenAttentionTP internally
  via torch.zeros. GPU placeholders are NOT created here for TP path — those are
  exclusively for RealModelRunner (HF fallback) per the nano-vllm override.
"""


class KVMemoryPool:
    """KV cache memory pool.

    TP path: estimate_num_blocks only (logging).
    HF path: also provides GPU placeholder tensors (not implemented in this scope).

    Blueprint responsibility_boundary:
      KVMemoryPool: 仅显存预算(estimate_num_blocks)+GPU placeholder.
      BlockManager:  运行时分配/释放/prefix caching+get_num_free_blocks() API.
    """

    def __init__(self, num_blocks, block_size, num_layers, num_kv_heads, head_dim,
                 dtype_size=2):
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.dtype_size = dtype_size
        # GPU placeholders: NOT created in TP path
        # (KV is managed by QwenAttentionTP internally)

    @staticmethod
    def estimate_num_blocks_dense(free_bytes, reserve_bytes, mem_utilization,
                                   num_layers, num_kv_heads, head_dim, block_size,
                                   dtype_size=2):
        """Dense KV budget estimation.

        Formula (blueprint _estimate_kv_blocks.dense_pseudocode):
          K+V per token = layers * kv_heads * head_dim * 2 * elem_bytes
          bytes_per_block = bytes_per_token * block_size
          budget = max(0, int((free_bytes - reserve_bytes) * mem_utilization))
          num_blocks = max(1, budget // max(bytes_per_block, 1))
        """
        bytes_per_token = num_layers * num_kv_heads * head_dim * 2 * dtype_size
        bytes_per_block = bytes_per_token * block_size
        budget = max(0, int((free_bytes - reserve_bytes) * mem_utilization))
        return max(1, budget // max(bytes_per_block, 1))
