# Qwen3-8B TP=4 nocompile Physical Tracing Summary

**Date**: 2026-05-27 16:07:01
**PyTorch**: 2.9.1+cu128
**GPU**: NVIDIA A800-SXM4-80GB

## Config (verified from config.json)

| Param | Value |
|-------|-------|
| max_position_embeddings | **40960** |
| intermediate_size | **12288** |
| hidden_size | 4096 |
| num_attention_heads | 32 |
| num_key_value_heads | 8 |
| max_blocks (256) | **160** |

## Per-Rank Dimensions (TP=4)

| Param | Value |
|-------|-------|
| qkv_proj weight | **[1536, 4096]** |
| gate_up_proj weight | **[6144, 4096]** |
| per_rank_attn_heads | 8 |
| per_rank_kv_heads | 2 |
| q_size | 1024 |
| kv_size | 256 |

## KV Cache (Actual Paged Format)

- block_size: **256**
- key_cache after inference: **[1, 256, 2, 128]**
- block_table dtype: **torch.int32**

## fused_add_rms_norm Weight Identity (FM-003 Verification)

- Total calls during inference: **1728**
- Unique weight data_ptrs: **72**
- Calls per layer: **48**
- All calls use self-layer weights: **True**

## Cross-Layer Weight Independence

- layer0 vs layer1 input_layernorm different ptr: **True**
- layer0 vs layer1 post_attention_layernorm different ptr: **True**

## Runtime

- Output: '（ ） A：建筑与园林结合 B：建筑与自然结合 C：建筑与山水结合 D：建筑'
- Greedy match: **True**
- Throughput: **10.6 tok/s**
- GPU memory: 4.69 GB allocated

## Dependencies

- flash_attn_varlen_func: available
- flash_attn_with_kvcache: available
- vllm._custom_ops: available
