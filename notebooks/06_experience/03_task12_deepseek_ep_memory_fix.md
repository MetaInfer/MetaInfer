---
name: DeepSeek-V2 TP memory issue and EP fix
description: Root cause and fix for DeepSeek-V2 high GPU memory usage (~36GB/card). Routed experts fully replicated with ep_size=1, fix is ep_size=2/4
type: feedback
---

## Issue
DeepSeek-V2-Lite (16B model) with TP=4 shows ~36GB per card increase instead of expected ~8GB.

## Root Cause
DeepSeek-V2-Lite config (from blueprint):
- 64 routed experts per MoE layer, moe_intermediate_size=1408
- 26 MoE layers (layers 1-26, layer 0 uses dense MLP)
- Each expert: gate_up_proj [2*1408x2048] + down_proj [1408x2048] = ~8.6M params (17MB in bf16)
- 64 experts x 26 layers x 8.6M ≈ 14.3B params → ~28.6 GB in bf16

With `ep_size=1` (default), ALL 64 experts are replicated on ALL 4 ranks:
- 28.6 GB (experts) + ~2 GB (attention/embeddings, TP-sliced) + ~2 GB (KV cache) = ~33 GB per card
- This matches the observed 50,495 - 13,885 = 36,610 MiB increase

## Fix
Change `ep_size` from 1 to 2 or 4 in test:
- `ep_size=2`: 2 EP groups, 32 experts/rank (memory halved to ~18 GB)
- `ep_size=4`: 4 EP groups, 16 experts/rank (memory quartered to ~11 GB)

The EP infrastructure (`DeepseekV2MoE._moe_infer_ep`) already implements proper all-to-all dispatch and gather for expert parallelism. See blueprint `deepseek_v2_v3_mla_moe.moe_routing_hybrid_parallel`.

## Implementation Details
- EP group init: `init_ep_process_group(ep_size)` in `engine/tp_layers/distributed.py`
  - Creates `ep_size` groups, each with `world_size // ep_size` ranks
  - With tp_size=4, world_size=4 → `ep_size=2` gives 2 groups of 2 ranks
- Expert instantiation: `experts_per_rank = n_routed_experts // ep_size`
- Dispatch: all-to-all sends tokens to EP rank owning the expert
- Process: each EP rank runs its local experts
- Gather: all-to-all returns results to original rank

## Verification
```bash
CUDA_VISIBLE_DEVICES=3,4,5,6 torchrun --nproc_per_node=4 -m pytest tests/test_deepseek_tp_real.py -v -s
```
Note: Requires GPUs with sufficient contiguous memory (~20 GB per card for ep_size=2).

## Related Files
- `tests/test_deepseek_tp_real.py` - change `ep_size` parameter
- `engine/models/deepseek_v2.py` - `DeepseekV2MoE` class with EP/TP support
- `engine/tp_layers/distributed.py` - `init_ep_process_group` for EP group setup
- `inference_blueprint.json` → `model_layer.deepseek_v2_v3_mla_moe.moe_routing_hybrid_parallel`
