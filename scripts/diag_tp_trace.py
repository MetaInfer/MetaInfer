#!/usr/bin/env python3
"""Trace get_tp_size() calls during model construction."""
import os, sys, traceback
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine.tp_layers import distributed
original_get_tp_size = distributed.get_tp_size
call_count = [0]

def traced_get_tp_size():
    val = original_get_tp_size()
    call_count[0] += 1
    if val not in (0, 1, 4):
        print(f"!!! get_tp_size() returned {val} at call #{call_count[0]}")
        traceback.print_stack(limit=8)
    return val

distributed.get_tp_size = traced_get_tp_size

# Also need to patch the imported version in linear.py
from engine.tp_layers import linear
if hasattr(linear, 'get_tp_size'):
    linear.get_tp_size = traced_get_tp_size

from engine.tp_layers.distributed import init_tp_distributed, get_tp_rank

init_tp_distributed()
rank = get_tp_rank()
model_dir = os.environ['MODEL_DIR']

print(f"Rank {rank}: tp_size before model construction = {distributed.get_tp_size()}")
call_count[0] = 0  # Reset

from engine.models.qwen import QwenTPConfig, QwenForCausalLMTP
cfg = QwenTPConfig(model_dir)
model = QwenForCausalLMTP(cfg)

print(f"\nTotal get_tp_size() calls during construction: {call_count[0]}")

# Check shapes
layer3 = model.layers[3]
fa = layer3.self_attn
print(f"\nLayer 3 FullAttention shapes:")
print(f"  q_proj.weight: {fa.q_proj.weight.shape}")
print(f"  k_proj.weight: {fa.k_proj.weight.shape}")
print(f"  v_proj.weight: {fa.v_proj.weight.shape}")
print(f"  o_proj.weight: {fa.o_proj.weight.shape}")

# Also check layer 0 GatedDeltaNet
layer0 = model.layers[0]
gdn = layer0.linear_attn
print(f"\nLayer 0 GatedDeltaNet shapes:")
print(f"  in_proj_qkv: {gdn.in_proj_qkv.weight.shape}")
print(f"  in_proj_a: {gdn.in_proj_a.weight.shape}")
print(f"  in_proj_b: {gdn.in_proj_b.weight.shape}")
print(f"  in_proj_z: {gdn.in_proj_z.weight.shape}")
print(f"  out_proj: {gdn.out_proj.weight.shape}")

import torch.distributed as dist
if rank == 0:
    dist.barrier()
    dist.destroy_process_group()
