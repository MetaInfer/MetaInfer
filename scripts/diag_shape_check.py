#!/usr/bin/env python3
"""Check actual weight shapes at runtime."""
import os, sys, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine.models.qwen import QwenTPConfig, QwenForCausalLMTP, load_weights
from engine.tp_layers.distributed import init_tp_distributed, get_tp_rank, get_tp_size

init_tp_distributed()
rank = get_tp_rank()
tp_size = get_tp_size()
model_dir = os.environ['MODEL_DIR']

cfg = QwenTPConfig(model_dir)
print(f"Rank {rank}: tp_size={tp_size}, heads_per_rank={cfg.heads_per_rank}, kv_heads_per_rank={cfg.kv_heads_per_rank}")
print(f"  k_heads_per_rank={cfg.k_heads_per_rank}, v_heads_per_rank={cfg.v_heads_per_rank}")

model = QwenForCausalLMTP(cfg)
model = model.to(torch.bfloat16).to(f'cuda:{rank}')

# Check shapes BEFORE weight loading
layer3 = model.layers[3]
fa = layer3.self_attn
print(f"\nFullAttention layer 3 (before weight loading):")
print(f"  q_proj.weight: {fa.q_proj.weight.shape}")
print(f"  k_proj.weight: {fa.k_proj.weight.shape}")
print(f"  v_proj.weight: {fa.v_proj.weight.shape}")
print(f"  o_proj.weight: {fa.o_proj.weight.shape}")
print(f"  q_norm.weight: {fa.q_norm.weight.shape}")
print(f"  k_norm.weight: {fa.k_norm.weight.shape}")

# Expected: q_proj=[3072, 5120], k_proj=[256, 5120], v_proj=[256, 5120], o_proj=[5120, 1536]

model = load_weights(model, model_dir)

print(f"\nFullAttention layer 3 (after weight loading):")
print(f"  q_proj.weight: {fa.q_proj.weight.shape}")
print(f"  k_proj.weight: {fa.k_proj.weight.shape}")
print(f"  v_proj.weight: {fa.v_proj.weight.shape}")
print(f"  o_proj.weight: {fa.o_proj.weight.shape}")

# Check GatedDeltaNet layer 0
layer0 = model.layers[0]
gdn = layer0.linear_attn
print(f"\nGatedDeltaNet layer 0:")
print(f"  in_proj_qkv.weight: {gdn.in_proj_qkv.weight.shape}")
print(f"  in_proj_a.weight: {gdn.in_proj_a.weight.shape}")
print(f"  in_proj_b.weight: {gdn.in_proj_b.weight.shape}")
print(f"  in_proj_z.weight: {gdn.in_proj_z.weight.shape}")
print(f"  out_proj.weight: {gdn.out_proj.weight.shape}")

import torch.distributed as dist
dist.barrier()
if rank == 0:
    dist.destroy_process_group()
