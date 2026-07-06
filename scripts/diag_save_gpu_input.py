#!/usr/bin/env python3
"""Step 1: Save GPU layer 3 input for CPU comparison.

Lightweight: only loads the model and runs layers 0-2, then saves layer 3 input.
Does NOT recompute FullAttention on GPU (avoids complexity of TP sharding comparison).
"""
import os, sys, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine.models.qwen import QwenTPConfig, QwenForCausalLMTP, load_weights
from engine.tp_layers.distributed import init_tp_distributed, get_tp_rank, get_tp_size
from engine.kernels.rms_norm import fused_add_rms_norm

init_tp_distributed()
rank = get_tp_rank()
tp_size = get_tp_size()
model_dir = os.environ['MODEL_DIR']
device = f'cuda:{rank}'

cfg = QwenTPConfig(model_dir)
model = QwenForCausalLMTP(cfg)
model = model.to(torch.bfloat16).to(device)
model = load_weights(model, model_dir)
model.eval()

tokens = [108618, 102066, 137351, 105017, 100462, 106808, 103105]
input_ids = torch.tensor([tokens], dtype=torch.long, device=device)
S = len(tokens)
positions = torch.arange(S, dtype=torch.int64, device=device)

with torch.inference_mode():
    hidden_states = model.embed_tokens(input_ids)
    residual = None
    for lx in range(3):
        hidden_states, residual = model.layers[lx](hidden_states, positions, S, residual)

    # Save the pre-norm states for layer 3 input
    hs_in = hidden_states.clone()
    resid_in = residual.clone()

    # Apply layer 3 input_layernorm (fused_add_rms_norm)
    layer3 = model.layers[3]
    w_ln = layer3.input_layernorm._effective_weight()
    hs_work = hs_in.clone()
    fused_add_rms_norm(hs_work, resid_in, w_ln, layer3.input_layernorm.eps)
    # fused_add_rms_norm: residual += hs_in, hs_work = rms_norm(residual)

    # Also run the actual layer 3 forward for comparison
    hs_out, resid_out = model.layers[3](hs_in, positions, S, resid_in)

    # Save everything from rank 0
    if rank == 0:
        torch.save({
            'hs_pre_ln': hs_in.cpu(),       # MLP output from layer 2
            'resid_pre_ln': resid_in.cpu(),  # residual before layer 3
            'hs_work': hs_work.cpu(),        # normed input to FullAttention
            'hs_out_gpu': hs_out.cpu(),      # Layer 3 output (mlp_out)
            'resid_out_gpu': resid_out.cpu(), # Residual after layer 3
            'tokens': tokens,
            'S': S,
        }, '/tmp/diag_gpu_layer3_input.pt')
        print(f"GPU layer 3 input saved to /tmp/diag_gpu_layer3_input.pt")
        print(f"  hs_work (normed input) norm: {hs_work.norm():.4f}")
        print(f"  hs_out_gpu (layer 3 mlp_out) norm: {hs_out.norm():.4f}")
        print(f"  resid_pre_ln norm: {resid_in.norm():.4f}")
        print(f"  resid_out_gpu norm: {resid_out.norm():.4f}")

import torch.distributed as dist
dist.barrier()
if rank == 0:
    dist.destroy_process_group()
