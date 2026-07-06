#!/usr/bin/env python3
"""Compare layer output norms: GPU (TP=4) vs CPU reference (layers 0-5)."""
import os, sys, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine.models.qwen import QwenTPConfig, QwenForCausalLMTP, load_weights
from engine.tp_layers.distributed import init_tp_distributed, get_tp_rank

init_tp_distributed()
rank = get_tp_rank()
model_dir = os.environ['MODEL_DIR']

# Load CPU reference
ref = torch.load('/tmp/diag_cpu_layers05_ref.pt', map_location='cpu', weights_only=True)
cp_ref = ref['checkpoints']
tokens = ref['tokens']
cpu_emb = ref['embedding'].float()

# Load model once
cfg = QwenTPConfig(model_dir)
model = QwenForCausalLMTP(cfg)
model = model.to(torch.bfloat16).to(f'cuda:{rank}')
model = load_weights(model, model_dir)
model.eval()

device = f'cuda:{rank}'
input_ids = torch.tensor([tokens], dtype=torch.long, device=device)
S = len(tokens)
positions = torch.arange(S, dtype=torch.int64, device=device)

with torch.inference_mode():
    hidden_states = model.embed_tokens(input_ids)
    residual = None

    if rank == 0:
        print(f"Embedding: GPU norm={hidden_states.norm():.4f}, CPU norm={cpu_emb.norm():.4f}")
        gpu_emb = hidden_states.cpu().float()
        emb_diff = (gpu_emb - cpu_emb).abs()
        print(f"  Emb diff: max={emb_diff.max():.6f}, mean={emb_diff.mean():.6f}")

    for layer_idx, layer in enumerate(model.layers):
        hidden_states, residual = layer(hidden_states, positions, S, residual)

        if layer_idx in cp_ref:
            gpu_hs_float = hidden_states.cpu().float()  # all_reduced already
            cpu_hs = cp_ref[layer_idx]['hs'].float()

            if rank == 0:
                norm_g = gpu_hs_float.norm()
                norm_c = cpu_hs.norm()
                diff = (gpu_hs_float - cpu_hs).abs()

                is_full = cp_ref[layer_idx].get('is_full', False)
                label = 'FULL' if is_full else 'LINE'
                pct = abs(norm_g - norm_c) / norm_c * 100
                print(f"\nLayer {layer_idx} ({label}): GPU norm={norm_g:.4f}, CPU norm={norm_c:.4f} ({pct:.2f}%)")
                print(f"  Diff: max={diff.max():.4f}, mean={diff.mean():.4f}")

                if diff.max() > 1.0:
                    print(f"  *** LARGE DIFFERENCE at layer {layer_idx}! ***")
                    max_idx = diff.argmax()
                    max_pos = torch.unravel_index(max_idx, diff.shape)
                    print(f"  Max diff at {max_pos}: GPU={gpu_hs_float[max_pos].item():.4f}, CPU={cpu_hs[max_pos].item():.4f}")

        if layer_idx >= 5:
            break

import torch.distributed as dist
dist.barrier()
if rank == 0:
    dist.destroy_process_group()
