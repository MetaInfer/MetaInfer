#!/usr/bin/env python3
"""Compare layer output norms: GPU (TP=4) vs CPU reference for layers 0-5."""
import os, sys, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine.models.qwen import QwenTPConfig, QwenForCausalLMTP, load_weights
from engine.tp_layers.distributed import init_tp_distributed, get_tp_rank

init_tp_distributed()
rank = get_tp_rank()
model_dir = os.environ['MODEL_DIR']

ref = torch.load('/tmp/diag_cpu_layers05_ref.pt', map_location='cpu', weights_only=True)
cp_ref = ref['checkpoints']
tokens = ref['tokens']

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

    for layer_idx, layer in enumerate(model.layers):
        hidden_states, residual = layer(hidden_states, positions, S, residual)

        if layer_idx in cp_ref and rank == 0:
            gpu_hs = hidden_states.cpu().float()
            cpu_hs = cp_ref[layer_idx]['hs'].float()
            norm_g = gpu_hs.norm()
            norm_c = cpu_hs.norm()
            diff = (gpu_hs - cpu_hs).abs()
            pct = abs(norm_g - norm_c) / norm_c * 100
            is_full = cp_ref[layer_idx].get('is_full', False)
            lt_label = 'FULL' if is_full else 'LINE'
            print(f"L{layer_idx:02d} ({lt_label}): GPU={norm_g:.4f} CPU={norm_c:.4f} diff={pct:.1f}% max={diff.max():.4f}")

        if layer_idx >= 5:
            break

import torch.distributed as dist
dist.barrier()
if rank == 0:
    # Show last position comparison for layer 3
    gpu_last = hidden_states[0, -1, :].cpu().float()
    cpu_last = cp_ref[5]['hs'][0, -1, :].float()
    print(f"\nL05 last token: GPU={gpu_last.norm():.4f} CPU={cpu_last.norm():.4f}")
    top_gpu = torch.topk(gpu_last, 5)
    top_cpu = torch.topk(cpu_last, 5)
    print(f"  GPU top5: {top_gpu.indices.tolist()} -> {top_gpu.values.tolist()}")
    print(f"  CPU top5: {top_cpu.indices.tolist()} -> {top_cpu.values.tolist()}")
    dist.destroy_process_group()
