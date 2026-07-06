#!/usr/bin/env python3
"""Compare model forward logits: GPU (TP=4) vs CPU safetensors reference."""
import os, sys, torch, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch.nn.functional as F

from engine.models.qwen import QwenTPConfig, QwenForCausalLMTP, load_weights
from engine.tp_layers.distributed import init_tp_distributed, get_tp_rank, get_tp_size

init_tp_distributed()
rank = get_tp_rank()
tp_size = get_tp_size()
model_dir = os.environ['MODEL_DIR']
device = f'cuda:{rank}'

cfg = QwenTPConfig(model_dir)
with open(os.path.join(model_dir, 'config.json')) as f:
    raw = json.load(f)
tc = raw.get('text_config', raw)
hidden_size = tc['hidden_size']

# Load GPU model
model = QwenForCausalLMTP(cfg)
model = model.to(torch.bfloat16).to(device)
model = load_weights(model, model_dir)
model.eval()

tokens = [108618, 102066, 137351, 105017, 100462, 106808, 103105]
input_ids = torch.tensor([tokens], dtype=torch.long, device=device)
B, S = 1, len(tokens)
positions = torch.arange(S, dtype=torch.int64, device=device)

# CPU weights
from safetensors import safe_open
with open(os.path.join(model_dir, 'model.safetensors.index.json')) as f:
    idx = json.load(f)
wm = idx['weight_map']

def load_cpu(key):
    fname = wm[key]
    fpath = os.path.join(model_dir, fname)
    with safe_open(fpath, framework='pt', device='cpu') as sf:
        return sf.get_tensor(key).float()

# Load all layer 0 weights for CPU reference
def load_layer_wts(layer_idx, device='cpu'):
    p = f'model.language_model.layers.{layer_idx}.'
    return {
        'input_ln': load_cpu(p + 'input_layernorm.weight'),
        'post_ln': load_cpu(p + 'post_attention_layernorm.weight'),
    }

with torch.inference_mode():
    # GPU forward pass
    gpu_logits = model(input_ids, positions, S)
    # Collect last token logits
    gpu_last = gpu_logits[:, -1, :].clone()

    if rank == 0:
        gpu_last_cpu = gpu_last.cpu().float()
        top5_vals, top5_ids = torch.topk(gpu_last_cpu[0], k=5)
        print(f"=== GPU Logits (last position) ===")
        print(f"  norm: {gpu_last_cpu.norm():.4f}")
        print(f"  min: {gpu_last_cpu.min():.6f}  max: {gpu_last_cpu.max():.6f}")
        print(f"  Top 5: ")
        for i in range(5):
            print(f"    token_id={top5_ids[i].item()} val={top5_vals[i].item():.4f}")

        # Get the token with highest logit
        best_id = gpu_last_cpu[0].argmax().item()
        print(f"  Greedy next token: {best_id}")

        # Also check first few layer outputs
        print(f"\n=== Layer output norms ===")
        # We can't directly get intermediate outputs, but we can check logits
        # Print last position embedding
        print(f"  last pos logits norm: {gpu_last_cpu.norm():.4f}")
        print(f"  Top 20 logits: {top5_ids.tolist()}")

        # Check if the output looks reasonable (no NaN/Inf)
        has_nan = torch.isnan(gpu_last_cpu).any().item()
        has_inf = torch.isinf(gpu_last_cpu).any().item()
        print(f"  NaN: {has_nan}, Inf: {has_inf}")

import torch.distributed as dist
dist.barrier()
if rank == 0:
    dist.destroy_process_group()
