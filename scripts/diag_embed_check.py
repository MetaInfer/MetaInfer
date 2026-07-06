#!/usr/bin/env python3
"""Check if embedding output matches expected values from safetensors."""
import os, sys, torch, json
model_dir = os.environ['MODEL_DIR']

from safetensors import safe_open

# Load embedding weight
with open(os.path.join(model_dir, 'model.safetensors.index.json')) as f:
    idx = json.load(f)

embed_key = 'model.language_model.embed_tokens.weight'
fname = idx['weight_map'][embed_key]
fpath = os.path.join(model_dir, fname)

with safe_open(fpath, framework='pt') as sf:
    embed_w = sf.get_tensor(embed_key)

print(f"Full embedding weight shape: {embed_w.shape}")
print(f"Full embedding weight: min={embed_w.min():.6f}, max={embed_w.max():.6f}, mean={embed_w.mean():.6f}, std={embed_w.std():.6f}")

# Check the tokens we use
tokens = {
    'BOS': 248044,
    '苏': 108618,
    '州': 102066,
    '园': 137351,
    '林': 105016,
    '的': 100462,
    '特': 105017,
    '点': 106808,
    '是': 103105,
    '讲': 105074,
    '究': 100026,
    '\n': 198,
    ' ': 220,
    '1': 16,
    '2': 17,
}

for name, tid in tokens.items():
    emb = embed_w[tid]
    norm = emb.norm().item()
    print(f"Token '{name}' (id={tid}): norm={norm:.4f}, min={emb.min():.4f}, max={emb.max():.4f}, mean={emb.mean():.4f}")

# Now load the model and check the same tokens
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine.models.qwen import QwenTPConfig, QwenForCausalLMTP, load_weights
from engine.tp_layers.distributed import init_tp_distributed, get_tp_rank

init_tp_distributed()
rank = get_tp_rank()
cfg = QwenTPConfig(model_dir)

print(f"\nRank {rank}: tp_size={cfg.tp_size}")
print(f"  vocab_start={rank * cfg.tp_size}..{(rank+1) * cfg.tp_size}" if cfg.tp_size > 1 else "")
print(f"  embed_tokens.weight shape: {cfg.tp_size}")

# Check the model's embedding for the same tokens
model = QwenForCausalLMTP(cfg)
model = model.to(torch.bfloat16).to(f'cuda:{rank}')
model = load_weights(model, model_dir)
model.eval()

device = f'cuda:{rank}'
with torch.inference_mode():
    for name, tid in tokens.items():
        ids = torch.tensor([[tid]], dtype=torch.long, device=device)
        emb = model.embed_tokens(ids)  # [1, 1, 5120]
        norm = emb.norm().item()
        print(f"Rank {rank} Model '{name}' (id={tid}): norm={norm:.4f}")

import torch.distributed as dist
dist.barrier()
if rank == 0:
    dist.destroy_process_group()
