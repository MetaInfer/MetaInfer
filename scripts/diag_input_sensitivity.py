#!/usr/bin/env python3
"""Check if model produces different outputs for different inputs (basic sanity)."""
import os, sys, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine.models.qwen import QwenTPConfig, QwenForCausalLMTP, load_weights
from engine.tp_layers.distributed import init_tp_distributed, get_tp_rank

init_tp_distributed()
rank = get_tp_rank()
model_dir = os.environ['MODEL_DIR']

cfg = QwenTPConfig(model_dir)
model = QwenForCausalLMTP(cfg)
model = model.to(torch.bfloat16).to(f'cuda:{rank}')
model = load_weights(model, model_dir)
model.eval()

device = f'cuda:{rank}'

inputs = [
    [248044],                    # BOS token
    [108618, 102066, 137351],    # 苏州园林的特点是
    [108618, 102066, 137351, 105017],  # 苏州园林的特点是讲究
]

for inp in inputs:
    ids = torch.tensor([inp], dtype=torch.long, device=device)
    S = ids.shape[1]
    pos = torch.arange(S, dtype=torch.int64, device=device)
    with torch.inference_mode():
        logits = model(ids, positions=pos, max_seq_len=S)
    last = logits[0, -1, :]
    top5_val, top5_idx = torch.topk(last, 5)
    if rank == 0:
        print(f"Input {inp}:")
        for i in range(5):
            print(f"  #{i+1}: id={top5_idx[i].item()}, logit={top5_val[i].item():.4f}")
        print()

import torch.distributed as dist
dist.barrier()
if rank == 0:
    dist.destroy_process_group()
