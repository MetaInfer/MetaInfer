#!/usr/bin/env python3
"""Test single-token decode: feed one known Chinese token, see what's predicted next."""
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

# Test with just BOS token — see what the model predicts as first character
device = f'cuda:{rank}'
bos_id = 248044  # <|endoftext|>

input_ids = torch.tensor([[bos_id]], dtype=torch.long, device=device)
positions = torch.tensor([0], dtype=torch.int64, device=device)
kv_len = 0

print(f"[Rank {rank}] Input: bos_token (id={bos_id})")

with torch.inference_mode():
    logits = model.forward_decode(input_ids, positions=positions, past_key_values=[kv_len])

last_logit = logits[0, -1, :]
topk = 20
values, indices = torch.topk(last_logit, topk)
print(f"\n[Rank {rank}] Top-{topk} predictions after BOS token:")
for i in range(topk):
    tid = indices[i].item()
    print(f"  {i+1}. id={tid}, logit={values[i]:.4f}")

# Try with a Chinese token that starts a common sequence
# Token for "苏" (first char of 苏州)
test_token_id = 108618  # from earlier diagnostic
input_ids2 = torch.tensor([[test_token_id]], dtype=torch.long, device=device)

with torch.inference_mode():
    logits2 = model.forward_decode(input_ids2, positions=positions, past_key_values=[kv_len])

last_logit2 = logits2[0, -1, :]
values2, indices2 = torch.topk(last_logit2, topk)
print(f"\n[Rank {rank}] Top-{topk} predictions after '苏' (id={test_token_id}):")
for i in range(topk):
    tid = indices2[i].item()
    print(f"  {i+1}. id={tid}, logit={values2[i]:.4f}")

# Also try prefill with the full prompt
prompt_ids = [108618, 102066, 137351]  # 苏州园林的特点是
input_ids3 = torch.tensor([prompt_ids], dtype=torch.long, device=device)
positions3 = torch.arange(len(prompt_ids), dtype=torch.int64, device=device)

print(f"\n[Rank {rank}] Prefill with {len(prompt_ids)} tokens: {prompt_ids}")

with torch.inference_mode():
    logits3 = model(input_ids3, positions=positions3, max_seq_len=len(prompt_ids))

last_logit3 = logits3[0, -1, :]
values3, indices3 = torch.topk(last_logit3, topk)
print(f"\n[Rank {rank}] Top-{topk} predictions after full prompt:")
for i in range(topk):
    tid = indices3[i].item()
    print(f"  {i+1}. id={tid}, logit={values3[i]:.4f}")

# Also check: what is id=198 actually?
print(f"\n[Rank {rank}] id=198 is likely '\\n' (newline)")

import torch.distributed as dist
dist.barrier()
if rank == 0:
    dist.destroy_process_group()
