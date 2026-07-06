#!/usr/bin/env python3
"""Test prefill logits on known input to check model quality."""
import os, sys, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine.models.qwen import QwenTPConfig, QwenForCausalLMTP, load_weights
from engine.tp_layers.distributed import init_tp_distributed, get_tp_rank, get_tp_size
from transformers import AutoTokenizer

init_tp_distributed()
rank = get_tp_rank()
model_dir = os.environ['MODEL_DIR']

cfg = QwenTPConfig(model_dir)
model = QwenForCausalLMTP(cfg)
model = model.to(torch.bfloat16).to(f'cuda:{rank}')
model = load_weights(model, model_dir)
model.eval()

tokenizer = AutoTokenizer.from_pretrained(model_dir)

prompt = "苏州园林的特点是"
input_ids = tokenizer.encode(prompt)
print(f"[Rank {rank}] Input tokens: {input_ids}")
print(f"[Rank {rank}] Input decoded: {tokenizer.decode(input_ids)}")

ids = torch.tensor([input_ids], dtype=torch.long, device=f'cuda:{rank}')
S = ids.shape[1]
positions = torch.arange(S, dtype=torch.int64, device=f'cuda:{rank}')

with torch.inference_mode():
    logits = model(ids, positions=positions, max_seq_len=S)

last_logit = logits[0, -1, :]  # [V]
print(f"\n[Rank {rank}] Logits shape: {logits.shape}")
print(f"[Rank {rank}] Last position logits: min={last_logit.min():.4f}, max={last_logit.max():.4f}, mean={last_logit.mean():.4f}")

# Top-k tokens
top_k = 20
values, indices = torch.topk(last_logit, top_k)
print(f"\n[Rank {rank}] Top-{top_k} predicted tokens:")
for i in range(top_k):
    token_id = indices[i].item()
    token_text = tokenizer.decode([token_id])
    print(f"  {i+1}. id={token_id}, logit={values[i]:.4f}, text={repr(token_text)}")

# Also check what the top prediction is
top_id = indices[0].item()
top_text = tokenizer.decode([top_id])
print(f"\n[Rank {rank}] Top prediction: id={top_id}, text={repr(top_text)}")

# Check if the expected first token is in the top-k
expected_char = "讲"
expected_id = tokenizer.encode(expected_char)[-1] if expected_char in tokenizer.get_vocab() else None
if expected_id:
    expected_logit = last_logit[expected_id].item()
    expected_rank = (last_logit > expected_logit).sum().item()
    print(f"\n[Rank {rank}] Expected token '{expected_char}' (id={expected_id}): logit={expected_logit:.4f}, rank={expected_rank}")

import torch.distributed as dist
dist.barrier()
if rank == 0:
    dist.destroy_process_group()
