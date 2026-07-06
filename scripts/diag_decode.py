#!/usr/bin/env python3
"""Test: single forward pass vs two-step decode (prefill + decode step)."""
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

model = QwenForCausalLMTP(cfg)
model = model.to(torch.bfloat16).to(device)
model = load_weights(model, model_dir)
model.eval()

# Full 8-token forward pass (7 prefix + 1 correct next token)
tokens_7 = [108618, 102066, 137351, 105017, 100462, 106808, 103105]
input_ids_7 = torch.tensor([tokens_7], dtype=torch.long, device=device)

with torch.inference_mode():
    # === Test 1: Full 7-token forward pass ===
    positions = torch.arange(7, dtype=torch.int64, device=device)
    logits_7 = model(input_ids_7, positions, 7)
    last_logits = logits_7[:, -1, :]
    next_token_id = last_logits.argmax(dim=-1).item()
    
    if rank == 0:
        print(f"=== Test 1: Full 7-token forward pass ===")
        print(f"  Top token: {next_token_id}")
        top5 = torch.topk(last_logits[0].float(), k=5)
        print(f"  Top 5: {[(id.item(), f'{v.item():.4f}') for id, v in zip(top5.indices, top5.values)]}")
    
    # === Test 2: Prefill + single decode step ===
    # Prefill: run 7 tokens to fill KV cache
    positions_prefill = torch.arange(7, dtype=torch.int64, device=device)
    logits_prefill = model(input_ids_7, positions_prefill, 7)
    last_logits_prefill = logits_prefill[:, -1, :]
    next_token = last_logits_prefill.argmax(dim=-1)
    
    # Decode step: run 1 token with KV cache
    next_token_id_from_decode = next_token.item()
    decode_input = next_token.unsqueeze(0)  # [1, 1]
    decode_pos = torch.tensor([7], dtype=torch.int64, device=device)
    logits_decode = model(decode_input, decode_pos, 1)
    next_next_token = logits_decode[:, -1, :].argmax(dim=-1).item()
    
    if rank == 0:
        print(f"\n=== Test 2: Prefill + Decode step ===")
        print(f"  Prefill top token: {next_token_id_from_decode}")
        print(f"  Decode next token: {next_next_token}")
        
        # Check: should match Test 1's result for first token + full 8-token forward pass
        # Full 8-token forward pass
        tokens_8 = tokens_7 + [next_token_id_from_decode]
        input_ids_8 = torch.tensor([tokens_8], dtype=torch.long, device=device)
        positions_8 = torch.arange(8, dtype=torch.int64, device=device)
        logits_8 = model(input_ids_8, positions_8, 8)
        last_logits_8 = logits_8[:, -1, :]
        expected_next = last_logits_8.argmax(dim=-1).item()
        
        print(f"\n=== Test 3: Full 8-token forward pass ===")
        print(f"  Expected next token (from full 8-token): {expected_next}")
        print(f"  Got from decode step:                   {next_next_token}")
        print(f"  Match: {'YES' if expected_next == next_next_token else 'NO - DECODE PATH BUG!'}")
        
        # Also compare logits at position 7 between both paths
        pos7_from_full = logits_8[:, -1, :].float()
        pos7_from_decode = logits_decode[:, -1, :].float()
        diff = (pos7_from_full.cpu() - pos7_from_decode.cpu()).abs()
        print(f"\n=== Logit comparison at position 7 (prediction for token 8) ===")
        print(f"  Full 8-token norm: {pos7_from_full.norm():.4f}")
        print(f"  Decode step norm:  {pos7_from_decode.norm():.4f}")
        print(f"  Diff max={diff.max():.6f} mean={diff.mean():.6f}")

import torch.distributed as dist
dist.barrier()
if rank == 0:
    dist.destroy_process_group()
