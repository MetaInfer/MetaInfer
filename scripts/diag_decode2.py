#!/usr/bin/env python3
"""Test: prefill + forward_decode with proper error handling."""
import os, sys, torch, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine.models.qwen import QwenTPConfig, QwenForCausalLMTP, load_weights
from engine.tp_layers.distributed import init_tp_distributed, get_tp_rank, get_tp_size

init_tp_distributed()
rank = get_tp_rank()
device = f'cuda:{rank}'
model_dir = os.environ['MODEL_DIR']

cfg = QwenTPConfig(model_dir)

model = QwenForCausalLMTP(cfg)
model = model.to(torch.bfloat16).to(device)
model = load_weights(model, model_dir)
model.eval()

tokens = [108618, 102066, 137351, 105017, 100462, 106808, 103105]
input_ids = torch.tensor([tokens], dtype=torch.long, device=device)

torch.cuda.synchronize(device)

with torch.inference_mode():
    # Step 1: Prefill 7 tokens
    if rank == 0: print("Step 1: Prefill 7 tokens...", flush=True)
    positions = torch.arange(7, dtype=torch.int64, device=device)
    logits = model(input_ids, positions=positions, max_seq_len=7)
    last_logits = logits[:, -1, :]
    next_tok = last_logits.argmax(dim=-1)
    if rank == 0: print(f"  Prefill done. Next token: {next_tok.item()}", flush=True)
    torch.cuda.synchronize(device)
    
    # Step 2: Decode 1 token
    if rank == 0: print("Step 2: Decode 1 token via forward_decode...", flush=True)
    decode_input = next_tok.unsqueeze(0)  # [1, 1]
    kv_len = 7
    positions_d = torch.tensor([kv_len], dtype=torch.int64, device=device)
    
    try:
        logits_d = model.forward_decode(
            decode_input, positions=positions_d, past_key_values=[kv_len])
        last_logits_d = logits_d[:, -1, :]
        next_tok2 = last_logits_d.argmax(dim=-1)
        if rank == 0: print(f"  Decode done. Next token: {next_tok2.item()}", flush=True)
    except Exception as e:
        if rank == 0: print(f"  Decode FAILED: {type(e).__name__}: {e}", flush=True)
        import traceback
        traceback.print_exc()
    
    torch.cuda.synchronize(device)
    
    # Step 3: Another decode step  
    if rank == 0: print("Step 3: Decode step 2...", flush=True)
    decode_input2 = next_tok2.unsqueeze(0)
    positions_d2 = torch.tensor([kv_len + 1], dtype=torch.int64, device=device)
    
    try:
        logits_d2 = model.forward_decode(
            decode_input2, positions=positions_d2, past_key_values=[kv_len + 1])
        next_tok3 = logits_d2[:, -1, :].argmax(dim=-1)
        if rank == 0: print(f"  Decode 2 done. Next token: {next_tok3.item()}", flush=True)
    except Exception as e:
        if rank == 0: print(f"  Decode 2 FAILED: {type(e).__name__}: {e}", flush=True)
        import traceback
        traceback.print_exc()

if rank == 0: print("All steps completed.", flush=True)

import torch.distributed as dist
dist.barrier()
if rank == 0:
    dist.destroy_process_group()
