#!/usr/bin/env python3
"""Compare our TP model output with HF reference on same input."""
import os, sys, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

model_dir = os.environ['MODEL_DIR']

# Load HF model with auto device mapping
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

print("Loading HF model...")
model_hf = AutoModelForCausalLM.from_pretrained(
    model_dir, torch_dtype=torch.bfloat16, device_map="auto",
    trust_remote_code=True)
model_hf.eval()
print(f"HF model loaded on devices: {model_hf.device}")

tokenizer = AutoTokenizer.from_pretrained(model_dir)

prompt = "苏州园林的特点是"
input_ids = tokenizer.encode(prompt)
print(f"Input tokens: {input_ids}")

# Forward through HF
ids = torch.tensor([input_ids], dtype=torch.long).to(model_hf.device)
with torch.inference_mode():
    outputs = model_hf(ids, use_cache=False)
    logits_hf = outputs.logits  # [1, S, V]

last_logit_hf = logits_hf[0, -1, :]
print(f"\nHF logits last position: min={last_logit_hf.min():.4f}, max={last_logit_hf.max():.4f}, mean={last_logit_hf.mean():.4f}")

top_k = 10
values_hf, indices_hf = torch.topk(last_logit_hf, top_k)
print(f"\nHF TOP-{top_k} predictions:")
for i in range(top_k):
    tok_id = indices_hf[i].item()
    tok_text = tokenizer.decode([tok_id])
    print(f"  {i+1}. id={tok_id}, logit={values_hf[i]:.4f}, text={repr(tok_text)}")

# Expected first character
expected = "讲"
expected_id = tokenizer.encode(expected)[-1] if expected in tokenizer.get_vocab() else None
if expected_id:
    expected_logit = last_logit_hf[expected_id].item()
    expected_rank = (last_logit_hf > expected_logit).sum().item()
    print(f"\nExpected '{expected}' (id={expected_id}): logit={expected_logit:.4f}, rank={expected_rank}")

# Check hidden states for first few layers (if possible)
print(f"\nHF logits shape: {logits_hf.shape}")
print(f"HF total output: min={logits_hf.min():.4f}, max={logits_hf.max():.4f}")
