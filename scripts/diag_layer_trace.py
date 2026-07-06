#!/usr/bin/env python3
"""Trace hidden state norms layer-by-layer during prefill.

Manually iterates layers and captures statistics without hooks,
to avoid any potential interference.
"""
import os, sys, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine.models.qwen import QwenTPConfig, QwenForCausalLMTP, load_weights
from engine.tp_layers.distributed import init_tp_distributed, get_tp_rank, get_tp_size
from engine.kernels.rms_norm import rms_norm

init_tp_distributed()
rank = get_tp_rank()
tp_size = get_tp_size()
model_dir = os.environ['MODEL_DIR']

cfg = QwenTPConfig(model_dir)
model = QwenForCausalLMTP(cfg)
model = model.to(torch.bfloat16).to(f'cuda:{rank}')
model = load_weights(model, model_dir)
model.eval()

device = f'cuda:{rank}'

# Same prompt
prompt_ids = [108618, 102066, 137351]  # 苏州园林的特点是
input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
S = len(prompt_ids)

with torch.inference_mode():
    hidden_states = model.embed_tokens(input_ids)
    positions = torch.arange(S, dtype=torch.int64, device=device)

    if rank == 0:
        def stats(t, label=""):
            n = float(t.norm().cpu())
            mean = float(t.mean().cpu())
            std = float(t.std().cpu())
            m = float(t.min().cpu())
            M = float(t.max().cpu())
            return f"{label:>24s}: norm={n:10.2f} min={m:8.2f} max={M:8.2f} mean={mean:8.2f} std={std:8.2f}"

        print(stats(hidden_states, "embedding"))

        residual = None
        # Trace every layer
        for i, layer in enumerate(model.layers):
            hs = hidden_states.clone()

            # --- input_layernorm ---
            if residual is None:
                # First layer: just norm the input
                residual_val = hs.clone()
                effective_w = layer.input_layernorm._effective_weight()
                rms_norm(hs, hs.contiguous(), effective_w, layer.input_layernorm.eps)
            else:
                # Subsequent layers: fused add + norm
                from engine.kernels.rms_norm import fused_add_rms_norm
                effective_w = layer.input_layernorm._effective_weight()
                fused_add_rms_norm(hs, residual, effective_w, layer.input_layernorm.eps)
                residual_val = residual

            # --- Before attention ---
            if i in [0, 1, 2, 3, 4, 5, 6, 7, 60, 61, 62, 63]:
                print(stats(hs, f"L{i:02d} ({layer.layer_type[:4]}) input_normed"))

            # --- attention ---
            if layer.layer_type == 'full_attention':
                attn_out = layer.self_attn(hs, positions, S)
            else:
                attn_out = layer.linear_attn(hs, positions, S)

            if i in [0, 1, 2, 3, 4, 5, 6, 7, 60, 61, 62, 63]:
                print(stats(attn_out, f"L{i:02d} ({layer.layer_type[:4]}) attn_out"))

            # --- post_attention_layernorm ---
            from engine.kernels.rms_norm import fused_add_rms_norm
            effective_w = layer.post_attention_layernorm._effective_weight()
            fused_add_rms_norm(attn_out, residual_val, effective_w, layer.post_attention_layernorm.eps)
            # attn_out is now the RMS-normed residual (before MLP) — but the original attn_out is lost

            if i in [0, 1, 2, 3, 4, 5, 6, 7, 60, 61, 62, 63]:
                print(stats(attn_out, f"L{i:02d} ({layer.layer_type[:4]}) pre-mlp_normed"))

            # --- MLP ---
            mlp_out = layer.mlp(attn_out)

            if i in [0, 1, 2, 3, 4, 5, 6, 7, 60, 61, 62, 63]:
                print(stats(mlp_out, f"L{i:02d} ({layer.layer_type[:4]}) mlp_out"))

            hidden_states = mlp_out
            residual = residual_val  # residual_val is the updated residual

        # --- Final processing ---
        if residual is not None:
            hidden_states = hidden_states + residual
            print(stats(hidden_states, "final_hs (after +residual)"))

        effective_w = model.norm._effective_weight()
        out = torch.empty_like(hidden_states)
        rms_norm(out, hidden_states.contiguous(), effective_w, model.norm.eps)
        logits = model.lm_head(out)

        last_logit = logits[0, -1, :]
        topk_values, topk_indices = torch.topk(last_logit, 20)
        print(f"\n[Rank {rank}] Logits: min={last_logit.min():.4f}, max={last_logit.max():.4f}, mean={last_logit.mean():.4f}")
        print("Top-20:")
        for i in range(20):
            tid = topk_indices[i].item()
            val = topk_values[i].item()
            print(f"  {i+1}. id={tid}, logit={val:.4f}")

        print(f"NaN: {torch.isnan(last_logit).any().item()}, Inf: {torch.isinf(last_logit).any().item()}")
    else:
        # Non-rank-0: just run forward silently
        residual = None
        for layer in model.layers:
            hidden_states, residual = layer(hidden_states, positions, S, residual)
        if residual is not None:
            hidden_states = hidden_states + residual

import torch.distributed as dist
dist.barrier()
if rank == 0:
    dist.destroy_process_group()
