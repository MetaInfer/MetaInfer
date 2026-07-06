#!/usr/bin/env python3
"""Compare GPU model layer outputs with CPU reference. Run with torchrun."""
import os, sys, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine.models.qwen import QwenTPConfig, QwenForCausalLMTP, load_weights
from engine.tp_layers.distributed import init_tp_distributed, get_tp_rank, get_tp_size
from engine.kernels.rms_norm import rms_norm

init_tp_distributed()
rank = get_tp_rank()
tp_size = get_tp_size()
model_dir = os.environ['MODEL_DIR']

# Load CPU reference
ref = torch.load('/tmp/diag_cpu_layers05_ref.pt', map_location='cpu', weights_only=True)
cp_ref = ref['checkpoints']
tokens = ref['tokens']

# Load model
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

    if rank == 0:
        print(f"[GPU] embedding norm: {hidden_states.norm():.4f}")
        cpu_emb = ref['embedding'].float()
        emb_diff = (hidden_states.cpu().float() - cpu_emb).abs()
        print(f"[DIFF] embedding: max={emb_diff.max():.6f}, mean={emb_diff.mean():.6f}")

    for layer_idx, layer in enumerate(model.layers):
        if layer_idx > 5:
            break  # Only trace first 6 layers

        hs_before = hidden_states.clone()

        # Run layer forward manually to capture intermediary states
        if residual is None:
            residual = hidden_states.clone()
            w = layer.input_layernorm._effective_weight() if hasattr(layer.input_layernorm, '_effective_weight') else layer.input_layernorm.weight
            rms_norm(hidden_states, hidden_states.contiguous(), w, layer.input_layernorm.eps)
        else:
            from engine.kernels.rms_norm import fused_add_rms_norm
            w = layer.input_layernorm._effective_weight() if hasattr(layer.input_layernorm, '_effective_weight') else layer.input_layernorm.weight
            fused_add_rms_norm(hidden_states, residual, w, layer.input_layernorm.eps)

        # Run attention
        if layer.layer_type == 'full_attention':
            attn_out = layer.self_attn(hidden_states, positions, S)
        else:
            attn_out = layer.linear_attn(hidden_states, positions, S)

        # The attn_out is after all_reduce (RowParallel o_proj does all_reduce)
        # But we need to compare with CPU reference which is also after o_proj (TP=1)

        if rank == 0 and layer_idx in cp_ref:
            cpu_val = cp_ref[layer_idx]
            # Note: attn_out at this point is pre-residual-add and pre-post_ln
            # We can't easily compare intermediate states since CPU reference stores
            # accumulated states differently. Let's instead check: is the model layer
            # producing reasonable output?

            gpu_norm = float(attn_out.norm())
            # CPU attn_out norms from reference (before residual add):
            cpu_layer = cp_ref[layer_idx]
            cpu_mlp_norm = cpu_layer['norm']
            print(f"\nLayer {layer_idx} ({layer.layer_type}): gpu_attn_out_norm={gpu_norm:.2f}")

        # Post-attention residual + norm (same as layer.forward)
        from engine.kernels.rms_norm import fused_add_rms_norm
        residual_init = residual
        fused_add_rms_norm(attn_out, residual, layer.post_attention_layernorm._effective_weight()
                           if hasattr(layer.post_attention_layernorm, '_effective_weight')
                           else layer.post_attention_layernorm.weight,
                           layer.post_attention_layernorm.eps)
        mlp_out = layer.mlp(attn_out)
        hidden_states = mlp_out
        # residual now holds accumulated: prev_residual + original_hs + attn_out
        # We don't need to track it further for this comparison

        if rank == 0 and layer_idx in cp_ref:
            gpu_mlp_norm = float(mlp_out.norm())
            cpu_mlp = cp_ref[layer_idx]
            cpu_mlp_norm_c = cpu_mlp['norm']
            print(f"  GPU mlp_out norm: {gpu_mlp_norm:.2f}, CPU mlp_out norm: {cpu_mlp_norm_c:.2f}")
            # These won't match directly because the CPU reference computes
            # residual = residual + attn_out differently (CPU uses accumulated)
            # But we can check if they're in the same ballpark

        # Important: after fused_add_rms_norm, residual is modified in-place
        # We need to ensure the layer.forward cycle is correctly tracked.
        # For the next iteration, we need the correct residual.
        # Let's use the layer's own forward but save the intermediate values.

    # Now run with the layer's own forward to get correct residual tracking
    # Reset
    del model
    model2 = QwenForCausalLMTP(cfg)
    model2 = model2.to(torch.bfloat16).to(f'cuda:{rank}')
    model2 = load_weights(model2, model_dir)
    model2.eval()

    hidden_states2 = model2.embed_tokens(input_ids)
    residual2 = None

    for layer_idx, layer in enumerate(model2.layers):
        if layer_idx > 5:
            break
        hidden_states2, residual2 = layer(hidden_states2, positions, S, residual2)

        if rank == 0 and layer_idx in cp_ref:
            gpu_mlp_norm = float(hidden_states2.norm())
            cpu_mlp = cp_ref[layer_idx]
            cpu_mlp_norm_c = cpu_mlp['norm']
            # The CPU reference stores hs (mlp_out, which is the hidden_state for next layer)
            # In our residual chain, hidden_states2 IS mlp_out (returned as first element)
            diff_pct = abs(gpu_mlp_norm - cpu_mlp_norm_c) / max(cpu_mlp_norm_c, 0.001) * 100
            print(f"\nLayer {layer_idx} ({layer.layer_type}): GPU={gpu_mlp_norm:.4f}, CPU={cpu_mlp_norm_c:.4f}, diff={diff_pct:.2f}%")

            # Try to compare actual values (after all_reduce if needed)
            gpu_hs_float = hidden_states2.cpu().float()
            cpu_hs = cpu_mlp['hs'].float()
            direct_diff = (gpu_hs_float - cpu_hs).abs()
            print(f"  Direct diff: max={direct_diff.max():.4f}, mean={direct_diff.mean():.4f}")

import torch.distributed as dist
dist.barrier()
if rank == 0:
    dist.destroy_process_group()
