#!/usr/bin/env python3
"""Compare single GatedDeltaNet layer output: our model vs manual computation using same weights.

If they match ➔ bug is elsewhere (later layers, or residual chain).
If they don't match ➔ bug is in our layer implementation.
"""
import os, sys, torch, json, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch.nn.functional as F

from engine.models.qwen import QwenTPConfig, QwenForCausalLMTP, load_weights
from engine.tp_layers.distributed import init_tp_distributed, get_tp_rank
from engine.kernels.rms_norm import rms_norm

init_tp_distributed()
rank = get_tp_rank()
model_dir = os.environ['MODEL_DIR']

cfg = QwenTPConfig(model_dir)
model = QwenForCausalLMTP(cfg)
model = model.to(torch.bfloat16).to(f'cuda:{rank}')
model = load_weights(model, model_dir)
model.eval()

device = f'cuda:{rank}'

# ============================================================
# Step 1: Extract weights for layer 0's GatedDeltaNet
# ============================================================
layer0 = model.layers[0]
gdn = layer0.linear_attn

w_in_proj_qkv = gdn.in_proj_qkv.weight.data.clone()
w_conv1d = gdn.conv1d_weight.data.clone()
b_conv1d = gdn.conv1d_bias.data.clone()
w_in_proj_a = gdn.in_proj_a.weight.data.clone()
w_in_proj_b = gdn.in_proj_b.weight.data.clone()
A_log = gdn.A_log.data.clone()
dt_bias = gdn.dt_bias.data.clone()
w_in_proj_z = gdn.in_proj_z.weight.data.clone()
w_norm = gdn.norm.weight.data.clone()
w_out_proj = gdn.out_proj.weight.data.clone()

input_ln_w = layer0.input_layernorm._effective_weight().clone()

# ============================================================
# Step 2: Run a simple input through both
# ============================================================
prompt_ids = [108618, 102066, 137351]  # 苏州园林的特点是
input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)

with torch.inference_mode():
    # Get the same input hidden states
    hs_full = model.embed_tokens(input_ids)  # [1, 3, 5120]

    # Apply input layernorm to match what layer would receive
    hs_backup = hs_full.clone()
    rms_norm(hs_full, hs_full.contiguous(), input_ln_w, layer0.input_layernorm.eps)
    hs = hs_full  # This is what the layer actually receives

    # ---- Run through our layer (the actual model) ----
    attn_out_model, residual_model = layer0(hs_backup.clone(), torch.arange(3, dtype=torch.int64, device=device), 3, None)
    # layer0.forward modifies its inputs. Let's also get the attention output directly.

    # ---- Manual computation of the same layer ----
    B, T, H = hs.shape  # 1, 3, 5120

    # 1. in_proj_qkv
    mixed_qkv = F.linear(hs, w_in_proj_qkv)  # [1, 3, 2560]

    # 2. Causal conv1d
    mixed_qkv_t = mixed_qkv.transpose(1, 2)
    mixed_qkv_pad = F.pad(mixed_qkv_t, (3, 0))  # kernel=4 → pad 3 on left
    conv_out = F.conv1d(mixed_qkv_pad, w_conv1d, b_conv1d, groups=mixed_qkv_t.shape[1])
    conv_out = F.silu(conv_out).transpose(1, 2)  # [1, 3, 2560]

    # 3. Split Q, K, V
    ksl = cfg.k_heads_per_rank       # 4
    vsl = cfg.v_heads_per_rank       # 12
    kdim = cfg.linear_key_head_dim    # 128
    vdim = cfg.linear_value_head_dim  # 128
    q = conv_out[:, :, :ksl*kdim].view(B, T, ksl, kdim)
    k = conv_out[:, :, ksl*kdim:2*ksl*kdim].view(B, T, ksl, kdim)
    v = conv_out[:, :, 2*ksl*kdim:].view(B, T, vsl, vdim)

    # 4. in_proj_a, in_proj_b
    a = F.linear(hs, w_in_proj_a)  # [1, 3, 48]
    b = F.linear(hs, w_in_proj_b)  # [1, 3, 48]

    vr = slice(rank*vsl, (rank+1)*vsl)
    a_local = a[:, :, vr]
    b_local = b[:, :, vr]

    # 5. Gate and beta
    g = -torch.exp(A_log) * F.softplus(a_local + dt_bias)
    beta = torch.sigmoid(b_local)

    # 6. Q/K L2 norm
    q_norm = F.normalize(q.float(), p=2, dim=-1).to(hs.dtype)
    k_norm = F.normalize(k.float(), p=2, dim=-1).to(hs.dtype)

    if vsl > ksl:
        q_norm = q_norm.repeat_interleave(vsl // ksl, dim=2)
        k_norm = k_norm.repeat_interleave(vsl // ksl, dim=2)

    q_scale = 1.0 / math.sqrt(kdim)
    q_norm = q_norm * q_scale

    # 7. Recurrence
    state = torch.zeros(B, vsl, kdim, vdim, dtype=torch.float32, device=device)
    core_out = torch.zeros(B, T, vsl, vdim, dtype=hs.dtype, device=device)

    for t in range(T):
        g_t = g[:, t, :]
        k_t = k_norm[:, t, :, :]
        v_t = v[:, t, :, :]
        q_t = q_norm[:, t, :, :]
        beta_t = beta[:, t, :]

        state = state * torch.exp(g_t.float())[:, :, None, None]
        kv_mem = torch.sum(state * k_t.float()[:, :, :, None], dim=-2)
        delta = (v_t.float() - kv_mem) * beta_t.float()[:, :, None]
        state = state + k_t.float()[:, :, :, None] * delta[:, :, None, :]
        o_t = torch.sum(state * q_t.float()[:, :, :, None], dim=-2)
        core_out[:, t, :, :] = o_t.to(hs.dtype)

    # 8. z + norm
    z = F.linear(hs, w_in_proj_z).view(B, T, vsl, vdim)
    core_flat = core_out.reshape(-1, vdim)
    z_flat = z.reshape(-1, vdim)

    rstd = 1.0 / torch.sqrt(core_flat.float().pow(2).mean(-1, keepdim=True) + 1e-6)
    x_norm = core_flat.float() * rstd
    w_eff = w_norm
    if w_eff.shape[0] != core_flat.shape[-1]:
        w_eff = w_eff.repeat(core_flat.shape[-1] // w_eff.shape[0])
    gated_out = (x_norm.to(hs.dtype) * w_eff * F.silu(z_flat)).view(B, T, vsl * vdim)

    # 9. out_proj (RowParallel)
    from engine.tp_layers.distributed import all_reduce_sum
    manual_out = all_reduce_sum(F.linear(gated_out, w_out_proj))

    # ---- Also get model's GatedDeltaNet output directly ----
    # Re-run with a fresh clone
    hs_fresh = hs_backup.clone()
    rms_norm(hs_fresh, hs_fresh.contiguous(), input_ln_w, layer0.input_layernorm.eps)
    model_gdn_out = gdn(hs_fresh, torch.arange(3, dtype=torch.int64, device=device), 3)

    if rank == 0:
        print(f"Manual GDN output: norm={manual_out.norm():.4f}, min={manual_out.min():.4f}, max={manual_out.max():.4f}")
        print(f"Model  GDN output: norm={model_gdn_out.norm():.4f}, min={model_gdn_out.min():.4f}, max={model_gdn_out.max():.4f}")
        diff = (manual_out - model_gdn_out).abs()
        print(f"Difference: max={diff.max():.8f}, mean={diff.mean():.8f}")
        print(f"Match? {'YES' if diff.max() < 1e-3 else 'NO - DIFFERENCE FOUND!'}")

        if diff.max() < 1e-3:
            print("\n✓ Layer 0 GatedDeltaNet matches manual computation.")
            print("  Bug is NOT in the layer itself — checking higher layers or residual chain.")
        else:
            print("\n✗ Layer 0 GatedDeltaNet differs from manual computation!")
            print(f"  Max diff: {diff.max():.8f}")
            # Find where the difference is largest
            max_idx = diff.argmax()
            max_pos = torch.unravel_index(max_idx, diff.shape)
            print(f"  Position of max diff: {max_pos}")
            print(f"  Manual value: {manual_out[max_pos].item():.6f}")
            print(f"  Model  value: {model_gdn_out[max_pos].item():.6f}")

import torch.distributed as dist
dist.barrier()
if rank == 0:
    dist.destroy_process_group()
