#!/usr/bin/env python3
"""Trace real model: compare GPU layer-3 FullAttention output vs CPU reference
using the ACTUAL hidden states from layers 0-2 of the GPU model."""
import os, sys, torch, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch.nn.functional as F

from engine.models.qwen import QwenTPConfig, QwenForCausalLMTP, load_weights
from engine.tp_layers.distributed import init_tp_distributed, get_tp_rank, get_tp_size
from engine.kernels.rotary_embedding import rotary_embedding, make_cos_sin_cache
from engine.kernels.attention import flash_attn_varlen_func

init_tp_distributed()
rank = get_tp_rank()
tp_size = get_tp_size()
model_dir = os.environ['MODEL_DIR']
device = f'cuda:{rank}'

# Config
cfg = QwenTPConfig(model_dir)
with open(os.path.join(model_dir, 'config.json')) as f:
    raw = json.load(f)
tc = raw.get('text_config', raw)
eps = tc['rms_norm_eps']
head_dim = tc['head_dim']
num_heads = tc['num_attention_heads']
num_kv_heads = tc['num_key_value_heads']
hidden_size = tc['hidden_size']
rp = tc.get('rope_parameters', tc.get('rope_scaling', {})) or {}
rotary_dim = int(head_dim * rp.get('partial_rotary_factor', 1.0))
mrope_section = rp.get('mrope_section')
rope_theta = tc.get('rope_theta') or rp.get('rope_theta', 1000000.0)
max_pos = tc['max_position_embeddings']
hpr = cfg.heads_per_rank
kvpr = cfg.kv_heads_per_rank
full_q_size = num_heads * head_dim

# Load model
model = QwenForCausalLMTP(cfg)
model = model.to(torch.bfloat16).to(device)
model = load_weights(model, model_dir)
model.eval()

# Input
tokens = [108618, 102066, 137351, 105017, 100462, 106808, 103105]
input_ids = torch.tensor([tokens], dtype=torch.long, device=device)
B, S = 1, len(tokens)
positions = torch.arange(S, dtype=torch.int64, device=device)

# CPU weights
from safetensors import safe_open
with open(os.path.join(model_dir, 'model.safetensors.index.json')) as f:
    idx = json.load(f)
wm = idx['weight_map']

def load_cpu(key):
    fname = wm[key]
    fpath = os.path.join(model_dir, fname)
    with safe_open(fpath, framework='pt', device='cpu') as sf:
        return sf.get_tensor(key)

w_q = load_cpu('model.language_model.layers.3.self_attn.q_proj.weight').float()
w_k = load_cpu('model.language_model.layers.3.self_attn.k_proj.weight').float()
w_v = load_cpu('model.language_model.layers.3.self_attn.v_proj.weight').float()
w_o = load_cpu('model.language_model.layers.3.self_attn.o_proj.weight').float()
w_qn = load_cpu('model.language_model.layers.3.self_attn.q_norm.weight').float()
w_kn = load_cpu('model.language_model.layers.3.self_attn.k_norm.weight').float()
w_iln = load_cpu('model.language_model.layers.3.input_layernorm.weight').float()
w_pln = load_cpu('model.language_model.layers.3.post_attention_layernorm.weight').float()
w_gate = load_cpu('model.language_model.layers.3.mlp.gate_proj.weight').float()
w_up = load_cpu('model.language_model.layers.3.mlp.up_proj.weight').float()
w_down = load_cpu('model.language_model.layers.3.mlp.down_proj.weight').float()

cos_sin_cpu = make_cos_sin_cache(
    max_pos, rotary_dim, rope_theta, dtype=torch.float32,
    mrope_section=mrope_section, mrope_interleaved=True, device='cpu')
positions_cpu = torch.arange(S, dtype=torch.int64)

def rms_norm_ref(x, w, eps=eps):
    rstd = 1.0 / torch.sqrt(x.float().pow(2).mean(-1, keepdim=True) + eps)
    return (x.float() * rstd * (1.0 + w.float())).to(x.dtype)

with torch.inference_mode():
    # Run layers 0-2 on GPU
    hidden_states = model.embed_tokens(input_ids)
    residual = None
    for lx in range(3):
        hidden_states, residual = model.layers[lx](hidden_states, positions, S, residual)

    # Save GPU layer 3 input
    hs_in_gpu = hidden_states.clone()
    resid_in_gpu = residual.clone()

    # GPU Layer 3 FullAttention (replicate what the decoder layer does)
    layer3 = model.layers[3]
    fa = layer3.self_attn

    # fused_add_rms_norm for input
    from engine.kernels.rms_norm import fused_add_rms_norm
    hs_work = hs_in_gpu.clone()
    fused_add_rms_norm(hs_work, resid_in_gpu,
                       layer3.input_layernorm._effective_weight(), layer3.input_layernorm.eps)
    # hs_work = rms_norm(hs_in + resid_in), resid_in_gpu = hs_in + resid_in

    # GPU FullAttention
    fa._ensure_cos_sin_gpu(device)
    q_gpu = fa.q_proj(hs_work)
    k_gpu = fa.k_proj(hs_work)
    v_gpu = fa.v_proj(hs_work)
    gate_gpu = fa.q_gate_proj(hs_work)

    q_gpu = q_gpu.view(B, S, hpr, head_dim)
    k_gpu = k_gpu.view(B, S, kvpr, head_dim)
    v_gpu = v_gpu.view(B, S, kvpr, head_dim)
    gate_gpu = gate_gpu.view(B, S, hpr, head_dim)

    q_gpu_normed = fa.q_norm(q_gpu)
    k_gpu_normed = fa.k_norm(k_gpu)

    q_flat = q_gpu_normed.reshape(-1, hpr, head_dim).clone()
    k_flat = k_gpu_normed.reshape(-1, kvpr, head_dim).clone()
    v_flat = v_gpu.reshape(-1, kvpr, head_dim).clone()

    q_rot = q_flat[..., :rotary_dim].contiguous().clone()
    k_rot = k_flat[..., :rotary_dim].contiguous().clone()
    rotary_embedding(positions, q_rot, k_rot, rotary_dim, fa._cos_sin_cache_gpu, is_neox=True)
    q_flat[..., :rotary_dim] = q_rot
    k_flat[..., :rotary_dim] = k_rot

    cu = torch.tensor([0, B*S], dtype=torch.int32, device=device)
    attn_gpu = flash_attn_varlen_func(
        q_flat, k_flat, v_flat, cu, cu, B*S, B*S,
        causal=True, softmax_scale=head_dim ** -0.5)

    attn_flat = attn_gpu.reshape(B, S, hpr * head_dim)
    gate_flat = gate_gpu.reshape(B, S, hpr * head_dim)
    attn_gated = attn_flat * torch.sigmoid(gate_flat)
    out_gpu = fa.o_proj(attn_gated)

    # GPU post-attention RMS norm + MLP
    attn_out_copy = out_gpu.clone()
    resid_copy = resid_in_gpu.clone()
    fused_add_rms_norm(attn_out_copy, resid_copy,
                       layer3.post_attention_layernorm._effective_weight(),
                       layer3.post_attention_layernorm.eps)
    mlp_out_gpu = layer3.mlp(attn_out_copy)

    if rank == 0:
        print(f"GPU layer 3 FullAttention output norm: {out_gpu.float().norm():.4f}")
        print(f"GPU layer 3 MLP output norm: {mlp_out_gpu.float().norm():.4f}")

        # ---- CPU reference using GPU's hs_in and resid_in ----
        hs_cpu_in = hs_in_gpu.cpu().float()
        resid_cpu_in = resid_in_gpu.cpu().float()

        # Step 1: fused_add_rms_norm
        # GPU does: residual += hs, then hs = rms_norm(residual) * weight
        # So residual after = hs_in + resid_in
        # And hs_normed = rms_norm(hs_in + resid_in) * (1 + w_iln)
        resid_expected = resid_cpu_in + hs_cpu_in
        hs_normed_cpu = rms_norm_ref(resid_expected, w_iln)

        # Check normed input
        hs_work_cpu = hs_work.cpu().float()  # GPU's normed input
        diff_norm = (hs_work_cpu - hs_normed_cpu).abs()
        print(f"\n=== Normed input ===")
        print(f"  GPU norm={hs_work_cpu.norm():.4f} CPU norm={hs_normed_cpu.norm():.4f}")
        print(f"  Diff max={diff_norm.max():.6f} mean={diff_norm.mean():.6f}")

        # Step 2: Q/K/V/Gate projections (full, not TP)
        q_full_cpu = F.linear(hs_normed_cpu, w_q)  # [1,7,12288]
        q_cpu = q_full_cpu[:, :, :full_q_size].view(B, S, num_heads, head_dim)
        gate_cpu = q_full_cpu[:, :, full_q_size:].view(B, S, num_heads, head_dim)
        k_cpu = F.linear(hs_normed_cpu, w_k).view(B, S, num_kv_heads, head_dim)
        v_cpu = F.linear(hs_normed_cpu, w_v).view(B, S, num_kv_heads, head_dim)

        # Compare rank 0's Q/K/V projections
        q_cpu_r0 = q_cpu[:, :, :hpr, :].float()
        q_gpu_r0 = q_gpu.cpu().float()
        q_diff = (q_gpu_r0 - q_cpu_r0).abs()
        print(f"\n=== Q projection (rank 0 heads 0:{hpr}) ===")
        print(f"  GPU norm={q_gpu_r0.norm():.4f} CPU norm={q_cpu_r0.norm():.4f}")
        print(f"  Diff max={q_diff.max():.6f} mean={q_diff.mean():.6f}")

        # Compare Q/K norms
        q_cpu_normed = rms_norm_ref(q_cpu.float(), w_qn)
        k_cpu_normed = rms_norm_ref(k_cpu.float(), w_kn)
        q_cpu_n_r0 = q_cpu_normed[:, :, :hpr, :]
        k_cpu_n_r0 = k_cpu_normed[:, :, rank:rank+1, :]
        qn_diff = (q_gpu_normed.cpu().float() - q_cpu_n_r0).abs()
        kn_diff = (k_gpu_normed.cpu().float() - k_cpu_n_r0).abs()
        print(f"\n=== Q norm ===")
        print(f"  GPU norm={q_gpu_normed.float().norm():.4f} CPU norm={q_cpu_n_r0.norm():.4f}")
        print(f"  Diff max={qn_diff.max():.6f} mean={qn_diff.mean():.6f}")
        print(f"=== K norm ===")
        print(f"  GPU norm={k_gpu_normed.float().norm():.4f} CPU norm={k_cpu_n_r0.norm():.4f}")
        print(f"  Diff max={kn_diff.max():.6f} mean={kn_diff.mean():.6f}")

        # Compare full o_proj output (after all_reduce on GPU)
        # CPU: compute full o_proj
        q_cpu_r0_ro = q_cpu_n_r0.clone().reshape(num_tokens, hpr, head_dim)
        k_cpu_r0_ro = k_cpu_n_r0.clone().reshape(num_tokens, kvpr, head_dim)

        q_rot_c = q_cpu_r0_ro[..., :rotary_dim].contiguous().clone()
        k_rot_c = k_cpu_r0_ro[..., :rotary_dim].contiguous().clone()
        rotary_embedding(positions_cpu, q_rot_c, k_rot_c, rotary_dim, cos_sin_cpu, is_neox=True)
        q_cpu_r0_ro[..., :rotary_dim] = q_rot_c
        k_cpu_r0_ro[..., :rotary_dim] = k_rot_c

        q_sdpa_c = q_cpu_r0_ro.reshape(1, S, hpr, head_dim).transpose(1, 2).bfloat16()
        k_sdpa_c = k_cpu_r0_ro.reshape(1, S, kvpr, head_dim).transpose(1, 2).bfloat16()
        v_cpu_r0 = v_cpu[:, :, rank:rank+1, :].reshape(num_tokens, kvpr, head_dim)
        v_sdpa_c = v_cpu_r0.reshape(1, S, kvpr, head_dim).transpose(1, 2).bfloat16()

        gqa = hpr // kvpr
        k_sdpa_c = k_sdpa_c.repeat_interleave(gqa, dim=1)
        v_sdpa_c = v_sdpa_c.repeat_interleave(gqa, dim=1)

        attn_cpu_r0 = F.scaled_dot_product_attention(
            q_sdpa_c, k_sdpa_c, v_sdpa_c, is_causal=True, scale=head_dim ** -0.5)
        attn_cpu_r0 = attn_cpu_r0.transpose(1, 2).reshape(num_tokens, hpr, head_dim).float()

        # Compare attention
        attn_diff = (attn_gpu.cpu().float() - attn_cpu_r0).abs()
        print(f"\n=== Attention (rank 0) ===")
        print(f"  GPU norm={attn_gpu.float().norm():.4f} CPU norm={attn_cpu_r0.norm():.4f}")
        print(f"  Diff max={attn_diff.max():.6f} mean={attn_diff.mean():.6f}")

        # Now compute the FULL o_proj output (all ranks)
        # CPU: compute full attention for all heads, then full o_proj
        q_cpu_full = q_cpu_normed.reshape(num_tokens, num_heads, head_dim).clone()
        k_cpu_full = k_cpu_normed.reshape(num_tokens, num_kv_heads, head_dim).clone()

        q_rot_full = q_cpu_full[..., :rotary_dim].contiguous().clone()
        k_rot_full = k_cpu_full[..., :rotary_dim].contiguous().clone()
        rotary_embedding(positions_cpu, q_rot_full, k_rot_full, rotary_dim, cos_sin_cpu, is_neox=True)
        q_cpu_full[..., :rotary_dim] = q_rot_full
        k_cpu_full[..., :rotary_dim] = k_rot_full

        q_sdpa_f = q_cpu_full.reshape(1, S, num_heads, head_dim).transpose(1, 2).bfloat16()
        k_sdpa_f = k_cpu_full.reshape(1, S, num_kv_heads, head_dim).transpose(1, 2).bfloat16()
        v_cpu_f = v_cpu.reshape(num_tokens, num_kv_heads, head_dim)
        v_sdpa_f = v_cpu_f.reshape(1, S, num_kv_heads, head_dim).transpose(1, 2).bfloat16()

        gqa_f = num_heads // num_kv_heads
        k_sdpa_f = k_sdpa_f.repeat_interleave(gqa_f, dim=1)
        v_sdpa_f = v_sdpa_f.repeat_interleave(gqa_f, dim=1)

        attn_cpu_full = F.scaled_dot_product_attention(
            q_sdpa_f, k_sdpa_f, v_sdpa_f, is_causal=True, scale=head_dim ** -0.5)
        attn_cpu_full = attn_cpu_full.transpose(1, 2).reshape(num_tokens, num_heads, head_dim).float()

        gate_cpu_full = gate_cpu.float().reshape(num_tokens, num_heads, head_dim)
        attn_gated_cpu = attn_cpu_full.reshape(B, S, num_heads * head_dim) * torch.sigmoid(gate_cpu_full.reshape(B, S, num_heads * head_dim))
        out_cpu_full = F.linear(attn_gated_cpu, w_o)

        # Compare with GPU output (which includes all_reduce)
        diff_full = (out_gpu.cpu().float() - out_cpu_full).abs()
        ratio_full = out_gpu.float().norm() / (out_cpu_full.float().norm() + 1e-8)
        print(f"\n=== FullAttention output (all ranks, after all_reduce) ===")
        print(f"  GPU norm={out_gpu.float().norm():.4f} CPU norm={out_cpu_full.float().norm():.4f}")
        print(f"  Diff max={diff_full.max():.6f} mean={diff_full.mean():.6f}")
        print(f"  Ratio GPU/CPU: {ratio_full:.4f}")

        # Check weight values on GPU vs CPU
        print(f"\n=== Weight verification ===")
        gpu_qw = fa.q_proj.weight.data.cpu().float()  # [1536, 5120]
        cpu_qw_r0 = w_q[:full_q_size][:hpr*head_dim, :]  # first 1536 rows of Q part
        qw_diff = (gpu_qw - cpu_qw_r0).abs()
        print(f"  q_proj weight: max diff={qw_diff.max():.6f}")

        gpu_kw = fa.k_proj.weight.data.cpu().float()  # [256, 5120]
        cpu_kw_r0 = w_k[:kvpr*head_dim, :]
        kw_diff = (gpu_kw - cpu_kw_r0).abs()
        print(f"  k_proj weight: max diff={kw_diff.max():.6f}")

        gpu_ow = fa.o_proj.weight.data.cpu().float()  # [5120, 1536]
        cpu_ow_r0 = w_o[:, :hpr*head_dim]
        ow_diff = (gpu_ow - cpu_ow_r0).abs()
        print(f"  o_proj weight: max diff={ow_diff.max():.6f}")

        gpu_qn = fa.q_norm.weight.data.cpu().float()
        qn_diff = (gpu_qn - w_qn).abs()
        print(f"  q_norm weight: max diff={qn_diff.max():.6f}")

        # Compare MLP output on CPU
        attn_out_cpu = out_cpu_full.float()
        resid_cpu_new = resid_cpu_in + hs_cpu_in + attn_out_cpu  # residual after both fused_add_rms_norm ops
        attn_normed_cpu = rms_norm_ref(resid_cpu_new, w_pln)

        mlp_hidden_cpu = F.linear(attn_normed_cpu, w_gate)
        mlp_gate_cpu = F.linear(attn_normed_cpu, w_up)
        mlp_cpu = F.linear(F.silu(mlp_hidden_cpu) * mlp_gate_cpu, w_down)

        mlp_diff = (mlp_out_gpu.cpu().float() - mlp_cpu).abs()
        print(f"\n=== MLP output ===")
        print(f"  GPU norm={mlp_out_gpu.float().norm():.4f} CPU norm={mlp_cpu.float().norm():.4f}")
        print(f"  Diff max={mlp_diff.max():.6f} mean={mlp_diff.mean():.6f}")
        mlp_ratio = mlp_out_gpu.float().norm() / (mlp_cpu.float().norm() + 1e-8)
        print(f"  Ratio GPU/CPU: {mlp_ratio:.4f}")

import torch.distributed as dist
dist.barrier()
if rank == 0:
    dist.destroy_process_group()
