#!/usr/bin/env python3
"""Isolate o_proj: compare RowParallelLinear output vs CPU reference."""
import os, sys, torch, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch.nn.functional as F

from engine.models.qwen import QwenTPConfig, QwenForCausalLMTP, load_weights
from engine.tp_layers.distributed import init_tp_distributed, get_tp_rank, get_tp_size
from engine.kernels.rms_norm import fused_add_rms_norm
from engine.kernels.rotary_embedding import rotary_embedding, make_cos_sin_cache
from engine.kernels.attention import flash_attn_varlen_func

init_tp_distributed()
rank = get_tp_rank()
tp_size = get_tp_size()
model_dir = os.environ['MODEL_DIR']
device = f'cuda:{rank}'

cfg = QwenTPConfig(model_dir)
with open(os.path.join(model_dir, 'config.json')) as f:
    raw = json.load(f)
tc = raw.get('text_config', raw)
eps = tc['rms_norm_eps']
head_dim = tc['head_dim']; num_heads = tc['num_attention_heads']
num_kv_heads = tc['num_key_value_heads']; hidden_size = tc['hidden_size']
rp = tc.get('rope_parameters', tc.get('rope_scaling', {})) or {}
rotary_dim = int(head_dim * rp.get('partial_rotary_factor', 1.0))
mrope_section = rp.get('mrope_section')
rope_theta = tc.get('rope_theta') or rp.get('rope_theta', 1000000.0)
max_pos = tc['max_position_embeddings']
hpr = cfg.heads_per_rank; kvpr = cfg.kv_heads_per_rank
full_q_size = num_heads * head_dim

model = QwenForCausalLMTP(cfg)
model = model.to(torch.bfloat16).to(device)
model = load_weights(model, model_dir)
model.eval()

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

prefix = 'model.language_model.layers.3.'
w_iln = load_cpu(prefix + 'input_layernorm.weight').float()
w_q = load_cpu(prefix + 'self_attn.q_proj.weight').float()
w_k = load_cpu(prefix + 'self_attn.k_proj.weight').float()
w_v = load_cpu(prefix + 'self_attn.v_proj.weight').float()
w_o = load_cpu(prefix + 'self_attn.o_proj.weight').float()
w_qn = load_cpu(prefix + 'self_attn.q_norm.weight').float()
w_kn = load_cpu(prefix + 'self_attn.k_norm.weight').float()

cos_sin_cpu = make_cos_sin_cache(
    max_pos, rotary_dim, rope_theta, dtype=torch.float32,
    mrope_section=mrope_section, mrope_interleaved=True, device='cpu')
positions_cpu = torch.arange(S, dtype=torch.int64)

def rms_norm_ref(x, w, eps_val=eps):
    rstd = 1.0 / torch.sqrt(x.float().pow(2).mean(-1, keepdim=True) + eps_val)
    return (x.float() * rstd * (1.0 + w.float())).to(torch.bfloat16)

with torch.inference_mode():
    hidden_states = model.embed_tokens(input_ids)
    residual = None
    for lx in range(3):
        hidden_states, residual = model.layers[lx](hidden_states, positions, S, residual)

    hs_in = hidden_states.clone()
    resid_in = residual.clone()
    layer3 = model.layers[3]
    fa = layer3.self_attn

    # Normed input
    hs_gpu = hs_in.clone()
    resid_gpu = resid_in.clone()
    fused_add_rms_norm(hs_gpu, resid_gpu, layer3.input_layernorm._effective_weight(),
                       layer3.input_layernorm.eps)

    # Q/K/V
    fa._ensure_cos_sin_gpu(device)
    q_gpu = fa.q_proj(hs_gpu); k_gpu = fa.k_proj(hs_gpu)
    v_gpu = fa.v_proj(hs_gpu); gate_gpu = fa.q_gate_proj(hs_gpu)
    q_gpu_h = q_gpu.view(B, S, hpr, head_dim); k_gpu_h = k_gpu.view(B, S, kvpr, head_dim)
    v_gpu_h = v_gpu.view(B, S, kvpr, head_dim); gate_gpu_h = gate_gpu.view(B, S, hpr, head_dim)
    q_gpu_n = fa.q_norm(q_gpu_h); k_gpu_n = fa.k_norm(k_gpu_h)

    num_tokens = B * S
    q_flat = q_gpu_n.reshape(num_tokens, hpr, head_dim).clone()
    k_flat = k_gpu_n.reshape(num_tokens, kvpr, head_dim).clone()
    v_flat = v_gpu_h.reshape(num_tokens, kvpr, head_dim).clone()
    q_rot = q_flat[..., :rotary_dim].contiguous().clone()
    k_rot = k_flat[..., :rotary_dim].contiguous().clone()
    rotary_embedding(positions, q_rot, k_rot, rotary_dim, fa._cos_sin_cache_gpu, is_neox=True)
    q_flat[..., :rotary_dim] = q_rot; k_flat[..., :rotary_dim] = k_rot
    cu = torch.tensor([0, num_tokens], dtype=torch.int32, device=device)
    attn_gpu = flash_attn_varlen_func(
        q_flat, k_flat, v_flat, cu, cu, num_tokens, num_tokens, causal=True,
        softmax_scale=head_dim ** -0.5)
    attn_flat_gpu = attn_gpu.reshape(B, S, hpr * head_dim)
    gate_flat_gpu = gate_gpu_h.reshape(B, S, hpr * head_dim)
    attn_gated_gpu = attn_flat_gpu * torch.sigmoid(gate_flat_gpu)

    # ===== o_proj comparison =====
    # Method 1: fa.o_proj (RowParallelLinear, may include all_reduce)
    out_via_fa = fa.o_proj(attn_gated_gpu)

    # Method 2: Manual matmul with o_proj.weight (BEFORE all_reduce)
    out_manual = F.linear(attn_gated_gpu, fa.o_proj.weight)

    if rank == 0:
        o_w_gpu = fa.o_proj.weight.data.cpu().float()
        o_w_cpu = w_o[:, :hpr * head_dim]  # rank 0 slice
        ow_diff = (o_w_gpu - o_w_cpu).abs()
        print(f"o_proj weight GPU shape={list(o_w_gpu.shape)} CPU slice shape={list(o_w_cpu.shape)}")
        print(f"  max diff={ow_diff.max():.6f} mean diff={ow_diff.mean():.6f}")
        print(f"  GPU norm={o_w_gpu.norm():.4f} CPU norm={o_w_cpu.norm():.4f}")
        print(f"  GPU first value: {o_w_gpu[0,0]:.6f}, CPU first value: {o_w_cpu[0,0]:.6f}")
        if o_w_gpu.shape != o_w_cpu.shape:
            print(f"  SHAPE MISMATCH! GPU={list(o_w_gpu.shape)} CPU={list(o_w_cpu.shape)}")
            # Print shapes to help debug
            print(f"  w_o full shape: {list(w_o.shape)}")
            print(f"  hpr*head_dim = {hpr*head_dim}")
            print(f"  hidden_size = {hidden_size}")

        # CPU: rank 0 partial output
        hs_cpu = hs_in.cpu().float()
        resid_cpu = resid_in.cpu().float()
        rsum = resid_cpu + hs_cpu
        rstd = 1.0 / torch.sqrt(rsum.pow(2).mean(-1, keepdim=True) + eps)
        hs_normed = (rsum * rstd * (1.0 + w_iln.float())).bfloat16()

        q_full = F.linear(hs_normed.float(), w_q)
        q_cpu_r0 = q_full[:, :, :hpr*head_dim].view(B, S, hpr, head_dim)
        gate_cpu = q_full[:, :, full_q_size:full_q_size+hpr*head_dim].view(B, S, hpr, head_dim)
        k_cpu_r0 = F.linear(hs_normed.float(), w_k)[:, :, :kvpr*head_dim].view(B, S, kvpr, head_dim)
        v_cpu_r0 = F.linear(hs_normed.float(), w_v)[:, :, :kvpr*head_dim].view(B, S, kvpr, head_dim)
        q_cpu_n = rms_norm_ref(q_cpu_r0.float(), w_qn)
        k_cpu_n = rms_norm_ref(k_cpu_r0.float(), w_kn)

        q_flat_c = q_cpu_n.float().reshape(num_tokens, hpr, head_dim).clone()
        k_flat_c = k_cpu_n.float().reshape(num_tokens, kvpr, head_dim).clone()
        q_rot_c = q_flat_c[..., :rotary_dim].contiguous().clone()
        k_rot_c = k_flat_c[..., :rotary_dim].contiguous().clone()
        rotary_embedding(positions_cpu, q_rot_c, k_rot_c, rotary_dim, cos_sin_cpu, is_neox=True)
        q_flat_c[..., :rotary_dim] = q_rot_c; k_flat_c[..., :rotary_dim] = k_rot_c

        v_cpu_f = v_cpu_r0.float().reshape(num_tokens, kvpr, head_dim)
        q_sdpa = q_flat_c.reshape(1, S, hpr, head_dim).transpose(1, 2).bfloat16()
        k_sdpa = k_flat_c.reshape(1, S, kvpr, head_dim).transpose(1, 2).bfloat16()
        v_sdpa = v_cpu_f.reshape(1, S, kvpr, head_dim).transpose(1, 2).bfloat16()
        gqa_f = hpr // kvpr
        if gqa_f > 1:
            k_sdpa = k_sdpa.repeat_interleave(gqa_f, dim=1)
            v_sdpa = v_sdpa.repeat_interleave(gqa_f, dim=1)
        attn_cpu_r0 = F.scaled_dot_product_attention(
            q_sdpa, k_sdpa, v_sdpa, is_causal=True, scale=head_dim ** -0.5)
        attn_cpu_r0 = attn_cpu_r0.transpose(1, 2).reshape(num_tokens, hpr, head_dim).float()
        gate_cpu_f = gate_cpu.float().reshape(B, S, hpr * head_dim)
        attn_gated_cpu = attn_cpu_r0.reshape(B, S, hpr * head_dim) * torch.sigmoid(gate_cpu_f)

        # CPU o_proj (rank 0 partial)
        out_cpu_r0 = F.linear(attn_gated_cpu, o_w_cpu)

        # ===== RESULTS =====
        print(f"\n=== o_proj comparison ===")
        print(f"  attn_gated GPU norm: {attn_gated_gpu.float().norm():.4f}")
        print(f"  attn_gated CPU norm: {attn_gated_cpu.float().norm():.4f}")
        ag_diff = (attn_gated_gpu.cpu().float() - attn_gated_cpu).abs()
        print(f"  attn_gated diff max={ag_diff.max():.6f}")

        print(f"\n  GPU via fa.o_proj:       {out_via_fa.float().norm():.4f}")
        print(f"  GPU via F.linear(w):     {out_manual.float().norm():.4f}")
        print(f"  CPU partial (rank 0):    {out_cpu_r0.float().norm():.4f}")

        diff_fa_vs_cpu = (out_via_fa.cpu().float() - out_cpu_r0).abs()
        diff_manual_vs_cpu = (out_manual.cpu().float() - out_cpu_r0).abs()
        print(f"  fa.o_proj vs CPU:        max={diff_fa_vs_cpu.max():.4f} mean={diff_fa_vs_cpu.mean():.4f}")
        print(f"  F.linear(w) vs CPU:      max={diff_manual_vs_cpu.max():.4f} mean={diff_manual_vs_cpu.mean():.4f}")

        # Check if fa.o_proj does all_reduce
        diff_fa_vs_manual = (out_via_fa.cpu().float() - out_manual.cpu().float()).abs()
        print(f"  fa.o_proj vs F.linear:   max={diff_fa_vs_manual.max():.4f} mean={diff_fa_vs_manual.mean():.4f}")
        if diff_fa_vs_manual.max() > 1.0:
            ratio_vs_ar = out_via_fa.float().norm() / (out_manual.float().norm() * tp_size + 1e-8)
            print(f"  (fa.o_proj includes all_reduce, 4*rank0 ratio={ratio_vs_ar:.4f})")

import torch.distributed as dist
dist.barrier()
if rank == 0:
    dist.destroy_process_group()
