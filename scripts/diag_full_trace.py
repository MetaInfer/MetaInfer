#!/usr/bin/env python3
"""Full layer-3 trace: identify where GPU diverges from CPU reference."""
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
w_pln = load_cpu(prefix + 'post_attention_layernorm.weight').float()
w_q = load_cpu(prefix + 'self_attn.q_proj.weight').float()
w_k = load_cpu(prefix + 'self_attn.k_proj.weight').float()
w_v = load_cpu(prefix + 'self_attn.v_proj.weight').float()
w_o = load_cpu(prefix + 'self_attn.o_proj.weight').float()
w_qn = load_cpu(prefix + 'self_attn.q_norm.weight').float()
w_kn = load_cpu(prefix + 'self_attn.k_norm.weight').float()
w_gate = load_cpu(prefix + 'mlp.gate_proj.weight').float()
w_up = load_cpu(prefix + 'mlp.up_proj.weight').float()
w_down = load_cpu(prefix + 'mlp.down_proj.weight').float()

cos_sin_cpu = make_cos_sin_cache(
    max_pos, rotary_dim, rope_theta, dtype=torch.float32,
    mrope_section=mrope_section, mrope_interleaved=True, device='cpu')
positions_cpu = torch.arange(S, dtype=torch.int64)

def rms_norm_ref(x, w, eps_val=eps):
    rstd = 1.0 / torch.sqrt(x.float().pow(2).mean(-1, keepdim=True) + eps_val)
    return (x.float() * rstd * (1.0 + w.float())).to(torch.bfloat16)

with torch.inference_mode():
    # Run layers 0-2
    hidden_states = model.embed_tokens(input_ids)
    residual = None
    for lx in range(3):
        hidden_states, residual = model.layers[lx](hidden_states, positions, S, residual)

    hs_in = hidden_states.clone()
    resid_in = residual.clone()

    # ===== GPU: Full layer 3 forward =====
    layer3 = model.layers[3]
    fa = layer3.self_attn

    # Step 1: fused_add_rms_norm (input norm)
    hs_gpu = hs_in.clone()
    resid_gpu = resid_in.clone()
    fused_add_rms_norm(hs_gpu, resid_gpu, layer3.input_layernorm._effective_weight(),
                       layer3.input_layernorm.eps)

    # Step 2: Q/K/V projections
    fa._ensure_cos_sin_gpu(device)
    q_gpu = fa.q_proj(hs_gpu)
    k_gpu = fa.k_proj(hs_gpu)
    v_gpu = fa.v_proj(hs_gpu)
    gate_gpu = fa.q_gate_proj(hs_gpu)

    q_gpu_h = q_gpu.view(B, S, hpr, head_dim)
    k_gpu_h = k_gpu.view(B, S, kvpr, head_dim)
    v_gpu_h = v_gpu.view(B, S, kvpr, head_dim)
    gate_gpu_h = gate_gpu.view(B, S, hpr, head_dim)

    # Step 3: Q/K norms
    q_gpu_n = fa.q_norm(q_gpu_h)
    k_gpu_n = fa.k_norm(k_gpu_h)

    # Step 4: RoPE
    num_tokens = B * S
    q_flat = q_gpu_n.reshape(num_tokens, hpr, head_dim).clone()
    k_flat = k_gpu_n.reshape(num_tokens, kvpr, head_dim).clone()
    v_flat = v_gpu_h.reshape(num_tokens, kvpr, head_dim).clone()
    q_rot = q_flat[..., :rotary_dim].contiguous().clone()
    k_rot = k_flat[..., :rotary_dim].contiguous().clone()
    rotary_embedding(positions, q_rot, k_rot, rotary_dim, fa._cos_sin_cache_gpu, is_neox=True)
    q_flat[..., :rotary_dim] = q_rot
    k_flat[..., :rotary_dim] = k_rot

    # Step 5: Attention
    cu = torch.tensor([0, num_tokens], dtype=torch.int32, device=device)
    attn_gpu = flash_attn_varlen_func(
        q_flat, k_flat, v_flat, cu, cu, num_tokens, num_tokens,
        causal=True, softmax_scale=head_dim ** -0.5)

    # Step 6: Gate + o_proj
    attn_flat = attn_gpu.reshape(B, S, hpr * head_dim)
    gate_flat = gate_gpu_h.reshape(B, S, hpr * head_dim)
    attn_gated = attn_flat * torch.sigmoid(gate_flat)
    out_gpu = fa.o_proj(attn_gated)

    if rank == 0:
        # ===== CPU reference for each step =====
        hs_cpu = hs_in.cpu().float()
        resid_cpu = resid_in.cpu().float()

        # Step 1: Normed input
        rsum = resid_cpu + hs_cpu
        rstd = 1.0 / torch.sqrt(rsum.pow(2).mean(-1, keepdim=True) + eps)
        hs_normed = (rsum * rstd * (1.0 + w_iln.float())).bfloat16()

        # Step 2: Q/K/V projections (rank 0 shard)
        q_full_cpu = F.linear(hs_normed.float(), w_q)
        q_cpu_r0 = q_full_cpu[:, :, :hpr * head_dim].view(B, S, hpr, head_dim)
        gate_cpu_full = q_full_cpu[:, :, full_q_size:]
        gate_cpu_r0 = gate_cpu_full[:, :, :hpr * head_dim].view(B, S, hpr, head_dim)
        k_full_cpu = F.linear(hs_normed.float(), w_k)
        k_cpu_r0 = k_full_cpu[:, :, :kvpr * head_dim].view(B, S, kvpr, head_dim)
        v_full_cpu = F.linear(hs_normed.float(), w_v)
        v_cpu_r0 = v_full_cpu[:, :, :kvpr * head_dim].view(B, S, kvpr, head_dim)

        # Step 3: Q/K norms
        q_cpu_n = rms_norm_ref(q_cpu_r0.float(), w_qn)
        k_cpu_n = rms_norm_ref(k_cpu_r0.float(), w_kn)

        # Step 4: RoPE
        q_cpu_flat = q_cpu_n.float().reshape(num_tokens, hpr, head_dim).clone()
        k_cpu_flat = k_cpu_n.float().reshape(num_tokens, kvpr, head_dim).clone()
        q_cpu_rot = q_cpu_flat[..., :rotary_dim].contiguous().clone()
        k_cpu_rot = k_cpu_flat[..., :rotary_dim].contiguous().clone()
        rotary_embedding(positions_cpu, q_cpu_rot, k_cpu_rot, rotary_dim, cos_sin_cpu, is_neox=True)
        q_cpu_flat[..., :rotary_dim] = q_cpu_rot
        k_cpu_flat[..., :rotary_dim] = k_cpu_rot

        # Step 5: Attention (SDPA on CPU, rank 0)
        v_cpu_flat = v_cpu_r0.float().reshape(num_tokens, kvpr, head_dim)
        q_sdpa = q_cpu_flat.reshape(1, S, hpr, head_dim).transpose(1, 2).bfloat16()
        k_sdpa = k_cpu_flat.reshape(1, S, kvpr, head_dim).transpose(1, 2).bfloat16()
        v_sdpa = v_cpu_flat.reshape(1, S, kvpr, head_dim).transpose(1, 2).bfloat16()
        gqa = hpr // kvpr
        if gqa > 1:
            k_sdpa = k_sdpa.repeat_interleave(gqa, dim=1)
            v_sdpa = v_sdpa.repeat_interleave(gqa, dim=1)
        attn_cpu_r0 = F.scaled_dot_product_attention(
            q_sdpa, k_sdpa, v_sdpa, is_causal=True, scale=head_dim ** -0.5)
        attn_cpu_r0 = attn_cpu_r0.transpose(1, 2).reshape(num_tokens, hpr, head_dim).float()

        # Step 6: Gate + o_proj (rank 0 partial)
        gate_cpu_r0_f = gate_cpu_r0.float().reshape(B, S, hpr * head_dim)
        attn_gated_cpu = attn_cpu_r0.reshape(B, S, hpr * head_dim) * torch.sigmoid(gate_cpu_r0_f)
        o_w_r0 = w_o[:, :hpr * head_dim]
        out_cpu_r0 = F.linear(attn_gated_cpu, o_w_r0)

        # ===== COMPARISONS =====
        print(f"{'='*60}")
        print(f"STEP-BY-STEP LAYER 3 TRACE (rank 0)")
        print(f"{'='*60}")

        def cmp(label, gpu, cpu):
            g = gpu.cpu().float()
            c = cpu.float()
            d = (g - c).abs()
            print(f"  {label}: GPU norm={g.norm():.4f} CPU norm={c.norm():.4f} "
                  f"max={d.max():.6f} mean={d.mean():.6f} ratio={g.norm()/(c.norm()+1e-8):.4f}")
            return d

        cmp("Rn", hs_gpu, hs_normed)
        cmp("Q_proj", q_gpu_h, q_cpu_r0)
        cmp("K_proj", k_gpu_h, k_cpu_r0)
        cmp("V_proj", v_gpu_h, v_cpu_r0)
        cmp("Q_norm", q_gpu_n, q_cpu_n)
        cmp("K_norm", k_gpu_n, k_cpu_n)
        cmp("Gate", gate_gpu_h, gate_cpu_r0)
        cmp("Q_RoPE", q_flat, q_cpu_flat)
        cmp("K_RoPE", k_flat, k_cpu_flat)
        cmp("Attn", attn_gpu, attn_cpu_r0)
        cmp("AttnGated", attn_gated, attn_gated_cpu)
        cmp("o_proj(r0)", out_gpu, out_cpu_r0)

        print(f"\n{'='*60}")
        print(f"Rank 0 partial output: GPU={out_gpu.float().norm():.4f} CPU={out_cpu_r0.float().norm():.4f}")

import torch.distributed as dist
dist.barrier()
if rank == 0:
    dist.destroy_process_group()
