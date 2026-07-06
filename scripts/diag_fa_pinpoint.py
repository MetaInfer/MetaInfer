#!/usr/bin/env python3
"""Pinpoint the FullAttention TP=4 divergence source.
Runs on single GPU (TP=1 simulated), compares with CPU reference step by step.
"""
import os, sys, torch, json, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch.nn.functional as F

os.environ.setdefault('LOCAL_RANK', '0')
os.environ.setdefault('RANK', '0')
os.environ.setdefault('WORLD_SIZE', '1')
os.environ.setdefault('MASTER_ADDR', '127.0.0.1')
os.environ.setdefault('MASTER_PORT', '29650')

model_dir = os.environ['MODEL_DIR']
device = 'cuda:0'
dtype = torch.bfloat16

# Load config
with open(os.path.join(model_dir, 'config.json')) as f:
    raw = json.load(f)
tc = raw.get('text_config', raw)
head_dim = tc['head_dim']
num_heads = tc['num_attention_heads']
num_kv_heads = tc['num_key_value_heads']
hidden_size = tc['hidden_size']
eps = tc['rms_norm_eps']
rp = tc.get('rope_parameters', tc.get('rope_scaling', {})) or {}
rotary_dim = int(head_dim * rp.get('partial_rotary_factor', 1.0))
mrope_section = rp.get('mrope_section')
rope_theta = tc.get('rope_theta', 1000000.0)
max_pos = tc['max_position_embeddings']

print(f"Config: hidden={hidden_size}, heads={num_heads}/{num_kv_heads}, hdim={head_dim}, rotary_dim={rotary_dim}")
print(f"MRoPE: section={mrope_section}, interleaved={rp.get('mrope_interleaved')}, theta={rope_theta}")

# Load layer 3 weights
from safetensors import safe_open
with open(os.path.join(model_dir, 'model.safetensors.index.json')) as f:
    idx = json.load(f)
wm = idx['weight_map']

def load_hf(key):
    fname = wm[key]
    fpath = os.path.join(model_dir, fname)
    with safe_open(fpath, framework='pt', device='cpu') as sf:
        return sf.get_tensor(key)

prefix = 'model.language_model.layers.3.'
w_input_ln = load_hf(prefix + 'input_layernorm.weight').float()
w_q = load_hf(prefix + 'self_attn.q_proj.weight').float()  # [12288, 5120]
w_k = load_hf(prefix + 'self_attn.k_proj.weight').float()  # [1024, 5120]
w_v = load_hf(prefix + 'self_attn.v_proj.weight').float()  # [1024, 5120]
w_o = load_hf(prefix + 'self_attn.o_proj.weight').float()  # [5120, 6144]
w_qn = load_hf(prefix + 'self_attn.q_norm.weight').float()  # [256]
w_kn = load_hf(prefix + 'self_attn.k_norm.weight').float()  # [256]

# Load embedding
w_embed = load_hf('model.language_model.embed_tokens.weight').float()

# Build MRoPE cache (CPU)
from engine.kernels.rotary_embedding import make_cos_sin_cache
cos_sin_cpu = make_cos_sin_cache(max_pos, rotary_dim, rope_theta,
                                  dtype=torch.float32,
                                  mrope_section=mrope_section,
                                  mrope_interleaved=True)

def cpu_rms_norm(x, w):
    rstd = 1.0 / torch.sqrt(x.float().pow(2).mean(-1, keepdim=True) + eps)
    return (x.float() * rstd * (1.0 + w.float())).to(x.dtype)

def cpu_apply_mrope(q, k, positions):
    """Apply MRoPE (NeoX style) in-place."""
    B, S = q.shape[0], q.shape[1]
    cos = cos_sin_cpu[positions, :rotary_dim//2].float()
    sin = cos_sin_cpu[positions, rotary_dim//2:].float()
    cos = cos.view(1, S, 1, rotary_dim//2)
    sin = sin.view(1, S, 1, rotary_dim//2)
    # NeoX: duplicate cos/sin for both halves
    cos_dup = torch.cat([cos, cos], dim=-1)
    sin_dup = torch.cat([sin, sin], dim=-1)

    q_rot = q[..., :rotary_dim].float()
    half = rotary_dim // 2
    q1, q2 = q_rot[..., :half], q_rot[..., half:]
    q[..., :rotary_dim] = (q_rot * cos_dup + torch.cat([-q2, q1], dim=-1) * sin_dup).to(q.dtype)

    k_rot = k[..., :rotary_dim].float()
    k1, k2 = k_rot[..., :half], k_rot[..., half:]
    k[..., :rotary_dim] = (k_rot * cos_dup + torch.cat([-k2, k1], dim=-1) * sin_dup).to(k.dtype)

# Test inputs
tokens = [108618, 102066, 137351, 105017, 100462, 106808, 103105]
input_ids = torch.tensor([tokens], dtype=torch.long)
B, S = 1, len(tokens)
positions = torch.arange(S, dtype=torch.int64)

# Embed
emb = F.embedding(input_ids, w_embed)
print(f"\n=== Step 1: Input norm ===")

# Simulate what input looks like at layer 3 (after 3 correct GatedDeltaNet layers)
# We need the actual residual state. Let's compute layers 0-2 on CPU first.
def cpu_forward_layer0_2():
    """Compute layers 0-2 on CPU (GatedDeltaNet only)."""
    prefix0 = 'model.language_model.layers.0.'
    prefix1 = 'model.language_model.layers.1.'
    prefix2 = 'model.language_model.layers.2.'

    # Layer 0
    w_in0 = load_hf(prefix0 + 'input_layernorm.weight').float()
    w_po0 = load_hf(prefix0 + 'post_attention_layernorm.weight').float()
    w_in_qkv0 = load_hf(prefix0 + 'linear_attn.in_proj_qkv.weight').float()
    w_conv0 = load_hf(prefix0 + 'linear_attn.conv1d.weight').float()
    w_a0 = load_hf(prefix0 + 'linear_attn.in_proj_a.weight').float()
    w_b0 = load_hf(prefix0 + 'linear_attn.in_proj_b.weight').float()
    A_log0 = load_hf(prefix0 + 'linear_attn.A_log').float()
    dt_b0 = load_hf(prefix0 + 'linear_attn.dt_bias').float()
    w_z0 = load_hf(prefix0 + 'linear_attn.in_proj_z.weight').float()
    w_nm0 = load_hf(prefix0 + 'linear_attn.norm.weight').float()
    w_ou0 = load_hf(prefix0 + 'linear_attn.out_proj.weight').float()
    w_g0 = load_hf(prefix0 + 'mlp.gate_proj.weight').float()
    w_u0 = load_hf(prefix0 + 'mlp.up_proj.weight').float()
    w_d0 = load_hf(prefix0 + 'mlp.down_proj.weight').float()

    lk_heads = tc['linear_num_key_heads']
    lv_heads = tc['linear_num_value_heads']
    lk_dim = tc['linear_key_head_dim']
    lv_dim = tc['linear_value_head_dim']

    hs = emb.clone()
    residual = hs.clone()
    hs_n = cpu_rms_norm(hs, w_in0)

    # GatedDeltaNet layer 0
    mixed0 = F.linear(hs_n, w_in_qkv0)
    mixed_t = mixed0.transpose(1, 2)
    mixed_p = F.pad(mixed_t, (3, 0))
    conv0 = F.conv1d(mixed_p.float(), w_conv0.float(), bias=None, groups=mixed_t.shape[1])
    conv0 = F.silu(conv0).transpose(1, 2)
    q0 = conv0[:, :, :lk_heads*lk_dim].view(B, S, lk_heads, lk_dim)
    k0 = conv0[:, :, lk_heads*lk_dim:2*lk_heads*lk_dim].view(B, S, lk_heads, lk_dim)
    v0 = conv0[:, :, 2*lk_heads*lk_dim:].view(B, S, lv_heads, lv_dim)
    a0 = F.linear(hs_n, w_a0)
    b0 = F.linear(hs_n, w_b0)
    g0 = -torch.exp(A_log0) * F.softplus(a0 + dt_b0)
    beta0 = torch.sigmoid(b0)
    q0_n = F.normalize(q0.float(), p=2, dim=-1).to(q0.dtype)
    k0_n = F.normalize(k0.float(), p=2, dim=-1).to(k0.dtype)
    rpt = lv_heads // lk_heads
    if rpt > 1:
        q0_n = q0_n.repeat_interleave(rpt, dim=2)
        k0_n = k0_n.repeat_interleave(rpt, dim=2)
    q0_n = q0_n * (1.0 / math.sqrt(lk_dim))

    state = torch.zeros(B, lv_heads, lk_dim, lv_dim, dtype=torch.float32)
    core_out = torch.zeros(B, S, lv_heads, lv_dim, dtype=torch.float32)
    for t in range(S):
        gt = g0[:, t, :]
        kt = k0_n[:, t, :, :]
        vt = v0[:, t, :, :]
        qt = q0_n[:, t, :, :]
        bt = beta0[:, t, :]
        state = state * torch.exp(gt.float())[:, :, None, None]
        kv_mem = torch.sum(state * kt.float()[:, :, :, None], dim=-2)
        delta = (vt.float() - kv_mem) * bt.float()[:, :, None]
        state = state + kt.float()[:, :, :, None] * delta[:, :, None, :]
        ot = torch.sum(state * qt.float()[:, :, :, None], dim=-2)
        core_out[:, t, :, :] = ot
    z0_out = F.linear(hs_n, w_z0).view(B, S, lv_heads, lv_dim)
    rstd = 1.0 / torch.sqrt(core_out.float().reshape(-1, lv_dim).pow(2).mean(-1, keepdim=True) + eps)
    gated0 = (rstd * core_out.float().reshape(-1, lv_dim) * w_nm0.float() *
              F.silu(z0_out.float().reshape(-1, lv_dim))).view(B, S, lv_heads*lv_dim)
    attn_out0 = F.linear(gated0, w_ou0)

    residual = residual + attn_out0.float()
    hs_n = cpu_rms_norm(residual, w_po0)
    gate_h = F.linear(hs_n, w_g0)
    up_h = F.linear(hs_n, w_u0)
    mlp_out = F.linear(F.silu(gate_h) * up_h.float(), w_d0).float()
    hs = mlp_out

    # --- Layer 1 (simplified: same pattern) ---
    w_in1 = load_hf(prefix1 + 'input_layernorm.weight').float()
    w_po1 = load_hf(prefix1 + 'post_attention_layernorm.weight').float()
    w_in_qkv1 = load_hf(prefix1 + 'linear_attn.in_proj_qkv.weight').float()
    w_conv1 = load_hf(prefix1 + 'linear_attn.conv1d.weight').float()
    w_a1 = load_hf(prefix1 + 'linear_attn.in_proj_a.weight').float()
    w_b1 = load_hf(prefix1 + 'linear_attn.in_proj_b.weight').float()
    A_log1 = load_hf(prefix1 + 'linear_attn.A_log').float()
    dt_b1 = load_hf(prefix1 + 'linear_attn.dt_bias').float()
    w_z1 = load_hf(prefix1 + 'linear_attn.in_proj_z.weight').float()
    w_nm1 = load_hf(prefix1 + 'linear_attn.norm.weight').float()
    w_ou1 = load_hf(prefix1 + 'linear_attn.out_proj.weight').float()
    w_g1 = load_hf(prefix1 + 'mlp.gate_proj.weight').float()
    w_u1 = load_hf(prefix1 + 'mlp.up_proj.weight').float()
    w_d1 = load_hf(prefix1 + 'mlp.down_proj.weight').float()

    residual = residual + hs
    hs_n = cpu_rms_norm(residual, w_in1)

    mixed1 = F.linear(hs_n, w_in_qkv1)
    mixed_t1 = mixed1.transpose(1, 2)
    mixed_p1 = F.pad(mixed_t1, (3, 0))
    conv1 = F.conv1d(mixed_p1.float(), w_conv1.float(), bias=None, groups=mixed_t1.shape[1])
    conv1 = F.silu(conv1).transpose(1, 2)
    q1 = conv1[:, :, :lk_heads*lk_dim].view(B, S, lk_heads, lk_dim)
    k1 = conv1[:, :, lk_heads*lk_dim:2*lk_heads*lk_dim].view(B, S, lk_heads, lk_dim)
    v1 = conv1[:, :, 2*lk_heads*lk_dim:].view(B, S, lv_heads, lv_dim)
    a1 = F.linear(hs_n, w_a1)
    b1 = F.linear(hs_n, w_b1)
    g1 = -torch.exp(A_log1) * F.softplus(a1 + dt_b1)
    beta1 = torch.sigmoid(b1)
    q1_n = F.normalize(q1.float(), p=2, dim=-1).to(q1.dtype)
    k1_n = F.normalize(k1.float(), p=2, dim=-1).to(k1.dtype)
    if rpt > 1:
        q1_n = q1_n.repeat_interleave(rpt, dim=2)
        k1_n = k1_n.repeat_interleave(rpt, dim=2)
    q1_n = q1_n * (1.0 / math.sqrt(lk_dim))

    state1 = torch.zeros(B, lv_heads, lk_dim, lv_dim, dtype=torch.float32)
    core_out1 = torch.zeros(B, S, lv_heads, lv_dim, dtype=torch.float32)
    for t in range(S):
        gt = g1[:, t, :]; kt = k1_n[:, t, :, :]; vt = v1[:, t, :, :]; qt = q1_n[:, t, :, :]; bt = beta1[:, t, :]
        state1 = state1 * torch.exp(gt.float())[:, :, None, None]
        kv_mem = torch.sum(state1 * kt.float()[:, :, :, None], dim=-2)
        delta1 = (vt.float() - kv_mem) * bt.float()[:, :, None]
        state1 = state1 + kt.float()[:, :, :, None] * delta1[:, :, None, :]
        ot = torch.sum(state1 * qt.float()[:, :, :, None], dim=-2)
        core_out1[:, t, :, :] = ot
    z1_out = F.linear(hs_n, w_z1).view(B, S, lv_heads, lv_dim)
    rstd1 = 1.0 / torch.sqrt(core_out1.float().reshape(-1, lv_dim).pow(2).mean(-1, keepdim=True) + eps)
    gated1 = (rstd1 * core_out1.float().reshape(-1, lv_dim) * w_nm1.float() *
              F.silu(z1_out.float().reshape(-1, lv_dim))).view(B, S, lv_heads*lv_dim)
    attn_out1 = F.linear(gated1, w_ou1)
    residual = residual + attn_out1.float()
    hs_n = cpu_rms_norm(residual, w_po1)
    gate_h1 = F.linear(hs_n, w_g1); up_h1 = F.linear(hs_n, w_u1)
    hs = F.linear(F.silu(gate_h1) * up_h1.float(), w_d1).float()

    # --- Layer 2 (same pattern) ---
    w_in2 = load_hf(prefix2 + 'input_layernorm.weight').float()
    w_po2 = load_hf(prefix2 + 'post_attention_layernorm.weight').float()
    w_in_qkv2 = load_hf(prefix2 + 'linear_attn.in_proj_qkv.weight').float()
    w_conv2 = load_hf(prefix2 + 'linear_attn.conv1d.weight').float()
    w_a2 = load_hf(prefix2 + 'linear_attn.in_proj_a.weight').float()
    w_b2 = load_hf(prefix2 + 'linear_attn.in_proj_b.weight').float()
    A_log2 = load_hf(prefix2 + 'linear_attn.A_log').float()
    dt_b2 = load_hf(prefix2 + 'linear_attn.dt_bias').float()
    w_z2 = load_hf(prefix2 + 'linear_attn.in_proj_z.weight').float()
    w_nm2 = load_hf(prefix2 + 'linear_attn.norm.weight').float()
    w_ou2 = load_hf(prefix2 + 'linear_attn.out_proj.weight').float()
    w_g2 = load_hf(prefix2 + 'mlp.gate_proj.weight').float()
    w_u2 = load_hf(prefix2 + 'mlp.up_proj.weight').float()
    w_d2 = load_hf(prefix2 + 'mlp.down_proj.weight').float()

    residual = residual + hs
    hs_n = cpu_rms_norm(residual, w_in2)

    mixed2 = F.linear(hs_n, w_in_qkv2)
    mixed_t2 = mixed2.transpose(1, 2)
    mixed_p2 = F.pad(mixed_t2, (3, 0))
    conv2 = F.conv1d(mixed_p2.float(), w_conv2.float(), bias=None, groups=mixed_t2.shape[1])
    conv2 = F.silu(conv2).transpose(1, 2)
    q2 = conv2[:, :, :lk_heads*lk_dim].view(B, S, lk_heads, lk_dim)
    k2 = conv2[:, :, lk_heads*lk_dim:2*lk_heads*lk_dim].view(B, S, lk_heads, lk_dim)
    v2 = conv2[:, :, 2*lk_heads*lk_dim:].view(B, S, lv_heads, lv_dim)
    a2 = F.linear(hs_n, w_a2); b2 = F.linear(hs_n, w_b2)
    g2 = -torch.exp(A_log2) * F.softplus(a2 + dt_b2); beta2 = torch.sigmoid(b2)
    q2_n = F.normalize(q2.float(), p=2, dim=-1).to(q2.dtype)
    k2_n = F.normalize(k2.float(), p=2, dim=-1).to(k2.dtype)
    if rpt > 1:
        q2_n = q2_n.repeat_interleave(rpt, dim=2); k2_n = k2_n.repeat_interleave(rpt, dim=2)
    q2_n = q2_n * (1.0 / math.sqrt(lk_dim))

    state2 = torch.zeros(B, lv_heads, lk_dim, lv_dim, dtype=torch.float32)
    core_out2 = torch.zeros(B, S, lv_heads, lv_dim, dtype=torch.float32)
    for t in range(S):
        gt = g2[:, t, :]; kt = k2_n[:, t, :, :]; vt = v2[:, t, :, :]; qt = q2_n[:, t, :, :]; bt = beta2[:, t, :]
        state2 = state2 * torch.exp(gt.float())[:, :, None, None]
        kv_mem = torch.sum(state2 * kt.float()[:, :, :, None], dim=-2)
        delta2 = (vt.float() - kv_mem) * bt.float()[:, :, None]
        state2 = state2 + kt.float()[:, :, :, None] * delta2[:, :, None, :]
        ot = torch.sum(state2 * qt.float()[:, :, :, None], dim=-2)
        core_out2[:, t, :, :] = ot
    z2_out = F.linear(hs_n, w_z2).view(B, S, lv_heads, lv_dim)
    rstd2 = 1.0 / torch.sqrt(core_out2.float().reshape(-1, lv_dim).pow(2).mean(-1, keepdim=True) + eps)
    gated2 = (rstd2 * core_out2.float().reshape(-1, lv_dim) * w_nm2.float() *
              F.silu(z2_out.float().reshape(-1, lv_dim))).view(B, S, lv_heads*lv_dim)
    attn_out2 = F.linear(gated2, w_ou2)
    residual = residual + attn_out2.float()
    hs_n = cpu_rms_norm(residual, w_po2)
    gate_h2 = F.linear(hs_n, w_g2); up_h2 = F.linear(hs_n, w_u2)
    hs = F.linear(F.silu(gate_h2) * up_h2.float(), w_d2).float()

    return hs, residual

hs_l3_input, residual_l3 = cpu_forward_layer0_2()
print(f"After L0-L2: hs_norm={hs_l3_input.float().norm():.4f}, residual_norm={residual_l3.float().norm():.4f}")

# ============================================================
# CPU FullAttention (TP=1, all heads)
# ============================================================
print("\n=== CPU FullAttention TP=1 ===")
resid_cpu = residual_l3.float() + hs_l3_input.float()
hs_normed_cpu = cpu_rms_norm(resid_cpu, w_input_ln)
print(f"  After input_ln: norm={hs_normed_cpu.float().norm():.4f}")

q_full = F.linear(hs_normed_cpu, w_q)  # [1,7,12288]
q_cpu, gate_cpu = torch.chunk(q_full, 2, dim=-1)  # [1,7,6144] each
k_cpu = F.linear(hs_normed_cpu, w_k)  # [1,7,1024]
v_cpu = F.linear(hs_normed_cpu, w_v)  # [1,7,1024]

q_cpu = q_cpu.view(B, S, num_heads, head_dim).clone()
k_cpu = k_cpu.view(B, S, num_kv_heads, head_dim).clone()
v_cpu = v_cpu.view(B, S, num_kv_heads, head_dim).clone()
gate_cpu = gate_cpu.view(B, S, num_heads, head_dim).clone()

print(f"  Q pre-norm: shape={list(q_cpu.shape)}, norm={q_cpu.float().norm():.4f}")
print(f"  K pre-norm: shape={list(k_cpu.shape)}, norm={k_cpu.float().norm():.4f}")

q_cpu = cpu_rms_norm(q_cpu, w_qn.unsqueeze(0).unsqueeze(0))
k_cpu = cpu_rms_norm(k_cpu, w_kn.unsqueeze(0).unsqueeze(0))

cpu_apply_mrope(q_cpu, k_cpu, positions)

print(f"  Q post-MRoPE: norm={q_cpu.float().norm():.4f}")
print(f"  K post-MRoPE: norm={k_cpu.float().norm():.4f}")

# GQA expansion for CPU SDPA
n_groups = num_heads // num_kv_heads  # 6
k_expanded = k_cpu.unsqueeze(2).expand(-1, -1, n_groups, -1, -1).reshape(B*S, num_heads, head_dim)
v_expanded = v_cpu.unsqueeze(2).expand(-1, -1, n_groups, -1, -1).reshape(B*S, num_heads, head_dim)
q_attn = q_cpu.reshape(B*S, num_heads, head_dim)

attn_out_cpu = F.scaled_dot_product_attention(
    q_attn.transpose(0,1).unsqueeze(0).float(),
    k_expanded.transpose(0,1).unsqueeze(0).float(),
    v_expanded.transpose(0,1).unsqueeze(0).float(),
    is_causal=True
).squeeze(0).transpose(0,1).to(torch.bfloat16)  # [7, 24, 256]

attn_out_cpu = attn_out_cpu.reshape(B, S, num_heads * head_dim)
gate_out_cpu = torch.sigmoid(gate_cpu.reshape(B, S, num_heads * head_dim))
attn_out_cpu = attn_out_cpu * gate_out_cpu
output_cpu = F.linear(attn_out_cpu.float(), w_o).bfloat16()

print(f"  Attn output (post-gate): norm={attn_out_cpu.float().norm():.4f}")
print(f"  Final output: norm={output_cpu.float().norm():.4f}")

# ============================================================
# GPU FullAttention (TP=4 simulation, rank-by-rank)
# ============================================================
print("\n=== GPU FullAttention TP=4 (simulated per rank) ===")

# Load GPU cos/sin cache
from engine.kernels.rotary_embedding import get_cos_sin_cache, rotary_embedding
cos_sin_gpu = get_cos_sin_cache(max_pos, rotary_dim, rope_theta, dtype=dtype, device=device,
                                 mrope_section=mrope_section, mrope_interleaved=True)

# Move weights and inputs to GPU
hs_normed_gpu = hs_normed_cpu.to(dtype=dtype, device=device)
w_q_gpu = w_q.to(device=device, dtype=dtype)
w_k_gpu = w_k.to(device=device, dtype=dtype)
w_v_gpu = w_v.to(device=device, dtype=dtype)
w_o_gpu = w_o.to(device=device, dtype=dtype)
w_qn_gpu = w_qn.to(device=device, dtype=dtype)
w_kn_gpu = w_kn.to(device=device, dtype=dtype)

tp_size = 4
heads_per_rank = num_heads // tp_size  # 6
kv_heads_per_rank = num_kv_heads // tp_size  # 1
q_per = heads_per_rank * head_dim  # 1536
kv_per = kv_heads_per_rank * head_dim  # 256

all_rank_attn = []
all_rank_gated = []
all_rank_partial = []

for r in range(tp_size):
    print(f"\n  Rank {r}:")

    # ColumnParallel: split q weight (first 6144 rows)
    q_w_r = w_q_gpu[r*q_per:(r+1)*q_per, :].clone()  # [1536, 5120]
    # Gate weight: next 6144 rows
    gate_w_r = w_q_gpu[6144 + r*q_per:6144 + (r+1)*q_per, :].clone()  # [1536, 5120]
    # KV: split by kv_heads
    k_w_r = w_k_gpu[r*kv_per:(r+1)*kv_per, :].clone()  # [256, 5120]
    v_w_r = w_v_gpu[r*kv_per:(r+1)*kv_per, :].clone()  # [256, 5120]
    # o_proj: RowParallel (split cols)
    o_w_r = w_o_gpu[:, r*q_per:(r+1)*q_per].clone()  # [5120, 1536]

    # Forward
    q_r = F.linear(hs_normed_gpu, q_w_r)  # [1,7,1536]
    k_r = F.linear(hs_normed_gpu, k_w_r)  # [1,7,256]
    v_r = F.linear(hs_normed_gpu, v_w_r)  # [1,7,256]
    gate_r = F.linear(hs_normed_gpu, gate_w_r)  # [1,7,1536]

    q_r = q_r.view(B, S, heads_per_rank, head_dim)
    k_r = k_r.view(B, S, kv_heads_per_rank, head_dim)
    v_r = v_r.view(B, S, kv_heads_per_rank, head_dim)
    gate_r = gate_r.view(B, S, heads_per_rank, head_dim)

    # Q/K norms (GPU-style: using rms_norm kernel with effective weight)
    from engine.kernels.rms_norm import rms_norm as gpu_rms_norm
    q_r = q_r.reshape(B*S, heads_per_rank, head_dim)
    k_r = k_r.reshape(B*S, kv_heads_per_rank, head_dim)

    q_r_normed = torch.empty_like(q_r)
    gpu_rms_norm(q_r_normed, q_r.contiguous(), (1.0 + w_qn_gpu), eps)
    k_r_normed = torch.empty_like(k_r)
    gpu_rms_norm(k_r_normed, k_r.contiguous(), (1.0 + w_kn_gpu), eps)

    # Compare with CPU Q/K norms for this rank
    q_cpu_r = q_cpu[:, :, r*heads_per_rank:(r+1)*heads_per_rank, :].reshape(B*S, heads_per_rank, head_dim)
    k_cpu_r = k_cpu[:, :, r:r+1, :].reshape(B*S, kv_heads_per_rank, head_dim)
    q_diff = (q_r_normed - q_cpu_r.to(device=device, dtype=dtype)).abs()
    k_diff = (k_r_normed - k_cpu_r.to(device=device, dtype=dtype)).abs()
    print(f"    Q norm diff: max={q_diff.max():.6f} mean={q_diff.mean():.6f}")
    print(f"    K norm diff: max={k_diff.max():.6f} mean={k_diff.mean():.6f}")

    # MRoPE
    q_rot = q_r_normed[..., :rotary_dim].contiguous()
    k_rot = k_r_normed[..., :rotary_dim].contiguous()
    rotary_embedding(positions.to(device), q_rot, k_rot, rotary_dim, cos_sin_gpu, is_neox=True)
    q_r_normed[..., :rotary_dim] = q_rot
    k_r_normed[..., :rotary_dim] = k_rot

    # Compare MRoPE output
    q_rope_diff = (q_r_normed - q_cpu_r.to(device=device, dtype=dtype)).abs()
    k_rope_diff = (k_r_normed - k_cpu_r.to(device=device, dtype=dtype)).abs()
    print(f"    Q post-MRoPE diff: max={q_rope_diff.max():.6f} mean={q_rope_diff.mean():.6f}")
    print(f"    K post-MRoPE diff: max={k_rope_diff.max():.6f} mean={k_rope_diff.mean():.6f}")

    # SDPA (GQA: 6 Q heads, 1 KV head)
    from engine.kernels.attention import flash_attn_varlen_func
    n_tokens = B * S
    cu = torch.tensor([0, n_tokens], dtype=torch.int32, device=device)
    attn_r = flash_attn_varlen_func(
        q_r_normed, k_r_normed, v_r.reshape(B*S, kv_heads_per_rank, head_dim),
        cu, cu, n_tokens, n_tokens,
        causal=True, softmax_scale=head_dim**-0.5)
    attn_r = attn_r.reshape(B, S, heads_per_rank * head_dim)

    # Compare attention output
    attn_cpu_r = attn_out_cpu[:, :, r*q_per:(r+1)*q_per]  # before gate
    attn_diff = (attn_r - attn_cpu_r.to(device=device, dtype=dtype)).abs()
    print(f"    Attn output diff (pre-gate): max={attn_diff.max():.6f} mean={attn_diff.mean():.6f}")

    # Gate
    gate_flat_r = gate_r.reshape(B, S, heads_per_rank * head_dim)
    gated_r = attn_r * torch.sigmoid(gate_flat_r)

    gate_cpu_r = gate_out_cpu[:, :, r*q_per:(r+1)*q_per]
    gated_diff = (gated_r - gate_cpu_r.to(device=device, dtype=dtype)).abs()
    print(f"    Gated output diff: max={gated_diff.max():.6f} mean={gated_diff.mean():.6f}")

    # o_proj per rank
    partial_r = F.linear(gated_r, o_w_r)  # [1,7,5120]
    all_rank_partial.append(partial_r)

# Sum across ranks (all_reduce_sum)
output_gpu = sum(all_rank_partial)

output_diff = (output_gpu - output_cpu.to(device=device, dtype=dtype)).abs()
print(f"\n=== Final Comparison ===")
print(f"  GPU output norm: {output_gpu.float().norm():.4f}")
print(f"  CPU output norm: {output_cpu.float().norm():.4f}")
print(f"  Ratio GPU/CPU: {output_gpu.float().norm() / output_cpu.float().norm():.4f}")
print(f"  Max diff: {output_diff.max():.6f}")
print(f"  Mean diff: {output_diff.mean():.6f}")

max_err = output_diff.max().item()
if max_err < 0.1:
    print("\nRESULT: PASS — GPU TP=4 simulation matches CPU reference")
else:
    print(f"\nRESULT: FAIL — GPU diverges from CPU (max diff={max_err:.4f})")
    # Find where the divergence starts
    print("\n--- Tracing divergence source ---")

    # Per-rank partial norms
    for r in range(tp_size):
        print(f"  Rank {r} partial o_proj norm: {all_rank_partial[r].float().norm():.4f}")
    print(f"  CPU full o_proj norm: {output_cpu.float().norm():.4f}")

    # Check: does sum of per-rank attn_outputs equal full CPU attn output?
    cpu_attn_pre_gate = attn_out_cpu_pre_gate.reshape(B, S, num_heads * head_dim)
    # We don't have this saved, let's compute individual comparison
    n_tokens = B * S
    cu = torch.tensor([0, n_tokens], dtype=torch.int32, device=device)

    # Re-do GPU attention but compare with CORRECT CPU per-rank attention
    # Actually let's check: is the SDPA on GPU producing correct attention?
    # Compare rank 0 GPU attention vs CPU attention for the same heads
    print("\n--- Checking SDPA GQA vs MHA equivalence ---")
    r = 0
    q_r_test = q_r_normed_saved[r]
    k_r_test = k_r_normed_saved[r]
    v_r_test = v_r_saved[r]

    # GPU attention for this rank
    attn_gpu_r = flash_attn_varlen_func(
        q_r_test, k_r_test, v_r_test,
        cu, cu, n_tokens, n_tokens,
        causal=True, softmax_scale=head_dim**-0.5)
    print(f"  GPU attn (rank {r}): shape={list(attn_gpu_r.shape)}, norm={attn_gpu_r.float().norm():.4f}")

    # Equivalent CPU computation: same 6 Q heads, 1 KV head (GQA 6:1)
    q_cpu_r = q_cpu[:, :, r*heads_per_rank:(r+1)*heads_per_rank, :].reshape(B*S, heads_per_rank, head_dim)
    k_cpu_r = k_cpu[:, :, r:r+1, :].reshape(B*S, 1, head_dim)
    v_cpu_r = v_cpu[:, :, r:r+1, :].reshape(B*S, 1, head_dim)
    # v_cpu was NOT modified in-place (only q and k were)

    attn_cpu_r = F.scaled_dot_product_attention(
        q_cpu_r.transpose(0,1).unsqueeze(0).float(),
        k_cpu_r.transpose(0,1).unsqueeze(0).float(),
        v_cpu_r.transpose(0,1).unsqueeze(0).float(),
        is_causal=True
    ).squeeze(0).transpose(0,1).to(dtype)
    print(f"  CPU attn (rank {r}): shape={list(attn_cpu_r.shape)}, norm={attn_cpu_r.float().norm():.4f}")
    attn_direct_diff = (attn_gpu_r - attn_cpu_r.to(device=device)).abs()
    print(f"  Direct attn diff: max={attn_direct_diff.max():.6f} mean={attn_direct_diff.mean():.6f}")

    # Also check: CPU attention with full MHA vs concatenated per-rank GQA
    # The full CPU MHA attention output (pre-gate) is already in q_full 
    # Actually we didn't save it! Let me recompute it
    cpu_attn_full = F.scaled_dot_product_attention(
        q_attn_cpu_saved.transpose(0,1).unsqueeze(0).float(),
        k_expanded_saved.transpose(0,1).unsqueeze(0).float(),
        v_expanded_saved.transpose(0,1).unsqueeze(0).float(),
        is_causal=True
    ).squeeze(0).transpose(0,1).to(dtype)
    print(f"  CPU MHA attn: shape={list(cpu_attn_full.shape)}, norm={cpu_attn_full.float().norm():.4f}")
