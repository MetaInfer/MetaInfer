#!/usr/bin/env python3
"""CPU-only TP=4 FullAttention simulation.

Computes FullAttention layer 3 on CPU using:
  1. The exact input from the CPU reference (diag_cpu_layers05_ref.pt)
  2. Simulates TP=4 sharding: split Q/K/V heads across 4 ranks
  3. Recombines via RowParallel o_proj with all_reduce
  4. Compares with:
     a. CPU reference (no TP, full heads)
     b. GPU output (actual TP=4, from norms_compare2)

This tells us if the TP sharding math itself has a bug.
"""
import os, sys, torch, json, math
import torch.nn.functional as F

model_dir = os.environ['MODEL_DIR']
from safetensors import safe_open

# Load weights and config once
with open(os.path.join(model_dir, 'model.safetensors.index.json')) as f:
    idx = json.load(f)
wm = idx['weight_map']

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

def load_cpu(key):
    fname = wm[key]
    fpath = os.path.join(model_dir, fname)
    with safe_open(fpath, framework='pt', device='cpu') as sf:
        return sf.get_tensor(key)

def qwen35_rms_norm(x, w):
    rstd = 1.0 / torch.sqrt(x.float().pow(2).mean(-1, keepdim=True) + eps)
    return (x.float() * rstd * (1.0 + w.float())).float()

# Load layer 3 weights
prefix = 'model.language_model.layers.3.'
w_input_ln = load_cpu(prefix + 'input_layernorm.weight').float()
w_post_ln = load_cpu(prefix + 'post_attention_layernorm.weight').float()
w_q_fused = load_cpu(prefix + 'self_attn.q_proj.weight').float()  # [12288, 5120]
full_q_size = num_heads * head_dim  # 6144
w_q_full = w_q_fused[:full_q_size, :]    # [6144, 5120] - All Q heads
w_gate_full = w_q_fused[full_q_size:, :] # [6144, 5120] - All Gate heads
w_k_full = load_cpu(prefix + 'self_attn.k_proj.weight').float()  # [1024, 5120]
w_v_full = load_cpu(prefix + 'self_attn.v_proj.weight').float()  # [1024, 5120]
w_o_full = load_cpu(prefix + 'self_attn.o_proj.weight').float()  # [5120, 6144]
w_qn = load_cpu(prefix + 'self_attn.q_norm.weight').float()     # [256]
w_kn = load_cpu(prefix + 'self_attn.k_norm.weight').float()     # [256]
w_gate_proj = load_cpu(prefix + 'mlp.gate_proj.weight').float()
w_up_proj = load_cpu(prefix + 'mlp.up_proj.weight').float()
w_down_proj = load_cpu(prefix + 'mlp.down_proj.weight').float()

# MRoPE cache (simple interleaving, matches both CPU refs)
num_sections = len(mrope_section)
half_dim = rotary_dim // 2
section_freqs = []
t = torch.arange(max_pos, dtype=torch.float32)
for sec_size in mrope_section:
    inv_freq = 1.0 / (rope_theta ** (torch.arange(0, sec_size, dtype=torch.float32) * 2 / rotary_dim))
    section_freqs.append(torch.einsum("i,j->ij", t, inv_freq))
pairs = []
for i in range(half_dim):
    s = i % num_sections
    idx = i // num_sections
    pairs.append(section_freqs[s][:, idx])
freqs = torch.stack(pairs, dim=-1)
cos_sin_cache = torch.cat((freqs.cos(), freqs.sin()), dim=-1)

# TP=4 configuration
TP = 4
heads_per_tp = num_heads // TP       # 6
kv_heads_per_tp = num_kv_heads // TP  # 1
head_dim_q = head_dim * heads_per_tp  # 1536
head_dim_kv = head_dim * kv_heads_per_tp  # 256

# ============================================================
# REFERENCE (no TP): from diag_cpu_layers05
# ============================================================
ref = torch.load('/tmp/diag_cpu_layers05_ref.pt', map_location='cpu', weights_only=True)
cp_ref = ref['checkpoints']
ref_layer3_norm = cp_ref[3]['norm']
print(f"CPU Reference (diag_cpu_layers05) layer 3 norm: {ref_layer3_norm:.4f}")

# Reconstruct layer 3 input from reference data
# We need the residual + hs (mlp_out_2) to form the layer 3 input
# cp_ref[2]['hs'] is the mlp_out of layer 2
# We need to reconstruct the residual from embedding through layer 2

# Instead of reconstructing, let's compute it directly:
# Load the saved GPU layer 3 input if available, or recompute

# Recompute layers 0-2 inline using the same logic as diag_cpu_layers05.py
tokens = [108618, 102066, 137351, 105017, 100462, 106808, 103105]
input_ids = torch.tensor([tokens], dtype=torch.long)
S = len(tokens)
positions = torch.arange(S, dtype=torch.int64)

embed_w = load_cpu('model.language_model.embed_tokens.weight').float()
emb = F.embedding(input_ids, embed_w)

# Layers 0-2: GatedDeltaNet
hs = emb.clone()
residual = None

for layer_idx in range(3):
    pfx = f'model.language_model.layers.{layer_idx}.'
    w_iln = load_cpu(pfx + 'input_layernorm.weight').float()
    w_pln = load_cpu(pfx + 'post_attention_layernorm.weight').float()

    if residual is None:
        residual = hs.clone()
        hs_normed = qwen35_rms_norm(hs, w_iln)
    else:
        residual = residual + hs
        hs_normed = qwen35_rms_norm(residual, w_iln)
        hs.copy_(residual)

    B, T, H = hs_normed.shape

    # GatedDeltaNet forward
    lk_heads = tc['linear_num_key_heads']      # 16
    lv_heads = tc['linear_num_value_heads']     # 48
    lk_dim = tc['linear_key_head_dim']          # 128
    lv_dim = tc['linear_value_head_dim']        # 128

    w_in_qkv = load_cpu(pfx + 'linear_attn.in_proj_qkv.weight').float()
    w_conv1d = load_cpu(pfx + 'linear_attn.conv1d.weight').float()
    w_in_a = load_cpu(pfx + 'linear_attn.in_proj_a.weight').float()
    w_in_b = load_cpu(pfx + 'linear_attn.in_proj_b.weight').float()
    A_log = load_cpu(pfx + 'linear_attn.A_log').float()
    dt_bias = load_cpu(pfx + 'linear_attn.dt_bias').float()
    w_in_z = load_cpu(pfx + 'linear_attn.in_proj_z.weight').float()
    w_norm_gdn = load_cpu(pfx + 'linear_attn.norm.weight').float()
    w_out_gdn = load_cpu(pfx + 'linear_attn.out_proj.weight').float()

    mixed_qkv = F.linear(hs_normed, w_in_qkv)
    mixed_qkv_t = mixed_qkv.transpose(1, 2)
    mixed_qkv_pad = F.pad(mixed_qkv_t, (3, 0))
    conv_out = F.conv1d(mixed_qkv_pad, w_conv1d, bias=None, groups=mixed_qkv_t.shape[1])
    conv_out = F.silu(conv_out).transpose(1, 2)

    q_lin = conv_out[:, :, :lk_heads*lk_dim].view(B, T, lk_heads, lk_dim)
    k_lin = conv_out[:, :, lk_heads*lk_dim:2*lk_heads*lk_dim].view(B, T, lk_heads, lk_dim)
    v_lin = conv_out[:, :, 2*lk_heads*lk_dim:].view(B, T, lv_heads, lv_dim)

    a = F.linear(hs_normed, w_in_a)
    b = F.linear(hs_normed, w_in_b)
    g = -torch.exp(A_log) * F.softplus(a + dt_bias)
    beta = torch.sigmoid(b)

    q_norm_gdn = F.normalize(q_lin.float(), p=2, dim=-1).to(q_lin.dtype)
    k_norm_gdn = F.normalize(k_lin.float(), p=2, dim=-1).to(k_lin.dtype)
    rpt = lv_heads // lk_heads
    if rpt > 1:
        q_norm_gdn = q_norm_gdn.repeat_interleave(rpt, dim=2)
        k_norm_gdn = k_norm_gdn.repeat_interleave(rpt, dim=2)
    q_norm_gdn = q_norm_gdn * (1.0 / math.sqrt(lk_dim))

    state = torch.zeros(B, lv_heads, lk_dim, lv_dim, dtype=torch.float32)
    core_out_gdn = torch.zeros(B, T, lv_heads, lv_dim, dtype=torch.float32)
    for t_idx in range(T):
        g_t = g[:, t_idx, :]
        k_t = k_norm_gdn[:, t_idx, :, :]
        v_t = v_lin[:, t_idx, :, :]
        q_t = q_norm_gdn[:, t_idx, :, :]
        beta_t = beta[:, t_idx, :]
        state = state * torch.exp(g_t.float())[:, :, None, None]
        kv_mem = torch.sum(state * k_t.float()[:, :, :, None], dim=-2)
        delta = (v_t.float() - kv_mem) * beta_t.float()[:, :, None]
        state = state + k_t.float()[:, :, :, None] * delta[:, :, None, :]
        o_t = torch.sum(state * q_t.float()[:, :, :, None], dim=-2)
        core_out_gdn[:, t_idx, :, :] = o_t

    z = F.linear(hs_normed, w_in_z).view(B, T, lv_heads, lv_dim)
    core_flat = core_out_gdn.reshape(-1, lv_dim)
    z_flat = z.reshape(-1, lv_dim)
    rstd = 1.0 / torch.sqrt(core_flat.float().pow(2).mean(-1, keepdim=True) + eps)
    x_norm = core_flat.float() * rstd
    gated = (x_norm * w_norm_gdn.float() * F.silu(z_flat)).view(B, T, lv_heads * lv_dim)
    attn_out = F.linear(gated.float(), w_out_gdn)

    # Post-attention + MLP
    residual = residual + attn_out.float()
    hs_normed_mlp = qwen35_rms_norm(residual, w_pln)
    w_gate_mlp = load_cpu(pfx + 'mlp.gate_proj.weight').float()
    w_up_mlp = load_cpu(pfx + 'mlp.up_proj.weight').float()
    w_down_mlp = load_cpu(pfx + 'mlp.down_proj.weight').float()
    gate_h = F.linear(hs_normed_mlp, w_gate_mlp)
    up_h = F.linear(hs_normed_mlp, w_up_mlp)
    mlp_out = F.linear(F.silu(gate_h) * up_h.float(), w_down_mlp).float()
    hs = mlp_out

print(f"Layer 2 output norm: {hs.norm():.4f}")

# Layer 3 input: residual (accumulated) + hs (mlp_out_2)
residual = residual + hs
hs_normed = qwen35_rms_norm(residual, w_input_ln)
print(f"Layer 3 normed input norm: {hs_normed.norm():.4f}")

B, T, _ = 1, S, hidden_size

# ============================================================
# PATH 1: Full computation (no TP), match diag_cpu_layers05
# ============================================================
q_full_raw = F.linear(hs_normed, w_q_fused)  # [1, 7, 12288]
q_full, gate_full = torch.chunk(q_full_raw, 2, dim=-1)
k_full = F.linear(hs_normed, w_k_full)  # [1, 7, 1024]
v_full = F.linear(hs_normed, w_v_full)  # [1, 7, 1024]

q_full_h = q_full.view(B, T, num_heads, head_dim)
k_full_h = k_full.view(B, T, num_kv_heads, head_dim)
v_full_h = v_full.view(B, T, num_kv_heads, head_dim)
gate_full_h = gate_full.view(B, T, num_heads, head_dim)

q_full_n = qwen35_rms_norm(q_full_h.float(), w_qn.unsqueeze(0).unsqueeze(0))
k_full_n = qwen35_rms_norm(k_full_h.float(), w_kn.unsqueeze(0).unsqueeze(0))

# MRoPE
q_full_r = q_full_n.clone()
k_full_r = k_full_n.clone()
pos_c = torch.arange(S, dtype=torch.int64)
cos = cos_sin_cache[pos_c, :rotary_dim//2].view(1, S, 1, rotary_dim//2)
sin = cos_sin_cache[pos_c, rotary_dim//2:].view(1, S, 1, rotary_dim//2)
cos_dup = torch.cat([cos, cos], dim=-1)
sin_dup = torch.cat([sin, sin], dim=-1)
q_rot = q_full_r[..., :rotary_dim].float()
half_r = rotary_dim // 2
q1, q2 = q_rot[..., :half_r], q_rot[..., half_r:]
q_full_r[..., :rotary_dim] = (q_rot * cos_dup + torch.cat([-q2, q1], dim=-1) * sin_dup)
k_rot = k_full_r[..., :rotary_dim].float()
k1, k2 = k_rot[..., :half_r], k_rot[..., half_r:]
k_full_r[..., :rotary_dim] = (k_rot * cos_dup + torch.cat([-k2, k1], dim=-1) * sin_dup)

# SDPA with GQA
n_groups = num_heads // num_kv_heads
k_attn = k_full_r.unsqueeze(2).expand(-1, -1, n_groups, -1, -1).reshape(B, num_heads, T, head_dim)
v_attn = v_full_h.unsqueeze(2).expand(-1, -1, n_groups, -1, -1).reshape(B, num_heads, T, head_dim)
q_attn = q_full_r.transpose(1, 2)
scale = head_dim ** -0.5
attn_out_full = F.scaled_dot_product_attention(
    q_attn.float(), k_attn.float(), v_attn.float(), is_causal=True, scale=scale
).transpose(1, 2).reshape(B, T, num_heads * head_dim)

# Output gate
gate_flat = gate_full_h.reshape(B, T, num_heads * head_dim)
attn_gated_full = attn_out_full.float() * torch.sigmoid(gate_flat.float())

# o_proj (full)
o_full_out = F.linear(attn_gated_full, w_o_full)  # [1, 7, 5120]

# MLP
resid_after_attn_full = residual + o_full_out.float()
hs_mlp_full = qwen35_rms_norm(resid_after_attn_full, w_post_ln)
gate_h_full = F.linear(hs_mlp_full, w_gate_proj)
up_h_full = F.linear(hs_mlp_full, w_up_proj)
mlp_full = F.linear(F.silu(gate_h_full) * up_h_full.float(), w_down_proj).float()

print(f"\nLayer 3 Full (no TP): mlp_out norm={mlp_full.norm():.4f}")
print(f"  vs CPU ref: {ref_layer3_norm:.4f}, diff={abs(mlp_full.norm()-ref_layer3_norm)/ref_layer3_norm*100:.2f}%")

# ============================================================
# PATH 2: TP=4 simulation
# ============================================================
# In TP, each rank has:
#   - Q weight: slice of w_q_full [rank*1536:(rank+1)*1536, :]
#   - Gate weight: slice of w_gate_full [rank*1536:(rank+1)*1536, :]
#   - K weight: slice of w_k_full [rank*256:(rank+1)*256, :]
#   - V weight: slice of w_v_full [rank*256:(rank+1)*256, :]
#   - o_proj weight: slice of w_o_full [:, rank*1536:(rank+1)*1536]

# For Q (and Gate): ColumnParallel of [num_heads*head_dim, hidden] = [6144, 5120] → [1536, 5120] per rank
# For K: ColumnParallel of [num_kv_heads*head_dim, hidden] = [1024, 5120] → [256, 5120] per rank
# For V: same as K
# For o_proj: RowParallel of [hidden, num_heads*head_dim] = [5120, 6144] → [5120, 1536] per rank

# Compute TP-sharded projections
q_tp = []   # list of [1, 7, 1536] for each rank
k_tp = []   # list of [1, 7, 256] for each rank
v_tp = []
gate_tp = []
o_tp = []

for r in range(TP):
    # Q: ColumnParallel shard
    r_start_q = r * head_dim_q       # r * 1536
    r_end_q = (r + 1) * head_dim_q
    w_q_r = w_q_full[r_start_q:r_end_q, :]  # [1536, 5120]
    w_gate_r = w_gate_full[r_start_q:r_end_q, :]

    # K/V: ColumnParallel shard
    r_start_kv = r * head_dim_kv      # r * 256
    r_end_kv = (r + 1) * head_dim_kv
    w_k_r = w_k_full[r_start_kv:r_end_kv, :]  # [256, 5120]
    w_v_r = w_v_full[r_start_kv:r_end_kv, :]

    # o_proj: RowParallel shard
    w_o_r = w_o_full[:, r_start_q:r_end_q]  # [5120, 1536]

    # Compute projections for this rank
    q_r = F.linear(hs_normed, w_q_r)        # [1, 7, 1536]
    gate_r = F.linear(hs_normed, w_gate_r)   # [1, 7, 1536]
    k_r = F.linear(hs_normed, w_k_r)         # [1, 7, 256]
    v_r = F.linear(hs_normed, w_v_r)         # [1, 7, 256]

    # Reshape to heads
    q_rh = q_r.view(B, T, heads_per_tp, head_dim)        # [1, 7, 6, 256]
    k_rh = k_r.view(B, T, kv_heads_per_tp, head_dim)      # [1, 7, 1, 256]
    v_rh = v_r.view(B, T, kv_heads_per_tp, head_dim)
    gate_rh = gate_r.view(B, T, heads_per_tp, head_dim)

    # Q/K norms
    q_rh_n = qwen35_rms_norm(q_rh.float(), w_qn.unsqueeze(0).unsqueeze(0))
    k_rh_n = qwen35_rms_norm(k_rh.float(), w_kn.unsqueeze(0).unsqueeze(0))

    # MRoPE
    q_rh_r = q_rh_n.clone()
    k_rh_r = k_rh_n.clone()
    q_rot_r = q_rh_r[..., :rotary_dim].float()
    qr1, qr2 = q_rot_r[..., :half_r], q_rot_r[..., half_r:]
    q_rh_r[..., :rotary_dim] = (q_rot_r * cos_dup + torch.cat([-qr2, qr1], dim=-1) * sin_dup)
    k_rot_r = k_rh_r[..., :rotary_dim].float()
    kr1, kr2 = k_rot_r[..., :half_r], k_rot_r[..., half_r:]
    k_rh_r[..., :rotary_dim] = (k_rot_r * cos_dup + torch.cat([-kr2, kr1], dim=-1) * sin_dup)

    # SDPA with GQA (each rank has 6 Q heads vs 1 KV head)
    n_groups_r = heads_per_tp // kv_heads_per_tp  # 6
    k_attn_r = k_rh_r.unsqueeze(2).expand(-1, -1, n_groups_r, -1, -1).reshape(B, heads_per_tp, T, head_dim)
    v_attn_r = v_rh.unsqueeze(2).expand(-1, -1, n_groups_r, -1, -1).reshape(B, heads_per_tp, T, head_dim)
    q_attn_r = q_rh_r.transpose(1, 2)

    attn_out_r = F.scaled_dot_product_attention(
        q_attn_r.float(), k_attn_r.float(), v_attn_r.float(), is_causal=True, scale=scale
    ).transpose(1, 2).reshape(B, T, heads_per_tp * head_dim)  # [1, 7, 1536]

    # Output gate
    gate_flat_r = gate_rh.reshape(B, T, heads_per_tp * head_dim)
    attn_gated_r = attn_out_r.float() * torch.sigmoid(gate_flat_r.float())

    # o_proj (RowParallel, NO all_reduce here - just local computation)
    o_r = F.linear(attn_gated_r, w_o_r)  # [1, 7, 5120]

    q_tp.append(q_r)
    k_tp.append(k_r)
    v_tp.append(v_r)
    gate_tp.append(gate_r)
    o_tp.append(o_r)

# Sum the RowParallel outputs (equivalent to all_reduce)
o_tp_sum = sum(o_tp)  # [1, 7, 5120]

# Compare o_tp_sum with o_full_out
o_diff = (o_tp_sum - o_full_out).abs()
print(f"\n=== TP=4 simulated o_proj comparison ===")
print(f"  TP sum norm: {o_tp_sum.norm():.4f}, Full norm: {o_full_out.norm():.4f}")
print(f"  Diff: max={o_diff.max():.6f}, mean={o_diff.mean():.6f}")

# Check if recombining Q across ranks gives the full Q
q_recombined = torch.cat(q_tp, dim=-1)  # [1, 7, 6144]
q_recombined_h = q_recombined.view(B, T, num_heads, head_dim)
q_recomb_diff = (q_recombined_h - q_full_h).abs()
print(f"\n=== Q recombination check ===")
print(f"  Recombined norm: {q_recombined_h.norm():.4f}, Full norm: {q_full_h.norm():.4f}")
print(f"  Diff: max={q_recomb_diff.max():.6f}, mean={q_recomb_diff.mean():.6f}")

# Run the full pipeline with TP=4 simulation
resid_after_attn_tp = residual + o_tp_sum.float()
hs_mlp_tp = qwen35_rms_norm(resid_after_attn_tp, w_post_ln)

# MLP: Using simplified ColumnParallel gate/up + RowParallel down
# For MLP, the intermediate dim is ffn_hidden (e.g. 13824)
# ColumnParallel: split the output dim across ranks
# RowParallel: split the input dim across ranks

# Actually, the MLP in our engine uses:
# - gate_proj: ColumnParallel (output sharded)
# - up_proj: ColumnParallel (output sharded)
# - down_proj: RowParallel (input sharded)
# For TP=4, each rank gets intermediate_dim // 4 of gate/up output

# But for simplicity and since the MLP is not our current focus,
# let me check: does the FullAttention o_proj output itself match?
# If TP-summed o_proj matches full o_proj, the issue is in the attention path.
# If not, the issue is in TP sharding of attention.

# But wait, the residual is DIFFERENT between TP=4 and no-TP cases!
# With TP=4, the residual after layer 3 = resid_pre + o_proj_rank(attn_gated_rank)
# which is different from resid_pre + o_proj_full(attn_gated_rank)
# because o_proj is RowParallel and needs all_reduce

# After all_reduce: o_proj_total = sum_i o_proj_i(attn_gated_i)
# = sum_i (attn_gated_i @ w_o_i)
# Does this equal (concat_i(attn_gated_i)) @ w_o_full?
#
# w_o_full = concat_i(w_o_i) along columns (dim -1)
# (concat_i(attn_gated_i)) @ w_o_full = sum_i(attn_gated_i @ w_o_i)
# Yes! Because matrix multiplication distributes over concatenation:
# [A | B | C | D] @ [W1; W2; W3; W4] = A@W1 + B@W2 + C@W3 + D@W4
#
# So the TP reduction is mathematically exact!
# This means if Q, K, V, norms, MRoPE, SDPA, and gate are all identical
# between TP and no-TP (just head partitions), the final output should match.

# Let me verify this more carefully: does q_full_h equal the concatenation of q_per_rank?
q_concat = torch.cat(q_tp, dim=-1).view(B, T, num_heads, head_dim)
print(f"\n=== Q heads: TP concat vs Full ===")
print(f"  TP concat norm: {q_concat.norm():.4f}")
print(f"  Full norm: {q_full_h.norm():.4f}")
print(f"  Match: {torch.allclose(q_concat, q_full_h, atol=1e-5)}")

# Check gate
gate_concat = torch.cat(gate_tp, dim=-1).view(B, T, num_heads, head_dim)
print(f"\nGate heads: TP concat={gate_concat.norm():.4f} Full={gate_full_h.norm():.4f}")
# Check if gate loading matches
print(f"  gate match: {torch.allclose(gate_concat, gate_full_h, atol=1e-5)}")
if not torch.allclose(gate_concat, gate_full_h, atol=1e-5):
    gd = (gate_concat - gate_full_h).abs()
    print(f"  gate diff: max={gd.max():.6f} mean={gd.mean():.6f}")

# Debug: Compare SDPA output per rank vs full SDPA output slices
print(f"\n=== Debug: SDPA output comparison ===")
for r in range(TP):
    r_start_q = r * head_dim_q
    r_end_q = (r + 1) * head_dim_q
    # attn_gated_r is from the TP simulation — we need to capture it.
    # Actually we recomputed it above. Let me capture in the loop.
    pass

# Let me redo the TP simulation more carefully with SDPA capture
tp_sdpa_outputs = []
tp_gated_outputs = []

for r in range(TP):
    r_start_q = r * head_dim_q
    r_end_q = (r + 1) * head_dim_q
    r_start_kv = r * head_dim_kv
    r_end_kv = (r + 1) * head_dim_kv

    w_q_r = w_q_full[r_start_q:r_end_q, :]
    w_gate_r = w_gate_full[r_start_q:r_end_q, :]
    w_k_r = w_k_full[r_start_kv:r_end_kv, :]
    w_v_r = w_v_full[r_start_kv:r_end_kv, :]
    w_o_r = w_o_full[:, r_start_q:r_end_q]

    q_r = F.linear(hs_normed, w_q_r).view(B, T, heads_per_tp, head_dim)
    gate_r = F.linear(hs_normed, w_gate_r).view(B, T, heads_per_tp, head_dim)
    k_r = F.linear(hs_normed, w_k_r).view(B, T, kv_heads_per_tp, head_dim)
    v_r = F.linear(hs_normed, w_v_r).view(B, T, kv_heads_per_tp, head_dim)

    q_rn = qwen35_rms_norm(q_r.float(), w_qn.unsqueeze(0).unsqueeze(0))
    k_rn = qwen35_rms_norm(k_r.float(), w_kn.unsqueeze(0).unsqueeze(0))

    # MRoPE
    q_rr = q_rn.clone()
    k_rr = k_rn.clone()
    q_rot_tp = q_rr[..., :rotary_dim].float()
    qr1, qr2 = q_rot_tp[..., :half_r], q_rot_tp[..., half_r:]
    q_rr[..., :rotary_dim] = (q_rot_tp * cos_dup + torch.cat([-qr2, qr1], dim=-1) * sin_dup)
    k_rot_tp = k_rr[..., :rotary_dim].float()
    kr1, kr2 = k_rot_tp[..., :half_r], k_rot_tp[..., half_r:]
    k_rr[..., :rotary_dim] = (k_rot_tp * cos_dup + torch.cat([-kr2, kr1], dim=-1) * sin_dup)

    # SDPA
    n_groups_r = heads_per_tp // kv_heads_per_tp
    k_attn_r = k_rr.unsqueeze(2).expand(-1, -1, n_groups_r, -1, -1).reshape(B, heads_per_tp, T, head_dim)
    v_attn_r = v_r.unsqueeze(2).expand(-1, -1, n_groups_r, -1, -1).reshape(B, heads_per_tp, T, head_dim)
    q_attn_r = q_rr.transpose(1, 2)
    sdpa_r = F.scaled_dot_product_attention(
        q_attn_r.float(), k_attn_r.float(), v_attn_r.float(), is_causal=True, scale=scale
    ).transpose(1, 2).reshape(B, T, heads_per_tp * head_dim)

    gate_flat_r = gate_r.reshape(B, T, heads_per_tp * head_dim)
    gated_r = sdpa_r.float() * torch.sigmoid(gate_flat_r.float())

    tp_sdpa_outputs.append(sdpa_r)
    tp_gated_outputs.append(gated_r)

# Compare SDPA: concatenate TP outputs and compare with full SDPA output
sdpa_tp_concat = torch.cat(tp_sdpa_outputs, dim=-1)
sdpa_diff = (sdpa_tp_concat - attn_out_full).abs()
print(f"SDPA output: TP concat norm={sdpa_tp_concat.norm():.4f}, Full norm={attn_out_full.norm():.4f}")
print(f"  Diff: max={sdpa_diff.max():.6f}, mean={sdpa_diff.mean():.6f}")

# Compare per-token
for t in range(T):
    sdpa_tp_t = sdpa_tp_concat[0, t, :]
    sdpa_full_t = attn_out_full[0, t, :]
    d_t = (sdpa_tp_t - sdpa_full_t).abs()
    print(f"  Token {t}: TP={sdpa_tp_t.norm():.4f}, Full={sdpa_full_t.norm():.4f}, diff={d_t.norm():.4f}")

# Compare gated
gated_tp_concat = torch.cat(tp_gated_outputs, dim=-1)
gated_diff = (gated_tp_concat - attn_gated_full).abs()
print(f"\nGated output: TP concat norm={gated_tp_concat.norm():.4f}, Full norm={attn_gated_full.norm():.4f}")
print(f"  Diff: max={gated_diff.max():.6f}, mean={gated_diff.mean():.6f}")

# Check per-head comparison
# Full SDPA has 24 heads. Rank 0 has heads 0:6.
for r in range(TP):
    full_start = r * heads_per_tp
    full_end = (r + 1) * heads_per_tp
    tp_slice = tp_sdpa_outputs[r]  # [1, 7, 1536]
    full_slice = attn_out_full[:, :, full_start*head_dim:full_end*head_dim]  # [1, 7, 1536]
    diff_slice = (tp_slice - full_slice).abs()
    print(f"\nRank {r} SDPA slice (heads {full_start}:{full_end-1}):")
    print(f"  TP norm={tp_slice.norm():.4f}, Full slice norm={full_slice.norm():.4f}")
    print(f"  Diff: max={diff_slice.max():.6f}, mean={diff_slice.mean():.6f}")

print(f"\n=== Conclusion ===")
print(f"Check SDPA per-rank slicing above. If TP concat == Full SDPA but")
print(f"o_proj differs, then the issue is in o_proj weight loading/reduction.")

