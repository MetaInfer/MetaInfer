#!/usr/bin/env python3
"""GPU vs CPU FullAttention layer 3 substep comparison.

Strategy:
1. On GPU (all 4 ranks): run layers 0-2 to get layer 3 input
2. Rank 0 gathers the normed hidden state from all ranks
3. Rank 0 computes GPU FullAttention intermediates (rank's shard)
4. Rank 0 computes CPU FullAttention intermediates (full, unsharded)
5. Compare at each substep
"""
import os, sys, torch, json, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch.nn.functional as F
import torch.distributed as dist

from engine.models.qwen import QwenTPConfig, QwenForCausalLMTP, load_weights
from engine.tp_layers.distributed import init_tp_distributed, get_tp_rank, get_tp_size
from engine.kernels.rms_norm import fused_add_rms_norm

init_tp_distributed()
rank = get_tp_rank()
tp_size = get_tp_size()
model_dir = os.environ['MODEL_DIR']
device = f'cuda:{rank}'

# Load model
cfg = QwenTPConfig(model_dir)
model = QwenForCausalLMTP(cfg)
model = model.to(torch.bfloat16).to(device)
model = load_weights(model, model_dir)
model.eval()

# ============================================================
# CPU weight loading (all ranks do this to get config)
# ============================================================
from safetensors import safe_open
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
mrope_section = rp.get('mrope_section', None)
rope_theta = tc.get('rope_theta') or rp.get('rope_theta', 1000000.0)

def load_cpu(key):
    fname = wm[key]
    fpath = os.path.join(model_dir, fname)
    with safe_open(fpath, framework='pt', device='cpu') as sf:
        return sf.get_tensor(key)

def qwen35_rms_norm(x, w):
    rstd = 1.0 / torch.sqrt(x.float().pow(2).mean(-1, keepdim=True) + eps)
    return (x.float() * rstd * (1.0 + w.float())).float()

# MRoPE cache
num_sections = len(mrope_section)
half_dim = rotary_dim // 2
section_freqs = []
t = torch.arange(tc['max_position_embeddings'], dtype=torch.float32)
for sec_size in mrope_section:
    inv_freq = 1.0 / (rope_theta ** (torch.arange(0, sec_size, dtype=torch.float32) * 2 / rotary_dim))
    freqs_s = torch.einsum("i,j->ij", t, inv_freq)
    section_freqs.append(freqs_s)
pairs = []
for i in range(half_dim):
    s = i % num_sections
    idx = i // num_sections
    pairs.append(section_freqs[s][:, idx])
freqs_cpu = torch.stack(pairs, dim=-1)
cos_sin_cpu = torch.cat((freqs_cpu.cos(), freqs_cpu.sin()), dim=-1)

@torch.inference_mode()
def run():
    tokens = [108618, 102066, 137351, 105017, 100462, 106808, 103105]
    input_ids = torch.tensor([tokens], dtype=torch.long, device=device)
    S = len(tokens)
    positions = torch.arange(S, dtype=torch.int64, device=device)

    # Run layers 0-2 on GPU
    hidden_states = model.embed_tokens(input_ids)
    residual = None
    for lx in range(3):
        hidden_states, residual = model.layers[lx](hidden_states, positions, S, residual)

    # Capture pre-norm states
    hs_in = hidden_states.clone()
    resid_in = residual.clone()

    # Apply layer 3 input_layernorm (fused_add_rms_norm)
    layer3 = model.layers[3]
    fa = layer3.self_attn
    w_ln = layer3.input_layernorm._effective_weight()
    hs_work = hs_in.clone()
    fused_add_rms_norm(hs_work, resid_in, w_ln, layer3.input_layernorm.eps)
    # hs_work = rms_norm(hs_in + resid_in)

    B, T = 1, S

    # GPU projections
    q_gpu = fa.q_proj(hs_work)        # ColumnParallel → [1, 7, 1536] (6 heads * 256)
    k_gpu = fa.k_proj(hs_work)        # ColumnParallel → [1, 7, 256] (1 head * 256)
    v_gpu = fa.v_proj(hs_work)        # ColumnParallel → [1, 7, 256]
    gate_gpu = fa.q_gate_proj(hs_work)  # ColumnParallel → [1, 7, 1536]

    # Reshape to heads
    num_h = cfg.heads_per_rank  # 6
    num_kvh = cfg.kv_heads_per_rank  # 1
    q_gpu_h = q_gpu.view(B, T, num_h, head_dim)  # [1, 7, 6, 256]
    k_gpu_h = k_gpu.view(B, T, num_kvh, head_dim)  # [1, 7, 1, 256]
    v_gpu_h = v_gpu.view(B, T, num_kvh, head_dim)
    gate_gpu_h = gate_gpu.view(B, T, num_h, head_dim)

    # GPU Q/K norms
    q_gpu_normed = fa.q_norm(q_gpu_h)
    k_gpu_normed = fa.k_norm(k_gpu_h)

    # GPU MRoPE (capture after RoPE by running the internal method)
    q_gpu_rope = q_gpu_normed.clone()
    k_gpu_rope = k_gpu_normed.clone()
    # Ensure cos/sin cache is on GPU (lazy init, only triggered during forward())
    fa._ensure_cos_sin_gpu(device)
    gpu_cos_sin = fa._cos_sin_cache_gpu
    # Manual MRoPE application matching GPU code
    cos = gpu_cos_sin[positions, :rotary_dim//2].view(1, S, 1, rotary_dim//2)
    sin = gpu_cos_sin[positions, rotary_dim//2:].view(1, S, 1, rotary_dim//2)
    cos_dup = torch.cat([cos, cos], dim=-1)
    sin_dup = torch.cat([sin, sin], dim=-1)

    q_rot = q_gpu_rope[..., :rotary_dim].float()
    half_r = rotary_dim // 2
    q1, q2 = q_rot[..., :half_r], q_rot[..., half_r:]
    q_gpu_rope[..., :rotary_dim] = (q_rot * cos_dup + torch.cat([-q2, q1], dim=-1) * sin_dup).to(q_gpu_rope.dtype)

    k_rot = k_gpu_rope[..., :rotary_dim].float()
    k1, k2 = k_rot[..., :half_r], k_rot[..., half_r:]
    k_gpu_rope[..., :rotary_dim] = (k_rot * cos_dup + torch.cat([-k2, k1], dim=-1) * sin_dup).to(k_gpu_rope.dtype)

    # GPU SDPA
    num_heads_local = cfg.heads_per_rank  # 6
    num_kv_heads_local = cfg.kv_heads_per_rank  # 1
    n_groups = num_heads_local // num_kv_heads_local  # 6

    q_attn = q_gpu_rope.transpose(1, 2)  # [1, 6, 7, 256]
    k_attn = k_gpu_rope.unsqueeze(2).expand(-1, -1, n_groups, -1, -1).reshape(B, num_heads_local, T, head_dim)
    v_attn = v_gpu_h.unsqueeze(2).expand(-1, -1, n_groups, -1, -1).reshape(B, num_heads_local, T, head_dim)

    scale = head_dim ** -0.5
    attn_out_gpu = F.scaled_dot_product_attention(
        q_attn.float(), k_attn.float(), v_attn.float(),
        is_causal=True, scale=scale
    ).transpose(1, 2).reshape(B, T, num_heads_local * head_dim)

    # GPU gate + output
    gate_flat = gate_gpu_h.reshape(B, T, num_heads_local * head_dim)
    attn_gated_gpu = attn_out_gpu.float() * torch.sigmoid(gate_flat.float())

    # GPU o_proj (RowParallel → all_reduce inside)
    o_proj_out_gpu = fa.o_proj(F.linear.__func__ if False else None)
    # Actually just call o_proj directly
    o_proj_out_gpu = fa.o_proj(attn_gated_gpu.to(torch.bfloat16))

    # GPU post_ln + MLP
    w_pln = layer3.post_attention_layernorm._effective_weight()
    resid_post_attn = resid_in + o_proj_out_gpu.to(torch.bfloat16)
    hs_mlpin = hs_in.clone()
    fused_add_rms_norm(hs_mlpin, resid_in, w_pln, layer3.post_attention_layernorm.eps)
    # Actually: fused_add_rms_norm modifies input to be rms_norm(residual + input)
    # Let me use resid_post_attn as the input and hs_mlpin gets overwritten
    hs_mlpin2 = torch.zeros_like(hs_in)
    resid_copy = resid_in.clone()
    o_proj_b = o_proj_out_gpu.to(torch.bfloat16)
    fused_add_rms_norm(hs_mlpin2, resid_copy, w_pln, layer3.post_attention_layernorm.eps)
    # Hmm, this is tricky. Let me just compute manually

    if rank == 0:
        # ============================================================
        # CPU COMPUTATION for rank 0's head shard
        # ============================================================
        # Load layer 3 weights
        prefix = 'model.language_model.layers.3.'
        w_input_ln = load_cpu(prefix + 'input_layernorm.weight').float()
        w_post_ln = load_cpu(prefix + 'post_attention_layernorm.weight').float()
        w_q_fused = load_cpu(prefix + 'self_attn.q_proj.weight').float()
        w_k = load_cpu(prefix + 'self_attn.k_proj.weight').float()
        w_v = load_cpu(prefix + 'self_attn.v_proj.weight').float()
        w_o = load_cpu(prefix + 'self_attn.o_proj.weight').float()
        w_qn = load_cpu(prefix + 'self_attn.q_norm.weight').float()
        w_kn = load_cpu(prefix + 'self_attn.k_norm.weight').float()

        full_q_size = num_heads * head_dim  # 6144
        w_q_proj = w_q_fused[:full_q_size, :]   # [6144, 5120]
        w_gate_proj = w_q_fused[full_q_size:, :]  # [6144, 5120]

        # CPU input: resid + hs
        resid_cpu = resid_in.cpu().float() + hs_in.cpu().float()
        hs_normed_cpu = qwen35_rms_norm(resid_cpu, w_input_ln)
        B_cpu, T_cpu = 1, S

        # Check the normed input matches
        gpu_normed = hs_work.cpu().float()
        cpu_normed = hs_normed_cpu
        normed_diff = (gpu_normed - cpu_normed).abs()
        print(f"=== Layer 3 input (normed) comparison ===")
        print(f"  GPU norm: {gpu_normed.norm():.4f}, CPU norm: {cpu_normed.norm():.4f}")
        print(f"  Diff: max={normed_diff.max():.6f}, mean={normed_diff.mean():.6f}")

        # CPU Q, Gate, K, V
        q_cpu_full = F.linear(hs_normed_cpu, w_q_proj)    # [1, 7, 6144]
        gate_cpu_full = F.linear(hs_normed_cpu, w_gate_proj)  # [1, 7, 6144]
        k_cpu_full = F.linear(hs_normed_cpu, w_k)          # [1, 7, 1024]
        v_cpu_full = F.linear(hs_normed_cpu, w_v)          # [1, 7, 1024]

        q_cpu_h = q_cpu_full.view(B_cpu, T_cpu, num_heads, head_dim)
        k_cpu_h = k_cpu_full.view(B_cpu, T_cpu, num_kv_heads, head_dim)
        v_cpu_h = v_cpu_full.view(B_cpu, T_cpu, num_kv_heads, head_dim)
        gate_cpu_h = gate_cpu_full.view(B_cpu, T_cpu, num_heads, head_dim)

        # Compare Q proj: rank 0 has heads 0:6
        q_cpu_r0 = q_cpu_h[:, :, :num_h, :].reshape(B_cpu, T_cpu, -1)
        q_gpu_r0 = q_gpu_h.cpu().float().reshape(B_cpu, T_cpu, -1)
        q_diff = (q_gpu_r0 - q_cpu_r0).abs()
        print(f"\n=== Q projection (heads 0:6) ===")
        print(f"  GPU norm: {q_gpu_r0.norm():.4f}, CPU norm: {q_cpu_r0.norm():.4f}")
        print(f"  Diff: max={q_diff.max():.6f}, mean={q_diff.mean():.6f}")

        # Compare K proj: rank 0 has KV head 0
        k_cpu_r0 = k_cpu_h[:, :, 0:1, :].reshape(B_cpu, T_cpu, -1)
        k_gpu_r0 = k_gpu_h.cpu().float().reshape(B_cpu, T_cpu, -1)
        k_diff = (k_gpu_r0 - k_cpu_r0).abs()
        print(f"\n=== K projection (head 0) ===")
        print(f"  GPU norm: {k_gpu_r0.norm():.4f}, CPU norm: {k_cpu_r0.norm():.4f}")
        print(f"  Diff: max={k_diff.max():.6f}, mean={k_diff.mean():.6f}")

        # Compare V proj
        v_cpu_r0 = v_cpu_h[:, :, 0:1, :].reshape(B_cpu, T_cpu, -1)
        v_gpu_r0 = v_gpu_h.cpu().float().reshape(B_cpu, T_cpu, -1)
        v_diff = (v_gpu_r0 - v_cpu_r0).abs()
        print(f"\n=== V projection (head 0) ===")
        print(f"  GPU norm: {v_gpu_r0.norm():.4f}, CPU norm: {v_cpu_r0.norm():.4f}")
        print(f"  Diff: max={v_diff.max():.6f}, mean={v_diff.mean():.6f}")

        # Compare Gate proj
        gate_cpu_r0 = gate_cpu_h[:, :, :num_h, :].reshape(B_cpu, T_cpu, -1)
        gate_gpu_r0 = gate_gpu_h.cpu().float().reshape(B_cpu, T_cpu, -1)
        gate_diff = (gate_gpu_r0 - gate_cpu_r0).abs()
        print(f"\n=== Gate projection (heads 0:6) ===")
        print(f"  GPU norm: {gate_gpu_r0.norm():.4f}, CPU norm: {gate_cpu_r0.norm():.4f}")
        print(f"  Diff: max={gate_diff.max():.6f}, mean={gate_diff.mean():.6f}")

        # CPU Q/K norms
        q_cpu_normed = qwen35_rms_norm(q_cpu_h.float(), w_qn.unsqueeze(0).unsqueeze(0))
        k_cpu_normed = qwen35_rms_norm(k_cpu_h.float(), w_kn.unsqueeze(0).unsqueeze(0))

        q_cpu_n_r0 = q_cpu_normed[:, :, :num_h, :].reshape(B_cpu, T_cpu, -1)
        q_gpu_n_r0 = q_gpu_normed.cpu().float().reshape(B_cpu, T_cpu, -1)
        qn_diff = (q_gpu_n_r0 - q_cpu_n_r0).abs()
        print(f"\n=== Q norm (after Qwen3_5RMSNorm, heads 0:6) ===")
        print(f"  GPU norm: {q_gpu_n_r0.norm():.4f}, CPU norm: {q_cpu_n_r0.norm():.4f}")
        print(f"  Diff: max={qn_diff.max():.6f}, mean={qn_diff.mean():.6f}")

        k_cpu_n_r0 = k_cpu_normed[:, :, 0:1, :].reshape(B_cpu, T_cpu, -1)
        k_gpu_n_r0 = k_gpu_normed.cpu().float().reshape(B_cpu, T_cpu, -1)
        kn_diff = (k_gpu_n_r0 - k_cpu_n_r0).abs()
        print(f"K norm (head 0): GPU={k_gpu_n_r0.norm():.4f} CPU={k_cpu_n_r0.norm():.4f} diff={kn_diff.max():.6f}")

        # CPU MRoPE
        q_cpu_rope = q_cpu_normed.clone()
        k_cpu_rope = k_cpu_normed.clone()
        pos_c = torch.arange(S, dtype=torch.int64)
        cos_c = cos_sin_cpu[pos_c, :rotary_dim//2].view(1, S, 1, rotary_dim//2)
        sin_c = cos_sin_cpu[pos_c, rotary_dim//2:].view(1, S, 1, rotary_dim//2)
        cos_dup_c = torch.cat([cos_c, cos_c], dim=-1)
        sin_dup_c = torch.cat([sin_c, sin_c], dim=-1)

        q_rot_c = q_cpu_rope[..., :rotary_dim].float()
        half_c = rotary_dim // 2
        q1_c, q2_c = q_rot_c[..., :half_c], q_rot_c[..., half_c:]
        q_cpu_rope[..., :rotary_dim] = (q_rot_c * cos_dup_c + torch.cat([-q2_c, q1_c], dim=-1) * sin_dup_c)

        k_rot_c = k_cpu_rope[..., :rotary_dim].float()
        k1_c, k2_c = k_rot_c[..., :half_c], k_rot_c[..., half_c:]
        k_cpu_rope[..., :rotary_dim] = (k_rot_c * cos_dup_c + torch.cat([-k2_c, k1_c], dim=-1) * sin_dup_c)

        # Compare MRoPE
        q_rope_r0 = q_cpu_rope[:, :, :num_h, :].reshape(B_cpu, T_cpu, -1)
        q_gpu_rope_r0 = q_gpu_rope.cpu().float().reshape(B_cpu, T_cpu, -1)
        qrope_diff = (q_gpu_rope_r0 - q_rope_r0).abs()
        print(f"\n=== Q after MRoPE (heads 0:6) ===")
        print(f"  GPU norm: {q_gpu_rope_r0.norm():.4f}, CPU norm: {q_rope_r0.norm():.4f}")
        print(f"  Diff: max={qrope_diff.max():.6f}, mean={qrope_diff.mean():.6f}")

        k_rope_r0 = k_cpu_rope[:, :, 0:1, :].reshape(B_cpu, T_cpu, -1)
        k_gpu_rope_r0 = k_gpu_rope.cpu().float().reshape(B_cpu, T_cpu, -1)
        krope_diff = (k_gpu_rope_r0 - k_rope_r0).abs()
        print(f"K after MRoPE (head 0): GPU={k_gpu_rope_r0.norm():.4f} CPU={k_rope_r0.norm():.4f} diff={krope_diff.max():.6f}")

        # CPU SDPA (full 24 heads with GQA)
        n_groups_full = num_heads // num_kv_heads  # 6
        k_attn_cpu = k_cpu_rope.unsqueeze(2).expand(-1, -1, n_groups_full, -1, -1).reshape(B_cpu, T_cpu, num_heads, head_dim)
        v_attn_cpu = v_cpu_h.unsqueeze(2).expand(-1, -1, n_groups_full, -1, -1).reshape(B_cpu, T_cpu, num_heads, head_dim)
        q_attn_cpu = q_cpu_rope.transpose(1, 2)
        k_attn_cpu = k_attn_cpu.transpose(1, 2)
        v_attn_cpu = v_attn_cpu.transpose(1, 2)

        attn_out_cpu = F.scaled_dot_product_attention(
            q_attn_cpu.float(), k_attn_cpu.float(), v_attn_cpu.float(),
            is_causal=True, scale=scale
        ).transpose(1, 2).reshape(B_cpu, T_cpu, num_heads * head_dim)

        # Compare SDPA output: rank 0 has heads 0:6
        attn_cpu_r0 = attn_out_cpu[:, :, :num_h * head_dim]
        attn_gpu_r0 = attn_out_gpu.cpu().float()
        attn_diff = (attn_gpu_r0 - attn_cpu_r0).abs()
        print(f"\n=== SDPA output (heads 0:6) ===")
        print(f"  GPU norm: {attn_gpu_r0.norm():.4f}, CPU norm: {attn_cpu_r0.norm():.4f}")
        print(f"  Diff: max={attn_diff.max():.6f}, mean={attn_diff.mean():.6f}")

        # CPU gate
        gate_cpu_flat = gate_cpu_h.reshape(B_cpu, T_cpu, num_heads * head_dim)
        gate_cpu_sig = torch.sigmoid(gate_cpu_flat.float())
        attn_gated_cpu = attn_out_cpu.float() * gate_cpu_sig

        # Compare gated output (rank 0)
        gated_cpu_r0 = attn_gated_cpu[:, :, :num_h * head_dim]
        gated_gpu_r0 = attn_gated_gpu.cpu().float()
        gated_diff = (gated_gpu_r0 - gated_cpu_r0).abs()
        print(f"\n=== After output gate (heads 0:6) ===")
        print(f"  GPU norm: {gated_gpu_r0.norm():.4f}, CPU norm: {gated_cpu_r0.norm():.4f}")
        print(f"  Diff: max={gated_diff.max():.6f}, mean={gated_diff.mean():.6f}")

        # CPU o_proj: full [5120, 6144] x [1, 7, 6144] = [1, 7, 5120]
        o_proj_cpu = F.linear(attn_gated_cpu.float(), w_o.float())

        # GPU o_proj: RowParallel → output is [1, 7, 5120] (already reduced)
        # But we can't easily get the pre-reduce output... the RowParallel does all_reduce internally
        # Let me compare the final o_proj output
        o_gpu_full = o_proj_out_gpu.cpu().float()
        o_diff = (o_gpu_full - o_proj_cpu).abs()
        print(f"\n=== o_proj output (full, all_reduced) ===")
        print(f"  GPU norm: {o_gpu_full.norm():.4f}, CPU norm: {o_proj_cpu.norm():.4f}")
        print(f"  Diff: max={o_diff.max():.6f}, mean={o_diff.mean():.6f}")

        # CPU post_ln + MLP
        cpu_resid = resid_cpu + o_proj_cpu.float()  # accumulate attention output
        hs_mlp_in_cpu = qwen35_rms_norm(cpu_resid, w_post_ln)

        w_gate_proj = load_cpu(prefix + 'mlp.gate_proj.weight').float()
        w_up_proj = load_cpu(prefix + 'mlp.up_proj.weight').float()
        w_down_proj = load_cpu(prefix + 'mlp.down_proj.weight').float()

        gate_h_cpu = F.linear(hs_mlp_in_cpu, w_gate_proj)
        up_h_cpu = F.linear(hs_mlp_in_cpu, w_up_proj)
        mlp_cpu = F.linear(F.silu(gate_h_cpu) * up_h_cpu.float(), w_down_proj).float()

        # Compare with GPU MLP output
        # GPU: run post_ln + MLP
        hs_post = hs_in.clone()
        resid_for_mlp = resid_in.clone()
        fused_add_rms_norm(hs_post, resid_for_mlp, w_pln, layer3.post_attention_layernorm.eps)
        # Actually this is wrong, we need to add o_proj output to residual first
        # Let me redo
        resid_post_attn_gpu = resid_in + o_proj_out_gpu.to(torch.bfloat16)
        hs_post2 = hidden_states.clone()
        # fused_add_rms_norm(x, residual, weight, eps): residual += x, x = rms_norm(residual, weight, eps)
        # So if we want norm(resid_post_attn_gpu), we should call it differently
        hs_mlp_input_gpu = hs_in.clone()
        fused_add_rms_norm(hs_mlp_input_gpu, resid_post_attn_gpu, w_pln, layer3.post_attention_layernorm.eps)

        print(f"\nCPU post_ln input norm: {cpu_resid.norm():.4f}")
        print(f"GPU post_ln input: resid_post_attn norm={resid_post_attn_gpu.norm():.4f}")

        # GPU MLP
        # Actually, let me just call the MLP directly
        gate_gpu_mlp = layer3.mlp.gate_proj(hs_mlp_input_gpu)
        up_gpu_mlp = layer3.mlp.up_proj(hs_mlp_input_gpu)
        # Both are ColumnParallel
        mlp_gpu_out = layer3.mlp.down_proj(F.silu(gate_gpu_mlp) * up_gpu_mlp.float())

        mlp_gpu_cpu = mlp_gpu_out.cpu().float()
        mlp_diff = (mlp_gpu_cpu - mlp_cpu).abs()
        print(f"\n=== Layer 3 MLP output ===")
        print(f"  GPU norm: {mlp_gpu_cpu.norm():.4f}, CPU norm: {mlp_cpu.norm():.4f}")
        print(f"  Diff: max={mlp_diff.max():.6f}, mean={mlp_diff.mean():.6f}")
        print(f"  (Expected CPU ref norm from diag_cpu_layers05: 22.5283)")

        # Also check GPU vs GPU forward through actual layer 3
        # Let's compare what model.layers[3] actually outputs
        hs_test, resid_test = model.layers[3](hs_in, positions, S, resid_in)
        print(f"\n=== Actual layer3.forward() output ===")
        print(f"  GPU hs (mlp_out) norm: {hs_test.norm():.4f}")
        print(f"  CPU mlp_out norm: {mlp_cpu.norm():.4f}")
        l3_diff = (hs_test.cpu().float() - mlp_cpu).abs()
        print(f"  Diff: max={l3_diff.max():.6f}, mean={l3_diff.mean():.6f}")

if __name__ == '__main__':
    run()
    dist.barrier()
    if rank == 0:
        dist.destroy_process_group()
