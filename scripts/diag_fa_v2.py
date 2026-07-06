#!/usr/bin/env python3
"""Focused diagnostic: compare GPU layer 3 normed input and Q/K/V projections with CPU."""
import os, sys, torch, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch.nn.functional as F

from engine.models.qwen import QwenTPConfig, QwenForCausalLMTP, load_weights
from engine.tp_layers.distributed import init_tp_distributed, get_tp_rank, get_tp_size

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

# Load model
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

w_q = load_cpu('model.language_model.layers.3.self_attn.q_proj.weight').float()
w_k = load_cpu('model.language_model.layers.3.self_attn.k_proj.weight').float()
w_v = load_cpu('model.language_model.layers.3.self_attn.v_proj.weight').float()
w_o = load_cpu('model.language_model.layers.3.self_attn.o_proj.weight').float()
w_qn = load_cpu('model.language_model.layers.3.self_attn.q_norm.weight').float()
w_kn = load_cpu('model.language_model.layers.3.self_attn.k_norm.weight').float()
w_iln = load_cpu('model.language_model.layers.3.input_layernorm.weight').float()

def rms_norm_ref(x, w, eps=eps):
    rstd = 1.0 / torch.sqrt(x.float().pow(2).mean(-1, keepdim=True) + eps)
    return (x.float() * rstd * (1.0 + w.float())).to(x.dtype)

with torch.inference_mode():
    # Run layers 0-2
    hidden_states = model.embed_tokens(input_ids)
    residual = None
    for lx in range(3):
        hidden_states, residual = model.layers[lx](hidden_states, positions, S, residual)

    hs_in_gpu = hidden_states.clone()
    resid_in_gpu = residual.clone()

    # Layer 3: fused_add_rms_norm
    from engine.kernels.rms_norm import fused_add_rms_norm
    hs_work = hs_in_gpu.clone()
    fused_add_rms_norm(hs_work, resid_in_gpu,
                       model.layers[3].input_layernorm._effective_weight(),
                       model.layers[3].input_layernorm.eps)

    # Q/K/V projections (GPU)
    fa = model.layers[3].self_attn
    fa._ensure_cos_sin_gpu(device)
    q_gpu = fa.q_proj(hs_work)
    k_gpu = fa.k_proj(hs_work)
    v_gpu = fa.v_proj(hs_work)
    gate_gpu = fa.q_gate_proj(hs_work)

    q_gpu_h = q_gpu.view(B, S, hpr, head_dim)
    k_gpu_h = k_gpu.view(B, S, kvpr, head_dim)
    v_gpu_h = v_gpu.view(B, S, kvpr, head_dim)
    gate_gpu_h = gate_gpu.view(B, S, hpr, head_dim)

    q_gpu_normed = fa.q_norm(q_gpu_h)
    k_gpu_normed = fa.k_norm(k_gpu_h)

    if rank == 0:
        # Verify weights
        print("=== Weight Verification ===")
        gpu_qw = fa.q_proj.weight.data.cpu().float()
        cpu_qw = w_q[:full_q_size][:hpr*head_dim, :]
        qw_diff = (gpu_qw - cpu_qw).abs()
        print(f"q_proj weight: GPU shape={list(gpu_qw.shape)} CPU shape={list(cpu_qw.shape)}")
        print(f"  max diff={qw_diff.max():.8f} mean diff={qw_diff.mean():.8f}")
        print(f"  GPU norm={gpu_qw.norm():.4f} CPU norm={cpu_qw.norm():.4f}")
        if qw_diff.max() > 0.01:
            print(f"  MISMATCH: showing first 5 diff positions:")
            idxs = torch.nonzero(qw_diff > 0.01)[:5]
            for idx in idxs:
                i, j = idx.tolist()
                print(f"    [{i},{j}]: GPU={gpu_qw[i,j]:.6f} CPU={cpu_qw[i,j]:.6f}")

        gpu_ow = fa.o_proj.weight.data.cpu().float()
        cpu_ow = w_o[:, :hpr*head_dim]
        ow_diff = (gpu_ow - cpu_ow).abs()
        print(f"o_proj weight: GPU shape={list(gpu_ow.shape)} CPU shape={list(cpu_ow.shape)}")
        print(f"  max diff={ow_diff.max():.8f} mean diff={ow_diff.mean():.8f}")

        gpu_qn = fa.q_norm.weight.data.cpu().float()
        qn_diff = (gpu_qn - w_qn).abs()
        print(f"q_norm weight: max diff={qn_diff.max():.8f}")

        gpu_iln = model.layers[3].input_layernorm.weight.data.cpu().float()
        iln_diff = (gpu_iln - w_iln).abs()
        print(f"input_layernorm weight: max diff={iln_diff.max():.8f}")

        # Verify normed input
        print("\n=== Normed Input ===")
        hs_cpu_in = hs_in_gpu.cpu().float()
        resid_cpu_in = resid_in_gpu.cpu().float()
        resid_expected = resid_cpu_in + hs_cpu_in
        hs_normed_cpu = rms_norm_ref(resid_expected, w_iln)
        hs_work_cpu = hs_work.cpu().float()
        diff_norm = (hs_work_cpu - hs_normed_cpu).abs()
        print(f"GPU norm={hs_work_cpu.norm():.4f} CPU norm={hs_normed_cpu.norm():.4f}")
        print(f"Diff: max={diff_norm.max():.6f} mean={diff_norm.mean():.6f}")

        # Verify Q/K/V projections
        print("\n=== Q Projection (rank 0) ===")
        q_full_cpu = F.linear(hs_normed_cpu, w_q)
        q_cpu = q_full_cpu[:, :, :full_q_size].view(B, S, num_heads, head_dim)
        q_cpu_r0 = q_cpu[:, :, :hpr, :].float()
        q_gpu_cpu = q_gpu_h.cpu().float()
        q_diff = (q_gpu_cpu - q_cpu_r0).abs()
        print(f"GPU norm={q_gpu_cpu.norm():.4f} CPU norm={q_cpu_r0.norm():.4f}")
        print(f"Diff: max={q_diff.max():.6f} mean={q_diff.mean():.6f}")

        # Verify Q using GPU weight directly (bypass ColumnParallelLinear)
        print("\n=== Q Projection (manual, bypass ColumnParallelLinear) ===")
        q_manual = F.linear(hs_work_cpu, gpu_qw)  # [1,7,1536] using GPU weight
        q_manual_h = q_manual.view(B, S, hpr, head_dim).float()
        q_manual_diff = (q_gpu_cpu - q_manual_h).abs()
        print(f"GPU Q (via fa.q_proj) norm={q_gpu_cpu.norm():.4f}")
        print(f"Manual Q (hs_work @ gpu_weight) norm={q_manual_h.norm():.4f}")
        print(f"Diff: max={q_manual_diff.max():.6f} mean={q_manual_diff.mean():.6f}")

        # Verify Q using CPU weight
        q_cpu_wt = F.linear(hs_work_cpu, cpu_qw)  # [1,7,1536] using CPU weight
        q_cpu_wt_h = q_cpu_wt.view(B, S, hpr, head_dim).float()
        q_cpu_wt_diff = (q_gpu_cpu - q_cpu_wt_h).abs()
        print(f"Q (hs_work @ cpu_weight) norm={q_cpu_wt_h.norm():.4f}")
        print(f"Diff vs GPU: max={q_cpu_wt_diff.max():.6f} mean={q_cpu_wt_diff.mean():.6f}")

        # Check if the issue is normed input or weight
        print("\n=== Cross Check ===")
        # GPU weight @ CPU normed input
        q_gpu_wt_cpu_in = F.linear(hs_normed_cpu.float().to(device), fa.q_proj.weight.data).cpu().float()
        q_gpu_wt_cpu_in_h = q_gpu_wt_cpu_in.view(B, S, hpr, head_dim)
        print(f"GPU_wt@CPU_normed: norm={q_gpu_wt_cpu_in_h.norm():.4f}")

        # CPU weight @ GPU normed input (hs_work)
        q_cpu_wt_gpu_in = F.linear(hs_work_cpu, cpu_qw)
        q_cpu_wt_gpu_in_h = q_cpu_wt_gpu_in.view(B, S, hpr, head_dim)
        print(f"CPU_wt@GPU_normed: norm={q_cpu_wt_gpu_in_h.norm():.4f}")

        print(f"\nSummary: GPU Q norm = {q_gpu_cpu.norm():.4f}")
        print(f"  Q via CPU_wt + CPU_normed = {q_cpu_r0.norm():.4f}")
        print(f"  Q via GPU_wt + CPU_normed = {q_gpu_wt_cpu_in_h.norm():.4f}")
        print(f"  Q via CPU_wt + GPU_normed = {q_cpu_wt_gpu_in_h.norm():.4f}")

import torch.distributed as dist
dist.barrier()
if rank == 0:
    dist.destroy_process_group()
