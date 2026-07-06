#!/usr/bin/env python3
"""V3: pinpoint the exact source of normed-input divergence with step-by-step tracing."""
import os, sys, torch, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch.nn.functional as F

from engine.models.qwen import QwenTPConfig, QwenForCausalLMTP, load_weights
from engine.tp_layers.distributed import init_tp_distributed, get_tp_rank, get_tp_size
from engine.kernels.rms_norm import fused_add_rms_norm

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

w_iln_raw = load_cpu('model.language_model.layers.3.input_layernorm.weight').float()
w_iln_eff = 1.0 + w_iln_raw  # Qwen3_5RMSNorm effective weight

def rms_norm_ref(x, w, eps_val=eps):
    rstd = 1.0 / torch.sqrt(x.float().pow(2).mean(-1, keepdim=True) + eps_val)
    return (x.float() * rstd * (1.0 + w.float())).to(x.dtype)

# Also manual re-implementation of fused_add_rms_norm for comparison
def fused_add_rms_norm_manual(input_t, residual_t, weight_t, eps_val):
    """Manual reference matching the vLLM kernel behavior."""
    residual_t.add_(input_t)
    x_fp32 = residual_t.to(torch.float32)
    rms = torch.sqrt(x_fp32.pow(2).mean(-1, keepdim=True) + eps_val)
    input_t.copy_((weight_t.float() * (x_fp32 / rms)).to(input_t.dtype))

with torch.inference_mode():
    # Run layers 0-2
    hidden_states = model.embed_tokens(input_ids)
    residual = None
    for lx in range(3):
        hidden_states, residual = model.layers[lx](hidden_states, positions, S, residual)

    if rank == 0:
        print(f"After layers 0-2: hs norm={hidden_states.float().norm():.4f}, "
              f"residual norm={residual.float().norm():.4f}")

    hs_in_gpu = hidden_states.clone()
    resid_in_gpu = residual.clone()

    # ==== Method 1: fused_add_rms_norm (the wrapper from engine) ====
    hs_wrapper = hs_in_gpu.clone()
    resid_wrapper = resid_in_gpu.clone()
    ln_layer = model.layers[3].input_layernorm
    eff_w = ln_layer._effective_weight()
    ln_eps = ln_layer.eps

    if rank == 0:
        iln_gpu_w = ln_layer.weight.data.cpu().float()
        iln_diff = (iln_gpu_w - w_iln_raw).abs()
        print(f"\ninput_layernorm weight on GPU: shape={list(iln_gpu_w.shape)}")
        print(f"  max diff vs safetensors: {iln_diff.max():.10f}")
        print(f"  first 5 values: {iln_gpu_w[:5].tolist()}")
        print(f"  effective weight first 5: {eff_w.cpu().float()[:5].tolist()}")
        print(f"  eps: GPU={ln_eps}, config={eps}")

    fused_add_rms_norm(hs_wrapper, resid_wrapper, eff_w, ln_eps)

    if rank == 0:
        print(f"\n=== Method 1: fused_add_rms_norm (engine wrapper) ===")
        print(f"  output norm={hs_wrapper.float().norm():.4f}")
        print(f"  residual norm={resid_wrapper.float().norm():.4f}")

    # ==== Method 2: Direct vLLM kernel call ====
    hs_vllm = hs_in_gpu.clone()
    resid_vllm = resid_in_gpu.clone()
    from vllm._custom_ops import fused_add_rms_norm as _vllm_fn
    try:
        _vllm_fn(hs_vllm, resid_vllm, eff_w, ln_eps)
        if rank == 0:
            print(f"\n=== Method 2: Direct vLLM kernel ===")
            print(f"  output norm={hs_vllm.float().norm():.4f}")
            print(f"  residual norm={resid_vllm.float().norm():.4f}")
            diff_1_2 = (hs_wrapper.float() - hs_vllm.float()).abs()
            print(f"  diff vs Method 1: max={diff_1_2.max():.6f}")
    except Exception as e:
        if rank == 0:
            print(f"\n=== Method 2: Direct vLLM kernel FAILED: {type(e).__name__}: {e} ===")

    # ==== Method 3: Manual fallback (matching engine's fallback) ====
    hs_manual = hs_in_gpu.clone()
    resid_manual = resid_in_gpu.clone()
    fused_add_rms_norm_manual(hs_manual, resid_manual, eff_w, ln_eps)
    if rank == 0:
        print(f"\n=== Method 3: Manual fallback ===")
        print(f"  output norm={hs_manual.float().norm():.4f}")
        print(f"  residual norm={resid_manual.float().norm():.4f}")
        diff_1_3 = (hs_wrapper.float() - hs_manual.float()).abs()
        print(f"  diff vs Method 1: max={diff_1_3.max():.6f}")

    # ==== Method 4: CPU reference with raw weights (1+w) ====
    hs_cpu = hs_in_gpu.cpu().float()
    resid_cpu = resid_in_gpu.cpu().float()
    resid_cpu_sum = resid_cpu + hs_cpu
    hs_normed_raw = rms_norm_ref(resid_cpu_sum, w_iln_raw, eps)  # uses (1+w)

    if rank == 0:
        print(f"\n=== Method 4: CPU reference (1+w) ===")
        print(f"  output norm={hs_normed_raw.float().norm():.4f}")
        diff_1_4 = (hs_wrapper.cpu().float() - hs_normed_raw).abs()
        print(f"  diff vs Method 1: max={diff_1_4.max():.6f}")

    # ==== Method 5: CPU reference with effective weight (w_eff, NOT 1+w again) ====
    # rms_norm_ref already does (1+w), so if we want effective weight w_eff=1+w,
    # we need to pass w=0 so that (1+0) = 1, then multiply by w_eff
    resid_cpu_sum2 = resid_cpu + hs_cpu
    rstd = 1.0 / torch.sqrt(resid_cpu_sum2.float().pow(2).mean(-1, keepdim=True) + eps)
    hs_normed_eff = (resid_cpu_sum2.float() * rstd * w_iln_eff.float()).bfloat16()

    if rank == 0:
        print(f"\n=== Method 5: CPU with effective weight directly ===")
        print(f"  output norm={hs_normed_eff.float().norm():.4f}")
        diff_1_5 = (hs_wrapper.cpu().float() - hs_normed_eff.float()).abs()
        print(f"  diff vs Method 1: max={diff_1_5.max():.6f}")

    # ==== Method 6: Check if dtype conversions matter ====
    # The engine manual fallback does: residual.add_(input) then compute in fp32
    # then copy back. Our CPU ref does it all in fp32.
    # Let's simulate the EXACT engine fallback path on CPU
    resid_exact_cpu = resid_in_gpu.cpu().float()
    hs_exact_cpu = hs_in_gpu.cpu().float()
    resid_exact_cpu += hs_exact_cpu  # manual fallback: residual.add_(input)
    x_fp32 = resid_exact_cpu.float()
    rms = torch.sqrt(x_fp32.pow(2).mean(-1, keepdim=True) + eps)
    hs_exact_out = (eff_w.float() * (x_fp32 / rms)).bfloat16()

    if rank == 0:
        print(f"\n=== Method 6: EXACT engine fallback on CPU ===")
        print(f"  output norm={hs_exact_out.float().norm():.4f}")
        diff_1_6 = (hs_wrapper.cpu().float() - hs_exact_out.float()).abs()
        print(f"  diff vs Method 1: max={diff_1_6.max():.6f}")

    # ==== Summary ====
    if rank == 0:
        print(f"\n========== SUMMARY ==========")
        print(f"Method 1 (engine wrapper):    {hs_wrapper.float().norm():.4f}")
        if 'hs_vllm' in dir():
            print(f"Method 2 (direct vLLM):       {hs_vllm.float().norm():.4f}")
        print(f"Method 3 (manual fallback):   {hs_manual.float().norm():.4f}")
        print(f"Method 4 (CPU ref 1+w):       {hs_normed_raw.float().norm():.4f}")
        print(f"Method 5 (CPU eff weight):    {hs_normed_eff.float().norm():.4f}")
        print(f"Method 6 (exact fallback):    {hs_exact_out.float().norm():.4f}")

import torch.distributed as dist
dist.barrier()
if rank == 0:
    dist.destroy_process_group()
