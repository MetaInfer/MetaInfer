#!/usr/bin/env python3
"""V4: pinpoint normed-input divergence using TP=4 (torchrun), REAL model."""
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
hidden_size = tc['hidden_size']
hpr = cfg.heads_per_rank
full_q_size = num_heads * head_dim

# Load model (TP=4 distributes weights)
model = QwenForCausalLMTP(cfg)
model = model.to(torch.bfloat16).to(device)
model = load_weights(model, model_dir)
model.eval()

tokens = [108618, 102066, 137351, 105017, 100462, 106808, 103105]
input_ids = torch.tensor([tokens], dtype=torch.long, device=device)
B, S = 1, len(tokens)
positions = torch.arange(S, dtype=torch.int64, device=device)

# CPU weights for layer 3
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
w_iln_eff = 1.0 + w_iln_raw  # Qwen3_5RMSNorm: effective weight

with torch.inference_mode():
    # Run layers 0-2
    hidden_states = model.embed_tokens(input_ids)
    residual = None
    for lx in range(3):
        hidden_states, residual = model.layers[lx](hidden_states, positions, S, residual)

    hs_in = hidden_states.clone()
    resid_in = residual.clone()
    ln = model.layers[3].input_layernorm

    # ==== GPU: fused_add_rms_norm ====
    hs_gpu = hs_in.clone()
    resid_gpu = resid_in.clone()
    eff_w_gpu = ln._effective_weight()
    fused_add_rms_norm(hs_gpu, resid_gpu, eff_w_gpu, ln.eps)

    # ==== CPU reference: different weight interpretations ====
    if rank == 0:
        hs_cpu = hs_in.cpu().float()
        resid_cpu = resid_in.cpu().float()

        # CPU Method A: rms_norm(resid+hs) * (1+w_raw)  [what rms_norm_ref did]
        rsum_a = resid_cpu + hs_cpu
        rstd_a = 1.0 / torch.sqrt(rsum_a.pow(2).mean(-1, keepdim=True) + eps)
        normed_a = (rsum_a * rstd_a * (1.0 + w_iln_raw)).to(torch.bfloat16)

        # CPU Method B: rms_norm(resid+hs) * w_eff (w_eff = 1+w_raw)
        rsum_b = resid_cpu + hs_cpu
        rstd_b = 1.0 / torch.sqrt(rsum_b.pow(2).mean(-1, keepdim=True) + eps)
        normed_b = (rsum_b * rstd_b * w_iln_eff).to(torch.bfloat16)

        # CPU Method C: EXACT replica of the manual fallback in fused_add_rms_norm
        x_c = hs_cpu.clone()
        r_c = resid_cpu.clone()
        r_c += x_c  # residual.add_(input)
        rms_c = torch.sqrt(r_c.float().pow(2).mean(-1, keepdim=True) + eps)
        normed_c = (w_iln_eff.float() * (r_c.float() / rms_c)).bfloat16()

        # Check: are A, B, C identical?
        print(f"=== CPU Methods ===")
        print(f"  Method A (1+w):     {normed_a.float().norm():.4f}")
        print(f"  Method B (w_eff):   {normed_b.float().norm():.4f}")
        print(f"  Method C (exact fb):{normed_c.float().norm():.4f}")
        print(f"  A-B diff max={(normed_a.float()-normed_b.float()).abs().max():.8f}")
        print(f"  A-C diff max={(normed_a.float()-normed_c.float()).abs().max():.8f}")
        print(f"  B-C diff max={(normed_b.float()-normed_c.float()).abs().max():.8f}")
        # These should all be 0 since w_raw is all zeros -> 1+w = 1.0, w_eff = 1.0

        # Compare with GPU
        hs_gpu_cpu = hs_gpu.cpu().float()
        print(f"\n=== GPU vs CPU ===")
        print(f"  GPU norm:    {hs_gpu_cpu.norm():.4f}")
        print(f"  CPU-A norm:  {normed_a.float().norm():.4f}")
        d_gpu_a = (hs_gpu_cpu - normed_a.float()).abs()
        d_gpu_c = (hs_gpu_cpu - normed_c.float()).abs()
        print(f"  GPU vs A: max={d_gpu_a.max():.6f} mean={d_gpu_a.mean():.6f}")
        print(f"  GPU vs C: max={d_gpu_c.max():.6f} mean={d_gpu_c.mean():.6f}")

        if d_gpu_a.max() > 0.1 or d_gpu_c.max() > 0.1:
            # Print the first few elements where they differ
            print(f"\n  === Top 10 largest diffs (GPU vs C) ===")
            flat_d = d_gpu_c.flatten()
            top_k = torch.topk(flat_d, k=min(10, flat_d.numel()))
            for k, (idx, val) in enumerate(zip(top_k.indices, top_k.values)):
                # Convert flat index to multi-dim
                b = idx // (S * hidden_size)
                rem = idx % (S * hidden_size)
                s = rem // hidden_size
                h = rem % hidden_size
                print(f"    [{b},{s},{h}]: GPU={hs_gpu_cpu[b,s,h]:.6f} CPU={normed_c.float()[b,s,h]:.6f} diff={val:.6f}")

        # Check weight match
        gpu_iln_w = ln.weight.data.cpu().float()
        iln_diff = (gpu_iln_w - w_iln_raw).abs()
        print(f"\n=== Weight Check ===")
        print(f"  GPU ln weight shape={list(gpu_iln_w.shape)}")
        print(f"  vs safetensors: max diff={iln_diff.max():.10f}")
        print(f"  GPU ln all zeros? {(gpu_iln_w == 0).all().item()}")
        eff_w_check = ln._effective_weight().cpu().float()
        print(f"  GPU effective weight min={eff_w_check.min():.6f} max={eff_w_check.max():.6f}")

        # Check eps
        print(f"  GPU eps={ln.eps}, config eps={eps}")

        # Check residual after fused_add_rms_norm
        resid_expected = resid_cpu + hs_cpu
        resid_gpu_cpu = resid_gpu.cpu().float()
        rdiff = (resid_gpu_cpu - resid_expected).abs()
        print(f"\n=== Residual Check ===")
        print(f"  GPU resid norm={resid_gpu_cpu.norm():.4f}")
        print(f"  Expected resid norm={resid_expected.norm():.4f}")
        print(f"  diff max={rdiff.max():.6f} mean={rdiff.mean():.6f}")

        if rdiff.max() > 0.1:
            print(f"  Top 5 residual diffs:")
            flat_r = rdiff.flatten()
            top_k = torch.topk(flat_r, k=5)
            for idx, val in zip(top_k.indices, top_k.values):
                b = idx // (S * hidden_size)
                rem = idx % (S * hidden_size)
                s = rem // hidden_size
                h = rem % hidden_size
                print(f"    [{b},{s},{h}]: GPU={resid_gpu_cpu[b,s,h]:.6f} CPU={resid_expected[b,s,h]:.6f} diff={val:.6f}")

import torch.distributed as dist
dist.barrier()
if rank == 0:
    dist.destroy_process_group()
