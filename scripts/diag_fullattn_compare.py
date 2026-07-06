#!/usr/bin/env python3
"""Compare FullAttention layer 3 computation: GPU (TP=4) vs CPU (TP=1) at each substep."""
import os, sys, torch, json, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch.nn.functional as F

from engine.models.qwen import QwenTPConfig, QwenForCausalLMTP, load_weights
from engine.tp_layers.distributed import init_tp_distributed, get_tp_rank, get_tp_size
from engine.kernels.rms_norm import rms_norm
from safetensors import safe_open

init_tp_distributed()
rank = get_tp_rank()
tp_size = get_tp_size()
model_dir = os.environ['MODEL_DIR']

# Load model
cfg = QwenTPConfig(model_dir)
model = QwenForCausalLMTP(cfg)
model = model.to(torch.bfloat16).to(f'cuda:{rank}')
model = load_weights(model, model_dir)
model.eval()

# ============================================================
# CPU reference setup
# ============================================================
if rank == 0:
    with open(os.path.join(model_dir, 'model.safetensors.index.json')) as f:
        idx = json.load(f)
    wm = idx['weight_map']
    with open(os.path.join(model_dir, 'config.json')) as f:
        raw = json.load(f)
    tc = raw.get('text_config', raw)

    eps = tc['rms_norm_eps']
    head_dim = tc['head_dim']  # 256
    num_heads = tc['num_attention_heads']  # 24
    num_kv_heads = tc['num_key_value_heads']  # 4
    hidden_size = tc['hidden_size']
    layer_types = tc['layer_types']
    rp = tc.get('rope_parameters', tc.get('rope_scaling', {})) or {}
    rotary_dim = int(head_dim * rp.get('partial_rotary_factor', 1.0))  # 64
    mrope_section = rp.get('mrope_section')
    mrope_interleaved = rp.get('mrope_interleaved', False)
    rope_theta = tc.get('rope_theta') or rp.get('rope_theta', 1000000.0)

    def load_cpu(key):
        fname = wm[key]
        fpath = os.path.join(model_dir, fname)
        with safe_open(fpath, framework='pt', device='cpu') as sf:
            return sf.get_tensor(key)

    def qwen35_rms_norm(x, w):
        rstd = 1.0 / torch.sqrt(x.pow(2).mean(-1, keepdim=True) + eps)
        return x * rstd * (1.0 + w)

    # Load layer 3 weights
    prefix = 'model.language_model.layers.3.'
    w_input_ln = load_cpu(prefix + 'input_layernorm.weight').float()
    w_post_ln = load_cpu(prefix + 'post_attention_layernorm.weight').float()
    w_q = load_cpu(prefix + 'self_attn.q_proj.weight').float()  # [12288, 5120]
    w_k = load_cpu(prefix + 'self_attn.k_proj.weight').float()  # [1024, 5120]
    w_v = load_cpu(prefix + 'self_attn.v_proj.weight').float()  # [1024, 5120]
    w_o = load_cpu(prefix + 'self_attn.o_proj.weight').float()  # [5120, 6144]
    w_qn = load_cpu(prefix + 'self_attn.q_norm.weight').float()  # [256]
    w_kn = load_cpu(prefix + 'self_attn.k_norm.weight').float()  # [256]

    # Build MRoPE cache
    num_sections = len(mrope_section)
    half_dim = rotary_dim // 2
    section_freqs = []
    t = torch.arange(tc['max_position_embeddings'], dtype=torch.float32)
    for section_size in mrope_section:
        inv_freq = 1.0 / (rope_theta ** (torch.arange(0, section_size, dtype=torch.float32) * 2 / rotary_dim))
        freqs_s = torch.einsum("i,j->ij", t, inv_freq)
        section_freqs.append(freqs_s)
    interleaved_pairs = []
    for i in range(half_dim):
        s = i % num_sections
        idx = i // num_sections
        interleaved_pairs.append(section_freqs[s][:, idx])
    freqs = torch.stack(interleaved_pairs, dim=-1)
    cos_sin_cpu = torch.cat((freqs.cos(), freqs.sin()), dim=-1)

    from engine.kernels.rotary_embedding import make_cos_sin_cache as our_make_cos_sin_cache
    our_cache = our_make_cos_sin_cache(
        tc['max_position_embeddings'], rotary_dim, rope_theta,
        dtype=torch.bfloat16, device='cpu',
        mrope_section=mrope_section, mrope_interleaved=True)

    cos_diff = (cos_sin_cpu.float() - our_cache.float()).abs()
    print(f"MRoPE cache comparison:")
    print(f"  CPU cache shape: {cos_sin_cpu.shape}")
    print(f"  Our  cache shape: {our_cache.shape}")
    print(f"  Cos/sin diff: max={cos_diff.max():.6f}, mean={cos_diff.mean():.6f}")

device = f'cuda:{rank}'

tokens = [108618, 102066, 137351, 105017, 100462, 106808, 103105]
input_ids = torch.tensor([tokens], dtype=torch.long, device=device)
S = len(tokens)

with torch.inference_mode():
    # Get to layer 3 by running layers 0-2
    hidden_states = model.embed_tokens(input_ids)
    positions = torch.arange(S, dtype=torch.int64, device=device)
    residual = None

    for lx in range(3):
        hidden_states, residual = model.layers[lx](hidden_states, positions, S, residual)

    if rank == 0:
        print(f"\nAfter layer 2: GPU hs norm={hidden_states.norm():.4f}")

    # Now we're at layer 3 input. hidden_states = mlp_out from layer 2.

    # On GPU: simulate what layer 3 does
    layer3 = model.layers[3]

    # Save pre-input-ln values
    hs_in = hidden_states.clone()
    resid_in = residual.clone()

    # input layernorm
    w_ln = layer3.input_layernorm._effective_weight()
    rms_norm(hidden_states, hidden_states.contiguous(), w_ln, layer3.input_layernorm.eps)

    # On CPU: compute what the layer should produce
    if rank == 0:
        # CPU reference: need hs_out_layer2 (accumulated residual so far)
        # residual after layer 2 on GPU is the accumulated value
        resid_cpu = resid_in.cpu().float()  # This should be h_out_layer2
        hs_in_cpu = hs_in.cpu().float()  # This is the input to layer 3 (mlp_out from layer 2)

        # CPU input_layernorm
        resid_cpu_test = resid_cpu + hs_in_cpu  # residual += hs (as layer.forward would do)
        hs_normed_cpu = qwen35_rms_norm(resid_cpu_test, w_input_ln)

        B, T = 1, S
        # CPU FullAttention
        q_full_cpu = F.linear(hs_normed_cpu, w_q)  # [1, 7, 12288]
        q_cpu, gate_cpu = torch.chunk(q_full_cpu, 2, dim=-1)  # [1,7,6144] each
        k_cpu = F.linear(hs_normed_cpu, w_k)  # [1, 7, 1024]
        v_cpu = F.linear(hs_normed_cpu, w_v)

        q_cpu = q_cpu.view(B, T, num_heads, head_dim)
        k_cpu = k_cpu.view(B, T, num_kv_heads, head_dim)
        v_cpu = v_cpu.view(B, T, num_kv_heads, head_dim)
        gate_cpu = gate_cpu.view(B, T, num_heads, head_dim)

        # Q/K norms
        q_cpu_normed = qwen35_rms_norm(q_cpu.float(), w_qn.unsqueeze(0).unsqueeze(0))
        k_cpu_normed = qwen35_rms_norm(k_cpu.float(), w_kn.unsqueeze(0).unsqueeze(0))

        # Now compare with GPU intermediate values
        layer3_gpu = layer3.self_attn

        # GPU q_full: from q_proj
        q_full_gpu = layer3_gpu.q_proj(hidden_states)  # ColumnParallel → [1, 7, 3072]
        q_gpu, gate_gpu = torch.chunk(q_full_gpu, 2, dim=-1)
        q_gpu_heads = q_gpu.view(B, T, cfg.heads_per_rank, head_dim)  # [1, 7, 6, 256]
        k_gpu = layer3_gpu.k_proj(hidden_states).view(B, T, cfg.kv_heads_per_rank, head_dim)
        v_gpu = layer3_gpu.v_proj(hidden_states).view(B, T, cfg.kv_heads_per_rank, head_dim)
        gate_gpu_heads = gate_gpu.view(B, T, cfg.heads_per_rank, head_dim)

        print(f"\nGPU q_full: shape={q_full_gpu.shape}, norm={q_full_gpu.norm():.4f}")
        print(f"CPU q_full: shape={q_full_cpu.shape}, norm={q_full_cpu.norm():.4f}")

        # Compare CPU q for our rank's heads (heads 0:6 → dims 0:1536)
        q_cpu_rank0 = q_cpu[:, :, :cfg.heads_per_rank, :]  # [1, 7, 6, 256]
        q_cpu_rank0_flat = q_cpu_rank0.reshape(B, T, -1)
        q_gpu_flat = q_gpu_heads.reshape(B, T, -1)
        q_diff = (q_gpu_flat.cpu().float() - q_cpu_rank0_flat.float()).abs()
        print(f"\nQ (heads 0:5, before norm) GPU vs CPU:")
        print(f"  GPU norm={q_gpu_flat.norm():.4f}, CPU norm={q_cpu_rank0_flat.norm():.4f}")
        print(f"  Diff: max={q_diff.max():.6f}, mean={q_diff.mean():.6f}")

        # Compare after Qwen3_5RMSNorm
        # GPU q_norm is applied in QwenFullAttentionTP.forward
        q_gpu_normed = layer3_gpu.q_norm(q_gpu_heads)
        q_norm_diff = (q_gpu_normed.cpu().float() - q_cpu_normed[:, :, :cfg.heads_per_rank, :].float()).abs()
        print(f"\nQ (after Qwen3_5RMSNorm) GPU vs CPU:")
        print(f"  GPU norm={q_gpu_normed.norm():.4f}, CPU norm={q_cpu_normed[:,:,:cfg.heads_per_rank,:].norm():.4f}")
        print(f"  Diff: max={q_norm_diff.max():.6f}, mean={q_norm_diff.mean():.6f}")

        # Compare K
        k_cpu_rank0 = k_cpu[:, :, rank:rank+1, :].float()  # KV head 0
        k_gpu_flat = k_gpu.reshape(B, T, -1)
        k_cpu_flat = k_cpu_rank0.reshape(B, T, -1)
        k_diff = (k_gpu_flat.cpu().float() - k_cpu_flat).abs()
        print(f"\nK (head 0, before norm) GPU vs CPU:")
        print(f"  GPU norm={k_gpu_flat.norm():.4f}, CPU norm={k_cpu_flat.norm():.4f}")
        print(f"  Diff: max={k_diff.max():.6f}, mean={k_diff.mean():.6f}")

        # Compare q_norm weights
        print(f"\nGPU q_norm weight: min={layer3_gpu.q_norm.weight.min():.4f}, max={layer3_gpu.q_norm.weight.max():.4f}")
        print(f"CPU q_norm weight: min={w_qn.min():.4f}, max={w_qn.max():.4f}")

        # After MRoPE, run SDPA comparison
        # GPU does: apply_mrope, then flash_attn_varlen_func
        # CPU does: apply_mrope explicitly, then SDPA

        # Let's check MRoPE cache format
        cos_sin_gpu = layer3_gpu._cos_sin_cache_gpu
        print(f"\nGPU cos_sin_cache shape: {cos_sin_gpu.shape}")
        print(f"CPU cos_sin_cache shape: {cos_sin_cpu.shape}")

        # Check first few positions
        for p in [0, 1, 6]:
            gpu_slice = cos_sin_gpu[p].cpu().float()
            cpu_slice = cos_sin_cpu[p].float()
            d = (gpu_slice - cpu_slice).abs()
            print(f"  Pos {p}: diff max={d.max():.6f}, mean={d.mean():.6f}")
            if d.max() > 0.001:
                # Show first 8 values
                print(f"    GPU[:8]: {gpu_slice[:8].tolist()}")
                print(f"    CPU[:8]: {cpu_slice[:8].tolist()}")

import torch.distributed as dist
dist.barrier()
if rank == 0:
    dist.destroy_process_group()
