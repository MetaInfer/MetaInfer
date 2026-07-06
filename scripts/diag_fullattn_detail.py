#!/usr/bin/env python3
"""Compare FullAttention layer 3 at each substep: Q, K, V, norms, MRoPE, attention, gate, o_proj."""
import os, sys, torch, json, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch.nn.functional as F

from engine.models.qwen import QwenTPConfig, QwenForCausalLMTP, load_weights
from engine.tp_layers.distributed import init_tp_distributed, get_tp_rank, get_tp_size

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

# CPU reference setup
if rank == 0:
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
    mrope_section = rp.get('mrope_section')
    rope_theta = tc.get('rope_theta') or rp.get('rope_theta', 1000000.0)
    max_pos = tc['max_position_embeddings']

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
    t = torch.arange(max_pos, dtype=torch.float32)
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

device = f'cuda:{rank}'
tokens = [108618, 102066, 137351, 105017, 100462, 106808, 103105]
input_ids = torch.tensor([tokens], dtype=torch.long, device=device)
S = len(tokens)
positions = torch.arange(S, dtype=torch.int64, device=device)

with torch.inference_mode():
    # Run layers 0-2 to get to layer 3 input
    hidden_states = model.embed_tokens(input_ids)
    residual = None
    for lx in range(3):
        hidden_states, residual = model.layers[lx](hidden_states, positions, S, residual)

    # Save pre-norm states
    hs_in = hidden_states.clone()
    resid_in = residual.clone()

    # Apply input_layernorm (GPU way)
    layer3 = model.layers[3]
    fa = layer3.self_attn
    w_ln = layer3.input_layernorm._effective_weight()
    from engine.kernels.rms_norm import fused_add_rms_norm
    hs_work = hs_in.clone()
    fused_add_rms_norm(hs_work, resid_in, w_ln, layer3.input_layernorm.eps)

    # Now hs_work is the normed input to FullAttention (= norm(hs_in + resid_in))
    # GPU: q_proj, k_proj, v_proj, q_gate_proj
    q_gpu = fa.q_proj(hs_work)        # [1, 7, 1536] = 6 heads * 256
    k_gpu = fa.k_proj(hs_work)        # [1, 7, 256]  = 1 head * 256
    v_gpu = fa.v_proj(hs_work)        # [1, 7, 256]
    gate_gpu = fa.q_gate_proj(hs_work)  # [1, 7, 1536]

    # Reshape to heads
    num_h = cfg.heads_per_rank  # 6
    num_kvh = cfg.kv_heads_per_rank  # 1
    B, T = 1, S

    q_gpu_h = q_gpu.view(B, T, num_h, head_dim)
    k_gpu_h = k_gpu.view(B, T, num_kvh, head_dim)
    v_gpu_h = v_gpu.view(B, T, num_kvh, head_dim)
    gate_gpu_h = gate_gpu.view(B, T, num_h, head_dim)

    # Apply Q/K norms (GPU)
    q_gpu_normed = fa.q_norm(q_gpu_h)
    k_gpu_normed = fa.k_norm(k_gpu_h)

    if rank == 0:
        # CPU reference computation
        resid_cpu = resid_in.cpu().float() + hs_in.cpu().float()
        hs_normed_cpu = qwen35_rms_norm(resid_cpu, w_input_ln)

        q_full_cpu = F.linear(hs_normed_cpu, w_q)
        # w_q is [12288, 5120]: first 6144 = Q_all_head, next 6144 = Gate_all_head
        q_cpu_raw = q_full_cpu[:, :, :6144]
        gate_cpu_raw = q_full_cpu[:, :, 6144:]
        k_cpu_raw = F.linear(hs_normed_cpu, w_k)
        v_cpu_raw = F.linear(hs_normed_cpu, w_v)

        q_cpu = q_cpu_raw.view(B, T, num_heads, head_dim)
        k_cpu = k_cpu_raw.view(B, T, num_kv_heads, head_dim)
        v_cpu = v_cpu_raw.view(B, T, num_kv_heads, head_dim)
        gate_cpu = gate_cpu_raw.view(B, T, num_heads, head_dim)

        # Compare Q: rank 0's Q heads (0:6)
        q_cpu_rank0 = q_cpu[:, :, :num_h, :]  # heads 0:6
        q_diff = (q_gpu_h.cpu().float() - q_cpu_rank0.float()).abs()
        print(f"Q (pre-norm): GPU norm={q_gpu_h.norm():.4f} CPU norm={q_cpu_rank0.norm():.4f}")
        print(f"  Diff: max={q_diff.max():.6f} mean={q_diff.mean():.6f}")

        # Compare K: rank 0's KV head (head 0)
        k_cpu_rank0 = k_cpu[:, :, rank:rank+1, :]
        k_diff = (k_gpu_h.cpu().float() - k_cpu_rank0.float()).abs()
        print(f"K (pre-norm): GPU norm={k_gpu_h.norm():.4f} CPU norm={k_cpu_rank0.norm():.4f}")
        print(f"  Diff: max={k_diff.max():.6f} mean={k_diff.mean():.6f}")

        # Q/K norms
        q_cpu_normed = qwen35_rms_norm(q_cpu.float(), w_qn.unsqueeze(0).unsqueeze(0))
        k_cpu_normed = qwen35_rms_norm(k_cpu.float(), w_kn.unsqueeze(0).unsqueeze(0))

        q_cpu_n_rank0 = q_cpu_normed[:, :, :num_h, :]
        qn_diff = (q_gpu_normed.cpu().float() - q_cpu_n_rank0.float()).abs()
        print(f"\nQ (post-norm): GPU norm={q_gpu_normed.norm():.4f} CPU norm={q_cpu_n_rank0.norm():.4f}")
        print(f"  Diff: max={qn_diff.max():.6f} mean={qn_diff.mean():.6f}")

        k_cpu_n_rank0 = k_cpu_normed[:, :, rank:rank+1, :]
        kn_diff = (k_gpu_normed.cpu().float() - k_cpu_n_rank0.float()).abs()
        print(f"K (post-norm): GPU norm={k_gpu_normed.norm():.4f} CPU norm={k_cpu_n_rank0.norm():.4f}")
        print(f"  Diff: max={kn_diff.max():.6f} mean={kn_diff.mean():.6f}")

        # Now check MRoPE
        from engine.kernels.rotary_embedding import rotary_embedding
        q_rot_cpu = q_cpu_normed.clone()
        k_rot_cpu = k_cpu_normed.clone()
        q_cpu_flat = q_rot_cpu.reshape(-1, num_heads, head_dim)
        k_cpu_flat = k_rot_cpu.reshape(-1, num_kv_heads, head_dim)

        # CPU manual MRoPE
        pos_cpu = torch.arange(S, dtype=torch.int64)
        cos = cos_sin_cpu[pos_cpu, :rotary_dim//2]
        sin = cos_sin_cpu[pos_cpu, rotary_dim//2:]
        cos = cos.view(1, S, 1, rotary_dim//2)
        sin = sin.view(1, S, 1, rotary_dim//2)
        cos_dup = torch.cat([cos, cos], dim=-1)
        sin_dup = torch.cat([sin, sin], dim=-1)

        q_rot = q_rot_cpu[..., :rotary_dim].float()
        half = rotary_dim // 2
        q1, q2 = q_rot[..., :half], q_rot[..., half:]
        q_rot_cpu[..., :rotary_dim] = (q_rot * cos_dup + torch.cat([-q2, q1], dim=-1) * sin_dup)

        k_rot = k_rot_cpu[..., :rotary_dim].float()
        k1, k2 = k_rot[..., :half], k_rot[..., half:]
        k_rot_cpu[..., :rotary_dim] = (k_rot * cos_dup + torch.cat([-k2, k1], dim=-1) * sin_dup)

        # Compare with GPU MRoPE output
        # On GPU, MRoPE was already applied during the forward pass
        # Let me redo with fresh tensors and apply MRoPE on GPU
        # Actually, the fa.forward() applies MRoPE internally. Let me compare.
        # We need to capture post-MRoPE Q/K from GPU, but we already have them through fa.forward()

        # Actually, let me just run the full forward and compare attention outputs

        # Check q_norm, k_norm weights match
        gpu_qn = fa.q_norm.weight.data.cpu().float()
        gpu_kn = fa.k_norm.weight.data.cpu().float()
        qn_w_diff = (gpu_qn - w_qn).abs()
        kn_w_diff = (gpu_kn - w_kn).abs()
        print(f"\nq_norm weights: GPU mean={gpu_qn.mean():.4f} CPU mean={w_qn.mean():.4f} diff max={qn_w_diff.max():.6f}")
        print(f"k_norm weights: GPU mean={gpu_kn.mean():.4f} CPU mean={w_kn.mean():.4f} diff max={kn_w_diff.max():.6f}")

        # Check q_proj weight for rank 0
        gpu_qw = fa.q_proj.weight.data.cpu().float()  # [1536, 5120]
        cpu_qw_full = w_q  # [12288, 5120]
        # rank 0 gets Q part: first 6144, then first 1536 of that (head 0:6)
        # Actually the q_proj is ColumnParallel of full_q_size=6144, not full_q_output=12288
        # So rank 0 gets rows 0:1536 of the first 6144 rows of w_q
        cpu_qw_rank0 = cpu_qw_full[:6144][:1536, :]  # Q partition, rank 0 slice
        qw_diff = (gpu_qw - cpu_qw_rank0).abs()
        print(f"\nq_proj weights: GPU shape={gpu_qw.shape} CPU slice shape={cpu_qw_rank0.shape}")
        print(f"  Diff max={qw_diff.max():.6f} mean={qw_diff.mean():.6f}")

        # Check q_gate_proj weight
        gpu_gw = fa.q_gate_proj.weight.data.cpu().float()
        cpu_gw_rank0 = cpu_qw_full[6144:][:1536, :]
        gw_diff = (gpu_gw - cpu_gw_rank0).abs()
        print(f"q_gate_proj weights: GPU shape={gpu_gw.shape} CPU slice shape={cpu_gw_rank0.shape}")
        print(f"  Diff max={gw_diff.max():.6f} mean={gw_diff.mean():.6f}")

        # Check k_proj weight
        gpu_kw = fa.k_proj.weight.data.cpu().float()  # [256, 5120]
        cpu_kw_rank0 = cpu_qw_full = w_k[:256, :]  # rank 0 KV head
        kw_diff = (gpu_kw - cpu_kw_rank0).abs()
        print(f"k_proj weights: GPU shape={gpu_kw.shape} CPU slice shape={cpu_kw_rank0.shape}")
        print(f"  Diff max={kw_diff.max():.6f} mean={kw_diff.mean():.6f}")

        # Check o_proj weight
        gpu_ow = fa.o_proj.weight.data.cpu().float()  # [5120, 1536]
        cpu_ow_rank0 = w_o[:, :1536]  # columns 0:1536
        ow_diff = (gpu_ow - cpu_ow_rank0).abs()
        print(f"\no_proj weights: GPU shape={gpu_ow.shape} CPU slice shape={cpu_ow_rank0.shape}")
        print(f"  Diff max={ow_diff.max():.6f} mean={ow_diff.mean():.6f}")

import torch.distributed as dist
dist.barrier()
if rank == 0:
    dist.destroy_process_group()
