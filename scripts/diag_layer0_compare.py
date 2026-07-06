#!/usr/bin/env python3
"""Compare TP model layer 0 output with CPU reference from diag_cpu_verify.py."""
import os, sys, torch, json, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine.models.qwen import QwenTPConfig, QwenForCausalLMTP, load_weights
from engine.tp_layers.distributed import init_tp_distributed, get_tp_rank, get_tp_size, all_reduce_sum
from engine.kernels.rms_norm import rms_norm

init_tp_distributed()
rank = get_tp_rank()
tp_size = get_tp_size()
model_dir = os.environ['MODEL_DIR']

# Load CPU reference
ref = torch.load('/tmp/diag_layer0_cpu_ref.pt', map_location='cpu', weights_only=True)
hs_ref = ref['hs']        # [1, 7, 5120] - after input_layernorm
out_ref = ref['out']      # [1, 7, 5120] - layer 0 full output
tokens = ref['tokens']

# Load model
cfg = QwenTPConfig(model_dir)
model = QwenForCausalLMTP(cfg)
model = model.to(torch.bfloat16).to(f'cuda:{rank}')
model = load_weights(model, model_dir)
model.eval()

device = f'cuda:{rank}'
input_ids = torch.tensor([tokens], dtype=torch.long, device=device)
S = len(tokens)

with torch.inference_mode():
    # Get embedding
    emb = model.embed_tokens(input_ids)  # [1, 7, 5120]

    if rank == 0:
        emb_norm = emb.norm().item()
        print(f"[CPU] embedding norm: {hs_ref.norm():.4f}")
        print(f"[GPU] embedding norm: {emb_norm:.4f}")
        # Compare: CPU uses full embedding; GPU uses VocabParallelEmbedding (mask + all_reduce)

    # Run just layer 0
    layer0 = model.layers[0]
    positions = torch.arange(S, dtype=torch.int64, device=device)

    # Apply input_layernorm same way as model.forward
    hs = emb.clone()
    effective_w = layer0.input_layernorm._effective_weight()
    rms_norm(hs, hs.contiguous(), effective_w, layer0.input_layernorm.eps)

    if rank == 0:
        print(f"[CPU] after input_ln: norm={hs_ref.norm():.4f}")
        print(f"[GPU] after input_ln: norm={hs.norm().item():.4f}")
        hs_cpu_err = (hs.cpu().float() - hs_ref).abs()
        print(f"[DIFF] input_ln max={hs_cpu_err.max():.4f}, mean={hs_cpu_err.mean():.4f}")
        if hs_cpu_err.max() < 0.01:
            print("  -> input_layernorm MATCHES CPU reference")
        else:
            print("  -> input_layernorm MISMATCH! Investigating...")

    # Run GatedDeltaNet forward
    attn_out = layer0.linear_attn(hs, positions, S)

    # The output of gdn.forward is the RowParallelLinear output (before all_reduce for partial case,
    # but QwenGatedDeltaNetTP already calls all_reduce_sum inside out_proj using RowParallelLinear)
    # Actually, let me check if the output is already all-reduced.
    # RowParallelLinear.forward calls all_reduce_sum if reduce_results=True (default).

    # After layer0 returns attn_out, the full layer adds it to residual in layer0.forward.
    # But here we only called linear_attn.forward, not layer0.forward.
    # So attn_out is the GatedDeltaNet output (after out_proj with all_reduce).

    model_out = attn_out  # [1, 7, 5120] (should be all-reduced)

    if rank == 0:
        out_cpu = out_ref
        out_gpu = model_out.cpu().float()

        print(f"\n[CPU] layer0 output: norm={out_cpu.norm():.4f}, last_pos norm={out_cpu[0,-1].norm():.4f}")
        print(f"[GPU] layer0 output: norm={out_gpu.norm():.4f}, last_pos norm={out_gpu[0,-1].norm():.4f}")

        diff = (out_gpu - out_cpu).abs()
        print(f"[DIFF] layer0 output: max={diff.max():.6f}, mean={diff.mean():.6f}")

        # Check each position
        for t in range(S):
            pos_diff = (out_gpu[0,t] - out_cpu[0,t]).abs()
            print(f"  pos {t}: diff max={pos_diff.max():.6f}, mean={pos_diff.mean():.6f}")

        if diff.max() < 0.05:
            print("\n*** LAYER 0 MATCHES! Bug is in later layers or the forward loop. ***")
        else:
            print(f"\n*** LAYER 0 MISMATCH! Max diff = {diff.max():.6f} ***")
            # Find where difference is largest
            max_idx = diff.argmax()
            max_pos = torch.unravel_index(max_idx, diff.shape)
            print(f"  Max diff at {max_pos}: cpu={out_cpu[max_pos].item():.6f}, gpu={out_gpu[max_pos].item():.6f}")

        # Debug: Check if one rank dominates the error
        # Actually let's also check without all_reduce
        print("\n--- Detailed GatedDeltaNet diagnostics ---")

        # Re-run with saved intermediate values
        gdn = layer0.linear_attn
        hs2 = emb.clone()
        rms_norm(hs2, hs2.contiguous(), effective_w, layer0.input_layernorm.eps)

        B, T, H = hs2.shape
        from engine.tp_layers.linear import ColumnParallelLinear, RowParallelLinear

        # Step 1: in_proj_qkv (ColumnParallel)
        mixed_qkv = gdn.in_proj_qkv(hs2)  # [1, T, local_dim]

        # Step 2: Causal conv1d
        # Copy weights for manual verification
        w_conv = gdn.conv1d_weight
        b_conv = gdn.conv1d_bias

        mixed_t = mixed_qkv.transpose(1, 2)
        mixed_pad = F.pad(mixed_t, (3, 0))
        conv_out_local = F.conv1d(mixed_pad, w_conv, b_conv, groups=mixed_t.shape[1])
        conv_out_local = F.silu(conv_out_local).transpose(1, 2)

        # Also get the model's conv output for comparison
        conv_out_model = mixed_qkv.transpose(1, 2)
        conv_out_model = F.pad(conv_out_model, (3, 0))
        conv_out_model = F.conv1d(conv_out_model, w_conv, b_conv, groups=conv_out_model.shape[1])
        conv_out_model = F.silu(conv_out_model).transpose(1, 2)

        # Step 3: Split Q, K, V (local shards)
        ksl = cfg.k_heads_per_rank
        vsl = cfg.v_heads_per_rank
        kdim = 128
        vdim = 128
        q = conv_out_model[:, :, :ksl*kdim].view(B, T, ksl, kdim)
        k = conv_out_model[:, :, ksl*kdim:2*ksl*kdim].view(B, T, ksl, kdim)
        v = conv_out_model[:, :, 2*ksl*kdim:].view(B, T, vsl, vdim)

        # Step 4: in_proj_a, in_proj_b
        a_full = gdn.in_proj_a(hs2)  # ColumnParallel or full?
        b_full = gdn.in_proj_b(hs2)

        # in_proj_a and in_proj_b might be ColumnParallelLinear or regular Linear
        # Check row_slice for local heads
        vr = slice(rank*vsl, (rank+1)*vsl)

        # Need to check: are a and b ColumnParallel? Let me look at config
        # The output dim should be v_heads (48). With TP=4, each rank gets 12.
        print(f"  a_full shape: {a_full.shape}, b_full shape: {b_full.shape}")
        print(f"  vsl={vsl}, rank={rank}, vr={vr}")

        # If ColumnParallel, a_full is [1, T, vsl=12]
        a_local = a_full
        b_local = b_full

        # Step 5: Gate and beta
        g = -torch.exp(gdn.A_log) * F.softplus(a_local + gdn.dt_bias)
        beta = torch.sigmoid(b_local)

        # Step 6: Q/K L2 norm
        q_norm = F.normalize(q.float(), p=2, dim=-1).to(q.dtype)
        k_norm = F.normalize(k.float(), p=2, dim=-1).to(k.dtype)

        if vsl > ksl:
            q_norm = q_norm.repeat_interleave(vsl // ksl, dim=2)
            k_norm = k_norm.repeat_interleave(vsl // ksl, dim=2)

        q_scale = 1.0 / math.sqrt(kdim)
        q_norm = q_norm * q_scale

        # Now compare with what the actual model produces at each step
        print(f"\n  q norm (local): {q_norm.norm():.4f}")
        print(f"  k norm (local): {k_norm.norm():.4f}")
        print(f"  v (local): norm={v.norm():.4f}")
        print(f"  g (local): min={g.min():.4f}, max={g.max():.4f}, mean={g.mean():.4f}")
        print(f"  beta (local): min={beta.min():.4f}, max={beta.max():.4f}, mean={beta.mean():.4f}")

import torch.distributed as dist
dist.barrier()
if rank == 0:
    dist.destroy_process_group()
