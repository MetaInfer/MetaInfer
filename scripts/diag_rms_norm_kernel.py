#!/usr/bin/env python3
"""Test whether vLLM fused_add_rms_norm kernel runs correctly on this platform."""
import os, sys, torch

model_dir = os.environ.get('MODEL_DIR', '/data/models')
device = 'cuda:0'

# Test 1: Does the vLLM kernel run at all?
print("=== Test 1: vLLM kernel availability ===")
try:
    from vllm._custom_ops import fused_add_rms_norm as _vllm_fn
    print("vLLM fused_add_rms_norm: IMPORTED OK")
except Exception as e:
    print(f"vLLM fused_add_rms_norm: IMPORT FAILED ({type(e).__name__}: {e})")
    sys.exit(1)

# Test 2: Run with float32 inputs (should fall back)
print("\n=== Test 2: Run with fp32 input (expect fallback) ===")
try:
    x = torch.randn(1, 7, 256, dtype=torch.float32, device=device)
    r = torch.randn(1, 7, 256, dtype=torch.float32, device=device)
    w = torch.ones(256, dtype=torch.float32, device=device)
    x_ref = x.clone()
    r_ref = r.clone()
    _vllm_fn(x, r, w, 1e-6)
    # Manual reference
    r_ref2 = r_ref + x_ref
    manual = (r_ref2.float() / torch.sqrt(r_ref2.float().pow(2).mean(-1, keepdim=True) + 1e-6) * w.float()).to(x.dtype)
    print(f"  vLLM output norm: {x.float().norm():.4f}")
    print(f"  Manual output norm: {manual.float().norm():.4f}")
    diff = (x.float() - manual.float()).abs()
    print(f"  Diff max={diff.max():.6f} mean={diff.mean():.6f}")
    print(f"  residual vLLM norm: {r.float().norm():.4f}")
    print(f"  residual expected norm: {r_ref2.float().norm():.4f}")
except Exception as e:
    print(f"  {type(e).__name__}: {e}")

# Test 3: Run with bf16 inputs
print("\n=== Test 3: Run with bf16 input ===")
try:
    x_bf = torch.randn(1, 7, 256, dtype=torch.bfloat16, device=device)
    r_bf = torch.randn(1, 7, 256, dtype=torch.bfloat16, device=device)
    w_bf = torch.ones(256, dtype=torch.bfloat16, device=device)
    x_ref_bf = x_bf.clone()
    r_ref_bf = r_bf.clone()

    # Try vLLM kernel
    _vllm_fn(x_bf, r_bf, w_bf, 1e-6)

    # Manual reference
    r_expected = r_ref_bf + x_ref_bf
    manual_bf = (r_expected.float() / torch.sqrt(r_expected.float().pow(2).mean(-1, keepdim=True) + 1e-6) * w_bf.float()).bfloat16()

    print(f"  vLLM output norm: {x_bf.float().norm():.4f}")
    print(f"  Manual output norm: {manual_bf.float().norm():.4f}")
    diff = (x_bf.float() - manual_bf.float()).abs()
    print(f"  Diff max={diff.max():.6f} mean={diff.mean():.6f}")

    # Also check residual
    r_expected_2 = r_ref_bf + x_ref_bf
    r_diff = (r_bf.float() - r_expected_2.float()).abs()
    print(f"  residual diff max={r_diff.max():.6f}")

    if diff.max() > 0.1:
        print("  ❌ KERNEL PRODUCES WRONG RESULTS!")
    else:
        print("  ✅ Kernel results match within bf16 tolerance")
except Exception as e:
    print(f"  {type(e).__name__}: {e}")

# Test 4: Same as Test 3 but with Normal(0, 10) scale
print("\n=== Test 4: Larger magnitudes (mu=0, std=10) ===")
try:
    x_large = torch.randn(1, 7, 256, dtype=torch.bfloat16, device=device) * 10
    r_large = torch.randn(1, 7, 256, dtype=torch.bfloat16, device=device) * 10
    w_large = torch.ones(256, dtype=torch.bfloat16, device=device)
    xl_ref = x_large.clone()
    rl_ref = r_large.clone()

    _vllm_fn(x_large, r_large, w_large, 1e-6)

    r_expected_l = rl_ref + xl_ref
    manual_large = (r_expected_l.float() / torch.sqrt(r_expected_l.float().pow(2).mean(-1, keepdim=True) + 1e-6) * w_large.float()).bfloat16()

    print(f"  vLLM output norm: {x_large.float().norm():.4f}")
    print(f"  Manual output norm: {manual_large.float().norm():.4f}")
    diff = (x_large.float() - manual_large.float()).abs()
    print(f"  Diff max={diff.max():.6f} mean={diff.mean():.6f}")
except Exception as e:
    print(f"  {type(e).__name__}: {e}")

# Test 5: Check if the REAL issue is multi-step accumulation
print("\n=== Test 5: Multi-step (simulate 3-layer residual chain) ===")
try:
    # Simulate what happens across 3 decoder layers
    hs = torch.randn(1, 7, 256, dtype=torch.bfloat16, device=device)
    resid = torch.randn(1, 7, 256, dtype=torch.bfloat16, device=device)

    def fused_manual(x, r, w, eps):
        r.add_(x)
        r_fp32 = r.float()
        rms = torch.sqrt(r_fp32.pow(2).mean(-1, keepdim=True) + eps)
        x_new = (w.float() * (r_fp32 / rms)).bfloat16()
        x.copy_(x_new)

    # GPU path (vLLM kernel)
    hs_gpu = hs.clone()
    resid_gpu = resid.clone()
    w_gpu = torch.ones(256, dtype=torch.bfloat16, device=device)
    for i in range(6):  # 3 layers * 2 norms per layer
        _vllm_fn(hs_gpu, resid_gpu, w_gpu, 1e-6)
        # Simulate attention/mlp producing some output
        hs_gpu.copy_(torch.randn(1, 7, 256, dtype=torch.bfloat16, device=device))

    # CPU path (manual)
    hs_cpu = hs.clone()
    resid_cpu = resid.clone()
    w_cpu = torch.ones(256)
    torch.manual_seed(torch.initial_seed())  # re-seed for same random sequence
    # Can't easily replicate random sequence - just compare final residual state

    print(f"  GPU residual after 6 steps: {resid_gpu.float().norm():.4f}")
except Exception as e:
    print(f"  {type(e).__name__}: {e}")

print("\n=== DONE ===")
