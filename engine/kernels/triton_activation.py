"""Fused activation helpers (reserved for future Triton kernels)."""
# P5a-light uses PyTorch F.silu * mul instead of Triton kernel.
# Triton kernel had -43.7% regression on small decode tensors (launch overhead).
