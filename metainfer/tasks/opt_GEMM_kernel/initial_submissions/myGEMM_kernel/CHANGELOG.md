# Baseline extraction

- Extracted only the W8A8 GEMV/GEMM kernels and launch routing from
  `harness/user_gemm/myGEMM_kernel.cpp`.
- Removed activation quantization, allocation, random input generation,
  correctness checking, timing, printing, and `main()`; these are frozen
  Harness responsibilities.
- Exported the system evaluator ABI as `launch_w8a8_gemm`.
