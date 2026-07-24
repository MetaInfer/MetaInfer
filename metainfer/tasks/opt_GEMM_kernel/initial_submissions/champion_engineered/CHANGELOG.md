# Engineered Champion initial submission

Source snapshot from `/data/work/int8-w8a8-gemm/benchmark/gemm_champon.cpp`.
Includes Wave64 DPP+SDOT4, split-K SDOT4, general MMAC, safe aligned/unaligned global loads, explicit LDS alignment, and arbitrary-shape fallbacks.
