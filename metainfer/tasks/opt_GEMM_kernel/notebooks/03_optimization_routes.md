# GEMM optimization routes

Start from measured shape classes rather than applying every technique at
once.

For compute-bound shapes, examine instruction selection, tensor-core/MFMA
tile compatibility, register reuse, K-loop unrolling and epilogue fusion. For
memory- or launch-bound shapes, examine vector width, coalescing, split-K
overhead, persistent scheduling, kernel count and fusion. For skinny or small-M
decode GEMMs, occupancy and launch overhead often matter more than peak FLOPS.

Change one major dimension per iteration:

1. block and warp/wave tile;
2. pipeline depth and global-to-shared movement;
3. vectorization and alignment paths;
4. split-K or persistent scheduling;
5. fused dequantization, bias or activation epilogue;
6. shape-specialized dispatch with a safe fallback.

Always preserve an explicit fallback for shapes whose alignment or dimensions
do not satisfy a specialized path.

