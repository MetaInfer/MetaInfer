// @@variant shape=tp8_shared_down_proj_m16 commit=119c281feefbb681f88dba89897484424bb07937 added=2026-08-26
//   median_us=8.453 p90_us=9.13
//   source=hy3-dsh-tp8-m16-1-adf021ab
// W8A8 INT8 GEMM bootstrap kernel for Hygon K500SM_AI / gfx928.
//
// Worker_3 (physical GPU 3) assigned shapes:
//   tp8_shared_gate_up_proj_m16 : M=16, N=512,  K=4096
//   tp8_shared_down_proj_m16    : M=16, N=4096, K=256
//
// TP8 shared_down_proj (M=16, N=4096, K=256): iteration 1 was the minimal
// gfx928 INT8 DUMMA m16n16k32 bootstrap (w8a8_dumma_m16_n16_k256_kernel,
// one 64-thread wavefront per block, 8 k-ascending steps, zero barriers).
// Iteration 2 (architecture round) replaces it with the launch-geometry
// probe w8a8_dumma_m16_n16_k256_sk2_kernel: 2 wavefronts per block (128
// threads), one 16x16 output tile per block, grid = N/16 = 256 blocks,
// in-block split-K = 2 along K (K-half = 128 per wave, 4 m16n16k32 steps),
// one END-of-K barrier, LDS-plane ascending split combine. The generic
// scalar kernel remains the fallback for every other (m,n,k).
//
// Iteration 1 strategy: scalar correctness bootstrap + minimal DUMMA path.
//   * w8a8_dumma_m16_n16_kernel: exact M=16,N=512,K=4096 gate_up shape,
//     one 64-thread wavefront per block, one 16x16 output tile per block,
//     LDS-staged A/B, m16n16k32 int8 DUMMA with explicit int32 accumulation
//     and a direct register epilogue (scale + bf16 store).
//   * w8a8_scalar_gemm_kernel: generic fallback for every unmatched (m,n,k).
//   * No split-K, double buffering, or inline asm in this round.
//
// Iteration 2 strategy (architecture round): launch-geometry exploration.
//   * w8a8_dumma_m16_n16_k4_kernel: same exact M=16,N=512,K=4096 gate_up
//     shape, now 4 wavefronts per block (256 threads) with in-block
//     split-K = 4: each wave accumulates one 1024-row K quarter in
//     k-ascending m16n16k32 steps from its own private 8 KiB LDS staging
//     region; the four 16x16 int32 partials are combined in ascending split
//     order through a 4 KiB LDS plane (bit-exact int32). Grid stays
//     N/16 = 32 blocks (the N axis caps independent blocks at 32 of 120 CUs;
//     grid split-K is the next lever if per-CU chains alone do not close the
//     gap vs the 59.21 us Triton baseline). Accepted at 30.87 us median.
//
// Iteration 5 strategy (packed-weight/staging round): replace the B path of
// the accepted kernel with the packed [N,K] n-major layout
// (w8a8_dumma_m16_n16_k4_packedb_kernel): the iteration-2 B fragment costs 8
// ds_read_u8 per v_mmac (row-major [k][16] stage samples 8 k-rows at 16-byte
// stride per lane; 7.7 LDS bank conflicts per LDS instruction). The packed
// layout + matrix_b col_major fragments turn each B fragment into one 8-byte
// LDS read. One-time pack (w8a8_pack_gateup_nmajor_kernel) runs
// out-of-timed-region/out-of-Graph for (k,n)==(4096,512); the generic scalar
// fallback decodes the same pack for paired M=2 API shapes via b_packed.
//
// The host contract is fixed by csrc/bindings.cpp. This file owns the two
// stable C symbols:
//   launch_w8a8_gemm(...)        -- timed GEMM (all shapes, generic fallback)
//   launch_pack_w8a8_weight(...) -- optional out-of-timed-region packing
//                                    (bootstrap: identity device-to-device
//                                    copy, valid for every (K,N))
//
// Header order is load-bearing for this DTK: hip_runtime first, then
// hip_bfloat16, then du_mma (du_mma.h is not self-contained when included
// before the HIP runtime headers). du_mma.h is included up front so later
// DUMMA optimization rounds only add kernels, not header reordering.

#include <hip/hip_runtime.h>
#include <hip/hip_bfloat16.h>
#include <du_mma.h>

#include <cstdint>

namespace {

// gfx928 wavefront size is 64 lanes; every blockDim below is a multiple of
// it (128 / 256 / 512 threads).

// ---------------------------------------------------------------------------
// Scalar correctness kernel: one thread per output element.
//
// idx -> (row = idx / n, col = idx % n). Consecutive threads therefore walk
// consecutive columns of the same row: B reads b[k*n + col] are byte-adjacent
// across lanes and out writes are adjacent bf16, while the A row is broadcast
// within a warp (every lane of one row reads the same a[row*k + kk]).
//
// This is also the generic scalar fallback for every unmatched (m,n,k):
// later rounds must keep any exact-shape specialization guarded AFTER this
// path in the dispatch so paired M=2 API shapes with the same (N,K) still
// reach the scalar kernel. b_packed selects the packed [N,K] n-major layout
// produced by launch_pack_w8a8_weight for (k,n)==(4096,512):
// b[col*k + kk] == W[kk*n + col] (P[n][k] = W[k][n]). Every other (k,n)
// keeps the byte-identical original [K,N] row-major decode.
// ---------------------------------------------------------------------------
__global__ __launch_bounds__(128) void w8a8_scalar_gemm_kernel(
    const int8_t* __restrict__ a,
    const int8_t* __restrict__ b,
    const float* __restrict__ x_scale,
    const float* __restrict__ weight_scale,
    hip_bfloat16* __restrict__ out,
    int m,
    int n,
    int k,
    int b_packed) {
  const int idx = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x) +
                  static_cast<int>(threadIdx.x);
  const int total = m * n;
  if (idx >= total) {
    return;
  }

  const int row = idx / n;
  const int col = idx - row * n;

  // Exact int32 dot product over the full K dimension.
  const int8_t* __restrict__ a_row = a + static_cast<int64_t>(row) * k;
  const int8_t* __restrict__ b_col =
      b_packed ? (b + static_cast<int64_t>(col) * k) : (b + col);
  int32_t acc = 0;
  for (int kk = 0; kk < k; ++kk) {
    acc += static_cast<int32_t>(a_row[kk]) *
           static_cast<int32_t>(b_packed
                                    ? b_col[kk]
                                    : b_col[static_cast<int64_t>(kk) * n]);
  }

  // out = bf16(dot * x_scale[m] * weight_scale[n]); scales applied
  // left-to-right exactly like the Python reference expression.
  const float scaled = static_cast<float>(acc) * x_scale[row] * weight_scale[col];
  out[static_cast<int64_t>(row) * n + col] =
      static_cast<hip_bfloat16>(scaled);
}

// ---------------------------------------------------------------------------
// TP8 shared_down_proj (M=16, N=4096, K=256) iteration 2: architecture
// round -- launch-geometry probe (2 waves per block, 1 adjacent N tile).
// Two 64-thread wavefronts per block, one 16x16 output tile per block,
// grid = N/16 = 256 blocks (2.13 blocks/CU on 120 CUs; 256 independent
// blocks keep all CUs active), in-block split-K = 2 along K: wave w owns K
// rows [w*128, (w+1)*128) and runs a 4-step k-ascending m16n16k32 chain --
// half the iteration-1 serial 8-step chain -- doubling wavefronts per CU
// from 2.13 to 4.27 and giving the scheduler two independent MMAC chains per
// block to hide DUMMA/LDS latency behind. The two 16x16 int32 partials are
// published to two 1 KiB LDS planes; one END-of-K __syncthreads() orders
// wave0's plane before wave1's ascending split sum (s0 + s1), which
// reproduces the scalar reference's k-ascending int32 accumulation
// bit-exactly (max |acc| = 256*128*128 << 2^31). Wave1 only runs the fused
// scale+bf16 epilogue and one merged 8-byte store per lane; wave0 exits
// after the barrier.
//
// Data path (byte-identical to iteration 1): the whole K=256 slice is
// staged ONCE into LDS with cooperative 16-byte int4 loads -- each wave
// stages only its own K half (2 x int4 loads per lane per matrix), so no
// cross-wave staging barrier is needed. A stays logical row-major [16][256]
// (4 KiB, L2-hot across all 256 blocks); B is read from the packed [N,K]
// n-major layout (P[n*256+k] = W[k*4096+n], produced one-time out-of-timed-
// region for (k,n)==(256,4096)), which makes every matrix_b col_major
// fragment one contiguous 8-byte LDS read. Both stages use the 272 = 256+16
// bank-skewed stride (same skew class as the accepted gate_up 400-stride
// stages). Zero barriers per k-step; exactly one barrier per block.
//
// Epilogue: wave1 reads both planes with the iteration-1 lane mapping
// (row = lane>>2, col0 = (lane&3)*4) and finalizes 4 consecutive columns of
// one row: (s_part0 + s_part1) * x_scale[row] * weight_scale[n0+col0+i] ->
// bf16, four adjacent 2-byte stores the compiler merges into one 8-byte
// store (16 rows x 32 B per wavefront instruction). No workspace use (the
// split is in-block). LDS/block = 2 x 4,352 + 2 x 1,024 = 10,752 B -> ~6
// blocks/CU by LDS, ~48 VGPR x 128 thr -> no residency cliff.
// ---------------------------------------------------------------------------
__global__ __launch_bounds__(128) void w8a8_dumma_m16_n16_k256_sk2_kernel(
    const int8_t* __restrict__ a,
    const int8_t* __restrict__ b,  // packed [N,K] n-major for this shape
    const float* __restrict__ x_scale,
    const float* __restrict__ weight_scale,
    hip_bfloat16* __restrict__ out,
    int n,
    int k) {
  constexpr int kTileM = 16;
  constexpr int kTileN = 16;
  constexpr int kTileK = 32;
  constexpr int kK = 256;
  constexpr int kSplit = 2;
  constexpr int kHalfK = kK / kSplit;  // 128 K rows per wave
  constexpr int kLdsStride = kK + 16;  // 272 = 256 + 16 bank skew
  const int tid = static_cast<int>(threadIdx.x);  // 0..127
  const int wave = tid >> 6;                      // 0..1 (split index)
  const int lane = tid & 63;                      // 0..63
  const int n0 = static_cast<int>(blockIdx.x) * kTileN;

  __shared__ __align__(16) int8_t lds_a[kTileM * kLdsStride];  // 4,352 B
  __shared__ __align__(16) int8_t lds_b[kTileN * kLdsStride];  // 4,352 B
  __shared__ __align__(16) int32_t s_part0[kTileM * kTileN];   // 1,024 B
  __shared__ __align__(16) int32_t s_part1[kTileM * kTileN];   // 1,024 B

  // Stage the whole K slice once, wave w covering its own K half:
  // lane -> (row = lane>>2, kk16 = (lane&3)<<4); i covers the half's two
  // 64-k quarters, so 16 rows x 128 k = 2,048 B per wave per matrix, each
  // (row, chunk) written exactly once (4,096 B per matrix total, same global
  // traffic as iteration 1). Wave-private regions: no staging barrier.
  const int row = lane >> 2;
  const int kk16 = (lane & 3) << 4;
  const int kBase = wave * kHalfK;
  int4 va[2], vb[2];
#pragma unroll
  for (int i = 0; i < 2; ++i) {
    va[i] = *reinterpret_cast<const int4*>(a + row * k + kBase + i * 64 + kk16);
    vb[i] = *reinterpret_cast<const int4*>(
        b + (n0 + row) * k + kBase + i * 64 + kk16);
  }
#pragma unroll
  for (int i = 0; i < 2; ++i) {
    *reinterpret_cast<int4*>(lds_a + row * kLdsStride + kBase + i * 64 + kk16) =
        va[i];
    *reinterpret_cast<int4*>(lds_b + row * kLdsStride + kBase + i * 64 + kk16) =
        vb[i];
  }

  du::dumma::DUFragment<du::dumma::matrix_a, kTileM, kTileN, kTileK,
                        signed char, du::dumma::row_major>
      a_frag;
  du::dumma::DUFragment<du::dumma::matrix_b, kTileM, kTileN, kTileK,
                        signed char, du::dumma::col_major>
      b_frag;
  du::dumma::DUFragment<du::dumma::accumulator, kTileM, kTileN, kTileK, int>
      acc_frag;
  du::dumma::du_fill_fragment(acc_frag, 0);

  // Zero-in-loop-barrier LDS-only K-half loop: 4 k-ascending m16n16k32 steps
  // per wave (each wave reads only its own staged region).
#pragma unroll
  for (int kk = kBase; kk < kBase + kHalfK; kk += kTileK) {
    du::dumma::du_load_matrix_sync(a_frag, lds_a + kk, kLdsStride);
    du::dumma::du_load_matrix_sync(b_frag, lds_b + kk, kLdsStride);
    du::dumma::du_mma_sync(acc_frag, a_frag, b_frag, acc_frag);
  }

  // Publish the wave's 16x16 int32 partial to its private LDS plane, then
  // one END-of-K barrier: orders wave0's plane before wave1's read (wave1
  // reads its own plane in same-wave program order; the compiler emits the
  // lgkmcnt wait). Wave0 exits after the barrier.
  if (wave == 0) {
    du::dumma::du_store_matrix_sync(s_part0, acc_frag, kTileN,
                                    du::dumma::mem_row_major);
  } else {
    du::dumma::du_store_matrix_sync(s_part1, acc_frag, kTileN,
                                    du::dumma::mem_row_major);
  }
  __syncthreads();

  if (wave == 1) {
    // Fused epilogue: ascending split sum (s0 + s1) * x_scale * weight_scale
    // -> bf16; each lane finalizes 4 consecutive columns of one row.
    const int orow = lane >> 2;        // 0..15
    const int ocol0 = (lane & 3) * 4;  // 0,4,8,12
    const float xs = x_scale[orow];
    const float4 ws =
        *reinterpret_cast<const float4*>(weight_scale + n0 + ocol0);
    const float wsf[4] = {ws.x, ws.y, ws.z, ws.w};
    hip_bfloat16* op = out + orow * n + n0 + ocol0;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int32_t s =
          s_part0[orow * kTileN + ocol0 + i] +
          s_part1[orow * kTileN + ocol0 + i];
      const float scaled = static_cast<float>(s) * xs * wsf[i];
      op[i] = static_cast<hip_bfloat16>(scaled);
    }
  }
}

// ---------------------------------------------------------------------------
// Iteration 2 (architecture round): launch-geometry exploration for the exact
// M=16, N=512, K=4096 gate_up shape (TP8). Geometry: 4 wavefronts per block
// (256 threads), one independent 16x16 output tile per block (grid = N/16 =
// 32 blocks), in-block split-K = 4: wave w accumulates K rows
// [w*1024, (w+1)*1024) in k-ascending m16n16k32 steps. Every wave owns a
// private 8 KiB LDS staging region (A 16x256 + B 256x16) reloaded in four
// 256-row sub-stages with the same cooperative 16-byte int4 global loads as
// iteration 1; one __syncthreads() per sub-stage orders each wave's staging
// before its fragment reads (regions are wave-private, so the reuse barrier
// only guards the wave's own write-after-read). The four 16x16 int32
// accumulators are published to a 4 KiB LDS partial plane, then summed in
// ascending split order by all four waves (4 rows each), which reproduces the
// scalar reference's k-ascending int32 dot exactly. LDS total = 36 KiB -> 1
// block per CU. The grid is still capped at 32 independent blocks (N axis),
// i.e. 32 of 120 CUs; this round measures whether 4 concurrent MMAC chains
// per CU (128 total wavefronts, 9 barriers/block vs 32 in iteration 1) close
// the latency gap vs the 59.21 us Triton baseline. If not, the wall is
// memory-side and the next lever is grid split-K (fill all 120 CUs) plus a
// packed-B layout.
//
// Iteration 5 (packed-weight/staging round): the accepted iteration-2 kernel
// costs ~9.25 LDS reads per v_mmac in the steady state, of which 8 are
// ds_read_u8 for the matrix_b fragment: B is staged [k][16] row-major and the
// row_major m16n16k32 B fragment samples 8 k-rows at 16-byte LDS stride per
// lane (k-step stride 512 = 32 k-rows x 16 B), so every B fragment is eight
// single-byte reads with 4-way+ bank conflicts (7.7 conflicts per LDS
// instruction overall). This round replaces ONLY the B path with the packed
// [N,K] n-major layout (P[n*K+k] = W[k*N+n], one-time out-of-timed-region
// transpose by w8a8_pack_gateup_nmajor_kernel for (k,n)==(4096,512)): B is
// staged per wave as [16 n][272 k] (272 = 256 + 16 bank skew) and loaded with
// matrix_b col_major fragments, turning each B fragment into ONE 8-byte
// ds_read2_b32/b64 (lane's 8 k bytes are contiguous; 64 lanes x 8 B = 512 B
// spread over 64 banks, max 2-way conflict). Geometry, A path (row-major
// [16][256], already one ds_read2_b32 per fragment), barrier structure,
// in-block split-K=4 exact int32 combine, and the direct register epilogue
// are byte-identical to iteration 2. Expected LDS reads per MMAC: ~9.25 -> ~2
// (1 A + 1 B). Falsifiable: if the median does not drop below 30.87 us, the
// 30.87 us wall is not the B-fragment LDS read path / bank conflicts, and the
// next lever is occupancy (grid split-K with a fused or measured combine) or
// A-side staging.
//
// Iteration 7 (pipeline round): single-buffering vs double-buffering
// comparison. All three gates hold (K=4096>=1024; fresh PMC L2 hit 48.716% <
// 70%; doubled LDS budget 40,960 B < 48 KiB after the per-stage chunk is
// shrunk from 256 to 128 K rows). The exact accepted-code-object ISA (source
// digest bb625614..., 32 blocks x 256 thr, 40 arch VGPR / 37,888 B LDS) shows
// the K loop is a serialized latency chain: each of the 8 staging vectors per
// 256-k sub-stage issues as global_load_dwordx4 immediately followed by
// s_waitcnt vmcnt(0) + ds_write_b128 (loop body at +0x148/+0x1c4), so every
// global load's full L2/DRAM latency is exposed with only 128 wavefronts on
// 120 CUs (1.07 waves/CU) to hide behind -- the dominant cost at 25.82 us.
// This round restructures staging into a legal double-buffered pipeline on
// the accepted geometry (32 blocks x 4 waves, in-block split-K=4, packed B):
// per-wave A/B stages become two 16-row x 144-stride (128 + 16 skew) buffers
// of 128-K-row chunks; the loop issues chunk c+1's 4 int4 loads per lane
// BEFORE chunk c's 4-MMAC burst and writes them to the other buffer after the
// burst, so the vmcnt wait overlaps the MMACs and one barrier per 128-k chunk
// suffices (1 prologue + 7 chunk + 1 combine = 9 barriers/block -- the same
// total as the single-buffered 9, i.e. 0.00220 per k-step both ways, but each
// new barrier gates a chunk whose loads were already in flight). A gains the
// same 144 bank skew (single-buffered A was [16][256] = exactly 64 LDS banks,
// all 16 rows aliased onto the same banks, ~16-way fragment conflicts): A
// fragment conflicts drop to ~4-way. Exact int32 k-ascending order per wave,
// split order, in-block combine, epilogue, exact-shape guard, packed layout,
// and the generic scalar fallback (incl. the b_packed M=2 decode) are
// byte-identical. No async-copy claim: the overlap is the buffer rotation
// plus compiler-scheduled early load issue (waits land at the ds_write).
//
// Iteration 8 (occupancy/resource round, mandated: tune one occupancy
// limiter -- waves per block / VGPR live range / LDS footprint / spill
// removal -- using the fresh PMC evidence, without trading repeated HBM
// reads for occupancy). Fresh PMC of the accepted iteration-7 kernel
// (source digest 7cbc3b41..., 32 blocks x 256 thr, 56 arch VGPR / 32 SGPR /
// 0 scratch, 40,960 B static LDS, profiled 18.72 us) shows the only
// occupancy limiter with headroom is waves per block: registers fit ~18
// waves (56 VGPR x 64 lanes = 3,584 VGPRs/wave vs 65,536/CU), scratch is 0,
// and the 40,960 B LDS caps residency at 1 block/CU -- but the grid is only
// 32 blocks, so the block-level LDS limit is not binding and the per-CU
// wave count is the lever. 8 is in the trusted split-K probe set for the
// 120-CU part, and the control-plane warning (32 blocks < 2 blocks/CU) is
// answered in wavefront terms: this probe doubles waves per block 4 -> 8,
// giving 32 blocks x 8 waves = 256 wavefronts = 2.13 waves/CU (the
// two-blocks-per-CU latency-hiding target) with NO grid split-K, NO combine
// kernel, and NO extra HBM reads (B stays the packed [N,K] once-read 2 MiB
// DRAM stream; A stays the L2-hot 64 KiB logical re-read; per-replay global
// reads unchanged at 2 MiB A (L2) + 2 MiB B (DRAM)). Per-wave K slices
// halve (1024 -> 512), so the double-buffer chunk shrinks 128 -> 64 K rows
// to keep LDS under 64 KiB/CU: per-wave A/B stages become 2 x [16][80]
// (64 + 16 bank skew), 8 chunks of 64 k per wave, ONE A + ONE B int4 load
// per lane per chunk (64 lanes x 16 B = 16 rows x 64 k per matrix), 2 MMACs
// per chunk, 1 prologue + 7 chunk + 1 combine = 9 barriers/block (same
// total as iteration 7, each gating a 64-k chunk with 8 independent waves
// behind it). The partial plane grows to 8 x 16 x 16 int32 = 8,192 B; the
// eight planes are summed in ascending split order (bit-exact int32:
// max |partial| = 512*128*128 = 8,388,608; full-K total 67,108,864 < 2^31,
// so no overflow and any grouping is exact) and each wave finalizes 2 rows
// (row = wave*2 + lane/32; each output written by 2 lanes with the same
// value). LDS/block 40,960 -> 49,152 B (still 1 block/CU
// < 64 KiB). Exact-shape guard, packed layout, generic scalar fallback
// (incl. the b_packed M=2 decode), and the Graph/current-stream contract
// are untouched. Falsifiable: if the median does not drop below 20.0658 us
// (or p90 fails <= 20.0786), doubling waves per block at 64-k chunks does
// not beat the 128-k double-buffered 4-wave chain, and the next lever is
// grid split-K with a measured combine (trusted sweep [2,3,4,7,8,10,11,15])
// or a reverted geometry. Raw inline asm: none (policy forbids it this
// round).
//
// Iteration 12 (register depth-2 prefetch round, final conditional inline-asm
// round: raw asm stays forbidden -- plateau=false, recent_valid_improvements
// [-4.52, -1.47, +0.28] has a regression, not three within [-2%, +2%); one
// HIP-only consolidation change). The control-plane pre-micro-optimization
// mandate was settled in iteration 9 (finer one-wave zero-barrier grid
// split-K=8 + separate combine: 12.5149/12.7389, rejected -4.52%) and the
// barrier/conflict levers were measured in iterations 10 (barrier removal
// 9->1: 12.1278/12.3229, median regressed) and 11 (per-wave LDS bank-phase
// rotation: 11.9168/12.5016, +0.28% -- noise, not accepted). The accepted
// kernel's exact-source ISA (digest 12fc6260..., 32 blocks x 512 thr, 40 arch
// VGPR / 32 SGPR / 0 scratch, 49,152 B LDS, profiled 14.4 us vs unprofiled
// 11.9497 us) pins the remaining wall: every 64-k chunk iteration is a
// serialized convoy -- chunk c+1's two global_load_dwordx4 issue at
// 0x36D0/0x36DC, the 2 v_mmac run at 0x3790/0x3800, then s_waitcnt vmcnt(1) +
// ds_write_b128 (A) at 0x3820/0x3824 and s_waitcnt vmcnt(0) + ds_write_b128
// (B) at 0x3830/0x3834 expose the full L2/DRAM latency at the staging store
// with only 2 x 16 B in flight per lane (16 KiB in flight per block vs ~76
// KiB needed to sustain the 175 GB/s request rate across ~700 cycles of DRAM
// latency -- latency-bound by ~5x). Iteration 9 proved the batch-load
// mechanism works mechanically (its loss was the 256-block geometry/combine
// cost, not the staging); this round consolidates that finding onto the
// accepted geometry. Change (single variable: register staging depth 1 -> 2
// chunks in flight per lane): the loop now issues chunk c+2's loads at the
// top of iteration c (a_pre2/b_pre2) and the prologue preloads chunk 1
// (a_pre/b_pre) before the prologue barrier, so each chunk's data lives in
// registers for one full extra chunk period + a barrier before its ds_write;
// the vmcnt wait at the staging store is then covered by the previous chunk's
// MMAC burst and the barrier slack instead of stalling the wave. 4 x int4 in
// flight per lane (64 B), 16 loads in flight per wave, 32 KiB per block; VGPR
// 40 -> ~48-56, scratch stays 0, and residency is unchanged (49,152 B LDS
// still caps at 1 block/CU; 8 waves x 64 lanes x ~56 VGPR = 28,672 << 65,536
// VGPRs/CU). Everything else is byte-identical: grid 32 x 512 thr, in-block
// split-K=8, 64-k double-buffer chunks, 80-stride stages, 9 barriers/block,
// A 2 MiB (L2-hot 64 KiB re-read 32x) + B 2 MiB (packed [N,K] once-read
// DRAM), exact int32 k-ascending per-wave order + ascending split combine,
// direct register epilogue, exact-shape guard (m==16 && n==512 && k==4096),
// one-time out-of-timed-region pack, generic scalar fallback (incl. the
// b_packed M=2 decode), and the Graph/current-stream contract. No
// asynchronous-copy claim: the overlap is compiler-scheduled load issue at
// depth 2 only. Falsifiable gate: normal-benchmark median_us <
// 11.949679851531982 AND p90_us <= 12.632080316543579 (accepted best).
// Predicted PMC/ISA signature if the mechanism is real: the vmcnt waits at
// the staging ds_write (0x3820/0x3830) move off the critical path -- lds_wait
// dropping from 5,212 and the profiled-vs-unprofiled gap shrinking -- with
// vmem_read_instructions unchanged at 4,608 (same loads, same bytes), LDS
// 49,152 B unchanged, grid 32, wg 512, 9 barriers, 40 -> ~48-56 VGPR, 0
// scratch. If flat or slower, the wall is the DRAM request service rate /
// barrier-locked slowest-wave tail at depth 2, and the next lever is a deeper
// occupancy probe (split-K=16, 512 blocks, planes exactly filling the 512 KiB
// workspace) or a reverted kernel. Raw inline asm: none (policy forbids it
// this round).
// ---------------------------------------------------------------------------
__global__ __launch_bounds__(512) void w8a8_dumma_m16_n16_k8_packedb_kernel(
    const int8_t* __restrict__ a,
    const int8_t* __restrict__ b,  // packed [N,K] n-major for this shape
    const float* __restrict__ x_scale,
    const float* __restrict__ weight_scale,
    hip_bfloat16* __restrict__ out,
    int n,
    int k) {
  constexpr int kTileM = 16;
  constexpr int kTileN = 16;
  constexpr int kTileK = 32;
  constexpr int kSplitK = 8;            // wavefronts per block
  constexpr int kStageK = 64;           // K rows per double-buffer chunk
  constexpr int kLdsStride = kStageK + 16;  // 80: 64 + 16 bank skew
  const int tid = static_cast<int>(threadIdx.x);
  const int wave = tid >> 6;  // 0..7
  const int lane = tid & 63;
  const int n0 = static_cast<int>(blockIdx.x) * kTileN;

  // Per-wave private double-buffered A/B stages: 8 waves x 2 buffers x
  // 16 x 80 = 20,480 B each; partial plane 8 x 16 x 16 int32 = 8,192 B;
  // total 49,152 B/block (under the 64 KiB LDS/CU limit, 1 block/CU).
  __shared__ __align__(16) int8_t lds_a[kSplitK * 2 * kTileM * kLdsStride];
  __shared__ __align__(16) int8_t lds_b[kSplitK * 2 * kTileN * kLdsStride];
  __shared__ __align__(16) int32_t s_part[kSplitK * kTileM * kTileN];

  du::dumma::DUFragment<du::dumma::matrix_a, kTileM, kTileN, kTileK,
                        signed char, du::dumma::row_major>
      a_frag;
  du::dumma::DUFragment<du::dumma::matrix_b, kTileM, kTileN, kTileK,
                        signed char, du::dumma::col_major>
      b_frag;
  du::dumma::DUFragment<du::dumma::accumulator, kTileM, kTileN, kTileK, int>
      acc_frag;
  du::dumma::du_fill_fragment(acc_frag, 0);

  // Each wave stages only its own K quarter (no cross-wave LDS sharing), so
  // the block-wide barriers act as ordering/pacing fences exactly as in the
  // single-buffered iteration-5 kernel; per-chunk hazards are covered by the
  // rotation: buffer (c+1)&1 is last read by chunk c-1, which every wave
  // finished before the previous barrier.
  int8_t* a_buf = lds_a + wave * (2 * kTileM * kLdsStride);
  int8_t* b_buf = lds_b + wave * (2 * kTileN * kLdsStride);

  const int k_lo = wave * (k / kSplitK);
  const int kChunks = (k / kSplitK) / kStageK;  // 8 for k=4096, split 8

  // Prologue: issue chunk 0's 1 A + 1 B int4 load per lane (64 lanes x
  // 16 B = 16 rows x 64 k per matrix; lane -> (row = lane>>2, kk16 =
  // (lane&3)<<4)) and stage them into buffer 0. Iteration 12 (register
  // depth-2 prefetch): chunk 1's loads (a_pre/b_pre) are also issued here,
  // BEFORE the prologue barrier, so they are in flight across the barrier and
  // land while chunk 0 computes -- the vmcnt wait at their staging store in
  // iteration 0 is covered instead of exposed. The rotation (a_pre2/b_pre2
  // loaded at the top of each loop iteration) keeps two chunks in flight per
  // lane for the whole K loop.
  int4 a_pre, b_pre, a_pre2, b_pre2;
  {
    const int row = lane >> 2;
    const int kk16 = (lane & 3) << 4;
    *reinterpret_cast<int4*>(a_buf + row * kLdsStride + kk16) =
        *reinterpret_cast<const int4*>(a + row * k + k_lo + kk16);
    *reinterpret_cast<int4*>(b_buf + row * kLdsStride + kk16) =
        *reinterpret_cast<const int4*>(b + (n0 + row) * k + k_lo + kk16);
    if (kChunks > 1) {
      const int k1 = k_lo + kStageK;
      a_pre = *reinterpret_cast<const int4*>(a + row * k + k1 + kk16);
      b_pre =
          *reinterpret_cast<const int4*>(b + (n0 + row) * k + k1 + kk16);
    }
  }
  __syncthreads();  // chunk 0 staged before any wave computes it

  for (int c = 0; c < kChunks; ++c) {
    const bool has_next = (c + 1 < kChunks);
    const bool has_next2 = (c + 2 < kChunks);

    // Iteration 12 (register depth-2 prefetch): issue chunk c+2's global
    // loads NOW, before this chunk's MMAC burst, so two chunks (c+1 in
    // a_pre/b_pre, loaded one full iteration ago; c+2 in a_pre2/b_pre2, in
    // flight now) are outstanding per lane -- 4 x 16 B vs 2 today. The
    // compiler's vmcnt waits land at the ds_write below, but by then the
    // waited-on loads have had a full extra chunk period + the previous
    // barrier to complete, so the staging store no longer exposes the raw
    // L2/DRAM latency.
    if (has_next2) {
      const int k_abs = k_lo + (c + 2) * kStageK;
      const int row = lane >> 2;
      const int kk16 = (lane & 3) << 4;
      a_pre2 = *reinterpret_cast<const int4*>(a + row * k + k_abs + kk16);
      b_pre2 =
          *reinterpret_cast<const int4*>(b + (n0 + row) * k + k_abs + kk16);
    }

    // Compute this chunk from buffer (c & 1): 2 k-ascending m16n16k32 steps.
    int8_t* a_cur = a_buf + (c & 1) * (kTileM * kLdsStride);
    int8_t* b_cur = b_buf + (c & 1) * (kTileN * kLdsStride);
#pragma unroll
    for (int kk = 0; kk < kStageK; kk += kTileK) {
      du::dumma::du_load_matrix_sync(a_frag, a_cur + kk, kLdsStride);
      du::dumma::du_load_matrix_sync(b_frag, b_cur + kk, kLdsStride);
      du::dumma::du_mma_sync(acc_frag, a_frag, b_frag, acc_frag);
    }

    // Stage chunk c+1 into the other buffer from the registers loaded one
    // full iteration ago (uniform branch: safe to fence).
    if (has_next) {
      int8_t* a_dst = a_buf + ((c + 1) & 1) * (kTileM * kLdsStride);
      int8_t* b_dst = b_buf + ((c + 1) & 1) * (kTileN * kLdsStride);
      const int row = lane >> 2;
      const int kk16 = (lane & 3) << 4;
      *reinterpret_cast<int4*>(a_dst + row * kLdsStride + kk16) = a_pre;
      *reinterpret_cast<int4*>(b_dst + row * kLdsStride + kk16) = b_pre;
      __syncthreads();  // chunk c+1 visible; buffer (c&1) free for c+2 staging
    }

    // Rotate the register pipeline: chunk c+2 becomes the next one-ahead.
    a_pre = a_pre2;
    b_pre = b_pre2;
  }

  // Publish the eight 16x16 int32 partials to LDS, then sum them in
  // ascending split order (bit-exact int32; max |partial| = 512*128*128 =
  // 8,388,608 and the full-K total 67,108,864 stays below 2^31, so no
  // overflow) and store the scaled bf16 output directly. Each wave
  // finalizes 2 rows of the tile: row = wave*2 + lane/32, col = lane%16
  // (each of the 256 outputs is written by 2 lanes with the identical
  // value, a benign duplicate store), and the partial plane layout is
  // row-major [16][16] int32, so the read address is linear in the lane.
  du::dumma::du_store_matrix_sync(s_part + wave * (kTileM * kTileN), acc_frag,
                                  kTileN, du::dumma::mem_row_major);
  __syncthreads();

  const int row = wave * 2 + (lane >> 5);  // 2 rows per wave, 16 total
  const int col = lane & 15;
  int32_t acc = 0;
#pragma unroll
  for (int s = 0; s < kSplitK; ++s) {
    acc += s_part[s * (kTileM * kTileN) + row * kTileN + col];
  }
  const float scaled =
      static_cast<float>(acc) * x_scale[row] * weight_scale[n0 + col];
  out[row * n + n0 + col] = static_cast<hip_bfloat16>(scaled);
}

// ---------------------------------------------------------------------------
// Iteration 14 (final conditional inline-asm round: raw asm stays forbidden
// because the control plane has NOT confirmed a HIP plateau -- plateau=false,
// recent_valid_improvements_percent [+0.28, +2.93, +9.03] is not three values
// within [-2%, +2%) -- so this is one HIP-only consolidation change). The
// accepted iteration-13 kernel (10.6473/10.7777 us; fresh/exact-source PMC:
// grid 480 x 64 thr, 64 arch VGPR/32 SGPR/0 scratch, 12,800 B LDS, profiled
// 9.76 us, lds_bank_conflicts 65,536 on 12,288 LDS instructions, vmem_read
// 4,096, L2 42.74%) established the one-wave zero-barrier grid family's
// measured occupancy trend: split-K=8 (256 blocks, 2.13 blocks/CU) measured
// 12.5149/12.7389 us (rejected, iter 9) while split-K=15 (480 blocks, exactly
// 4.0 blocks/CU) measured 10.6473/10.7777 us (+9.03% vs the 32-block in-block
// kernel) -- more blocks/CU within the same mechanism is the measured
// direction, and iteration 12's accepted hypothesis explicitly named the next
// lever as "split-K=16, 512 blocks = 4.27 blocks/CU, planes exactly filling
// the 512 KiB workspace capacity". This round probes exactly that point,
// outside the trusted probe set {2,3,4,7,8,10,11,15} because evidence
// supports it: (a) the 2.13 -> 4.0 blocks/CU sweep is monotonic in the
// measured direction; (b) S=15's non-uniform slicing (13 x 256 + 2 x 384)
// leaves two 1.5x-work 384-k straggler blocks (s=13,14) whose host CUs define
// the grid-kernel tail, and S=16 makes ALL 512 blocks uniform 256-k work (8
// MMAC steps, 4 int4 loads per lane per matrix) -- the stragglers disappear;
// (c) the shape contract allocates workspace_split_k_capacity = min(16,
// budget, k/32) = 16 planes of m*n int32 = exactly 524,288 B, so S=16 fits
// with zero slack (the "exact-fill, zero-margin" concern that selected 15
// last round is now falsifiable; if the guard fails, the accepted in-block
// kernel runs and correctness is preserved). Change (single variable: split-K
// 15 -> 16 on the accepted one-wave grid mechanism; everything else
// byte-identical): grid 480 -> 512 blocks x 64 thr (4.27 blocks/CU; 5
// co-resident blocks on 32 CUs: LDS 5 x 12,800 = 64,000 B <= 64 KiB/CU, VGPR
// 64 x 64 x 5 = 20,480 <= 65,536/CU, so no residency cliff); uniform
// ascending 64-aligned slices k0 = s*256, sliceK = 256 (max |partial| =
// 256*128*128 = 4,194,304; full-K total 67,108,864 < 2^31; ascending split
// sum order preserved, bit-exact int32 identical to the accepted combine);
// the staging code collapses to the single 4-load path (no 6-load branch;
// one static unrolled 8-step zero-barrier K loop, one staging wait per
// block); LDS stride 400 unchanged (12,800 B/block); each block publishes
// its 16x16 int32 partial to workspace plane [tile][s] (32 x 16 x 1,024 B =
// 524,288 B = exactly the 512 KiB this shape allocates); the stateless
// combine kernel (grid 32 x 256 thr, same stream, inside the Graph -- the
// timed region includes the combine per the operator contract) sums the 16
// planes per tile in ascending split order and applies x_scale/weight_scale/
// bf16. Byte accounting: global reads unchanged (A 2 MiB = L2-hot 64 KiB
// re-read 32x, B 2 MiB once from DRAM); workspace plane writes 512 KiB +
// combine plane reads 512 KiB (L2-hot) + bf16 out 16 KiB per replay. The
// accepted iteration-12 in-block kernel stays in the file byte-identical as
// the workspace-insufficient fallback (defensive). Exact-shape guard
// (m==16 && n==512 && k==4096), packed [N,K] layout + one-time pack, generic
// scalar fallback (incl. the b_packed M=2 decode), and the Graph/current-
// stream contract are untouched. No asynchronous-copy claim: the batch load
// is compiler-scheduled memory-level parallelism only. Falsifiable gate:
// normal-benchmark median_us < 10.647284984588623 AND p90_us <=
// 10.777679681777954 (accepted best). Predicted PMC signature if the
// mechanism is real: primary kernel grid_blocks 480 -> 512 with kernel name
// w8a8_dumma_m16_n16_sk16_grid_kernel, workgroup_size 64, lds_bytes 12,800,
// barriers/block 0, vmem_read 4,096, lds_bank_conflicts unchanged (~65k: the
// 400-stride residual is not the lever -- iter 11 measured conflict removal
// as noise), and the per-block uniformity (no 384-k blocks) shortening the
// grid-kernel tail -- the normal-benchmark median/P90 are the score. If flat
// or slower, the wall is the grid-level DRAM request service rate / 5-block-
// CU imbalance at 4.27 blocks/CU and the occupancy family is exhausted at
// this shape: the next lever is the combine launch (a fused last-arrival
// combine needs an out-of-Graph counter pre-zero in the Python layer) or a
// revert to the accepted S=15 kernel. Raw inline asm: none (policy forbids
// it this round).
// ---------------------------------------------------------------------------
__global__ __launch_bounds__(64) void w8a8_dumma_m16_n16_sk16_grid_kernel(
    const int8_t* __restrict__ a,
    const int8_t* __restrict__ b,  // packed [N,K] n-major for this shape
    int32_t* __restrict__ planes,  // [32 tiles][16 splits][16][16] int32
    int n,
    int k) {
  constexpr int kTileM = 16;
  constexpr int kTileN = 16;
  constexpr int kTileK = 32;
  constexpr int kSplitK = 16;
  constexpr int kStride = 384 + 16;  // 400: 256-k slice + bank skew
  const int lane = static_cast<int>(threadIdx.x);  // 64 lanes, one wavefront
  const int tile = static_cast<int>(blockIdx.x) / kSplitK;  // 0..31
  const int s = static_cast<int>(blockIdx.x) % kSplitK;     // 0..15
  const int n0 = tile * kTileN;
  // Uniform ascending 64-aligned slicing: 16 x 256 = 4,096.
  const int sliceK = 256;
  const int k0 = s * 256;

  __shared__ __align__(16) int8_t lds_a[kTileM * kStride];
  __shared__ __align__(16) int8_t lds_b[kTileN * kStride];

  // Stage the whole slice once: all int4 global loads issue back-to-back
  // before any ds_write (no __syncthreads -- single wavefront; the compiler
  // orders the LDS write-before-read with s_waitcnt lgkmcnt). Uniform 256-k
  // slice -> single static unrolled 4-load path for every block.
  const int row = lane >> 2;         // 0..15
  const int kk16 = (lane & 3) << 4;  // 0,16,32,48
  int4 va[4], vb[4];
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    va[i] = *reinterpret_cast<const int4*>(a + row * k + k0 + i * 64 + kk16);
    vb[i] = *reinterpret_cast<const int4*>(
        b + (n0 + row) * k + k0 + i * 64 + kk16);
  }
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    *reinterpret_cast<int4*>(lds_a + row * kStride + i * 64 + kk16) = va[i];
    *reinterpret_cast<int4*>(lds_b + row * kStride + i * 64 + kk16) = vb[i];
  }

  du::dumma::DUFragment<du::dumma::matrix_a, kTileM, kTileN, kTileK,
                        signed char, du::dumma::row_major>
      a_frag;
  du::dumma::DUFragment<du::dumma::matrix_b, kTileM, kTileN, kTileK,
                        signed char, du::dumma::col_major>
      b_frag;
  du::dumma::DUFragment<du::dumma::accumulator, kTileM, kTileN, kTileK, int>
      acc_frag;
  du::dumma::du_fill_fragment(acc_frag, 0);

  // Zero-barrier LDS-only K loop: 8 k-ascending m16n16k32 steps.
  for (int kk = 0; kk < sliceK; kk += kTileK) {
    du::dumma::du_load_matrix_sync(a_frag, lds_a + kk, kStride);
    du::dumma::du_load_matrix_sync(b_frag, lds_b + kk, kStride);
    du::dumma::du_mma_sync(acc_frag, a_frag, b_frag, acc_frag);
  }

  // Publish the 16x16 int32 partial to workspace plane [tile][s]; the
  // separate combine kernel sums the 16 planes in ascending split order.
  du::dumma::du_store_matrix_sync(
      planes + (tile * kSplitK + s) * (kTileM * kTileN), acc_frag, kTileN,
      du::dumma::mem_row_major);
}

// ---------------------------------------------------------------------------
// Iteration 14: stateless combine for the sk16 one-wave grid. One thread per
// output element (grid 32 tiles x 256 threads = 8,192 outputs); each thread
// reads its 16 int32 partials from the [tile][s] planes in ascending split
// order (bit-exact int32, same order as the accepted combine), applies
// x_scale/weight_scale, and stores bf16. Launched on the same stream inside
// the Graph immediately after the grid kernel, so the timed region includes
// the split-K combine exactly as the operator contract requires.
// ---------------------------------------------------------------------------
__global__ __launch_bounds__(256) void w8a8_gateup_m16_sk16_combine_kernel(
    const int32_t* __restrict__ planes,  // [32][16][16][16] int32
    const float* __restrict__ x_scale,
    const float* __restrict__ weight_scale,
    hip_bfloat16* __restrict__ out,
    int n) {
  constexpr int kTileM = 16;
  constexpr int kTileN = 16;
  constexpr int kSplitK = 16;
  const int tile = static_cast<int>(blockIdx.x);
  const int row = static_cast<int>(threadIdx.x) >> 4;
  const int col = static_cast<int>(threadIdx.x) & 15;
  const int32_t* p =
      planes + tile * (kSplitK * kTileM * kTileN) + row * kTileN + col;
  int32_t acc = 0;
#pragma unroll
  for (int s = 0; s < kSplitK; ++s) {
    acc += p[s * (kTileM * kTileN)];
  }
  const float scaled = static_cast<float>(acc) * x_scale[row] *
                       weight_scale[tile * kTileN + col];
  out[row * n + tile * kTileN + col] = static_cast<hip_bfloat16>(scaled);
}

// ---------------------------------------------------------------------------
// Iteration 5: one-time out-of-timed-region packing for (k,n)==(4096,512).
// Transposes the logical [K,N] weight into the packed [N,K] n-major layout
// packed[n*K + k] = W[k*N + n] so the exact-shape kernel's matrix_b col_major
// fragment loads read 8 contiguous k bytes (one 8-byte LDS read per fragment)
// and the generic scalar fallback decodes b[col*K + kk] for the same (k,n).
// 64(k) x 64(n) tile per block via an LDS transpose (grid = (K/64, N/64)).
// ---------------------------------------------------------------------------
__global__ __launch_bounds__(256) void w8a8_pack_gateup_nmajor_kernel(
    const int8_t* __restrict__ src,  // W[K][N] row-major
    int8_t* __restrict__ dst,        // P[N][K] n-major
    int k,
    int n) {
  constexpr int kTile = 64;
  constexpr int kStride = 80;  // 64 + 16 bank skew
  __shared__ __align__(16) int8_t lds[kTile * kStride];
  const int k0 = static_cast<int>(blockIdx.x) * kTile;
  const int n0 = static_cast<int>(blockIdx.y) * kTile;
  const int tid = static_cast<int>(threadIdx.x);

  // Load: W row k0+kk, 16 consecutive n columns -> LDS row kk.
  const int kk = tid >> 2;             // 0..63
  const int nn16 = (tid & 3) * 16;     // 0,16,32,48
  *reinterpret_cast<int4*>(&lds[kk * kStride + nn16]) =
      *reinterpret_cast<const int4*>(src + (k0 + kk) * n + n0 + nn16);
  __syncthreads();

  // Store: P[n0+nn][k0 + kk16*16 .. +15] is 16 contiguous k bytes.
  const int nn = tid & 63;             // 0..63
  const int kk16 = tid >> 6;           // 0..3
  int4 outv;
  int8_t* op = reinterpret_cast<int8_t*>(&outv);
#pragma unroll
  for (int i = 0; i < 16; ++i) {
    op[i] = lds[(kk16 * 16 + i) * kStride + nn];
  }
  *reinterpret_cast<int4*>(dst + (n0 + nn) * k + k0 + kk16 * 16) = outv;
}

// ---------------------------------------------------------------------------
// Identity device-to-device packing helpers (bootstrap pack_weight).
// packed_weight[i] = raw_weight[i] (K*N int8), packed_scale[i] = scale[i]
// (N fp32). Later Parallel explore rounds may replace these kernels with a
// real packed layout plus the matching GEMM interpretation; the generic
// identity copy stays the fallback for unmatched (K,N).
// ---------------------------------------------------------------------------
__global__ __launch_bounds__(256) void w8a8_identity_pack_i8_kernel(
    const int8_t* __restrict__ src,
    int8_t* __restrict__ dst,
    int64_t count) {
  const int64_t idx =
      static_cast<int64_t>(blockIdx.x) * static_cast<int64_t>(blockDim.x) +
      static_cast<int64_t>(threadIdx.x);
  if (idx < count) {
    dst[idx] = src[idx];
  }
}

__global__ __launch_bounds__(256) void w8a8_identity_pack_f32_kernel(
    const float* __restrict__ src,
    float* __restrict__ dst,
    int count) {
  const int idx = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x) +
                  static_cast<int>(threadIdx.x);
  if (idx < count) {
    dst[idx] = src[idx];
  }
}

}  // namespace

// ---------------------------------------------------------------------------
// Timed GEMM entry point. Launches only on the caller-provided stream; no
// allocation, no synchronization inside the timed region. The iteration-13
// exact-shape path uses the caller-provided workspace (preallocated before
// Graph capture) as the split-K partial-plane buffer; the scalar path needs
// no split-K partials and leaves workspace unused.
// ---------------------------------------------------------------------------
extern "C" void launch_w8a8_gemm(
    const int8_t* a,
    const int8_t* b,
    const float* x_scale,
    const float* weight_scale,
    void* out,
    void* workspace,
    int64_t workspace_bytes,
    int m,
    int n,
    int k,
    hipStream_t stream) {
  if (m <= 0 || n <= 0 || k <= 0) {
    return;
  }

  // Exact-shape specialization: TP8 shared_gate_up_proj decode (M=16).
  // Iteration 14: one-wave zero-barrier grid, split-K=16 (512 blocks x 64
  // thr = 4.27 blocks/CU, uniform ascending 256-k slices, whole-slice
  // 400-stride LDS staging, zero barriers) writing 512 KiB of int32 partial
  // planes to the shape's workspace (capacity = exactly 16 planes), plus a
  // stateless combine kernel on the same stream inside the Graph (the timed
  // region includes the combine exactly per the operator contract). The
  // accepted iteration-12 in-block split-K=8 depth-2-prefetch kernel (32
  // blocks x 512 thr, workspace-free) remains the fallback if the workspace
  // is smaller than the plane contract (defensive: the Python shape contract
  // allocates 512 KiB, so this is defensive only). b is the packed [N,K]
  // n-major layout produced one-time out-of-timed-region by
  // launch_pack_w8a8_weight for this (k,n). The generic scalar path below
  // remains the fallback for every other shape, including the paired M=2
  // API shapes with the same (N,K), which decode the packed [N,K] B layout
  // through the b_packed flag.
  if (m == 16 && n == 512 && k == 4096) {
    constexpr int64_t kPlanesBytes = 32 * 16 * 16 * 16 * 4;  // 524,288
    if (workspace != nullptr && workspace_bytes >= kPlanesBytes) {
      hipLaunchKernelGGL(
          w8a8_dumma_m16_n16_sk16_grid_kernel,
          dim3(512),
          dim3(64),
          0,
          stream,
          a,
          b,
          reinterpret_cast<int32_t*>(workspace),
          n,
          k);
      hipLaunchKernelGGL(
          w8a8_gateup_m16_sk16_combine_kernel,
          dim3(32),
          dim3(256),
          0,
          stream,
          reinterpret_cast<const int32_t*>(workspace),
          x_scale,
          weight_scale,
          reinterpret_cast<hip_bfloat16*>(out),
          n);
    } else {
      hipLaunchKernelGGL(
          w8a8_dumma_m16_n16_k8_packedb_kernel,
          dim3(static_cast<unsigned>(n / 16)),
          dim3(512),
          0,
          stream,
          a,
          b,
          x_scale,
          weight_scale,
          reinterpret_cast<hip_bfloat16*>(out),
          n,
          k);
    }
    return;
  }

  // Exact-shape specialization: TP8 shared_down_proj decode (M=16, N=4096,
  // K=256). Iteration 2 (architecture round): launch-geometry probe -- 2
  // wavefronts per block (128 thr), one 16x16 output tile per block (grid =
  // N/16 = 256 blocks), in-block split-K = 2 along K (K-half = 128 per wave,
  // 4 m16n16k32 steps each), whole K=256 slice staged once into 272-stride
  // LDS from the packed [N,K] n-major B layout, one END-of-K barrier,
  // LDS-plane ascending split combine + fused scale/bf16 epilogue. b is the
  // packed [N,K] layout produced one-time out-of-timed-region by
  // launch_pack_w8a8_weight for (k,n)==(256,4096); the generic scalar path
  // below stays the fallback for every other shape, including the paired
  // M=2 API shapes with the same (N,K), which decode the packed [N,K] B
  // layout through the b_packed flag.
  if (m == 16 && n == 4096 && k == 256) {
    hipLaunchKernelGGL(
        w8a8_dumma_m16_n16_k256_sk2_kernel,
        dim3(static_cast<unsigned>(n / 16)),
        dim3(128),
        0,
        stream,
        a,
        b,
        x_scale,
        weight_scale,
        reinterpret_cast<hip_bfloat16*>(out),
        n,
        k);
    return;
  }

  constexpr int kBlockThreads = 128;  // multiple of wavefront size 64
  const int64_t total = static_cast<int64_t>(m) * n;
  const int grid =
      static_cast<int>((total + kBlockThreads - 1) / kBlockThreads);
  const int b_packed =
      (k == 4096 && n == 512) || (k == 256 && n == 4096) ? 1 : 0;

  hipLaunchKernelGGL(
      w8a8_scalar_gemm_kernel,
      dim3(static_cast<unsigned>(grid)),
      dim3(static_cast<unsigned>(kBlockThreads)),
      0,
      stream,
      a,
      b,
      x_scale,
      weight_scale,
      reinterpret_cast<hip_bfloat16*>(out),
      m,
      n,
      k,
      b_packed);
}

// ---------------------------------------------------------------------------
// Optional out-of-timed-region weight packing. For the TP8 gate_up shape
// (K=4096, N=512) and the TP8 shared_down_proj shape (K=256, N=4096) this
// writes the packed [N,K] n-major transpose (P[n*K+k] = W[k*N+n]) consumed
// by the exact-shape DUMMA kernels (col_major B fragments) and decoded by
// the generic scalar fallback (b_packed flag, incl. paired M=2 API shapes).
// Every other (K,N) keeps the identity device-to-device copy, valid for all
// shapes.
// ---------------------------------------------------------------------------
extern "C" void launch_pack_w8a8_weight(
    const int8_t* raw_weight,
    const float* weight_scale,
    int8_t* packed_weight,
    float* packed_weight_scale,
    int k,
    int n,
    hipStream_t stream) {
  constexpr int kBlockThreads = 256;
  const int64_t count = static_cast<int64_t>(k) * n;

  if ((k == 4096 && n == 512) || (k == 256 && n == 4096)) {
    hipLaunchKernelGGL(
        w8a8_pack_gateup_nmajor_kernel,
        dim3(static_cast<unsigned>(k / 64), static_cast<unsigned>(n / 64)),
        dim3(static_cast<unsigned>(kBlockThreads)),
        0,
        stream,
        raw_weight,
        packed_weight,
        k,
        n);
  } else if (count > 0) {
    const int grid =
        static_cast<int>((count + kBlockThreads - 1) / kBlockThreads);
    hipLaunchKernelGGL(
        w8a8_identity_pack_i8_kernel,
        dim3(static_cast<unsigned>(grid)),
        dim3(static_cast<unsigned>(kBlockThreads)),
        0,
        stream,
        raw_weight,
        packed_weight,
        count);
  }

  if (n > 0) {
    const int grid = (n + kBlockThreads - 1) / kBlockThreads;
    hipLaunchKernelGGL(
        w8a8_identity_pack_f32_kernel,
        dim3(static_cast<unsigned>(grid)),
        dim3(static_cast<unsigned>(kBlockThreads)),
        0,
        stream,
        weight_scale,
        packed_weight_scale,
        n);
  }
}
// @@end
