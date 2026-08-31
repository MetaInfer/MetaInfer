// @@variant shape=minimax_tp8_qkv_proj_m16 commit=b86aaad82e2a39632cd692ec7fe88dcaeeadaaed added=2026-08-28
//   median_us=33.2 p90_us=33.31
//   source=minimax-dsh-tp8-m16-1-78402260
// MetaInfer W8A8 INT8 GEMM — gfx928 (K500SM_AI) bootstrap.
//
// worker_0 owns physical GPU 0 and the exact shape
//     minimax_tp8_qkv_proj_m16 : M=16, N=1280, K=6144
//
// Iteration 0 (bootstrap, correctness-first): one simple scalar int8
// dot-product kernel is used for every shape, plus an identity device-to-
// device `pack_weight`. No DUMMA, no split-K, no LDS staging, no inline
// assembly: the goal is a kernel that compiles on gfx928, is CUDA/HIP-Graph
// capture safe, and reproduces the exact int64 CPU reference bit-for-bit.
//
// Iteration 1 (DUMMA bootstrap): the exact assigned shape
// minimax_tp8_qkv_proj_m16 (M=16, N=1280, K=6144) ran the minimal
// DUMMA m16n16k32 INT8 kernel: one 64-lane wavefront per 16x16 output
// tile (80 one-wave blocks, no cross-wave barrier, no LDS staging, no
// split-K), direct global fragment loads and explicit int32 accumulation,
// then x_scale[m] * packed_weight_scale[n] scaling and bf16 store.
// Measured 206.9 us median (logical 1.216 TOPs, 38.7 GB/s): latency/
// occupancy-bound, not bandwidth-bound.
//
// Iteration 2 (architecture round, grid parallelism): same minimal direct
// DUMMA m16n16k32 tile and one-wave zero-barrier blocks, grid raised from
// 80 blocks to 240 with split-K=3 (exactly 2 one-wave blocks per CU on all
// 120 CUs). Each block computes its K/3=2048 (64 k32 steps) int32 partial
// over a uniform K-slice with the same direct global fragment loads, stores
// it to a workspace plane, and a separate 80-block combine kernel sums the
// 3 planes ascending (exact int32), applies the scales and stores bf16.
// Measured floor improved (min 162 us) but the median exploded
// (416.7 us, p90 781.4): the direct byte-load path's per-step latency is
// already inflated ~1.27x at 2 waves/CU, so pure split-K occupancy past
// 2 blocks/CU has to be probed before any load-path micro-optimization.
//
// Iteration 3 (architecture round, split-K occupancy sweep completion):
// keep the identical direct-load DUMMA m16n16k32 partial + combine family,
// but make the split-K count a compile-time template over the two legal
// probes that remain unmeasured for this shape:
//     SPLIT_K=2 (default, measured): grid = 160 one-wave zero-barrier
//         blocks = a finer one-wave grid (80 -> 160) with the mildest
//         occupancy step (40 CUs carry 2 waves, 80 CUs carry 1 wave) and
//         the per-wave K-chain halved from 192 to 96 k32 steps; predicted
//         to stay in the stable latency regime of iteration 1 while
//         shortening every wave's serial load->mma chain.
//     SPLIT_K=6 (compiled alternative, one constant away): grid = 480
//         one-wave blocks = exactly 4 blocks per CU (CU-aligned,
//         non-power-of-two), per-wave K-chain = 32 k32 steps; the strongest
//         CU-aligned occupancy probe in the trusted set {2,3,4,5,6,8,9,12}
//         whose K=6144 chunk (1024) stays k32-aligned.
// Both configurations write int32 partials into the caller workspace plane
// (tile, s) and run the ascending-sum combine+scale+bf16 kernel in the
// timed Graph replay, so split-K combine cost is part of the operator wall.
// All other (m, n, k) keep the generic scalar fallback.
//
// Iteration 10 (final conditional inline-asm round; HIP-only consolidation
// per the control-plane policy: plateau=false, raw_inline_asm_allowed=false):
// the measured geometry is flipped from SPLIT_K=2 to SPLIT_K=6, completing
// the split-K occupancy sweep the control plane keeps mandating before any
// micro-optimization ("benchmark a finer one-wave zero-barrier grid or
// multiple legal split-K candidates including combine cost"). Iteration 3
// (SPLIT_K=2: 160 blocks, 96 k32 steps/wave) is the accepted champion at
// 106.8 us; iteration 2 (SPLIT_K=3: 240 blocks, 64 steps/wave) was
// pathological (median 416.7 us); iteration 9's LDS staging on the S=2
// geometry regressed to 115.5 us and thereby falsified 'byte loads are the
// per-step poison' (per its own acceptance criterion), pointing at the
// one-wave grid / per-CU occupancy as the remaining lever. SPLIT_K=6 is the
// strongest CU-aligned candidate in the trusted set {2,3,4,5,6,8,9,12} for
// this shape: grid = (N/16)*6 = 480 one-wave zero-barrier blocks = exactly
// 4 blocks per CU on all 120 CUs (non-power-of-two, CU-aligned), K-chunk =
// 1024 = 32 k32 steps per wave (k32-aligned), 6 int32 partial planes
// (491,520 B) inside the 16-plane / 1,310,720 B workspace capacity, and the
// 80-block ascending-sum combine stays in the timed Graph replay, so the
// full combine cost is measured. Load path, layout, accumulation order and
// kernels are otherwise byte-identical to the accepted S=2 geometry, so
// correctness stays exact int32/bf16 (0 mismatches) and graph capture is
// unchanged. Prediction: if per-step latency does NOT inflate with waves/CU
// (S=2 measured ~0.56 us/step), 4 waves x 32 steps = 128 serial step-times
// vs 2 x 96 = 192 -> median ~72-80 us including combine (1.35-1.5x vs
// 106.8); if it does inflate like the S=3 probe (~1.27x at 2 waves/CU), the
// median lands well above 106.8 and falsifies further occupancy probing.
//
// Iteration 11 (HIP-only, rejected 66.34 us): SPLIT_K=12 (960 blocks = 8/CU
// = 2 waves/SIMD) sat just above the S=6 champion (64.46 us) with the same
// 491,520 L2 sector requests -> the L2 request-throughput ceiling is
// ~7.4-7.6 G sector/s and wave count >= 480 is sufficient, so pure split-K
// occupancy is closed (the remaining trusted candidates S=4/S=8 have uneven
// 2.67/5.33 blocks-per-CU tails and cannot beat the champion's 128 step
// path). Iteration 12 (HIP-only, rejected 83.65 us): depth-1 register
// prefetch (double-buffered fragments) regressed, falsifying HIP-level load
// scheduling. The untouched axis is the request COUNT itself: A is re-read
// once per n16 tile (80x), i.e. A contributes 245,760 of the 491,520 L2
// sectors. Iteration 13: one wave owns TWO adjacent n16 tiles (n32 block)
// sharing a single A fragment; with SPLIT_K=12 the grid stays 480 one-wave
// zero-barrier blocks (4/CU = 1/SIMD, the champion's MLP level), steps per
// wave halve to 16 (2 mmacs per step), and L2 sectors drop to 368,640
// (-25%). See the kernel comment for the full prediction.
//
// Iteration 14 (HIP-only consolidation, champion 60.374 us): the partial
// kernel's per-wave 384-load chain (16 steps x 24 byte loads) and the
// combine kernel's per-lane 48-load chain (12 planes x 4 strided dword
// loads) are both serialized-load latency bound. The remaining HIP lever at
// the champion's 480-block / 1-wave-per-SIMD geometry is the combine: the
// plane element order is permuted to lane-major (lane l owns plane[l*4..l*4+3])
// in BOTH kernels, so the partial stores one dwordx4 per lane per plane
// (vmem_write 3,840 -> 960) and the combine reads one dwordx4 per lane per
// split (48 scalar loads -> 12 dwordx4 loads per lane). Values, ascending
// int32 order, workspace bytes, grid, graph structure and shape guards are
// unchanged; sub-480-block probes (n64 S=12 -> 240 blocks, n32 S=6 -> 240
// blocks) are closed by the session's own sector model (S=2 160 blocks 4.60
// G/s, S=3 240 blocks ~1.2-3 G/s vs 7.4-7.6 G/s at >= 480 waves).
//
// Iteration 15 (final conditional inline-asm round; HIP-only consolidation
// per the control-plane policy: plateau=false, raw_inline_asm_allowed=false,
// skill_allowed=false): apply iteration 14's "vector loads beat serialized
// scalar chains" medicine to the partial kernel's A fragment. The accepted
// code object emits 24 scalar global_load_ubyte per lane per k32 step in six
// 4-load groups, each group followed by a full count-limited s_waitcnt
// vmcnt(3..0) chain and a 4-instruction byte-pack (v_and/v_lshlrev/v_or3),
// i.e. six serialized wait+pack segments per step. du_load_matrix_sync
// (matrix_a, row_major, ldm = k) maps lane l to the EIGHT CONTIGUOUS K
// elements a[row = l & 15][col = ((l >> 4) << 3) .. +7], and du_mma_sync
// consumes the fragment as one 64-bit little-endian value (8 bytes), so one
// 8-byte-aligned global_load_dwordx2 replaces A's 8 scalar byte loads + 2
// wait chains + 8 pack VALU per step with identical register values (0
// mismatches). B fragments stay scalar: matrix_b row_major bytes are strided
// by the K-major row stride (n) per lane, i.e. contiguous only in a repacked
// n-major layout, the axis iteration 5 already falsified. This cuts
// vmem_read 184,320 -> 130,560 (-29%, requests only; the 368,640-sector L2
// footprint is unchanged), so the round discriminates the session's two
// candidate ceilings: if per-wait-chain serialization is on the critical
// path, per-step ~3.3 us -> ~2.2-2.6 us (partial ~35-42 us); if the L2
// sector-rate ceiling (~7 G sector/s) is binding instead, the median stays
// flat ~52-55 us partial.
//
// Contract (see int8_w8a8_gemm_api.py):
//     out[m, n] = bf16( int32_dot(x_q[m, :], weight[:, n])
//                       * x_scale[m, 0] * weight_scale[n, 0] )
// x_q        : int8  [M, K]  row-major
// weight     : int8  [K, N]  row-major (identity packed layout this round;
//                       later rounds may change pack_weight + GEMM together)
// x_scale    : fp32  [M, 1]
// weight_scale : fp32 [N, 1]
// out        : bf16  [M, N]  row-major, caller-provided
// workspace  : uint8, caller-allocated before Graph capture; the exact
//                       shape path uses SPLIT_K x 80 x 256 int32
//                       (SPLIT_K=12: 983,040 B default this round;
//                        SPLIT_K=6: 491,520 B; SPLIT_K=2: 163,840 B) for
//                       partial planes, all inside the 16-plane /
//                       1,310,720 B capacity for this shape (see
//                       workspace_split_k_capacity).
//
// The timed operator (`launch_w8a8_gemm`) performs no allocation,
// compilation, autotuning, packing, host/device synchronization, or
// default-stream launch: it only validates the exact shape guard and
// launches the split-K partial + combine kernels on the caller-provided
// stream.

#include <hip/hip_runtime.h>
#include <hip/hip_bfloat16.h>
#include <du_mma.h>

#include <cstdint>

namespace {

// gfx928 wavefront = 64 lanes; block sizes must be multiples of 64.
constexpr int kScalarBlockThreads = 128;
constexpr int kPackBlockThreads = 256;

// ---------------------------------------------------------------------------
// Generic scalar INT8 dot-product GEMM.
//
// One thread computes one output element out[m, n]. The full K loop runs in
// int32 (exact for every assigned K; max K=6144 -> |dot| <= 6144*127*127
// ~= 9.9e7 << 2^31), then the two fp32 scales are applied and the result is
// stored as bf16.
//
// Thread t maps to (row = t / n, col = t % n): adjacent lanes land on
// adjacent addresses in the fastest-changing N dimension, so the per-K-row
// B reads (b[k*n + col]) stay lane-coalesced and the A reads are broadcast
// within a row.
// ---------------------------------------------------------------------------
__global__ __launch_bounds__(kScalarBlockThreads) void
w8a8_scalar_gemm_kernel(const int8_t* __restrict__ a,       // [M, K]
                        const int8_t* __restrict__ b,       // [K, N]
                        const float* __restrict__ x_scale,  // [M]
                        const float* __restrict__ weight_scale,  // [N]
                        hip_bfloat16* __restrict__ out,     // [M, N]
                        int m,
                        int n,
                        int k) {
  const int64_t total = static_cast<int64_t>(m) * n;
  const int64_t t =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (t >= total) {
    return;
  }
  const int row = static_cast<int>(t / n);
  const int col = static_cast<int>(t - static_cast<int64_t>(row) * n);

  const int8_t* a_row = a + static_cast<int64_t>(row) * k;
  const int8_t* b_col = b + col;

  int32_t acc = 0;
  const int8_t* b_ptr = b_col;
  for (int kk = 0; kk < k; ++kk) {
    acc += static_cast<int32_t>(a_row[kk]) *
           static_cast<int32_t>(b_ptr[0]);
    b_ptr += n;
  }

  // Same arithmetic order as the harness reference
  // (dot.to(fp32) * x_scale * weight_scale.T): fp32 multiply left-to-right,
  // then round-to-nearest-even bf16.
  const float scaled = static_cast<float>(acc) * x_scale[row] *
                       weight_scale[col];
  out[t] = __float2bfloat16(scaled);
}

// ---------------------------------------------------------------------------
// Identity device-to-device weight packing (bootstrap).
// raw_weight [K, N] -> packed_weight [K, N], weight_scale [N] ->
// packed_weight_scale [N]. Later optimization rounds may replace this with a
// real layout transform and a matching GEMM interpretation.
// ---------------------------------------------------------------------------
__global__ __launch_bounds__(kPackBlockThreads) void
w8a8_pack_weight_identity_kernel(const int8_t* __restrict__ raw_weight,
                                 int8_t* __restrict__ packed_weight,
                                 int64_t count) {
  const int64_t t =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (t < count) {
    packed_weight[t] = raw_weight[t];
  }
}

__global__ __launch_bounds__(kPackBlockThreads) void
w8a8_pack_scale_identity_kernel(const float* __restrict__ raw_scale,
                                float* __restrict__ packed_scale,
                                int count) {
  const int t = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (t < count) {
    packed_scale[t] = raw_scale[t];
  }
}

// Shared scalar launch helper: works for any (m, n, k) satisfying the API
// contract, so it doubles as the generic fallback.
void launch_scalar_gemm(const int8_t* a,
                        const int8_t* b,
                        const float* x_scale,
                        const float* weight_scale,
                        void* out,
                        int m,
                        int n,
                        int k,
                        hipStream_t stream) {
  const int64_t total = static_cast<int64_t>(m) * n;
  const int64_t blocks =
      (total + kScalarBlockThreads - 1) / kScalarBlockThreads;
  hipLaunchKernelGGL(w8a8_scalar_gemm_kernel,
                     dim3(static_cast<unsigned>(blocks)),
                     dim3(kScalarBlockThreads), 0, stream, a, b, x_scale,
                     weight_scale, reinterpret_cast<hip_bfloat16*>(out), m,
                     n, k);
}

// ---------------------------------------------------------------------------
// DUMMA INT8 m16n16k32 split-K partial kernel (exact assigned shape M=16).
//
// Occupancy-probe family (iterations 1-12): one 64-lane wavefront owned one
// (16x16) partial tile over one uniform K-slice of the template SPLIT_K with
// direct global byte-load fragments; grid = (N/16)*SPLIT_K. Measured
// geometry ladder: S=1 80 blocks 206.9 us, S=3 240 blocks pathological
// 416.7 us, S=2 160 blocks 106.8 us (iter-3 champion), S=6 480 blocks
// 64.46 us (iter-10 champion), S=12 960 blocks 66.34 us. S=6/S=12 both move
// the same 491,520 L2 sectors at ~7.4-7.6 G sector/s, so pure split-K
// occupancy is closed; load width (iter 5) and LDS staging (iter 9) and
// register prefetch (iter 12) all regressed.
//
// Iteration 13 (n32 block, A-sharing): each wavefront now owns TWO adjacent
// n16 tiles (tiles 2*tile2 and 2*tile2+1) over the same K-slice. Both mmacs
// consume the SAME A fragment (A is loaded once per k32 step for both
// tiles), which halves A's L2 sector traffic (A was re-read once per n16
// tile, 80x total). B is two separate n16 fragments at n0 and n0+16. The
// partials layout is unchanged ([tile][s][16][16] with tile the n16-tile
// index, so the combine kernel is untouched) and every output element still
// accumulates the identical k-ascending int32 sequence, so correctness
// stays bit-exact. Grid = (N/32)*SPLIT_K: with SPLIT_K=12 -> 40*12 = 480
// one-wave zero-barrier blocks = exactly 4 blocks per CU (1 wave per SIMD,
// the iteration-10 champion's MLP level), 16 k32 steps x 2 mmacs per wave.
// ---------------------------------------------------------------------------
constexpr int kDummaTileM = 16;
constexpr int kDummaTileN = 16;
constexpr int kDummaTileK = 32;
constexpr int kDummaBlockThreads = 64;
// Iteration 13: adjacent n16 tiles processed per wave (shared A fragment).
constexpr int kDummaTilesPerBlock = 2;

template <int SPLIT_K>
__global__ __launch_bounds__(kDummaBlockThreads) void
w8a8_dumma_m16n16k32_splitk_partial_kernel(
    const int8_t* __restrict__ a,             // [16, K] row-major
    const int8_t* __restrict__ b,             // [K, N] row-major
    int32_t* __restrict__ partials,           // [N/16][SPLIT_K][16][16]
    int n,
    int k) {
  const int tile2 = static_cast<int>(blockIdx.x) / SPLIT_K;
  const int s = static_cast<int>(blockIdx.x) % SPLIT_K;
  const int n0 = tile2 * kDummaTilesPerBlock * kDummaTileN;

  // Uniform split; the exact shape (K=6144) divides evenly by every
  // SPLIT_K in {2, 6, 12} (3072 / 1024 / 512), each k32-aligned.
  const int k_chunk = k / SPLIT_K;
  const int k_begin = k_chunk * s;

  du::dumma::DUFragment<du::dumma::matrix_a, kDummaTileM, kDummaTileN,
                        kDummaTileK, signed char, du::dumma::row_major>
      a_frag;
  du::dumma::DUFragment<du::dumma::matrix_b, kDummaTileM, kDummaTileN,
                        kDummaTileK, signed char, du::dumma::row_major>
      b_frag0;
  du::dumma::DUFragment<du::dumma::matrix_b, kDummaTileM, kDummaTileN,
                        kDummaTileK, signed char, du::dumma::row_major>
      b_frag1;
  du::dumma::DUFragment<du::dumma::accumulator, kDummaTileM, kDummaTileN,
                        kDummaTileK, int>
      acc_frag0;
  du::dumma::DUFragment<du::dumma::accumulator, kDummaTileM, kDummaTileN,
                        kDummaTileK, int>
      acc_frag1;
  du::dumma::du_fill_fragment(acc_frag0, 0);
  du::dumma::du_fill_fragment(acc_frag1, 0);

  const int lane = static_cast<int>(threadIdx.x);
  for (int k0 = k_begin; k0 < k_begin + k_chunk; k0 += kDummaTileK) {
    // One A fragment feeds both n16 tiles (halves A's L2 sector traffic).
    // Iteration 15: du_load_matrix_sync (matrix_a row_major, ldm = k) maps
    // lane l to the eight CONTIGUOUS K elements a[row = l & 15][col =
    // ((l >> 4) << 3) .. +7], and du_mma_sync consumes the fragment as one
    // 64-bit little-endian value, so one 8-byte-aligned dwordx2 load
    // replaces A's 8 scalar global_load_ubyte + 2 count-limited s_waitcnt
    // vmcnt chains + 8 byte-pack VALU ops per k32 step with identical
    // register values (bit-identical int32 accumulation; 0 mismatches).
    // B stays scalar: matrix_b row_major bytes are strided by the K-major
    // row stride (n), not contiguous, so they cannot be vector-loaded from
    // the identity layout (that axis was falsified at iteration 5).
    const int a_row = lane & 15;
    const int a_col = (lane >> 4) << 3;
    const int8_t* a_ptr = static_cast<const int8_t*>(
        __builtin_assume_aligned(a + k0 + a_row * k + a_col, 8));
    *reinterpret_cast<int2*>(a_frag.x) =
        *reinterpret_cast<const int2*>(a_ptr);
    du::dumma::du_load_matrix_sync(b_frag0, b + k0 * n + n0, n);
    du::dumma::du_load_matrix_sync(b_frag1, b + k0 * n + n0 + kDummaTileN,
                                   n);
    du::dumma::du_mma_sync(acc_frag0, a_frag, b_frag0, acc_frag0);
    du::dumma::du_mma_sync(acc_frag1, a_frag, b_frag1, acc_frag1);
  }

  // Single wave per block: no barrier needed before the direct global
  // stores. Iteration 14: each lane's four int32 accumulator elements are
  // stored CONTIGUOUSLY (lane-major plane: lane l occupies plane[l*4 .. l*4+3],
  // one dwordx4 store per lane per plane instead of four strided dword
  // stores), so the combine kernel reads every plane with one dwordx4 load
  // per lane per split instead of four strided scalar loads. The raw
  // fragment values are unchanged: element i of lane l is the partial of
  // (row = l & 15, col = (l >> 4) + 4*i) (gfx928 m16n16k32 int32 accumulator
  // layout, verified against du_store_matrix_sync and the accepted pre-edit
  // ISA; see the combine kernel for the matching interpretation). The plane
  // layout stays [tile][s][16][16] with tile the n16-tile index (2*tile2 and
  // 2*tile2+1); only the intra-plane element order is permuted, consistently
  // in both kernels, so every int32 partial value and the ascending-sum
  // combine order are bit-identical.
  int32_t* plane0 =
      partials +
      ((tile2 * kDummaTilesPerBlock) * SPLIT_K + s) *
          (kDummaTileM * kDummaTileN);
  *reinterpret_cast<int4*>(plane0 + lane * 4) =
      make_int4(acc_frag0.x[0], acc_frag0.x[1], acc_frag0.x[2],
                acc_frag0.x[3]);
  int32_t* plane1 =
      partials +
      ((tile2 * kDummaTilesPerBlock + 1) * SPLIT_K + s) *
          (kDummaTileM * kDummaTileN);
  *reinterpret_cast<int4*>(plane1 + lane * 4) =
      make_int4(acc_frag1.x[0], acc_frag1.x[1], acc_frag1.x[2],
                acc_frag1.x[3]);
}

// ---------------------------------------------------------------------------
// Split-K combine kernel (same exact shape). One 64-lane wavefront per 16x16
// output tile: 4 elements per lane, each summed ascending over the SPLIT_K
// planes (exact int32, split-k-ascending + k-ascending inside each split is
// bit-identical to the reference int64 dot), then x_scale[m] *
// packed_weight_scale[n] and bf16 store. Part of the timed operator wall.
// Iteration 14: reads the lane-major plane permutation written by the
// partial kernel (lane l owns plane[l*4 .. l*4+3]) with one dwordx4 load per
// lane per split instead of four strided dword loads.
// ---------------------------------------------------------------------------
template <int SPLIT_K>
__global__ __launch_bounds__(kDummaBlockThreads) void
w8a8_dumma_m16_splitk_combine_kernel(
    const int32_t* __restrict__ partials,     // [N/16][SPLIT_K][16][16]
    const float* __restrict__ x_scale,        // [16]
    const float* __restrict__ weight_scale,   // [N]
    hip_bfloat16* __restrict__ out,           // [16, N] row-major
    int n) {
  const int tile = static_cast<int>(blockIdx.x);
  const int n0 = tile * kDummaTileN;
  const int lane = static_cast<int>(threadIdx.x);
  // Iteration 14 lane-major plane layout: lane l's four elements are stored
  // contiguously (one dwordx4 per lane per split), so the combine reads
  // plane[l*4 .. l*4+3] with one dwordx4 load per lane per split. Repair
  // (iteration 14 round 1): the gfx928 m16n16k32 int32 accumulator fragment
  // layout (verified against du_store_matrix_sync and the accepted pre-edit
  // ISA) is row = lane & 15, col = (lane >> 4) + 4*i for element i, i.e.
  // lane l's i-th element is the partial of (row = l & 15, col = (l >> 4) +
  // 4*i). Element i of lane l is stored at plane[l*4 + i], so acc0..acc3
  // (v.x..v.w) map to row = lane % 16 and columns (lane / 16) + {0, 4, 8,
  // 12}. The split-ascending int32 sum per output element is unchanged
  // (bit-identical values; only the (lane, i) -> (row, col) interpretation
  // was wrong and is fixed here).
  const int row = lane % kDummaTileN;     // 0..15: fragment row = l & 15
  const int col_lo = lane / kDummaTileN;  // 0..3:  fragment col base = l >> 4
  const int plane_base = tile * SPLIT_K * (kDummaTileM * kDummaTileN);

  int32_t acc0 = 0, acc1 = 0, acc2 = 0, acc3 = 0;
#pragma unroll
  for (int s = 0; s < SPLIT_K; ++s) {
    const int4 v = *reinterpret_cast<const int4*>(
        partials + plane_base + s * (kDummaTileM * kDummaTileN) +
        lane * 4);
    acc0 += v.x;
    acc1 += v.y;
    acc2 += v.z;
    acc3 += v.w;
  }
  const float s0 = static_cast<float>(acc0) * x_scale[row] *
                   weight_scale[n0 + col_lo];
  out[row * n + n0 + col_lo] = __float2bfloat16(s0);
  const float s1 = static_cast<float>(acc1) * x_scale[row] *
                   weight_scale[n0 + col_lo + 4];
  out[row * n + n0 + col_lo + 4] = __float2bfloat16(s1);
  const float s2 = static_cast<float>(acc2) * x_scale[row] *
                   weight_scale[n0 + col_lo + 8];
  out[row * n + n0 + col_lo + 8] = __float2bfloat16(s2);
  const float s3 = static_cast<float>(acc3) * x_scale[row] *
                   weight_scale[n0 + col_lo + 12];
  out[row * n + n0 + col_lo + 12] = __float2bfloat16(s3);
}

// Split-K DUMMA launch for the exact M=16 shape with N % 16 == 0,
// K % 32 == 0 and K % SPLIT_K == 0. Partial + combine both run inside the
// timed Graph replay on the caller stream. The partial kernel processes
// kDummaTilesPerBlock adjacent n16 tiles per wave (grid = N/(16*2) *
// SPLIT_K); the combine still launches one block per n16 tile over the
// unchanged [tile][s][16][16] plane layout.
template <int SPLIT_K>
void launch_dumma_m16_splitk(const int8_t* a,
                             const int8_t* b,
                             const float* x_scale,
                             const float* weight_scale,
                             void* out,
                             void* workspace,
                             int64_t workspace_bytes,
                             int n,
                             int k,
                             hipStream_t stream) {
  const int tiles = n / kDummaTileN;
  const int64_t required_bytes =
      static_cast<int64_t>(SPLIT_K) * tiles * kDummaTileM * kDummaTileN *
      static_cast<int64_t>(sizeof(int32_t));
  if (workspace_bytes < required_bytes) {
    // Defensive: workspace capacity for this shape is 16 planes, so this
    // never triggers for SPLIT_K in {2, 6, 12}; keep the scalar path as the
    // generic fallback anyway.
    launch_scalar_gemm(a, b, x_scale, weight_scale, out, kDummaTileM, n, k,
                       stream);
    return;
  }
  int32_t* partials = reinterpret_cast<int32_t*>(workspace);
  const int partial_blocks =
      tiles / kDummaTilesPerBlock * SPLIT_K;
  hipLaunchKernelGGL(w8a8_dumma_m16n16k32_splitk_partial_kernel<SPLIT_K>,
                     dim3(static_cast<unsigned>(partial_blocks)),
                     dim3(kDummaBlockThreads), 0, stream, a, b, partials, n,
                     k);
  hipLaunchKernelGGL(w8a8_dumma_m16_splitk_combine_kernel<SPLIT_K>,
                     dim3(static_cast<unsigned>(tiles)),
                     dim3(kDummaBlockThreads), 0, stream, partials, x_scale,
                     weight_scale, reinterpret_cast<hip_bfloat16*>(out), n);
}

}  // namespace

// ---------------------------------------------------------------------------
// Host launch symbols consumed by csrc/bindings.cpp (TORCH_LIBRARY zth_w8a8).
// ---------------------------------------------------------------------------
extern "C" void launch_w8a8_gemm(const int8_t* a,
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
  // Exact assigned shape: minimax_tp8_qkv_proj_m16 (M=16, N=1280, K=6144).
  // Iteration 13: default SPLIT_K=12 on the n32 (2 tiles/wave, shared A)
  // partial kernel -> 40*12 = 480 one-wave zero-barrier blocks = exactly
  // 4 blocks per CU on all 120 CUs (1 wave per SIMD, the iteration-10
  // champion's MLP level), 16 k32 steps x 2 mmacs per wave, 12 int32
  // partial planes (983,040 B) inside the 16-plane / 1,310,720 B capacity.
  // Iteration 14: plane element order is lane-major (both kernels changed
  // together, values/order bit-identical), so the combine's per-lane
  // serialized load chain drops from 48 strided dword loads to 12 dwordx4
  // loads and the partial's stores drop from 8 to 2 dwordx4 per block.
  // Iteration 15: the partial kernel's A fragment load is vectorized to one
  // 8-byte dwordx2 per lane per k32 step (the 8 bytes per lane are
  // contiguous K elements in the identity row-major A layout; du_mma_sync
  // consumes the fragment as one 64-bit value), cutting A's 8 scalar
  // global_load_ubyte + 2 wait chains + 8 byte-pack VALU ops per step with
  // bit-identical values; B stays scalar (strided by n). vmem_read drops
  // 184,320 -> 130,560 at unchanged grid (480) / planes (12) / sectors.
  // SPLIT_K=6 (240 blocks) and SPLIT_K=2 (80 blocks) stay instantiated as
  // measured alternatives; flip kDefaultSplitK to switch the measured
  // geometry without any other source change. The guard is exact, so paired
  // M=2 shapes with the same (N, K) keep taking the generic scalar fallback
  // below.
  constexpr int kDefaultSplitK = 12;
  if (m == 16 && n == 1280 && k == 6144) {
    switch (kDefaultSplitK) {
      case 12:
        launch_dumma_m16_splitk<12>(a, b, x_scale, weight_scale, out,
                                    workspace, workspace_bytes, n, k, stream);
        break;
      case 6:
        launch_dumma_m16_splitk<6>(a, b, x_scale, weight_scale, out,
                                   workspace, workspace_bytes, n, k, stream);
        break;
      case 2:
        launch_dumma_m16_splitk<2>(a, b, x_scale, weight_scale, out,
                                   workspace, workspace_bytes, n, k, stream);
        break;
      default:
        launch_scalar_gemm(a, b, x_scale, weight_scale, out, kDummaTileM, n,
                           k, stream);
        break;
    }
    return;
  }

  // Generic scalar fallback for every unmatched (m, n, k).
  launch_scalar_gemm(a, b, x_scale, weight_scale, out, m, n, k, stream);
}

extern "C" void launch_pack_w8a8_weight(const int8_t* raw_weight,
                                        const float* weight_scale,
                                        int8_t* packed_weight,
                                        float* packed_weight_scale,
                                        int k,
                                        int n,
                                        hipStream_t stream) {
  // Identity device-to-device copy, valid for every (K, N). Out of the timed
  // region (pack_weight op), so multiple launches are fine here.
  const int64_t weight_count = static_cast<int64_t>(k) * n;
  const int64_t weight_blocks =
      (weight_count + kPackBlockThreads - 1) / kPackBlockThreads;
  hipLaunchKernelGGL(w8a8_pack_weight_identity_kernel,
                     dim3(static_cast<unsigned>(weight_blocks)),
                     dim3(kPackBlockThreads), 0, stream, raw_weight,
                     packed_weight, weight_count);

  const int64_t scale_blocks =
      (static_cast<int64_t>(n) + kPackBlockThreads - 1) / kPackBlockThreads;
  hipLaunchKernelGGL(w8a8_pack_scale_identity_kernel,
                     dim3(static_cast<unsigned>(scale_blocks)),
                     dim3(kPackBlockThreads), 0, stream, weight_scale,
                     packed_weight_scale, n);
}
// @@end
