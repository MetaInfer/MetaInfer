// @@variant shape=minimax_tp8_o_proj_m16 commit=35384b7737eacb9fb8946dc1c5f984e8bf364aae added=2026-08-28
//   median_us=11.37 p90_us=11.39
//   source=minimax-dsh-tp8-m16-1-78402260
// W8A8 INT8 GEMM - gfx928 (K500SM_AI) - worker_2 bootstrap (iteration 1).
//
// Operator contract (logical):
//   out[m, n] = bf16( int32_dot(x_q[m, :], packed_weight[:, n])
//                     * x_scale[m, 0] * packed_weight_scale[n, 0] )
//
// Bootstrap strategy (correctness first, no DUMMA/split-K machinery):
//   * one thread computes exactly one output element;
//   * 2D grid: blockIdx.y = row (M), blockIdx.x = 256-wide N tile;
//     256 threads/block = 4 wavefronts of 64;
//   * complete K loop accumulated exactly in int32 (|acc| <= 127*127*K fits
//     trivially for every supported K);
//   * float32 scaling in the exact reference order
//     (float(dot) * x_scale[row]) * weight_scale[col], then bf16 RN store;
//   * pack_weight is an identity device-to-device copy (raw layout), so the
//     GEMM reads packed_weight as plain [K, N] row-major int8.
//
// This scalar kernel is the generic fallback for every (m, n, k), including
// the paired M=2 API shapes. Later optimization rounds may add exact-shape
// guards BEFORE this fallback inside launch_w8a8_gemm, and may change the
// pack_weight HIP implementation together with the matching GEMM
// interpretation. The host launcher signatures below are stable and consumed
// by csrc/bindings.cpp.
//
// Iteration 5 (HIP-only packed-weight round): the exact (M=16, N=6144,
// K=1024) shape now runs the packed-B DUMMA kernel below (packed
// [n_tile][k_step][lane][8] B fragment-slot layout, dwordx2 register
// transport, A kept staged in LDS), and launch_pack_w8a8_weight permutes
// that shape's weight into the packed layout (one-time, untimed, outside
// Graph capture). The scalar fallback decodes the packed layout for the
// exact (n, k) == (6144, 1024) pair so the paired M=2 API shape stays exact;
// every other (m, n, k) keeps the identity pack and direct row-major reads.
//
// Iteration 6 (staging-family round): the staging decision stays A-only LDS
// (A staged once per block, B direct register transport -- B is the cold
// once-read stream with zero cross-block reuse, so an LDS hop for B would add
// 6 MiB of LDS writes + reads without removing any global request). The
// focused change is the B transport width: the packed layout becomes
// [n_tile][k_step_pair][lane][16] -- each lane's 16 B slot holds the
// fragments of TWO consecutive k_steps (2 x 8 B, byte-for-byte the
// du_mma.hpp matrix_b row_major lane order) -- so each lane fetches two
// k-steps' whole B fragments with ONE aligned 16-byte global_load_dwordx4
// (1 KiB contiguous per wave per instruction = 8 sectors) instead of two
// dwordx2. Same-size permutation (6 MiB), same k-ascending int32
// accumulation, so output stays bit-identical to iterations 1/4/5.
//
// Iteration 7 (pipeline round: single vs double buffering): the exact ISA of
// the iteration-6 kernel shows the depth-1 "prefetch" is shallower than its
// source comment claims -- per K=128 stage the 2 B global_load_dwordx4 are
// issued before the 4-MMAC burst but the single s_waitcnt vmcnt(0) lands
// immediately AFTER the burst (before the 16-mov fragment rotate), so each B
// HBM round trip has only ~1 MMAC-burst (~120-200 cycles) of in-flight time
// against a ~1,700+ cycle contended round trip; 17.2 us = 8 stages x ~3,100
// cycles, of which ~3,000 is the exposed B wait. All three pipeline gates
// hold (K=1024 >= 1024; L2 36.6% < 70%; doubled LDS 2 x 16,640 = 33,280 B <
// 48 KiB). The round resolves as: register buffering depth 1 (single) -> 2
// (double) via a lagged 3-set rotation (burst stage s from set 0, rotate
// sets 0<-1<-2, refill set 2 with stage s+3 AFTER the rotate, so the
// compiler's wait for the refill loads lands at the NEXT iteration's rotate
// -- one full loop iteration of in-flight time per B dwordx4). LDS-ring
// double buffering is rejected by evidence (adds 2 ds_write + 2 ds_read + a
// self-barrier per stage while removing zero B round trips). Barriers per
// k-step stay 0 (the single-wave __syncthreads compiles to s_waitcnt only).
// Same fragments, k-ascending int32 order -> output bit-identical to
// iterations 1/4/5/6.
//
// Iteration 8 (HIP-only occupancy round, mandate: tune ONE occupancy limiter
// with PMC evidence -- waves per block / VGPR live range / LDS footprint /
// spills): the exact ISA + PMC of the accepted iteration-7 kernel show the
// limiter is the WAVE COUNT, not registers or LDS. arch_vgpr = 88 allows
// 2 waves/SIMD (256/88) and LDS 16,640 B/block leaves the measured 3.2
// blocks/CU resident, but grid = 384 one-wave blocks on 120 CUs = 3.2
// waves/CU = 0.8 waves/SIMD ACHIEVED: no SIMD in the whole GPU runs a second
// wave, so every B HBM round trip (~3,000 cycles per K=128 stage, the
// iteration-7 measured limiter) is fully exposed with nothing to switch to.
// The round resolves the trusted occupancy-probe split set {2,3,4,5} as an
// IN-BLOCK split-K = 2 ("waves per block" is the tuned limiter): workgroup
// 64 -> 128 (two 64-lane wavefronts), grid stays 384, each wave computes the
// SAME 16x16 tile over a disjoint K half (512 K = 4 stages, stage-aligned),
// so waves/CU doubles to ~6.4 (1.6 waves/SIMD) and the two co-resident waves
// of a block hide each other's B waits. A/B HBM bytes are UNCHANGED (each
// byte still read exactly once -- disjoint K slices, no repeated reads), the
// packed [n_tile][k_step_pair][lane][16] layout is untouched, and the int32
// partials are combined in LDS (1,024 B) with one real 2-wave s_barrier:
// integer addition is associative, so the int32 total (and therefore the bf16
// output) stays BIT-IDENTICAL to iterations 1/4/5/6/7. LDS per block grows
// 16,640 -> 17,664 B (fits the measured >= 4-block/CU capacity with margin).
// The single-wave iteration-7 kernel remains defined below as the recorded
// accepted best (fallback reference only).
//
// Falsifiable prediction (iteration 8): if the B round-trip wait was the
// limiter and 2 co-resident waves per block hide it (1.6 waves/SIMD
// achieved), the per-wave K halves and median should fall to ~9-12 us with
// P90 tracking median (1.4-1.9x vs the 16.847 us official best); if the
// hardware cannot co-schedule the second wave on the same SIMD slot, median
// stays ~16-17 us within noise and the next round must attack the 4-way LDS
// bank conflicts (49,152 events) or the 4-MMAC dependency chain instead.

#include <hip/hip_runtime.h>
#include <hip/hip_bfloat16.h>
#include <du_mma.h>

#include <cstdint>

namespace {

// 256 threads = 4 gfx928 wavefronts of 64 lanes.
constexpr int kScalarBlockThreads = 256;
constexpr int kCopyBlockThreads = 256;

// ---------------------------------------------------------------------------
// Exact-shape constants for the packed-B DUMMA specialization (iteration 5).
// kTargetN/kTargetK identify the minimax_tp8_o_proj_m16 weight (K=1024,
// N=6144) that gets the non-identity pack; the paired M=2 API shape shares
// this (N, K) weight and decodes the packed layout in the scalar fallback.
// ---------------------------------------------------------------------------
constexpr int kTargetN = 6144;
constexpr int kTargetK = 1024;
constexpr int kDummaBlockThreads = 64;   // one gfx928 wavefront per block

// gfx928 DUMMA INT8 tile geometry (m16n16k32).
constexpr int kDummaTileM = 16;
constexpr int kDummaTileN = 16;
constexpr int kDummaTileK = 32;

// Iteration-5/6 packed-B layout constants (depend on the tile constants
// above). Bytes of one (n_tile, k_step) B fragment: 16 cols x 32 k rows =
// 512 B, stored as 64 lanes x 8 B. Iteration 6 pairs two consecutive
// k_steps per lane: [n_tile][k_step_pair][lane][16] with 16 B per lane (=
// the 8-B fragment slots of k_step 2p and 2p+1), so one aligned 16-byte
// global_load_dwordx4 per lane per pair fetches two whole fragments
// (1 KiB contiguous per wave per instruction = 8 sectors).
constexpr int kPackedFragBytes = kDummaTileN * kDummaTileK;  // 512
constexpr int kPackedPairBytes = 2 * kPackedFragBytes;  // 1024: one pair slot
constexpr int kPackedLaneBytes = kPackedPairBytes / kDummaBlockThreads;  // 16
// K=128 stage = 4 DUMMA steps = 2 k_step pairs: 2 B global dwordx4 (2 KiB)
// + 4 A ds_read2 from LDS per stage refill, 4 v_mmac per stage. Iteration 7:
// the loop keeps THREE stage register sets (current / next / next-next); one
// refill (2 KiB of B) is in flight per wave in steady state with the prologue
// issuing 3 refills (6 KiB) back-to-back.
constexpr int kStageK = 128;
constexpr int kStageSteps = kStageK / kDummaTileK;  // 4

// Iteration 8 (occupancy round): in-block split-K = 2. Each block runs TWO
// 64-lane wavefronts (workgroup 128) that split the K=1024 range in half
// (512 K = 4 stages per wave); int32 partials are combined in LDS. The pack
// layout and kDummaBlockThreads (= 64 lanes per wavefront) are unchanged.
constexpr int kDummaSplitK = 2;  // in-block K split (waves per block)
constexpr int kDummaBlockThreads2 = kDummaBlockThreads * kDummaSplitK;  // 128
constexpr int kSplitWaveK = kTargetK / kDummaSplitK;  // 512
constexpr int kSplitStages = kSplitWaveK / kStageK;   // 4

// ---------------------------------------------------------------------------
// Packed-B decode for the generic scalar fallback: returns logical
// weight[kk, col] from the packed [n_tile][k_step_pair][lane][16] layout
// produced by w8a8_pack_o_proj_m16_kernel (see the pack kernel for the exact
// permutation). Only consulted when (n, k) == (kTargetN, kTargetK), i.e. the
// paired M=2 API shape shares the packed weight buffer with the M=16 path.
// ---------------------------------------------------------------------------
__device__ __forceinline__ int8_t
packed_b_element(const int8_t* __restrict__ packed, int col, int kk) {
  const int n_tile = col >> 4;
  const int nn = col & 15;
  const int k_step = kk >> 5;
  const int kk8 = kk & 31;
  const int lane = (kk8 >> 3) * 16 + nn;
  const int i = kk8 & 7;
  const int pair = k_step >> 1;
  const int half = k_step & 1;  // 0: low 8 B of the lane slot, 1: high 8 B
  const int k_pairs = (kTargetK / kDummaTileK) >> 1;  // 16
  return packed[((static_cast<int64_t>(n_tile) * k_pairs + pair) *
                     kPackedPairBytes +
                 static_cast<int64_t>(lane) * kPackedLaneBytes + half * 8 +
                 i)];
}

// ---------------------------------------------------------------------------
// Scalar INT8 GEMM kernel (generic fallback for every (m, n, k)).
//
// One thread per output element. Adjacent lanes map to adjacent N columns
// (fastest-changing dimension), giving coalesced B and out accesses. For the
// exact (n, k) == (6144, 1024) pair the weight tensor is the packed
// [n_tile][k_step_pair][lane][16] layout (o_proj M=16 pack), decoded
// elementwise.
// ---------------------------------------------------------------------------
__global__ __launch_bounds__(kScalarBlockThreads) void w8a8_scalar_gemm_kernel(
    const int8_t* __restrict__ a,       // [M, K] row-major int8
    const int8_t* __restrict__ b,       // [K, N] (packed for 6144x1024)
    const float* __restrict__ x_scale,  // [M, 1]
    const float* __restrict__ weight_scale,  // [N, 1]
    hip_bfloat16* __restrict__ out,     // [M, N] bf16
    int n,
    int k) {
  const int row = static_cast<int>(blockIdx.y);
  const int col = static_cast<int>(blockIdx.x) * kScalarBlockThreads +
                  static_cast<int>(threadIdx.x);
  if (col >= n) {
    return;
  }

  const int8_t* __restrict__ a_row = a + static_cast<int64_t>(row) * k;
  const bool packed_b = (n == kTargetN && k == kTargetK);
  int32_t acc = 0;
  for (int kk = 0; kk < k; ++kk) {
    const int8_t b_val = packed_b ? packed_b_element(b, col, kk)
                                  : b[static_cast<int64_t>(kk) * n + col];
    acc += static_cast<int32_t>(a_row[kk]) * static_cast<int32_t>(b_val);
  }

  // Exact reference order: (float(dot) * x_scale[row]) * weight_scale[col].
  // For every supported K the int32 dot is exact and |dot| < 2^24, so the
  // float32 conversion is exact and both multiplications round identically
  // to the trusted int64 reference (0 mismatch expected).
  const float scaled =
      static_cast<float>(acc) * x_scale[row] * weight_scale[col];
  out[static_cast<int64_t>(row) * n + col] = __float2bfloat16(scaled);
}

// ---------------------------------------------------------------------------
// M=16 INT8 DUMMA m16n16k32 one-wave kernel (accepted best, iteration 1).
//
// One 64-lane wavefront per block computes one 16x16 output tile with
// explicit int32 accumulation:
//   * du_fill_fragment(acc, 0), then k/32 du_mma_sync steps;
//   * A/B fragments loaded directly from the row-major global tensors
//     (lda = k, ldb = n); identity packed weight = raw [K, N] layout;
//   * no LDS, no __syncthreads (single wave, no cross-wave barrier);
//   * direct accumulator-fragment epilogue using the verified gfx928 INT8
//     m16n16k32 ownership: row = lane & 15, col_mod4 = lane >> 4,
//     acc.x[i] maps to column col_mod4 + 4*i;
//   * exact reference scaling order float(acc) * x_scale[row] *
//     weight_scale[col], bf16 RN store.
//
// Superseded for the exact (16,6144,1024) shape by the A-only-staged kernel
// below (iteration 4); kept unchanged as the recorded accepted best.
// ---------------------------------------------------------------------------
__global__ __launch_bounds__(64) void w8a8_dumma_m16n16k32_onewave_kernel(
    const int8_t* __restrict__ a,           // [M, K] row-major int8
    const int8_t* __restrict__ b,           // [K, N] row-major int8 (identity)
    const float* __restrict__ x_scale,      // [M, 1]
    const float* __restrict__ weight_scale, // [N, 1]
    hip_bfloat16* __restrict__ out,         // [M, N] bf16
    int n,
    int k) {
  const int lane = static_cast<int>(threadIdx.x);
  const int n0 = static_cast<int>(blockIdx.x) * kDummaTileN;

  du::dumma::DUFragment<du::dumma::matrix_a, kDummaTileM, kDummaTileN,
                        kDummaTileK, signed char, du::dumma::row_major>
      a_frag;
  du::dumma::DUFragment<du::dumma::matrix_b, kDummaTileM, kDummaTileN,
                        kDummaTileK, signed char, du::dumma::row_major>
      b_frag;
  du::dumma::DUFragment<du::dumma::accumulator, kDummaTileM, kDummaTileN,
                        kDummaTileK, int> acc_frag;
  du::dumma::du_fill_fragment(acc_frag, 0);

  for (int k0 = 0; k0 < k; k0 += kDummaTileK) {
    du::dumma::du_load_matrix_sync(a_frag, a + k0, k);
    du::dumma::du_load_matrix_sync(b_frag, b + static_cast<int64_t>(k0) * n + n0,
                                   n);
    du::dumma::du_mma_sync(acc_frag, a_frag, b_frag, acc_frag);
  }

  const int row = lane & 15;
  const int col_mod4 = lane >> 4;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const int out_col = n0 + col_mod4 + 4 * i;
    const float scaled = static_cast<float>(acc_frag.x[i]) * x_scale[row] *
                         weight_scale[out_col];
    out[static_cast<int64_t>(row) * n + out_col] =
        __float2bfloat16(scaled);
  }
}

// ---------------------------------------------------------------------------
// M=16 INT8 DUMMA m16n16k32 A-only-staged kernel (iteration 4).
//
// Identical geometry and math to the accepted iteration-1 one-wave kernel:
// grid = 384 blocks x 64 lanes (>= 120 CUs), one 16x16 output tile per wave,
// no split-K, direct accumulator-fragment epilogue, B loaded directly from
// the row-major identity-packed global tensor. The single pipeline change is
// A-only staging: the whole A tile (16 x 1024 = 16 KiB) is written ONCE per
// block into LDS by the prologue with 16 fully-coalesced dwordx4 loads per
// lane (64 lanes x 16 B = 1 KiB contiguous per load instruction), stored at
// a padded row stride of 1040 B (1024 + 16) for bank skew and 8-B alignment;
// the K loop then reads the A fragment from LDS via du_load_matrix_sync
// (same (row, k) -> byte mapping as the proven global path, stride 1040).
// Per k-step this removes 8 scattered global_load_ubyte (A), ~8 incremental
// s_waitcnt vmcnt waits, ~32 fragment-reassembly VALU and ~20 A-address VALU
// (all A addresses become loop-invariant LDS offsets), i.e. roughly half the
// loop-body instructions and half the wait chain, and it removes the L1/L2
// sector pressure of 8 loads that each touch 16 rows x 4 k-chunks. A staged
// bytes are reused 16x per MMA across the tile's 16 output columns and are
// L2-shared across all 384 blocks; B bytes remain the cold once-read stream
// (each B[k][n] read from global once, reused 16x inside the tensor core).
// Single wave -> the one __syncthreads is a self-barrier that only orders
// this wave's own LDS writes before its LDS reads (no cross-wave barrier).
// Exact int32 accumulation order is unchanged (single accumulator, k
// ascending), so the bf16 output is bit-identical to iteration 1.
//
// Superseded for the exact (16,6144,1024) shape by the packed-B kernel below
// (iteration 5); kept unchanged as the recorded accepted best.
// ---------------------------------------------------------------------------
constexpr int kAStageStride = 1040;  // K = 1024 + 16 pad: 8-B aligned + skew
// Iteration 16 (repair 1): the in-block split-K=2 kernel
// (w8a8_dumma_m16n16k32_packedb_sk2_kernel) stages each wave's K-HALF of A
// only (wave w consumes k in [w*512, (w+1)*512)) into its OWN half-size A LDS
// tile (s_a4[wave]): 16 rows x 33 int4 = 512 cols + 16 B pad = 528 B row
// stride (8-B aligned; 33 dwords mod 32 = 1 -> the 16 rows and 4 col groups
// still land on distinct LDS bank phases, same skew property as 1040). Each
// tile is 8,448 B; two tiles (one per wave) are 16,896 B/block, so A LDS per
// block is 16,896 B (vs the accepted 16,640-B single full tile) and total LDS
// 18,048 B -> still 3 blocks/CU. Repair note: the iteration-16 draft used ONE
// shared 8,448-B tile for both waves, which cannot hold both K-halves
// concurrently (its staging loop also covered only 8 of 16 rows per wave) ->
// wrong A data for 8 rows per wave; the repair gives each wave its own tile.
constexpr int kSk2AStageStride = 528;  // K-half = 512 + 16 pad

__global__ __launch_bounds__(64) void w8a8_dumma_m16n16k32_astage_kernel(
    const int8_t* __restrict__ a,           // [M, K] row-major int8
    const int8_t* __restrict__ b,           // [K, N] row-major int8 (identity)
    const float* __restrict__ x_scale,      // [M, 1]
    const float* __restrict__ weight_scale, // [N, 1]
    hip_bfloat16* __restrict__ out,         // [M, N] bf16
    int n,
    int k) {
  // 16,640 B LDS per block: 16 rows x 1040 B padded stride.
  __shared__ int4 s_a4[(kDummaTileM * kAStageStride) / 16];

  const int lane = static_cast<int>(threadIdx.x);
  const int n0 = static_cast<int>(blockIdx.x) * kDummaTileN;

  // Prologue: stage the full A tile (16,384 B = 1024 x int4 chunks). Lane l
  // owns chunk g = i*64 + l, so each of the 16 load instructions covers 1 KiB
  // of contiguous bytes (fully coalesced) and every 16-B chunk maps to its
  // padded LDS slot exactly once. Rows land 65 int4 (1040 B) apart, which is
  // 8-B aligned and spreads the 16 rows across distinct LDS bank phases.
  {
    const int4* __restrict__ a4 = reinterpret_cast<const int4*>(a);
#pragma unroll
    for (int i = 0; i < 16; ++i) {
      const int g = i * 64 + lane;  // 16-B chunk index in [0, 1024)
      s_a4[(g >> 6) * 65 + (g & 63)] = a4[g];
    }
  }
  // Single wave: self-barrier ordering this wave's own LDS writes before its
  // LDS reads; no cross-wave dependency, no grid sync.
  __syncthreads();

  du::dumma::DUFragment<du::dumma::matrix_a, kDummaTileM, kDummaTileN,
                        kDummaTileK, signed char, du::dumma::row_major>
      a_frag;
  du::dumma::DUFragment<du::dumma::matrix_b, kDummaTileM, kDummaTileN,
                        kDummaTileK, signed char, du::dumma::row_major>
      b_frag;
  du::dumma::DUFragment<du::dumma::accumulator, kDummaTileM, kDummaTileN,
                        kDummaTileK, int> acc_frag;
  du::dumma::du_fill_fragment(acc_frag, 0);

  const int8_t* __restrict__ s_a =
      reinterpret_cast<const int8_t*>(s_a4);  // padded row-major A in LDS
  for (int k0 = 0; k0 < k; k0 += kDummaTileK) {
    du::dumma::du_load_matrix_sync(a_frag, s_a + k0, kAStageStride);
    du::dumma::du_load_matrix_sync(b_frag, b + static_cast<int64_t>(k0) * n + n0,
                                   n);
    du::dumma::du_mma_sync(acc_frag, a_frag, b_frag, acc_frag);
  }

  const int row = lane & 15;
  const int col_mod4 = lane >> 4;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const int out_col = n0 + col_mod4 + 4 * i;
    const float scaled = static_cast<float>(acc_frag.x[i]) * x_scale[row] *
                         weight_scale[out_col];
    out[static_cast<int64_t>(row) * n + out_col] =
        __float2bfloat16(scaled);
  }
}

// ---------------------------------------------------------------------------
// M=16 INT8 DUMMA m16n16k32 packed-B kernel (iterations 5/6).
//
// HIP-only packed-weight round: the SAME accepted geometry as iteration 4
// (grid = 384 blocks x 64 lanes >= 120 CUs, one 16x16 output tile per wave,
// no split-K, A staged once into LDS at padded stride 1040, direct
// accumulator-fragment epilogue) but the B operand now comes from a packed
// weight layout instead of 8 scattered global ubyte loads per k-step.
//
// Iteration-6 packed layout [n_tile][k_step_pair][lane][16] (see
// w8a8_pack_o_proj_m16_kernel): the 16x32 B fragments of the two consecutive
// k_steps (2p, 2p+1) of a pair are stored as 64 lanes x 16 B with lane's 16 B
// = fragment bytes of k_step 2p (low 8 B) followed by k_step 2p+1 (high 8 B),
// each fragment byte-for-byte the du_mma.hpp matrix_b row_major loader order
// (lane's 8 B = logical B[k_step*32 + col + i, n_tile*16 + row] for i =
// 0..7, row = lane & 15, col = (lane >> 4) << 3), so one aligned 16-byte
// (dwordx4) load per lane per pair fills TWO consecutive b_frag slots
// exactly (1 KiB contiguous per wave per instruction = 8 sectors; 2 pairs =
// 2 KiB contiguous per stage). The A fragment is read from the staged LDS
// tile with the same u64 transport (matrix_a row_major: 8 contiguous bytes
// at row*1040 + col, identical to the ds_read2_b32 the iteration-4 library
// loader compiled to).
//
// K-loop transport (iteration 7, pipeline round): the loop is a lagged
// THREE-set register pipeline with 0 barriers per k-step. Prologue fills
// stages 0/1/2 back-to-back (3 cold-start DRAM latencies, 6 KiB in flight,
// overlapped with the A-staging prologue); each loop iteration bursts stage
// s from the current set, rotates current<-next<-next-next (consuming the
// stage loaded a FULL iteration earlier, so the compiler's s_waitcnt
// vmcnt(0)/lgkmcnt(0) lands after the burst that precedes the rotate), then
// refills the freed third set with stage s+3's six loads (2 B global
// dwordx4 + 4 A ds_read2 = 2 KiB) whose first read is the NEXT iteration's
// rotate. Every B dwordx4 therefore stays in flight for one whole loop
// iteration (~4x the in-flight time of iteration 6's compiled depth-1, where
// the wait landed right after the same-iteration burst), and B vmem
// instructions per replay drop from 6,912 (iteration 6) to 6,144 (384 x 16
// dwordx4 = 3 prologue + 5 in-loop refills). A staged bytes are
// unchanged (16 KiB per block, read once from HBM, 16x reused per MMA), B
// HBM bytes unchanged (6 MiB, every byte read exactly once). Single wave ->
// the one __syncthreads is a self-barrier (compiles to s_waitcnt, zero
// s_barrier instructions in the kernel). Exact int32 accumulation is
// unchanged (single accumulator, k ascending), so the bf16 output is
// bit-identical to iterations 1, 4, 5 and 6.
// ---------------------------------------------------------------------------
__device__ __forceinline__ void load_stage_fragments_packedb(
    const int8_t* __restrict__ s_a,      // staged A in LDS (wave-local K-half
                                         // stride for the sk2 kernel, full-K
                                         // stride 1040 for the one-wave kernel)
    const int8_t* __restrict__ bpacked,  // packed [n_tile][pair][lane][16]
    int lane,
    int a_off,
    int64_t b_base,
    int s,        // GLOBAL stage index (0..7): selects the wave's disjoint
                  // K-half for B (pairs 2s, 2s+1)
    int a_stage,  // A stage index within the wave's staged K-half (0..3 for
                  // the sk2 kernel; == s for the full-K one-wave kernel)
    uint64_t& a0, uint64_t& a1, uint64_t& a2, uint64_t& a3,
    uint64_t& b0, uint64_t& b1, uint64_t& b2, uint64_t& b3) {
  const int64_t k0 = static_cast<int64_t>(s) * kStageK;
  const int64_t ak0 = static_cast<int64_t>(a_stage) * kStageK;
  a0 = *reinterpret_cast<const uint64_t*>(s_a + a_off + ak0);
  a1 = *reinterpret_cast<const uint64_t*>(s_a + a_off + ak0 + kDummaTileK);
  a2 = *reinterpret_cast<const uint64_t*>(s_a + a_off + ak0 + 2 * kDummaTileK);
  a3 = *reinterpret_cast<const uint64_t*>(s_a + a_off + ak0 + 3 * kDummaTileK);
  // B: one aligned 16-B (dwordx4) cooperative load per k_step pair; the
  // pair index covers steps 4s..4s+3 as pairs (2s, 2s+1). q.x = the 8-B
  // fragment slot of the even step, q.y = the odd step (little-endian u64,
  // same byte order du_mma_sync consumes).
  const int64_t pair = (k0 / kDummaTileK) >> 1;
  const int64_t b_off = b_base + pair * kPackedPairBytes +
                        static_cast<int64_t>(lane) * kPackedLaneBytes;
  const ulonglong2 q0 = *reinterpret_cast<const ulonglong2*>(bpacked + b_off);
  const ulonglong2 q1 =
      *reinterpret_cast<const ulonglong2*>(bpacked + b_off + kPackedPairBytes);
  b0 = q0.x;  // k_step 4s
  b1 = q0.y;  // k_step 4s+1
  b2 = q1.x;  // k_step 4s+2
  b3 = q1.y;  // k_step 4s+3
}

__global__ __launch_bounds__(kDummaBlockThreads) void
w8a8_dumma_m16n16k32_packedb_kernel(
    const int8_t* __restrict__ a,           // [M, K] row-major int8
    const int8_t* __restrict__ b,           // packed [n_tile][pair][lane][16]
    const float* __restrict__ x_scale,      // [M, 1]
    const float* __restrict__ weight_scale, // [N, 1]
    hip_bfloat16* __restrict__ out,         // [M, N] bf16
    int n,
    int k) {
  // 16,640 B LDS per block: 16 rows x 1040 B padded stride (as iteration 4).
  __shared__ int4 s_a4[(kDummaTileM * kAStageStride) / 16];

  const int lane = static_cast<int>(threadIdx.x);
  const int n0 = static_cast<int>(blockIdx.x) * kDummaTileN;

  // Prologue: stage the full A tile (16,384 B = 1024 x int4 chunks). Lane l
  // owns chunk g = i*64 + l, so each of the 16 load instructions covers 1 KiB
  // of contiguous bytes (fully coalesced); rows land 65 int4 (1040 B) apart.
  {
    const int4* __restrict__ a4 = reinterpret_cast<const int4*>(a);
#pragma unroll
    for (int i = 0; i < 16; ++i) {
      const int g = i * 64 + lane;  // 16-B chunk index in [0, 1024)
      s_a4[(g >> 6) * 65 + (g & 63)] = a4[g];
    }
  }
  // Single wave: self-barrier ordering this wave's own LDS writes before its
  // LDS reads; no cross-wave dependency, no grid sync.
  __syncthreads();

  du::dumma::DUFragment<du::dumma::matrix_a, kDummaTileM, kDummaTileN,
                        kDummaTileK, signed char, du::dumma::row_major>
      a_frag;
  du::dumma::DUFragment<du::dumma::matrix_b, kDummaTileM, kDummaTileN,
                        kDummaTileK, signed char, du::dumma::row_major>
      b_frag;
  du::dumma::DUFragment<du::dumma::accumulator, kDummaTileM, kDummaTileN,
                        kDummaTileK, int> acc_frag;
  du::dumma::du_fill_fragment(acc_frag, 0);

  // Per-lane A ownership is loop-invariant: row = lane & 15,
  // col = (lane >> 4) << 3, x[i] = s_a[row*1040 + col + i] (du_mma.hpp
  // matrix_a row_major). Per-lane B slot offset is loop-invariant too:
  // b_base + pair*1024 + lane*16.
  const int8_t* __restrict__ s_a = reinterpret_cast<const int8_t*>(s_a4);
  const int a_row = lane & 15;
  const int a_col = (lane >> 4) << 3;
  const int a_off = a_row * kAStageStride + a_col;
  const int64_t b_base = static_cast<int64_t>(blockIdx.x) *
                         (k / kDummaTileK / 2) * kPackedPairBytes;

  // Iteration-7 lagged 3-set pipeline. Prologue: stages 0, 1 and 2 land
  // directly in registers (three cold-start DRAM latencies per block, issued
  // back-to-back = 6 KiB in flight, overlapped with the A-staging prologue).
  const int n_stages = k / kStageK;  // 8 for K = 1024
  uint64_t ca0, ca1, ca2, ca3, cb0, cb1, cb2, cb3;  // stage s
  uint64_t na0, na1, na2, na3, nb0, nb1, nb2, nb3;  // stage s+1
  uint64_t ma0, ma1, ma2, ma3, mb0, mb1, mb2, mb3;  // stage s+2
  load_stage_fragments_packedb(s_a, b, lane, a_off, b_base, 0, 0, ca0, ca1,
                               ca2, ca3, cb0, cb1, cb2, cb3);
  load_stage_fragments_packedb(s_a, b, lane, a_off, b_base, 1, 1, na0, na1,
                               na2, na3, nb0, nb1, nb2, nb3);
  load_stage_fragments_packedb(s_a, b, lane, a_off, b_base, 2, 2, ma0, ma1,
                               ma2, ma3, mb0, mb1, mb2, mb3);

  for (int s = 0; s < n_stages; ++s) {
    // Stage-s MMAC burst straight from registers, in the exact a_frag.x /
    // b_frag.x byte order du_mma_sync consumes (little-endian u64 slot =
    // bytes 0..7 of the fragment).
#pragma unroll
    for (int t = 0; t < kStageSteps; ++t) {
      const uint64_t av =
          (t == 0) ? ca0 : (t == 1) ? ca1 : (t == 2) ? ca2 : ca3;
      const uint64_t bv =
          (t == 0) ? cb0 : (t == 1) ? cb1 : (t == 2) ? cb2 : cb3;
      __builtin_memcpy(a_frag.x, &av, 8);
      __builtin_memcpy(b_frag.x, &bv, 8);
      du::dumma::du_mma_sync(acc_frag, a_frag, b_frag, acc_frag);
    }
    // Lagged rotation: stage s+1 -> current, stage s+2 -> next. The ma/mb
    // set (stage s+2) was loaded a FULL iteration earlier, so the compiler's
    // s_waitcnt vmcnt(0)/lgkmcnt(0) for it lands at THIS rotate -- after the
    // burst, one whole loop iteration of in-flight time per B dwordx4 (vs
    // ~1 MMAC burst in iteration 6's compiled depth-1).
    ca0 = na0;
    ca1 = na1;
    ca2 = na2;
    ca3 = na3;
    cb0 = nb0;
    cb1 = nb1;
    cb2 = nb2;
    cb3 = nb3;
    na0 = ma0;
    na1 = ma1;
    na2 = ma2;
    na3 = ma3;
    nb0 = mb0;
    nb1 = mb1;
    nb2 = mb2;
    nb3 = mb3;
    // Refill the just-freed third set with stage s+3's six loads (2 B global
    // dwordx4 + 4 A ds_read2 = 2 KiB in flight). The rotate above reads
    // ma/mb (WAR), so the compiler cannot hoist these loads above it; their
    // first read is the NEXT iteration's rotate, one full iteration after
    // issue. The guard keeps the tail stages from reading past the packed B
    // buffer (stage 8+ never loads; stale ma/mb at s=6/7 are never consumed).
    if (s + 3 < n_stages) {
      load_stage_fragments_packedb(s_a, b, lane, a_off, b_base, s + 3, s + 3,
                                   ma0, ma1, ma2, ma3, mb0, mb1, mb2, mb3);
    }
  }

  // Direct accumulator-fragment epilogue, unchanged from iterations 1 and 4
  // (gfx928 INT8 m16n16k32 ownership: row = lane & 15, col_mod4 = lane >> 4,
  // acc.x[i] maps to column col_mod4 + 4*i).
  const int row = lane & 15;
  const int col_mod4 = lane >> 4;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const int out_col = n0 + col_mod4 + 4 * i;
    const float scaled = static_cast<float>(acc_frag.x[i]) * x_scale[row] *
                         weight_scale[out_col];
    out[static_cast<int64_t>(row) * n + out_col] =
        __float2bfloat16(scaled);
  }
}

// ---------------------------------------------------------------------------
// M=16 INT8 DUMMA m16n16k32 packed-B IN-BLOCK split-K=2 kernel (iteration 8,
// occupancy round). Same packed [n_tile][k_step_pair][lane][16] layout, same
// 16x16 tile geometry, same iteration-7 lagged 3-set pipeline, same A-only
// LDS staging (16 x 1040 B) -- but the workgroup is now TWO 64-lane
// wavefronts (128 threads) that split K in half: wave w computes the same
// tile over k in [w*512, w*512+512) (4 K=128 stages each, disjoint B pairs
// 8w..8w+7 via the global stage index s = wave*4 + sw). Waves per block 1->2
// is the tuned occupancy limiter: grid stays 384 blocks, so waves/CU goes
// 3.2 -> ~6.4 (1.6 waves/SIMD vs the 88-VGPR 2-waves/SIMD capacity), giving
// the wave scheduler a second co-resident wavefront to switch to during each
// wave's exposed B HBM round trip. A/B HBM bytes are unchanged (disjoint K
// slices, every byte read exactly once -- no repeated reads for occupancy).
// Combine: wave 1 publishes its int32 accumulator (4 ints per lane = 1,024 B)
// to LDS, one real 2-wave s_barrier orders it, wave 0 adds it in int32 and
// runs the unchanged direct bf16 epilogue (wave 1 exits). Integer addition is
// associative, so the int32 total and the bf16 output stay bit-identical to
// iterations 1/4/5/6/7. LDS per block: 16,640 (A) + 1,024 (partials) =
// 17,664 B (fits the measured >= 4-block/CU capacity with margin). 2 real
// s_barrier per block (post-staging + pre-combine) -> 0.0625 per k-step.
//
// Iteration 13 (final conditional HIP-only round; control plane keeps
// plateau=false, raw_inline_asm_allowed=false, so ONE HIP-only consolidation
// change on this accepted kernel -- the file is restored to the accepted
// iteration-8 digest f8b708... after the iteration-12 barrier-removal variant
// regressed to 12.716 us). The accepted ISA's epilogue (after the pre-combine
// s_barrier) issues FIVE per-lane global_load_dword (1 x_scale + 4
// weight_scale) with FOUR serialized s_waitcnt vmcnt(1)/vmcnt(0) windows --
// each column's weight_scale load is issued only after the previous column's
// store, then waited: ~4 x L2 latency of serial global round trips on the
// block's drain tail. Iteration 13 prefetches the scales into LDS: wave 0
// stores its 16-row x_scale slice + this block's 16-column weight_scale slice
// (n0..n0+15, 128 B) right after the A-staging loop, before the staging
// barrier; the two scale loads issue at kernel top (waits resolve during the
// barrier/B-wait) and the epilogue reads conflict-free LDS broadcasts with
// zero global latency. Same float values, same scaling order, +128 B LDS
// (17,792 B/block; 3 blocks/CU = 53,376 B <= 64 KiB unchanged), no new
// barrier, VGPR unchanged (~80) -> bf16 output BIT-IDENTICAL.
//
// Iteration 16, repair 1 (final conditional HIP-only round; control plane
// keeps plateau=false, raw_inline_asm_allowed=false: recent valid candidates
// are +6.80% it13 accepted, -4.05% it14 balanced-epilogue REGRESSED
// (reverted), -5.63% it15 epilogue-LDS-hoist REGRESSED (reverted); the
// epilogue axis is closed, so this round makes ONE HIP-only consolidation
// change on the untouched structural axis). The round proposes wave-private
// A-side staging: each wave stages ONLY its own K-half into a half-size
// 528-B-stride LDS tile instead of the full 16,640-B tile staged by both
// waves. Repair 1 corrects the draft's staging: the single shared 8,448-B
// tile cannot hold both waves' K-halves concurrently, and the draft loop
// covered only 8 of 16 rows per wave (per-thread loads 8 -> 4 halves the
// staged bytes, not the tile) -> wrong A fragments for half the rows of each
// wave's partial. The repaired kernel gives EACH wave its own 8,448-B half
// tile (s_a4[wave][528], 16,896 B/block A LDS; total LDS 16,896 + 1,024 +
// 128 = 18,048 B -> 3 blocks/CU unchanged) and stages each wave's FULL
// 512-chunk K-half (8 dwordx4 per thread, same per-thread count and
// coalescing as the accepted cooperative staging; every A byte read exactly
// once). The two s_barrier per block are kept, A fragment values, per-wave K
// halves, int32 accumulation order and the scaling math are UNCHANGED ->
// bf16 output stays BIT-IDENTICAL to iterations 8/13. Grid 384 x workgroup
// 128 unchanged; A/B HBM bytes unchanged; packed
// [n_tile][k_step_pair][lane][16] layout untouched; shape guards and scalar
// fallback untouched.
// ---------------------------------------------------------------------------
__global__ __launch_bounds__(kDummaBlockThreads2) void
w8a8_dumma_m16n16k32_packedb_sk2_kernel(
    const int8_t* __restrict__ a,           // [M, K] row-major int8
    const int8_t* __restrict__ b,           // packed [n_tile][pair][lane][16]
    const float* __restrict__ x_scale,      // [M, 1]
    const float* __restrict__ weight_scale, // [N, 1]
    hip_bfloat16* __restrict__ out,         // [M, N] bf16
    int n,
    int k) {
  // 18,048 B LDS per block: TWO per-wave 16 x 528 B padded A-stride K-half
  // tiles (iteration 16 repair 1: each wave stages ONLY its own K-half into
  // its own 8,448-B tile, s_a4[0]/s_a4[1]) + 1,024 B int32 partial buffer
  // (16-B aligned int4 x 64 lanes, wave 1 -> wave 0) + 128 B epilogue scale
  // cache (iteration 13: wave 0's 16-row x_scale slice + this block's
  // 16-column weight_scale slice). A LDS per block is 16,896 B; total LDS
  // 18,048 B keeps co-residency at 3 blocks/CU (18,048 x 3 = 54,144 B <= 64
  // KiB; 4 x 18,048 = 72,192 B > 64 KiB), same as the accepted 17,792-B
  // layout. A single shared 8,448-B tile (iteration-16 draft) could not hold
  // both waves' K-halves concurrently, so the repair allocates one tile per
  // wave.
  __shared__ int4 s_a4[2][(kDummaTileM * kSk2AStageStride) / 16];
  __shared__ int4 s_partial4[kDummaBlockThreads];
  __shared__ float s_xscale[kDummaTileM];  // x_scale[0..15] (64 B)
  __shared__ float s_wscale[kDummaTileN];  // weight_scale[n0..n0+15] (64 B)

  const int tid = static_cast<int>(threadIdx.x);  // 0..127
  const int wave = tid >> 6;                      // 0..1: in-block K split
  const int lane = tid & 63;
  const int n0 = static_cast<int>(blockIdx.x) * kDummaTileN;

  // Prologue (iteration 16, repair 1): each wave stages ONLY its own K-half
  // into ITS OWN half-size tile s_a4[wave]. Wave w owns global chunk
  // g = r*64 + (w*32) + c32 for r in 0..15, c32 in 0..31 (k in
  // [w*512, (w+1)*512) -- all 512 chunks of the wave's half), mapped to its
  // tile's slot r*33+c32. Per-thread staging loads stay 8 dwordx4 (one full
  // K-half per wave; the draft's "8 -> 4" halved the staged bytes, not the
  // tile, and left 8 rows per wave unstaged); each instruction covers the
  // wave's 512-B segment of 2 rows (1,024 B contiguous, fully coalesced --
  // the two waves together cover the accepted loop's 2-KiB contiguous
  // segment) and every 16-B chunk maps to its padded LDS slot exactly once.
  // Rows land 33 int4 (528 B) apart (8-B aligned, 33 mod 32 = 1 -> the 16
  // rows and 4 col groups sit on distinct LDS bank phases, same skew class
  // as the 1040-B stride). Wave w later reads ONLY slots it wrote in its own
  // tile, so the staging barrier still orders each wave's own writes before
  // its own reads (barrier kept: the accepted ISA's waitcnt placement stays
  // unchanged).
  {
    const int4* __restrict__ a4 = reinterpret_cast<const int4*>(a);
#pragma unroll
    for (int i = 0; i < kSplitWaveK / 64; ++i) {  // 8 iterations
      const int flat = i * 64 + lane;  // 16-B chunk in [0, 512) of this half
      const int r = flat >> 5;         // M row 0..15 (32 chunks per row half)
      const int c32 = flat & 31;       // 16-B chunk within the row's K-half
      const int g = r * 64 + (wave << 5) + c32;  // global chunk in [0, 1024)
      s_a4[wave][r * 33 + c32] = a4[g];
    }
  }
  // Iteration 13 (epilogue scale prefetch): wave 0 caches its 16-row x_scale
  // slice and this block's 16-column weight_scale slice (n0..n0+15) in LDS
  // right after the A staging, before the staging barrier. The accepted ISA
  // shows the epilogue issuing FIVE per-lane global_load_dword (1 x_scale +
  // 4 weight_scale) AFTER the pre-combine s_barrier with FOUR serialized
  // s_waitcnt vmcnt(1)/vmcnt(0) windows (each column's weight_scale load is
  // issued only after the previous column's store, then waited) -- ~4 x L2
  // latency of serial global round trips on the block's drain tail. Anchored
  // by these LDS stores, the two scale loads issue here at kernel top (their
  // vmcnt waits resolve during the barrier/B-wait) and the epilogue reads
  // LDS broadcast slots (4-way broadcast, conflict-free) with zero global
  // latency. Same float values, same scaling order
  // (float(dot) * x_scale[row]) * weight_scale[col] -> bf16 output stays
  // BIT-IDENTICAL. Wave 0 both writes and reads these slots (same-wave LDS
  // ordering is the compiler's lgkmcnt waits; no new barrier).
  if (wave == 0) {
    if (lane < kDummaTileM) {
      s_xscale[lane] = x_scale[lane];
      s_wscale[lane] = weight_scale[n0 + lane];
    }
  }
  // Real 2-wave barrier: both waves' staging writes are visible before any
  // fragment read (first of two s_barrier per block).
  __syncthreads();

  du::dumma::DUFragment<du::dumma::matrix_a, kDummaTileM, kDummaTileN,
                        kDummaTileK, signed char, du::dumma::row_major>
      a_frag;
  du::dumma::DUFragment<du::dumma::matrix_b, kDummaTileM, kDummaTileN,
                        kDummaTileK, signed char, du::dumma::row_major>
      b_frag;
  du::dumma::DUFragment<du::dumma::accumulator, kDummaTileM, kDummaTileN,
                        kDummaTileK, int> acc_frag;
  du::dumma::du_fill_fragment(acc_frag, 0);

  const int8_t* __restrict__ s_a = reinterpret_cast<const int8_t*>(s_a4[wave]);
  const int a_row = lane & 15;
  const int a_col = (lane >> 4) << 3;
  const int a_off = a_row * kSk2AStageStride + a_col;
  const int64_t b_base = static_cast<int64_t>(blockIdx.x) *
                         (k / kDummaTileK / 2) * kPackedPairBytes;

  // Iteration-7 lagged 3-set pipeline, kSplitStages = 4 stages per wave. The
  // GLOBAL stage index s = wave*4 + sw picks the wave's disjoint K half:
  // B pairs s*2/s*2+1 via s, and the A slice is read from the wave's OWN
  // half-size LDS tile (s_a4[wave], 8,448 B) via the LOCAL stage index sw
  // (a_stage) -- each wave reads only slots it staged itself.
  const int n_stages = kSplitStages;  // 4
  uint64_t ca0, ca1, ca2, ca3, cb0, cb1, cb2, cb3;  // stage s
  uint64_t na0, na1, na2, na3, nb0, nb1, nb2, nb3;  // stage s+1
  uint64_t ma0, ma1, ma2, ma3, mb0, mb1, mb2, mb3;  // stage s+2
  const int s0 = wave * n_stages;  // global stage of the wave's first K-half
  load_stage_fragments_packedb(s_a, b, lane, a_off, b_base, s0 + 0, 0, ca0,
                               ca1, ca2, ca3, cb0, cb1, cb2, cb3);
  load_stage_fragments_packedb(s_a, b, lane, a_off, b_base, s0 + 1, 1, na0,
                               na1, na2, na3, nb0, nb1, nb2, nb3);
  load_stage_fragments_packedb(s_a, b, lane, a_off, b_base, s0 + 2, 2, ma0,
                               ma1, ma2, ma3, mb0, mb1, mb2, mb3);

  for (int sw = 0; sw < n_stages; ++sw) {
    const int s = wave * n_stages + sw;
    // Stage-s MMAC burst straight from registers, in the exact a_frag.x /
    // b_frag.x byte order du_mma_sync consumes (little-endian u64 slot =
    // bytes 0..7 of the fragment).
#pragma unroll
    for (int t = 0; t < kStageSteps; ++t) {
      const uint64_t av =
          (t == 0) ? ca0 : (t == 1) ? ca1 : (t == 2) ? ca2 : ca3;
      const uint64_t bv =
          (t == 0) ? cb0 : (t == 1) ? cb1 : (t == 2) ? cb2 : cb3;
      __builtin_memcpy(a_frag.x, &av, 8);
      __builtin_memcpy(b_frag.x, &bv, 8);
      du::dumma::du_mma_sync(acc_frag, a_frag, b_frag, acc_frag);
    }
    // Lagged rotation: stage s+1 -> current, stage s+2 -> next. The ma/mb
    // set (stage s+2) was loaded a FULL iteration earlier, so the compiler's
    // s_waitcnt vmcnt(0)/lgkmcnt(0) for it lands at THIS rotate -- one whole
    // loop iteration of in-flight time per B dwordx4 (unchanged from
    // iteration 7).
    ca0 = na0;
    ca1 = na1;
    ca2 = na2;
    ca3 = na3;
    cb0 = nb0;
    cb1 = nb1;
    cb2 = nb2;
    cb3 = nb3;
    na0 = ma0;
    na1 = ma1;
    na2 = ma2;
    na3 = ma3;
    nb0 = mb0;
    nb1 = mb1;
    nb2 = mb2;
    nb3 = mb3;
    // Refill the freed third set with stage s+3's six loads (2 B global
    // dwordx4 + 4 A ds_read2 = 2 KiB in flight). Only sw = 0 refills within
    // the wave's 4 stages (global stage wave*4 + 3); the guard keeps tail
    // stages from reading past the wave's packed-B half.
    if (sw + 3 < n_stages) {
      // a_stage = sw + 3 stays within the wave's own 4-stage K-half (the
      // guard bounds it to 3), so the A reads never leave the 8,448-B tile.
      load_stage_fragments_packedb(s_a, b, lane, a_off, b_base, s + 3, sw + 3,
                                   ma0, ma1, ma2, ma3, mb0, mb1, mb2, mb3);
    }
  }

  // In-block split-K combine. Wave 1 publishes its int32 accumulator (one
  // aligned 16-B int4 per lane); the real 2-wave barrier orders the publish
  // before wave 0's read; wave 0 adds in int32 (exact: integer addition is
  // associative, so the total is bit-identical to the single-pass k-ascending
  // order) and runs the unchanged direct epilogue; wave 1 exits.
  if (wave == 1) {
    s_partial4[lane] =
        make_int4(acc_frag.x[0], acc_frag.x[1], acc_frag.x[2], acc_frag.x[3]);
  }
  __syncthreads();  // second (and last) real s_barrier per block
  if (wave == 0) {
    const int4 p = s_partial4[lane];
    const int total[4] = {acc_frag.x[0] + p.x, acc_frag.x[1] + p.y,
                          acc_frag.x[2] + p.z, acc_frag.x[3] + p.w};
    const int row = lane & 15;
    const int col_mod4 = lane >> 4;
    // Iteration 13: scales come from the LDS cache staged at kernel top
    // (s_xscale / s_wscale) instead of five serialized global loads; the
    // scaling math and order are unchanged.
    const float xs = s_xscale[row];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int out_col = n0 + col_mod4 + 4 * i;
      const float scaled = static_cast<float>(total[i]) * xs *
                           s_wscale[col_mod4 + 4 * i];
      out[static_cast<int64_t>(row) * n + out_col] =
          __float2bfloat16(scaled);
    }
  }
}

// ---------------------------------------------------------------------------
// One-time pack for the exact (k, n) == (1024, 6144) o_proj weight: permute
// the logical [K, N] int8 layout into packed[n_tile][k_step_pair][lane][16]
// B fragment slots (iteration 6: two consecutive k_steps per lane, see the
// packed-B kernel comment for the exact du_mma.hpp matrix_b row_major lane
// mapping: row = lane & 0xf, col = (lane >> 4) << 3, x[i] = p[(col + i) *
// ldm + row]). One thread per (n_tile, k_step_pair, lane): 16 pairs x 384
// n_tiles x 64 lanes = 393,216 threads, each writing one aligned 16-byte
// slot. Runs once per weight, outside the timed GEMM and outside Graph
// capture; byte count is unchanged (6 MiB), so the packed tensor is a
// same-size permutation.
// ---------------------------------------------------------------------------
__global__ __launch_bounds__(kCopyBlockThreads) void
w8a8_pack_o_proj_m16_kernel(const int8_t* __restrict__ raw,
                            int8_t* __restrict__ packed,
                            int k,
                            int n) {
  const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x +
                      static_cast<int64_t>(threadIdx.x);
  const int k_pairs = (static_cast<int64_t>(k) / kDummaTileK) >> 1;  // 16
  const int64_t slots_per_tile =
      static_cast<int64_t>(k_pairs) * kDummaBlockThreads;  // 1024
  const int64_t total = slots_per_tile * (n / kDummaTileN);
  if (idx >= total) {
    return;
  }
  const int n_tile = static_cast<int>(idx / slots_per_tile);
  const int64_t rem = idx % slots_per_tile;
  const int pair = static_cast<int>(rem / kDummaBlockThreads);
  const int lane = static_cast<int>(rem % kDummaBlockThreads);
  const int row = lane & 15;
  const int col = (lane >> 4) << 3;
  // v0 = 8-B fragment slot of k_step 2p (low 8 B of the 16-B lane slot),
  // v1 = fragment slot of k_step 2p+1 (high 8 B).
  uint64_t v0 = 0;
  uint64_t v1 = 0;
#pragma unroll
  for (int s = 0; s < 2; ++s) {
    const int k_step = 2 * pair + s;
    const int8_t* __restrict__ src =
        raw + (static_cast<int64_t>(k_step) * kDummaTileK + col) * n +
        static_cast<int64_t>(n_tile) * kDummaTileN + row;
    uint64_t v = 0;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      v |= (static_cast<uint64_t>(static_cast<uint8_t>(src[i * n]))) << (8 * i);
    }
    if (s == 0) {
      v0 = v;
    } else {
      v1 = v;
    }
  }
  ulonglong2* __restrict__ dst = reinterpret_cast<ulonglong2*>(
      packed + ((static_cast<int64_t>(n_tile) * k_pairs + pair) *
                    kPackedPairBytes +
                static_cast<int64_t>(lane) * kPackedLaneBytes));
  ulonglong2 u;
  u.x = v0;
  u.y = v1;
  *dst = u;
}

// ---------------------------------------------------------------------------
// Identity device-to-device pack (generic fallback for every (K, N) except
// the exact (1024, 6144) o_proj weight, which uses the pack kernel above).
// ---------------------------------------------------------------------------
__global__ __launch_bounds__(kCopyBlockThreads) void
w8a8_identity_copy_i8_kernel(const int8_t* __restrict__ src,
                             int8_t* __restrict__ dst,
                             int64_t count) {
  const int64_t i = static_cast<int64_t>(blockIdx.x) * kCopyBlockThreads +
                    static_cast<int64_t>(threadIdx.x);
  if (i < count) {
    dst[i] = src[i];
  }
}

__global__ __launch_bounds__(kCopyBlockThreads) void
w8a8_identity_copy_f32_kernel(const float* __restrict__ src,
                              float* __restrict__ dst,
                              int64_t count) {
  const int64_t i = static_cast<int64_t>(blockIdx.x) * kCopyBlockThreads +
                    static_cast<int64_t>(threadIdx.x);
  if (i < count) {
    dst[i] = src[i];
  }
}

}  // namespace

// ---------------------------------------------------------------------------
// Stable host launcher symbols consumed by csrc/bindings.cpp.
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
  // The scalar bootstrap needs no workspace; the buffer is still required by
  // the API contract so later split-K rounds can use it without ABI changes.
  (void)workspace;
  (void)workspace_bytes;

  if (m <= 0 || n <= 0 || k <= 0) {
    return;
  }

  // Exact-shape specialization: minimax_tp8_o_proj_m16 (M=16, N=6144, K=1024)
  // -> packed-B DUMMA m16n16k32 in-block split-K=2 (iteration 8): 384 blocks
  // x 128 lanes (TWO 64-lane wavefronts per block), each wave computes the
  // same 16x16 output tile over a disjoint K half (512 K = 4 stages), int32
  // partials combined in LDS, A tile staged once into LDS, B read from the
  // packed [n_tile][k_step_pair][lane][16] fragment-slot layout (produced by
  // launch_pack_w8a8_weight for this (k, n)). Waves/CU doubles from 3.2 to
  // ~6.4 (1.6 waves/SIMD) with no repeated A/B HBM reads. All other
  // (m, n, k) -- including the paired M=2 TP8 o_proj shape with the same
  // (N, K) -- keep the scalar generic fallback below (which decodes the
  // packed layout for that exact (n, k)).
  if (m == 16 && n == kTargetN && k == kTargetK) {
    const dim3 block(kDummaBlockThreads2);  // 128 = 2 x 64-lane wavefronts
    const dim3 grid(kTargetN / kDummaTileN);
    hipLaunchKernelGGL(w8a8_dumma_m16n16k32_packedb_sk2_kernel, grid, block, 0,
                       stream, a, b, x_scale, weight_scale,
                       reinterpret_cast<hip_bfloat16*>(out), n, k);
    return;
  }

  const dim3 block(kScalarBlockThreads);
  const dim3 grid(
      static_cast<unsigned>((n + kScalarBlockThreads - 1) /
                            kScalarBlockThreads),
      static_cast<unsigned>(m));
  hipLaunchKernelGGL(w8a8_scalar_gemm_kernel, grid, block, 0, stream, a, b,
                     x_scale, weight_scale,
                     reinterpret_cast<hip_bfloat16*>(out), n, k);
}

extern "C" void launch_pack_w8a8_weight(
    const int8_t* raw_weight,
    const float* weight_scale,
    int8_t* packed_weight,
    float* packed_weight_scale,
    int k,
    int n,
    hipStream_t stream) {
  const int64_t weight_count = static_cast<int64_t>(k) * n;
  const dim3 block(kCopyBlockThreads);
  if (k == kTargetK && n == kTargetN) {
    // Exact o_proj M=16 weight: permute into [n_tile][k_step_pair][lane][16]
    // B fragment slots (one-time, untimed, outside Graph capture).
    const int64_t total = (static_cast<int64_t>(k) / kDummaTileK / 2) *
                          (n / kDummaTileN) * kDummaBlockThreads;
    const dim3 grid(static_cast<unsigned>(
        (total + kCopyBlockThreads - 1) / kCopyBlockThreads));
    hipLaunchKernelGGL(w8a8_pack_o_proj_m16_kernel, grid, block, 0, stream,
                       raw_weight, packed_weight, k, n);
  } else if (weight_count > 0) {
    const dim3 grid(static_cast<unsigned>(
        (weight_count + kCopyBlockThreads - 1) / kCopyBlockThreads));
    hipLaunchKernelGGL(w8a8_identity_copy_i8_kernel, grid, block, 0, stream,
                       raw_weight, packed_weight, weight_count);
  }
  if (n > 0) {
    const dim3 grid(static_cast<unsigned>(
        (n + kCopyBlockThreads - 1) / kCopyBlockThreads));
    hipLaunchKernelGGL(w8a8_identity_copy_f32_kernel, grid, block, 0, stream,
                       weight_scale, packed_weight_scale, n);
  }
}
// @@end
