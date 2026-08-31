// @@variant shape=hy3_tp8_shared_gate_up_proj_m4096 commit=95895c8eadee15332d1799fbfd36789ff57dfee7 added=2026-08-27
//   median_us=116 p90_us=116.2 speedup=144.1 baseline_us=1.671e+04
//   source=hy3-dsh-tp8-m4096-1-0ebb994d
// INT8 W8A8 GEMM HIP implementation for Hygon K500SM_AI / gfx928.
//
// Worker 2 assigned shape: hy3_tp8_shared_gate_up_proj_m4096
//   (M=4096, N=384, K=4096).
//
// Iteration 2 (architecture round): the bootstrap (iteration 1) was a
// single-buffered 64x128 macro-tile with identity [K, N] packed B. Its ISA
// (PMC bundle) shows the dominant steady-state costs are the B-fragment LDS
// path (with any 16-byte-aligned [K, N] row stride the int8 DUMMA row-major
// B loader reads 8 strided k-rows per lane -> eight ds_read_u8, 16-way bank
// aliasing) plus the per-dword mask/OR fragment reassembly VALU, and the
// single-buffered K loop serializes one global-load round trip in front of
// every stage. This round ports the sibling-validated architecture family
// (hy3 TP4 M4096 shared_gate_up_proj, median 220.7 us at 2x this shape's
// FLOPs, 116.8 TOPS) for the exact (K, N) = (4096, 384) pair:
//   * launch_pack_w8a8_weight stores the (K=4096, N=384) weight transposed
//     [N, K] (packing runs outside the timed/CUDA-Graph region; all other
//     (k, n) pairs keep the identity copy).
//   * The prefill kernel gains a kBuffers==2 software-pipelined K loop with
//     K stage 64 (one barrier per stage): stage s+1's global loads are
//     issued into registers before the stage-s MMA compute and committed to
//     the second LDS buffer after it, overlapping the global round trip with
//     the LDS-wait-bound compute. Buffer footprint 2 x (64x80 A + 128x80 B)
//     = 30,720 B/block -> 2 blocks/CU -> 16 waves/CU (occupancy preserved).
//   * B is staged [N, K] in LDS from the transposed packed weight and loaded
//     with the col_major fragment loader; each lane's 8 elements are 8
//     consecutive k-values at an 8-byte-aligned LDS address (row stride 80),
//     so each B fragment is ONE 8-byte LDS read (ds_read2_b32, ~4-way bank
//     floor) instead of eight ds_read_u8 at a 16-way aliasing phase.
//   * load_fragment8 writes the same 8 bytes directly into the fragment
//     storage (identical element-to-slot mapping and byte order), so the
//     v_mmac operand bit patterns and the exact int32 accumulation are
//     unchanged while the per-dword mask/OR reassembly VALU disappears.
//   * The generic single-buffer 64x128/128x64/64x64 arms, identity packing,
//     the scalar fallback (now transposed-aware for the exact (k, n) pair so
//     the paired M=2/M=16 decode shapes stay correct), and the A staging /
//     epilogue are unchanged in behavior.
//
// Iteration 9 (wave-structure round): the exact assigned shape (M=4096,
// N=384, K=4096) dispatches to w8a8_gemm_prefill_tiled_kernel_w4<64,128,64>
// (see that kernel's comment): identical 64x128 macro-tile, K stage 64
// double buffer, [N,K] packed B, 30,720 B LDS, 192-block grid, and
// bit-identical int32 accumulation, with 256 threads (4 waves) owning 32x64
// quadrants and eight independent accumulators per wave instead of 512
// threads owning 32x32 quadrants with four accumulators. Every other
// large-M shape with the packed (k, n) = (4096, 384) pair keeps the
// accepted 8-wave <64,128,64,2> path.
//
// Iteration 11 (address-math consolidation, HIP only): the accepted w4
// kernel's exact-source ISA recomputes the whole global/LDS address chain
// every K stage (~34 loop-invariant VALU/SALU per wave per stage: the row*K
// products, the quadrant/lane slot constants, and the stage-count bound),
// although only the 64-byte stage offset and the double-buffer select vary.
// This round hoists every loop-invariant address into per-thread staging
// pointers / LDS commit slots / per-wave fragment-read bases computed once
// before the K loop and walks the three global pointers by kStageK per
// stage. Same bytes, same element-to-slot mapping, same load/store order,
// same barrier structure, grid, LDS, and occupancy -> bit-identical int32
// accumulation (expected mismatch 0).
//
// Iteration 13 (burst-scheduling probe, HIP only): the w4 steady state
// issues each stage's sixteen v_mmac_i32_16x16x32_i8 as eight kk=0 MMACs
// followed by eight kk=kTileK MMACs into the same eight accumulators. This
// round pins the fragment-load schedule: all twelve load_fragment8 calls
// (six per kk slice) are issued EXPLICITLY before the whole sixteen-MMAC
// burst, so the second slice's six ds_read2_b64 no longer compete with the
// first slice's burst for a liveness-driven placement (the compiler keeps
// both slices' fragments live across the burst, ~+12 VGPR, still <= 128 for
// 2 blocks/CU). No load, byte, element-to-slot mapping, LDS commit, barrier
// (65/block), grid (192 blocks), LDS (30,720 B), threads (256), or MMAC
// emission order changes -> the int32 accumulation is bit-identical
// (expected mismatch 0).
//
// Iteration 18 (wave-count probe, ISA-guided round 2 of 2): the accepted w4
// cycle accounting (48 single-block CUs at 1 wave/SIMD take 164.7 cycles per
// v_mmac, 72 dual-block CUs at 2 waves/SIMD take 82.4, both landing exactly
// on the 168,654-cycle GRBM wall) isolates the per-SIMD v_mmac issue rate as
// the wall, and rounds 13-15 proved the burst dependency structure and the
// operand-register alternation do not move it. The untested axis is the
// WAVEFRONT COUNT per SIMD: the wall is invariant to occupancy while each
// wavefront still issues 1,024 v_mmac (165 cyc x 1,024 = 168,960 cyc), so
// the exact shape now dispatches to w8a8_gemm_prefill_tiled_kernel_w8
// <64,128,64> -- an 8-wave (512-thread) clone of the accepted w4 steady
// state (iteration-11 address hoist, iteration-13 burst pinning, same
// 64x128 tile, same 192-block grid, same 30,720-B LDS, same 65 barriers/
// block, same staging bytes and element-to-slot mapping -> bit-identical
// int32): 32x32 quadrant per wave, four accumulators, 8 MMACs/stage,
// 512 MMACs/wavefront, target <= 64 VGPR -> 2 blocks/CU -> 4 waves/SIMD.
// Falsifiable: 4-deep MMAC pipe -> wall 84,480 cyc (~58 us, ~2x); 2-deep
// pipe (throughput cap 1/82.4 per SIMD) -> wall ~168,755 (flat).
//
// Mathematical contract (exact int32 dot before float scaling):
//   out[m, n] = bf16(int32_dot(a[m, :], b[:, n]) * x_scale[m] * weight_scale[n])
//
// Header order is fixed by the control plane for this DTK:
//   hip_runtime.h -> hip_bfloat16.h -> du_mma.h

#include <hip/hip_runtime.h>
#include <hip/hip_bfloat16.h>
#include <du_mma.h>

#include <cstdint>

namespace {

// gfx928 INT8 DUMMA primitive: m16n16k32, int8 x int8 -> int32.
constexpr int kTileM = 16;
constexpr int kTileN = 16;
constexpr int kTileK = 32;

// gfx928 wavefront is 64 lanes; every blockDim must be a multiple of 64.
constexpr int kWaveSize = 64;

// Large-prefill dispatch geometry. Each macro-tile config uses a 32x32
// quadrant per wave, so waves per block = (BlockM/32) * (BlockN/32).
constexpr int kPrefillMinM = 128;
constexpr int kPrefillStageK = 128;
constexpr int kDefaultBlockM = 64;
constexpr int kDefaultBlockN = 64;

constexpr int kScalarThreads = 256;

using namespace du::dumma;

// Verified gfx928 INT8 DUMMA accumulator ownership:
//   row = lane & 15, col_mod4 = lane >> 4, frag.x[i] -> columns
//   col_mod4 + 4*i. Scale by x_scale[row] and weight_scale[col] and store
//   bf16 directly from the fragment (no accumulator LDS round trip).
template <typename AccFragment>
__device__ __forceinline__ void store_prefill_fragment(
    const AccFragment& frag,
    const float* __restrict__ x_scale,
    const float* __restrict__ weight_scale,
    hip_bfloat16* __restrict__ out,
    int m,
    int n,
    int base_row,
    int base_col,
    int lane) {
  const int row = base_row + (lane & 15);
  if (row >= m) {
    return;
  }
  const int col_mod4 = lane >> 4;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const int col = base_col + col_mod4 + 4 * i;
    const float scaled = static_cast<float>(frag.x[i]) *
                         x_scale[row] * weight_scale[col];
    out[row * n + col] = __float2bfloat16(scaled);
  }
}

// Direct 8-byte LDS fragment load for the double-buffered exact-shape path.
// du_load_matrix_sync's int8 loaders assign x[0..7] = 8 consecutive bytes at
// (lane & 15) * ldm + ((lane >> 4) << 3) for both matrix_a row_major and
// matrix_b col_major, and du_mma_sync passes the fragment to the v_mmac
// builtin as one packed 8-byte operand, so the compiler emits a per-byte
// mask/OR reassembly chain (v_and 0xff00/0xff0000/0xff000000 + v_or_b32_sdwa
// + v_or3_b32) for every loaded dword. Writing the same 8 bytes directly into
// the fragment storage keeps the operand bit pattern identical (exact int32
// accumulation unchanged) and lets the compiler feed the ds_read2_b32 pair
// straight to the v_mmac (still one ds_read2_b32 per fragment, 4-way floor).
__device__ __forceinline__ void load_fragment8(
    DUFragmentBase<signed char, 8>& frag,
    const signed char* __restrict__ base,
    int stride,
    int lane) {
  const int off = (lane & 15) * stride + ((lane >> 4) << 3);
  *reinterpret_cast<unsigned long long*>(&frag.x[0]) =
      *reinterpret_cast<const unsigned long long*>(base + off);
}

// Large-prefill path: one block computes a kBlockM x kBlockN output tile.
// kWaveRows x kWaveCols wavefronts each own a 32x32 quadrant (four 16x16
// DUMMA accumulators), while the block cooperatively stages A[kBlockM,
// kStageK] and B[kStageK, kBlockN] in bank-skewed LDS. Tail M rows are
// zero-filled on load and masked on store, so any M >= 128 is supported.
//
// kBuffers == 1: single-buffered K loop (K stage 128, two barriers per stage):
//   stage s+1's global loads cannot start until the stage-s barrier, so each
//   stage serializes one global-load round trip (global_load -> vmcnt(0) ->
//   ds_write -> lgkmcnt(0) -> barrier) in front of the compute.
// kBuffers == 2: software-pipelined K loop (K stage 64, one barrier per stage):
//   stage s+1's loads are issued into registers before the stage-s MMA compute
//   and committed to the other LDS buffer after it, overlapping the global
//   round trip with the LDS-wait-bound compute. Requires at most one int4 per
//   thread per operand (static_asserted), which holds for the exact shape.
//   The (K, N) = (4096, 384) packed weight is transposed [N, K], so B is
//   staged [N, K] and loaded with the col_major fragment loader.
template <int kBlockM, int kBlockN, int kStageK, int kBuffers = 1>
__global__ __launch_bounds__((kBlockM / 32) * (kBlockN / 32) * kWaveSize) void
w8a8_gemm_prefill_tiled_kernel(
    const int8_t* __restrict__ x_q,
    const int8_t* __restrict__ weight,
    const float* __restrict__ x_scale,
    const float* __restrict__ weight_scale,
    hip_bfloat16* __restrict__ out,
    int m,
    int n,
    int k) {
  constexpr int kWaveRows = kBlockM / 32;
  constexpr int kWaveCols = kBlockN / 32;
  constexpr int kThreads = kWaveRows * kWaveCols * kWaveSize;
  // 16-byte bank skew: keeps int4 staging stores aligned while spreading the
  // fragment rows across distinct LDS banks (~4-way floor for 8-byte/lane
  // reads instead of the 16-way aliasing of power-of-two strides).
  constexpr int kAStride = kStageK + sizeof(int4);
  // Double-buffered path (exact shape) stages B transposed [N, K] so the
  // DUMMA B fragments read 8 consecutive k-values per lane (one ds_read2_b32,
  // ~4-way) instead of 8 strided k-rows (eight ds_read_u8, 16-way: with any
  // 16-byte-aligned row stride S, B k-rows 8 apart alias onto one bank group
  // because (S/4)*8*g == 0 mod 32 for all g). The generic single-buffer path
  // keeps [K, N] B and the row-major B loader.
  constexpr int kBStride = (kBuffers == 2) ? (kStageK + sizeof(int4))
                                           : (kBlockN + sizeof(int4));
  constexpr int kBRows = (kBuffers == 2) ? kBlockN : kStageK;
  constexpr int kAVectorsPerRow = kStageK / sizeof(int4);
  constexpr int kAVectors = kBlockM * kAVectorsPerRow;
  constexpr int kBVectorsPerRow =
      (kBuffers == 2) ? (kStageK / sizeof(int4)) : (kBlockN / sizeof(int4));
  constexpr int kBVectors = kBRows * kBVectorsPerRow;

  const int tid = static_cast<int>(threadIdx.x);
  const int wave = tid / kWaveSize;
  const int lane = tid % kWaveSize;
  const int wave_row = wave / kWaveCols;
  const int wave_col = wave - wave_row * kWaveCols;
  const int m0 = static_cast<int>(blockIdx.y) * kBlockM;
  const int n0 = static_cast<int>(blockIdx.x) * kBlockN;

  __shared__ __align__(16) int8_t a_tile[kBuffers][kBlockM * kAStride];
  __shared__ __align__(16) int8_t b_tile[kBuffers][kBRows * kBStride];

  DUFragment<matrix_a, kTileM, kTileN, kTileK, signed char, row_major>
      a_frag0, a_frag1;
  DUFragment<matrix_b, kTileM, kTileN, kTileK, signed char, row_major>
      b_frag0, b_frag1;
  DUFragment<accumulator, kTileM, kTileN, kTileK, int>
      acc00, acc01, acc10, acc11;
  du_fill_fragment(acc00, 0);
  du_fill_fragment(acc01, 0);
  du_fill_fragment(acc10, 0);
  du_fill_fragment(acc11, 0);

  if constexpr (kBuffers == 2) {
    static_assert(kAVectors <= kThreads && kBVectors <= kThreads,
                  "double-buffered path stages at most one int4 per thread");
    // The exact shape's packed weight is [N, K] (transposed by
    // launch_pack_w8a8_weight), so B fragments load 8 consecutive k-values
    // per lane from the [N, K] LDS tile via the col_major loader; the
    // element-to-slot mapping (slot i = B[k = kk + 8*g + i][n = col + row])
    // is identical to the row-major loader on a [K, N] tile, so the v_mmac
    // operand values and the exact int32 accumulation are unchanged.
    DUFragment<matrix_b, kTileM, kTileN, kTileK, signed char, col_major>
        b_frag_t0, b_frag_t1;
    // Prologue: stage K tile 0 into buffer 0, then one barrier.
    for (int vec = tid; vec < kAVectors; vec += kThreads) {
      const int local_row = vec / kAVectorsPerRow;
      const int v = vec - local_row * kAVectorsPerRow;
      const int global_row = m0 + local_row;
      reinterpret_cast<int4*>(a_tile[0] + local_row * kAStride)[v] =
          global_row < m
              ? *reinterpret_cast<const int4*>(
                    x_q + global_row * k + v * static_cast<int>(sizeof(int4)))
              : int4{0, 0, 0, 0};
    }
    // B is packed [N, K] for the exact shape: each thread stages 16
    // consecutive k-values of one n row (n-stride in global is k).
    for (int vec = tid; vec < kBVectors; vec += kThreads) {
      const int n_local = vec / kBVectorsPerRow;
      const int v = vec - n_local * kBVectorsPerRow;
      reinterpret_cast<int4*>(b_tile[0] + n_local * kBStride)[v] =
          *reinterpret_cast<const int4*>(
              weight + (n0 + n_local) * k +
              v * static_cast<int>(sizeof(int4)));
    }
    __syncthreads();

    const int n_stages = k / kStageK;
    for (int s = 0; s < n_stages; ++s) {
      const int cur = s & 1;
      const int nxt = cur ^ 1;
      const bool has_next = (s + 1) < n_stages;
      // Issue the stage s+1 global loads now; the data is consumed by the
      // ds_write after the compute, so the vmcnt wait lands after the MMA
      // loop instead of stalling the front of the stage.
      int4 a_reg{0, 0, 0, 0};
      int4 b_reg{0, 0, 0, 0};
      if (has_next) {
        const int k1 = (s + 1) * kStageK;
        if (tid < kAVectors) {
          const int local_row = tid / kAVectorsPerRow;
          const int v = tid - local_row * kAVectorsPerRow;
          const int global_row = m0 + local_row;
          a_reg = global_row < m
                      ? *reinterpret_cast<const int4*>(
                            x_q + global_row * k + k1 +
                            v * static_cast<int>(sizeof(int4)))
                      : int4{0, 0, 0, 0};
        }
        const int n_local = tid / kBVectorsPerRow;
        const int v = tid - n_local * kBVectorsPerRow;
        b_reg = *reinterpret_cast<const int4*>(
            weight + (n0 + n_local) * k + k1 +
            v * static_cast<int>(sizeof(int4)));
      }

      // Compute stage s from the buffer staged last iteration.
      // Load the int8 fragments as raw 8-byte LDS reads into the fragment
      // storage (same bytes, same element-to-slot mapping), so the compiler
      // feeds the ds_read2_b32 pair straight to the v_mmac and the per-dword
      // mask/OR reassembly VALU disappears from the steady state.
#pragma unroll
      for (int kk = 0; kk < kStageK; kk += kTileK) {
        const int local_row = wave_row * 32;
        const int local_col = wave_col * 32;
        load_fragment8(
            a_frag0, a_tile[cur] + local_row * kAStride + kk, kAStride,
            lane);
        load_fragment8(
            a_frag1, a_tile[cur] + (local_row + kTileM) * kAStride + kk,
            kAStride, lane);
        load_fragment8(
            b_frag_t0, b_tile[cur] + local_col * kBStride + kk, kBStride,
            lane);
        load_fragment8(
            b_frag_t1, b_tile[cur] + (local_col + kTileN) * kBStride + kk,
            kBStride, lane);
        du_mma_sync(acc00, a_frag0, b_frag_t0, acc00);
        du_mma_sync(acc01, a_frag0, b_frag_t1, acc01);
        du_mma_sync(acc10, a_frag1, b_frag_t0, acc10);
        du_mma_sync(acc11, a_frag1, b_frag_t1, acc11);
      }

      // Commit the prefetched stage s+1 into the other LDS buffer, then one
      // barrier: it orders both this iteration's compute reads of buffer cur
      // (against the next-next prefetch, which reuses cur) and the prefetch
      // writes of buffer nxt (against the next iteration's compute reads).
      if (has_next) {
        if (tid < kAVectors) {
          const int local_row = tid / kAVectorsPerRow;
          const int v = tid - local_row * kAVectorsPerRow;
          reinterpret_cast<int4*>(a_tile[nxt] + local_row * kAStride)[v] =
              a_reg;
        }
        const int n_local = tid / kBVectorsPerRow;
        const int v = tid - n_local * kBVectorsPerRow;
        reinterpret_cast<int4*>(b_tile[nxt] + n_local * kBStride)[v] = b_reg;
      }
      __syncthreads();
    }
  } else {
    for (int k0 = 0; k0 < k; k0 += kStageK) {
      // Stage A[kBlockM, kStageK] into LDS (zero-filled tail M rows).
      for (int vec = tid; vec < kAVectors; vec += kThreads) {
        const int local_row = vec / kAVectorsPerRow;
        const int v = vec - local_row * kAVectorsPerRow;
        const int global_row = m0 + local_row;
        reinterpret_cast<int4*>(a_tile[0] + local_row * kAStride)[v] =
            global_row < m
                ? *reinterpret_cast<const int4*>(
                      x_q + global_row * k + k0 +
                      v * static_cast<int>(sizeof(int4)))
                : int4{0, 0, 0, 0};
      }
      // Stage B[kStageK, kBlockN] into LDS from the packed weight. The
      // generic path keeps the identity [K, N] layout (pack_weight is an
      // identity copy for every (k, n) pair other than the exact shape).
      for (int vec = tid; vec < kBVectors; vec += kThreads) {
        const int kk = vec / kBVectorsPerRow;
        const int v = vec - kk * kBVectorsPerRow;
        reinterpret_cast<int4*>(b_tile[0] + kk * kBStride)[v] =
            *reinterpret_cast<const int4*>(
                weight + (k0 + kk) * n + n0 +
                v * static_cast<int>(sizeof(int4)));
      }
      __syncthreads();

      // Each wave computes its 32x32 quadrant: four m16n16k32 DUMMA tiles.
#pragma unroll
      for (int kk = 0; kk < kStageK; kk += kTileK) {
        const int local_row = wave_row * 32;
        const int local_col = wave_col * 32;
        du_load_matrix_sync(
            a_frag0, a_tile[0] + local_row * kAStride + kk, kAStride);
        du_load_matrix_sync(
            a_frag1, a_tile[0] + (local_row + kTileM) * kAStride + kk,
            kAStride);
        du_load_matrix_sync(
            b_frag0, b_tile[0] + kk * kBStride + local_col, kBStride);
        du_load_matrix_sync(
            b_frag1, b_tile[0] + kk * kBStride + local_col + kTileN,
            kBStride);
        du_mma_sync(acc00, a_frag0, b_frag0, acc00);
        du_mma_sync(acc01, a_frag0, b_frag1, acc01);
        du_mma_sync(acc10, a_frag1, b_frag0, acc10);
        du_mma_sync(acc11, a_frag1, b_frag1, acc11);
      }
      __syncthreads();
    }
  }

  const int base_row = m0 + wave_row * 32;
  const int base_col = n0 + wave_col * 32;
  store_prefill_fragment(
      acc00, x_scale, weight_scale, out, m, n,
      base_row, base_col, lane);
  store_prefill_fragment(
      acc01, x_scale, weight_scale, out, m, n,
      base_row, base_col + kTileN, lane);
  store_prefill_fragment(
      acc10, x_scale, weight_scale, out, m, n,
      base_row + kTileM, base_col, lane);
  store_prefill_fragment(
      acc11, x_scale, weight_scale, out, m, n,
      base_row + kTileM, base_col + kTileN, lane);
}

// Iteration 9: 4-wave / 8-accumulator variant of the exact-shape
// double-buffered 64x128 kernel (w8a8_gemm_prefill_tiled_kernel_w4, only
// instantiated as <64,128,64>). The macro-tile, K stage 64 double buffer
// (one barrier/stage), [N,K] packed B, bank-skewed LDS layout (30,720
// B/block), element-to-slot fragment mapping, byte order, and the
// k0-outer/kk-inner int32 accumulation order are IDENTICAL to the accepted
// 8-wave kernel, so the result is bit-identical; the only deltas are the
// wave partition and the per-thread staging split:
//   * 512 -> 256 threads (4 wavefronts); each wave owns a 32x64 quadrant
//     (two 16-row bands x four 16-col bands) = EIGHT independent m16n16k32
//     accumulators (8-deep MMAC ILP per kk slice) instead of a 32x32
//     quadrant with four accumulators (4-deep ILP).
//   * A staging is one int4/thread (256 vectors), B staging two int4/thread
//     (512 vectors) at invariant staged bytes (12 KiB/stage/block).
// Per block per stage the LDS fragment-read instructions drop 32 -> 24 (A is
// read once per 64-col wave group instead of once per 32-col group) and the
// LDS write instructions drop 16 -> 12, at invariant total MMA count (64
// v_mmac per block stage) and invariant bank-conflict floors (8-cycle 2-way
// reads, 16-cycle 4-way writes). Grid (N/128) x (M/64) = 3 x 64 = 192 blocks;
// 30,720 B x 2 = 61,440 B <= 64 KiB LDS and ~80-96 VGPR x 256 x 2 <= 512 KiB
// -> 2 blocks/CU -> 8 waves/CU.
//
// Iteration 11 (address-math consolidation): the steady-state K loop's
// address expressions are strength-reduced to loop-invariant bases computed
// once before the loop -- the three global staging pointers (A/B0/B1) walk
// +kStageK per stage, the three LDS commit slots add only the buffer offset
// nxt * kBufInt4, and the six fragment reads add only the buffer offset
// cur * kBufBytes and the unrolled kk slice. The exact-source ISA of the
// accepted kernel recomputes the full row*K product chains and slot
// constants every stage (~34 VALU/SALU per wave per stage at 0x4A0C-0x4AEC,
// 0x4ABC-0x4AE8, 0x4BFC-0x4C1C of the iteration-9 gfx928.co: v_mul_lo_u32,
// v_mad_u32_u24, v_add3_u32, v_lshlrev_b32, s_ashr_i32 chains whose operands
// are all stage-invariant); this is the concrete compiler limitation recorded
// for the ISA-guided gate. LDS contents, fragment operands, element-to-slot
// mapping, k0-outer/kk-inner int32 accumulation order, barrier count
// (65/block), grid (192 blocks), LDS (30,720 B), threads (256), and VGPR
// class are unchanged -> the result is bit-identical to iteration 9.
template <int kBlockM, int kBlockN, int kStageK>
__global__ __launch_bounds__(4 * kWaveSize) void
w8a8_gemm_prefill_tiled_kernel_w4(
    const int8_t* __restrict__ x_q,
    const int8_t* __restrict__ weight,
    const float* __restrict__ x_scale,
    const float* __restrict__ weight_scale,
    hip_bfloat16* __restrict__ out,
    int m,
    int n,
    int k) {
  constexpr int kThreads = 4 * kWaveSize;
  constexpr int kAStride = kStageK + sizeof(int4);
  constexpr int kBStride = kStageK + sizeof(int4);
  constexpr int kAVectorsPerRow = kStageK / sizeof(int4);
  constexpr int kAVectors = kBlockM * kAVectorsPerRow;
  constexpr int kBVectorsPerRow = kStageK / sizeof(int4);
  constexpr int kBVectors = kBlockN * kBVectorsPerRow;
  static_assert(kAVectors == kThreads,
                "w4 path stages exactly one A int4 per thread");
  static_assert(kBVectors == 2 * kThreads,
                "w4 path stages exactly two B int4 per thread");

  const int tid = static_cast<int>(threadIdx.x);
  const int wave = tid / kWaveSize;
  const int lane = tid % kWaveSize;
  const int wave_row = wave / 2;
  const int wave_col = wave - wave_row * 2;
  const int m0 = static_cast<int>(blockIdx.y) * kBlockM;
  const int n0 = static_cast<int>(blockIdx.x) * kBlockN;

  __shared__ __align__(16) int8_t a_tile[2][kBlockM * kAStride];
  __shared__ __align__(16) int8_t b_tile[2][kBlockN * kBStride];

  DUFragment<accumulator, kTileM, kTileN, kTileK, int> acc[2][4];
#pragma unroll
  for (int r = 0; r < 2; ++r) {
#pragma unroll
    for (int c = 0; c < 4; ++c) {
      du_fill_fragment(acc[r][c], 0);
    }
  }

  // Prologue: stage K tile 0 into buffer 0 (same vectors and addresses as
  // the 512-thread kernel, distributed over 256 threads), then one barrier.
  for (int vec = tid; vec < kAVectors; vec += kThreads) {
    const int local_row = vec / kAVectorsPerRow;
    const int v = vec - local_row * kAVectorsPerRow;
    const int global_row = m0 + local_row;
    reinterpret_cast<int4*>(a_tile[0] + local_row * kAStride)[v] =
        global_row < m
            ? *reinterpret_cast<const int4*>(
                  x_q + global_row * k + v * static_cast<int>(sizeof(int4)))
            : int4{0, 0, 0, 0};
  }
  for (int vec = tid; vec < kBVectors; vec += kThreads) {
    const int n_local = vec / kBVectorsPerRow;
    const int v = vec - n_local * kBVectorsPerRow;
    reinterpret_cast<int4*>(b_tile[0] + n_local * kBStride)[v] =
        *reinterpret_cast<const int4*>(
            weight + (n0 + n_local) * k +
            v * static_cast<int>(sizeof(int4)));
  }
  __syncthreads();

  const int n_stages = k / kStageK;
  // Iteration 11 (address-math consolidation): hoist every loop-invariant
  // address of the steady state out of the K loop. The three global staging
  // pointers (A/B0/B1) are precomputed at the stage-1 offset and walk
  // +kStageK per stage; the three LDS commit slots are precomputed in buffer
  // 0 and add only the buffer offset nxt * kBufInt4; the six fragment reads
  // use the per-wave buffer-0 bases and add only the buffer offset
  // cur * kBufBytes and the unrolled kk slice. The bytes, the
  // element-to-slot mapping, and the load/store order are identical to the
  // accepted kernel, so the int32 accumulation is bit-identical.
  const int local_row_a = tid / kAVectorsPerRow;
  const int v_a = tid - local_row_a * kAVectorsPerRow;
  const int global_row_a = m0 + local_row_a;
  const bool a_row_valid = global_row_a < m;
  const int8_t* __restrict__ a_g =
      x_q + static_cast<int64_t>(global_row_a) * k +
      v_a * static_cast<int>(sizeof(int4)) + kStageK;
  const int n_local0 = tid / kBVectorsPerRow;
  const int v_b0 = tid - n_local0 * kBVectorsPerRow;
  const int8_t* __restrict__ b_g0 =
      weight + static_cast<int64_t>(n0 + n_local0) * k +
      v_b0 * static_cast<int>(sizeof(int4)) + kStageK;
  const int vec1 = tid + kThreads;
  const int n_local1 = vec1 / kBVectorsPerRow;
  const int v_b1 = vec1 - n_local1 * kBVectorsPerRow;
  const int8_t* __restrict__ b_g1 =
      weight + static_cast<int64_t>(n0 + n_local1) * k +
      v_b1 * static_cast<int>(sizeof(int4)) + kStageK;
  // LDS commit slots as int4 indexes into buffer 0 (the loop adds the
  // buffer-1 offset nxt * kBufInt4; both buffers are 16-byte aligned so the
  // int4 stores are exactly the accepted ds_write_b128 pattern).
  int4* const a_wslot =
      reinterpret_cast<int4*>(a_tile[0] + local_row_a * kAStride) + v_a;
  int4* const b_wslot0 =
      reinterpret_cast<int4*>(b_tile[0] + n_local0 * kBStride) + v_b0;
  int4* const b_wslot1 =
      reinterpret_cast<int4*>(b_tile[0] + n_local1 * kBStride) + v_b1;
  constexpr int kABufInt4 =
      (kBlockM * kAStride) / static_cast<int>(sizeof(int4));
  constexpr int kBBufInt4 =
      (kBlockN * kBStride) / static_cast<int>(sizeof(int4));
  // Per-wave fragment-read bases in buffer 0; the loop adds the buffer
  // offset cur * kBufBytes and the kk slice (same addresses as
  // a_tile[cur] + (wave_row * 32) * kAStride + kk, etc.).
  const int8_t* const a_rd = a_tile[0] + (wave_row * 32) * kAStride;
  const int8_t* const b_rd = b_tile[0] + (wave_col * 64) * kBStride;
  constexpr int kABufBytes = kBlockM * kAStride;
  constexpr int kBBufBytes = kBlockN * kBStride;

  for (int s = 0; s < n_stages; ++s) {
    const int cur = s & 1;
    const int nxt = cur ^ 1;
    const bool has_next = (s + 1) < n_stages;
    // Issue the stage s+1 global loads from the precomputed pointers (A: one
    // int4/thread, B: two int4/thread); the vmcnt wait lands after the MMA
    // burst, at the commit.
    int4 a_reg{0, 0, 0, 0};
    int4 b_reg0{0, 0, 0, 0};
    int4 b_reg1{0, 0, 0, 0};
    if (has_next) {
      a_reg = a_row_valid ? *reinterpret_cast<const int4*>(a_g)
                          : int4{0, 0, 0, 0};
      b_reg0 = *reinterpret_cast<const int4*>(b_g0);
      b_reg1 = *reinterpret_cast<const int4*>(b_g1);
      a_g += kStageK;
      b_g0 += kStageK;
      b_g1 += kStageK;
    }

    // Compute stage s: each wave owns a 32x64 quadrant (two 16-row bands x
    // four 16-col bands), eight independent m16n16k32 accumulators. The
    // k0-outer/kk-inner order per output element is unchanged, so the int32
    // accumulation is bit-identical to the accepted kernel.
    //
    // Iteration 13 (burst-scheduling probe): the twelve fragment loads of
    // the stage (six per kk slice) are all issued here, ahead of the whole
    // sixteen-MMAC burst, in the same address order as the accepted kernel.
    // This pins the second slice's ds_reads above the first slice's MMAC
    // burst (both slices' fragments live across the burst); the MMAC
    // emission order (kk=0 group, then kk=kTileK group, per accumulator) is
    // unchanged, so the result is bit-identical.
    static_assert(kStageK == 2 * kTileK,
                  "w4 burst schedule expects exactly two kk slices per stage");
    DUFragment<matrix_a, kTileM, kTileN, kTileK, signed char, row_major>
        a_frag00, a_frag01, a_frag10, a_frag11;
    DUFragment<matrix_b, kTileM, kTileN, kTileK, signed char, col_major>
        b_frag00, b_frag01, b_frag02, b_frag03, b_frag10, b_frag11, b_frag12,
        b_frag13;
    const int8_t* const a_cur = a_rd + cur * kABufBytes;
    const int8_t* const b_cur = b_rd + cur * kBBufBytes;
    // kk = 0 slice loads (row bands x0/x1, col bands 0..3).
    load_fragment8(a_frag00, a_cur, kAStride, lane);
    load_fragment8(a_frag01, a_cur + kTileM * kAStride, kAStride, lane);
    load_fragment8(b_frag00, b_cur, kBStride, lane);
    load_fragment8(
        b_frag01, b_cur + kTileN * kBStride, kBStride, lane);
    load_fragment8(
        b_frag02, b_cur + 2 * kTileN * kBStride, kBStride, lane);
    load_fragment8(
        b_frag03, b_cur + 3 * kTileN * kBStride, kBStride, lane);
    // kk = kTileK slice loads (same rows/cols, k offset kTileK).
    load_fragment8(a_frag10, a_cur + kTileK, kAStride, lane);
    load_fragment8(
        a_frag11, a_cur + kTileM * kAStride + kTileK, kAStride, lane);
    load_fragment8(b_frag10, b_cur + kTileK, kBStride, lane);
    load_fragment8(
        b_frag11, b_cur + kTileN * kBStride + kTileK, kBStride, lane);
    load_fragment8(
        b_frag12, b_cur + 2 * kTileN * kBStride + kTileK, kBStride, lane);
    load_fragment8(
        b_frag13, b_cur + 3 * kTileN * kBStride + kTileK, kBStride, lane);
    // kk = 0 MMAC group.
    du_mma_sync(acc[0][0], a_frag00, b_frag00, acc[0][0]);
    du_mma_sync(acc[0][1], a_frag00, b_frag01, acc[0][1]);
    du_mma_sync(acc[0][2], a_frag00, b_frag02, acc[0][2]);
    du_mma_sync(acc[0][3], a_frag00, b_frag03, acc[0][3]);
    du_mma_sync(acc[1][0], a_frag01, b_frag00, acc[1][0]);
    du_mma_sync(acc[1][1], a_frag01, b_frag01, acc[1][1]);
    du_mma_sync(acc[1][2], a_frag01, b_frag02, acc[1][2]);
    du_mma_sync(acc[1][3], a_frag01, b_frag03, acc[1][3]);
    // kk = kTileK MMAC group (same accumulators, same order).
    du_mma_sync(acc[0][0], a_frag10, b_frag10, acc[0][0]);
    du_mma_sync(acc[0][1], a_frag10, b_frag11, acc[0][1]);
    du_mma_sync(acc[0][2], a_frag10, b_frag12, acc[0][2]);
    du_mma_sync(acc[0][3], a_frag10, b_frag13, acc[0][3]);
    du_mma_sync(acc[1][0], a_frag11, b_frag10, acc[1][0]);
    du_mma_sync(acc[1][1], a_frag11, b_frag11, acc[1][1]);
    du_mma_sync(acc[1][2], a_frag11, b_frag12, acc[1][2]);
    du_mma_sync(acc[1][3], a_frag11, b_frag13, acc[1][3]);

    // Commit the prefetched stage s+1 into the other LDS buffer, then one
    // barrier (same ordering semantics as the accepted kernel).
    if (has_next) {
      *(a_wslot + nxt * kABufInt4) = a_reg;
      *(b_wslot0 + nxt * kBBufInt4) = b_reg0;
      *(b_wslot1 + nxt * kBBufInt4) = b_reg1;
    }
    __syncthreads();
  }

  const int base_row = m0 + wave_row * 32;
  const int base_col = n0 + wave_col * 64;
#pragma unroll
  for (int r = 0; r < 2; ++r) {
#pragma unroll
    for (int c = 0; c < 4; ++c) {
      store_prefill_fragment(
          acc[r][c], x_scale, weight_scale, out, m, n,
          base_row + r * kTileM, base_col + c * kTileN, lane);
    }
  }
}

// Iteration 18: 8-wave / 4-accumulator variant of the exact-shape
// double-buffered 64x128 kernel (w8a8_gemm_prefill_tiled_kernel_w8, only
// instantiated as <64,128,64>). The macro-tile, K stage 64 double buffer
// (one barrier/stage), [N,K] packed B, bank-skewed LDS layout (30,720
// B/block), staging bytes, element-to-slot fragment mapping, byte order,
// and the k0-outer/kk-inner int32 accumulation order are IDENTICAL to the
// accepted w4 kernel, so the result is bit-identical; the only deltas are
// the wave partition and the per-thread staging split:
//   * 256 -> 512 threads (8 wavefronts); each wave owns a 32x32 quadrant
//     (two 16-row bands x two 16-col bands) = FOUR independent m16n16k32
//     accumulators (4-deep MMAC ILP per kk slice) instead of a 32x64
//     quadrant with eight accumulators. Per-wavefront MMACs per stage drop
//     16 -> 8 (1,024 -> 512 MMACs per wavefront over the K loop).
//   * A staging is one int4/thread for the first 256 threads, B staging one
//     int4/thread for all 512 threads (same 12,288 B/stage/block as w4).
// Grid (N/128) x (M/64) = 192 blocks; LDS 30,720 B x 2 = 61,440 B <= 64 KiB;
// target <= 64 VGPR x 512 x 2 <= 512 KiB -> 2 blocks/CU -> 4 waves/SIMD
// (must be confirmed in the code object; > 64 VGPR degenerates the probe to
// 1 block/CU = 2 waves/SIMD). Same address hoist (iteration 11) and
// burst-pinned fragment schedule (iteration 13) as the accepted w4.
template <int kBlockM, int kBlockN, int kStageK>
__global__ __launch_bounds__(8 * kWaveSize) void
w8a8_gemm_prefill_tiled_kernel_w8(
    const int8_t* __restrict__ x_q,
    const int8_t* __restrict__ weight,
    const float* __restrict__ x_scale,
    const float* __restrict__ weight_scale,
    hip_bfloat16* __restrict__ out,
    int m,
    int n,
    int k) {
  constexpr int kThreads = 8 * kWaveSize;
  constexpr int kAStride = kStageK + sizeof(int4);
  constexpr int kBStride = kStageK + sizeof(int4);
  constexpr int kAVectorsPerRow = kStageK / sizeof(int4);
  constexpr int kAVectors = kBlockM * kAVectorsPerRow;
  constexpr int kBVectorsPerRow = kStageK / sizeof(int4);
  constexpr int kBVectors = kBlockN * kBVectorsPerRow;
  static_assert(kAVectors <= kThreads,
                "w8 path stages A with at most one int4 per thread");
  static_assert(kBVectors == kThreads,
                "w8 path stages exactly one B int4 per thread");

  const int tid = static_cast<int>(threadIdx.x);
  const int wave = tid / kWaveSize;
  const int lane = tid % kWaveSize;
  const int wave_row = wave / 4;
  const int wave_col = wave - wave_row * 4;
  const int m0 = static_cast<int>(blockIdx.y) * kBlockM;
  const int n0 = static_cast<int>(blockIdx.x) * kBlockN;

  __shared__ __align__(16) int8_t a_tile[2][kBlockM * kAStride];
  __shared__ __align__(16) int8_t b_tile[2][kBlockN * kBStride];

  DUFragment<accumulator, kTileM, kTileN, kTileK, int> acc[2][2];
#pragma unroll
  for (int r = 0; r < 2; ++r) {
#pragma unroll
    for (int c = 0; c < 2; ++c) {
      du_fill_fragment(acc[r][c], 0);
    }
  }

  // Prologue: stage K tile 0 into buffer 0 (A: first 256 threads, B: all 512
  // threads; same vectors and addresses as the w4 kernel distributed over
  // 512 threads), then one barrier.
  for (int vec = tid; vec < kAVectors; vec += kThreads) {
    const int local_row = vec / kAVectorsPerRow;
    const int v = vec - local_row * kAVectorsPerRow;
    const int global_row = m0 + local_row;
    reinterpret_cast<int4*>(a_tile[0] + local_row * kAStride)[v] =
        global_row < m
            ? *reinterpret_cast<const int4*>(
                  x_q + global_row * k + v * static_cast<int>(sizeof(int4)))
            : int4{0, 0, 0, 0};
  }
  for (int vec = tid; vec < kBVectors; vec += kThreads) {
    const int n_local = vec / kBVectorsPerRow;
    const int v = vec - n_local * kBVectorsPerRow;
    reinterpret_cast<int4*>(b_tile[0] + n_local * kBStride)[v] =
        *reinterpret_cast<const int4*>(
            weight + (n0 + n_local) * k +
            v * static_cast<int>(sizeof(int4)));
  }
  __syncthreads();

  const int n_stages = k / kStageK;
  // Iteration 11 (address-math consolidation), same as w4: hoist every
  // loop-invariant address of the steady state out of the K loop. The A
  // staging pointer is only dereferenced by threads tid < kAVectors.
  const bool a_active = tid < kAVectors;
  const int local_row_a = tid / kAVectorsPerRow;
  const int v_a = tid - local_row_a * kAVectorsPerRow;
  const int global_row_a = m0 + local_row_a;
  const bool a_row_valid = global_row_a < m;
  const int8_t* __restrict__ a_g =
      x_q + static_cast<int64_t>(global_row_a) * k +
      v_a * static_cast<int>(sizeof(int4)) + kStageK;
  const int n_local = tid / kBVectorsPerRow;
  const int v_b = tid - n_local * kBVectorsPerRow;
  const int8_t* __restrict__ b_g =
      weight + static_cast<int64_t>(n0 + n_local) * k +
      v_b * static_cast<int>(sizeof(int4)) + kStageK;
  int4* const a_wslot =
      reinterpret_cast<int4*>(a_tile[0] + local_row_a * kAStride) + v_a;
  int4* const b_wslot =
      reinterpret_cast<int4*>(b_tile[0] + n_local * kBStride) + v_b;
  constexpr int kABufInt4 =
      (kBlockM * kAStride) / static_cast<int>(sizeof(int4));
  constexpr int kBBufInt4 =
      (kBlockN * kBStride) / static_cast<int>(sizeof(int4));
  // Per-wave fragment-read bases in buffer 0 (32x32 quadrant per wave); the
  // loop adds the buffer offset cur * kBufBytes and the kk slice.
  const int8_t* const a_rd = a_tile[0] + (wave_row * 32) * kAStride;
  const int8_t* const b_rd = b_tile[0] + (wave_col * 32) * kBStride;
  constexpr int kABufBytes = kBlockM * kAStride;
  constexpr int kBBufBytes = kBlockN * kBStride;

  for (int s = 0; s < n_stages; ++s) {
    const int cur = s & 1;
    const int nxt = cur ^ 1;
    const bool has_next = (s + 1) < n_stages;
    // Issue the stage s+1 global loads from the precomputed pointers (A: one
    // int4/thread for tid < kAVectors, B: one int4/thread); the vmcnt wait
    // lands after the MMA burst, at the commit.
    int4 a_reg{0, 0, 0, 0};
    int4 b_reg{0, 0, 0, 0};
    if (has_next) {
      if (a_active) {
        a_reg = a_row_valid ? *reinterpret_cast<const int4*>(a_g)
                            : int4{0, 0, 0, 0};
      }
      b_reg = *reinterpret_cast<const int4*>(b_g);
      a_g += kStageK;
      b_g += kStageK;
    }

    // Compute stage s: each wave owns a 32x32 quadrant (two 16-row bands x
    // two 16-col bands), four independent m16n16k32 accumulators. The
    // k0-outer/kk-inner order per output element is unchanged, so the int32
    // accumulation is bit-identical to the accepted w4 kernel.
    //
    // Iteration 13 (burst-scheduling probe), same as w4: the eight fragment
    // loads of the stage (four per kk slice) are all issued here, ahead of
    // the whole eight-MMAC burst, in the same address order as the accepted
    // w4 kernel.
    static_assert(kStageK == 2 * kTileK,
                  "w8 burst schedule expects exactly two kk slices per stage");
    DUFragment<matrix_a, kTileM, kTileN, kTileK, signed char, row_major>
        a_frag00, a_frag01, a_frag10, a_frag11;
    DUFragment<matrix_b, kTileM, kTileN, kTileK, signed char, col_major>
        b_frag00, b_frag01, b_frag10, b_frag11;
    const int8_t* const a_cur = a_rd + cur * kABufBytes;
    const int8_t* const b_cur = b_rd + cur * kBBufBytes;
    // kk = 0 slice loads (row bands x0/x1, col bands 0..1).
    load_fragment8(a_frag00, a_cur, kAStride, lane);
    load_fragment8(a_frag01, a_cur + kTileM * kAStride, kAStride, lane);
    load_fragment8(b_frag00, b_cur, kBStride, lane);
    load_fragment8(b_frag01, b_cur + kTileN * kBStride, kBStride, lane);
    // kk = kTileK slice loads (same rows/cols, k offset kTileK).
    load_fragment8(a_frag10, a_cur + kTileK, kAStride, lane);
    load_fragment8(
        a_frag11, a_cur + kTileM * kAStride + kTileK, kAStride, lane);
    load_fragment8(b_frag10, b_cur + kTileK, kBStride, lane);
    load_fragment8(
        b_frag11, b_cur + kTileN * kBStride + kTileK, kBStride, lane);
    // kk = 0 MMAC group.
    du_mma_sync(acc[0][0], a_frag00, b_frag00, acc[0][0]);
    du_mma_sync(acc[0][1], a_frag00, b_frag01, acc[0][1]);
    du_mma_sync(acc[1][0], a_frag01, b_frag00, acc[1][0]);
    du_mma_sync(acc[1][1], a_frag01, b_frag01, acc[1][1]);
    // kk = kTileK MMAC group (same accumulators, same order).
    du_mma_sync(acc[0][0], a_frag10, b_frag10, acc[0][0]);
    du_mma_sync(acc[0][1], a_frag10, b_frag11, acc[0][1]);
    du_mma_sync(acc[1][0], a_frag11, b_frag10, acc[1][0]);
    du_mma_sync(acc[1][1], a_frag11, b_frag11, acc[1][1]);

    // Commit the prefetched stage s+1 into the other LDS buffer, then one
    // barrier (same ordering semantics as the accepted w4 kernel).
    if (has_next) {
      if (a_active) {
        *(a_wslot + nxt * kABufInt4) = a_reg;
      }
      *(b_wslot + nxt * kBBufInt4) = b_reg;
    }
    __syncthreads();
  }

  const int base_row = m0 + wave_row * 32;
  const int base_col = n0 + wave_col * 32;
#pragma unroll
  for (int r = 0; r < 2; ++r) {
#pragma unroll
    for (int c = 0; c < 2; ++c) {
      store_prefill_fragment(
          acc[r][c], x_scale, weight_scale, out, m, n,
          base_row + r * kTileM, base_col + c * kTileN, lane);
    }
  }
}

// Generic scalar fallback: one output element per grid-stride iteration with
// exact int32 accumulation. Correct for any (m, n, k), including the small-M
// API shapes (M=2/16) and any unmatched geometry. The exact (k, n) =
// (4096, 384) pair reads the transposed [N, K] packed weight (the pack
// specialization is keyed on (k, n), so every consumer of that packed pair,
// including the paired M=2/M=16 decode shapes that reach this fallback, must
// interpret it transposed).
__global__ __launch_bounds__(kScalarThreads) void
w8a8_gemm_scalar_kernel(
    const int8_t* __restrict__ x_q,
    const int8_t* __restrict__ weight,
    const float* __restrict__ x_scale,
    const float* __restrict__ weight_scale,
    hip_bfloat16* __restrict__ out,
    int m,
    int n,
    int k) {
  const bool packed_transposed = (k == 4096 && n == 384);
  const int64_t total = static_cast<int64_t>(m) * n;
  const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x +
                     threadIdx.x;
       idx < total; idx += stride) {
    const int row = static_cast<int>(idx / n);
    const int col = static_cast<int>(idx - static_cast<int64_t>(row) * n);
    int32_t acc = 0;
    const int8_t* a_row = x_q + static_cast<int64_t>(row) * k;
    if (packed_transposed) {
      // packed[n * k + kk] = raw[kk * n + n].
      const int8_t* b_col = weight + static_cast<int64_t>(col) * k;
      for (int kk = 0; kk < k; ++kk) {
        acc += static_cast<int32_t>(a_row[kk]) *
               static_cast<int32_t>(b_col[kk]);
      }
    } else {
      for (int kk = 0; kk < k; ++kk) {
        acc += static_cast<int32_t>(a_row[kk]) *
               static_cast<int32_t>(
                   weight[static_cast<int64_t>(kk) * n + col]);
      }
    }
    const float scaled = static_cast<float>(acc) * x_scale[row] *
                         weight_scale[col];
    out[idx] = __float2bfloat16(scaled);
  }
}

// pack_weight. The packed layout is opaque per the API contract; for the
// exact (K=4096, N=384) pair the weight is stored transposed [N, K]
// (n-major) so the double-buffered GEMM stages B in [N, K] LDS order and the
// DUMMA B fragments read 8 consecutive k-values per lane. Every other (k, n)
// pair keeps the identity [K, N] copy (generic fallback unchanged).
__global__ __launch_bounds__(kScalarThreads) void
w8a8_pack_kernel(
    const int8_t* __restrict__ raw_weight,
    const float* __restrict__ weight_scale,
    int8_t* __restrict__ packed_weight,
    float* __restrict__ packed_weight_scale,
    int64_t weight_elems,
    int n,
    int k) {
  const int64_t idx =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (k == 4096 && n == 384) {
    // Transpose [K, N] -> [N, K]: packed[n * k + kk] = raw[kk * n + n].
    if (idx < weight_elems) {
      const int64_t n_idx = idx / static_cast<int64_t>(k);
      const int64_t k_idx = idx - n_idx * static_cast<int64_t>(k);
      packed_weight[idx] = raw_weight[k_idx * n + n_idx];
    }
  } else if (idx < weight_elems) {
    packed_weight[idx] = raw_weight[idx];
  }
  if (idx < n) {
    packed_weight_scale[idx] = weight_scale[idx];
  }
}

}  // namespace

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
  // No split-K: the provided workspace is not touched. The timed operator
  // performs no allocation, synchronization, packing, or default-stream
  // launch; only the caller-provided out is written.
  (void)workspace;
  (void)workspace_bytes;
  auto* out_bf16 = static_cast<hip_bfloat16*>(out);

  if (m >= kPrefillMinM && (n % kDefaultBlockN == 0) &&
      (k % kPrefillStageK == 0)) {
    if (n == 384 && k == 4096) {
      // Packed (k, n) = (4096, 384) weight is transposed [N, K], so every
      // large-M shape with this pair uses the double-buffered transposed-B
      // kernel (tail M rows zero-filled on load, masked on store). For the
      // exact assigned shape (M=4096, N=384, K=4096): 64x128 macro-tile,
      // grid 3x64 = 192 blocks, K stage 64 double-buffered, 30,720 B
      // LDS/block -> 2 blocks/CU (see the iteration-18 branch below for the
      // current 8-wave dispatch; the accepted 4-wave w4 path is the
      // bit-identical 117.50 us best).
      const dim3 grid(n / 128, (m + 63) / 64);
      if (m == 4096) {
        // Iteration 18 (wave-count probe, ISA-guided round 2 of 2): the
        // exact shape dispatches to the 8-wave variant w8<64,128,64> (512
        // threads, 32x32 quadrant per wave, 4 accumulators, 512
        // MMACs/wavefront) at the same tile, grid (192 blocks), LDS (30,720
        // B), and barrier count (65/block) as the accepted w4 path.
        const dim3 block(8 * kWaveSize);
        hipLaunchKernelGGL(
            HIP_KERNEL_NAME(w8a8_gemm_prefill_tiled_kernel_w8<64, 128, 64>),
            grid, block, 0, stream,
            a, b, x_scale, weight_scale, out_bf16, m, n, k);
        return;
      }
      // Other large-M shapes with the packed (k, n) pair keep the accepted
      // 8-wave <64,128,64,2> kernel (bit-identical path, 169.46 us median
      // for M=4096).
      const dim3 block(8 * kWaveSize);
      hipLaunchKernelGGL(
          HIP_KERNEL_NAME(w8a8_gemm_prefill_tiled_kernel<64, 128, 64, 2>),
          grid, block, 0, stream,
          a, b, x_scale, weight_scale, out_bf16, m, n, k);
      return;
    }
    if (n % 128 == 0) {
      // 64x128 tile, 8 waves.
      const dim3 grid(n / 128, (m + 63) / 64);
      const dim3 block(8 * kWaveSize);
      hipLaunchKernelGGL(
          HIP_KERNEL_NAME(w8a8_gemm_prefill_tiled_kernel<64, 128, 128>),
          grid, block, 0, stream,
          a, b, x_scale, weight_scale, out_bf16, m, n, k);
      return;
    }
    if (m % 128 == 0) {
      // 128x64 tile, 8 waves.
      const dim3 grid(n / 64, m / 128);
      const dim3 block(8 * kWaveSize);
      hipLaunchKernelGGL(
          HIP_KERNEL_NAME(w8a8_gemm_prefill_tiled_kernel<128, 64, 128>),
          grid, block, 0, stream,
          a, b, x_scale, weight_scale, out_bf16, m, n, k);
      return;
    }
    {
      // 64x64 tile, 4 waves (generic default).
      const dim3 grid(n / 64, (m + 63) / 64);
      const dim3 block(4 * kWaveSize);
      hipLaunchKernelGGL(
          HIP_KERNEL_NAME(w8a8_gemm_prefill_tiled_kernel<64, 64, 128>),
          grid, block, 0, stream,
          a, b, x_scale, weight_scale, out_bf16, m, n, k);
      return;
    }
  }

  const int64_t total = static_cast<int64_t>(m) * n;
  const unsigned blocks = static_cast<unsigned>(
      (total + kScalarThreads - 1) / kScalarThreads);
  hipLaunchKernelGGL(
      w8a8_gemm_scalar_kernel,
      dim3(blocks), dim3(kScalarThreads), 0, stream,
      a, b, x_scale, weight_scale, out_bf16, m, n, k);
}

extern "C" void launch_pack_w8a8_weight(
    const int8_t* raw_weight,
    const float* weight_scale,
    int8_t* packed_weight,
    float* packed_weight_scale,
    int k,
    int n,
    hipStream_t stream) {
  // Exact (K, N) = (4096, 384) is stored transposed [N, K]; every other pair
  // keeps the identity [K, N] copy. Runs outside the timed/CUDA-Graph region.
  const int64_t weight_elems = static_cast<int64_t>(k) * n;
  const int64_t total = weight_elems > n ? weight_elems : n;
  const unsigned blocks = static_cast<unsigned>(
      (total + kScalarThreads - 1) / kScalarThreads);
  hipLaunchKernelGGL(
      w8a8_pack_kernel,
      dim3(blocks), dim3(kScalarThreads), 0, stream,
      raw_weight, weight_scale, packed_weight, packed_weight_scale,
      weight_elems, n, k);
}
// @@end
// @@end
