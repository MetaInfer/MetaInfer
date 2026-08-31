// @@variant shape=minimax_tp8_shared_gate_up_proj_m4096 commit=7c4c216bf06688139ccf5d4bbc70df27a420f661 added=2026-08-29
//   median_us=319.2 p90_us=320.6 speedup=99.89 baseline_us=3.188e+04
//   source=minimaxm3-dsh-tp8-m4096-1-b0482833
// MetaInfer W8A8 INT8 GEMM backend for Hygon gfx928 (K500SM_AI).
// worker_3 native INT8 DUMMA m16n16k32 prefill kernels for the assigned
// MiniMax TP8 M=4096 shapes:
//   * minimax_tp8_shared_gate_up_proj_m4096: (M=4096, N=768,  K=6144)
//   * minimax_tp8_shared_down_proj_m4096:    (M=4096, N=6144, K=384)
//
// This file provides the two stable host launch symbols consumed by
// csrc/bindings.cpp:
//   * launch_w8a8_gemm(...)        - timed, graph-safe GEMM on the caller's
//                                    HIP stream; no allocation, packing,
//                                    autotuning, or synchronization. Uses
//                                    only the caller-provided out and
//                                    workspace (workspace untouched by this
//                                    no-split-K path).
//   * launch_pack_w8a8_weight(...) - one-time out-of-timed-region packing;
//                                    the exact (K, N) = (6144, 768) pair is
//                                    stored transposed [N, K], every other
//                                    pair keeps the identity copy.
//
// Strategy (round 1: establish the 2-D macro-tile DUMMA baseline for the
// exact gate_up shape by porting the sibling-validated gate_up architecture
// family; the bootstrap's single-buffered 64x128 tile was measured at
// 595 us / 64.95 TOPS):
//   * Large-M prefill (M >= 128): native INT8 DUMMA m16n16k32 2-D macro-tiles
//     (64x64 / 64x128 / 128x64), each wave owning a 32x32 quadrant (four
//     16x16 int32 accumulators) resident across the whole K loop. A and B are
//     cooperatively staged into bank-skewed LDS with 16-byte vectorized
//     coalesced global loads.
//   * Exact gate_up shape (M=4096, N=768, K=6144): dispatches to
//     w8a8_gemm_prefill_tiled_kernel_w4<64,128,64> -- the 64x128 macro-tile,
//     256 threads = 4 wavefronts, each wave owning a 32x64 quadrant (eight
//     m16n16k32 accumulators), K stage 64 DOUBLE-buffered with one
//     __syncthreads per stage (stage s+1's global loads issue into registers
//     before the stage-s MMAC burst and commit after it), B packed [N, K]
//     and loaded with col_major fragments as ONE 8-byte LDS read per lane
//     (load_fragment8: identical element-to-slot mapping and byte order, no
//     per-dword mask/OR reassembly VALU), fused fragment -> scale -> bf16
//     epilogue. LDS 30,720 B/block -> 2 blocks/CU; grid (N/128) x (M/64) =
//     6 x 64 = 384 blocks; 96 K stages. No split-K: the M x N output-tile
//     grid dwarfs the 120 CUs.
//   * Other large-M shapes sharing the packed (k, n) = (6144, 768) pair use
//     the generic 8-wave double-buffered <64,128,64,2> kernel (bit-identical
//     path); the generic single-buffer 64x128/128x64/64x64 arms and the
//     identity packing for unmatched (k, n) pairs are unchanged.
//   * The scalar fallback is transposed-aware for the exact (k, n) pair so
//     every consumer of the packed (6144, 768) buffer (including the paired
//     M=2/M=16 decode shapes) stays correct.
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

// Large-prefill dispatch geometry.
constexpr int kPrefillMinM = 128;
constexpr int kPrefillStageK = 128;
constexpr int kDefaultBlockN = 64;

// LDS padding in bytes added to each staged row. A keeps 16 so every A row
// start stays 16-byte aligned for int4 staging while shifting fragment rows
// off the same 32-bank phase (power-of-two strides alias every row onto
// identical banks). B deliberately uses only 4 bytes of pad: with the gfx928
// INT8 DUMMA B-fragment layout (lane (c, g) reads bytes at k-rows 8g..8g+7,
// one byte per row, ldm apart), any 16-byte-aligned row stride makes
// 2*stride*g == 0 (mod 32), so all four k-groups of every ds_read_u8 land on
// the same bank phase -> 4 banks x 16 lanes (16-way). Stride 132 (== 4 mod
// 16) moves the k-groups to bank phases {0,8,16,24} -> 16 banks x 4 lanes.
constexpr int kLdsPad = 16;
constexpr int kBLdsPad = 4;

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
// mask/OR reassembly chain for every loaded dword. Writing the same 8 bytes
// directly into the fragment storage keeps the operand bit pattern identical
// (exact int32 accumulation unchanged) and lets the compiler feed the
// ds_read2_b32 pair straight to the v_mmac (one ds_read2_b32 per fragment,
// ~4-way bank floor instead of the 16-way aliasing of strided byte reads).
__device__ __forceinline__ void load_fragment8(
    DUFragmentBase<signed char, 8>& frag,
    const signed char* __restrict__ base,
    int stride,
    int lane) {
  const int off = (lane & 15) * stride + ((lane >> 4) << 3);
  *reinterpret_cast<unsigned long long*>(&frag.x[0]) =
      *reinterpret_cast<const unsigned long long*>(base + off);
}

// Large-M prefill path: one block computes a kBlockM x kBlockN output tile.
// kWaveRows x kWaveCols wavefronts each own a 32x32 quadrant (four 16x16
// DUMMA accumulators), while the block cooperatively stages
// A[kBlockM, kStageK] and B[kStageK, kBlockN] in bank-skewed LDS. The LDS is
// single-buffered with two barriers per stage (stage in, compute, protect).
// Tail-M rows are zero-filled on load and masked on store, so any M >= 128
// is supported. The packed weight is the bootstrap identity [K, N] layout,
// so B rows are staged n-major contiguous chunks.
template <int kBlockM, int kBlockN, int kStageK>
__global__ __launch_bounds__((kBlockM / 32) * (kBlockN / 32) * kWaveSize) void
w8a8_dumma_prefill_kernel(
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
  constexpr int kAStride = kStageK + kLdsPad;
  constexpr int kBStride = kBlockN + kBLdsPad;
  static_assert(kAStride % sizeof(int4) == 0,
                "A LDS row stride must stay 16-byte aligned");
  static_assert(kBStride % sizeof(int32_t) == 0,
                "B LDS row stride must stay 4-byte aligned (int32 stores)");
  static_assert(kStageK % kTileK == 0,
                "K stage must be a multiple of the DUMMA K unit");
  constexpr int kAVectorsPerRow = kStageK / sizeof(int4);
  constexpr int kAVectors = kBlockM * kAVectorsPerRow;
  constexpr int kBVectorsPerRow = kBlockN / sizeof(int4);
  constexpr int kBVectors = kStageK * kBVectorsPerRow;

  const int tid = static_cast<int>(threadIdx.x);
  const int wave = tid / kWaveSize;
  const int lane = tid % kWaveSize;
  const int wave_row = wave / kWaveCols;
  const int wave_col = wave - wave_row * kWaveCols;
  const int m0 = static_cast<int>(blockIdx.y) * kBlockM;
  const int n0 = static_cast<int>(blockIdx.x) * kBlockN;

  __shared__ __align__(16) int8_t a_tile[kBlockM * kAStride];
  __shared__ __align__(16) int8_t b_tile[kStageK * kBStride];

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

  for (int k0 = 0; k0 < k; k0 += kStageK) {
    // Cooperatively stage A[kBlockM, kStageK] rows into LDS. Out-of-range
    // (tail-M) rows are zero-filled so the DUMMA math stays defined.
    for (int vec = tid; vec < kAVectors; vec += kThreads) {
      const int local_row = vec / kAVectorsPerRow;
      const int v = vec - local_row * kAVectorsPerRow;
      const int global_row = m0 + local_row;
      reinterpret_cast<int4*>(a_tile + local_row * kAStride)[v] =
          global_row < m
              ? *reinterpret_cast<const int4*>(
                    x_q + global_row * k + k0 +
                    v * static_cast<int>(sizeof(int4)))
              : int4{0, 0, 0, 0};
    }
    // Cooperatively stage B[kStageK, kBlockN] rows into LDS. The packed
    // weight is the bootstrap identity [K, N] layout, so each B row is a
    // contiguous chunk of kBlockN columns at row stride n. Global loads stay
    // 16-byte vectorized and fully coalesced (one int4 per element); the LDS
    // row stride kBStride = kBlockN + 4 is not 16-byte aligned by design (see
    // kBLdsPad), so ds_write_b128 is unavailable and each int4 is committed
    // as four int32 stores (identical total bank traffic to one b128).
    for (int vec = tid; vec < kBVectors; vec += kThreads) {
      const int kk = vec / kBVectorsPerRow;
      const int v = vec - kk * kBVectorsPerRow;
      const int4 b = *reinterpret_cast<const int4*>(
          weight + (k0 + kk) * n + n0 +
          v * static_cast<int>(sizeof(int4)));
      int32_t* b_row = reinterpret_cast<int32_t*>(
          b_tile + kk * kBStride + v * static_cast<int>(sizeof(int4)));
      b_row[0] = b.x;
      b_row[1] = b.y;
      b_row[2] = b.z;
      b_row[3] = b.w;
    }
    __syncthreads();

    // Compute the kTileK steps of this stage. Each wave owns a 32x32
    // quadrant: two A fragments (rows 0-15 / 16-31) and two B fragments
    // (cols 0-15 / 16-31) per step, four du_mma_sync per step. The kk
    // ordering (k0-outer, kk-inner step kTileK) keeps the int32 accumulation
    // in ascending k order, bit-identical to the scalar fallback.
    const int local_row = wave_row * 32;
    const int local_col = wave_col * 32;
#pragma unroll
    for (int kk = 0; kk < kStageK; kk += kTileK) {
      du_load_matrix_sync(
          a_frag0, a_tile + local_row * kAStride + kk, kAStride);
      du_load_matrix_sync(
          a_frag1, a_tile + (local_row + kTileM) * kAStride + kk, kAStride);
      du_load_matrix_sync(
          b_frag0, b_tile + kk * kBStride + local_col, kBStride);
      du_load_matrix_sync(
          b_frag1, b_tile + kk * kBStride + local_col + kTileN, kBStride);
      du_mma_sync(acc00, a_frag0, b_frag0, acc00);
      du_mma_sync(acc01, a_frag0, b_frag1, acc01);
      du_mma_sync(acc10, a_frag1, b_frag0, acc10);
      du_mma_sync(acc11, a_frag1, b_frag1, acc11);
    }

    // Protect the single LDS buffer from being overwritten by the next
    // stage's staging while this stage's fragment loads are still in flight.
    __syncthreads();
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

// Double-buffered variant of the large-prefill path above (ported from the
// sibling-validated gate_up lineage; only instantiated as <64,128,64,2>).
//
// kBuffers == 1: single-buffered K loop (K stage 128, two barriers per stage):
//   stage s+1's global loads cannot start until the stage-s barrier, so each
//   stage serializes one global-load round trip in front of the compute.
// kBuffers == 2: software-pipelined K loop (K stage 64, one barrier per
//   stage): stage s+1's loads are issued into registers before the stage-s
//   MMA compute and committed to the other LDS buffer after it, overlapping
//   the global round trip with the LDS-wait-bound compute. Requires at most
//   one int4 per thread per operand (static_asserted), which holds for the
//   exact shape. The (K, N) = (6144, 768) packed weight is transposed
//   [N, K], so B is staged [N, K] and loaded with the col_major fragment
//   loader (8 consecutive k-values per lane -> one ds_read2_b32 per fragment,
//   ~4-way bank floor).
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

// 8-wave / 4-accumulator variant of the exact-shape double-buffered 64x128
// kernel (w8a8_gemm_prefill_tiled_kernel_w8, instantiated as <64,128,64>).
// The macro-tile, K stage 64 double buffer (one barrier/stage), [N,K] packed
// B, bank-skewed LDS layout (30,720 B/block), staging bytes, element-to-slot
// fragment mapping, byte order, and the k0-outer/kk-inner int32 accumulation
// order are IDENTICAL to the sibling-accepted 4-wave kernel, so the result is
// bit-identical; the only deltas are the wave partition and the per-thread
// staging split:
//   * 512 threads (8 wavefronts); each wave owns a 32x32 quadrant (two
//     16-row bands x two 16-col bands) = FOUR independent m16n16k32
//     accumulators (4-deep MMAC ILP per kk slice). Per-wavefront MMACs per
//     stage = 8 (512 MMACs/wavefront over a K=4096 loop).
//   * A staging is one int4/thread for the first 256 threads, B staging one
//     int4/thread for all 512 threads (same 12,288 B/stage/block).
// Grid (N/128) x (M/64) = 6 x 64 = 384 blocks; LDS 30,720 B x 2 = 61,440 B
// <= 64 KiB; target <= 64 VGPR x 512 x 2 <= 512 KiB -> 2 blocks/CU -> 4
// waves/SIMD (must be confirmed in the code object; > 64 VGPR degenerates to
// 1 block/CU = 2 waves/SIMD). Loop-invariant staging/fragment addresses are
// hoisted before the K loop, and the eight fragment loads of each stage are
// pinned ahead of the eight-MMAC burst.
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
  // threads; same vectors and addresses as the generic kernel distributed
  // over 512 threads), then one barrier.
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
  // Hoist every loop-invariant address of the steady state out of the K
  // loop. The A staging pointer is only dereferenced by threads tid <
  // kAVectors. The three global staging pointers (A/B) walk +kStageK per
  // stage; the LDS commit slots add only the buffer offset nxt * kBufInt4;
  // the fragment reads use the per-wave buffer-0 bases and add only the
  // buffer offset cur * kBufBytes and the unrolled kk slice.
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

  // Prologue prefetch: issue the tile-1 global loads into the loop-carried
  // payload now (a_g/b_g already address tile 1). The prologue barrier above
  // already ordered the tile-0 staging; the publish of this payload happens
  // at the top of stage 0, one full stage later.
  int4 a_pld{0, 0, 0, 0};
  int4 b_pld{0, 0, 0, 0};
  if (a_active) {
    a_pld = a_row_valid ? *reinterpret_cast<const int4*>(a_g)
                        : int4{0, 0, 0, 0};
  }
  b_pld = *reinterpret_cast<const int4*>(b_g);
  a_g += kStageK;
  b_g += kStageK;

  // Steady state (stages 0..n_stages-2): publish-at-top double buffering.
  // The tile-(s+1) payload prefetched one iteration earlier is written into
  // the idle LDS buffer at the TOP of stage s, BEFORE the MMAC burst, so the
  // ds_write latency (and the already-satisfied vmcnt wait) overlaps the
  // compute instead of sitting between the MMACs and the barrier (the exact
  // code object shows both ds_write_b128 after the v_mmac group, immediately
  // before s_barrier). The tile-(s+2) prefetch then issues from the walking
  // pointers; its vmcnt wait lands at the next stage's publish. One barrier
  // per stage orders both this stage's fragment reads of buffer cur (against
  // the next publish into it) and this stage's publish into buffer nxt
  // (against the next stage's reads of it) -- identical semantics to the old
  // commit-after-compute loop, so every global load, LDS write, fragment
  // read, and v_mmac is unchanged and the int32 accumulation stays
  // bit-identical.
  DUFragment<matrix_a, kTileM, kTileN, kTileK, signed char, row_major>
      a_frag00, a_frag01, a_frag10, a_frag11;
  DUFragment<matrix_b, kTileM, kTileN, kTileK, signed char, col_major>
      b_frag00, b_frag01, b_frag10, b_frag11;
  static_assert(kStageK == 2 * kTileK,
                "w8 burst schedule expects exactly two kk slices per stage");
  for (int s = 0; s < n_stages - 1; ++s) {
    const int cur = s & 1;
    const int nxt = cur ^ 1;
    // Publish tile s+1 into buffer nxt at the top of the stage.
    if (a_active) {
      *(a_wslot + nxt * kABufInt4) = a_pld;
    }
    *(b_wslot + nxt * kBBufInt4) = b_pld;
    // Prefetch tile s+2 (skipped on the last loop iteration; the peeled
    // final stage below needs no payload).
    if (s + 2 < n_stages) {
      if (a_active) {
        a_pld = a_row_valid ? *reinterpret_cast<const int4*>(a_g)
                            : int4{0, 0, 0, 0};
      }
      b_pld = *reinterpret_cast<const int4*>(b_g);
      a_g += kStageK;
      b_g += kStageK;
    }

    // Compute stage s: each wave owns a 32x32 quadrant (two 16-row bands x
    // two 16-col bands), four independent m16n16k32 accumulators. The
    // k0-outer/kk-inner order per output element is unchanged, so the int32
    // accumulation is bit-identical to the generic kernel.
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

    __syncthreads();
  }

  // Peeled final stage: compute tile n_stages-1 from buffer (n_stages-1)&1,
  // which the last loop iteration published above its barrier. No prefetch,
  // no publish, and no dead barrier after the last stage: each wave's
  // epilogue writes its own disjoint 32x32 output quadrant, so no
  // cross-wave ordering is required.
  {
    const int cur = (n_stages - 1) & 1;
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

// 4-wave / 8-accumulator variant of the exact-shape double-buffered 64x128
// kernel (w8a8_gemm_prefill_tiled_kernel_w4, instantiated as <64,128,64>).
// The macro-tile, K stage 64 double buffer (one barrier/stage, publish-at-top,
// peeled final stage), [N,K] packed B, bank-skewed LDS layout (30,720
// B/block), staging bytes, element-to-slot fragment mapping, byte order, and
// the k0-outer/kk-inner int32 accumulation order are IDENTICAL to the
// accepted 8-wave kernel, so the result is bit-identical; the only deltas are
// the wave partition and the per-thread staging split:
//   * 256 threads (4 wavefronts); each wave owns a 32x64 quadrant (two
//     16-row bands x four 16-col bands) = EIGHT independent m16n16k32 int32
//     accumulators (8-deep MMAC ILP per kk slice). Per block-stage the LDS
//     fragment reads drop from 8 waves x 8 = 64 to 4 waves x 12 = 48
//     load_fragment8 (six ds_read2_b64 per wave) and the LDS read BYTES drop
//     from 32 KB to 24 KB: the 32x32 quadrant layout re-reads A 4x and B 2x,
//     the 32x64 quadrant re-reads both 2x (per-wave per stage: 4 A fragments
//     + 8 B fragments), so per block-stage the LDS read instruction count
//     drops 64 -> 48 and the family lds_instructions 1.62M -> ~1.33M.
//   * A staging is one int4/thread for all 256 threads, B staging two
//     int4/thread for all 256 threads (same 12,288 B/stage/block).
// Grid (N/128) x (M/64) = 6 x 64 = 384 blocks; LDS 30,720 B x 2 = 61,440 B
// <= 64 KiB; 256 threads x 2 blocks = 8 waves/CU = 2 waves/SIMD at
// arch_vgpr <= 128 (8 accumulators = 32 VGPR + 12 loop-carried payload +
// 12 in-flight fragment VGPR + hoisted addresses ~= 80; 2 x 256 x 128 =
// 65,536 exactly fits; > 128 VGPR degenerates to 1 block/CU and predicts a
// regression -- falsifiable). Loop-invariant staging/fragment addresses are
// hoisted before the K loop, and the twelve fragment loads of each stage are
// pinned ahead of the sixteen-MMAC burst.
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
                "w4 path stages A with exactly one int4 per thread");
  static_assert(kBVectors == 2 * kThreads,
                "w4 path stages B with exactly two int4 per thread");

  const int tid = static_cast<int>(threadIdx.x);
  const int wave = tid / kWaveSize;
  const int lane = tid % kWaveSize;
  const int wave_row = wave >> 1;
  const int wave_col = wave & 1;
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

  // Prologue: stage K tile 0 into buffer 0 (A: one int4/thread, B: two
  // int4/thread; the same vectors and addresses as the 8-wave kernel
  // distributed over 256 threads), then one barrier.
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
  // Hoist every loop-invariant address of the steady state out of the K
  // loop (same scheme as the 8-wave kernel; B staging is two int4/thread:
  // vectors tid for rows tid/4 (0..63) and tid+256 for rows 64 + tid/4).
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
  // Per-wave fragment-read bases in buffer 0 (32x64 quadrant per wave); the
  // loop adds the buffer offset cur * kBufBytes and the kk slice.
  const int8_t* const a_rd = a_tile[0] + (wave_row * 32) * kAStride;
  const int8_t* const b_rd = b_tile[0] + (wave_col * 64) * kBStride;
  constexpr int kABufBytes = kBlockM * kAStride;
  constexpr int kBBufBytes = kBlockN * kBStride;

  // Prologue prefetch: issue the tile-1 global loads into the loop-carried
  // payload now (a_g/b_g0/b_g1 already address tile 1). The publish of this
  // payload happens at the top of stage 0, one full stage later.
  int4 a_pld{0, 0, 0, 0};
  int4 b_pld0{0, 0, 0, 0};
  int4 b_pld1{0, 0, 0, 0};
  a_pld = a_row_valid ? *reinterpret_cast<const int4*>(a_g)
                      : int4{0, 0, 0, 0};
  b_pld0 = *reinterpret_cast<const int4*>(b_g0);
  b_pld1 = *reinterpret_cast<const int4*>(b_g1);
  a_g += kStageK;
  b_g0 += kStageK;
  b_g1 += kStageK;

  // Steady state (stages 0..n_stages-2): publish-at-top double buffering
  // with identical semantics to the 8-wave kernel (one barrier per stage;
  // 1 prologue + 95 loop = 96 dynamic barriers/block for K=6144, stage 64).
  DUFragment<matrix_a, kTileM, kTileN, kTileK, signed char, row_major>
      a_frag00, a_frag01, a_frag10, a_frag11;
  DUFragment<matrix_b, kTileM, kTileN, kTileK, signed char, col_major>
      b_frag00, b_frag01, b_frag02, b_frag03;
  DUFragment<matrix_b, kTileM, kTileN, kTileK, signed char, col_major>
      b_frag10, b_frag11, b_frag12, b_frag13;
  static_assert(kStageK == 2 * kTileK,
                "w4 burst schedule expects exactly two kk slices per stage");
  for (int s = 0; s < n_stages - 1; ++s) {
    const int cur = s & 1;
    const int nxt = cur ^ 1;
    // Publish tile s+1 into buffer nxt at the top of the stage.
    *(a_wslot + nxt * kABufInt4) = a_pld;
    *(b_wslot0 + nxt * kBBufInt4) = b_pld0;
    *(b_wslot1 + nxt * kBBufInt4) = b_pld1;
    // Prefetch tile s+2 (skipped on the last loop iteration; the peeled
    // final stage below needs no payload).
    if (s + 2 < n_stages) {
      a_pld = a_row_valid ? *reinterpret_cast<const int4*>(a_g)
                          : int4{0, 0, 0, 0};
      b_pld0 = *reinterpret_cast<const int4*>(b_g0);
      b_pld1 = *reinterpret_cast<const int4*>(b_g1);
      a_g += kStageK;
      b_g0 += kStageK;
      b_g1 += kStageK;
    }

    // Compute stage s: each wave owns a 32x64 quadrant (two 16-row bands x
    // four 16-col bands), eight independent m16n16k32 accumulators. The
    // k0-outer/kk-inner order per output element is unchanged, so the int32
    // accumulation is bit-identical to the 8-wave/generic kernels.
    const int8_t* const a_cur = a_rd + cur * kABufBytes;
    const int8_t* const b_cur = b_rd + cur * kBBufBytes;
    // kk = 0 slice loads (row bands x0/x1, col bands 0..3).
    load_fragment8(a_frag00, a_cur, kAStride, lane);
    load_fragment8(a_frag01, a_cur + kTileM * kAStride, kAStride, lane);
    load_fragment8(b_frag00, b_cur, kBStride, lane);
    load_fragment8(b_frag01, b_cur + kTileN * kBStride, kBStride, lane);
    load_fragment8(b_frag02, b_cur + 2 * kTileN * kBStride, kBStride, lane);
    load_fragment8(b_frag03, b_cur + 3 * kTileN * kBStride, kBStride, lane);
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

    __syncthreads();
  }

  // Peeled final stage: compute tile n_stages-1 from buffer (n_stages-1)&1,
  // which the last loop iteration published above its barrier. No prefetch,
  // no publish, and no dead barrier after the last stage: each wave's
  // epilogue writes its own disjoint 32x64 output quadrant, so no
  // cross-wave ordering is required.
  {
    const int cur = (n_stages - 1) & 1;
    const int8_t* const a_cur = a_rd + cur * kABufBytes;
    const int8_t* const b_cur = b_rd + cur * kBBufBytes;
    // kk = 0 slice loads (row bands x0/x1, col bands 0..3).
    load_fragment8(a_frag00, a_cur, kAStride, lane);
    load_fragment8(a_frag01, a_cur + kTileM * kAStride, kAStride, lane);
    load_fragment8(b_frag00, b_cur, kBStride, lane);
    load_fragment8(b_frag01, b_cur + kTileN * kBStride, kBStride, lane);
    load_fragment8(b_frag02, b_cur + 2 * kTileN * kBStride, kBStride, lane);
    load_fragment8(b_frag03, b_cur + 3 * kTileN * kBStride, kBStride, lane);
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

// Generic scalar fallback: one output element per thread with exact int32
// accumulation in k order. Correct for any (m, n, k), including the small-M
// API shapes (M=2/M=16) and any unmatched geometry. The exact (k, n) =
// (6144, 768) pair reads the transposed [N, K] packed weight (the pack
// specialization is keyed on (k, n), so every consumer of that packed pair,
// including the paired M=2/M=16 decode shapes that reach this fallback, must
// interpret it transposed). b_col_stride carries the identity [K, N] column
// stride (n) for all other pairs.
__global__ __launch_bounds__(kScalarThreads) void w8a8_gemm_scalar_kernel(
    const int8_t* __restrict__ x_q,
    const int8_t* __restrict__ weight,
    const float* __restrict__ x_scale,
    const float* __restrict__ weight_scale,
    hip_bfloat16* __restrict__ out,
    int m,
    int n,
    int k,
    int b_col_stride) {
  const bool packed_transposed = (k == 6144 && n == 768);
  const int64_t total = static_cast<int64_t>(m) * n;
  const int64_t idx =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= total) {
    return;
  }
  const int row = static_cast<int>(idx / n);
  const int col = static_cast<int>(idx - static_cast<int64_t>(row) * n);
  const int8_t* a_row = x_q + static_cast<int64_t>(row) * k;
  // packed[n * k + kk] = raw[kk * n + n]: the column's k values are
  // contiguous at packed + col * k.
  const int8_t* b_col = packed_transposed
                            ? weight + static_cast<int64_t>(col) * k
                            : weight + static_cast<int64_t>(col) * b_col_stride;
  int32_t acc = 0;
  if (packed_transposed) {
    for (int kk = 0; kk < k; ++kk) {
      acc += static_cast<int32_t>(a_row[kk]) *
             static_cast<int32_t>(b_col[kk]);
    }
  } else {
    for (int kk = 0; kk < k; ++kk) {
      acc += static_cast<int32_t>(a_row[kk]) *
             static_cast<int32_t>(
                 b_col[static_cast<int64_t>(kk) * b_col_stride]);
    }
  }
  const float scaled =
      static_cast<float>(acc) * x_scale[row] * weight_scale[col];
  out[idx] = __float2bfloat16(scaled);
}

// pack_weight. The packed layout is opaque per the API contract; for the
// exact (K=6144, N=768) pair the weight is stored transposed [N, K]
// (n-major) so the double-buffered GEMM stages B in [N, K] LDS order and the
// DUMMA B fragments read 8 consecutive k-values per lane. Every other (k, n)
// pair keeps the identity [K, N] copy (generic fallback unchanged). Runs
// outside the timed/CUDA-Graph region.
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
  if (k == 6144 && n == 768) {
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

// ---------------------------------------------------------------------------
// Host launch symbols (extern "C", consumed by csrc/bindings.cpp).
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
  // No split-K: the workspace is not touched. The timed operator performs no
  // allocation, synchronization, packing, or default-stream launch; it
  // writes only the caller-provided out.
  (void)workspace;
  (void)workspace_bytes;
  auto* out_bf16 = static_cast<hip_bfloat16*>(out);

  // Native INT8 DUMMA m16n16k32 tiled path for large-M prefill. All three
  // macro-tile instantiations share the exact guard; the launch geometry is
  // selected explicitly so every assigned shape is covered (no
  // single-shape specialization). The guard divisibility (k % 128 == 0,
  // n % 64 == 0) holds for both assigned shapes and for every supported
  // catalog shape.
  if (m >= kPrefillMinM && (n % kDefaultBlockN == 0) &&
      (k % kPrefillStageK == 0)) {
    if (n == 768 && k == 6144) {
      // Packed (k, n) = (6144, 768) weight is transposed [N, K], so every
      // large-M shape with this pair uses the double-buffered transposed-B
      // kernel (the generic [K, N] arms below would misinterpret it). Exact
      // assigned shape (M=4096, N=768, K=6144): 64x128 macro-tile, grid
      // 6x64 = 384 blocks, K stage 64 double-buffered, 30,720 B LDS/block
      // -> 2 blocks/CU.
      const dim3 grid(
          static_cast<unsigned>(n / 128),
          static_cast<unsigned>((m + 63) / 64));
      if (m == 4096) {
        // 4-wave variant w4<64,128,64> (256 threads, 32x64 quadrant per
        // wave, eight accumulators, 96 K stages of 64; 25% fewer LDS
        // fragment reads per block-stage than the 8-wave arm).
        const dim3 block(4 * kWaveSize);
        hipLaunchKernelGGL(
            HIP_KERNEL_NAME(w8a8_gemm_prefill_tiled_kernel_w4<64, 128, 64>),
            grid, block, 0, stream,
            a, b, x_scale, weight_scale, out_bf16, m, n, k);
        return;
      }
      // Other large-M shapes with the packed (k, n) pair keep the generic
      // 8-wave double-buffered <64,128,64,2> kernel (bit-identical path).
      const dim3 block(8 * kWaveSize);
      hipLaunchKernelGGL(
          HIP_KERNEL_NAME(w8a8_gemm_prefill_tiled_kernel<64, 128, 64, 2>),
          grid, block, 0, stream,
          a, b, x_scale, weight_scale, out_bf16, m, n, k);
      return;
    }
    if (n % 128 == 0) {
      // 64x128 tile, 8 wavefronts / 512 threads, single-buffered 128-K stage
      // (generic [K, N] packed-B arm for unmatched (k, n) pairs; the packed
      // (6144, 768) pair never reaches this arm):
      //   down_proj: grid (48, 64) = 3072 blocks x 3 stages
      const dim3 grid(
          static_cast<unsigned>(n / 128),
          static_cast<unsigned>((m + 63) / 64));
      const dim3 block(8 * kWaveSize);
      hipLaunchKernelGGL(
          HIP_KERNEL_NAME(w8a8_dumma_prefill_kernel<64, 128, kPrefillStageK>),
          grid, block, 0, stream,
          a, b, x_scale, weight_scale, out_bf16, m, n, k);
      return;
    }
    if (m % 128 == 0) {
      // 128x64 tile, 8 wavefronts / 512 threads.
      const dim3 grid(
          static_cast<unsigned>(n / 64),
          static_cast<unsigned>(m / 128));
      const dim3 block(8 * kWaveSize);
      hipLaunchKernelGGL(
          HIP_KERNEL_NAME(w8a8_dumma_prefill_kernel<128, 64, kPrefillStageK>),
          grid, block, 0, stream,
          a, b, x_scale, weight_scale, out_bf16, m, n, k);
      return;
    }
    {
      // 64x64 tile, 4 wavefronts / 256 threads (generic default).
      const dim3 grid(
          static_cast<unsigned>(n / 64),
          static_cast<unsigned>((m + 63) / 64));
      const dim3 block(4 * kWaveSize);
      hipLaunchKernelGGL(
          HIP_KERNEL_NAME(w8a8_dumma_prefill_kernel<64, 64, kPrefillStageK>),
          grid, block, 0, stream,
          a, b, x_scale, weight_scale, out_bf16, m, n, k);
      return;
    }
  }

  // Generic scalar fallback for every unmatched (m, n, k) and M < 128
  // (including the paired M=2/M=16 API shapes). The kernel decodes the
  // transposed [N, K] pack for the exact (k, n) = (6144, 768) pair and uses
  // the identity [K, N] column stride n for every other pair.
  const int64_t total = static_cast<int64_t>(m) * n;
  const dim3 grid(static_cast<unsigned>(
      (total + kScalarThreads - 1) / kScalarThreads));
  const dim3 block(static_cast<unsigned>(kScalarThreads));
  hipLaunchKernelGGL(
      w8a8_gemm_scalar_kernel,
      grid, block, 0, stream,
      a, b, x_scale, weight_scale, out_bf16, m, n, k, n);
}

extern "C" void launch_pack_w8a8_weight(
    const int8_t* raw_weight,
    const float* weight_scale,
    int8_t* packed_weight,
    float* packed_weight_scale,
    int k,
    int n,
    hipStream_t stream) {
  // Exact (K, N) = (6144, 768) is stored transposed [N, K]; every other pair
  // keeps the identity [K, N] copy. The scale copy is N-length and
  // order-independent. Runs outside the timed/CUDA-Graph region.
  const int64_t weight_elems = static_cast<int64_t>(k) * n;
  const int64_t total = weight_elems > n ? weight_elems : n;
  const dim3 grid(static_cast<unsigned>(
      (total + kScalarThreads - 1) / kScalarThreads));
  hipLaunchKernelGGL(
      w8a8_pack_kernel,
      grid, dim3(kScalarThreads), 0, stream,
      raw_weight, weight_scale, packed_weight, packed_weight_scale,
      weight_elems, n, k);
}
// @@end
