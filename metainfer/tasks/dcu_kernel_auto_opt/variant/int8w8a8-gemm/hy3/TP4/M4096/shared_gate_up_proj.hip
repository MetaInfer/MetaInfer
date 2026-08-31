// @@variant shape=hy3_tp4_shared_gate_up_proj_m4096 commit=5a1e581a09dc66853a94d300b508d6db6bd6731e added=2026-08-23
//   median_us=220.7 p90_us=221.5 tops=116.8 bandwidth_gb_s=118.9 speedup=30.26 baseline_us=6678
//   source=hy3-tp4-m4096-dsh-test1-bfdf3002
// INT8 W8A8 GEMM HIP implementation for Hygon K500SM_AI / gfx928.
//
// Worker 2 assigned shape: hy3_tp4_shared_gate_up_proj_m4096
//   (M=4096, N=768, K=4096).
//
// DUMMA 2-D macro-tile family (iteration 1, refined iteration 3):
//   * Large-prefill path (M >= 128, N % 64 == 0, K % 128 == 0): native INT8
//     DUMMA m16n16k32 tiled kernel with int32 accumulation. One templated
//     kernel covers the 64x64 (4 waves), 64x128 (8 waves) and 128x64
//     (8 waves) output tiles; every wavefront owns a 32x32 quadrant of four
//     m16n16k32 DUMMA accumulators, so per-wave DUMMA/LDS work is identical
//     across the family and only grid, LDS footprint and occupancy differ.
//     Operands are staged cooperatively with 16-byte vector loads into
//     bank-skewed LDS (A row stride kStageK+16, B row stride kBlockN+16):
//     the +16-byte skew keeps every int4 store aligned while breaking the
//     16-way fragment bank conflicts of the power-of-two strides. No
//     split-K, no raw asm.
//   * The exact assigned shape (4096, 768, 4096) is dispatched to the
//     64x128 tile: 6x64 = 384 blocks, 8 waves/block, grid 384. Iteration 3
//     switches this shape to a double-buffered (software-pipelined) K loop
//     with K stage 64: staging of stage s+1 is issued into registers before
//     the MMA compute of stage s and committed to the second LDS buffer after
//     it, so the global-load round trips overlap the LDS-wait-bound compute
//     phase instead of serializing at the top of every stage (the iteration-1
//     ISA shows each global_load_dwordx4 followed by an immediate
//     s_waitcnt vmcnt(0) before its ds_write). Buffer footprint is 2 x
//     (64x80 A + 64x144 B) = 28,672 B/block -> still 2 blocks/CU -> 16 waves/CU,
//     i.e. the pipeline is added without sacrificing the 16-wave occupancy
//     that the 64x128 tile established. Barrier count stays ~64/block (one
//     prologue + one per stage vs. two per stage in the single-buffer loop).
//     All other routed shapes keep the proven single-buffer K stage 128 loop.
//     Iteration 10 (exact shape only) fixes the dominant LDS cost of this
//     path: the DUMMA int8 B-fragment loader reads 8 k-rows per lane, and
//     with any 16-byte-aligned [K, N] row stride the four k-octets alias onto
//     one LDS bank group (16-way conflict, eight ds_read_u8 per fragment;
//     PMC: 16.5M bank conflicts, 23.15M LDS waits on the stage-64 winner).
//     pack_weight now stores the exact shape's weight transposed [N, K], the
//     double-buffered path stages B in [N, K] LDS order and loads B fragments
//     with the col_major loader, so each B fragment becomes one ds_read2_b32
//     of 8 consecutive k-values (~4-way floor). LDS grows 28,672 -> 30,720
//     B/block, still 2 blocks/CU (16 waves); same MMAC sequence and operand
//     values -> exact int32 accumulation unchanged; grid and barriers
//     unchanged. Generic fallback and all other shapes are untouched.
//     Iteration 13 (exact shape only) removes the residual per-byte fragment
//     reassembly: du_mma_sync passes each int8 fragment to the v_mmac builtin
//     as one packed 8-byte operand, and du_load_matrix_sync assigns x[0..7] =
//     8 separate byte loads, so the compiler emits a 5-op mask/OR chain
//     (v_and 0xff00/0xff0000/0xff000000 + v_or_b32_sdwa + v_or3_b32) for every
//     loaded dword (~50-66 VALU per wave per stage, ~13M of the 16.08M VALU
//     per replay; PMC of the accepted source). The exact-shape k-chunk now
//     writes the same 8 consecutive bytes directly into the fragment storage
//     (same element-to-slot mapping, same byte order), so the v_mmac operand
//     bit patterns and the exact int32 accumulation are unchanged while the
//     reassembly VALU disappears and the LDS->MMA dependency chain shortens.
//     LDS reads stay one ds_read2_b32 per fragment at the 4-way floor; grid,
//     occupancy, barriers and the generic paths are untouched.
//   * Everything else (small M, unmatched shapes): simple scalar int8/int32
//     fallback with exact int32 accumulation.
//   * pack_weight: identity copy for every shape except the exact assigned
//     shape (K=4096, N=768), which is stored transposed [N, K]. The packed
//     layout is opaque per the API contract (prepare_weight may return any
//     contiguous layout); the correctness reference computes from raw
//     weights, and pack runs outside the timed/CUDA-Graph region.
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
// DUMMA accumulators), while the block cooperatively stages A[kBlockM, kStageK]
// and B[kStageK, kBlockN] in bank-skewed LDS. Tail M rows are zero-filled on
// load and masked on store, so any M >= 128 is supported.
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
  // fragment rows across distinct LDS banks (4-way floor for 8-byte/lane
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
      // Iteration 13: load the four int8 fragments as raw 8-byte LDS reads
      // into the fragment storage instead of du_load_matrix_sync's per-byte
      // assignments (same bytes, same element-to-slot mapping), so the
      // compiler feeds the ds_read2_b32 pair straight to the v_mmac and the
      // per-dword mask/OR reassembly VALU disappears from the steady state.
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

      for (int vec = tid; vec < kBVectors; vec += kThreads) {
        const int kk = vec / kBVectorsPerRow;
        const int v = vec - kk * kBVectorsPerRow;
        reinterpret_cast<int4*>(b_tile[0] + kk * kBStride)[v] =
            *reinterpret_cast<const int4*>(
                weight + (k0 + kk) * n + n0 +
                v * static_cast<int>(sizeof(int4)));
      }
      __syncthreads();

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

// Generic scalar fallback: one output element per grid-stride iteration with
// exact int32 accumulation. Correct for any (m, n, k), including the small-M
// API shapes (M=2/16) and any unmatched geometry.
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
  const int64_t total = static_cast<int64_t>(m) * n;
  const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x +
                     threadIdx.x;
       idx < total; idx += stride) {
    const int row = static_cast<int>(idx / n);
    const int col = static_cast<int>(idx - static_cast<int64_t>(row) * n);
    int32_t acc = 0;
    const int8_t* a_row = x_q + static_cast<int64_t>(row) * k;
    for (int kk = 0; kk < k; ++kk) {
      acc += static_cast<int32_t>(a_row[kk]) *
             static_cast<int32_t>(
                 weight[static_cast<int64_t>(kk) * n + col]);
    }
    const float scaled = static_cast<float>(acc) * x_scale[row] *
                         weight_scale[col];
    out[idx] = __float2bfloat16(scaled);
  }
}

// pack_weight. The packed layout is opaque per the API contract; for the
// exact assigned shape (K=4096, N=768) the weight is stored transposed
// [N, K] (n-major) so the double-buffered GEMM stages B in [N, K] LDS order
// and the DUMMA B fragments read 8 consecutive k-values per lane. Every
// other shape keeps the identity [K, N] copy (generic fallback unchanged).
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
  if (k == 4096 && n == 768) {
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
    if (m == 4096 && n == 768 && k == 4096) {
      // Exact assigned shape: 64x128 macro-tile, 8 waves, grid 6x64 = 384.
      // Double-buffered K loop, K stage 64, B staged transposed [N, K] from
      // the transposed packed weight (col_major B fragments): 30,720 B LDS
      // -> 2 blocks/CU -> 16 waves/CU (occupancy preserved), global staging
      // overlapped with the MMA compute, B fragment loads cut from 8 strided
      // ds_read_u8 (16-way conflicts) to one ds_read2_b32 (~4-way) each.
      const dim3 grid(768 / 128, 4096 / 64);
      const dim3 block(8 * kWaveSize);
      hipLaunchKernelGGL(
          HIP_KERNEL_NAME(w8a8_gemm_prefill_tiled_kernel<64, 128, 64, 2>),
          grid, block, 0, stream,
          a, b, x_scale, weight_scale, out_bf16, m, n, k);
      return;
    }
    if (n % 128 == 0) {
      // 64x128 tile, 8 waves; A rows are re-read by only 6 N-blocks.
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
