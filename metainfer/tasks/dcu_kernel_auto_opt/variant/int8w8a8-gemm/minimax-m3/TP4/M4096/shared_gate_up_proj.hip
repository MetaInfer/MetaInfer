// @@variant shape=minimax_tp4_shared_gate_up_proj_m4096 commit=5c8bfa17fe45a29749afdfc9124782e9068e415c added=2026-08-24
//   median_us=986.5 p90_us=987.7
//   source=minimaxm3-dsh-tp4-m4096-1-0c2f84a9
// MetaInfer W8A8 INT8 GEMM backend for Hygon gfx928 (K500SM_AI).
// worker_3 bootstrap (iteration 1): correctness-first native INT8 DUMMA
// m16n16k32 tiled prefill kernels for the assigned MiniMax TP4 M=4096
// shapes:
//   * minimax_tp4_shared_gate_up_proj_m4096: (M=4096, N=1536, K=6144)
//   * minimax_tp4_shared_down_proj_m4096:    (M=4096, N=6144, K=768)
//
// This file provides the two stable host launch symbols consumed by
// csrc/bindings.cpp:
//   * launch_w8a8_gemm(...)        - timed, graph-safe GEMM on the caller's
//                                    HIP stream; no allocation, packing,
//                                    autotuning, or synchronization.
//   * launch_pack_w8a8_weight(...) - bootstrap identity device-to-device
//                                    copy (opaque packed layout = raw
//                                    [K, N] layout; later rounds may change
//                                    only this and the GEMM interpretation).
//
// Strategy (correctness-first bootstrap, then measured tuning):
//   * Large-M prefill (M >= 128): a single templated native INT8 DUMMA
//     m16n16k32 kernel computes a (kBlockM x kBlockN) output tile from
//     (kBlockM/32) x (kBlockN/32) wavefronts, each wave owning one 32x32
//     quadrant (four 16x16 int32 accumulators) resident across the whole K
//     loop. A and B are cooperatively staged into bank-skewed LDS with
//     16-byte vectorized coalesced global loads. The K pipeline is chosen by
//     the kDoubleBuffer template flag:
//       - false: single-buffered LDS, kStageK=128 chunks, two barriers per
//         stage (stage in, compute, buffer-protect). Used by the 128x64 and
//         64x64 arms.
//       - true: double-buffered LDS, kStageK=32 chunks (iteration 6 halved
//         the stage depth 64 -> 32 so the two 32-K LDS buffers fit 4
//         blocks/CU), one barrier per stage, loop-carried int4 register
//         payload prefetching one stage ahead (global loads issued before the
//         MMAC burst, published into the idle buffer before the next stage's
//         MMAs). Used by the 64x128 arm (both assigned shapes). No split-K,
//         no raw asm.
//   * LDS row strides are bank-phased for the DUMMA fragment loads: A rows
//     keep a 16-byte pad (stride kStageK+16: 144 for 128-K stages, 80 for
//     64-K stages, 48 for 32-K stages) so the A-fragment ds_read2_b32 spread
//     over all 32 banks
//     (2-way), while B rows use a 4-byte pad (stride kBlockN+4 = 132)
//     because any 16-byte-aligned B stride collapses the four k-groups of
//     every B-fragment byte load onto the same bank phase (4 banks, 16-way;
//     stride 132 moves the groups to phases {0,8,16,24}, 16 banks, 4-way).
//   * Three macro-tiles are instantiated (64x64, 64x128, 128x64) and the
//     active one is selected at launch time by explicit geometry dispatch:
//       - N % 128 == 0      -> 64x128 tile (8 wavefronts, 512 threads)
//       - else M % 128 == 0 -> 128x64 tile (8 wavefronts, 512 threads)
//       - else              -> 64x64 tile  (4 wavefronts, 256 threads)
//     Both assigned shapes route to the 64x128 tile (double-buffered 64-K
//     stages since iteration 3, halved to 32-K stages in iteration 6):
//       - gate_up  : grid (12, 64) = 768 blocks, 192 K stages, 6144 waves
//       - down_proj: grid (48, 64) = 3072 blocks, 24 K stages, 24576 waves
//     The M x N output-tile grid already dwarfs the 120 CUs, so no split-K.
//   * Tail M rows are zero-filled on A load and masked on store, so any
//     M >= 128 is correct; M < 128 (decode API shapes, e.g. M=2/M=16)
//     reaches the scalar fallback below.
//   * Everything else (M < 128 or unmatched geometry): scalar int8/int32
//     fallback with exact int32 accumulation in k order.
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
// Stage depth of the double-buffered arm (iteration 6: halved 64 -> 32 so
// the two 32-K LDS buffers (a_tile[2][64x48] + b_tile[2][32x132] = 14,592
// B/block) fit 4 blocks/CU (58,368 B <= 65,536 B) -> up to 32 resident
// waves/CU vs 16 for the 64-K depth (27,136 B/block -> 2 blocks/CU). Staged
// bytes per MAC, (1/kBlockM + 1/kBlockN), are independent of the stage depth,
// so total global loads and LDS traffic stay bit-identical to the 64-K
// champion; the change is purely an occupancy/latency-hiding experiment
// against the measured LDS-latency-bound profile (1.65 lds_wait per LDS
// instruction). A naive 128-K double buffer (52,224 B/block -> 1 block/CU)
// remains rejected.
constexpr int kPrefillDoubleStageK = 32;
constexpr int kDefaultBlockM = 64;
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

// Load one flat staging vector (index v over the A-then-B region) of the K
// stage starting at k0 into a register slot. Tail-M A rows are zero-filled so
// the DUMMA math stays defined; B rows are contiguous kBlockN chunks at global
// row stride k (bootstrap identity [K, N] packed layout). Used only by the
// double-buffered pipeline's loop-carried payload.
template <int kBlockM, int kStageK, int kBlockN>
__device__ __forceinline__ int4 prefill_load_stage_vector(
    const int8_t* __restrict__ x_q,
    const int8_t* __restrict__ weight,
    int v,
    int k0,
    int m,
    int k,
    int n,
    int m0,
    int n0) {
  constexpr int kAVectorsPerRow = kStageK / static_cast<int>(sizeof(int4));
  constexpr int kAVectors = kBlockM * kAVectorsPerRow;
  constexpr int kBVectorsPerRow = kBlockN / static_cast<int>(sizeof(int4));
  if (v < kAVectors) {
    const int local_row = v / kAVectorsPerRow;
    const int vv = v - local_row * kAVectorsPerRow;
    const int global_row = m0 + local_row;
    return global_row < m
               ? *reinterpret_cast<const int4*>(
                     x_q + global_row * k + k0 +
                     vv * static_cast<int>(sizeof(int4)))
               : int4{0, 0, 0, 0};
  }
  const int bv = v - kAVectors;
  const int kk = bv / kBVectorsPerRow;
  const int vv = bv - kk * kBVectorsPerRow;
  return *reinterpret_cast<const int4*>(
      weight + (k0 + kk) * n + n0 +
      vv * static_cast<int>(sizeof(int4)));
}

// Publish one flat staging vector into the given LDS buffers. A rows keep
// 16-byte-aligned int4 stores at row stride kAStride; B rows are committed as
// four int32 stores at row stride kBStride (4-byte pad, see kBLdsPad).
template <int kBlockM, int kStageK, int kBlockN, int kAStride, int kBStride>
__device__ __forceinline__ void prefill_publish_stage_vector(
    int8_t* __restrict__ a_tile,
    int8_t* __restrict__ b_tile,
    int v,
    int4 val) {
  constexpr int kAVectorsPerRow = kStageK / static_cast<int>(sizeof(int4));
  constexpr int kAVectors = kBlockM * kAVectorsPerRow;
  constexpr int kBVectorsPerRow = kBlockN / static_cast<int>(sizeof(int4));
  if (v < kAVectors) {
    const int local_row = v / kAVectorsPerRow;
    const int vv = v - local_row * kAVectorsPerRow;
    reinterpret_cast<int4*>(a_tile + local_row * kAStride)[vv] = val;
  } else {
    const int bv = v - kAVectors;
    const int kk = bv / kBVectorsPerRow;
    const int vv = bv - kk * kBVectorsPerRow;
    int32_t* b_row = reinterpret_cast<int32_t*>(
        b_tile + kk * kBStride + vv * static_cast<int>(sizeof(int4)));
    b_row[0] = val.x;
    b_row[1] = val.y;
    b_row[2] = val.z;
    b_row[3] = val.w;
  }
}

// Large-M prefill path: one block computes a kBlockM x kBlockN output tile.
// kWaveRows x kWaveCols wavefronts each own a 32x32 quadrant (four 16x16
// DUMMA accumulators), while the block cooperatively stages
// A[kBlockM, kStageK] and B[kStageK, kBlockN] in bank-skewed LDS. With
// kDoubleBuffer=false the LDS is single-buffered with two barriers per stage
// (stage in, compute, protect); with kDoubleBuffer=true two LDS buffers
// alternate across the stage with a loop-carried int4 register payload and
// one barrier per stage (see the branch for the pipeline). Tail M rows are
// zero-filled on load and masked on store, so any M >= 128 is supported. The
// packed weight is the bootstrap identity [K, N] layout, so B rows are staged
// n-major contiguous chunks.
template <int kBlockM, int kBlockN, int kStageK, bool kDoubleBuffer>
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
  constexpr int kStageVectors = kAVectors + kBVectors;

  const int tid = static_cast<int>(threadIdx.x);
  const int wave = tid / kWaveSize;
  const int lane = tid % kWaveSize;
  const int wave_row = wave / kWaveCols;
  const int wave_col = wave - wave_row * kWaveCols;
  const int m0 = static_cast<int>(blockIdx.y) * kBlockM;
  const int n0 = static_cast<int>(blockIdx.x) * kBlockN;

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

  if constexpr (kDoubleBuffer) {
    // Double-buffered K pipeline (one barrier per stage):
    //   * two LDS buffers; stage s lives in buffer s & 1;
    //   * a loop-carried int4 payload prefetches one stage ahead: the global
    //     loads for stage s+2 are issued before stage s's MMAC burst, and
    //     stage s+1's payload is published into the idle buffer (vmcnt wait
    //     implicit) before stage s+1's MMAs. The stage-top vmem wait and the
    //     staging LDS-write burst no longer sit on the barrier's critical
    //     path the way the single-buffer stage-in barrier forces.
    static_assert(kStageVectors <= 2 * kThreads,
                  "double-buffered staging needs at most 2 payload slots");
    constexpr int kPayloadSlots = (kStageVectors + kThreads - 1) / kThreads;
    __shared__ __align__(16) int8_t a_tile[2][kBlockM * kAStride];
    __shared__ __align__(16) int8_t b_tile[2][kStageK * kBStride];

    const int nstages = k / kStageK;
    int4 payload[kPayloadSlots];

    // Prologue: stage 0 is staged straight into LDS buffer 0, then the
    // global prefetch of stage 1 is issued into the payload so its latency
    // overlaps the prologue barrier and stage 0's compute.
#pragma unroll
    for (int p = 0; p < kPayloadSlots; ++p) {
      const int v = tid + p * kThreads;
      if (v < kStageVectors) {
        prefill_publish_stage_vector<kBlockM, kStageK, kBlockN, kAStride,
                                     kBStride>(
            a_tile[0], b_tile[0], v,
            prefill_load_stage_vector<kBlockM, kStageK, kBlockN>(
                x_q, weight, v, 0, m, k, n, m0, n0));
      }
    }
    if (nstages > 1) {
#pragma unroll
      for (int p = 0; p < kPayloadSlots; ++p) {
        const int v = tid + p * kThreads;
        if (v < kStageVectors) {
          payload[p] = prefill_load_stage_vector<kBlockM, kStageK, kBlockN>(
              x_q, weight, v, kStageK, m, k, n, m0, n0);
        }
      }
    }
    __syncthreads();

    for (int s = 0; s < nstages; ++s) {
      const int buf = s & 1;
      // Publish the prefetched stage s+1 into the idle buffer before this
      // stage's MMAC burst. Buffer (s+1)&1 was last read by stage s-1's MMAs,
      // which the previous barrier ordered ahead of this publish. The barrier
      // at the bottom of the iteration then (a) makes the published stage s+1
      // visible to every thread before iteration s+1's MMAs, and (b) keeps all
      // reads of buffer buf ahead of the stage s+2 publish (same-parity
      // buffer) in iteration s+1.
      if (s + 1 < nstages) {
        const int nbuf = buf ^ 1;
#pragma unroll
        for (int p = 0; p < kPayloadSlots; ++p) {
          const int v = tid + p * kThreads;
          if (v < kStageVectors) {
            prefill_publish_stage_vector<kBlockM, kStageK, kBlockN, kAStride,
                                         kBStride>(a_tile[nbuf], b_tile[nbuf],
                                                   v, payload[p]);
          }
        }
        // Issue the global prefetch of stage s+2 (latency hidden behind this
        // stage's MMAC burst and the barrier).
        if (s + 2 < nstages) {
#pragma unroll
          for (int p = 0; p < kPayloadSlots; ++p) {
            const int v = tid + p * kThreads;
            if (v < kStageVectors) {
              payload[p] = prefill_load_stage_vector<kBlockM, kStageK, kBlockN>(
                  x_q, weight, v, (s + 2) * kStageK, m, k, n, m0, n0);
            }
          }
        }
      }

      // MMAC burst on buffer buf. Each wave owns a 32x32 quadrant: two A
      // fragments (rows 0-15 / 16-31) and two B fragments (cols 0-15 / 16-31)
      // per kTileK step, four du_mma_sync per step. The per-kk sequence
      // (k0-outer, kk-inner) is identical to the single-buffered arm, so the
      // int32 accumulation order stays bit-identical.
      const int local_row = wave_row * 32;
      const int local_col = wave_col * 32;
#pragma unroll
      for (int kk = 0; kk < kStageK; kk += kTileK) {
        du_load_matrix_sync(
            a_frag0, a_tile[buf] + local_row * kAStride + kk, kAStride);
        du_load_matrix_sync(
            a_frag1, a_tile[buf] + (local_row + kTileM) * kAStride + kk,
            kAStride);
        du_load_matrix_sync(
            b_frag0, b_tile[buf] + kk * kBStride + local_col, kBStride);
        du_load_matrix_sync(
            b_frag1, b_tile[buf] + kk * kBStride + local_col + kTileN,
            kBStride);
        du_mma_sync(acc00, a_frag0, b_frag0, acc00);
        du_mma_sync(acc01, a_frag0, b_frag1, acc01);
        du_mma_sync(acc10, a_frag1, b_frag0, acc10);
        du_mma_sync(acc11, a_frag1, b_frag1, acc11);
      }
      __syncthreads();
    }
  } else {
    __shared__ __align__(16) int8_t a_tile[kBlockM * kAStride];
    __shared__ __align__(16) int8_t b_tile[kStageK * kBStride];

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
      // (cols 0-15 / 16-31) per step, four du_mma_sync per step.
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

// Generic scalar fallback: one output element per thread with exact int32
// accumulation in k order. Correct for any (m, n, k), including the small-M
// API shapes (M=2/M=16) and any unmatched geometry. The packed weight is the
// bootstrap identity [K, N] layout (column stride n).
__global__ __launch_bounds__(kScalarThreads) void w8a8_gemm_scalar_kernel(
    const int8_t* __restrict__ x_q,
    const int8_t* __restrict__ weight,
    const float* __restrict__ x_scale,
    const float* __restrict__ weight_scale,
    hip_bfloat16* __restrict__ out,
    int m,
    int n,
    int k) {
  const int64_t total = static_cast<int64_t>(m) * n;
  const int64_t idx =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= total) {
    return;
  }
  const int row = static_cast<int>(idx / n);
  const int col = static_cast<int>(idx - static_cast<int64_t>(row) * n);
  const int8_t* a_row = x_q + static_cast<int64_t>(row) * k;
  const int8_t* b_col = weight + static_cast<int64_t>(col);
  int32_t acc = 0;
  for (int kk = 0; kk < k; ++kk) {
    acc += static_cast<int32_t>(a_row[kk]) *
           static_cast<int32_t>(b_col[static_cast<int64_t>(kk) * n]);
  }
  const float scaled =
      static_cast<float>(acc) * x_scale[row] * weight_scale[col];
  out[idx] = __float2bfloat16(scaled);
}

// Bootstrap pack: identity device-to-device copies. The packed layout is
// opaque per the API contract; until a later round changes the pack layout
// (and the matching GEMM interpretation together), packed == raw [K, N].
__global__ __launch_bounds__(kScalarThreads) void w8a8_identity_copy_i8_kernel(
    const int8_t* __restrict__ src,
    int8_t* __restrict__ dst,
    int64_t numel) {
  const int64_t idx =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < numel) {
    dst[idx] = src[idx];
  }
}

__global__ __launch_bounds__(kScalarThreads) void
w8a8_identity_copy_f32_kernel(
    const float* __restrict__ src,
    float* __restrict__ dst,
    int64_t numel) {
  const int64_t idx =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < numel) {
    dst[idx] = src[idx];
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
  // Bootstrap uses no split-K, so the workspace is not touched. The timed
  // operator performs no allocation, synchronization, packing, or
  // default-stream launch; it writes only the caller-provided out.
  (void)workspace;
  (void)workspace_bytes;
  auto* out_bf16 = static_cast<hip_bfloat16*>(out);

  // Native INT8 DUMMA m16n16k32 tiled path for large-M prefill. All three
  // macro-tile instantiations share the exact guard; the launch geometry is
  // selected explicitly so every assigned shape is covered (no
  // single-shape specialization). The guard divisibility (k % 128 == 0,
  // n % 64 == 0) holds for all supported catalog shapes.
  if (m >= kPrefillMinM && (n % kDefaultBlockN == 0) &&
      (k % kPrefillStageK == 0)) {
    if (n % 128 == 0) {
      // 64x128 tile, 8 wavefronts / 512 threads, double-buffered 32-K stage.
      // Both assigned shapes: gate_up grid (12, 64) = 768 blocks x 192 stages,
      // down_proj grid (48, 64) = 3072 blocks x 24 stages.
      const dim3 grid(
          static_cast<unsigned>(n / 128),
          static_cast<unsigned>((m + 63) / 64));
      const dim3 block(8 * kWaveSize);
      hipLaunchKernelGGL(
          HIP_KERNEL_NAME(w8a8_dumma_prefill_kernel<64, 128,
                                                    kPrefillDoubleStageK, true>),
          grid, block, 0, stream,
          a, b, x_scale, weight_scale, out_bf16, m, n, k);
      return;
    }
    if (m % 128 == 0) {
      // 128x64 tile, 8 wavefronts / 512 threads (single-buffered arm).
      const dim3 grid(
          static_cast<unsigned>(n / 64),
          static_cast<unsigned>(m / 128));
      const dim3 block(8 * kWaveSize);
      hipLaunchKernelGGL(
          HIP_KERNEL_NAME(
              w8a8_dumma_prefill_kernel<128, 64, kPrefillStageK, false>),
          grid, block, 0, stream,
          a, b, x_scale, weight_scale, out_bf16, m, n, k);
      return;
    }
    {
      // 64x64 tile, 4 wavefronts / 256 threads (generic default,
      // single-buffered arm).
      const dim3 grid(
          static_cast<unsigned>(n / 64),
          static_cast<unsigned>((m + 63) / 64));
      const dim3 block(4 * kWaveSize);
      hipLaunchKernelGGL(
          HIP_KERNEL_NAME(
              w8a8_dumma_prefill_kernel<64, 64, kPrefillStageK, false>),
          grid, block, 0, stream,
          a, b, x_scale, weight_scale, out_bf16, m, n, k);
      return;
    }
  }

  // Generic scalar fallback for every unmatched (m, n, k) and M < 128
  // (including the paired M=2/M=16 API shapes).
  const int64_t total = static_cast<int64_t>(m) * n;
  const dim3 grid(static_cast<unsigned>(
      (total + kScalarThreads - 1) / kScalarThreads));
  const dim3 block(static_cast<unsigned>(kScalarThreads));
  hipLaunchKernelGGL(
      w8a8_gemm_scalar_kernel,
      grid, block, 0, stream,
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
  // Bootstrap: identity device-to-device copy for every (k, n). The scale
  // copy is unchanged in later rounds (N-length, order-independent).
  const int64_t weight_numel = static_cast<int64_t>(k) * n;
  const dim3 weight_grid(static_cast<unsigned>(
      (weight_numel + kScalarThreads - 1) / kScalarThreads));
  hipLaunchKernelGGL(
      w8a8_identity_copy_i8_kernel,
      weight_grid, dim3(kScalarThreads), 0, stream,
      raw_weight, packed_weight, weight_numel);

  const dim3 scale_grid(static_cast<unsigned>(
      (n + kScalarThreads - 1) / kScalarThreads));
  hipLaunchKernelGGL(
      w8a8_identity_copy_f32_kernel,
      scale_grid, dim3(kScalarThreads), 0, stream,
      weight_scale, packed_weight_scale, static_cast<int64_t>(n));
}
// @@end
