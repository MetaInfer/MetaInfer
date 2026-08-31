// @@variant shape=hy3_tp4_o_proj_m4096 commit=7c36244e756baf1e8268807cb6481eb118a30133 added=2026-08-23
//   median_us=803.9 p90_us=805.1 tops=85.48 bandwidth_gb_s=62.65 speedup=24.73 baseline_us=1.988e+04
//   source=hy3-tp4-m4096-dsh-test1-bfdf3002
// INT8 W8A8 GEMM HIP implementation for Hygon K500SM_AI / gfx928.
//
// Worker: worker_1 (physical GPU 1), assigned shape:
//   hy3_tp4_o_proj_m4096 : M=4096, N=4096, K=2048
//
// Iteration 1 - DUMMA throughput baseline with a 2-D macro-tile (128x64).
//   * Large-prefill path (exact (m, n, k) == (4096, 4096, 2048)):
//       native INT8 DUMMA m16n16k32 with int32 accumulation; one 128x64
//       output tile per block; four wavefronts (256 threads); each wave owns
//       a 64x32 quadrant built from eight m16n16k32 fragments; block
//       cooperatively vector-loads A[128,64] and packed B panels into 26 KiB
//       of LDS with a padded 80-byte B row stride; 64-deep K double buffering
//       with a register prefetch of the next stage (one loop-carried barrier
//       per 64-K stage); fused dot * x_scale[m] * weight_scale[n] epilogue
//       stored directly as bf16 from the accumulator fragments.
//       Rationale vs the bootstrap 64x64x128 single-buffer tile (verified on
//       the exact shape in the trusted ledger): 128x64 doubles B reuse across
//       rows, halves the grid to 2048 blocks, stages each B stage as one
//       coalesced 4 KiB stream from pre-packed [N/64, K, 64] panels, breaks
//       the 64-byte LDS bank periodicity with an 80-byte stride, and hides
//       global->LDS latency with double buffering.
//   * launch_pack_w8a8_weight packs exactly (k, n) == (2048, 4096) into
//       contiguous [N/64, K, 64] int8 panels; every other (k, n) keeps the
//       identity copy. Packing never happens inside the timed GEMM.
//   * Generic scalar int8/int32 fallback for every unmatched (m, n, k),
//       including the paired M=2 decode shape with the same (K, N); the
//       fallback reads the packed panel layout whenever (k, n) == (2048, 4096)
//       and the identity layout otherwise.
//
// Iteration 3 - A-tile LDS bank-conflict fix:
//   PMC on the accepted iteration-1 source: 77,594,624 LDS bank conflicts
//   (~6.88 per LDS instruction). Root cause in the fragment loads: the A
//   tile still uses an unpadded 64-byte row stride (16 banks/row), so the
//   bank pattern repeats every two rows and an m16n16k32 A-fragment load
//   (64 lanes x 8 B over 16 rows) puts eight lanes on the same bank pair
//   (8-way conflicts), while the already padded 80-byte B rows (20
//   banks/row, period 8) only reach 4-way over their 32-row span - matching
//   the PMC mix (8 A-loads x 8-way + 4 B-loads x 4-way per 64-K stage).
//   Focused change: pad the A LDS rows 64 -> 80 bytes (five bank phases,
//   same as B) so each A-fragment load drops from 8-way to 2-way bank
//   conflicts. LDS grows 26,624 -> 30,720 B and stays at 2 blocks/CU
//   (61,440 B of the 64 KiB LDS budget); 80 is a multiple of 16 so every
//   cooperative int4 stage store stays aligned; the packed [N/64, K, 64]
//   global layout, exact int32 accumulation, grid, and fallback are
//   untouched.
//
// Iteration 10 - stage-boundary restructure (early publish, one full stage of
//   write->read distance):
//   Rounds 7 (in-loop A-fragment prefetch, 845.0 us) and 8 (occupancy probe,
//   866.6 us) falsified the in-kk LDS latency and wave-count theories and
//   indicted the structural stage-start chain: [last MMACs] -> [s_waitcnt
//   vmcnt for the prefetch loads] -> [3x ds_write_b128] -> [s_waitcnt
//   lgkmcnt(0)] -> [s_barrier] -> [stage-start fragment loads] (ISA 0x3BC0-
//   0x3BF4 on the accepted source). The next-stage LDS publish currently
//   happens AFTER the MMAC burst, so the store issue+retire latency and the
//   fresh-write visibility sit directly between the barrier and the next
//   stage's first reads. Focused change: the loop-carried int4 payload for
//   stage k0+kStageK is published into the idle ping-pong buffer at the TOP
//   of stage k0 (before the burst) and the global prefetch for stage
//   k0+2*kStageK now has a full extra stage of lead, so the store latency
//   retires under the burst and the stage-end barrier publishes data that has
//   been resident in LDS for a full stage before the next stage reads it.
//   One barrier per stage, 30,720 B LDS (2 blocks/CU), grid, packed layout,
//   exact int32 accumulation order, shape guards, and the scalar fallback are
//   unchanged.

#include <hip/hip_runtime.h>
#include <hip/hip_bfloat16.h>
#include <du_mma.h>

#include <cstdint>

namespace {

using namespace du::dumma;

constexpr int kTileM = 16;
constexpr int kTileN = 16;
constexpr int kTileK = 32;  // gfx928 INT8 DUMMA primitive: m16n16k32
constexpr int kWaveSize = 64;  // gfx928 native wavefront
constexpr int kTargetM = 4096;
constexpr int kTargetN = 4096;
constexpr int kTargetK = 2048;
constexpr int kBlockM = 128;
constexpr int kBlockN = 64;
constexpr int kStageK = 64;
constexpr int kBPad = 16;  // 64 -> 80-byte LDS row stride (five bank phases)
constexpr int kBStride = kBlockN + kBPad;
constexpr int kAStride = kStageK + kBPad;  // 64 -> 80-byte A row stride
constexpr int kBlockThreads = 4 * kWaveSize;

using bf16_t = hip_bfloat16;

// ---------------------------------------------------------------------------
// Large-M prefill: 128x64 output tile per block, four wavefronts of 64 lanes.
// Each wave owns a 64x32 quadrant (eight m16n16k32 accumulators); the block
// cooperatively stages A[128,64] and packed-B [64,64] panels in two ping-pong
// buffers of 64-K stages (26 KiB total LDS). One barrier per stage.
// ---------------------------------------------------------------------------
template <typename AccFragment>
__device__ __forceinline__ void store_prefill_fragment(
    const AccFragment& frag,
    const float* __restrict__ x_scale,
    const float* __restrict__ weight_scale,
    bf16_t* __restrict__ out,
    int base_row,
    int base_col,
    int m,
    int n,
    int lane) {
  // Verified gfx928 int8 m16n16k32 accumulator ownership (matches
  // du_store_matrix_sync): lane & 15 selects the row, lane >> 4 selects
  // col % 4, and x[i] maps to columns col%4 + 4*i.
  const int row = base_row + (lane & 15);
  if (row >= m) {
    return;  // tail-M masking: padded rows never write
  }
  const int col_mod4 = lane >> 4;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const int col = base_col + col_mod4 + 4 * i;
    const float scaled = static_cast<float>(frag.x[i]) *
                         x_scale[row] * weight_scale[col];
    out[static_cast<int64_t>(row) * n + col] = __float2bfloat16(scaled);
  }
}

__global__ __launch_bounds__(kBlockThreads) void
w8a8_dumma_128x64x64_packed_kernel(
    const int8_t* __restrict__ x_q,
    const int8_t* __restrict__ weight,
    const float* __restrict__ x_scale,
    const float* __restrict__ weight_scale,
    bf16_t* __restrict__ out,
    int m,
    int n,
    int k) {
  const int tid = static_cast<int>(threadIdx.x);
  const int wave = tid / kWaveSize;
  const int lane = tid % kWaveSize;
  const int wave_row = wave >> 1;
  const int wave_col = wave & 1;
  const int m0 = static_cast<int>(blockIdx.y) * kBlockM;
  const int n0 = static_cast<int>(blockIdx.x) * kBlockN;
  // Packed layout: panel p = B[:, p*64:(p+1)*64] stored as [K][64] rows.
  const int b_panel_offset =
      static_cast<int>(blockIdx.x) * kTargetK * kBlockN;
  const int local_row = wave_row * 64;
  const int local_col = wave_col * 32;

  __shared__ __align__(16) int8_t a_tile[2][kBlockM * kAStride];
  __shared__ __align__(16) int8_t b_tile[2][kStageK * kBStride];

  DUFragment<matrix_a, kTileM, kTileN, kTileK, signed char, row_major>
      a_frag0, a_frag1;
  DUFragment<matrix_b, kTileM, kTileN, kTileK, signed char, row_major>
      b_frag0, b_frag1;
  DUFragment<accumulator, kTileM, kTileN, kTileK, int>
      acc00, acc01, acc10, acc11, acc20, acc21, acc30, acc31;
  du_fill_fragment(acc00, 0);
  du_fill_fragment(acc01, 0);
  du_fill_fragment(acc10, 0);
  du_fill_fragment(acc11, 0);
  du_fill_fragment(acc20, 0);
  du_fill_fragment(acc21, 0);
  du_fill_fragment(acc30, 0);
  du_fill_fragment(acc31, 0);

  // Prologue: cooperative vectorized staging of stage 0. Each thread owns one
  // int4 in A rows [0,64) and one in A rows [64,128), plus one int4 of packed
  // B. A[128,64] is 512 int4s; B[64,64] is 256 int4s.
  const int vector_byte_offset = tid * sizeof(int4);
  const int stage_row = vector_byte_offset / kStageK;
  const int stage_col = vector_byte_offset - stage_row * kStageK;
  *reinterpret_cast<int4*>(a_tile[0] + stage_row * kAStride + stage_col) =
      *reinterpret_cast<const int4*>(
          x_q + static_cast<int64_t>(m0 + stage_row) * k + stage_col);
  *reinterpret_cast<int4*>(a_tile[0] +
                           (stage_row + kBlockM / 2) * kAStride + stage_col) =
      *reinterpret_cast<const int4*>(
          x_q + static_cast<int64_t>(m0 + stage_row + kBlockM / 2) * k +
          stage_col);
  *reinterpret_cast<int4*>(b_tile[0] + stage_row * kBStride + stage_col) =
      *reinterpret_cast<const int4*>(
          weight + b_panel_offset + stage_row * kBlockN + stage_col);
  __syncthreads();

  // Loop-carried register payload for the stage after the one being computed
  // (stage k0 + kStageK): loaded one full stage ahead and published into the
  // idle ping-pong buffer at the TOP of the following stage. The publish's
  // LDS store latency then retires under the MMAC burst, and the stage-end
  // barrier publishes it a full stage before the next stage's first fragment
  // loads - so the structural LDS write -> barrier -> read chain leaves the
  // stage boundary (rounds 7/8 falsified in-loop A-fragment prefetch and
  // occupancy; this is the prescribed stage-boundary restructure).
  int4 cur_a0{};
  int4 cur_a1{};
  int4 cur_b{};
  if (k > kStageK) {
    cur_a0 = *reinterpret_cast<const int4*>(
        x_q + static_cast<int64_t>(m0 + stage_row) * k + kStageK + stage_col);
    cur_a1 = *reinterpret_cast<const int4*>(
        x_q + static_cast<int64_t>(m0 + stage_row + kBlockM / 2) * k +
        kStageK + stage_col);
    cur_b = *reinterpret_cast<const int4*>(
        weight + b_panel_offset + (kStageK + stage_row) * kBlockN + stage_col);
  }

  int current = 0;
  for (int k0 = 0; k0 < k; k0 += kStageK) {
    const int next_k = k0 + kStageK;
    const int next = current ^ 1;
    // Publish the loop-carried payload (stage next_k) into the idle buffer
    // BEFORE the MMAC burst; the stage-end barrier below publishes it a full
    // stage before it is read, hiding the store latency under the burst.
    if (next_k < k) {
      *reinterpret_cast<int4*>(a_tile[next] + stage_row * kAStride +
                               stage_col) = cur_a0;
      *reinterpret_cast<int4*>(a_tile[next] +
                               (stage_row + kBlockM / 2) * kAStride +
                               stage_col) = cur_a1;
      *reinterpret_cast<int4*>(
          b_tile[next] + stage_row * kBStride + stage_col) = cur_b;
    }
    // Prefetch stage k0 + 2*kStageK into the payload registers (a full stage
    // of extra global->LDS lead) so the publish never waits on the loads.
    const int load_k = k0 + 2 * kStageK;
    if (load_k < k) {
      cur_a0 = *reinterpret_cast<const int4*>(
          x_q + static_cast<int64_t>(m0 + stage_row) * k + load_k + stage_col);
      cur_a1 = *reinterpret_cast<const int4*>(
          x_q + static_cast<int64_t>(m0 + stage_row + kBlockM / 2) * k +
          load_k + stage_col);
      cur_b = *reinterpret_cast<const int4*>(
          weight + b_panel_offset + (load_k + stage_row) * kBlockN +
          stage_col);
    }

    // Each wave consumes its 64x32 quadrant: eight m16n16k32 MMACs per kk.
#pragma unroll
    for (int kk = 0; kk < kStageK; kk += kTileK) {
      du_load_matrix_sync(
          b_frag0, b_tile[current] + kk * kBStride + local_col, kBStride);
      du_load_matrix_sync(
          b_frag1, b_tile[current] + kk * kBStride + local_col + kTileN,
          kBStride);
      du_load_matrix_sync(
          a_frag0, a_tile[current] + local_row * kAStride + kk, kAStride);
      du_load_matrix_sync(
          a_frag1, a_tile[current] + (local_row + kTileM) * kAStride + kk,
          kAStride);
      du_mma_sync(acc00, a_frag0, b_frag0, acc00);
      du_mma_sync(acc01, a_frag0, b_frag1, acc01);
      du_mma_sync(acc10, a_frag1, b_frag0, acc10);
      du_mma_sync(acc11, a_frag1, b_frag1, acc11);
      du_load_matrix_sync(
          a_frag0, a_tile[current] + (local_row + 2 * kTileM) * kAStride + kk,
          kAStride);
      du_load_matrix_sync(
          a_frag1, a_tile[current] + (local_row + 3 * kTileM) * kAStride + kk,
          kAStride);
      du_mma_sync(acc20, a_frag0, b_frag0, acc20);
      du_mma_sync(acc21, a_frag0, b_frag1, acc21);
      du_mma_sync(acc30, a_frag1, b_frag0, acc30);
      du_mma_sync(acc31, a_frag1, b_frag1, acc31);
    }

    if (next_k < k) {
      // The stage-end barrier publishes the payload written at the top of
      // this stage (a full stage before the next stage reads it) and protects
      // buffer `current` from the publish at the top of the next stage.
      __syncthreads();
      current = next;
    }
  }

  const int base_row = m0 + local_row;
  const int base_col = n0 + local_col;
  store_prefill_fragment(acc00, x_scale, weight_scale, out,
                         base_row, base_col, m, n, lane);
  store_prefill_fragment(acc01, x_scale, weight_scale, out,
                         base_row, base_col + kTileN, m, n, lane);
  store_prefill_fragment(acc10, x_scale, weight_scale, out,
                         base_row + kTileM, base_col, m, n, lane);
  store_prefill_fragment(acc11, x_scale, weight_scale, out,
                         base_row + kTileM, base_col + kTileN, m, n, lane);
  store_prefill_fragment(acc20, x_scale, weight_scale, out,
                         base_row + 2 * kTileM, base_col, m, n, lane);
  store_prefill_fragment(acc21, x_scale, weight_scale, out,
                         base_row + 2 * kTileM, base_col + kTileN, m, n, lane);
  store_prefill_fragment(acc30, x_scale, weight_scale, out,
                         base_row + 3 * kTileM, base_col, m, n, lane);
  store_prefill_fragment(acc31, x_scale, weight_scale, out,
                         base_row + 3 * kTileM, base_col + kTileN, m, n, lane);
}

// ---------------------------------------------------------------------------
// Generic scalar fallback: one thread per output element. Exact int32 dot
// over K, then fused x_scale * weight_scale, then bf16 store. Handles every
// unmatched (m, n, k), including all small-M API cases. When (k, n) matches
// the packed pair (2048, 4096) the weight is read from the [N/64, K, 64]
// panel layout; otherwise from the identity layout.
// ---------------------------------------------------------------------------
__global__ __launch_bounds__(256) void w8a8_gemm_scalar_fallback_kernel(
    const int8_t* __restrict__ x_q,
    const int8_t* __restrict__ weight,
    const float* __restrict__ x_scale,
    const float* __restrict__ weight_scale,
    bf16_t* __restrict__ out,
    int m,
    int n,
    int k) {
  const int64_t linear =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total = static_cast<int64_t>(m) * n;
  if (linear >= total) {
    return;
  }
  const int row = static_cast<int>(linear / n);
  const int col = static_cast<int>(linear - static_cast<int64_t>(row) * n);
  int32_t acc = 0;
  if (n == kTargetN && k == kTargetK) {
    const int panel = col / kBlockN;
    const int panel_col = col - panel * kBlockN;
    const int64_t panel_offset =
        static_cast<int64_t>(panel) * kTargetK * kBlockN;
    for (int kk = 0; kk < k; ++kk) {
      acc += static_cast<int32_t>(x_q[static_cast<int64_t>(row) * k + kk]) *
             static_cast<int32_t>(weight[panel_offset +
                                         static_cast<int64_t>(kk) * kBlockN +
                                         panel_col]);
    }
  } else {
    const int8_t* a_row = x_q + static_cast<int64_t>(row) * k;
    const int8_t* b_col = weight + col;
    for (int kk = 0; kk < k; ++kk) {
      acc += static_cast<int32_t>(a_row[kk]) *
             static_cast<int32_t>(b_col[static_cast<int64_t>(kk) * n]);
    }
  }
  const float scaled = static_cast<float>(acc) *
                       x_scale[row] * weight_scale[col];
  out[linear] = __float2bfloat16(scaled);
}

// ---------------------------------------------------------------------------
// Weight packing: exact (k, n) == (2048, 4096) becomes contiguous
// [N/64, K, 64] int8 panels (each int4 stays within one 64-column row, so
// loads are coalesced 4 KiB streams per 64-K stage); identity copy for any
// other (k, n). Scales are always copied identity.
// ---------------------------------------------------------------------------
__global__ __launch_bounds__(256) void w8a8_pack_64n_panel_kernel(
    const int8_t* __restrict__ raw_weight,
    int8_t* __restrict__ packed_weight) {
  const int vec = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
  constexpr int kVectors =
      kTargetK * kTargetN / static_cast<int>(sizeof(int4));
  if (vec >= kVectors) {
    return;
  }
  constexpr int kPanelBytes = kTargetK * kBlockN;
  const int byte_offset = vec * sizeof(int4);
  const int panel = byte_offset / kPanelBytes;
  const int panel_offset = byte_offset - panel * kPanelBytes;
  const int kk = panel_offset / kBlockN;
  const int col = panel_offset - kk * kBlockN;
  reinterpret_cast<int4*>(packed_weight)[vec] =
      *reinterpret_cast<const int4*>(
          raw_weight + static_cast<int64_t>(kk) * kTargetN +
          panel * kBlockN + col);
}

__global__ __launch_bounds__(256) void w8a8_pack_identity_kernel(
    const int8_t* __restrict__ raw,
    int8_t* __restrict__ packed,
    int64_t count) {
  const int64_t linear =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (linear < count) {
    packed[linear] = raw[linear];
  }
}

__global__ __launch_bounds__(256) void w8a8_pack_scale_identity_kernel(
    const float* __restrict__ raw,
    float* __restrict__ packed,
    int count) {
  const int linear = blockIdx.x * blockDim.x + threadIdx.x;
  if (linear < count) {
    packed[linear] = raw[linear];
  }
}

}  // namespace

// ---------------------------------------------------------------------------
// Stable host launch symbols consumed by csrc/bindings.cpp.
// Both launchers are pure dispatch: no allocation, no packing, no
// synchronization, no default-stream launch; they run on the caller-provided
// PyTorch stream and are CUDA-Graph safe.
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
  (void)workspace;        // no split-K: the GEMM does not use the workspace
  (void)workspace_bytes;
  auto* out_bf16 = reinterpret_cast<bf16_t*>(out);

  // Explicit dispatch. The assigned shape (M=4096, N=4096, K=2048) takes the
  // packed-panel DUMMA 128x64 path; every other (m, n, k) - including
  // small-M API cases and the paired M=2 shape with the same (K, N) - takes
  // the scalar fallback, which understands the packed panel layout.
  if (m == kTargetM && n == kTargetN && k == kTargetK) {
    const dim3 grid(kTargetN / kBlockN, kTargetM / kBlockM);
    const dim3 block(kBlockThreads);
    hipLaunchKernelGGL(w8a8_dumma_128x64x64_packed_kernel,
                       grid, block, 0, stream,
                       a, b, x_scale, weight_scale, out_bf16, m, n, k);
  } else {
    constexpr int kBlock = 256;
    const int64_t total = static_cast<int64_t>(m) * n;
    const dim3 grid(static_cast<unsigned>((total + kBlock - 1) / kBlock));
    const dim3 block(kBlock);
    hipLaunchKernelGGL(w8a8_gemm_scalar_fallback_kernel,
                       grid, block, 0, stream,
                       a, b, x_scale, weight_scale, out_bf16, m, n, k);
  }
  (void)hipGetLastError();
}

extern "C" void launch_pack_w8a8_weight(
    const int8_t* raw_weight,
    const float* weight_scale,
    int8_t* packed_weight,
    float* packed_weight_scale,
    int k,
    int n,
    hipStream_t stream) {
  constexpr int kBlock = 256;
  if (k == kTargetK && n == kTargetN) {
    constexpr int kPackVectors =
        kTargetK * kTargetN / static_cast<int>(sizeof(int4));
    const dim3 weight_grid(
        static_cast<unsigned>((kPackVectors + kBlock - 1) / kBlock));
    const dim3 block(kBlock);
    hipLaunchKernelGGL(w8a8_pack_64n_panel_kernel,
                       weight_grid, block, 0, stream,
                       raw_weight, packed_weight);
  } else {
    const int64_t weight_count = static_cast<int64_t>(k) * n;
    const dim3 weight_grid(
        static_cast<unsigned>((weight_count + kBlock - 1) / kBlock));
    const dim3 block(kBlock);
    hipLaunchKernelGGL(w8a8_pack_identity_kernel,
                       weight_grid, block, 0, stream,
                       raw_weight, packed_weight, weight_count);
  }

  const int64_t scale_count = n;
  const dim3 scale_grid(
      static_cast<unsigned>((scale_count + kBlock - 1) / kBlock));
  const dim3 block(kBlock);
  hipLaunchKernelGGL(w8a8_pack_scale_identity_kernel,
                     scale_grid, block, 0, stream,
                     weight_scale, packed_weight_scale, n);
  (void)hipGetLastError();
}
// @@end
