// MetaInfer optional W8A8 implementation variants for gfx928.
//
// Provenance: ISA_test_codex/wqkv_a_compute_tuning. This file contains
// measured kernels and experiment launchers, including the M=16,K=4096,
// N=1536 split10 / 4-wave / stage64 family. It is staged read-only under
// references/ in newly generated kernel repositories and is NOT included by
// setup.py or the default extension build.
//
// This is evidence, not policy: optimization workers may adapt, combine, or
// ignore any implementation here and must explore other legal strategies.
// Every reused idea requires current-shape exact correctness, Graph replay,
// median/P90, workspace, resource, and ISA validation before acceptance.

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <du_mma.h>
#include <hip/hip_bf16.h>
#include <hip/hip_runtime.h>
#include <hip/hip_cooperative_groups.h>
#include <torch/extension.h>

#include <cstdint>

namespace {

constexpr int kTileM = 16;
constexpr int kTileN = 16;
constexpr int kTileK = 32;
constexpr int kWaveSize = 64;
constexpr int kStageK = 256;
constexpr int kPrefillBlockM = 64;
constexpr int kPrefillBlockN = 64;
constexpr int kPrefillStageK = 128;

using namespace du::dumma;

// Fast path for M=16. Both DUMMA operands are loaded directly from their
// row-major global tensors, removing the per-K-stage LDS copy and barriers in
// the generic padded path below.
template <int kWaves>
__global__ __launch_bounds__(kWaveSize * kWaves) void
w8a8_dumma_m16_direct_kernel(
    const int8_t* __restrict__ x_q,
    const int8_t* __restrict__ weight,
    const float* __restrict__ x_scale,
    const float* __restrict__ weight_scale,
    __hip_bfloat16* __restrict__ out,
    int n,
    int k) {
  const int wave = static_cast<int>(threadIdx.x) / kWaveSize;
  const int lane = static_cast<int>(threadIdx.x) % kWaveSize;
  const int n0 =
      (static_cast<int>(blockIdx.x) * kWaves + wave) * kTileN;

  __shared__ __align__(16) int32_t
      acc_tiles[kWaves * kTileM * kTileN];
  int32_t* acc_tile = acc_tiles + wave * kTileM * kTileN;

  DUFragment<matrix_a, kTileM, kTileN, kTileK, signed char, row_major>
      a_frag;
  DUFragment<matrix_b, kTileM, kTileN, kTileK, signed char, row_major>
      b_frag;
  DUFragment<accumulator, kTileM, kTileN, kTileK, int> acc_frag;
  du_fill_fragment(acc_frag, 0);

  for (int k0 = 0; k0 < k; k0 += kTileK) {
    du_load_matrix_sync(a_frag, x_q + k0, k);
    du_load_matrix_sync(b_frag, weight + k0 * n + n0, n);
    du_mma_sync(acc_frag, a_frag, b_frag, acc_frag);
  }

  du_store_matrix_sync(acc_tile, acc_frag, kTileN, mem_row_major);
  __syncthreads();

#pragma unroll
  for (int linear = lane; linear < kTileM * kTileN;
       linear += kWaveSize) {
    const int row = linear / kTileN;
    const int col = linear - row * kTileN;
    const float scaled = static_cast<float>(acc_tile[linear]) *
                         x_scale[row] * weight_scale[n0 + col];
    out[row * n + n0 + col] = __float2bfloat16(scaled);
  }
  return;
}

// One or two waves cooperatively stage A[16,stage_k] and
// B[stage_k,16*waves], then consume DUMMA K=32 fragments per barrier.
template <int kWaves, int kBPad, bool kDirectEpilogue = false,
          int kStageKValue = kStageK>
__global__ __launch_bounds__(kWaves * kWaveSize) void
w8a8_dumma_m16_staged_kernel(
    const int8_t* __restrict__ x_q,
    const int8_t* __restrict__ weight,
    const float* __restrict__ x_scale,
    const float* __restrict__ weight_scale,
    __hip_bfloat16* __restrict__ out,
    int n,
    int k) {
  constexpr int kBlockN = kWaves * kTileN;
  constexpr int kBStride = kBlockN + kBPad;
  constexpr int kThreads = kWaves * kWaveSize;
  const int tid = static_cast<int>(threadIdx.x);
  const int wave = tid / kWaveSize;
  const int lane = tid % kWaveSize;
  const int n0 = static_cast<int>(blockIdx.x) * kBlockN;

  __shared__ __align__(16) int8_t a_tile[kTileM * kStageKValue];
  __shared__ __align__(16) int8_t b_tile[kStageKValue * kBStride];
  // The legacy epilogue materializes the fragment in LDS.  The direct
  // epilogue needs only a dummy word here; if constexpr removes all accesses.
  __shared__ __align__(16) int32_t
      acc_tiles[kDirectEpilogue ? 1 : kWaves * kTileM * kTileN];
  int32_t* acc_tile = acc_tiles + wave * kTileM * kTileN;

  DUFragment<matrix_a, kTileM, kTileN, kTileK, signed char, row_major>
      a_frag;
  DUFragment<matrix_b, kTileM, kTileN, kTileK, signed char, row_major>
      b_frag;
  DUFragment<accumulator, kTileM, kTileN, kTileK, int> acc_frag;
  du_fill_fragment(acc_frag, 0);

  for (int k0 = 0; k0 < k; k0 += kStageKValue) {
    // Vectorized 16-byte cooperative copies. All target K/N values and tile
    // origins are 16-byte aligned.
    constexpr int kAVectors = kTileM * kStageKValue / sizeof(int4);
    for (int vec = tid; vec < kAVectors; vec += kThreads) {
      const int byte_offset = vec * sizeof(int4);
      const int row = byte_offset / kStageKValue;
      const int kk = byte_offset - row * kStageKValue;
      reinterpret_cast<int4*>(a_tile)[vec] =
          *reinterpret_cast<const int4*>(x_q + row * k + k0 + kk);
    }

    if constexpr (kBPad == 0) {
      constexpr int kBVectorsPerRow = kBlockN / sizeof(int4);
      constexpr int kBVectors = kStageKValue * kBVectorsPerRow;
      for (int vec = tid; vec < kBVectors; vec += kThreads) {
        const int kk = vec / kBVectorsPerRow;
        const int segment = vec - kk * kBVectorsPerRow;
        const int col = segment * sizeof(int4);
        *reinterpret_cast<int4*>(b_tile + kk * kBStride + col) =
            *reinterpret_cast<const int4*>(
                weight + (k0 + kk) * n + n0 + col);
      }
    } else if constexpr (kBPad % sizeof(int32_t) == 0) {
      constexpr int kBWordsPerRow = kBlockN / sizeof(int32_t);
      constexpr int kBWords = kStageKValue * kBWordsPerRow;
      for (int word = tid; word < kBWords; word += kThreads) {
        const int kk = word / kBWordsPerRow;
        const int segment = word - kk * kBWordsPerRow;
        const int col = segment * sizeof(int32_t);
        *reinterpret_cast<int32_t*>(b_tile + kk * kBStride + col) =
            *reinterpret_cast<const int32_t*>(
                weight + (k0 + kk) * n + n0 + col);
      }
    } else {
      constexpr int kBElements = kStageKValue * kBlockN;
      for (int elem = tid; elem < kBElements; elem += kThreads) {
        const int kk = elem / kBlockN;
        const int col = elem - kk * kBlockN;
        b_tile[kk * kBStride + col] = weight[(k0 + kk) * n + n0 + col];
      }
    }
    __syncthreads();

#pragma unroll
    for (int kk = 0; kk < kStageKValue; kk += kTileK) {
      du_load_matrix_sync(a_frag, a_tile + kk, kStageKValue);
      du_load_matrix_sync(
          b_frag, b_tile + kk * kBStride + wave * kTileN, kBStride);
      du_mma_sync(acc_frag, a_frag, b_frag, acc_frag);
    }
    __syncthreads();
  }

  if constexpr (kDirectEpilogue) {
    // gfx928 int8 m16n16k32 accumulator ownership, established against
    // du_store_matrix_sync: lane%16 selects the row, lane/16 selects col%4,
    // and x[i] selects columns separated by four.
    const int row = lane & 15;
    const int col_mod4 = lane >> 4;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int out_col = n0 + wave * kTileN + col_mod4 + 4 * i;
      const float scaled = static_cast<float>(acc_frag.x[i]) *
                           x_scale[row] * weight_scale[out_col];
      out[row * n + out_col] = __float2bfloat16(scaled);
    }
  } else {
    du_store_matrix_sync(acc_tile, acc_frag, kTileN, mem_row_major);
    __syncthreads();

#pragma unroll
    for (int linear = lane; linear < kTileM * kTileN;
         linear += kWaveSize) {
      const int row = linear / kTileN;
      const int col = linear - row * kTileN;
      const int out_col = n0 + wave * kTileN + col;
      const float scaled = static_cast<float>(acc_tile[linear]) *
                           x_scale[row] * weight_scale[out_col];
      out[row * n + out_col] = __float2bfloat16(scaled);
    }
  }
  return;
}

// Split-K partial kernel. Each block computes one 16x32 tile for one K slice
// and stores int32 accumulators into caller-owned workspace. No atomics or
// workspace clearing are required.
template <int kWaves, int kStageKValue>
__global__ __launch_bounds__(kWaves * kWaveSize) void
w8a8_dumma_m16_staged2_splitk_kernel(
    const int8_t* __restrict__ x_q,
    const int8_t* __restrict__ weight,
    int32_t* __restrict__ workspace,
    int n,
    int k,
    int split_k) {
  constexpr int kBlockN = kWaves * kTileN;
  constexpr int kThreads = kWaves * kWaveSize;
  const int tid = static_cast<int>(threadIdx.x);
  const int wave = tid / kWaveSize;
  const int num_n_tiles = n / kBlockN;
  const int split_id = static_cast<int>(blockIdx.x) / num_n_tiles;
  const int tile_n = static_cast<int>(blockIdx.x) - split_id * num_n_tiles;
  const int n0 = tile_n * kBlockN;
  const int k_per_split = k / split_k;
  const int k_begin = split_id * k_per_split;
  const int k_end = k_begin + k_per_split;

  __shared__ __align__(16) int8_t a_tile[kTileM * kStageKValue];
  __shared__ __align__(16) int8_t b_tile[kStageKValue * kBlockN];

  DUFragment<matrix_a, kTileM, kTileN, kTileK, signed char, row_major>
      a_frag;
  DUFragment<matrix_b, kTileM, kTileN, kTileK, signed char, row_major>
      b_frag;
  DUFragment<accumulator, kTileM, kTileN, kTileK, int> acc_frag;
  du_fill_fragment(acc_frag, 0);

  for (int k0 = k_begin; k0 < k_end; k0 += kStageKValue) {
    constexpr int kAVectors = kTileM * kStageKValue / sizeof(int4);
    for (int vec = tid; vec < kAVectors; vec += kThreads) {
      const int byte_offset = vec * sizeof(int4);
      const int row = byte_offset / kStageKValue;
      const int kk = byte_offset - row * kStageKValue;
      reinterpret_cast<int4*>(a_tile)[vec] =
          *reinterpret_cast<const int4*>(x_q + row * k + k0 + kk);
    }

    constexpr int kBVectors = kStageKValue * kBlockN / sizeof(int4);
    for (int vec = tid; vec < kBVectors; vec += kThreads) {
      const int byte_offset = vec * sizeof(int4);
      const int kk = byte_offset / kBlockN;
      const int col = byte_offset - kk * kBlockN;
      reinterpret_cast<int4*>(b_tile)[vec] =
          *reinterpret_cast<const int4*>(
              weight + (k0 + kk) * n + n0 + col);
    }
    __syncthreads();

#pragma unroll
    for (int kk = 0; kk < kStageKValue; kk += kTileK) {
      du_load_matrix_sync(a_frag, a_tile + kk, kStageKValue);
      du_load_matrix_sync(
          b_frag, b_tile + kk * kBlockN + wave * kTileN, kBlockN);
      du_mma_sync(acc_frag, a_frag, b_frag, acc_frag);
    }
    __syncthreads();
  }

  int32_t* partial = workspace +
      split_id * kTileM * n + n0 + wave * kTileN;
  du_store_matrix_sync(partial, acc_frag, n, mem_row_major);
  return;
}

// Fixed-shape software-prefetch family. The current winner is 4-wave/stage64
// with non-uniform split10. Global int4 loads for stage i+1 are issued before
// the two MMACs of
// stage i, then committed to the alternate LDS buffer after all waves finish
// consuming the current buffer. Raw memory asm is intentionally avoided; the
// only inline ISA is the conservative, previously validated VMEM wait.
template <int kWaves, int kSyncMode, bool kSpreadA = false>
__global__ __launch_bounds__(kWaves * kWaveSize) void
w8a8_dumma_m16_splitk8_w3_s64_prefetch_kernel(
    const int8_t* __restrict__ x_q,
    const int8_t* __restrict__ weight,
    int32_t* __restrict__ workspace,
    int n,
    int k,
    int split_k) {
  constexpr int kStage = 64;
  constexpr int kBlockN = kWaves * kTileN;
  constexpr int kThreads = kWaves * kWaveSize;
  const int tid = static_cast<int>(threadIdx.x);
  const int wave = tid / kWaveSize;
  const int num_n_tiles = n / kBlockN;
  const int split_id = static_cast<int>(blockIdx.x) / num_n_tiles;
  const int tile_n = static_cast<int>(blockIdx.x) - split_id * num_n_tiles;
  const int n0 = tile_n * kBlockN;
  // split10 is a fixed-shape occupancy experiment for the 120-CU K500SM_AI:
  // 24 N tiles x 10 gives exactly two blocks/CU. Distribute the 64 K/64
  // stages as 4x7 + 6x6 so every boundary remains stage aligned.
  const int base_stages = (k / kStage) / split_k;
  const int extra_stages = (k / kStage) - base_stages * split_k;
  const int begin_stage = split_id * base_stages +
                          (split_id < extra_stages ? split_id : extra_stages);
  const int split_stages = base_stages + (split_id < extra_stages ? 1 : 0);
  const int k_begin = begin_stage * kStage;
  const int k_end = k_begin + split_stages * kStage;

  __shared__ __align__(16) int8_t a_tile[2][kTileM * kStage];
  __shared__ __align__(16) int8_t b_tile[2][kStage * kBlockN];

  DUFragment<matrix_a, kTileM, kTileN, kTileK, signed char, row_major>
      a_frag;
  DUFragment<matrix_b, kTileM, kTileN, kTileK, signed char, row_major>
      b_frag;
  DUFragment<accumulator, kTileM, kTileN, kTileK, int> acc_frag;
  du_fill_fragment(acc_frag, 0);

  int4 a_reg{};
  int4 b_reg{};
  const bool owns_a = kSpreadA ? ((tid & 3) == 0)
                               : (tid < kTileM * kStage /
                                            static_cast<int>(sizeof(int4)));
  const int a_vec = kSpreadA ? (tid >> 2) : tid;
  if (owns_a) {
    const int byte_offset = a_vec * sizeof(int4);
    const int row = byte_offset / kStage;
    const int kk = byte_offset - row * kStage;
    a_reg = *reinterpret_cast<const int4*>(x_q + row * k + k_begin + kk);
  }
  {
    const int byte_offset = tid * sizeof(int4);
    const int kk = byte_offset / kBlockN;
    const int col = byte_offset - kk * kBlockN;
    b_reg = *reinterpret_cast<const int4*>(
        weight + (k_begin + kk) * n + n0 + col);
  }
  if constexpr (kSyncMode != 1) {
    asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
  }
  if (owns_a) {
    reinterpret_cast<int4*>(a_tile[0])[a_vec] = a_reg;
  }
  reinterpret_cast<int4*>(b_tile[0])[tid] = b_reg;
  if constexpr (kSyncMode == 3) {
    asm volatile("s_waitcnt lgkmcnt(0)\n\ts_barrier" ::: "memory");
  } else {
    __syncthreads();
  }

  int current = 0;
  for (int k0 = k_begin; k0 < k_end; k0 += kStage) {
    const int next_k = k0 + kStage;
    int4 next_a{};
    int4 next_b{};
    if (next_k < k_end) {
      if (owns_a) {
        const int byte_offset = a_vec * sizeof(int4);
        const int row = byte_offset / kStage;
        const int kk = byte_offset - row * kStage;
        next_a = *reinterpret_cast<const int4*>(
            x_q + row * k + next_k + kk);
      }
      const int byte_offset = tid * sizeof(int4);
      const int kk = byte_offset / kBlockN;
      const int col = byte_offset - kk * kBlockN;
      next_b = *reinterpret_cast<const int4*>(
          weight + (next_k + kk) * n + n0 + col);
    }

#pragma unroll
    for (int kk = 0; kk < kStage; kk += kTileK) {
      du_load_matrix_sync(a_frag, a_tile[current] + kk, kStage);
      du_load_matrix_sync(
          b_frag, b_tile[current] + kk * kBlockN + wave * kTileN,
          kBlockN);
      du_mma_sync(acc_frag, a_frag, b_frag, acc_frag);
    }
    if (next_k < k_end) {
      if constexpr (kSyncMode >= 2) {
        asm volatile("s_barrier" ::: "memory");
      } else {
        __syncthreads();
      }
      if constexpr (kSyncMode != 1) {
        asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
      }
      const int next = current ^ 1;
      if (owns_a) {
        reinterpret_cast<int4*>(a_tile[next])[a_vec] = next_a;
      }
      reinterpret_cast<int4*>(b_tile[next])[tid] = next_b;
      current = next;
      if constexpr (kSyncMode == 3) {
        asm volatile("s_waitcnt lgkmcnt(0)\n\ts_barrier" ::: "memory");
      } else {
        __syncthreads();
      }
    }
  }

  int32_t* partial = workspace +
      split_id * kTileM * n + n0 + wave * kTileN;
  du_store_matrix_sync(partial, acc_frag, n, mem_row_major);
  return;
}

// One-kernel split-K finalization for the 4-wave/stage64 winner. Each N=64
// tile owns one counter placed after the eight partial planes. The last split
// block to arrive combines its tile and resets the counter for the next
// stream-ordered Graph replay.
__global__ __launch_bounds__(4 * kWaveSize) void
w8a8_dumma_m16_splitk8_w4_s64_fused_finalize_kernel(
    const int8_t* __restrict__ x_q,
    const int8_t* __restrict__ weight,
    const float* __restrict__ x_scale,
    const float* __restrict__ weight_scale,
    __hip_bfloat16* __restrict__ out,
    int32_t* __restrict__ workspace,
    int n,
    int k) {
  constexpr int kWaves = 4;
  constexpr int kStage = 64;
  constexpr int kBlockN = kWaves * kTileN;
  constexpr int kThreads = kWaves * kWaveSize;
  constexpr int kSplitK = 8;
  const int tid = static_cast<int>(threadIdx.x);
  const int wave = tid / kWaveSize;
  const int num_n_tiles = n / kBlockN;
  const int split_id = static_cast<int>(blockIdx.x) / num_n_tiles;
  const int tile_n = static_cast<int>(blockIdx.x) - split_id * num_n_tiles;
  const int n0 = tile_n * kBlockN;
  constexpr int kPerSplit = 4096 / kSplitK;
  const int k_begin = split_id * kPerSplit;
  const int k_end = k_begin + kPerSplit;

  __shared__ __align__(16) int8_t a_tile[2][kTileM * kStage];
  __shared__ __align__(16) int8_t b_tile[2][kStage * kBlockN];
  __shared__ int arrival_ticket;

  DUFragment<matrix_a, kTileM, kTileN, kTileK, signed char, row_major>
      a_frag;
  DUFragment<matrix_b, kTileM, kTileN, kTileK, signed char, row_major>
      b_frag;
  DUFragment<accumulator, kTileM, kTileN, kTileK, int> acc_frag;
  du_fill_fragment(acc_frag, 0);

  int4 a_reg{};
  int4 b_reg{};
  if (tid < 64) {
    const int byte_offset = tid * sizeof(int4);
    const int row = byte_offset / kStage;
    const int kk = byte_offset - row * kStage;
    a_reg = *reinterpret_cast<const int4*>(x_q + row * k + k_begin + kk);
  }
  {
    const int byte_offset = tid * sizeof(int4);
    const int kk = byte_offset / kBlockN;
    const int col = byte_offset - kk * kBlockN;
    b_reg = *reinterpret_cast<const int4*>(
        weight + (k_begin + kk) * n + n0 + col);
  }
  asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
  if (tid < 64) reinterpret_cast<int4*>(a_tile[0])[tid] = a_reg;
  reinterpret_cast<int4*>(b_tile[0])[tid] = b_reg;
  __syncthreads();

  int current = 0;
  for (int k0 = k_begin; k0 < k_end; k0 += kStage) {
    const int next_k = k0 + kStage;
    int4 next_a{};
    int4 next_b{};
    if (next_k < k_end) {
      if (tid < 64) {
        const int byte_offset = tid * sizeof(int4);
        const int row = byte_offset / kStage;
        const int kk = byte_offset - row * kStage;
        next_a = *reinterpret_cast<const int4*>(
            x_q + row * k + next_k + kk);
      }
      const int byte_offset = tid * sizeof(int4);
      const int kk = byte_offset / kBlockN;
      const int col = byte_offset - kk * kBlockN;
      next_b = *reinterpret_cast<const int4*>(
          weight + (next_k + kk) * n + n0 + col);
    }

#pragma unroll
    for (int kk = 0; kk < kStage; kk += kTileK) {
      du_load_matrix_sync(a_frag, a_tile[current] + kk, kStage);
      du_load_matrix_sync(
          b_frag, b_tile[current] + kk * kBlockN + wave * kTileN,
          kBlockN);
      du_mma_sync(acc_frag, a_frag, b_frag, acc_frag);
    }
    if (next_k < k_end) {
      asm volatile("s_barrier" ::: "memory");
      asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
      const int next = current ^ 1;
      if (tid < 64) reinterpret_cast<int4*>(a_tile[next])[tid] = next_a;
      reinterpret_cast<int4*>(b_tile[next])[tid] = next_b;
      current = next;
      __syncthreads();
    }
  }

  int32_t* partial = workspace +
      split_id * kTileM * n + n0 + wave * kTileN;
  du_store_matrix_sync(partial, acc_frag, n, mem_row_major);
  __syncthreads();

  int32_t* counters = workspace + kSplitK * kTileM * n;
  if (tid == 0) {
    __threadfence();
    arrival_ticket = atomicAdd(counters + tile_n, 1);
  }
  __syncthreads();

  if (arrival_ticket == kSplitK - 1) {
#pragma unroll
    for (int linear = tid; linear < kTileM * kBlockN;
         linear += kThreads) {
      const int row = linear / kBlockN;
      const int col = linear - row * kBlockN;
      const int global_linear = row * n + n0 + col;
      int32_t acc = 0;
#pragma unroll
      for (int split = 0; split < kSplitK; ++split) {
        acc += workspace[split * kTileM * n + global_linear];
      }
      const float scaled = static_cast<float>(acc) * x_scale[row] *
                           weight_scale[n0 + col];
      out[global_linear] = __float2bfloat16(scaled);
    }
    __syncthreads();
    if (tid == 0) {
      __threadfence();
      atomicExch(counters + tile_n, 0);
    }
  }
  return;
}

// Cooperative alternative to the atomic last-arriver experiment. The full
// 192-block grid is resident together on K500SM_AI (120 CUs, two blocks/CU
// required). A device-wide barrier makes every partial visible, after which
// all grid threads share the 16x1536 epilogue without a second Graph node.
__global__ __launch_bounds__(4 * kWaveSize) void
w8a8_dumma_m16_splitk8_w4_s64_cooperative_finalize_kernel(
    const int8_t* __restrict__ x_q,
    const int8_t* __restrict__ weight,
    const float* __restrict__ x_scale,
    const float* __restrict__ weight_scale,
    __hip_bfloat16* __restrict__ out,
    int32_t* __restrict__ workspace,
    int n,
    int k) {
  constexpr int kWaves = 4;
  constexpr int kStage = 64;
  constexpr int kBlockN = kWaves * kTileN;
  constexpr int kThreads = kWaves * kWaveSize;
  constexpr int kSplitK = 8;
  const int tid = static_cast<int>(threadIdx.x);
  const int wave = tid / kWaveSize;
  const int num_n_tiles = n / kBlockN;
  const int split_id = static_cast<int>(blockIdx.x) / num_n_tiles;
  const int tile_n = static_cast<int>(blockIdx.x) - split_id * num_n_tiles;
  const int n0 = tile_n * kBlockN;
  constexpr int kPerSplit = 4096 / kSplitK;
  const int k_begin = split_id * kPerSplit;
  const int k_end = k_begin + kPerSplit;

  __shared__ __align__(16) int8_t a_tile[2][kTileM * kStage];
  __shared__ __align__(16) int8_t b_tile[2][kStage * kBlockN];

  DUFragment<matrix_a, kTileM, kTileN, kTileK, signed char, row_major>
      a_frag;
  DUFragment<matrix_b, kTileM, kTileN, kTileK, signed char, row_major>
      b_frag;
  DUFragment<accumulator, kTileM, kTileN, kTileK, int> acc_frag;
  du_fill_fragment(acc_frag, 0);

  int4 a_reg{};
  int4 b_reg{};
  if (tid < 64) {
    const int byte_offset = tid * sizeof(int4);
    const int row = byte_offset / kStage;
    const int kk = byte_offset - row * kStage;
    a_reg = *reinterpret_cast<const int4*>(x_q + row * k + k_begin + kk);
  }
  {
    const int byte_offset = tid * sizeof(int4);
    const int kk = byte_offset / kBlockN;
    const int col = byte_offset - kk * kBlockN;
    b_reg = *reinterpret_cast<const int4*>(
        weight + (k_begin + kk) * n + n0 + col);
  }
  asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
  if (tid < 64) reinterpret_cast<int4*>(a_tile[0])[tid] = a_reg;
  reinterpret_cast<int4*>(b_tile[0])[tid] = b_reg;
  __syncthreads();

  int current = 0;
  for (int k0 = k_begin; k0 < k_end; k0 += kStage) {
    const int next_k = k0 + kStage;
    int4 next_a{};
    int4 next_b{};
    if (next_k < k_end) {
      if (tid < 64) {
        const int byte_offset = tid * sizeof(int4);
        const int row = byte_offset / kStage;
        const int kk = byte_offset - row * kStage;
        next_a = *reinterpret_cast<const int4*>(
            x_q + row * k + next_k + kk);
      }
      const int byte_offset = tid * sizeof(int4);
      const int kk = byte_offset / kBlockN;
      const int col = byte_offset - kk * kBlockN;
      next_b = *reinterpret_cast<const int4*>(
          weight + (next_k + kk) * n + n0 + col);
    }

#pragma unroll
    for (int kk = 0; kk < kStage; kk += kTileK) {
      du_load_matrix_sync(a_frag, a_tile[current] + kk, kStage);
      du_load_matrix_sync(
          b_frag, b_tile[current] + kk * kBlockN + wave * kTileN,
          kBlockN);
      du_mma_sync(acc_frag, a_frag, b_frag, acc_frag);
    }
    if (next_k < k_end) {
      asm volatile("s_barrier" ::: "memory");
      asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
      const int next = current ^ 1;
      if (tid < 64) reinterpret_cast<int4*>(a_tile[next])[tid] = next_a;
      reinterpret_cast<int4*>(b_tile[next])[tid] = next_b;
      current = next;
      __syncthreads();
    }
  }

  int32_t* partial = workspace +
      split_id * kTileM * n + n0 + wave * kTileN;
  du_store_matrix_sync(partial, acc_frag, n, mem_row_major);

  cooperative_groups::this_grid().sync();
  const int linear = static_cast<int>(blockIdx.x) * kThreads + tid;
  const int elements = kTileM * n;
  if (linear < elements) {
    int32_t acc = 0;
#pragma unroll
    for (int split = 0; split < kSplitK; ++split) {
      acc += workspace[split * elements + linear];
    }
    const int row = linear / n;
    const int col = linear - row * n;
    const float scaled = static_cast<float>(acc) * x_scale[row] *
                         weight_scale[col];
    out[linear] = __float2bfloat16(scaled);
  }
  return;
}

// Deeper prefetch candidate: four MMACs per global stage. This intentionally
// remains a separate symbol so the stage64 winner is preserved byte-for-byte.
__global__ __launch_bounds__(3 * kWaveSize) void
w8a8_dumma_m16_splitk8_w3_s128_prefetch_kernel(
    const int8_t* __restrict__ x_q,
    const int8_t* __restrict__ weight,
    int32_t* __restrict__ workspace,
    int n,
    int k,
    int split_k) {
  constexpr int kWaves = 3;
  constexpr int kStage = 128;
  constexpr int kBlockN = kWaves * kTileN;
  constexpr int kThreads = kWaves * kWaveSize;
  constexpr int kAVectors = kTileM * kStage / sizeof(int4);
  const int tid = static_cast<int>(threadIdx.x);
  const int wave = tid / kWaveSize;
  const int num_n_tiles = n / kBlockN;
  const int split_id = static_cast<int>(blockIdx.x) / num_n_tiles;
  const int tile_n = static_cast<int>(blockIdx.x) - split_id * num_n_tiles;
  const int n0 = tile_n * kBlockN;
  const int k_per_split = k / split_k;
  const int k_begin = split_id * k_per_split;
  const int k_end = k_begin + k_per_split;

  __shared__ __align__(16) int8_t a_tile[2][kTileM * kStage];
  __shared__ __align__(16) int8_t b_tile[2][kStage * kBlockN];

  DUFragment<matrix_a, kTileM, kTileN, kTileK, signed char, row_major>
      a_frag;
  DUFragment<matrix_b, kTileM, kTileN, kTileK, signed char, row_major>
      b_frag;
  DUFragment<accumulator, kTileM, kTileN, kTileK, int> acc_frag;
  du_fill_fragment(acc_frag, 0);

  int4 a_reg{};
  int4 b_reg0{};
  int4 b_reg1{};
  if (tid < kAVectors) {
    const int byte_offset = tid * sizeof(int4);
    const int row = byte_offset / kStage;
    const int kk = byte_offset - row * kStage;
    a_reg = *reinterpret_cast<const int4*>(x_q + row * k + k_begin + kk);
  }
  {
    const int byte_offset0 = tid * sizeof(int4);
    const int kk0 = byte_offset0 / kBlockN;
    const int col0 = byte_offset0 - kk0 * kBlockN;
    b_reg0 = *reinterpret_cast<const int4*>(
        weight + (k_begin + kk0) * n + n0 + col0);
    const int vec1 = tid + kThreads;
    const int byte_offset1 = vec1 * sizeof(int4);
    const int kk1 = byte_offset1 / kBlockN;
    const int col1 = byte_offset1 - kk1 * kBlockN;
    b_reg1 = *reinterpret_cast<const int4*>(
        weight + (k_begin + kk1) * n + n0 + col1);
  }
  asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
  if (tid < kAVectors) reinterpret_cast<int4*>(a_tile[0])[tid] = a_reg;
  reinterpret_cast<int4*>(b_tile[0])[tid] = b_reg0;
  reinterpret_cast<int4*>(b_tile[0])[tid + kThreads] = b_reg1;
  __syncthreads();

  int current = 0;
  for (int k0 = k_begin; k0 < k_end; k0 += kStage) {
    const int next_k = k0 + kStage;
    int4 next_a{};
    int4 next_b0{};
    int4 next_b1{};
    if (next_k < k_end) {
      if (tid < kAVectors) {
        const int byte_offset = tid * sizeof(int4);
        const int row = byte_offset / kStage;
        const int kk = byte_offset - row * kStage;
        next_a = *reinterpret_cast<const int4*>(
            x_q + row * k + next_k + kk);
      }
      const int byte_offset0 = tid * sizeof(int4);
      const int kk0 = byte_offset0 / kBlockN;
      const int col0 = byte_offset0 - kk0 * kBlockN;
      next_b0 = *reinterpret_cast<const int4*>(
          weight + (next_k + kk0) * n + n0 + col0);
      const int vec1 = tid + kThreads;
      const int byte_offset1 = vec1 * sizeof(int4);
      const int kk1 = byte_offset1 / kBlockN;
      const int col1 = byte_offset1 - kk1 * kBlockN;
      next_b1 = *reinterpret_cast<const int4*>(
          weight + (next_k + kk1) * n + n0 + col1);
    }

#pragma unroll
    for (int kk = 0; kk < kStage; kk += kTileK) {
      du_load_matrix_sync(a_frag, a_tile[current] + kk, kStage);
      du_load_matrix_sync(
          b_frag, b_tile[current] + kk * kBlockN + wave * kTileN,
          kBlockN);
      du_mma_sync(acc_frag, a_frag, b_frag, acc_frag);
    }
    if (next_k < k_end) {
      __syncthreads();
      asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
      const int next = current ^ 1;
      if (tid < kAVectors) reinterpret_cast<int4*>(a_tile[next])[tid] = next_a;
      reinterpret_cast<int4*>(b_tile[next])[tid] = next_b0;
      reinterpret_cast<int4*>(b_tile[next])[tid + kThreads] = next_b1;
      current = next;
      __syncthreads();
    }
  }

  int32_t* partial = workspace +
      split_id * kTileM * n + n0 + wave * kTileN;
  du_store_matrix_sync(partial, acc_frag, n, mem_row_major);
  return;
}

template <typename AccFragment>
__device__ __forceinline__ void store_prefill_fragment(
    const AccFragment& frag,
    const float* __restrict__ x_scale,
    const float* __restrict__ weight_scale,
    __hip_bfloat16* __restrict__ out,
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

// Large-M prefill path. Four waves compute a 64x64 output tile. Each wave
// owns a 32x32 quadrant (four 16x16 DUMMA accumulators), while the block
// cooperatively stages A[64,128] and B[128,64] in 16KB LDS.
__global__ __launch_bounds__(4 * kWaveSize) void
w8a8_dumma_prefill_64x64x128_kernel(
    const int8_t* __restrict__ x_q,
    const int8_t* __restrict__ weight,
    const float* __restrict__ x_scale,
    const float* __restrict__ weight_scale,
    __hip_bfloat16* __restrict__ out,
    int m,
    int n,
    int k) {
  constexpr int kThreads = 4 * kWaveSize;
  const int tid = static_cast<int>(threadIdx.x);
  const int wave = tid / kWaveSize;
  const int lane = tid % kWaveSize;
  const int wave_row = wave >> 1;
  const int wave_col = wave & 1;
  const int m0 = static_cast<int>(blockIdx.y) * kPrefillBlockM;
  const int n0 = static_cast<int>(blockIdx.x) * kPrefillBlockN;

  __shared__ __align__(16) int8_t
      a_tile[kPrefillBlockM * kPrefillStageK];
  __shared__ __align__(16) int8_t
      b_tile[kPrefillStageK * kPrefillBlockN];

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

  for (int k0 = 0; k0 < k; k0 += kPrefillStageK) {
    constexpr int kAVectors =
        kPrefillBlockM * kPrefillStageK / sizeof(int4);
    for (int vec = tid; vec < kAVectors; vec += kThreads) {
      const int byte_offset = vec * sizeof(int4);
      const int local_row = byte_offset / kPrefillStageK;
      const int kk = byte_offset - local_row * kPrefillStageK;
      const int global_row = m0 + local_row;
      reinterpret_cast<int4*>(a_tile)[vec] =
          global_row < m
              ? *reinterpret_cast<const int4*>(
                    x_q + global_row * k + k0 + kk)
              : int4{0, 0, 0, 0};
    }

    constexpr int kBVectors =
        kPrefillStageK * kPrefillBlockN / sizeof(int4);
    for (int vec = tid; vec < kBVectors; vec += kThreads) {
      const int byte_offset = vec * sizeof(int4);
      const int kk = byte_offset / kPrefillBlockN;
      const int col = byte_offset - kk * kPrefillBlockN;
      reinterpret_cast<int4*>(b_tile)[vec] =
          *reinterpret_cast<const int4*>(
              weight + (k0 + kk) * n + n0 + col);
    }
    __syncthreads();

#pragma unroll
    for (int kk = 0; kk < kPrefillStageK; kk += kTileK) {
      const int local_row = wave_row * 32;
      const int local_col = wave_col * 32;
      du_load_matrix_sync(
          a_frag0, a_tile + local_row * kPrefillStageK + kk,
          kPrefillStageK);
      du_load_matrix_sync(
          a_frag1, a_tile + (local_row + 16) * kPrefillStageK + kk,
          kPrefillStageK);
      du_load_matrix_sync(
          b_frag0, b_tile + kk * kPrefillBlockN + local_col,
          kPrefillBlockN);
      du_load_matrix_sync(
          b_frag1, b_tile + kk * kPrefillBlockN + local_col + 16,
          kPrefillBlockN);
      du_mma_sync(acc00, a_frag0, b_frag0, acc00);
      du_mma_sync(acc01, a_frag0, b_frag1, acc01);
      du_mma_sync(acc10, a_frag1, b_frag0, acc10);
      du_mma_sync(acc11, a_frag1, b_frag1, acc11);
    }
    __syncthreads();
  }

  const int base_row = m0 + wave_row * 32;
  const int base_col = n0 + wave_col * 32;
  store_prefill_fragment(
      acc00, x_scale, weight_scale, out, m, n,
      base_row, base_col, lane);
  store_prefill_fragment(
      acc01, x_scale, weight_scale, out, m, n,
      base_row, base_col + 16, lane);
  store_prefill_fragment(
      acc10, x_scale, weight_scale, out, m, n,
      base_row + 16, base_col, lane);
  store_prefill_fragment(
      acc11, x_scale, weight_scale, out, m, n,
      base_row + 16, base_col + 16, lane);
  return;
}

__global__ __launch_bounds__(256) void w8a8_splitk_combine_kernel(
    const int32_t* __restrict__ workspace,
    const float* __restrict__ x_scale,
    const float* __restrict__ weight_scale,
    __hip_bfloat16* __restrict__ out,
    int n,
    int split_k) {
  const int linear = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int elements = kTileM * n;
  if (linear >= elements) {
    return;
  }
  int32_t acc = 0;
#pragma unroll
  for (int split = 0; split < split_k; ++split) {
    acc += workspace[split * elements + linear];
  }
  const int row = linear / n;
  const int col = linear - row * n;
  const float scaled = static_cast<float>(acc) *
                       x_scale[row] * weight_scale[col];
  out[linear] = __float2bfloat16(scaled);
}

struct __align__(4) BFloat16x2 {
  __hip_bfloat16 x;
  __hip_bfloat16 y;
};

struct __align__(8) BFloat16x4 {
  __hip_bfloat16 x;
  __hip_bfloat16 y;
  __hip_bfloat16 z;
  __hip_bfloat16 w;
};

__global__ __launch_bounds__(256) void w8a8_splitk_combine_vec2_kernel(
    const int32_t* __restrict__ workspace,
    const float* __restrict__ x_scale,
    const float* __restrict__ weight_scale,
    __hip_bfloat16* __restrict__ out,
    int n,
    int split_k) {
  const int pair = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
  constexpr int kVec = 2;
  const int pairs = kTileM * 1536 / kVec;
  if (pair >= pairs) return;
  const int base = pair * kVec;
  int2 acc{0, 0};
#pragma unroll
  for (int split = 0; split < split_k; ++split) {
    const int2 value = *reinterpret_cast<const int2*>(
        workspace + split * kTileM * n + base);
    acc.x += value.x;
    acc.y += value.y;
  }
  const int row = base / n;
  const int col = base - row * n;
  const float xs = x_scale[row];
  const float2 ws = *reinterpret_cast<const float2*>(weight_scale + col);
  const BFloat16x2 result{
      __float2bfloat16(static_cast<float>(acc.x) * xs * ws.x),
      __float2bfloat16(static_cast<float>(acc.y) * xs * ws.y)};
  *reinterpret_cast<BFloat16x2*>(out + base) = result;
}

__global__ __launch_bounds__(256) void w8a8_splitk_combine_vec4_kernel(
    const int32_t* __restrict__ workspace,
    const float* __restrict__ x_scale,
    const float* __restrict__ weight_scale,
    __hip_bfloat16* __restrict__ out,
    int n,
    int split_k) {
  const int quad = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
  constexpr int kVec = 4;
  const int quads = kTileM * 1536 / kVec;
  if (quad >= quads) return;
  const int base = quad * kVec;
  int4 acc{0, 0, 0, 0};
#pragma unroll
  for (int split = 0; split < split_k; ++split) {
    const int4 value = *reinterpret_cast<const int4*>(
        workspace + split * kTileM * n + base);
    acc.x += value.x;
    acc.y += value.y;
    acc.z += value.z;
    acc.w += value.w;
  }
  const int row = base / n;
  const int col = base - row * n;
  const float xs = x_scale[row];
  const float4 ws = *reinterpret_cast<const float4*>(weight_scale + col);
  const BFloat16x4 result{
      __float2bfloat16(static_cast<float>(acc.x) * xs * ws.x),
      __float2bfloat16(static_cast<float>(acc.y) * xs * ws.y),
      __float2bfloat16(static_cast<float>(acc.z) * xs * ws.z),
      __float2bfloat16(static_cast<float>(acc.w) * xs * ws.w)};
  *reinterpret_cast<BFloat16x4*>(out + base) = result;
}

// Generic 1 <= M < 16 path. DUMMA always consumes 16 rows, so absent rows are
// zero-padded in LDS before each 16x16x32 matrix operation.
__global__ __launch_bounds__(kWaveSize) void w8a8_dumma_m_lt16_padded_kernel(
    const int8_t* __restrict__ x_q,
    const int8_t* __restrict__ weight,
    const float* __restrict__ x_scale,
    const float* __restrict__ weight_scale,
    __hip_bfloat16* __restrict__ out,
    int m,
    int n,
    int k) {
  const int lane = static_cast<int>(threadIdx.x);
  const int n0 = static_cast<int>(blockIdx.x) * kTileN;

  __shared__ __align__(16) int8_t a_tile[kTileM * kTileK];
  __shared__ __align__(16) int32_t acc_tile[kTileM * kTileN];

  DUFragment<matrix_a, kTileM, kTileN, kTileK, signed char, row_major>
      a_frag;
  DUFragment<matrix_b, kTileM, kTileN, kTileK, signed char, row_major>
      b_frag;
  DUFragment<accumulator, kTileM, kTileN, kTileK, int> acc_frag;
  du_fill_fragment(acc_frag, 0);

  for (int k0 = 0; k0 < k; k0 += kTileK) {
    // Stage A because DUMMA always consumes 16 rows, while decode M may be
    // smaller. Each lane copies eight int8 values for a full 16x32 tile.
#pragma unroll
    for (int linear = lane; linear < kTileM * kTileK;
         linear += kWaveSize) {
      const int row = linear / kTileK;
      const int kk = linear - row * kTileK;
      a_tile[linear] = row < m ? x_q[row * k + k0 + kk] : int8_t{0};
    }
    __syncthreads();

    du_load_matrix_sync(a_frag, a_tile, kTileK);
    du_load_matrix_sync(b_frag, weight + k0 * n + n0, n);
    du_mma_sync(acc_frag, a_frag, b_frag, acc_frag);

    // Ensure every lane has consumed a_tile before it is overwritten.
    __syncthreads();
  }

  du_store_matrix_sync(acc_tile, acc_frag, kTileN, mem_row_major);
  __syncthreads();

  // A 16x16 tile has 256 values, so each of the 64 lanes writes four.
#pragma unroll
  for (int linear = lane; linear < kTileM * kTileN;
       linear += kWaveSize) {
    const int row = linear / kTileN;
    const int col = linear - row * kTileN;
    if (row < m) {
      const float scaled = static_cast<float>(acc_tile[linear]) *
                           x_scale[row] * weight_scale[n0 + col];
      out[row * n + n0 + col] = __float2bfloat16(scaled);
    }
  }
  return;
}

void check_inputs(
    const torch::Tensor& x_q,
    const torch::Tensor& weight,
    const torch::Tensor& x_scale,
    const torch::Tensor& weight_scale,
    const torch::Tensor& out) {
  TORCH_CHECK(x_q.is_cuda(), "x_q must be a CUDA/HIP tensor");
  TORCH_CHECK(weight.is_cuda(), "weight must be a CUDA/HIP tensor");
  TORCH_CHECK(x_scale.is_cuda(), "x_scale must be a CUDA/HIP tensor");
  TORCH_CHECK(weight_scale.is_cuda(),
              "weight_scale must be a CUDA/HIP tensor");
  TORCH_CHECK(out.is_cuda(), "out must be a CUDA/HIP tensor");
  TORCH_CHECK(x_q.device() == weight.device() &&
                  x_q.device() == x_scale.device() &&
                  x_q.device() == weight_scale.device() &&
                  x_q.device() == out.device(),
              "all tensors must be on the same device");

  TORCH_CHECK(x_q.scalar_type() == at::kChar, "x_q must be int8");
  TORCH_CHECK(weight.scalar_type() == at::kChar, "weight must be int8");
  TORCH_CHECK(x_scale.scalar_type() == at::kFloat,
              "x_scale must be float32");
  TORCH_CHECK(weight_scale.scalar_type() == at::kFloat,
              "weight_scale must be float32");
  TORCH_CHECK(out.scalar_type() == at::kBFloat16, "out must be bfloat16");

  TORCH_CHECK(x_q.dim() == 2, "x_q must have shape [M, K]");
  TORCH_CHECK(weight.dim() == 2, "weight must have shape [K, N]");
  TORCH_CHECK(x_scale.dim() == 2 && x_scale.size(1) == 1,
              "x_scale must have shape [M, 1]");
  TORCH_CHECK(weight_scale.dim() == 2 && weight_scale.size(1) == 1,
              "weight_scale must have shape [N, 1]");
  TORCH_CHECK(out.dim() == 2, "out must have shape [M, N]");

  const auto m = x_q.size(0);
  const auto k = x_q.size(1);
  TORCH_CHECK(m >= 1, "M must be positive");
  TORCH_CHECK(weight.size(0) == k, "K mismatch between x_q and weight");
  const auto n = weight.size(1);
  TORCH_CHECK(k % kTileK == 0, "K must be divisible by 32");
  TORCH_CHECK(n % kTileN == 0, "N must be divisible by 16");
  TORCH_CHECK(x_scale.size(0) == m, "x_scale shape mismatch");
  TORCH_CHECK(weight_scale.size(0) == n, "weight_scale shape mismatch");
  TORCH_CHECK(out.size(0) == m && out.size(1) == n,
              "out shape must be [M, N]");

  TORCH_CHECK(x_q.is_contiguous(), "x_q must be contiguous");
  TORCH_CHECK(weight.is_contiguous(), "weight must be contiguous");
  TORCH_CHECK(x_scale.is_contiguous(), "x_scale must be contiguous");
  TORCH_CHECK(weight_scale.is_contiguous(),
              "weight_scale must be contiguous");
  TORCH_CHECK(out.is_contiguous(), "out must be contiguous");
}

void launch_m16_variant(
    int waves,
    const int8_t* x_q,
    const int8_t* weight,
    const float* x_scale,
    const float* weight_scale,
    __hip_bfloat16* out,
    int n,
    int k,
    hipStream_t stream) {
  TORCH_CHECK(
      waves == 1 || waves == 2 || waves == 4 || waves == 16 ||
          waves == 32 || waves == 33 || waves == 36 || waves == 40 ||
          waves == 132 || waves == 1128 || waves == 1512,
      "variant must be direct 1/2/4, staged1=16, staged2 stride "
      "32/33/36/40, or staged2 direct-epilogue K-stage 256/128/512="
      "132/1128/1512");
  if (waves == 16 || waves == 32 || waves == 33 || waves == 36 ||
      waves == 40 || waves == 132 || waves == 1128 || waves == 1512) {
    const int stage_waves = waves == 16 ? 1 : 2;
    TORCH_CHECK(n % (stage_waves * kTileN) == 0,
                "N is not divisible by the staged block N");
    const int requested_stage_k =
        waves == 1128 ? 128 : (waves == 1512 ? 512 : kStageK);
    TORCH_CHECK(k % requested_stage_k == 0,
                "K must be divisible by the staged kernel K depth");
    const dim3 grid(static_cast<unsigned>(n / (stage_waves * kTileN)));
    const dim3 block(static_cast<unsigned>(stage_waves * kWaveSize));
    if (stage_waves == 1) {
      hipLaunchKernelGGL(
          HIP_KERNEL_NAME(w8a8_dumma_m16_staged_kernel<1, 0>),
          grid, block, 0, stream,
          x_q, weight, x_scale, weight_scale, out, n, k);
    } else {
      if (waves == 1128) {
        hipLaunchKernelGGL(
            HIP_KERNEL_NAME(w8a8_dumma_m16_staged_kernel<2, 0, true, 128>),
            grid, block, 0, stream,
            x_q, weight, x_scale, weight_scale, out, n, k);
      } else if (waves == 1512) {
        hipLaunchKernelGGL(
            HIP_KERNEL_NAME(w8a8_dumma_m16_staged_kernel<2, 0, true, 512>),
            grid, block, 0, stream,
            x_q, weight, x_scale, weight_scale, out, n, k);
      } else if (waves == 132) {
        hipLaunchKernelGGL(
            HIP_KERNEL_NAME(w8a8_dumma_m16_staged_kernel<2, 0, true>),
            grid, block, 0, stream,
            x_q, weight, x_scale, weight_scale, out, n, k);
      } else if (waves == 32) {
        hipLaunchKernelGGL(
            HIP_KERNEL_NAME(w8a8_dumma_m16_staged_kernel<2, 0>),
            grid, block, 0, stream,
            x_q, weight, x_scale, weight_scale, out, n, k);
      } else if (waves == 33) {
        hipLaunchKernelGGL(
            HIP_KERNEL_NAME(w8a8_dumma_m16_staged_kernel<2, 1>),
            grid, block, 0, stream,
            x_q, weight, x_scale, weight_scale, out, n, k);
      } else if (waves == 36) {
        hipLaunchKernelGGL(
            HIP_KERNEL_NAME(w8a8_dumma_m16_staged_kernel<2, 4>),
            grid, block, 0, stream,
            x_q, weight, x_scale, weight_scale, out, n, k);
      } else {
        hipLaunchKernelGGL(
            HIP_KERNEL_NAME(w8a8_dumma_m16_staged_kernel<2, 8>),
            grid, block, 0, stream,
            x_q, weight, x_scale, weight_scale, out, n, k);
      }
    }
    return;
  }
  TORCH_CHECK(n % (kTileN * waves) == 0,
              "N must be divisible by 16 * waves");
  const dim3 grid(static_cast<unsigned>(n / (kTileN * waves)));
  const dim3 block(static_cast<unsigned>(kWaveSize * waves));

  if (waves == 1) {
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(w8a8_dumma_m16_direct_kernel<1>),
        grid, block, 0, stream,
        x_q, weight, x_scale, weight_scale, out, n, k);
  } else if (waves == 2) {
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(w8a8_dumma_m16_direct_kernel<2>),
        grid, block, 0, stream,
        x_q, weight, x_scale, weight_scale, out, n, k);
  } else {
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(w8a8_dumma_m16_direct_kernel<4>),
        grid, block, 0, stream,
        x_q, weight, x_scale, weight_scale, out, n, k);
  }
}

}  // namespace

torch::Tensor w8a8_gemm_out_hip(
    const torch::Tensor& x_q,
    const torch::Tensor& weight,
    const torch::Tensor& x_scale,
    const torch::Tensor& weight_scale,
    torch::Tensor out) {
  check_inputs(x_q, weight, x_scale, weight_scale, out);
  TORCH_CHECK(x_q.size(0) <= kTileM, "gemm_out requires M in [1, 16]");

  const c10::cuda::CUDAGuard device_guard(x_q.device());
  const int m = static_cast<int>(x_q.size(0));
  const int k = static_cast<int>(x_q.size(1));
  const int n = static_cast<int>(weight.size(1));
  const dim3 grid(static_cast<unsigned>(n / kTileN));
  const dim3 block(kWaveSize);
  const auto stream = at::cuda::getCurrentCUDAStream(x_q.get_device());

  if (m == kTileM) {
    // Offline TP4/M=16 tuning: shallow K stages improve occupancy for the
    // K<=2048 decode GEMMs, while K=4096 retains the 256-deep stage.
    const int variant = k <= 2048 ? 1128 : 132;
    launch_m16_variant(
        variant,
        x_q.data_ptr<int8_t>(),
        weight.data_ptr<int8_t>(),
        x_scale.data_ptr<float>(),
        weight_scale.data_ptr<float>(),
        reinterpret_cast<__hip_bfloat16*>(out.data_ptr<at::BFloat16>()),
        n,
        k,
        stream);
  } else {
    hipLaunchKernelGGL(
        w8a8_dumma_m_lt16_padded_kernel,
        grid,
        block,
        0,
        stream,
        x_q.data_ptr<int8_t>(),
        weight.data_ptr<int8_t>(),
        x_scale.data_ptr<float>(),
        weight_scale.data_ptr<float>(),
        reinterpret_cast<__hip_bfloat16*>(out.data_ptr<at::BFloat16>()),
        m,
        n,
        k);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

torch::Tensor w8a8_gemm_out_variant_hip(
    const torch::Tensor& x_q,
    const torch::Tensor& weight,
    const torch::Tensor& x_scale,
    const torch::Tensor& weight_scale,
    torch::Tensor out,
    int64_t waves) {
  check_inputs(x_q, weight, x_scale, weight_scale, out);
  TORCH_CHECK(x_q.size(0) == kTileM,
              "gemm_out_variant currently requires M=16");

  const c10::cuda::CUDAGuard device_guard(x_q.device());
  const int k = static_cast<int>(x_q.size(1));
  const int n = static_cast<int>(weight.size(1));
  const auto stream = at::cuda::getCurrentCUDAStream(x_q.get_device());
  launch_m16_variant(
      static_cast<int>(waves),
      x_q.data_ptr<int8_t>(),
      weight.data_ptr<int8_t>(),
      x_scale.data_ptr<float>(),
      weight_scale.data_ptr<float>(),
      reinterpret_cast<__hip_bfloat16*>(out.data_ptr<at::BFloat16>()),
      n,
      k,
      stream);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

torch::Tensor w8a8_gemm_out_prefill_hip(
    const torch::Tensor& x_q,
    const torch::Tensor& weight,
    const torch::Tensor& x_scale,
    const torch::Tensor& weight_scale,
    torch::Tensor out) {
  check_inputs(x_q, weight, x_scale, weight_scale, out);
  TORCH_CHECK(x_q.size(0) > kTileM, "prefill kernel requires M > 16");
  TORCH_CHECK(x_q.size(1) % kPrefillStageK == 0,
              "K must be divisible by 128");
  TORCH_CHECK(weight.size(1) % kPrefillBlockN == 0,
              "N must be divisible by 64");

  const c10::cuda::CUDAGuard device_guard(x_q.device());
  const int m = static_cast<int>(x_q.size(0));
  const int k = static_cast<int>(x_q.size(1));
  const int n = static_cast<int>(weight.size(1));
  const auto stream = at::cuda::getCurrentCUDAStream(x_q.get_device());
  const dim3 grid(
      static_cast<unsigned>(n / kPrefillBlockN),
      static_cast<unsigned>((m + kPrefillBlockM - 1) / kPrefillBlockM));
  const dim3 block(4 * kWaveSize);
  hipLaunchKernelGGL(
      w8a8_dumma_prefill_64x64x128_kernel,
      grid,
      block,
      0,
      stream,
      x_q.data_ptr<int8_t>(),
      weight.data_ptr<int8_t>(),
      x_scale.data_ptr<float>(),
      weight_scale.data_ptr<float>(),
      reinterpret_cast<__hip_bfloat16*>(out.data_ptr<at::BFloat16>()),
      m,
      n,
      k);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

torch::Tensor w8a8_gemm_out_splitk_hip(
    const torch::Tensor& x_q,
    const torch::Tensor& weight,
    const torch::Tensor& x_scale,
    const torch::Tensor& weight_scale,
    torch::Tensor out,
    torch::Tensor workspace,
    int64_t split_k) {
  check_inputs(x_q, weight, x_scale, weight_scale, out);
  TORCH_CHECK(x_q.size(0) == kTileM, "split-K currently requires M=16");
  TORCH_CHECK(split_k == 2 || split_k == 4 || split_k == 8 || split_k == 16,
              "split_k must be 2, 4, 8, or 16");
  TORCH_CHECK(workspace.is_cuda() && workspace.device() == x_q.device(),
              "workspace must be on the same CUDA/HIP device");
  TORCH_CHECK(workspace.scalar_type() == at::kInt,
              "workspace must be int32");
  TORCH_CHECK(workspace.is_contiguous(), "workspace must be contiguous");

  const int k = static_cast<int>(x_q.size(1));
  const int n = static_cast<int>(weight.size(1));
  TORCH_CHECK((k / split_k) % kStageK == 0,
              "each K split must be divisible by 256");
  TORCH_CHECK(workspace.numel() >= split_k * kTileM * n,
              "workspace needs at least split_k * 16 * N int32 elements");

  const c10::cuda::CUDAGuard device_guard(x_q.device());
  const auto stream = at::cuda::getCurrentCUDAStream(x_q.get_device());
  const dim3 partial_grid(
      static_cast<unsigned>((n / (2 * kTileN)) * split_k));
  const dim3 partial_block(2 * kWaveSize);
  hipLaunchKernelGGL(
      HIP_KERNEL_NAME(w8a8_dumma_m16_staged2_splitk_kernel<2, kStageK>),
      partial_grid,
      partial_block,
      0,
      stream,
      x_q.data_ptr<int8_t>(),
      weight.data_ptr<int8_t>(),
      workspace.data_ptr<int32_t>(),
      n,
      k,
      static_cast<int>(split_k));

  const int elements = kTileM * n;
  const dim3 combine_block(256);
  const dim3 combine_grid(static_cast<unsigned>((elements + 255) / 256));
  hipLaunchKernelGGL(
      w8a8_splitk_combine_kernel,
      combine_grid,
      combine_block,
      0,
      stream,
      workspace.data_ptr<int32_t>(),
      x_scale.data_ptr<float>(),
      weight_scale.data_ptr<float>(),
      reinterpret_cast<__hip_bfloat16*>(out.data_ptr<at::BFloat16>()),
      n,
      static_cast<int>(split_k));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

torch::Tensor w8a8_gemm_out_splitk_config_hip(
    const torch::Tensor& x_q,
    const torch::Tensor& weight,
    const torch::Tensor& x_scale,
    const torch::Tensor& weight_scale,
    torch::Tensor out,
    torch::Tensor workspace,
    int64_t split_k,
    int64_t waves,
    int64_t stage_k,
    int64_t combine_block,
    bool prefetch,
    int64_t combine_mode,
    int64_t phase,
    int64_t sync_mode,
    bool spread_a,
    bool fused_finalize,
    bool cooperative_finalize) {
  check_inputs(x_q, weight, x_scale, weight_scale, out);
  TORCH_CHECK(x_q.size(0) == kTileM, "configured split-K requires M=16");
  TORCH_CHECK(split_k == 2 || split_k == 4 || split_k == 8 || split_k == 10 ||
                  split_k == 16,
              "split_k must be 2, 4, 8, 10, or 16");
  TORCH_CHECK(waves == 1 || waves == 2 || waves == 3 || waves == 4 ||
                  waves == 6 || waves == 8,
              "waves must be 1, 2, 3, 4, 6, or 8");
  TORCH_CHECK(stage_k == 32 || stage_k == 64 || stage_k == 128 || stage_k == 256 ||
                  stage_k == 512,
              "stage_k must be 32, 64, 128, 256, or 512");
  TORCH_CHECK(combine_block == 64 || combine_block == 128 ||
                  combine_block == 256,
              "combine_block must be 64, 128, or 256");
  TORCH_CHECK(combine_mode >= 0 && combine_mode <= 2,
              "combine_mode must be 0 scalar, 1 vec2, or 2 vec4");
  TORCH_CHECK(phase >= 0 && phase <= 2,
              "phase must be 0 both, 1 partial, or 2 combine");
  TORCH_CHECK(sync_mode >= 0 && sync_mode <= 3,
              "sync_mode must be 0 baseline, 1 compiler wait, 2 raw first "
              "barrier, or 3 full raw synchronization");
  TORCH_CHECK(!spread_a ||
                  (prefetch && split_k == 8 && waves == 4 && stage_k == 64 &&
                   sync_mode == 2),
              "spread_a is specialized for split8/waves4/stage64/sync2");
  TORCH_CHECK(!fused_finalize ||
                  (phase == 0 && prefetch && split_k == 8 && waves == 4 &&
                   stage_k == 64 && sync_mode == 2 && combine_mode == 0),
              "fused_finalize requires the split8/waves4/stage64/sync2 path");
  TORCH_CHECK(!cooperative_finalize ||
                  (phase == 0 && prefetch && split_k == 8 && waves == 4 &&
                   stage_k == 64 && sync_mode == 2 && combine_mode == 0 &&
                   !fused_finalize),
              "cooperative_finalize requires the split8/waves4/stage64/sync2 path");
  TORCH_CHECK(workspace.is_cuda() && workspace.device() == x_q.device(),
              "workspace must be on the same CUDA/HIP device");
  TORCH_CHECK(workspace.scalar_type() == at::kInt && workspace.is_contiguous(),
              "workspace must be contiguous int32");

  const int k = static_cast<int>(x_q.size(1));
  const int n = static_cast<int>(weight.size(1));
  TORCH_CHECK(k == 4096 && n == 1536,
              "configured kernel is specialized for K=4096,N=1536");
  TORCH_CHECK(split_k == 10 || (k / split_k) % stage_k == 0,
              "each uniform K split must be divisible by stage_k");
  TORCH_CHECK(workspace.numel() >= split_k * kTileM * n,
              "workspace needs split_k * 16 * N int32 elements");

  const c10::cuda::CUDAGuard device_guard(x_q.device());
  const auto stream = at::cuda::getCurrentCUDAStream(x_q.get_device());
  const unsigned grid_x = static_cast<unsigned>((n / (waves * kTileN)) * split_k);

#define LAUNCH_SPLITK_CONFIG(W, S)                                          \
  hipLaunchKernelGGL(                                                       \
      HIP_KERNEL_NAME(w8a8_dumma_m16_staged2_splitk_kernel<W, S>),         \
      dim3(grid_x), dim3(W * kWaveSize), 0, stream,                         \
      x_q.data_ptr<int8_t>(), weight.data_ptr<int8_t>(),                    \
      workspace.data_ptr<int32_t>(), n, k, static_cast<int>(split_k))

  if (cooperative_finalize) {
    const int8_t* x_ptr = x_q.data_ptr<int8_t>();
    const int8_t* w_ptr = weight.data_ptr<int8_t>();
    const float* xs_ptr = x_scale.data_ptr<float>();
    const float* ws_ptr = weight_scale.data_ptr<float>();
    auto* out_ptr = reinterpret_cast<__hip_bfloat16*>(
        out.data_ptr<at::BFloat16>());
    int32_t* workspace_ptr = workspace.data_ptr<int32_t>();
    int kernel_n = n;
    int kernel_k = k;
    void* kernel_args[] = {&x_ptr, &w_ptr, &xs_ptr, &ws_ptr, &out_ptr,
                           &workspace_ptr, &kernel_n, &kernel_k};
    C10_CUDA_CHECK(hipLaunchCooperativeKernel(
        reinterpret_cast<const void*>(
            w8a8_dumma_m16_splitk8_w4_s64_cooperative_finalize_kernel),
        dim3(grid_x), dim3(4 * kWaveSize), kernel_args, 0, stream));
  } else if (fused_finalize) {
    hipLaunchKernelGGL(
        w8a8_dumma_m16_splitk8_w4_s64_fused_finalize_kernel,
        dim3(grid_x), dim3(4 * kWaveSize), 0, stream,
        x_q.data_ptr<int8_t>(), weight.data_ptr<int8_t>(),
        x_scale.data_ptr<float>(), weight_scale.data_ptr<float>(),
        reinterpret_cast<__hip_bfloat16*>(out.data_ptr<at::BFloat16>()),
        workspace.data_ptr<int32_t>(), n, k);
  } else if (phase != 2) {
    if (prefetch) {
      TORCH_CHECK((split_k == 4 || split_k == 8 || split_k == 10 ||
                   split_k == 16) &&
                      ((stage_k == 64 &&
                        (waves == 1 || waves == 2 || waves == 3 ||
                         waves == 4 || waves == 6 || waves == 8)) ||
                       (stage_k == 128 && waves == 3)),
                  "prefetch supports split4|8|10|16/stage64 waves1|2|3|4|6|8, or "
                  "split8/stage128 waves3");
      if (stage_k == 64) {
        if (waves != 3) {
          TORCH_CHECK(sync_mode == 2,
                      "non-3-wave prefetch candidates require sync_mode=2");
          if (waves == 1) {
            hipLaunchKernelGGL(
                HIP_KERNEL_NAME(w8a8_dumma_m16_splitk8_w3_s64_prefetch_kernel<1, 2>),
                dim3(grid_x), dim3(kWaveSize), 0, stream,
                x_q.data_ptr<int8_t>(), weight.data_ptr<int8_t>(),
                workspace.data_ptr<int32_t>(), n, k, static_cast<int>(split_k));
          } else if (waves == 2) {
            hipLaunchKernelGGL(
                HIP_KERNEL_NAME(w8a8_dumma_m16_splitk8_w3_s64_prefetch_kernel<2, 2>),
                dim3(grid_x), dim3(2 * kWaveSize), 0, stream,
                x_q.data_ptr<int8_t>(), weight.data_ptr<int8_t>(),
                workspace.data_ptr<int32_t>(), n, k, static_cast<int>(split_k));
          } else if (waves == 4) {
            if (spread_a) {
              hipLaunchKernelGGL(
                  HIP_KERNEL_NAME(w8a8_dumma_m16_splitk8_w3_s64_prefetch_kernel<4, 2, true>),
                  dim3(grid_x), dim3(4 * kWaveSize), 0, stream,
                  x_q.data_ptr<int8_t>(), weight.data_ptr<int8_t>(),
                  workspace.data_ptr<int32_t>(), n, k, static_cast<int>(split_k));
            } else {
              hipLaunchKernelGGL(
                  HIP_KERNEL_NAME(w8a8_dumma_m16_splitk8_w3_s64_prefetch_kernel<4, 2, false>),
                  dim3(grid_x), dim3(4 * kWaveSize), 0, stream,
                  x_q.data_ptr<int8_t>(), weight.data_ptr<int8_t>(),
                  workspace.data_ptr<int32_t>(), n, k, static_cast<int>(split_k));
            }
          } else if (waves == 6) {
            hipLaunchKernelGGL(
                HIP_KERNEL_NAME(w8a8_dumma_m16_splitk8_w3_s64_prefetch_kernel<6, 2>),
                dim3(grid_x), dim3(6 * kWaveSize), 0, stream,
                x_q.data_ptr<int8_t>(), weight.data_ptr<int8_t>(),
                workspace.data_ptr<int32_t>(), n, k, static_cast<int>(split_k));
          } else {
            hipLaunchKernelGGL(
                HIP_KERNEL_NAME(w8a8_dumma_m16_splitk8_w3_s64_prefetch_kernel<8, 2>),
                dim3(grid_x), dim3(8 * kWaveSize), 0, stream,
                x_q.data_ptr<int8_t>(), weight.data_ptr<int8_t>(),
                workspace.data_ptr<int32_t>(), n, k, static_cast<int>(split_k));
          }
        } else if (sync_mode == 0) {
          hipLaunchKernelGGL(
              HIP_KERNEL_NAME(w8a8_dumma_m16_splitk8_w3_s64_prefetch_kernel<3, 0>),
              dim3(grid_x), dim3(3 * kWaveSize), 0, stream,
              x_q.data_ptr<int8_t>(), weight.data_ptr<int8_t>(),
              workspace.data_ptr<int32_t>(), n, k, static_cast<int>(split_k));
        } else if (sync_mode == 1) {
          hipLaunchKernelGGL(
              HIP_KERNEL_NAME(w8a8_dumma_m16_splitk8_w3_s64_prefetch_kernel<3, 1>),
              dim3(grid_x), dim3(3 * kWaveSize), 0, stream,
              x_q.data_ptr<int8_t>(), weight.data_ptr<int8_t>(),
              workspace.data_ptr<int32_t>(), n, k, static_cast<int>(split_k));
        } else if (sync_mode == 2) {
          hipLaunchKernelGGL(
              HIP_KERNEL_NAME(w8a8_dumma_m16_splitk8_w3_s64_prefetch_kernel<3, 2>),
              dim3(grid_x), dim3(3 * kWaveSize), 0, stream,
              x_q.data_ptr<int8_t>(), weight.data_ptr<int8_t>(),
              workspace.data_ptr<int32_t>(), n, k, static_cast<int>(split_k));
        } else {
          hipLaunchKernelGGL(
              HIP_KERNEL_NAME(w8a8_dumma_m16_splitk8_w3_s64_prefetch_kernel<3, 3>),
              dim3(grid_x), dim3(3 * kWaveSize), 0, stream,
              x_q.data_ptr<int8_t>(), weight.data_ptr<int8_t>(),
              workspace.data_ptr<int32_t>(), n, k, static_cast<int>(split_k));
        }
      } else {
        hipLaunchKernelGGL(
            w8a8_dumma_m16_splitk8_w3_s128_prefetch_kernel,
            dim3(grid_x), dim3(3 * kWaveSize), 0, stream,
            x_q.data_ptr<int8_t>(), weight.data_ptr<int8_t>(),
            workspace.data_ptr<int32_t>(), n, k, static_cast<int>(split_k));
      }
    } else if (waves == 1 && stage_k == 32) LAUNCH_SPLITK_CONFIG(1, 32);
  else if (waves == 1 && stage_k == 64) LAUNCH_SPLITK_CONFIG(1, 64);
  else if (waves == 1 && stage_k == 128) LAUNCH_SPLITK_CONFIG(1, 128);
  else if (waves == 1 && stage_k == 256) LAUNCH_SPLITK_CONFIG(1, 256);
  else if (waves == 1 && stage_k == 512) LAUNCH_SPLITK_CONFIG(1, 512);
  else if (waves == 2 && stage_k == 32) LAUNCH_SPLITK_CONFIG(2, 32);
  else if (waves == 2 && stage_k == 64) LAUNCH_SPLITK_CONFIG(2, 64);
  else if (waves == 2 && stage_k == 128) LAUNCH_SPLITK_CONFIG(2, 128);
  else if (waves == 2 && stage_k == 256) LAUNCH_SPLITK_CONFIG(2, 256);
  else if (waves == 2 && stage_k == 512) LAUNCH_SPLITK_CONFIG(2, 512);
  else if (waves == 3 && stage_k == 32) LAUNCH_SPLITK_CONFIG(3, 32);
  else if (waves == 3 && stage_k == 64) LAUNCH_SPLITK_CONFIG(3, 64);
  else if (waves == 3 && stage_k == 128) LAUNCH_SPLITK_CONFIG(3, 128);
  else if (waves == 3 && stage_k == 256) LAUNCH_SPLITK_CONFIG(3, 256);
  else if (waves == 3 && stage_k == 512) LAUNCH_SPLITK_CONFIG(3, 512);
  else if (waves == 4 && stage_k == 32) LAUNCH_SPLITK_CONFIG(4, 32);
  else if (waves == 4 && stage_k == 64) LAUNCH_SPLITK_CONFIG(4, 64);
  else if (waves == 4 && stage_k == 128) LAUNCH_SPLITK_CONFIG(4, 128);
  else if (waves == 4 && stage_k == 256) LAUNCH_SPLITK_CONFIG(4, 256);
  else if (waves == 4 && stage_k == 512) LAUNCH_SPLITK_CONFIG(4, 512);
  else if (waves == 6 && stage_k == 32) LAUNCH_SPLITK_CONFIG(6, 32);
  else if (waves == 6 && stage_k == 64) LAUNCH_SPLITK_CONFIG(6, 64);
  else if (waves == 6 && stage_k == 128) LAUNCH_SPLITK_CONFIG(6, 128);
  else if (waves == 6 && stage_k == 256) LAUNCH_SPLITK_CONFIG(6, 256);
  else LAUNCH_SPLITK_CONFIG(6, 512);
  }
#undef LAUNCH_SPLITK_CONFIG

  if (!fused_finalize && !cooperative_finalize && phase != 1) {
    const int elements = kTileM * n;
    const int vector_width = combine_mode == 0 ? 1 : combine_mode == 1 ? 2 : 4;
    const dim3 combine_grid(static_cast<unsigned>(
        ((elements / vector_width) + combine_block - 1) / combine_block));
    const dim3 combine_threads(static_cast<unsigned>(combine_block));
    if (combine_mode == 0) {
      hipLaunchKernelGGL(
          w8a8_splitk_combine_kernel, combine_grid, combine_threads, 0,
          stream, workspace.data_ptr<int32_t>(), x_scale.data_ptr<float>(),
          weight_scale.data_ptr<float>(),
          reinterpret_cast<__hip_bfloat16*>(out.data_ptr<at::BFloat16>()), n,
          static_cast<int>(split_k));
    } else if (combine_mode == 1) {
      hipLaunchKernelGGL(
          w8a8_splitk_combine_vec2_kernel, combine_grid, combine_threads, 0,
          stream, workspace.data_ptr<int32_t>(), x_scale.data_ptr<float>(),
          weight_scale.data_ptr<float>(),
          reinterpret_cast<__hip_bfloat16*>(out.data_ptr<at::BFloat16>()), n,
          static_cast<int>(split_k));
    } else {
      hipLaunchKernelGGL(
          w8a8_splitk_combine_vec4_kernel, combine_grid, combine_threads, 0,
          stream, workspace.data_ptr<int32_t>(), x_scale.data_ptr<float>(),
          weight_scale.data_ptr<float>(),
          reinterpret_cast<__hip_bfloat16*>(out.data_ptr<at::BFloat16>()), n,
          static_cast<int>(split_k));
    }
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}
