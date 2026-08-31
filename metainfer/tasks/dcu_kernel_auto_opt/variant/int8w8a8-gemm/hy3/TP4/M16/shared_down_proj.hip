// @@variant shape=hy3_tp4_shared_down_proj_m16 commit=8f7319d888b0772736fa86ed3dc42afb6b56aa7b added=2026-08-26
//   median_us=9.174 p90_us=9.754
//   source=hy3-dsh-tp4-m16-2-368c654c
// INT8 W8A8 GEMM - worker_3, round 1 (DUMMA bootstrap).
//
// Assigned shape (worker_3, physical GPU 3):
//   hy3_tp4_shared_down_proj_m16 : M=16, N=4096, K=384
//
// Mandated architecture: minimal native gfx928 DUMMA INT8 m16n16k32 tile,
// ONE 64-lane wavefront per block, one 16x16 output tile per block
// (grid = N/16 = 256 blocks), explicit int32 accumulation, NO cross-wave
// barrier anywhere (single wave per block: no LDS, no __syncthreads).
//
//   * A (x_q, logical [16][384] row-major) is loaded with
//     du::dumma::du_load_matrix_sync(matrix_a, row_major): each lane's 8
//     fragment bytes A[row][k0 + i] (row = lane&15, k0 = (lane>>4)*8) are
//     contiguous in global memory, so one 8-byte load per lane per step.
//   * B (weight, logical [384][4096] row-major) is packed ONCE outside the
//     timed region by launch_pack_w8a8_weight for the exact (k,n) ==
//     (384,4096) pair into the [N][K] n-major transpose
//         P[n*384 + k] = W[k*4096 + n].
//     The B fragment is loaded with du::dumma::du_load_matrix_sync(
//     matrix_b, col_major) over P: lane (row = n column, col = 8-k group)
//     reads 8 contiguous bytes P[(n0+row)*384 + k0 + (lane>>4)*8 + i], i.e.
//     one 8-byte load per lane per step instead of the strided byte-load
//     expansion of a row-major [K][N] B fragment.
//   * K=384 = 12 fully-unrolled m16n16k32 steps accumulate into one int32
//     accumulator fragment (du_fill_fragment + du_mma_sync).
//   * Fused direct epilogue using the verified gfx928 int8 m16n16k32
//     accumulator ownership (lane%16 = row, lane/16 = col%4, x[i] = columns
//     col%4 + 4*i): each lane scales its four int32 accumulators by
//     (float(dot) * x_scale[row]) * weight_scale[col] in the same
//     left-associative float32 order as the harness reference, then stores
//     round-to-nearest-even bf16 via __float2bfloat16.
//
// Exact int32 accumulation order (k-ascending, no split-K) is preserved, so
// the bf16 output is expected bit-identical to the scalar bootstrap.
//
// Dispatch:
//   * exact-shape guard (m == 16 && n == 4096 && k == 384) -> DUMMA arm;
//   * every other (m, n, k) - including the paired M=2 API shape with the
//     same (N, K) - reaches the generic scalar fallback, which decodes the
//     n-major pack for (n, k) == (4096, 384) via a b_transposed flag and
//     keeps the identity row-major read for every other pair.
//
// Header order is the known-good DTK order: HIP runtime, then bfloat16,
// then du_mma.h (du_mma.h is not self-contained before the HIP runtime
// headers). gfx928 wavefront is 64 lanes; every block size is a multiple
// of 64.

#include <hip/hip_runtime.h>
#include <hip/hip_bfloat16.h>
#include <du_mma.h>

#include <cstdint>

namespace {

// One block per 256 threads; 256 is a multiple of the gfx928 wavefront (64).
constexpr int kScalarBlockThreads = 256;

// DUMMA INT8 m16n16k32 tile constants for the M=16 exact-shape arm.
constexpr int kDummaTileM = 16;
constexpr int kDummaTileN = 16;
constexpr int kDummaTileK = 32;
constexpr int kDummaBlockThreads = 64;  // one wavefront per block
constexpr int kDummaK = 384;  // exact-guard K; 12 m16n16k32 steps

// Exact pack guard for the assigned (K, N) pair: (384, 4096).
constexpr int kPackK = 384;
constexpr int kPackN = 4096;

// ---------------------------------------------------------------------------
// Generic scalar fallback: one thread per output element.
//   out[m, n] = bf16( int32_dot(x_q[m, :], weight[:, n])
//                     * x_scale[m] * weight_scale[n] )
// x_q is [M, K] int8 row-major (stride K).
// weight is the packed [K, N] int8 buffer (stride N in the identity layout).
// b_transposed selects the exact-shape packed [N][K] layout
// (P[n*K + k] = W[k*N + n], produced by launch_pack_w8a8_weight for
// (k,n) == (384,4096)): the logical column col starts at P[col*K] with unit
// stride; the identity pack keeps the legacy row-major [K][N] layout with
// stride n. The branch is hoisted out of the K loop by the compiler.
// Consecutive threads own consecutive N columns, so adjacent lanes touch
// adjacent addresses in the fastest-changing N dimension.
__global__ __launch_bounds__(kScalarBlockThreads) void w8a8_scalar_gemm_kernel(
    const int8_t* __restrict__ x_q,
    const int8_t* __restrict__ weight,
    const float* __restrict__ x_scale,
    const float* __restrict__ weight_scale,
    __hip_bfloat16* __restrict__ out,
    int m,
    int n,
    int k,
    int b_transposed) {
  const int64_t linear =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total = static_cast<int64_t>(m) * n;
  if (linear >= total) {
    return;
  }

  const int row = static_cast<int>(linear / n);
  const int col = static_cast<int>(linear - static_cast<int64_t>(row) * n);

  const int8_t* a_row = x_q + static_cast<int64_t>(row) * k;
  const int8_t* b_col =
      weight + (b_transposed ? static_cast<int64_t>(col) * k : col);
  const int64_t b_stride = b_transposed ? 1 : static_cast<int64_t>(n);

  // Exact int32 accumulation over the full K loop. The maximum assigned K
  // keeps the exact int8 dot within int32 range (K * 127 * 127 << 2^31).
  int32_t acc = 0;
  for (int kk = 0; kk < k; ++kk) {
    acc += static_cast<int32_t>(a_row[kk]) *
           static_cast<int32_t>(b_col[static_cast<int64_t>(kk) * b_stride]);
  }

  // Convert to float only for the two scales, then store bf16.
  const float scaled = static_cast<float>(acc) * x_scale[row] *
                       weight_scale[col];
  out[linear] = __float2bfloat16(scaled);
}

// ---------------------------------------------------------------------------
// Exact-shape M=16 DUMMA arm (mandated minimal geometry): blockDim = 64 =
// one 64-lane wavefront, one 16x16 output tile per block, grid = N/16 = 256
// blocks. Direct global fragment loads (A row_major from logical x_q, B
// col_major from the exact-shape [N][K] n-major pack), explicit int32
// accumulation over the full K=384 (12 unrolled m16n16k32 steps), fused
// direct scale+bf16 epilogue. No LDS and no barriers anywhere. Only launched
// for (m, n, k) == (16, 4096, 384).
// ---------------------------------------------------------------------------
__global__ __launch_bounds__(kDummaBlockThreads) void
w8a8_dumma_m16n16k32_1wave_kernel(
    const int8_t* __restrict__ x_q,
    const int8_t* __restrict__ packed_b,
    const float* __restrict__ x_scale,
    const float* __restrict__ weight_scale,
    __hip_bfloat16* __restrict__ out,
    int n) {
  constexpr int kSteps = kDummaK / kDummaTileK;  // 12

  const int lane = static_cast<int>(threadIdx.x);  // 0 .. 63
  const int n0 = static_cast<int>(blockIdx.x) * kDummaTileN;

  du::dumma::DUFragment<du::dumma::matrix_a, kDummaTileM, kDummaTileN,
                        kDummaTileK, signed char, du::dumma::row_major>
      a_frag;
  du::dumma::DUFragment<du::dumma::matrix_b, kDummaTileM, kDummaTileN,
                        kDummaTileK, signed char, du::dumma::col_major>
      b_frag;
  du::dumma::DUFragment<du::dumma::accumulator, kDummaTileM, kDummaTileN,
                        kDummaTileK, int>
      acc_frag;
  du::dumma::du_fill_fragment(acc_frag, 0);

  // The col_major B loader adds (lane%16)*ldm internally, so the pointer is
  // the tile base only: P[n0*384 + s*32 + (lane%16)*384 + (lane>>4)*8 + i],
  // i.e. each lane's 8 fragment bytes are contiguous in P (8-byte aligned).
  const int64_t b_tile_base = static_cast<int64_t>(n0) * kDummaK;

#pragma unroll
  for (int s = 0; s < kSteps; ++s) {
    du::dumma::du_load_matrix_sync(a_frag, x_q + s * kDummaTileK, kDummaK);
    du::dumma::du_load_matrix_sync(
        b_frag, packed_b + b_tile_base + s * kDummaTileK, kDummaK);
    du::dumma::du_mma_sync(acc_frag, a_frag, b_frag, acc_frag);
  }

  // gfx928 int8 m16n16k32 accumulator ownership, established against
  // du_store_matrix_sync: lane%16 selects the row, lane/16 selects col%4,
  // and x[i] holds the columns col%4 + 4*i.
  const int row = lane & 15;
  const int col_mod4 = lane >> 4;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const int out_col = n0 + col_mod4 + 4 * i;
    const float scaled = static_cast<float>(acc_frag.x[i]) *
                         x_scale[row] * weight_scale[out_col];
    out[static_cast<int64_t>(row) * n + out_col] =
        __float2bfloat16(scaled);
  }
}

// Simple device-to-device copy used by the identity packing (every (K, N)
// pair that is not the exact-shape transpose pair, plus all scales).
template <typename T>
__global__ __launch_bounds__(kScalarBlockThreads) void
w8a8_identity_copy_kernel(const T* __restrict__ src,
                          T* __restrict__ dst,
                          int64_t numel) {
  const int64_t i =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < numel) {
    dst[i] = src[i];
  }
}

// Exact-shape pack: writes the [N][K] transpose
//   dst[n*K + k] = src[k*N + n]
// for (k, n) == (384, 4096). One thread per 16-byte k-chunk of one column;
// runs once per weight outside the timed region/Graph, so the strided byte
// loads are acceptable. All other shapes keep the identity copy kernel.
__global__ __launch_bounds__(kScalarBlockThreads) void
w8a8_pack_b_transpose_kernel(const int8_t* __restrict__ src,
                             int8_t* __restrict__ dst,
                             int k,
                             int n) {
  const int kchunks = (k + 15) >> 4;
  const int64_t total = static_cast<int64_t>(n) * kchunks;
  const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t t = static_cast<int64_t>(blockIdx.x) * blockDim.x +
                   static_cast<int64_t>(threadIdx.x);
       t < total; t += stride) {
    const int col = static_cast<int>(t / kchunks);
    const int k0 = static_cast<int>(t % kchunks) * 16;
    alignas(16) int8_t chunk[16];
#pragma unroll
    for (int j = 0; j < 16; ++j) {
      chunk[j] = src[(static_cast<int64_t>(k0) + j) * n + col];
    }
    *reinterpret_cast<int4*>(dst + static_cast<int64_t>(col) * k + k0) =
        *reinterpret_cast<const int4*>(chunk);
  }
}

}  // namespace

// Optional out-of-timed-region weight packing. For the exact (k, n) ==
// (384, 4096) pair it produces the [N][K] n-major transpose
// P[n*384 + k] = W[k*4096 + n] that the M=16 DUMMA arm (col_major B
// fragments) and the scalar fallback (b_transposed flag) both decode; every
// other (K, N) keeps the identity layout, and all scales keep the identity
// copy.
extern "C" void launch_pack_w8a8_weight(
    const int8_t* raw_weight,
    const float* weight_scale,
    int8_t* packed_weight,
    float* packed_weight_scale,
    int k,
    int n,
    hipStream_t stream) {
  const int64_t weight_numel = static_cast<int64_t>(k) * n;
  const dim3 block(kScalarBlockThreads);
  const dim3 weight_grid(static_cast<unsigned>(
      (weight_numel + kScalarBlockThreads - 1) / kScalarBlockThreads));
  if (k == kPackK && n == kPackN) {
    // Exact-shape pack: [N][K] transpose P[n*384 + k] = W[k*4096 + n], so
    // every B fragment is one contiguous 8-byte load per lane per step.
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(w8a8_pack_b_transpose_kernel),
        weight_grid, block, 0, stream,
        raw_weight, packed_weight, k, n);
  } else {
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(w8a8_identity_copy_kernel<int8_t>),
        weight_grid, block, 0, stream,
        raw_weight, packed_weight, weight_numel);
  }

  const dim3 scale_grid(static_cast<unsigned>(
      (n + kScalarBlockThreads - 1) / kScalarBlockThreads));
  hipLaunchKernelGGL(
      HIP_KERNEL_NAME(w8a8_identity_copy_kernel<float>),
      scale_grid, block, 0, stream,
      weight_scale, packed_weight_scale, n);
}

// Timed GEMM operator: no allocation, no compilation, no autotuning, no
// weight packing, no host/device synchronization, no default-stream launch.
// Uses only the caller-provided out and workspace, on the caller's stream.
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
  // The one-wave DUMMA arm performs no split-K, so the workspace is unused.
  (void)workspace;
  (void)workspace_bytes;

  // The exact (k, n) == (384, 4096) pack is the [N][K] n-major transpose;
  // any other (k, n) keeps the identity pack (row-major [K][N]).
  const int b_transposed = (k == kPackK && n == kPackN) ? 1 : 0;

  // Exact-shape M=16 DUMMA specialization: one 64-lane wavefront per 16x16
  // output tile -> grid = N/16 = 256 blocks, full K=384 reduction, no
  // barriers. Guarded by the full (m, n, k) triple so the paired M=2 API
  // shape with the same (N, K) still reaches the generic scalar fallback.
  if (m == 16 && n == kPackN && k == kPackK) {
    const dim3 grid(static_cast<unsigned>(n / kDummaTileN));
    const dim3 block(kDummaBlockThreads);
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(w8a8_dumma_m16n16k32_1wave_kernel),
        grid, block, 0, stream,
        a, b, x_scale, weight_scale,
        reinterpret_cast<__hip_bfloat16*>(out), n);
    return;
  }

  // Generic scalar fallback for every other (m, n, k), including the paired
  // M=2 shape with the same (N, K): decodes the n-major pack when
  // b_transposed, row-major otherwise.
  const int64_t total = static_cast<int64_t>(m) * n;
  const dim3 grid(static_cast<unsigned>(
      (total + kScalarBlockThreads - 1) / kScalarBlockThreads));
  const dim3 block(kScalarBlockThreads);
  hipLaunchKernelGGL(
      HIP_KERNEL_NAME(w8a8_scalar_gemm_kernel),
      grid, block, 0, stream,
      a, b, x_scale, weight_scale,
      reinterpret_cast<__hip_bfloat16*>(out), m, n, k, b_transposed);
}
// @@end
