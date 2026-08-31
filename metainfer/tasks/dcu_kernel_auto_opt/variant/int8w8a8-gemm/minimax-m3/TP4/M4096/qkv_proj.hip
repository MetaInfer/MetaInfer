// @@variant shape=minimax_tp4_qkv_proj_m4096 commit=8e400dcec14217e9b03ccc7512c4d99149820b06 added=2026-08-24
//   median_us=1560 p90_us=1575
//   source=minimaxm3-dsh-tp4-m4096-1-0c2f84a9
// MetaInfer W8A8 INT8 GEMM bootstrap for gfx928 (K500SM_AI).
//
// Worker: worker_0 (physical GPU 0)
// Assigned shape: minimax_tp4_qkv_proj_m4096  (M=4096, N=2304, K=6144)
//
// Operator contract (fixed by the control plane):
//   out[m, n] = bf16( int32_dot(x_q[m, :], raw_weight[:, n])
//                     * x_scale[m] * weight_scale[n] )
//
// Bootstrap (iteration 1) strategy, kept unchanged by the iteration 2 repair:
// correctness-first but usable as a profiling baseline. A single native INT8
// DUMMA m16n16k32 tiled kernel
// (64x64 output tile per block, 256 threads = 4 wavefronts, each wave owns a
// 32x32 quadrant = four m16n16k32 int32 accumulator fragments) handles every
// large-M shape with compatible geometry (M >= 128, N % 64 == 0, K % 128 ==
// 0) -- this covers the assigned (4096, 2304, 6144): grid = (36, 64) = 2304
// blocks against 120 CUs. K is staged cooperatively in a SINGLE LDS buffer at
// 128-K granularity (A[64,128] + B[128,64], padded row strides 144 B / 72 B
// to break the int8 fragment-load bank conflicts, 18,432 B/block) with two
// __syncthreads per stage; no split-K, no double buffering, no raw asm, no
// speculative pipelining. A is zero-filled past M and output rows are masked
// in the epilogue for M tails. The epilogue is the verified gfx928 direct
// fragment store (row = lane & 15, col_mod4 = lane >> 4, frag.x[i] -> column
// col_mod4 + 4*i) fused with float(dot) * x_scale[row] * weight_scale[col]
// and bf16 conversion.
//
// launch_pack_w8a8_weight is the bootstrap identity device-to-device copy:
// the packed weight is the logical [K, N] row-major int8 weight and the
// packed scales are the logical [N, 1] fp32 scales, valid for every (K, N).
// Later Parallel explore rounds may change only the HIP implementation and
// the matching GEMM interpretation.
//
// Every unmatched (m, n, k) -- including the paired small-M API shapes
// (M=2 / M=16) -- reaches the generic scalar int8/int32 fallback, which
// reads the same logical [K, N] row-major layout.
//
// The timed operator (launch_w8a8_gemm) performs no allocation, compilation,
// autotuning, packing, host/device synchronization, or default-stream launch:
// it only dispatches kernels on the caller-provided HIP stream.
//
// Include order is fixed by the control plane: hip_runtime, hip_bfloat16,
// then du_mma (this DTK's du_mma.h is not self-contained otherwise).

#include <hip/hip_runtime.h>
#include <hip/hip_bfloat16.h>
#include <du_mma.h>

#include <cstdint>

namespace {

constexpr int kWaveSize = 64;
constexpr int kTileM = 16;
constexpr int kTileN = 16;
constexpr int kTileK = 32;
constexpr int kBlockM = 64;
constexpr int kBlockN = 64;
constexpr int kStageK = 128;
// Padded LDS row strides (bytes) for the int8 fragment loads. With the
// natural strides (A 128, B 64) every du_load_matrix_sync ds_read is
// 8-way (A) / 16-way (B) bank-conflicted: A lane l reads (l&15)*ldm +
// 8*(l>>4) and the row term 32*(l&15) mod 64 collapses to {0,32} (8 lanes
// per bank); B lane l reads 8*(l>>4)*ldm + (l&15) and the k-chunk term
// 8*64 = 512 B is 0 mod 256 B, stacking all four (l>>4) k-groups on the
// same 16 banks. Padding A rows to 144 B (36 dwords) makes 36*(l&15) mod
// 64 take 16 distinct dword-banks; padding B rows to 72 B moves the four
// k-groups 16 banks apart (16*(l>>4)), leaving at most a 4-way residue on
// the byte-granular (l&15)+i diagonal. Pure layout remap: every staged
// value and every fragment element is unchanged, so the int32 accumulation
// order and results stay bit-identical. LDS grows 16,384 -> 18,432 B/block;
// residency drops 4 -> 3 blocks/CU (3 x 18,432 = 55,296 <= 65,536 B LDS,
// 3 x 56 x 256 = 43,008 <= 65,536 VGPR).
constexpr int kAStride = 144;
constexpr int kBStride = 72;
// 4 wavefronts; must remain a multiple of the gfx928 wavefront size (64).
constexpr int kThreadsPerBlock = 256;

using namespace du::dumma;

// Direct accumulator epilogue for gfx928 INT8 m16n16k32 (verified against
// du_store_matrix_sync): row = lane & 15, col_mod4 = lane >> 4,
// frag.x[i] maps to columns col_mod4 + 4*i. Stores
// bf16(float(dot) * x_scale[row] * weight_scale[col]); out-of-range rows are
// masked (tail-M handling).
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

// Large-M prefill kernel: 64x64 output tile per block, K staged in LDS
// (single buffer at 128-K granularity), four waves each owning a 32x32
// quadrant = four m16n16k32 int8->int32 DUMMA accumulators.
__global__ __launch_bounds__(kThreadsPerBlock) void
w8a8_dumma_prefill_64x64_kernel(
    const int8_t* __restrict__ x_q,
    const int8_t* __restrict__ weight,
    const float* __restrict__ x_scale,
    const float* __restrict__ weight_scale,
    hip_bfloat16* __restrict__ out,
    int m,
    int n,
    int k) {
  const int tid = static_cast<int>(threadIdx.x);
  const int wave = tid / kWaveSize;
  const int lane = tid % kWaveSize;
  const int wave_row = wave >> 1;  // 0 or 1
  const int wave_col = wave & 1;   // 0 or 1
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
    // Stage A[64,128] (row-major, zero-filled past M) into the padded
    // 144-B-row layout (row stride 144, still 16-B vector stores: 144/16
    // = 9 int4 slots per row).
    constexpr int kAVectors = kBlockM * kStageK / static_cast<int>(sizeof(int4));
#pragma unroll(2)
    for (int vec = tid; vec < kAVectors; vec += kThreadsPerBlock) {
      const int byte_offset = vec * static_cast<int>(sizeof(int4));
      const int local_row = byte_offset / kStageK;
      const int kk = byte_offset - local_row * kStageK;
      const int global_row = m0 + local_row;
      const int4 v =
          global_row < m
              ? *reinterpret_cast<const int4*>(
                    x_q + global_row * k + k0 + kk)
              : int4{0, 0, 0, 0};
      reinterpret_cast<int4*>(a_tile)[local_row * (kAStride / 16) + kk / 16] =
          v;
    }

    // Stage B[128,64] (row-major [K, N] logical weight layout) into the
    // padded 72-B-row layout. 72 is not a multiple of 16, so each 16-B
    // global vector is stored as two 8-B int2 halves (72*kk + col is
    // always 8-B aligned).
    constexpr int kBVectors = kStageK * kBlockN / static_cast<int>(sizeof(int4));
#pragma unroll(2)
    for (int vec = tid; vec < kBVectors; vec += kThreadsPerBlock) {
      const int byte_offset = vec * static_cast<int>(sizeof(int4));
      const int kk = byte_offset / kBlockN;
      const int col = byte_offset - kk * kBlockN;
      const int4 v = *reinterpret_cast<const int4*>(
          weight + (k0 + kk) * n + n0 + col);
      int2* dst = reinterpret_cast<int2*>(b_tile + kk * kBStride + col);
      dst[0] = *reinterpret_cast<const int2*>(&v);
      dst[1] = *reinterpret_cast<const int2*>(
          reinterpret_cast<const char*>(&v) + 8);
    }
    __syncthreads();

    // Consume the stage: each wave does four m16n16k32 MMAs per 32-K step.
    // Four 32-K tiles per stage: pin the full unroll so the int32
    // accumulation order (k0-outer, kk-inner) is deterministic codegen, not
    // compiler-heuristic dependent.
    const int local_row = wave_row * 32;
    const int local_col = wave_col * 32;
#pragma unroll(4)
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
    // Single buffer: make sure every wave finished reading LDS before the
    // next stage overwrites it.
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

// Generic scalar int8/int32 fallback for unmatched shapes and small-M API
// cases. One output element per grid-stride step; exact int32 accumulation
// (the assigned K keeps the int8 dot well within int32 range), then the fused
// float scale and bf16 store. Reads the logical [K, N] row-major weight
// layout (bootstrap identity pack).
__global__ __launch_bounds__(256) void w8a8_scalar_gemm_kernel(
    const int8_t* __restrict__ a,
    const int8_t* __restrict__ b,
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
       idx < total;
       idx += stride) {
    const int row = static_cast<int>(idx / n);
    const int col = static_cast<int>(idx - static_cast<int64_t>(row) * n);
    const int8_t* a_row = a + static_cast<int64_t>(row) * k;
    const int8_t* b_col = b + col;
    int32_t acc = 0;
    for (int kk = 0; kk < k; ++kk) {
      acc += static_cast<int32_t>(a_row[kk]) *
             static_cast<int32_t>(b_col[static_cast<int64_t>(kk) * n]);
    }
    const float scaled = static_cast<float>(acc) *
                         x_scale[row] * weight_scale[col];
    out[idx] = __float2bfloat16(scaled);
  }
}

// Identity device-to-device byte copy (pack_weight bootstrap, valid for any
// (K, N)).
__global__ __launch_bounds__(256) void w8a8_pack_identity_i8_kernel(
    const int8_t* __restrict__ src,
    int8_t* __restrict__ dst,
    int64_t count) {
  const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x +
                   threadIdx.x;
       i < count;
       i += stride) {
    dst[i] = src[i];
  }
}

// Identity device-to-device float copy for the weight scales.
__global__ __launch_bounds__(256) void w8a8_pack_identity_f32_kernel(
    const float* __restrict__ src,
    float* __restrict__ dst,
    int64_t count) {
  const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x +
                   threadIdx.x;
       i < count;
       i += stride) {
    dst[i] = src[i];
  }
}

}  // namespace

// Stable host launch symbol consumed by csrc/bindings.cpp. Dispatches on the
// caller-provided stream only; never allocates, synchronizes, or touches the
// default stream.
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
  (void)workspace;
  (void)workspace_bytes;
  if (m <= 0 || n <= 0 || k <= 0) {
    return;
  }
  auto* out_ptr = static_cast<hip_bfloat16*>(out);

  // Native INT8 DUMMA m16n16k32 prefill path (2-D macro-tile 64x64,
  // single-buffered 128-K K stage) for large-M shapes with compatible
  // geometry. Covers the assigned shape (M=4096, N=2304, K=6144):
  // 2304 % 64 == 0, 6144 % 128 == 0. The M >= 128 guard keeps the paired
  // small-M API shapes (M=2 / M=16) on the scalar fallback.
  if (m >= 128 && (n % kBlockN) == 0 && (k % kStageK) == 0) {
    const dim3 grid(
        static_cast<unsigned>(n / kBlockN),
        static_cast<unsigned>((m + kBlockM - 1) / kBlockM));
    const dim3 block(static_cast<unsigned>(kThreadsPerBlock));
    hipLaunchKernelGGL(
        w8a8_dumma_prefill_64x64_kernel,
        grid,
        block,
        0,
        stream,
        a,
        b,
        x_scale,
        weight_scale,
        out_ptr,
        m,
        n,
        k);
    return;
  }

  // Scalar int8/int32 fallback for unmatched (m, n, k) and small-M API cases.
  const int64_t total = static_cast<int64_t>(m) * n;
  constexpr int kFallbackThreads = 256;
  int64_t blocks = (total + kFallbackThreads - 1) / kFallbackThreads;
  if (blocks > 4096) {
    blocks = 4096;
  }
  if (blocks < 1) {
    blocks = 1;
  }
  const dim3 grid(static_cast<unsigned>(blocks));
  const dim3 block(kFallbackThreads);
  hipLaunchKernelGGL(
      w8a8_scalar_gemm_kernel,
      grid,
      block,
      0,
      stream,
      a,
      b,
      x_scale,
      weight_scale,
      out_ptr,
      m,
      n,
      k);
}

// Stable host launch symbol consumed by csrc/bindings.cpp. Bootstrap:
// identity device-to-device packing, valid for every (K, N) -- the packed
// weight keeps the logical [K, N] row-major layout and the packed scales are
// copied unchanged. Runs only from the optional pack_weight op, outside the
// timed region and outside Graph capture.
extern "C" void launch_pack_w8a8_weight(
    const int8_t* raw_weight,
    const float* weight_scale,
    int8_t* packed_weight,
    float* packed_weight_scale,
    int k,
    int n,
    hipStream_t stream) {
  constexpr int kPackThreads = 256;
  const int64_t weight_count = static_cast<int64_t>(k) * n;
  if (weight_count > 0) {
    int64_t blocks = (weight_count + kPackThreads - 1) / kPackThreads;
    if (blocks > 4096) {
      blocks = 4096;
    }
    if (blocks < 1) {
      blocks = 1;
    }
    hipLaunchKernelGGL(
        w8a8_pack_identity_i8_kernel,
        dim3(static_cast<unsigned>(blocks)),
        dim3(kPackThreads),
        0,
        stream,
        raw_weight,
        packed_weight,
        weight_count);
  }
  if (n > 0) {
    int64_t blocks = (static_cast<int64_t>(n) + kPackThreads - 1) /
                     kPackThreads;
    if (blocks > 4096) {
      blocks = 4096;
    }
    if (blocks < 1) {
      blocks = 1;
    }
    hipLaunchKernelGGL(
        w8a8_pack_identity_f32_kernel,
        dim3(static_cast<unsigned>(blocks)),
        dim3(kPackThreads),
        0,
        stream,
        weight_scale,
        packed_weight_scale,
        n);
  }
}
// @@end
