// @@variant shape=hy3_tp4_qkv_proj_m4096 commit=61a0f079ab9622cc6c83a9c44af6ecd626ef0640 added=2026-08-23
//   median_us=756.7 p90_us=760.2 tops=113.5 bandwidth_gb_s=63.78 speedup=31.78 baseline_us=2.405e+04
//   source=hy3-tp4-m4096-dsh-test1-bfdf3002
// MetaInfer W8A8 INT8 GEMM bootstrap for gfx928 (K500SM_AI).
//
// Worker: worker_0 (physical GPU 0)
// Assigned shape: hy3_tp4_qkv_proj_m4096  (M=4096, N=2560, K=4096)
//
// Operator contract (fixed by the control plane):
//   out[m, n] = bf16( int32_dot(x_q[m, :], raw_weight[:, n])
//                     * x_scale[m] * weight_scale[n] )
// with the packed weight layout equal to the logical [K, N] row-major int8
// weight for the bootstrap (launch_pack_w8a8_weight is an identity D2D copy).
//
// Iteration 3 (harness) strategy: single-buffered 64x128 macro-tile with
// cooperative A+B LDS staging -- the matched control arm of the mandated
// pipeline comparison (single vs double buffering across K tiles).
//   * Large-M path (M >= 128, N % 128 == 0, K % 128 == 0): native INT8 DUMMA
//     m16n16k32 tiled kernel with a 64x128 output tile per block and 256
//     threads (4 wavefronts). Each wave owns a 32x64 quadrant (eight
//     m16n16k32 int32 accumulator fragments). K is staged cooperatively in a
//     SINGLE LDS buffer at 64-K granularity: the next stage's 3 x int4
//     (1 A + 2 B vectors) are loaded from global only after the current
//     stage's 16-MMAC DUMMA burst and its all-consumption barrier, waited
//     on, and committed as int32 LDS stores into the same buffer, followed
//     by a store-visibility barrier -- TWO __syncthreads per 64-K stage
//     (128 over K=4096). The global-load latency is fully exposed (no
//     compute overlap), which isolates the value of the double-buffer
//     overlap at the same 64-K stage size, odd-word LDS strides, and
//     2-blocks/CU residency as the accepted iteration-2 double-buffered
//     kernel (1108.58 us median; iteration-1 single-buffered 128-K stage:
//     1296.50 us). LDS/block = 64x68 + 64x132 = 12,800 B -> 2 resident
//     blocks/CU (VGPR-bound). The grid (N/128) x (ceil(M/64)) supplies 1280
//     blocks for the assigned shape.
//   * Comparison evidence (fact ledger): double-64 vs single-128 improved
//     the benchmark median 1296.50 -> 1108.58 us, but PMC moved against the
//     'reduced VMEM stalls' reading: vmem_read_instructions stayed flat at
//     1,177,600, lds_wait_instructions rose 19.48M -> 20.69M, bank conflicts
//     rose 5.24M -> 7.86M, L2 hit fell 89.46% -> 88.25%. This round re-runs
//     the single-buffered pipeline at the SAME 64-K stage to decide
//     retention on matched evidence (benchmark median/P90 is authoritative
//     per the interpretation guard; PMC counters are diagnostic).
//   * Operand-reuse accounting per 64x128 macro-tile (A[64,64] + B[64,128]
//     per 64-K stage): cooperative staging loads each A element from global
//     once per stage and reuses it for 8 MMAs (512 MMA-uses per element over
//     K=4096), and each B element for 2 MMAs per stage (128 MMA-uses over
//     K=4096). Without staging, the two waves sharing each A row / B column
//     segment would refetch everything (2x global traffic) with per-lane
//     8-byte fragment loads (4x more global-load instructions), so the
//     cooperative A+B staging path is kept.
//   * LDS row strides stay padded to odd word counts (A: 68 B = 17 words,
//     B: 132 B = 33 words). The 64x64 bootstrap used unpadded 128/64-byte
//     rows, so every INT8 m16n16k32 fragment load collapsed onto a handful
//     of LDS banks (16-way conflicts; 199.2M conflict events over 24.9M LDS
//     instructions in PMC). The odd word stride makes the lane row/k-group
//     index contribute to the bank, spreading fragment loads across banks
//     (A and B: 16-way -> 2-4-way per LDS instruction) at 2 blocks/CU.
//   * The epilogue applies float(dot) * x_scale[m] * weight_scale[n] and
//     stores bf16 directly from the accumulator fragments using the verified
//     gfx928 int8 m16n16k32 lane mapping (row = lane & 15, col_mod4 =
//     lane >> 4, frag.x[i] -> columns col_mod4 + 4*i).
//   * Fallbacks are preserved unchanged: the 64x64 kernel for large-M shapes
//     with N % 64 == 0 but N % 128 != 0, and the scalar int8/int32
//     grid-stride fallback for every other (m, n, k) including the paired
//     M=2 API shape. launch_pack_w8a8_weight stays an identity
//     device-to-device copy valid for every (K, N).
//
// Iteration 5 (packing round) strategy: keep the accepted iteration-3
// pipeline (single-buffered 64-K stage, 64x128 macro-tile, 3 blocks/CU) and
// add ONE weight packing/swizzle for the exact assigned shape (K=4096,
// N=2560): launch_pack_w8a8_weight transposes the logical [K, N] weight to
// an n-major packed[n][k] layout once, outside the timed region and outside
// Graph capture (same byte count k*n, same allocation, graph-stable packed
// layout). The exact shape routes (before the generic 64x128 path) to a new
// w8a8_dumma_prefill_64x128_packedb_kernel that stages B into an n-major LDS
// tile with a 16-byte-aligned 80-byte row stride (20 words, five bank
// phases) and consumes it with col_major m16n16k32 B fragments via an
// explicit 8-byte loader (load_b_frag8): each lane's 8 fragment elements at
// p[row*ldm + col + i] are one contiguous 8-byte-aligned chunk (one
// ds_read2_b64 per B fragment instead of 8 byte-granular ds_read_u8 at
// stride 132 in the accepted kernel's ISA), and each 16-byte staging vector
// commits with one ds_write_b128 instead of four ds_write_b32. A stays
// row-major (already vectorized ds_read2_b32 in the accepted ISA); the int32
// accumulation order (k0-outer, kk-inner) and the fused scale/bf16 epilogue
// are unchanged, so results stay bit-identical. All other shapes keep the
// identity pack and the accepted iteration-3 kernel byte-for-byte; the 64x64
// kernel, the scalar fallback (which decodes packed[n][k] when (k, n)
// matches), M tails, and the exact shape guards are preserved.
//
// Iteration 6 (epilogue round) strategy: keep the accepted iteration-5
// packedb kernel's pipeline byte-identical (single-buffered 64-K stage,
// n-major B with 80-B rows, load_b_frag8, 3 blocks/CU) and replace ONLY its
// final store. The scales and bf16 conversion are already fused in the
// kernel and launch_w8a8_gemm has no workspace/combine pass (workspace is
// unused), so the remaining epilogue inefficiency is the store pattern: the
// m16n16k32 accumulator lane mapping (row = lane & 15, column group
// c4 = lane >> 4, frag.x[i] -> column c4 + 4*i) makes the direct store four
// 2-byte scalar stores per lane whose wavefront addresses touch every 32-B
// sector at 25% utilization. store_prefill_fragment_coalesced transposes the
// 4-element groups within each 4-lane column group (lanes r, r+16, r+32,
// r+48) with two 2x2 shuffle steps (8 shfl_xor + 8 v_cndmask; this DTK
// lowers each shfl_xor to one ds_bpermute -- +64 LDS permutes per wavefront
// at the block tail, where the LDS pipe is otherwise idle), so lane (r, c4)
// owns the four CONTIGUOUS columns 4*c4 .. 4*c4+3, then converts to bf16,
// packs 4 bf16 (8 B), and issues ONE 8-byte store per lane: 64 lanes x 8 B =
// 512 B per fragment per wavefront in 16 fully-used 32-B sectors (100%
// store sector efficiency; the assigned shape's vmem_write_instructions drop
// 163,840 -> 40,960). Only the int32 values are re-routed between lanes; the
// per-element float scale/bf16 math is unchanged, so output bits are
// identical. The generic 64x128/64x64 kernels and the scalar fallback are
// untouched (their store_prefill_fragment epilogue is preserved
// byte-for-byte).
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
// 4 wavefronts; must remain a multiple of the gfx928 wavefront size (64).
constexpr int kThreadsPerBlock = 256;

// Iteration 3 (pipeline round): 64x128 macro-tile, SINGLE-buffered 64-K K
// stage (matched control for the single-vs-double buffering comparison).
// LDS row strides stay padded to odd word counts (A: 68 B = 17 words,
// B: 132 B = 33 words) so the INT8 m16n16k32 fragment loads spread across
// banks instead of collapsing 16-way onto one bank (128-byte rows are 32
// words, i.e. 0 mod 32 for every row -> single-bank collisions).
constexpr int kBlockM128 = 64;
constexpr int kBlockN128 = 128;
// Single-buffered K stage: one 64-K buffer = 12,800 B/block (the accepted
// iteration-2 double buffer was 2 x 64 = 25,600 B; the iteration-1
// single-buffered 128-K stage was 25,344 B). Occupancy stays 2 blocks/CU
// (VGPR-bound at arch_vgpr ~112), identical to both prior kernels.
constexpr int kStageK128 = 64;
constexpr int kAStride128 = kStageK128 + 4;  // 68 bytes per A row (17 words)
constexpr int kBStride128 = kBlockN128 + 4;  // 132 bytes per B row (33 words)

// Iteration 5 (packing round): packed n-major B layout for the exact
// assigned shape (K=4096, N=2560). B is transposed once, outside timing, to
// packed[n][k]; the packed kernel stages B into an n-major LDS tile with a
// 16-byte-aligned non-power-of-two row stride (80 B = 20 words, five bank
// phases) so every DUMMA m16n16k32 B fragment load (col_major) reads its
// lane's 8 bytes contiguously (ds_read2_b64) and every 16-byte staging
// vector commits with one ds_write_b128 (b_n*80 + b_k16 is 0 mod 16).
constexpr int kPackedBStride = 80;  // 64 data + 16 pad bytes (20 words)
constexpr int kPackedBK = 4096;     // exact K of the packed assigned shape
constexpr int kPackedBN = 2560;     // exact N of the packed assigned shape

// Iteration 18 (fragment-pipeline round): the packedb compute loop splits the
// 64-K stage into two explicit 32-K steps (kk = 0, kk = 32) and hoists ALL
// fragment loads for both steps before the first v_mmac. This requires
// exactly two kTileK steps per stage.
static_assert(kStageK128 == 2 * kTileK,
              "pipelined kk loop requires exactly two 32-K steps per stage");

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

// Iteration 6 (epilogue round): coalesced fragment store. The m16n16k32
// accumulator lane mapping (row = lane & 15, column group c4 = lane >> 4,
// frag.x[i] -> column c4 + 4*i) gives each lane four elements strided by 4
// columns, so the direct store (store_prefill_fragment) is four 2-byte
// scalar stores per lane whose wavefront addresses touch each 32-B sector at
// 25% utilization (8 B used of 32 B; 4 store instructions per fragment per
// wave). This epilogue transposes the 4-element groups within each 4-lane
// column group (lanes r, r+16, r+32, r+48 -- a 4x4 transpose, two 2x2 steps
// with shfl_xor 16 then 32, one v_cndmask per element per step; no staging
// tile round trip), so lane (r, c4) ends up holding the four CONTIGUOUS
// columns 4*c4 .. 4*c4+3, converts them to bf16, packs 4 bf16 (8 B), and
// writes ONE 8-byte store per lane: 64 lanes x 8 B = 512 B per fragment per
// wavefront in 16 fully-used 32-B sectors (100% store sector efficiency;
// vmem_write_instructions for the assigned shape drops 163,840 -> 40,960).
// Note: this DTK's __shfl_xor lowers to ds_bpermute (one LDS permute per
// shuffle, i.e. +64 LDS instructions per wavefront per block -- accepted
// because it runs at the block tail where the LDS pipe is otherwise idle;
// there is no global round trip and no staging tile). Only the int32 values
// are re-routed between lanes -- the per-element scale multiply order
// (float(dot) * x_scale[row] * weight_scale[col]) and the bf16 rounding are
// unchanged, so the stored bits are identical to store_prefill_fragment.
// The row>=m guard is wavefront-uniform (lane & 15 cycles the same 16 rows
// in every 16-lane group), so the shuffles never mix active and inactive
// lanes; base_col is a multiple of 64 and n*2 a multiple of 8, so the float4
// weight_scale load (col0 % 4 == 0) and the 8-byte store are aligned.
template <typename AccFragment>
__device__ __forceinline__ void store_prefill_fragment_coalesced(
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
  const int c4 = lane >> 4;  // 0..3
  const int x0 = frag.x[0];
  const int x1 = frag.x[1];
  const int x2 = frag.x[2];
  const int x3 = frag.x[3];
  // Step 1: transpose the 2x2 in-block pairs (c4 bit0 <-> element bit0).
  const int s0 = __shfl_xor(x0, 16, kWaveSize);
  const int s1 = __shfl_xor(x1, 16, kWaveSize);
  const int s2 = __shfl_xor(x2, 16, kWaveSize);
  const int s3 = __shfl_xor(x3, 16, kWaveSize);
  const bool lo = (c4 & 1) == 0;
  const int a0 = lo ? x0 : s1;
  const int a1 = lo ? s0 : x1;
  const int a2 = lo ? x2 : s3;
  const int a3 = lo ? s2 : x3;
  // Step 2: transpose the 2x2 blocks (c4 bit1 <-> element bit1).
  const int t0 = __shfl_xor(a0, 32, kWaveSize);
  const int t1 = __shfl_xor(a1, 32, kWaveSize);
  const int t2 = __shfl_xor(a2, 32, kWaveSize);
  const int t3 = __shfl_xor(a3, 32, kWaveSize);
  const bool hi = (c4 & 2) == 0;
  const int f0 = hi ? a0 : t2;
  const int f1 = hi ? a1 : t3;
  const int f2 = hi ? t0 : a2;
  const int f3 = hi ? t1 : a3;

  // Lane (r, c4) now owns columns base_col + 4*c4 .. +3 (8 B, 8-byte
  // aligned: base_col is a multiple of 64, n*2 is a multiple of 8).
  const int col0 = base_col + 4 * c4;
  const float xs = x_scale[row];
  const float4 ws = *reinterpret_cast<const float4*>(weight_scale + col0);
  const float v0 = static_cast<float>(f0) * xs * ws.x;
  const float v1 = static_cast<float>(f1) * xs * ws.y;
  const float v2 = static_cast<float>(f2) * xs * ws.z;
  const float v3 = static_cast<float>(f3) * xs * ws.w;
  const uint64_t packed =
      static_cast<uint64_t>(
          static_cast<unsigned short>(__float2bfloat16(v0))) |
      (static_cast<uint64_t>(
           static_cast<unsigned short>(__float2bfloat16(v1)))
       << 16) |
      (static_cast<uint64_t>(
           static_cast<unsigned short>(__float2bfloat16(v2)))
       << 32) |
      (static_cast<uint64_t>(
           static_cast<unsigned short>(__float2bfloat16(v3)))
       << 48);
  *reinterpret_cast<uint64_t*>(out + row * n + col0) = packed;
}

// Large-M prefill kernel: 64x64 output tile per block, K staged in LDS
// (single buffer), four waves each owning a 32x32 quadrant = four
// m16n16k32 int8->int32 DUMMA accumulators.
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

  __shared__ __align__(16) int8_t a_tile[kBlockM * kStageK];
  __shared__ __align__(16) int8_t b_tile[kStageK * kBlockN];

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
    // Stage A[64,128] (row-major, zero-filled past M).
    constexpr int kAVectors = kBlockM * kStageK / static_cast<int>(sizeof(int4));
    for (int vec = tid; vec < kAVectors; vec += kThreadsPerBlock) {
      const int byte_offset = vec * static_cast<int>(sizeof(int4));
      const int local_row = byte_offset / kStageK;
      const int kk = byte_offset - local_row * kStageK;
      const int global_row = m0 + local_row;
      reinterpret_cast<int4*>(a_tile)[vec] =
          global_row < m
              ? *reinterpret_cast<const int4*>(
                    x_q + global_row * k + k0 + kk)
              : int4{0, 0, 0, 0};
    }

    // Stage B[128,64] (row-major [K, N] logical weight layout).
    constexpr int kBVectors = kStageK * kBlockN / static_cast<int>(sizeof(int4));
    for (int vec = tid; vec < kBVectors; vec += kThreadsPerBlock) {
      const int byte_offset = vec * static_cast<int>(sizeof(int4));
      const int kk = byte_offset / kBlockN;
      const int col = byte_offset - kk * kBlockN;
      reinterpret_cast<int4*>(b_tile)[vec] =
          *reinterpret_cast<const int4*>(
              weight + (k0 + kk) * n + n0 + col);
    }
    __syncthreads();

    // Consume the stage: each wave does four m16n16k32 MMAs per 32-K step.
    const int local_row = wave_row * 32;
    const int local_col = wave_col * 32;
#pragma unroll
    for (int kk = 0; kk < kStageK; kk += kTileK) {
      du_load_matrix_sync(
          a_frag0, a_tile + local_row * kStageK + kk, kStageK);
      du_load_matrix_sync(
          a_frag1, a_tile + (local_row + kTileM) * kStageK + kk, kStageK);
      du_load_matrix_sync(
          b_frag0, b_tile + kk * kBlockN + local_col, kBlockN);
      du_load_matrix_sync(
          b_frag1, b_tile + kk * kBlockN + local_col + kTileN, kBlockN);
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

// Large-M prefill kernel #2 (iteration 3, pipeline round): 64x128 output
// tile per block, 256 threads (4 wavefronts). Each wave owns a 32x64
// quadrant = eight m16n16k32 int32 accumulators (2 row halves x 4 column
// quarters). K is staged cooperatively in a SINGLE LDS buffer at 64-K
// granularity as the matched control for the mandated single-vs-double
// buffering comparison: it shares the iteration-2 double-buffered kernel's
// 64-K stage size, odd-word LDS strides (A: 68 B = 17 words, B: 132 B = 33
// words), 16-MMAC DUMMA burst, and 2-blocks/CU residency, but uses ONE
// 12,800-B buffer and TWO __syncthreads per stage (128 over K): the next
// stage's 3 x int4 (1 A + 2 B) are loaded from global only after the
// current burst and its all-consumption barrier, vmcnt-waited, and
// committed into the same buffer, then a store-visibility barrier. The
// global-load latency is fully exposed (no compute overlap), isolating the
// value of the double-buffer overlap at matched stage size and occupancy.
__global__ __launch_bounds__(kThreadsPerBlock) void
w8a8_dumma_prefill_64x128_kernel(
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
  const int m0 = static_cast<int>(blockIdx.y) * kBlockM128;
  const int n0 = static_cast<int>(blockIdx.x) * kBlockN128;

  __shared__ __align__(16) int8_t a_tile[kBlockM128 * kAStride128];
  __shared__ __align__(16) int8_t b_tile[kStageK128 * kBStride128];

  DUFragment<matrix_a, kTileM, kTileN, kTileK, signed char, row_major>
      a_frag0, a_frag1;
  DUFragment<matrix_b, kTileM, kTileN, kTileK, signed char, row_major>
      b_frag0, b_frag1, b_frag2, b_frag3;
  DUFragment<accumulator, kTileM, kTileN, kTileK, int>
      acc00, acc01, acc02, acc03, acc10, acc11, acc12, acc13;
  du_fill_fragment(acc00, 0);
  du_fill_fragment(acc01, 0);
  du_fill_fragment(acc02, 0);
  du_fill_fragment(acc03, 0);
  du_fill_fragment(acc10, 0);
  du_fill_fragment(acc11, 0);
  du_fill_fragment(acc12, 0);
  du_fill_fragment(acc13, 0);

  const int num_stages = k / kStageK128;

  // Cooperative staging mapping (fixed per thread, reused every stage):
  //   A[64,64]  -> 256 int4 vectors; thread `tid` owns vector `tid`
  //                (row = tid/4, 16-B column group = (tid%4)*16), so each
  //                wavefront covers 16 rows x 64 B contiguous per row.
  //   B[64,128] -> 512 int4 vectors; thread `tid` owns vectors `tid` and
  //                `tid+256` (kk = tid/8, column group = (tid%8)*16), so
  //                each wavefront covers 8 kk rows x 128 B contiguous.
  const int a_row = tid >> 2;
  const int a_col16 = (tid & 3) << 4;
  const int b_kk0 = tid >> 3;
  const int b_col16 = (tid & 7) << 4;

  // VGPR payload for one full 64-K stage (3 x int4 = 12 VGPR), loaded just
  // before it is committed (no one-stage-ahead overlap in this control).
  int4 vA, vB0, vB1;

  // Prologue: load stage 0 into registers, commit it to the single buffer,
  // and make it visible before the first burst.
  {
    const int g_row = m0 + a_row;
    vA = (g_row < m)
             ? *reinterpret_cast<const int4*>(x_q + g_row * k + a_col16)
             : int4{0, 0, 0, 0};
    vB0 = *reinterpret_cast<const int4*>(
        weight + b_kk0 * n + n0 + b_col16);
    vB1 = *reinterpret_cast<const int4*>(
        weight + (b_kk0 + kStageK128 / 2) * n + n0 + b_col16);
    // Commit as int32 stores (skips the 4-byte pad column; 4 x ds_write_b32
    // per 16-byte vector because the odd strides are 4 B mod 16).
    int32_t* adst = reinterpret_cast<int32_t*>(
        a_tile + a_row * kAStride128 + a_col16);
    adst[0] = vA.x;
    adst[1] = vA.y;
    adst[2] = vA.z;
    adst[3] = vA.w;
    int32_t* bdst0 = reinterpret_cast<int32_t*>(
        b_tile + b_kk0 * kBStride128 + b_col16);
    bdst0[0] = vB0.x;
    bdst0[1] = vB0.y;
    bdst0[2] = vB0.z;
    bdst0[3] = vB0.w;
    int32_t* bdst1 = bdst0 + (kStageK128 / 2) * (kBStride128 / 4);
    bdst1[0] = vB1.x;
    bdst1[1] = vB1.y;
    bdst1[2] = vB1.z;
    bdst1[3] = vB1.w;
    __syncthreads();
  }

  for (int s = 0; s < num_stages; ++s) {
    // Consume the single buffer: each wave does eight m16n16k32 MMAs per
    // 32-K step over its 32x64 quadrant.
    const int local_row = wave_row * 32;
    const int local_col = wave_col * 64;
#pragma unroll
    for (int kk = 0; kk < kStageK128; kk += kTileK) {
      du_load_matrix_sync(
          a_frag0, a_tile + local_row * kAStride128 + kk, kAStride128);
      du_load_matrix_sync(
          a_frag1, a_tile + (local_row + kTileM) * kAStride128 + kk,
          kAStride128);
      du_load_matrix_sync(
          b_frag0, b_tile + kk * kBStride128 + local_col, kBStride128);
      du_load_matrix_sync(
          b_frag1, b_tile + kk * kBStride128 + local_col + kTileN,
          kBStride128);
      du_load_matrix_sync(
          b_frag2, b_tile + kk * kBStride128 + local_col + 2 * kTileN,
          kBStride128);
      du_load_matrix_sync(
          b_frag3, b_tile + kk * kBStride128 + local_col + 3 * kTileN,
          kBStride128);
      du_mma_sync(acc00, a_frag0, b_frag0, acc00);
      du_mma_sync(acc01, a_frag0, b_frag1, acc01);
      du_mma_sync(acc02, a_frag0, b_frag2, acc02);
      du_mma_sync(acc03, a_frag0, b_frag3, acc03);
      du_mma_sync(acc10, a_frag1, b_frag0, acc10);
      du_mma_sync(acc11, a_frag1, b_frag1, acc11);
      du_mma_sync(acc12, a_frag1, b_frag2, acc12);
      du_mma_sync(acc13, a_frag1, b_frag3, acc13);
    }
    // All consumption of the single buffer is complete only after every
    // wavefront passes this barrier; the stage s+1 stores below overwrite
    // that same buffer, so they must wait for it.
    __syncthreads();

    // Load stage s+1 and commit it into the single buffer. The global-load
    // latency is fully exposed here: the loads are issued after the burst
    // (nothing to overlap) and the compiler's vmcnt wait before the DS
    // stores is on the critical path. `s + 1 < num_stages` is block-uniform,
    // so the branch and its barrier are free of divergence.
    if (s + 1 < num_stages) {
      const int s1 = (s + 1) * kStageK128;
      const int g_row = m0 + a_row;
      vA = (g_row < m)
               ? *reinterpret_cast<const int4*>(x_q + g_row * k + s1 + a_col16)
               : int4{0, 0, 0, 0};
      vB0 = *reinterpret_cast<const int4*>(
          weight + (s1 + b_kk0) * n + n0 + b_col16);
      vB1 = *reinterpret_cast<const int4*>(
          weight + (s1 + b_kk0 + kStageK128 / 2) * n + n0 + b_col16);
      int32_t* adst = reinterpret_cast<int32_t*>(
          a_tile + a_row * kAStride128 + a_col16);
      adst[0] = vA.x;
      adst[1] = vA.y;
      adst[2] = vA.z;
      adst[3] = vA.w;
      int32_t* bdst0 = reinterpret_cast<int32_t*>(
          b_tile + b_kk0 * kBStride128 + b_col16);
      bdst0[0] = vB0.x;
      bdst0[1] = vB0.y;
      bdst0[2] = vB0.z;
      bdst0[3] = vB0.w;
      int32_t* bdst1 = bdst0 + (kStageK128 / 2) * (kBStride128 / 4);
      bdst1[0] = vB1.x;
      bdst1[1] = vB1.y;
      bdst1[2] = vB1.z;
      bdst1[3] = vB1.w;
      // Make the stage s+1 stores visible to every wavefront before the
      // next burst.
      __syncthreads();
    }
  }

  const int base_row = m0 + wave_row * 32;
  const int base_col = n0 + wave_col * 64;
  store_prefill_fragment(
      acc00, x_scale, weight_scale, out, m, n,
      base_row, base_col, lane);
  store_prefill_fragment(
      acc01, x_scale, weight_scale, out, m, n,
      base_row, base_col + kTileN, lane);
  store_prefill_fragment(
      acc02, x_scale, weight_scale, out, m, n,
      base_row, base_col + 2 * kTileN, lane);
  store_prefill_fragment(
      acc03, x_scale, weight_scale, out, m, n,
      base_row, base_col + 3 * kTileN, lane);
  store_prefill_fragment(
      acc10, x_scale, weight_scale, out, m, n,
      base_row + kTileM, base_col, lane);
  store_prefill_fragment(
      acc11, x_scale, weight_scale, out, m, n,
      base_row + kTileM, base_col + kTileN, lane);
  store_prefill_fragment(
      acc12, x_scale, weight_scale, out, m, n,
      base_row + kTileM, base_col + 2 * kTileN, lane);
  store_prefill_fragment(
      acc13, x_scale, weight_scale, out, m, n,
      base_row + kTileM, base_col + 3 * kTileN, lane);
}

// Iteration 5 (packing round): explicit 8-byte loader for the col_major
// m16n16k32 B fragment (the gate_up-lineage-validated load_fragment8
// pattern). The col_major fragment mapping is n = lane&15,
// k = 8*(lane>>4) + i; with the n-major LDS tile at row stride 80 the
// lane's 8 elements are contiguous (p[row*ldm + col .. +7], 8-byte aligned
// because ldm=80 and col are multiples of 8), so this compiles to ONE
// ds_read2_b64 instead of 8 ds_read_u8 + mask/OR reassembly. The byte
// placement is identical to du_load_matrix_sync<col_major>, so the
// v_mmac_i32_16x16x32_i8 fragment registers receive the same values.
__device__ __forceinline__ void load_b_frag8(
    DUFragment<matrix_b, kTileM, kTileN, kTileK, signed char, col_major>& f,
    const int8_t* __restrict__ p,
    int ldm) {
  const unsigned row = static_cast<unsigned>(__lane_id()) & 0xfu;
  const unsigned col = (static_cast<unsigned>(__lane_id()) >> 4) << 3;
  const int64_t v = *reinterpret_cast<const int64_t*>(p + row * ldm + col);
  *reinterpret_cast<int64_t*>(&f.x[0]) = v;
}

// Large-M prefill kernel #3 (iteration 5, packing round): 64x128 macro-tile
// for the exact assigned shape (K=4096, N=2560) with the weight packed
// n-major. The pipeline is byte-identical to the accepted iteration-3 kernel
// (single-buffered 64-K stage, TWO __syncthreads per stage with fully
// exposed global-load latency, 3 x int4 VGPR payload, A staged row-major
// with the 68-byte odd-word stride); ONLY the B operand path changes:
//   * pack: the logical [K, N] weight is transposed once (outside the timed
//     region and Graph capture) to packed[n][k] by launch_pack_w8a8_weight.
//   * staging: thread tid owns rows n_local = tid>>2 and n_local+64, each
//     with the 16-byte k-run (tid&3)*16; two global_load_dwordx4 are
//     committed with ONE ds_write_b128 each (16-byte-aligned LDS address:
//     row stride 80 and the k16 offset are both 0 mod 16), vs four
//     ds_write_b32 per vector in the accepted kernel.
//   * fragments: load_b_frag8 (col_major mapping n = local_col + lane&15,
//     k = kk + 8*(lane>>4) + i) reads each lane's 8 elements as one
//     8-byte-aligned contiguous chunk -> one ds_read2_b64 per B fragment,
//     vs 8 byte-granular ds_read_u8 at stride 132 in the accepted kernel's
//     ISA. The mapping is the transpose of the accepted row_major mapping,
//     so the v_mmac_i32_16x16x32_i8 accumulation order is unchanged and the
//     int32 results are bit-identical. Bank safety: the 20-word row stride
//     spreads each k-group's 16 n-lanes over 8 distinct banks with five
//     bank phases.
//   * A path, epilogue, tail masking, and occupancy are unchanged.
//     LDS/block = 64x68 + 128x80 = 14,592 B -> 3 resident blocks/CU
//     (VGPR-bound at arch_vgpr ~80), same as the accepted kernel.
//
// Iteration 18 (fragment-pipeline round) strategy: keep the accepted
// iteration-6 packedb kernel byte-identical in EVERYTHING except the inner
// fragment schedule of the 64-K-stage compute loop. The accepted kernel's
// exact gfx928 ISA (iteration 7 evidence) shows the compiler already
// software-pipelines the kk=32 A fragment loads under the kk=0 MMAC burst,
// but its four kk=32 B-fragment ds_read_b64 loads are issued only AFTER
// MMACs 0-7, with s_waitcnt lgkmcnt(4..1) interleaved into MMACs 8-11 (16+
// lgkmcnt waits per stage; PMC lds_wait 6,560,572 = 1.18x lds_instructions,
// the dominant stall class of a SIMD ~36%-issue-busy kernel). This round
// hoists ALL twelve ds_read fragment loads of both 32-K steps before the
// first v_mmac of the stage (explicit two-step unroll of the kk loop with
// separate kk=32 fragment variables), so the LDS latency concentrates in one
// wait and the 16-MMAC stream has no lgkmcnt waits. The MMAC sequence per
// (m,n) output tile and the int32 k0-outer/kk-inner accumulation order are
// unchanged, so output bits stay identical. Occupancy target is retained:
// arch_vgpr 72 -> <= 85 at 3 blocks/CU (65,280 <= 65,536 VGPRs), LDS 14,592
// B, grid 1280 x 256, TWO __syncthreads per stage, staging mapping, packed
// B layout, coalesced epilogue, and the exact shape guard (m >= 128 &&
// n == 2560 && k == 4096) all unchanged; generic 64x128/64x64 kernels, the
// pack kernel, and the scalar fallback are untouched.
__global__ __launch_bounds__(kThreadsPerBlock) void
w8a8_dumma_prefill_64x128_packedb_kernel(
    const int8_t* __restrict__ x_q,
    const int8_t* __restrict__ packed_b,
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
  const int m0 = static_cast<int>(blockIdx.y) * kBlockM128;
  const int n0 = static_cast<int>(blockIdx.x) * kBlockN128;

  __shared__ __align__(16) int8_t a_tile[kBlockM128 * kAStride128];
  // n-major B tile: 128 n-rows of 80 bytes (64 data + 16 pad).
  __shared__ __align__(16) int8_t b_tile[kBlockN128 * kPackedBStride];

  DUFragment<matrix_a, kTileM, kTileN, kTileK, signed char, row_major>
      a_frag0, a_frag1;
  DUFragment<matrix_b, kTileM, kTileN, kTileK, signed char, col_major>
      b_frag0, b_frag1, b_frag2, b_frag3;
  DUFragment<accumulator, kTileM, kTileN, kTileK, int>
      acc00, acc01, acc02, acc03, acc10, acc11, acc12, acc13;
  du_fill_fragment(acc00, 0);
  du_fill_fragment(acc01, 0);
  du_fill_fragment(acc02, 0);
  du_fill_fragment(acc03, 0);
  du_fill_fragment(acc10, 0);
  du_fill_fragment(acc11, 0);
  du_fill_fragment(acc12, 0);
  du_fill_fragment(acc13, 0);

  const int num_stages = k / kStageK128;

  // A staging mapping unchanged (row-major x_q, odd-word stride 68).
  const int a_row = tid >> 2;
  const int a_col16 = (tid & 3) << 4;
  // B staging mapping for packed[n][k]: thread tid owns the 16-byte k-run
  // (b_k16..b_k16+15) of n-rows b_n and b_n + 64.
  const int b_n = tid >> 2;
  const int b_k16 = (tid & 3) << 4;

  // VGPR payload for one full 64-K stage (3 x int4 = 12 VGPR).
  int4 vA, vB0, vB1;

  // Prologue: load stage 0 into registers, commit it to the single buffers,
  // and make it visible before the first burst.
  {
    const int g_row = m0 + a_row;
    vA = (g_row < m)
             ? *reinterpret_cast<const int4*>(x_q + g_row * k + a_col16)
             : int4{0, 0, 0, 0};
    vB0 = *reinterpret_cast<const int4*>(
        packed_b + (n0 + b_n) * k + b_k16);
    vB1 = *reinterpret_cast<const int4*>(
        packed_b + (n0 + b_n + kBlockN128 / 2) * k + b_k16);
    // Commit A as int32 stores (68-byte rows are 4 mod 16 -> no b128).
    int32_t* adst = reinterpret_cast<int32_t*>(
        a_tile + a_row * kAStride128 + a_col16);
    adst[0] = vA.x;
    adst[1] = vA.y;
    adst[2] = vA.z;
    adst[3] = vA.w;
    // Commit B as 16-byte vector stores: b_n*80 + b_k16 is 0 mod 16.
    *reinterpret_cast<int4*>(b_tile + b_n * kPackedBStride + b_k16) = vB0;
    *reinterpret_cast<int4*>(
        b_tile + (b_n + kBlockN128 / 2) * kPackedBStride + b_k16) = vB1;
    __syncthreads();
  }

  for (int s = 0; s < num_stages; ++s) {
    // Consume the single buffer: each wave does eight m16n16k32 MMAs per
    // 32-K step over its 32x64 quadrant.
    const int local_row = wave_row * 32;
    const int local_col = wave_col * 64;
    const int8_t* abase = a_tile + local_row * kAStride128;
    const int8_t* bbase = b_tile + local_col * kPackedBStride;
    // Iteration 18 (fragment-pipeline round): software-pipeline the kk=32
    // fragment loads UNDER the kk=0 MMAC burst. The accepted kernel's exact
    // ISA already prefetches the kk=32 A fragments this way, but its four
    // kk=32 B ds_read_b64 loads are issued only AFTER MMACs 0-7, with
    // s_waitcnt lgkmcnt(4..1) interleaved into MMACs 8-11 (16+ lgkmcnt waits
    // per stage; PMC lds_wait 6,560,572 = 1.18x lds_instructions). Loading
    // both 32-K steps' operands before the first v_mmac concentrates the LDS
    // latency into one wait and leaves the 16-MMAC stream free of lgkmcnt
    // waits. Same 12 ds_read instructions per stage, same fragment values,
    // same int32 k0-outer/kk-inner accumulation order (the MMAC sequence per
    // output tile is unchanged) -> bit-identical results.
    du_load_matrix_sync(a_frag0, abase, kAStride128);
    du_load_matrix_sync(a_frag1, abase + kTileM * kAStride128, kAStride128);
    load_b_frag8(b_frag0, bbase, kPackedBStride);
    load_b_frag8(b_frag1, bbase + kTileN * kPackedBStride, kPackedBStride);
    load_b_frag8(b_frag2, bbase + 2 * kTileN * kPackedBStride, kPackedBStride);
    load_b_frag8(b_frag3, bbase + 3 * kTileN * kPackedBStride, kPackedBStride);
    DUFragment<matrix_a, kTileM, kTileN, kTileK, signed char, row_major>
        a2_frag0, a2_frag1;
    DUFragment<matrix_b, kTileM, kTileN, kTileK, signed char, col_major>
        b2_frag0, b2_frag1, b2_frag2, b2_frag3;
    du_load_matrix_sync(a2_frag0, abase + kTileK, kAStride128);
    du_load_matrix_sync(
        a2_frag1, abase + kTileM * kAStride128 + kTileK, kAStride128);
    load_b_frag8(b2_frag0, bbase + kTileK, kPackedBStride);
    load_b_frag8(
        b2_frag1, bbase + kTileN * kPackedBStride + kTileK, kPackedBStride);
    load_b_frag8(
        b2_frag2, bbase + 2 * kTileN * kPackedBStride + kTileK,
        kPackedBStride);
    load_b_frag8(
        b2_frag3, bbase + 3 * kTileN * kPackedBStride + kTileK,
        kPackedBStride);
    // kk = 0 burst (eight m16n16k32 MMAs).
    du_mma_sync(acc00, a_frag0, b_frag0, acc00);
    du_mma_sync(acc01, a_frag0, b_frag1, acc01);
    du_mma_sync(acc02, a_frag0, b_frag2, acc02);
    du_mma_sync(acc03, a_frag0, b_frag3, acc03);
    du_mma_sync(acc10, a_frag1, b_frag0, acc10);
    du_mma_sync(acc11, a_frag1, b_frag1, acc11);
    du_mma_sync(acc12, a_frag1, b_frag2, acc12);
    du_mma_sync(acc13, a_frag1, b_frag3, acc13);
    // kk = 32 burst (same accumulators, same MMAC order as the accepted
    // two-iteration kk loop).
    du_mma_sync(acc00, a2_frag0, b2_frag0, acc00);
    du_mma_sync(acc01, a2_frag0, b2_frag1, acc01);
    du_mma_sync(acc02, a2_frag0, b2_frag2, acc02);
    du_mma_sync(acc03, a2_frag0, b2_frag3, acc03);
    du_mma_sync(acc10, a2_frag1, b2_frag0, acc10);
    du_mma_sync(acc11, a2_frag1, b2_frag1, acc11);
    du_mma_sync(acc12, a2_frag1, b2_frag2, acc12);
    du_mma_sync(acc13, a2_frag1, b2_frag3, acc13);
    // All consumption of the single buffer is complete only after every
    // wavefront passes this barrier; the stage s+1 stores below overwrite
    // that same buffer, so they must wait for it.
    __syncthreads();

    // Load stage s+1 and commit it into the single buffer (fully exposed
    // global-load latency, exactly as in the accepted iteration-3 kernel).
    if (s + 1 < num_stages) {
      const int s1 = (s + 1) * kStageK128;
      const int g_row = m0 + a_row;
      vA = (g_row < m)
               ? *reinterpret_cast<const int4*>(x_q + g_row * k + s1 + a_col16)
               : int4{0, 0, 0, 0};
      vB0 = *reinterpret_cast<const int4*>(
          packed_b + (n0 + b_n) * k + s1 + b_k16);
      vB1 = *reinterpret_cast<const int4*>(
          packed_b + (n0 + b_n + kBlockN128 / 2) * k + s1 + b_k16);
      int32_t* adst = reinterpret_cast<int32_t*>(
          a_tile + a_row * kAStride128 + a_col16);
      adst[0] = vA.x;
      adst[1] = vA.y;
      adst[2] = vA.z;
      adst[3] = vA.w;
      *reinterpret_cast<int4*>(b_tile + b_n * kPackedBStride + b_k16) = vB0;
      *reinterpret_cast<int4*>(
          b_tile + (b_n + kBlockN128 / 2) * kPackedBStride + b_k16) = vB1;
      // Make the stage s+1 stores visible to every wavefront before the
      // next burst.
      __syncthreads();
    }
  }

  const int base_row = m0 + wave_row * 32;
  const int base_col = n0 + wave_col * 64;
  // Iteration 6 (epilogue round): coalesced stores -- register 4x4
  // transpose (8 shfl_xor -> ds_bpermute + 8 v_cndmask, no staging tile)
  // then ONE 8-byte store per lane per fragment (4x fewer VMEM store
  // instructions, 100% store sector utilization vs 25% for the scalar
  // 2-byte stores).
  store_prefill_fragment_coalesced(
      acc00, x_scale, weight_scale, out, m, n,
      base_row, base_col, lane);
  store_prefill_fragment_coalesced(
      acc01, x_scale, weight_scale, out, m, n,
      base_row, base_col + kTileN, lane);
  store_prefill_fragment_coalesced(
      acc02, x_scale, weight_scale, out, m, n,
      base_row, base_col + 2 * kTileN, lane);
  store_prefill_fragment_coalesced(
      acc03, x_scale, weight_scale, out, m, n,
      base_row, base_col + 3 * kTileN, lane);
  store_prefill_fragment_coalesced(
      acc10, x_scale, weight_scale, out, m, n,
      base_row + kTileM, base_col, lane);
  store_prefill_fragment_coalesced(
      acc11, x_scale, weight_scale, out, m, n,
      base_row + kTileM, base_col + kTileN, lane);
  store_prefill_fragment_coalesced(
      acc12, x_scale, weight_scale, out, m, n,
      base_row + kTileM, base_col + 2 * kTileN, lane);
  store_prefill_fragment_coalesced(
      acc13, x_scale, weight_scale, out, m, n,
      base_row + kTileM, base_col + 3 * kTileN, lane);
}

// Generic scalar int8/int32 fallback for unmatched shapes and small-M API
// cases. One output element per grid-stride step; exact int32 accumulation
// (K <= 4096 keeps the int8 dot well within int32 range), then the fused
// float scale and bf16 store.
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
    // Iteration 5 (packing round): the exact assigned shape (K=4096,
    // N=2560) uses the packed n-major layout packed[n][k] (set up once by
    // launch_pack_w8a8_weight); every other (k, n) keeps the logical
    // [K, N] row-major layout. The branch is grid-uniform per launch.
    const bool packed_b = (k == kPackedBK && n == kPackedBN);
    const int8_t* b_col =
        packed_b ? b + static_cast<int64_t>(col) * k : b + col;
    int32_t acc = 0;
    for (int kk = 0; kk < k; ++kk) {
      const int32_t bw = packed_b
                             ? static_cast<int32_t>(b_col[kk])
                             : static_cast<int32_t>(
                                   b_col[static_cast<int64_t>(kk) * n]);
      acc += static_cast<int32_t>(a_row[kk]) * bw;
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

// Iteration 5 (packing round): one-time n-major transpose of the logical
// [K, N] row-major weight into packed[n][k] for the exact assigned shape
// (K=4096, N=2560). Runs only from launch_pack_w8a8_weight, outside the
// timed region and outside Graph capture. The packed buffer keeps the same
// byte count (k*n) as the identity pack, so allocations and graph-stable
// addresses are unchanged. Element (k, n) of the logical weight lands at
// packed[n * K + k]; each thread copies one 16-byte k-run (coalesced read
// side; the strided write side is off the critical path). The guard
// guarantees n % 16 == 0, so every 16-byte chunk lies inside one logical
// row and the transpose is exact.
__global__ __launch_bounds__(256) void w8a8_pack_nmajor_b_kernel(
    const int8_t* __restrict__ src,
    int8_t* __restrict__ dst,
    int k,
    int n) {
  const int64_t chunks = (static_cast<int64_t>(k) * n) >> 4;
  const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t c = static_cast<int64_t>(blockIdx.x) * blockDim.x +
                   threadIdx.x;
       c < chunks;
       c += stride) {
    const int64_t off = c << 4;
    const int kk = static_cast<int>(off / n);
    const int col = static_cast<int>(off - static_cast<int64_t>(kk) * n);
    const int4 v = *reinterpret_cast<const int4*>(src + off);
    const int8_t* bytes = reinterpret_cast<const int8_t*>(&v);
#pragma unroll
    for (int i = 0; i < 16; ++i) {
      // Logical element (k = kk, n = col + i) lands at packed[(col+i)*K+kk].
      dst[static_cast<int64_t>(col + i) * k + kk] = bytes[i];
    }
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

  // Iteration 5 (packing round): exact assigned shape (K=4096, N=2560)
  // routes to the packed n-major B variant of the 64x128 kernel. The guard
  // is exact (m >= 128, n == 2560, k == 4096) and sits BEFORE the generic
  // 64x128 path; every other shape keeps its existing path (generic 64x128
  // for n % 128 == 0, 64x64 for n % 64 == 0, scalar fallback otherwise) and
  // the identity-packed layout.
  if (m >= 128 && n == kPackedBN && k == kPackedBK) {
    const dim3 grid(
        static_cast<unsigned>(n / kBlockN128),
        static_cast<unsigned>((m + kBlockM128 - 1) / kBlockM128));
    const dim3 block(static_cast<unsigned>(kThreadsPerBlock));
    hipLaunchKernelGGL(
        w8a8_dumma_prefill_64x128_packedb_kernel,
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

  // Native INT8 DUMMA m16n16k32 prefill path (2-D macro-tile 64x128,
  // single-buffered 64-K K stage; pipeline-round control arm) for every
  // large-M shape with compatible geometry (covers the assigned M=4096,
  // N=2560, K=4096 shape: 2560 % 128 == 0). The 128-K guard is written
  // explicitly so the dispatch condition is bit-identical to the accepted
  // iteration-1 kernel even though the internal K stage is now 64
  // (single-buffered).
  if (m >= 128 && (n % kBlockN128) == 0 && (k % 128) == 0) {
    const dim3 grid(
        static_cast<unsigned>(n / kBlockN128),
        static_cast<unsigned>((m + kBlockM128 - 1) / kBlockM128));
    const dim3 block(static_cast<unsigned>(kThreadsPerBlock));
    hipLaunchKernelGGL(
        w8a8_dumma_prefill_64x128_kernel,
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

  // Native INT8 DUMMA m16n16k32 prefill path (64x64 tile) for large-M shapes
  // whose N is not a multiple of 128 but is a multiple of 64.
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

// Stable host launch symbol consumed by csrc/bindings.cpp. For the exact
// assigned shape (K=4096, N=2560) it performs the one-time n-major B pack
// (logical [K, N] -> packed[n][k]) outside the timed region and outside
// Graph capture; every other (K, N) keeps the identity device-to-device
// copy, valid for every (K, N). The packed buffer size is k*n in both
// cases, so the allocation and graph-stable packed layout are unchanged.
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
    if (k == kPackedBK && n == kPackedBN) {
      const int64_t chunks = weight_count >> 4;
      int64_t blocks = (chunks + kPackThreads - 1) / kPackThreads;
      if (blocks > 4096) {
        blocks = 4096;
      }
      hipLaunchKernelGGL(
          w8a8_pack_nmajor_b_kernel,
          dim3(static_cast<unsigned>(blocks)),
          dim3(kPackThreads),
          0,
          stream,
          raw_weight,
          packed_weight,
          k,
          n);
    } else {
      int64_t blocks = (weight_count + kPackThreads - 1) / kPackThreads;
      if (blocks > 4096) {
        blocks = 4096;
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
  }
  if (n > 0) {
    int64_t blocks = (static_cast<int64_t>(n) + kPackThreads - 1) /
                     kPackThreads;
    if (blocks > 4096) {
      blocks = 4096;
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
