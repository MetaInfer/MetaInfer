// @@variant shape=hy3_tp8_qkv_proj_m4096 commit=54bce94f966dd5cb4dc5962db5258abd9fb392dc added=2026-08-27
//   median_us=418.2 p90_us=431.6 speedup=46.34 baseline_us=1.938e+04
//   source=hy3-dsh-tp8-m4096-1-0ebb994d
// MetaInfer W8A8 INT8 GEMM for gfx928 (K500SM_AI).
//
// Worker: worker_0 (physical GPU 0)
// Assigned shape: hy3_tp8_qkv_proj_m4096  (M=4096, N=1280, K=4096)
//
// Operator contract (fixed by the control plane):
//   out[m, n] = bf16( int32_dot(x_q[m, :], raw_weight[:, n])
//                     * x_scale[m] * weight_scale[n] )
//
// Iteration 1 (DUMMA throughput baseline round): port the TP4-validated
// 64x128 packed-n-major-B architecture (113.5 TOPS on the sibling M=4096,
// N=2560, K=4096 shape) to the assigned TP8 shape as a templated 2-D
// macro-tile family, and benchmark the three mandated block tiles (64x64,
// 64x128, 128x64).
//   * The timed exact shape (m >= 128, n == 1280, k == 4096) dispatches the
//     <64,128> instantiation: 256 threads = 4 wavefronts, each wave owns a
//     32x64 quadrant = eight m16n16k32 int32 accumulator fragments,
//     single-buffered 64-K LDS stage with TWO __syncthreads per stage,
//     3 resident blocks/CU (LDS 14,592 B/block, ~80 VGPR).
//   * B is packed once (outside the timed region and Graph capture) to an
//     n-major packed[n][k] layout for the exact (K=4096, N=1280): the LDS B
//     tile rows are 80 B (64 data + 16 pad, five bank phases) and every
//     m16n16k32 col_major B fragment is loaded with the explicit 8-byte
//     loader load_b_frag8 (one ds_read2_b64 per fragment instead of 8
//     byte-granular reads at stride 132), while every 16-byte staging vector
//     commits with one ds_write_b128.
//   * The 64-K stage's two 32-K steps have ALL fragment loads hoisted before
//     the first v_mmac (the TP4 iter-18 schedule), concentrating LDS latency
//     into one wait per stage and leaving the 16-MMAC stream free of
//     lgkmcnt waits.
//   * The epilogue is the TP4 iter-6 coalesced store: a register 4x4
//     transpose (8 shfl_xor + 8 v_cndmask) puts four contiguous bf16 columns
//     per lane, converted and stored as ONE 8-byte store per lane per
//     fragment (100% store sector efficiency vs 25% for the scalar 2-byte
//     stores). Iteration 6 (epilogue round) keeps this structure but peels
//     the final K stage so the per-row/per-column scales are prefetched into
//     registers once per lane (2 scalar + 4 float4 loads instead of 8 + 8
//     per-fragment reloads) after the last MMAC burst and the provably dead
//     final all-consumption barrier is dropped (127 barriers per block); the
//     per-element multiply order and bf16 rounding are unchanged, so output
//     bits are identical. The timed path was audited for this round: scales,
//     bf16 conversion and the final coalesced store are already fused inside
//     the compute kernel and launch_w8a8_gemm ignores the workspace argument
//     (no workspace/combine pass exists to remove; the only D2D copy is the
//     one-time pack-time weight_scale copy, outside the timed region and
//     outside Graph capture, required by the immutable API).
//   * The <64,64> (LDS 9,472 B, ~4 blocks/CU) and <128,64> (LDS 13,824 B,
//     ~2-3 blocks/CU) instantiations are compiled alongside and are
//     benchmark candidates for later rounds; the exact-shape dispatch stays
//     on <64,128> this round.
//   * Generic arms preserved byte-for-byte: the 64x128 identity-layout
//     kernel for large-M shapes with N % 128 == 0 (k % 64 == 0), the 64x64
//     identity-layout kernel for N % 64 == 0, and the scalar int8/int32
//     grid-stride fallback for every other (m, n, k), including the paired
//     M=2 API shape (which decodes the packed n-major layout when
//     (k, n) == (4096, 1280)).
//   * launch_pack_w8a8_weight transposes the logical [K, N] weight to
//     packed[n][k] for the exact (k, n) == (4096, 1280) and keeps the
//     identity D2D copy for every other (K, N). Same byte count k*n, same
//     allocation, graph-stable addresses.
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
constexpr int kStageK = 128;  // K stage of the 64x64 kernel
// 4 wavefronts; must remain a multiple of the gfx928 wavefront size (64).
constexpr int kThreadsPerBlock = 256;

// 64x128 macro-tile, single-buffered 64-K K stage. LDS row strides are
// padded to odd word counts (A: 68 B = 17 words, B: 132 B = 33 words) so the
// INT8 m16n16k32 fragment loads spread across LDS banks instead of
// collapsing onto a single bank phase (128-byte rows would be 0 mod 32
// words -> single-bank collisions).
constexpr int kBlockM128 = 64;
constexpr int kBlockN128 = 128;
constexpr int kStageK128 = 64;
constexpr int kAStride128 = kStageK128 + 4;  // 68 bytes per A row (17 words)
constexpr int kBStride128 = kBlockN128 + 4;  // 132 bytes per B row (33 words)

// Iteration 1 (throughput baseline): packed n-major B layout for the exact
// assigned shape (K=4096, N=1280). B is transposed once, outside timing, to
// packed[n][k]; the packed kernel stages B into an n-major LDS tile with a
// 16-byte-aligned non-power-of-two row stride (80 B = 20 words, five bank
// phases) so every DUMMA m16n16k32 B fragment load (col_major) reads its
// lane's 8 bytes contiguously (ds_read2_b64) and every 16-byte staging
// vector commits with one ds_write_b128.
constexpr int kPackedBStride = 80;  // 64 data + 16 pad bytes (20 words)
constexpr int kPackedBK = 4096;     // exact K of the packed assigned shape
constexpr int kPackedBN = 1280;     // exact N of the packed assigned shape

using namespace du::dumma;

// Direct accumulator epilogue for gfx928 INT8 m16n16k32 (verified against
// du_store_matrix_sync): row = lane & 15, col_mod4 = lane >> 4,
// frag.x[i] maps to columns col_mod4 + 4*i. Stores
// bf16(float(dot) * x_scale[row] * weight_scale[col]); out-of-range rows are
// masked (tail-M handling). The float multiply order (dot * x_scale, then
// * weight_scale) matches the reference exactly.
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

// Iteration 1 (throughput baseline): coalesced fragment store. The m16n16k32
// accumulator lane mapping (row = lane & 15, column group c4 = lane >> 4,
// frag.x[i] -> column c4 + 4*i) gives each lane four elements strided by 4
// columns, so the direct store (store_prefill_fragment) is four 2-byte
// scalar stores per lane whose wavefront addresses touch each 32-B sector at
// 25% utilization. This epilogue transposes the 4-element groups within each
// 4-lane column group (lanes r, r+16, r+32, r+48 -- a 4x4 transpose, two 2x2
// steps with shfl_xor 16 then 32, one v_cndmask per element per step; no
// staging tile round trip), so lane (r, c4) ends up holding the four
// CONTIGUOUS columns 4*c4 .. 4*c4+3, converts them to bf16, packs 4 bf16
// (8 B), and writes ONE 8-byte store per lane: 64 lanes x 8 B = 512 B per
// fragment per wavefront in 16 fully-used 32-B sectors (100% store sector
// efficiency; vmem_write_instructions for the assigned shape drop from
// 81,920 to 20,480). Only the int32 values are re-routed between lanes --
// the per-element scale multiply order (float(dot) * x_scale[row] *
// weight_scale[col]) and the bf16 rounding are unchanged, so the stored bits
// are identical to store_prefill_fragment. The row>=m guard is
// wavefront-uniform (lane & 15 cycles the same 16 rows in every 16-lane
// group), so the shuffles never mix active and inactive lanes; base_col is a
// multiple of 64 and n*2 a multiple of 8, so the float4 weight_scale load
// (col0 % 4 == 0) and the 8-byte store are aligned.
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

// Iteration 6 (epilogue round): register-scale variant of the coalesced
// store. The per-row scale (xs) and the 4 per-column scales (ws) are loaded
// ONCE per lane by the peeled final stage of the packed kernel (2 scalar + 4
// float4 loads instead of 8 + 8 per-fragment reloads) and passed in; the
// per-element multiply order (float(dot) * xs * ws.i) and the bf16 rounding
// are byte-identical to store_prefill_fragment_coalesced, so the stored
// output bits are unchanged. The row>=m guard stays inside (tail-M blocks:
// the guard is uniform across each 4-lane row group, so the shuffle
// transpose never mixes active and inactive lanes).
template <typename AccFragment>
__device__ __forceinline__ void store_prefill_fragment_coalesced_rs(
    const AccFragment& frag,
    float xs,
    const float4 ws,
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

// Iteration 1 (throughput baseline): explicit 8-byte loader for the
// col_major m16n16k32 B fragment. The col_major fragment mapping is
// n = lane&15, k = 8*(lane>>4) + i; with the n-major LDS tile at row stride
// 80 the lane's 8 elements are contiguous (p[row*ldm + col .. +7], 8-byte
// aligned because ldm=80 and col are multiples of 8), so this compiles to
// ONE ds_read2_b64 instead of 8 ds_read_u8 + mask/OR reassembly. The byte
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

// Iteration 1 (throughput baseline): templated 2-D macro-tile packed-B
// prefill kernel (the TP4-validated 64x128 architecture generalized to the
// mandated tile set {64x64, 64x128, 128x64}). 256 threads = 4 wavefronts;
// each wave owns a (BM/2) x (BN/2) quadrant of m16n16k32 int32 accumulator
// fragments. K is staged cooperatively in a SINGLE LDS buffer at 64-K
// granularity with TWO __syncthreads per stage (iteration 6 peels the final
// stage and drops its provably dead all-consumption barrier: 127 barriers
// per block instead of 128):
//   * A is staged row-major into a_tile[BM][68] (odd-word stride: 17 words,
//     five bank phases); global-load latency is fully exposed, exactly as in
//     the accepted TP4 pipeline (co-resident blocks hide it).
//   * B is staged n-major from packed[n][k] into b_tile[BN][80] (20 words,
//     five bank phases): each 16-byte staging vector commits with one
//     ds_write_b128 (b_n*80 + b_k16 is 0 mod 16).
//   * The 64-K stage is split into two explicit 32-K steps and ALL fragment
//     loads (kWaveM16 A + kWaveN16 B per step, two steps) are hoisted before
//     the first v_mmac of the stage, so LDS latency concentrates in one wait
//     and the MMAC stream has no lgkmcnt waits.
//   * The int32 accumulation order (k0-outer over 64-K stages, kk-inner
//     kk=0 then kk=32 within a stage) and the element-to-slot fragment
//     mapping are unchanged, so the results are bit-identical to the scalar
//     reference.
//   * The epilogue is the coalesced store: iteration 1
//     (store_prefill_fragment_coalesced); iteration 6 hoists the per-row and
//     per-column scales into registers loaded once per lane after the peeled
//     final stage's MMAC burst and stores via
//     store_prefill_fragment_coalesced_rs (same math order, same stored
//     bits).
template <int BM, int BN>
__global__ __launch_bounds__(kThreadsPerBlock) void
w8a8_dumma_prefill_packedb_tiled_kernel(
    const int8_t* __restrict__ x_q,
    const int8_t* __restrict__ packed_b,
    const float* __restrict__ x_scale,
    const float* __restrict__ weight_scale,
    hip_bfloat16* __restrict__ out,
    int m,
    int n,
    int k) {
  // m16n16k32 accumulator tiles per wave quadrant: rows BM/32, cols BN/32.
  constexpr int kWaveM16 = BM / 32;
  constexpr int kWaveN16 = BN / 32;
  // 16-byte staging vectors per thread (1 for 64-row/64-col sides, 2 for
  // 128-row/128-col sides).
  constexpr int kAVecsPerThread =
      (BM * kStageK128 / static_cast<int>(sizeof(int4))) / kThreadsPerBlock;
  constexpr int kBVecsPerThread =
      (BN * kStageK128 / static_cast<int>(sizeof(int4))) / kThreadsPerBlock;
  static_assert(BM % 32 == 0 && BN % 32 == 0,
                "block tile must be a multiple of the wave quadrant");
  static_assert(kAVecsPerThread >= 1 && kBVecsPerThread >= 1,
                "staging must fit the block thread count");

  const int tid = static_cast<int>(threadIdx.x);
  const int wave = tid / kWaveSize;
  const int lane = tid % kWaveSize;
  const int wave_row = wave >> 1;  // 0 or 1
  const int wave_col = wave & 1;   // 0 or 1
  const int m0 = static_cast<int>(blockIdx.y) * BM;
  const int n0 = static_cast<int>(blockIdx.x) * BN;

  __shared__ __align__(16) int8_t a_tile[BM * kAStride128];
  __shared__ __align__(16) int8_t b_tile[BN * kPackedBStride];

  DUFragment<matrix_a, kTileM, kTileN, kTileK, signed char, row_major>
      a_frag[kWaveM16][2];
  DUFragment<matrix_b, kTileM, kTileN, kTileK, signed char, col_major>
      b_frag[kWaveN16][2];
  DUFragment<accumulator, kTileM, kTileN, kTileK, int>
      acc[kWaveM16][kWaveN16];
#pragma unroll
  for (int j = 0; j < kWaveM16; ++j) {
#pragma unroll
    for (int i = 0; i < kWaveN16; ++i) {
      du_fill_fragment(acc[j][i], 0);
    }
  }

  const int num_stages = k / kStageK128;

  // Cooperative staging mapping (fixed per thread, reused every stage):
  //   A[BM,64]: vector v = tid + t*256 owns row v>>2 and 16-B column group
  //             (v&3)*16; each wavefront covers 16 rows x 64 contiguous B.
  //   B[BN,64]: n-major; vector v = tid + t*256 owns packed row (n0 + v>>2)
  //             and the 16-byte k-run (v&3)*16.
  int4 vA[kAVecsPerThread];
  int4 vB[kBVecsPerThread];

  // Prologue: load stage 0 into registers, commit it to the single buffers,
  // and make it visible before the first burst.
  {
#pragma unroll
    for (int t = 0; t < kAVecsPerThread; ++t) {
      const int v = tid + t * kThreadsPerBlock;
      const int a_row = v >> 2;
      const int a_k16 = (v & 3) << 4;
      const int g_row = m0 + a_row;
      vA[t] = (g_row < m)
                  ? *reinterpret_cast<const int4*>(x_q + g_row * k + a_k16)
                  : int4{0, 0, 0, 0};
      // Commit A as int32 stores (68-byte rows are 4 mod 16 -> no b128).
      int32_t* adst = reinterpret_cast<int32_t*>(
          a_tile + a_row * kAStride128 + a_k16);
      adst[0] = vA[t].x;
      adst[1] = vA[t].y;
      adst[2] = vA[t].z;
      adst[3] = vA[t].w;
    }
#pragma unroll
    for (int t = 0; t < kBVecsPerThread; ++t) {
      const int v = tid + t * kThreadsPerBlock;
      const int b_n = v >> 2;
      const int b_k16 = (v & 3) << 4;
      vB[t] = *reinterpret_cast<const int4*>(
          packed_b + (n0 + b_n) * k + b_k16);
      // Commit B as 16-byte vector stores: b_n*80 + b_k16 is 0 mod 16.
      *reinterpret_cast<int4*>(b_tile + b_n * kPackedBStride + b_k16) =
          vB[t];
    }
    __syncthreads();
  }

  // Stage consumption shared by the loop body and the peeled final stage:
  // hoist all fragment loads of both 32-K steps before the first v_mmac,
  // then run the MMAC bursts.
  auto consume_stage = [&]() {
    const int local_row = wave_row * (BM / 2);
    const int local_col = wave_col * (BN / 2);
    const int8_t* abase = a_tile + local_row * kAStride128;
    const int8_t* bbase = b_tile + local_col * kPackedBStride;
#pragma unroll
    for (int st = 0; st < 2; ++st) {
      const int kk = st * kTileK;
#pragma unroll
      for (int j = 0; j < kWaveM16; ++j) {
        du_load_matrix_sync(
            a_frag[j][st], abase + j * kTileM * kAStride128 + kk,
            kAStride128);
      }
#pragma unroll
      for (int i = 0; i < kWaveN16; ++i) {
        load_b_frag8(
            b_frag[i][st],
            bbase + i * kTileN * kPackedBStride + kk, kPackedBStride);
      }
    }
    // kk = 0 then kk = 32 bursts (same accumulators, same MMAC order as the
    // scalar k-ascending reference).
#pragma unroll
    for (int st = 0; st < 2; ++st) {
#pragma unroll
      for (int j = 0; j < kWaveM16; ++j) {
#pragma unroll
        for (int i = 0; i < kWaveN16; ++i) {
          du_mma_sync(acc[j][i], a_frag[j][st], b_frag[i][st], acc[j][i]);
        }
      }
    }
  };

  for (int s = 0; s < num_stages - 1; ++s) {
    consume_stage();
    // All consumption of the single buffer is complete only after every
    // wavefront passes this barrier; the stage s+1 stores below overwrite
    // that same buffer, so they must wait for it.
    __syncthreads();

    // Load stage s+1 and commit it into the single buffer. The global-load
    // latency is fully exposed here (co-resident blocks hide it); the
    // compiler's vmcnt wait before the DS stores is on this path.
    if (s + 1 < num_stages) {
      const int s1 = (s + 1) * kStageK128;
#pragma unroll
      for (int t = 0; t < kAVecsPerThread; ++t) {
        const int v = tid + t * kThreadsPerBlock;
        const int a_row = v >> 2;
        const int a_k16 = (v & 3) << 4;
        const int g_row = m0 + a_row;
        vA[t] = (g_row < m)
                    ? *reinterpret_cast<const int4*>(
                          x_q + g_row * k + s1 + a_k16)
                    : int4{0, 0, 0, 0};
        int32_t* adst = reinterpret_cast<int32_t*>(
            a_tile + a_row * kAStride128 + a_k16);
        adst[0] = vA[t].x;
        adst[1] = vA[t].y;
        adst[2] = vA[t].z;
        adst[3] = vA[t].w;
      }
#pragma unroll
      for (int t = 0; t < kBVecsPerThread; ++t) {
        const int v = tid + t * kThreadsPerBlock;
        const int b_n = v >> 2;
        const int b_k16 = (v & 3) << 4;
        vB[t] = *reinterpret_cast<const int4*>(
            packed_b + (n0 + b_n) * k + s1 + b_k16);
        *reinterpret_cast<int4*>(b_tile + b_n * kPackedBStride + b_k16) =
            vB[t];
      }
      // Make the stage s+1 stores visible to every wavefront before the
      // next burst.
      __syncthreads();
    }
  }

  // Iteration 6 (epilogue round): peel the final stage. After its MMAC
  // burst no wavefront reads LDS again -- the epilogue touches only
  // registers and global memory -- so the final all-consumption barrier is
  // provably dead and is dropped (127 barriers per block instead of 128).
  // The per-row (xs) and per-column (ws) scale registers are prefetched
  // right after the last MMACs: 2 scalar + 4 float4 loads per lane instead
  // of the 8 + 8 per-fragment reloads, with the load latency hidden behind
  // the transpose/shuffle chains. The per-element multiply order and bf16
  // rounding are unchanged, so the stored bits are identical.
  consume_stage();

  const int base_row = m0 + wave_row * (BM / 2);
  const int base_col = n0 + wave_col * (BN / 2);
  const int lane_row = lane & 15;
  const int c4 = lane >> 4;
  float xs[kWaveM16];
#pragma unroll
  for (int j = 0; j < kWaveM16; ++j) {
    const int row = base_row + j * kTileM + lane_row;
    // Tail rows are clamped to 0.0f (never dereferenced); the store guard
    // below skips them, and the guard is uniform across each 4-lane row
    // group so the shuffle transpose never mixes active and inactive lanes.
    xs[j] = (row < m) ? x_scale[row] : 0.0f;
  }
  float4 ws[kWaveN16];
#pragma unroll
  for (int i = 0; i < kWaveN16; ++i) {
    // col0 = base_col + i*kTileN + 4*c4 is a multiple of 4 (base_col is a
    // multiple of 64), so the float4 load is 16-byte aligned.
    ws[i] = *reinterpret_cast<const float4*>(
        weight_scale + base_col + i * kTileN + 4 * c4);
  }
#pragma unroll
  for (int j = 0; j < kWaveM16; ++j) {
#pragma unroll
    for (int i = 0; i < kWaveN16; ++i) {
      store_prefill_fragment_coalesced_rs(
          acc[j][i], xs[j], ws[i], out, m, n,
          base_row + j * kTileM, base_col + i * kTileN, lane);
    }
  }
}

// Large-M prefill kernel #1: 64x128 output tile per block, 256 threads
// (4 wavefronts). Each wave owns a 32x64 quadrant = eight m16n16k32 int32
// accumulators (2 row halves x 4 column quarters). K is staged cooperatively
// in a SINGLE LDS buffer at 64-K granularity with TWO __syncthreads per
// stage (128 over K=4096): the next stage's 3 x int4 (1 A + 2 B vectors) are
// loaded from global only after the current stage's 16-MMAC DUMMA burst and
// its all-consumption barrier, vmcnt-waited, and committed into the same
// buffer, then a store-visibility barrier. The global-load latency is fully
// exposed (no compute overlap) -- the accepted direct/single-buffered
// bootstrap design.
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
  // before it is committed (no one-stage-ahead overlap in this bootstrap).
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

// Large-M prefill kernel #2: 64x64 output tile per block, K staged in LDS
// (single 128-K buffer), four waves each owning a 32x32 quadrant = four
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

// Generic scalar int8/int32 fallback for unmatched shapes and small-M API
// cases (including the paired M=2 shape). One output element per grid-stride
// step; exact int32 accumulation (K <= 4096 keeps the int8 dot well within
// int32 range), then the fused float scale and bf16 store. The exact packed
// shape (K=4096, N=1280) uses the packed n-major layout packed[n][k] (set up
// once by launch_pack_w8a8_weight); every other (k, n) keeps the logical
// [K, N] row-major layout. The branch is grid-uniform per launch.
__global__ __launch_bounds__(256) void w8a8_scalar_gemm_kernel(
    const int8_t* __restrict__ a,
    const int8_t* __restrict__ b,
    const float* __restrict__ x_scale,
    const float* __restrict__ weight_scale,
    hip_bfloat16* __restrict__ out,
    int m,
    int n,
    int k) {
  const bool packed_b = (k == kPackedBK && n == kPackedBN);
  const int64_t total = static_cast<int64_t>(m) * n;
  const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x +
                     threadIdx.x;
       idx < total;
       idx += stride) {
    const int row = static_cast<int>(idx / n);
    const int col = static_cast<int>(idx - static_cast<int64_t>(row) * n);
    const int8_t* a_row = a + static_cast<int64_t>(row) * k;
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

// Iteration 1 (throughput baseline): one-time n-major transpose of the
// logical [K, N] row-major weight into packed[n][k] for the exact assigned
// shape (K=4096, N=1280). Runs only from launch_pack_w8a8_weight, outside
// the timed region and outside Graph capture. The packed buffer keeps the
// same byte count (k*n) as the identity pack, so allocations and
// graph-stable addresses are unchanged. Element (k, n) of the logical weight
// lands at packed[n * K + k]; each thread copies one 16-byte k-run (coalesced
// read side; the strided write side is off the critical path). The guard
// guarantees n % 16 == 0, so every 16-byte chunk lies inside one logical row
// and the transpose is exact.
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

  // Iteration 1 (throughput baseline): exact assigned shape (K=4096,
  // N=1280) routes to the packed n-major B variant of the 2-D macro-tile
  // kernel, dispatched as the <64,128> instantiation. The guard is exact
  // (m >= 128, n == 1280, k == 4096) and sits BEFORE the generic 64x128
  // path; every other shape keeps its existing path (generic 64x128 for
  // n % 128 == 0, 64x64 for n % 64 == 0, scalar fallback otherwise) and the
  // identity-packed layout.
  if (m >= 128 && n == kPackedBN && k == kPackedBK) {
    const dim3 grid(
        static_cast<unsigned>(n / kBlockN128),
        static_cast<unsigned>((m + kBlockM128 - 1) / kBlockM128));
    const dim3 block(static_cast<unsigned>(kThreadsPerBlock));
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(w8a8_dumma_prefill_packedb_tiled_kernel<64, 128>),
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

  // Mandated 2-D macro-tile benchmark family: keep the <64,64> and <128,64>
  // instantiations of the packed kernel compiled (and correct) so later
  // rounds can flip the exact-shape dispatch without source surgery. This
  // branch can never run (m <= 0 already returned above); it only forces
  // template instantiation of the two sibling tiles.
  if (m < 0) {
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(w8a8_dumma_prefill_packedb_tiled_kernel<64, 64>),
        dim3(1),
        dim3(kThreadsPerBlock),
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
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(w8a8_dumma_prefill_packedb_tiled_kernel<128, 64>),
        dim3(1),
        dim3(kThreadsPerBlock),
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

  // Native INT8 DUMMA m16n16k32 prefill path #1 (64x128 macro-tile,
  // single-buffered 64-K K stage) for every large-M shape with compatible
  // geometry. Covers large-M shapes with N % 128 == 0 and K % 64 == 0 that
  // are NOT the packed exact shape (which returned above). Grid =
  // (N/128) x (ceil(M/64)).
  if (m >= 128 && (n % kBlockN128) == 0 && (k % kStageK128) == 0) {
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

  // Native INT8 DUMMA m16n16k32 prefill path #2 (64x64 tile) for large-M
  // shapes whose N is not a multiple of 128 but is a multiple of 64 (or
  // whose K is a multiple of 128 while not a multiple of 64).
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
// assigned shape (K=4096, N=1280) it performs the one-time n-major B pack
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
