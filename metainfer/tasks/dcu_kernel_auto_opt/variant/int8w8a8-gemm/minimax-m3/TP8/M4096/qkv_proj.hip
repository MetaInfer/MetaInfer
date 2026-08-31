// @@variant shape=minimax_tp8_qkv_proj_m4096 commit=76e2b8659429332829a877855ef1178692d453d5 added=2026-08-29
//   median_us=475.5 p90_us=476.5 speedup=86.22 baseline_us=4.1e+04
//   source=minimaxm3-dsh-tp8-m4096-1-b0482833
// MetaInfer W8A8 INT8 GEMM bootstrap for gfx928 (K500SM_AI).
//
// Worker: worker_0 (physical GPU 0)
// Assigned shape: minimax_tp8_qkv_proj_m4096  (M=4096, N=1280, K=6144)
//
// Operator contract (fixed by the control plane):
//   out[m, n] = bf16( int32_dot(x_q[m, :], raw_weight[:, n])
//                     * x_scale[m] * weight_scale[n] )
//
// Iteration 2 (2-D macro-tile DUMMA throughput baseline): the exact assigned
// shape routes to the templated packed-B 2-D macro-tile kernel
// w8a8_dumma_prefill_packedb_tiled_kernel<BM, BN>, dispatched as the
// <64, 128> instantiation (64x128 output tile per block, 256 threads =
// 4 wavefronts, each wave owns a 32x64 quadrant = eight m16n16k32 int32
// accumulator fragments). The mandated tile benchmark {64x64, 64x128,
// 128x64} is realized by the same template; the <64, 64> and <128, 64>
// instantiations stay compiled (dead-branch instantiation in
// launch_w8a8_gemm) so a later round can flip the exact-shape dispatch
// without source surgery. Per-tile resources (kStageK = 64, A stride 68 B,
// packed-B n-major stride 80 B, 4 waves/block in all three):
//   tile     LDS B/block   acc frags   arch VGPR est.   blocks/CU   grid
//   64x64       9,472          4            ~80            3-4       1280
//   64x128     14,592          8            ~96            2          640
//   128x64     13,824          8            ~96            2          640
// (<64,128> is the exact-shape timing candidate; the control plane verifies
// VGPR/LDS/occupancy from the code object and measures median/P90/TOPS.)
//
// B is packed once (outside the timed region and Graph capture) to an
// n-major packed[n][k] layout for the exact (K=6144, N=1280); the packed
// kernel stages B into an n-major LDS tile (row stride 80 B = 20 words, five
// bank phases) so every col_major m16n16k32 B fragment load is the explicit
// 8-byte loader load_b_frag8 (one ds_read2_b64 per fragment instead of 8
// ds_read_u8 + mask/OR reassembly) and every 16-byte staging vector commits
// with one ds_write_b128. A is staged row-major into a_tile[BM][68] (odd
// 17-word stride, five bank phases); global-load latency is fully exposed,
// exactly as in the accepted sibling TP8 qkv pipeline (co-resident blocks
// hide it).
//
// K is staged cooperatively in a SINGLE LDS buffer at 64-K granularity with
// TWO __syncthreads per stage; the final stage is peeled and its provably
// dead all-consumption barrier is dropped (191 barriers over K=6144). All
// fragment loads of both 32-K steps are hoisted before the first v_mmac so
// LDS latency concentrates in one wait and the MMAC stream has no lgkmcnt
// waits. The int32 accumulation order (k0-outer, kk-inner) and the
// element-to-slot fragment mapping are unchanged, so results are
// bit-identical to the scalar reference.
//
// The epilogue is the fused coalesced store: a register 4x4 shuffle
// transpose (one 8-byte store per lane per fragment, 100% store sector
// efficiency), with per-row (xs) and per-column (ws) scale registers hoisted
// once per lane after the final MMAC burst.
//
// Every unmatched (m, n, k) keeps the prior generic path: the 64x64
// single-buffered 128-K staged DUMMA kernel for large-M compatible shapes
// (identity [K, N] layout), and the scalar int8/int32 fallback, which
// decodes the packed n-major layout when (k, n) == (6144, 1280) (including
// the paired small-M API shapes with the same (N, K)).
//
// launch_pack_w8a8_weight packs (K=6144, N=1280) to packed[n][k] once,
// outside the timed region and outside Graph capture; every other (k, n)
// keeps the identity copy. The packed buffer keeps the same byte count
// (k*n), so allocations and graph-stable addresses are unchanged.
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
// 4 wavefronts; must remain a multiple of the gfx928 wavefront size (64).
constexpr int kThreadsPerBlock = 256;

// Generic 64x64 large-M path (unchanged bootstrap, identity [K, N] layout):
// single-buffered 128-K K stage with padded row strides.
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

// Packed-B 2-D macro-tile family (iteration 2, exact assigned shape):
// 64-K stage, A row stride 68 B (17 words, five bank phases), packed-B
// n-major row stride 80 B (20 words, five bank phases).
constexpr int kStageK128 = 64;
constexpr int kAStride128 = kStageK128 + 4;  // 68 bytes per A row
constexpr int kPackedBStride = 80;  // 64 data + 16 pad bytes (20 words)
constexpr int kPackedBK = 6144;     // exact K of the packed assigned shape
constexpr int kPackedBN = 1280;     // exact N of the packed assigned shape
// Default tile of the packed family used by the exact-shape dispatch.
constexpr int kBlockM128 = 64;
constexpr int kBlockN128 = 128;

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

// Iteration-2 register-scale coalesced store used by the packed-B macro-tile
// kernel. The m16n16k32 accumulator lane mapping (row = lane & 15, column
// group c4 = lane >> 4, frag.x[i] -> column c4 + 4*i) gives each lane four
// elements strided by 4 columns; a register 4x4 transpose (two 2x2 steps
// with shfl_xor 16 then 32) re-routes the four int32 values so lane (r, c4)
// owns the four CONTIGUOUS columns 4*c4 .. 4*c4+3, then converts them to
// bf16 and writes ONE 8-byte store per lane (64 lanes x 8 B = 512 B per
// fragment per wavefront in 16 fully-used 32-B sectors; vmem_write drops
// 4x). Only the int32 values are re-routed -- the per-element multiply order
// (float(dot) * xs * ws.i) and the bf16 rounding are unchanged, so the
// stored bits are identical to store_prefill_fragment. The row>=m guard is
// wavefront-uniform (lane & 15 cycles the same 16 rows in every 16-lane
// group), so the shuffles never mix active and inactive lanes; base_col is a
// multiple of 64 and n*2 a multiple of 8, so the float4 weight_scale load
// (col0 % 4 == 0) and the 8-byte store are aligned. The per-row (xs) and
// per-column (ws) scales are loaded ONCE per lane after the final MMAC burst
// by the caller and passed in (2 scalar + 4 float4 loads instead of the
// per-fragment reloads).
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

// Explicit 8-byte loader for the col_major m16n16k32 B fragment. The
// col_major fragment mapping is n = lane&15, k = 8*(lane>>4) + i; with the
// n-major LDS tile at row stride 80 the lane's 8 elements are contiguous
// (p[row*ldm + col .. +7], 8-byte aligned because ldm=80 and col are
// multiples of 8), so this compiles to ONE ds_read2_b64 instead of 8
// ds_read_u8 + mask/OR reassembly. The byte placement is identical to
// du_load_matrix_sync<col_major>, so the v_mmac_i32_16x16x32_i8 fragment
// registers receive the same values.
__device__ __forceinline__ void load_b_frag8(
    DUFragment<matrix_b, kTileM, kTileN, kTileK, signed char, col_major>& f,
    const int8_t* __restrict__ p,
    int ldm) {
  const unsigned row = static_cast<unsigned>(__lane_id()) & 0xfu;
  const unsigned col = (static_cast<unsigned>(__lane_id()) >> 4) << 3;
  const int64_t v = *reinterpret_cast<const int64_t*>(p + row * ldm + col);
  *reinterpret_cast<int64_t*>(&f.x[0]) = v;
}

// Iteration-2 throughput baseline: templated 2-D macro-tile packed-B DUMMA
// prefill kernel for the exact assigned shape (M>=128, N=1280, K=6144). The
// mandated tile set {64x64, 64x128, 128x64} is instantiated from this one
// template; <64,128> is the exact-shape dispatch. 256 threads = 4
// wavefronts; each wave owns a (BM/2) x (BN/2) quadrant of m16n16k32 int32
// accumulator fragments (kWaveM16 x kWaveN16 per wave). K is staged
// cooperatively in a SINGLE LDS buffer at 64-K granularity with TWO
// __syncthreads per stage; the final stage is peeled and its provably dead
// all-consumption barrier dropped:
//   * A is staged row-major into a_tile[BM][68] (odd 17-word stride, five
//     bank phases); global-load latency is fully exposed (co-resident
//     blocks hide it).
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
//   * The epilogue is the fused coalesced store
//     (store_prefill_fragment_coalesced_rs): per-row and per-column scales
//     are hoisted into registers loaded once per lane after the peeled final
//     stage's MMAC burst; one 8-byte store per lane per fragment.
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

  // Peel the final stage. After its MMAC burst no wavefront reads LDS again
  // -- the epilogue touches only registers and global memory -- so the final
  // all-consumption barrier is provably dead and is dropped. The per-row
  // (xs) and per-column (ws) scale registers are prefetched right after the
  // last MMACs: 2 scalar + 4 float4 loads per lane instead of the per-
  // fragment reloads, with the load latency hidden behind the
  // transpose/shuffle chains. The per-element multiply order and bf16
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
// cases (including the paired M=2 shape). One output element per grid-stride
// step; exact int32 accumulation (the assigned K keeps the int8 dot well
// within int32 range), then the fused float scale and bf16 store. The exact
// packed shape (K=6144, N=1280) uses the packed n-major layout packed[n][k]
// (set up once by launch_pack_w8a8_weight); every other (k, n) keeps the
// logical [K, N] row-major layout. The branch is grid-uniform per launch.
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

// One-time n-major transpose of the logical [K, N] row-major weight into
// packed[n][k] for the exact assigned shape (K=6144, N=1280). Runs only from
// launch_pack_w8a8_weight, outside the timed region and outside Graph
// capture. The packed buffer keeps the same byte count (k*n) as the identity
// pack, so allocations and graph-stable addresses are unchanged. Element
// (k, n) of the logical weight lands at packed[n * K + k]; each thread
// copies one 16-byte k-run (coalesced read side; the strided write side is
// off the critical path). The guard guarantees n % 16 == 0, so every 16-byte
// chunk lies inside one logical row and the transpose is exact.
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

  // Iteration-2 throughput baseline: the exact assigned shape (M>=128,
  // N=1280, K=6144) routes to the packed n-major B variant of the 2-D
  // macro-tile kernel, dispatched as the <64,128> instantiation (64x128
  // tile, 4 wavefronts, 8 accumulator fragments per wave, grid
  // (N/128)x(M/64) = (10, 64) = 640 blocks). The guard is exact
  // (m >= 128, n == 1280, k == 6144) and sits BEFORE the generic large-M
  // paths; the packed layout is only valid for this (k, n).
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

  // Native INT8 DUMMA m16n16k32 prefill path (2-D macro-tile 64x64,
  // single-buffered 128-K K stage, identity [K, N] weight layout) for
  // large-M shapes with compatible geometry that are NOT the packed exact
  // shape (which returned above), e.g. (4096, 1536, 6144):
  // 1536 % 64 == 0, 6144 % 128 == 0. The M >= 128 guard keeps the paired
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

// Stable host launch symbol consumed by csrc/bindings.cpp. For the exact
// assigned shape (K=6144, N=1280) it performs the one-time n-major B pack
// (logical [K, N] -> packed[n][k]) outside the timed region and outside
// Graph capture; every other (K, N) keeps the identity device-to-device
// copy, valid for every (K, N). The packed buffer size is k*n in both cases,
// so the allocation and graph-stable packed layout are unchanged.
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
      if (blocks < 1) {
        blocks = 1;
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
