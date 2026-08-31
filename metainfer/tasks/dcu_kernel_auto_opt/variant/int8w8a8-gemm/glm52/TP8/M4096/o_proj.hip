// @@variant shape=glm_tp8_o_proj_m4096 commit=486f3ec14b6d88c045aa35a5c85d1d44816c8c2d added=2026-08-31
//   median_us=759.9 p90_us=760.8 speedup=72.08 baseline_us=5.478e+04
//   source=glm5-2-dsh-tp8-m4096-1-e6a280a2
// @@variant shape=glm_tp8_o_proj_m4096
// INT8 W8A8 GEMM HIP implementation for Hygon K500SM_AI / gfx928.
//
// Worker: worker_2 (physical GPU 2), assigned shape:
//   glm_tp8_o_proj_m4096 : M=4096, N=6144, K=2048
//
// Bootstrap strategy (iteration 1, correctness-first but a usable profiling
// baseline - a large-Prefill scalar K loop would not be):
//   * Large-prefill path (exact (m, n, k) == (4096, 6144, 2048)):
//       native INT8 DUMMA m16n16k32 with int32 accumulation; one 128x64
//       output tile per block; four wavefronts (256 threads); each wave owns
//       a 64x32 quadrant built from eight m16n16k32 accumulator fragments;
//       the block cooperatively vector-loads A[128,64] and B[64,64] into one
//       single-buffered 64-K LDS stage (15,360 B total: 128*80 + 64*80 with
//       16 B padding per row for bank skew); two __syncthreads per stage;
//       fused dot * x_scale[m] * weight_scale[n] epilogue stored directly as
//       bf16 from the accumulator fragments.  Grid dim3(96, 32) = 3072 blocks
//       dwarfs the 120 CUs, so no split-K is needed.
//       The weight is read in the identity [K, N] int8 layout (bootstrap:
//       launch_pack_w8a8_weight is an identity device-to-device copy for
//       every (k, n)).  A is staged m-major into a_tile[m][k] and its eight
//       fragment k-values per lane are contiguous, so A fragments use the
//       direct 8-byte LDS fill (load_fragment8, the exact pattern validated
//       on the worker-29 TP4 gate_up lineage iteration 13 and the TP8 hy3
//       o_proj lineage iteration 12).  B is staged k-major into
//       b_tile[k][n] and its fragments use the library du_load_matrix_sync
//       row_major loader (8 per-byte LDS reads strided by ldm per lane -
//       correct, direct, and simple; the packed n-major layout + 8-byte B
//       fragment fill is a later-round optimization once the packed layout
//       is introduced).
//   * Generic scalar int8/int32 fallback for every unmatched (m, n, k),
//       including all small-M API cases (M=2, M=16) and M=3072 with the same
//       (K, N) = (2048, 6144); the fallback reads the identity [K, N] layout
//       for every (k, n).
//   * launch_pack_w8a8_weight: identity device-to-device copy of the raw
//       [K, N] weight and the [N] weight_scale for every (k, n) (bootstrap;
//       later rounds may change only the pack HIP implementation and the
//       matching GEMM interpretation).  Packing never happens inside the
//       timed GEMM.
//       Iteration 5 (packing round): for the exact (k, n) == (2048, 6144)
//       launch_pack_w8a8_weight now produces the n-major [N, K] layout
//       packed[n*K + kk] = raw[kk*N + n] (same byte count and buffer ->
//       graph-stable), and the exact-shape dispatch uses the new
//       w8a8_dumma_256x64x64_packedb_kernel, which stages B n-major into
//       bank-safe 72-byte LDS rows and fills each col_major B fragment with
//       one contiguous 8-byte ds_read_b64 (zero per-byte reads, zero
//       reassembly VALU); the scalar fallback decodes the n-major layout
//       when (k, n) matches so the paired M=2/M=16 and M=3072 API shapes
//       stay byte-exact.  Every other (k, n) keeps the identity copy.
//       Iteration 6 (epilogue round): the fused scale->bf16->coalesced
//       store epilogue of w8a8_dumma_256x64x64_packedb_kernel is
//       restructured to register-batch the per-row and per-column scales:
//       each lane loads its 8 x_scale rows and its 2 weight_scale float4
//       columns ONCE (issued on the last K stage, before the final
//       protective barrier) and the 16 per-fragment coalesced stores reuse
//       them; the previous per-call epilogue re-loaded the scales inside
//       each store (PMC vmem_read surplus ~153,600 wavefront loads above
//       the pure staging floor).  Output bits are identical (same transpose,
//       same float(dot) * x_scale[row] * weight_scale[col] order, same bf16
//       rounding).  No workspace/combine pass exists: the workspace
//       argument stays unused and the epilogue is fully fused in-kernel.
//
// Rules honored: headers in the known-good order (hip/hip_runtime.h,
// hip/hip_bfloat16.h, du_mma.h); wavefront = 64 and blockDim a multiple of
// 64; int32 accumulation with float conversion only for
// dot * x_scale[m] * weight_scale[n]; hipLaunchKernelGGL on the
// caller-provided PyTorch stream; no stream-0 / synchronization / temporary
// allocation in gemm_out; no NVIDIA wmma/mma.sync/PTX/warp-32 masks; all
// barriers on paths reached by every thread in the block; explicit dispatch
// for the assigned shape with a scalar generic fallback for everything else.

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
constexpr int kTargetN = 6144;
constexpr int kTargetK = 2048;
constexpr int kBlockM = 128;
constexpr int kBlockN = 64;
constexpr int kStageK = 64;
constexpr int kBPad = 16;  // 64 -> 80-byte LDS row stride (five bank phases)
constexpr int kBStride = kBlockN + kBPad;
constexpr int kAStride = kStageK + kBPad;  // 64 -> 80-byte A row stride
constexpr int kBlockThreads = 4 * kWaveSize;

// Iteration-4 tile-shape constants: 256x64 output tile (M-major doubling of
// the accepted 128x64 tile) at 256 threads = 4 wavefronts, each wave owning a
// 128x32 quadrant (16 m16n16k32 int32 accumulators).
constexpr int kBlockM256 = 2 * kBlockM;

// Iteration-5 packed-B constants: for the exact (k, n) == (2048, 6144) the
// weight is packed once (outside the timed region) into the n-major [N, K]
// layout packed[n*K + kk] = raw[kk*N + n]; B is then staged n-major into LDS
// rows of 72 B (64 + 8). 72 is an odd 8-byte count, so the 16 lanes of each
// ds_read_b64 fragment-load phase land on 16 distinct bank pairs (18r mod 32,
// r = 0..15 -> all 16 even dword starts: the conflict-free 4-phase floor),
// and 72 % 16 == 8, so staging stores use two int64 halves (ds_write2_b64)
// instead of one misaligned int4.
constexpr int kBPadPacked = 8;  // 64 -> 72-byte n-major B LDS row stride
constexpr int kBStridePacked = kStageK + kBPadPacked;

// Iteration-9 (A-stride round) constant: packedb A LDS row stride 64 -> 88
// bytes, replacing the shared 80-byte kAStride for w8a8_dumma_256x64x64_
// packedb_kernel only (the identity-layout kernels keep kAStride = 80).
// 88 = 22 dwords: 22r mod 32 is a permutation of the 16 even dword starts
// over r = 0..15, so the 16 lanes of each ds_read_b64 fragment-load phase
// land on 16 distinct bank pairs (the same conflict-free 4-phase floor the
// packed-B stride 72 already uses); the current 80-byte (20-dword) stride
// has 20r mod 32 with period 8, so rows r and r+8 alias onto one bank phase
// and EVERY A fragment read pays a 2-way bank conflict (PMC
// lds_bank_conflicts 20,447,232, deterministic across iterations 5..8).
// 88 % 16 == 8, so the 16-byte staging stores go through two int64 halves
// (ds_write2_b64, the exact accepted packed-B staging pattern) - the same
// instruction count as the current ds_write_b128. A[256,88] = 22,528 B +
// B[64,72] = 4,608 B = 27,136 B/block; 2 blocks/CU = 54,272 <= 65,536, so
// co-residency, VGPR, barrier count and the global-load pattern are all
// unchanged; only the physical LDS addresses move (same bytes stored, same
// bytes read, fragment values bit-identical).
constexpr int kAStridePacked = kStageK + 24;  // 64 -> 88-byte A LDS row stride

using bf16_t = hip_bfloat16;

// Direct 8-byte LDS fragment fill for matrix_a on the m-major A tile.
// The m16n16k32 matrix_a row_major loader assigns x[i] =
// p[row*ldm + col_group*8 + i] (row = lane & 15, col_group = lane >> 4),
// i.e. eight consecutive k-values per lane; the library path makes the
// compiler emit per-byte ds_read_u8 loads plus a mask/OR reassembly chain
// per dword that is an arithmetic identity on the v_mmac A operand.  Writing
// the same 8 bytes straight into the fragment storage keeps the operand bit
// pattern and the int32 accumulation identical and lets the compiler feed a
// single 8-byte LDS read to the v_mmac.  This is the exact pattern validated
// on the worker-29 TP4 gate_up lineage (accepted iteration 13) and the TP8
// hy3 o_proj lineage (accepted iteration 12).
template <typename Frag>
__device__ __forceinline__ void load_fragment8(
    Frag& frag,
    const signed char* __restrict__ base,
    int stride,
    int lane) {
  const int off = (lane & 15) * stride + ((lane >> 4) << 3);
  *reinterpret_cast<unsigned long long*>(&frag.x[0]) =
      *reinterpret_cast<const unsigned long long*>(base + off);
}

// Coalesced fragment store.  The m16n16k32 accumulator lane mapping
// (row = lane & 15, column group c4 = lane >> 4, frag.x[i] -> column
// c4 + 4*i) gives each lane four elements strided by 4 columns, so a direct
// store would be four 2-byte scalar stores per lane whose wavefront
// addresses touch each 32-B sector at 25% utilization.  This epilogue
// transposes the 4-element groups within each 4-lane column group (lanes r,
// r+16, r+32, r+48 -- a 4x4 transpose, two 2x2 steps with shfl_xor 16 then
// 32), so lane (r, c4) ends up holding the four CONTIGUOUS columns
// 4*c4 .. 4*c4+3, converts them to bf16, packs 4 bf16 (8 B), and writes ONE
// 8-byte store per lane (100% store sector efficiency).  This DTK lowers
// __shfl_xor to ds_bpermute at the block tail where the LDS pipe is idle.
// Only the int32 values are re-routed between lanes; the per-element scale
// multiply order (float(dot) * x_scale[row] * weight_scale[col]) and the
// bf16 rounding are unchanged, so the stored bits are identical to the
// direct per-element store.  The row >= m guard is wavefront-uniform (all
// 64 lanes of a wave share the same 16-row window), so the shuffles never
// mix active and inactive lanes; base_col is a multiple of 16 and n*2 a
// multiple of 8, so the float4 weight_scale load and the 8-byte store are
// aligned.
template <typename AccFragment>
__device__ __forceinline__ void store_prefill_fragment_coalesced(
    const AccFragment& frag,
    const float* __restrict__ x_scale,
    const float* __restrict__ weight_scale,
    bf16_t* __restrict__ out,
    int base_row,
    int base_col,
    int m,
    int n,
    int lane) {
  const int row = base_row + (lane & 15);
  if (row >= m) {
    return;  // tail-M masking: padded rows never write (wavefront-uniform)
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
  // aligned: base_col is a multiple of 16, n*2 is a multiple of 8).
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
  *reinterpret_cast<uint64_t*>(out + static_cast<int64_t>(row) * n + col0) =
      packed;
}

// Register-batched variant of the coalesced fragment store (iteration 6,
// epilogue round).  Identical to store_prefill_fragment_coalesced in every
// output bit: same 4x4 shuffle transpose, same per-element multiply order
// float(dot) * x_scale[row] * weight_scale[col], same __float2bfloat16
// rounding, same single 8-byte coalesced store.  The only difference is that
// the per-row x_scale and per-column weight_scale values are passed in from
// registers: the caller loads each lane's 8 x_scale rows and 2 weight_scale
// float4 columns exactly once per wave (the per-call epilogue re-loaded the
// scales inside each of the 16 fragment stores), so the stored bits are
// unchanged while the epilogue vmem read count per lane per wave drops from
// up to 8 x_scale + 16 float4 loads to exactly 8 + 2 and the loads are
// independent of (and issue before) the shuffle/convert/store chains.
template <typename AccFragment>
__device__ __forceinline__ void store_prefill_fragment_coalesced_scaled(
    const AccFragment& frag,
    float xs,
    const float4& ws,
    bf16_t* __restrict__ out,
    int base_row,
    int base_col,
    int m,
    int n,
    int lane) {
  const int row = base_row + (lane & 15);
  if (row >= m) {
    return;  // tail-M masking: padded rows never write (wavefront-uniform)
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

  // Lane (r, c4) owns columns base_col + 4*c4 .. +3 (8 B, 8-byte aligned:
  // base_col is a multiple of 16, n*2 is a multiple of 8); ws is the float4
  // at weight_scale + base_col + 4*c4 preloaded by the caller.
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
  *reinterpret_cast<uint64_t*>(out + static_cast<int64_t>(row) * n + col0) =
      packed;
}

// ---------------------------------------------------------------------------
// Large-M prefill: 128x64 output tile per block, four wavefronts of 64 lanes.
// Each wave owns a 64x32 quadrant (eight m16n16k32 int32 accumulators); the
// block cooperatively stages A[128,64] from x_q (row-major, stride k) and
// B[64,64] from the identity [K, N] weight into a single-buffered LDS
// buffer. Two barriers per stage: one after the cooperative load, one before
// the next stage overwrites LDS. Direct fragment epilogue (coalesced store).
//
// Stage layout (bootstrap, identity [K, N] weight):
//   A[128, 80]:  m-major, 512 int4s; thread t loads int4 #0 at
//                a_tile[(t/4)*80 + (t%4)*16] from x_q[(m0 + t/4)*k + k0 +
//                (t%4)*16] and int4 #1 at row +64 (rows 64..127).
//   B[64, 80]:   k-major, 256 int4s; thread t loads int4 at
//                b_tile[(t/4)*80 + (t%4)*16] from
//                weight[(k0 + t/4)*N + n0 + (t%4)*16] (16 consecutive n at
//                fixed k; one int4 per thread).  B fragments are loaded with
//                the library row_major loader (ldm = 80): lane (r, c4) holds
//                x[i] = b_tile[(kk + r + i)*80 + local_col + c4], i.e. eight
//                k-rows strided by ldm - correct for the identity [K, N]
//                layout; later rounds can switch to an n-major packed B with
//                the 8-byte fragment fill.
// ---------------------------------------------------------------------------
__global__ __launch_bounds__(kBlockThreads) void w8a8_dumma_128x64x64_kernel(
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
  const int local_row = wave_row * 64;
  const int local_col = wave_col * 32;

  // Single-buffered 64-K stage: A[128, 80] + B[64, 80] = 15,360 B/block
  // (4 blocks/CU fit the 64 KiB LDS budget; the padded 80-byte strides are
  // 16-byte-aligned and break the 64-byte LDS bank periodicity).
  __shared__ __align__(16) int8_t a_tile[kBlockM * kAStride];
  __shared__ __align__(16) int8_t b_tile[kBlockN * kBStride];

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

  // Cooperative staging: each thread owns one 16-byte vector in A rows
  // [0,64), one in A rows [64,128), and one 16-byte vector of B.
  // A[128,64] is 512 int4s; B[64,64] is 256 int4s.
  const int vector_byte_offset = tid * static_cast<int>(sizeof(int4));
  const int stage_row = vector_byte_offset / kStageK;  // tid / 4, 0..63
  const int stage_col = vector_byte_offset - stage_row * kStageK;  // (tid%4)*16
  const int b_row = tid >> 2;            // k row of this thread's B int4
  const int b_n16 = (tid & 3) * 16;      // n offset (stage-local) of B int4

  for (int k0 = 0; k0 < k; k0 += kStageK) {
    *reinterpret_cast<int4*>(a_tile + stage_row * kAStride + stage_col) =
        *reinterpret_cast<const int4*>(
            x_q + static_cast<int64_t>(m0 + stage_row) * k + k0 + stage_col);
    *reinterpret_cast<int4*>(a_tile +
                             (stage_row + kBlockM / 2) * kAStride + stage_col) =
        *reinterpret_cast<const int4*>(
            x_q + static_cast<int64_t>(m0 + stage_row + kBlockM / 2) * k +
            k0 + stage_col);
    *reinterpret_cast<int4*>(b_tile + b_row * kBStride + b_n16) =
        *reinterpret_cast<const int4*>(
            weight + (static_cast<int64_t>(k0 + b_row) * n + n0 + b_n16));
    __syncthreads();

    // Each wave consumes its 64x32 quadrant: eight m16n16k32 MMACs per kk.
#pragma unroll
    for (int kk = 0; kk < kStageK; kk += kTileK) {
      // A fragments: direct 8-byte LDS fill (m-major A tile; each lane's
      // eight fragment k-values are contiguous).
      load_fragment8(a_frag0, a_tile + local_row * kAStride + kk, kAStride,
                     lane);
      load_fragment8(a_frag1, a_tile + (local_row + kTileM) * kAStride + kk,
                     kAStride, lane);
      // B fragments: library row_major loader on the k-major [K, N] LDS
      // tile (bootstrap, identity layout; ldm = kBStride).
      du_load_matrix_sync(b_frag0, b_tile + kk * kBStride + local_col,
                          kBStride);
      du_load_matrix_sync(b_frag1,
                          b_tile + kk * kBStride + local_col + kTileN,
                          kBStride);
      du_mma_sync(acc00, a_frag0, b_frag0, acc00);
      du_mma_sync(acc01, a_frag0, b_frag1, acc01);
      du_mma_sync(acc10, a_frag1, b_frag0, acc10);
      du_mma_sync(acc11, a_frag1, b_frag1, acc11);
      load_fragment8(
          a_frag0, a_tile + (local_row + 2 * kTileM) * kAStride + kk,
          kAStride, lane);
      load_fragment8(
          a_frag1, a_tile + (local_row + 3 * kTileM) * kAStride + kk,
          kAStride, lane);
      du_mma_sync(acc20, a_frag0, b_frag0, acc20);
      du_mma_sync(acc21, a_frag0, b_frag1, acc21);
      du_mma_sync(acc30, a_frag1, b_frag0, acc30);
      du_mma_sync(acc31, a_frag1, b_frag1, acc31);
    }

    // Protect the LDS buffer from the next stage's cooperative overwrite.
    __syncthreads();
  }

  const int base_row = m0 + local_row;
  const int base_col = n0 + local_col;
  // Coalesced epilogue store (one 8-byte store per lane per fragment; scales
  // + bf16 conversion fused in-kernel, no workspace pass).
  store_prefill_fragment_coalesced(acc00, x_scale, weight_scale, out,
                                   base_row, base_col, m, n, lane);
  store_prefill_fragment_coalesced(acc01, x_scale, weight_scale, out,
                                   base_row, base_col + kTileN, m, n, lane);
  store_prefill_fragment_coalesced(acc10, x_scale, weight_scale, out,
                                   base_row + kTileM, base_col, m, n, lane);
  store_prefill_fragment_coalesced(acc11, x_scale, weight_scale, out,
                                   base_row + kTileM, base_col + kTileN, m, n,
                                   lane);
  store_prefill_fragment_coalesced(acc20, x_scale, weight_scale, out,
                                   base_row + 2 * kTileM, base_col, m, n,
                                   lane);
  store_prefill_fragment_coalesced(acc21, x_scale, weight_scale, out,
                                   base_row + 2 * kTileM, base_col + kTileN,
                                   m, n, lane);
  store_prefill_fragment_coalesced(acc30, x_scale, weight_scale, out,
                                   base_row + 3 * kTileM, base_col, m, n,
                                   lane);
  store_prefill_fragment_coalesced(acc31, x_scale, weight_scale, out,
                                   base_row + 3 * kTileM, base_col + kTileN,
                                   m, n, lane);
}

// ---------------------------------------------------------------------------
// Iteration-3 pipeline-control variant: double-buffered K-stage of the same
// 128x64 tile / 64-K stage / fragment loads / epilogue as the accepted
// single-buffered kernel.  Stage s+1's three global loads (2 A int4s + 1 B
// int4 per thread) are issued BEFORE the stage s MMAC burst and the loaded
// vectors stay in VGPR; the publish (ds_write into the idle buffer) happens
// AFTER the burst, so the compiler's vmcnt(0) wait sits behind 16 v_mmac +
// ~70 LDS reads instead of directly behind the load issue (the accepted
// single-buffered kernel waits immediately after 3 loads per wave per
// stage).  One __syncthreads per stage (32/block vs 64/block: the barrier
// doubles as read-done for stage s+1 and write-done for stage s; the last
// stage needs no fence before the register epilogue).  Costs: LDS
// 15,360 -> 30,720 B/block (A[2][128,80] + B[2][64,80]), residency
// 4 -> 2 blocks/CU = 8 waves/CU; VGPR cap at 2 blocks/CU is 128, so the
// ~12-VGPR loop-carried prefetch payload cannot change residency.  Retained
// only if ISA/PMC evidence shows the vmcnt wait moved off the load->ds_write
// path (covered by the MMAC burst) and the measured median/p90 beat the
// 1162.15/1164.03 guard.
// ---------------------------------------------------------------------------
__global__ __launch_bounds__(kBlockThreads) void w8a8_dumma_128x64x64_db_kernel(
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
  const int local_row = wave_row * 64;
  const int local_col = wave_col * 32;

  // Double-buffered 64-K stage: A[2][128,80] + B[2][64,80] = 30,720 B/block
  // (2 blocks/CU fit the 64 KiB LDS budget; padded 80-byte strides and bank
  // phases unchanged from the accepted kernel).
  __shared__ __align__(16) int8_t a_tile[2][kBlockM * kAStride];
  __shared__ __align__(16) int8_t b_tile[2][kBlockN * kBStride];

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

  // Cooperative staging indexing is identical to the accepted kernel: each
  // thread owns two 16-byte A vectors (rows [0,64) and [64,128)) and one
  // 16-byte B vector.
  const int vector_byte_offset = tid * static_cast<int>(sizeof(int4));
  const int stage_row = vector_byte_offset / kStageK;  // tid / 4, 0..63
  const int stage_col = vector_byte_offset - stage_row * kStageK;  // (tid%4)*16
  const int b_row = tid >> 2;            // k row of this thread's B int4
  const int b_n16 = (tid & 3) * 16;      // n offset (stage-local) of B int4

  const int k_stages = k / kStageK;

  // Loop-carried prefetch payload (stage s+1 global vectors in VGPR).
  int4 pa0, pa1, pb0;

  // Prologue: fetch stage 0 into registers, publish into buffer 0, fence.
  {
    const int k0 = 0;
    pa0 = *reinterpret_cast<const int4*>(
        x_q + static_cast<int64_t>(m0 + stage_row) * k + k0 + stage_col);
    pa1 = *reinterpret_cast<const int4*>(
        x_q + static_cast<int64_t>(m0 + stage_row + kBlockM / 2) * k + k0 +
        stage_col);
    pb0 = *reinterpret_cast<const int4*>(
        weight + (static_cast<int64_t>(k0 + b_row) * n + n0 + b_n16));
    *reinterpret_cast<int4*>(a_tile[0] + stage_row * kAStride + stage_col) =
        pa0;
    *reinterpret_cast<int4*>(a_tile[0] +
                             (stage_row + kBlockM / 2) * kAStride +
                             stage_col) = pa1;
    *reinterpret_cast<int4*>(b_tile[0] + b_row * kBStride + b_n16) = pb0;
    __syncthreads();
  }

  for (int s = 0; s < k_stages; ++s) {
    const int cur = s & 1;
    const int nxt = cur ^ 1;

    // Prefetch stage s+1: issue the global loads BEFORE the stage s MMAC
    // burst; the loaded vectors stay in VGPR and the compiler's vmcnt(0) wait
    // is inserted only at the publish below (covered by the burst).
    if (s + 1 < k_stages) {
      const int k0 = (s + 1) * kStageK;
      pa0 = *reinterpret_cast<const int4*>(
          x_q + static_cast<int64_t>(m0 + stage_row) * k + k0 + stage_col);
      pa1 = *reinterpret_cast<const int4*>(
          x_q + static_cast<int64_t>(m0 + stage_row + kBlockM / 2) * k + k0 +
          stage_col);
      pb0 = *reinterpret_cast<const int4*>(
          weight + (static_cast<int64_t>(k0 + b_row) * n + n0 + b_n16));
    }

    // Consume stage s from the current buffer: bit-identical fragment values,
    // MMAC sequence, int32 accumulation order and lane mapping to the
    // accepted kernel (only the buffer base changes).
#pragma unroll
    for (int kk = 0; kk < kStageK; kk += kTileK) {
      load_fragment8(a_frag0, a_tile[cur] + local_row * kAStride + kk,
                     kAStride, lane);
      load_fragment8(a_frag1, a_tile[cur] + (local_row + kTileM) * kAStride +
                                 kk,
                     kAStride, lane);
      du_load_matrix_sync(b_frag0,
                          b_tile[cur] + kk * kBStride + local_col, kBStride);
      du_load_matrix_sync(b_frag1,
                          b_tile[cur] + kk * kBStride + local_col + kTileN,
                          kBStride);
      du_mma_sync(acc00, a_frag0, b_frag0, acc00);
      du_mma_sync(acc01, a_frag0, b_frag1, acc01);
      du_mma_sync(acc10, a_frag1, b_frag0, acc10);
      du_mma_sync(acc11, a_frag1, b_frag1, acc11);
      load_fragment8(
          a_frag0, a_tile[cur] + (local_row + 2 * kTileM) * kAStride + kk,
          kAStride, lane);
      load_fragment8(
          a_frag1, a_tile[cur] + (local_row + 3 * kTileM) * kAStride + kk,
          kAStride, lane);
      du_mma_sync(acc20, a_frag0, b_frag0, acc20);
      du_mma_sync(acc21, a_frag0, b_frag1, acc21);
      du_mma_sync(acc30, a_frag1, b_frag0, acc30);
      du_mma_sync(acc31, a_frag1, b_frag1, acc31);
    }

    // Publish stage s+1 into the idle buffer after the burst (vmcnt wait
    // covered by the burst), then one fence: read-done for s+1 and
    // write-done for s (buffer cur is free for the s+2 publish).
    if (s + 1 < k_stages) {
      *reinterpret_cast<int4*>(a_tile[nxt] + stage_row * kAStride +
                               stage_col) = pa0;
      *reinterpret_cast<int4*>(a_tile[nxt] +
                               (stage_row + kBlockM / 2) * kAStride +
                               stage_col) = pa1;
      *reinterpret_cast<int4*>(b_tile[nxt] + b_row * kBStride + b_n16) = pb0;
      __syncthreads();
    }
  }

  const int base_row = m0 + local_row;
  const int base_col = n0 + local_col;
  store_prefill_fragment_coalesced(acc00, x_scale, weight_scale, out,
                                   base_row, base_col, m, n, lane);
  store_prefill_fragment_coalesced(acc01, x_scale, weight_scale, out,
                                   base_row, base_col + kTileN, m, n, lane);
  store_prefill_fragment_coalesced(acc10, x_scale, weight_scale, out,
                                   base_row + kTileM, base_col, m, n, lane);
  store_prefill_fragment_coalesced(acc11, x_scale, weight_scale, out,
                                   base_row + kTileM, base_col + kTileN, m, n,
                                   lane);
  store_prefill_fragment_coalesced(acc20, x_scale, weight_scale, out,
                                   base_row + 2 * kTileM, base_col, m, n,
                                   lane);
  store_prefill_fragment_coalesced(acc21, x_scale, weight_scale, out,
                                   base_row + 2 * kTileM, base_col + kTileN,
                                   m, n, lane);
  store_prefill_fragment_coalesced(acc30, x_scale, weight_scale, out,
                                   base_row + 3 * kTileM, base_col, m, n,
                                   lane);
  store_prefill_fragment_coalesced(acc31, x_scale, weight_scale, out,
                                   base_row + 3 * kTileM, base_col + kTileN,
                                   m, n, lane);
}

// ---------------------------------------------------------------------------
// Iteration-4 tile-shape variant: 256x64 output tile (M-major doubling of the
// accepted 128x64 tile) at 256 threads = 4 wavefronts, each wave owning a
// 128x32 quadrant (sixteen m16n16k32 int32 accumulators).  This is the
// measured tile axis pushed one step further: per-FLOP B-fragment LDS reads
// scale as 1/(quadrant m-height) (the 64x128 flip's quadrants were 32x64 and
// lost +8% with a doubled per-FLOP B cost), so doubling the quadrant height
// to 128 HALVES the per-FLOP B ds_read_u8 + reassembly VALU - the measured
// LDS-read/wait bottleneck (PMC lds_wait ~= lds_instructions) - while keeping
// the same 64-column quadrant width (B library row_major loader ldm = 80).
// B reuse doubles (each staged B element is consumed by all 256 block rows),
// so the grid-level B re-read drops from M/128 = 32x to M/256 = 16x (staged B
// traffic 402.7 -> 201.3 MB/replay; A re-read stays N/64 = 96x = 805.3 MB;
// total 1.208 -> 1.007 GB, -17%).  Single-buffered 64-K stage: A[256,80] +
// B[64,80] = 25,600 B/block -> 2 blocks/CU = 8 waves/CU, the SAME
// co-residency structure as the accepted double-buffered 128x64 kernel
// (2 blocks x 4 waves), so the comparison is not residency-confounded.
// Two __syncthreads per stage, direct 8-byte A fragment fills, fused
// coalesced scale->bf16 epilogue, row>=m guard.  Grid dim3(96,16) = 1536
// blocks = 12.8 blocks/CU, so the MxN output-tile grid still dwarfs the
// 120 CUs.  Fragment values, MMAC sequence, int32 accumulation order
// (k0-outer 64, kk-inner 32, hardware k-order in m16n16k32),
// element-to-lane mapping, barrier count and the fused epilogue are
// bit-identical to the accepted kernels (only the tile height, the quadrant
// split and the staging vectorization change), so exact bf16 equality
// (mismatch 0 / max_abs_error 0.0) is expected without tolerance debate.
// ---------------------------------------------------------------------------
__global__ __launch_bounds__(kBlockThreads)
    void w8a8_dumma_256x64x64_kernel(
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
  // 4 waves: wave_row = (wave>>1)*128 covers rows 0..255 in two 128-row bands
  // (waves 2w, 2w+1 share the band), wave_col picks the 32-col half.
  const int wave_row = (wave >> 1) * (kBlockM256 / 2);
  const int wave_col = wave & 1;
  const int m0 = static_cast<int>(blockIdx.y) * kBlockM256;
  const int n0 = static_cast<int>(blockIdx.x) * kBlockN;
  const int local_row = wave_row;
  const int local_col = wave_col * 32;

  // Single-buffered 64-K stage: A[256, 80] + B[64, 80] = 25,600 B/block
  // (2 blocks/CU fit the 64 KiB LDS budget; padded 80-byte strides and bank
  // phases unchanged from the accepted kernels).
  __shared__ __align__(16) int8_t a_tile[kBlockM256 * kAStride];
  __shared__ __align__(16) int8_t b_tile[kBlockN * kBStride];

  DUFragment<matrix_a, kTileM, kTileN, kTileK, signed char, row_major>
      a_frag0, a_frag1, a_frag2, a_frag3, a_frag4, a_frag5, a_frag6, a_frag7;
  DUFragment<matrix_b, kTileM, kTileN, kTileK, signed char, row_major>
      b_frag0, b_frag1;
  // Sixteen m16n16k32 int32 accumulators (8 m-groups x 2 n-groups per wave).
  DUFragment<accumulator, kTileM, kTileN, kTileK, int>
      acc00, acc01, acc10, acc11, acc20, acc21, acc30, acc31,
      acc40, acc41, acc50, acc51, acc60, acc61, acc70, acc71;
  du_fill_fragment(acc00, 0);
  du_fill_fragment(acc01, 0);
  du_fill_fragment(acc10, 0);
  du_fill_fragment(acc11, 0);
  du_fill_fragment(acc20, 0);
  du_fill_fragment(acc21, 0);
  du_fill_fragment(acc30, 0);
  du_fill_fragment(acc31, 0);
  du_fill_fragment(acc40, 0);
  du_fill_fragment(acc41, 0);
  du_fill_fragment(acc50, 0);
  du_fill_fragment(acc51, 0);
  du_fill_fragment(acc60, 0);
  du_fill_fragment(acc61, 0);
  du_fill_fragment(acc70, 0);
  du_fill_fragment(acc71, 0);

  // Cooperative staging: A[256,64] is 1,024 int4s (4 per thread: rows
  // [0,64), [64,128), [128,192), [192,256)) and B[64,64] is 256 int4s (one
  // per thread), 5 loads per thread per stage.
  const int a_row0 = tid >> 2;        // 0..63
  const int a_col = (tid & 3) * 16;   // 0,16,32,48

  for (int k0 = 0; k0 < k; k0 += kStageK) {
    *reinterpret_cast<int4*>(a_tile + a_row0 * kAStride + a_col) =
        *reinterpret_cast<const int4*>(
            x_q + static_cast<int64_t>(m0 + a_row0) * k + k0 + a_col);
    *reinterpret_cast<int4*>(a_tile +
                             (a_row0 + kBlockM256 / 4) * kAStride + a_col) =
        *reinterpret_cast<const int4*>(
            x_q + static_cast<int64_t>(m0 + a_row0 + kBlockM256 / 4) * k +
            k0 + a_col);
    *reinterpret_cast<int4*>(a_tile +
                             (a_row0 + kBlockM256 / 2) * kAStride + a_col) =
        *reinterpret_cast<const int4*>(
            x_q + static_cast<int64_t>(m0 + a_row0 + kBlockM256 / 2) * k +
            k0 + a_col);
    *reinterpret_cast<int4*>(a_tile +
                             (a_row0 + 3 * kBlockM256 / 4) * kAStride +
                             a_col) =
        *reinterpret_cast<const int4*>(
            x_q + static_cast<int64_t>(m0 + a_row0 + 3 * kBlockM256 / 4) * k +
            k0 + a_col);
    const int b_row = tid >> 2;      // k row of this thread's B int4, 0..63
    const int b_n16 = (tid & 3) * 16;
    *reinterpret_cast<int4*>(b_tile + b_row * kBStride + b_n16) =
        *reinterpret_cast<const int4*>(
            weight + (static_cast<int64_t>(k0 + b_row) * n + n0 + b_n16));
    __syncthreads();

    // Each wave consumes its 128x32 quadrant: eight m-groups x two n-groups,
    // 16 m16n16k32 MMACs per kk.  The A fragment loads and the MMAC sequence
    // keep the exact int32 accumulation order (k0-outer 64, kk-inner 32,
    // hardware k-order in m16n16k32) and the element-to-lane mapping of the
    // accepted kernels; only the number of m-groups per wave changes.
#pragma unroll
    for (int kk = 0; kk < kStageK; kk += kTileK) {
      load_fragment8(a_frag0, a_tile + local_row * kAStride + kk, kAStride,
                     lane);
      load_fragment8(a_frag1, a_tile + (local_row + kTileM) * kAStride + kk,
                     kAStride, lane);
      load_fragment8(a_frag2, a_tile + (local_row + 2 * kTileM) * kAStride +
                                  kk,
                     kAStride, lane);
      load_fragment8(a_frag3, a_tile + (local_row + 3 * kTileM) * kAStride +
                                  kk,
                     kAStride, lane);
      du_load_matrix_sync(b_frag0, b_tile + kk * kBStride + local_col,
                          kBStride);
      du_load_matrix_sync(b_frag1,
                          b_tile + kk * kBStride + local_col + kTileN,
                          kBStride);
      du_mma_sync(acc00, a_frag0, b_frag0, acc00);
      du_mma_sync(acc01, a_frag0, b_frag1, acc01);
      du_mma_sync(acc10, a_frag1, b_frag0, acc10);
      du_mma_sync(acc11, a_frag1, b_frag1, acc11);
      du_mma_sync(acc20, a_frag2, b_frag0, acc20);
      du_mma_sync(acc21, a_frag2, b_frag1, acc21);
      du_mma_sync(acc30, a_frag3, b_frag0, acc30);
      du_mma_sync(acc31, a_frag3, b_frag1, acc31);
      load_fragment8(a_frag4, a_tile + (local_row + 4 * kTileM) * kAStride +
                                  kk,
                     kAStride, lane);
      load_fragment8(a_frag5, a_tile + (local_row + 5 * kTileM) * kAStride +
                                  kk,
                     kAStride, lane);
      load_fragment8(a_frag6, a_tile + (local_row + 6 * kTileM) * kAStride +
                                  kk,
                     kAStride, lane);
      load_fragment8(a_frag7, a_tile + (local_row + 7 * kTileM) * kAStride +
                                  kk,
                     kAStride, lane);
      du_mma_sync(acc40, a_frag4, b_frag0, acc40);
      du_mma_sync(acc41, a_frag4, b_frag1, acc41);
      du_mma_sync(acc50, a_frag5, b_frag0, acc50);
      du_mma_sync(acc51, a_frag5, b_frag1, acc51);
      du_mma_sync(acc60, a_frag6, b_frag0, acc60);
      du_mma_sync(acc61, a_frag6, b_frag1, acc61);
      du_mma_sync(acc70, a_frag7, b_frag0, acc70);
      du_mma_sync(acc71, a_frag7, b_frag1, acc71);
    }

    // Protect the LDS buffer from the next stage's cooperative overwrite.
    __syncthreads();
  }

  const int base_row = m0 + local_row;
  const int base_col = n0 + local_col;
  store_prefill_fragment_coalesced(acc00, x_scale, weight_scale, out,
                                   base_row, base_col, m, n, lane);
  store_prefill_fragment_coalesced(acc01, x_scale, weight_scale, out,
                                   base_row, base_col + kTileN, m, n, lane);
  store_prefill_fragment_coalesced(acc10, x_scale, weight_scale, out,
                                   base_row + kTileM, base_col, m, n, lane);
  store_prefill_fragment_coalesced(acc11, x_scale, weight_scale, out,
                                   base_row + kTileM, base_col + kTileN, m, n,
                                   lane);
  store_prefill_fragment_coalesced(acc20, x_scale, weight_scale, out,
                                   base_row + 2 * kTileM, base_col, m, n,
                                   lane);
  store_prefill_fragment_coalesced(acc21, x_scale, weight_scale, out,
                                   base_row + 2 * kTileM, base_col + kTileN,
                                   m, n, lane);
  store_prefill_fragment_coalesced(acc30, x_scale, weight_scale, out,
                                   base_row + 3 * kTileM, base_col, m, n,
                                   lane);
  store_prefill_fragment_coalesced(acc31, x_scale, weight_scale, out,
                                   base_row + 3 * kTileM, base_col + kTileN,
                                   m, n, lane);
  store_prefill_fragment_coalesced(acc40, x_scale, weight_scale, out,
                                   base_row + 4 * kTileM, base_col, m, n,
                                   lane);
  store_prefill_fragment_coalesced(acc41, x_scale, weight_scale, out,
                                   base_row + 4 * kTileM, base_col + kTileN,
                                   m, n, lane);
  store_prefill_fragment_coalesced(acc50, x_scale, weight_scale, out,
                                   base_row + 5 * kTileM, base_col, m, n,
                                   lane);
  store_prefill_fragment_coalesced(acc51, x_scale, weight_scale, out,
                                   base_row + 5 * kTileM, base_col + kTileN,
                                   m, n, lane);
  store_prefill_fragment_coalesced(acc60, x_scale, weight_scale, out,
                                   base_row + 6 * kTileM, base_col, m, n,
                                   lane);
  store_prefill_fragment_coalesced(acc61, x_scale, weight_scale, out,
                                   base_row + 6 * kTileM, base_col + kTileN,
                                   m, n, lane);
  store_prefill_fragment_coalesced(acc70, x_scale, weight_scale, out,
                                   base_row + 7 * kTileM, base_col, m, n,
                                   lane);
  store_prefill_fragment_coalesced(acc71, x_scale, weight_scale, out,
                                   base_row + 7 * kTileM, base_col + kTileN,
                                   m, n, lane);
}

// ---------------------------------------------------------------------------
// Iteration-5 packed-B variant of the accepted 256x64 kernel: the B operand
// switches to the one-time n-major [N, K] pack (packed[n*K + kk] =
// raw[kk*N + n], produced by launch_pack_w8a8_weight outside the timed
// region and Graph capture; same byte count and same buffer, so captured
// addresses are unchanged) and the B fragment path switches from the
// library row_major loader (8 per-byte ds_read_u8 strided by ldm per lane
// plus mask/OR reassembly VALU per dword - the measured LDS-read axis) to
// the lineage-validated 8-byte fragment fill (load_fragment8, one
// ds_read_b64 per col_major fragment, zero reassembly VALU) on the n-major
// staged tile.  The m16n16k32 matrix_b hardware slot for lane
// (r = lane & 15, kq = lane >> 4) is B[k = kk + kq*8 + i][n = local_col +
// i*16 + r] - eight consecutive k bytes at fixed n - so staging B n-major
// (b_tile[n][k]) makes each lane's eight fragment bytes CONTIGUOUS in LDS:
// the fill reads the same elements into the same fragment slots with the
// same v_mmac inputs and the exact int32 accumulation order (k0-outer 64,
// kk-inner 32, hardware k-order), so exact bf16 equality (mismatch 0 /
// max_abs_error 0.0) is expected without tolerance debate.  LDS bank
// safety: the n-major B tile uses the 72-byte row stride (odd 8-byte
// count) so the 16 lanes of each ds_read_b64 phase read 18r mod 32
// (r = 0..15) distinct bank pairs - conflict-free; 72 % 16 == 8 forces the
// two-int64-half staging stores (ds_write2_b64).  Iteration 9 gives the A
// tile the same conflict-free floor: the A row stride is 88 bytes (22
// dwords; 22r mod 32 over r = 0..15 is a permutation of the 16 even dword
// starts), replacing the 80-byte stride's 2-way class (20r mod 32, period
// 8, rows r and r+8 alias), and A staging uses the same two-int64-half
// ds_write2_b64 pattern (88 % 16 == 8).  Everything else is bit-identical
// to the accepted kernel: same bytes staged/read (only the physical LDS
// addresses move), same 128x32 quadrant per wave, same MMAC sequence, two
// __syncthreads per stage, fused coalesced scale->bf16 epilogue, row>=m
// guard, grid dim3(96,16) = 1536 blocks.  LDS 22,528 (A) + 4,608 (B) =
// 27,136 B/block -> 2 blocks/CU = 8 waves/CU, the same co-residency
// structure as the accepted kernel.
// ---------------------------------------------------------------------------
__global__ __launch_bounds__(kBlockThreads)
    void w8a8_dumma_256x64x64_packedb_kernel(
        const int8_t* __restrict__ x_q,
        const int8_t* __restrict__ packed_w,  // [N, K] n-major (exact shape)
        const float* __restrict__ x_scale,
        const float* __restrict__ weight_scale,
        bf16_t* __restrict__ out,
        int m,
        int n,
        int k) {
  const int tid = static_cast<int>(threadIdx.x);
  const int wave = tid / kWaveSize;
  const int lane = tid % kWaveSize;
  // 4 waves: wave_row = (wave>>1)*128 covers rows 0..255 in two 128-row bands
  // (waves 2w, 2w+1 share the band), wave_col picks the 32-col half.
  const int wave_row = (wave >> 1) * (kBlockM256 / 2);
  const int wave_col = wave & 1;
  const int m0 = static_cast<int>(blockIdx.y) * kBlockM256;
  const int n0 = static_cast<int>(blockIdx.x) * kBlockN;
  const int local_row = wave_row;
  const int local_col = wave_col * 32;

  // Single-buffered 64-K stage: A[256, 88] (iteration-9 stride) + B[64, 72]
  // n-major = 27,136 B/block (2 blocks/CU fit the 64 KiB LDS budget).
  __shared__ __align__(16) int8_t a_tile[kBlockM256 * kAStridePacked];
  __shared__ __align__(16) int8_t b_tile[kBlockN * kBStridePacked];

  DUFragment<matrix_a, kTileM, kTileN, kTileK, signed char, row_major>
      a_frag0, a_frag1, a_frag2, a_frag3, a_frag4, a_frag5, a_frag6, a_frag7;
  DUFragment<matrix_b, kTileM, kTileN, kTileK, signed char, col_major>
      b_frag0, b_frag1;
  // Sixteen m16n16k32 int32 accumulators (8 m-groups x 2 n-groups per wave).
  DUFragment<accumulator, kTileM, kTileN, kTileK, int>
      acc00, acc01, acc10, acc11, acc20, acc21, acc30, acc31,
      acc40, acc41, acc50, acc51, acc60, acc61, acc70, acc71;
  du_fill_fragment(acc00, 0);
  du_fill_fragment(acc01, 0);
  du_fill_fragment(acc10, 0);
  du_fill_fragment(acc11, 0);
  du_fill_fragment(acc20, 0);
  du_fill_fragment(acc21, 0);
  du_fill_fragment(acc30, 0);
  du_fill_fragment(acc31, 0);
  du_fill_fragment(acc40, 0);
  du_fill_fragment(acc41, 0);
  du_fill_fragment(acc50, 0);
  du_fill_fragment(acc51, 0);
  du_fill_fragment(acc60, 0);
  du_fill_fragment(acc61, 0);
  du_fill_fragment(acc70, 0);
  du_fill_fragment(acc71, 0);

  // Cooperative staging: A[256,64] is 1,024 int4s (4 per thread: rows
  // [0,64), [64,128), [128,192), [192,256)) and B[64,64] is 256 int4s (one
  // per thread), 5 loads per thread per stage.
  const int a_row0 = tid >> 2;        // 0..63
  const int a_col = (tid & 3) * 16;   // 0,16,32,48
  // B staging is n-major: thread owns the 16-byte k-chunk at
  // (n_row = tid >> 2, k16 = (tid & 3) * 16).
  const int b_n_row = tid >> 2;       // 0..63
  const int b_k16 = (tid & 3) * 16;   // 0,16,32,48

  const int base_row = m0 + local_row;
  const int base_col = n0 + local_col;
  // Iteration-6 (epilogue round) register-batched scales: each lane's 16
  // fragment stores cover the 8 m-groups (rows base_row + r + 16*i,
  // r = lane & 15) x 2 n-groups (cols base_col + 4*c4 and
  // base_col + 16 + 4*c4, c4 = lane >> 4), so each lane needs exactly 8
  // x_scale values and 2 weight_scale float4s.  They are loaded ONCE on the
  // last K stage, issued BEFORE the final protective barrier (uniform
  // branch: k0 is block-uniform, so the barrier below is reached by every
  // thread), so the vmem latency overlaps the barrier and the 16
  // shuffle/convert/store chains; the previous per-call epilogue re-loaded
  // the scales inside each of the 16 fragment stores (PMC vmem_read
  // surplus ~24-25 wavefront loads/wave above the 5/thread/stage staging
  // floor matches ~8 x_scale + ~16 float4 re-loads per lane per wave).
  float xs_m[8];
  float4 ws0, ws1;

  for (int k0 = 0; k0 < k; k0 += kStageK) {
    const int4 av0 = *reinterpret_cast<const int4*>(
        x_q + static_cast<int64_t>(m0 + a_row0) * k + k0 + a_col);
    const int4 av1 = *reinterpret_cast<const int4*>(
        x_q + static_cast<int64_t>(m0 + a_row0 + kBlockM256 / 4) * k + k0 +
        a_col);
    const int4 av2 = *reinterpret_cast<const int4*>(
        x_q + static_cast<int64_t>(m0 + a_row0 + kBlockM256 / 2) * k + k0 +
        a_col);
    const int4 av3 = *reinterpret_cast<const int4*>(
        x_q + static_cast<int64_t>(m0 + a_row0 + 3 * kBlockM256 / 4) * k +
        k0 + a_col);
    // A rows are 88 B apart (8 mod 16), so each 16-byte staging store is two
    // int64 halves (ds_write2_b64, one instruction) - the exact accepted
    // packed-B staging pattern below; a single int4 store would be
    // misaligned on odd rows.
    const int64_t* asrc0 = reinterpret_cast<const int64_t*>(&av0);
    const int64_t* asrc1 = reinterpret_cast<const int64_t*>(&av1);
    const int64_t* asrc2 = reinterpret_cast<const int64_t*>(&av2);
    const int64_t* asrc3 = reinterpret_cast<const int64_t*>(&av3);
    int64_t* adst0 = reinterpret_cast<int64_t*>(
        a_tile + a_row0 * kAStridePacked + a_col);
    int64_t* adst1 = reinterpret_cast<int64_t*>(
        a_tile + (a_row0 + kBlockM256 / 4) * kAStridePacked + a_col);
    int64_t* adst2 = reinterpret_cast<int64_t*>(
        a_tile + (a_row0 + kBlockM256 / 2) * kAStridePacked + a_col);
    int64_t* adst3 = reinterpret_cast<int64_t*>(
        a_tile + (a_row0 + 3 * kBlockM256 / 4) * kAStridePacked + a_col);
    adst0[0] = asrc0[0];
    adst0[1] = asrc0[1];
    adst1[0] = asrc1[0];
    adst1[1] = asrc1[1];
    adst2[0] = asrc2[0];
    adst2[1] = asrc2[1];
    adst3[0] = asrc3[0];
    adst3[1] = asrc3[1];
    // Stage B[64, 64] n-major from the packed [N, K] weight: 16 consecutive
    // k bytes per thread at fixed n (one coalesced int4 global load).  Row
    // stride 72 is 8 mod 16, so the LDS store is two int64 halves
    // (ds_write2_b64) - a single int4 store would be misaligned on odd rows.
    const int4 bv = *reinterpret_cast<const int4*>(
        packed_w +
        (static_cast<int64_t>(n0 + b_n_row) * k + k0 + b_k16));
    const int64_t* bsrc64 = reinterpret_cast<const int64_t*>(&bv);
    int64_t* bdst64 = reinterpret_cast<int64_t*>(
        b_tile + b_n_row * kBStridePacked + b_k16);
    bdst64[0] = bsrc64[0];
    bdst64[1] = bsrc64[1];
    __syncthreads();

    // Each wave consumes its 128x32 quadrant: eight m-groups x two n-groups,
    // 16 m16n16k32 MMACs per kk.  The A fragment loads and the MMAC sequence
    // keep the exact int32 accumulation order (k0-outer 64, kk-inner 32,
    // hardware k-order in m16n16k32) and the element-to-lane mapping of the
    // accepted kernels; only the B fragment fill (one 8-byte contiguous LDS
    // read per col_major fragment on the n-major tile) changes.
#pragma unroll
    for (int kk = 0; kk < kStageK; kk += kTileK) {
      load_fragment8(a_frag0, a_tile + local_row * kAStridePacked + kk,
                     kAStridePacked, lane);
      load_fragment8(a_frag1,
                     a_tile + (local_row + kTileM) * kAStridePacked + kk,
                     kAStridePacked, lane);
      load_fragment8(a_frag2,
                     a_tile + (local_row + 2 * kTileM) * kAStridePacked + kk,
                     kAStridePacked, lane);
      load_fragment8(a_frag3,
                     a_tile + (local_row + 3 * kTileM) * kAStridePacked + kk,
                     kAStridePacked, lane);
      load_fragment8(b_frag0, b_tile + local_col * kBStridePacked + kk,
                     kBStridePacked, lane);
      load_fragment8(b_frag1,
                     b_tile + (local_col + kTileN) * kBStridePacked + kk,
                     kBStridePacked, lane);
      du_mma_sync(acc00, a_frag0, b_frag0, acc00);
      du_mma_sync(acc01, a_frag0, b_frag1, acc01);
      du_mma_sync(acc10, a_frag1, b_frag0, acc10);
      du_mma_sync(acc11, a_frag1, b_frag1, acc11);
      du_mma_sync(acc20, a_frag2, b_frag0, acc20);
      du_mma_sync(acc21, a_frag2, b_frag1, acc21);
      du_mma_sync(acc30, a_frag3, b_frag0, acc30);
      du_mma_sync(acc31, a_frag3, b_frag1, acc31);
      load_fragment8(a_frag4,
                     a_tile + (local_row + 4 * kTileM) * kAStridePacked + kk,
                     kAStridePacked, lane);
      load_fragment8(a_frag5,
                     a_tile + (local_row + 5 * kTileM) * kAStridePacked + kk,
                     kAStridePacked, lane);
      load_fragment8(a_frag6,
                     a_tile + (local_row + 6 * kTileM) * kAStridePacked + kk,
                     kAStridePacked, lane);
      load_fragment8(a_frag7,
                     a_tile + (local_row + 7 * kTileM) * kAStridePacked + kk,
                     kAStridePacked, lane);
      du_mma_sync(acc40, a_frag4, b_frag0, acc40);
      du_mma_sync(acc41, a_frag4, b_frag1, acc41);
      du_mma_sync(acc50, a_frag5, b_frag0, acc50);
      du_mma_sync(acc51, a_frag5, b_frag1, acc51);
      du_mma_sync(acc60, a_frag6, b_frag0, acc60);
      du_mma_sync(acc61, a_frag6, b_frag1, acc61);
      du_mma_sync(acc70, a_frag7, b_frag0, acc70);
      du_mma_sync(acc71, a_frag7, b_frag1, acc71);
    }

    // Last stage only: prefetch the epilogue's per-row x_scale and
    // per-column weight_scale values into registers (uniform branch: k0 is
    // block-uniform, so the barrier below is reached by every thread; the
    // row < m guard keeps the x_scale reads in bounds for any m tail).
    if (k0 + kStageK >= k) {
      const int r = lane & 15;
      const int c4 = lane >> 4;
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        const int row = base_row + i * kTileM + r;
        xs_m[i] = (row < m) ? x_scale[row] : 0.0f;
      }
      ws0 = *reinterpret_cast<const float4*>(weight_scale + base_col +
                                             4 * c4);
      ws1 = *reinterpret_cast<const float4*>(weight_scale + base_col +
                                             kTileN + 4 * c4);
    }
    // Protect the LDS buffer from the next stage's cooperative overwrite.
    __syncthreads();
  }

  store_prefill_fragment_coalesced_scaled(acc00, xs_m[0], ws0, out, base_row,
                                          base_col, m, n, lane);
  store_prefill_fragment_coalesced_scaled(acc01, xs_m[0], ws1, out, base_row,
                                          base_col + kTileN, m, n, lane);
  store_prefill_fragment_coalesced_scaled(acc10, xs_m[1], ws0, out,
                                          base_row + kTileM, base_col, m, n,
                                          lane);
  store_prefill_fragment_coalesced_scaled(acc11, xs_m[1], ws1, out,
                                          base_row + kTileM,
                                          base_col + kTileN, m, n, lane);
  store_prefill_fragment_coalesced_scaled(acc20, xs_m[2], ws0, out,
                                          base_row + 2 * kTileM, base_col, m,
                                          n, lane);
  store_prefill_fragment_coalesced_scaled(acc21, xs_m[2], ws1, out,
                                          base_row + 2 * kTileM,
                                          base_col + kTileN, m, n, lane);
  store_prefill_fragment_coalesced_scaled(acc30, xs_m[3], ws0, out,
                                          base_row + 3 * kTileM, base_col, m,
                                          n, lane);
  store_prefill_fragment_coalesced_scaled(acc31, xs_m[3], ws1, out,
                                          base_row + 3 * kTileM,
                                          base_col + kTileN, m, n, lane);
  store_prefill_fragment_coalesced_scaled(acc40, xs_m[4], ws0, out,
                                          base_row + 4 * kTileM, base_col, m,
                                          n, lane);
  store_prefill_fragment_coalesced_scaled(acc41, xs_m[4], ws1, out,
                                          base_row + 4 * kTileM,
                                          base_col + kTileN, m, n, lane);
  store_prefill_fragment_coalesced_scaled(acc50, xs_m[5], ws0, out,
                                          base_row + 5 * kTileM, base_col, m,
                                          n, lane);
  store_prefill_fragment_coalesced_scaled(acc51, xs_m[5], ws1, out,
                                          base_row + 5 * kTileM,
                                          base_col + kTileN, m, n, lane);
  store_prefill_fragment_coalesced_scaled(acc60, xs_m[6], ws0, out,
                                          base_row + 6 * kTileM, base_col, m,
                                          n, lane);
  store_prefill_fragment_coalesced_scaled(acc61, xs_m[6], ws1, out,
                                          base_row + 6 * kTileM,
                                          base_col + kTileN, m, n, lane);
  store_prefill_fragment_coalesced_scaled(acc70, xs_m[7], ws0, out,
                                          base_row + 7 * kTileM, base_col, m,
                                          n, lane);
  store_prefill_fragment_coalesced_scaled(acc71, xs_m[7], ws1, out,
                                          base_row + 7 * kTileM,
                                          base_col + kTileN, m, n, lane);
}

// ---------------------------------------------------------------------------
// Generic scalar fallback: one thread per output element. Exact int32 dot
// over K, then fused x_scale * weight_scale, then bf16 store. Handles every
// unmatched (m, n, k), including all small-M API cases (M=2, M=16) and any
// M with the same (K, N). For the exact (k, n) == (2048, 6144) the weight is
// the n-major packed [N, K] layout (produced once out of the timed region),
// which the fallback decodes; every other (k, n) keeps the identity [K, N]
// layout.
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
  const int8_t* a_row = x_q + static_cast<int64_t>(row) * k;
  // Iteration 5: for the exact (k, n) == (2048, 6144) the packed_weight
  // buffer holds the n-major transpose packed[col*k + kk] = raw[kk*n + col]
  // (produced once out of the timed region), so the fallback decodes that
  // layout there (keeps the paired M=2/M=16 and M=3072 API shapes with the
  // same (K, N) byte-exact); every other (k, n) keeps the identity [K, N]
  // row-major copy.
  const bool packed_nmajor = (k == kTargetK && n == kTargetN);
  const int8_t* b_ptr =
      packed_nmajor ? weight + static_cast<int64_t>(col) * k : weight + col;
  const int b_stride = packed_nmajor ? 1 : n;
  for (int kk = 0; kk < k; ++kk) {
    acc += static_cast<int32_t>(a_row[kk]) * static_cast<int32_t>(b_ptr[0]);
    b_ptr += b_stride;
  }
  const float scaled =
      static_cast<float>(acc) * x_scale[row] * weight_scale[col];
  out[linear] = __float2bfloat16(scaled);
}

// ---------------------------------------------------------------------------
// Identity packing (bootstrap; outside the timed region and outside Graph
// capture; the packed buffer keeps the same byte count K*N and the same
// allocated address, so the layout is graph-stable).  launch_pack_w8a8_weight
// is an identity device-to-device copy for every (k, n): the packed weight
// is byte-identical to the raw [K, N] weight and the packed scale is
// byte-identical to the raw [N] scale.  Later Parallel explore rounds may
// replace these kernels with a real layout transform and must then update
// the matching GEMM interpretation; the identity D2D copy is the generic
// fallback for unmatched (k, n).
// ---------------------------------------------------------------------------
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

// Exact-shape n-major pack (iteration 5): for (k, n) == (2048, 6144) the
// weight is transposed once, outside the timed region and out of Graph
// capture, into packed[col * k + kk] = raw[kk * n + col] so every col_major
// B fragment is 8 contiguous K bytes in the packed buffer (and in the
// staged n-major LDS tile).  Same byte count and same buffer as the
// identity pack, so captured addresses are unchanged; correctness is
// checked against the raw logical [K, N] weight.  One thread per output
// byte; runs once during weight prep.
__global__ __launch_bounds__(256) void w8a8_pack_nmajor_i8_kernel(
    const int8_t* __restrict__ raw,
    int8_t* __restrict__ packed,
    int k,
    int n) {
  const int64_t total = static_cast<int64_t>(k) * n;
  const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x +
                     threadIdx.x;
       idx < total;
       idx += stride) {
    const int col = static_cast<int>(idx / k);
    const int kk = static_cast<int>(idx - static_cast<int64_t>(col) * k);
    packed[idx] = raw[static_cast<int64_t>(kk) * n + col];
  }
}

__global__ __launch_bounds__(256) void w8a8_pack_scale_identity_kernel(
    const float* __restrict__ raw,
    float* __restrict__ packed,
    int count) {
  const int linear = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
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
  (void)workspace;  // no split-K: the GEMM does not use the workspace
  (void)workspace_bytes;
  auto* out_bf16 = reinterpret_cast<bf16_t*>(out);

  // Explicit dispatch. The assigned shape (M=4096, N=6144, K=2048) takes the
  // iteration-5 packed-B DUMMA 256x64 path (w8a8_dumma_256x64x64_packedb_kernel
  // consumes the n-major packed [N, K] weight produced by
  // launch_pack_w8a8_weight; the identity-layout kernels stay compiled for
  // re-dispatch only together with an identity pack revert); every other
  // (m, n, k) - including small-M API cases (M=2, M=16) and M=3072 with the
  // same (K, N) - takes the scalar fallback, which decodes the n-major
  // packed layout when (k, n) == (2048, 6144) and the identity [K, N]
  // layout otherwise.
  if (m == kTargetM && n == kTargetN && k == kTargetK) {
    const dim3 grid(kTargetN / kBlockN, kTargetM / kBlockM256);
    const dim3 block(kBlockThreads);
    hipLaunchKernelGGL(w8a8_dumma_256x64x64_packedb_kernel,
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
  // Iteration 5: the exact (k, n) == (2048, 6144) weight is packed once,
  // outside the timed region and out of Graph capture, into the n-major
  // [N, K] layout packed[col*k + kk] = raw[kk*n + col] (same byte count and
  // same buffer -> graph-stable addresses) so every col_major B fragment is
  // 8 contiguous K bytes; every other (k, n) keeps the identity
  // device-to-device copy (the scalar fallback decodes the n-major layout
  // when (k, n) matches and the identity layout otherwise).
  constexpr int kBlock = 256;
  const dim3 block(kBlock);

  const int64_t weight_count = static_cast<int64_t>(k) * n;
  const dim3 weight_grid(
      static_cast<unsigned>((weight_count + kBlock - 1) / kBlock));
  if (k == kTargetK && n == kTargetN) {
    hipLaunchKernelGGL(w8a8_pack_nmajor_i8_kernel,
                       weight_grid, block, 0, stream,
                       raw_weight, packed_weight, k, n);
  } else {
    hipLaunchKernelGGL(w8a8_pack_identity_kernel,
                       weight_grid, block, 0, stream,
                       raw_weight, packed_weight, weight_count);
  }

  const int64_t scale_count = n;
  const dim3 scale_grid(
      static_cast<unsigned>((scale_count + kBlock - 1) / kBlock));
  hipLaunchKernelGGL(w8a8_pack_scale_identity_kernel,
                     scale_grid, block, 0, stream,
                     weight_scale, packed_weight_scale, n);
  (void)hipGetLastError();
}
// @@end
