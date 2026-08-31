// @@variant shape=glm_tp8_q_b_proj_m4096 commit=a9fcfc5e2ac1234af51849905834a850b5ef3448 added=2026-08-31
//   median_us=220.1 p90_us=220.5 speedup=65.35 baseline_us=1.439e+04
//   source=glm5-2-dsh-tp8-m4096-1-e6a280a2
// INT8 W8A8 GEMM HIP implementation for Hygon K500SM_AI / gfx928.
//
// Worker: worker_1 (physical GPU 1), assigned shapes:
//   glm_tp8_q_b_proj_m4096  : M=4096, N=2048, K=2048
//   glm_tp8_kv_b_proj_m4096 : M=4096, N=3584, K=512
//
// Logical operation (exact contract):
//   out[m, n] = bf16( int32_dot(x_q[m, :], raw_weight[:, n])
//                     * x_scale[m] * weight_scale[n] )
//
// Iteration 3 (first valid HIP experiment; iterations 1-2 were killed in the
// agent infrastructure before any proposal).  Accepted-best source digest
// a64b6250... with one bounded mechanism: n-major packed B + col_major B
// fragments for the exact glm_tp8_q_b_proj_m4096 (k,n) == (2048, 2048).
//   * Large-prefill path (m >= 128 with exact 128x64x64 geometry): native
//     INT8 DUMMA m16n16k32 with int32 accumulation; one 128x64 output tile
//     per block; four wavefronts (256 threads); each wave owns a 64x32
//     quadrant built from eight m16n16k32 int32 accumulator fragments; the
//     block cooperatively vector-loads A[128,64] and B[64,64] into one
//     single-buffered LDS stage (15,360 B total: 128*80 + 64*80 with 16 B
//     padding per row for bank skew); two __syncthreads per stage; fused
//     dot * x_scale[m] * weight_scale[n] epilogue stored directly as bf16
//     from the accumulator fragments using the verified gfx928 int8
//     m16n16k32 lane mapping (row = lane & 15, col = (lane >> 4) + 4*i).
//     A is read with the library du_load_matrix_sync row-major loader.
//     B is layout-templated:
//       - kNMajorB == true  (exact (k,n) == (2048, 2048), q_b_proj): weight
//         is packed once to n-major packed[n*K + k] == raw[k*N + n] outside
//         the timed region; the stage stores B n-major (n row, k contiguous,
//         80 B row stride); each B fragment is one contiguous 8-byte LDS run
//         per lane, loaded with the lineage-validated load_fragment8 (one
//         ds_read2_b64) - kills the 32 ds_read_u8 byte-gathers + mask/OR
//         reassembly VALU seen in the accepted-best ISA.
//       - kNMajorB == false (any other shape, e.g. kv_b_proj (512, 3584)):
//         raw [K, N] row-major identity layout, row-major B loader (the
//         accepted-best path, byte-identical).
//     Grid dim3(N/64, M/128) = 1024 blocks (q_b_proj) / 1792 blocks
//     (kv_b_proj) dwarfs the 120 CUs, so no split-K is needed.
//   * Generic scalar int8/int32 fallback for every unmatched (m, n, k),
//     including all small-M API cases (M=2, M=16), M tails, and M in
//     (0, 128) with the same (K, N); it decodes the n-major packed layout
//     for the exact (2048, 2048) and the identity [K, N] layout otherwise.
//   * launch_pack_w8a8_weight: n-major device-to-device permutation for
//     (k, n) == (2048, 2048), identity copy for every other (k, n) (both the
//     int8 weight and the fp32 scale).  Packing never happens inside the
//     timed GEMM and keeps the same byte count (graph-stable addresses).
//
// Iteration 4 (tile-aspect round): the exact (k, n) == (2048, 2048) q_b_proj
// arm switches from the 128x64 tile to a 128x128 tile
// (w8a8_dumma_prefill_128x128_kernel).  With the n-major packed B + 8-byte
// load_fragment8 B reads in place (iteration 3), the remaining LDS-read cost
// is dominated by the A-side row-major fragment loads (4 x ds_read2_b32 per
// kk per wave).  A 128x128 tile with four 64x64 quadrants keeps the same
// 4-wave x 8-accumulator-fragment count per wave (8 -> 16 m16n16k32 MMACs
// per kk from 8 LDS fragment reads, i.e. per-MMAC LDS reads drop 6/8 -> 8/16
// = -33%), and the A-side global re-read halves (A reuse 32 -> 16, A traffic
// 268 -> 134 MiB) while B reuse stays 32 (134 MiB), balancing the per-byte
// A:B traffic at 1:1 and cutting total tile traffic 402 -> 268 MiB (-33%).
// LDS grows 15,360 -> 20,480 B/block (still 2 blocks/CU = 40,960 B <= 64
// KiB); the grid becomes (N/128, M/128) = 512 blocks (~4.3/CU).  The 16
// accumulator fragments cost ~+32 VGPR; arch VGPR is expected <= 128 so the
// accepted kernel's 2-blocks/CU co-residency is preserved (falsified if the
// exact code object shows > 128 VGPR -> 1 block/CU).  kv_b_proj keeps the
// byte-identical 128x64 kNMajorB=false arm and the scalar fallback is
// untouched.
//
// Iteration 5 (packing round): the exact (k, n) == (2048, 2048) q_b_proj
// weight is re-packed once (outside timing/Graph, same byte count, same
// graph-stable buffer) into a swizzled 64-k-stage-major layout
//   packed[((k0*8 + kc)*n + col)*8 + b] == raw[kk*n + col],
//   kk = k0*64 + kc*8 + b  (k0 = 64-k stage, kc = 8-byte sub-chunk, b = byte),
// and the 128x128 kernel stages each B tile as 16-byte plane chunks into an
// LDS plane layout b_tile[kc][n][8] (plane stride 1024 B, n stride 8 B).
// The exact code object of the accepted kernel shows every B fragment read
// is an 8-byte LDS access at (lane&15)*80 + (lane>>4)*8 (n-major stride-80
// rows), and PMC counts 6,291,456 bank conflicts over 1,048,576 LDS slots
// (6.0/slot) with 4,078,494 LDS waits (3.9/slot): the loop is LDS-latency
// bound and the stride-80 rows alias bank phases every 8 rows (80*8 == 640
// == 0 mod 128), so every fragment read conflicts ~4-6-way.  In the plane
// layout a lane's 8 bytes sit at (lane>>4)*1024 + (lane&15)*8 relative to
// the fragment origin: 16 lanes per 128-B phase cover all 32 banks exactly
// once, i.e. B fragment reads become lane-linear and zero-conflict (4-cycle
// minimum for 512 B).  Staging reads stay coalesced (each 16-B chunk is two
// n rows of one sub-chunk -> one aligned int4) and LDS writes are 2-way
// (the 16-B optimum).  Same fragment operand bytes, same k0-outer/kk-inner
// int32 accumulation -> bit-identical results.  The scalar fallback decodes
// the new pack for (2048,2048) only; kv_b_proj keeps the byte-identical
// 128x64 kNMajorB=false identity path.
//
// Iteration 6 (epilogue round): the fused dot * x_scale[m] * weight_scale[n]
// -> bf16 epilogue has been in-kernel since iteration 1 and the workspace is
// unused ((void)workspace; no split-K, no combine pass anywhere in the call
// chain), so the remaining epilogue inefficiency of the exact (2048, 2048)
// 128x128 arm is the STORE pattern: store_prefill_fragment issues one
// 2-byte bf16 store per lane per element - 64 scattered stores per wave per
// fragment set, each wavefront store touching 16 rows x 4 columns so every
// 32-B sector is only 25% utilized (PMC: 131,072 vmem_write_instructions =
// 512 blocks x 4 waves x 64).  The new epilogue transposes each 16x16
// fragment's 4-element groups inside the 4-lane column group (lanes r,
// r+16, r+32, r+48; two 2x2 steps with __shfl_xor 16 then 32 - the
// lineage-validated 4x4 register transpose accepted on the sibling TP8
// workers 0/2, same DTK 26.04/gfx928), so lane (r, c4) owns the four
// CONTIGUOUS columns 4*c4 .. +3 and writes ONE 8-byte store per lane (100%
// store sector efficiency; vmem_write 131,072 -> 32,768).  Only int32
// values move between lanes: the per-element multiply order
// (float(dot) * x_scale[row] * weight_scale[col]) and the bf16 rounding are
// unchanged, so stored bits are identical.  The per-row x_scale (4 rows per
// lane) and per-column weight_scale float4s (4 per lane) are additionally
// register-batched on the last K stage (before the final protective
// __syncthreads), so the epilogue is pure compute + 16 coalesced stores with
// no interleaved vmem loads (vmem_read ~413,696 -> ~278k).  kv_b_proj keeps
// the byte-identical 128x64 kNMajorB=false arm (old per-element epilogue)
// and the generic scalar fallback is untouched.
//
// Iteration 7 (compute-pipeline round): the exact (k, n) == (2048, 2048)
// 128x128 arm loads its four A fragments per kk with the lineage-validated
// direct load_fragment8 (one ds_read2_b64 straight into the v_mmac operand)
// instead of the library du_load_matrix_sync row_major loader.  The exact
// code object shows the library loader lowers to 8 x ds_read2_b32 plus a
// redundant byte-reassembly chain per fragment (~7 VALU: v_and 0xff00 /
// 0xff0000 / 0xff000000 + v_or_b32_sdwa + v_or3, ~50-56 VALU per stage per
// wave) between the LDS read and the first MMAC, and PMC counts 3,009,148
// LDS waits (2.3/slot) against a latency-bound loop running at 1 block/CU
// (160 VGPR).  du_mma.hpp defines matrix_a row_major int8 as x[i] =
// p[(lane&15)*ldm + (lane>>4)*8 + i] (8 consecutive bytes, memory order)
// and du_mma_sync passes reinterpret<long>(x) unchanged to v_mmac, so
// load_fragment8 produces byte-identical operand values and the int32
// accumulation is bit-identical; only the redundant VALU reassembly (and its
// lgkmcnt wait states) is removed from the LDS->MMAC critical path, letting
// the 32-MMAC burst issue back-to-back.  Expected PMC: valu_instructions
// 8.66M -> ~4.8-5.6M (-35..-45%), lds_instructions 1.31M -> ~1.05M (-20%,
// 8 A ds_read2_b32 + 4 B ds_read2_b64 -> 8 ds_read2_b64 per stage),
// lds_wait_instructions 3.01M -> ~2.2-2.5M, lds_bank_conflicts
// approximately unchanged (A rows keep the 2-way stride-80 aliasing),
// vmem_read/vmem_write and the 32 v_mmac/stage unchanged.  kv_b_proj keeps
// the byte-identical 128x64 kNMajorB=false arm (unchanged lowering) and the
// generic scalar fallback is untouched.

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
constexpr int kBlockM = 128;
constexpr int kBlockN = 64;
constexpr int kStageK = 64;
constexpr int kBPad = 16;  // 64 -> 80-byte LDS row stride (five bank phases)
constexpr int kBStride = kBlockN + kBPad;
constexpr int kAStride = kStageK + kBPad;  // 64 -> 80-byte A row stride
constexpr int kBlockThreads = 4 * kWaveSize;

// Exact (k, n) pair whose weight buffer is packed n-major
// (packed[n*K + k] == raw[k*N + n]): glm_tp8_q_b_proj_m4096 only.
constexpr int kPackNMajorK = 2048;
constexpr int kPackNMajorN = 2048;

using bf16_t = hip_bfloat16;

// ---------------------------------------------------------------------------
// Direct fragment epilogue for one m16n16k32 accumulator fragment.
// Verified gfx928 int8 m16n16k32 accumulator ownership (matches
// du_store_matrix_sync): lane & 15 selects the row, lane >> 4 selects
// col % 4, and x[i] maps to columns col%4 + 4*i.  The scale multiply order
// (float(dot) * x_scale[row] * weight_scale[col]) and the bf16 rounding are
// identical to the harness reference ((dot.float() * x_scale) * ws.T then
// .to(bfloat16)), so stored bf16 bits match exactly.
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

// ---------------------------------------------------------------------------
// Iteration 6 (epilogue round): register-batched COALESCED direct-fragment
// epilogue for the exact (2048, 2048) 128x128 arm.  The per-element variant
// (store_prefill_fragment) above issues one 2-byte bf16 store per lane per
// element: 64 scattered stores per wave per fragment set, each touching 16
// rows x 4 columns, i.e. every 32-B store sector only 25% utilized.  This
// epilogue transposes the 4-element groups within each 4-lane column group
// (lanes r, r+16, r+32, r+48 - a 4x4 transpose, two 2x2 steps with
// shfl_xor 16 then 32; the lineage-validated pattern accepted on the sibling
// TP8 workers 0/2 on this DTK, which lowers __shfl_xor to ds_bpermute at the
// block tail where the LDS pipe is idle), so lane (r, c4) ends up holding
// the four CONTIGUOUS columns 4*c4 .. 4*c4+3, converts them to bf16, packs
// 4 bf16 (8 B) and writes ONE 8-byte store per lane (100% store sector
// efficiency).  Only the int32 values are re-routed between lanes; the
// per-element scale multiply order (float(dot) * x_scale[row] *
// weight_scale[col]) and the __float2bfloat16 rounding are unchanged, so
// the stored bits are identical to the per-element store.  xs/ws come from
// caller registers (batched per wave on the last K stage), so the epilogue
// issues no vmem loads.  The row >= m guard is wavefront-uniform (all 64
// lanes of a wave share the same 16-row window), so the shuffles never mix
// active and inactive lanes; base_col is a multiple of 64 and n*2 a
// multiple of 8, so the 8-byte store is aligned.
// ---------------------------------------------------------------------------
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

  // Lane (r, c4) now owns columns base_col + 4*c4 .. +3 (8 B, 8-byte
  // aligned: base_col is a multiple of 16, n*2 is a multiple of 8); ws is
  // the float4 at weight_scale + base_col + 4*c4 preloaded by the caller.
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
// Direct 8-byte LDS fragment load (lineage-validated on gfx928/DTK 26.04 for
// m16n16k32 int8 fragments).  du_load_matrix_sync's int8 loaders assign
// x[0..7] = 8 consecutive bytes at (lane & 15) * ldm + ((lane >> 4) << 3) for
// matrix_b col_major, but the compiler lowers that to per-byte ds_read_u8 +
// mask/OR reassembly VALU.  Writing the same 8 bytes directly into the
// fragment storage keeps the operand bit pattern identical (exact int32
// accumulation unchanged) and lets the compiler emit one ds_read2_b64 per
// fragment straight into the v_mmac operand.
// ---------------------------------------------------------------------------
__device__ __forceinline__ void load_fragment8(
    DUFragmentBase<signed char, 8>& frag,
    const signed char* __restrict__ base,
    int stride,
    int lane) {
  const int off = (lane & 15) * stride + ((lane >> 4) << 3);
  *reinterpret_cast<unsigned long long*>(&frag.x[0]) =
      *reinterpret_cast<const unsigned long long*>(base + off);
}

// ---------------------------------------------------------------------------
// Iteration 5 (packing round): 8-byte fragment load for the swizzled plane
// LDS layout b_tile[kc][n][8] (kc = 8-byte k sub-chunk, n row stride 8 B,
// plane stride 1024 B).  Lane l reads the same 8 consecutive k bytes as
// load_fragment8 (n row = lane&15, k chunk = lane>>4) but from
//   (lane >> 4) * 1024 + (lane & 15) * 8
// relative to the fragment origin, so the 64 lanes hit 16 distinct bank
// pairs per 128-B phase (all 32 banks once) -> zero bank conflicts, the
// 4-cycle minimum for a 512-B fragment.
// ---------------------------------------------------------------------------
__device__ __forceinline__ void load_fragment8_plane(
    DUFragmentBase<signed char, 8>& frag,
    const signed char* __restrict__ base,
    int lane) {
  const int off = ((lane >> 4) << 10) + ((lane & 15) << 3);
  *reinterpret_cast<unsigned long long*>(&frag.x[0]) =
      *reinterpret_cast<const unsigned long long*>(base + off);
}

// ---------------------------------------------------------------------------
// Compile-time B-fragment layout selector.  row_major / col_major are tag
// types in du::dumma, so the layout template argument must be selected as a
// type (a conditional expression over type names is not a valid template
// argument).  kNMajorB == true  (n-major packed B, exact (2048, 2048)) ->
// col_major fragments (8 contiguous k bytes per lane, load_fragment8);
// kNMajorB == false (raw [K, N] row-major B) -> row_major fragments (the
// accepted-best loader).
// ---------------------------------------------------------------------------
template <bool kNMajorB>
struct b_frag_layout {
  using type = row_major;
};
template <>
struct b_frag_layout<true> {
  using type = col_major;
};

// ---------------------------------------------------------------------------
// Large-M prefill: 128x64 output tile per block, four wavefronts of 64 lanes.
// Each wave owns a 64x32 quadrant (eight m16n16k32 int32 accumulators); the
// block cooperatively stages A[128,64] from x_q (row-major, stride k) and
// B[64,64] into a single-buffered LDS stage.  Two barriers per stage: one
// after the cooperative load, one before the next stage overwrites LDS.
// A fragments use the library du_load_matrix_sync row-major loader.  B is
// templated on its staged layout:
//   * kNMajorB == true:  B is packed n-major (packed[n*K + k]) and staged
//     n-major (n row, 80 B stride, k contiguous); each B fragment is 8
//     contiguous k bytes per lane, loaded by load_fragment8 (one 8-byte LDS
//     read per fragment), eliminating the ds_read_u8 byte-gather + VALU
//     reassembly of the accepted-best kernel.
//   * kNMajorB == false: B stays raw [K, N] row-major, staged k-major with
//     row-major library fragments (accepted-best path, unchanged).
// Dispatch guarantees m % 128 == 0, n % 64 == 0, k % 64 == 0, so every
// global load/store is in-bounds and 16-byte aligned.
// ---------------------------------------------------------------------------
template <bool kNMajorB>
__global__ __launch_bounds__(kBlockThreads) void w8a8_dumma_prefill_128x64_kernel(
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

  // Single-buffered stage: A[128, 80] + B[64, 80] = 15,360 B/block
  // (4 blocks/CU fit the 64 KiB LDS budget; the padded 80-byte strides are
  // 16-byte-aligned and break the 64-byte LDS bank periodicity).
  __shared__ __align__(16) int8_t a_tile[kBlockM * kAStride];
  __shared__ __align__(16) int8_t b_tile[kStageK * kBStride];

  DUFragment<matrix_a, kTileM, kTileN, kTileK, signed char, row_major>
      a_frag0, a_frag1;
  DUFragment<matrix_b, kTileM, kTileN, kTileK, signed char,
             typename b_frag_layout<kNMajorB>::type>
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

  // Cooperative staging: A[128,64] is 512 int4s (two per thread), B[64,64]
  // is 256 int4s (one per thread).
  //   A: thread tid owns row (tid*16)/64 and 16-byte column group
  //      (tid*16)%64 for rows [0,64) and [64,128).
  //   B (kNMajorB == false): raw [K, N] row-major; thread tid owns K row
  //      tid>>2 and the 16-byte N-column group (tid&3)*16 (16 consecutive N
  //      values at a fixed K row -> one aligned int4 in global and in LDS).
  //   B (kNMajorB == true): packed n-major; thread tid owns N row tid>>2 and
  //      the 16-byte K-column group (tid&3)*16 (16 consecutive K values at a
  //      fixed N row -> one aligned int4 in global and in the n-major LDS).
  const int vector_byte_offset = tid * static_cast<int>(sizeof(int4));
  const int stage_row = vector_byte_offset / kStageK;  // 0..63
  const int stage_col = vector_byte_offset - stage_row * kStageK;
  const int b_k = tid >> 2;            // K row (false) / N row (true)
  const int b_nc = (tid & 3) * 16;     // N group (false) / K group (true)

  for (int k0 = 0; k0 < k; k0 += kStageK) {
    *reinterpret_cast<int4*>(a_tile + stage_row * kAStride + stage_col) =
        *reinterpret_cast<const int4*>(
            x_q + static_cast<int64_t>(m0 + stage_row) * k + k0 + stage_col);
    *reinterpret_cast<int4*>(a_tile +
                             (stage_row + kBlockM / 2) * kAStride + stage_col) =
        *reinterpret_cast<const int4*>(
            x_q + static_cast<int64_t>(m0 + stage_row + kBlockM / 2) * k +
            k0 + stage_col);
    if constexpr (kNMajorB) {
      // n-major packed B: 16 consecutive k bytes of the (n0 + b_k) row.
      *reinterpret_cast<int4*>(b_tile + b_k * kBStride + b_nc) =
          *reinterpret_cast<const int4*>(
              weight + static_cast<int64_t>(n0 + b_k) * k + k0 + b_nc);
    } else {
      // raw [K, N] row-major B: 16 consecutive n bytes of the (k0 + b_k) row.
      *reinterpret_cast<int4*>(b_tile + b_k * kBStride + b_nc) =
          *reinterpret_cast<const int4*>(
              weight + static_cast<int64_t>(k0 + b_k) * n + n0 + b_nc);
    }
    __syncthreads();

    // Each wave consumes its 64x32 quadrant: eight m16n16k32 MMACs per kk.
    // Accumulation order: k0-outer over 64-K stages, kk-inner (kk=0 then
    // kk=32), matching the reference int32 accumulation.
#pragma unroll
    for (int kk = 0; kk < kStageK; kk += kTileK) {
      if constexpr (kNMajorB) {
        // n-major LDS tile: n rows stride kBStride, k contiguous within a
        // row; col_major fragment slots are 8 consecutive k bytes at
        // (lane & 15) * kBStride + ((lane >> 4) << 3) relative to the
        // fragment origin -> one ds_read2_b64 per fragment per lane.
        load_fragment8(b_frag0, b_tile + local_col * kBStride + kk, kBStride,
                       lane);
        load_fragment8(b_frag1,
                       b_tile + (local_col + kTileN) * kBStride + kk,
                       kBStride, lane);
      } else {
        du_load_matrix_sync(b_frag0, b_tile + kk * kBStride + local_col,
                            kBStride);
        du_load_matrix_sync(b_frag1,
                            b_tile + kk * kBStride + local_col + kTileN,
                            kBStride);
      }
      du_load_matrix_sync(a_frag0, a_tile + local_row * kAStride + kk,
                          kAStride);
      du_load_matrix_sync(a_frag1,
                          a_tile + (local_row + kTileM) * kAStride + kk,
                          kAStride);
      du_mma_sync(acc00, a_frag0, b_frag0, acc00);
      du_mma_sync(acc01, a_frag0, b_frag1, acc01);
      du_mma_sync(acc10, a_frag1, b_frag0, acc10);
      du_mma_sync(acc11, a_frag1, b_frag1, acc11);
      du_load_matrix_sync(a_frag0,
                          a_tile + (local_row + 2 * kTileM) * kAStride + kk,
                          kAStride);
      du_load_matrix_sync(a_frag1,
                          a_tile + (local_row + 3 * kTileM) * kAStride + kk,
                          kAStride);
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
  store_prefill_fragment(acc00, x_scale, weight_scale, out, base_row,
                         base_col, m, n, lane);
  store_prefill_fragment(acc01, x_scale, weight_scale, out, base_row,
                         base_col + kTileN, m, n, lane);
  store_prefill_fragment(acc10, x_scale, weight_scale, out,
                         base_row + kTileM, base_col, m, n, lane);
  store_prefill_fragment(acc11, x_scale, weight_scale, out,
                         base_row + kTileM, base_col + kTileN, m, n, lane);
  store_prefill_fragment(acc20, x_scale, weight_scale, out,
                         base_row + 2 * kTileM, base_col, m, n, lane);
  store_prefill_fragment(acc21, x_scale, weight_scale, out,
                         base_row + 2 * kTileM, base_col + kTileN, m, n,
                         lane);
  store_prefill_fragment(acc30, x_scale, weight_scale, out,
                         base_row + 3 * kTileM, base_col, m, n, lane);
  store_prefill_fragment(acc31, x_scale, weight_scale, out,
                         base_row + 3 * kTileM, base_col + kTileN, m, n,
                         lane);
}

// ---------------------------------------------------------------------------
// Iteration 4 (tile-aspect round) + iteration 5 (packing round): 128x128
// tile for the exact q_b_proj (k, n) == (2048, 2048) swizzled-pack B arm.
// Four wavefronts of 64 lanes; each wave owns a 64x64 quadrant (four A x
// four B m16n16k32 fragments = 16 MMACs per kk, 16 int32 accumulator
// fragments).  The block cooperatively stages A[128,64] (2 int4/thread,
// 80-byte-strided rows) and B[128,64] from the iteration-5 swizzled pack (2
// int4/thread, 16-byte plane chunks) into a single-buffered LDS stage:
// A[128,80] (10,240 B) + B plane tile [8][128][8] (8,192 B) = 18,432
// B/block (2 blocks/CU = 36,864 B <= 64 KiB); two __syncthreads per stage.
// A fragments use the library du_load_matrix_sync row-major loader
// (4 x ds_read2_b32 per kk per wave), B fragments use load_fragment8_plane
// (lane-linear 8-byte reads, zero LDS bank conflicts).  A-side global
// re-read halves (A reuse 32 -> 16, 268 -> 134 MiB) while B reuse stays 32
// (134 MiB): per-byte A:B tile traffic balances 1:1, total 402 -> 268 MiB
// (-33%).  Grid dim3(N/128, M/128) = 512 blocks (~4.3/CU).  Dispatch
// guarantees m % 128 == 0, n == 2048, k % 64 == 0, so every global
// load/store is in-bounds and 16-byte aligned.  Accumulation order stays
// k0-outer / kk-inner with the same element-to-slot fragment mapping, so the
// int32 accumulation is bit-identical to the accepted kernel.
// ---------------------------------------------------------------------------
__global__ __launch_bounds__(kBlockThreads) void w8a8_dumma_prefill_128x128_kernel(
    const int8_t* __restrict__ x_q,
    const int8_t* __restrict__ weight,
    const float* __restrict__ x_scale,
    const float* __restrict__ weight_scale,
    bf16_t* __restrict__ out,
    int m,
    int n,
    int k) {
  constexpr int kBlockM128 = 128;
  constexpr int kBlockN128 = 128;
  constexpr int kTileStride = kStageK + kBPad;  // 64 -> 80-byte row stride
  constexpr int kTileBytes = kBlockM128 * kTileStride;  // 10,240 B per tile
  // Iteration 5 (packing round): B tile is the swizzled plane layout
  // [kc][n][8] (8 planes of 128 n rows x 8 B, plane stride 1024 B) so each
  // lane's 8-byte fragment chunk is lane-linear and bank-conflict-free.
  constexpr int kBTileBytes = 8 * kBlockN128 * 8;  // 8,192 B/block
  constexpr int kBTilePlane = 1024;  // 128 n rows x 8 B

  const int tid = static_cast<int>(threadIdx.x);
  const int wave = tid / kWaveSize;
  const int lane = tid % kWaveSize;
  const int wave_row = wave >> 1;
  const int wave_col = wave & 1;
  const int m0 = static_cast<int>(blockIdx.y) * kBlockM128;
  const int n0 = static_cast<int>(blockIdx.x) * kBlockN128;
  const int local_row = wave_row * (kBlockM128 / 2);  // 0 or 64
  const int local_col = wave_col * (kBlockN128 / 2);  // 0 or 64
  const int base_row = m0 + local_row;
  const int base_col = n0 + local_col;

  // Iteration 6 (epilogue round): per-lane register-batched scales.  Each
  // lane owns 4 rows (base_row + 16*i + (lane & 15), i = 0..3) and 4
  // weight_scale float4s (base_col + 16*j + 4*(lane >> 4), j = 0..3) across
  // its 16 fragments; they are loaded exactly once on the last K stage
  // before the final protective barrier, so the coalesced epilogue below is
  // pure compute + 16 eight-byte stores with no interleaved vmem loads.
  float xs_m[4];
  float4 ws_m[4];

  // Single-buffered stage: A[128,80] (10,240 B) + B plane tile [8][128][8]
  // (8,192 B) = 18,432 B/block (2 blocks/CU = 36,864 B <= 64 KiB; the A rows
  // keep the padded 80-byte stride, the B tile is plane-swizzled so fragment
  // reads are lane-linear with zero bank conflicts).
  __shared__ __align__(16) int8_t a_tile[kTileBytes];
  __shared__ __align__(16) int8_t b_tile[kBTileBytes];

  DUFragment<matrix_a, kTileM, kTileN, kTileK, signed char, row_major>
      a_frag0, a_frag1, a_frag2, a_frag3;
  DUFragment<matrix_b, kTileM, kTileN, kTileK, signed char, col_major>
      b_frag0, b_frag1, b_frag2, b_frag3;
  DUFragment<accumulator, kTileM, kTileN, kTileK, int>
      acc00, acc01, acc02, acc03, acc10, acc11, acc12, acc13,
      acc20, acc21, acc22, acc23, acc30, acc31, acc32, acc33;
  du_fill_fragment(acc00, 0);
  du_fill_fragment(acc01, 0);
  du_fill_fragment(acc02, 0);
  du_fill_fragment(acc03, 0);
  du_fill_fragment(acc10, 0);
  du_fill_fragment(acc11, 0);
  du_fill_fragment(acc12, 0);
  du_fill_fragment(acc13, 0);
  du_fill_fragment(acc20, 0);
  du_fill_fragment(acc21, 0);
  du_fill_fragment(acc22, 0);
  du_fill_fragment(acc23, 0);
  du_fill_fragment(acc30, 0);
  du_fill_fragment(acc31, 0);
  du_fill_fragment(acc32, 0);
  du_fill_fragment(acc33, 0);

  // Cooperative staging: A[128,64] is 512 int4s (two per thread) and
  // B[128,64] from the iteration-5 swizzled pack is 512 int4s (two per
  // thread).  Thread tid owns linear int4 slots tid and tid + 256; for A a
  // slot maps to row linear >> 2 and the 16-byte column group
  // (linear & 3) * 16 (an M row of x_q, one aligned int4 in global and in
  // LDS); for B a slot maps to plane linear >> 6 and n-pair linear & 63
  // (two n rows of one 8-byte k sub-chunk, one aligned int4 in the pack and
  // in the plane LDS tile).
  for (int k0 = 0; k0 < k; k0 += kStageK) {
    const int kstage = k0 >> 6;  // 64-k group index inside the swizzled pack
#pragma unroll
    for (int i = 0; i < 2; ++i) {
      const int linear = tid + i * kBlockThreads;
      const int row = linear >> 2;        // 0..127
      const int col = (linear & 3) * 16;  // 0/16/32/48
      *reinterpret_cast<int4*>(a_tile + row * kTileStride + col) =
          *reinterpret_cast<const int4*>(
              x_q + static_cast<int64_t>(m0 + row) * k + k0 + col);
      // Swizzled pack: slot linear -> (kc = linear >> 6, j = linear & 63);
      // the 16-byte chunk is two n rows (n0 + 2j, n0 + 2j + 1) of the 8-byte
      // k sub-chunk kc of stage kstage: one aligned int4 in global (two
      // consecutive n rows of one plane) and one aligned int4 in the plane
      // LDS tile (plane kc, rows 2j..2j+1 -> contiguous 16 B).
      const int b_kc = linear >> 6;  // 0..7
      const int b_j = linear & 63;   // 0..63 (n-pair within the plane)
      *reinterpret_cast<int4*>(b_tile + b_kc * kBTilePlane + b_j * 16) =
          *reinterpret_cast<const int4*>(
              weight + (static_cast<int64_t>(kstage * 8 + b_kc) * n +
                        (n0 + 2 * b_j)) * 8);
    }
    __syncthreads();

    // Each wave consumes its 64x64 quadrant: sixteen m16n16k32 MMACs per kk.
    // Accumulation order: k0-outer over 64-K stages, kk-inner (kk=0 then
    // kk=32), matching the reference int32 accumulation.  All eight fragment
    // loads are issued before the sixteen MMACs (load-all-then-MMAC-all).
    //
    // Iteration 7 (compute-pipeline round): the A fragments are loaded with
    // the same direct 8-byte LDS read as B (load_fragment8) instead of the
    // library du_load_matrix_sync row_major loader.  The library's int8
    // matrix_a row_major loader assigns x[i] = p[(lane&15)*ldm +
    // (lane>>4)*8 + i] (8 consecutive bytes, memory order) and du_mma_sync
    // feeds reinterpret<long>(x) straight into v_mmac, but on this DTK the
    // loader lowers to 8 x ds_read2_b32 + a redundant per-dword byte
    // reassembly chain (~7 VALU: v_and 0xff00/0xff0000/0xff000000 +
    // v_or_b32_sdwa + v_or3 per second dword) sitting between the LDS read
    // and the MMAC issue (exact code object, both the 128x128 and the 128x64
    // symbols).  load_fragment8 fills the same x[0..7] with one 64-bit
    // little-endian write, so the operand bytes are bit-identical and the
    // compiler emits one ds_read2_b64 straight into the v_mmac operand (the
    // same lineage as the B side since iteration 3/5): the LDS->MMAC
    // critical path shortens by ~50 VALU + their lgkmcnt wait states per
    // stage, and the 32-MMAC burst can issue back-to-back after the barrier.
#pragma unroll
    for (int kk = 0; kk < kStageK; kk += kTileK) {
      load_fragment8(a_frag0, a_tile + local_row * kTileStride + kk,
                     kTileStride, lane);
      load_fragment8(a_frag1,
                     a_tile + (local_row + kTileM) * kTileStride + kk,
                     kTileStride, lane);
      load_fragment8(a_frag2,
                     a_tile + (local_row + 2 * kTileM) * kTileStride + kk,
                     kTileStride, lane);
      load_fragment8(a_frag3,
                     a_tile + (local_row + 3 * kTileM) * kTileStride + kk,
                     kTileStride, lane);
      // B fragments in the plane layout: fragment (f, kk) origin is
      // b_tile + (kk >> 3) * 1024 + (local_col + 16f) * 8 and lane l reads
      // the 8 bytes at (lane >> 4) * 1024 + (lane & 15) * 8 (lane-linear,
      // zero bank conflicts); operand bytes are identical to the accepted
      // n-major load_fragment8, so the int32 accumulation is bit-identical.
      const int b_kc0 = kk >> 3;  // 0 (kk == 0) or 4 (kk == 32)
      load_fragment8_plane(b_frag0,
                           b_tile + b_kc0 * kBTilePlane + local_col * 8,
                           lane);
      load_fragment8_plane(b_frag1,
                           b_tile + b_kc0 * kBTilePlane +
                               (local_col + kTileN) * 8,
                           lane);
      load_fragment8_plane(b_frag2,
                           b_tile + b_kc0 * kBTilePlane +
                               (local_col + 2 * kTileN) * 8,
                           lane);
      load_fragment8_plane(b_frag3,
                           b_tile + b_kc0 * kBTilePlane +
                               (local_col + 3 * kTileN) * 8,
                           lane);
      du_mma_sync(acc00, a_frag0, b_frag0, acc00);
      du_mma_sync(acc01, a_frag0, b_frag1, acc01);
      du_mma_sync(acc02, a_frag0, b_frag2, acc02);
      du_mma_sync(acc03, a_frag0, b_frag3, acc03);
      du_mma_sync(acc10, a_frag1, b_frag0, acc10);
      du_mma_sync(acc11, a_frag1, b_frag1, acc11);
      du_mma_sync(acc12, a_frag1, b_frag2, acc12);
      du_mma_sync(acc13, a_frag1, b_frag3, acc13);
      du_mma_sync(acc20, a_frag2, b_frag0, acc20);
      du_mma_sync(acc21, a_frag2, b_frag1, acc21);
      du_mma_sync(acc22, a_frag2, b_frag2, acc22);
      du_mma_sync(acc23, a_frag2, b_frag3, acc23);
      du_mma_sync(acc30, a_frag3, b_frag0, acc30);
      du_mma_sync(acc31, a_frag3, b_frag1, acc31);
      du_mma_sync(acc32, a_frag3, b_frag2, acc32);
      du_mma_sync(acc33, a_frag3, b_frag3, acc33);
    }

    // Last stage only: prefetch the epilogue's per-row x_scale and
    // per-column weight_scale values into registers (uniform branch: k0 is
    // block-uniform, so the barrier below is reached by every thread; the
    // row < m guard keeps the x_scale reads in bounds for any m tail).  The
    // vmem latency overlaps the barrier below (s_barrier waits on
    // lgkmcnt/arrival, not vmcnt) and the dead a_frag/b_frag VGPR slots are
    // reused, so the epilogue is pure compute + coalesced stores.
    if (k0 + kStageK >= k) {
      const int r = lane & 15;
      const int c4 = lane >> 4;
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const int row = base_row + i * kTileM + r;
        xs_m[i] = (row < m) ? x_scale[row] : 0.0f;
      }
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        ws_m[j] = *reinterpret_cast<const float4*>(
            weight_scale + base_col + j * kTileN + 4 * c4);
      }
    }
    // Protect the LDS buffers from the next stage's cooperative overwrite.
    __syncthreads();
  }

  // Iteration 6 (epilogue round): coalesced direct-fragment epilogue.  Each
  // lane owns four contiguous bf16 columns of its fragment row and writes
  // ONE 8-byte store per fragment (16 stores per wave vs 64 scattered 2-byte
  // stores before); the scales come from the registers batched on the last
  // K stage, so no vmem loads are interleaved with the stores.  The int32
  // accumulation is untouched, and the multiply order / bf16 rounding are
  // identical to the old per-element store, so output bits are unchanged.
  store_prefill_fragment_coalesced_scaled(acc00, xs_m[0], ws_m[0], out,
                                          base_row, base_col, m, n, lane);
  store_prefill_fragment_coalesced_scaled(acc01, xs_m[0], ws_m[1], out,
                                          base_row, base_col + kTileN, m, n,
                                          lane);
  store_prefill_fragment_coalesced_scaled(acc02, xs_m[0], ws_m[2], out,
                                          base_row, base_col + 2 * kTileN, m,
                                          n, lane);
  store_prefill_fragment_coalesced_scaled(acc03, xs_m[0], ws_m[3], out,
                                          base_row, base_col + 3 * kTileN, m,
                                          n, lane);
  store_prefill_fragment_coalesced_scaled(acc10, xs_m[1], ws_m[0], out,
                                          base_row + kTileM, base_col, m, n,
                                          lane);
  store_prefill_fragment_coalesced_scaled(acc11, xs_m[1], ws_m[1], out,
                                          base_row + kTileM,
                                          base_col + kTileN, m, n, lane);
  store_prefill_fragment_coalesced_scaled(acc12, xs_m[1], ws_m[2], out,
                                          base_row + kTileM,
                                          base_col + 2 * kTileN, m, n, lane);
  store_prefill_fragment_coalesced_scaled(acc13, xs_m[1], ws_m[3], out,
                                          base_row + kTileM,
                                          base_col + 3 * kTileN, m, n, lane);
  store_prefill_fragment_coalesced_scaled(acc20, xs_m[2], ws_m[0], out,
                                          base_row + 2 * kTileM, base_col, m,
                                          n, lane);
  store_prefill_fragment_coalesced_scaled(acc21, xs_m[2], ws_m[1], out,
                                          base_row + 2 * kTileM,
                                          base_col + kTileN, m, n, lane);
  store_prefill_fragment_coalesced_scaled(acc22, xs_m[2], ws_m[2], out,
                                          base_row + 2 * kTileM,
                                          base_col + 2 * kTileN, m, n, lane);
  store_prefill_fragment_coalesced_scaled(acc23, xs_m[2], ws_m[3], out,
                                          base_row + 2 * kTileM,
                                          base_col + 3 * kTileN, m, n, lane);
  store_prefill_fragment_coalesced_scaled(acc30, xs_m[3], ws_m[0], out,
                                          base_row + 3 * kTileM, base_col, m,
                                          n, lane);
  store_prefill_fragment_coalesced_scaled(acc31, xs_m[3], ws_m[1], out,
                                          base_row + 3 * kTileM,
                                          base_col + kTileN, m, n, lane);
  store_prefill_fragment_coalesced_scaled(acc32, xs_m[3], ws_m[2], out,
                                          base_row + 3 * kTileM,
                                          base_col + 2 * kTileN, m, n, lane);
  store_prefill_fragment_coalesced_scaled(acc33, xs_m[3], ws_m[3], out,
                                          base_row + 3 * kTileM,
                                          base_col + 3 * kTileN, m, n, lane);
}

// ---------------------------------------------------------------------------
// Generic scalar fallback: one thread per output element.  Exact int32 dot
// over K, then fused x_scale * weight_scale, then bf16 store.  Handles every
// unmatched (m, n, k), including all small-M API cases (M=2, M=16), M tails
// and M < 128 with the same (K, N).  kNMajorPack == true decodes the
// iteration-5 swizzled 64-k-stage layout
// packed[((k0*8+kc)*n+col)*8+b] == raw[kk*n+col] (kk = k0*64+kc*8+b) for the
// exact (k, n) == (2048, 2048); otherwise the weight is the raw [K, N]
// row-major identity layout.
// ---------------------------------------------------------------------------
template <bool kNMajorPack>
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
  if constexpr (kNMajorPack) {
    // Swizzled 64-k-stage pack (iteration 5, exact (2048,2048) only):
    // packed[((k0*8 + kc)*n + col)*8 + b] == raw[kk*n + col] with
    // kk = k0*64 + kc*8 + b, so raw[kk][col] = packed[((k0*8+kc)*n+col)*8+b].
    for (int kk = 0; kk < k; ++kk) {
      const int k0 = kk >> 6;
      const int kc = (kk >> 3) & 7;
      const int b = kk & 7;
      const int8_t* p = weight +
                        (((static_cast<int64_t>(k0) * 8 + kc) * n + col) * 8) +
                        b;
      acc += static_cast<int32_t>(a_row[kk]) * static_cast<int32_t>(*p);
    }
  } else {
    // identity [K, N] row-major: column col is strided by n.
    const int8_t* b_col = weight + col;
    for (int kk = 0; kk < k; ++kk) {
      acc += static_cast<int32_t>(a_row[kk]) *
             static_cast<int32_t>(b_col[static_cast<int64_t>(kk) * n]);
    }
  }
  const float scaled =
      static_cast<float>(acc) * x_scale[row] * weight_scale[col];
  out[linear] = __float2bfloat16(scaled);
}

// ---------------------------------------------------------------------------
// Identity device-to-device weight packing (outside the timed region and
// outside Graph capture) for every (k, n) except the exact (2048, 2048)
// q_b_proj pair (which uses the iteration-5 swizzled pack).  The packed
// buffer keeps the same byte count K*N and the same allocated address, so
// the layout is graph-stable.
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

// Iteration 5 (packing round): swizzled 64-k-stage-major permutation for the
// exact (k, n) == (2048, 2048) q_b_proj weight:
//   packed[((k0*8 + kc)*n + col)*8 + b] = raw[kk*n + col],
//   kk = k0*64 + kc*8 + b  (k0 = 64-k stage, kc = 8-byte sub-chunk, b = byte).
// Byte-wise (one thread per byte) so the permutation is trivially correct;
// runs once outside the timed region and outside Graph capture, keeping the
// same byte count K*N and the same graph-stable buffer address.  The layout
// lets the 128x128 kernel stage each B tile as 16-byte plane chunks (one
// aligned int4 global read per chunk, perfectly coalesced) into the plane
// LDS tile [kc][n][8] whose fragment loads are lane-linear with zero bank
// conflicts.
__global__ __launch_bounds__(256) void w8a8_pack_swizzle_kernel(
    const int8_t* __restrict__ raw,
    int8_t* __restrict__ packed,
    int k,
    int n) {
  const int64_t linear =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total = static_cast<int64_t>(k) * n;
  if (linear < total) {
    const int kk = static_cast<int>(linear / n);
    const int col = static_cast<int>(linear - static_cast<int64_t>(kk) * n);
    const int k0 = kk >> 6;
    const int kc = (kk >> 3) & 7;
    const int b = kk & 7;
    packed[(((static_cast<int64_t>(k0) * 8 + kc) * n + col) * 8) + b] =
        raw[linear];
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

  // Explicit dispatch.  Large-M shapes with exact tiled geometry (both
  // assigned shapes: M=4096, N in {2048, 3584}, K in {2048, 512}) take the
  // native INT8 DUMMA tiled path; every other (m, n, k) - including small-M
  // API cases (M=2, M=16) and M < 128 - takes the scalar fallback.  The
  // exact (k, n) == (2048, 2048) q_b_proj pair takes the iteration-4 128x128
  // tile with the iteration-5 swizzled-pack B layout; every other (k, n) -
  // including kv_b_proj (512, 3584) - uses the byte-identical 128x64
  // kNMajorB=false arm (kNMajorPack == true for the scalar fallback's
  // (2048,2048) decode).
  if (m >= kBlockM && m % kBlockM == 0 && n % kBlockN == 0 &&
      k % kStageK == 0) {
    const dim3 grid(static_cast<unsigned>(n / kBlockN),
                    static_cast<unsigned>(m / kBlockM));
    const dim3 block(kBlockThreads);
    if (k == kPackNMajorK && n == kPackNMajorN) {
      // Exact q_b_proj (2048, 2048): iteration-4 tile-aspect arm, 128x128
      // tile with the iteration-5 swizzled-pack plane B layout; grid
      // (N/128, M/128) = 512 blocks for M=4096.  n == 2048 is divisible by
      // 128 by the exact guard; the outer arm already guarantees
      // m % 128 == 0, k % 64 == 0.
      const dim3 grid128(static_cast<unsigned>(n / 128),
                         static_cast<unsigned>(m / 128));
      hipLaunchKernelGGL(
          HIP_KERNEL_NAME(w8a8_dumma_prefill_128x128_kernel),
          grid128, block, 0, stream,
          a, b, x_scale, weight_scale, out_bf16, m, n, k);
    } else {
      hipLaunchKernelGGL(
          HIP_KERNEL_NAME(w8a8_dumma_prefill_128x64_kernel<false>),
          grid, block, 0, stream,
          a, b, x_scale, weight_scale, out_bf16, m, n, k);
    }
  } else {
    constexpr int kBlock = 256;
    const int64_t total = static_cast<int64_t>(m) * n;
    const dim3 grid(static_cast<unsigned>((total + kBlock - 1) / kBlock));
    const dim3 block(kBlock);
    if (k == kPackNMajorK && n == kPackNMajorN) {
      hipLaunchKernelGGL(
          HIP_KERNEL_NAME(w8a8_gemm_scalar_fallback_kernel<true>),
          grid, block, 0, stream,
          a, b, x_scale, weight_scale, out_bf16, m, n, k);
    } else {
      hipLaunchKernelGGL(
          HIP_KERNEL_NAME(w8a8_gemm_scalar_fallback_kernel<false>),
          grid, block, 0, stream,
          a, b, x_scale, weight_scale, out_bf16, m, n, k);
    }
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
  const dim3 block(kBlock);
  // Bootstrap pack: iteration-5 swizzled 64-k-stage permutation for the exact
  // (2048, 2048) q_b_proj weight; identity device-to-device copy for every
  // other (k, n).
  const int64_t weight_count = static_cast<int64_t>(k) * n;
  const dim3 weight_grid(
      static_cast<unsigned>((weight_count + kBlock - 1) / kBlock));
  if (k == kPackNMajorK && n == kPackNMajorN) {
    hipLaunchKernelGGL(w8a8_pack_swizzle_kernel,
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
