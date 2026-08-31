// @@variant shape=glm_tp8_kv_b_proj_m4096 commit=bb4aec7f2d04a07b6c3c5cf4c19e1b9fc97fc8a3 added=2026-08-31
//   median_us=137.4 p90_us=137.9 speedup=50.51 baseline_us=6940
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
//
// Iteration 1 (kv_b_proj tile-baseline round): the exact (k, n) == (512,
// 3584) kv_b_proj arm is still the iteration-2 bootstrap pipeline (library
// fragment loaders, identity [K, N] B, per-element scattered epilogue) and
// runs at 65 TOPS vs 156 TOPS on the fully-modernized q_b_proj arm (same
// GPU, same DTK, measured 220.14 us on K=2048/N=2048).  This round ports the
// proven pipeline (swizzled plane-pack B, direct 8-byte A/B fragment loads,
// register-batched coalesced epilogue, fully unrolled kk) to kv_b_proj as
// one templated 2-D macro-tile family (w8a8_dumma_prefill_tile_kernel) over
// the three mandated block tiles plus the iteration-4 aspect flip, all
// compiled, active selected by kActiveKvTile (0 = 64x64, 1 = 64x128,
// 2 = 128x64, 3 = 64x128 with 32x32 quadrants):
//   config  tile   waves  threads  LDS B      grid (blocks)   est. VGPR
//     0     64x64    1      64      9,216      (56, 64) = 3584   ~128-140
//     1     64x128   2     128     13,312      (28, 64) = 1792   ~128-140
//     2     128x64   4     256     14,336      (56, 32) = 1792   ~88-100
//     3     64x128   8     512     13,312      (28, 64) = 1792   ~55-60
// Per-wave LDS fragment-read bytes per MMAC: 256 B for configs 0/1 (4 A + 4
// B fragments feed 16 MMACs per kk) vs 384 B for config 2 (4 A + 2 B feed 8
// MMACs) and vs 384 B for the old 128x64 arm; B reads are zero-conflict in
// the plane layout for every plane stride, and the A stride-80 2-way
// aliasing matches the accepted q_b_proj arm.  Occupancy: 64x64 -> 7
// blocks/CU = 7 waves/CU (LDS- and VGPR-limited), 64x128 -> 4 blocks/CU = 8
// waves/CU (LDS-limited; VGPR <= 128 keeps 4 blocks), 128x64 -> 2 blocks/CU
// = 8 waves/CU (VGPR-limited) - vs 2 blocks/CU = 8 waves/CU today but with
// ~2.9x fewer LDS instructions (direct 8-byte reads vs library byte-gathers)
// and 4x fewer store instructions (coalesced epilogue, 229,376 -> 57,344
// vmem_write).  Active config for the measurement: 64x128 (config 1):
// deepest per-kk MMAC burst of the three alongside the lowest per-MMAC LDS
// read volume, half the A-side global re-read of 128x64, and 8 waves/CU at
// 4 blocks/CU.
//   * Iteration-1 measurement: config 1 landed at arch_vgpr 136 (code object
//     135) -> 3 blocks/CU = 6 waves/CU (1.5/SIMD), median 209.4 us = 71.8
//     TOPS.  The kernel is latency-bound (per-stage wall ~2,454 cycles vs
//     ~250 cycles of pure issue), so 6 waves/CU leaves the barrier + LDS +
//     global-latency chain exposed on the SIMDs that hold only one wave.
//   * Iteration-2 (operand-reuse) round selects config 2 (128x64): same
//     cooperative A+B staging machinery and swizzled-pack B, same
//     bit-identical int32 order; its +50% A-side LDS fragment-read bytes per
//     MMAC (384 vs 256 B; A rows are re-read by the two n-half waves in both
//     configs, but config 2's narrower 64x32 quadrant halves the MMACs per
//     A-frag load) is absorbed by the LDS pipe (~10% of per-stage cycles)
//     while the compiled 78 VGPR / 14,336 B gives 3 blocks x 4 waves = 12
//     waves/CU (3/SIMD) - a 2x occupancy increase over the measured config-1
//     residency.
//  The swizzled pack is extended to (512, 3584) in
// launch_pack_w8a8_weight (same byte count, outside the timed region) and
// the scalar fallback decodes it for both swizzled pairs, so the paired
// M=2/(3584,512) API case stays exact.  All other arms (q_b_proj 128x128,
// generic 128x64<false>, identity packs) are byte-identical.
//
// Iteration 5 (kv_b_proj packing round): the exact (k, n) == (512, 3584)
// weight is re-packed once (outside timing/Graph, same byte count, same
// graph-stable buffer) from the k-stage-major swizzle to a TILE-CONTIGUOUS
// stage-major layout
//   packed[(((k0*(n/128) + nt)*8 + kc)*128 + row)*8 + b] == raw[kk*n + col]
//   kk = k0*64 + kc*8 + b, nt = col/128, row = col%128,
// so every (64-k stage, 128-n tile) B tile is ONE contiguous 8192-B global
// region.  The tile-family kernel (w8a8_dumma_prefill_tile_kernel, all four
// configs) stages it as 8 consecutive 1024-B wavefront streams instead of 8
// streams 28,672 B apart in the k-stage-major layout; the LDS plane tile
// [kc][n][8], the zero-conflict lane-linear 8-byte B fragment reads
// (load_fragment8_plane_stride), the 16-B staging writes and the
// k0-outer/kk-inner int32 accumulation are byte-identical, so mismatch 0 /
// max_abs_error 0 is expected without tolerance debate.  q_b_proj (2048,
// 2048) keeps the old k-stage-major swizzle byte-identical (its 128x128
// kernel, pack kernel and fallback decode are untouched); the scalar
// fallback decodes the new tile-contiguous layout for (512,3584) and the
// old swizzle for (2048,2048).
//
// Iteration 6 (kv_b_proj epilogue round): the mandate (per-row x_scale,
// per-column weight_scale, bf16 conversion and the final coalesced store
// fused into the compute kernel; no workspace/combine pass) has been in
// place since iteration 1 - the launcher is a single kernel with
// (void)workspace and the epilogue is register-batched (xs scalar + ws
// float4 prefetched on the last K stage) with one 8-byte bf16 store per lane
// per fragment (57,344 vmem_write).  The remaining epilogue inefficiency of
// the exact (512, 3584) tile family is the bf16 CONVERSION lowering: the
// exact code object shows hip's __float2bfloat16 emits, per element, an
// exec-masked inf/NaN fixup (v_and exp + v_cmp_ne + s_and_saveexec + s_xor +
// v_mov 0 + v_or 0x10000 + v_cmp_eq_u32_sdwa + v_cndmask + s_or_b64) around
// the 3-instruction RNE rounding (v_bfe + v_add3 + shift-in-pack) - ~11
// instructions and three scalar exec manipulations that serialize the
// block-tail epilogue.  This round replaces that path, ONLY in the
// w8a8_dumma_prefill_tile_kernel epilogue, with a plain RNE conversion
// (bf16_rne_u16: u += 0x7fff + ((u >> 16) & 1); u >> 16) behind a new
// kRneOnly template flag on store_prefill_fragment_coalesced_scaled
// (default false, so the q_b_proj 128x128 / 128x64 arms and the scalar
// fallback stay byte-identical).  The RNE formula is bit-identical to hip's
// own non-masked branch for every finite float - the only values this GEMM
// can produce (int32 accumulators x finite scales) - so mismatch 0 /
// max_abs_error 0 is expected on the unchanged pre-seeded reference data.
// PMC prediction: valu_instructions 7.31M -> ~5.9-6.1M (-15..-20%),
// lds_instructions 1,089,536 unchanged, lds_bank_conflicts 2,293,760
// unchanged, vmem_read 229,376 / vmem_write 57,344 unchanged, arch_vgpr <=
// 64 / 0 scratch / 13,312 B LDS / 2 blocks/CU = 16 waves/CU residency
// unchanged; the exact code object must show no s_and_saveexec/s_xor/
// s_andn2_saveexec around the conversion (v_bfe + v_add3 straight into the
// 8-byte pack).

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

// Exact (k, n) pairs whose weight buffer is packed with the iteration-5
// swizzled 64-k-stage layout (packed[((k0*8+kc)*n+col)*8+b] == raw[kk*n+col]):
// the q_b_proj pair (2048, 2048) and - since this kv_b_proj tile round - the
// kv_b_proj pair (512, 3584).  Every other (k, n) keeps the identity layout.
constexpr int kPackSwizzleK1 = 2048;
constexpr int kPackSwizzleN1 = 2048;
constexpr int kPackSwizzleK2 = 512;
constexpr int kPackSwizzleN2 = 3584;

inline bool is_swizzled_pack(int k, int n) {
  return (k == kPackSwizzleK1 && n == kPackSwizzleN1) ||
         (k == kPackSwizzleK2 && n == kPackSwizzleN2);
}

// Iteration 5 (kv_b_proj packing round): the (512, 3584) pair switched from
// the k-stage-major swizzle to the tile-contiguous layout
// packed[(((k0*(n/128) + nt)*8 + kc)*128 + row)*8 + b]; (2048, 2048) keeps
// the k-stage-major swizzle byte-identical.
inline bool is_tile_major_pack(int k, int n) {
  return k == kPackSwizzleK2 && n == kPackSwizzleN2;
}

// Active 2-D macro-tile for the exact glm_tp8_kv_b_proj_m4096 shape among the
// three mandated candidates plus the iteration-4 aspect flip (0 = 64x64,
// 1 = 64x128, 2 = 128x64, 3 = 64x128 with 32x32 quadrants).  All four
// instantiations are compiled and dispatched behind the exact (k, n) guard;
// flipping this constant re-routes the kv_b_proj arm without source surgery.
constexpr int kActiveKvTile = 3;  // 3 = 64x128 with 32x32 quadrants
                                  // (iteration-4 tile-shape round): the
                                  // down_proj-validated 16-waves/CU recipe
                                  // (512 threads, 8 wavefronts/block, 4
                                  // accs/wave, arch_vgpr <= 64 -> 2 blocks/CU
                                  // = 16 waves/CU = 4/SIMD) applied as the
                                  // M-tile 128 -> 64 / N-tile 64 -> 128 aspect
                                  // flip; config 2 (128x64, 3 blocks x 4 waves
                                  // = 12 waves/CU, accepted best 169.257 us)
                                  // stays compiled and is the retention
                                  // fallback if config 3 regresses (arch_vgpr
                                  // > 64 -> 1 block/CU = 8 waves -> reject).
constexpr int kKvTileM = 64;
constexpr int kKvTileN = 128;
constexpr int kKvQuadN = 64;

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
// Iteration 6 (kv_b_proj epilogue round): plain round-to-nearest-even bf16
// conversion without the inf/NaN fixup path.  hip's __float2bfloat16 lowers
// (verified on this exact DTK 26.04 / gfx928 toolchain with a micro compile)
// to an exec-masked sequence per element: v_and (exp mask) + v_cmp_ne +
// s_and_saveexec + s_xor + v_bfe + v_add3 + v_mov 0 + v_or 0x10000 +
// v_cmp_eq_u32_sdwa + v_cndmask + s_or_b64 (~11 instructions including three
// scalar exec manipulations that serialize the wavefront).  The masking only
// changes the result for inf/NaN inputs; for every finite float - the only
// values this GEMM can produce from int32 accumulators and finite scales -
// the RNE rounding is exactly u += 0x7fff + ((u >> 16) & 1); return u >> 16,
// bit-identical to hip's own non-masked branch (same v_bfe + v_add3 +
// shift).  The kernel epilogue is the only place that converts all M*N =
// 14.68M elements; dropping the mask removes ~6 VALU + 5 SALU per element
// (~1.38M VALU + 1.15M SALU per launch, ~19% of the measured 7.31M VALU
// stream) and - the falsifiable axis - removes the
// s_and_saveexec / s_xor / s_andn2_saveexec / s_or_b64 pairs that serialize
// the block-tail epilogue behind scalar exec manipulation.
// ---------------------------------------------------------------------------
__device__ __forceinline__ unsigned short bf16_rne_u16(float x) {
  union {
    float f;
    uint32_t u;
  } c;
  c.f = x;
  const uint32_t u = c.u + 0x7fff + ((c.u >> 16) & 1);
  return static_cast<unsigned short>(u >> 16);
}

template <bool kRneOnly = false, typename AccFragment>
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
  uint64_t packed;
  if constexpr (kRneOnly) {
    // Iteration 6 (kv_b_proj epilogue round): plain RNE (see bf16_rne_u16) -
    // bit-identical to __float2bfloat16 for every finite input, without the
    // exec-masked inf/NaN fixup path.
    packed = static_cast<uint64_t>(bf16_rne_u16(v0)) |
             (static_cast<uint64_t>(bf16_rne_u16(v1)) << 16) |
             (static_cast<uint64_t>(bf16_rne_u16(v2)) << 32) |
             (static_cast<uint64_t>(bf16_rne_u16(v3)) << 48);
  } else {
    packed =
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
  }
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
// Plane-layout 8-byte fragment load with a parameterized plane stride
// (plane_stride = kBlockN * 8 bytes).  Identical to load_fragment8_plane for
// kBlockN == 128 (plane stride 1024 B); for narrower tiles the plane is
// smaller but the bank pattern is unchanged: lane l reads the 8 bytes at
// (lane >> 4) * plane_stride + (lane & 15) * 8, so each 16-lane group of a
// 128-B phase covers all 32 banks exactly once (zero bank conflicts) for any
// plane stride.
// ---------------------------------------------------------------------------
__device__ __forceinline__ void load_fragment8_plane_stride(
    DUFragmentBase<signed char, 8>& frag,
    const signed char* __restrict__ base,
    int plane_stride,
    int lane) {
  const int off = ((lane >> 4) * plane_stride) + ((lane & 15) << 3);
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
// Iteration 1 (kv_b_proj tile-baseline round): 2-D macro-tile DUMMA family
// for the exact glm_tp8_kv_b_proj_m4096 (k, n) == (512, 3584).  The prior
// kv_b_proj arm (w8a8_dumma_prefill_128x64_kernel<false>) is the iteration-2
// bootstrap pipeline: library du_load_matrix_sync fragment loaders (per-byte
// ds_read_u8 byte-gather + mask/OR reassembly VALU, ~2.47M LDS instructions
// and 6.88M bank conflicts measured), raw [K, N] B identity layout, and a
// per-element 2-byte scattered epilogue (229,376 vmem_write instructions =
// 25% store sector efficiency) - i.e. 65 TOPS vs 156 TOPS on the sibling
// q_b_proj arm that already carries the full modern pipeline (plane-swizzled
// B, direct 8-byte fragment loads, coalesced epilogue).  This round ports
// that proven pipeline to kv_b_proj as one templated family over the three
// mandated macro-tiles:
//   config 0: 64x64   (1 wavefront, 64 threads; wave owns the whole tile)
//   config 1: 64x128  (2 wavefronts, 128 threads; each wave owns 64x64)
//   config 2: 128x64  (4 wavefronts, 256 threads; each wave owns 64x32)
//   config 3: 64x128  (8 wavefronts, 512 threads; each wave owns 32x32;
//                      iteration-4 tile-shape round)
// (kActiveKvTile selects the active config; all four are compiled.)
//
// Pipeline (byte-identical to the validated q_b_proj arm):
//   * B is staged from the iteration-5 swizzled 64-k-stage pack, now also
//     produced for (512, 3584) by launch_pack_w8a8_weight, into the plane
//     LDS tile b_tile[kc][n][8] (plane stride kBlockN*8 B, n stride 8 B);
//     every B fragment read is one 8-byte lane-linear ds_read (zero bank
//     conflicts) via load_fragment8_plane_stride.
//   * A is staged [kBlockM, 80] (16 B padded row stride) and each A fragment
//     is one direct 8-byte LDS read (load_fragment8), no VALU reassembly.
//   * int32 accumulation stays resident across the full K loop (k0-outer over
//     64-K stages, kk-inner over two m16n16k32 DUMMA steps), fully unrolled
//     per stage like the q_b_proj winner (deep 16-MMAC bursts per kk).
//   * The fused dot * x_scale * weight_scale -> bf16 epilogue is the
//     register-batched coalesced variant (one 8-byte store per lane per
//     fragment; scales prefetched on the last K stage).
//   * Single-buffered LDS, two __syncthreads per stage, grid dim3(N/BN,
//     M/BM) = 1792 blocks (64x128 / 128x64) or 3584 blocks (64x64) - no
//     split-K (the MxN grid already over-subscribes the 120 CUs).
//
// Dispatch guarantees m % kBlockM == 0, n % kBlockN == 0, k % 64 == 0 for the
// exact (512, 3584) pair (m == 4096 in the assigned shape), so every global
// load/store is in bounds and 16-byte aligned; the row < m guards stay for
// generic safety (wavefront-uniform, so the coalesced shuffles never mix
// active and inactive lanes).
//
// Iteration 16 (kv_b_proj load-placement round): the exact (k, n) == (512,
// 3584) active config-3 kernel (w8a8_dumma_prefill_tile_kernel<64,128,32,32,
// true>, median 139.567 us = 107.7 TOPS, 49.72x vs fixed Triton 6939.939 us;
// code object 61 VGPR / 28 SGPR / 13,312 B LDS / 0 scratch / 2 blocks/CU =
// 16 waves/CU; grid 1792, workgroup 512) is LDS-latency/issue-bound (lineage
// rule 5) and its stage critical path still exposes, per 64-K stage, one
// grouped A+B staging L2 round trip at the stage top (iteration-7 grouping
// gained +2.24% by merging the two serialized round trips of iteration 6,
// so the round trip sits on the barrier-to-barrier wall).  The double-
// buffered one-barrier pipeline is arithmetic-infeasible at 2 blocks/CU
// (iters 9/10/12 compiled to 72/72/68 VGPR -> 1 block/CU = 8 waves/CU, the
// measured regression regime; every full double-buffer costs >= 7 VGPR on
// this 64-VGPR-constrained 512-thread config) and the 128-K stage-depth
// rework was attempted in the killed iteration-15 candidate (NOT replayed).
// This round keeps the SINGLE-buffered two-barrier stage byte-identical and
// moves ONLY the global-load ISSUE point: the grouped A+B staging loads for
// stage s+1 are issued at the END of stage s (after the MMAC burst, before
// the protective trailing __syncthreads) instead of at the top of stage s+1.
// Global loads never touch LDS, so the early issue is hazard-free; the
// vmcnt wait is inserted by the compiler before the first LDS write of stage
// s+1 that consumes the chunk, and the in-flight round trip overlaps the
// trailing barrier + loop back-edge of stage s (partial hiding of each of
// the 8 exposed staging round trips).  Crucially NO register payload is
// carried across the burst: a_chunk/b_chunk are dead during the burst and
// are re-issued after it, so peak VGPR stays <= 64 -> 2 blocks/CU = 16
// waves/CU (this is NOT the iteration-9/10/12 register-carrying placement,
// and NOT the iteration-12 double-buffer: LDS stays 13,312 B/block, 16
// s_barriers/block).  Bit-identity: identical global bytes -> identical LDS
// writes at identical (row, kk) slots -> identical operand bytes in the
// same k0-outer/kk-inner int32 accumulation -> identical register-batched
// RNE epilogue (only the load issue point moves), so mismatch 0 /
// max_abs_error 0 is expected without tolerance debate.  Scoping: a new
// defaulted template parameter kPrefetchNext = false keeps configs 0-2 on
// the exact iteration-7 issue-at-stage-top placement (compiled identically
// via the if constexpr retention branch); the q_b_proj 128x128 / 128x64
// arms, the generic scalar fallback and the pack kernels are untouched.
// Iteration 17 (kv_b_proj consolidation round): the two independently
// measured single-buffered wins are consolidated into ONE active config-3
// instantiation (w8a8_dumma_prefill_tile_kernel<64,128,32,32,true,24>):
//   (a) iteration 16's next-stage load-issue prefetch (kPrefetchNext = true,
//       the current shadow candidate, median 138.229 us / p90 138.781,
//       +0.968% vs the accepted best): the grouped A+B staging loads for
//       stage s+1 are issued after the MMAC burst of stage s, so each of the
//       8 staging L2 round trips overlaps the trailing barrier + loop
//       back-edge instead of stalling the stage-top LDS writes;
//   (b) iteration 13's A-side LDS bank skew (kALdsPad = 24 -> 88-byte A row
//       stride, standalone median 138.783 us / p90 139.185, +0.565% vs the
//       accepted best): stride-80 rows are 8-periodic in the 128-B bank
//       phase, so the 16 fragment rows (lane & 15) collapse onto 8 phases
//       (r and r+8 collide, 2-way); 88 = 64+24 is 8-byte aligned and 88r mod
//       128 is 16-periodic, so the 16 fragment rows hit 16 DISTINCT phases
//       -> zero-conflict A fragment reads (lineage rule 3, the qkv stride-68
//       / down_proj stride-72 pattern).  Because stride % 16 == 8, the
//       16-byte A staging chunk is written as two int64 halves
//       (ds_write2_b64, lineage rule 3), with the SAME 16 bytes at the SAME
//       logical (row, kk) slots.
// The two mechanisms are orthogonal: (a) removes the staging L2 round trip
// from the stage-top critical path, (b) removes the 2,293,760-cycle LDS
// bank-conflict term from the LDS pipe (the conflict model: A fragment reads
// 32 x 2-way 8-B reads/stage -> 8 vs 4 cycles, + A staging writes 4 x 2-way
// 16-B writes/stage -> 16 vs 8 cycles, x 8 stages = 1,280 extra cycles/block
// x 1792 blocks).  Both keep the single-buffered two-barrier stage (16
// s_barriers/block), 2 blocks/CU = 16 waves/CU (vgpr <= 64: the write split
// reuses the same 4-VGPR a_chunk and NO payload crosses the burst), and LDS
// 13,824 B/block (2 blocks = 27,648 B <= 64 KiB).  Bit-identity: identical
// global bytes -> identical logical A tile content at every (row, kk) (only
// inter-row padding changes) -> identical 8-byte m16n16k32 operand bytes per
// lane -> identical k0-outer/kk-inner int32 accumulation and register-batched
// RNE epilogue -> mismatch 0 / max_abs_error 0 expected without tolerance
// debate.  Scoping: kALdsPad defaults to kBPad = 16, so configs 0-2 keep the
// exact iteration-7 stride-80 int4 write path (token-identical); the
// q_b_proj 128x128 / 128x64 arms, the generic scalar fallback and the pack
// kernels are untouched; the exact-shape guard (k == 512 && n == 3584 &&
// m >= 64 && m % 64 == 0) is preserved.
// ---------------------------------------------------------------------------
template <int kBlockM, int kBlockN, int kQuadN, int kQuadM = 64,
          bool kPrefetchNext = false, int kALdsPad = kBPad>
__global__ __launch_bounds__(
    ((kBlockM / kQuadM) * (kBlockN / kQuadN)) * kWaveSize)
void w8a8_dumma_prefill_tile_kernel(
    const int8_t* __restrict__ x_q,
    const int8_t* __restrict__ weight,
    const float* __restrict__ x_scale,
    const float* __restrict__ weight_scale,
    bf16_t* __restrict__ out,
    int m,
    int n,
    int k) {
  // kQuadM = rows owned by each wavefront (64 in configs 0-2; 32 in the
  // iteration-4 config 3 so a 64x128 block hosts 8 wavefronts).
  constexpr int kThreads =
      (kBlockM / kQuadM) * (kBlockN / kQuadN) * kWaveSize;
  // Iteration 17 (kv_b_proj consolidation round): A row stride is
  // kStageK + kALdsPad.  The default pad 16 keeps stride 80 (16-byte
  // aligned, 2-way bank aliasing) for the retention configs 0-2; the active
  // config-3 instantiation passes kALdsPad = 24 -> stride 88 (8-byte
  // aligned, 16 distinct 128-B phases for the 16 fragment rows -> zero
  // conflict A fragment reads; lineage rule: stride % 16 == 8 stages with
  // two int64 halves, see the A staging write below).
  constexpr int kTileStride = kStageK + kALdsPad;  // 64 -> 80/88 B A stride
  constexpr int kBPlaneStride = kBlockN * 8;    // n rows x 8 B per plane
  constexpr int kATileBytes = kBlockM * kTileStride;
  constexpr int kBTileBytes = 8 * kBlockN * 8;  // 8 k-planes of n rows x 8 B
  constexpr int kASlots = (kBlockM * kStageK) / 16;  // int4 A chunks
  constexpr int kBSlots = kBTileBytes / 16;          // int4 B chunks
  constexpr int kALoads = (kASlots + kThreads - 1) / kThreads;
  constexpr int kBLoads = (kBSlots + kThreads - 1) / kThreads;
  constexpr int kAccM = kQuadM / kTileM;   // A fragments per wave (4 / 2)
  constexpr int kAccN = kQuadN / kTileN;   // B fragments per wave
  constexpr int kWavesN = kBlockN / kQuadN;

  const int tid = static_cast<int>(threadIdx.x);
  const int wave = tid / kWaveSize;
  const int lane = tid % kWaveSize;
  const int wave_row = wave / kWavesN;
  const int wave_col = wave % kWavesN;
  const int m0 = static_cast<int>(blockIdx.y) * kBlockM;
  const int n0 = static_cast<int>(blockIdx.x) * kBlockN;
  const int local_row = wave_row * kQuadM;
  const int local_col = wave_col * kQuadN;
  const int base_row = m0 + local_row;
  const int base_col = n0 + local_col;

  // Single-buffered stage: A[kBlockM, 80] + B plane tile [8][kBlockN][8].
  __shared__ __align__(16) int8_t a_tile[kATileBytes];
  __shared__ __align__(16) int8_t b_tile[kBTileBytes];

  DUFragment<matrix_a, kTileM, kTileN, kTileK, signed char, row_major>
      a_frag[kAccM];
  DUFragment<matrix_b, kTileM, kTileN, kTileK, signed char, col_major>
      b_frag[kAccN];
  DUFragment<accumulator, kTileM, kTileN, kTileK, int> acc[kAccM][kAccN];
#pragma unroll
  for (int i = 0; i < kAccM; ++i) {
#pragma unroll
    for (int j = 0; j < kAccN; ++j) {
      du_fill_fragment(acc[i][j], 0);
    }
  }

  // Register-batched epilogue scales (prefetched on the last K stage).
  float xs_m[kAccM];
  float4 ws_m[kAccN];

  if constexpr (kPrefetchNext) {
    // -----------------------------------------------------------------------
    // Iteration 16 (kv_b_proj load-placement round) - ACTIVE config-3 path:
    // single-buffered two-barrier stage with the grouped A+B staging loads
    // for stage s+1 issued at the END of stage s (after the MMAC burst,
    // before the protective trailing barrier).  a_chunk/b_chunk live across
    // the trailing barrier + the loop back-edge only (never across a burst),
    // so peak VGPR stays <= 64 -> 2 blocks/CU = 16 waves/CU unchanged; LDS
    // 13,312 B/block and 16 s_barriers/block unchanged.  The vmcnt wait for
    // the in-flight loads is inserted by the compiler before the first LDS
    // write of the next stage, i.e. each staging L2 round trip overlaps the
    // trailing barrier + back-edge instead of stalling the LDS writes.
    // -----------------------------------------------------------------------
    int4 a_chunk[kALoads];
    bool a_ok[kALoads];
    int4 b_chunk[kBLoads];
    {
      // Prologue: issue the stage-0 A+B loads (the same grouped pair and the
      // same addresses as the iteration-7 stage-top placement) so the first
      // iteration starts directly with the LDS writes.
      const int k0 = 0;
      const int kstage = 0;  // 64-k stage index inside the swizzled pack
#pragma unroll
      for (int i = 0; i < kALoads; ++i) {
        const int slot = tid + i * kThreads;
        a_ok[i] = slot < kASlots;
        if (a_ok[i]) {
          const int row = slot >> 2;        // 0..kBlockM-1
          const int col = (slot & 3) * 16;  // 0/16/32/48
          a_chunk[i] = *reinterpret_cast<const int4*>(
              x_q + static_cast<int64_t>(m0 + row) * k + k0 + col);
        }
      }
#pragma unroll
      for (int i = 0; i < kBLoads; ++i) {
        const int slot = tid + i * kThreads;
        if (slot < kBSlots) {
          const int b_kc = slot / (kBlockN / 2);  // plane 0..7
          const int b_j = slot - b_kc * (kBlockN / 2);  // n-pair 0..BN/2-1
          const int b_row0 = (n0 + 2 * b_j) & 127;
          const int b_nt = (n0 + 2 * b_j) >> 7;
          b_chunk[i] = *reinterpret_cast<const int4*>(
              weight + (((static_cast<int64_t>(kstage) * (n >> 7) + b_nt) * 8 +
                         b_kc) * 128 + b_row0) * 8);
        }
      }
    }
    for (int k0 = 0; k0 < k; k0 += kStageK) {
      const int kstage = k0 >> 6;  // 64-k stage index inside the swizzled pack
#pragma unroll
      for (int i = 0; i < kALoads; ++i) {
        if (a_ok[i]) {
          const int slot = tid + i * kThreads;
          const int row = slot >> 2;
          const int col = (slot & 3) * 16;
          if constexpr (kTileStride % 16 == 8) {
            // Iteration 17 (kv_b_proj consolidation round): 88-byte A rows
            // are only 8-byte aligned, so stage the 16-byte chunk as two
            // int64 halves (ds_write2_b64, not ds_write_b128 - lineage rule
            // 3).  Same 16 staging bytes at the same logical (row, kk)
            // slots, so bit-identity holds.
            int8_t* dst = a_tile + row * kTileStride + col;
            const int64_t* src =
                reinterpret_cast<const int64_t*>(&a_chunk[i]);
            *reinterpret_cast<int64_t*>(dst) = src[0];
            *reinterpret_cast<int64_t*>(dst + 8) = src[1];
          } else {
            *reinterpret_cast<int4*>(a_tile + row * kTileStride + col) =
                a_chunk[i];
          }
        }
      }
#pragma unroll
      for (int i = 0; i < kBLoads; ++i) {
        const int slot = tid + i * kThreads;
        if (slot < kBSlots) {
          const int b_kc = slot / (kBlockN / 2);
          const int b_j = slot - b_kc * (kBlockN / 2);
          *reinterpret_cast<int4*>(b_tile + b_kc * kBPlaneStride + b_j * 16) =
              b_chunk[i];
        }
      }
      __syncthreads();

      // Each wave consumes its 64 x kQuadN quadrant: kAccM * kAccN
      // m16n16k32 MMACs per kk.  Accumulation order: k0-outer over 64-K
      // stages, kk-inner (kk = 0 then 32), matching the reference int32
      // accumulation.  All fragment loads are issued before the MMAC burst
      // (load-all-then-MMAC-all, fully unrolled like the q_b_proj winner).
#pragma unroll
      for (int kk = 0; kk < kStageK; kk += kTileK) {
#pragma unroll
        for (int i = 0; i < kAccM; ++i) {
          load_fragment8(a_frag[i],
                         a_tile + (local_row + i * kTileM) * kTileStride + kk,
                         kTileStride, lane);
        }
#pragma unroll
        for (int j = 0; j < kAccN; ++j) {
          load_fragment8_plane_stride(
              b_frag[j],
              b_tile + (kk >> 3) * kBPlaneStride +
                  (local_col + j * kTileN) * 8,
              kBPlaneStride, lane);
        }
#pragma unroll
        for (int j = 0; j < kAccN; ++j) {
#pragma unroll
          for (int i = 0; i < kAccM; ++i) {
            du_mma_sync(acc[i][j], a_frag[i], b_frag[j], acc[i][j]);
          }
        }
      }

      // Last stage only: prefetch the epilogue's per-row x_scale and
      // per-column weight_scale values into registers (uniform branch: k0 is
      // block-uniform, so the barrier below is reached by every thread; the
      // row < m guard keeps the x_scale reads in bounds for any m tail).
      if (k0 + kStageK >= k) {
        const int r = lane & 15;
        const int c4 = lane >> 4;
#pragma unroll
        for (int i = 0; i < kAccM; ++i) {
          const int row = base_row + i * kTileM + r;
          xs_m[i] = (row < m) ? x_scale[row] : 0.0f;
        }
#pragma unroll
        for (int j = 0; j < kAccN; ++j) {
          ws_m[j] = *reinterpret_cast<const float4*>(
              weight_scale + base_col + j * kTileN + 4 * c4);
        }
      }
      // Iteration 16 (kv_b_proj load-placement round): issue the grouped A+B
      // staging loads for the NEXT stage here - after the burst and the
      // epilogue scale prefetch, before the protective trailing barrier - so
      // their L2 round trip overlaps the barrier + loop back-edge instead of
      // stalling the next stage's LDS writes.  Uniform branch (k0 is
      // block-uniform); only GLOBAL memory is touched, so the barrier
      // semantics are unchanged.  On the last stage no loads are issued and
      // the placement degenerates to the accepted schedule.
      if (k0 + kStageK < k) {
        const int k0n = k0 + kStageK;
        const int kstagen = k0n >> 6;  // 64-k stage index of the next stage
#pragma unroll
        for (int i = 0; i < kALoads; ++i) {
          const int slot = tid + i * kThreads;
          a_ok[i] = slot < kASlots;
          if (a_ok[i]) {
            const int row = slot >> 2;        // 0..kBlockM-1
            const int col = (slot & 3) * 16;  // 0/16/32/48
            a_chunk[i] = *reinterpret_cast<const int4*>(
                x_q + static_cast<int64_t>(m0 + row) * k + k0n + col);
          }
        }
#pragma unroll
        for (int i = 0; i < kBLoads; ++i) {
          const int slot = tid + i * kThreads;
          if (slot < kBSlots) {
            const int b_kc = slot / (kBlockN / 2);  // plane 0..7
            const int b_j = slot - b_kc * (kBlockN / 2);  // n-pair 0..BN/2-1
            const int b_row0 = (n0 + 2 * b_j) & 127;
            const int b_nt = (n0 + 2 * b_j) >> 7;
            b_chunk[i] = *reinterpret_cast<const int4*>(
                weight +
                (((static_cast<int64_t>(kstagen) * (n >> 7) + b_nt) * 8 +
                  b_kc) * 128 + b_row0) * 8);
          }
        }
      }
      // Protect the LDS buffers from the next stage's cooperative overwrite.
      __syncthreads();
    }
  } else {
    // -----------------------------------------------------------------------
    // Retention path (configs 0-2): EXACT iteration-7 loop, byte-identical.
    // -----------------------------------------------------------------------
    for (int k0 = 0; k0 < k; k0 += kStageK) {
      const int kstage = k0 >> 6;  // 64-k stage index inside the swizzled pack
      // Iteration 7 (compute-pipeline round): issue the A and B staging global
      // loads for the WHOLE stage before either vmcnt wait (issue grouping /
      // prefetch distance = one VMEM round trip), then one wait and both LDS
      // writes.  The iteration-6 exact code object serializes the staging as
      //   global_load_dwordx4 A -> s_waitcnt vmcnt(0) -> ds_write_b128 A
      //   global_load_dwordx4 B -> s_waitcnt vmcnt(0) -> ds_write_b128 B
      // (zero instructions between each load and its wait, and the compiler
      // reuses one 4-VGPR slot for both loads), so each 64-K stage exposes ~2
      // L2 round trips back-to-back on the barrier-to-barrier critical path.
      // The A and B loads are independent (A: x_q rows; B: swizzled-pack tile),
      // so grouping them as one issued pair overlaps the two latencies into ~1
      // round trip.  Staging content, LDS addresses, the int32 accumulation
      // order (k0-outer / kk-inner) and the epilogue are byte-identical; the
      // loads simply complete concurrently instead of serially.
      int4 a_chunk[kALoads];
      bool a_ok[kALoads];
#pragma unroll
      for (int i = 0; i < kALoads; ++i) {
        const int slot = tid + i * kThreads;
        a_ok[i] = slot < kASlots;
        if (a_ok[i]) {
          const int row = slot >> 2;        // 0..kBlockM-1
          const int col = (slot & 3) * 16;  // 0/16/32/48
          a_chunk[i] = *reinterpret_cast<const int4*>(
              x_q + static_cast<int64_t>(m0 + row) * k + k0 + col);
        }
      }
      int4 b_chunk[kBLoads];
#pragma unroll
      for (int i = 0; i < kBLoads; ++i) {
        const int slot = tid + i * kThreads;
        if (slot < kBSlots) {
          const int b_kc = slot / (kBlockN / 2);  // plane 0..7
          const int b_j = slot - b_kc * (kBlockN / 2);  // n-pair 0..BN/2-1
          // Iteration 5 (packing round): tile-contiguous (512, 3584) pack -
          // the (kstage, n-tile) B tile is ONE contiguous 8192-B region
          // packed[(((k0*(n/128) + nt)*8 + kc)*128 + row)*8 + b]; the 16-B
          // chunk is two consecutive n rows (n0 + 2j, +1) of one 8-byte k
          // sub-chunk.  It never crosses a 128-row tile boundary (n0 % 64 == 0
          // and a 64/128-row window starting at a multiple of 64 stays inside
          // one 128-row pack tile), so the int4 is aligned in global and LDS.
          const int b_row0 = (n0 + 2 * b_j) & 127;
          const int b_nt = (n0 + 2 * b_j) >> 7;
          b_chunk[i] = *reinterpret_cast<const int4*>(
              weight + (((static_cast<int64_t>(kstage) * (n >> 7) + b_nt) * 8 +
                         b_kc) * 128 + b_row0) * 8);
        }
      }
#pragma unroll
      for (int i = 0; i < kALoads; ++i) {
        if (a_ok[i]) {
          const int slot = tid + i * kThreads;
          const int row = slot >> 2;
          const int col = (slot & 3) * 16;
          if constexpr (kTileStride % 16 == 8) {
            // Iteration 17 (kv_b_proj consolidation round): 88-byte A rows
            // are only 8-byte aligned, so stage the 16-byte chunk as two
            // int64 halves (ds_write2_b64, not ds_write_b128 - lineage rule
            // 3).  Same 16 staging bytes at the same logical (row, kk)
            // slots, so bit-identity holds.
            int8_t* dst = a_tile + row * kTileStride + col;
            const int64_t* src =
                reinterpret_cast<const int64_t*>(&a_chunk[i]);
            *reinterpret_cast<int64_t*>(dst) = src[0];
            *reinterpret_cast<int64_t*>(dst + 8) = src[1];
          } else {
            *reinterpret_cast<int4*>(a_tile + row * kTileStride + col) =
                a_chunk[i];
          }
        }
      }
#pragma unroll
      for (int i = 0; i < kBLoads; ++i) {
        const int slot = tid + i * kThreads;
        if (slot < kBSlots) {
          const int b_kc = slot / (kBlockN / 2);
          const int b_j = slot - b_kc * (kBlockN / 2);
          *reinterpret_cast<int4*>(b_tile + b_kc * kBPlaneStride + b_j * 16) =
              b_chunk[i];
        }
      }
      __syncthreads();

      // Each wave consumes its 64 x kQuadN quadrant: kAccM * kAccN
      // m16n16k32 MMACs per kk.  Accumulation order: k0-outer over 64-K
      // stages, kk-inner (kk = 0 then 32), matching the reference int32
      // accumulation.  All fragment loads are issued before the MMAC burst
      // (load-all-then-MMAC-all, fully unrolled like the q_b_proj winner).
#pragma unroll
      for (int kk = 0; kk < kStageK; kk += kTileK) {
#pragma unroll
        for (int i = 0; i < kAccM; ++i) {
          load_fragment8(a_frag[i],
                         a_tile + (local_row + i * kTileM) * kTileStride + kk,
                         kTileStride, lane);
        }
#pragma unroll
        for (int j = 0; j < kAccN; ++j) {
          load_fragment8_plane_stride(
              b_frag[j],
              b_tile + (kk >> 3) * kBPlaneStride +
                  (local_col + j * kTileN) * 8,
              kBPlaneStride, lane);
        }
#pragma unroll
        for (int j = 0; j < kAccN; ++j) {
#pragma unroll
          for (int i = 0; i < kAccM; ++i) {
            du_mma_sync(acc[i][j], a_frag[i], b_frag[j], acc[i][j]);
          }
        }
      }

      // Last stage only: prefetch the epilogue's per-row x_scale and
      // per-column weight_scale values into registers (uniform branch: k0 is
      // block-uniform, so the barrier below is reached by every thread; the
      // row < m guard keeps the x_scale reads in bounds for any m tail).
      if (k0 + kStageK >= k) {
        const int r = lane & 15;
        const int c4 = lane >> 4;
#pragma unroll
        for (int i = 0; i < kAccM; ++i) {
          const int row = base_row + i * kTileM + r;
          xs_m[i] = (row < m) ? x_scale[row] : 0.0f;
        }
#pragma unroll
        for (int j = 0; j < kAccN; ++j) {
          ws_m[j] = *reinterpret_cast<const float4*>(
              weight_scale + base_col + j * kTileN + 4 * c4);
        }
      }
      // Protect the LDS buffers from the next stage's cooperative overwrite.
      __syncthreads();
    }
  }

  // Coalesced direct-fragment epilogue: one 8-byte bf16 store per lane per
  // fragment (100% store sector efficiency), scales from caller registers.
  // Iteration 6 (kv_b_proj epilogue round): kRneOnly = true - plain RNE bf16
  // conversion without the exec-masked inf/NaN fixup (bit-identical for all
  // finite inputs; see bf16_rne_u16).
#pragma unroll
  for (int j = 0; j < kAccN; ++j) {
#pragma unroll
    for (int i = 0; i < kAccM; ++i) {
      store_prefill_fragment_coalesced_scaled<true>(
          acc[i][j], xs_m[i], ws_m[j], out, base_row + i * kTileM,
          base_col + j * kTileN, m, n, lane);
    }
  }
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
    if (k == kPackSwizzleK2 && n == kPackSwizzleN2) {
      // Tile-contiguous pack (iteration-5 kv_b_proj packing round, exact
      // (512, 3584)): packed[(((k0*(n/128) + nt)*8 + kc)*128 + row)*8 + b] ==
      // raw[kk*n + col] with kk = k0*64 + kc*8 + b, nt = col/128,
      // row = col%128, so raw[kk][col] = packed[...].
      for (int kk = 0; kk < k; ++kk) {
        const int k0 = kk >> 6;
        const int kc = (kk >> 3) & 7;
        const int b = kk & 7;
        const int nt = col >> 7;
        const int row = col & 127;
        const int8_t* p =
            weight +
            ((((static_cast<int64_t>(k0) * (n >> 7) + nt) * 8 + kc) * 128 +
              row) * 8) +
            b;
        acc += static_cast<int32_t>(a_row[kk]) * static_cast<int32_t>(*p);
      }
    } else {
      // Swizzled 64-k-stage pack (iteration 5; exact (2048,2048) q_b_proj
      // pair): packed[((k0*8 + kc)*n + col)*8 + b] == raw[kk*n + col] with
      // kk = k0*64 + kc*8 + b, so raw[kk][col] =
      // packed[((k0*8+kc)*n+col)*8+b].
      for (int kk = 0; kk < k; ++kk) {
        const int k0 = kk >> 6;
        const int kc = (kk >> 3) & 7;
        const int b = kk & 7;
        const int8_t* p = weight +
                          (((static_cast<int64_t>(k0) * 8 + kc) * n + col) * 8) +
                          b;
        acc += static_cast<int32_t>(a_row[kk]) * static_cast<int32_t>(*p);
      }
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
// outside Graph capture) for every (k, n) except the exact swizzled pairs
// (2048, 2048) q_b_proj and (512, 3584) kv_b_proj (which use the
// iteration-5 swizzled pack).  The packed buffer keeps the same byte count
// K*N and the same allocated address, so the layout is graph-stable.
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
// exact swizzled (k, n) pair q_b_proj (2048, 2048) only (kv_b_proj
// (512, 3584) switched to the tile-contiguous w8a8_pack_tile_major_kernel
// in the kv_b_proj iteration-5 packing round):
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

// Iteration 5 (kv_b_proj packing round): tile-contiguous 64-k-stage-major
// permutation for the exact (k, n) == (512, 3584) pair only:
//   packed[(((k0*(n/128) + nt)*8 + kc)*128 + row)*8 + b] = raw[kk*n + col],
//   kk = k0*64 + kc*8 + b, nt = col/128, row = col%128.
// Every (64-k stage, 128-n tile) B tile is ONE contiguous 8192-B global
// region; the tile-family kernel stages it as 8 consecutive 1024-B
// wavefront streams (the k-stage-major swizzle read 8 streams 28,672 B
// apart).  Byte-wise (one thread per byte) so the permutation is trivially
// correct; runs once outside the timed region and outside Graph capture,
// keeping the same byte count K*N and the same graph-stable buffer address.
// The (2048, 2048) q_b_proj pair keeps the k-stage-major swizzle
// (w8a8_pack_swizzle_kernel) byte-identical.
__global__ __launch_bounds__(256) void w8a8_pack_tile_major_kernel(
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
    const int nt = col >> 7;
    const int row = col & 127;
    packed[((((static_cast<int64_t>(k0) * (n >> 7) + nt) * 8 + kc) * 128 +
             row) *
            8) +
           b] = raw[linear];
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
  // API cases (M=2, M=16) and M < 64 - takes the scalar fallback.  The
  // exact (k, n) == (2048, 2048) q_b_proj pair takes the iteration-4 128x128
  // tile with the iteration-5 swizzled-pack plane B layout; the exact
  // (k, n) == (512, 3584) kv_b_proj pair takes the iteration-1 tile family
  // (w8a8_dumma_prefill_tile_kernel, active config kActiveKvTile, also
  // swizzled-pack B); every other (k, n) uses the byte-identical 128x64
  // kNMajorB=false arm.  The scalar fallback decodes the swizzled pack for
  // both swizzled (k, n) pairs.
  if (k == kPackSwizzleK2 && n == kPackSwizzleN2 && m >= 64 && m % 64 == 0) {
    // Exact kv_b_proj (512, 3584): 2-D macro-tile family.  All four
    // instantiations are referenced (so all are compiled); only
    // kActiveKvTile is launched.  grid (N/BN, M/BM), block (waves*64):
    //   0 -> 64x64,   1 wavefront/block,  (56, 64)    = 3584 blocks
    //   1 -> 64x128,  2 wavefronts/block, (28, 64)    = 1792 blocks
    //   2 -> 128x64,  4 wavefronts/block, (56, 32)    = 1792 blocks
    //   3 -> 64x128,  8 wavefronts/block, (28, 64)    = 1792 blocks
    const dim3 grid64x64(static_cast<unsigned>(n / 64),
                         static_cast<unsigned>(m / 64));
    const dim3 block64x64(1 * kWaveSize);
    const dim3 grid64x128(static_cast<unsigned>(n / 128),
                          static_cast<unsigned>(m / 64));
    const dim3 block64x128(2 * kWaveSize);
    const dim3 block64x128_8w(8 * kWaveSize);
    const dim3 grid128x64(static_cast<unsigned>(n / 64),
                          static_cast<unsigned>(m / 128));
    const dim3 block128x64(4 * kWaveSize);
    if (kActiveKvTile == 0) {
      hipLaunchKernelGGL(
          HIP_KERNEL_NAME(w8a8_dumma_prefill_tile_kernel<64, 64, 64>),
          grid64x64, block64x64, 0, stream,
          a, b, x_scale, weight_scale, out_bf16, m, n, k);
    } else if (kActiveKvTile == 1) {
      hipLaunchKernelGGL(
          HIP_KERNEL_NAME(
              w8a8_dumma_prefill_tile_kernel<kKvTileM, kKvTileN, kKvQuadN>),
          grid64x128, block64x128, 0, stream,
          a, b, x_scale, weight_scale, out_bf16, m, n, k);
    } else if (kActiveKvTile == 2) {
      // config 2 (128x64) additionally needs m % 128 == 0; any (512, 3584)
      // m not covered by the active tile falls back to the swizzle decode.
      if (m % 128 == 0) {
        hipLaunchKernelGGL(
            HIP_KERNEL_NAME(w8a8_dumma_prefill_tile_kernel<128, 64, 32>),
            grid128x64, block128x64, 0, stream,
            a, b, x_scale, weight_scale, out_bf16, m, n, k);
      } else {
        constexpr int kBlock = 256;
        const int64_t total = static_cast<int64_t>(m) * n;
        const dim3 grid(static_cast<unsigned>((total + kBlock - 1) / kBlock));
        const dim3 block(kBlock);
        hipLaunchKernelGGL(
            HIP_KERNEL_NAME(w8a8_gemm_scalar_fallback_kernel<true>),
            grid, block, 0, stream,
            a, b, x_scale, weight_scale, out_bf16, m, n, k);
      }
    } else {
      // config 3 (iteration-4 tile-shape round): 64x128 with 32x32
      // quadrants, 512 threads.  m % 64 == 0 is guaranteed by the exact-shape
      // guard above, so no tail fallback is needed.
      // Iteration 16 (kv_b_proj load-placement round): kPrefetchNext = true
      // -> the grouped A+B staging loads for stage s+1 are issued after the
      // burst of stage s (before the protective trailing barrier), partially
      // hiding each of the 8 staging L2 round trips behind the barrier +
      // loop back-edge at unchanged 2 blocks/CU = 16 waves/CU (no register
      // payload crosses the burst).  Iteration 17 (kv_b_proj consolidation
      // round): kALdsPad = 24 additionally applies the iteration-13 A-side
      // LDS bank skew (88-byte A row stride -> zero-conflict A fragment
      // reads; ds_write2_b64 halves at the staging writes) on top of the
      // prefetch.  LDS 13,312 -> 13,824 B/block (2 blocks = 27,648 B <= 64
      // KiB); 16 s_barriers/block unchanged.  Configs 0-2 keep the defaults
      // (kPrefetchNext = false, kALdsPad = 16) -> the exact iteration-7
      // issue-at-stage-top / stride-80 placement (byte-identical).
      hipLaunchKernelGGL(
          HIP_KERNEL_NAME(
              w8a8_dumma_prefill_tile_kernel<64, 128, 32, 32, true, 24>),
          grid64x128, block64x128_8w, 0, stream,
          a, b, x_scale, weight_scale, out_bf16, m, n, k);
    }
  } else if (m >= kBlockM && m % kBlockM == 0 && n % kBlockN == 0 &&
             k % kStageK == 0) {
    const dim3 grid(static_cast<unsigned>(n / kBlockN),
                    static_cast<unsigned>(m / kBlockM));
    const dim3 block(kBlockThreads);
    if (is_swizzled_pack(k, n)) {
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
    if (is_swizzled_pack(k, n)) {
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
  // Pack: iteration-5 kv_b_proj packing round - tile-contiguous permutation
  // for (512, 3584) kv_b_proj, k-stage-major swizzle for (2048, 2048)
  // q_b_proj, identity device-to-device copy for every other (k, n).
  // Packing runs once outside the timed region and keeps the same byte count
  // (graph-stable addresses).
  const int64_t weight_count = static_cast<int64_t>(k) * n;
  const dim3 weight_grid(
      static_cast<unsigned>((weight_count + kBlock - 1) / kBlock));
  if (is_tile_major_pack(k, n)) {
    // Iteration 5 (kv_b_proj packing round): tile-contiguous pack for the
    // exact (512, 3584) pair (one contiguous 8192-B region per (stage,
    // n-tile) B tile); (2048, 2048) keeps the k-stage-major swizzle below.
    hipLaunchKernelGGL(w8a8_pack_tile_major_kernel,
                       weight_grid, block, 0, stream,
                       raw_weight, packed_weight, k, n);
  } else if (is_swizzled_pack(k, n)) {
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
