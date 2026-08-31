// @@variant shape=glm_tp8_shared_down_proj_m4096 commit=9b56135e1fec933006c477066840f9c84f497cef added=2026-08-31
//   median_us=278.4 p90_us=279.2 speedup=63.57 baseline_us=1.77e+04
//   source=glm5-2-dsh-tp8-m4096-1-e6a280a2
// MetaInfer W8A8 INT8 GEMM backend for Hygon gfx928 (K500SM_AI).
// worker_3 (iteration 3), assigned shapes:
//   glm_tp8_shared_gate_up_proj_m4096 : (M, N, K) = (4096, 512, 6144)
//   glm_tp8_shared_down_proj_m4096    : (M, N, K) = (4096, 6144, 256)
//
// This file provides the two stable host launch symbols consumed by
// csrc/bindings.cpp:
//   * launch_w8a8_gemm(...)        - timed, graph-safe GEMM on the caller's
//                                    HIP stream; no allocation, packing,
//                                    autotuning, or synchronization. Only the
//                                    caller-provided out and workspace are
//                                    touched (workspace is unused this round).
//   * launch_pack_w8a8_weight(...) - out-of-timed-region packing. Identity
//                                    device-to-device copy for every (k, n)
//                                    pair EXCEPT the assigned gate_up weight
//                                    (k, n) == (6144, 512), which is packed
//                                    once as a [K, N] -> [N, K] transpose
//                                    (n-major rows, k contiguous) so the timed
//                                    GEMM consumes B fragments as
//                                    k-contiguous bytes. The pack stays
//                                    out-of-timed, out-of-Graph, same byte
//                                    count, same graph-stable buffer.
//
// Bootstrap strategy (correctness-first, usable profiling baseline):
//   * Native INT8 DUMMA m16n16k32 tiled kernel with int32 accumulation for
//     every assigned shape with M >= 128 (both assigned shapes qualify).
//   * 2-D macro-tile family (64x64, 64x128, 128x64) with one 32x32 quadrant
//     per wavefront (four m16n16k32 int32 accumulators kept resident over the
//     whole K loop). Single-buffered K loop (K stage 128, two barriers per
//     stage): no split-K, no raw asm, no speculative double-buffer pipeline.
//
// Round 1 (baseline): the 64x128 tile measured 517.4 us median / 49.81 TOPS;
// routing the exact gate_up shape to the 64x64 tile measured 432.5 us
// (accepted; 19.6% improvement).
//
// Round 2 (operand-reuse, mandated decision): cooperative A+B LDS staging is
// kept and direct per-wave global loads are rejected (A is reused 64x per
// macro-tile via kBlockN and B 64x via kBlockM; direct loads would refetch
// both operands once per wavefront, 4x the 28.3 MB staged global traffic ->
// ~1.73M vmem_read instructions vs 432K, and the TP4 gate_up direct-A arm
// measured 7.9x slower). The focused change makes the staged B layout
// bank-safe for the fragment consumers: the gate_up weight is packed once
// [K,N] -> [N,K] (n-major, k contiguous) and B fragments are loaded
// col_major from a 144-B (128+16) row-strided LDS tile, replacing the
// byte-scattered row_major B reads on the identity [K,N] layout that are the
// 17.3M PMC bank-conflict source. Global loads stay 16-byte vectorized and
// coalesced on both operands. LDS stays 19,456 B/block -> 3 blocks/CU
// (58,368 B, unchanged; round-1's "4 blocks/CU" estimate was impossible on
// both LDS (4x19,456 = 77,824 B > 64 KiB) and VGPR (4x256x72 = 73,728 >
// 65,536)). down_proj keeps the identity 64x128 arm; every unmatched shape
// keeps the generic arms and the scalar fallback (which decodes the packed
// [N,K] layout when (k, n) == (6144, 512) so the paired M=2/M=16 gate_up
// tails stay correct).
//   * A[kBlockM, kStageK] and B staged in bank-skewed LDS (row strides
//     144/80/144 = 16-byte-aligned non-power-of-two), consumed with the
//     library du_load_matrix_sync loaders: A row_major (unchanged, load-
//     bearing on every shape), B row_major for the identity [K, N] layout
//     (down_proj + unmatched shapes) and B col_major for the gate_up
//     [N, K] packed layout (round 2).
//   * Fused direct fragment -> x_scale -> weight_scale -> bf16 epilogue using
//     the verified gfx928 accumulator ownership (row = lane & 15,
//     col_mod4 = lane >> 4, x[i] -> columns col_mod4 + 4*i). The int32 dot is
//     exact for every assigned shape (max |dot| = 6144*127*127 << 2^31), the
//     int32 -> float conversion is exact, and the single fp32 multiply
//     followed by one bfloat16 rounding matches the reference bit-for-bit.
//   * Tail M rows are zero-filled on load and masked on store, so any
//     M >= 128 is supported by the tiled path.
//   * All other (m, n, k) - including every M < 128 API case and the paired
//     M=2/M=16 shapes with the same (N, K) - go to a scalar int8/int32
//     fallback that decodes the [N, K] packed layout when (k, n) ==
//     (6144, 512) and the identity [K, N] layout otherwise.
//
// Round 3 (pipeline, mandated decision): compare single and double buffering
// across K tiles. The exact (4096, 512, 6144) guard now routes to
// w8a8_dumma_prefill_tiled_kernel<64, 64, 64, true, true>: a software-
// pipelined 64-K double-buffered K loop (prologue + one barrier per stage,
// 97 barriers for 96 stages) that issues stage s+1's global loads into the
// idle LDS buffer before the stage-s MMAC burst, overlapping the
// global-load latency that the round-2 single buffer serializes in front of
// every stage (ISA-verified order: global_load_dwordx4 -> ds_write_b128 ->
// s_waitcnt -> s_barrier -> ds_read2 x16 -> v_mmac x16 -> s_barrier, with
// the next stage's loads gated behind the previous stage's barrier). The
// 128-K double-buffered variant is arithmetically impossible at 3 blocks/CU
// (2 x 19,456 = 38,912 B/block x 3 = 116,736 > 64 KiB -> 1 block/CU, a
// harmful occupancy loss), so the pipelined comparison uses the 64-K
// double buffer: LDS 20,480 B/block x 3 = 61,440 <= 64 KiB and
// VGPR 64-72 x 256 x 3 <= 65,536, i.e. 3 blocks/CU preserved (no occupancy
// confound). Everything else is byte-identical: same int4 staging pattern
// (same 28.3 MB global traffic and ~432K vmem_read instructions), same
// k0-outer/kk-inner int32 order (stage size changes only regroup exact
// integer sums, so results stay bit-identical), same col_major B fragments
// (stride 80 = 64+16 bank skew), same fused epilogue, same pack, same
// generic single-buffered arms and scalar fallback. Decision rule (mandate):
// retain double buffering only if PMC shows reduced VMEM/LDS stall evidence
// (lds_wait_instructions drops below ~1.2M, lds_instructions stays ~1.97M,
// occupancy stays 3 blocks/CU) AND the median improves >2%; otherwise the
// round's evidence retains the single-buffered round-2 structure.
//
// Round 10 (fragment-load-form round, restarted from the accepted round-3
// source after the round-9 infra kill): replace the library
// du_load_matrix_sync fragment loads of the exact-shape gate_up kernel with
// the lineage-validated explicit load_frag8 8-byte ds_read_b64 loader. The
// exact (4096, 512, 6144) guard now routes to
// w8a8_dumma_prefill_tiled_kernel<64, 64, 64, true, true, true> (new
// kDirectFrag template arm). The round-3 ISA shows every fragment load as
// ds_read2_b32 + per-byte v_and/v_or_b32_sdwa/v_or3 mask-OR "identity"
// reassembly (~9.4M of the 15.43M VALU instructions per replay); loading the
// same 8 bytes per lane as one int64 (ds_read_b64) removes that VALU and
// halves the fragment bank-conflict factor (32-lane dword phases at stride
// 80 -> 16-lane 8-byte phases). Element-to-slot fragment mapping, v_mmac
// operands, exact k0-outer/kk-inner int32 accumulation, staging/global
// traffic, barriers, tile, grid, LDS bytes, occupancy (3 blocks/CU) and the
// fused epilogue are unchanged; results stay bit-identical. The generic arms
// (<64,128,128>, <128,64,128>, <64,64,128>, all kBNMajor=false) keep the
// library loaders and byte-identical codegen/resources (if constexpr).
//
// Round 11 (LDS bank-skew round, HIP-only; ISA policy keeps raw asm
// disallowed because the HIP plateau rule - 8 valid HIP rounds within +/-2%
// of the then-current best - is unmet). The accepted round-10 kernel still
// shows PMC lds_bank_conflicts = 9,437,184, EXACTLY unchanged from the
// round-3 code object (the -9.07% round-10 win came from VALU removal, not
// from conflicts). 9,437,184 = 192 x 512 blocks x 96 stages: every
// block-stage pays 192 conflict cycles = 128 from the 16 ds_read2_b64
// fragment reads + 64 from the 8 ds_write_b128 staging stores (the 16-B/lane
// store floor). The read half is a stride defect: at stride 80 = 20 dwords,
// 20*r mod 32 has period 8, so every 16-lane read phase groups rows r and
// r+8 onto the same bank pair - a 2-way conflict on ALL 4 sub-phases of
// every fragment read. The exact (4096, 512, 6144) guard now routes to
// w8a8_dumma_prefill_tiled_kernel<64, 64, 64, true, true, true, true> (new
// kSkew8 template arm) which stages both operands at the down_proj-validated
// 8-mod-16 row strides A = 88 (64+24) and B = 72 (64+8) instead of 80:
// 22*r and 18*r mod 32 are distinct over r = 0..15 and gcd(22,32) =
// gcd(18,32) = 2 does not divide 1, so every 16-lane read phase lands on 16
// DISTINCT bank pairs - the fragment reads become conflict-free (128 of the
// 192 cycles removed; only the 2-way 16-B/lane staging-store floor of 64
// cycles remains -> lds_bank_conflicts expected ~3.15M). The 8-mod-16 row
// starts forbid 16-byte staging stores, so each 16-byte int4 is staged as
// two 8-byte halves (ds_write2_b64), the validated down_proj staging form
// (its B tile is exactly stride 72; its A tile exactly stride 88). Global
// loads (int4, same 432,128 vmem_read), pack layout, barrier structure
// (97/block), tile/grid (512 blocks), LDS bytes (A[64,88] + B[64,72] x 2 =
// 20,480 B/block -> 3 blocks/CU unchanged), VGPR, the k0-outer/kk-inner
// int32 order, the element-to-slot fragment mapping and the fused epilogue
// are untouched; only the LDS row strides and the staging store granularity
// change, so results stay bit-identical. Generic arms (<64,128,128>,
// <128,64,128>, <64,64,128>) keep kSkew8 = false and byte-identical
// codegen/resources (if constexpr).
//
// Round 16 (lead-2 loop-carried payload round, HIP-only; raw asm stays
// DISALLOWED - plateau=false, the recent-valid-improvement window
// [-34.44%, +2.82%, +9.97%] does not meet three valid HIP rounds within
// [-2%, +2%) of the then-current best, and no prior ISA-guided round
// recorded a compiler limitation with target instructions). The accepted
// round-11 exact code object (digest 9889820e, median 293.30 us / p90
// 293.545) shows the software-pipeline intent is DEFEATED by the compiler's
// wait placement: per stage the steady-state loop is
//   global_load_dwordx4 (A prefetch s+1) -> global_load_dwordx4 (B prefetch
//   s+1) -> s_waitcnt vmcnt(1) -> s_waitcnt vmcnt(0)   <- waits IMMEDIATELY
//   -> 2 x ds_write2_b64 (staging stores) -> 4 x ds_read2_b64 (fragment
//   reads, stage s) -> 8 x v_mmac CONTIGUOUS (stage s) -> s_barrier -> loop.
// i.e. the stage-(s+1) global loads are waited ~12-16 instructions after
// issue, BEFORE the stage-s MMAC burst, so the full global-load latency
// (~600-1000+ cycles under 120-CU L2 contention) sits on the per-stage
// critical path with zero overlap (the round-12 analysis'
// "prefetch global load -> ds_write2_b64 -> s_barrier -> ds_read2_b64 ->
// v_mmac" chain; PMC: 6.0M VALU + 1.18M LDS + 465K VMEM wave-instructions
// over 293 us x 120 CUs, i.e. ~95-99% issue-stalled). Iteration history
// rules out the other levers on this exact shape: tile aspect (iter 4
// 32x64: 749.9 us; iter 8 64x128 w8: 395.4 us), 2-wave quadrants (iter 6:
// 629.0 us - wave count per CU is load-bearing), LDS-side fragment
// prefetch (iter 7: 337.97 us, flat/worse), stage 32 / 6 blocks per CU
// (iter 12: 447.4 us - per-stage fixed cost, not occupancy, is the lever).
// The one untested mechanism that directly removes the exposed vmcnt wait:
// a LEAD-2 LOOP-CARRIED REGISTER PAYLOAD (the TP4 o_proj lineage's
// validated "stage-top publish" form). The exact (4096, 512, 6144) guard
// now routes to w8a8_dumma_prefill_tiled_kernel<64,64,64,true,true,true,
// true,true> (new kPayload template arm): each thread keeps the one A int4
// + one B int4 of stage s+2 in VGPR across the stage-(s+1) barrier, issues
// those global loads at the TOP of iteration s (no wait), and PUBLISHES
// the payload carried from iteration s-1 into the idle LDS buffer at the
// TOP of iteration s (before the stage-s MMAC burst). The first use of the
// payload registers is the next iteration's publish, and the s_barrier
// fences outstanding vmem, so the compiler can no longer place
// s_waitcnt vmcnt(0) immediately after the loads: the wait lands after the
// stage-s fragment reads + MMAC burst (immediately before the barrier), and
// the global-load latency overlaps the whole compute segment instead of
// serializing in front of it. One barrier per 64-K stage is preserved
// (prologue + 96 = 97/block); the publish targets the idle buffer (WAR
// retired by the previous barrier, RAW retired by the stage barrier), so no
// second barrier is introduced. Everything else is byte-identical: 64x64
// tile / 256 threads / 4 waves / 32x32 quadrant per wave, grid (8,64) =
// 512 blocks, kSkew8 strides 88/72 with ds_write2_b64 staging (8 per
// block-stage), same 16 ds_read2_b64 fragment reads per block-stage (same
// lds_instructions ~1,179,648), same 20,480 B LDS/block -> 3 blocks/CU
// (payload costs 8 VGPRs: ~54 total, 54 x 256 x 3 = 41,472 <= 65,536; no
// spill risk at 3 blocks/CU), same int4 global loads (same ~432,128
// vmem_read / 32,768 vmem_write), same k0-outer/kk-inner int32 order and
// the same element-to-slot fragment mapping - the payload changes only WHEN
// global bytes land in LDS, never which bytes or the accumulation order, so
// results stay bit-identical (mismatch 0 / max_abs_error 0.0 expected).
// Generic arms (<64,128,128>, <128,64,128>, <64,64,128>, all kBNMajor=
// false) keep kPayload = false and byte-identical codegen/resources
// (if constexpr).
//
// Round 2 (this session, glm_tp8_shared_down_proj_m4096 exact-shape arm):
// the generic <64,128,128> identity-layout arm that served down_proj
// (4096, 6144, 256) measured 314.72 us median with PMC
// lds_bank_conflicts = 7,864,320, ~600 LDS instructions and heavy per-byte
// fragment-reassembly VALU per block-stage (library row_major loaders on
// the identity [K, N] B layout). The exact (4096, 6144, 256) guard now
// routes to w8a8_dumma_prefill_tiled_kernel<64, 128, 64, true, true, true,
// true> - the file's own lineage-validated gate_up machinery transplanted
// to down_proj: B packed once [K,N] -> [N,K] n-major (kBNMajor, same
// transpose kernel as gate_up, same byte count / graph-stable buffer,
// out-of-timed/out-of-Graph), 64-K double-buffered K loop (kDoubleBuffer,
// prologue + 4 stages for K=256, one barrier per stage), explicit 8-byte
// load_frag8 fragment reads (kDirectFrag, one ds_read_b64 per fragment,
// zero reassembly VALU) and the 8-mod-16 bank-skewed LDS strides A=88 /
// B=72 (kSkew8, down_proj-validated, conflict-free 16-lane fragment
// phases). LDS A[64,88] + B[128,72] x 2 = 29,696 B/block -> 2 blocks/CU
// (occupancy unchanged vs the 27,648-B single buffer); grid (48, 64) =
// 3072 blocks; the stage size 128->64 only regroups exact integer sums
// (k0-outer / kk-inner int32 order over k = 0,32,64,96,128,160,192,224 is
// unchanged) and load_frag8 fills the same x[0..7] slots with the same
// little-endian bytes, so results stay bit-identical (mismatch 0 /
// max_abs_error 0.0 expected).
//
// Round 3 (this session, glm_tp8_shared_down_proj_m4096 pipeline round):
// mandated single vs double buffering comparison across K tiles. The
// accepted round-2 arm (<64,128,64,true,true,true,true>, 286.53 us median /
// 286.64 us p90, digest cd9791d0) is the 64-K DOUBLE-buffered pipeline; its
// exact code object shows the software pipeline already works on this shape
// (steady-state loop: ds_write2_b64 publish s+1 -> global_load_dwordx4 x2
// for s+2 -> ds_read2_b64 x4 (fragment reads s+1) -> v_mmac x8 -> s_waitcnt
// vmcnt(0) lgkmcnt(0) -> s_barrier: the stage-(s+2) loads stay in flight
// across the fragment reads + MMAC burst and are waited only after it, i.e.
// reduced VMEM stalls), and PMC shows no harmful LDS or occupancy growth vs
// the pre-round-2 single buffer (29,696 vs 27,648 B/block, 2 blocks/CU
// pinned; lds_wait 162,664 / lds_instructions 540,672 / conflicts
// 1,179,648). The (4096, 6144, 256) guard now routes to the SINGLE-buffered
// form <64,128,64,true,false,true,true>: identical stage size 64, identical
// kBNMajor/kDirectFrag/kSkew8/tile/grid/epilogue, kDoubleBuffer = false, so
// each stage serializes its global-load round trip (load -> wait ->
// ds_write2_b64) in front of the MMAC burst with two barriers per stage
// (8/block vs 5/block) and LDS halves to 14,848 B/block. The single-buffer
// branch now threads kSkew8 and kDirectFrag through its staging/MMAC calls
// (previously hardcoded to the library-loader defaults; generic arms keep
// byte-identical codegen via if constexpr). Falsifiable gate: if the
// single-buffered median is NOT strictly below 286.53368949890137 (p90
// guard < 286.6385078430176), double buffering is retained with the ISA/PMC
// evidence above; if it IS below, the single buffer wins on this shape
// (LDS halved; VGPR may drop <= 42 and open 3 blocks/CU, the trusted
// occupancy-probe set [2, 3, 4]). Results stay bit-identical either way
// (same stage size, same k0-outer/kk-inner int32 order, same bytes).
//
// Round 4 (this session, glm_tp8_shared_down_proj_m4096 tile-shape round):
// the balanced 1:1 aspect probe <64,64,64,true,false,true,true> (256 thr =
// 4 waves, 6144 blocks, LDS 10,240 B/block -> 4 blocks/CU, doubled A
// staging) measured 297.15 us median / 297.97 us p90 and was REJECTED
// (guard restored to the round-3 single-buffered <64,128,64,true,false,
// true,true> winner, 281.65 us / 282.52 us): co-residency was already
// saturated at 2 blocks/CU and the +33% A-side staging cost was not repaid.
//
// Round 5 (this session, glm_tp8_shared_down_proj_m4096 PACKING round): test
// one weight packing/swizzle that makes each DUMMA B tile vector-loadable
// and LDS-bank-safe, with the packing outside timing and the graph-stable
// packed layout validated. The accepted round-3 arm stages its 128x64 B
// tile from the plain n-major [N,K] packed weight: per block-stage the 512
// int4 loads are 16 rows x 4 int4 per warp at 256-B row stride, i.e. every
// warp instruction touches 16 DISTINCT 64-B segments (50% of each 128-B L2
// line) - vector loads, but scattered at L2 granularity. The round changes
// the one-time pack for (k,n) == (256,6144) ONLY to the panel layout
// [N/128][K/64][128][64] (n-major 64-k-byte rows per panel; byte-exact index
// relation packed[((nt*(k/64)+kt)*128+nn)*64+kk64] == raw[(nt*128+nn)*k +
// kt*64+kk64], verified by simulation): each block's per-stage B tile is now
// ONE contiguous 8 KiB region, so every warp instruction reads 1 KiB
// contiguous (full sector/line utilization; per block-stage the B staging
// collapses from 128 half-used 128-B lines to 64 fully-used lines). The new
// exact-shape arm w8a8_dumma_prefill_tiled_kernel<64,128,64,true,false,
// true,true,false,true> (new kBPanel template flag, default false) is
// byte-identical in EVERYTHING else: same 64-K single-buffered stage, same
// kBNMajor/kDirectFrag/kSkew8, same 64x128 tile / 512 thr / 8 waves / 32x32
// quadrant, same grid (48,64) = 3072 blocks, same LDS A[64,88]+B[128,72] =
// 14,848 B/block @ 2 blocks/CU, same ds_write2_b64 staging stores and same
// conflict-free load_frag8 ds_read2_b64 fragment reads (LDS-bank-safe
// preserved - only the GLOBAL byte order of the packed buffer changes,
// never the staged LDS bytes), same k0-outer/kk-inner int32 order, same
// fused epilogue -> mismatch 0 / max_abs_error 0.0 expected. The scalar
// fallback decodes the panel layout for (k,n) == (256,6144) so M<128 tails
// stay correct; gate_up (6144,512) keeps its plain [N,K] transpose pack and
// its own arm; generic arms are untouched (kBPanel=false). Packing runs
// out-of-timed/out-of-Graph in launch_pack_w8a8_weight, same byte count
// (256*6144), same graph-stable buffer. Expected effect: B global staging
// becomes perfectly coalesced 1 KiB-per-warp streams with half the L2 line
// count; PMC vmem_read_instructions unchanged (~614,400 - same 512 int4
// loads/block-stage), lds_* identical (LDS layout untouched), l2_hits/l2_
// misses shift slightly in the direction of fewer miss sectors, valu ~flat
// (address math per thread unchanged: one base + offset). FALSIFIABLE GATE:
// median_us < 281.6507148742676 AND p90_us < 282.5235176086426 -> B-global
// coalescing was binding on this shape and the panel pack wins; otherwise
// the plain n-major pack is retained and this round records that B reads
// are L2-resident (90.4% hit) and NOT binding on the LDS-bound critical
// path (consistent with the lineage rule that pack layout stops paying once
// fragment loads/strides are fixed).
//
// Header order is fixed by the control plane: hip_runtime first (du_mma.h
// is not self-contained before it), hip_bfloat16 second, du_mma.h last.

#include <hip/hip_runtime.h>
#include <hip/hip_bfloat16.h>
#include <du_mma.h>

#include <cstdint>

namespace {

// gfx928 INT8 DUMMA primitive: m16n16k32, int8 x int8 -> int32.
constexpr int kTileM = 16;
constexpr int kTileN = 16;
constexpr int kTileK = 32;

// gfx928 wavefront is 64 lanes; every blockDim must be a multiple of 64.
constexpr int kWaveSize = 64;

// Large-prefill dispatch geometry. Each macro-tile config uses a 32x32
// quadrant per wave, so waves per block = (BlockM/32) * (BlockN/32).
constexpr int kPrefillMinM = 128;
constexpr int kPrefillStageK = 128;
constexpr int kDefaultBlockM = 64;
constexpr int kDefaultBlockN = 64;

constexpr int kScalarThreads = 256;

using namespace du::dumma;

// Verified gfx928 INT8 DUMMA accumulator ownership:
//   row = lane & 15, col_mod4 = lane >> 4, frag.x[i] -> columns
//   col_mod4 + 4*i. Scale by x_scale[row] and weight_scale[col] and store
//   bf16 directly from the fragment (no accumulator LDS round trip).
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
  const float xs = x_scale[row];
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const int col = base_col + col_mod4 + 4 * i;
    const float scaled = static_cast<float>(frag.x[i]) * xs * weight_scale[col];
    out[row * n + col] = __float2bfloat16(scaled);
  }
}

// Round 10 (fragment-load-form round): explicit 8-byte fragment loader
// replacing the library du_load_matrix_sync for the exact-shape gate_up
// kernel. Validated on the sibling TP8 down_proj accepted kernel (iter 8,
// -25.9%: 230.91 -> 171.19 us) and the TP4 gate_up lineage (iter 13
// load_fragment8, -21%: 280.05 -> 220.71 us). For BOTH the library row_major
// matrix_a and col_major matrix_b m16n16k32 int8 loaders, lane l -> row
// (l & 15), k-quarter (l >> 4), and the loader fetches the same eight
// contiguous bytes at p[row*ldm + (l>>4)*8 .. +7] and stores them into the
// same x[0..7] slots in the same little-endian order (the round-3 ISA shows
// the compiler lowering this to ds_read2_b32 followed by a per-byte
// v_and/v_or_b32_sdwa/v_or3 mask-OR "identity" reassembly - ~9.4M dead-ish
// VALU instructions per replay, ~61% of the 15.43M VALU total). Loading the
// same 8 bytes as one int64 forces the 8-byte ds_read_b64 form instead:
// (a) the mask/OR reassembly VALU disappears, and (b) the LDS access
// changes from ds_read2_b32 (32-lane dword phases - at the stride-80 row
// pattern every phase groups rows 0-15 of k-quarter q with rows 0-15 of
// k-quarter q+1 onto the same 16 bank phases, a 4-way conflict per phase)
// to ds_read_b64 (16-lane phases: rows 0-15 land on 16 bank-pairs
// {20*r mod 32} = 8 distinct pairs hit twice - a 2-way conflict per phase,
// half of the current fragment-conflict factor). The element-to-slot
// mapping is unchanged (f.x[0..7] = the same 8 bytes in the same
// little-endian order), the v_mmac operands and the exact k0-outer/kk-inner
// int32 accumulation sequence are untouched, and the staging stores, LDS
// strides, barriers, tile, grid, occupancy and epilogue are byte-identical.
template <typename Frag>
__device__ __forceinline__ void load_frag8(Frag& f, const int8_t* p,
                                           unsigned ldm) {
  const unsigned row = __lane_id() & 0xf;
  const unsigned kq = __lane_id() >> 4;
  const int64_t v =
      *reinterpret_cast<const int64_t*>(p + row * ldm + (kq << 3));
  reinterpret_cast<int64_t*>(f.x)[0] = v;
}

// Stage one K tile (k0) of A[kBlockM, kStageK] and B into the caller-offset
// LDS buffers (bank-skewed row strides; int4 vectorized coalesced global
// loads, zero-filled tail M rows). kBNMajor == true stages B from the packed
// [N, K] n-major gate_up layout (n-major rows, k contiguous); otherwise from
// the identity [K, N] layout. Shared by the single-buffered and
// double-buffered K loops; the buffer offset (0 or kABytes/kBMaxBytes) is
// applied by the caller so one body serves both pipeline forms.
//
// kSkew8 (round 11, exact-shape arm only): A and B row strides become the
// down_proj-validated 8-mod-16 values 88 (kStageK + 24) and 72 (kStageK + 8)
// so every 16-lane fragment-read phase lands on 16 distinct bank pairs (see
// the round-11 header comment). The 8-mod-16 row starts forbid 16-byte
// stores, so each 16-byte int4 chunk is staged as two 8-byte halves
// (ds_write2_b64); the global int4 loads are byte-identical.
template <int kBlockM, int kBlockN, int kStageK, bool kBNMajor,
          bool kSkew8 = false, bool kBPanel = false>
__device__ __forceinline__ void stage_prefill_tiles(
    const int8_t* __restrict__ x_q,
    const int8_t* __restrict__ weight,
    int8_t* a_tile,
    int8_t* b_tile,
    int tid,
    int m0,
    int n0,
    int m,
    int n,
    int k,
    int k0) {
  constexpr int kThreads = kBlockM * kBlockN / 1024 * kWaveSize;
  // 16-byte bank skew: keeps int4 staging stores aligned while spreading the
  // fragment rows across distinct LDS banks.
  constexpr int kAStride = kSkew8 ? kStageK + 24 : kStageK + sizeof(int4);
  constexpr int kBStride = kBlockN + sizeof(int4);   // identity [K,N] rows
  constexpr int kBNStride = kSkew8 ? kStageK + 8 : kStageK + sizeof(int4);
  static_assert(!kSkew8 || kStageK == 64,
                "kSkew8 strides 88/72 assume kStageK == 64");
  constexpr int kAInt4PerRow = kStageK / sizeof(int4);
  constexpr int kAInt4 = kBlockM * kAInt4PerRow;
  constexpr int kBInt4PerRow = kBlockN / sizeof(int4);
  constexpr int kBInt4 = kStageK * kBInt4PerRow;
  constexpr int kBNInt4PerRow = kStageK / sizeof(int4);
  constexpr int kBNInt4 = kBlockN * kBNInt4PerRow;
  // Stage A[kBlockM, kStageK] into LDS (zero-filled tail M rows).
  for (int vec = tid; vec < kAInt4; vec += kThreads) {
    const int local_row = vec / kAInt4PerRow;
    const int v = vec - local_row * kAInt4PerRow;
    const int global_row = m0 + local_row;
    const int4 val =
        global_row < m
            ? *reinterpret_cast<const int4*>(
                  x_q + static_cast<int64_t>(global_row) * k + k0 +
                  v * static_cast<int>(sizeof(int4)))
            : int4{0, 0, 0, 0};
    if constexpr (kSkew8) {
      // 8-mod-16 row starts: write the 16 bytes as two 8-byte halves so every
      // store stays 8-byte aligned (ds_write2_b64 staging; same LDS bytes).
      int8_t* dst =
          a_tile + local_row * kAStride + v * static_cast<int>(sizeof(int4));
      reinterpret_cast<int64_t*>(dst)[0] =
          reinterpret_cast<const int64_t*>(&val)[0];
      reinterpret_cast<int64_t*>(dst)[1] =
          reinterpret_cast<const int64_t*>(&val)[1];
    } else {
      reinterpret_cast<int4*>(a_tile + local_row * kAStride)[v] = val;
    }
  }
  if constexpr (kBNMajor) {
    // Stage B[kBlockN, kStageK] from the packed [N, K] weight (n-major:
    // row = n_local, k contiguous). Same vectorized coalesced int4 pattern
    // as the identity arm, one per 16 consecutive k bytes.
    //
    // kBPanel (round 5, down_proj exact-shape arm only): the one-time pack
    // for (k,n) == (256,6144) is the panel layout [N/128][K/64][128][64] -
    // each block's per-stage B tile is one contiguous 8 KiB region, so a
    // 64-lane warp instruction reads 1 KiB contiguous (full L2 line/sector
    // utilization) instead of 16 rows x 64 B scattered at 256-B stride
    // (50% of each 128-B line). The per-thread address is still one base +
    // offset: packed byte offset = panel_base + local_col*64 + v*16 where
    // panel_base = (n0/128)*(k/64)*8192 + (k0/64)*8192. The LDS stores are
    // untouched (same stride-72 ds_write2_b64 halves -> LDS-bank-safe
    // staging and conflict-free fragment reads preserved); only the global
    // byte order of the loaded int4 changes, so the staged LDS bytes are
    // identical to the plain n-major arm (bit-identical results).
    for (int vec = tid; vec < kBNInt4; vec += kThreads) {
      const int local_col = vec / kBNInt4PerRow;
      const int v = vec - local_col * kBNInt4PerRow;
      int4 val;
      if constexpr (kBPanel) {
        const int64_t panel_base =
            (static_cast<int64_t>(n0) / kBlockN) * (k / kStageK) *
                (kBlockN * kStageK) +
            (k0 / kStageK) * (kBlockN * kStageK);
        val = *reinterpret_cast<const int4*>(
            weight + panel_base +
            static_cast<int64_t>(local_col) * kStageK +
            v * static_cast<int>(sizeof(int4)));
      } else {
        val = *reinterpret_cast<const int4*>(
            weight + static_cast<int64_t>(n0 + local_col) * k + k0 +
            v * static_cast<int>(sizeof(int4)));
      }
      if constexpr (kSkew8) {
        int8_t* dst =
            b_tile + local_col * kBNStride +
            v * static_cast<int>(sizeof(int4));
        reinterpret_cast<int64_t*>(dst)[0] =
            reinterpret_cast<const int64_t*>(&val)[0];
        reinterpret_cast<int64_t*>(dst)[1] =
            reinterpret_cast<const int64_t*>(&val)[1];
      } else {
        reinterpret_cast<int4*>(b_tile + local_col * kBNStride)[v] = val;
      }
    }
  } else {
    // Stage B[kStageK, kBlockN] into LDS from the identity [K, N] weight.
    for (int vec = tid; vec < kBInt4; vec += kThreads) {
      const int kk = vec / kBInt4PerRow;
      const int v = vec - kk * kBInt4PerRow;
      reinterpret_cast<int4*>(b_tile + kk * kBStride)[v] =
          *reinterpret_cast<const int4*>(
              weight + static_cast<int64_t>(k0 + kk) * n + n0 +
              v * static_cast<int>(sizeof(int4)));
    }
  }
}

// Round 16 (lead-2 loop-carried payload, exact-shape arm only): the register
// half of the lead-2 software pipeline. Each thread prefetches exactly one A
// int4 and one B int4 of a later K stage into thread-local registers and
// returns WITHOUT waiting (the vmcnt wait is emitted by the compiler at the
// first use, one iteration later). The <64,64,64> arm stages kStageK * 64
// bytes per operand per stage, i.e. kAInt4 == kBNInt4 == kThreads == 256:
// one int4 per thread per operand, so the whole payload costs 8 VGPRs. The
// (row, v) mapping is the same as stage_prefill_tiles, so
// publish_stage_payload writes the exact same LDS addresses the staging
// loop would (bit-identical staging bytes, just loaded one stage earlier).
template <int kBlockM, int kBlockN, int kStageK, bool kBNMajor, bool kSkew8>
__device__ __forceinline__ void prefetch_stage_payload(
    const int8_t* __restrict__ x_q,
    const int8_t* __restrict__ weight,
    int tid,
    int m0,
    int n0,
    int m,
    int k,
    int k0,
    int4& a_payload,
    int4& b_payload) {
  constexpr int kThreads = kBlockM * kBlockN / 1024 * kWaveSize;
  constexpr int kAInt4PerRow = kStageK / static_cast<int>(sizeof(int4));
  constexpr int kAInt4 = kBlockM * kAInt4PerRow;
  constexpr int kBNInt4PerRow = kStageK / static_cast<int>(sizeof(int4));
  constexpr int kBNInt4 = kBlockN * kBNInt4PerRow;
  static_assert(kAInt4 == kThreads && kBNInt4 == kThreads,
                "payload arm requires exactly one int4 per thread per "
                "operand (exact-shape <64,64,64> arm)");
  static_assert(kBNMajor, "payload arm stages the packed n-major B layout");
  // Same vec -> (local_row, v) mapping as stage_prefill_tiles, so the
  // publish half stores to the same LDS addresses.
  const int a_local_row = tid / kAInt4PerRow;
  const int a_v = tid - a_local_row * kAInt4PerRow;
  const int global_row = m0 + a_local_row;
  // Tail M rows are zero-filled exactly as in stage_prefill_tiles.
  a_payload = global_row < m
                  ? *reinterpret_cast<const int4*>(
                        x_q + static_cast<int64_t>(global_row) * k + k0 +
                        a_v * static_cast<int>(sizeof(int4)))
                  : int4{0, 0, 0, 0};
  const int b_local_col = tid / kBNInt4PerRow;
  const int b_v = tid - b_local_col * kBNInt4PerRow;
  b_payload = *reinterpret_cast<const int4*>(
      weight + static_cast<int64_t>(n0 + b_local_col) * k + k0 +
      b_v * static_cast<int>(sizeof(int4)));
}

// Round 16: the LDS half of the lead-2 payload pipeline. Writes the payload
// carried in registers (loaded by prefetch_stage_payload one stage earlier)
// into the caller-offset LDS tile with the exact same 8-byte-half
// ds_write2_b64 stores (kSkew8 strides 88/72) that stage_prefill_tiles
// emits, so each thread republishes exactly the bytes it loaded.
template <int kBlockM, int kBlockN, int kStageK, bool kBNMajor, bool kSkew8>
__device__ __forceinline__ void publish_stage_payload(
    int4 a_payload,
    int4 b_payload,
    int8_t* a_tile,
    int8_t* b_tile,
    int tid) {
  constexpr int kThreads = kBlockM * kBlockN / 1024 * kWaveSize;
  constexpr int kAStride = kSkew8 ? kStageK + 24 : kStageK + sizeof(int4);
  constexpr int kBNStride = kSkew8 ? kStageK + 8 : kStageK + sizeof(int4);
  constexpr int kAInt4PerRow = kStageK / static_cast<int>(sizeof(int4));
  constexpr int kAInt4 = kBlockM * kAInt4PerRow;
  constexpr int kBNInt4PerRow = kStageK / static_cast<int>(sizeof(int4));
  constexpr int kBNInt4 = kBlockN * kBNInt4PerRow;
  static_assert(kAInt4 == kThreads && kBNInt4 == kThreads,
                "payload arm requires exactly one int4 per thread per "
                "operand (exact-shape <64,64,64> arm)");
  static_assert(kBNMajor, "payload arm stages the packed n-major B layout");
  const int a_local_row = tid / kAInt4PerRow;
  const int a_v = tid - a_local_row * kAInt4PerRow;
  int8_t* a_dst =
      a_tile + a_local_row * kAStride + a_v * static_cast<int>(sizeof(int4));
  if constexpr (kSkew8) {
    // 8-mod-16 row starts: write the 16 bytes as two 8-byte halves
    // (ds_write2_b64 staging, identical to stage_prefill_tiles).
    reinterpret_cast<int64_t*>(a_dst)[0] =
        reinterpret_cast<const int64_t*>(&a_payload)[0];
    reinterpret_cast<int64_t*>(a_dst)[1] =
        reinterpret_cast<const int64_t*>(&a_payload)[1];
  } else {
    reinterpret_cast<int4*>(a_dst)[0] = a_payload;
  }
  const int b_local_col = tid / kBNInt4PerRow;
  const int b_v = tid - b_local_col * kBNInt4PerRow;
  int8_t* b_dst =
      b_tile + b_local_col * kBNStride + b_v * static_cast<int>(sizeof(int4));
  if constexpr (kSkew8) {
    reinterpret_cast<int64_t*>(b_dst)[0] =
        reinterpret_cast<const int64_t*>(&b_payload)[0];
    reinterpret_cast<int64_t*>(b_dst)[1] =
        reinterpret_cast<const int64_t*>(&b_payload)[1];
  } else {
    reinterpret_cast<int4*>(b_dst)[0] = b_payload;
  }
}

// One wave's 32x32-quadrant MMAC burst for the K tile staged at a_tile/b_tile
// (caller-offset to the target buffer). k0-outer / kk-inner int32 accumulation
// order; stage-size changes only regroup exact integer sums (max |dot| =
// 6144*127*127 << 2^31), so results stay bit-identical.
template <int kBlockM, int kBlockN, int kStageK, bool kBNMajor,
          bool kDirectFrag = false, bool kSkew8 = false>
__device__ __forceinline__ void mmac_prefill_stage(
    DUFragment<matrix_a, kTileM, kTileN, kTileK, signed char, row_major>&
        a_frag0,
    DUFragment<matrix_a, kTileM, kTileN, kTileK, signed char, row_major>&
        a_frag1,
    DUFragment<accumulator, kTileM, kTileN, kTileK, int>& acc00,
    DUFragment<accumulator, kTileM, kTileN, kTileK, int>& acc01,
    DUFragment<accumulator, kTileM, kTileN, kTileK, int>& acc10,
    DUFragment<accumulator, kTileM, kTileN, kTileK, int>& acc11,
    int8_t* a_tile,
    int8_t* b_tile,
    int local_row,
    int local_col) {
  // kSkew8 (round 11, exact-shape arm only): A 88 / B 72 row strides; the
  // load_frag8 ldm arguments below pick them up unchanged (same 8-byte
  // aligned fragment reads, conflict-free 16-lane bank phases).
  constexpr int kAStride = kSkew8 ? kStageK + 24 : kStageK + sizeof(int4);
  constexpr int kBStride = kBlockN + sizeof(int4);   // identity [K,N] rows
  constexpr int kBNStride = kSkew8 ? kStageK + 8 : kStageK + sizeof(int4);
  if constexpr (kBNMajor) {
    if constexpr (kDirectFrag) {
      // Round 10 exact-shape path: explicit load_frag8 8-byte ds_read_b64
      // fragment reads replacing the library loaders (see load_frag8
      // comment). Both steps of the 64-K stage are unrolled into
      // independent fragment register sets so all eight reads of a stage
      // issue before the MMACs; the kk = 0 MMAC group runs first, then the
      // kk = kTileK group - the exact k0-outer/kk-inner int32 order and the
      // element-to-slot fragment mapping of the library path are preserved
      // (f.x[0..7] = the same 8 bytes in the same little-endian order), so
      // results stay bit-identical.
      static_assert(kStageK == 2 * kTileK,
                    "kDirectFrag two-step unroll assumes kStageK == 2*kTileK");
      DUFragment<matrix_a, kTileM, kTileN, kTileK, signed char, row_major>
          a_frag0_1, a_frag1_1;
      DUFragment<matrix_b, kTileM, kTileN, kTileK, signed char, col_major>
          b_frag0, b_frag1, b_frag0_1, b_frag1_1;
      load_frag8(a_frag0, a_tile + local_row * kAStride, kAStride);
      load_frag8(a_frag1, a_tile + (local_row + kTileM) * kAStride, kAStride);
      load_frag8(b_frag0, b_tile + local_col * kBNStride, kBNStride);
      load_frag8(b_frag1, b_tile + (local_col + kTileN) * kBNStride,
                 kBNStride);
      load_frag8(a_frag0_1, a_tile + local_row * kAStride + kTileK,
                 kAStride);
      load_frag8(a_frag1_1,
                 a_tile + (local_row + kTileM) * kAStride + kTileK, kAStride);
      load_frag8(b_frag0_1, b_tile + local_col * kBNStride + kTileK,
                 kBNStride);
      load_frag8(b_frag1_1,
                 b_tile + (local_col + kTileN) * kBNStride + kTileK,
                 kBNStride);
      du_mma_sync(acc00, a_frag0, b_frag0, acc00);
      du_mma_sync(acc01, a_frag0, b_frag1, acc01);
      du_mma_sync(acc10, a_frag1, b_frag0, acc10);
      du_mma_sync(acc11, a_frag1, b_frag1, acc11);
      du_mma_sync(acc00, a_frag0_1, b_frag0_1, acc00);
      du_mma_sync(acc01, a_frag0_1, b_frag1_1, acc01);
      du_mma_sync(acc10, a_frag1_1, b_frag0_1, acc10);
      du_mma_sync(acc11, a_frag1_1, b_frag1_1, acc11);
    } else {
      DUFragment<matrix_b, kTileM, kTileN, kTileK, signed char, col_major>
          b_frag0, b_frag1;
#pragma unroll
      for (int kk = 0; kk < kStageK; kk += kTileK) {
        du_load_matrix_sync(
            a_frag0, a_tile + local_row * kAStride + kk, kAStride);
        du_load_matrix_sync(
            a_frag1, a_tile + (local_row + kTileM) * kAStride + kk, kAStride);
        du_load_matrix_sync(
            b_frag0, b_tile + local_col * kBNStride + kk, kBNStride);
        du_load_matrix_sync(
            b_frag1, b_tile + (local_col + kTileN) * kBNStride + kk,
            kBNStride);
        du_mma_sync(acc00, a_frag0, b_frag0, acc00);
        du_mma_sync(acc01, a_frag0, b_frag1, acc01);
        du_mma_sync(acc10, a_frag1, b_frag0, acc10);
        du_mma_sync(acc11, a_frag1, b_frag1, acc11);
      }
    }
  } else {
    DUFragment<matrix_b, kTileM, kTileN, kTileK, signed char, row_major>
        b_frag0, b_frag1;
#pragma unroll
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
  }
}

// Large-prefill path: one block computes a kBlockM x kBlockN output tile.
// (kBlockM/32) x (kBlockN/32) wavefronts each own a 32x32 quadrant (four
// m16n16k32 DUMMA accumulators), while the block cooperatively stages
// A[kBlockM, kStageK] and B in bank-skewed LDS.
//
// kDoubleBuffer == false: single-buffered K loop (two barriers per stage);
// stage s+1's global loads cannot start until the stage-s barrier, so each
// stage serializes one global-load round trip in front of the compute
// (bootstrap structure; co-resident block streams hide it at the CU level).
//
// kDoubleBuffer == true: software-pipelined K loop (round 3). Stage s+1's
// global loads are issued into the idle LDS buffer before the stage-s MMAC
// burst, overlapping the serialized global-load latency with compute. One
// barrier per stage (plus one prologue): it both retires stage-s fragment
// reads (WAR on the buffer just read) and makes stage s+1's ds_writes
// visible (RAW on the prefetched buffer). The gate_up shape uses
// <64, 64, 64, true, true>: two 64-K tiles = 20,480 B/block x 3 blocks/CU =
// 61,440 <= 64 KiB (occupancy preserved; a 128-K double buffer would need
// 38,912 B/block -> 1 block/CU, a harmful occupancy loss).
//
// kBNMajor selects the B operand layout:
//   * false (identity): B staged [K, N] row-major, library row_major B
//     fragments (byte-scattered LDS reads; correct for the raw [K, N] weight
//     and every non-gate_up shape).
//   * true (gate_up only): B staged [N, K] n-major with k contiguous (row
//     stride 144 = 128+16, or 80 = 64+16 for the 64-K double buffer),
//     library col_major B fragments -> each lane's elements are k-contiguous
//     bytes instead of byte-scattered columns. This is the bank-safe consumer
//     layout for the B side (the 17.3M PMC bank-conflict source in round 1).
//     The one-time pack in launch_pack_w8a8_weight provides the [N, K] buffer
//     for (k, n) == (6144, 512) only.
template <int kBlockM, int kBlockN, int kStageK, bool kBNMajor = false,
          bool kDoubleBuffer = false, bool kDirectFrag = false,
          bool kSkew8 = false, bool kPayload = false, bool kBPanel = false>
__global__ __launch_bounds__(kBlockM * kBlockN / 1024 * kWaveSize) void
w8a8_dumma_prefill_tiled_kernel(
    const int8_t* __restrict__ x_q,
    const int8_t* __restrict__ weight,
    const float* __restrict__ x_scale,
    const float* __restrict__ weight_scale,
    hip_bfloat16* __restrict__ out,
    int m,
    int n,
    int k) {
  constexpr int kWavesN = kBlockN / 32;
  constexpr int kThreads = kBlockM * kBlockN / 1024 * kWaveSize;
  // 16-byte bank skew: keeps int4 staging stores aligned while spreading the
  // fragment rows across distinct LDS banks. The staging loops live in
  // stage_prefill_tiles below; the strides remain here as the static-asserted
  // alignment contract of every instantiation. kSkew8 (round 11, exact-shape
  // arm only) switches A/B to the down_proj-validated 8-mod-16 strides
  // 88/72; LDS bytes stay A[64,88] + B[64,72] x 2 = 20,480 B/block so the
  // 3-blocks/CU occupancy of the accepted kernel is preserved exactly.
  constexpr int kAStride = kSkew8 ? kStageK + 24 : kStageK + sizeof(int4);
  constexpr int kBStride = kBlockN + sizeof(int4);   // identity [K,N] rows
  constexpr int kBNStride = kSkew8 ? kStageK + 8 : kStageK + sizeof(int4);
  constexpr int kBNInt4 = kBlockN * (kStageK / sizeof(int4));
  // Double buffering uses kBuffers = 2 LDS tile copies; the gate_up shape
  // pairs it with kStageK = 64 so 3 blocks/CU is preserved (see kernel
  // comment above).
  constexpr int kBuffers = kDoubleBuffer ? 2 : 1;
  constexpr int kABytes = kBlockM * kAStride;
  // One shared B tile sized for the larger of the two layouts so an inactive
  // arm never allocates dead LDS (keeps <64,64,128> at 19,456 B total and
  // 3 blocks/CU). kSkew8 implies kBNMajor, so only the n-major arm is live.
  constexpr int kBMaxBytes =
      kSkew8 ? kBlockN * kBNStride
             : (kStageK * kBStride > kBlockN * kBNStride
                    ? kStageK * kBStride
                    : kBlockN * kBNStride);
  static_assert(kSkew8 ? (kAStride % 8 == 0) : (kAStride % sizeof(int4) == 0),
                "A LDS row stride must stay int4 aligned (or 8-byte aligned "
                "under kSkew8)");
  static_assert(kBStride % static_cast<int>(sizeof(int4)) == 0,
                "B LDS row stride must stay int4 aligned");
  static_assert(
      kSkew8 ? (kBNStride % 8 == 0) : (kBNStride % sizeof(int4) == 0),
      "n-major B LDS row stride must stay int4 aligned (or 8-byte aligned "
      "under kSkew8)");
  static_assert(!kSkew8 || kBNMajor,
                "kSkew8 is only defined for the packed n-major B layout");
  static_assert(!kSkew8 || kStageK == 64,
                "kSkew8 strides 88/72 assume kStageK == 64");
  static_assert(kStageK % kTileK == 0,
                "K stage must be a multiple of the DUMMA K unit");
  static_assert(kStageK % static_cast<int>(sizeof(int4)) == 0,
                "K stage must keep int4 staging aligned");
  static_assert(kBNInt4 % kThreads == 0,
                "n-major B staging must divide evenly across threads");
  static_assert(!kPayload || (kBNMajor && kDoubleBuffer && kDirectFrag &&
                              kSkew8 && kStageK == 64 && kBlockM == 64 &&
                              kBlockN == 64),
                "payload arm is specialized to the exact-shape <64,64,64> "
                "double-buffered load_frag8/kSkew8 arm (one int4 per thread "
                "per operand)");
  static_assert(
      !kBPanel || (kBNMajor && kStageK == 64 && kBlockN == 128 && !kPayload),
      "kBPanel is specialized to the down_proj panel pack (n-major 128x64 "
      "per-stage panels, 64-K stages, non-payload pipeline)");

  const int tid = static_cast<int>(threadIdx.x);
  const int wave = tid / kWaveSize;
  const int lane = tid % kWaveSize;
  const int wave_row = wave / kWavesN;
  const int wave_col = wave - wave_row * kWavesN;
  const int m0 = static_cast<int>(blockIdx.y) * kBlockM;
  const int n0 = static_cast<int>(blockIdx.x) * kBlockN;
  const int local_row = wave_row * 32;
  const int local_col = wave_col * 32;

  __shared__ __align__(16) int8_t a_tile[kBuffers * kABytes];
  __shared__ __align__(16) int8_t b_tile[kBuffers * kBMaxBytes];

  DUFragment<matrix_a, kTileM, kTileN, kTileK, signed char, row_major>
      a_frag0, a_frag1;
  DUFragment<accumulator, kTileM, kTileN, kTileK, int>
      acc00, acc01, acc10, acc11;
  du_fill_fragment(acc00, 0);
  du_fill_fragment(acc01, 0);
  du_fill_fragment(acc10, 0);
  du_fill_fragment(acc11, 0);

  if constexpr (kDoubleBuffer) {
    if constexpr (kPayload) {
      // Round 16 lead-2 loop-carried payload pipeline (exact-shape arm
      // only): stage s+1's global loads are issued a FULL iteration early
      // (top of iteration s-1) into a two-int4-per-thread register payload
      // and stay outstanding across the stage-(s-1) fragment reads + MMAC
      // burst and the stage barrier; the vmcnt wait lands at the top of
      // iteration s (first use of the payload registers), immediately
      // before the ds_write2 publish into the idle LDS buffer. The
      // s_barrier fences outstanding vmem, so the wait can no longer sit
      // right after the load issue: the global-load latency overlaps the
      // whole compute segment instead of serializing in front of every
      // stage (the round-11 ISA defect). Same 2 buffers, same one barrier
      // per stage (prologue + 96 = 97/block), same 16 ds_read2_b64 +
      // 8 ds_write2_b64 per block-stage, same staging bytes/addresses
      // (bit-identical int32 order).
      int4 a_payload, b_payload;
      // Prologue: issue the stage-1 global loads first (so they ride behind
      // the stage-0 inline staging), stage k0 = 0 into buffer 0 inline
      // (same load+wait+write path as the accepted kernel), then the first
      // barrier.
      prefetch_stage_payload<kBlockM, kBlockN, kStageK, kBNMajor, kSkew8>(
          x_q, weight, tid, m0, n0, m, k, kStageK, a_payload, b_payload);
      stage_prefill_tiles<kBlockM, kBlockN, kStageK, kBNMajor, kSkew8,
                         kBPanel>(
          x_q, weight, a_tile, b_tile, tid, m0, n0, m, n, k, 0);
      __syncthreads();
      for (int k0 = 0; k0 < k; k0 += kStageK) {
        const int buf = (k0 / kStageK) & 1;
        if (k0 + kStageK < k) {
          // Publish the payload carried from the previous iteration (stage
          // k0+kStageK) into the idle buffer: the vmcnt wait for the loads
          // issued one iteration earlier is emitted here at the first use,
          // after the previous MMAC burst + barrier.
          publish_stage_payload<kBlockM, kBlockN, kStageK, kBNMajor, kSkew8>(
              a_payload, b_payload, a_tile + (buf ^ 1) * kABytes,
              b_tile + (buf ^ 1) * kBMaxBytes, tid);
        }
        if (k0 + 2 * kStageK < k) {
          // Issue stage k0+2*kStageK's global loads into the payload
          // registers (no wait); they stay in flight across the MMAC burst
          // and the barrier below and are waited at the next iteration's
          // publish (RAW).
          prefetch_stage_payload<kBlockM, kBlockN, kStageK, kBNMajor, kSkew8>(
              x_q, weight, tid, m0, n0, m, k, k0 + 2 * kStageK, a_payload,
              b_payload);
        }
        // Each wave computes its 32x32 quadrant: four m16n16k32 DUMMA tiles.
        // k0-outer / kk-inner int32 accumulation order.
        mmac_prefill_stage<kBlockM, kBlockN, kStageK, kBNMajor, kDirectFrag,
                           kSkew8>(
            a_frag0, a_frag1, acc00, acc01, acc10, acc11,
            a_tile + buf * kABytes, b_tile + buf * kBMaxBytes,
            local_row, local_col);
        // One barrier per stage: (a) WAR - all stage-s fragment reads of the
        // buffer just read are retired before it becomes the publish target
        // two stages later; (b) RAW - every thread's stage-(s+1) ds_writes
        // are visible before the next iteration reads them.
        __syncthreads();
      }
    } else {
      // Prologue: stage k0 = 0 into buffer 0, then the pipelined loop.
      stage_prefill_tiles<kBlockM, kBlockN, kStageK, kBNMajor, kSkew8,
                         kBPanel>(
          x_q, weight, a_tile, b_tile, tid, m0, n0, m, n, k, 0);
      __syncthreads();
      for (int k0 = 0; k0 < k; k0 += kStageK) {
        const int buf = (k0 / kStageK) & 1;
        if (k0 + kStageK < k) {
          // Prefetch stage s+1 into the idle buffer: global loads issue here,
          // before the stage-s MMAC burst, and complete (RAW) behind the
          // single barrier below.
          stage_prefill_tiles<kBlockM, kBlockN, kStageK, kBNMajor, kSkew8,
                         kBPanel>(
              x_q, weight, a_tile + (buf ^ 1) * kABytes,
              b_tile + (buf ^ 1) * kBMaxBytes, tid, m0, n0, m, n, k,
              k0 + kStageK);
        }
        // Each wave computes its 32x32 quadrant: four m16n16k32 DUMMA tiles.
        // k0-outer / kk-inner int32 accumulation order.
        mmac_prefill_stage<kBlockM, kBlockN, kStageK, kBNMajor, kDirectFrag,
                           kSkew8>(
            a_frag0, a_frag1, acc00, acc01, acc10, acc11,
            a_tile + buf * kABytes, b_tile + buf * kBMaxBytes,
            local_row, local_col);
        // One barrier per stage: (a) WAR - all stage-s fragment reads of the
        // buffer just read are retired before it becomes the prefetch target
        // two stages later; (b) RAW - every thread's stage-(s+1) ds_writes are
        // visible before the next iteration reads them.
        __syncthreads();
      }
    }
  } else {
    for (int k0 = 0; k0 < k; k0 += kStageK) {
      // Stage A[kBlockM, kStageK] and B into the single LDS buffer
      // (kSkew8 8-mod-16 row strides 88/72 with ds_write2_b64 staging).
      stage_prefill_tiles<kBlockM, kBlockN, kStageK, kBNMajor, kSkew8,
                         kBPanel>(
          x_q, weight, a_tile, b_tile, tid, m0, n0, m, n, k, k0);
      __syncthreads();

      // Each wave computes its 32x32 quadrant: four m16n16k32 DUMMA tiles.
      // k0-outer / kk-inner int32 accumulation order (kDirectFrag: eight
      // load_frag8 ds_read2_b64 fragment reads, zero reassembly VALU).
      mmac_prefill_stage<kBlockM, kBlockN, kStageK, kBNMajor, kDirectFrag,
                         kSkew8>(
          a_frag0, a_frag1, acc00, acc01, acc10, acc11,
          a_tile, b_tile, local_row, local_col);
      // Retire this stage's reads before the next stage's staging writes
      // overwrite the single LDS buffer (WAR hazard across waves).
      __syncthreads();
    }
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

// Generic scalar fallback: one output element per grid-stride iteration with
// exact int32 accumulation. Correct for any (m, n, k), including the small-M
// API shapes (M=2/16) and any unmatched geometry. The weight is the identity
// [K, N] row-major layout for every (k, n) EXCEPT the gate_up weight
// (k, n) == (6144, 512), whose buffer is packed [N, K] n-major; this kernel
// decodes that layout so the paired gate_up tails stay correct.
__global__ __launch_bounds__(kScalarThreads) void
w8a8_gemm_scalar_kernel(
    const int8_t* __restrict__ x_q,
    const int8_t* __restrict__ weight,
    const float* __restrict__ x_scale,
    const float* __restrict__ weight_scale,
    hip_bfloat16* __restrict__ out,
    int m,
    int n,
    int k) {
  // Decode the packed layout when this weight was packed that way:
  // gate_up (k, n) == (6144, 512) and, since round 2 of this session,
  // down_proj (k, n) == (256, 6144) - since round 5 in the PANEL layout
  // [N/128][K/64][128][64] (n-major 64-k-byte rows per panel; byte-exact
  // relation packed[((nt*4+kt)*128+nn)*64+kk64] == raw[(nt*128+nn)*k +
  // kt*64+kk64] with nt = col/128, nn = col%128, kt = kk/64, kk64 = kk%64);
  // identity [K, N] otherwise.
  const bool b_panel = (k == 256 && n == 6144);
  const bool b_nmajor = (k == 6144 && n == 512) || b_panel;
  const int64_t total = static_cast<int64_t>(m) * n;
  const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x +
                     threadIdx.x;
       idx < total; idx += stride) {
    const int row = static_cast<int>(idx / n);
    const int col = static_cast<int>(idx - static_cast<int64_t>(row) * n);
    int32_t acc = 0;
    const int8_t* a_row = x_q + static_cast<int64_t>(row) * k;
    for (int kk = 0; kk < k; ++kk) {
      int64_t b_off;
      if (b_panel) {
        const int nt = col / 128;
        const int nn = col - nt * 128;
        const int kt = kk / 64;
        const int kk64 = kk - kt * 64;
        b_off = (static_cast<int64_t>(nt * 4 + kt) * 128 + nn) * 64 + kk64;
      } else {
        b_off = b_nmajor ? static_cast<int64_t>(col) * k + kk
                         : static_cast<int64_t>(kk) * n + col;
      }
      acc += static_cast<int32_t>(a_row[kk]) *
             static_cast<int32_t>(weight[b_off]);
    }
    const float scaled = static_cast<float>(acc) * x_scale[row] *
                         weight_scale[col];
    out[idx] = __float2bfloat16(scaled);
  }
}

// Bootstrap pack: identity device-to-device copy for every (k, n) pair
// (raw [K, N] layout preserved). Runs outside the timed/CUDA-Graph region.
__global__ __launch_bounds__(kScalarThreads) void
w8a8_pack_identity_kernel(
    const int8_t* __restrict__ raw_weight,
    const float* __restrict__ weight_scale,
    int8_t* __restrict__ packed_weight,
    float* __restrict__ packed_weight_scale,
    int64_t weight_elems,
    int n) {
  const int64_t idx =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < weight_elems) {
    packed_weight[idx] = raw_weight[idx];
  }
  if (idx < n) {
    packed_weight_scale[idx] = weight_scale[idx];
  }
}

// Shape-specific gate_up pack: transpose raw [K, N] -> packed [N, K]
// (n-major rows, k contiguous) for (k, n) == (6144, 512) only. Byte-exact
// index relation: packed[cc * k + kk] == raw[kk * n + cc]. Each thread
// gathers the 16 bytes of one aligned int4 k-block from 16 raw rows (stride
// n) and stores one aligned int4, keeping the packed write side fully
// coalesced (same gather + aligned int4-store pattern as the validated
// o_proj transpose repair). Runs outside the timed/CUDA-Graph region; same
// byte count, same graph-stable buffer.
__global__ __launch_bounds__(kScalarThreads) void
w8a8_pack_gateup_transpose_kernel(
    const int8_t* __restrict__ raw_weight,
    const float* __restrict__ weight_scale,
    int8_t* __restrict__ packed_weight,
    float* __restrict__ packed_weight_scale,
    int k,
    int n) {
  const int k4 = k / static_cast<int>(sizeof(int4));
  const int64_t total4 = static_cast<int64_t>(n) * k4;
  const int64_t stride4 = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t idx4 = static_cast<int64_t>(blockIdx.x) * blockDim.x +
                     threadIdx.x;
       idx4 < total4; idx4 += stride4) {
    const int cc = static_cast<int>(idx4 / k4);
    const int kblk = static_cast<int>(idx4 - static_cast<int64_t>(cc) * k4);
    const int kk = kblk * static_cast<int>(sizeof(int4));
    const int8_t* src = raw_weight + static_cast<int64_t>(kk) * n + cc;
    int4 v;
    char* vp = reinterpret_cast<char*>(&v);
#pragma unroll
    for (int i = 0; i < static_cast<int>(sizeof(int4)); ++i) {
      vp[i] = src[static_cast<int64_t>(i) * n];
    }
    reinterpret_cast<int4*>(packed_weight + static_cast<int64_t>(cc) * k +
                            kk)[0] = v;
  }
  const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x +
                     threadIdx.x;
       idx < n; idx += stride) {
    packed_weight_scale[idx] = weight_scale[idx];
  }
}

// Round 5 (this session): shape-specific down_proj panel pack for
// (k, n) == (256, 6144) only. Raw [K, N] -> panel layout
// [N/128][K/64][128][64]: n-major rows of 64 k-bytes per panel, panels
// ordered (n-tile outer, k-tile inner). Byte-exact index relation (verified
// by simulation):
//   packed[((nt * (k/64) + kt) * 128 + nn) * 64 + kk64] ==
//     raw[(nt * 128 + nn) * k + (kt * 64 + kk64)]
// with nt in [0, n/128), kt in [0, k/64), nn in [0, 128), kk64 in [0, 64).
// Each packed int4 (idx4) decodes as v = idx4 % 4, row_p = idx4 / 4,
// nn = row_p % 128, kt = (row_p / 128) % (k/64), nt = (row_p / 128) / (k/64),
// kk = kt * 64 + v * 16, cc = nt * 128 + nn, and
// packed[idx4*16 .. +16] == raw[(kk + i) * n + cc] for i in 0..15 (the same
// 16-row gather at column cc as the gate_up transpose, stored into a
// different packed offset; each thread stores one aligned int4, so the
// packed write side stays fully coalesced). The panel layout makes every
// per-block B stage tile (128 n x 64 k) ONE contiguous 8 KiB region, so the
// timed kernel's B staging reads 1 KiB per warp instruction contiguously.
// Runs outside the timed/CUDA-Graph region; same byte count, same
// graph-stable buffer, byte-exact vs the raw logical [K, N] weight.
__global__ __launch_bounds__(kScalarThreads) void
w8a8_pack_downproj_panel_kernel(
    const int8_t* __restrict__ raw_weight,
    const float* __restrict__ weight_scale,
    int8_t* __restrict__ packed_weight,
    float* __restrict__ packed_weight_scale,
    int k,
    int n) {
  constexpr int kBlockN = 128;
  constexpr int kStageK = 64;
  const int64_t total4 = (static_cast<int64_t>(n) * k) / 16;
  const int64_t stride4 = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t idx4 = static_cast<int64_t>(blockIdx.x) * blockDim.x +
                     threadIdx.x;
       idx4 < total4; idx4 += stride4) {
    const int v = static_cast<int>(idx4 & 3);
    const int64_t row_p = idx4 >> 2;
    const int nn = static_cast<int>(row_p % kBlockN);
    const int64_t kt_tmp = row_p / kBlockN;  // nt * (k/kStageK) + kt
    const int kt = static_cast<int>(kt_tmp % (k / kStageK));
    const int nt = static_cast<int>(kt_tmp / (k / kStageK));
    const int kk = kt * kStageK + v * 16;
    const int cc = nt * kBlockN + nn;
    const int8_t* src = raw_weight + static_cast<int64_t>(kk) * n + cc;
    int4 val;
    char* vp = reinterpret_cast<char*>(&val);
#pragma unroll
    for (int i = 0; i < static_cast<int>(sizeof(int4)); ++i) {
      vp[i] = src[static_cast<int64_t>(i) * n];
    }
    reinterpret_cast<int4*>(packed_weight + idx4 * 16)[0] = val;
  }
  const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x +
                     threadIdx.x;
       idx < n; idx += stride) {
    packed_weight_scale[idx] = weight_scale[idx];
  }
}

}  // namespace

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
  // No split-K: the provided workspace is not touched. The timed operator
  // performs no allocation, synchronization, packing, or default-stream
  // launch; only the caller-provided out is written.
  (void)workspace;
  (void)workspace_bytes;
  auto* out_bf16 = static_cast<hip_bfloat16*>(out);

  // Native INT8 DUMMA m16n16k32 tiled path for every large-M shape. Both
  // assigned shapes qualify: gate_up (4096, 512, 6144) and down_proj
  // (4096, 6144, 256) both have M >= 128, N % 128 == 0 and K % 128 == 0.
  // Explicit dispatch by launch geometry; unmatched shapes fall through to
  // the scalar path (no shape is specialized away).
  if (m >= kPrefillMinM && (n % kDefaultBlockN == 0) &&
      (k % kPrefillStageK == 0)) {
    // Exact-shape guard for the assigned gate_up shape (round-3 pipeline
    // decision, round-10 load_frag8, round-11 kSkew8, round-16 kPayload):
    // cooperative A+B staging with the n-major [N, K] packed B operand
    // (kBNMajor) consumed as col_major fragments, software-pipelined across
    // K tiles (kDoubleBuffer) with a 64-K double buffer: 64x64 tile, 512
    // blocks, 20,480 B LDS/block, 3 blocks/CU (61,440 B) - occupancy
    // preserved vs the accepted round-2 single buffer (19,456 B x 3 =
    // 58,368 B). One barrier per 64-K stage + prologue (97 for 96 stages).
    // Round 11 stages A/B at the 8-mod-16 strides 88/72 (kSkew8) so the
    // fragment reads are bank-conflict-free (LDS bytes and occupancy
    // unchanged). Round 16 (kPayload) issues the stage-(s+1) global loads a
    // full iteration early into a two-int4-per-thread register payload and
    // publishes them into the idle LDS buffer at the top of iteration s, so
    // the vmcnt wait lands after the stage-s MMAC burst instead of in front
    // of it (LDS bytes, barriers, traffic, occupancy and results unchanged).
    if (m == 4096 && n == 512 && k == 6144) {
      const dim3 grid(n / 64, (m + 63) / 64);
      const dim3 block(4 * kWaveSize);
      hipLaunchKernelGGL(
          HIP_KERNEL_NAME(w8a8_dumma_prefill_tiled_kernel<
                          64, 64, 64, true, true, true, true, true>),
          grid, block, 0, stream,
          a, b, x_scale, weight_scale, out_bf16, m, n, k);
      return;
    }
    // Exact-shape guard for the assigned down_proj shape (round 2 of this
    // session): the generic <64,128,128> identity-layout arm measured
    // 314.72 us median / 315.18 us p90 with PMC lds_bank_conflicts =
    // 7,864,320, ~600 LDS instructions and heavy per-byte fragment
    // reassembly VALU per block-stage (library row_major loaders on the
    // identity [K,N] B layout). The round-2 guard transplanted the file's
    // lineage-validated gate_up machinery: n-major packed B (kBNMajor, pack
    // in launch_pack_w8a8_weight for (k,n) == (256,6144)), 64-K
    // double-buffered K loop (kDoubleBuffer), explicit 8-byte load_frag8
    // fragment reads (kDirectFrag), 8-mod-16 bank-skewed LDS strides
    // A=88/B=72 (kSkew8); accepted at 286.53 us median / 286.64 us p90 with
    // PMC lds_instructions 540,672, conflicts 1,179,648, lds_wait 162,664.
    //
    // Round 3 (this session, pipeline round): compare single and double
    // buffering across K tiles. The (4096, 6144, 256) guard now routes to
    // w8a8_dumma_prefill_tiled_kernel<64, 128, 64, true, false, true, true>
    // - the SINGLE-buffered form of the accepted round-2 arm: same 64-K
    // stage, same kBNMajor/kDirectFrag/kSkew8, same tile/grid/epilogue, but
    // kDoubleBuffer = false, so each stage serializes one global-load round
    // trip (load -> wait -> ds_write2_b64) in front of its MMAC burst with
    // two barriers per stage (8/block vs the accepted 5/block), and the LDS
    // halves from 29,696 to 14,848 B/block (A[64,88] + B[128,72] x 1). The
    // accepted kernel's exact code object shows the double-buffered loop
    // already overlaps stage-(s+1) loads with the stage-s MMAC burst
    // (global_load x2 -> ds_read2_b64 x4 -> v_mmac x8 -> s_waitcnt
    // vmcnt(0) lgkmcnt(0) -> s_barrier), so this round measures whether that
    // overlap is worth the extra 14,848 B of LDS: if the single-buffered
    // median is not strictly below the accepted 286.53 us (with the p90
    // noise guard), double buffering is retained per the pipeline mandate.
    // Results stay bit-identical: stage size, k0-outer/kk-inner int32 order
    // and the element-to-slot fragment mapping are unchanged (only the
    // buffer count and barrier count differ), so mismatch 0 is expected.
    //
    // Round 5 (this session, PACKING round): the (4096, 6144, 256) guard
    // now routes to w8a8_dumma_prefill_tiled_kernel<64, 128, 64, true,
    // false, true, true, false, true> (new kBPanel template flag): the
    // one-time pack for (k, n) == (256, 6144) becomes the panel layout
    // [N/128][K/64][128][64] (n-major 64-k-byte rows per panel, byte-exact
    // vs the raw [K,N] weight, same byte count / graph-stable buffer,
    // out-of-timed/out-of-Graph) so each block's per-stage B tile is one
    // contiguous 8 KiB region and every warp's int4 loads read 1 KiB
    // contiguous (full L2 line utilization) instead of 16 rows x 64 B at
    // 256-B stride (half-used lines). Everything else is byte-identical to
    // the round-3 winner: same single-buffered 64-K stage, same
    // kBNMajor/kDirectFrag/kSkew8, same 64x128 tile / 512 threads / 8 waves,
    // same grid (48,64) = 3072 blocks, same LDS A[64,88] + B[128,72] =
    // 14,848 B/block @ 2 blocks/CU, same ds_write2_b64 staging and
    // load_frag8 fragment reads (LDS bytes and bank behavior untouched),
    // same k0-outer/kk-inner int32 order and element-to-slot fragment
    // mapping -> mismatch 0 / max_abs_error 0.0 expected.
    if (m == 4096 && n == 6144 && k == 256) {
      const dim3 grid(n / 128, (m + 63) / 64);
      const dim3 block(8 * kWaveSize);
      hipLaunchKernelGGL(
          HIP_KERNEL_NAME(w8a8_dumma_prefill_tiled_kernel<
                          64, 128, 64, true, false, true, true, false, true>),
          grid, block, 0, stream,
          a, b, x_scale, weight_scale, out_bf16, m, n, k);
      return;
    }
    if (n % 128 == 0) {
      // 64x128 tile, 8 waves (32x32 quadrant per wave).
      //   down_proj: grid (6144/128, 4096/64) = (48, 64) = 3072 blocks.
      const dim3 grid(n / 128, (m + 63) / 64);
      const dim3 block(8 * kWaveSize);
      hipLaunchKernelGGL(
          HIP_KERNEL_NAME(w8a8_dumma_prefill_tiled_kernel<64, 128, 128>),
          grid, block, 0, stream,
          a, b, x_scale, weight_scale, out_bf16, m, n, k);
      return;
    }
    if (m % 128 == 0) {
      // 128x64 tile, 8 waves.
      const dim3 grid(n / 64, m / 128);
      const dim3 block(8 * kWaveSize);
      hipLaunchKernelGGL(
          HIP_KERNEL_NAME(w8a8_dumma_prefill_tiled_kernel<128, 64, 128>),
          grid, block, 0, stream,
          a, b, x_scale, weight_scale, out_bf16, m, n, k);
      return;
    }
    {
      // 64x64 tile, 4 waves (generic default).
      const dim3 grid(n / 64, (m + 63) / 64);
      const dim3 block(4 * kWaveSize);
      hipLaunchKernelGGL(
          HIP_KERNEL_NAME(w8a8_dumma_prefill_tiled_kernel<64, 64, 128>),
          grid, block, 0, stream,
          a, b, x_scale, weight_scale, out_bf16, m, n, k);
      return;
    }
  }

  // Generic scalar fallback for every unmatched (m, n, k) and M < 128
  // (including the paired M=2/M=16 API shapes with the same (N, K), which
  // decode the packed [N, K] gate_up layout when (k, n) == (6144, 512)).
  const int64_t total = static_cast<int64_t>(m) * n;
  const unsigned blocks = static_cast<unsigned>(
      (total + kScalarThreads - 1) / kScalarThreads);
  hipLaunchKernelGGL(
      w8a8_gemm_scalar_kernel,
      dim3(blocks), dim3(kScalarThreads), 0, stream,
      a, b, x_scale, weight_scale, out_bf16, m, n, k);
}

extern "C" void launch_pack_w8a8_weight(
    const int8_t* raw_weight,
    const float* weight_scale,
    int8_t* packed_weight,
    float* packed_weight_scale,
    int k,
    int n,
    hipStream_t stream) {
  // Assigned gate_up weight (K=6144, N=512): one-time [K,N] -> [N,K]
  // transpose pack (n-major, k contiguous) so the timed GEMM reads B
  // fragments as k-contiguous bytes from the staged LDS tile. Runs outside
  // the timed/CUDA-Graph region; same byte count, same graph-stable buffer,
  // byte-exact vs the raw logical [K,N] weight.
  if (k == 6144 && n == 512) {
    const int64_t total4 = static_cast<int64_t>(n) * (k / 16);
    const unsigned blocks = static_cast<unsigned>(
        (total4 + kScalarThreads - 1) / kScalarThreads);
    hipLaunchKernelGGL(
        w8a8_pack_gateup_transpose_kernel,
        dim3(blocks), dim3(kScalarThreads), 0, stream,
        raw_weight, weight_scale, packed_weight, packed_weight_scale, k, n);
    return;
  }
  // Round 5 (this session): the assigned down_proj weight (K=256, N=6144)
  // gets the PANEL pack [N/128][K/64][128][64] (n-major 64-k-byte rows per
  // panel; each per-block B stage tile is one contiguous 8 KiB region so
  // the timed kernel's B staging reads 1 KiB per warp instruction
  // contiguously). One-time, out-of-timed/out-of-Graph, same byte count,
  // same graph-stable buffer, byte-exact vs the raw [K,N] weight.
  if (k == 256 && n == 6144) {
    const int64_t total4 = (static_cast<int64_t>(n) * k) / 16;
    const unsigned blocks = static_cast<unsigned>(
        (total4 + kScalarThreads - 1) / kScalarThreads);
    hipLaunchKernelGGL(
        w8a8_pack_downproj_panel_kernel,
        dim3(blocks), dim3(kScalarThreads), 0, stream,
        raw_weight, weight_scale, packed_weight, packed_weight_scale, k, n);
    return;
  }
  // Identity device-to-device copy for every other (k, n) pair, including
  // unmatched (K, N). Runs outside the timed/CUDA-Graph region.
  const int64_t weight_elems = static_cast<int64_t>(k) * n;
  const int64_t total = weight_elems > n ? weight_elems : n;
  const unsigned blocks = static_cast<unsigned>(
      (total + kScalarThreads - 1) / kScalarThreads);
  hipLaunchKernelGGL(
      w8a8_pack_identity_kernel,
      dim3(blocks), dim3(kScalarThreads), 0, stream,
      raw_weight, weight_scale, packed_weight, packed_weight_scale,
      weight_elems, n);
}
// @@end
