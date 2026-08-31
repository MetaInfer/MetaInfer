// @@variant shape=glm_tp8_fused_qkv_a_proj_m4096 commit=935c285f78b3c36ce8f49ef13a68b0cf4e8ff4d5 added=2026-08-31
//   median_us=937.6 p90_us=939.5 speedup=73.23 baseline_us=6.866e+04
//   source=glm5-2-dsh-tp8-m4096-1-e6a280a2
// @@variant shape=glm_tp8_fused_qkv_a_proj_m4096
// MetaInfer W8A8 INT8 GEMM for Hygon K500SM_AI / gfx928 (worker_0, GPU 0).
//
// Assigned shape:
//   glm_tp8_fused_qkv_a_proj_m4096 : M=4096, N=2624, K=6144
//
// Operator contract (fixed by the control plane):
//   out[m, n] = bf16( int32_dot(x_q[m, :], raw_weight[:, n])
//                     * x_scale[m] * weight_scale[n] )
//
// Iteration 1 (correctness-first DUMMA bootstrap):
//   * Packing bootstrap: launch_pack_w8a8_weight is the mandated identity
//     device-to-device copy for every (K, N) (raw [K, N] row-major weight is
//     copied unchanged and the fp32 scales are copied unchanged). The GEMM
//     therefore interprets packed_weight as the logical [K, N] row-major
//     layout everywhere. Later Parallel explore rounds may change the pack
//     kernel and the matching GEMM interpretation together.
//   * Iteration 1 dispatched tile <64, 64, 128> (41 x 64 = 2624 blocks,
//     16,384 B LDS/block, 4 blocks/CU, official median 6391.92 us = 20.66
//     logical TOPS). Its exact-source ISA shows the stage loop is
//     LDS-issue-bound, not MMA-bound: per stage per wavefront 136 ds_read
//     (mostly du_load_matrix_sync byte-reassembly ds_read_u8) + 107 waitcnt
//     vs only 32 v_mmac and 2 barriers; 2624 blocks x 96 barriers =
//     251,904 barriers and 2.06 GB of staged global traffic per replay.
//   * The epilogue is the coalesced fragment store: the verified gfx928
//     accumulator mapping (row = lane & 15, col_mod4 = lane >> 4,
//     frag.x[i] -> column col_mod4 + 4*i) is transposed in registers (8
//     shfl_xor + 8 v_cndmask) so each lane owns four contiguous bf16 columns
//     and writes ONE 8-byte store (100% store sector efficiency). The
//     per-element float multiply order (float(dot) * x_scale[m] *
//     weight_scale[n]) and the bf16 rounding are the reference order, so the
//     stored bits are identical to the scalar path.
//   * Generic arms: the <64,64,128> tiled kernel also serves every
//     large-M shape with N % 64 == 0 and K % 128 == 0 (identity layout); a
//     scalar int8/int32 grid-stride kernel is the fallback for every other
//     (m, n, k), including the paired M=2 and M=16 API shapes with the same
//     (N, K), which must never reach the tiled path (m >= 128 guard).
//   * The timed operator (launch_w8a8_gemm) performs no allocation,
//     compilation, autotuning, packing, host/device synchronization, or
//     default-stream launch: it only dispatches kernels on the
//     caller-provided HIP stream and ignores the workspace (split-K capacity
//     is zero for this shape: m*n*4 = 42,990,080 B > 16 MiB budget).
//
// Iteration 1 (mandated 2-D macro-tile baseline: benchmark 64x64 / 64x128 /
// 128x64, pick with code-object resources + occupancy):
//   * Tile comparison (all three instantiations already compiled by the
//     iteration-1 build; VGPR/SGPR/LDS/spills are exact code-object facts
//     from profiles/.../iteration1/current-best-isa/metadata.txt; grid and
//     staged traffic are arithmetic for M=4096, N=2624, K=6144):
//         tile          VGPR SGPR LDS B  blocks/CU waves/CU grid    staged GB
//         <64, 64,128>     57   36 16384      4       16    2624    2.06
//         <64,128,128>     92   36 24576      2        8    1344    1.56*
//         <128,64,128>     88   34 24576      2        8    1312    1.55
//       (*) 64x128 needs an N tail: N=2624 = 41*64 is not a multiple of
//       128, so the n0=2560 block would zero-fill B columns >= 2624 and an
//       epilogue column guard, wasting ~2.4% of all MMAs. No split-K in any
//       arm: 1312 blocks dwarf the 120 CUs.
//   * Iteration 1 dispatched the exact-fit 2-D macro-tile <128, 64, 128>:
//     M=4096 = 32*128 and N=2624 = 41*64 are both exact, so no tail
//     predication is introduced anywhere (the generic A row guard and the
//     epilogue row guard are retained for the generic path). Grid = (N/64) x
//     (M/128) = 41 x 32 = 1312 blocks; 256 threads = 4 wavefronts, each wave
//     owns a 64x32 quadrant = eight m16n16k32 int32 accumulator fragments;
//     single-buffered 128-K LDS stage (A[128,128] + B[128,64] = 24,576
//     B/block) with TWO __syncthreads per stage, K=6144 = 48 stages. Versus
//     the iteration-1 64x64 tile this halves B re-reads (B staged by 32
//     M-blocks instead of 64; staged global traffic 2.06 -> 1.55 GB per
//     replay) and cuts total barriers 251,904 -> 125,952, at the cost of
//     occupancy 4 -> 2 blocks/CU (16 -> 8 resident waves/CU). The int32
//     accumulation order (k0-outer over 128-K stages, kk-inner ascending,
//     same m16n16k32 element-to-slot mapping) is bit-identical to the
//     iteration-1 kernel and the scalar reference.
//   * The <64,128,128> instantiation stays compiled as the next benchmark
//     candidate (needs the N-tail handling described above before dispatch);
//     the generic <64,64,128> arm and the scalar fallback are untouched.
//
// Iteration 2 (mandated operand-reuse round: cooperative A+B LDS staging
// vs direct loads, quantified A/B reuse, vectorized coalesced global loads,
// bank-safe LDS layout):
//   * Operand-reuse accounting for the <128,64,128> macro-tile (M=4096,
//     N=2624, K=6144): per block A = 128 x 6144 = 786,432 B and B = 6144 x
//     64 = 393,216 B of unique bytes; the 41 N-blocks re-read every A
//     row-slab (A global traffic M*K*41 = 1.03 GB/replay, 41x amplification
//     of the 25.2 MB unique A) and the 32 M-blocks re-read every B
//     column-slab (K*N*32 = 0.52 GB/replay, 32x of the 16.1 MB unique B);
//     total staged traffic 1.55 GB/replay. Cooperative staging additionally
//     buys 4x wave-level reuse inside each block (one 16-B vectorized global
//     load per byte, four waves re-read it from LDS), so it is retained: a
//     direct per-wave global fragment path would issue 4x more vmem (4 waves
//     x 48 x 8-B fragment loads per stage = ~9,216 vs ~288 staged
//     16-B loads per wave) and every 8-B fragment load would be a
//     row-strided 16-sector access (vs 4 sectors for the coalesced 16-B
//     staging) -- both directions strictly worse.
//   * The exact-source ISA of the iteration-1 <128,64,128> kernel shows the
//     per-stage per-wavefront consume issues 80 ds_read (64 ds_read_u8 +
//     16 ds_read2_b32) + 99 waitcnt + ~96 byte-reassembly VALU
//     (v_and/v_or3/v_lshlrev) per stage against only 32 v_mmac: the
//     row_major matrix_b loader of du_load_matrix_sync lowers each B
//     fragment to 8 ds_read_u8 at 16-B spacing plus a byte-reassembly chain,
//     and the LDS row strides 128 (A) / 64 (B) alias every row onto one
//     bank phase (32r and 16r mod 32 == 0), matching the PMC
//     lds_bank_conflicts = 274,071,552 (~12.5 per LDS instruction). The
//     kernel is LDS-issue/conflict-bound, not MMA- or HBM-bound.
//   * Focused change (one mechanism, all inside csrc/w8a8_gemm_hip.hip):
//     (1) launch_pack_w8a8_weight now packs the exact (k,n) == (6144,2624)
//     weight once, outside the timed region and out of Graph capture, into
//     the n-major layout packed[n*K + kk] = raw[kk*N + n] (same byte count
//     and buffer, captured addresses unchanged; every other (K, N) keeps
//     the identity copy);
//     (2) the exact-shape guard dispatches a new
//     w8a8_dumma_prefill_packedb_kernel<128,64,128> (grid 41x32 = 1312
//     blocks, 256 threads = 4 waves, same single-buffered 128-K stage with
//     two __syncthreads, same k0-outer/kk-inner int32 order, same epilogue)
//     that stages A row-major and B n-major into bank-safe LDS rows of 136 B
//     (StageK + 8: 8-byte-aligned so every fragment read is one ds_read_b64,
//     34 dwords so the 16 rows of each 16-lane ds_read_b64 phase land on 16
//     distinct bank pairs -> 4-phase conflict-free floor) and replaces both
//     library loaders with the lineage-validated explicit load_frag8 (one
//     aligned 8-B load per lane per fragment: x[0..7] = 8 consecutive bytes
//     of row (lane&15) at k-offset ((lane>>4)*8) for row_major A and the
//     same 8 k-bytes of column n0+(lane&15) for col_major B over the
//     packed [N, K] tile -- identical fragment values, so the m16n16k32
//     v_mmac inputs and the exact int32 accumulation order are bit-identical
//     to the iteration-1 kernel and the scalar reference);
//     (3) the scalar fallback decodes the n-major layout when
//     (k,n) == (6144,2624) so the paired M=2/M=16 API shapes with the same
//     (N, K) stay byte-exact; the generic <64,64,128> identity arm and the
//     <64,128,128>/<128,64,128> sibling instantiations are untouched.
//   * Expected ISA/PMC deltas: ds_read per stage per wavefront 80 -> 24
//     (16 A + 8 B ds_read_b64), waitcnt 99 -> ~24-32, reassembly VALU ~0,
//     lds_bank_conflicts 274M -> conflict-free floor (~1-3M), LDS/block
//     24,576 -> 26,112 B (2 blocks/CU unchanged; no spills).
//
// Iteration 3 (mandated pipeline round: single vs double buffering across K
// tiles; retain double buffering only on ISA/PMC evidence of reduced VMEM
// stalls without harmful LDS or occupancy growth):
//   * Bottleneck evidence from the accepted iteration-2 source (fresh PMC
//     digest 20f3bc9a..., profiles/.../iteration3/pmc.json: profiled
//     2041.095 us, vmem_read 1,574,400, lds_instructions 7,892,992 with
//     lds_wait 4,571,574, lds_bank_conflicts 0, 26,112 B LDS, 2 blocks/CU):
//     the exact-source ISA of w8a8_dumma_prefill_packedb_kernel<128,64,128>
//     shows the stage loop is [global_load_dwordx4 ... -> s_waitcnt
//     vmcnt(0) -> ds_write2_b64 -> s_barrier -> 24x ds_read_b64 -> 32x
//     v_mmac -> s_barrier] -- the FULL global-load round trip (issue, vmcnt
//     wait, LDS flush) is serialized in front of every consume with ZERO
//     compute overlap, and the 14 vmcnt + 57 lgkmcnt waits in the symbol sit
//     on that critical path. The K=6144 grid (1312 blocks) and 2-block
//     co-residency do not hide this per-stage latency.
//   * Focused change (one mechanism, all inside csrc/w8a8_gemm_hip.hip):
//     the exact-shape guard now dispatches a new
//     w8a8_dumma_prefill_packedb_db_kernel<128,64,64> -- SAME macro-tile
//     (grid 41x32 = 1312 blocks, 256 threads = 4 waves, 64x32 quadrant per
//     wave, eight m16n16k32 int32 accumulators), SAME load_frag8 fragment
//     reads, SAME coalesced epilogue -- with the K stage changed from a
//     single-buffered 128-K stage (2 __syncthreads, full VMEM latency
//     exposed) to a DOUBLE-buffered 64-K stage with ONE __syncthreads per
//     stage: the stage-(s+1) global loads are issued into 3 int4 staging
//     VGPR per thread BEFORE the stage-s consume, and the prefetched vectors
//     are flushed into the idle LDS buffer after the consume (the validated
//     M=3072 prefetch order global_load_dwordx4 -> v_mmac -> vmcnt(0) ->
//     ds_write2_b64, here in pure HIP: the vmcnt wait lands on the data
//     dependency before the ds_write, i.e. AFTER the MMAC burst).
//   * Resources: LDS = 2 x (A[128,72] + B[64,72]) = 2 x 13,824 = 27,648
//     B/block (26,112 -> 27,648 B, +5.9%); 2 x 27,648 = 55,296 <= 65,536 B
//     keeps TWO resident blocks/CU = 8 waves/CU, occupancy UNCHANGED (the
//     StageK=128 double-buffer alternative needs 52,224 B/block -> 1
//     block/CU = 4 waves/CU and is rejected up front as harmful occupancy
//     loss). kLdsStride = 72 = 64 + 8: 8-aligned with an odd 8-byte count
//     (72/8 = 9) so the 16 rows of each 16-lane ds_read_b64 phase land on 16
//     distinct bank pairs (18r mod 32, r = 0..15) -- the same conflict-free
//     4-phase floor as iteration 2 -- and 72 % 16 == 8 keeps the two-half
//     ds_write2_b64 staging form. VGPR grows by the 12 staging registers
//     (~72 -> ~84), far below the 128/thread budget for 2 blocks/CU; 0
//     spills expected. Barriers per block: 1 (prologue) + 95 (loop) = 96,
//     the same as iteration 2's 2 x 48 (the final stage is consumed without
//     a trailing barrier).
//   * Correctness: k0-outer over 96 ASCENDING 64-K stages with kk-inner
//     ascending over two m16n16k32 steps gives the identical (k0+kk)
//     k-chunk sequence 0, 32, 64, 96, ... as iteration 2's 48 x 128-K
//     stages, and the element-to-slot fragment mapping is unchanged, so
//     every int32 accumulator is bit-identical (mismatch 0 / max_abs_error
//     0.0 expected).
//   * Falsifiable retention: double buffering is retained only if the exact
//     shape shows median_us < 2276.930923461914 AND p90_us <
//     2540.756378173828 with mismatch_count == 0, max_abs_error == 0.0,
//     graph_capture_passed == true, the exact code object shows 0 spills /
//     LDS == 27,648 B (2 blocks/CU still fits: 55,296 <= 65,536) and the
//     stage ISA shows the prefetch loads issued BEFORE the ds_read_b64 /
//     v_mmac burst with the vmcnt wait after the MMACs (load -> mmac ->
//     vmcnt(0) -> ds_write2_b64), i.e. reduced exposed VMEM stalls; any
//     median/p90 >= current best, any mismatch, any spill, any occupancy
//     loss, or any ISA that still serializes the vmcnt wait in front of the
//     consume falsifies the round and the control plane restores the
//     iteration-2 source (single buffering retained).
//
// Iteration 5 (mandated packing round: test one weight packing/swizzle that
// makes each DUMMA B tile vector-loadable and LDS-bank-safe):
//   * Baseline: the accepted iteration-3 kernel
//     w8a8_dumma_prefill_packedb_db_kernel<128,64,64> (official median
//     1070.23 us = 123.4 logical TOPS; fresh PMC digest 47542b38...,
//     profiles/.../iteration4/pmc.json of the current source: profiled
//     1101.96 us, grid 1312, 96 arch_vgpr / 27,648 B LDS, lds_instructions
//     4,901,632 = 9.7 per wave-stage (6 merged ds_read2_b64 + 3
//     ds_write2_b64), lds_wait 3,701,906, lds_bank_conflicts 12,091,392 =
//     6 x 4 x 4 x 96 x 1312 EXACTLY -- one conflict per 16-lane phase of
//     every merged ds_read2_b64, the inherent 2-cycle alias of the +16-row
//     (16 x 72 = 1152 B = 9 x 128 B) second-fragment offset, NOT a
//     layout defect (the iteration-2 unmerged ds_read_b64 control had 0
//     conflicts at the same 72/136-B strides), vmem_read 1,574,400, L2 hit
//     79.3% with 5,155,404 misses, TCC read 269.3 MB/replay). The kernel is
//     LDS-latency-bound, not HBM-bound; the global B operand is the
//     remaining layout lever.
//   * Mechanism: replace the iteration-2 n-major pack
//     packed[n*K + kk] (each 64-K B tile = 64 column chunks of 64 B at
//     6144-B stride -> 64 x 128-B lines touched per tile, 50% line
//     utilization, scattered L2 footprint) with a B-PANEL pack
//     packed[(n>>6)*(k*64) + (kk>>4)*1024 + (n&63)*16 + (kk&15)] =
//     raw[kk*n + n] (41 panels of [K, 64] int8, 16-byte k-runs per column,
//     byte-exact bijection verified by simulation): every [64,64] B tile of
//     a 64-K stage is then FOUR CONTIGUOUS 1-KiB slabs (one per 16-k-run),
//     each staging vector is a 16-byte-aligned contiguous k-run, and a wave
//     reads 4 x 256 B of one panel region per stage -- the tile is
//     vector-loadable as one coalesced stream with 8x fewer 128-B lines
//     touched per tile (32 vs 256) and full line utilization, so L2 tag
//     pressure and B HBM fetch traffic drop without any change in LDS
//     layout or instruction count.
//   * Kernel-side changes are ONLY the two B global-load sites (prologue +
//     steady-state prefetch) of w8a8_dumma_prefill_packedb_db_kernel
//     (dispatched solely by the exact-shape guard; static_assert BN == 64):
//     vector (n_row, k16) is loaded from byte
//     (n0>>6)*(k*64) + ((k0>>4)+k16)*1024 + n_row*16 instead of
//     (n0+n_row)*k + k0 + k16*16. LDS staging layout (b_tile[64][72],
//     two-half ds_write2_b64), the col_major load_frag8 consumes, the
//     k0-outer/kk-inner int32 accumulation order, the coalesced epilogue,
//     the grid (41x32 = 1312 blocks), threads (256 = 4 waves), occupancy
//     (2 blocks/CU, LDS 27,648 B) and VGPR budget are ALL unchanged, so the
//     round is a pure packing/swizzle A/B.
//   * Correctness: the same weight elements reach the same fragment slots
//     in the same ascending k-chunk sequence, so every int32 accumulator is
//     bit-identical (mismatch 0 / max_abs_error 0.0 expected); the scalar
//     fallback decodes the panel layout when (k,n) == (6144,2624) (ascending
//     16-byte k-runs keep the exact int32 order for the paired M=2/M=16 API
//     shapes); the generic identity arms and sibling kernels are untouched.
//   * Packing stays outside the timed region and out of Graph capture
//     (launch_pack_w8a8_weight, same byte count k*n = 16,121,856 and same
//     buffer -> graph-stable captured addresses unchanged; pack relation
//     verified by a byte-exact index simulation before dispatch).
//   * Predicted deltas: vmem_read_instructions unchanged (~1,574,400), LDS
//     counters unchanged (conflicts stay at the 12,091,392 merged-read alias
//     floor), L2 misses < 5,155,404 with hit rate UP, TCC read bytes <
//     269,329,728 (B line fetches halve), median ~1000-1070 us. Falsifiable:
//     retained only if median_us < 1070.2328491210938 AND p90_us <
//     1081.9308471679688 with mismatch_count == 0, max_abs_error == 0.0,
//     graph_capture_passed == true, the exact code object shows 0 spills /
//     LDS == 27,648 B / VGPR <= 128 (2 blocks/CU), and PMC confirms L2
//     misses and TCC read bytes strictly below the iteration-4 values with
//     vmem_read/lds/conflicts not growing; any median/p90 >= current best,
//     any mismatch, any spill, any occupancy loss, or any L2/HBM regression
//     falsifies the pack and the control plane restores the iteration-3
//     source.
//
// Include order is fixed by the control plane: hip_runtime, hip_bfloat16,
// then du_mma (this DTK's du_mma.h is not self-contained otherwise).

#include <hip/hip_runtime.h>
#include <hip/hip_bfloat16.h>
#include <du_mma.h>

#include <cstdint>

namespace {

constexpr int kWaveSize = 64;  // gfx928 native wavefront
constexpr int kTileM = 16;
constexpr int kTileN = 16;
constexpr int kTileK = 32;  // INT8 DUMMA unit: m16n16k32, int32 accumulation
// 4 wavefronts; must remain a multiple of the gfx928 wavefront size (64).
constexpr int kThreadsPerBlock = 256;

// Exact assigned shape (K, N) after TP=8 partitioning.
constexpr int kExactN = 2624;
constexpr int kExactK = 6144;

using namespace du::dumma;

// Coalesced accumulator epilogue for gfx928 INT8 m16n16k32 (verified against
// du_store_matrix_sync in the accepted worker-29 lineage): the accumulator
// lane mapping is row = lane & 15, col_mod4 = lane >> 4, frag.x[i] maps to
// columns col_mod4 + 4*i. A register 4x4 transpose (two 2x2 steps with
// shfl_xor 16 then 32, one v_cndmask per element per step) re-routes the
// int32 values so lane (r, c4) owns the four CONTIGUOUS columns
// base_col + 4*c4 .. +3; they are scaled (float(dot) * x_scale[row] *
// weight_scale[col], the exact reference multiply order), converted to bf16,
// packed and stored with ONE 8-byte store per lane. The row >= m guard is
// wavefront-uniform (lane & 15 cycles the same 16 rows in every 16-lane
// group), so the shuffles never mix active and inactive lanes. base_col is a
// multiple of 64 for the dispatched tiles, so the float4 weight_scale load
// (col0 % 4 == 0) and the 8-byte store (n even -> out offset a multiple of
// 8) are aligned.
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
  *reinterpret_cast<uint64_t*>(out + static_cast<int64_t>(row) * n + col0) =
      packed;
}

// Simple native INT8 DUMMA m16n16k32 tiled prefill kernel (identity [K, N]
// weight layout; launch_pack_w8a8_weight is the identity bootstrap, so
// packed_weight == raw logical weight). Template parameters: BM x BN output
// tile per block, StageK K rows per single-buffered LDS stage. 256 threads =
// 4 wavefronts; each wave owns a (BM/2) x (BN/2) quadrant of m16n16k32
// int32 accumulator fragments. Grid = (N/BN) x (ceil(M/BM)).
//
// K is staged cooperatively in a SINGLE LDS buffer at StageK granularity
// with TWO __syncthreads per stage (one after the staging stores, one after
// all waves consumed the buffer):
//   * A[BM, StageK] is staged row-major with flat 16-byte vectors (each
//     thread strides over kAVectors = BM*StageK/16 vectors); rows past M are
//     zero-filled and masked in the epilogue.
//   * B[StageK, BN] is staged row-major from the logical [K, N] weight with
//     flat 16-byte vectors (each thread strides over kBVectors =
//     StageK*BN/16 vectors). The dispatch guarantees n % BN == 0 and
//     k % StageK == 0, so every global vector is aligned and in-bounds.
//   * Fragment loads use the library du_load_matrix_sync (row_major for both
//     operands) and accumulate with du_mma_sync in ascending k order --
//     k0-outer over stages, kk-inner over kTileK steps -- preserving the
//     exact int32 accumulation order of the scalar reference.
//   * The epilogue is the coalesced fragment store.
//
// Bootstrap keeps this direct/single-buffered by design (no double buffering,
// no split-K, no raw asm); later rounds tune tile, stage, and packing.
template <int BM, int BN, int StageK>
__global__ __launch_bounds__(kThreadsPerBlock) void
w8a8_dumma_prefill_tiled_kernel(
    const int8_t* __restrict__ x_q,       // [M, K] row-major
    const int8_t* __restrict__ weight,    // [K, N] row-major (identity pack)
    const float* __restrict__ x_scale,    // [M]
    const float* __restrict__ weight_scale,  // [N]
    hip_bfloat16* __restrict__ out,       // [M, N]
    int m,
    int n,
    int k) {
  constexpr int kWaveM16 = BM / 32;
  constexpr int kWaveN16 = BN / 32;
  static_assert(BM % 32 == 0 && BN % 32 == 0,
                "block tile must be a multiple of the wave quadrant");
  static_assert(kThreadsPerBlock % kWaveSize == 0,
                "blockDim must be a multiple of the gfx928 wavefront size");

  const int tid = static_cast<int>(threadIdx.x);
  const int wave = tid / kWaveSize;
  const int lane = tid % kWaveSize;
  const int wave_row = wave >> 1;  // 0 or 1
  const int wave_col = wave & 1;   // 0 or 1
  const int m0 = static_cast<int>(blockIdx.y) * BM;
  const int n0 = static_cast<int>(blockIdx.x) * BN;

  __shared__ __align__(16) int8_t a_tile[BM * StageK];
  __shared__ __align__(16) int8_t b_tile[StageK * BN];

  DUFragment<matrix_a, kTileM, kTileN, kTileK, signed char, row_major>
      a_frag[kWaveM16];
  DUFragment<matrix_b, kTileM, kTileN, kTileK, signed char, row_major>
      b_frag[kWaveN16];
  DUFragment<accumulator, kTileM, kTileN, kTileK, int>
      acc[kWaveM16][kWaveN16];
#pragma unroll
  for (int j = 0; j < kWaveM16; ++j) {
#pragma unroll
    for (int i = 0; i < kWaveN16; ++i) {
      du_fill_fragment(acc[j][i], 0);
    }
  }

  // Flat 16-byte staging vectors per stage (compile-time; the dispatch
  // guarantees k % StageK == 0 and n % BN == 0).
  constexpr int kAVectors = BM * StageK / static_cast<int>(sizeof(int4));
  constexpr int kBVectors = StageK * BN / static_cast<int>(sizeof(int4));

  for (int k0 = 0; k0 < k; k0 += StageK) {
    // Stage A[BM, StageK]: thread `vec` owns the 16-byte chunk at flat byte
    // offset vec*16 (row = offset / StageK, k-run = offset % StageK).
    for (int vec = tid; vec < kAVectors; vec += kThreadsPerBlock) {
      const int byte_off = vec * static_cast<int>(sizeof(int4));
      const int local_row = byte_off / StageK;
      const int kk = byte_off - local_row * StageK;
      const int g_row = m0 + local_row;
      reinterpret_cast<int4*>(a_tile)[vec] =
          (g_row < m)
              ? *reinterpret_cast<const int4*>(
                    x_q + static_cast<int64_t>(g_row) * k + k0 + kk)
              : int4{0, 0, 0, 0};
    }
    // Stage B[StageK, BN] from the logical [K, N] weight: thread `vec` owns
    // the 16-byte chunk at (kk = offset / BN, col = offset % BN).
    for (int vec = tid; vec < kBVectors; vec += kThreadsPerBlock) {
      const int byte_off = vec * static_cast<int>(sizeof(int4));
      const int kk = byte_off / BN;
      const int col = byte_off - kk * BN;
      reinterpret_cast<int4*>(b_tile)[vec] =
          *reinterpret_cast<const int4*>(
              weight + static_cast<int64_t>(k0 + kk) * n + n0 + col);
    }
    // Make the staged tiles visible to every wavefront.
    __syncthreads();

    // Consume the stage: each wave accumulates its (BM/2) x (BN/2) quadrant
    // with kk-ascending m16n16k32 MMAs.
    const int local_row = wave_row * (BM / 2);
    const int local_col = wave_col * (BN / 2);
#pragma unroll
    for (int kk = 0; kk < StageK; kk += kTileK) {
#pragma unroll
      for (int j = 0; j < kWaveM16; ++j) {
        du_load_matrix_sync(
            a_frag[j], a_tile + (local_row + j * kTileM) * StageK + kk,
            StageK);
      }
#pragma unroll
      for (int i = 0; i < kWaveN16; ++i) {
        du_load_matrix_sync(
            b_frag[i], b_tile + kk * BN + local_col + i * kTileN, BN);
      }
#pragma unroll
      for (int j = 0; j < kWaveM16; ++j) {
#pragma unroll
        for (int i = 0; i < kWaveN16; ++i) {
          du_mma_sync(acc[j][i], a_frag[j], b_frag[i], acc[j][i]);
        }
      }
    }
    // Single buffer: every wavefront must finish reading LDS before the next
    // stage's stores overwrite it.
    __syncthreads();
  }

  const int base_row = m0 + wave_row * (BM / 2);
  const int base_col = n0 + wave_col * (BN / 2);
#pragma unroll
  for (int j = 0; j < kWaveM16; ++j) {
#pragma unroll
    for (int i = 0; i < kWaveN16; ++i) {
      store_prefill_fragment_coalesced(
          acc[j][i], x_scale, weight_scale, out, m, n,
          base_row + j * kTileM, base_col + i * kTileN, lane);
    }
  }
}

// Explicit 8-byte int8 fragment loader (lineage-validated load_frag8): for
// both the row_major matrix_a and the col_major matrix_b m16n16k32
// fragments, lane l -> row (l & 15), k-quarter (l >> 4) holds the EIGHT
// CONTIGUOUS bytes at p[row*ldm + (l>>4)*8 .. +7]. Loading the 8 bytes as
// one int64 forces the ds_read_b64 form (16-lane phases, distinct bank
// pairs per group -> 4-phase conflict-free floor with an odd 8-byte LDS row
// stride), identical fragment values to the library loader, so the v_mmac
// inputs and the exact int32 accumulation order are unchanged.
template <typename Frag>
__device__ __forceinline__ void load_frag8(Frag& f, const int8_t* p,
                                           unsigned ldm) {
  const unsigned row = __lane_id() & 0xf;
  const unsigned kq = __lane_id() >> 4;
  const int64_t v =
      *reinterpret_cast<const int64_t*>(p + row * ldm + (kq << 3));
  reinterpret_cast<int64_t*>(f.x)[0] = v;
}

// Iteration-2 exact-shape packed-B prefill kernel (dispatched only for
// (m >= 128, n == 2624, k == 6144), where launch_pack_w8a8_weight produced
// the n-major packed_weight[n*K + kk] = raw[kk*N + n] layout). Same
// geometry as w8a8_dumma_prefill_tiled_kernel<128,64,128> (grid (N/64) x
// (M/128), 256 threads = 4 wavefronts, 64x32 quadrant per wave, eight
// m16n16k32 int32 accumulators, single-buffered 128-K LDS stage with TWO
// __syncthreads per stage, same k0-outer/kk-inner int32 accumulation order,
// same coalesced fragment epilogue). The two differences, both inside the
// operand path:
//   * B is staged n-major (b_tile[n][kk] from packed [N, K]) so col_major
//     B fragments read 8 contiguous K bytes per lane;
//   * A and B LDS row strides are 136 B (StageK + 8): 8-byte aligned for
//     the ds_read_b64 fragment loads (odd 8-byte stride count -> the 16
//     rows of each 16-lane phase land on 16 distinct bank pairs, 34r mod 32
//     = 2r mod 32, r = 0..15 -> all 16 even dword starts, conflict-free
//     4-phase floor) and 136 % 16 == 8 so staging stores are the two-half
//     int64 ds_write2_b64 form (single ds_write_b128 would be misaligned on
//     odd rows).
// Both fragment loaders are the explicit load_frag8 above; the staged bytes
// are the same matrix elements in the same fragment slots as the library
// loaders produced from the identity layout, so every v_mmac input and the
// exact int32 accumulation (k0-outer over 48 stages, kk-inner ascending
// over four m16n16k32 steps) are bit-identical to the accepted iteration-1
// kernel and to the scalar reference.
template <int BM, int BN, int StageK>
__global__ __launch_bounds__(kThreadsPerBlock) void
w8a8_dumma_prefill_packedb_kernel(
    const int8_t* __restrict__ x_q,       // [M, K] row-major
    const int8_t* __restrict__ packed_w,  // [N, K] n-major (exact shape)
    const float* __restrict__ x_scale,    // [M]
    const float* __restrict__ weight_scale,  // [N]
    hip_bfloat16* __restrict__ out,       // [M, N]
    int m,
    int n,
    int k) {
  constexpr int kWaveM16 = BM / 32;
  constexpr int kWaveN16 = BN / 32;
  static_assert(BM % 32 == 0 && BN % 32 == 0,
                "block tile must be a multiple of the wave quadrant");
  static_assert(kThreadsPerBlock % kWaveSize == 0,
                "blockDim must be a multiple of the gfx928 wavefront size");
  // Bank-safe LDS row stride: StageK + 8 bytes = 34 dwords. 8-aligned with
  // an odd 8-byte count -> every lane's 8-byte fragment is ds_read_b64 at
  // the 4-phase conflict-free floor (see the kernel comment above).
  constexpr int kLdsStride = StageK + 8;
  static_assert(kLdsStride % 8 == 0 && (kLdsStride / 8) % 2 == 1,
                "LDS row stride must be 8-aligned with an odd 8-byte count");
  static_assert(kLdsStride % 16 == 8,
                "LDS row stride must be 8 mod 16 for the two-half staging");

  const int tid = static_cast<int>(threadIdx.x);
  const int wave = tid / kWaveSize;
  const int lane = tid % kWaveSize;
  const int wave_row = wave >> 1;  // 0 or 1
  const int wave_col = wave & 1;   // 0 or 1
  const int m0 = static_cast<int>(blockIdx.y) * BM;
  const int n0 = static_cast<int>(blockIdx.x) * BN;

  __shared__ __align__(16) int8_t a_tile[BM * kLdsStride];
  __shared__ __align__(16) int8_t b_tile[BN * kLdsStride];

  DUFragment<matrix_a, kTileM, kTileN, kTileK, signed char, row_major>
      a_frag[kWaveM16];
  DUFragment<matrix_b, kTileM, kTileN, kTileK, signed char, col_major>
      b_frag[kWaveN16];
  DUFragment<accumulator, kTileM, kTileN, kTileK, int>
      acc[kWaveM16][kWaveN16];
#pragma unroll
  for (int j = 0; j < kWaveM16; ++j) {
#pragma unroll
    for (int i = 0; i < kWaveN16; ++i) {
      du_fill_fragment(acc[j][i], 0);
    }
  }

  // Flat 16-byte staging vectors per stage (dispatch guarantees
  // k % StageK == 0 and n % BN == 0). 8 vectors per 128-K row.
  constexpr int kAVectors = BM * StageK / static_cast<int>(sizeof(int4));
  constexpr int kBVectors = StageK * BN / static_cast<int>(sizeof(int4));
  constexpr int kVecsPerRow = StageK / static_cast<int>(sizeof(int4));

  for (int k0 = 0; k0 < k; k0 += StageK) {
    // Stage A[BM, StageK] row-major with row stride kLdsStride: thread
    // `vec` owns the 16-byte chunk at (local_row, k16); rows past M are
    // zero-filled and masked in the epilogue.
    for (int vec = tid; vec < kAVectors; vec += kThreadsPerBlock) {
      const int local_row = vec / kVecsPerRow;
      const int k16 = vec - local_row * kVecsPerRow;
      const int g_row = m0 + local_row;
      int4 v{0, 0, 0, 0};
      if (g_row < m) {
        v = *reinterpret_cast<const int4*>(
            x_q + static_cast<int64_t>(g_row) * k + k0 + k16 * 16);
      }
      // Two 8-byte halves (stride % 16 == 8 -> odd rows are not 16-byte
      // aligned, so a single int4 store is illegal; the int64 pair lowers
      // to one ds_write2_b64 per thread).
      const int64_t* src64 = reinterpret_cast<const int64_t*>(&v);
      int64_t* dst64 = reinterpret_cast<int64_t*>(
          a_tile + local_row * kLdsStride + k16 * 16);
      dst64[0] = src64[0];
      dst64[1] = src64[1];
    }
    // Stage B[BN, StageK] n-major from the packed [N, K] weight: thread
    // `vec` owns the 16-byte chunk at (n_row, k16); the K axis is always
    // in range.
    for (int vec = tid; vec < kBVectors; vec += kThreadsPerBlock) {
      const int n_row = vec / kVecsPerRow;
      const int k16 = vec - n_row * kVecsPerRow;
      const int4 v = *reinterpret_cast<const int4*>(
          packed_w + static_cast<int64_t>(n0 + n_row) * k + k0 + k16 * 16);
      const int64_t* src64 = reinterpret_cast<const int64_t*>(&v);
      int64_t* dst64 = reinterpret_cast<int64_t*>(
          b_tile + n_row * kLdsStride + k16 * 16);
      dst64[0] = src64[0];
      dst64[1] = src64[1];
    }
    // Make the staged tiles visible to every wavefront.
    __syncthreads();

    // Consume the stage: each wave accumulates its (BM/2) x (BN/2) quadrant
    // with kk-ascending m16n16k32 MMAs; one ds_read_b64 per fragment.
    const int local_row = wave_row * (BM / 2);
    const int local_col = wave_col * (BN / 2);
#pragma unroll
    for (int kk = 0; kk < StageK; kk += kTileK) {
#pragma unroll
      for (int j = 0; j < kWaveM16; ++j) {
        load_frag8(
            a_frag[j],
            a_tile + (local_row + j * kTileM) * kLdsStride + kk,
            kLdsStride);
      }
#pragma unroll
      for (int i = 0; i < kWaveN16; ++i) {
        load_frag8(
            b_frag[i],
            b_tile + (local_col + i * kTileN) * kLdsStride + kk,
            kLdsStride);
      }
#pragma unroll
      for (int j = 0; j < kWaveM16; ++j) {
#pragma unroll
        for (int i = 0; i < kWaveN16; ++i) {
          du_mma_sync(acc[j][i], a_frag[j], b_frag[i], acc[j][i]);
        }
      }
    }
    // Single buffer: every wavefront must finish reading LDS before the
    // next stage's stores overwrite it.
    __syncthreads();
  }

  const int base_row = m0 + wave_row * (BM / 2);
  const int base_col = n0 + wave_col * (BN / 2);
#pragma unroll
  for (int j = 0; j < kWaveM16; ++j) {
#pragma unroll
    for (int i = 0; i < kWaveN16; ++i) {
      store_prefill_fragment_coalesced(
          acc[j][i], x_scale, weight_scale, out, m, n,
          base_row + j * kTileM, base_col + i * kTileN, lane);
    }
  }
}

// Iteration-3 double-buffered packed-B prefill kernel (dispatched only for
// (m >= 128, n == 2624, k == 6144)): SAME macro-tile as the accepted
// iteration-2 w8a8_dumma_prefill_packedb_kernel<128,64,128> (grid (N/64) x
// (M/128) = 41x32 = 1312 blocks, 256 threads = 4 wavefronts, 64x32 quadrant
// per wave, eight m16n16k32 int32 accumulators, same load_frag8 fragment
// reads, same bank-safe LDS row strides, same coalesced epilogue). The ONLY
// change is the K pipeline (see the file-header iteration-3 section);
// iteration 5 additionally reads the packed B operand from the B-panel
// layout (see the file-header iteration-5 section and
// w8a8_pack_panel_i8_kernel) -- the LDS staging layout, fragment consumes,
// and accumulation order are untouched:
//   * The single-buffered 128-K stage (two __syncthreads; the exact-source
//     ISA serializes global_load_dwordx4 -> s_waitcnt vmcnt(0) ->
//     ds_write2_b64 -> s_barrier -> consume with no compute overlapping the
//     VMEM round trip) becomes a 64-K DOUBLE-buffered stage with ONE
//     __syncthreads per stage: the stage-(s+1) global loads are issued into
//     3 int4 staging VGPR per thread BEFORE the stage-s consume, and the
//     prefetched vectors are flushed into the idle buffer after the consume,
//     so the VMEM latency of stage s+1 overlaps the LDS/MMAC work of stage s
//     (validated M=3072 prefetch order, in pure HIP: the vmcnt wait lands on
//     the register data dependency right before the ds_write2_b64, i.e.
//     after the v_mmac burst).
//   * LDS = 2 x (A[128,72] + B[64,72]) = 2 x 13,824 = 27,648 B/block
//     (26,112 -> 27,648 B, +5.9%); 2 x 27,648 = 55,296 <= 65,536 B keeps
//     TWO resident blocks/CU = 8 waves/CU, occupancy unchanged. kLdsStride =
//     72 = StageK + 8 (8-aligned, odd 8-byte count 9 -> the 16 rows of each
//     16-lane ds_read_b64 phase land on 16 distinct bank pairs, 18r mod 32,
//     r = 0..15 -> the same 4-phase conflict-free floor; 72 % 16 == 8 keeps
//     the two-half ds_write2_b64 staging form). 12 extra VGPR for the
//     staging registers (~72 -> ~84, still <= 128 for 2 blocks/CU).
//   * k0-outer over 96 ASCENDING 64-K stages with kk-inner ascending over
//     two m16n16k32 steps: the (k0+kk) k-chunk sequence 0, 32, 64, 96, ...
//     is identical to iteration 2 (48 x 128-K stages), so every int32
//     accumulator is bit-identical (mismatch 0 / max_abs_error 0.0).
//   * Barriers per block: 1 (prologue) + 95 (loop) = 96, the same as
//     iteration 2's 2 x 48; the final stage is consumed without a trailing
//     barrier.
template <int BM, int BN, int StageK>
__global__ __launch_bounds__(kThreadsPerBlock) void
w8a8_dumma_prefill_packedb_db_kernel(
    const int8_t* __restrict__ x_q,       // [M, K] row-major
    const int8_t* __restrict__ packed_w,  // [N/64][K][64] B panels (exact shape, iter 5)
    const float* __restrict__ x_scale,    // [M]
    const float* __restrict__ weight_scale,  // [N]
    hip_bfloat16* __restrict__ out,       // [M, N]
    int m,
    int n,
    int k) {
  constexpr int kWaveM16 = BM / 32;
  constexpr int kWaveN16 = BN / 32;
  static_assert(BM % 32 == 0 && BN % 32 == 0,
                "block tile must be a multiple of the wave quadrant");
  static_assert(kThreadsPerBlock % kWaveSize == 0,
                "blockDim must be a multiple of the gfx928 wavefront size");
  // Bank-safe LDS row stride: StageK + 8 bytes = 18 dwords for StageK = 64.
  constexpr int kLdsStride = StageK + 8;
  static_assert(kLdsStride % 8 == 0 && (kLdsStride / 8) % 2 == 1,
                "LDS row stride must be 8-aligned with an odd 8-byte count");
  static_assert(kLdsStride % 16 == 8,
                "LDS row stride must be 8 mod 16 for the two-half staging");
  // Iteration 5: the packed B weight is the panel layout (see
  // w8a8_pack_panel_i8_kernel), which is defined for BN == 64 columns per
  // panel and 16-byte k-runs (StageK % 16 == 0). Only <128,64,64> is ever
  // dispatched by the exact-shape guard, so the panel index math below is
  // exact for every dispatch.
  static_assert(BN == 64 && StageK % 16 == 0,
                "panel B pack requires BN == 64 and StageK % 16 == 0");

  const int tid = static_cast<int>(threadIdx.x);
  const int wave = tid / kWaveSize;
  const int lane = tid % kWaveSize;
  const int wave_row = wave >> 1;  // 0 or 1
  const int wave_col = wave & 1;   // 0 or 1
  const int m0 = static_cast<int>(blockIdx.y) * BM;
  const int n0 = static_cast<int>(blockIdx.x) * BN;
  const int kStages = k / StageK;  // dispatch guarantees k % StageK == 0

  __shared__ __align__(16) int8_t a_tile[2][BM * kLdsStride];
  __shared__ __align__(16) int8_t b_tile[2][BN * kLdsStride];

  DUFragment<matrix_a, kTileM, kTileN, kTileK, signed char, row_major>
      a_frag[kWaveM16];
  DUFragment<matrix_b, kTileM, kTileN, kTileK, signed char, col_major>
      b_frag[kWaveN16];
  DUFragment<accumulator, kTileM, kTileN, kTileK, int>
      acc[kWaveM16][kWaveN16];
#pragma unroll
  for (int j = 0; j < kWaveM16; ++j) {
#pragma unroll
    for (int i = 0; i < kWaveN16; ++i) {
      du_fill_fragment(acc[j][i], 0);
    }
  }

  // Flat 16-byte staging vectors per stage. For the dispatched <128,64,64>
  // instantiation the 512 A vectors and 256 B vectors divide evenly over the
  // 256 threads (2 + 1 per thread), so the prefetch payload is 3 int4
  // staging registers held live across the consume.
  constexpr int kVecsPerRow = StageK / static_cast<int>(sizeof(int4));
  constexpr int kAVectors = BM * StageK / static_cast<int>(sizeof(int4));
  constexpr int kBVectors = StageK * BN / static_cast<int>(sizeof(int4));
  constexpr int kAVecPerThread = kAVectors / kThreadsPerBlock;
  constexpr int kBVecPerThread = kBVectors / kThreadsPerBlock;
  static_assert(kAVectors % kThreadsPerBlock == 0 &&
                    kBVectors % kThreadsPerBlock == 0,
                "staging vectors must divide evenly across the block");

  int4 a_pre[kAVecPerThread];
  int4 b_pre[kBVecPerThread];

  // ---- Prologue: issue the stage-0 global loads, flush them into buffer 0
  // and make the tile visible to every wavefront.
  {
    const int k0 = 0;
#pragma unroll
    for (int v = 0; v < kAVecPerThread; ++v) {
      const int vec = tid + v * kThreadsPerBlock;
      const int local_row = vec / kVecsPerRow;
      const int k16 = vec - local_row * kVecsPerRow;
      const int g_row = m0 + local_row;
      a_pre[v] = (g_row < m)
                     ? *reinterpret_cast<const int4*>(
                           x_q + static_cast<int64_t>(g_row) * k + k0 +
                           k16 * 16)
                     : int4{0, 0, 0, 0};
    }
#pragma unroll
    for (int v = 0; v < kBVecPerThread; ++v) {
      const int vec = tid + v * kThreadsPerBlock;
      const int n_row = vec / kVecsPerRow;
      const int k16 = vec - n_row * kVecsPerRow;
      // Iteration 5: B is read from the panel-packed layout; the [64, 64]
      // tile is 4 contiguous 1-KiB slabs (one per 16-k-run) and vector
      // (n_row, k16) sits at byte (n0>>6)*(k*64) + ((k0>>4)+k16)*1024 +
      // n_row*16 (16-byte aligned, contiguous k-run of column n0+n_row).
      const int64_t panel_off =
          static_cast<int64_t>(n0 >> 6) * (static_cast<int64_t>(k) * 64) +
          (static_cast<int64_t>(k0 >> 4) + k16) * 1024 + n_row * 16;
      b_pre[v] = *reinterpret_cast<const int4*>(packed_w + panel_off);
    }
#pragma unroll
    for (int v = 0; v < kAVecPerThread; ++v) {
      const int vec = tid + v * kThreadsPerBlock;
      const int local_row = vec / kVecsPerRow;
      const int k16 = vec - local_row * kVecsPerRow;
      const int64_t* src64 = reinterpret_cast<const int64_t*>(&a_pre[v]);
      int64_t* dst64 = reinterpret_cast<int64_t*>(
          a_tile[0] + local_row * kLdsStride + k16 * 16);
      dst64[0] = src64[0];
      dst64[1] = src64[1];
    }
#pragma unroll
    for (int v = 0; v < kBVecPerThread; ++v) {
      const int vec = tid + v * kThreadsPerBlock;
      const int n_row = vec / kVecsPerRow;
      const int k16 = vec - n_row * kVecsPerRow;
      const int64_t* src64 = reinterpret_cast<const int64_t*>(&b_pre[v]);
      int64_t* dst64 = reinterpret_cast<int64_t*>(
          b_tile[0] + n_row * kLdsStride + k16 * 16);
      dst64[0] = src64[0];
      dst64[1] = src64[1];
    }
  }
  __syncthreads();

  // ---- Steady state: for s = 0..kStages-2 issue the stage-(s+1) global
  // loads into the staging registers BEFORE consuming stage s from buffer
  // s&1, then flush the prefetched vectors into the idle buffer (s+1)&1 and
  // close the stage with ONE __syncthreads (the barrier's vmcnt wait lands
  // after the consume, so the VMEM latency of stage s+1 overlaps the
  // LDS/MMAC work of stage s). The last stage is consumed after the loop
  // without a trailing barrier.
  for (int s = 0; s < kStages - 1; ++s) {
    // Prefetch stage s+1: issue the global loads (kept in VGPR while the
    // stage-s consume runs).
    const int k0 = (s + 1) * StageK;
#pragma unroll
    for (int v = 0; v < kAVecPerThread; ++v) {
      const int vec = tid + v * kThreadsPerBlock;
      const int local_row = vec / kVecsPerRow;
      const int k16 = vec - local_row * kVecsPerRow;
      const int g_row = m0 + local_row;
      a_pre[v] = (g_row < m)
                     ? *reinterpret_cast<const int4*>(
                           x_q + static_cast<int64_t>(g_row) * k + k0 +
                           k16 * 16)
                     : int4{0, 0, 0, 0};
    }
#pragma unroll
    for (int v = 0; v < kBVecPerThread; ++v) {
      const int vec = tid + v * kThreadsPerBlock;
      const int n_row = vec / kVecsPerRow;
      const int k16 = vec - n_row * kVecsPerRow;
      // Iteration 5: panel-packed B tile read (see the prologue comment).
      const int64_t panel_off =
          static_cast<int64_t>(n0 >> 6) * (static_cast<int64_t>(k) * 64) +
          (static_cast<int64_t>(k0 >> 4) + k16) * 1024 + n_row * 16;
      b_pre[v] = *reinterpret_cast<const int4*>(packed_w + panel_off);
    }

    // Consume stage s from buffer s&1: each wave accumulates its (BM/2) x
    // (BN/2) quadrant with kk-ascending m16n16k32 MMAs; one ds_read_b64 per
    // fragment.
    const int local_row = wave_row * (BM / 2);
    const int local_col = wave_col * (BN / 2);
#pragma unroll
    for (int kk = 0; kk < StageK; kk += kTileK) {
#pragma unroll
      for (int j = 0; j < kWaveM16; ++j) {
        load_frag8(
            a_frag[j],
            a_tile[s & 1] + (local_row + j * kTileM) * kLdsStride + kk,
            kLdsStride);
      }
#pragma unroll
      for (int i = 0; i < kWaveN16; ++i) {
        load_frag8(
            b_frag[i],
            b_tile[s & 1] + (local_col + i * kTileN) * kLdsStride + kk,
            kLdsStride);
      }
#pragma unroll
      for (int j = 0; j < kWaveM16; ++j) {
#pragma unroll
        for (int i = 0; i < kWaveN16; ++i) {
          du_mma_sync(acc[j][i], a_frag[j], b_frag[i], acc[j][i]);
        }
      }
    }

    // Flush the prefetched stage-(s+1) vectors into the idle buffer.
    {
      const int buf = (s + 1) & 1;
#pragma unroll
      for (int v = 0; v < kAVecPerThread; ++v) {
        const int vec = tid + v * kThreadsPerBlock;
        const int local_row = vec / kVecsPerRow;
        const int k16 = vec - local_row * kVecsPerRow;
        const int64_t* src64 = reinterpret_cast<const int64_t*>(&a_pre[v]);
        int64_t* dst64 = reinterpret_cast<int64_t*>(
            a_tile[buf] + local_row * kLdsStride + k16 * 16);
        dst64[0] = src64[0];
        dst64[1] = src64[1];
      }
#pragma unroll
      for (int v = 0; v < kBVecPerThread; ++v) {
        const int vec = tid + v * kThreadsPerBlock;
        const int n_row = vec / kVecsPerRow;
        const int k16 = vec - n_row * kVecsPerRow;
        const int64_t* src64 = reinterpret_cast<const int64_t*>(&b_pre[v]);
        int64_t* dst64 = reinterpret_cast<int64_t*>(
            b_tile[buf] + n_row * kLdsStride + k16 * 16);
        dst64[0] = src64[0];
        dst64[1] = src64[1];
      }
    }
    __syncthreads();
  }

  // ---- Tail: consume the last stage (its data was flushed before the last
  // barrier; no trailing barrier needed).
  {
    const int s = kStages - 1;
    const int local_row = wave_row * (BM / 2);
    const int local_col = wave_col * (BN / 2);
#pragma unroll
    for (int kk = 0; kk < StageK; kk += kTileK) {
#pragma unroll
      for (int j = 0; j < kWaveM16; ++j) {
        load_frag8(
            a_frag[j],
            a_tile[s & 1] + (local_row + j * kTileM) * kLdsStride + kk,
            kLdsStride);
      }
#pragma unroll
      for (int i = 0; i < kWaveN16; ++i) {
        load_frag8(
            b_frag[i],
            b_tile[s & 1] + (local_col + i * kTileN) * kLdsStride + kk,
            kLdsStride);
      }
#pragma unroll
      for (int j = 0; j < kWaveM16; ++j) {
#pragma unroll
        for (int i = 0; i < kWaveN16; ++i) {
          du_mma_sync(acc[j][i], a_frag[j], b_frag[i], acc[j][i]);
        }
      }
    }
  }

  const int base_row = m0 + wave_row * (BM / 2);
  const int base_col = n0 + wave_col * (BN / 2);
#pragma unroll
  for (int j = 0; j < kWaveM16; ++j) {
#pragma unroll
    for (int i = 0; i < kWaveN16; ++i) {
      store_prefill_fragment_coalesced(
          acc[j][i], x_scale, weight_scale, out, m, n,
          base_row + j * kTileM, base_col + i * kTileN, lane);
    }
  }
}

// Generic scalar int8/int32 fallback for unmatched shapes and small-M API
// cases (including the paired M=2 and M=16 shapes with the same (N, K) --
// they never reach the m >= 128 tiled path). One output element per
// grid-stride step; exact int32 accumulation (max assigned K = 6144 keeps
// the int8 dot well within int32 range: |dot| <= 6144*127*127 < 2^31), then
// the fused float scale (dot * x_scale[m] * weight_scale[n]) and bf16 store.
// The bootstrap pack is the identity copy, so packed_weight is always the
// logical [K, N] row-major layout here.
__global__ __launch_bounds__(256) void w8a8_scalar_gemm_kernel(
    const int8_t* __restrict__ a,       // [M, K] row-major
    const int8_t* __restrict__ b,       // [K, N] row-major (identity pack)
    const float* __restrict__ x_scale,  // [M]
    const float* __restrict__ weight_scale,  // [N]
    hip_bfloat16* __restrict__ out,     // [M, N]
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
    // Iteration 5: for the exact (k, n) == (6144, 2624) the packed_weight
    // buffer holds the B-panel layout packed[(col>>6)*(k*64) + (kk>>4)*1024
    // + (col&63)*16 + (kk&15)] = raw[kk*n + col] (41 panels of [K, 64] with
    // 16-byte k-runs per column, produced once out of the timed region), so
    // the fallback decodes that layout there in ascending-kk 16-byte runs
    // (keeps the paired M=2/M=16 API shapes with the same (N, K)
    // byte-exact); every other (k, n) keeps the identity [K, N] row-major
    // copy. The int32 accumulation stays kk-ascending 0..k-1 in both arms.
    const bool packed_panel = (k == kExactK && n == kExactN);
    int32_t acc = 0;
    if (packed_panel) {
      const int8_t* panel =
          b + static_cast<int64_t>(col >> 6) * (static_cast<int64_t>(k) * 64) +
          (col & 63) * 16;
      for (int kkb = 0; kkb < k; kkb += 16) {
        const int8_t* p = panel + (kkb >> 4) * 1024;
        for (int bb = 0; bb < 16; ++bb) {
          acc += static_cast<int32_t>(a_row[kkb + bb]) *
                 static_cast<int32_t>(p[bb]);
        }
      }
    } else {
      const int8_t* b_ptr = b + col;
      for (int kk = 0; kk < k; ++kk) {
        acc += static_cast<int32_t>(a_row[kk]) *
               static_cast<int32_t>(b_ptr[0]);
        b_ptr += n;
      }
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

// Exact-shape n-major pack (iteration 2): for (k, n) == (6144, 2624) the
// weight is transposed once, outside the timed region and out of Graph
// capture, into packed[n * k + kk] = raw[kk * n + n_row] so every col_major
// B fragment is 8 contiguous K bytes in the packed buffer. Same byte count
// and buffer as the identity pack, so captured addresses are unchanged. One
// thread per output byte; runs once during weight prep. (Kept compiled as
// the iteration-2/3 layout control; iteration 5 dispatches the panel pack
// below for the exact shape.)
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

// Exact-shape B-panel pack (iteration 5): for (k, n) == (6144, 2624) the
// weight is packed once, outside the timed region and out of Graph capture,
// into 41 panels of [K, 64] int8 with 16-byte k-runs per column:
//   packed[(n >> 6) * (k * 64) + (kk >> 4) * 1024 + (n & 63) * 16 +
//          (kk & 15)] = raw[kk * n + n]
// (the byte-exact bijection 41 x 384 x 64 x 16 = k*n is verified by
// simulation; same byte count and buffer, so captured addresses are
// unchanged). Every [64, 64] B tile of a 64-K stage is then 4 CONTIGUOUS
// 1-KiB slabs (one per 16-k-run) and every 16-byte staging vector is a
// 16-byte-aligned contiguous k-run of one column, so each DUMMA B tile is
// vector-loadable as one coalesced stream with full 128-B line utilization
// (vs 64 scattered 64-B chunks at 6144-B stride in the n-major layout).
// One thread per output byte; runs once during weight prep.
__global__ __launch_bounds__(256) void w8a8_pack_panel_i8_kernel(
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
    // (col >> 6) < n/64 <= 41, (kk >> 4) < k/16 <= 384, (col & 63) < 64,
    // (kk & 15) < 16 -> the mixed-radix index stays < k*n in int.
    packed[(col >> 6) * (k * 64) + (kk >> 4) * 1024 + (col & 63) * 16 +
           (kk & 15)] = raw[static_cast<int64_t>(kk) * n + col];
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
// caller-provided stream only; never allocates, synchronizes, packs, or
// touches the default stream.
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

  // Exact assigned shape guard: glm_tp8_fused_qkv_a_proj_m4096
  // (M=4096, N=2624, K=6144), m >= 128 so M=2/M=16 paired API shapes with
  // the same (N, K) keep reaching the scalar fallback below. Iteration 5:
  // dispatch the exact-fit 2-D macro-tile <128, 64, 128> geometry with the
  // B-PANEL packed weight (launch_pack_w8a8_weight packs (6144, 2624) once,
  // out of the timed region, into packed[(n>>6)*(k*64) + (kk>>4)*1024 +
  // (n&63)*16 + (kk&15)] = raw[kk*N + n] so every 64-K B tile is 4
  // contiguous 1-KiB slabs), the bank-safe LDS row strides, the explicit
  // 8-byte load_frag8 fragment loads, and the double-buffered 64-K K
  // pipeline (one __syncthreads per stage, stage-(s+1) global loads
  // prefetched into staging VGPR before the stage-s consume) -- the
  // mandated packing/swizzle round at unchanged tile/occupancy (2
  // blocks/CU, 27,648 B LDS/block). The single-buffered iteration-2
  // control kernel stays compiled (see the m < 0 forcing block below) for
  // the A/B and for rollback.
  if (m >= 128 && n == kExactN && k == kExactK) {
    const dim3 grid(
        static_cast<unsigned>(n / 64),
        static_cast<unsigned>((m + 127) / 128));
    const dim3 block(static_cast<unsigned>(kThreadsPerBlock));
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(w8a8_dumma_prefill_packedb_db_kernel<128, 64, 64>),
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

  // Force the sibling tile instantiations to compile (<64,128,128> is the
  // next benchmark candidate; this branch can never run because m <= 0
  // already returned above). The single-buffered iteration-2 packed-B kernel
  // is kept here so the control arm of the iteration-3 A/B stays compiled
  // (and rollback only needs the guard repointed).
  if (m < 0) {
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(w8a8_dumma_prefill_tiled_kernel<64, 128, 128>),
        dim3(1),
        dim3(static_cast<unsigned>(kThreadsPerBlock)),
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
        HIP_KERNEL_NAME(w8a8_dumma_prefill_tiled_kernel<128, 64, 128>),
        dim3(1),
        dim3(static_cast<unsigned>(kThreadsPerBlock)),
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
        HIP_KERNEL_NAME(w8a8_dumma_prefill_packedb_kernel<128, 64, 128>),
        dim3(1),
        dim3(static_cast<unsigned>(kThreadsPerBlock)),
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

  // Generic native INT8 DUMMA prefill path for every large-M shape with
  // N % 64 == 0 and K % 128 == 0 (identity [K, N] layout). The exact shape
  // above already returned; this covers compatible sibling shapes.
  if (m >= 128 && (n % 64) == 0 && (k % 128) == 0) {
    const dim3 grid(
        static_cast<unsigned>(n / 64),
        static_cast<unsigned>((m + 63) / 64));
    const dim3 block(static_cast<unsigned>(kThreadsPerBlock));
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(w8a8_dumma_prefill_tiled_kernel<64, 64, 128>),
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

  // Scalar int8/int32 fallback for every unmatched (m, n, k) and all
  // small-M API cases.
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

// Stable host launch symbol consumed by csrc/bindings.cpp. Iteration 5: the
// exact (k, n) == (6144, 2624) int8 weight is packed once (outside the
// timed region and out of Graph capture) into the B-panel layout
// packed[(n>>6)*(k*64) + (kk>>4)*1024 + (n&63)*16 + (kk&15)] = raw[kk*N + n]
// consumed by the packed-B prefill kernel's panel tile reads and by the
// scalar fallback's panel decode; every other (K, N) keeps the identity copy
// (the logical [K, N] row-major weight). The [N] fp32 scales are always
// copied unchanged. Same byte count (k*n and n), same allocation,
// graph-stable addresses.
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
  // Iteration 5: the exact (k, n) == (6144, 2624) weight is packed once,
  // outside the timed region and out of Graph capture, into the B-panel
  // layout (same byte count and buffer addresses, so captured pointers stay
  // valid; the packed-B prefill kernel and the scalar fallback for this
  // (k, n) both decode it). Every other (K, N) keeps the identity copy
  // (raw row-major [K, N]).
  if (k == kExactK && n == kExactN) {
    int64_t blocks = (weight_count + kPackThreads - 1) / kPackThreads;
    if (blocks > 4096) {
      blocks = 4096;
    }
    if (blocks < 1) {
      blocks = 1;
    }
    hipLaunchKernelGGL(
        w8a8_pack_panel_i8_kernel,
        dim3(static_cast<unsigned>(blocks)),
        dim3(kPackThreads),
        0,
        stream,
        raw_weight,
        packed_weight,
        k,
        n);
  } else if (weight_count > 0) {
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
// @@end
