// @@variant shape=minimax_tp8_qkv_proj_and_indexer_qk_m16 commit=e72025dfcfa50ff3bf3fcb00b87663e65f4e0fc4 added=2026-08-28
//   median_us=39.73 p90_us=39.81
//   source=minimax-dsh-tp8-m16-1-78402260
// csrc/w8a8_gemm_hip.hip
//
// Worker 1 (physical GPU 1) — assigned shape:
//   minimax_tp8_qkv_proj_and_indexer_qk_m16 : M=16, N=1536, K=6144
//
// Iteration 0: correctness-first scalar bootstrap (one thread per output
// element, blockDim 128).  Preserved below as the generic fallback for every
// unmatched (m, n, k) shape, including the paired M=2 shape with the same
// (N, K).
//
// Iteration 1 (DUMMA bootstrap, accepted): minimal native gfx928 DUMMA INT8
// m16n16k32 tile, one 64-lane wavefront per block, one 16x16 output tile per
// block, direct global fragment loads, explicit int32 accumulation, direct
// scaled-bf16 fragment epilogue.  Measured 203.2 us median vs the 111.5 us
// fixed Triton baseline; grid = N/16 = 96 blocks on 120 CUs (0.8 blocks/CU,
// below the two-blocks-per-CU latency-hiding target).
//
// Iteration 2 (architecture round: measure grid parallelism before
// polishing): replace the exact (m==16 && n==1536 && k==6144) arm with a
// split-K=4 partial GEMM + separate combine:
//   * partial kernel: grid = 96 N-tiles x 4 K-splits = 384 one-wave blocks
//     (64 lanes each, one 16x16 output tile per block); uniform K=1536
//     slice per split (48 m16n16k32 steps, k ascending within the slice,
//     exact int32 order); direct global-to-fragment loads (same load path
//     as iteration 1 — vectorized loads/LDS staging are deliberately
//     deferred to a later polishing round); each block stores its 16x16
//     int32 partial to its exclusive plane partials[split_id][16][N];
//   * combine kernel: grid = 96 one-wave blocks; sums the 4 planes in
//     ascending split order (split s covers k in [s*1536,(s+1)*1536) →
//     bit-identical int32 accumulation order), applies
//     x_scale[row] * weight_scale[col], stores scaled bf16.
// Zero LDS, zero barriers, zero atomics in both kernels.  The two launches
// are ordered by the caller-provided stream, so the combine kernel always
// observes the partial kernel's writes and no workspace clear is needed
// (every launch overwrites every plane element before combine reads it).
// 384 blocks = 3.2 blocks/CU on the 120-CU device: all CUs are covered and
// co-resident one-wave blocks can overlap the global-load latency chains —
// the falsifiable grid-parallelism claim of this round.
//
// launch_pack_w8a8_weight stays an identity device-to-device copy (packed
// layout == logical [K, N] row-major), unchanged from iteration 0.
//
// Iteration 6 (pipeline round: B-only LDS staging with coalesced 8/16-byte
// cooperative loads): keep the accepted iteration-2 geometry (split-K=4,
// 384 one-wave blocks), workspace, combine kernel, identity pack, scalar
// fallback, and bit-exact int32 order; replace ONLY the partial kernel's
// fragment transport.  L2/VMEM evidence picks B as the operand to stage: B
// is the only compulsory-HBM operand (9,437,184 B read exactly once, zero
// reuse) while A is 96 KiB unique, re-read 96x, and L2-hot (measured 76.5%
// L2 hit rate).  Each block stages its [1536][16] B K-slice once into a
// compact 24,704 B n-major LDS tile s_b[16][1544] (stride 1544 = 1536 + 8:
// rows are 8-B aligned for ds_read_b64 and the stride spreads the 64 lanes'
// bank pairs to the 4-cycle conflict floor; 2 blocks/CU co-residency, vs
// iteration 4's whole-slice A+B tile at 1 block/CU) via 24 coalesced
// dwordx4 (16 B/lane) cooperative loads + conflict-free ds_write_b8 + one
// __syncthreads; the 48-step K loop then contains no global memory op for B
// at all: A is one aligned global_load_dwordx2 per step (verified 8-byte
// loader) and B is one ds_read_b64 per lane per step, with a depth-1
// register pipeline issuing step s+1's loads before step s's v_mmac.
// Whether the issued-ahead loads actually overlap the MMAC is measured,
// not assumed.  HBM bytes are unchanged (B is still fetched exactly once,
// now in 24 wide loads per block instead of 384 byte-loads); LDS bytes go
// 0 -> 49,152 B/block (24,576 staged + 24,576 read) = 18.9 MB over the
// launch; vmem_read instructions drop 294,912 -> 27,648 (9,216 staging +
// 18,432 A).
//
// Iteration 7 (pipeline round decision: SINGLE buffering kept — the
// double-buffering comparison gate fails on both criteria: L2 hit rate is
// 77.8% (not below 70%, because A is L2-hot and B is already staged) and
// doubled LDS would be 2 x 24,704 B = 49,408 B (not below 48 KiB), so
// double buffering is not indicated; barriers per K step are counted at 0
// both before and after — the staging __syncthreads compiles out for
// one-wave blocks (0 s_barrier in the iteration-6 ISA)).  Deepen the
// single-buffered pipeline instead: the iteration-6 depth-1 ping-pong
// compiled to ONE 8-step unrolled group iterated 6x whose group ENTRY
// issues the group's first A global load and then fully drains it with
// s_waitcnt vmcnt(0) lgkmcnt(0) before the first MMAC — 6 full L2 round
// trips per block on the critical path (steps 1..7 of each group get
// staggered-wait coverage, step 0 of each group does not).  Replace it
// with a depth-8 single-buffered register ring for A: every A global load
// is issued exactly 8 steps before its MMAC so no MMAC ever waits on an
// L2 round trip; B stays on the 24,704 B LDS tile (one ds_read_b64 per
// step, hoisted/batched by the scheduler); the K loop still has zero
// barriers; LDS stays 24,704 B/block (2 blocks/CU).  Everything else
// (split-K=4 grid, workspace, combine, identity pack, scalar fallback,
// exact int32 order) is unchanged.
//
// Iteration 8 (HIP-only resource round: LDS-footprint occupancy limiter):
// the fresh exact-source PMC launch records (iteration8/pmc.csv) show the
// iteration-7 kernel at arch_vgpr=168, lds=24,704 B, scratch=0, 1 wave per
// block, 384 blocks — the BINDING occupancy limiter is the LDS footprint:
// 64 KiB/CU / 24,704 B = 2.65 -> 2 blocks/CU co-residency (3 x 24,704 B =
// 74,112 B > 64 KiB), i.e. only 2 of the CU's 4 SIMDs carry a wave (VGPRs
// at 168 allow 1 wave/SIMD = 4 waves/CU, not binding at 2 blocks).  The
// exact-source ISA (iteration8/current-best-isa/isa.txt) also shows the
// depth-8 ring compiled to a FULL 48-step unroll: all 48 A
// global_load_dwordx2 issued in one burst (v[69:70]..v[163:164]), 16
// ds_read2_b64 for the B fragments, ONE s_waitcnt vmcnt(47) lgkmcnt(14) at
// 0x2D64, then 48 dependent v_mmac on the single accumulator with no
// inter-MMAC waits — the per-wave critical path is one A round trip + a
// 48-deep dependent MMAC chain hidden by only 2 resident waves/CU.
// Falsifiable claim: halve the per-block staged B slice by moving the
// exact (16,1536,6144) arm from split-K=4 to split-K=8 — grid 96 N-tiles x
// 8 splits = 768 one-wave blocks (6.4 blocks/CU), B tile s_b[16][776]
// (12,288 B payload + 8 B/row pad, 776/4 = 194 == 2 mod 32 keeps the
// 4-cycle bank-conflict floor, 12,416 B/block -> LDS co-residency 5
// blocks/CU, VGPR-capped at 4-5), K loop 24 steps (24 A loads, 12
// ds_read2_b64, one wait, 24 dependent MMACs — half the serial chain per
// wave), workspace 8 x 98,304 B = 786,432 B (<= the 16-plane 1,572,864 B
// capacity allocate_workspace guarantees for this shape), combine sums the
// 8 planes in ascending split order (int32 addition is exact mod 2^32, so
// the split grouping does not change the accumulated value), identity pack
// and scalar fallback unchanged.  HBM byte accounting is unchanged: B is
// still read exactly once (768 blocks x 12,288 B = 9,437,184 B), A read
// volume is unchanged (768 x 12,288 B = 9,437,184 B, still 96x re-read of
// the 96 KiB L2-hot slice), vmem_read instructions stay 27,648 (9,216
// staging + 18,432 A) and LDS bytes stay 18,874,368 B/launch (12,288
// staged + 12,288 read per block x 768).  No repeated HBM operand reads are
// traded for the occupancy gain.  If median does not improve below 74.018
// us / p90 below 74.777 us, the LDS-footprint/co-residency lever is
// falsified and the next round must attack the 24-48-deep dependent MMAC
// accumulation chain itself (e.g. split accumulators) at a fixed split-K.
//
// Iteration 9 (rejected, NOT in this source): the staging-burst candidate
// (median 66.626 us, -26.07% vs iteration 8) tried to collapse the three
// serial staging rounds into one 12-load burst.  Its exact code object
// (iterations/.../iteration9/isa/isa.txt, digest 1290fe7b) shows the
// compiler limitation: it refused to emit a single 12-load burst — it
// unrolled the staging into straight-line 8+2+2 global_load_dwordx4 with
// partial vmcnt drains (vmcnt(5) after 8 loads, vmcnt(6)->vmcnt(2) after
// the 9th/10th, vmcnt(1)->vmcnt(0) after the 11th/12th) — and the 48 live
// staging VGPRs lifted arch_vgpr 104 -> 137, dropping co-residency 5 -> 4
// blocks/CU (1 wave/SIMD).  Both the unchanged round structure AND the
// register budget lost; the round is not in the source (restored to the
// accepted iteration-8 digest 4a6e82d before this round).
//
// Iteration 10 (this round, HIP-only occupancy probe): the accepted
// iteration-8 exact-source ISA (profiles/.../iteration9/current-best-isa/
// isa.txt) shows the staging #pragma unroll 4 loop compiled to a 3-round
// loop whose body issues 4 global_load_dwordx4, drains them progressively
// to vmcnt(0), writes 64 ds_write_b8, and branches back — 3 serial HBM
// round trips per wave on the critical path (600 concurrent waves x
// 12,288 B = 7.2 MB in flight per round), i.e. the per-wave serial staging
// RTT chain and co-residency, not the MMAC chain, bind the partial kernel
// (measured co-res 2 -> 5 moved 74.018 -> 49.259 us; co-res 5 -> 4 moved
// it back to 66.626 us).  Falsifiable claim: move ONLY the split count
// 8 -> 12 inside the exact (16,1536,6144) arm — keeping the m16n16k32
// DUMMA tile, the one-wave/one-tile block structure, the B-only LDS
// staging loop recipe (coalesced dwordx4 + conflict-free ds_write_b8 + one
// __syncthreads), the depth-8 A register ring (kSliceSteps 16 % 8 == 0,
// ring closes exactly on step 15), the combine kernel (now summing 12
// planes in ascending split order, same code path), the identity pack, the
// scalar fallback, and the bit-exact int32 order — with: grid 96 N-tiles x
// 12 splits = 1,152 one-wave blocks (9.6 blocks/CU); per-block B slice
// 512 k -> LDS tile s_b[16][520] = 8,320 B (8,192 B payload + 8 B/row
// pad; 520 % 8 == 0 keeps rows 8-B aligned and 520/4 = 130 == 2 mod 32
// keeps the 4-cycle bank-conflict floor) -> co-residency 7 blocks/CU by
// LDS (65,536 / 8,320 = 7.88; VGPR-capped at 8, arch_vgpr must stay
// <= 128 or co-res drops to 4 and the probe is falsified); K loop 16 steps
// (16 A loads, 8 ds_read2_b64, one wait, 16 dependent MMACs); staging 8
// dwordx4 chunks -> the unroll-4 loop runs 2 serial rounds instead of 3
// (one fewer HBM RTT per wave); workspace 12 x 98,304 B = 1,179,648 B <=
// the 16-plane 1,572,864 B capacity (the workspace_bytes guard passes);
// combine sums the 12 planes in ascending split order.  HBM and LDS byte
// accounting is unchanged — no repeated HBM reads are traded: B is still
// read exactly once (1,152 blocks x 8,192 B = 9,437,184 B), A read volume
// is unchanged (1,152 x 8,192 B = 9,437,184 B, still the 96x re-read of
// the 96 KiB L2-hot slice), vmem_read instructions stay 27,648 (9,216
// staging + 18,432 A), LDS bytes stay 18,874,368 B/launch (8,192 staged +
// 8,192 read per block x 1,152).  Predicted deltas: (a) partial-kernel
// staging rounds 3 -> 2, (b) co-residency 5 -> 7 (840 concurrent waves,
// 1.37 rounds vs 1.28), (c) MMAC chain 24 -> 16, (d) combine reads 12
// planes (1,179,648 B) instead of 8 (786,432 B) — combine grows ~5.28 ->
// ~6.5-7 us while the partial kernel is predicted to drop ~38.9 ->
// ~27-33 us.  If median does not improve below 49.259 us / p90 below
// 49.415 us, the occupancy probe at co-res 7 is falsified and the next
// round must either probe split-K=16 (co-res 8, but 1.6 rounds and 16
// combine planes) or attack the remaining serial staging rounds
// (e.g. wider per-round loads) at a fixed split-K.
//
// Iteration 14 (accepted, 43.186 us median / 43.361 us p90): the
// staging-burst (iteration 11: 85.049 us), fused-combine (iteration 12:
// 45.115 us) and split-K=16 occupancy (iteration 13: 45.345 us median /
// 202.93 us p90) probes all falsified their levers at fixed split-K=12
// geometry; the remaining serial wall component was the STANDALONE combine
// kernel's load pattern: each lane read its 4 elements x 12 planes as 48
// scalar int32 global loads grouped in four serial load-drain rounds (ISA
// staircase vmcnt drains), and the combine runs stream-serial after the
// 1,152-block partial kernel, so its HBM-latency-bound drains (L2 hit
// 16.288%, measured at split-K=8) add directly to the operator wall.
// Change was ONLY the combine kernel: each lane owns 4 consecutive n
// columns of one row (row = lane>>2, tcol4 = (lane&3)<<2) so the 12 plane
// reads per lane become 16-B-aligned int4 loads (one drain group instead
// of four) and the bf16 epilogue becomes one 8-B uint64 store; per-element
// ascending split sum s = 0..11 is unchanged -> bit-identical int32 order.
// Accepted at 43.186 us median / 43.361 us p90 (below the 44.283/44.502
// falsifier), so the combine's load-drain structure WAS a wall component.
//
// Iteration 15 (this round, HIP-only consolidation): the accepted
// iteration-14 exact-source ISA (iteration14/isa/isa.txt, source digest
// d67fd4d5) records the partial kernel at arch_vgpr=72, sgpr=25, lds=8,320
// B, scratch=0, grid 1,152 and shows the 16-step K loop compiled as one
// pre-loop 16-load A burst + 8 ds_read2_b64 batch + 16 v_mmac chained on a
// SINGLE accumulator v[1:4] with only a vmcnt(15)..vmcnt(0)/
// lgkmcnt(7)..lgkmcnt(0) staircase between them — a 16-deep dependent MMAC
// chain per wave (the iteration-14 falsifier names this as the remaining
// probe: 'the next round must attack the dependent 16-MMAC accumulator
// chain with split accumulators (value-identical mod 2^32) at fixed
// split-K=12, or stop at co-res 7').  Change ONLY the partial kernel's
// accumulator structure: two independent accumulator fragments acc0/acc1 —
// steps 0..7 accumulate into acc0, steps 8..15 into acc1 (k ascending
// within each chain), then acc0 += acc1 before the plane store; int32
// addition is exact mod 2^32 so the grouped ascending sum is bit-identical
// (0 mismatches expected).  The dependent MMAC chain per wave halves 16 ->
// 8; MMAC issue count (16), staging recipe (2-round unroll-4, 8,320 B LDS
// tile), depth-8 A ring (refill loads stay unconditional, ring closes
// exactly on step 15), plane store, combine kernel (vectorized int4,
// grid 96), identity pack, scalar fallback, paired M=2 fallback, and
// workspace 1,179,648 B are untouched.  Predicted deltas: (a) per-wave
// MMAC critical path 16 x MMAC-latency -> 8 x MMAC-latency + 4 int32 adds
// (the adds are VALU on a different pipe and overlap other waves), exposed
// mainly in the 1.37-round tail where fewer than 7 waves/CU remain to hide
// the chain; (b) arch_vgpr expected ~72-80 (acc1 adds 4 VGPRs; 65,536 /
// (7 x 64) = 146.3, so co-res 7 by LDS is unaffected below arch_vgpr
// 146); scratch_bytes > 0 in the next PMC would falsify the no-spill
// premise.  Falsifiers: median not below 43.186 us / p90 not below 43.361
// us (the accumulator chain is not the remaining wall component at co-res
// 7; the family is at its measured floor and the next round must stop or
// re-probe with fresh PMC); any correctness mismatch falsifies the
// bit-exact premise.

#include <hip/hip_runtime.h>
#include <hip/hip_bfloat16.h>
#include <du_mma.h>

#include <cstdint>

namespace {

// gfx928 wavefront is 64 lanes; blockDim must be a multiple of 64.
constexpr int kScalarBlockThreads = 128;

// Minimal DUMMA tile constants (gfx928 INT8 support is m16n16k32).
constexpr int kDummaM = 16;
constexpr int kDummaN = 16;
constexpr int kDummaK = 32;
constexpr int kDummaThreads = 64;  // one wavefront (64 lanes) per block

// Iteration 2 architecture: split-K = 4 for the assigned shape.  96 N-tiles
// x 4 splits = 384 blocks = 3.2 blocks/CU on the 120-CU device (>= the
// two-blocks-per-CU latency-hiding target; every CU covered).  Uniform
// k32-aligned slices: K=6144 / 4 = 1536 = 48 m16n16k32 steps.  Workspace:
// 4 x [16][1536] int32 planes = 393,216 B, inside the 16-plane capacity
// (16 x 98,304 B = 1,572,864 B) that allocate_workspace guarantees for this
// shape.  Trusted occupancy-probe split list [2,3,4,5,7,8,10,15]: only
// {2,3,4,8,12} divide 6144 into k32-aligned slices.  Iteration 8 moved the
// arm from 4 to 8 (96 N-tiles x 8 splits = 768 one-wave blocks, LDS tile
// 24,704 B -> 12,416 B, co-residency 2 -> 5, MMAC chain 48 -> 24 steps;
// accepted 74.018 -> 49.259 us).  Iteration 9 (staging burst at split-K=8)
// regressed to 66.626 us and is NOT in this source: its exact code object
// shows the compiler refused to emit one 12-load burst (it scheduled 8+2+2
// dwordx4 loads with partial vmcnt drains) and arch_vgpr 104 -> 137 dropped
// co-residency 5 -> 4.  Iteration 10 (this round) probes split-K = 12:
// 96 N-tiles x 12 splits = 1,152 one-wave blocks, LDS tile 12,416 B ->
// 8,320 B -> co-residency 7 by LDS (VGPR-capped at 8), staging 12 -> 8
// dwordx4 chunks (the unroll-4 staging loop runs 2 serial rounds instead of
// 3), MMAC chain 24 -> 16 steps, workspace 12 x 98,304 B = 1,179,648 B
// inside the 16-plane 1,572,864 B capacity.
constexpr int kSplitK = 12;

// Iteration 6 pipeline constants, valid only under the exact-shape guard
// (m == 16 && n == 1536 && k == 6144) that is the sole caller of this
// kernel: uniform K slice = k / kSplitK = 512 at split-K=12 (was 768 at
// split-K=8, 1536 at split-K=4) = 16 m16n16k32 steps.
constexpr int kSliceK = 512;       // K per split (6144 / 12)
constexpr int kSliceKStride = 520; // LDS row stride: 512 + 8 pad
constexpr int kSliceSteps = kSliceK / kDummaK;  // 16
constexpr int kStageInstrs = (kSliceK * kDummaN) / (kDummaThreads * 16);  // 8
// Iteration 7 pipeline depth: A register-ring depth (must be a power of two
// and divide kSliceSteps so the ring closes exactly: 16 % 8 == 0).
constexpr int kPrefetchDepth = 8;

// Scalar int8 GEMM: out[m,n] = bf16(int32_dot(a[m,:], b[:,n]) *
//                                   x_scale[m] * weight_scale[n]).
// a           : [M, K] int8, row-major
// b           : [K, N] int8, row-major
// x_scale     : [M, 1] float
// weight_scale: [N, 1] float
// out         : [M, N] bf16, row-major
__global__ __launch_bounds__(kScalarBlockThreads) void
w8a8_scalar_gemm_kernel(
    const int8_t* __restrict__ a,
    const int8_t* __restrict__ b,
    const float* __restrict__ x_scale,
    const float* __restrict__ weight_scale,
    hip_bfloat16* __restrict__ out,
    int m,
    int n,
    int k) {
  const int total = m * n;
  const int linear = blockIdx.x * kScalarBlockThreads + threadIdx.x;
  if (linear >= total) {
    return;
  }

  // Adjacent lanes land on adjacent addresses in the fastest-changing N
  // dimension (linear = row * n + col).
  const int row = linear / n;
  const int col = linear - row * n;

  const int8_t* __restrict__ a_row = a + row * k;
  const int8_t* __restrict__ b_col = b + col;

  int32_t dot = 0;
  for (int kk = 0; kk < k; ++kk) {
    dot += static_cast<int32_t>(a_row[kk]) *
           static_cast<int32_t>(b_col[kk * n]);
  }

  const float scaled =
      static_cast<float>(dot) * x_scale[row] * weight_scale[col];
  out[linear] = __float2bfloat16(scaled);
}

// Iteration 6 fragment transport helpers (byte placement identical to the
// library loaders, verified in the iteration-5 round):
//   du_load_matrix_sync(matrix_a, row_major) and (matrix_b, col_major)
//   both place lane l's 8 elements at p[(l & 15) * ldm + ((l >> 4) << 3) +
//   i], i = 0..7 (confirmed against du_mma.hpp).  With ldm and offsets that
//   are multiples of 8, each lane's fragment is ONE 8-B-aligned vector
//   load: A = global_load_dwordx2 from the logical [16][K] row-major
//   activation (L2-hot); B = ds_read_b64 from the n-major LDS tile
//   s_b[16][kSliceKStride] (stride 1544 at split-K=4, 776 at split-K=8,
//   520 at split-K=12; stride % 8 == 0 keeps every row 8-B aligned and
//   stride/4 == 2 mod 32 spreads the 64 lanes' bank pairs to the 4-cycle
//   conflict floor for a full-wave 512-B ds_read_b64).
__device__ __forceinline__ void load_a_frag8(
    du::dumma::DUFragment<du::dumma::matrix_a, kDummaM, kDummaN, kDummaK,
                          signed char, du::dumma::row_major>& f,
    const int8_t* __restrict__ p,
    int ldm) {
  const unsigned row = static_cast<unsigned>(threadIdx.x) & 0xfu;
  const unsigned col = (static_cast<unsigned>(threadIdx.x) >> 4) << 3;
  const int64_t v = *reinterpret_cast<const int64_t*>(p + row * ldm + col);
  *reinterpret_cast<int64_t*>(&f.x[0]) = v;
}

__device__ __forceinline__ void load_b_lds8(
    du::dumma::DUFragment<du::dumma::matrix_b, kDummaM, kDummaN, kDummaK,
                          signed char, du::dumma::col_major>& f,
    const int8_t* __restrict__ s_b,
    int k_local) {
  const unsigned row = static_cast<unsigned>(threadIdx.x) & 0xfu;
  const unsigned col = (static_cast<unsigned>(threadIdx.x) >> 4) << 3;
  const int64_t v = *reinterpret_cast<const int64_t*>(
      s_b + row * kSliceKStride + k_local + col);
  *reinterpret_cast<int64_t*>(&f.x[0]) = v;
}

// Split-K partial GEMM kernel (exact (16, 1536, 6144) arm, S = kSplitK):
//   block b -> tile_id = b / S (0..95), split_id = b % S (0..11);
//   K slice [split_id * 512, (split_id + 1) * 512), k ascending in
//   m16n16k32 steps; stores the 16x16 int32 partial into its exclusive
//   plane partials[split_id][16][N] (no atomic, no zeroing needed: each
//   split block owns its plane and every launch rewrites all elements).
// One 64-lane wavefront per block; iteration-6 B-only LDS staging
// (iteration 8 halves the staged slice with split-K=8; iteration 10 halves
// it again with split-K=12):
//   * stage: 8 coalesced dwordx4 global loads (16 B/lane, one B row per
//     lane) transpose the block's [512][16] B slice into the n-major LDS
//     tile s_b[n][kk] = raw[(k_base + kk) * n + n0 + n] via 16
//     conflict-free ds_write_b8 per lane per instruction (64 lanes hit 16
//     distinct LDS words), then one __syncthreads; the #pragma unroll 4
//     staging loop compiles to 2 serial rounds (4 loads drained to
//     vmcnt(0) each) instead of the 3 rounds at split-K=8; iteration 17
//     splits the staging by K-half: rows 0..255 are staged and barriered
//     as before, rows 256..511 loads are ISSUED right after the barrier
//     (before MMAC 0) and their ds_writes are DEFERRED to between the two
//     K half-loops with one extra __syncthreads, so the second staging
//     round trip overlaps MMACs 0..7 instead of sitting serially behind
//     round 1's drain (same 4-load in-flight quantum — no burst widening);
//   * K loop (16 steps, no global op for B, no barrier): A = one
//     global_load_dwordx2 (load_a_frag8), B = one ds_read_b64
//     (load_b_lds8); iteration 7 deepens the single-buffered pipeline to a
//     depth-8 register ring for A: load_a_frag8(a_ring[slot], step s+8) is
//     issued in iteration s, consumed by the MMAC of iteration s+8, so no
//     MMAC ever waits on an L2 round trip (the iteration-6 ISA drained
//     vmcnt(0)+lgkmcnt(0) at every 8-step group entry, 6 full L2 round
//     trips per block on the critical path — removed here).  B stays LDS
//     and is loaded per step (scheduler hoists/batches the ds_reads).
//     With split-K=12 the per-wave dependent MMAC chain is 16 steps instead
//     of 24, and the 8,320 B LDS tile (16 x 520) raises co-residency to
//     7 blocks/CU by LDS (VGPR-capped at 8; was 5 at 12,416 B) — the
//     iteration-10 occupancy probe.
// a        : [16, K] int8 row-major (ldm = k)
// b        : [K, N] int8 row-major (ldm = n), raw logical layout (no pack)
// partials : [S][16][N] int32 planes (workspace, preallocated pre-Graph)
__global__ __launch_bounds__(kDummaThreads) void
w8a8_dumma_m16n16k32_sk4_partial_kernel(
    const int8_t* __restrict__ a,
    const int8_t* __restrict__ b,
    int* __restrict__ partials,
    int n,
    int k) {
  __shared__ int8_t s_b[kDummaN][kSliceKStride];

  const int lane = static_cast<int>(threadIdx.x);
  const int tile_id = static_cast<int>(blockIdx.x) / kSplitK;
  const int split_id = static_cast<int>(blockIdx.x) % kSplitK;
  const int n0 = tile_id * kDummaN;
  const int k_base = split_id * (k / kSplitK);

  // Stage the block's B K-slice once (8,192 B read exactly once from
  // global, same bytes as the previous per-step loads; 8 dwordx4
  // cooperative loads instead of 128 byte loads).  Iteration 17 splits the
  // staging by K-half: rows 0..255 are staged and barriered as before, but
  // the rows 256..511 loads are ISSUED right after the barrier (before any
  // MMAC) and their LDS writes are DEFERRED to between the two K
  // half-loops (one extra __syncthreads), so round 2's L2/DRAM round trip
  // overlaps MMACs 0..7 instead of sitting serially behind round 1's drain
  // (the accepted iteration-15 ISA drains the two 4-load rounds strictly
  // one after the other).  The in-flight VMEM quantum stays at 4 staging
  // dwordx4 + the 16 A dwordx2 burst — no wider burst (the iteration-9/11
  // falsification) — and the deferred write + second barrier preserve the
  // exact same LDS contents and k-ascending MMAC order (bit-identical).
  const int8_t* __restrict__ b_slice =
      b + static_cast<int64_t>(k_base) * n + n0;
#pragma unroll
  for (int i = 0; i < kStageInstrs / 2; ++i) {
    const int row = i * kDummaThreads + lane;  // 0..255
    const int4 v = *reinterpret_cast<const int4*>(b_slice + row * n);
    const int8_t* src = reinterpret_cast<const int8_t*>(&v);
#pragma unroll
    for (int c = 0; c < kDummaN; ++c) {
      s_b[c][row] = src[c];
    }
  }
  __syncthreads();
  // Round 2's loads are issued AFTER the A burst below (not before it): the
  // waitcnt staircase counts ALL outstanding VMEM ops in issue order, so the
  // MMAC 0..7 waits must not be gated by the cold B loads.  Issued last,
  // they complete in the background while MMACs 0..7 drain the L2-hot A
  // burst, and the deferred writes below wait only vmcnt(3)..vmcnt(0) for
  // them after the first half-loop.
  int4 b_stage2[kStageInstrs / 2];

  du::dumma::DUFragment<du::dumma::matrix_a, kDummaM, kDummaN, kDummaK,
                        signed char, du::dumma::row_major>
      a_ring[kPrefetchDepth];
  // Iteration 15: two split accumulators.  Steps 0..7 accumulate into acc0,
  // steps 8..15 into acc1 (each chain k-ascending), then acc0 += acc1 —
  // int32 addition is exact mod 2^32, so the grouped sum is bit-identical
  // to the single-accumulator ascending sum (and to the reference order).
  // The dependent MMAC chain per wave halves 16 -> 8 while the MMAC issue
  // count (16), the A/B/plane byte traffic, the LDS tile, and the plane
  // layout stay unchanged.
  du::dumma::DUFragment<du::dumma::accumulator, kDummaM, kDummaN, kDummaK,
                        int>
      acc0;
  du::dumma::DUFragment<du::dumma::accumulator, kDummaM, kDummaN, kDummaK,
                        int>
      acc1;
  du::dumma::du_fill_fragment(acc0, 0);
  du::dumma::du_fill_fragment(acc1, 0);

  // Iteration 7: depth-8 single-buffered register ring for A.  The
  // iteration-6 depth-1 ping-pong compiled to one 8-step unrolled group
  // iterated 6x, and each group entry issued its first A load and then
  // fully drained it (s_waitcnt vmcnt(0) lgkmcnt(0)) before the first MMAC
  // — 6 full L2 round trips per block on the critical path.  Here every A
  // load for step s+8 is issued in iteration s (8 MMAC issue slots before
  // its use), so no MMAC sits on an L2 round trip.  kSliceSteps (16 at
  // split-K=12) is a multiple of kPrefetchDepth (8), so the ring closes
  // exactly on step 15; the modulo index is a compile-time constant under
  // full unroll, keeping the ring in registers.  B is read from LDS per
  // step; the scheduler hoists and batches the ds_reads as before.  MMAC
  // order stays k ascending 0..15 -> bit-identical int32 accumulation.
#pragma unroll
  for (int i = 0; i < kPrefetchDepth; ++i) {
    load_a_frag8(a_ring[i], a + k_base + i * kDummaK, k);
  }
  // Round-2 (rows 256..511) B loads issued after the A burst: newest in the
  // VMEM issue order, so the MMAC 0..7 waitcnt staircase tracks the A loads
  // only, and these DRAM/L2 round trips overlap the first K half-loop.
#pragma unroll
  for (int i = 0; i < kStageInstrs / 2; ++i) {
    const int row = (kStageInstrs / 2 + i) * kDummaThreads + lane;  // 256..511
    b_stage2[i] = *reinterpret_cast<const int4*>(b_slice + row * n);
  }
#pragma unroll
  for (int s = 0; s < kSliceSteps / 2; ++s) {
    const int slot = s & (kPrefetchDepth - 1);
    du::dumma::DUFragment<du::dumma::matrix_b, kDummaM, kDummaN, kDummaK,
                          signed char, du::dumma::col_major>
        b_frag;
    load_b_lds8(b_frag, s_b[0], s * kDummaK);
    du::dumma::du_mma_sync(acc0, a_ring[slot], b_frag, acc0);
    // Unconditional refill (s + kPrefetchDepth <= 15 < kSliceSteps): the
    // depth-8 ring closes exactly on step 15; the compiler keeps all 16 A
    // fragments in the accepted iteration-14 pre-loop burst (one
    // global_load_dwordx2 burst + one s_waitcnt vmcnt(15) lgkmcnt(7)).
    load_a_frag8(a_ring[slot], a + k_base + (s + kPrefetchDepth) * kDummaK, k);
  }
  // Drain round 2's loads (rows 256..511) and write them to LDS.  These
  // ds_writes target s_b rows 256..511 only, disjoint from the rows 0..255
  // that MMACs 0..7 read, so they do not race the first half-loop; the
  // __syncthreads makes them visible to every lane before MMAC 8 reads
  // them.  Same bytes, same LDS layout, same k-ascending MMAC order — the
  // result is bit-identical to the single-staged iteration-15 kernel.
#pragma unroll
  for (int i = 0; i < kStageInstrs / 2; ++i) {
    const int row = (kStageInstrs / 2 + i) * kDummaThreads + lane;
    const int8_t* src = reinterpret_cast<const int8_t*>(&b_stage2[i]);
#pragma unroll
    for (int c = 0; c < kDummaN; ++c) {
      s_b[c][row] = src[c];
    }
  }
  __syncthreads();
  // Steps 8..15 -> acc1.  No refill needed: s + kPrefetchDepth >=
  // kSliceSteps, all remaining A fragments were already loaded above.
#pragma unroll
  for (int s = kSliceSteps / 2; s < kSliceSteps; ++s) {
    const int slot = s & (kPrefetchDepth - 1);
    du::dumma::DUFragment<du::dumma::matrix_b, kDummaM, kDummaN, kDummaK,
                          signed char, du::dumma::col_major>
        b_frag;
    load_b_lds8(b_frag, s_b[0], s * kDummaK);
    du::dumma::du_mma_sync(acc1, a_ring[slot], b_frag, acc1);
  }
  // Combine the two chains: acc0 (steps 0..7) += acc1 (steps 8..15) —
  // ascending split sum, bit-identical mod 2^32.
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    acc0.x[i] += acc1.x[i];
  }

  // Store the 16x16 int32 partial to the split plane (verified gfx928
  // accumulator ownership: row = lane & 15, col_mod4 = lane >> 4,
  // frag.x[i] holds column col_mod4 + 4*i).
  const int row = lane & 15;
  const int col_mod4 = lane >> 4;
  int* __restrict__ plane = partials + split_id * (kDummaM * n) + row * n + n0;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    plane[col_mod4 + 4 * i] = acc0.x[i];
  }
}

// Split-K combine kernel (exact (16, 1536, 6144) arm, S = kSplitK): 96
// one-wave blocks; each block sums the S int32 partial planes for its 16x16
// tile in ascending split order (split 0 covers k [0,512), ..., split 11
// covers [5632,6144)) — bit-identical int32 accumulation order to the
// reference (int32 addition is exact mod 2^32, so any ascending split
// grouping of the same products is value-identical) — then applies
// x_scale[row] * weight_scale[col] and stores scaled bf16.  Same-stream
// ordering guarantees the partial kernel's writes are visible; no barrier,
// no atomic, no LDS.
//
// Iteration 14 (vectorized plane reads, fixed split-K=12): each lane now
// owns 4 CONSECUTIVE n columns of one row (row = lane >> 2, tcol4 =
// (lane & 3) << 2) instead of 4 column-aligned elements in different rows
// (e = lane + 64*i), so all S plane reads per lane group into 16-B-aligned
// int4 loads: S int4 global_load_dwordx4 per lane (was 4*S scalar
// global_load_dword spread over four serial load-drain groups per lane, see
// the iteration-13 ISA staircase vmcnt drains) and the bf16 epilogue
// becomes one 8-B-aligned uint64 store (was four 2-B stores).  The
// per-element split sum is unchanged (ascending s = 0..S-1 over the same
// planes) so the output is bit-identical; the partial kernel, grid, LDS
// tile, workspace, identity pack, scalar fallback, and exact int32 order
// are untouched.
// partials     : [S][16][N] int32 planes
// x_scale      : [16, 1] float
// weight_scale : [N, 1] float
// out          : [16, N] bf16 row-major
__global__ __launch_bounds__(kDummaThreads) void
w8a8_dumma_m16n16k32_sk4_combine_kernel(
    const int* __restrict__ partials,
    const float* __restrict__ x_scale,
    const float* __restrict__ weight_scale,
    hip_bfloat16* __restrict__ out,
    int n) {
  const int n0 = static_cast<int>(blockIdx.x) * kDummaN;
  const int lane = static_cast<int>(threadIdx.x);
  const int plane_stride = kDummaM * n;  // int32 per split plane

  // 16-B-aligned int32 base: row * n and n0 are multiples of 4 int32 and
  // tcol4 is a multiple of 4 int32 (16 B).
  const int row = lane >> 2;
  const int tcol4 = (lane & 3) << 2;
  const int idx = row * n + n0 + tcol4;

  int4 acc = *reinterpret_cast<const int4*>(partials + idx);
#pragma unroll
  for (int s = 1; s < kSplitK; ++s) {
    const int4 v = *reinterpret_cast<const int4*>(
        partials + s * plane_stride + idx);
    acc.x += v.x;
    acc.y += v.y;
    acc.z += v.z;
    acc.w += v.w;
  }

  const float xs = x_scale[row];
  const float4 ws = *reinterpret_cast<const float4*>(weight_scale + n0 + tcol4);
  // Repair 2 (compile fix, no design change): __float2bfloat16 returns the
  // INTERNAL __hip_bfloat16, which exposes ~11 implicit integer/float
  // conversion operators (amd_hip_bf16.h) and has no implicit conversion to
  // the public hip_bfloat16 (whose float ctor is explicit, so copy-init
  // `const hip_bfloat16 bN = __float2bfloat16(...)` fails overload
  // resolution with "no viable conversion" — the 4 compile errors).  Each
  // bf16 is therefore direct-initialized from the float scalar via the
  // public explicit hip_bfloat16(float) ctor, whose float_to_bfloat16 RNE
  // is byte-identical to __float2bfloat16's float_2_bfloatraw on gfx928
  // (same rounding and NaN handling), so every output bit is unchanged.
  const hip_bfloat16 b0(static_cast<float>(acc.x) * xs * ws.x);
  const hip_bfloat16 b1(static_cast<float>(acc.y) * xs * ws.y);
  const hip_bfloat16 b2(static_cast<float>(acc.z) * xs * ws.z);
  const hip_bfloat16 b3(static_cast<float>(acc.w) * xs * ws.w);
  // hip_bfloat16 (public type, amd_hip_bfloat16.h) exposes its raw 16-bit
  // bf16 pattern as the public `data` member; it has no conversion to
  // __hip_bfloat16_raw (that operator exists only on the internal
  // __hip_bfloat16 class), so pack the four 16-bit patterns directly.
  const uint64_t packed =
      (static_cast<uint64_t>(b0.data)) |
      (static_cast<uint64_t>(b1.data) << 16) |
      (static_cast<uint64_t>(b2.data) << 32) |
      (static_cast<uint64_t>(b3.data) << 48);
  *reinterpret_cast<uint64_t*>(out + idx) = packed;
}

}  // namespace

// Host launch symbol consumed by csrc/bindings.cpp (extern "C").
// Graph-safe: launches only on the caller-provided stream; performs no
// allocation, compilation, autotuning, weight packing, host/device
// synchronization, or default-stream launch.  `workspace` holds the
// split-K int32 partial planes for the assigned shape (preallocated before
// Graph capture by the API contract).
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
  hip_bfloat16* out_ptr = reinterpret_cast<hip_bfloat16*>(out);
  const int total = m * n;
  const int blocks = (total + kScalarBlockThreads - 1) / kScalarBlockThreads;

  if (m == 16 && n == 1536 && k == 6144) {
    // Assigned shape minimax_tp8_qkv_proj_and_indexer_qk_m16.
    // Iteration 10 (HIP-only occupancy probe): split-K=12 partial GEMM
    // (grid 1,152 = 9.6 blocks/CU; B-only LDS staging at 8,320 B/block ->
    // 7 blocks/CU co-residency by LDS, up from 5 at split-K=8; 16-step K
    // loop; one barrier; combine included in the operator wall).  The guard
    // keeps the paired M=2 (and any other) shape on the generic scalar
    // fallback below.  Workspace 12 x 98,304 B = 1,179,648 B <= the
    // 1,572,864 B (16-plane) capacity allocate_workspace guarantees for
    // this shape.
    const int64_t sk_need =
        static_cast<int64_t>(kSplitK) * kDummaM * n * sizeof(int32_t);
    if (workspace != nullptr && workspace_bytes >= sk_need) {
      int* partials = reinterpret_cast<int*>(workspace);
      const int tiles = n / kDummaN;  // 96
      hipLaunchKernelGGL(w8a8_dumma_m16n16k32_sk4_partial_kernel,
                         dim3(tiles * kSplitK), dim3(kDummaThreads), 0,
                         stream, a, b, partials, n, k);
      hipLaunchKernelGGL(w8a8_dumma_m16n16k32_sk4_combine_kernel,
                         dim3(tiles), dim3(kDummaThreads), 0, stream,
                         partials, x_scale, weight_scale, out_ptr, n);
      return;
    }
    // Workspace too small for kSplitK partial planes: correctness-first
    // scalar fallback for the exact shape (never writes out of bounds).
  }

  // Generic scalar fallback for every unmatched (m, n, k), including the
  // paired M=2 API shape with the same (N, K) and the M=3072 prefill shape.
  hipLaunchKernelGGL(w8a8_scalar_gemm_kernel, dim3(blocks),
                     dim3(kScalarBlockThreads), 0, stream, a, b, x_scale,
                     weight_scale, out_ptr, m, n, k);
}

// Host launch symbol consumed by csrc/bindings.cpp (extern "C").
// Identity device-to-device copy (packed layout == logical [K, N] row-major
// layout).  Runs through the optional zth_w8a8::pack_weight op, outside the
// timed region and outside Graph capture.  For unmatched (K, N) this
// identity copy remains the generic fallback.
extern "C" void launch_pack_w8a8_weight(
    const int8_t* raw_weight,
    const float* weight_scale,
    int8_t* packed_weight,
    float* packed_weight_scale,
    int k,
    int n,
    hipStream_t stream) {
  hipMemcpyAsync(packed_weight, raw_weight,
                 static_cast<size_t>(k) * static_cast<size_t>(n) *
                     sizeof(int8_t),
                 hipMemcpyDeviceToDevice, stream);
  hipMemcpyAsync(packed_weight_scale, weight_scale,
                 static_cast<size_t>(n) * sizeof(float),
                 hipMemcpyDeviceToDevice, stream);
}
// @@end
