// @@variant shape=glm_tp8_shared_gate_up_proj_m16 commit=fa6ea50630e35036b22141edc48f72975fca041d added=2026-08-31
//   median_us=10.82 p90_us=10.87
//   source=glm5-2-dsh-tp8-m16-test1-cb4d262c
// W8A8 INT8 GEMM for Hygon K500SM_AI / gfx928 (worker_3).
//
// Exact-shape split-K DUMMA specialization for
// glm_tp8_shared_gate_up_proj_m16 (M=16, N=512, K=6144) established before
// the generic scalar fallback:
//
//   glm_tp8_shared_gate_up_proj_m16 : (M=16, N=512,  K=6144)
//   glm_tp8_shared_down_proj_m16    : (M=16, N=6144, K=256)
//
// Iteration 1 measured the minimal gfx928 INT8 m16n16k32 DUMMA tile (one
// 64-thread wavefront per 16x16 output tile, grid = N/16 = 32 blocks) at
// 173.6 us median: PMC showed grbm_count == grbm_gui_active with one
// wavefront per CU and 88 of 120 CUs idle, i.e. a latency-bound launch
// geometry where the 192-step K loop exposes its full load->vmcnt->mmac
// chain with no co-resident wave to hide it.
//
// Iteration 2 (architecture round) split K=6144 into 8 uniform 768-element
// slices (24 m16n16k32 steps each) so grid = 32 N-tiles x 8 splits = 256
// independent one-wave zero-barrier blocks = 2.13 blocks/CU, reaching the
// two-blocks-per-CU latency-hiding target with all 120 CUs covered.  Each
// block kept the minimal zero-barrier DUMMA K loop over its own slice and
// stored its int32 partial plane to the caller workspace (8 x 16 x 512 x 4 =
// 256 KiB, within the 16-plane contract budget); a separate 128-thread
// combine kernel (second dispatch) summed the 8 planes in ascending split
// order (bit-exact int32 accumulation order, identical to the unsplit
// kernel) with the fused scale/bf16 epilogue: 76.9 us median (1.29x over the
// 99.3 us Triton Graph baseline).
//
// Iteration 3 (architecture round, this source): the fresh PMC showed the
// two-launch wall is dominated by the combine dispatch, not the GEMM --
// w8a8_dumma_m16_sk8_combine_kernel (grid = 1 block, 128 threads on one CU,
// only 256 KiB of L2-hot planes) profiles at 47.2 us vs 39.4 us for all 256
// partial blocks (86.6 us profiled aggregate vs 76.9 us unprofiled
// operator): a one-CU serial dependent-load-chain cliff worth roughly half
// the wall.  The decision: keep the accepted split-K=8 one-wave zero-barrier
// grid and remove the one-CU combine dispatch by fusing the combine into the
// partial kernel as a validated qkv-style last-arrival tail.  Every block
// ends with __threadfence() + atomicAdd on a monotonic per-N-tile counter in
// the workspace tail; the 8th arriver of each tile (32 of 256 blocks) sums
// the tile's 8 planes in ascending split order (bit-exact int32, identical
// to the unsplit kernel) and applies the fused scale/bf16 epilogue.
// Counters are monotonic ((arrived & 7) == 7 fires exactly once per tile per
// replay), so no reset is needed between Graph replays; one dispatch per
// replay, still Graph-safe.
//
// Repair 1 (in-round, iteration 3) suspected uninitialized counters and added
// a hipStreamIsCapturing-gated one-time hipMemsetAsync; the exact check still
// failed with the identical 7168/8192 mismatch signature, so that diagnosis
// was wrong.
//
// Repair 2 (in-round, iteration 3, this source): the real defect is that the
// fused tail let ALL 64 lanes of every block execute the arrival atomicAdd
// (64 increments per block = 512 per tile per replay), so (arrived & 7) == 7
// fired ~64 times per tile -- the first fires inside the first-arriving
// block, before its 7 siblings had stored their planes, and the premature
// combiner summed unwritten zero planes: exact check failed with
// first_mismatch (0,0) actual=0.0 and 28/32 tiles (7168/8192 elements) wrong,
// the identical signature already documented by the validated hy3 TP8 gate_up
// lineage.  Fix (mirroring that lineage): exactly ONE arrival per block --
// lane 0 fences + atomicAdds + fences and publishes the result through an LDS
// flag with barriers; only the last arriver of each tile runs the combine.
// The one-time counter zero is now a workspace-pointer-static-guarded
// hipMemsetAsync (the validated qkv/gate_up pattern; no capture-status
// query), issued on the caller stream by the first launch with the workspace
// (the pre-capture eager warmup), so the timed replay keeps one dispatch and
// the monotonic counters need no reset between Graph replays.
//
// Iteration 4 (architecture round, this source): A-only staging.  The fresh
// iteration-4 PMC/ISA of the accepted iteration-3 kernel shows the K-loop
// body is the documented decode poison: 16 global_load_ubyte per lane per
// step (8 A + 8 B bytes), a 16-deep s_waitcnt vmcnt cascade, ~20 byte-OR
// reassembly VALU, and ~30 address-carry VALU -- ~64 instructions per step
// with the whole A side of the load chain inside the mmac critical path.
// The DTK du_mma.hpp implementation confirms the m16n16k32 row_major
// matrix_a fragment is 8 CONSECUTIVE bytes at (row = lane&15, k =
// (lane>>4)*8 + {0..7}) -- vectorizable -- while the matrix_b fragment's 8
// bytes are strided by the leading dimension (not vectorizable without
// packing, out of scope).  Decision: keep the accepted split-K=8 geometry
// (grid = 256 blocks = 2.13 blocks/CU, all 120 CUs covered, fused
// last-arrival combine tail) and stage each block's 16x768 A slice into LDS
// once with vectorized 16-B loads (12 global_load_dwordx4 + 12
// ds_write_b128 per lane, one __syncthreads), then read every k-step's A
// fragment from LDS as one 8-B ds_read_b64 (x[0..7] = A[row][k0 +
// (lane>>4)*8 + {0..7}], bit-identical to du_load_matrix_sync).  B keeps
// its direct global path.  A bytes reused: the 12 KiB slice is loaded from
// global exactly once per block and each byte is read 24x from LDS (once
// per k-step); cross-block A/B reuse via L2 is unchanged.  No workspace,
// plane, counter, or combine-tail change; exact int32 accumulation order
// (k-ascending within a split, ascending split sum) is preserved because
// the mmac consumes identical fragment bytes.
//
// Iteration 5 (packed-weight round, this source): the fresh iteration-5
// PMC of the accepted iteration-4 code object (digest 7d5673b9..., vmem_read
// 53152, valu_instructions 255872) shows the K-loop body is still half
// decode poison: per lane per k-step B contributes 8 global_load_ubyte, ~15
// of the 16-deep s_waitcnt vmcnt cascade, ~20 byte-OR reassembly VALU and
// ~30 v_add/v_addc address-carry VALU -- the whole B side of the load chain
// inside the mmac critical path, because the m16n16k32 row_major matrix_b
// fragment's 8 bytes are strided by the leading dimension N and cannot be
// vectorized in the raw [K][N] layout (the iteration-4 round staged A only,
// per its mandate).  Decision (mandated option: packed layout): pack B once,
// outside the timed region and outside Graph capture, into the n-major
// transpose packed[n*K + kk] = raw[kk*n + n] for the exact (k, n) ==
// (6144, 512) pair (byte count unchanged, so captured addresses stay
// valid).  In the packed layout the B fragment is 8 CONTIGUOUS bytes at
// packed[(n0 + lane&15)*k + k0 + (lane>>4)*8 + {0..7}] -- the exact byte
// pattern du_load_matrix_sync(b_frag, b + n0*k + k0, k) reads -- so each
// k-step's B fragment becomes ONE aligned global_load_dwordx2 per lane with
// no byte-OR reassembly and no per-byte waitcnt cascade; the A-side LDS
// stage (iteration 4) is unchanged.  This is the exact load-path pattern
// validated on the K=6144 m16n16k32 gate_up lineage (MiniMax TP8 gate_up,
// accepted iterations 5-16: n-major pack + one 8-byte dwordx2 load per
// fragment).  The generic scalar fallback decodes the same n-major pack
// whenever (n, k) == (512, 6144), so the paired M=2 API shape with the same
// (N, K) stays correct through the packed buffer; every other (K, N) keeps
// the identity pack and the raw row-major decode, and the exact int32
// accumulation order (k-ascending within a split, ascending split sum) is
// unchanged because the mmac consumes identical fragment bytes.
//
// Iteration 7 (pipeline round): double-buffered B transport.  The fresh
// iteration-5 ISA (digest d2ffb4f6...) shows the single-buffered K-loop
// body is exactly global_load_dwordx2 (B) -> ds_read_b64 (A) ->
// s_waitcnt vmcnt(0) lgkmcnt(0) -> v_mmac per step, with no unroll, so the
// next B load cannot issue until the branch returns to the loop head: every
// per-block B load (a zero-reuse HBM stream -- each (tile, split) block
// reads a disjoint 12 KiB B slice, l2_misses 55733 ~ the compulsory 3.57
// MiB traffic) exposes its full ~590-cycle HBM round trip on the critical
// path.  Mandated this round: compare single buffering with double
// buffering (K=6144 >= 1024, L2 hit 55.11% < 70%, doubled LDS budget
// 25,096 B < 48 KiB all hold).  Chosen double buffering: depth-1 register
// prefetch of the packed B fragment (the o_proj-validated pattern -- next
// step's loads issued before the current mmac, rotated in after, compiler
// wait lands at the next fill): step i+1's aligned global_load_dwordx2 is
// issued BEFORE step i's ds_read_b64 + v_mmac, so its HBM latency overlaps
// the previous step's compute instead of sitting on the critical path; the
// final step is peeled so the prefetch address never leaves the block's K
// slice (the packed buffer ends exactly at 512*6144 B; an unconditional
// k0+32 read on the last step would be out of bounds).  No second LDS
// buffer: the LDS-fed B path regressed twice (MiniMax TP8 gate_up 25.9 us;
// our iteration 6 at 19.624 us vs 16.778 us), so packed-global dwordx2 B
// remains the transport.  LDS per block stays 12,544 + 4 B (doubled budget
// 25,096 B < 48 KiB; 2.13 blocks/CU unchanged), barriers per K step stay 0
// (only the single pre-loop staging barrier and the two tail barriers,
// unchanged from iteration 5), and the mmac consumes bit-identical fragment
// bytes in k-ascending order, so the output stays bit-identical to the CPU
// reference (0 mismatches, exact check).
//
// Iteration 9 (MLP round, this source): grouped depth-8 register prefetch.
// The iteration-8 occupancy probe (split-K 15 -> 480 one-wave blocks = exactly
// 4 blocks/CU, integer-balanced) REGRESSED to 21.54 us median, so co-resident
// wave count is not the lever: the kernel is DRAM-latency-bound on per-wave
// in-flight B bytes, not wave-starved (l2_misses ~55.7k = the compulsory 3.57
// MiB A+B traffic; each block still serially waits ~24 ~440-540-cycle HBM
// round trips).  This round raises memory-level parallelism INSIDE the wave
// while keeping the exact 256-block split-K=8 grid and its contiguous 12 KiB
// per-block B slices (no DRAM scatter): the 24-step K loop is regrouped into
// 3 x 8-step groups; before each group's 8 mmacs consume the previously
// loaded B fragments, the NEXT group's 8 packed-B dwordx2 loads are issued
// back-to-back (8 independent loads in flight per lane = 4 KiB per wave), so
// ONE vmcnt fill covers 8 steps and each group's loads have the whole
// preceding 8-mmac burst (~8x the iteration-7 single-step window) to expire
// off the critical path.  Grid, split-K=8, A LDS staging, packed-B transport,
// zero in-loop barriers, workspace, and the fused last-arrival tail are
// untouched; the final group is peeled (no prefetch) so the prefetch address
// never leaves the block's [k_start, k_start + k_per_split) slice, and the
// mmacs consume bit-identical fragment bytes in k-ascending order, so the
// output stays bit-identical to the CPU reference (0 mismatches, exact
// check).
//
// Iteration 11 (layout consolidation, this source): fragment-slot B pack.
// The iteration-10 direct-A probe REGRESSED to 15.06 us median: its ISA shows
// the compiler could not hoist the 48 global A+B dwordx2 loads (register
// pressure) and fell back to a depth-2 rotation with `s_waitcnt vmcnt(8)`
// before EVERY mmac, re-exposing a full global round trip per step -- so the
// A-side LDS staging stays (the 12 KiB stage is served by ~30-40-cycle
// ds_read2_b64 with 2-step lookahead, not by global loads).  The accepted
// iteration-9 kernel's remaining B-side inefficiency is LAYOUT, not MLP:
// the 24 packed-B dwordx2 loads are fully hoisted (all in flight from the
// prologue, per-step vmcnt waits mostly satisfied), but the iteration-5
// n-major pack packed[n*K+kk] places a block's 16 rows x 768 B B-slice in 16
// chunks 6,144 B apart, so each 64-lane load instruction touches 16 strided
// 64-B lines (8 B used each) and the 256 concurrent blocks stream 4,096
// strided DRAM row streams -- the kernel delivers only ~250-270 GB/s
// (counter-derived) where the o_proj-validated fragment-slot pack
// (packed[n_tile][k_step][lane][8]) carried ~461 GB/s on this part.  This
// round re-packs the exact (k, n) == (6144, 512) pair once (outside the
// timed region, same buffer bytes) into b2[(tile*(k/32) + step)*512 + row*32
// + kk]: one 512-B slot per m16n16k32 step, so each lane still reads its 8
// CONSECUTIVE fragment bytes as ONE aligned global_load_dwordx2 (the exact
// du_load_matrix_sync fragment bytes, k-ascending order, bit-exact int32),
// a full 64-lane load instruction covers 512 CONTIGUOUS bytes (8 full 64-B
// lines, 100% line utilization), and each block's 24-slot slice is one
// contiguous 12 KiB stream -- the 256 blocks now stream the 3 MiB pack in
// 256 sequential DRAM row runs.  Grid (256), split-K=8, A LDS staging,
// zero in-loop barriers, the b8 grouped rotation, the workspace, and the
// fused last-arrival tail are untouched; the scalar fallback decodes the
// same fragment-slot pack for (n, k) == (512, 6144) so the paired M=2 API
// shape stays correct through the same packed buffer; l2_misses stay the
// compulsory ~55.6k (same bytes, no over-fetch) while the B second-half
// line L2 hits disappear (l2_hits drop), and the output stays bit-identical
// to the CPU reference (0 mismatches, exact check).
//
// Iteration 20 (stream-shape probe, this source): split-K 8 -> 6.  The
// measured split-K/stream lattice for this shape is {blocks x per-wave KiB
// slice -> median}: 64 x 48 -> 13.64 us (iter 15), 128 x 24 -> 12.124 us
// (iter 13), 256 x 12 -> 11.736 us (accepted, iter 11), 480 x 4 -> 21.54 us
// (iter 8).  Two readings survive every probe so far: (a) the counter-derived
// delivery rate rises with the number of concurrent per-wave streams
// (260 -> 287 -> 302 GB/s at 64 -> 128 -> 256 streams), with diminishing
// elasticity; (b) there is a steep DRAM-efficiency cliff below ~12-KiB
// per-stream length (4-KiB slices collapse to ~165 GB/s even at 480 streams
// and 4 blocks/CU).  Those two readings leave exactly ONE untested lattice
// point that keeps every stream >= 12 KiB with an integer uniform split:
// 192 blocks x 16-KiB slices = split-K 6 (6144/6 = 1024 = 32 k32 steps per
// slice; 32 N-tiles x 6 = 192 blocks = 1.6 blocks/CU, fully resident at
// every plausible occupancy >= 2/CU, so no queued tail at any occupancy).
// 6 is deliberately OUTSIDE the round's trusted set {2,3,4,7,8,10,11,15}:
// the set's untested members are either length-cliffed (10/11 would need
// 8.7-9.6-KiB slices), underfilling (3 -> 96 blocks < 120 CUs), or require
// ragged non-constexpr splits (7/10/11: 192 is not divisible by 7/10/11),
// while 6 divides 192 evenly and keeps the exact 8-step grouped structure
// (32 = 4 x 8).  The decode skill's non-power-of-two sweep family includes
// 6 explicitly, and the validated wqkv_a winner used split-K 6.  Mechanism:
// the 3.55-MiB compulsory traffic, the 3-MiB A L2 re-reads (192 x 16 KiB ==
// 256 x 12 KiB), the 512-B per-instruction B request granularity, the
// fragment-slot pack, the grouped depth-8 rotation, the zero-barrier
// single-wave loop, the plane store, and the fused last-arrival tail are
// all unchanged -- only the per-wave B stream grows 12 -> 16 KiB and the
// concurrent stream count falls 256 -> 192.  Falsifiable: median stays
// 11.736 +/- 0.3 if ~302 GB/s is a count- and length-independent DRAM-mix
// floor for this 3.55-MiB traffic (the session's second plateau-eligible
// flat candidate after iteration 19), drops toward ~11.5-11.65 if the
// longer 16-KiB rows lift delivery to ~305-310 GB/s, or regresses to
// ~12.0-12.1 if count-dominance dominates (falsifying the
// length-compensation model).  Outputs stay bit-identical to the CPU
// reference: the pack is split-agnostic (pure (tile, k-step) slots), each
// split's 32 steps are consumed in k-ascending order, and the fused tail
// sums the 6 planes in ascending split order (int32 addition commutative,
// every partial sum within int32 range as before), firing on
// (arrived % 6) == 5 -- the monotonic modulo fire lands exactly once per
// tile per replay on the true 6th arrival.
//
// The scalar kernel stays the fallback for every unmatched (m, n, k),
// including the paired M=2 API shape with the same (N, K) and the
// down_proj shape.
//
// Math contract:
//   out[m, n] = bf16( int32_dot(a[m, :], b[:, n]) * x_scale[m] * weight_scale[n] )
// The int8 dot product is accumulated exactly in int32 (the largest assigned
// K=6144 keeps every partial sum within int32 range:
// 6144 * 127 * 127 = 99,058,176 < 2^31), so the result is bit-identical to
// the CPU int64 reference regardless of accumulation order.

// Include the installed headers in this known-good order: the DTK's
// du_mma.h is not self-contained when included before the HIP runtime
// headers.
#include <hip/hip_runtime.h>
#include <hip/hip_bfloat16.h>
#include <du_mma.h>

#include <cstdint>

namespace {

constexpr int kScalarBlockThreads = 128;
constexpr int kPackBlockThreads = 256;

// DUMMA tile constants for the exact-shape specialization.  gfx928 INT8
// tensor cores execute m16n16k32 (signed char x signed char -> int32) with
// wavefront=64; each 16x16 output tile is owned by exactly one wavefront.
constexpr int kDummaTileM = 16;
constexpr int kDummaTileN = 16;
constexpr int kDummaTileK = 32;
constexpr int kDummaBlockThreads = 64;  // one 64-thread wavefront per block

// Split-K geometry for the exact-shape specialization (iteration 2;
// iteration 20: 8 -> 6).  K=6144 / 32 = 192 m16n16k32 steps, split into 6
// uniform 32-step slices; grid = (N/16) * kDummaSplitK = 32 * 6 = 192 blocks
// = 1.6 blocks/CU on the 120-CU device (fully resident at every occupancy
// >= 2/CU, no queued tail), each slice a contiguous 16-KiB fragment-slot
// run.  6 divides 192 evenly, so every K boundary stays aligned to the
// DUMMA k=32 unit with zero wasted work.  Probe rationale (iteration 20):
// the measured stream lattice (64 x 48 KiB -> 13.64 us, 128 x 24 KiB ->
// 12.124 us, 256 x 12 KiB -> 11.736 us, 480 x 4 KiB -> 21.54 us) reads
// monotone in concurrent-stream count (260 -> 287 -> 302 GB/s) with a steep
// length cliff below ~12 KiB; 192 x 16 KiB is the only untested lattice
// point that keeps every stream >= 12 KiB with an integer uniform split.
constexpr int kDummaSplitK = 6;
constexpr int kDummaCombineThreads = 128;

// Iteration-4 A-only staging constants (iteration 20: k_per_split 768 ->
// 1024 with split-K 6).  The exact-shape guard pins k=6144, so each split
// slice is k_per_split = 6144/6 = 1024 int8 columns.  The A stage is the
// 16x1024 slice padded to a 1040-column row stride: 1040 is 16-aligned
// (staging stores compile to ds_write_b128) and 8-aligned (the per-k-step
// A-fragment read compiles to one ds_read_b64); the +16 skew bounds LDS
// bank conflicts to 2-way.  16 * 1040 = 16,640 B LDS per block; with ~2-3
// blocks/CU that is ~33-50 KiB of the 64 KiB/CU LDS, so the accepted
// occupancy class is unchanged (2/CU = 33,280 B < 64 KiB still holds).
constexpr int kPerSplitK = kDummaTileK * 32;   // 1024 = 6144 / 6
constexpr int kAStageStride = kPerSplitK + 16; // 1040

// ---------------------------------------------------------------------------
// Generic scalar W8A8 GEMM fallback.
//
// One thread computes exactly one output element out[row, col].  Threads are
// mapped linear = row * n + col, so adjacent lanes sit on adjacent addresses
// in the fastest-changing N dimension: B[kk, col] loads coalesce across a
// wavefront and A[row, kk] broadcasts within a row.  blockDim is a multiple
// of the gfx928 wavefront size 64.
// ---------------------------------------------------------------------------
__global__ __launch_bounds__(kScalarBlockThreads) void
w8a8_gemm_scalar_kernel(
    const int8_t* __restrict__ a,
    const int8_t* __restrict__ b,
    const float* __restrict__ x_scale,
    const float* __restrict__ weight_scale,
    __hip_bfloat16* __restrict__ out,
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
  const int col = static_cast<int>(linear % n);

  const int8_t* __restrict__ a_row = a + static_cast<int64_t>(row) * k;
  // The exact gate_up (n, k) == (512, 6144) uses the iteration-11
  // fragment-slot packed weight b2[(tile*(k/32) + step)*512 + row*32 + kk]
  // (produced once by launch_pack_w8a8_weight outside the timed region); all
  // other shapes keep the identity [K, N] row-major layout.  Decode both so
  // the generic fallback -- including the paired M=2 API shape with the same
  // (N, K) -- stays correct through the same packed pointer.
  const bool fragslot_packed = (n == 512 && k == 6144);
  const int8_t* __restrict__ b_col = b + col;
  int64_t b_stride = static_cast<int64_t>(n);  // identity layout per-kk stride
  if (fragslot_packed) {
    // b2[(col>>4)*(k>>5)*512 + (col&15)*32 + (kk>>5)*512 + (kk&31)]
    b_col = b + static_cast<int64_t>(col >> 4) * ((k >> 5) * 512) +
            static_cast<int64_t>(col & 15) * 32;
    b_stride = 0;  // addressed via the per-kk slot formula below
  }
  int32_t acc = 0;
  for (int kk = 0; kk < k; ++kk) {
    const int64_t b_off = fragslot_packed
                              ? (static_cast<int64_t>(kk >> 5) * 512 + (kk & 31))
                              : kk * b_stride;
    acc += static_cast<int32_t>(a_row[kk]) *
           static_cast<int32_t>(b_col[b_off]);
  }

  const float scaled =
      static_cast<float>(acc) * x_scale[row] * weight_scale[col];
  out[linear] = __float2bfloat16(scaled);
}

// ---------------------------------------------------------------------------
// Exact-shape split-K DUMMA specialization for glm_tp8_shared_gate_up_proj_m16:
// (M=16, N=512, K=6144).  Guarded by an exact (m, n, k) match in
// launch_w8a8_gemm so paired M=2 API shapes and every other shape keep the
// scalar fallback.
//
// Architecture round (iteration 2): the iteration-1 grid of 32 one-wave
// blocks left 88 of 120 CUs idle and each 192-step K loop ran with no
// co-resident wave (PMC: grbm_count == grbm_gui_active, 173.6 us).  Splitting
// K into 8 uniform 768-element slices turns the launch into 256 independent
// zero-barrier one-wave blocks (2.13 blocks/CU), so while one block's
// fragment loads sit on vmcnt, the co-resident block issues its mmac chain.
// Each block keeps the minimal gfx928 INT8 m16n16k32 DUMMA tile with explicit
// int32 accumulation over its own k-slice (split-major block order keeps the
// first 32 blocks on one 384 KiB B slice: L2-hot), then stores the int32
// partial plane to the workspace using the verified gfx928 accumulator lane
// ownership (row = lane & 15, col_mod4 = lane >> 4, acc_frag.x[i] -> columns
// col_mod4 + 4*i).
//
// Iteration 3 fused combine (this source): the standalone 128-thread combine
// dispatch profiled at 47.2 us (grid = 1 block on one CU) -- the largest
// single component of the 76.9 us wall.  The combine is now a last-arrival
// tail inside this kernel: after the plane store each block publishes one
// monotonic arrival on counters[tile] (tile = the block's N-tile); the block
// whose arrival makes (arrived & 7) == 7 is the last of the tile's 8 splits,
// so it sums the 8 planes in ascending split order (bit-exact int32
// accumulation order, identical to the unsplit kernel) and applies the fused
// scale/bf16 epilogue.  Repair 2: exactly ONE arrival per block (lane 0 fence
// + atomicAdd + LDS flag + barriers; the round-3 draft's all-64-lane atomicAdd
// fired the combine prematurely, before sibling planes were stored -- see the
// tail comment).  Only 32 of 256 blocks run the combine; counters live in the
// workspace tail, are zeroed once per workspace by launch_w8a8_gemm, and
// never need a reset (monotonic across Graph replays).  Single Graph-safe
// dispatch per replay.
// ---------------------------------------------------------------------------
__global__ __launch_bounds__(kDummaBlockThreads) void
w8a8_dumma_m16_sk8_partial_kernel(
    const int8_t* __restrict__ a,
    const int8_t* __restrict__ b,
    int32_t* __restrict__ partials,  // [kDummaSplitK][M][N] int32 planes
    const float* __restrict__ x_scale,      // [M]
    const float* __restrict__ weight_scale, // [N]
    __hip_bfloat16* __restrict__ out,       // [M][N]
    int32_t* __restrict__ counters,         // [N/16] monotonic arrival counters
    int n,
    int k) {
  const int lane = static_cast<int>(threadIdx.x);
  const int n_tiles = n / kDummaTileN;
  const int split = static_cast<int>(blockIdx.x) / n_tiles;
  const int tile = static_cast<int>(blockIdx.x) % n_tiles;
  const int n0 = tile * kDummaTileN;

  const int k_per_split = k / kDummaSplitK;  // 1024, a multiple of 32
  const int k_start = split * k_per_split;

  // Iteration 4 (A-only staging): stage this block's 16 x k_per_split A
  // slice into LDS once with vectorized 16-B loads, then serve every
  // k-step's A fragment from LDS as one 8-B read.  The DTK du_mma.hpp
  // row_major matrix_a loader maps lane l to 8 CONSECUTIVE bytes
  // A[lane&15][k0 + ((lane>>4)<<3) + {0..7}], so the staged fragment bytes
  // are bit-identical to du_load_matrix_sync while the accepted kernel's 8
  // global ubyte loads + waitcnt cascade + byte-OR reassembly for A leave
  // the per-step critical path (see the header comment for the iteration-4
  // ISA evidence).  B (iteration 5) is the n-major packed weight
  // (packed[n*K+kk] = raw[kk*n+n] for (k, n) == (6144, 512), produced once
  // by launch_pack_w8a8_weight outside the timed region), whose fragment is
  // 8 CONTIGUOUS bytes at packed[(n0 + lane&15)*k + k0 + (lane>>4)*8 +
  // {0..7}], loaded below as one aligned global_load_dwordx2 per lane
  // (bit-identical to du_load_matrix_sync on the packed layout).
  __shared__ int8_t s_a[kDummaTileM * kAStageStride];
  {
    // 16 x 1024 B = 16,384 B = 1024 x 16-B groups; 64 lanes x 16 = 1024.
    constexpr int kGroupsPerRow = kPerSplitK / 16;  // 64
    constexpr int kStageVecs =
        (kDummaTileM * kGroupsPerRow) / kDummaBlockThreads;  // 16
    static_assert(kDummaTileM * kGroupsPerRow ==
                      kDummaBlockThreads * kStageVecs,
                  "A slice must divide evenly across the staging lanes");
#pragma unroll
    for (int t = 0; t < kStageVecs; ++t) {
      const int g = lane + kDummaBlockThreads * t;
      const int row = g / kGroupsPerRow;
      const int c16 = (g % kGroupsPerRow) * 16;
      // 16-aligned: a base, k_start (multiple of 1024), row*k (k=6144) and
      // c16 are all multiples of 16, so this is one global_load_dwordx4.
      const int4 v =
          *reinterpret_cast<const int4*>(a + k_start + row * k + c16);
      // 16-aligned in LDS: kAStageStride = 1040 is a multiple of 16, so this
      // is one ds_write_b128.
      *reinterpret_cast<int4*>(&s_a[row * kAStageStride + c16]) = v;
    }
  }
  __syncthreads();  // every lane reads rows staged by other lanes below

  du::dumma::DUFragment<du::dumma::matrix_a, kDummaTileM, kDummaTileN,
                        kDummaTileK, signed char, du::dumma::row_major>
      a_frag;
  du::dumma::DUFragment<du::dumma::matrix_b, kDummaTileM, kDummaTileN,
                        kDummaTileK, signed char, du::dumma::row_major>
      b_frag;
  du::dumma::DUFragment<du::dumma::accumulator, kDummaTileM, kDummaTileN,
                        kDummaTileK, int>
      acc_frag;
  du::dumma::du_fill_fragment(acc_frag, 0);

  const int a_row = lane & (kDummaTileM - 1);
  const int a_kq = (lane >> 4) << 3;  // 8 consecutive k bytes per lane
  // Iteration 5 (packed weight): the same lane bits select the B fragment
  // of the n-major packed weight -- n = lane & 15, k offset (lane>>4)*8 --
  // so b_row and b_kq coincide with a_row / a_kq.
  const int b_row = lane & (kDummaTileM - 1);
  const int b_kq = (lane >> 4) << 3;
  // Iteration 11 (layout consolidation, this source): the per-step packed-B
  // address is loop-invariant except the step index, so precompute its base
  // here.  The packed weight is now the o_proj-validated fragment-slot
  // layout b2[(tile*(k/32) + step)*512 + row*32 + kk] (one 512-B slot per
  // m16n16k32 step): lane l still owns the 8 CONSECUTIVE bytes
  // B[n0 + lane&15][k0 + (lane>>4)*8 + {0..7}] -- the exact
  // du_load_matrix_sync fragment -- but a full 64-lane load instruction now
  // covers 512 CONTIGUOUS bytes (8 full 64-B lines, 100% line utilization)
  // instead of 16 rows x 8 B strided 6,144 B apart under the iteration-5
  // n-major pack.  Each block's 24-slot slice is one contiguous 12 KiB
  // stream (the contiguity the iteration-9 hypothesis assumed), so the 256
  // blocks stream the 3 MiB pack in 256 sequential DRAM row runs instead of
  // 4,096 strided row streams -- the same fragment-slot transport that
  // carried the validated o_proj kernel at ~461 GB/s effective read
  // bandwidth on this part (vs ~250-270 GB/s counter-derived for this
  // kernel's strided n-major pack).  Current and prefetch addresses stay
  // 8-byte aligned (b_base and kBFragStep are multiples of 8).
  constexpr int kBFragStep = kDummaTileN * kDummaTileK;  // 512 B per k32 slot
  const int64_t b_base =
      (static_cast<int64_t>(tile) * (k >> 5) +
       split * (k_per_split / kDummaTileK)) *
          kBFragStep +
      static_cast<int64_t>(b_row) * kDummaTileK + b_kq;

  // Iteration 7 (pipeline round): double-buffered B transport.  The
  // iteration-5 K-loop body (one global_load_dwordx2 (B) + one ds_read_b64
  // (A) + s_waitcnt vmcnt(0) lgkmcnt(0) + one v_mmac per step, no unroll)
  // exposes the full ~590-cycle HBM round trip of every B load on the
  // critical path because the next load cannot issue until the branch
  // returns to the loop head, and B is a zero-reuse HBM stream (disjoint
  // 12 KiB slice per (tile, split) block).  This round is the mandated
  // single-vs-double buffering comparison.  Double buffering here = one
  // register-resident B fragment in flight (depth-1 software prefetch, the
  // o_proj-validated pattern): step i+1's packed dwordx2 is issued BEFORE
  // step i's ds_read + mmac and rotated in after, so its HBM latency
  // overlaps the previous step's compute and the compiler places the vmcnt
  // wait at the next fill point.  LDS stays 12,544 + 4 B per block (no
  // second LDS buffer: the LDS-fed B path regressed twice -- MiniMax TP8
  // gate_up 25.9 us and our iteration 6 at 19.624 us -- while packed-global
  // dwordx2 B is the validated transport), so the doubled LDS budget
  // (25,096 B) stays below 48 KiB and the 2.13 blocks/CU geometry is
  // unchanged.  Barriers per K step: 0 -- the only __syncthreads are the
  // single pre-loop A-staging barrier and the two tail barriers, unchanged
  // from iteration 5.  The final step is peeled so the prefetch address
  // never leaves the block's [k_start, k_start + k_per_split) slice (the
  // packed buffer ends exactly at 512*6144 B; an unconditional k0+32 read
  // on the last step would read past its end).  The mmac consumes
  // bit-identical fragment bytes in k-ascending order, so the output stays
  // bit-identical to the CPU reference (0 mismatches, exact check).
  // Iteration 9 (MLP round, this source): grouped depth-8 register prefetch.
  // The iteration-8 occupancy probe (split-K 15 -> 480 one-wave blocks =
  // exactly 4 blocks/CU, integer-balanced, no tail wave) REGRESSED to 21.54
  // us median, so co-resident wave count is NOT the lever: this kernel is
  // DRAM-latency-bound on per-wave in-flight B bytes, not wave-starved
  // (l2_misses ~55.7k = the compulsory 3.57 MiB A+B traffic; every block
  // still serially waits ~24 ~440-540-cycle HBM round trips).  This round
  // raises memory-level parallelism INSIDE the wave while keeping the exact
  // 256-block split-K=8 grid and its contiguous 12 KiB per-block B slices
  // (no DRAM scatter): the 24-step K loop is regrouped into 3 x 8-step
  // groups; before each group's 8 mmacs consume the previously loaded B
  // fragments, the NEXT group's 8 packed-B dwordx2 loads are issued
  // back-to-back (8 independent loads in flight per lane = 4 KiB per wave),
  // so ONE vmcnt fill covers 8 steps and each group's loads have the whole
  // preceding 8-mmac burst (~8x the iteration-7 single-step window) to expire
  // off the critical path.  Grid, split-K=8, A LDS staging, packed-B
  // transport, zero in-loop barriers, workspace, and the fused last-arrival
  // tail are untouched; the final group is peeled (no prefetch) so the
  // prefetch address never leaves the block's [k_start, k_start +
  // k_per_split) slice (the packed buffer ends exactly at 512*6144 B), and
  // the mmacs consume bit-identical fragment bytes in k-ascending order, so
  // the output stays bit-identical to the CPU reference (0 mismatches,
  // exact check).
  constexpr int kUnroll = 8;
  constexpr int kGroupK = kUnroll * kDummaTileK;  // 256 = 8 k32 steps
  static_assert(kPerSplitK % kGroupK == 0,
                "32 steps must split evenly into 8-step groups");
  uint64_t b8[kUnroll];
  // b_first is the base of this block's 32-slot (16 KiB) contiguous
  // fragment-slot slice; each slot is kBFragStep = 512 B.  Declared at
  // function scope (repair 2: the draft scoped it inside the prologue block,
  // so the grouped loop below could not see it) so the prologue, the grouped
  // loop and the peeled tail all reuse the same base.
  const int64_t b_first = b_base;
  {
    // Prologue: issue group 0 (the split slice's first 8 fragment slots) so
    // its round trip is in flight before the first group fill.
#pragma unroll
    for (int i = 0; i < kUnroll; ++i) {
      b8[i] = *reinterpret_cast<const uint64_t*>(b + b_first + i * kBFragStep);
    }
  }
  int k0 = k_start;
  int64_t b_off = 0;  // B byte offset of the current 8-step group's first slot
  // Three full groups: issue the NEXT group's 8 B loads before this group's
  // 8 mmacs (each next-group load is in flight during the whole 8-mmac
  // burst) and rotate the registers in after; the compiler places the vmcnt
  // fill at the group head (the iteration-7 wait placement on this shape).
  for (int g = 0; g < kPerSplitK / kGroupK - 1; ++g) {
    uint64_t b8_next[kUnroll];
#pragma unroll
    for (int i = 0; i < kUnroll; ++i) {
      // 8-aligned: b_first, b_off, kUnroll*kBFragStep, i*kBFragStep and b_kq
      // are all multiples of 8, so each of these is one global_load_dwordx2
      // over 8 consecutive bytes of one 512-B fragment slot (8 full 64-B
      // lines per full 64-lane load instruction).
      b8_next[i] = *reinterpret_cast<const uint64_t*>(
          b + b_first + b_off + kUnroll * kBFragStep + i * kBFragStep);
    }
#pragma unroll
    for (int i = 0; i < kUnroll; ++i) {
      // A fragment from LDS: one 8-B read reproduces the exact
      // du_load_matrix_sync bytes (x[0..7] = A[row][k0 + a_kq + {0..7}]);
      // the offset is 8-aligned (kAStageStride and a_kq are multiples of 8
      // and (k0 - k_start) + i*kDummaTileK is a multiple of 32), so this is
      // one ds_read_b64.
      const uint64_t a8 = *reinterpret_cast<const uint64_t*>(
          &s_a[a_row * kAStageStride + (k0 - k_start) + i * kDummaTileK +
               a_kq]);
      __builtin_memcpy(a_frag.x, &a8, sizeof(a8));
      // B fragment from the iteration-11 fragment-slot pack: 8 contiguous
      // bytes at b2[(tile*(k/32) + 24*split + s)*512 + b_row*32 + b_kq] for
      // step s of the block's split slice (i = 0..7) -- the exact bytes
      // du_load_matrix_sync(b_frag, b + n0*k + k0, k) reads from the packed
      // layout, delivered as ONE aligned global_load_dwordx2 per lane
      // instead of 8 strided global_load_ubyte + waitcnt cascade + byte-OR
      // reassembly.
      __builtin_memcpy(b_frag.x, &b8[i], sizeof(b8[i]));
      du::dumma::du_mma_sync(acc_frag, a_frag, b_frag, acc_frag);
    }
#pragma unroll
    for (int i = 0; i < kUnroll; ++i) {
      b8[i] = b8_next[i];
    }
    k0 += kGroupK;
    b_off += kUnroll * kBFragStep;  // next 8-step group's first slot
  }
  {
    // Peeled final group (no prefetch): consumes the last group loaded by the
    // loop's final issue (b8 holds the split slice's slots 24 .. 31).
#pragma unroll
    for (int i = 0; i < kUnroll; ++i) {
      const uint64_t a8 = *reinterpret_cast<const uint64_t*>(
          &s_a[a_row * kAStageStride + (k0 - k_start) + i * kDummaTileK +
               a_kq]);
      __builtin_memcpy(a_frag.x, &a8, sizeof(a8));
      __builtin_memcpy(b_frag.x, &b8[i], sizeof(b8[i]));
      du::dumma::du_mma_sync(acc_frag, a_frag, b_frag, acc_frag);
    }
  }

  // Register-only partial store with the verified gfx928 accumulator
  // ownership: lane l owns row = l & 15 and columns col_mod4 + 4*i with
  // col_mod4 = l >> 4, so acc_frag.x[i] belongs to output column
  // n0 + col_mod4 + 4*i of row (l & 15).
  const int plane = kDummaTileM * n;  // stride between split planes
  int32_t* plane_p = partials + split * plane;
  const int row = lane & (kDummaTileM - 1);
  const int col_mod4 = lane >> 4;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const int col = n0 + col_mod4 + 4 * i;
    plane_p[row * n + col] = acc_frag.x[i];
  }

  // Fused last-arrival combine tail (iteration 3; repair 2: exactly ONE
  // arrival per block).  The iteration-3 draft let all 64 lanes of every
  // block execute the arrival atomicAdd (64 increments per block = 512 per
  // tile per replay), so (arrived & 7) == 7 fired ~64 times per tile -- the
  // first fires inside the first-arriving block, before its 7 siblings had
  // stored their planes, and the premature combiner summed unwritten zero
  // planes (exact check: 7168/8192 mismatched, first_mismatch (0,0)
  // actual=0.0 -- the identical signature documented by the validated hy3
  // TP8 gate_up lineage).  Fix: exactly one arrival per block -- lane 0
  // fences (release), atomicAdds, fences (acquire) and publishes the result
  // through LDS; the rest of the block waits at a barrier, and only the last
  // arriver of each tile (32 of 192 blocks) runs the combine.  The monotonic
  // modulo fire (arrived % kDummaSplitK) == kDummaSplitK-1 then lands
  // exactly once per tile per replay, on the true 6th arrival (iteration 20:
  // split-K 6 is non-power-of-two, so the power-of-two mask of iteration 3
  // becomes the generic modulo), after all 6 planes of the tile are visible;
  // the counters are zeroed once per
  // workspace by launch_w8a8_gemm (repair 2: static-pointer-guarded
  // hipMemsetAsync, the validated qkv/gate_up pattern) and stay monotonic
  // across Graph replays.
  __syncthreads();
  __shared__ int s_is_last;
  if (lane == 0) {
    __threadfence();  // release: this block's plane stores visible to the
                      // observer of this block's arrival atomic
    const int arrived = atomicAdd(&counters[tile], 1);
    __threadfence();  // acquire: reads below see every sibling plane store
                      // that was released before its arrival atomic
    // Iteration 20: kDummaSplitK = 6 is non-power-of-two, so the modulo-6
    // fire replaces the power-of-two mask of iteration 3; the monotonic
    // counters still land the fire exactly once per tile per replay on the
    // true 6th arrival (counters zeroed once per workspace, monotonic
    // across Graph replays).
    s_is_last =
        ((arrived % kDummaSplitK) == (kDummaSplitK - 1)) ? 1 : 0;
  }
  __syncthreads();
  if (s_is_last == 0) {
    return;
  }
  {
    // Last arriver of this tile: sum the kDummaSplitK planes in ascending
    // split order -- bit-exact int32 accumulation order, identical to the
    // unsplit kernel -- and fuse the scale/bf16 epilogue.
    const int e_local = lane * 4;  // 64 lanes x 4 = 256 tile elements
    const int tile_row = e_local >> 4;
    const int col0 = n0 + (e_local & 15);
    int32_t v[kDummaSplitK][4];
#pragma unroll
    for (int s = 0; s < kDummaSplitK; ++s) {
      const int32_t* p = partials + s * plane + tile_row * n + col0;
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        v[s][i] = p[i];
      }
    }
    const float xs = x_scale[tile_row];
    __hip_bfloat16* orow = out + tile_row * n + col0;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      int32_t acc = 0;
#pragma unroll
      for (int s = 0; s < kDummaSplitK; ++s) {
        acc += v[s][i];
      }
      const float scaled =
          static_cast<float>(acc) * xs * weight_scale[col0 + i];
      orow[i] = __float2bfloat16(scaled);
    }
  }
}

// Superseded by the iteration-3 fused last-arrival tail inside
// w8a8_dumma_m16_sk8_partial_kernel (single dispatch per replay); retained
// only for reference.  The iteration-2 version of this kernel (grid = 1
// block, 128 threads on one CU, serial per-element 8-plane dependent load
// chain) profiled at 47.2 us -- the largest single component of the 76.9 us
// operator wall.
__global__ __launch_bounds__(kDummaCombineThreads) void
w8a8_dumma_m16_sk8_combine_kernel(
    const int32_t* __restrict__ partials,  // [kDummaSplitK][M][N] int32
    const float* __restrict__ x_scale,     // [M]
    const float* __restrict__ weight_scale,  // [N]
    __hip_bfloat16* __restrict__ out,      // [M][N]
    int n) {
  const int total = kDummaTileM * n;  // 8192 output elements
  const int plane = kDummaTileM * n;  // stride between split planes
  for (int e = static_cast<int>(threadIdx.x); e < total;
       e += kDummaCombineThreads) {
    int32_t acc = 0;
#pragma unroll
    for (int s = 0; s < kDummaSplitK; ++s) {
      acc += partials[s * plane + e];
    }
    const int row = e / n;
    const int col = e - row * n;
    const float scaled =
        static_cast<float>(acc) * x_scale[row] * weight_scale[col];
    out[e] = __float2bfloat16(scaled);
  }
}

// ---------------------------------------------------------------------------
// Device-to-device packing kernels (optional pack_weight op, outside the
// timed region).  Iteration 5 introduced the packed weight for the exact
// gate_up (k, n) == (6144, 512) pair so every m16n16k32 B fragment is 8
// contiguous bytes; iteration 11 re-packs that pair into the o_proj-validated
// fragment-slot layout (w8a8_pack_fragslot_i8_kernel) so each full 64-lane
// load instruction reads 512 contiguous bytes (see the kernel comment); every
// other (K, N) keeps the identity copy of raw_weight [K, N] int8 and
// weight_scale [N, 1] fp32, the generic fallback layout.
// ---------------------------------------------------------------------------
__global__ __launch_bounds__(kPackBlockThreads) void
w8a8_pack_identity_kernel(
    const int8_t* __restrict__ src,
    int8_t* __restrict__ dst,
    int64_t count) {
  const int64_t i =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < count) {
    dst[i] = src[i];
  }
}

__global__ __launch_bounds__(kPackBlockThreads) void
w8a8_pack_identity_scale_kernel(
    const float* __restrict__ src,
    float* __restrict__ dst,
    int64_t count) {
  const int64_t i =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < count) {
    dst[i] = src[i];
  }
}

// Fragment-slot pack (iteration 11) for the exact gate_up (k, n) ==
// (6144, 512) pair: dst[(tile*(k/32) + step)*512 + row*32 + kk] =
// src[kk*n + tile*16 + row], with tile = n_idx/16, row = n_idx%16,
// step = k_idx/32, kk = k_idx%32.  Runs once, outside the timed region and
// outside Graph capture, into the same caller-owned packed_weight buffer
// (byte count unchanged, so captured addresses stay valid).  Each
// m16n16k32 B-fragment lane still owns 8 CONTIGUOUS bytes at
// b2[(tile*(k/32) + 24*split + s)*512 + (lane&15)*32 + (lane>>4)*8 + {0..7}]
// (one 512-B slot per k32 step, see w8a8_dumma_m16_sk8_partial_kernel), so
// each fragment stays ONE aligned global_load_dwordx2 per lane -- while a
// full 64-lane load instruction now covers 512 contiguous bytes (8 full
// 64-B lines, 100% line utilization) and each block's 24-slot slice is one
// contiguous 12 KiB stream instead of 16 rows x 768 B strided 6,144 B
// apart under the iteration-5 n-major pack.  This is the o_proj-validated
// fragment-slot transport that carried ~461 GB/s effective read bandwidth on
// this part.  The identity copy remains the fallback for every other (K, N).
__global__ __launch_bounds__(kPackBlockThreads) void
w8a8_pack_fragslot_i8_kernel(
    const int8_t* __restrict__ src,
    int8_t* __restrict__ dst,
    int64_t count,
    int n,
    int k) {
  const int64_t i =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < count) {
    const int n_idx = static_cast<int>(i / k);
    const int k_idx = static_cast<int>(i - static_cast<int64_t>(n_idx) * k);
    const int tile = n_idx >> 4;  // n_idx / kDummaTileN (16)
    const int row = n_idx & 15;   // n_idx % kDummaTileN
    const int step = k_idx >> 5;  // k_idx / kDummaTileK (32)
    const int kk = k_idx & 31;    // k_idx % kDummaTileK
    const int64_t dst_idx =
        (static_cast<int64_t>(tile) * (k >> 5) + step) * 512 +
        static_cast<int64_t>(row) * 32 + kk;
    dst[dst_idx] = src[static_cast<int64_t>(k_idx) * n + n_idx];
  }
}

}  // namespace

// Optional out-of-timed-region weight pack (stable host symbol used by
// csrc/bindings.cpp).  Bootstrap: identity device-to-device copy.
// Iteration 5 packed the exact gate_up (k, n) == (6144, 512) pair into the
// n-major transpose packed[n*K+kk] = raw[kk*n+n]; iteration 11 re-packs that
// pair into the o_proj-validated fragment-slot layout
// b2[(tile*(k/32) + step)*512 + row*32 + kk] (one contiguous 512-B slot per
// m16n16k32 step; byte count unchanged); every other (K, N) keeps the
// identity copy.  The matching GEMM interpretation is selected by the exact
// (m, n, k) guard in launch_w8a8_gemm and by the scalar fallback's
// (n, k) == (512, 6144) decode.
extern "C" void launch_pack_w8a8_weight(
    const int8_t* raw_weight,
    const float* weight_scale,
    int8_t* packed_weight,
    float* packed_weight_scale,
    int k,
    int n,
    hipStream_t stream) {
  const int64_t weight_count = static_cast<int64_t>(k) * n;
  const int weight_grid = static_cast<int>(
      (weight_count + kPackBlockThreads - 1) / kPackBlockThreads);
  if (k == 6144 && n == 512) {
    hipLaunchKernelGGL(
        w8a8_pack_fragslot_i8_kernel,
        dim3(weight_grid),
        dim3(kPackBlockThreads),
        0,
        stream,
        raw_weight,
        packed_weight,
        weight_count,
        n,
        k);
  } else {
    hipLaunchKernelGGL(
        w8a8_pack_identity_kernel,
        dim3(weight_grid),
        dim3(kPackBlockThreads),
        0,
        stream,
        raw_weight,
        packed_weight,
        weight_count);
  }

  const int64_t scale_count = static_cast<int64_t>(n);
  const int scale_grid = static_cast<int>(
      (scale_count + kPackBlockThreads - 1) / kPackBlockThreads);
  hipLaunchKernelGGL(
      w8a8_pack_identity_scale_kernel,
      dim3(scale_grid),
      dim3(kPackBlockThreads),
      0,
      stream,
      weight_scale,
      packed_weight_scale,
      scale_count);
}

// Timed W8A8 GEMM entry point (stable host symbol used by csrc/bindings.cpp).
// Graph-safe: launches only on the caller-provided current HIP stream, with
// no allocation, compilation, autotuning, packing, host synchronization,
// device synchronization, or default-stream use.  Only the caller-provided
// `out` and `workspace` are touched; the split-K specialization uses 6
// int32 planes (6 * 16 * N * 4 bytes) of the workspace as partial
// accumulators plus a 32-int monotonic arrival-counter tail.  Every launch
// overwrites every partial tile, and the counters are zeroed once per
// workspace by the first launch (repair 2: static-pointer-guarded async
// hipMemsetAsync on the caller stream), after which they are monotonic and
// need no reset between Graph replays.
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

  // -------------------------------------------------------------------------
  // Exact-shape specialization for glm_tp8_shared_gate_up_proj_m16:
  // (M=16, N=512, K=6144).  The guard requires all three dimensions so the
  // paired M=2 API shape with the same (N, K) and every other (m, n, k) still
  // reach the scalar fallback below.  For this shape `b` is the iteration-11
  // fragment-slot packed weight b2[(tile*(k/32) + step)*512 + row*32 + kk]
  // produced once by launch_pack_w8a8_weight outside the timed region, and
  // the scalar fallback decodes the same pack for (n, k) == (512, 6144).  The workspace
  // contract for this shape guarantees 16 int32 planes (512 KiB); the
  // split-K=6 path needs 6 planes (192 KiB) plus a 128-byte counter tail,
  // and falls back to the scalar kernel defensively if the caller ever
  // passes a smaller buffer.  Single dispatch per replay: the fused
  // last-arrival combine tail (iteration 3, repair 2) produces the final
  // scaled bf16 output inside the partial kernel, and the counter tail is
  // zeroed once per workspace (repair 2) without adding a dispatch to the
  // timed replay.
  // -------------------------------------------------------------------------
  if (m == 16 && n == 512 && k == 6144) {
    const int64_t partial_bytes =
        static_cast<int64_t>(kDummaSplitK) * kDummaTileM * n *
        static_cast<int64_t>(sizeof(int32_t));
    const int64_t counter_bytes =
        static_cast<int64_t>(n / kDummaTileN) * sizeof(int32_t);
    if (workspace_bytes >= partial_bytes + counter_bytes) {
      int32_t* partials = reinterpret_cast<int32_t*>(workspace);
      int32_t* counters = reinterpret_cast<int32_t*>(
          reinterpret_cast<char*>(partials) + partial_bytes);
      // Repair 2: the fused last-arrival tail requires every per-tile
      // arrival counter to start at a multiple of kDummaSplitK (0), so the
      // modulo-6 fire lands only on the true 6th arrival of each tile per
      // replay.  The caller workspace is torch.empty (not guaranteed
      // zeroed), so zero the 128-byte counter tail once per workspace with
      // the validated workspace-pointer-static guard: the first launch with
      // this workspace (the pre-capture eager warmup in the normal flow)
      // issues an async hipMemsetAsync on the caller stream, and the static
      // prevents any later launch from repeating it, so the timed replay
      // keeps exactly one dispatch.  Once initialized the counters are
      // monotonic (6 increments per tile per replay) and never need a reset
      // between Graph replays.
      static const void* s_zeroed_counters_ws = nullptr;
      if (s_zeroed_counters_ws != workspace) {
        hipMemsetAsync(counters, 0, counter_bytes, stream);
        s_zeroed_counters_ws = workspace;
      }
      const int grid = (n / kDummaTileN) * kDummaSplitK;
      hipLaunchKernelGGL(
          w8a8_dumma_m16_sk8_partial_kernel,
          dim3(grid),
          dim3(kDummaBlockThreads),
          0,
          stream,
          a,
          b,
          partials,
          x_scale,
          weight_scale,
          reinterpret_cast<__hip_bfloat16*>(out),
          counters,
          n,
          k);
      return;
    }
  }

  const int64_t total = static_cast<int64_t>(m) * n;
  const int grid = static_cast<int>(
      (total + kScalarBlockThreads - 1) / kScalarBlockThreads);
  hipLaunchKernelGGL(
      w8a8_gemm_scalar_kernel,
      dim3(grid),
      dim3(kScalarBlockThreads),
      0,
      stream,
      a,
      b,
      x_scale,
      weight_scale,
      reinterpret_cast<__hip_bfloat16*>(out),
      m,
      n,
      k);
}
// @@end
