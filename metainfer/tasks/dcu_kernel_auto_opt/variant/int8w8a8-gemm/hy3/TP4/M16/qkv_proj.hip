// @@variant shape=hy3_tp4_qkv_proj_m16 commit=7d6959345400cb268cfd9289338d0891db2297f9 added=2026-08-26
//   median_us=26.05 p90_us=26.09
//   source=hy3-dsh-tp4-m16-1-7f1fb1d1
// MetaInfer W8A8 INT8 GEMM - worker_0 (iteration 17, fused combine tail).
//
// Assigned shape : hy3_tp4_qkv_proj_m16, M=16, N=2560, K=4096, gfx928
//                  (K500SM_AI, wavefront=64, 64 KiB LDS/CU, 120 CUs).
//
// Round history (all timings are full-operator Graph median / p90 vs the
// fixed 80.19us Triton baseline; exact int32 accumulation has been preserved
// bit-for-bit in every round):
//   R1-R2  direct per-byte fragment loads (grid/occupancy variants):
//          ~133-141us; each K step issued 16 scalar global ubyte loads, each
//          followed by its own s_waitcnt vmcnt(N) cascade before the single
//          v_mmac_i32_16x16x32_i8.
//   R3     workspace split-K pair, one 64-thread wavefront per block, grid
//          (160 tiles x SPLIT_K): 124.92/125.13us at split-K=2 (320 blocks =
//          2.67 blocks/CU). Falsified grid parallelism as the lever:
//          vmem_read stayed 327,680 (16 per-lane ubyte loads per K step,
//          invariant to split-K) and per-step wall time ~1.95us was the
//          serialized load->pack chain, not bandwidth or MMA issue.
//   R4     double-buffered LDS pipeline (s_a[2][16][32] + s_b[2][32][20],
//          2,304 B/block): one 8-byte vector load per lane per tile per K
//          step; 54.21/54.32us (1.48x, current best). PMC: vmem_read
//          327,680 -> 40,960; LDS 225,280 instr / 491,520 bank conflicts
//          (the matrix_b fragment's 8 strided ds_read_u8 at ldm 20 are
//          4-way conflicted); ~15-VALU byte-OR reassembly per step; 13
//          waitcnts/step; 76.2% L2 hit rate; per-step wall time still
//          ~0.85us.
//   R5     n-major packed weight + DIRECT vectorized 8-byte B fragment loads
//          (A-only LDS staging): 55.73/55.84us - flat/slightly WORSE than
//          R4. Removed the B LDS staging, the 8 conflicted ds_read_u8, the
//          ~15-VALU OR chain and 2-3 waitcnts per step and nothing moved:
//          the limiter is NOT the fragment read path (LSU/L2 request
//          throughput or MMA issue). It is the one-wave geometry: 2 loads +
//          1 MMA per step cannot cover the per-step load->stage->fragment->
//          MMA chain, and only 2.67 waves/CU are resident to absorb it.
//
// Round-6 change (staging family choice from L2/VMEM evidence - chosen:
// A+B LDS): port the validated shape-sibling wqkv_a (M=16, K=4096, N=1536;
// 17.3us median, 570 GB/s logical request bandwidth) split-K=10 /
// 4-wavefront / stage-K=64 / block-N=64 architecture to N=2560:
//   * block-N=64: one block computes a 16x64 output tile; four 64-thread
//     wavefronts each own one 16x16 n-quadrant and SHARE the A fragments.
//   * stage-K=64 double-buffered LDS: s_a[2][16][64] + s_b[2][64][64] =
//     10,240 B/block - the sibling's exact unpadded resource profile.
//   * one 16-byte cooperative global load per lane per tile per stage (wave
//     0 loads the 1,024 B A stage at 16 B/lane; all four waves load the
//     4,096 B B stage at 16 B/lane) = 5 wave-level vmem_read per stage vs 2
//     per step in R4; vmem_read 40,960 -> 12,800 (-68.75%); request bytes
//     20.97 MB -> 13.11 MB (A is re-read once per 64-N tile = 40x instead of
//     160x; B streamed once, 10.49 MB).
//   * stage i+1 loads issue at the TOP of stage i (16 B, 16-B-aligned) into
//     VGPRs; the two v_mmac_i32_16x16x32_i8 per wave on the current stage
//     cover their latency; the compiler-ordered s_waitcnt vmcnt(0) lands the
//     bytes right before the ds_write_b128 pair that publishes the alternate
//     slot; one __syncthreads() per stage (cross-wave barrier, no raw asm).
//   * split-K default 6: 40 tiles x 6 = 240 blocks = exactly 2 blocks/CU x 4
//     wavefronts = 8 resident waves/CU (the sibling's validated 240-block /
//     2-blocks-per-CU occupancy; K=4096 = 64 stage-64 units split as 4x11 +
//     2x10, all stage-aligned, tiling [0, K) exactly once ascending).
//     ZTH_W8A8_QKV_SPLIT_K still selects any trusted candidate
//     {2,3,4,5,6,8,9,12} (block counts 80..480) without source edits.
//   * combine kernel updated to 16x64 planes (40 blocks x 256 threads x 4
//     elements): exact int32 sum of the SPLIT_K partials in ascending slice
//     order + x_scale/weight_scale + bf16 (RN-even), same stream, in Graph.
//   * Exact int32 accumulation preserved bit-for-bit: same
//     v_mmac_i32_16x16x32_i8 hardware k-order, same row_major fragment lane
//     mapping as rounds 1-5, slices tile [0, K) once ascending, combine sums
//     in int32 in ascending slice order - every per-tile dot is identical to
//     rounds 1-5 and the scalar fallback.
//
// Round-6 result (built, ran, FAILED exact correctness): 72.32us median /
// 72.32us p90 (1.109x vs the 80.19us Triton baseline) but 30,697 / 40,960
// mismatches, max_abs_error 126.0, first mismatch at flat_index 16 (m=0,
// n=16) - the first element of wave 1's quadrant, with wave 0's whole
// quadrant (n = 0..15) correct. Root cause (fixed this round): the B-stage
// cooperative load's lane role `b_nchunk = lane >> 6` is 0 for EVERY lane
// (lane = tid & 63 ranges 0..63), so all four waves redundantly staged the
// SAME 16 columns [n0, n0+16) of each 64x64 B stage and LDS columns 16..63
// were never written (garbage). Each wave's fragment read starts at LDS
// column 16*wave, so waves 1..3 (3/4 of the outputs) consumed unstaged
// garbage - exactly matching the first mismatch at n=16 and the ~3/4 wrong
// outputs. Round-7 (this file) fixes the single lane role: `b_nchunk = wave`
// (== tid >> 6), so wave w stages B[kk + krow][n0 + 16w .. +16w + 16) - the
// same 16-column quadrant its fragments read. No instruction count changes
// (each lane still issues exactly one 16-B load per tile per stage; only the
// addresses change), so the R6 raw speed, the 12,800 vmem_read / 13.11 MB
// request profile and the 240-block / 8-waves-per-CU occupancy are preserved
// unchanged.
//
// Expected outcome (falsifiable): the repaired kernel must pass exact
// correctness (0 mismatches) at the R6 raw speed (~72us median, 1.109x vs
// the 80.19us Triton baseline). R6's sibling-scaled ~29-35us target was NOT
// reached (72.32us measured), so the accepted R4 54.21us official best may
// still lead; this round is the mandated smallest-correctness-fix of the
// faster-but-incorrect candidate, not a redesign.
//
// Round-11 change (combine-parallelism round, partial kernel BYTE-IDENTICAL
// to the accepted iteration-7 source): the operator aggregate profile of the
// accepted R7 tree (profiled 26.08us partial + 10.88us combine = 36.96us,
// matching the 36.88us official median, so the split is real, not PMC
// inflation) shows the combine kernel is ~29% of the operator wall despite
// reading only 6 x 40 x 4,096 B = 983,040 B of int32 partials. Root cause:
// the n64 combine launches 40 blocks x 256 threads x 4 elements per thread
// (160 waves on ~40 CUs, 24 strided int32 loads per thread from 6 planes
// 163,840 B apart) - a tiny, latency-bound grid with no wave-level slack.
// Round 11 (this file) splits the 16x64 tile into its four 16x16 quadrants:
// grid 40 -> 160 blocks (exactly 1.33 blocks/CU over 120 CUs), 256 threads
// x 1 element per thread, each element still sums its SPLIT_K planes in
// ascending slice order and applies the identical x_scale/weight_scale bf16
// (RN-even) store. The element mapping is bit-identical to the R6-R10
// combine (lin = quad*256 + t == the old t + q*256 set), so every output
// byte is unchanged; only the grid/parallelism changes.
//
// Expected outcome (falsifiable): combine 10.88us -> ~2-4us and operator
// median 36.88us -> ~29-32us (~1.15-1.27x vs the R7 accepted best), p90
// follows. Null result (combine stays ~10.9us): the combine cost is NOT
// wave-level latency exposure - it is fixed launch/teardown or DRAM
// latency that 4x parallelism cannot hide, and the next round must fuse the
// combine into the partial kernel tail or shrink partial traffic.
//
// Round-12: killed before a valid proposal (no source change; the archived
// iteration-12 source is byte-identical to the accepted iteration-11 tree).
//
// Round-13 (partial-kernel prefetch depth, archived digest 3abfd6bb, NOT
// accepted): issued stage-(i+2)'s loads one full stage earlier via VGPR-held
// a_hold/b_hold. Measured 32.395/32.445us median/p90 - a 2.69% REGRESSION vs
// the accepted 31.522/31.642us best; rejected and the tree reverted to the
// R11 source (this file was byte-identical to iteration 11 before this
// round's edit). Its archived ISA shows why it was a no-op: the compiler
// still orders s_waitcnt vmcnt(0) before the s_barrier, which covers the
// stage-(i+2) fetches issued at the TOP of the same stage, so the load-to-
// wait window was never widened - only +4-8 VGPR and +2 in-flight vmem per
// lane were added. The per-stage wall (~2.44us) is invariant to grid
// parallelism (R3/R9), LDS bank conflicts (R8), stage width (R10) and
// prefetch depth (R13); the operator-aggregate PMC (12.68 MB moved in ~26us
// partial = ~450 GB/s plateau with ~10 KB in flight per CU) means the wall
// tracks REQUEST throughput, not exposed latency.
//
// Round-14 change (this file, B-stage cooperative-load lane coalescing):
// the B stage load is the request hog - each wave's 64 lanes load 16 B at
// stride n = 2560, i.e. 64 DISTINCT 128-B lines per wave-load, so each block
// issues 256 L1->L2 line-requests per stage for only 64 unique lines
// (~26.8 G requests/s kernel-wide, plausibly saturating L2 request slots and
// inflating the queueing latency the per-stage chain exposes). The staging
// lane roles become tid-linear: b_krow = tid >> 2 (each wave stages 16 of
// the 64 stage rows) and b_nchunk = tid & 3 (the four 16-B column chunks of
// that row), so a wave-load's 64 lanes touch only 16 unique lines (4 lanes
// cover one row's 64 B) and the stage publishes the SAME bytes to the SAME
// LDS addresses with 4x fewer L1->L2 requests (256 -> 64 per stage per
// block). Published LDS content, fragment reads, the ascending-K v_mmac
// sequence (bit-identical int32 dots; slices still tile [0, K) exactly
// once), the one __syncthreads()/stage, the 10,240 B/block LDS footprint,
// the 240-block / 2-blocks-per-CU occupancy, the 12,800 vmem_read
// instruction count and the combine kernel are all unchanged.
//
// Expected outcome (falsifiable): if L1/L2 request throughput was the
// per-stage limiter, the partial kernel drops from ~26.08us to ~12-20us and
// the operator median from 31.52us to ~17-25us (~1.26-1.85x vs the R11
// accepted best), p90 follows. Null result (median flat ~31.5us): L1/L2
// request count is NOT the limiter (the plateau is DRAM behavior or
// launch/tail), and the next round must attack request BYTES (wider
// n-tiling, block-N=128 with split-K=12 to halve the 40x A re-reads at
// unchanged 2-blocks/CU occupancy) - NOT more prefetch depth and NOT more
// request coalescing.
//
// Round-17 change (this file, fused combine tail): the operator-aggregate
// PMC of the accepted R14 tree (profile iteration15/pmc.json, reused as the
// current-best evidence) splits the 28.483us wall into partial ~22.879us
// profiled and combine 3.68us profiled (160 blocks, 24 VGPR), and the
// combine's L2 hit rate is only 6.3% (l2_hits 1,166 / misses 17,205): the
// 983,040 B of int32 partials were evicted from L2 by the 12.68 MB A/B
// stream of the partial kernel and must be re-read from HBM, on top of the
// second-kernel launch gap. R16's block-N=128 / split-K=12 "more compute
// per stage" attack measured 32.087us - an 11.2% REGRESSION - falsifying
// per-stage compute width as the fix for the partial kernel, exactly as its
// null-result clause predicted ("the wall is launch/tail overhead or DRAM
// latency - the next round must fuse/trim the combine (19% of the wall) or
// the grid tail"). This round FUSES the combine into the partial kernel
// tail, the smallest change that removes the combine kernel launch, its gap
// and its HBM re-read together:
//   * The partial kernel's stage loop, fragment mapping, tid-linear B
//     staging, K tiling and per-tile int32 dots are BYTE-IDENTICAL to the
//     accepted R14 source; only the tail changes.
//   * After the (unchanged) private du_store_matrix_sync, each block's
//     tid-0 does __threadfence() + atomicAdd(&counters[tile], 1). The block
//     whose atomicAdd returns SPLIT_K-1 is the LAST arrival for its tile -
//     hardware-ordered by the atomics themselves, so there is NO spin loop
//     and NO residency assumption (safe for every trusted split-K, even if
//     register pressure ever dropped occupancy).
//   * The last block sums the tile's SPLIT_K planes in ascending slice
//     order (identical per-element int32 accumulation order as the R11
//     combine, so every output byte is bit-identical) and writes the scaled
//     bf16 output with the same x_scale/weight_scale RN-even conversion and
//     store addresses. Element mapping lin = tid*4 + e covers the 16x64
//     tile exactly once; per plane each wave loads 64 x 16 B = one
//     contiguous 1,024 B chunk (coalesced), so the fused tail's 6 plane
//     reads per thread are wide v4 loads.
//   * The fused tail reads the sibling planes while they are still L2-hot
//     (co-resident blocks write them ~0-2us earlier), so the combine's
//     HBM re-read (~2.5us at the measured 6.3% hit rate) largely disappears
//     as well. The operator becomes ONE kernel launch on the caller's
//     stream; the round-11 combine kernel and its launch gap (~1-2us) are
//     gone.
//   * The per-tile counters (num_tiles x int32 = 160 B) live in the last
//     160 B of the 16-plane contract workspace (plane 15's tail; max used
//     plane index for any trusted split-K <= 12 is 479, far below 600, so
//     no overlap with partial data). The combiner self-clears its tile's
//     counter (atomicExch 0) after combining, so replays start from zero;
//     the very first launch with a given workspace is zeroed by one async
//     hipMemsetAsync on the caller's stream, guarded by a workspace-pointer
//     static so the captured graph contains only the single kernel launch
//     and every replay is memset-free. No allocation, no host/device
//     synchronization, no default stream - the timed-region rules are
//     unchanged.
//   * The two-kernel path (partial kernel with FUSED=false + the round-11
//     quadrant combine kernel) is preserved for workspaces smaller than the
//     16-plane contract, so the fallback behavior is byte-identical to R14.
//
// Expected outcome (falsifiable): if the combine kernel + its launch gap
// (~4-5.5us of the 28.48us wall) were latency/traffic, the operator median
// drops to ~24-26.5us (~1.08-1.19x vs the R14 accepted best, ~3.0-3.35x vs
// the 80.19us Triton baseline), p90 follows, and the profiled kernel count
// goes 2 -> 1. Null result (median flat ~28.5us): the fused tail's own cost
// (last-tile combine ~1-2us on the critical path) cancels the savings - the
// wall is inside the partial stage loop itself, and the next round must
// attack per-stage DRAM latency (wider B lines / different B layout) rather
// than launch structure.
//
// Round-20/21/22: R20 (cold-plane L2 warm, discarded volatile reads of the
// base-stage slices) was killed before measurement; R21 (byte-identical R17
// re-issue) was killed the same way, proving the kills environmental; R22
// finally measured the warm on the byte-identical R17 tree: 28.409/28.469us
// median/p90 - a 7.9% REGRESSION (rejected; tree reverted to the R17 source,
// digest 264dbb1f). The warm's 1.92 MB of extra reads cost ~4.3us of wall
// during the bandwidth-saturated main phase and the tail did NOT get faster,
// falsifying the tail-as-DRAM-re-read model: the fused tail is NOT partial
// re-read DRAM traffic (the sibling planes are L2-hot), it is the arrival
// barrier/atomic serialization + last-stage skew + the combiner's serial
// critical path, and the main phase is at the DRAM/request floor (R18's
// packed-B layout - half the L2->DRAM line requests - was flat; R19's 3
// blocks/CU occupancy was a regression; R14's lane-coalescing and R17's
// fusion are the only real wins).
//
// Round-23 change (this file, fused-tail critical-path trim): with the tail
// established as serial latency, the two smallest arrival-protocol cuts:
//   * The trailing `atomicExch(&counters[tile], 0)` self-clear is REMOVED.
//     The last-arrival test becomes modulo-SPLIT_K on the monotonic arrival
//     count: replay r's arrivals return (r-1)*SPLIT_K + 0..SPLIT_K-1, so the
//     SPLIT_K-th arrival of EVERY replay is exactly the one with
//     `arrived % SPLIT_K == SPLIT_K - 1` (int32 wraparound would need
//     2^31/6 ~ 358M replays). The combiner no longer pays a dependent
//     atomicExch round trip (~0.2-0.4us) after its output stores; the
//     one-time host hipMemsetAsync of the counters before the first launch
//     is unchanged (and now purely a first-launch precondition), the
//     FUSED=false two-kernel fallback never touches counters, and every
//     replay stays exactly one kernel launch on the caller's stream.
//   * The tail's four per-thread bf16 stores (four `global_store_short_d16_hi`
//     with an intervening waitcnt in the archived ISA) are packed into ONE
//     8-B store: the 4 elements (lin = tid*4 + 0..3) are always 4 consecutive
//     columns of one row (4 | 64, so a 4-element run never crosses the
//     64-column row boundary) and the byte address is 8-B aligned (n0, col0
//     and row*n are all multiples of 4), so the four RN-even bf16 bit
//     patterns pack little-endian into one uint64 - identical output bytes,
//     identical store addresses, one store instruction instead of four plus
//     their address mads and the intervening wait.
//   The stage loop, tid-linear B staging, fragment mapping, K tiling,
//   per-tile int32 dots (ascending-K v_mmac + ascending-slice plane sums),
//   the arrival atomicAdd/fence order and the plane layout are byte-identical
//   to the accepted R17 source, so exact int32 accumulation and every output
//   byte are preserved bit-for-bit.
//
// Expected outcome (falsifiable): if the trailing atomic round trip + the
// serialized 2-B store issue are on the last-tile combiner's critical path,
// the operator median drops from 26.331us toward ~25.8-26.0us (~1-2%,
// ~1.02x vs the R17 accepted best, ~3.08-3.11x vs the 80.19us Triton
// baseline), p90 follows, and the profiled fused kernel drops from 25.439us
// toward ~25.0-25.2us with the global_store instruction count down slightly
// in the tail and everything else (vmem_read 14,720, LDS 198,320, grid 240,
// one launch) unchanged. Null result (median flat ~26.3us): the tail wall is
// the arrival atomicAdd + plane-load latency + last-stage skew (the
// self-clear and store issue are off it), and the next round must split the
// combine across the last TWO arrivals (bitmap + scratch + bounded-flag
// handoff; the producer is provably executing when the consumer spins, so
// it is deadlock-free) or attack the per-stage load batch directly.
//
// Timed-region rules honored: launch_w8a8_gemm performs no allocation,
// compilation, autotuning, packing, host/device synchronization or
// default-stream launch; it uses only the caller-provided tensors (the
// workspace holds the SPLIT_K int32 partial planes; round 17 additionally
// uses its last 160 B for the per-tile fused-combine arrival counters) and
// issues the kernel launch(es) on the caller's stream - one fused kernel
// for the contract workspace (round 17), two kernels (partial + combine)
// for the preserved fallback. The one-time counter zero is an async
// hipMemsetAsync on the same stream before the first launch with a given
// workspace, so it is never captured into the replayed Graph. The split-K
// choice is a host-side static dispatch decision (env override or default
// 6) made once per launch call, never inside the Graph replay, so
// capture/replay determinism holds.
//
// Header order is the DTK known-good order: hip_runtime first (du_mma.h is
// not self-contained when included before the HIP runtime headers), then
// the bfloat16 header, then du_mma.h.

#include <hip/hip_runtime.h>
#include <hip/hip_bfloat16.h>
#include <du_mma.h>

#include <cstdint>
#include <cstdlib>

namespace {

// gfx928 native wavefront is 64 lanes; blockDim must be a multiple of 64.
constexpr int kBlockThreads = 256;

// Minimal DUMMA tile constants (gfx928 INT8 support is m16n16k32).
constexpr int kDummaM = 16;
constexpr int kDummaN = 16;
constexpr int kDummaK = 32;
constexpr int kDummaThreads = 64;   // one wavefront (64 lanes) per block
constexpr int kCombineThreads = 256;  // 16x16 = 256 output elements per block

// Round-4 LDS staging (preserved kernel): padded B row stride (bytes). The
// matrix_b row_major fragment reads p[(col + j) * ldm + row]; at ldm = 16
// all 64 lanes of a ds_read_u8 collide on 4 LDS banks (16-way conflict),
// while ldm = 20 (4 mod 8) spreads each read over 16 banks (4-way).
constexpr int kDummaLdsBRow = kDummaN + 4;  // 20

// Round-6 geometry constants: block-N=64 (four 16x16 quadrants per block),
// stage-K=64 (two 32-K DUMMA steps per stage), four wavefronts per block.
constexpr int kDummaN64 = 64;     // N-tile width per round-6 block
constexpr int kStageK = 64;       // K rows staged per double-buffer slot
constexpr int kDummaThreads4 = 256;  // 4 wavefronts x 64 lanes per block

// Trusted occupancy-probe split-K candidates (control plane): all fit the
// contract workspace for this shape (2,621,440 B = 16 partial planes of
// 163,840 B each) and produce 64-aligned K slices. Round-6 block counts =
// 40*S (80..480).
constexpr int kTrustedSplitK[] = {2, 3, 4, 5, 6, 8, 9, 12};

// Round-to-nearest-even float -> bfloat16 bit pattern. bf16 tensors store
// uint16, so a manual uint16 store avoids depending on any bf16 type
// conversion API and is bit-identical to __float2bfloat16 in its default
// (RN-even) rounding mode.
__device__ __forceinline__ uint16_t float_to_bf16_bits(float f) {
  uint32_t u = 0;
  __builtin_memcpy(&u, &f, sizeof(u));
  const uint32_t bias = 0x7FFFu + ((u >> 16) & 1u);
  u += bias;
  return static_cast<uint16_t>(u >> 16);
}

// 16-byte vector used by the round-6 stage loads/stores (one 16-byte
// cooperative load per lane per tile; both global and LDS addresses are
// 16-B aligned - see the kernel comment).
struct alignas(16) v16_t {
  int64_t lo;
  int64_t hi;
};

// Stage one DUMMA K-step's A (16x32) and B (32x16) int8 tiles from global
// into one LDS double-buffer slot using one 8-byte vector load per lane per
// tile (2 vmem instructions per K step total, vs the 16 scalar
// global_load_ubyte the direct fragment loads issue). All 64 lanes
// participate; both global addresses are 8-byte aligned (kk is a multiple of
// 32, n0 of 16, n of 2560, x_q/weight tensor bases are 256 B aligned):
//   A: lane = row(0..15) | chunk(0..3)*16 -> 8 B of A row `row` at k-columns
//      kk + chunk*8 .. +8, written to s_a[slot][row][chunk*8 .. +8)
//      (LDS offset row*32 + chunk*8, 8 B aligned).
//   B: lane = krow(0..31) | half(0..1)*32 -> 8 B of K-row kk+krow at
//      n-columns n0 + half*8 .. +8, written to s_b[slot][krow][half*8 .. +8)
//      (LDS offset krow*20 + half*8; odd krows are 4 B aligned, so the
//      compiler splits those writes into two ds_write_b32 - harmless).
__device__ __forceinline__ void stage_dumma_step(
    const int8_t* __restrict__ x_q,
    const int8_t* __restrict__ weight,
    int kk,
    int k,
    int n,
    int n0,
    int lane,
    int slot,
    int8_t* __restrict__ s_a_slot,
    int8_t* __restrict__ s_b_slot) {
  const int a_row = lane & 15;
  const int a_chunk = lane >> 4;  // 0..3
  const int b_krow = lane & 31;
  const int b_half = lane >> 5;  // 0..1
  const int64_t va = *reinterpret_cast<const int64_t*>(
      x_q + static_cast<int64_t>(a_row) * k + kk + a_chunk * 8);
  *reinterpret_cast<int64_t*>(s_a_slot + a_row * kDummaK + a_chunk * 8) = va;
  const int64_t vb = *reinterpret_cast<const int64_t*>(
      weight + static_cast<int64_t>(kk + b_krow) * n + n0 + b_half * 8);
  *reinterpret_cast<int64_t*>(s_b_slot + b_krow * kDummaLdsBRow + b_half * 8) =
      vb;
}

// Workspace split-K partial kernel (template over SPLIT_K), M == 16:
//   partial[tile, slice][m, n] = int32_dot(x_q[m, k0:k1], weight[k0:k1, n])
// Layouts (all contiguous):
//   x_q      [16, K] int8 row-major (ldm = K)
//   weight   [K, N] int8 row-major (ldm = N, identity pack layout)
//   partials [(SPLIT_K * num_tiles) * 256] int32, plane = slice*num_tiles +
//            tile, row-major 16x16 tile per plane (workspace)
// One block = one wavefront = one (tile, slice). The K slice is 32-aligned
// (non-uniform: slices differ by at most one 32-K DUMMA step) and covers
// [k0, k0 + steps*32) in ascending order. Round-4: the per-step fragment
// loads are served from a double-buffered LDS pipeline (s_a[2][16][32] +
// s_b[2][32][20], 2,304 B per block) instead of direct 16 scalar byte loads
// from global; step i+1's two staging loads issue at the top of step i and
// complete under the step-i fragment reads + v_mmac, with one __syncthreads
// per step ordering them. Each block still writes exactly one private int32
// partial plane.
template <int SPLIT_K>
__global__ __launch_bounds__(kDummaThreads) void w8a8_dumma_m16_splitk_partial_kernel(
    const int8_t* __restrict__ x_q,
    const int8_t* __restrict__ weight,
    int32_t* __restrict__ partials,
    int n,
    int k) {
  const int tile = static_cast<int>(blockIdx.x);
  const int slice = static_cast<int>(blockIdx.y);
  const int n0 = tile * kDummaN;
  const int lane = static_cast<int>(threadIdx.x);  // 0..63, one wavefront

  // 32-aligned non-uniform slice bounds over the K dimension.
  const int total_steps = k / kDummaK;  // 4096 / 32 = 128
  const int base = total_steps / SPLIT_K;
  const int rem = total_steps - base * SPLIT_K;
  const int steps = base + (slice < rem ? 1 : 0);
  const int k0 = (slice * base + (slice < rem ? slice : rem)) * kDummaK;

  // Double-buffered LDS staging slots (round 4).
  __shared__ int8_t s_a[2][kDummaM][kDummaK];
  __shared__ int8_t s_b[2][kDummaK][kDummaLdsBRow];

  du::dumma::DUFragment<du::dumma::matrix_a, kDummaM, kDummaN, kDummaK,
                        signed char, du::dumma::row_major>
      a_frag;
  du::dumma::DUFragment<du::dumma::matrix_b, kDummaM, kDummaN, kDummaK,
                        signed char, du::dumma::row_major>
      b_frag;
  du::dumma::DUFragment<du::dumma::accumulator, kDummaM, kDummaN, kDummaK,
                        int>
      acc_frag;
  du::dumma::du_fill_fragment(acc_frag, 0);

  const int a_row = lane & 15;
  const int a_chunk = lane >> 4;  // 0..3
  const int b_krow = lane & 31;
  const int b_half = lane >> 5;  // 0..1

  // Prologue: stage step 0 into slot 0, then make it visible.
  if (steps > 0) {
    stage_dumma_step(x_q, weight, k0, k, n, n0, lane, 0, &s_a[0][0][0],
                     &s_b[0][0][0]);
  }
  __syncthreads();

  for (int i = 0; i < steps; ++i) {
    const int cur = i & 1;
    const int nxt = cur ^ 1;

    // Prefetch step i+1: issue its two 8-byte global loads FIRST (into
    // registers, no wait) so the ~hundreds of cycles of global latency
    // overlap the step-i fragment LDS reads and the v_mmac below; the
    // trailing ds_writes + barrier land the bytes before the next
    // iteration's fragment reads.
    int64_t a_nxt = 0;
    int64_t b_nxt = 0;
    if (i + 1 < steps) {
      const int kk_nxt = k0 + (i + 1) * kDummaK;
      a_nxt = *reinterpret_cast<const int64_t*>(
          x_q + static_cast<int64_t>(a_row) * k + kk_nxt + a_chunk * 8);
      b_nxt = *reinterpret_cast<const int64_t*>(
          weight + static_cast<int64_t>(kk_nxt + b_krow) * n + n0 +
          b_half * 8);
    }

    du::dumma::du_load_matrix_sync(a_frag, &s_a[cur][0][0], kDummaK);
    du::dumma::du_load_matrix_sync(b_frag, &s_b[cur][0][0], kDummaLdsBRow);
    du::dumma::du_mma_sync(acc_frag, a_frag, b_frag, acc_frag);

    if (i + 1 < steps) {
      *reinterpret_cast<int64_t*>(&s_a[nxt][0][0] + a_row * kDummaK +
                                  a_chunk * 8) = a_nxt;
      *reinterpret_cast<int64_t*>(&s_b[nxt][0][0] + b_krow * kDummaLdsBRow +
                                  b_half * 8) = b_nxt;
    }
    __syncthreads();
  }

  // Private int32 partial plane: slice-major so a tile's S planes are
  // stride-uniform in the combine kernel.
  const int num_tiles = static_cast<int>(gridDim.x);
  du::dumma::du_store_matrix_sync(
      partials + (slice * num_tiles + tile) * (kDummaM * kDummaN),
      acc_frag, kDummaN, du::dumma::mem_row_major);
}

// ---------------------------------------------------------------------------
// Round-6 partial kernel: block-N=64 / 4 wavefronts / stage-K=64 double-
// buffered A+B LDS (the validated sibling wqkv_a split-K=10 architecture,
// ported to N=2560).
//
// One block computes the 16x64 output tile [0..16) x [n0..n0+64). Four
// wavefronts (256 threads) each own one 16x16 n-quadrant (wave w -> columns
// n0 + 16w .. n0 + 16w + 16) and share the A fragments.
//
// Staging (one 16-byte cooperative load per lane per tile per stage):
//   A stage 16x64 int8 = 1,024 B: wave 0 loads it at 16 B/lane
//     (lane = row(0..15) | chunk(0..3)*16 -> A[row][kk + chunk*16 .. +16),
//     16-B aligned: row*4096 + kk + chunk*16, kk % 64 == 0).
//   B stage 64x64 int8 = 4,096 B: all four waves load it at 16 B/lane
//     (tid-linear roles: stage row b_krow = tid >> 2, 16-B column chunk
//     b_nchunk = tid & 3 -> B[kk + b_krow][n0 + b_nchunk*16 .. +16),
//     16-B aligned: (kk + b_krow)*2560 + n0 + b_nchunk*16). Round-7
//     correctness property preserved: every (row, chunk) of the stage is
//     written exactly once (wave w stages rows [16w, 16w+16) x all four
//     chunks, so columns [n0, n0+64) are fully covered; the round-6
//     `b_nchunk = lane >> 6` bug that left columns 16..63 unwritten is
//     structurally impossible now). Round-14: the tid-linear roles coalesce
//     each wave-load onto 16 unique 128-B lines (4 consecutive lanes cover
//     one row's 64 B) instead of rounds 6-11's one-lane-per-row mapping
//     (64 unique lines per wave-load) - 4x fewer L1->L2 requests, identical
//     bytes, identical LDS addresses, identical fragment reads.
//   Stage i+1's five wave-level loads issue at the TOP of stage i (into
//   VGPRs, no wait); the two v_mmac per wave on the current stage cover
//   their latency; the compiler-ordered s_waitcnt vmcnt(0) lands the bytes
//   right before the ds_write_b128 pair that publishes the alternate slot;
//   one __syncthreads() per stage makes the slot visible to all four waves.
//
// Fragment reads keep the same row_major DUMMA lane mapping as rounds 1-5
// (so the per-tile int32 dots are bit-identical): a_frag = 8 consecutive K
// bytes of one M row (ldm 64); b_frag = 8 strided ds_read_u8 (ldm 64, 16-way
// LDS bank conflict, accepted: round 5 proved the B fragment read path is
// off the critical path, and the sibling's 10,240 B profile is exactly
// unpadded).
//
// Partial store: each wave du_store_matrix_sync's its 16x16 quadrant into
// the (slice, tile) 16x64 int32 plane (slice-major, same convention as
// rounds 3-5).
template <int SPLIT_K, bool FUSED = true>
__global__ __launch_bounds__(kDummaThreads4)
void w8a8_dumma_m16_n64_splitk_partial_kernel(
    const int8_t* __restrict__ x_q,
    const int8_t* __restrict__ weight,
    int32_t* __restrict__ partials,
    const float* __restrict__ x_scale,
    const float* __restrict__ weight_scale,
    uint16_t* __restrict__ out,
    int32_t* __restrict__ counters,
    int n,
    int k) {
  const int tile = static_cast<int>(blockIdx.x);   // 0..(n/64-1) = 0..39
  const int slice = static_cast<int>(blockIdx.y);  // 0..SPLIT_K-1
  const int tid = static_cast<int>(threadIdx.x);   // 0..255
  const int wave = tid >> 6;                       // 0..3
  const int lane = tid & 63;
  const int n0 = tile * kDummaN64;

  // 64-aligned non-uniform slice bounds over the K dimension (units of
  // 64-K stages): slices tile [0, K) exactly once in ascending order.
  const int total_stages = k / kStageK;  // 4096 / 64 = 64
  const int base = total_stages / SPLIT_K;
  const int rem = total_stages - base * SPLIT_K;
  const int stages = base + (slice < rem ? 1 : 0);
  const int s0 = (slice * base + (slice < rem ? slice : rem)) * kStageK;

  // Double-buffered stage slots (round-6 geometry; 10,240 B/block total).
  __shared__ int8_t s_a[2][kDummaM][kStageK];    // 2 x 1,024 B
  __shared__ int8_t s_b[2][kStageK][kDummaN64];  // 2 x 4,096 B

  du::dumma::DUFragment<du::dumma::matrix_a, kDummaM, kDummaN, kDummaK,
                        signed char, du::dumma::row_major>
      a_frag;
  du::dumma::DUFragment<du::dumma::matrix_b, kDummaM, kDummaN, kDummaK,
                        signed char, du::dumma::row_major>
      b_frag;
  du::dumma::DUFragment<du::dumma::accumulator, kDummaM, kDummaN, kDummaK,
                        int>
      acc_frag;
  du::dumma::du_fill_fragment(acc_frag, 0);

  // Staging lane roles.
  const int a_row = (wave == 0) ? (lane & 15) : 0;
  const int a_chunk = (wave == 0) ? (lane >> 4) : 0;  // 0..3
  // Round-14 (B-stage load coalescing): tid-linear staging roles - each wave
  // stages 16 of the 64 stage rows x all four 16-B column chunks (wave w
  // covers rows [16w, 16w+16)). Round-7's correctness property (every stage
  // row/column written exactly once, all four waves participate) is
  // preserved: b_nchunk = tid & 3 ranges 0..3 inside every wave, and
  // b_krow = tid >> 2 ranges 0..63 across the block, so the round-6
  // `lane >> 6` redundant-staging bug cannot recur.
  const int b_krow = tid >> 2;   // 0..63: 16 stage rows per wave
  const int b_nchunk = tid & 3;  // 0..3: 16-B column chunk of that row

  // Prologue: stage stage-0 into slot 0, then make it visible.
  if (stages > 0) {
    v16_t a_v{};
    v16_t b_v;
    if (wave == 0) {
      a_v = *reinterpret_cast<const v16_t*>(
          x_q + static_cast<int64_t>(a_row) * k + s0 + a_chunk * 16);
    }
    b_v = *reinterpret_cast<const v16_t*>(
        weight + static_cast<int64_t>(s0 + b_krow) * n + n0 + b_nchunk * 16);
    if (wave == 0) {
      *reinterpret_cast<v16_t*>(&s_a[0][a_row][a_chunk * 16]) = a_v;
    }
    *reinterpret_cast<v16_t*>(&s_b[0][b_krow][b_nchunk * 16]) = b_v;
  }
  __syncthreads();

  for (int i = 0; i < stages; ++i) {
    const int cur = i & 1;
    const int nxt = cur ^ 1;

    // Prefetch stage i+1: issue its 16-byte global loads FIRST (into
    // registers, no wait) so their latency overlaps the stage-i fragment LDS
    // reads and the two v_mmac below; the compiler orders the s_waitcnt
    // vmcnt(0) immediately before the LDS stores that publish the alternate
    // slot.
    v16_t a_v{};
    v16_t b_v;
    if (i + 1 < stages) {
      const int kk_nxt = s0 + (i + 1) * kStageK;
      if (wave == 0) {
        a_v = *reinterpret_cast<const v16_t*>(
            x_q + static_cast<int64_t>(a_row) * k + kk_nxt + a_chunk * 16);
      }
      b_v = *reinterpret_cast<const v16_t*>(
          weight + static_cast<int64_t>(kk_nxt + b_krow) * n + n0 +
          b_nchunk * 16);
    }

    // Two 32-K DUMMA steps of the current stage (ascending K).
    for (int j = 0; j < 2; ++j) {
      du::dumma::du_load_matrix_sync(a_frag, &s_a[cur][0][j * kDummaK],
                                     kStageK);
      du::dumma::du_load_matrix_sync(b_frag,
                                     &s_b[cur][j * kDummaK][wave * kDummaN],
                                     kStageK);
      du::dumma::du_mma_sync(acc_frag, a_frag, b_frag, acc_frag);
    }

    if (i + 1 < stages) {
      if (wave == 0) {
        *reinterpret_cast<v16_t*>(&s_a[nxt][a_row][a_chunk * 16]) = a_v;
      }
      *reinterpret_cast<v16_t*>(&s_b[nxt][b_krow][b_nchunk * 16]) = b_v;
    }
    __syncthreads();
  }

  // Private int32 partial plane quadrant: slice-major planes of 16x64 so a
  // tile's S planes are stride-uniform in the combine kernel.
  const int num_tiles = static_cast<int>(gridDim.x);
  du::dumma::du_store_matrix_sync(
      partials + (slice * num_tiles + tile) * (kDummaM * kDummaN64) +
          wave * kDummaN,
      acc_frag, kDummaN64, du::dumma::mem_row_major);

  if (FUSED) {
    // Round-17 fused combine tail (replaces the separate combine kernel):
    // every block signals its arrival for its tile with one tid-0
    // __threadfence() + atomicAdd(&counters[tile], 1); the LAST arrival
    // (atomicAdd returns the value that identifies it, hardware-ordered
    // after all SPLIT_K sibling stores+fences - no spin loop, no residency
    // assumption) sums the tile's SPLIT_K planes in ascending slice order
    // and writes the scaled bf16 output with the same x_scale/weight_scale
    // RN-even conversion and store addresses as the R11 combine. The
    // per-element int32 sums are bit-identical to R11 (ascending slice
    // order), so the exact-K-order accumulation contract is unchanged.
    //
    // Round-23 (arrival-protocol trim): the counters are now MONOTONIC -
    // the trailing atomicExch(&counters[tile], 0) self-clear is gone. Each
    // replay adds exactly SPLIT_K to its tile's counter, so replay r's
    // arrivals return (r-1)*SPLIT_K + 0..SPLIT_K-1 and the SPLIT_K-th
    // arrival of EVERY replay is exactly the one with
    // (arrived % SPLIT_K) == SPLIT_K - 1 (int32 wraparound would need
    // 2^31/6 ~ 358M replays). The combiner no longer pays a dependent
    // atomicExch round trip after its output stores; the host still zeroes
    // the counters once per workspace before the first launch (see
    // launch_w8a8_gemm) and the FUSED=false fallback never touches them.
    __syncthreads();
    __shared__ int s_is_last;
    if (tid == 0) {
      __threadfence();  // release: sibling plane stores visible to the
                        // observer of this block's arrival atomic
      const int arrived = atomicAdd(&counters[tile], 1);
      __threadfence();  // acquire: this block's reads (below, after the
                        // barrier) see every sibling store that was released
                        // before its arrival atomic
      s_is_last = ((arrived % SPLIT_K) == SPLIT_K - 1);
    }
    __syncthreads();
    if (s_is_last) {
      // 1,024 output elements per 16x64 tile; 256 threads x 4 consecutive
      // elements (lin = tid*4 + e). Per plane, the 64 lanes of a wave load
      // 16 B each = one contiguous 1,024 B plane chunk (perfectly
      // coalesced).
      const int n0 = tile * kDummaN64;
      const int lin0 = tid * 4;
      int32_t sums[4] = {0, 0, 0, 0};
      for (int s = 0; s < SPLIT_K; ++s) {
        const int32_t* __restrict__ plane =
            partials + (s * num_tiles + tile) * (kDummaM * kDummaN64) + lin0;
        const int4 p = *reinterpret_cast<const int4*>(plane);
        sums[0] += p.x;
        sums[1] += p.y;
        sums[2] += p.z;
        sums[3] += p.w;
      }
      // Round-23 (store packing): the 4 elements (lin = tid*4 + 0..3) are
      // always 4 consecutive columns of one row (4 | 64, so a 4-element run
      // never crosses the 64-column row boundary) and the byte address is
      // 8-B aligned (n0, col0 and row*n are all multiples of 4), so the
      // four 2-B bf16 patterns pack little-endian into ONE 8-B store -
      // identical RN-even conversions, identical output bytes and store
      // addresses, one store instruction instead of four plus their address
      // mads and the intervening waitcnt seen in the R17 ISA.
      const int row = lin0 >> 6;
      const int col0 = lin0 & 63;
      uint64_t packed = 0;
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        const float scaled = static_cast<float>(sums[e]) * x_scale[row] *
                             weight_scale[n0 + col0 + e];
        packed |= static_cast<uint64_t>(float_to_bf16_bits(scaled)) << (16 * e);
      }
      *reinterpret_cast<uint64_t*>(out + static_cast<int64_t>(row) * n + n0 +
                                   col0) = packed;
    }
  }
}


// Combine + scale kernel (M == 16): one 16x16 tile per block, one thread per
// output element. Exact int32 sum of the SPLIT_K workspace partials, then
// x_scale/weight_scale scaling and the bf16 (RN-even) store. No barrier is
// needed: every thread reads only its own 16x16 element's S partial values
// (written by the previous kernel on the same stream).
__global__ __launch_bounds__(kCombineThreads) void w8a8_dumma_m16_combine_scale_kernel(
    const int32_t* __restrict__ partials,
    const float* __restrict__ x_scale,
    const float* __restrict__ weight_scale,
    uint16_t* __restrict__ out,
    int split_k,
    int num_tiles,
    int n) {
  const int tile = static_cast<int>(blockIdx.x);
  const int linear = static_cast<int>(threadIdx.x);  // 0..255
  const int row = linear >> 4;
  const int col = linear & 15;
  const int out_col = tile * kDummaN + col;

  int32_t sum = 0;
  for (int s = 0; s < split_k; ++s) {
    sum += partials[(s * num_tiles + tile) * (kDummaM * kDummaN) + linear];
  }

  const float scaled =
      static_cast<float>(sum) * x_scale[row] * weight_scale[out_col];
  out[row * n + out_col] = float_to_bf16_bits(scaled);
}

// Round-11 combine + scale kernel (M == 16): one 16x16 n-quadrant per block
// (grid = 4 x num_tiles = 160 blocks for the assigned shape), 256 threads x
// 1 element per thread. Exact int32 sum of the SPLIT_K workspace partials in
// ascending slice order, then x_scale/weight_scale scaling and the bf16
// (RN-even) store. No barrier is needed: every thread reads only its own
// element's S partial values (written by the previous kernel on the same
// stream). Element mapping is bit-identical to the R6-R10 combine: lin =
// quad*256 + t covers exactly the old t + q*256 set, so row/col, the int32
// sum order and the store addresses are unchanged. Stores are coalesced:
// each block's 256 lanes cover one full 16x16 quadrant (64 B per row-slice
// of 64 lanes x 2 B).
__global__ __launch_bounds__(kCombineThreads)
void w8a8_dumma_m16_n64_combine_scale_kernel(
    const int32_t* __restrict__ partials,
    const float* __restrict__ x_scale,
    const float* __restrict__ weight_scale,
    uint16_t* __restrict__ out,
    int split_k,
    int num_tiles,
    int n) {
  const int quad = static_cast<int>(blockIdx.x) & 3;  // 0..3
  const int tile = static_cast<int>(blockIdx.x) >> 2; // 0..num_tiles-1
  const int t = static_cast<int>(threadIdx.x);        // 0..255
  const int lin = quad * kCombineThreads + t;         // 0..1023 in the tile
  const int row = lin >> 6;
  const int col = lin & 63;
  const int n0 = tile * kDummaN64;

  int32_t sum = 0;
  for (int s = 0; s < split_k; ++s) {
    sum += partials[(s * num_tiles + tile) * (kDummaM * kDummaN64) + lin];
  }
  const float scaled =
      static_cast<float>(sum) * x_scale[row] * weight_scale[n0 + col];
  out[row * n + n0 + col] = float_to_bf16_bits(scaled);
}

// Scalar W8A8 GEMM (generic fallback for every shape not matched by the
// exact-shape DUMMA launch):
//   out[m, n] = bf16( int32_dot(x_q[m, :], weight[:, n])
//                     * x_scale[m] * weight_scale[n] )
// Layouts (all contiguous):
//   x_q          [M, K] int8 row-major
//   weight       [K, N] int8 row-major (identity pack layout)
//   x_scale      [M]    fp32
//   weight_scale [N]    fp32
//   out          [M, N] bf16 (uint16 storage)
// One thread per output element; adjacent lanes map to adjacent N addresses
// (linear index over row-major [M, N] output), keeping stores coalesced.
__global__ __launch_bounds__(kBlockThreads) void w8a8_scalar_gemm_kernel(
    const int8_t* __restrict__ x_q,
    const int8_t* __restrict__ weight,
    const float* __restrict__ x_scale,
    const float* __restrict__ weight_scale,
    uint16_t* __restrict__ out,
    int m,
    int n,
    int k) {
  const int64_t linear = static_cast<int64_t>(blockIdx.x) * kBlockThreads +
                         static_cast<int64_t>(threadIdx.x);
  const int64_t total = static_cast<int64_t>(m) * static_cast<int64_t>(n);
  if (linear >= total) {
    return;
  }
  const int row = static_cast<int>(linear / static_cast<int64_t>(n));
  const int col = static_cast<int>(linear - static_cast<int64_t>(row) * n);

  const int8_t* __restrict__ a_row =
      x_q + static_cast<int64_t>(row) * static_cast<int64_t>(k);
  const int8_t* __restrict__ b_col = weight + col;

  // Exact int32 accumulation over the full K dimension.
  int32_t acc = 0;
  for (int kk = 0; kk < k; ++kk) {
    acc += static_cast<int32_t>(a_row[kk]) *
           static_cast<int32_t>(b_col[static_cast<int64_t>(kk) * n]);
  }

  const float scaled =
      static_cast<float>(acc) * x_scale[row] * weight_scale[col];
  out[linear] = float_to_bf16_bits(scaled);
}

// Template launch helper for the partial kernel (one wavefront per block,
// grid = (num_tiles, SPLIT_K)).
template <int SPLIT_K>
void launch_splitk_partial(const int8_t* a,
                           const int8_t* b,
                           int32_t* partials,
                           int n,
                           int k,
                           hipStream_t stream,
                           int num_tiles) {
  hipLaunchKernelGGL(
      HIP_KERNEL_NAME(w8a8_dumma_m16_splitk_partial_kernel<SPLIT_K>),
      dim3(static_cast<unsigned>(num_tiles), static_cast<unsigned>(SPLIT_K)),
      dim3(static_cast<unsigned>(kDummaThreads)),
      0,
      stream,
      a,
      b,
      partials,
      n,
      k);
}

// Template launch helper for the round-6 partial kernel (four wavefronts per
// block, grid = (num_tiles_64, SPLIT_K)). FUSED selects the round-17 fused
// combine tail (single-kernel operator); FUSED=false keeps the round-11
// two-kernel path for non-contract workspaces (x_scale/weight_scale/out/
// counters are passed but unused by the kernel when FUSED=false).
template <int SPLIT_K, bool FUSED>
void launch_splitk_partial_n64(const int8_t* a,
                               const int8_t* b,
                               int32_t* partials,
                               const float* x_scale,
                               const float* weight_scale,
                               uint16_t* out,
                               int32_t* counters,
                               int n,
                               int k,
                               hipStream_t stream,
                               int num_tiles) {
  hipLaunchKernelGGL(
      HIP_KERNEL_NAME(w8a8_dumma_m16_n64_splitk_partial_kernel<SPLIT_K,
                                                                FUSED>),
      dim3(static_cast<unsigned>(num_tiles), static_cast<unsigned>(SPLIT_K)),
      dim3(static_cast<unsigned>(kDummaThreads4)),
      0,
      stream,
      a,
      b,
      partials,
      x_scale,
      weight_scale,
      out,
      counters,
      n,
      k);
}

// Scalar fallback used by the dispatcher below (defined after it; forward
// declaration keeps the call sites valid).
void launch_scalar_gemm(const int8_t* a,
                        const int8_t* b,
                        const float* x_scale,
                        const float* weight_scale,
                        void* out,
                        int m,
                        int n,
                        int k,
                        hipStream_t stream);

// Dispatcher over the trusted split-K set for the n64 partial kernel.
// FUSED=false (the two-kernel path) passes null scale/output/counter
// pointers; the kernel never dereferences them in that instantiation.
template <bool FUSED>
void launch_splitk_partial_n64_switch(int split_k,
                                      const int8_t* a,
                                      const int8_t* b,
                                      int32_t* partials,
                                      const float* x_scale,
                                      const float* weight_scale,
                                      uint16_t* out,
                                      int32_t* counters,
                                      int m,
                                      int n,
                                      int k,
                                      hipStream_t stream,
                                      int num_tiles) {
  switch (split_k) {
    case 2:
      launch_splitk_partial_n64<2, FUSED>(a, b, partials, x_scale,
                                          weight_scale, out, counters, n, k,
                                          stream, num_tiles);
      break;
    case 3:
      launch_splitk_partial_n64<3, FUSED>(a, b, partials, x_scale,
                                          weight_scale, out, counters, n, k,
                                          stream, num_tiles);
      break;
    case 4:
      launch_splitk_partial_n64<4, FUSED>(a, b, partials, x_scale,
                                          weight_scale, out, counters, n, k,
                                          stream, num_tiles);
      break;
    case 5:
      launch_splitk_partial_n64<5, FUSED>(a, b, partials, x_scale,
                                          weight_scale, out, counters, n, k,
                                          stream, num_tiles);
      break;
    case 6:
      launch_splitk_partial_n64<6, FUSED>(a, b, partials, x_scale,
                                          weight_scale, out, counters, n, k,
                                          stream, num_tiles);
      break;
    case 8:
      launch_splitk_partial_n64<8, FUSED>(a, b, partials, x_scale,
                                          weight_scale, out, counters, n, k,
                                          stream, num_tiles);
      break;
    case 9:
      launch_splitk_partial_n64<9, FUSED>(a, b, partials, x_scale,
                                          weight_scale, out, counters, n, k,
                                          stream, num_tiles);
      break;
    case 12:
      launch_splitk_partial_n64<12, FUSED>(a, b, partials, x_scale,
                                           weight_scale, out, counters, n, k,
                                           stream, num_tiles);
      break;
    default:
      launch_scalar_gemm(a, b, x_scale, weight_scale, out, m, n, k, stream);
      break;
  }
}

void launch_scalar_gemm(const int8_t* a,
                        const int8_t* b,
                        const float* x_scale,
                        const float* weight_scale,
                        void* out,
                        int m,
                        int n,
                        int k,
                        hipStream_t stream) {
  const int64_t total = static_cast<int64_t>(m) * static_cast<int64_t>(n);
  const int blocks =
      static_cast<int>((total + kBlockThreads - 1) / kBlockThreads);
  hipLaunchKernelGGL(
      w8a8_scalar_gemm_kernel,
      dim3(static_cast<unsigned>(blocks)),
      dim3(static_cast<unsigned>(kBlockThreads)),
      0,
      stream,
      a,
      b,
      x_scale,
      weight_scale,
      static_cast<uint16_t*>(out),
      m,
      n,
      k);
}

}  // namespace

// Host launcher for torch.ops.zth_w8a8.gemm_out (timed / graph-capturable).
// Called on PyTorch's current HIP stream. The exact assigned shape
// (m,n,k) == (16,2560,4096) is dispatched to the workspace split-K DUMMA
// pair (partial kernel + combine/scale kernel, both in the timed Graph);
// every other shape reaches the scalar fallback.
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
  if (m <= 0 || n <= 0 || k <= 0) {
    return;
  }
  if (m == 16 && n == 2560 && k == 4096) {
    // Round-6 geometry: block-N=64, so 2560 / 64 = 40 tiles.
    const int num_tiles = n / kDummaN64;  // 2560 / 64 = 40
    const int64_t plane_bytes =
        static_cast<int64_t>(num_tiles) * kDummaM * kDummaN64 *
        static_cast<int64_t>(sizeof(int32_t));  // 40 * 1024 * 4 = 163,840

    // Split-K selection: default 6 (mandated primary geometry: 40 x 6 = 240
    // four-wave blocks = exactly 2 blocks/CU x 4 wavefronts = 8 resident
    // waves/CU, the validated sibling wqkv_a occupancy). The
    // ZTH_W8A8_QKV_SPLIT_K environment variable selects any trusted
    // occupancy-probe candidate {2,3,4,5,6,8,9,12} (block counts 80..480)
    // so the CU-aligned sweep can be measured without source edits. The read
    // happens once per launch call on the host, never during Graph replay,
    // so capture/replay determinism is preserved.
    int split_k = 6;
    if (const char* env = std::getenv("ZTH_W8A8_QKV_SPLIT_K")) {
      char* end = nullptr;
      const long parsed = std::strtol(env, &end, 10);
      if (end != env && *end == '\0') {
        for (int t : kTrustedSplitK) {
          if (static_cast<int>(parsed) == t) {
            split_k = t;
            break;
          }
        }
      }
    }

    // Workspace fit: clamp to the largest trusted candidate that fits the
    // caller workspace (contract guarantees 16 planes = 2,621,440 B, so the
    // clamp never fires for the exact shape; kept defensive).
    int best_fit = 0;
    for (int t : kTrustedSplitK) {
      if (static_cast<int64_t>(t) * plane_bytes <= workspace_bytes) {
        best_fit = t;
      }
    }
    if (best_fit == 0) {
      launch_scalar_gemm(a, b, x_scale, weight_scale, out, m, n, k, stream);
      return;
    }
    if (split_k > best_fit) {
      split_k = best_fit;
    }

    int32_t* partials = static_cast<int32_t*>(workspace);

    // Round-17 fused single-kernel path: the combine runs inside the
    // partial kernel (last arrival per tile), so the operator is ONE launch
    // and the round-11 combine kernel + its launch gap disappear. The
    // per-tile arrival counters (num_tiles x int32) live in the last 160 B
    // of the 16-plane contract workspace (plane 15's tail; planes 0..12*40-1
    // are the max used by any trusted split-K, so no overlap). Round-23: the
    // counters are monotonic (the combiner no longer self-clears; the
    // last-arrival test is modulo-SPLIT_K on the arrival count), so only the
    // FIRST launch with a given workspace needs an async zero (guarded by a
    // workspace-pointer static: the memset is a stream op on the caller's
    // stream - no host sync - and it is NOT part of the captured graph, so
    // every replay is just the one kernel).
    if (workspace_bytes >= 16 * plane_bytes) {
      int32_t* counters = reinterpret_cast<int32_t*>(
                              static_cast<char*>(workspace) + 16 * plane_bytes) -
                          num_tiles;
      static const void* s_fused_counters_ws = nullptr;
      if (s_fused_counters_ws != workspace) {
        hipMemsetAsync(counters, 0,
                       static_cast<size_t>(num_tiles) * sizeof(int32_t),
                       stream);
        s_fused_counters_ws = workspace;
      }
      launch_splitk_partial_n64_switch<true>(
          split_k, a, b, partials, x_scale, weight_scale,
          static_cast<uint16_t*>(out), counters, m, n, k, stream, num_tiles);
      return;
    }

    // Fallback for workspaces smaller than the 16-plane contract: the
    // round-11 two-kernel path (partial kernel without the fused tail,
    // then the quadrant combine kernel), byte-identical to the accepted
    // R14 behavior. The FUSED=false instantiation never dereferences the
    // scale/output/counter pointers.
    launch_splitk_partial_n64_switch<false>(
        split_k, a, b, partials, x_scale, weight_scale,
        static_cast<uint16_t*>(out), static_cast<int32_t*>(nullptr), m, n, k,
        stream, num_tiles);
    hipLaunchKernelGGL(
        w8a8_dumma_m16_n64_combine_scale_kernel,
        dim3(static_cast<unsigned>(num_tiles * 4)),
        dim3(static_cast<unsigned>(kCombineThreads)),
        0,
        stream,
        partials,
        x_scale,
        weight_scale,
        static_cast<uint16_t*>(out),
        split_k,
        num_tiles,
        n);
    return;
  }
  launch_scalar_gemm(a, b, x_scale, weight_scale, out, m, n, k, stream);
}

// Optional out-of-timed-region weight packing. Bootstrap: identity
// device-to-device copy of the logical [K, N] int8 weight and the [N] fp32
// scales, valid for every (K, N). Later rounds may replace this with a
// packed layout as long as launch_w8a8_gemm interprets it consistently.
extern "C" void launch_pack_w8a8_weight(
    const int8_t* raw_weight,
    const float* weight_scale,
    int8_t* packed_weight,
    float* packed_weight_scale,
    int k,
    int n,
    hipStream_t stream) {
  if (k <= 0 || n <= 0) {
    return;
  }
  hipMemcpyAsync(
      packed_weight,
      raw_weight,
      static_cast<size_t>(k) * static_cast<size_t>(n) * sizeof(int8_t),
      hipMemcpyDeviceToDevice,
      stream);
  hipMemcpyAsync(
      packed_weight_scale,
      weight_scale,
      static_cast<size_t>(n) * sizeof(float),
      hipMemcpyDeviceToDevice,
      stream);
}
// @@end
