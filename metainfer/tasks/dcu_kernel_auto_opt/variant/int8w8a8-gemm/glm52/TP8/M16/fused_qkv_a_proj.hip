// @@variant shape=glm_tp8_fused_qkv_a_proj_m16 commit=bf306c40c48713a81efbd7ace430c9cf21cbb429 added=2026-08-31
//   median_us=41.97 p90_us=42.09
//   source=glm5-2-dsh-tp8-m16-test1-cb4d262c
// INT8 W8A8 GEMM kernel for Hygon K500SM_AI / gfx928 (worker_0).
//
// Assigned shape:
//   glm_tp8_fused_qkv_a_proj_m16 : M=16, N=2624, K=6144
//
// Iteration 0 (correctness-first scalar bootstrap): one thread computes one
// output element with a plain int32 dot product. Generic fallback, still
// used for every shape that is NOT the exact assigned (m,n,k) below,
// including the paired M=2 API shapes with the same (N,K).
//
// Iteration 1 (DUMMA m16n16k32 bootstrap): the exact assigned shape ran the
// minimal gfx928 INT8 DUMMA kernel -- one 64-lane wavefront per 16x16 output
// tile (grid = N/16 = 164 one-wave blocks, no cross-wave barrier, no LDS
// staging, no split-K), direct global fragment loads via du_load_matrix_sync,
// explicit int32 accumulation over the full K=6144 (192 m16n16k32 steps),
// then a direct register epilogue that applies x_scale[m] *
// packed_weight_scale[n] and stores bf16. Measured median 205.41 us
// (0.48x vs the fixed 97.99 us Triton baseline): the 164-block grid leaves
// ~1.37 one-wave blocks per CU, below the two-blocks-per-CU latency-hiding
// target, so each wave's direct fragment-load vmcnt stalls are serialized
// with no co-resident wave to overlap them.
//
// Iteration 2 (architecture round, split-K occupancy probe): keep the
// zero-barrier one-wave block, but split K across SPLIT_K=4 blocks, giving a
// finer one-wave zero-barrier grid of 164*4 = 656 independent blocks (5.47
// blocks/CU >= 2 blocks/CU target) with each MMA chain shortened to
// K/SPLIT_K = 48 k32 steps. Each block publishes its 16x16 int32 partial
// plane into the preallocated workspace ([4][164][256] int32 = 671,744 B,
// within the 16-plane API budget), and a tiny same-stream combine kernel
// (164 one-wave blocks) sums the planes in ascending split order -- exact
// int32 accumulation, bit-identical to the reference -- then applies the
// x_scale * weight_scale bf16 epilogue. Both launches stay inside the timed
// Graph replay (split-K combine is part of the operator wall).
//
// Iteration 3 (rejected occupancy probe, source reverted): an env-selectable
// SPLIT_K sweep defaulted to 2 measured 184.38 us median (0.53x baseline) --
// worse than SPLIT_K=4 -- so the three-point curve {1,2,4} = {205.41, 184.38,
// 138.56} us shows 5.47 blocks/CU co-residency plus the 48-step chains is the
// good operating point and 2.73 blocks/CU regresses.
//
// Iteration 4 (architecture/pipeline round, multi-N-tile reuse): the
// iteration-2 partial kernel's exact code object still serializes 16
// global_load_ubyte fragment loads with a full per-load s_waitcnt vmcnt
// ladder (vmcnt(15)..vmcnt(0)) before every single v_mmac (48 chains per
// block at SPLIT_K=4, zero LDS), so the residual limiter is the load-path
// latency ladder, not split count. This round reuses the A fragment across
// two adjacent 16x16 N-tiles in the same one-wave zero-barrier block: each
// k32 step issues one shared A fragment load plus two B fragment loads (24
// byte loads total) and then two v_mmacs, halving the per-MMA load-ladder
// stalls while keeping the proven 656-block grid (82 N-tile pairs x 8 K
// splits = 5.47 one-wave blocks/CU) and shortening each block's MMA chain to
// K/SPLIT_K = 24 k32 steps. Reused bytes: the per-block A slice
// (A[0:16, split-slice], 12,288 B at SPLIT_K=8) is loaded once per k32 step
// and shared by both 16x16 N-tiles (A total reads halve from 16.12 MB to
// 8.06 MB, L2-hot); every B byte is still read exactly once by its owning
// tile (cold 16.12 MB stream, no B reuse). Workspace layout (iteration 4:
// [8][164][256] int32; reordered to tile-major [164][8][256] in iteration
// 18 so a tile's 8 split planes are one contiguous 8-KB region) totals
// 1,343,488 B, within the API's 16-plane budget
// (allocate_workspace yields 2,686,976 B for this shape). The combine kernel
// (SPLIT_K=8, 164 one-wave blocks) is unchanged apart from its template
// instantiation: ascending split sum keeps the int32 accumulation
// bit-identical to the k-ascending reference.
//
// Iteration 5 (HIP-only packed-weight round): the iteration-4 partial
// kernel's exact code object still compiles every fragment load of a k32
// step to one global_load_ubyte per byte (8 A + 16 B) with a progressive
// s_waitcnt vmcnt ladder and byte reassembly before the two v_mmacs -- the
// library's int8 du_load_matrix_sync scalarizes to 8 per-lane byte loads
// regardless of layout. This round replaces the B data path with the
// lineage-validated n-major pack: launch_pack_w8a8_weight transposes the
// exact (k,n)==(6144,2624) weight once, outside the timed region and out of
// Graph capture, into packed[n*K + k] = raw[k*N + n] (same byte count, same
// buffer, graph-stable addresses; the identity copy is retained for every
// other (K,N)). In the partial kernel the B fragments are declared col_major
// against the packed buffer (du_mma.hpp lane mapping: lane l holds the 8
// CONSECUTIVE k bytes of column n0+(l&15) at k-offset ((l>>4)*8)), and the
// row-major A fragment already maps to 8 consecutive bytes of one row, so
// each of the three fragments of a k32 step is fetched with ONE aligned 8-B
// vector load per lane (int2 + __builtin_memcpy into the fragment storage,
// the same register-to-register fill the library loader produces): 3
// global_load_dwordx2 per step instead of 24 global_load_ubyte (47,232 vs
// 377,856 vmem_read instructions per replay, -87.5%) and one vmcnt drain per
// step instead of the per-byte ladder. Geometry (grid (82,8) = 656 one-wave
// blocks, two tiles per block, SPLIT_K=8), the zero-barrier zero-LDS
// structure, the combine kernel, the workspace planes and the exact int32
// accumulation order are all unchanged, so outputs stay bit-identical (0
// mismatches). The generic scalar fallback decodes the n-major pack for
// (n,k)==(2624,6144) (keeps the paired M=2 API shape byte-exact).
//
// Iteration 6 (pipeline round, B-only LDS staging): the iteration-5 code
// object's steady-state loop still serializes, per k32 step, 3
// global_load_dwordx2 + s_waitcnt vmcnt(1) + v_mmac + s_waitcnt vmcnt(0) +
// v_mmac with the next step's loads issued only AFTER both MMAs (24 cold-B
// latency exposures per block, no cross-step overlap), while the L2/VMEM
// evidence shows B is the cold once-read stream (16.12 MB/replay, ~all L2
// misses / HBM bytes) and A is a 98 KiB L2-resident stream re-read 82x.
// This round stages ONLY B through LDS: each 2-k32-step chunk's two B tiles
// are fetched with one aligned 16-B cooperative load per lane per tile
// (global_load_dwordx4, 64 contiguous packed B bytes per row, was 32 B),
// drained ONCE per chunk (12 drains/block, was 24), written to a 2,304-B
// bank-skewed (stride 72) LDS buffer, and re-read per fragment with 8-B
// ds_read_b64 while the next chunk's loads are issued into fresh registers
// before the current chunk is consumed (structure only; overlap is verified
// from the exact ISA after this round). A stays direct (L2-hot, 8-B int2
// loads). vmem_read 47,232 -> 31,488 per replay (-33%); HBM read unchanged
// at the compulsory floor (~17.66 MB); LDS +32.2 MB/replay (16.12 MB
// write + 16.12 MB read); 5.47 one-wave blocks/CU preserved (LDS 2,304 B ->
// 28 blocks/CU, VGPR ~40-46 -> >= 10 waves/CU). Fragment values are
// byte-identical to iteration 5, so partials stay bit-identical (0
// mismatches) and Graph replay with changed contents must pass.
//
// Iteration 7 (pipeline round, B-only LDS DOUBLE buffering, mandated by the
// pipeline decision rule: K=6144 >= 1024, L2 hit 40.6% < 70%, doubled LDS
// 4,608 B < 48 KiB): the iteration-6 exact code object keeps the next
// chunk's B loads in flight across the back edge but drains the next chunk's
// A loads at the BOTTOM of every body (vmcnt(1)/vmcnt(0) before the rotation
// v_movs at 0x56E8/0x56F0) -- the register rotation forces a per-chunk vmcnt
// drain on the A path (zero cross-chunk overlap for A, ~1 chunk for cold B).
// This round replaces the single LDS plane + rotation pipeline with TWO
// 2,304-B LDS planes (ping-pong) and TWO register sets alternating by chunk
// parity, each chunk's four loads (2 dwordx4 B + 2 dwordx2 A) issued TWO
// chunks ahead and consumed in place by the same-parity stage: no rotation
// copies, no bottom-of-body drains, one vmcnt drain per chunk at its own
// staging write, barriers per K step = 0 (one wave; parity replaces the
// cross-wave barrier). Fragment values are unchanged, so partials stay
// bit-identical and Graph replay with changed contents must pass.
//
// Iteration 8 (HIP-only occupancy probe, REJECTED and reverted): SPLIT_K=12
// (984 one-wave blocks = 2.05 waves/SIMD) regressed to 50.46 us median (vs
// the iteration-7 best 47.59 us), falsifying per-SIMD co-residency as the
// binding limiter: the residual is per-block serial cost, not the number of
// co-resident waves.
//
// Iteration 9 (HIP-only pipeline round, stage width 64 K -> 128 K): the
// iteration-7 partial kernel still pays one near-full cold-B vmcnt drain per
// 64-K chunk (12 chunks/block) with only ~1 pass of work between issue and
// drain. This round WIDENS each staged chunk to FOUR k32 steps (128 K) while
// keeping the exact double-buffer ping-pong structure (two 136-stride LDS
// planes, two parity register sets, two-chunk prefetch, 3 passes of 2
// chunks, grid (82,8) = 656 one-wave blocks, zero barriers): drains halve
// 12 -> 6, the in-flight window doubles (each set now carries 2 x 32 B of B
// + 32 B of A per lane), LDS becomes 8,704 B/block (7 blocks/CU headroom vs
// the scheduled 5.47), VGPR ~58 -> ~80-95 (no spills expected), vmem_read
// stays 31,488/replay, LDS instructions stay 31,488/replay, HBM read stays
// at the compulsory ~17.66 MB floor. B fragment VALUES are byte-identical
// (same packed bytes staged, same 8-B ds_read_b64 fills), the k-ascending
// int32 accumulation order is unchanged, so partials and the combine output
// stay bit-identical (0 mismatches) and Graph replay with changed contents
// must pass.
//
// Iteration 10 (HIP-only pipeline round, stage width 128 K -> 192 K): the
// exact iteration-9 code object (digest b766fc17..., 96 VGPR/32 SGPR/8,704 B
// LDS/0 scratch, profiled 37.44 us) emits ONE full progressive vmcnt drain
// per PASS at the pass top (s_waitcnt vmcnt(7)..vmcnt(0) ladder at 0x571C-
// 0x574C covering all 16 loads issued in the previous pass; the B staging
// writes and the A capture after it are wait-free) -- so the real per-block
// drain-event count is 3 (passes), not 6 (chunks), each exposing the cold-B
// round trip of 2 x 128-K chunks against an issue-to-drain window of only
// ~1 pass (~16 MMAs + 8 ds_read2 + 8 stagings + 16 loads). This round
// WIDENS each staged chunk to SIX k32 steps (192 K) while keeping the exact
// double-buffer ping-pong structure (two 200-stride LDS planes, two parity
// register sets, two-chunk prefetch, 2 passes of 2 chunks, grid (82,8) =
// 656 one-wave blocks, zero barriers): drain events drop 3 -> 2 per block
// and each event now covers ~50% more in-flight bytes (2 x 192-K chunks =
// 12,288 B of B + 6,144 B of A) against an issue-to-drain window of one
// full pass (24 MMAs + 12 ds_read2 + 12 stagings + 24 loads). LDS becomes
// s_b[2][2][16][200] = 12,800 B/block (stride 200 = 192 + 8: 50 dwords = 18
// mod 32 -> row bank bases 18*n mod 32 for n = 0..15, the same 16-distinct-
// base conflict-free pattern as iteration 9's 2*n mod 32; the plane offset
// 6,400 B = 1,600 dwords is 0 mod 32, so both planes keep the same pattern).
// LDS 12,800 B -> 5 blocks/CU of headroom vs the scheduled 5.47 (5 slots
// still >= the 4 SIMDs per CU, so co-execution is unchanged; the 56-block
// shortfall only delays block START by at most one block-launch latency on
// those CUs); VGPR ~96 -> ~110-130 (in-flight set registers 48 -> 72; no
// spills expected, 65,536/(130*64) ~= 7.9 waves/CU >= 5.47). vmem_read stays
// 31,488/replay (4 chunks x 6 dwordx4 B + 6 dwordx2 A = 48 loads/block, same
// total), LDS instructions stay 31,488/replay (the same staged bytes and the
// same 8-B ds_read fills), LDS traffic unchanged, HBM read unchanged at the
// compulsory ~17.66 MB floor. Fragment VALUES are byte-identical (the staged
// rows hold the same packed bytes; the per-step 8-B ds_read_b64 fills are
// unchanged), the k-ascending int32 accumulation order is unchanged, so the
// partial planes and the combine output stay bit-identical (0 mismatches)
// and Graph replay with changed contents must pass.
//
// Iteration 13 (combine latency consolidation): iteration 12's single-drain
// partial kernel landed at 43.64 us median (43.81 -> 43.64, +0.38%),
// falsifying the cold-B drain chain as the residual limiter: the operator is
// at the once-read-B memory-pipe floor (PMC: 19.09 MB of traffic per replay
// at ~436 GB/s derived HBM bandwidth; partial kernel profiled 41.92 us,
// combine kernel profiled 4.48 us with 31.4% L2 hit rate -- the 16.1-MB B
// stream thrashes L2 ahead of the combine). The exact combine ISA shows the
// four weight_scale elements loaded as FOUR serialized 4-B loads, each
// waited to completion before its multiply and its own 2-B store: four
// round trips per lane. This round consolidates the combine epilogue: ONE
// aligned 16-B float4 scale load (issued before the 8 plane reads) + ONE
// 8-B packed bf16 store per lane. The partial kernel, its 656-block
// geometry, the workspace, and the per-element arithmetic (int32 ascending
// split sum, fp32 left-to-right scale multiplies, RNE bf16) are unchanged,
// so the combine output stays bit-identical (0 mismatches) and Graph replay
// with changed contents must pass.
//
// Iteration 18 (workspace plane order [tile][split]): the accepted best is
// 43.00 us median and the iteration-14 PMC shows the operator is at the
// once-read-B DRAM floor: 19.08 MB of traffic per replay (17.65 MB read +
// 1.43 MB write) at 443.7 GB/s derived; partial kernel profiled 36.32 us
// (16.27 MB read + 1.34 MB write), combine kernel profiled 3.52 us
// (1.38 MB read, 31.4% L2 hit rate -- the 16.12-MB B stream thrashes L2, so
// ~0.95 MB of the combine's plane re-reads are DRAM-cold). The fusion
// attempts (iteration 15: x-major grid + last-arriver tail, 48.24 us;
// iteration 16: split-major grid + last-arriver tail, 43.96 us) both lost to
// the two-launch structure: the tail's plane re-reads are DRAM-cold under
// the x-major spread, and the tile-pair-major B order perturbs the main
// loop; no cache-policy hint is available (raw inline asm banned; the DTK's
// __ldcs/__ldcg are plain dereferences, unimplemented for int8/int4). The
// remaining exposed cost inside the combine kernel is the DRAM locality of
// its plane reads: with the [split][tile] layout each of the 164 combine
// blocks reads its tile's eight 1-KB planes at 164-KB strides -- 1,312
// scattered 1-KB cold streams. CHANGE: swap the plane order to [tile][split]
// (plane (tile, split) at (tile*SPLIT_K + split)*256 int32), so each combine
// block reads its tile's eight planes as ONE contiguous 8-KB region: 164
// sequential 8-KB streams instead of 1,312 scattered 1-KB streams (8x fewer
// concurrent streams, sequential-line access). The partial kernel's stores
// carry byte-identical values (same planes, same mem_row_major fragment
// stores; only the base index changes -- each block still writes two
// line-complete 1-KB planes), the ascending split-sum order and the
// per-element fp32 scale arithmetic are unchanged, so the combine output
// stays bit-identical (0 mismatches) and Graph replay with changed contents
// must pass. Exact shape guards, the M=2 scalar fallback, the n-major pack,
// the generic scalar fallback and the workspace guard (1,343,488 B, within
// the API's 16-plane budget) are untouched. Resource deltas: none (no
// register/LDS/grid changes; combine vmem_read stays 1,640/replay, vmem_write
// 164/replay; partial vmem_write stays 5,248/replay). Falsifiable
// prediction: if the combine's scattered cold 1-KB reads are part of its
// ~3.5 us wall, the contiguous 8-KB streams cut its DRAM service time and
// the median drops from 43.00 toward ~42.0-42.6 us with p90 below the
// current 43.07; a flat ~43.0-43.6 median falsifies it (the combine is
// launch/ramp-bound or already at the machine DRAM rate -- the residual is
// the once-read-B floor plus the combine launch, and the next round attacks
// the launch gap or re-tests byte reduction); a median above ~44.5 would
// show the store-address scatter hurt the partial kernel's write path and
// the round must be reverted.
//
// Contract (see int8_w8a8_gemm_api.py):
//   out[m, n] = bf16( int32_dot(x_q[m, :], weight[:, n])
//                     * x_scale[m, 0] * weight_scale[n, 0] )
// x_q         : int8  [M, K]  row-major
// weight      : int8  [K, N]  row-major logical; the packed_weight buffer
//               holds packed[n*K + k] = weight[k*N + n] (n-major transpose)
//               for the exact (k,n)==(6144,2624) and the identity [K, N]
//               copy for every other (K, N) (iteration 5)
// x_scale     : fp32  [M, 1]
// weight_scale: fp32  [N, 1]
// out         : bf16  [M, N]  row-major, caller-provided
// workspace   : uint8, caller-allocated before Graph capture; holds the
//               [164][8][256] int32 split-K partial planes (tile-major,
//               iteration 18: a tile's 8 split planes are one contiguous
//               8-KB region) for the exact shape (1,343,488 B, within the
//               API 16-plane budget)
//
// The timed operator (launch_w8a8_gemm) performs no allocation, compilation,
// autotuning, packing, host/device synchronization, or default-stream
// launch: it only validates the exact shape guard and launches the DUMMA
// kernel (or the generic scalar fallback) on the caller-provided stream.

#include <hip/hip_runtime.h>
#include <hip/hip_bfloat16.h>
#include <du_mma.h>

#include <cstdint>

namespace {

// gfx928 wavefront = 64 lanes; block sizes must be multiples of 64.
constexpr int kScalarBlockThreads = 128;
constexpr int kCopyBlockThreads = 256;

// DUMMA INT8 primitive: m16n16k32, int8 x int8 -> int32.
constexpr int kDummaTileM = 16;
constexpr int kDummaTileN = 16;
constexpr int kDummaTileK = 32;
constexpr int kDummaBlockThreads = 64;  // one wavefront per output tile

// ---------------------------------------------------------------------------
// Generic scalar INT8 dot-product GEMM (fallback for every shape).
//
// One thread computes one output element out[m, n]. The full K loop runs in
// int32 (exact for every assigned K; max K=6144 -> |dot| <= 6144*127*127
// ~= 9.9e7 << 2^31), then the two fp32 scales are applied and the result is
// stored as bf16.
//
// Thread t maps to (row = t / n, col = t % n): adjacent lanes land on
// adjacent addresses in the fastest-changing N dimension, so the per-K-row
// B reads (b[k*n + col]) stay lane-coalesced and the A reads are broadcast
// within a row.
// ---------------------------------------------------------------------------
__global__ __launch_bounds__(kScalarBlockThreads) void
w8a8_scalar_gemm_kernel(const int8_t* __restrict__ a,       // [M, K]
                        const int8_t* __restrict__ b,       // [K, N]
                        const float* __restrict__ x_scale,  // [M]
                        const float* __restrict__ weight_scale,  // [N]
                        hip_bfloat16* __restrict__ out,     // [M, N]
                        int m,
                        int n,
                        int k) {
  const int64_t total = static_cast<int64_t>(m) * n;
  const int64_t t =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (t >= total) {
    return;
  }
  const int row = static_cast<int>(t / n);
  const int col = static_cast<int>(t - static_cast<int64_t>(row) * n);

  const int8_t* a_row = a + static_cast<int64_t>(row) * k;
  // Iteration 5: for the exact (n, k) == (2624, 6144) the packed_weight
  // buffer holds the n-major transpose packed[col * k + kk] = raw[kk * n +
  // col] (produced once out of the timed region), so the fallback decodes
  // that layout there (keeps the paired M=2 API shape with the same (N, K)
  // byte-exact); every other (k, n) keeps the identity copy (raw row-major
  // [K, N]).
  const bool packed_nmajor = (n == 2624 && k == 6144);
  const int8_t* b_ptr = packed_nmajor ? b + static_cast<int64_t>(col) * k
                                      : b + col;
  const int b_stride = packed_nmajor ? 1 : n;
  int32_t acc = 0;
  for (int kk = 0; kk < k; ++kk) {
    acc += static_cast<int32_t>(a_row[kk]) *
           static_cast<int32_t>(b_ptr[0]);
    b_ptr += b_stride;
  }

  // Same arithmetic order as the harness reference
  // (dot.to(fp32) * x_scale * weight_scale.T): fp32 multiply left-to-right,
  // then round-to-nearest-even bf16.
  const float scaled = static_cast<float>(acc) * x_scale[row] *
                       weight_scale[col];
  out[t] = __float2bfloat16(scaled);
}

// ---------------------------------------------------------------------------
// Iteration 2 + iteration 4 + iteration 5: split-K partial + combine
// kernels (exact assigned shape M=16, N=2624, K=6144).
//
// Partial kernel geometry (iteration 4): one 64-lane wavefront per (two
// adjacent 16x16 N-tiles, K split) pair; grid = (N/32, SPLIT_K) = (82, 8) ->
// 656 one-wave zero-barrier blocks, each computing two 16x16 tiles over its
// own contiguous K slice (K/SPLIT_K = 768 = 24 k32 steps, k-ascending within
// the slice). The A fragment of each k32 step is loaded once and reused by
// both tiles (A slice bytes are the reused bytes); B fragments are loaded
// per tile, each B byte exactly once. No LDS, no __syncthreads, no atomics:
// every block writes only its own two 256-int32 planes, so the partial grid
// is deterministic and Graph-safe. Partial planes are published with
// du_store_matrix_sync (mem_row_major), the canonical accumulator-fragment
// store matching the verified gfx928 ownership (row = lane & 15,
// col = (lane >> 4) + 4*i).
//
// Iteration 5 (packed-weight round): the B fragments are now declared
// col_major against the packed [N, K] n-major weight
// (packed[n*K + k] = raw[k*N + n], produced once out of the timed region;
// see launch_pack_w8a8_weight). With the du_mma.hpp lane mappings, every
// lane's fragment bytes are 8 CONSECUTIVE bytes of one row/column, so the
// three fragments of a k32 step are fetched with one aligned 8-B vector
// load per lane each (int2 + __builtin_memcpy into the fragment storage --
// the same register-to-register fill the library loader produces): 3
// global_load_dwordx2 per step instead of 24 global_load_ubyte with the
// per-byte vmcnt ladder. Fragment VALUES are identical to the iteration-4
// library loads (row-major A: 8 bytes of row (lane&15) at k-offset
// ((lane>>4)*8); col_major B: 8 k bytes of column n0+(lane&15) at the same
// k-offset), so the k-ascending int32 accumulation and the partial planes
// stay bit-identical.
//
// Iteration 6 (pipeline round): the B data path now stages through LDS (see
// the kernel body): per 2-k32-step chunk, each of the two B tiles is fetched
// with ONE aligned 16-B cooperative load per lane (int4 / dwordx4, 64
// contiguous packed bytes per row), drained once, written to s_b (stride
// 72), and the per-step 8-B fragments are re-read from LDS with ds_read_b64
// while the next chunk's loads are already in flight. A stays direct (8-B
// int2, L2-hot). Fragment VALUES are unchanged, so partials stay
// bit-identical.
//
// Iteration 7 (pipeline round, double buffering): the single LDS plane and
// the rotation-copy pipeline are replaced by TWO ping-pong LDS planes
// (4,608 B) and TWO parity-alternating register sets with a two-chunk
// prefetch (see the kernel body): every load stays in flight across the
// back edge and is drained only at its own staging write. Fragment VALUES
// are unchanged, so partials stay bit-identical.
//
// Combine kernel geometry (unchanged from iteration 2, SPLIT_K=8): one
// 64-lane wavefront per 16x16 output tile (164 one-wave blocks, zero
// barriers); lane l owns the four contiguous row-major elements e = 4*l + i
// (row = e >> 4, col = e & 15), sums them across the SPLIT_K planes in
// ascending split order (exact int32 accumulation), applies
// x_scale[row] * weight_scale[col], and stores bf16.
//
// Iteration 13 (combine latency consolidation): the exact iteration-12
// code object's combine kernel (164 blocks, 48 VGPR, profiled 4.48 us)
// issues the 8 plane dwordx4 loads up front, but loads the four
// weight_scale elements with four SEPARATE 4-B loads, each waited to
// completion (vmcnt(0)) before its multiply and its own 2-B bf16 store --
// four serialized round trips per lane. This round replaces them with ONE
// aligned 16-B float4 scale load (issued before the plane reads) and ONE
// 8-B packed store per lane; per-element arithmetic (int32 ascending sum,
// fp32 left-to-right scale multiplies, RNE bf16) is byte-for-byte
// unchanged, so the combine output stays bit-identical.
// ---------------------------------------------------------------------------
template <int SPLIT_K>
__global__ __launch_bounds__(kDummaBlockThreads) void
w8a8_dumma_m16n16k32_splitk_partial_kernel(
    const int8_t* __restrict__ a,          // [16, K] row-major (logical A)
    const int8_t* __restrict__ b,          // packed [N, K] n-major (iter 5)
    int* __restrict__ partials,            // [N/16][SPLIT_K][256] int32
                                           // (tile-major, iteration 18)
    int n,
    int k) {
  constexpr int kNTiles = 2;  // adjacent 16x16 N-tiles per block
  const int tile0 = static_cast<int>(blockIdx.x) * kNTiles;  // 0..N/16-2
  const int split = static_cast<int>(blockIdx.y);  // 0..SPLIT_K-1
  const int n0 = tile0 * kDummaTileN;
  const int k_len = k / SPLIT_K;  // K % SPLIT_K == 0 (exact shape guard)
  const int k_lo = split * k_len;
  const int lane = static_cast<int>(threadIdx.x);  // 0..63

  // Iteration 7 (pipeline round, B-only LDS DOUBLE buffering): the exact
  // iteration-6 code object (source digest f04e11e9..., 41 VGPR, single
  // 2,304-B LDS plane) keeps the next chunk's B loads in flight across the
  // back edge, but drains the next chunk's A loads at the BOTTOM of every
  // body (s_waitcnt vmcnt(1)/vmcnt(0) at 0x56E8/0x56F0 immediately before
  // the rotation v_movs v24->v22/v26->v20): the register rotation forces a
  // per-chunk vmcnt drain on the A path, so A (L2-hot) is re-exposed to L2
  // latency 12x per block with ZERO cross-chunk overlap, and the cold B
  // stream gets only ~1 chunk of hiding. This round double-buffers the B
  // staging: TWO 2,304-B LDS planes (4,608 B/block total; 64 KiB/4,608 B =
  // 14 blocks/CU of LDS headroom vs the scheduled 5.47) and TWO register
  // sets alternating by chunk parity, each chunk's four loads (2 dwordx4 B
  // + 2 dwordx2 A) issued TWO chunks ahead and consumed IN PLACE by the
  // matching parity stage -- there are no rotation copies, so no bottom-of-
  // body vmcnt drains: every load stays in flight across the back edge and
  // the single vmcnt drain per chunk lands only at its own LDS staging
  // write. A now gets ~2 chunks of latency hiding (was ~0), B ~2 (was ~1).
  // Row stride 72 = 64 + 8 keeps every ds_read_b64/ds_write_b64 address 8-B
  // aligned AND gives 18*n mod 32 distinct row bank bases for n = 0..15 (no
  // LDS bank conflicts on either the fragment reads or the b64 staging
  // writes; the ping-pong plane offset 2,304 B = 576 dwords is 0 mod 32, so
  // both planes keep the same conflict-free bank pattern). One wavefront per
  // block: no __syncthreads, no barriers -- the ping-pong parity replaces
  // the cross-wave barrier a multi-wave block would need.
  //
  // Iteration 9 (HIP-only pipeline round, stage width 64 K -> 128 K): the
  // iteration-8 occupancy probe (SPLIT_K=12, 984 blocks = 2.05 waves/SIMD)
  // REGRESSED to 50.46 us median (vs the iteration-7 best 47.59), falsifying
  // per-SIMD co-residency as the limiter and pointing back at the per-block
  // serial cost: each 64-K chunk still pays one near-full cold-B vmcnt drain
  // (12 chunks/block), and the issue-to-drain window is only ~1 pass of work
  // (~8 MMAs + 4 ds_read2 + 2 stagings) against a ~600+ cycle HBM cold
  // latency. This round keeps the winning 656-block geometry and the exact
  // double-buffer ping-pong structure but WIDENS each staged chunk to FOUR
  // k32 steps (128 K): per chunk each B tile is fetched with TWO aligned
  // 16-B cooperative loads per lane (global_load_dwordx4 at b_wide and
  // b_wide+64; 128 CONTIGUOUS packed B bytes per row, was 64), the drain
  // count halves (12 -> 6), and the in-flight window doubles (each reload
  // set now carries 2 x 32 B of B + 32 B of A per lane). LDS planes become
  // s_b[2][2][16][136] = 8,704 B/block (stride 136 = 128 + 8: 34 dwords = 2
  // mod 32 -> row bank bases 2*n mod 32, still 16 distinct bases for n =
  // 0..15, so reads (ds_read2_b64 pairs at +0/+32 and +64/+96) and staging
  // writes (ds_write2_b64 at b_wide and b_wide+64) stay conflict-free; the
  // plane offset 4,352 B = 1,088 dwords is 0 mod 32, so both planes keep the
  // same pattern). LDS 8,704 B -> 7 blocks/CU of headroom vs the scheduled
  // 5.47; VGPR ~58 -> ~80-95 (in-flight set registers 24 -> 48; no spills
  // expected, 65,536/(95*64) ~= 10 waves/CU >= 5.47). vmem_read stays
  // 31,488/replay (6 chunks x 4 dwordx4 B + 4 dwordx2 A = 48 loads/block,
  // same total), LDS instructions stay 31,488/replay (24 ds_write2 + 24
  // ds_read2 per block), LDS traffic unchanged (every B byte still staged
  // exactly once), HBM read unchanged at the compulsory ~17.66 MB floor.
  // Fragment VALUES are byte-identical (the staged rows hold the same packed
  // bytes; the per-step 8-B ds_read_b64 fills are unchanged), the k-ascending
  // int32 accumulation order is unchanged, so the partial planes and the
  // combine output stay bit-identical (0 mismatches) and Graph replay with
  // changed contents must pass.
  //
  // Iteration 10 (stage width 128 K -> 192 K): the exact iteration-9 code
  // object emits one full progressive vmcnt ladder per PASS at the pass top
  // (vmcnt(7)..vmcnt(0) at 0x571C-0x574C, covering all 16 loads issued in the
  // previous pass; the B staging writes and the A capture after it are
  // wait-free), so the real drain-event count is 3 per block (passes), each
  // exposing the cold-B round trip of 2 x 128-K chunks. This round WIDENS
  // each staged chunk to SIX k32 steps (192 K) with the same double-buffer
  // ping-pong structure: per chunk each B tile is fetched with THREE aligned
  // 16-B cooperative loads per lane (global_load_dwordx4 at b_wide,
  // b_wide+64, b_wide+128; 192 CONTIGUOUS packed B bytes per row, was 128),
  // drain events drop 3 -> 2 per block, and each pass now carries 24 loads
  // (2 chunks x 6 dwordx4 B + 6 dwordx2 A) against an issue-to-drain window
  // of one full pass (24 MMAs + 12 ds_read2 + 12 stagings + 24 loads). LDS
  // planes become s_b[2][2][16][200] = 12,800 B/block (stride 200 = 192 + 8:
  // 50 dwords = 18 mod 32 -> row bank bases 18*n mod 32, still 16 distinct
  // bases for n = 0..15, so reads (ds_read2_b64 at +0/+32/+64/+96/+128/+160)
  // and staging writes (ds_write2_b64 at b_wide, b_wide+64, b_wide+128) stay
  // conflict-free; the plane offset 6,400 B = 1,600 dwords is 0 mod 32, so
  // both planes keep the same pattern). LDS 12,800 B -> 5 blocks/CU of
  // headroom vs the scheduled 5.47 (5 slots still >= the 4 SIMDs per CU);
  // VGPR ~96 -> ~110-130 (in-flight set registers 48 -> 72; no spills
  // expected, 65,536/(130*64) ~= 7.9 waves/CU >= 5.47). vmem_read stays
  // 31,488/replay (4 chunks x 12 loads = 48 loads/block, same total), LDS
  // instructions stay 31,488/replay (24 ds_write2 + 24 ds_read2 per block),
  // LDS traffic unchanged (every B byte still staged exactly once), HBM read
  // unchanged at the compulsory ~17.66 MB floor. Fragment VALUES are
  // byte-identical (the staged rows hold the same packed bytes; the per-step
  // 8-B ds_read_b64 fills are unchanged), the k-ascending int32 accumulation
  // order is unchanged, so the partial planes and the combine output stay
  // bit-identical (0 mismatches) and Graph replay with changed contents must
  // pass.
  alignas(16) __shared__ int8_t s_b[2][kNTiles][kDummaTileN][200];

  du::dumma::DUFragment<du::dumma::matrix_a, kDummaTileM, kDummaTileN,
                        kDummaTileK, signed char, du::dumma::row_major>
      a_frag0;
  du::dumma::DUFragment<du::dumma::matrix_a, kDummaTileM, kDummaTileN,
                        kDummaTileK, signed char, du::dumma::row_major>
      a_frag1;
  du::dumma::DUFragment<du::dumma::matrix_a, kDummaTileM, kDummaTileN,
                        kDummaTileK, signed char, du::dumma::row_major>
      a_frag2;
  du::dumma::DUFragment<du::dumma::matrix_a, kDummaTileM, kDummaTileN,
                        kDummaTileK, signed char, du::dumma::row_major>
      a_frag3;
  du::dumma::DUFragment<du::dumma::matrix_a, kDummaTileM, kDummaTileN,
                        kDummaTileK, signed char, du::dumma::row_major>
      a_frag4;
  du::dumma::DUFragment<du::dumma::matrix_a, kDummaTileM, kDummaTileN,
                        kDummaTileK, signed char, du::dumma::row_major>
      a_frag5;
  du::dumma::DUFragment<du::dumma::matrix_b, kDummaTileM, kDummaTileN,
                        kDummaTileK, signed char, du::dumma::col_major>
      b_frag0;
  du::dumma::DUFragment<du::dumma::matrix_b, kDummaTileM, kDummaTileN,
                        kDummaTileK, signed char, du::dumma::col_major>
      b_frag1;
  du::dumma::DUFragment<du::dumma::accumulator, kDummaTileM, kDummaTileN,
                        kDummaTileK, int>
      acc0_frag;
  du::dumma::DUFragment<du::dumma::accumulator, kDummaTileM, kDummaTileN,
                        kDummaTileK, int>
      acc1_frag;
  du::dumma::du_fill_fragment(acc0_frag, 0);
  du::dumma::du_fill_fragment(acc1_frag, 0);

  // Lane roles are fixed by the m16n16k32 fragment layouts in du_mma.hpp
  // (see the iteration-5 note): row-major A lane l holds 8 consecutive bytes
  // of row (l&15) at k-offset ((l>>4)*8); col_major B over the packed buffer
  // holds 8 consecutive k bytes of column n0+(l&15) at the same k-offset.
  // Iteration 6 widens the B LOADS to 16 B per lane (int4 / dwordx4): lane
  // (row, c) fetches packed bytes [c*16, c*16+16) of its column across a
  // 2-step (64-K) chunk, i.e. 64 CONTIGUOUS packed B bytes per row per
  // instruction (was 32 B per row per step), stages them into LDS, and re-
  // reads the per-step 8-B fragment from LDS -- so one global drain covers
  // the whole 2-step chunk and the MMA path is decoupled from global latency.
  const int a_row = lane & 0xf;
  const int a_col = (lane >> 4) << 3;   // {0, 8, 16, 24} (A fragment offset)
  const int b_row = lane & 0xf;         // n index within the 16-col tile
  const int b_col = (lane >> 4) << 3;   // {0, 8, 16, 24} (B fragment offset)
  const int b_wide = (lane >> 4) << 4;  // {0, 16, 32, 48} (B 16-B load offset)
  const int8_t* a_base =
      a + static_cast<int64_t>(a_row) * k + k_lo + a_col;
  // B chunk bases are 16-B aligned: k = 6144 is a multiple of 16, n0 and
  // k_lo are multiples of 16/64, and b_wide is a multiple of 16.
  const int8_t* b0_base =
      b + static_cast<int64_t>(n0 + b_row) * k + k_lo;
  const int8_t* b1_base = b0_base + static_cast<int64_t>(kDummaTileN) * k;

  constexpr int kChunkK = 6 * kDummaTileK;  // 192 K per staged chunk (iter 10)

  // Iteration 12 (single cold-B drain per block): the exact iteration-10
  // code object (source digest 6ca5190e..., 130 VGPR/12,800 B LDS/0 scratch,
  // compile_cache_key 34ee6cfd...) emits TWO cold-B vmcnt drains per block:
  // (1) the prologue ladder s_waitcnt vmcnt(13)..vmcnt(0) at 0x575C-0x57A4
  // covering chunks 0+1 with a ZERO issue-to-drain window, and (2) the
  // pass-0 bottom ladder s_waitcnt vmcnt(5)..vmcnt(0) at 0x5AD8-0x5B00
  // covering the 24 reloaded pass-1 loads (chunks 2+3) before the 24
  // rotation v_movs -- so every block still serially exposes TWO cold-B
  // round trips. This round issues ALL FOUR chunks' B loads (24
  // global_load_dwordx4) plus the chunks 0/1 A loads in the prologue, so
  // the FIRST staging consumer drains the whole K-slice with ONE vmcnt
  // ladder: one cold-B round trip per block, the pass-1 staging (chunks
  // 2+3) becomes wait-free, and the 24 rotation v_movs disappear. The loop
  // is UNROLLED into two explicit passes because the B register sets are no
  // longer reloaded in place (the exact-shape launcher guard guarantees
  // k_len = 1536 at SPLIT_K = 4, i.e. chunks == 8 -- the only launch
  // configuration of this kernel; iteration 20 adds the second 4-chunk
  // body below). A stays L2-pipelined exactly as in
  // iteration 10: the chunk 2/3 A loads are issued inside pass 0 (after the
  // chunk 0/1 captures, reusing the dead set-0/1 A registers) and are
  // consumed by the pass-1 captures one full pass later. Resource deltas:
  // in-flight load registers 72 -> 120 (24 dwordx4 B = 96 regs + 12 int2 A
  // = 24 regs at the prologue), expected arch_vgpr ~136 -> ~150-175
  // (65,536/(175*64) ~= 5.85 waves/CU >= the LDS-capped 5 slots, no spills
  // at < 256 VGPR, scratch must stay 0), LDS
  // stays 12,800 B (5 blocks/CU, unchanged), grid stays (82,8) = 656
  // one-wave zero-barrier blocks. vmem_read stays 31,488/replay (the same
  // 48 loads/block: 24 dwordx4 B + 24 dwordx2 A -- 36 issued in the
  // prologue, 12 A issued in pass 0), LDS instructions stay 31,488/replay
  // (24 ds_write2 + 24 ds_read2 per block), LDS traffic unchanged (every B
  // byte still staged exactly once), HBM read unchanged at the compulsory
  // ~17.66 MB floor. Exactness is preserved: the staged LDS rows hold
  // byte-for-byte the same packed B bytes (same planes, same b_wide /
  // b_wide+64 / b_wide+128 staging addresses, same 8-B ds_read_b64 fragment
  // fills), the A fragments are unchanged, and the k-ascending int32
  // accumulation order within each split slice (chunks 0,1,2,3) is
  // unchanged, so the partial planes and the combine output stay
  // bit-identical (0 mismatches) and Graph replay with changed contents
  // must pass. Falsifiable prediction: if the residual limiter is the
  // per-block serial cold-B drain chain (the mechanism behind iterations 9
  // and 10), removing the second cold-B round trip should land the median
  // in roughly 42.0-43.3 us with p90 below the current 43.92 (combine
  // ~2.5-3 us unchanged); a flat ~43.8 us median falsifies the drain-chain
  // hypothesis (the residual would be the memory-pipe / launch+combine
  // floor, and the next round would attack the combine or the B request
  // partition); a median above ~45 us would show register-pressure or
  // scheduling loss (e.g. the compiler spilling the 96 in-flight B
  // registers or re-inserting a per-pass wait at the pass-1 staging).
  //
  // Iteration 20 (SPLIT_K 8 -> 4 byte-reduction round): the operator is at
  // the once-read-B floor -- the counter-derived whole-operator HBM rate is
  // ~448 GB/s and the partial kernel alone moves 16.22 MB read + 1.34 MB
  // write in ~36.5 us (~480 GB/s effective), so the iteration-18/19
  // falsification branches' remaining lever is BYTE REDUCTION: the workspace
  // plane traffic (1.34 MB write + 1.31 MB combine re-read at SPLIT_K=8)
  // halves at SPLIT_K=4. The partial kernel keeps the exact iteration-12
  // single-drain-per-group structure but covers EIGHT 192-K chunks (k_len =
  // 1536): body 1 (chunks 0-3) is the accepted iteration-12 code path
  // verbatim; body 2 (chunks 4-7) re-issues its 24 dwordx4 B loads plus the
  // chunks 4/5 A loads into the body-1 dead registers (same 96-B + 48-B live
  // load sets -> arch_vgpr stays ~120-140, LDS stays 12,800 B, no spills)
  // and drains them with ONE ladder at its chunk-4 staging -- the block's
  // second cold-B round trip, but device-wide the drain count is unchanged
  // (328 blocks x 2 = 656 vs the accepted 656 blocks x 1), and drain count
  // was measured ~flat at the memory ceiling (iteration 10: 3 drains 43.81
  // us vs iteration 12: 1 drain 43.64 us). Grid becomes (82,4) = 328 one-wave
  // zero-barrier blocks = 2.73 blocks/CU, a SINGLE wave (all resident), vs
  // the old 656-block 1.31-wave grid. The combine kernel is templated on
  // SPLIT_K and needs NO change: 164 one-wave blocks each sum the four
  // planes in ascending split order with the fused x_scale * weight_scale
  // bf16 epilogue. Exactness: the int32 accumulation stays k-ascending
  // within each split slice (chunks 0..7 in order, 48 k32 steps) and the
  // combine sums s=0..3 ascending, so the output stays bit-identical (0
  // mismatches) and Graph replay with changed contents must pass. The
  // [tile][split] workspace layout (iteration 18), the n-major pack
  // (iteration 5), the exact shape guards, the M=2 scalar fallback and the
  // two-launch Graph structure are untouched (the pack and the scalar
  // fallback do not depend on SPLIT_K). Resource deltas: partial vmem_read
  // 31,488 -> 23,616/replay (48 loads/block x 328), vmem_write 5,248 ->
  // 2,624/replay (plane bytes halved), LDS instructions 31,488 -> 15,744/
  // replay, LDS stays 12,800 B/block, scratch stays 0; combine vmem_read
  // 1,640 -> ~984/replay, vmem_write stays 164/replay; HBM traffic drops
  // 19,080,640 -> ~17,771,008 B per replay (partial read 16.22 MB compulsory
  // + partial write 0.66 MB + combine read 0.66 MB + out 0.08 MB).
  // Falsifiable prediction: if the machine's effective HBM rate holds at
  // ~480 GB/s for the smaller grid, the 1.31-MB plane-traffic cut lands the
  // median in roughly 39.5-40.8 us with p90 below the current 42.61 (partial
  // toward ~35.2 us profiled, combine toward ~2.5 us, launch gap unchanged);
  // a flat ~42.5 us median falsifies it (the smaller grid or the second
  // per-block drain costs as much as the byte cut -- the machine's effective
  // rate drops with grid size -- and the next round reverts or tries
  // SPLIT_K=2 / the launch gap); a median above ~44.5 us shows the second
  // zero-window drain serialized per-block latency or a register/scheduling
  // loss and the round must be reverted.

  // ---- Prologue: all four chunks' B loads (24 dwordx4) + chunks 0/1 A.
  // The first staging consumer below drains the whole set with ONE vmcnt
  // ladder: a single cold-B round trip per block (was two). ----
  int4 b0s0a = *reinterpret_cast<const int4*>(b0_base + b_wide);
  int4 b0s0b = *reinterpret_cast<const int4*>(b0_base + b_wide + 64);
  int4 b0s0c = *reinterpret_cast<const int4*>(b0_base + b_wide + 128);
  int4 b1s0a = *reinterpret_cast<const int4*>(b1_base + b_wide);
  int4 b1s0b = *reinterpret_cast<const int4*>(b1_base + b_wide + 64);
  int4 b1s0c = *reinterpret_cast<const int4*>(b1_base + b_wide + 128);
  int4 b0s1a = *reinterpret_cast<const int4*>(b0_base + kChunkK + b_wide);
  int4 b0s1b =
      *reinterpret_cast<const int4*>(b0_base + kChunkK + b_wide + 64);
  int4 b0s1c =
      *reinterpret_cast<const int4*>(b0_base + kChunkK + b_wide + 128);
  int4 b1s1a = *reinterpret_cast<const int4*>(b1_base + kChunkK + b_wide);
  int4 b1s1b =
      *reinterpret_cast<const int4*>(b1_base + kChunkK + b_wide + 64);
  int4 b1s1c =
      *reinterpret_cast<const int4*>(b1_base + kChunkK + b_wide + 128);
  int4 b0s2a = *reinterpret_cast<const int4*>(b0_base + 2 * kChunkK + b_wide);
  int4 b0s2b =
      *reinterpret_cast<const int4*>(b0_base + 2 * kChunkK + b_wide + 64);
  int4 b0s2c =
      *reinterpret_cast<const int4*>(b0_base + 2 * kChunkK + b_wide + 128);
  int4 b1s2a = *reinterpret_cast<const int4*>(b1_base + 2 * kChunkK + b_wide);
  int4 b1s2b =
      *reinterpret_cast<const int4*>(b1_base + 2 * kChunkK + b_wide + 64);
  int4 b1s2c =
      *reinterpret_cast<const int4*>(b1_base + 2 * kChunkK + b_wide + 128);
  int4 b0s3a = *reinterpret_cast<const int4*>(b0_base + 3 * kChunkK + b_wide);
  int4 b0s3b =
      *reinterpret_cast<const int4*>(b0_base + 3 * kChunkK + b_wide + 64);
  int4 b0s3c =
      *reinterpret_cast<const int4*>(b0_base + 3 * kChunkK + b_wide + 128);
  int4 b1s3a = *reinterpret_cast<const int4*>(b1_base + 3 * kChunkK + b_wide);
  int4 b1s3b =
      *reinterpret_cast<const int4*>(b1_base + 3 * kChunkK + b_wide + 64);
  int4 b1s3c =
      *reinterpret_cast<const int4*>(b1_base + 3 * kChunkK + b_wide + 128);
  int2 a0s0 = *reinterpret_cast<const int2*>(a_base);
  int2 a1s0 = *reinterpret_cast<const int2*>(a_base + kDummaTileK);
  int2 a2s0 = *reinterpret_cast<const int2*>(a_base + 2 * kDummaTileK);
  int2 a3s0 = *reinterpret_cast<const int2*>(a_base + 3 * kDummaTileK);
  int2 a4s0 = *reinterpret_cast<const int2*>(a_base + 4 * kDummaTileK);
  int2 a5s0 = *reinterpret_cast<const int2*>(a_base + 5 * kDummaTileK);
  int2 a0s1 = *reinterpret_cast<const int2*>(a_base + kChunkK);
  int2 a1s1 =
      *reinterpret_cast<const int2*>(a_base + kChunkK + kDummaTileK);
  int2 a2s1 =
      *reinterpret_cast<const int2*>(a_base + kChunkK + 2 * kDummaTileK);
  int2 a3s1 =
      *reinterpret_cast<const int2*>(a_base + kChunkK + 3 * kDummaTileK);
  int2 a4s1 =
      *reinterpret_cast<const int2*>(a_base + kChunkK + 4 * kDummaTileK);
  int2 a5s1 =
      *reinterpret_cast<const int2*>(a_base + kChunkK + 5 * kDummaTileK);

  // ---- Pass 0: chunks 0 and 1. Each B tile's 48 B per lane (three dwordx4
  // halves at b_wide, b_wide+64, b_wide+128) land in the 192-B row: six
  // ds_write2_b64 per chunk, all conflict-free (row bank bases 18*n mod 32).
  {
    // Stage chunk 0 (set 0) into LDS plane 0 -- the single cold-B drain of
    // the whole block lands here (first consumer of the 36 prologue loads).
    {
      int2* d0 = reinterpret_cast<int2*>(&s_b[0][0][b_row][b_wide]);
      int2* d1 = reinterpret_cast<int2*>(&s_b[0][1][b_row][b_wide]);
      d0[0] = *reinterpret_cast<const int2*>(&b0s0a);
      d0[1] = *reinterpret_cast<const int2*>(
          reinterpret_cast<const int*>(&b0s0a) + 2);
      d1[0] = *reinterpret_cast<const int2*>(&b1s0a);
      d1[1] = *reinterpret_cast<const int2*>(
          reinterpret_cast<const int*>(&b1s0a) + 2);
      int2* d0b = reinterpret_cast<int2*>(&s_b[0][0][b_row][b_wide + 64]);
      int2* d1b = reinterpret_cast<int2*>(&s_b[0][1][b_row][b_wide + 64]);
      d0b[0] = *reinterpret_cast<const int2*>(&b0s0b);
      d0b[1] = *reinterpret_cast<const int2*>(
          reinterpret_cast<const int*>(&b0s0b) + 2);
      d1b[0] = *reinterpret_cast<const int2*>(&b1s0b);
      d1b[1] = *reinterpret_cast<const int2*>(
          reinterpret_cast<const int*>(&b1s0b) + 2);
      int2* d0c = reinterpret_cast<int2*>(&s_b[0][0][b_row][b_wide + 128]);
      int2* d1c = reinterpret_cast<int2*>(&s_b[0][1][b_row][b_wide + 128]);
      d0c[0] = *reinterpret_cast<const int2*>(&b0s0c);
      d0c[1] = *reinterpret_cast<const int2*>(
          reinterpret_cast<const int*>(&b0s0c) + 2);
      d1c[0] = *reinterpret_cast<const int2*>(&b1s0c);
      d1c[1] = *reinterpret_cast<const int2*>(
          reinterpret_cast<const int*>(&b1s0c) + 2);
    }
    // Capture chunk 0's six A fragments (steps 0..5) BEFORE set 0's A
    // registers are overwritten below by the chunk-2 A issue; these captures
    // are the first consumers of the chunks 0/1 A loads (L2-hot, complete
    // during the single cold-B drain).
    __builtin_memcpy(a_frag0.x, &a0s0, 8);
    __builtin_memcpy(a_frag1.x, &a1s0, 8);
    __builtin_memcpy(a_frag2.x, &a2s0, 8);
    __builtin_memcpy(a_frag3.x, &a3s0, 8);
    __builtin_memcpy(a_frag4.x, &a4s0, 8);
    __builtin_memcpy(a_frag5.x, &a5s0, 8);
    // Issue chunk 2's A loads (L2-hot) into the now-dead set-0 A registers;
    // consumed by the pass-1 capture one full pass later.
    a0s0 = *reinterpret_cast<const int2*>(a_base + 2 * kChunkK);
    a1s0 = *reinterpret_cast<const int2*>(
        a_base + 2 * kChunkK + kDummaTileK);
    a2s0 = *reinterpret_cast<const int2*>(
        a_base + 2 * kChunkK + 2 * kDummaTileK);
    a3s0 = *reinterpret_cast<const int2*>(
        a_base + 2 * kChunkK + 3 * kDummaTileK);
    a4s0 = *reinterpret_cast<const int2*>(
        a_base + 2 * kChunkK + 4 * kDummaTileK);
    a5s0 = *reinterpret_cast<const int2*>(
        a_base + 2 * kChunkK + 5 * kDummaTileK);
    // Consume chunk 0 from LDS plane 0: six k32 steps covering k offsets
    // [0, 32), [32, 64), [64, 96), [96, 128), [128, 160), [160, 192) of the
    // chunk. All twelve fragment reads issue up front so the LDS latency of
    // the whole chunk is hidden behind the twelve-MMA burst.
    {
      const int2 rb00 =
          *reinterpret_cast<const int2*>(&s_b[0][0][b_row][b_col]);
      const int2 rb10 =
          *reinterpret_cast<const int2*>(&s_b[0][1][b_row][b_col]);
      const int2 rb01 = *reinterpret_cast<const int2*>(
          &s_b[0][0][b_row][b_col + kDummaTileK]);
      const int2 rb11 = *reinterpret_cast<const int2*>(
          &s_b[0][1][b_row][b_col + kDummaTileK]);
      const int2 rb02 = *reinterpret_cast<const int2*>(
          &s_b[0][0][b_row][b_col + 2 * kDummaTileK]);
      const int2 rb12 = *reinterpret_cast<const int2*>(
          &s_b[0][1][b_row][b_col + 2 * kDummaTileK]);
      const int2 rb03 = *reinterpret_cast<const int2*>(
          &s_b[0][0][b_row][b_col + 3 * kDummaTileK]);
      const int2 rb13 = *reinterpret_cast<const int2*>(
          &s_b[0][1][b_row][b_col + 3 * kDummaTileK]);
      const int2 rb04 = *reinterpret_cast<const int2*>(
          &s_b[0][0][b_row][b_col + 4 * kDummaTileK]);
      const int2 rb14 = *reinterpret_cast<const int2*>(
          &s_b[0][1][b_row][b_col + 4 * kDummaTileK]);
      const int2 rb05 = *reinterpret_cast<const int2*>(
          &s_b[0][0][b_row][b_col + 5 * kDummaTileK]);
      const int2 rb15 = *reinterpret_cast<const int2*>(
          &s_b[0][1][b_row][b_col + 5 * kDummaTileK]);
      __builtin_memcpy(b_frag0.x, &rb00, 8);
      __builtin_memcpy(b_frag1.x, &rb10, 8);
      du::dumma::du_mma_sync(acc0_frag, a_frag0, b_frag0, acc0_frag);
      du::dumma::du_mma_sync(acc1_frag, a_frag0, b_frag1, acc1_frag);
      __builtin_memcpy(b_frag0.x, &rb01, 8);
      __builtin_memcpy(b_frag1.x, &rb11, 8);
      du::dumma::du_mma_sync(acc0_frag, a_frag1, b_frag0, acc0_frag);
      du::dumma::du_mma_sync(acc1_frag, a_frag1, b_frag1, acc1_frag);
      __builtin_memcpy(b_frag0.x, &rb02, 8);
      __builtin_memcpy(b_frag1.x, &rb12, 8);
      du::dumma::du_mma_sync(acc0_frag, a_frag2, b_frag0, acc0_frag);
      du::dumma::du_mma_sync(acc1_frag, a_frag2, b_frag1, acc1_frag);
      __builtin_memcpy(b_frag0.x, &rb03, 8);
      __builtin_memcpy(b_frag1.x, &rb13, 8);
      du::dumma::du_mma_sync(acc0_frag, a_frag3, b_frag0, acc0_frag);
      du::dumma::du_mma_sync(acc1_frag, a_frag3, b_frag1, acc1_frag);
      __builtin_memcpy(b_frag0.x, &rb04, 8);
      __builtin_memcpy(b_frag1.x, &rb14, 8);
      du::dumma::du_mma_sync(acc0_frag, a_frag4, b_frag0, acc0_frag);
      du::dumma::du_mma_sync(acc1_frag, a_frag4, b_frag1, acc1_frag);
      __builtin_memcpy(b_frag0.x, &rb05, 8);
      __builtin_memcpy(b_frag1.x, &rb15, 8);
      du::dumma::du_mma_sync(acc0_frag, a_frag5, b_frag0, acc0_frag);
      du::dumma::du_mma_sync(acc1_frag, a_frag5, b_frag1, acc1_frag);
    }
  }
  {
    // Stage chunk 1 (set 1) into LDS plane 1 (wait-free: already drained
    // with chunk 0 by the single prologue ladder).
    {
      int2* d0 = reinterpret_cast<int2*>(&s_b[1][0][b_row][b_wide]);
      int2* d1 = reinterpret_cast<int2*>(&s_b[1][1][b_row][b_wide]);
      d0[0] = *reinterpret_cast<const int2*>(&b0s1a);
      d0[1] = *reinterpret_cast<const int2*>(
          reinterpret_cast<const int*>(&b0s1a) + 2);
      d1[0] = *reinterpret_cast<const int2*>(&b1s1a);
      d1[1] = *reinterpret_cast<const int2*>(
          reinterpret_cast<const int*>(&b1s1a) + 2);
      int2* d0b = reinterpret_cast<int2*>(&s_b[1][0][b_row][b_wide + 64]);
      int2* d1b = reinterpret_cast<int2*>(&s_b[1][1][b_row][b_wide + 64]);
      d0b[0] = *reinterpret_cast<const int2*>(&b0s1b);
      d0b[1] = *reinterpret_cast<const int2*>(
          reinterpret_cast<const int*>(&b0s1b) + 2);
      d1b[0] = *reinterpret_cast<const int2*>(&b1s1b);
      d1b[1] = *reinterpret_cast<const int2*>(
          reinterpret_cast<const int*>(&b1s1b) + 2);
      int2* d0c = reinterpret_cast<int2*>(&s_b[1][0][b_row][b_wide + 128]);
      int2* d1c = reinterpret_cast<int2*>(&s_b[1][1][b_row][b_wide + 128]);
      d0c[0] = *reinterpret_cast<const int2*>(&b0s1c);
      d0c[1] = *reinterpret_cast<const int2*>(
          reinterpret_cast<const int*>(&b0s1c) + 2);
      d1c[0] = *reinterpret_cast<const int2*>(&b1s1c);
      d1c[1] = *reinterpret_cast<const int2*>(
          reinterpret_cast<const int*>(&b1s1c) + 2);
    }
    // Capture chunk 1's six A fragments before the chunk-3 A issue below.
    __builtin_memcpy(a_frag0.x, &a0s1, 8);
    __builtin_memcpy(a_frag1.x, &a1s1, 8);
    __builtin_memcpy(a_frag2.x, &a2s1, 8);
    __builtin_memcpy(a_frag3.x, &a3s1, 8);
    __builtin_memcpy(a_frag4.x, &a4s1, 8);
    __builtin_memcpy(a_frag5.x, &a5s1, 8);
    // Issue chunk 3's A loads (L2-hot) into the now-dead set-1 A registers.
    a0s1 = *reinterpret_cast<const int2*>(a_base + 3 * kChunkK);
    a1s1 = *reinterpret_cast<const int2*>(
        a_base + 3 * kChunkK + kDummaTileK);
    a2s1 = *reinterpret_cast<const int2*>(
        a_base + 3 * kChunkK + 2 * kDummaTileK);
    a3s1 = *reinterpret_cast<const int2*>(
        a_base + 3 * kChunkK + 3 * kDummaTileK);
    a4s1 = *reinterpret_cast<const int2*>(
        a_base + 3 * kChunkK + 4 * kDummaTileK);
    a5s1 = *reinterpret_cast<const int2*>(
        a_base + 3 * kChunkK + 5 * kDummaTileK);
    // Consume chunk 1 from LDS plane 1.
    {
      const int2 rb00 =
          *reinterpret_cast<const int2*>(&s_b[1][0][b_row][b_col]);
      const int2 rb10 =
          *reinterpret_cast<const int2*>(&s_b[1][1][b_row][b_col]);
      const int2 rb01 = *reinterpret_cast<const int2*>(
          &s_b[1][0][b_row][b_col + kDummaTileK]);
      const int2 rb11 = *reinterpret_cast<const int2*>(
          &s_b[1][1][b_row][b_col + kDummaTileK]);
      const int2 rb02 = *reinterpret_cast<const int2*>(
          &s_b[1][0][b_row][b_col + 2 * kDummaTileK]);
      const int2 rb12 = *reinterpret_cast<const int2*>(
          &s_b[1][1][b_row][b_col + 2 * kDummaTileK]);
      const int2 rb03 = *reinterpret_cast<const int2*>(
          &s_b[1][0][b_row][b_col + 3 * kDummaTileK]);
      const int2 rb13 = *reinterpret_cast<const int2*>(
          &s_b[1][1][b_row][b_col + 3 * kDummaTileK]);
      const int2 rb04 = *reinterpret_cast<const int2*>(
          &s_b[1][0][b_row][b_col + 4 * kDummaTileK]);
      const int2 rb14 = *reinterpret_cast<const int2*>(
          &s_b[1][1][b_row][b_col + 4 * kDummaTileK]);
      const int2 rb05 = *reinterpret_cast<const int2*>(
          &s_b[1][0][b_row][b_col + 5 * kDummaTileK]);
      const int2 rb15 = *reinterpret_cast<const int2*>(
          &s_b[1][1][b_row][b_col + 5 * kDummaTileK]);
      __builtin_memcpy(b_frag0.x, &rb00, 8);
      __builtin_memcpy(b_frag1.x, &rb10, 8);
      du::dumma::du_mma_sync(acc0_frag, a_frag0, b_frag0, acc0_frag);
      du::dumma::du_mma_sync(acc1_frag, a_frag0, b_frag1, acc1_frag);
      __builtin_memcpy(b_frag0.x, &rb01, 8);
      __builtin_memcpy(b_frag1.x, &rb11, 8);
      du::dumma::du_mma_sync(acc0_frag, a_frag1, b_frag0, acc0_frag);
      du::dumma::du_mma_sync(acc1_frag, a_frag1, b_frag1, acc1_frag);
      __builtin_memcpy(b_frag0.x, &rb02, 8);
      __builtin_memcpy(b_frag1.x, &rb12, 8);
      du::dumma::du_mma_sync(acc0_frag, a_frag2, b_frag0, acc0_frag);
      du::dumma::du_mma_sync(acc1_frag, a_frag2, b_frag1, acc1_frag);
      __builtin_memcpy(b_frag0.x, &rb03, 8);
      __builtin_memcpy(b_frag1.x, &rb13, 8);
      du::dumma::du_mma_sync(acc0_frag, a_frag3, b_frag0, acc0_frag);
      du::dumma::du_mma_sync(acc1_frag, a_frag3, b_frag1, acc1_frag);
      __builtin_memcpy(b_frag0.x, &rb04, 8);
      __builtin_memcpy(b_frag1.x, &rb14, 8);
      du::dumma::du_mma_sync(acc0_frag, a_frag4, b_frag0, acc0_frag);
      du::dumma::du_mma_sync(acc1_frag, a_frag4, b_frag1, acc1_frag);
      __builtin_memcpy(b_frag0.x, &rb05, 8);
      __builtin_memcpy(b_frag1.x, &rb15, 8);
      du::dumma::du_mma_sync(acc0_frag, a_frag5, b_frag0, acc0_frag);
      du::dumma::du_mma_sync(acc1_frag, a_frag5, b_frag1, acc1_frag);
    }
  }

  // ---- Pass 1: chunks 2 and 3. Both stagings are wait-free (their B loads
  // were drained by the single prologue ladder); the only waits here are
  // the L2 A captures of the chunk 2/3 A loads issued in pass 0. ----
  {
    // Stage chunk 2 (set 2) into LDS plane 0.
    {
      int2* d0 = reinterpret_cast<int2*>(&s_b[0][0][b_row][b_wide]);
      int2* d1 = reinterpret_cast<int2*>(&s_b[0][1][b_row][b_wide]);
      d0[0] = *reinterpret_cast<const int2*>(&b0s2a);
      d0[1] = *reinterpret_cast<const int2*>(
          reinterpret_cast<const int*>(&b0s2a) + 2);
      d1[0] = *reinterpret_cast<const int2*>(&b1s2a);
      d1[1] = *reinterpret_cast<const int2*>(
          reinterpret_cast<const int*>(&b1s2a) + 2);
      int2* d0b = reinterpret_cast<int2*>(&s_b[0][0][b_row][b_wide + 64]);
      int2* d1b = reinterpret_cast<int2*>(&s_b[0][1][b_row][b_wide + 64]);
      d0b[0] = *reinterpret_cast<const int2*>(&b0s2b);
      d0b[1] = *reinterpret_cast<const int2*>(
          reinterpret_cast<const int*>(&b0s2b) + 2);
      d1b[0] = *reinterpret_cast<const int2*>(&b1s2b);
      d1b[1] = *reinterpret_cast<const int2*>(
          reinterpret_cast<const int*>(&b1s2b) + 2);
      int2* d0c = reinterpret_cast<int2*>(&s_b[0][0][b_row][b_wide + 128]);
      int2* d1c = reinterpret_cast<int2*>(&s_b[0][1][b_row][b_wide + 128]);
      d0c[0] = *reinterpret_cast<const int2*>(&b0s2c);
      d0c[1] = *reinterpret_cast<const int2*>(
          reinterpret_cast<const int*>(&b0s2c) + 2);
      d1c[0] = *reinterpret_cast<const int2*>(&b1s2c);
      d1c[1] = *reinterpret_cast<const int2*>(
          reinterpret_cast<const int*>(&b1s2c) + 2);
    }
    // Capture chunk 2's six A fragments (issued in pass 0; L2-hot).
    __builtin_memcpy(a_frag0.x, &a0s0, 8);
    __builtin_memcpy(a_frag1.x, &a1s0, 8);
    __builtin_memcpy(a_frag2.x, &a2s0, 8);
    __builtin_memcpy(a_frag3.x, &a3s0, 8);
    __builtin_memcpy(a_frag4.x, &a4s0, 8);
    __builtin_memcpy(a_frag5.x, &a5s0, 8);
    // Consume chunk 2 from LDS plane 0.
    {
      const int2 rb00 =
          *reinterpret_cast<const int2*>(&s_b[0][0][b_row][b_col]);
      const int2 rb10 =
          *reinterpret_cast<const int2*>(&s_b[0][1][b_row][b_col]);
      const int2 rb01 = *reinterpret_cast<const int2*>(
          &s_b[0][0][b_row][b_col + kDummaTileK]);
      const int2 rb11 = *reinterpret_cast<const int2*>(
          &s_b[0][1][b_row][b_col + kDummaTileK]);
      const int2 rb02 = *reinterpret_cast<const int2*>(
          &s_b[0][0][b_row][b_col + 2 * kDummaTileK]);
      const int2 rb12 = *reinterpret_cast<const int2*>(
          &s_b[0][1][b_row][b_col + 2 * kDummaTileK]);
      const int2 rb03 = *reinterpret_cast<const int2*>(
          &s_b[0][0][b_row][b_col + 3 * kDummaTileK]);
      const int2 rb13 = *reinterpret_cast<const int2*>(
          &s_b[0][1][b_row][b_col + 3 * kDummaTileK]);
      const int2 rb04 = *reinterpret_cast<const int2*>(
          &s_b[0][0][b_row][b_col + 4 * kDummaTileK]);
      const int2 rb14 = *reinterpret_cast<const int2*>(
          &s_b[0][1][b_row][b_col + 4 * kDummaTileK]);
      const int2 rb05 = *reinterpret_cast<const int2*>(
          &s_b[0][0][b_row][b_col + 5 * kDummaTileK]);
      const int2 rb15 = *reinterpret_cast<const int2*>(
          &s_b[0][1][b_row][b_col + 5 * kDummaTileK]);
      __builtin_memcpy(b_frag0.x, &rb00, 8);
      __builtin_memcpy(b_frag1.x, &rb10, 8);
      du::dumma::du_mma_sync(acc0_frag, a_frag0, b_frag0, acc0_frag);
      du::dumma::du_mma_sync(acc1_frag, a_frag0, b_frag1, acc1_frag);
      __builtin_memcpy(b_frag0.x, &rb01, 8);
      __builtin_memcpy(b_frag1.x, &rb11, 8);
      du::dumma::du_mma_sync(acc0_frag, a_frag1, b_frag0, acc0_frag);
      du::dumma::du_mma_sync(acc1_frag, a_frag1, b_frag1, acc1_frag);
      __builtin_memcpy(b_frag0.x, &rb02, 8);
      __builtin_memcpy(b_frag1.x, &rb12, 8);
      du::dumma::du_mma_sync(acc0_frag, a_frag2, b_frag0, acc0_frag);
      du::dumma::du_mma_sync(acc1_frag, a_frag2, b_frag1, acc1_frag);
      __builtin_memcpy(b_frag0.x, &rb03, 8);
      __builtin_memcpy(b_frag1.x, &rb13, 8);
      du::dumma::du_mma_sync(acc0_frag, a_frag3, b_frag0, acc0_frag);
      du::dumma::du_mma_sync(acc1_frag, a_frag3, b_frag1, acc1_frag);
      __builtin_memcpy(b_frag0.x, &rb04, 8);
      __builtin_memcpy(b_frag1.x, &rb14, 8);
      du::dumma::du_mma_sync(acc0_frag, a_frag4, b_frag0, acc0_frag);
      du::dumma::du_mma_sync(acc1_frag, a_frag4, b_frag1, acc1_frag);
      __builtin_memcpy(b_frag0.x, &rb05, 8);
      __builtin_memcpy(b_frag1.x, &rb15, 8);
      du::dumma::du_mma_sync(acc0_frag, a_frag5, b_frag0, acc0_frag);
      du::dumma::du_mma_sync(acc1_frag, a_frag5, b_frag1, acc1_frag);
    }
  }
  {
    // Stage chunk 3 (set 3) into LDS plane 1 (wait-free).
    {
      int2* d0 = reinterpret_cast<int2*>(&s_b[1][0][b_row][b_wide]);
      int2* d1 = reinterpret_cast<int2*>(&s_b[1][1][b_row][b_wide]);
      d0[0] = *reinterpret_cast<const int2*>(&b0s3a);
      d0[1] = *reinterpret_cast<const int2*>(
          reinterpret_cast<const int*>(&b0s3a) + 2);
      d1[0] = *reinterpret_cast<const int2*>(&b1s3a);
      d1[1] = *reinterpret_cast<const int2*>(
          reinterpret_cast<const int*>(&b1s3a) + 2);
      int2* d0b = reinterpret_cast<int2*>(&s_b[1][0][b_row][b_wide + 64]);
      int2* d1b = reinterpret_cast<int2*>(&s_b[1][1][b_row][b_wide + 64]);
      d0b[0] = *reinterpret_cast<const int2*>(&b0s3b);
      d0b[1] = *reinterpret_cast<const int2*>(
          reinterpret_cast<const int*>(&b0s3b) + 2);
      d1b[0] = *reinterpret_cast<const int2*>(&b1s3b);
      d1b[1] = *reinterpret_cast<const int2*>(
          reinterpret_cast<const int*>(&b1s3b) + 2);
      int2* d0c = reinterpret_cast<int2*>(&s_b[1][0][b_row][b_wide + 128]);
      int2* d1c = reinterpret_cast<int2*>(&s_b[1][1][b_row][b_wide + 128]);
      d0c[0] = *reinterpret_cast<const int2*>(&b0s3c);
      d0c[1] = *reinterpret_cast<const int2*>(
          reinterpret_cast<const int*>(&b0s3c) + 2);
      d1c[0] = *reinterpret_cast<const int2*>(&b1s3c);
      d1c[1] = *reinterpret_cast<const int2*>(
          reinterpret_cast<const int*>(&b1s3c) + 2);
    }
    // Capture chunk 3's six A fragments (issued in pass 0; L2-hot).
    __builtin_memcpy(a_frag0.x, &a0s1, 8);
    __builtin_memcpy(a_frag1.x, &a1s1, 8);
    __builtin_memcpy(a_frag2.x, &a2s1, 8);
    __builtin_memcpy(a_frag3.x, &a3s1, 8);
    __builtin_memcpy(a_frag4.x, &a4s1, 8);
    __builtin_memcpy(a_frag5.x, &a5s1, 8);
    // Consume chunk 3 from LDS plane 1.
    {
      const int2 rb00 =
          *reinterpret_cast<const int2*>(&s_b[1][0][b_row][b_col]);
      const int2 rb10 =
          *reinterpret_cast<const int2*>(&s_b[1][1][b_row][b_col]);
      const int2 rb01 = *reinterpret_cast<const int2*>(
          &s_b[1][0][b_row][b_col + kDummaTileK]);
      const int2 rb11 = *reinterpret_cast<const int2*>(
          &s_b[1][1][b_row][b_col + kDummaTileK]);
      const int2 rb02 = *reinterpret_cast<const int2*>(
          &s_b[1][0][b_row][b_col + 2 * kDummaTileK]);
      const int2 rb12 = *reinterpret_cast<const int2*>(
          &s_b[1][1][b_row][b_col + 2 * kDummaTileK]);
      const int2 rb03 = *reinterpret_cast<const int2*>(
          &s_b[1][0][b_row][b_col + 3 * kDummaTileK]);
      const int2 rb13 = *reinterpret_cast<const int2*>(
          &s_b[1][1][b_row][b_col + 3 * kDummaTileK]);
      const int2 rb04 = *reinterpret_cast<const int2*>(
          &s_b[1][0][b_row][b_col + 4 * kDummaTileK]);
      const int2 rb14 = *reinterpret_cast<const int2*>(
          &s_b[1][1][b_row][b_col + 4 * kDummaTileK]);
      const int2 rb05 = *reinterpret_cast<const int2*>(
          &s_b[1][0][b_row][b_col + 5 * kDummaTileK]);
      const int2 rb15 = *reinterpret_cast<const int2*>(
          &s_b[1][1][b_row][b_col + 5 * kDummaTileK]);
      __builtin_memcpy(b_frag0.x, &rb00, 8);
      __builtin_memcpy(b_frag1.x, &rb10, 8);
      du::dumma::du_mma_sync(acc0_frag, a_frag0, b_frag0, acc0_frag);
      du::dumma::du_mma_sync(acc1_frag, a_frag0, b_frag1, acc1_frag);
      __builtin_memcpy(b_frag0.x, &rb01, 8);
      __builtin_memcpy(b_frag1.x, &rb11, 8);
      du::dumma::du_mma_sync(acc0_frag, a_frag1, b_frag0, acc0_frag);
      du::dumma::du_mma_sync(acc1_frag, a_frag1, b_frag1, acc1_frag);
      __builtin_memcpy(b_frag0.x, &rb02, 8);
      __builtin_memcpy(b_frag1.x, &rb12, 8);
      du::dumma::du_mma_sync(acc0_frag, a_frag2, b_frag0, acc0_frag);
      du::dumma::du_mma_sync(acc1_frag, a_frag2, b_frag1, acc1_frag);
      __builtin_memcpy(b_frag0.x, &rb03, 8);
      __builtin_memcpy(b_frag1.x, &rb13, 8);
      du::dumma::du_mma_sync(acc0_frag, a_frag3, b_frag0, acc0_frag);
      du::dumma::du_mma_sync(acc1_frag, a_frag3, b_frag1, acc1_frag);
      __builtin_memcpy(b_frag0.x, &rb04, 8);
      __builtin_memcpy(b_frag1.x, &rb14, 8);
      du::dumma::du_mma_sync(acc0_frag, a_frag4, b_frag0, acc0_frag);
      du::dumma::du_mma_sync(acc1_frag, a_frag4, b_frag1, acc1_frag);
      __builtin_memcpy(b_frag0.x, &rb05, 8);
      __builtin_memcpy(b_frag1.x, &rb15, 8);
      du::dumma::du_mma_sync(acc0_frag, a_frag5, b_frag0, acc0_frag);
      du::dumma::du_mma_sync(acc1_frag, a_frag5, b_frag1, acc1_frag);
    }
  }

  // ---- Iteration 20, body 2 (chunks 4-7, slice offsets 768..1535): the
  // body-1 B/A registers are dead after the pass-1 captures, so re-issue the
  // 24 dwordx4 B loads plus the chunks 4/5 A loads in place; the chunk-4
  // staging below drains them with ONE ladder (the block's second cold-B
  // round trip; 328 blocks x 2 = 656 drains device-wide, the same total as
  // the accepted 656-block kernel). All offsets stay 16-B aligned: 4*kChunkK
  // = 768 = 16*48, max offset 7*192 + 128 + 15 = 1487 < 1536. ----
  b0s0a = *reinterpret_cast<const int4*>(b0_base + 4 * kChunkK + b_wide);
  b0s0b = *reinterpret_cast<const int4*>(b0_base + 4 * kChunkK + b_wide + 64);
  b0s0c = *reinterpret_cast<const int4*>(b0_base + 4 * kChunkK + b_wide + 128);
  b1s0a = *reinterpret_cast<const int4*>(b1_base + 4 * kChunkK + b_wide);
  b1s0b = *reinterpret_cast<const int4*>(b1_base + 4 * kChunkK + b_wide + 64);
  b1s0c = *reinterpret_cast<const int4*>(b1_base + 4 * kChunkK + b_wide + 128);
  b0s1a = *reinterpret_cast<const int4*>(b0_base + 5 * kChunkK + b_wide);
  b0s1b = *reinterpret_cast<const int4*>(b0_base + 5 * kChunkK + b_wide + 64);
  b0s1c = *reinterpret_cast<const int4*>(b0_base + 5 * kChunkK + b_wide + 128);
  b1s1a = *reinterpret_cast<const int4*>(b1_base + 5 * kChunkK + b_wide);
  b1s1b = *reinterpret_cast<const int4*>(b1_base + 5 * kChunkK + b_wide + 64);
  b1s1c = *reinterpret_cast<const int4*>(b1_base + 5 * kChunkK + b_wide + 128);
  b0s2a = *reinterpret_cast<const int4*>(b0_base + 6 * kChunkK + b_wide);
  b0s2b = *reinterpret_cast<const int4*>(b0_base + 6 * kChunkK + b_wide + 64);
  b0s2c = *reinterpret_cast<const int4*>(b0_base + 6 * kChunkK + b_wide + 128);
  b1s2a = *reinterpret_cast<const int4*>(b1_base + 6 * kChunkK + b_wide);
  b1s2b = *reinterpret_cast<const int4*>(b1_base + 6 * kChunkK + b_wide + 64);
  b1s2c = *reinterpret_cast<const int4*>(b1_base + 6 * kChunkK + b_wide + 128);
  b0s3a = *reinterpret_cast<const int4*>(b0_base + 7 * kChunkK + b_wide);
  b0s3b = *reinterpret_cast<const int4*>(b0_base + 7 * kChunkK + b_wide + 64);
  b0s3c = *reinterpret_cast<const int4*>(b0_base + 7 * kChunkK + b_wide + 128);
  b1s3a = *reinterpret_cast<const int4*>(b1_base + 7 * kChunkK + b_wide);
  b1s3b = *reinterpret_cast<const int4*>(b1_base + 7 * kChunkK + b_wide + 64);
  b1s3c = *reinterpret_cast<const int4*>(b1_base + 7 * kChunkK + b_wide + 128);
  a0s0 = *reinterpret_cast<const int2*>(a_base + 4 * kChunkK);
  a1s0 = *reinterpret_cast<const int2*>(a_base + 4 * kChunkK + kDummaTileK);
  a2s0 = *reinterpret_cast<const int2*>(a_base + 4 * kChunkK + 2 * kDummaTileK);
  a3s0 = *reinterpret_cast<const int2*>(a_base + 4 * kChunkK + 3 * kDummaTileK);
  a4s0 = *reinterpret_cast<const int2*>(a_base + 4 * kChunkK + 4 * kDummaTileK);
  a5s0 = *reinterpret_cast<const int2*>(a_base + 4 * kChunkK + 5 * kDummaTileK);
  a0s1 = *reinterpret_cast<const int2*>(a_base + 5 * kChunkK);
  a1s1 = *reinterpret_cast<const int2*>(a_base + 5 * kChunkK + kDummaTileK);
  a2s1 = *reinterpret_cast<const int2*>(a_base + 5 * kChunkK + 2 * kDummaTileK);
  a3s1 = *reinterpret_cast<const int2*>(a_base + 5 * kChunkK + 3 * kDummaTileK);
  a4s1 = *reinterpret_cast<const int2*>(a_base + 5 * kChunkK + 4 * kDummaTileK);
  a5s1 = *reinterpret_cast<const int2*>(a_base + 5 * kChunkK + 5 * kDummaTileK);

  // ---- Pass 2: chunks 4 and 5. The chunk-4 staging is the block's second
  // cold-B drain; chunk 5 is wait-free. A rotation identical to body 1. ----
  {
    // Stage chunk 4 (set 0) into LDS plane 0.
    {
      int2* d0 = reinterpret_cast<int2*>(&s_b[0][0][b_row][b_wide]);
      int2* d1 = reinterpret_cast<int2*>(&s_b[0][1][b_row][b_wide]);
      d0[0] = *reinterpret_cast<const int2*>(&b0s0a);
      d0[1] = *reinterpret_cast<const int2*>(
          reinterpret_cast<const int*>(&b0s0a) + 2);
      d1[0] = *reinterpret_cast<const int2*>(&b1s0a);
      d1[1] = *reinterpret_cast<const int2*>(
          reinterpret_cast<const int*>(&b1s0a) + 2);
      int2* d0b = reinterpret_cast<int2*>(&s_b[0][0][b_row][b_wide + 64]);
      int2* d1b = reinterpret_cast<int2*>(&s_b[0][1][b_row][b_wide + 64]);
      d0b[0] = *reinterpret_cast<const int2*>(&b0s0b);
      d0b[1] = *reinterpret_cast<const int2*>(
          reinterpret_cast<const int*>(&b0s0b) + 2);
      d1b[0] = *reinterpret_cast<const int2*>(&b1s0b);
      d1b[1] = *reinterpret_cast<const int2*>(
          reinterpret_cast<const int*>(&b1s0b) + 2);
      int2* d0c = reinterpret_cast<int2*>(&s_b[0][0][b_row][b_wide + 128]);
      int2* d1c = reinterpret_cast<int2*>(&s_b[0][1][b_row][b_wide + 128]);
      d0c[0] = *reinterpret_cast<const int2*>(&b0s0c);
      d0c[1] = *reinterpret_cast<const int2*>(
          reinterpret_cast<const int*>(&b0s0c) + 2);
      d1c[0] = *reinterpret_cast<const int2*>(&b1s0c);
      d1c[1] = *reinterpret_cast<const int2*>(
          reinterpret_cast<const int*>(&b1s0c) + 2);
    }
    // Capture chunk 4's six A fragments BEFORE set 0's A registers are
    // overwritten below by the chunk-6 A issue.
    __builtin_memcpy(a_frag0.x, &a0s0, 8);
    __builtin_memcpy(a_frag1.x, &a1s0, 8);
    __builtin_memcpy(a_frag2.x, &a2s0, 8);
    __builtin_memcpy(a_frag3.x, &a3s0, 8);
    __builtin_memcpy(a_frag4.x, &a4s0, 8);
    __builtin_memcpy(a_frag5.x, &a5s0, 8);
    // Issue chunk 6's A loads (L2-hot) into the now-dead set-0 A registers;
    // consumed by the pass-3 capture one full pass later.
    a0s0 = *reinterpret_cast<const int2*>(a_base + 6 * kChunkK);
    a1s0 = *reinterpret_cast<const int2*>(
        a_base + 6 * kChunkK + kDummaTileK);
    a2s0 = *reinterpret_cast<const int2*>(
        a_base + 6 * kChunkK + 2 * kDummaTileK);
    a3s0 = *reinterpret_cast<const int2*>(
        a_base + 6 * kChunkK + 3 * kDummaTileK);
    a4s0 = *reinterpret_cast<const int2*>(
        a_base + 6 * kChunkK + 4 * kDummaTileK);
    a5s0 = *reinterpret_cast<const int2*>(
        a_base + 6 * kChunkK + 5 * kDummaTileK);
    // Consume chunk 4 from LDS plane 0: six k32 steps covering slice k
    // offsets [768, 800), [800, 832), [832, 864), [864, 896), [896, 928),
    // [928, 960).
    {
      const int2 rb00 =
          *reinterpret_cast<const int2*>(&s_b[0][0][b_row][b_col]);
      const int2 rb10 =
          *reinterpret_cast<const int2*>(&s_b[0][1][b_row][b_col]);
      const int2 rb01 = *reinterpret_cast<const int2*>(
          &s_b[0][0][b_row][b_col + kDummaTileK]);
      const int2 rb11 = *reinterpret_cast<const int2*>(
          &s_b[0][1][b_row][b_col + kDummaTileK]);
      const int2 rb02 = *reinterpret_cast<const int2*>(
          &s_b[0][0][b_row][b_col + 2 * kDummaTileK]);
      const int2 rb12 = *reinterpret_cast<const int2*>(
          &s_b[0][1][b_row][b_col + 2 * kDummaTileK]);
      const int2 rb03 = *reinterpret_cast<const int2*>(
          &s_b[0][0][b_row][b_col + 3 * kDummaTileK]);
      const int2 rb13 = *reinterpret_cast<const int2*>(
          &s_b[0][1][b_row][b_col + 3 * kDummaTileK]);
      const int2 rb04 = *reinterpret_cast<const int2*>(
          &s_b[0][0][b_row][b_col + 4 * kDummaTileK]);
      const int2 rb14 = *reinterpret_cast<const int2*>(
          &s_b[0][1][b_row][b_col + 4 * kDummaTileK]);
      const int2 rb05 = *reinterpret_cast<const int2*>(
          &s_b[0][0][b_row][b_col + 5 * kDummaTileK]);
      const int2 rb15 = *reinterpret_cast<const int2*>(
          &s_b[0][1][b_row][b_col + 5 * kDummaTileK]);
      __builtin_memcpy(b_frag0.x, &rb00, 8);
      __builtin_memcpy(b_frag1.x, &rb10, 8);
      du::dumma::du_mma_sync(acc0_frag, a_frag0, b_frag0, acc0_frag);
      du::dumma::du_mma_sync(acc1_frag, a_frag0, b_frag1, acc1_frag);
      __builtin_memcpy(b_frag0.x, &rb01, 8);
      __builtin_memcpy(b_frag1.x, &rb11, 8);
      du::dumma::du_mma_sync(acc0_frag, a_frag1, b_frag0, acc0_frag);
      du::dumma::du_mma_sync(acc1_frag, a_frag1, b_frag1, acc1_frag);
      __builtin_memcpy(b_frag0.x, &rb02, 8);
      __builtin_memcpy(b_frag1.x, &rb12, 8);
      du::dumma::du_mma_sync(acc0_frag, a_frag2, b_frag0, acc0_frag);
      du::dumma::du_mma_sync(acc1_frag, a_frag2, b_frag1, acc1_frag);
      __builtin_memcpy(b_frag0.x, &rb03, 8);
      __builtin_memcpy(b_frag1.x, &rb13, 8);
      du::dumma::du_mma_sync(acc0_frag, a_frag3, b_frag0, acc0_frag);
      du::dumma::du_mma_sync(acc1_frag, a_frag3, b_frag1, acc1_frag);
      __builtin_memcpy(b_frag0.x, &rb04, 8);
      __builtin_memcpy(b_frag1.x, &rb14, 8);
      du::dumma::du_mma_sync(acc0_frag, a_frag4, b_frag0, acc0_frag);
      du::dumma::du_mma_sync(acc1_frag, a_frag4, b_frag1, acc1_frag);
      __builtin_memcpy(b_frag0.x, &rb05, 8);
      __builtin_memcpy(b_frag1.x, &rb15, 8);
      du::dumma::du_mma_sync(acc0_frag, a_frag5, b_frag0, acc0_frag);
      du::dumma::du_mma_sync(acc1_frag, a_frag5, b_frag1, acc1_frag);
    }
  }
  {
    // Stage chunk 5 (set 1) into LDS plane 1 (wait-free: drained with
    // chunk 4 by the body-2 ladder).
    {
      int2* d0 = reinterpret_cast<int2*>(&s_b[1][0][b_row][b_wide]);
      int2* d1 = reinterpret_cast<int2*>(&s_b[1][1][b_row][b_wide]);
      d0[0] = *reinterpret_cast<const int2*>(&b0s1a);
      d0[1] = *reinterpret_cast<const int2*>(
          reinterpret_cast<const int*>(&b0s1a) + 2);
      d1[0] = *reinterpret_cast<const int2*>(&b1s1a);
      d1[1] = *reinterpret_cast<const int2*>(
          reinterpret_cast<const int*>(&b1s1a) + 2);
      int2* d0b = reinterpret_cast<int2*>(&s_b[1][0][b_row][b_wide + 64]);
      int2* d1b = reinterpret_cast<int2*>(&s_b[1][1][b_row][b_wide + 64]);
      d0b[0] = *reinterpret_cast<const int2*>(&b0s1b);
      d0b[1] = *reinterpret_cast<const int2*>(
          reinterpret_cast<const int*>(&b0s1b) + 2);
      d1b[0] = *reinterpret_cast<const int2*>(&b1s1b);
      d1b[1] = *reinterpret_cast<const int2*>(
          reinterpret_cast<const int*>(&b1s1b) + 2);
      int2* d0c = reinterpret_cast<int2*>(&s_b[1][0][b_row][b_wide + 128]);
      int2* d1c = reinterpret_cast<int2*>(&s_b[1][1][b_row][b_wide + 128]);
      d0c[0] = *reinterpret_cast<const int2*>(&b0s1c);
      d0c[1] = *reinterpret_cast<const int2*>(
          reinterpret_cast<const int*>(&b0s1c) + 2);
      d1c[0] = *reinterpret_cast<const int2*>(&b1s1c);
      d1c[1] = *reinterpret_cast<const int2*>(
          reinterpret_cast<const int*>(&b1s1c) + 2);
    }
    // Capture chunk 5's six A fragments before the chunk-7 A issue below.
    __builtin_memcpy(a_frag0.x, &a0s1, 8);
    __builtin_memcpy(a_frag1.x, &a1s1, 8);
    __builtin_memcpy(a_frag2.x, &a2s1, 8);
    __builtin_memcpy(a_frag3.x, &a3s1, 8);
    __builtin_memcpy(a_frag4.x, &a4s1, 8);
    __builtin_memcpy(a_frag5.x, &a5s1, 8);
    // Issue chunk 7's A loads (L2-hot) into the now-dead set-1 A registers.
    a0s1 = *reinterpret_cast<const int2*>(a_base + 7 * kChunkK);
    a1s1 = *reinterpret_cast<const int2*>(
        a_base + 7 * kChunkK + kDummaTileK);
    a2s1 = *reinterpret_cast<const int2*>(
        a_base + 7 * kChunkK + 2 * kDummaTileK);
    a3s1 = *reinterpret_cast<const int2*>(
        a_base + 7 * kChunkK + 3 * kDummaTileK);
    a4s1 = *reinterpret_cast<const int2*>(
        a_base + 7 * kChunkK + 4 * kDummaTileK);
    a5s1 = *reinterpret_cast<const int2*>(
        a_base + 7 * kChunkK + 5 * kDummaTileK);
    // Consume chunk 5 from LDS plane 1: slice k offsets [960, 1152).
    {
      const int2 rb00 =
          *reinterpret_cast<const int2*>(&s_b[1][0][b_row][b_col]);
      const int2 rb10 =
          *reinterpret_cast<const int2*>(&s_b[1][1][b_row][b_col]);
      const int2 rb01 = *reinterpret_cast<const int2*>(
          &s_b[1][0][b_row][b_col + kDummaTileK]);
      const int2 rb11 = *reinterpret_cast<const int2*>(
          &s_b[1][1][b_row][b_col + kDummaTileK]);
      const int2 rb02 = *reinterpret_cast<const int2*>(
          &s_b[1][0][b_row][b_col + 2 * kDummaTileK]);
      const int2 rb12 = *reinterpret_cast<const int2*>(
          &s_b[1][1][b_row][b_col + 2 * kDummaTileK]);
      const int2 rb03 = *reinterpret_cast<const int2*>(
          &s_b[1][0][b_row][b_col + 3 * kDummaTileK]);
      const int2 rb13 = *reinterpret_cast<const int2*>(
          &s_b[1][1][b_row][b_col + 3 * kDummaTileK]);
      const int2 rb04 = *reinterpret_cast<const int2*>(
          &s_b[1][0][b_row][b_col + 4 * kDummaTileK]);
      const int2 rb14 = *reinterpret_cast<const int2*>(
          &s_b[1][1][b_row][b_col + 4 * kDummaTileK]);
      const int2 rb05 = *reinterpret_cast<const int2*>(
          &s_b[1][0][b_row][b_col + 5 * kDummaTileK]);
      const int2 rb15 = *reinterpret_cast<const int2*>(
          &s_b[1][1][b_row][b_col + 5 * kDummaTileK]);
      __builtin_memcpy(b_frag0.x, &rb00, 8);
      __builtin_memcpy(b_frag1.x, &rb10, 8);
      du::dumma::du_mma_sync(acc0_frag, a_frag0, b_frag0, acc0_frag);
      du::dumma::du_mma_sync(acc1_frag, a_frag0, b_frag1, acc1_frag);
      __builtin_memcpy(b_frag0.x, &rb01, 8);
      __builtin_memcpy(b_frag1.x, &rb11, 8);
      du::dumma::du_mma_sync(acc0_frag, a_frag1, b_frag0, acc0_frag);
      du::dumma::du_mma_sync(acc1_frag, a_frag1, b_frag1, acc1_frag);
      __builtin_memcpy(b_frag0.x, &rb02, 8);
      __builtin_memcpy(b_frag1.x, &rb12, 8);
      du::dumma::du_mma_sync(acc0_frag, a_frag2, b_frag0, acc0_frag);
      du::dumma::du_mma_sync(acc1_frag, a_frag2, b_frag1, acc1_frag);
      __builtin_memcpy(b_frag0.x, &rb03, 8);
      __builtin_memcpy(b_frag1.x, &rb13, 8);
      du::dumma::du_mma_sync(acc0_frag, a_frag3, b_frag0, acc0_frag);
      du::dumma::du_mma_sync(acc1_frag, a_frag3, b_frag1, acc1_frag);
      __builtin_memcpy(b_frag0.x, &rb04, 8);
      __builtin_memcpy(b_frag1.x, &rb14, 8);
      du::dumma::du_mma_sync(acc0_frag, a_frag4, b_frag0, acc0_frag);
      du::dumma::du_mma_sync(acc1_frag, a_frag4, b_frag1, acc1_frag);
      __builtin_memcpy(b_frag0.x, &rb05, 8);
      __builtin_memcpy(b_frag1.x, &rb15, 8);
      du::dumma::du_mma_sync(acc0_frag, a_frag5, b_frag0, acc0_frag);
      du::dumma::du_mma_sync(acc1_frag, a_frag5, b_frag1, acc1_frag);
    }
  }

  // ---- Pass 3: chunks 6 and 7. Both stagings are wait-free (their B loads
  // were drained by the body-2 ladder at the chunk-4 staging); the only
  // waits here are the L2 A captures of the chunk 6/7 A loads issued in
  // pass 2. ----
  {
    // Stage chunk 6 (set 2) into LDS plane 0.
    {
      int2* d0 = reinterpret_cast<int2*>(&s_b[0][0][b_row][b_wide]);
      int2* d1 = reinterpret_cast<int2*>(&s_b[0][1][b_row][b_wide]);
      d0[0] = *reinterpret_cast<const int2*>(&b0s2a);
      d0[1] = *reinterpret_cast<const int2*>(
          reinterpret_cast<const int*>(&b0s2a) + 2);
      d1[0] = *reinterpret_cast<const int2*>(&b1s2a);
      d1[1] = *reinterpret_cast<const int2*>(
          reinterpret_cast<const int*>(&b1s2a) + 2);
      int2* d0b = reinterpret_cast<int2*>(&s_b[0][0][b_row][b_wide + 64]);
      int2* d1b = reinterpret_cast<int2*>(&s_b[0][1][b_row][b_wide + 64]);
      d0b[0] = *reinterpret_cast<const int2*>(&b0s2b);
      d0b[1] = *reinterpret_cast<const int2*>(
          reinterpret_cast<const int*>(&b0s2b) + 2);
      d1b[0] = *reinterpret_cast<const int2*>(&b1s2b);
      d1b[1] = *reinterpret_cast<const int2*>(
          reinterpret_cast<const int*>(&b1s2b) + 2);
      int2* d0c = reinterpret_cast<int2*>(&s_b[0][0][b_row][b_wide + 128]);
      int2* d1c = reinterpret_cast<int2*>(&s_b[0][1][b_row][b_wide + 128]);
      d0c[0] = *reinterpret_cast<const int2*>(&b0s2c);
      d0c[1] = *reinterpret_cast<const int2*>(
          reinterpret_cast<const int*>(&b0s2c) + 2);
      d1c[0] = *reinterpret_cast<const int2*>(&b1s2c);
      d1c[1] = *reinterpret_cast<const int2*>(
          reinterpret_cast<const int*>(&b1s2c) + 2);
    }
    // Capture chunk 6's six A fragments (issued in pass 2; L2-hot).
    __builtin_memcpy(a_frag0.x, &a0s0, 8);
    __builtin_memcpy(a_frag1.x, &a1s0, 8);
    __builtin_memcpy(a_frag2.x, &a2s0, 8);
    __builtin_memcpy(a_frag3.x, &a3s0, 8);
    __builtin_memcpy(a_frag4.x, &a4s0, 8);
    __builtin_memcpy(a_frag5.x, &a5s0, 8);
    // Consume chunk 6 from LDS plane 0: slice k offsets [1152, 1344).
    {
      const int2 rb00 =
          *reinterpret_cast<const int2*>(&s_b[0][0][b_row][b_col]);
      const int2 rb10 =
          *reinterpret_cast<const int2*>(&s_b[0][1][b_row][b_col]);
      const int2 rb01 = *reinterpret_cast<const int2*>(
          &s_b[0][0][b_row][b_col + kDummaTileK]);
      const int2 rb11 = *reinterpret_cast<const int2*>(
          &s_b[0][1][b_row][b_col + kDummaTileK]);
      const int2 rb02 = *reinterpret_cast<const int2*>(
          &s_b[0][0][b_row][b_col + 2 * kDummaTileK]);
      const int2 rb12 = *reinterpret_cast<const int2*>(
          &s_b[0][1][b_row][b_col + 2 * kDummaTileK]);
      const int2 rb03 = *reinterpret_cast<const int2*>(
          &s_b[0][0][b_row][b_col + 3 * kDummaTileK]);
      const int2 rb13 = *reinterpret_cast<const int2*>(
          &s_b[0][1][b_row][b_col + 3 * kDummaTileK]);
      const int2 rb04 = *reinterpret_cast<const int2*>(
          &s_b[0][0][b_row][b_col + 4 * kDummaTileK]);
      const int2 rb14 = *reinterpret_cast<const int2*>(
          &s_b[0][1][b_row][b_col + 4 * kDummaTileK]);
      const int2 rb05 = *reinterpret_cast<const int2*>(
          &s_b[0][0][b_row][b_col + 5 * kDummaTileK]);
      const int2 rb15 = *reinterpret_cast<const int2*>(
          &s_b[0][1][b_row][b_col + 5 * kDummaTileK]);
      __builtin_memcpy(b_frag0.x, &rb00, 8);
      __builtin_memcpy(b_frag1.x, &rb10, 8);
      du::dumma::du_mma_sync(acc0_frag, a_frag0, b_frag0, acc0_frag);
      du::dumma::du_mma_sync(acc1_frag, a_frag0, b_frag1, acc1_frag);
      __builtin_memcpy(b_frag0.x, &rb01, 8);
      __builtin_memcpy(b_frag1.x, &rb11, 8);
      du::dumma::du_mma_sync(acc0_frag, a_frag1, b_frag0, acc0_frag);
      du::dumma::du_mma_sync(acc1_frag, a_frag1, b_frag1, acc1_frag);
      __builtin_memcpy(b_frag0.x, &rb02, 8);
      __builtin_memcpy(b_frag1.x, &rb12, 8);
      du::dumma::du_mma_sync(acc0_frag, a_frag2, b_frag0, acc0_frag);
      du::dumma::du_mma_sync(acc1_frag, a_frag2, b_frag1, acc1_frag);
      __builtin_memcpy(b_frag0.x, &rb03, 8);
      __builtin_memcpy(b_frag1.x, &rb13, 8);
      du::dumma::du_mma_sync(acc0_frag, a_frag3, b_frag0, acc0_frag);
      du::dumma::du_mma_sync(acc1_frag, a_frag3, b_frag1, acc1_frag);
      __builtin_memcpy(b_frag0.x, &rb04, 8);
      __builtin_memcpy(b_frag1.x, &rb14, 8);
      du::dumma::du_mma_sync(acc0_frag, a_frag4, b_frag0, acc0_frag);
      du::dumma::du_mma_sync(acc1_frag, a_frag4, b_frag1, acc1_frag);
      __builtin_memcpy(b_frag0.x, &rb05, 8);
      __builtin_memcpy(b_frag1.x, &rb15, 8);
      du::dumma::du_mma_sync(acc0_frag, a_frag5, b_frag0, acc0_frag);
      du::dumma::du_mma_sync(acc1_frag, a_frag5, b_frag1, acc1_frag);
    }
  }
  {
    // Stage chunk 7 (set 3) into LDS plane 1 (wait-free).
    {
      int2* d0 = reinterpret_cast<int2*>(&s_b[1][0][b_row][b_wide]);
      int2* d1 = reinterpret_cast<int2*>(&s_b[1][1][b_row][b_wide]);
      d0[0] = *reinterpret_cast<const int2*>(&b0s3a);
      d0[1] = *reinterpret_cast<const int2*>(
          reinterpret_cast<const int*>(&b0s3a) + 2);
      d1[0] = *reinterpret_cast<const int2*>(&b1s3a);
      d1[1] = *reinterpret_cast<const int2*>(
          reinterpret_cast<const int*>(&b1s3a) + 2);
      int2* d0b = reinterpret_cast<int2*>(&s_b[1][0][b_row][b_wide + 64]);
      int2* d1b = reinterpret_cast<int2*>(&s_b[1][1][b_row][b_wide + 64]);
      d0b[0] = *reinterpret_cast<const int2*>(&b0s3b);
      d0b[1] = *reinterpret_cast<const int2*>(
          reinterpret_cast<const int*>(&b0s3b) + 2);
      d1b[0] = *reinterpret_cast<const int2*>(&b1s3b);
      d1b[1] = *reinterpret_cast<const int2*>(
          reinterpret_cast<const int*>(&b1s3b) + 2);
      int2* d0c = reinterpret_cast<int2*>(&s_b[1][0][b_row][b_wide + 128]);
      int2* d1c = reinterpret_cast<int2*>(&s_b[1][1][b_row][b_wide + 128]);
      d0c[0] = *reinterpret_cast<const int2*>(&b0s3c);
      d0c[1] = *reinterpret_cast<const int2*>(
          reinterpret_cast<const int*>(&b0s3c) + 2);
      d1c[0] = *reinterpret_cast<const int2*>(&b1s3c);
      d1c[1] = *reinterpret_cast<const int2*>(
          reinterpret_cast<const int*>(&b1s3c) + 2);
    }
    // Capture chunk 7's six A fragments (issued in pass 2; L2-hot).
    __builtin_memcpy(a_frag0.x, &a0s1, 8);
    __builtin_memcpy(a_frag1.x, &a1s1, 8);
    __builtin_memcpy(a_frag2.x, &a2s1, 8);
    __builtin_memcpy(a_frag3.x, &a3s1, 8);
    __builtin_memcpy(a_frag4.x, &a4s1, 8);
    __builtin_memcpy(a_frag5.x, &a5s1, 8);
    // Consume chunk 7 from LDS plane 1: slice k offsets [1344, 1536).
    {
      const int2 rb00 =
          *reinterpret_cast<const int2*>(&s_b[1][0][b_row][b_col]);
      const int2 rb10 =
          *reinterpret_cast<const int2*>(&s_b[1][1][b_row][b_col]);
      const int2 rb01 = *reinterpret_cast<const int2*>(
          &s_b[1][0][b_row][b_col + kDummaTileK]);
      const int2 rb11 = *reinterpret_cast<const int2*>(
          &s_b[1][1][b_row][b_col + kDummaTileK]);
      const int2 rb02 = *reinterpret_cast<const int2*>(
          &s_b[1][0][b_row][b_col + 2 * kDummaTileK]);
      const int2 rb12 = *reinterpret_cast<const int2*>(
          &s_b[1][1][b_row][b_col + 2 * kDummaTileK]);
      const int2 rb03 = *reinterpret_cast<const int2*>(
          &s_b[1][0][b_row][b_col + 3 * kDummaTileK]);
      const int2 rb13 = *reinterpret_cast<const int2*>(
          &s_b[1][1][b_row][b_col + 3 * kDummaTileK]);
      const int2 rb04 = *reinterpret_cast<const int2*>(
          &s_b[1][0][b_row][b_col + 4 * kDummaTileK]);
      const int2 rb14 = *reinterpret_cast<const int2*>(
          &s_b[1][1][b_row][b_col + 4 * kDummaTileK]);
      const int2 rb05 = *reinterpret_cast<const int2*>(
          &s_b[1][0][b_row][b_col + 5 * kDummaTileK]);
      const int2 rb15 = *reinterpret_cast<const int2*>(
          &s_b[1][1][b_row][b_col + 5 * kDummaTileK]);
      __builtin_memcpy(b_frag0.x, &rb00, 8);
      __builtin_memcpy(b_frag1.x, &rb10, 8);
      du::dumma::du_mma_sync(acc0_frag, a_frag0, b_frag0, acc0_frag);
      du::dumma::du_mma_sync(acc1_frag, a_frag0, b_frag1, acc1_frag);
      __builtin_memcpy(b_frag0.x, &rb01, 8);
      __builtin_memcpy(b_frag1.x, &rb11, 8);
      du::dumma::du_mma_sync(acc0_frag, a_frag1, b_frag0, acc0_frag);
      du::dumma::du_mma_sync(acc1_frag, a_frag1, b_frag1, acc1_frag);
      __builtin_memcpy(b_frag0.x, &rb02, 8);
      __builtin_memcpy(b_frag1.x, &rb12, 8);
      du::dumma::du_mma_sync(acc0_frag, a_frag2, b_frag0, acc0_frag);
      du::dumma::du_mma_sync(acc1_frag, a_frag2, b_frag1, acc1_frag);
      __builtin_memcpy(b_frag0.x, &rb03, 8);
      __builtin_memcpy(b_frag1.x, &rb13, 8);
      du::dumma::du_mma_sync(acc0_frag, a_frag3, b_frag0, acc0_frag);
      du::dumma::du_mma_sync(acc1_frag, a_frag3, b_frag1, acc1_frag);
      __builtin_memcpy(b_frag0.x, &rb04, 8);
      __builtin_memcpy(b_frag1.x, &rb14, 8);
      du::dumma::du_mma_sync(acc0_frag, a_frag4, b_frag0, acc0_frag);
      du::dumma::du_mma_sync(acc1_frag, a_frag4, b_frag1, acc1_frag);
      __builtin_memcpy(b_frag0.x, &rb05, 8);
      __builtin_memcpy(b_frag1.x, &rb15, 8);
      du::dumma::du_mma_sync(acc0_frag, a_frag5, b_frag0, acc0_frag);
      du::dumma::du_mma_sync(acc1_frag, a_frag5, b_frag1, acc1_frag);
    }
  }

  // Publish both 16x16 int32 partial planes (2 x 256 int32 = 2 KiB/block).
  // Iteration 18 (workspace plane order [tile][split]): tile t's split-s
  // plane lives at (t*SPLIT_K + s)*256 int32, so the four split planes of
  // an output tile form ONE contiguous 4-KB region that the combine kernel
  // reads as a sequential stream (164 streams instead of 1,312 scattered
  // 1-KB streams). The stored bytes are identical; only the base index
  // changes, and each block still writes two line-complete 1-KB planes.
  du::dumma::du_store_matrix_sync(
      partials + (tile0 * SPLIT_K + split) * (kDummaTileM * kDummaTileN),
      acc0_frag, static_cast<unsigned>(kDummaTileN),
      du::dumma::mem_row_major);
  du::dumma::du_store_matrix_sync(
      partials + ((tile0 + 1) * SPLIT_K + split) * (kDummaTileM * kDummaTileN),
      acc1_frag, static_cast<unsigned>(kDummaTileN),
      du::dumma::mem_row_major);
}

template <int SPLIT_K>
__global__ __launch_bounds__(kDummaBlockThreads) void
w8a8_dumma_m16n16k32_splitk_combine_kernel(
    const int* __restrict__ partials,        // [N/16][SPLIT_K][256] int32
                                              // (tile-major, iteration 18)
    const float* __restrict__ x_scale,       // [16]
    const float* __restrict__ weight_scale,  // [N]
    hip_bfloat16* __restrict__ out,          // [16, N]
    int n) {
  const int tile = static_cast<int>(blockIdx.x);
  const int n0 = tile * kDummaTileN;
  const int lane = static_cast<int>(threadIdx.x);
  // Iteration 18: tile-major plane order -- plane (tile, split) lives at
  // (tile*SPLIT_K + split)*256, so this tile's four split planes are one
  // contiguous 4-KB region (164 sequential 4-KB streams instead of 1,312
  // scattered 1-KB streams at 164-KB strides).
  const int* p =
      partials + tile * SPLIT_K * (kDummaTileM * kDummaTileN) + lane * 4;

  // Iteration 13 (combine latency consolidation): the exact iteration-12
  // code object's combine kernel issues the 8 plane loads up front (one
  // round trip) but then loads the four weight_scale elements with four
  // SEPARATE 4-B loads, each waited to completion (s_waitcnt vmcnt(0))
  // before its multiply and its own 2-B bf16 store -- four serialized
  // round trips plus four store instructions per lane (weight_scale is
  // only 10.5 KiB, but the 16.1-MB B stream thrashes L2 ahead of the
  // combine, whose l2 hit rate is 31.4%, so most of those round trips are
  // DRAM-cold). This round fetches the four contiguous scales (col_base is
  // a multiple of 4 -> 16-B aligned) with ONE aligned dwordx4 load issued
  // before the plane reads below, and packs the four 2-B bf16 outputs into
  // ONE 8-B store per lane (byte offset (row*n + col_base)*2 is a multiple
  // of 8: n is a multiple of 8 and col_base is a multiple of 4). The
  // per-element arithmetic is byte-for-byte unchanged -- int32 ascending
  // split sum, then float(acc)*xs*ws (fp32 left-to-right), then
  // round-to-nearest-even bf16 -- only the load grouping and the store
  // width change, so the combine output stays bit-identical (0 mismatches)
  // and Graph replay with changed contents must pass.
  const int e0 = lane * 4;
  const int row = e0 >> 4;     // 0..15
  const int col0 = e0 & 15;    // 0,4,8,12
  const int col_base = n0 + col0;
  const float xs = x_scale[row];
  const float4 ws = *reinterpret_cast<const float4*>(weight_scale + col_base);
  // Lane l owns row-major elements e = 4*l + i (i = 0..3). Sum the SPLIT_K
  // planes in ascending split order: exact int32 accumulation, bit-identical
  // to the k-ascending full-K reference dot.
  int acc0 = 0, acc1 = 0, acc2 = 0, acc3 = 0;
#pragma unroll
  for (int s = 0; s < SPLIT_K; ++s) {
    const int* ps = p + s * (kDummaTileM * kDummaTileN);
    acc0 += ps[0];
    acc1 += ps[1];
    acc2 += ps[2];
    acc3 += ps[3];
  }

  // DTK note: __float2bfloat16 returns __hip_bfloat16 (amd_hip_bf16.h), which
  // is a distinct type from hip_bfloat16 (amd_hip_bfloat16.h, explicit float
  // ctor only) -- keep these locals as __hip_bfloat16 so the RNE bits are
  // stored directly (same bits the iter-12 out[i] = __float2bfloat16(...)
  // assignment path wrote) with the single 8-B packed store.
  const __hip_bfloat16 out0 =
      __float2bfloat16(static_cast<float>(acc0) * xs * ws.x);
  const __hip_bfloat16 out1 =
      __float2bfloat16(static_cast<float>(acc1) * xs * ws.y);
  const __hip_bfloat16 out2 =
      __float2bfloat16(static_cast<float>(acc2) * xs * ws.z);
  const __hip_bfloat16 out3 =
      __float2bfloat16(static_cast<float>(acc3) * xs * ws.w);
  const __hip_bfloat16 packed[4] = {out0, out1, out2, out3};
  __builtin_memcpy(&out[row * n + col_base], packed, sizeof(packed));
}

// Shared scalar launch helper: works for any (m, n, k) satisfying the API
// contract, so it doubles as the generic fallback.
void launch_scalar_gemm(const int8_t* a,
                        const int8_t* b,
                        const float* x_scale,
                        const float* weight_scale,
                        void* out,
                        int m,
                        int n,
                        int k,
                        hipStream_t stream) {
  const int64_t total = static_cast<int64_t>(m) * n;
  const int64_t blocks =
      (total + kScalarBlockThreads - 1) / kScalarBlockThreads;
  hipLaunchKernelGGL(w8a8_scalar_gemm_kernel,
                     dim3(static_cast<unsigned>(blocks)),
                     dim3(kScalarBlockThreads), 0, stream, a, b, x_scale,
                     weight_scale, reinterpret_cast<hip_bfloat16*>(out), m,
                     n, k);
}

// Identity device-to-device copy; generic pack_weight fallback.
template <typename T>
__global__ __launch_bounds__(kCopyBlockThreads) void
w8a8_identity_copy_kernel(
    const T* __restrict__ src,
    T* __restrict__ dst,
    int count) {
  const int i = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < count) {
    dst[i] = src[i];
  }
}

// Exact-shape n-major pack (iteration 5): for (K, N) == (6144, 2624) the
// weight is transposed once, outside the timed region and out of Graph
// capture, into packed[n * K + k] = raw[k * N + n] so each lane's 8-byte
// col_major B fragment is contiguous and every m16n16k32 B fragment is a
// single 8-B vector load in the partial kernel. Same byte count and buffer
// as the identity pack, so captured addresses are unchanged. One thread per
// output byte; runs once during weight prep.
__global__ __launch_bounds__(kCopyBlockThreads) void
w8a8_pack_nmajor_bytes_kernel(
    const int8_t* __restrict__ raw,
    int8_t* __restrict__ packed,
    int k,
    int n) {
  const int64_t total = static_cast<int64_t>(k) * n;
  const int64_t stride =
      static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x +
                     threadIdx.x;
       idx < total; idx += stride) {
    const int col = static_cast<int>(idx / k);
    const int kk = static_cast<int>(idx - static_cast<int64_t>(col) * k);
    packed[idx] = raw[static_cast<int64_t>(kk) * n + col];
  }
}

}  // namespace

// ---------------------------------------------------------------------------
// Host launch symbols consumed by csrc/bindings.cpp (TORCH_LIBRARY zth_w8a8).
// ---------------------------------------------------------------------------
extern "C" void launch_w8a8_gemm(const int8_t* a,
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
  // Exact assigned shape: glm_tp8_fused_qkv_a_proj_m16 (M=16, N=2624,
  // K=6144). Iteration 4 architecture: multi-N-tile reuse -- partial grid
  // (82, 8) = 656 one-wave zero-barrier blocks (5.47 blocks/CU, same as the
  // iteration-2 SPLIT_K=4 winner), each block computing two adjacent 16x16
  // N-tiles over its K slice (K/8 = 768 = 24 k32 steps) with one shared A
  // fragment load per step feeding two MMAs, then a same-stream combine
  // kernel (164 one-wave blocks) summing the eight int32 planes in ascending
  // split order with the fused x_scale * weight_scale bf16 epilogue.
  // Iteration 20: SPLIT_K 8 -> 4 (byte-reduction round) -- the partial grid
  // becomes (82, 4) = 328 one-wave zero-barrier blocks (2.73 blocks/CU, a
  // single wave with all blocks resident) over K/4 = 1536 = 48 k32 steps,
  // and the combine sums the four int32 planes in ascending split order.
  // Iteration 5: `b` is the n-major packed weight packed[n*K + k] =
  // raw[k*N + n] (launch_pack_w8a8_weight transposes it once out of the
  // timed region for (k,n)==(6144,2624)); the partial kernel loads A and
  // both B fragments with one aligned 8-B vector load per lane (int2 +
  // __builtin_memcpy) instead of the library's per-byte loaders.
  // Iteration 6: the partial kernel stages ONLY the cold B stream through a
  // 2,304-B LDS buffer (16-B dwordx4 cooperative loads, one vmcnt drain per
  // 2-k32-step chunk, ds_read_b64 fragment re-reads, next-chunk loads issued
  // before consumption); A stays direct and L2-hot. B fragment values are
  // byte-identical, so the int32 partial planes and the combine output stay
  // bit-identical; geometry, workspace and Graph structure are unchanged.
  // Iteration 7: the partial kernel DOUBLE-buffers the B staging -- two
  // ping-pong 2,304-B LDS planes (4,608 B total, < 48 KiB) and two
  // parity-alternating register sets with a two-chunk prefetch, no rotation
  // copies, zero barriers per K step (one wave per block) -- so every load
  // stays in flight across the back edge and the single vmcnt drain per
  // chunk lands only at its own staging write. B fragment values are still
  // byte-identical, so partials and the combine output stay bit-identical;
  // geometry, workspace and Graph structure are unchanged.
  // Iteration 9: the partial kernel WIDENS each staged chunk to FOUR k32
  // steps (128 K, 6 chunks per block at SPLIT_K=8) with the same double-
  // buffer ping-pong structure -- two 136-stride LDS planes (8,704 B total,
  // < 48 KiB), two parity register sets carrying 2 x 32 B of B + 32 B of A
  // per lane, two-chunk prefetch, zero barriers per K step -- halving the
  // per-chunk vmcnt drains and doubling the in-flight window. vmem_read
  // stays 31,488/replay, LDS instructions stay 31,488/replay, HBM read
  // stays at the compulsory floor; B fragment values are still
  // byte-identical, so partials and the combine output stay bit-identical;
  // geometry, workspace and Graph structure are unchanged.
  // Iteration 10: the partial kernel WIDENS each staged chunk to
  // SIX k32 steps (192 K, 4 chunks per block at SPLIT_K=8, 2 passes of 2)
  // with the same double-buffer ping-pong structure -- two 200-stride LDS
  // planes (12,800 B total, < 48 KiB), two parity register sets carrying
  // 2 x 48 B of B + 48 B of A per lane, two-chunk prefetch, zero barriers
  // per K step -- cutting the per-pass vmcnt drain events from 3 to 2 per
  // block with ~50% more in-flight bytes per event. vmem_read stays
  // 31,488/replay, LDS instructions stay 31,488/replay, HBM read stays at
  // the compulsory floor; B fragment values are still byte-identical, so
  // partials and the combine output stay bit-identical; geometry, workspace
  // and Graph structure are unchanged.
  // Iteration 12 (current partial-kernel architecture): the partial kernel
  // issues ALL FOUR chunks' B loads (24 dwordx4) plus the chunks 0/1 A
  // loads in the prologue and unrolls the 2-pass loop, so the first staging
  // consumer drains the whole K-slice with ONE vmcnt ladder: one cold-B
  // round trip per block (was two), wait-free pass-1 staging, no rotation
  // v_movs. A stays L2-pipelined (chunk 2/3 A issued in pass 0, captured in
  // pass 1). Same 656-block geometry, same 12,800-B LDS planes, same staged
  // bytes and k-ascending int32 order -> bit-identical partials and combine
  // output; vmem_read stays 31,488/replay, LDS instructions stay
  // 31,488/replay.
  // Iteration 13 (combine latency consolidation): the partial kernel is
  // UNTOUCHED. The combine kernel (164 one-wave blocks) now fetches the
  // four contiguous weight_scale elements of each lane with ONE aligned
  // 16-B load (was four serialized 4-B loads, each waited to completion
  // before its multiply) and stores the four 2-B bf16 outputs with ONE 8-B
  // store per lane (was four). Per-element arithmetic is byte-for-byte
  // unchanged -> the combine output stays bit-identical; grid, workspace
  // and Graph structure (partial + combine launches on the caller stream)
  // are unchanged.
  // K % 4 == 0, N % 32 == 0 hold for this exact shape. The workspace layout
  // is [164][4][256] int32 (tile-major since iteration 18) = 671,744 B
  // (iteration 20: SPLIT_K 8 -> 4 halves the plane traffic), guaranteed by
  // the API's 16-plane split budget for this shape
  // (allocate_workspace yields 2,686,976 B); the
  // guard below falls back to the generic scalar path if the caller-provided
  // workspace is smaller. The shape guard stays exact, so paired M=2 shapes
  // with the same (N, K) keep taking the generic scalar fallback (which
  // decodes the n-major pack for (n,k)==(2624,6144)).
  if (m == kDummaTileM && n == 2624 && k == 6144) {
    constexpr int kSplitK = 4;
    constexpr int kTilesPerBlock = 2;  // adjacent 16x16 N-tiles per block
    constexpr int kTiles = 2624 / kDummaTileN;             // 164
    constexpr int kBlockTiles = kTiles / kTilesPerBlock;   // 82
    constexpr int kPlaneInts = kDummaTileM * kDummaTileN;  // 256
    constexpr int64_t kPartialBytes =
        static_cast<int64_t>(kSplitK) * kTiles * kPlaneInts *
        static_cast<int64_t>(sizeof(int));                 // 671,744
    if (workspace != nullptr && workspace_bytes >= kPartialBytes) {
      hipLaunchKernelGGL(
          HIP_KERNEL_NAME(
              w8a8_dumma_m16n16k32_splitk_partial_kernel<kSplitK>),
          dim3(static_cast<unsigned>(kBlockTiles),
               static_cast<unsigned>(kSplitK)),
          dim3(kDummaBlockThreads),
          0,  // no dynamic shared memory
          stream,
          a,
          b,
          reinterpret_cast<int*>(workspace),
          n,
          k);
      hipLaunchKernelGGL(
          HIP_KERNEL_NAME(
              w8a8_dumma_m16n16k32_splitk_combine_kernel<kSplitK>),
          dim3(static_cast<unsigned>(kTiles)),
          dim3(kDummaBlockThreads),
          0,  // no dynamic shared memory
          stream,
          reinterpret_cast<const int*>(workspace),
          x_scale,
          weight_scale,
          reinterpret_cast<hip_bfloat16*>(out),
          n);
      return;
    }
    // Workspace too small for split-K=8: fall through to the generic scalar
    // fallback (still exact for this shape).
  }

  // Generic scalar fallback for every unmatched (m, n, k).
  launch_scalar_gemm(a, b, x_scale, weight_scale, out, m, n, k, stream);
}

extern "C" void launch_pack_w8a8_weight(const int8_t* raw_weight,
                                        const float* weight_scale,
                                        int8_t* packed_weight,
                                        float* packed_weight_scale,
                                        int k,
                                        int n,
                                        hipStream_t stream) {
  // Iteration 5: the exact (k, n) == (6144, 2624) weight is packed once,
  // outside the timed region and out of Graph capture, into the n-major
  // layout packed[n * K + k] = raw[k * N + n] (same byte count and buffer
  // addresses, so captured pointers stay valid; the partial kernel reads it
  // with col_major B fragments, one 8-B vector load per lane). Every other
  // (K, N) keeps the identity copy (raw row-major [K, N]).
  const int elem_count = k * n;
  if (k == 6144 && n == 2624) {
    const int64_t pack_blocks =
        (static_cast<int64_t>(elem_count) + kCopyBlockThreads - 1) /
        kCopyBlockThreads;
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(w8a8_pack_nmajor_bytes_kernel),
        dim3(static_cast<unsigned>(pack_blocks)),
        dim3(kCopyBlockThreads),
        0,  // no dynamic shared memory
        stream,
        raw_weight,
        packed_weight,
        k,
        n);
  } else {
    const int elem_blocks =
        (elem_count + kCopyBlockThreads - 1) / kCopyBlockThreads;
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(w8a8_identity_copy_kernel<int8_t>),
        dim3(static_cast<unsigned>(elem_blocks)),
        dim3(kCopyBlockThreads),
        0,  // no dynamic shared memory
        stream,
        raw_weight,
        packed_weight,
        elem_count);
  }

  const int scale_blocks =
      (n + kCopyBlockThreads - 1) / kCopyBlockThreads;
  hipLaunchKernelGGL(
      HIP_KERNEL_NAME(w8a8_identity_copy_kernel<float>),
      dim3(static_cast<unsigned>(scale_blocks)),
      dim3(kCopyBlockThreads),
      0,  // no dynamic shared memory
      stream,
      weight_scale,
      packed_weight_scale,
      n);
}
// @@end
