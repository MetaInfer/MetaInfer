// @@variant shape=tp8_wq_b_m16 commit=bfaef2fc76d0e4a095b2ffae2b466f2863511d43 added=2026-08-26
//   median_us=14.67 p90_us=14.7
//   source=hy3-dsh-tp8-m16-1-adf021ab
// W8A8 INT8 GEMM — gfx928 implementation (worker_1, physical GPU 1)
//
// Assigned shapes:
//   tp8_wq_b_m16           M=16,  N=4096, K=1024
//   tp8_indexer_wq_b_m16   M=16,  N=8192, K=1024
//
// Iteration 0 (correctness-first bootstrap): one simple scalar int8
// dot-product kernel maps one thread to one output element and computes the
// complete K loop exactly in int32 before applying the two fp32 scales and
// storing bf16. This remains the generic fallback for every shape that does
// not match the exact tp8_wq_b_m16 guard below.
//
// Iteration 1 (DUMMA bootstrap): the exact tp8_wq_b_m16 shape ran a minimal
// 16x16x32 DUMMA kernel, one 64-thread wavefront per 16x16 output tile
// (256 blocks, 1 wave, direct row-major global fragment loads). It built,
// passed exact int32 correctness and Graph capture, and measured 59.9 us
// median (vs the 32.137 us fixed Triton Graph baseline). Its gfx928 ISA
// showed a latency-bound K loop: the library fragment loader expanded into
// 18 global_load_ubyte with 19 serialized s_waitcnt around a single
// v_mmac_i32_16x16x32_i8 per step, and with ~2 blocks resident per CU there
// were too few independent load->MMA streams to hide that latency.
//
// Iteration 2 (architecture round — measure grid parallelism): changed ONLY
// the launch geometry to two wavefronts per block owning two adjacent 16x16
// N tiles (block tile 16x32, grid N/32 = 128 blocks = 1.07 blocks/CU, two
// co-resident full-K MMA chains per CU). It measured 83.0 us median — FLAT
// OR WORSE than iteration 1 — falsifying grid parallelism as the binding
// lever: the binding constraint is the per-step byte-load fragment path
// itself (18 global_load_ubyte + 19 s_waitcnt per step), and the mandated
// architecture axis is split-K, not more geometry.
//
// Iteration 3 (architecture round — split-K, this file): the unsplit grid
// has 128 blocks / 120 CUs = 1.07 < 2 blocks/CU and K = 1024 >= 1024, so the
// mandate requires split-K=2 plus a CU-aligned candidate, with int32
// partials written into the caller workspace and a combine+scale kernel in
// the timed Graph. This round implements a ONE-WAVE ZERO-BARRIER workspace
// split-K pair (also the mandate's alternative geometry branch — it reaches
// double the unsplit parallelism with zero barriers):
//   (1) w8a8_dumma_m16n16k32_splitk_partial_kernel<kSplitK> — one 64-thread
//       wavefront per block, grid = (N/16) * kSplitK = 512 blocks for
//       split-K=2 (4.27 blocks/CU >= the two-blocks-per-CU target, 512
//       resident wave-tasks vs 256 in the unsplit grid). Block b owns
//       (split = b % kSplitK, tile = b / kSplitK): split s sums its
//       32-aligned K-chunk ([0,512) or [512,1024) for split-K=2, 16
//       m16n16k32 steps per wave, same k-ascending operand order as the
//       unsplit chain) with the UNCHANGED direct du_load_matrix_sync path,
//       and stores the raw int32 accumulator fragment directly into the
//       caller workspace plane partials[split][tile][16][16] via the
//       verified gfx928 accumulator ownership (row = lane&15,
//       col_mod4 = lane>>4, x[i] -> col_mod4 + 4*i). No LDS, no barriers.
//   (2) w8a8_dumma_m16n16k32_combine_kernel<kSplitK> — one thread per
//       output element, grid = 256 blocks x 256 threads, sums the kSplitK
//       planes ASCENDING (s = 0..kSplitK-1; int32 addition is exact and
//       order-independent below 2^24, so the total is bit-identical to the
//       unsplit accumulator), applies (acc * x_scale[row]) *
//       weight_scale[col] in the same left-associative float order as the
//       reference, and stores round-to-nearest-even bf16. Launched
//       immediately after the partial kernel on the same stream, so both
//       kernels are captured into the timed Graph.
// The split factor is the single compile-time constant kSplitKMeasured = 2.
// The CU-aligned candidate kSplitKCUAligned = 15 (grid 3840 = exactly
// 32 blocks/CU on the 120-CU device, non-power-of-two, non-uniform 32-aligned
// K chunks) is compiled into the object and selected by flipping the
// constant, so the mandated occupancy sweep is a one-line change.
//
// Iteration 8 (HIP-only resource round — tune ONE occupancy limiter, the
// LDS footprint, from the accepted kernel's PMC): iterations 4-6 proved the
// per-step direct global fragment loads are poison (reg-prefetch 51.0 us,
// B-only LDS 67.0 us) and that only a fully A+B-staged zero-global K loop
// can win (the iteration-6 fused kernel, 25.97 us) — but the fused kernel
// stages the WHOLE K=1024 per block (35,456 B/block: s_a+s_b 16,640 each +
// s_part 2,048 + s_scale 128), pinning it at 1 block/CU co-resident. This
// round moves that staging to the SPLIT GRID (the exact next step prescribed
// by iteration 7): the accepted iteration-3 grid/workspace/combine Graph is
// kept (512 blocks x 64 threads, split-K=2, 2 int32 partial planes, separate
// combine kernel in the timed Graph), but the partial kernel now stages ONLY
// its own 512-K A chunk (16 x 512 B, row-major x_q) and B chunk (packed
// [N][K] P, 512 contiguous B per column) into LDS once: 16,896 B/block
// (s_a/s_b 16 x 528 = 512+16 padded stride) -> 3 blocks/CU co-resident on
// the 64 KiB LDS/CU (vs 1 block/CU for the fused kernel, 4.27 avg for the
// accepted kernel). The K loop then collapses to the fused kernel's proven
// zero-global chain: per step exactly two ds_read_b64 (A and B fragments,
// raw byte order == the du_load_matrix_sync expansion) + one v_mmac, same
// k-ascending chunk [0,512)/[512,1024) as iteration 3. Byte accounting per
// replay is UNCHANGED (no repeated HBM reads: B 4 MiB unique, A 16 KiB
// L2-hot re-read by 256 tiles, 512 KiB partial write+read, 128 KiB out) —
// occupancy is raised by shrinking the per-block LDS footprint, not by
// re-reading global memory. Transport prerequisite (iteration-5/6-validated):
// for (k,n)==(1024,4096) launch_pack_w8a8_weight now writes the [N][K]
// transpose P[n*K+k] = W[k*N+n] once outside the timed region/Graph, and the
// generic scalar fallback decodes it via a b_transposed flag so the paired
// M=2 validation and every other caller of the exact (n,k) stay bit-exact;
// every other shape keeps the identity pack.
//
// Iteration 9 (HIP-only occupancy probe — the mandated one-constant staged
// split sweep): the accepted iteration-8 kernel's PMC (staged partial,
// profiled 14.56 us, 512 blocks x 64 threads, 16,896 B/block LDS -> 3 blocks
// /CU co-resident on the 64 KiB LDS/CU, 0.75 waves/SIMD resident with the
// 4th SIMD of every CU idle, grid 4.27 blocks/CU -> 1.42 serial block rounds)
// shows the staged transport's per-step chain is short (2 ds_read_b64 +
// v_mmac, ~LDS-latency), so the residual limiter is resident-stream count
// over the one-per-block HBM staging wait, not per-step global latency (the
// direct-load split-K=4 regression of iteration 7 does NOT transfer: that
// transport kept a ~500-600 cycle HBM chain inside EVERY K step). This round
// sweeps the dispatched split to S=4 — the exact next step iteration 8
// prescribed: 1024 blocks x 64 threads, uniform 32-aligned K chunks
// [0,256)/[256,512)/[512,768)/[768,1024), 8 m16n16k32 steps per block,
// staging shrinks to 8,704 B/block (s_a/s_b 16 x 272 = 256+16 padded
// stride, the same 4-dword-mod-32 bank skew as 528) -> 7 blocks/CU
// co-resident (floor(65536/8704) = 7), 1.75 waves/SIMD with ALL FOUR SIMDs
// busy, 1.22 serial block rounds, 4 int32 partial planes = 1 MiB <= the
// API's 16-plane / 4 MiB workspace. Exactness is preserved by construction:
// identical k-ascending m16n16k32 order within each chunk, chunks tile
// [0,1024) exactly once, the UNCHANGED ascending 4-plane combine sums
// (p0+p1+p2)+p3 == the S=2 two-plane total == the unsplit accumulator
// (int32 exact below 2^24), so the output is bit-identical to iteration 8
// (expected 0 mismatches). Graph contract unchanged: same two kernels, same
// stream order, 4 planes within the same 16-plane allocation, every partial
// overwritten each launch.
//
// Repair 1 (exact-correctness defect of the S=4 sweep, no architecture
// change): the cooperative staging loop decomposed the 16-B chunk index with
// the S=2-fixed mapping r = c>>5, ko = (c&31)*16 (32 chunks per row). That is
// exact at kLocal=512 but at S=4 (kLocal=256, kChunksLocal=256) c only
// reaches 255, so only A rows / B columns 0..7 were staged and the K loop
// read UNINITIALIZED LDS for rows/columns 8..15 (75% of every tile wrong —
// rows 8..15 in all columns plus columns 8..15 of rows 0..7; observed
// mismatch_count 49121/65536, first mismatch m=0 n=8). Fixed in place by
// deriving the per-row chunk count from the actual chunk length,
// kChunksPerRow = kLocal/16 (r = c/kChunksPerRow, ko = (c%kChunksPerRow)*16),
// which is bit-identical to the old mapping at S=2 (512/16 = 32) and stages
// all 16 rows/columns at S=4; kLocal stays a 32-multiple so every 16-B
// global load/store alignment is unchanged. The S=4 mapping/performance idea
// (1024 blocks, 8,704 B/block LDS, 7 blocks/CU) is preserved.
//
// Iteration 11 (HIP-only consolidation — fused combine tail): the accepted
// S=4 tree's operator-aggregate PMC (partial 10.879 us + combine 3.04 us =
// 13.919 us vs the 15.08 timed median) pins the residual wall on the combine
// STRUCTURE: a second kernel launch (~1.16 us inter-kernel/graph gap) plus a
// 1 MiB int32 plane re-read that is HBM-cold (combine L2 hit rate only 19.3%:
// the 4 MiB B stream evicts the partials), while the sibling TP4/M16 lineage
// (qkv_proj.hip rounds 17/23) proves the canonical M<=32 fix is the FUSED
// ATOMIC COMBINE. This round merges the combine into the partial kernel tail:
// every block stores its plane as before, then tid-0 does __threadfence() +
// atomicAdd(&counters[tile], 1), and the block whose atomicAdd returns the
// kSplitK-th arrival of its replay (monotonic mod-kSplitK: r % kSplitK ==
// kSplitK-1; int32 wraparound would need 2^31/4 ~ 536M replays) sums the
// tile's kSplitK planes in ascending slice order (identical per-element int32
// order as the removed combine), applies the identical left-associative
// fp32 scale and RN-even bf16 conversion, and stores byte-identical output
// bytes (0 mismatches expected). The Graph becomes ONE kernel launch per
// replay. The per-tile counters (256 x int32 = 1 KiB) live in the last
// kCountersBytes of the caller workspace (planes 0..3 end at 1 MiB, far below
// the 4 MiB allocation's tail; no overlap), zeroed once per workspace by an
// async hipMemsetAsync on the caller's stream before the first launch (never
// part of steady-state replay; the mod-kSplitK test is also correct if a
// reset were captured, since arrivals 0..kSplitK-1 still identify the last).
// The staged K loop, transport, pack, exact-shape guards, generic scalar
// fallback and the compile-only kSplitKCUAligned=15 two-kernel branch are
// unchanged; the generic fallback never touches the counters.
//
// Layout contract (logical tensors, contiguous / row-major):
//   x_q                   [M, K] int8
//   packed_weight         [K, N] int8 (identity D2D copy of raw_weight made
//                            by launch_pack_w8a8_weight)
//   x_scale               [M] fp32
//   packed_weight_scale   [N] fp32 (identity copy of weight_scale)
//   out                   [M, N] bf16
//   workspace             caller-owned int32 partial planes:
//                            partials[split][tile][16][16], plane stride
//                            (N/16)*256 int32 elements
//
// Math:
//   out[m, n] = bf16( (int32) sum_k x_q[m,k] * W[k,n] * x_scale[m]
//                     * weight_scale[n] )
//
// Exactness: the assigned K=1024 keeps every partial int8 dot below 2^24, so
// the int32 sum (in any grouping order) is bit-identical to the fp32 torch
// reference ((A.float() @ B.float()) * x_scale * weight_scale.T) for the
// timed shapes. The generic fallback is exact in int32 for every supported
// shape (max |dot| = K * 127 * 127 < 2^31).

#include <hip/hip_runtime.h>
#include <hip/hip_bfloat16.h>
#include <du_mma.h>

#include <cstdint>

namespace {

constexpr int kScalarBlock = 256;  // 4 wavefronts of 64 (blockDim % 64 == 0)
constexpr int kPackBlock = 256;

// DUMMA INT8 m16n16k32 constants: one 64-thread wavefront owns one fragment.
constexpr int kDummaM = 16;
constexpr int kDummaN = 16;
constexpr int kDummaK = 32;
constexpr int kDummaWave = 64;

// Exact-shape constants for tp8_wq_b_m16 (M=16, N=4096, K=1024).
constexpr int kExactK = 1024;
constexpr int kExactN = 4096;
constexpr int kExactNTiles = kExactN / kDummaN;  // 256

// Combine kernel block size.
constexpr int kCombineThreads = 256;

// Iteration 9 staged occupancy sweep — split factor dispatched this round.
// The exact-shape fast path launches the staged partial kernel below, which
// supports every trusted split in {2..15} via runtime 32-aligned chunks, so
// the occupancy sweep stays a one-constant change. S=4: 1024 blocks x 64
// threads, K-chunk 256 per block, LDS 8,704 B/block (s_a/s_b 16 x 272 =
// 256+16 padded stride) -> 7 blocks/CU co-resident (vs 3 at S=2), grid
// average 8.53 blocks/CU -> 1.22 serial block rounds (vs 1.42), 1.75
// waves/SIMD resident with all 4 SIMDs busy (vs 0.75 on 3 of 4 SIMDs), 4
// int32 partial planes = 1 MiB <= the API's 16-plane / 4 MiB workspace.
constexpr int kSplitKMeasured = 4;
// Implemented CU-aligned candidate: grid = 256 * 15 = 3840 blocks = exactly
// 32 blocks/CU on the 120-CU gfx928 (integer blocks per CU, non-power-of-two,
// non-uniform 32-aligned K chunks). Compiled into the object via the
// iteration-3 direct-load partial kernel; never dispatched (the
// workspace_bytes == -1 guard cannot hold and it does not decode the [N][K]
// pack).
constexpr int kSplitKCUAligned = 15;

// Iteration 11 fused combine tail: per-tile arrival counters (256 tiles x
// int32 = 1 KiB) live in the last kCountersBytes of the caller workspace
// (planes 0..3 end at 1 MiB, far below the counters' offset in the API's
// 16-plane / 4 MiB allocation, so there is no overlap with partial data).
constexpr int64_t kCountersBytes = kExactNTiles * sizeof(int32_t);  // 1 KiB

// Round-to-nearest-even float -> bfloat16 bit pattern (sibling-validated
// helper; identical RN-even conversion to the hip_bfloat16 stores used by
// the old combine kernel, so the packed tail store is byte-identical).
__device__ __forceinline__ uint16_t float_to_bf16_bits(float f) {
  uint32_t u = 0;
  __builtin_memcpy(&u, &f, sizeof(u));
  const uint32_t bias = 0x7FFFu + ((u >> 16) & 1u);
  u += bias;
  return static_cast<uint16_t>(u >> 16);
}

// One-time async zero of the fused-combine arrival counters, guarded by the
// workspace pointer: issued once per workspace before the first launch on
// the caller's stream, so the captured Graph contains only the single fused
// kernel launch and every steady-state replay is memset-free.
static void* g_fused_counters_zeroed = nullptr;

// ---------------------------------------------------------------------------
// Iteration 3 split-K partial GEMM for the exact tp8_wq_b_m16 shape.
// One 64-thread wavefront per block, grid = (N/16) * kSplitK blocks. Block b
// owns (split = b % kSplitK, tile = b / kSplitK): split s computes the
// 16x16 output tile over its 32-aligned K-chunk [kStart(s), kEnd(s)) in
// k-ascending m16n16k32 steps (16 steps for split-K=2), with the unchanged
// direct row-major global fragment loads, and stores the raw int32
// accumulator fragment into workspace plane s (tile-major, row stride 16)
// via the verified gfx928 accumulator ownership. Zero barriers, zero LDS.
// ---------------------------------------------------------------------------
template <int kSplitK>
__global__ __launch_bounds__(kDummaWave)
void w8a8_dumma_m16n16k32_splitk_partial_kernel(
    const int8_t* __restrict__ x_q,          // [16, K] row-major
    const int8_t* __restrict__ weight,       // [K, N] row-major (identity pack)
    int32_t* __restrict__ partials,          // [kSplitK][N/16][16][16] int32
    const int n) {
  const int split = static_cast<int>(blockIdx.x) % kSplitK;
  const int tile = static_cast<int>(blockIdx.x) / kSplitK;
  const int lane = static_cast<int>(threadIdx.x);
  const int n0 = tile * kDummaN;

  // 32-aligned non-uniform K chunks: kStart(s) = floor(s*K/S) rounded down
  // to a DUMMA-K multiple; the last chunk absorbs the remainder. For
  // split-K=2 this is exactly [0,512) and [512,1024).
  const int kStart = ((split * kExactK) / kSplitK) & ~(kDummaK - 1);
  const int kEnd = (split == kSplitK - 1)
                       ? kExactK
                       : ((((split + 1) * kExactK) / kSplitK) &
                          ~(kDummaK - 1));

  du::dumma::DUFragment<du::dumma::matrix_a, kDummaM, kDummaN, kDummaK,
                        signed char, du::dumma::row_major> a_frag;
  du::dumma::DUFragment<du::dumma::matrix_b, kDummaM, kDummaN, kDummaK,
                        signed char, du::dumma::row_major> b_frag;
  du::dumma::DUFragment<du::dumma::accumulator, kDummaM, kDummaN, kDummaK,
                        int> acc_frag;
  du::dumma::du_fill_fragment(acc_frag, 0);

  // Same per-step instruction stream as iterations 1/2 (direct global
  // fragment loads, exact int32 k-ascending accumulation); only the K range
  // per wave shrinks by the split factor. Fully unrolled (16 steps for
  // split-K=2) like the validated split-K lineage kernels.
#pragma unroll
  for (int k0 = kStart; k0 < kEnd; k0 += kDummaK) {
    du::dumma::du_load_matrix_sync(a_frag, x_q + k0, kExactK);
    du::dumma::du_load_matrix_sync(
        b_frag, weight + static_cast<int64_t>(k0) * n + n0, n);
    du::dumma::du_mma_sync(acc_frag, a_frag, b_frag, acc_frag);
  }

  // gfx928 int8 m16n16k32 accumulator ownership, established against
  // du_store_matrix_sync: lane%16 selects the row, lane/16 selects col%4,
  // and x[i] holds the columns col%4 + 4*i. Store the raw int32 partial
  // (no scaling) into the caller workspace plane.
  const int row = lane & 15;
  const int col_mod4 = lane >> 4;
  const int n_tiles = n / kDummaN;
  int32_t* __restrict__ p = partials +
      (static_cast<int64_t>(split) * n_tiles + tile) *
          (kDummaM * kDummaN);
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    p[row * kDummaN + col_mod4 + 4 * i] = acc_frag.x[i];
  }
}

// ---------------------------------------------------------------------------
// Iteration 8 split-grid staged partial GEMM for the exact tp8_wq_b_m16
// shape — the LDS-footprint occupancy tune. Same grid as iteration 3 (512
// blocks x 64 threads, split = b % kSplitK, tile = b / kSplitK, 32-aligned
// non-uniform chunks [kStart, kEnd)), same workspace partial-plane contract,
// same combine kernel in the timed Graph. The per-step direct global
// fragment loads are replaced by ONE cooperative staging pass per block:
// each lane moves 16-B chunks of the block's own A chunk (x_q rows, 512 B
// each at stride K) and B chunk (packed [N][K] P, 512 contiguous B per
// column) with coalesced dwordx4 global reads and 16-B aligned b128 LDS
// stores — 16 global loads + 16 LDS stores per lane at split-K=2 — followed
// by ONE __syncthreads() and a zero-global K loop whose every step is
// exactly two ds_read_b64 (A and B fragments; raw byte order == the
// du_load_matrix_sync expansion) + one v_mmac (the iteration-6-validated
// fused chain, minus the barrier/s_part handoff). LDS = 16,896 B/block at
// the dispatched S=2 (s_a/s_b 16 x 528 = 512+16 padded stride) -> 3 blocks
// /CU co-resident on the 64 KiB LDS/CU. Template supports every trusted
// split in {2..15}: the buffer covers ceil(K/S) rounded up to a DUMMA-K
// multiple and both loops use runtime chunk bounds, so the mandated sweep
// stays a one-constant change (kSplitKMeasured).
// ---------------------------------------------------------------------------
template <int kSplitK>
__global__ __launch_bounds__(kDummaWave)
void w8a8_dumma_m16n16k32_splitk_staged_partial_kernel(
    const int8_t* __restrict__ x_q,          // [16, K] row-major
    const int8_t* __restrict__ weight,       // packed [N, K] n-major (exact)
    int32_t* __restrict__ partials,          // [kSplitK][N/16][16][16] int32
    const float* __restrict__ x_scale,       // [16]
    const float* __restrict__ weight_scale,  // [N]
    hip_bfloat16* __restrict__ out,          // [16, N] bf16
    int32_t* __restrict__ counters,          // [N/16] per-tile arrival counts
    const int n) {
  constexpr int kChunkMax =
      (((kExactK + kSplitK - 1) / kSplitK + kDummaK - 1) / kDummaK) *
      kDummaK;  // 512 at split-K=2
  constexpr int kStageStride = kChunkMax + 16;  // 528 = 512+16 at split-K=2

  __shared__ __align__(16) int8_t s_a[kDummaM * kStageStride];
  __shared__ __align__(16) int8_t s_b[kDummaN * kStageStride];

  const int split = static_cast<int>(blockIdx.x) % kSplitK;
  const int tile = static_cast<int>(blockIdx.x) / kSplitK;
  const int lane = static_cast<int>(threadIdx.x);
  const int n0 = tile * kDummaN;

  // Same 32-aligned non-uniform K chunks as the iteration-3 kernel; for
  // split-K=2 this is exactly [0,512) and [512,1024).
  const int kStart = ((split * kExactK) / kSplitK) & ~(kDummaK - 1);
  const int kEnd = (split == kSplitK - 1)
                       ? kExactK
                       : ((((split + 1) * kExactK) / kSplitK) &
                          ~(kDummaK - 1));
  const int kLocal = kEnd - kStart;  // 32-multiple, <= kChunkMax

  // Cooperative staging of this block's A and B chunks: every 16-B chunk of
  // each matrix is moved exactly once. Per pass each lane issues one
  // contiguous dwordx4 global load per matrix (coalesced 16-B sectors: lanes
  // 0..31 cover one 512-B row/column segment, lanes 32..63 the next) and one
  // 16-B aligned b128 LDS store (stride 528 = 16*33 keeps every store
  // 16-B-aligned). c enumerates the kChunksLocal 16-B chunks of ONE matrix
  // (A rows == B columns, both 16 x kLocal) and the chunk -> (row, k-offset)
  // decomposition uses kChunksPerRow = kLocal/16 so ALL 16 rows/columns are
  // staged at every split; the fixed 32-chunks-per-row shift mapping only
  // covered rows/columns 0..7 once kLocal dropped to 256 at split-K=4
  // (repair 1: uninitialized-LDS reads for rows/columns 8..15).
  const int kChunksLocal = (kLocal * kDummaM) / 16;  // 16-B chunks per matrix
  const int kChunksPerRow = kLocal / 16;  // 16-B chunks per A row / B column
  const int64_t b_base =
      static_cast<int64_t>(n0) * kExactK + kStart;  // packed P column base
#pragma unroll
  for (int c = lane; c < kChunksLocal; c += kDummaWave) {
    const int r = c / kChunksPerRow;          // A row == B column in the 16
    const int ko = (c % kChunksPerRow) * 16;  // k offset in the row/column
    const int4 va = *reinterpret_cast<const int4*>(
        x_q + r * kExactK + kStart + ko);
    const int4 vb = *reinterpret_cast<const int4*>(
        weight + b_base + static_cast<int64_t>(r) * kExactK + ko);
    *reinterpret_cast<int4*>(s_a + r * kStageStride + ko) = va;
    *reinterpret_cast<int4*>(s_b + r * kStageStride + ko) = vb;
  }
  __syncthreads();

  du::dumma::DUFragment<du::dumma::matrix_a, kDummaM, kDummaN, kDummaK,
                        signed char, du::dumma::row_major> a_frag;
  du::dumma::DUFragment<du::dumma::matrix_b, kDummaM, kDummaN, kDummaK,
                        signed char, du::dumma::row_major> b_frag;
  du::dumma::DUFragment<du::dumma::accumulator, kDummaM, kDummaN, kDummaK,
                        int> acc_frag;
  du::dumma::du_fill_fragment(acc_frag, 0);

  // Fragment ownership == the du_load_matrix_sync expansion (validated in
  // iterations 3/5/6): A: row = lane&15, kg = lane>>4, x[i] = A[row][kg*8+i];
  // B: col = lane&15, kg = lane>>4, x[i] = P[n0+col][kg*8+i] -- 8 contiguous
  // staged bytes per fragment, one ds_read_b64 each, raw byte order (the
  // loader's byte-reassembly is a no-op on the 8-byte path, so the manual
  // fill is bit-identical). Zero-global, zero-barrier k-ascending chain over
  // the block's own chunk.
  const int frag_row = lane & 15;  // A row == B column
  const int kg = lane >> 4;
  const int lds_off = frag_row * kStageStride + kg * 8;
  const int kSteps = kLocal / kDummaK;
#pragma unroll
  for (int s = 0; s < kSteps; ++s) {
    *reinterpret_cast<int2*>(&a_frag.x[0]) =
        *reinterpret_cast<const int2*>(s_a + lds_off + s * kDummaK);
    *reinterpret_cast<int2*>(&b_frag.x[0]) =
        *reinterpret_cast<const int2*>(s_b + lds_off + s * kDummaK);
    du::dumma::du_mma_sync(acc_frag, a_frag, b_frag, acc_frag);
  }

  // Same raw int32 partial-plane store as iteration 3 (verified gfx928
  // accumulator ownership: row = lane&15, col_mod4 = lane>>4, x[i] ->
  // col_mod4 + 4*i).
  const int row = lane & 15;
  const int col_mod4 = lane >> 4;
  const int n_tiles = n / kDummaN;
  int32_t* __restrict__ p = partials +
      (static_cast<int64_t>(split) * n_tiles + tile) *
          (kDummaM * kDummaN);
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    p[row * kDummaN + col_mod4 + 4 * i] = acc_frag.x[i];
  }

  // Iteration 11 fused combine tail (replaces the separate combine kernel in
  // the timed Graph; sibling-validated arrival protocol): every block's tid-0
  // releases with __threadfence() + atomicAdd(&counters[tile], 1); the LAST
  // arrival for its tile (atomicAdd return value r with r % kSplitK ==
  // kSplitK-1 — hardware-ordered after all kSplitK sibling stores+fences, no
  // spin loop, no residency assumption) sums the tile's planes in ascending
  // slice order — the identical per-element int32 accumulation order as the
  // old combine — applies the identical (acc * x_scale[row]) *
  // weight_scale[col] fp32 scale order, and stores RN-even bf16 at the same
  // addresses. The counters are MONOTONIC across replays: replay r's arrivals
  // return (r-1)*kSplitK + 0..kSplitK-1, so the kSplitK-th arrival of EVERY
  // replay is exactly the r % kSplitK == kSplitK-1 one (int32 wraparound
  // would need 2^31/4 ~ 536M replays); launch_w8a8_gemm zeroes them once per
  // workspace before the first launch and the generic fallback never touches
  // them. Element mapping: lin0 = lane*4 covers the 256-element tile exactly
  // once (row = lin0>>4, col0 = lin0&15, 4 consecutive columns per lane — 4 |
  // 16 so a run never crosses the 16-column row boundary), each plane is read
  // as one perfectly coalesced dwordx4 per lane, and the four 2-B bf16
  // results pack into one 8-B aligned uint64 store — bit-identical output
  // bytes to the removed combine kernel (0 mismatches expected).
  __syncthreads();
  __shared__ int s_is_last;
  if (threadIdx.x == 0) {
    __threadfence();  // release: this block's plane stores are visible to
                      // the observer of its arrival atomic
    const int arrived = atomicAdd(&counters[tile], 1);
    __threadfence();  // acquire: the reads below (after the barrier) see
                      // every sibling store released before prior arrivals
    s_is_last = ((arrived % kSplitK) == kSplitK - 1);
  }
  __syncthreads();
  if (s_is_last) {
    const int lin0 = lane * 4;   // 4 consecutive columns of one row
    const int crow = lin0 >> 4;  // 0..15
    const int col0 = lin0 & 15;  // 0..12 (multiple of 4)
    int32_t sums[4] = {0, 0, 0, 0};
#pragma unroll
    for (int s = 0; s < kSplitK; ++s) {
      const int4 plane4 = *reinterpret_cast<const int4*>(
          partials + (static_cast<int64_t>(s) * n_tiles + tile) *
                         (kDummaM * kDummaN) +
          lin0);
      sums[0] += plane4.x;
      sums[1] += plane4.y;
      sums[2] += plane4.z;
      sums[3] += plane4.w;
    }
    const int out_col0 = tile * kDummaN + col0;
    uint64_t packed = 0;
#pragma unroll
    for (int e = 0; e < 4; ++e) {
      const float scaled = static_cast<float>(sums[e]) * x_scale[crow] *
                           weight_scale[out_col0 + e];
      packed |= static_cast<uint64_t>(float_to_bf16_bits(scaled)) << (16 * e);
    }
    *reinterpret_cast<uint64_t*>(out + static_cast<int64_t>(crow) * n +
                                 out_col0) = packed;
  }
}

// ---------------------------------------------------------------------------
// Iteration 3 split-K combine + scale for the exact tp8_wq_b_m16 shape:
// reduce the kSplitK int32 partial planes ascending (each plane is
// [n_tiles][16][16], tile-major with row stride 16), apply
// (acc * x_scale[row]) * weight_scale[col] in the same left-associative
// float order as the reference, and store round-to-nearest-even bf16.
// One thread per output element, grid = ceil(16*n / 256) = 256 blocks.
// ---------------------------------------------------------------------------
template <int kSplitK>
__global__ __launch_bounds__(kCombineThreads)
void w8a8_dumma_m16n16k32_combine_kernel(
    const int32_t* __restrict__ partials,
    const float* __restrict__ x_scale,       // [16]
    const float* __restrict__ weight_scale,  // [N]
    hip_bfloat16* __restrict__ out,          // [16, N] bf16
    const int n) {
  const int idx = static_cast<int>(blockIdx.x) * kCombineThreads +
                  static_cast<int>(threadIdx.x);
  const int total = kDummaM * n;
  if (idx >= total) {
    return;
  }

  const int row = idx / n;
  const int col = idx - row * n;
  const int tile = col / kDummaN;
  const int tile_base = tile * (kDummaM * kDummaN) +
                        row * kDummaN + (col & (kDummaN - 1));
  const int n_tiles = n / kDummaN;
  int32_t sum = 0;
#pragma unroll
  for (int s = 0; s < kSplitK; ++s) {
    sum += partials[static_cast<int64_t>(s) * n_tiles *
                        (kDummaM * kDummaN) +
                    tile_base];
  }

  // Same left-to-right fp32 evaluation order as the reference.
  const float scaled =
      static_cast<float>(sum) * x_scale[row] * weight_scale[col];
  out[idx] = hip_bfloat16(scaled);
}

// One thread computes exactly one out[row, col] element. Adjacent lanes map
// to adjacent addresses in the fastest-changing N dimension.
// Iteration 8: b_transposed selects the exact-shape packed [N][K] layout
// (P[n*K + k] = W[k*N + n], produced by launch_pack_w8a8_weight), so the
// logical column col starts at P[col*K] with unit stride; the identity pack
// keeps the legacy row-major [K][N] layout with stride n. The branch is
// hoisted out of the K loop by the compiler.
__global__ __launch_bounds__(kScalarBlock) void w8a8_gemm_scalar_kernel(
    const int8_t* __restrict__ x_q,          // [M, K] row-major
    const int8_t* __restrict__ weight,       // [K, N] row-major (identity pack)
    const float* __restrict__ x_scale,       // [M]
    const float* __restrict__ weight_scale,  // [N]
    hip_bfloat16* __restrict__ out,          // [M, N] bf16
    const int m,
    const int n,
    const int k,
    const int b_transposed) {
  const int64_t tid =
      static_cast<int64_t>(blockIdx.x) * kScalarBlock + threadIdx.x;
  const int64_t total = static_cast<int64_t>(m) * n;
  if (tid >= total) {
    return;
  }
  const int row = static_cast<int>(tid / n);
  const int col = static_cast<int>(tid - static_cast<int64_t>(row) * n);
  const int8_t* a_row = x_q + static_cast<int64_t>(row) * k;
  const int8_t* b_col =
      weight + (b_transposed ? static_cast<int64_t>(col) * k : col);
  const int64_t b_stride = b_transposed ? 1 : static_cast<int64_t>(n);
  int32_t acc = 0;
  for (int kk = 0; kk < k; ++kk) {
    acc += static_cast<int32_t>(a_row[kk]) *
           static_cast<int32_t>(b_col[static_cast<int64_t>(kk) * b_stride]);
  }
  // Same left-to-right fp32 evaluation order as the torch reference.
  const float scaled =
      static_cast<float>(acc) * x_scale[row] * weight_scale[col];
  out[tid] = hip_bfloat16(scaled);
}

// Generic element-wise device-to-device copy used by the identity pack.
template <typename T>
__global__ __launch_bounds__(kPackBlock) void w8a8_identity_copy_kernel(
    const T* __restrict__ src,
    T* __restrict__ dst,
    const int64_t count) {
  const int64_t tid =
      static_cast<int64_t>(blockIdx.x) * kPackBlock + threadIdx.x;
  if (tid < count) {
    dst[tid] = src[tid];
  }
}

// Iteration 8 exact-shape pack: writes the [N][K] transpose
// dst[n*K + k] = src[k*N + n] (used for K=1024, N=4096), so the staged
// partial kernel reads each block's 16 B columns as 16 contiguous 512-B
// segments and every B fragment as one 8-byte ds_read_b64. One thread per
// 16-byte k-chunk of one column; runs once per weight outside the timed
// region, so the strided byte loads are acceptable. All other shapes keep
// the identity copy kernel above.
__global__ __launch_bounds__(kPackBlock) void w8a8_pack_transpose_i8_kernel(
    const int8_t* __restrict__ src,
    int8_t* __restrict__ dst,
    const int k,
    const int n) {
  const int kchunks = (k + 15) >> 4;
  const int64_t total = static_cast<int64_t>(n) * kchunks;
  const int64_t stride =
      static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t t = static_cast<int64_t>(blockIdx.x) * blockDim.x +
                   static_cast<int64_t>(threadIdx.x);
       t < total; t += stride) {
    const int col = static_cast<int>(t / kchunks);
    const int k0 = static_cast<int>(t % kchunks) * 16;
    alignas(16) int8_t chunk[16];
#pragma unroll
    for (int j = 0; j < 16; ++j) {
      chunk[j] = src[(static_cast<int64_t>(k0) + j) * n + col];
    }
    *reinterpret_cast<int4*>(dst + static_cast<int64_t>(col) * k + k0) =
        *reinterpret_cast<const int4*>(chunk);
  }
}

inline int blocks_for(const int64_t count, const int block) {
  const int64_t b = (count + block - 1) / block;
  return b < 1 ? 1 : static_cast<int>(b);
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
  // Exact-shape fast path: tp8_wq_b_m16 (M=16, N=4096, K=1024). Iteration 9:
  // the split-grid STAGED partial kernel at the dispatched split S=4 — 1024
  // blocks x 64 threads, each block stages its own 256-K A and B chunks into
  // LDS once (8,704 B/block -> 7 blocks/CU co-resident, all 4 SIMDs busy)
  // and then runs a zero-global two-ds_read_b64 + v_mmac K loop — writes the
  // 4 int32 partial planes into the caller workspace (4 * 16 * 4096 * 4 =
  // 1 MiB; the API allocates 16 planes / 4 MiB for this shape, so this
  // always qualifies). Iteration 11: the combine is FUSED into the partial
  // kernel tail (per-tile arrival counters in the last 1 KiB of the caller
  // workspace; the last arrival per tile sums the planes and stores the
  // scaled bf16 output), so the timed Graph contains exactly ONE kernel
  // launch per replay on the same stream, every partial element overwritten
  // each launch (no clear), and the counters are zeroed once per workspace
  // by an async hipMemsetAsync before the first launch (never captured).
  // B is read from the exact-shape [N][K] pack P[n*K+k] = W[k*N+n] produced
  // once by launch_pack_w8a8_weight outside the timed region/Graph.
  if (m == kDummaM && n == kExactN && k == kExactK) {
    constexpr int kSplitK = kSplitKMeasured;
    constexpr int64_t kPartialsBytes =
        static_cast<int64_t>(kSplitK) * kDummaM * kExactN * sizeof(int32_t);
    // Iteration 11: the fused path needs the 4 int32 partial planes PLUS the
    // per-tile arrival counters (last kCountersBytes of the caller
    // workspace); the API's 16-plane / 4 MiB allocation always qualifies.
    if (workspace != nullptr &&
        workspace_bytes >= kPartialsBytes + kCountersBytes) {
      int32_t* counters = reinterpret_cast<int32_t*>(
          static_cast<char*>(workspace) + workspace_bytes - kCountersBytes);
      // One-time async zero of the arrival counters on the caller's stream,
      // guarded by the workspace pointer: issued before the first launch with
      // a given workspace, so steady-state replays (and the captured Graph)
      // contain only the single fused kernel launch.
      if (g_fused_counters_zeroed != workspace) {
        hipMemsetAsync(counters, 0, kCountersBytes, stream);
        g_fused_counters_zeroed = workspace;
      }
      // ONE kernel per replay: the staged partial kernel now fuses the
      // combine into its tail (last arrival per tile sums the planes and
      // stores the scaled bf16 output). Same stream, same workspace
      // addresses, every partial overwritten each launch.
      hipLaunchKernelGGL(
          HIP_KERNEL_NAME(
              w8a8_dumma_m16n16k32_splitk_staged_partial_kernel<kSplitK>),
          dim3(static_cast<unsigned>(kExactNTiles * kSplitK)),
          dim3(kDummaWave),
          0,
          stream,
          a,
          b,
          static_cast<int32_t*>(workspace),
          x_scale,
          weight_scale,
          static_cast<hip_bfloat16*>(out),
          counters,
          n);
      return;
    }
    // Workspace smaller than the fused budget (never true for the API's
    // 16-plane allocation): fall through to the generic scalar fallback.
  }

  // Compile-only reference for the mandated CU-aligned candidate
  // (kSplitK=15 -> grid 3840 = exactly 32 blocks/CU, non-power-of-two,
  // non-uniform 32-aligned K chunks). Never dispatched: the workspace_bytes
  // == -1 guard cannot hold for any caller-provided byte count, and this
  // iteration-3 direct-load instantiation does NOT decode the exact-shape
  // [N][K] pack (it predates it), so it must never be launched against the
  // packed buffer. It keeps the iteration-3 kernel compiled (and correct)
  // while kSplitKMeasured = 2 dispatches the staged partial kernel; the
  // staged kernel supports every trusted split in {2..15} with runtime
  // 32-aligned chunks, so the occupancy sweep stays a one-constant change.
  if (workspace_bytes == -1 && workspace != nullptr) {
    constexpr int kSplitK = kSplitKCUAligned;
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(w8a8_dumma_m16n16k32_splitk_partial_kernel<kSplitK>),
        dim3(static_cast<unsigned>(kExactNTiles * kSplitK)),
        dim3(kDummaWave),
        0,
        stream,
        a,
        b,
        static_cast<int32_t*>(workspace),
        n);
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(w8a8_dumma_m16n16k32_combine_kernel<kSplitK>),
        dim3(static_cast<unsigned>(
            (kDummaM * n + kCombineThreads - 1) / kCombineThreads)),
        dim3(kCombineThreads),
        0,
        stream,
        static_cast<const int32_t*>(workspace),
        x_scale,
        weight_scale,
        static_cast<hip_bfloat16*>(out),
        n);
  }

  // Every other (m, n, k) — including the paired M=2 API shape with the same
  // (N, K) and the indexer N=8192 shape — reaches this single generic scalar
  // launch. Exact-shape guards sit BEFORE this fallback and never remove it.
  // Iteration 8: the exact (n == 4096 && k == 1024) weight is packed as the
  // [N][K] transpose by launch_pack_w8a8_weight, so every non-M16 caller of
  // this pair — including the paired M=2 validation — reads it through the
  // scalar fallback with the transposed flag. All other shapes use the
  // identity pack and the legacy row-major read.
  const int b_transposed = (n == kExactN && k == kExactK) ? 1 : 0;
  const int64_t total = static_cast<int64_t>(m) * n;
  const int blocks = blocks_for(total, kScalarBlock);
  hipLaunchKernelGGL(
      HIP_KERNEL_NAME(w8a8_gemm_scalar_kernel),
      dim3(blocks),
      dim3(kScalarBlock),
      0,
      stream,
      a,
      b,
      x_scale,
      weight_scale,
      static_cast<hip_bfloat16*>(out),
      m,
      n,
      k,
      b_transposed);
}

extern "C" void launch_pack_w8a8_weight(
    const int8_t* raw_weight,
    const float* weight_scale,
    int8_t* packed_weight,
    float* packed_weight_scale,
    int k,
    int n,
    hipStream_t stream) {
  // Iteration 8 exact-shape pack: for (k,n)==(1024,4096) write the [N][K]
  // transpose P[n*K + k] = W[k*N + n], so the staged partial kernel reads
  // each block's 16 B columns as 16 contiguous 512-B segments and every B
  // fragment as one 8-byte ds_read_b64. Runs once per weight outside the
  // timed region/Graph. All other shapes keep the identity layout
  // (row-major [K][N]) consumed by the DUMMA and scalar kernels.
  const int64_t w_count = static_cast<int64_t>(k) * n;
  const int64_t s_count = n;
  if (w_count > 0) {
    if (k == kExactK && n == kExactN) {
      const int kchunks = (k + 15) >> 4;
      const int64_t chunks = static_cast<int64_t>(n) * kchunks;
      hipLaunchKernelGGL(
          HIP_KERNEL_NAME(w8a8_pack_transpose_i8_kernel),
          dim3(blocks_for(chunks, kPackBlock)),
          dim3(kPackBlock),
          0,
          stream,
          raw_weight,
          packed_weight,
          k,
          n);
    } else {
      hipLaunchKernelGGL(
          HIP_KERNEL_NAME(w8a8_identity_copy_kernel<int8_t>),
          dim3(blocks_for(w_count, kPackBlock)),
          dim3(kPackBlock),
          0,
          stream,
          raw_weight,
          packed_weight,
          w_count);
    }
  }
  hipLaunchKernelGGL(
      HIP_KERNEL_NAME(w8a8_identity_copy_kernel<float>),
      dim3(blocks_for(s_count, kPackBlock)),
      dim3(kPackBlock),
      0,
      stream,
      weight_scale,
      packed_weight_scale,
      s_count);
}
// @@end
