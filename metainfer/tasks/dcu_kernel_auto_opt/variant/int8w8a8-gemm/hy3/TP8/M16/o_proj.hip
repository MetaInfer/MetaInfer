// @@variant shape=hy3_tp8_o_proj_m16 commit=4ddcb92e2032eb19fef17c7d4e19ac6739780ba9 added=2026-08-27
//   median_us=12.43 p90_us=12.59
//   source=hy3-dsh-tp8-m16-2-dc53295a
// MetaInfer W8A8 INT8 GEMM HIP implementation for gfx928 (worker_1).
//
// Iteration 4: bounded LDS prefetch pipeline for the assigned shape
// hy3_tp8_o_proj_m16 (M=16, N=4096, K=1024).
//   * Iteration 1 (accepted): 256 blocks x 1 wavefront x 1 16x16 N tile,
//     direct global fragment loads, 32 K-steps; official median
//     97.3785 us / p90 99.7009 us.
//   * Iteration 2 (rejected): 128 blocks x 2 wavefronts x 2 N tiles,
//     median 168.109 us / p90 177.434 us -- intra-block wave pairing with
//     the direct byte-load B path regressed, so launch-geometry co-
//     residency alone is not the lever.
//   * Iteration 3 (accepted official best): split-K=2 pipeline with int32
//     partials in the caller workspace -- (1)
//     w8a8_dumma_m16_n16_splitk_partial_kernel<2>: grid = 2*256 = 512
//     one-wave blocks (one 16x16 N tile per block; split s owns the K-half
//     [s*512, s*512+512), 16 ascending m16n16k32 steps with the
//     iteration-1 direct-load double-buffered K loop), each block storing
//     its int32 tile to workspace plane s via the API LDS plane
//     (mem_row_major); (2) w8a8_dumma_m16_combine_kernel<2>: grid = 256
//     one-wave blocks summing the planes in ascending split order and
//     applying the exact reference fp32 scale/bf16 chain. Official median
//     31.446 us / p90 31.509 us (beats the fixed Triton Graph baseline
//     32.137/35.969 us).
//   * This round (mandated architecture/pipeline axis: bounded
//     register/LDS prefetch): the exact code object of the accepted
//     iteration-3 partial kernel shows per K-step ONE global_load_dwordx2
//     per lane for A (8 B, immediate K-offsets -- already vectorized and
//     L2-hot) but EIGHT per-lane global_load_ubyte for B (one byte per
//     k-row, k-rows strided by N=4096) with s_waitcnt vmcnt(N) interleaved
//     with the byte reassembly -- 16 dwordx2 + 128 ubyte + 150 waits + 16
//     mmac per block, i.e. ~9 serialized global-latency rounds per K-step.
//     A-only staging would remove only ~1 of those 9 rounds, and
//     multi-N-tile reuse would add 8 B-ubyte rounds per extra tile per
//     step (per-step chain grows while blocks/CU halves), so the load path
//     itself is replaced with a BOUNDED depth-1 LDS prefetch pipeline: the
//     partial kernel keeps the exact split-K=2 geometry (512 one-wave
//     blocks = 4.27/CU, zero barriers, same combine kernel, same workspace
//     planes, same ascending-K int32 order) and only the B load path
//     changes. Each block stages its 8-KiB B tile slice into a
//     double-buffered LDS window (2 x 64 rows x 48-B padded stride; one
//     window = 2 m16n16k32 steps; 8 windows for kLen=512) with ONE 16-B
//     global_load_dwordx4 per lane per window, issued one full window
//     ahead of consumption; the K loop then reads B fragments from LDS
//     (ds_read_ubyte x8 + reassembly, ~30-cycle LDS latency, no vmcnt)
//     while A stays on its proven single-dwordx2 direct path. Byte reuse:
//     A (8 KiB/block slice; 16 KiB total, L2-hot) is re-read per K-step
//     exactly as before; B (8 KiB per block) is read from global exactly
//     once per block through the vectorized prefetch and every staged byte
//     is consumed by exactly one m16n16k32 step (B is inherently
//     single-use per block -- the change converts 128 scattered
//     ubyte-load+wait rounds into 8 vectorized prefetch rounds plus LDS
//     reads).
//   * Falsifiable gate: median_us < 31.446 AND p90_us < 31.509 (strict
//     improvement over the iteration-3 official best with the p90 noise
//     guard); expected effect ~2x (partial kernel ~3-4x faster, combine
//     ~2-3 us unchanged).
//   * The CU-aligned non-power-of-two candidate (SPLIT_K=3, grid = 768
//     blocks = 6.4 blocks/CU, non-uniform 32-aligned K slices 384/384/256)
//     remains implemented in the same templated partial+combine pair and
//     explicitly instantiated; this round launches SPLIT_K=2 as the
//     primary candidate.
//   * Workspace contract unchanged: partials occupy SPLIT_K * 65536 int32
//     planes (512 KiB for S=2) and are overwritten on every launch; stream
//     order makes combine read the planes written by the same replay.
//     Graph capture/replay safe: two kernel launches, no allocation, no
//     sync, no host reads, no default stream.
//   * Iteration 6 (this round): the exact code object of the accepted
//     iteration-4 partial kernel shows the B fragment is still reassembled
//     per m16n16k32 step from EIGHT per-lane ds_read_u8 (k-rows strided by
//     the 48-B padded window stride, one byte per k-row) with ~8
//     s_waitcnt lgkmcnt + ~15 VALU byte-reassembly ops per step, while A is
//     already one dwordx2 per lane per step and software-pipelined 4-6
//     steps ahead by the compiler (offsets 128..288 issued early). The
//     remaining per-step serial chain is therefore the B-from-LDS byte
//     path, so this round replaces the B data path with the validated
//     gate_up/down_proj recipe (int8-w8a8-gemm-decode skill): pack the
//     exact-shape weight OUT of the timed region into n-major fragment
//     layout packed[n*K + k] = raw[k*N + n] (same 4 MiB byte count, same
//     buffer, graph-stable addresses; identity pack retained for every
//     other (K, N)), keep the depth-1 double-buffered LDS window pipeline,
//     but store each window n-major (16 n-cols x 64 k-rows, 80-B padded
//     k-stride = 64+16, 16-B aligned non-power-of-two bank skew) so the
//     col_major B fragment load (du_mma.hpp: lane l reads
//     p[(l&15)*ldm + (l>>4)*8 + i], i=0..7 -- 8 CONTIGUOUS bytes) becomes
//     one 8-B vectorized read per lane per step (emitted as two
//     ds_read_b32) with no reassembly. Stage-in stays
//     one 16-B global_load_dwordx4 per lane per window (now from the packed
//     n-major buffer), issued one window ahead of consumption. A path,
//     split-K=2 geometry (512 + 256 one-wave blocks), combine kernel,
//     workspace planes, ascending-K exact int32 order, and zero-barrier
//     structure are held fixed; the scalar fallback decodes the n-major
//     pack for (k,n)==(1024,4096) (keeps the paired M=2 API shape exact).
//   * Iteration 7 (rejected, reverted): single-buffering leg of the
//     mandated double-vs-single buffer comparison -- all 8 B windows staged
//     in a prologue burst (LDS 11264 B, zero B-related vmcnt in the K loop,
//     525 -> 371 instructions). Median 14.7632 / p90 14.7728 us vs the
//     iteration-6 gate 14.7233/14.7345 -> NOT accepted. Conclusion: the B
//     stage-in latency was not the binding constraint; the residual
//     per-wave serial latency is the direct A-path global loads waited
//     inside the K loop (vmcnt(0..6) pipelined waits) with only 1.07
//     waves/SIMD (4.27 one-wave blocks/CU) to overlap them.
//   * Iteration 8 (this round, mandated HIP-only occupancy probe): PMC for
//     the accepted iteration-6 build (arch_vgpr 40, sgpr 16, LDS 3584 B,
//     scratch 0, 512 blocks, partial kernel 12.8 us) shows LDS allows 18
//     blocks/CU and VGPR 24 blocks/CU with zero spills -- neither resource
//     binds at 4.27 blocks/CU, so the only occupancy limiter is the grid
//     (total one-wave blocks). This round flips ONLY the split factor to
//     SPLIT_K=4: 1024 one-wave partial blocks = 8.53 blocks/CU = 2.13
//     wavefronts/SIMD (first candidate in the trusted [2..8] probe set that
//     crosses 2 waves/SIMD), uniform 64-window-aligned K slices 256/256/
//     256/256 (numWindows = 4), all blocks resident (LDS 18.3 >= 8.53,
//     VGPR 24 >= 8.53). The kernel body is bit-identical to the accepted
//     iteration-6 code: depth-1 double-buffered LDS window pipeline
//     (b_lds[2][1280], one 16-B global_load_dwordx4 per lane per window
//     issued one window ahead), col_major B fragment (one 8-B vectorized
//     LDS read per lane per step), direct A dwordx2 path, zero barriers,
//     same combine/scale chain, same n-major pack, same fallbacks. A and B
//     per-replay HBM totals stay constant (A: 1024 x 4 KiB, B: 1024 x
//     4 KiB); only the combine kernel now sums 4 freshly-written
//     L2-resident planes (+512 KiB reads, no repeated A/B HBM reads).
//     Workspace: 4 planes x 256 KiB = 1 MiB <= the 16-plane (4 MiB) caller
//     budget; the launcher workspace guard keeps the scalar fallback for an
//     undersized workspace. Expected effect: the ~2x resident waves overlap
//     the A-load global stalls (and the 4 instead of 8 per-block stage-in
//     rounds halve the per-wave exposed-latency chain), partial kernel
//     ~12.8 us -> ~7-10 us, combine ~2 us -> ~2.5-3.5 us, total ~10-13 us.
//     Falsifiable gate: median_us < 14.72327470779419 AND p90_us <
//     14.734469652175903 (strict improvement over the iteration-6 official
//     best with the p90 noise guard). If the median stays ~14.7 us, the
//     kernel is not occupancy-limited on this axis and the next round must
//     attack the A-path itself (LDS-stage A, or A layout) or probe SPLIT_K=8
//     (4.27 waves/SIMD, combine reads 2 MiB).
//   * Iteration 11 (rejected, reverted): SPLIT_K=8 occupancy probe (2048
//     one-wave partial blocks = 17.07 blocks/CU = 4.27 waves/SIMD, kernel
//     body bit-identical to iteration 8) regressed to 17.3119/17.3270 vs the
//     14.4528/14.4672 gate -- the occupancy axis is falsified (1.07 through
//     4.27 waves/SIMD all flat or worse; 2.13 is the peak). Per-kernel
//     profile of the accepted build: partial kernel 12.16 us + combine
//     kernel 2.88 us (profiled; operator 15.04 us), i.e. the combine is a
//     ~20% SERIAL second phase plus its launch gap.
//   * Iteration 12 (this round, HIP-only consolidation): fuse the combine
//     into the partial kernel with the qkv_proj-lineage per-tile
//     last-arrival protocol -- monotonic counters in the workspace tail
//     (256 x int32 after plane 3; one async zero per workspace pointer on
//     its first eager use, before capture), each block does one
//     __threadfence() + atomicAdd(&counters[tile], 1), and the arrival with
//     (arrived % SPLIT_K) == SPLIT_K - 1 sums the tile's 4 planes in
//     ascending split order and writes the scaled bf16 output with the exact
//     combine-kernel math (bit-identical bytes). The operator becomes ONE
//     launch per replay: the ~2.9-us serial combine phase and its launch gap
//     disappear, each tile's combine runs as soon as its 4 splits land
//     (overlapped with sibling blocks), and the plane reads happen while the
//     planes are L2-fresh. SPLIT_K=4 geometry, kernel body, n-major pack,
//     workspace planes, exact int32 order, and fallbacks are unchanged; the
//     two-kernel path remains for workspaces that fit the planes but not the
//     counters. Expected: partial kernel ~12 us (combine tail on the
//     last-arrivers only) + no second kernel -> total ~12-13 us. Falsifiable
//     gate: median_us < 14.452790021896362 AND p90_us < 14.467190504074097
//     (strict improvement over the iteration-8 official best with the p90
//     noise guard). If the median stays ~14.4 us, the serial combine phase
//     was not the binding cost and the next round must attack the partial
//     kernel's own issue chain (dual accumulator fragments) or geometry.
//   * Iteration 21 (this round, HIP-only): A-LDS double-buffered window
//     staging -- A is staged through the same depth-1 double-buffered LDS
//     window pipeline as B (a_lds[2][1280], one 16-B dwordx4 per lane per
//     window issued one window ahead) and the per-step a_frag loads read LDS
//     (ldm = 80) instead of global (ldm = 1024). Iteration 7 showed removing
//     every B-related vmcnt wait from the K loop does not move the time, and
//     identified the DIRECT A-PATH global loads waited inside the loop as the
//     residual per-wave serial latency; iteration 10 hoisted all A dwordx2
//     loads into a prologue burst and regressed (14.8639) because the 8
//     steps of A sat in VGPRs. Staging A in LDS (not VGPRs) keeps the
//     prefetch in registers only (the compiler hoists all 8 A/B window
//     dwordx4 prefetches into the prologue with progressive vmcnt waits;
//     arch_vgpr 40 -> 50, 2-3 waves/SIMD -- lockstep evidence says extra
//     co-residency does not change the serial-chain makespan), LDS 3588 ->
//     6148 B/block (64 KiB / 6.15 KiB = 10.4 blocks/CU >= 8.53 needed, grid
//     stays fully resident), and moves the A vmcnt wait out of the per-step
//     mmac chain to the top-of-window ds_write where it overlaps a full
//     compute window -- the exact recipe that took the B path from 31.45 to
//     19.01 us in iteration 4. Fragment values are byte-identical (same
//     global bytes staged, same col_major-style 8-B fragment mapping), so
//     the k-ascending int32 accumulation, the ascending-split sum, the fp32
//     scale/bf16 chain and every output byte are unchanged. Geometry
//     (SPLIT_K=4, 1024 one-wave blocks, one launch per replay), B path,
//     fused last-arrival counter protocol, workspace/planes/counters budget,
//     Graph safety, two-kernel fallback (counters = nullptr) and generic
//     scalar fallback are UNCHANGED. Expected: per-block serial latency
//     drops by the exposed A-load round trips (~8 waited dwordx2 -> 4
//     prologue-hidden dwordx4), operator median toward ~8-11 us.
//     Falsifiable gate: median_us < 13.084909915924072 AND p90_us <
//     13.115299940109253 (strict improvement over the iteration-12 official
//     best with the p90 noise guard). If the median stays ~13 us, the A path
//     is falsified as the binding cost and the residual is the fused
//     atomic/combine tail, irreducible in this protocol.
//
// The guarded exact-shape branch keeps every other (m, n, k) -- including the
// paired M=2 API shape with the same (N, K) -- on the generic scalar fallback
// (which decodes the n-major pack only for (k,n)==(1024,4096) and otherwise
// reads the row-major weight, matching the identity pack below).
// Later optimization rounds may replace only the exact-shape branch
// implementation (and the pack op) while this dispatch structure stays.

#include <hip/hip_runtime.h>
#include <hip/hip_bfloat16.h>
#include <du_mma.h>

#include <cstdint>

namespace {

// gfx928 wavefront is 64; block dimensions must be multiples of 64.
constexpr int kScalarBlockThreads = 256;
constexpr int kWaveSize = 64;

// Exact-shape DUMMA geometry for hy3_tp8_o_proj_m16 (M=16, N=4096, K=1024).
constexpr int kTargetM = 16;
constexpr int kTargetN = 4096;
constexpr int kTargetK = 1024;
constexpr int kDummaTileM = 16;
constexpr int kDummaTileN = 16;
constexpr int kDummaTileK = 32;

// Split-K planning for the exact shape: one 16x16 N tile per block means
// kNTiles = N/16 = 256 tiles; a split-K=S partial grid has S*kNTiles
// one-wave blocks. Every K boundary is aligned to the 32-element DUMMA step
// and to the 64-k-row LDS window (uniform 512/512 for S=2; uniform 256/256/
// 256/256 for S=4; non-uniform 384/384/256 for S=3).
constexpr int kNTiles = kTargetN / kDummaTileN;  // 256
constexpr int kPlaneInts = kTargetM * kTargetN;  // 65536

__device__ __forceinline__ constexpr int kSplitStart(int split_k, int s) {
  if (split_k == 2) return s * 512;
  if (split_k == 4) return s * 256;
  return s * 384;  // S=3
}
__device__ __forceinline__ constexpr int kSplitLen(int split_k, int s) {
  if (split_k == 2) return 512;
  if (split_k == 4) return 256;
  return (s < 2) ? 384 : 256;  // S=3
}

// ---------------------------------------------------------------------------
// Generic scalar int8 dot-product kernel: one thread per output element.
// Fallback for every (m, n, k) not handled by an exact-shape specialization.
// For the exact (k, n) == (1024, 4096) the packed weight buffer holds the
// n-major transpose packed[n*K + k] = raw[k*N + n] (iteration 6), so the
// fallback decodes that layout there (this keeps the paired M=2 API shape
// with the same (N, K) byte-exact); every other (k, n) keeps the identity
// pack (raw row-major [K, N]).
// ---------------------------------------------------------------------------
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
  const int64_t linear =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total = static_cast<int64_t>(m) * n;
  if (linear >= total) {
    return;
  }
  const int row = static_cast<int>(linear / n);
  const int col = static_cast<int>(linear - static_cast<int64_t>(row) * n);

  const int8_t* a_row = a + static_cast<int64_t>(row) * k;
  const bool packed_nmajor = (k == kTargetK && n == kTargetN);
  const int8_t* b_col = b + col;
  int32_t acc = 0;
  if (packed_nmajor) {
    // packed[col * 1024 + kk] == raw[kk * 4096 + col] (iteration-6 pack).
    const int8_t* b_pack_col = b + static_cast<int64_t>(col) * kTargetK;
    for (int kk = 0; kk < k; ++kk) {
      acc += static_cast<int32_t>(a_row[kk]) *
             static_cast<int32_t>(b_pack_col[kk]);
    }
  } else {
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
// Split-K partial kernel: one 64-thread wavefront per block, one 16x16 N
// tile per block, one K-slice per split. Iteration 6 replaces the B data
// path with the validated n-major packed + col_major fragment recipe: the
// exact-shape weight is packed (out of the timed region) as
// packed[n*K + k] = raw[k*N + n], and each block stages its B slice
// (kLen k-rows x 16 n-cols) into a double-buffered LDS window stored
// n-major (16 n-cols x 64 k-rows, 80-B padded k-stride) with ONE 16-B
// global_load_dwordx4 per lane per window, issued one full window (2
// m16n16k32 steps) ahead of consumption; the per-step K loop then reads B
// fragments from LDS with the col_major load (du_mma.hpp lane mapping:
// p[(l&15)*ldm + (l>>4)*8 + i], i=0..7), i.e. one 8-B vectorized read per
// lane per step -- emitted by the compiler as two ds_read_b32 (8 contiguous
// bytes) -- so the 8 ds_read_u8 + ~8 lgkmcnt + ~15 VALU byte-reassembly
// chain of iteration 4 is gone (validated on gate_up/down_proj in this
// lineage).
// Iteration 21: A takes the same depth-1 double-buffered LDS window path as
// B (a_lds[2][1280], one 16-B dwordx4 per lane per window issued one window
// ahead; per-step a_frag reads from LDS at ldm = kWindowKStride), so the
// direct A-path vmcnt wait leaves the per-step mmac chain (iteration 7
// identified those waited A loads as the residual per-wave serial latency;
// iteration 10's register hoist regressed on VGPR pressure -- LDS staging
// avoids it). Explicit int32 accumulation (max |dot| per slice =
// 16,516,096 << 2^31, k-ascending).
// Each block publishes its int32 tile through the API LDS plane
// (mem_row_major) and stores it to workspace plane `split` with one 16-B
// vector store per lane. Single wavefront -> no barrier anywhere (LDS
// store->read ordering is per-wave program order; buffer w&1 is stored at
// the top of iteration w and read only after that store). Guarded in the
// launcher: only this exact (m, n, k) reaches it.
// Iteration 12 (fused last-arrival combine, qkv_proj lineage recipe): with
// the counters pointer the tile's LAST arrival (monotonic per-tile counters
// in the workspace tail, (arrived % SPLIT_K) == SPLIT_K - 1) sums the
// tile's SPLIT_K planes in ascending split order and emits the scaled bf16
// output in-kernel, byte-identical to the separate combine kernel, so the
// operator becomes ONE launch per replay; counters = nullptr keeps the
// iteration-8 two-kernel behavior for undersized workspaces.
// ---------------------------------------------------------------------------
constexpr int kWindowKRows = 64;   // one window = 2 m16n16k32 steps
constexpr int kWindowNCols = 16;   // block tile width (n-cols per window)
constexpr int kWindowKStride = 80; // 64 k-rows + 16 pad; 16-B aligned,
                                   // non-power-of-two bank skew (<=2-way)
constexpr int kNumBStages = 2;     // double-buffered (depth-1 prefetch)

template <int SPLIT_K>
__global__ __launch_bounds__(kWaveSize) void
w8a8_dumma_m16_n16_splitk_partial_kernel(
    const int8_t* __restrict__ a,
    const int8_t* __restrict__ b,
    int32_t* __restrict__ partials,
    int32_t* __restrict__ counters,
    const float* __restrict__ x_scale,
    const float* __restrict__ weight_scale,
    hip_bfloat16* __restrict__ out) {
  static_assert(SPLIT_K == 2 || SPLIT_K == 3 || SPLIT_K == 4,
                "exact-shape split-K plan supports SPLIT_K=2, 3 or 4");
  const int lane = static_cast<int>(threadIdx.x);
  const int split = static_cast<int>(blockIdx.x) / kNTiles;
  const int tile = static_cast<int>(blockIdx.x) % kNTiles;
  const int n0 = tile * kDummaTileN;
  const int kStart = kSplitStart(SPLIT_K, split);
  const int kLen = kSplitLen(SPLIT_K, split);
  const int numWindows = kLen / kWindowKRows;  // 8 (S=2), 6/4 (S=3)

  __shared__ __align__(16) int8_t b_lds[kNumBStages][kWindowNCols *
                                                      kWindowKStride];
  // Iteration 21: A is staged through the same double-buffered LDS window
  // pipeline as B (16 m-rows x 64 k-rows per window, 80-B padded stride,
  // one 16-B dwordx4 per lane per window issued one window ahead), so the
  // per-step a_frag loads read LDS (ldm = kWindowKStride) instead of global
  // (ldm = kTargetK). The A-path vmcnt wait leaves the per-step mmac chain
  // and lands at the top-of-window ds_write alongside B's (iteration 7
  // identified the direct A-path global loads as the residual per-wave
  // serial latency; iteration 10's register hoist was rejected -- staging in
  // LDS, not VGPRs, keeps occupancy and the windowed depth-1 prefetch).
  __shared__ __align__(16) int8_t a_lds[kNumBStages][kWindowNCols *
                                                      kWindowKStride];

  du::dumma::DUFragment<du::dumma::matrix_a, kDummaTileM, kDummaTileN,
                        kDummaTileK, signed char, du::dumma::row_major>
      a_frag[2];
  du::dumma::DUFragment<du::dumma::matrix_b, kDummaTileM, kDummaTileN,
                        kDummaTileK, signed char, du::dumma::col_major>
      b_frag[2];
  du::dumma::DUFragment<du::dumma::accumulator, kDummaTileM, kDummaTileN,
                        kDummaTileK, int>
      acc_frag;
  du::dumma::du_fill_fragment(acc_frag, 0);

  // Depth-1 double-buffered LDS prefetch. Window w covers B k-rows
  // [kStart + w*64, kStart + w*64 + 64) at the block's 16 n-cols. The
  // packed buffer is n-major, so lane l fetches 16 CONTIGUOUS k-rows
  // (16 B, one dwordx4) at n-col (n0 + (l&15)) and k-offset
  // 16*(l>>4) within the window; the data for window w is issued one
  // iteration ahead (prologue for w=0) so the vmcnt wait lands at the
  // top-of-iteration ds_write_b128, a full compute window (~2 mmac steps)
  // after issue.
  const int8_t* b_packed_col =
      b + static_cast<int64_t>(n0 + (lane & 15)) * kTargetK;
  uint4 prefetch = *reinterpret_cast<const uint4*>(
      b_packed_col + kStart + 16 * (lane >> 4));
  // A prefetch mirrors B: lane l fetches 16 contiguous k-rows (16 B, one
  // dwordx4) of its A row (lane & 15) at k-offset 16*(lane >> 4) within the
  // window; every staged byte is consumed by exactly one m16n16k32 step via
  // the row_major col_major-style fragment (the 16-B chunk covers two
  // 8-B k-groups, exactly like the B window).
  const int8_t* a_row = a + static_cast<int64_t>(lane & 15) * kTargetK;
  uint4 a_prefetch = *reinterpret_cast<const uint4*>(
      a_row + kStart + 16 * (lane >> 4));
#pragma unroll
  for (int w = 0; w < numWindows; ++w) {
    // Store the previously-prefetched window w into buffer w&1 (its vmcnt
    // wait has had the prior iteration's compute to complete), then issue
    // the next window's prefetches before computing this window from LDS.
    *reinterpret_cast<uint4*>(
        &b_lds[w & 1][(lane & 15) * kWindowKStride + 16 * (lane >> 4)]) =
        prefetch;
    *reinterpret_cast<uint4*>(
        &a_lds[w & 1][(lane & 15) * kWindowKStride + 16 * (lane >> 4)]) =
        a_prefetch;
    if (w + 1 < numWindows) {
      prefetch = *reinterpret_cast<const uint4*>(
          b_packed_col + kStart + (w + 1) * kWindowKRows + 16 * (lane >> 4));
      a_prefetch = *reinterpret_cast<const uint4*>(
          a_row + kStart + (w + 1) * kWindowKRows + 16 * (lane >> 4));
    }

    const int8_t* bwin = &b_lds[w & 1][0];
    const int8_t* awin = &a_lds[w & 1][0];
    du::dumma::du_load_matrix_sync(a_frag[0], awin, kWindowKStride);
    du::dumma::du_load_matrix_sync(b_frag[0], bwin, kWindowKStride);
    du::dumma::du_load_matrix_sync(a_frag[1], awin + kDummaTileK,
                                   kWindowKStride);
    du::dumma::du_load_matrix_sync(b_frag[1], bwin + kDummaTileK,
                                   kWindowKStride);
    du::dumma::du_mma_sync(acc_frag, a_frag[0], b_frag[0], acc_frag);
    du::dumma::du_mma_sync(acc_frag, a_frag[1], b_frag[1], acc_frag);
  }

  // Publish the int32 tile through the API (mem_row_major), then one 16-B
  // vector store per lane. Plane layout is per-tile contiguous
  // workspace[split][tile][256] (e = row*16 + col), so each block's 1-KiB
  // int32 tile is one coalesced 64-lane x 16-B store. Single wavefront per
  // block: the LDS write->read dependency is ordered by the compiler's
  // lgkmcnt wait, no barrier needed.
  __shared__ __align__(16) int32_t tile_lds[kDummaTileM * kDummaTileN];
  du::dumma::du_store_matrix_sync(tile_lds, acc_frag, kDummaTileN,
                                  du::dumma::mem_row_major);

  const int e = lane * 4;
  const uint4 packed =
      make_uint4(tile_lds[e], tile_lds[e + 1], tile_lds[e + 2],
                 tile_lds[e + 3]);
  *reinterpret_cast<uint4*>(partials + split * kPlaneInts +
                            tile * (kDummaTileM * kDummaTileN) + e) = packed;

  // Fused last-arrival combine tail (iteration 12): when counters != nullptr
  // the separate combine kernel is removed and the operator is ONE launch per
  // replay. Every block signals its arrival for its tile with one lane-0
  // __threadfence() + atomicAdd(&counters[tile], 1); the arrival whose
  // pre-increment value satisfies (arrived % SPLIT_K) == SPLIT_K - 1 is the
  // tile's LAST arrival of this replay (counters are monotonic: each replay
  // adds exactly SPLIT_K to every tile counter) and it sums the tile's
  // SPLIT_K planes in ascending split order and writes the scaled bf16
  // output -- byte-identical to the separate combine kernel (same plane
  // chunk reads, same ascending int32 sum, same
  // float(acc) * x_scale[row] * weight_scale[col] -> __float2bfloat16 chain,
  // same store addresses). Each tile's combine now runs as soon as its 4
  // splits land, while sibling blocks are still computing, so the ~2.9-us
  // serial combine phase and its launch gap disappear and the plane reads
  // happen while the planes are L2-fresh (per-tile, immediately after the
  // tile's last write). Exact int32 accumulation unchanged: per-slice max
  // |dot| << 2^31, ascending-split sum order (same as the reference's
  // ascending-K order). The counters live in the workspace tail (last 256
  // int32 = 1 KiB, past plane SPLIT_K-1) and are zeroed once per workspace
  // pointer by the launcher before the first eager use (never inside Graph
  // capture); the two-kernel fallback passes counters = nullptr and the tail
  // branch is dead there.
  if (counters != nullptr) {
    __syncthreads();
    __shared__ int s_is_last;
    if (lane == 0) {
      __threadfence();  // release: this block's plane store is visible to the
                        // observer of this block's arrival atomic
      const int arrived = atomicAdd(&counters[tile], 1);
      __threadfence();  // acquire: the reads below (after the barrier) see
                        // every sibling plane store released before its
                        // arrival atomic
      s_is_last = ((arrived % SPLIT_K) == SPLIT_K - 1);
    }
    __syncthreads();
    if (s_is_last) {
      const int row = lane >> 2;         // 0..15
      const int cbase = (lane & 3) * 4;  // 0, 4, 8, 12 within the tile
      // 16-B aligned reads of the per-tile-contiguous plane layout
      // workspace[split][tile][256]; plane stride is kPlaneInts/4 uint4s.
      const uint4* plane =
          reinterpret_cast<const uint4*>(partials) +
          tile * (kDummaTileM * kDummaTileN / 4) + row * (kDummaTileN / 4) +
          (lane & 3);
      int32_t acc[4] = {0, 0, 0, 0};
#pragma unroll
      for (int s = 0; s < SPLIT_K; ++s) {
        const uint4 v = plane[s * (kPlaneInts / 4)];
        acc[0] += v.x;
        acc[1] += v.y;
        acc[2] += v.z;
        acc[3] += v.w;
      }
      const float xs = x_scale[row];
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const int col = n0 + cbase + i;
        const float scaled =
            static_cast<float>(acc[i]) * xs * weight_scale[col];
        out[static_cast<int64_t>(row) * kTargetN + col] =
            __float2bfloat16(scaled);
      }
    }
  }
}

// ---------------------------------------------------------------------------
// Combine+scale kernel: one 64-thread wavefront per block, one 16x16 N tile
// per block. Each lane loads its 4-element chunk from every split plane with
// one 16-B vector load per plane, sums the SPLIT_K planes in ascending split
// order (exact int32; ascending-split sum == the reference's ascending-K
// order), then applies the exact reference fp32 chain
// float(acc) * x_scale[row] * weight_scale[n0 + col] -> __float2bfloat16
// (RN). Same expression order as the reference -> bit-identical output for
// the same int32 dot. Zero barriers. Reads only planes written by the
// partial kernel of the same stream-ordered launch.
// ---------------------------------------------------------------------------
template <int SPLIT_K>
__global__ __launch_bounds__(kWaveSize) void w8a8_dumma_m16_combine_kernel(
    const int32_t* __restrict__ partials,
    const float* __restrict__ x_scale,
    const float* __restrict__ weight_scale,
    hip_bfloat16* __restrict__ out) {
  const int lane = static_cast<int>(threadIdx.x);
  const int n0 = static_cast<int>(blockIdx.x) * kDummaTileN;
  const int row = lane >> 2;         // 0..15
  const int cbase = (lane & 3) * 4;  // 0, 4, 8, 12 within the tile

  // 16-B aligned reads of the per-tile-contiguous plane layout
  // workspace[split][tile][256]; plane stride is kPlaneInts/4 uint4s.
  const uint4* plane =
      reinterpret_cast<const uint4*>(partials) +
      blockIdx.x * (kDummaTileM * kDummaTileN / 4) + row * (kDummaTileN / 4) +
      (lane & 3);
  int32_t acc[4] = {0, 0, 0, 0};
#pragma unroll
  for (int s = 0; s < SPLIT_K; ++s) {
    const uint4 v = plane[s * (kPlaneInts / 4)];
    acc[0] += v.x;
    acc[1] += v.y;
    acc[2] += v.z;
    acc[3] += v.w;
  }

  const float xs = x_scale[row];
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const int col = n0 + cbase + i;
    const float scaled = static_cast<float>(acc[i]) * xs * weight_scale[col];
    out[static_cast<int64_t>(row) * kTargetN + col] =
        __float2bfloat16(scaled);
  }
}

// ---------------------------------------------------------------------------
// Identity device-to-device packing: byte-for-byte copy. This is the generic
// fallback for any (K, N) and is valid for every shape.
// pack_weight runs outside the timed region and outside Graph capture.
// ---------------------------------------------------------------------------
__global__ __launch_bounds__(kScalarBlockThreads) void
w8a8_pack_identity_bytes_kernel(
    const uint8_t* __restrict__ src,
    uint8_t* __restrict__ dst,
    int64_t num_bytes) {
  const int64_t stride =
      static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       i < num_bytes; i += stride) {
    dst[i] = src[i];
  }
}

// ---------------------------------------------------------------------------
// Exact-shape n-major pack (iteration 6): for (K, N) == (1024, 4096) the
// weight is transposed one-time (outside the timed region / Graph) into
// packed[n * K + k] = raw[k * N + n], so each lane's 8-byte col_major B
// fragment (du_mma.hpp lane mapping) is contiguous and every m16n16k32
// fragment is a single vectorized load. Same byte count and buffer as the
// identity pack, so captured addresses are unchanged. One thread per output
// byte; runs once during weight prep.
// ---------------------------------------------------------------------------
__global__ __launch_bounds__(kScalarBlockThreads) void
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

// Explicit instantiations: the CU-aligned non-power-of-two candidate
// (SPLIT_K=3, grid = 3*kNTiles = 768 blocks = 6.4 blocks/CU, non-uniform
// 32-aligned K slices 384/384/256) and the SPLIT_K=4 occupancy candidate
// (grid = 4*kNTiles = 1024 blocks = 8.53 blocks/CU, uniform 64-window-aligned
// K slices 256/256/256/256) are compiled so a future round flips one
// constant; this round launches SPLIT_K=4 as the primary mandated candidate.
template __global__ void w8a8_dumma_m16_n16_splitk_partial_kernel<3>(
    const int8_t*, const int8_t*, int32_t*, int32_t*, const float*,
    const float*, hip_bfloat16*);
template __global__ void w8a8_dumma_m16_combine_kernel<3>(
    const int32_t*, const float*, const float*, hip_bfloat16*);
template __global__ void w8a8_dumma_m16_n16_splitk_partial_kernel<4>(
    const int8_t*, const int8_t*, int32_t*, int32_t*, const float*,
    const float*, hip_bfloat16*);
template __global__ void w8a8_dumma_m16_combine_kernel<4>(
    const int32_t*, const float*, const float*, hip_bfloat16*);

}  // namespace

// ---------------------------------------------------------------------------
// Host launchers (stable symbols consumed by csrc/bindings.cpp).
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
  hip_bfloat16* out_bf16 = static_cast<hip_bfloat16*>(out);

  // Guarded exact-shape dispatch: only hy3_tp8_o_proj_m16 (16, 4096, 1024)
  // reaches the split-K DUMMA pipeline. Every other (m, n, k) -- including
  // the paired M=2 API shape with the same (N, K) -- takes the generic
  // scalar fallback (which reads the row-major weight, matching the
  // identity pack).
  if (m == kTargetM && n == kTargetN && k == kTargetK) {
    // Split-K pipeline (iteration 8: occupancy probe -- SPLIT_K=4, grid =
    // 4*kNTiles = 1024 one-wave partial blocks = 8.53 blocks/CU = 2.13
    // wavefronts/SIMD. PMC evidence for the accepted iteration-6 SPLIT_K=2
    // build (arch_vgpr 40, sgpr 16, LDS 3584 B, scratch 0, 512 blocks) shows
    // LDS allows 18 blocks/CU and VGPR 24 blocks/CU -- neither binds at the
    // 4.27 blocks/CU needed, so the only occupancy limiter is the grid
    // itself (1.07 waves/SIMD). Iteration 7 proved removing every B-related
    // vmcnt wait from the K loop does not move the time (14.763 vs 14.723
    // us), i.e. the per-wave serial latency is the direct A-path global
    // loads waited inside the loop with no sibling wave to overlap them.
    // SPLIT_K=4 doubles resident waves to 2.13/SIMD (all 1024 blocks stay
    // resident: LDS 18.3 >= 8.53, VGPR 24 >= 8.53) while keeping the exact
    // iteration-6 depth-1 double-buffered window pipeline, A/B totals and
    // per-replay HBM traffic constant (only the combine reads 4 instead of
    // 2 freshly-written L2-resident planes, +512 KiB). The partial kernel
    // reads B from the n-major packed buffer through the col_major fragment
    // path (one 8-B vectorized LDS read per lane per step, staged via the
    // double-buffered LDS window prefetched one window ahead): int32
    // partials in the caller workspace, combine+scale kernel in the timed
    // Graph. Both launches are stream ordered inside the same captured
    // region; the partial kernel overwrites every partial element on each
    // launch (no workspace clear needed).
    constexpr int kSplitK = 4;
    constexpr int kPartialBlocks = kSplitK * kNTiles;  // 1024
    constexpr int kCombineBlocks = kNTiles;            // 256
    const int64_t plane_bytes =
        static_cast<int64_t>(kPlaneInts) * sizeof(int32_t);  // 256 KiB
    const int64_t counters_bytes =
        static_cast<int64_t>(kNTiles) * sizeof(int32_t);  // 1 KiB
    int32_t* partials = static_cast<int32_t*>(workspace);
    if (workspace_bytes >= kSplitK * plane_bytes + counters_bytes) {
      // Fused single-launch path (iteration 12): the combine runs inside the
      // partial kernel (per-tile last arrival), so the operator is ONE kernel
      // launch per replay and the separate combine kernel + its launch gap
      // disappear. The per-tile arrival counters (256 x int32 = 1 KiB) live
      // immediately after plane kSplitK-1 (the partial kernel writes planes
      // 0..kSplitK-1 only; the two-kernel fallback below never touches the
      // counters). Counters are monotonic -- each replay adds exactly SPLIT_K
      // to every tile counter -- so the last-arrival test is
      // (arrived % SPLIT_K) == SPLIT_K - 1 and NO per-replay reset is needed.
      // One async zero per workspace pointer is issued on the workspace's
      // first eager use (the Graph capture flow always warms up eagerly
      // first): the guarded hipMemsetAsync is a stream op on the caller's
      // stream, no host sync, and is NOT part of the captured graph -- the
      // qkv_proj lineage validated this exact protocol.
      int32_t* counters = partials + kSplitK * kPlaneInts;
      static const void* s_fused_counters_ws = nullptr;
      if (s_fused_counters_ws != workspace) {
        hipMemsetAsync(counters, 0, static_cast<size_t>(counters_bytes),
                       stream);
        s_fused_counters_ws = workspace;
      }
      hipLaunchKernelGGL(
          HIP_KERNEL_NAME(w8a8_dumma_m16_n16_splitk_partial_kernel<kSplitK>),
          dim3(kPartialBlocks),
          dim3(kWaveSize),
          0,
          stream,
          a,
          b,
          partials,
          counters,
          x_scale,
          weight_scale,
          out_bf16);
      return;
    }
    if (workspace_bytes >= kSplitK * plane_bytes) {
      // Two-kernel fallback for workspaces that fit the partial planes but
      // not the counters: partial kernel without the fused tail
      // (counters = nullptr -> the tail branch is dead) then the combine
      // kernel, byte-identical to the accepted iteration-8 behavior.
      hipLaunchKernelGGL(
          HIP_KERNEL_NAME(w8a8_dumma_m16_n16_splitk_partial_kernel<kSplitK>),
          dim3(kPartialBlocks),
          dim3(kWaveSize),
          0,
          stream,
          a,
          b,
          partials,
          static_cast<int32_t*>(nullptr),
          x_scale,
          weight_scale,
          out_bf16);
      hipLaunchKernelGGL(
          HIP_KERNEL_NAME(w8a8_dumma_m16_combine_kernel<kSplitK>),
          dim3(kCombineBlocks),
          dim3(kWaveSize),
          0,
          stream,
          partials,
          x_scale,
          weight_scale,
          out_bf16);
      return;
    }
    // Workspace smaller than the split-K=4 partial planes: fall through to
    // the generic scalar fallback (correct for every (m, n, k)); never run
    // the tuned pipeline with an undersized workspace.
  }

  const int64_t total = static_cast<int64_t>(m) * n;
  const unsigned blocks = static_cast<unsigned>(
      (total + kScalarBlockThreads - 1) / kScalarBlockThreads);
  hipLaunchKernelGGL(
      HIP_KERNEL_NAME(w8a8_scalar_gemm_kernel),
      dim3(blocks),
      dim3(kScalarBlockThreads),
      0,
      stream,
      a,
      b,
      x_scale,
      weight_scale,
      out_bf16,
      m,
      n,
      k);
}

extern "C" void launch_pack_w8a8_weight(
    const int8_t* raw_weight,
    const float* weight_scale,
    int8_t* packed_weight,
    float* packed_weight_scale,
    int k,
    int n,
    hipStream_t stream) {
  const int64_t weight_bytes = static_cast<int64_t>(k) * n;
  const int64_t scale_bytes = static_cast<int64_t>(n) * sizeof(float);
  constexpr int kPackThreads = kScalarBlockThreads;

  // Exact-shape n-major pack (iteration 6): only (k, n) == (1024, 4096)
  // gets the transposed fragment layout consumed by the tuned partial
  // kernel; every other (K, N) keeps the byte-identical identity pack so
  // the generic scalar fallback stays correct for all other shapes.
  if (k == kTargetK && n == kTargetN) {
    const unsigned pack_blocks = static_cast<unsigned>(
        (weight_bytes + kPackThreads - 1) / kPackThreads);
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(w8a8_pack_nmajor_bytes_kernel),
        dim3(pack_blocks),
        dim3(kPackThreads),
        0,
        stream,
        raw_weight,
        packed_weight,
        k,
        n);
  } else {
    const unsigned weight_blocks = static_cast<unsigned>(
        (weight_bytes + kPackThreads - 1) / kPackThreads);
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(w8a8_pack_identity_bytes_kernel),
        dim3(weight_blocks),
        dim3(kPackThreads),
        0,
        stream,
        reinterpret_cast<const uint8_t*>(raw_weight),
        reinterpret_cast<uint8_t*>(packed_weight),
        weight_bytes);
  }

  const unsigned scale_blocks = static_cast<unsigned>(
      (scale_bytes + kPackThreads - 1) / kPackThreads);
  hipLaunchKernelGGL(
      HIP_KERNEL_NAME(w8a8_pack_identity_bytes_kernel),
      dim3(scale_blocks),
      dim3(kPackThreads),
      0,
      stream,
      reinterpret_cast<const uint8_t*>(weight_scale),
      reinterpret_cast<uint8_t*>(packed_weight_scale),
      scale_bytes);
}
// @@end
