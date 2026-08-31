// @@variant shape=glm_tp8_q_b_proj_m16 commit=8a9a28e409277a6f266a59e0c053a3d2bb516f12 added=2026-08-31
//   median_us=14.12 p90_us=14.14
//   source=glm5-2-dsh-tp8-m16-test1-cb4d262c
// csrc/w8a8_gemm_hip.hip
//
// Worker_1 bootstrap (iteration 1) for the GLM5.2 TP8 M=16 decode shapes on
// Hygon K500SM_AI / gfx928 (wavefront = 64, 64 KiB LDS per CU):
//
//   glm_tp8_q_b_proj_m16  : M=16, N=2048, K=2048
//   glm_tp8_kv_b_proj_m16 : M=16, N=3584, K=512
//
// Logical operation (frozen contract, see int8_w8a8_gemm_api.py):
//   out[m, n] = bf16( int32_dot(x_q[m, :], weight[:, n])
//                     * x_scale[m, 0] * weight_scale[n, 0] )
//
// Strategy: assigned M=16 shapes run gfx928 DUMMA INT8 fast paths
// (m16n16k32, exact int32 accumulation). q_b (K=2048, N=2048) runs the
// grid-level split-K partial kernel: one 64-thread wavefront per (tile,
// split) block, the 32-aligned B K slice is staged once into LDS with
// vectorized 16-B bulk loads, one staging barrier, then a zero-barrier K
// loop whose B fragments come from LDS and whose A fragments are loaded
// straight from the L2-resident row-major A (8 contiguous bytes per lane per
// k32 step, 2-deep software pipeline; iteration 7 deepened the pipeline
// because the exact iteration-5 code object showed the 1-deep source
// degrading to an issue-mid-body / wait-same-body schedule that still
// exposed most of the L2 latency per step, and iteration 6 proved the LDS
// side of the body is not the critical path). Iteration 8 (HIP-only
// occupancy round): SPLIT_K 3 -> 8 (128 x 8 = 1,024 blocks = 8.53 blocks/CU
// = 2.13 waves/SIMD on the 120-CU device, uniform 256-K slices, LDS 11,264 ->
// 4,096 B/block -> 16 resident blocks/CU); each block publishes its exact
// int32 partial to a caller workspace plane and a separate combine+scale
// kernel (in the timed Graph) sums the split planes in ascending order and
// emits scaled bf16. Iteration 9's batched-staging HIP change (all four
// global_load_dwordx4 -> ds_write_b128 chains collapsed into one L2 round
// trip, verified in its exact code object) regressed 21.191 -> 22.033 us, so
// the staging latency is hidden by the 2.13 co-resident waves/SIMD and the
// partial kernel is at its HIP floor. Iteration 10 (this code) rewrites the
// combine+scale kernel as a single-pass int4-vectorized wavefront (one drain
// instead of four serialized load->wait->add->store chains; vmem_read 4,736
// -> ~1,280 per replay) with bit-identical int32 sums and float rounding.
// kv_b (K=512, N=3584) keeps the one-wave direct kernel.
// Iteration 17 (this code): one-time tile-major B pack for (k, n) ==
// (2048, 2048) (P[(t*K + kk)*16 + col] = B[kk][t*16 + col], out of the timed
// region and the Graph). The exact current-best code object + PMC show the
// partial kernel is memory-service-bound: the B staging loop's 64-lane 16-B
// chunks are scattered over 64 different rows' 128-B sectors (262,144 sector
// requests/replay, 16 B used per request) and each sector is shared by 8
// tile-blocks whose accesses are spread over the whole kernel, so evicted
// sectors are re-fetched (PMC l2_misses 83,350 ~= 2.5x the 32,768 compulsory
// B sectors; ~10.7 MB/replay at ~660 GB/s effective). The pack makes every
// staging iteration one contiguous 1,024-B chunk (8 fully-consumed sectors,
// one consumer block, no re-fetch possible): B sector requests drop 262,144
// -> 32,768 and B DRAM traffic drops toward the 4.19 MB compulsory set. The
// staged bytes, lds_b, the single barrier, the hoisted uniform A path, the
// k-ascending v_mmac body, the exact int32 partials, the combine kernel, the
// 1,024-block grid and the 2-dispatch Graph are unchanged -> outputs stay
// bit-identical to iteration 11 (0 mismatches). The (n, k) == (2048, 2048)
// generic scalar fallback (paired M=2 validation shape) decodes the pack via
// the b_tilemajor flag; every other shape keeps the identity pack and the
// logical decode.
// Iteration 18 (HIP-only consolidation; final conditional inline-asm round,
// raw asm still gated): the uniform-path B staging loop's 4 chains are
// batched into one in-flight group (4 named int4 loads back-to-back, one
// drain, 4 ds_write_b128) - the exact current-best code object shows the
// rolled loop still executes one load in flight per block (4 serialized
// cold-DRAM round trips), and the pack made the 4 chunks contiguous so the
// batch is DRAM-row-friendly (iteration 9's pre-pack scatter regression does
// not apply). Named registers avoid iteration 14's scratch spill. Staged
// bytes, barrier, body, partials and combine are unchanged -> outputs stay
// bit-identical to iteration 17.
// The generic scalar kernel remains the fallback for every unmatched (m, n,
// k), including the paired M=2 API shapes that share (N, K) with the
// assigned M=16 shapes.
//
// Graph-safety: gemm_out only launches kernels on the caller-provided stream
// (PyTorch's current HIP stream); it performs no allocation, no compilation,
// no autotuning, no packing, and no host/device synchronization, and touches
// only the caller-provided `out` and `workspace` (split-K partial planes).

#include <hip/hip_runtime.h>
#include <hip/hip_bfloat16.h>
#include <du_mma.h>

#include <cstdint>

namespace {

// blockDim must be a multiple of the gfx928 wavefront size (64).
constexpr int kScalarBlockThreads = 256;
constexpr int kPackBlockThreads = 256;

// gfx928 DUMMA INT8 tile geometry for the M=16 decode fast path: the
// installed DTK exposes m16n16k32 signed-char fragments with int32
// accumulation, and one 64-thread wavefront owns one independent tile.
constexpr int kDummaTileM = 16;
constexpr int kDummaTileN = 16;
constexpr int kDummaTileK = 32;
constexpr int kDummaWaveSize = 64;

using namespace du::dumma;

// One thread -> one output element. Adjacent threads map to adjacent columns
// (fastest-changing N dimension), so both the B row loads and the bf16 stores
// are coalesced; the A row load is broadcast across the threads of the same
// row.
__global__ __launch_bounds__(kScalarBlockThreads) void
w8a8_gemm_scalar_kernel(
    const int8_t* __restrict__ a,
    const int8_t* __restrict__ b,
    const float* __restrict__ x_scale,
    const float* __restrict__ weight_scale,
    __hip_bfloat16* __restrict__ out,
    int m,
    int n,
    int k,
    int b_tilemajor) {
  const int64_t linear =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total = static_cast<int64_t>(m) * n;
  if (linear >= total) {
    return;
  }
  const int row = static_cast<int>(linear / n);
  const int col = static_cast<int>(linear - static_cast<int64_t>(row) * n);

  const int8_t* a_row = a + static_cast<int64_t>(row) * k;
  // For (k, n) == (2048, 2048) the B buffer is the one-time tile-major pack
  // P[(t*K + kk)*16 + col_tile] = raw[kk][t*16 + col_tile] (t = col >> 4,
  // col_tile = col & 15), so the logical [K][N] column stride n becomes a
  // 16-B stride in kk; every other shape keeps the logical row-major decode.
  const int8_t* b_col = b_tilemajor
      ? b + (static_cast<int64_t>(col >> 4) * k * 16 + (col & 15))
      : b + col;
  const int64_t b_stride = b_tilemajor ? 16 : static_cast<int64_t>(n);

  // Exact int32 dot product over the complete K loop. The maximum assigned K
  // is 2048, so |dot| <= 2048 * 128 * 128 = 2^25 stays far below INT32_MAX.
  int32_t acc = 0;
  for (int kk = 0; kk < k; ++kk) {
    acc += static_cast<int32_t>(a_row[kk]) *
           static_cast<int32_t>(b_col[static_cast<int64_t>(kk) * b_stride]);
  }

  const float scaled =
      static_cast<float>(acc) * x_scale[row] * weight_scale[col];
  out[linear] = __float2bfloat16(scaled);
}

// Minimal M=16 DUMMA fast path: one wavefront (blockDim=64) computes one
// 16x16 output tile with a 16x16x32 signed-char fragment pair and exact int32
// accumulation, so there is no cross-wave barrier and no LDS staging in the K
// loop. A and B fragments are loaded directly from the caller row-major
// global tensors (A leading dimension K, B leading dimension N). The epilogue
// materializes the accumulator fragment through du_store_matrix_sync into a
// per-block LDS tile, applies the two float scales in the same order as the
// scalar path, and stores bf16.
__global__ __launch_bounds__(kDummaWaveSize) void w8a8_dumma_m16_direct_kernel(
    const int8_t* __restrict__ a,
    const int8_t* __restrict__ b,
    const float* __restrict__ x_scale,
    const float* __restrict__ weight_scale,
    __hip_bfloat16* __restrict__ out,
    int n,
    int k) {
  constexpr int kTileElems = kDummaTileM * kDummaTileN;  // 256
  const int lane = static_cast<int>(threadIdx.x);        // 0..63
  const int n0 = static_cast<int>(blockIdx.x) * kDummaTileN;

  __shared__ __align__(16) int32_t acc_tile[kTileElems];

  DUFragment<matrix_a, kDummaTileM, kDummaTileN, kDummaTileK, signed char,
             row_major>
      a_frag;
  DUFragment<matrix_b, kDummaTileM, kDummaTileN, kDummaTileK, signed char,
             row_major>
      b_frag;
  DUFragment<accumulator, kDummaTileM, kDummaTileN, kDummaTileK, int> acc_frag;
  du_fill_fragment(acc_frag, 0);

#pragma unroll 4
  for (int k0 = 0; k0 < k; k0 += kDummaTileK) {
    du_load_matrix_sync(a_frag, a + k0, k);
    du_load_matrix_sync(
        b_frag, b + static_cast<int64_t>(k0) * n + n0, n);
    du_mma_sync(acc_frag, a_frag, b_frag, acc_frag);
  }

  du_store_matrix_sync(acc_tile, acc_frag, kDummaTileN, mem_row_major);
  __syncthreads();

#pragma unroll
  for (int linear = lane; linear < kTileElems; linear += kDummaWaveSize) {
    const int row = linear / kDummaTileN;
    const int col = linear - row * kDummaTileN;
    const float scaled = static_cast<float>(acc_tile[linear]) *
                         x_scale[row] * weight_scale[n0 + col];
    out[row * n + n0 + col] = __float2bfloat16(scaled);
  }
}

// ---------------------------------------------------------------------------
// Iteration 3: grid-level split-K partial + combine (CU-aligned probe).
//
// Iteration 1 (one wavefront per 16x16 tile, direct global fragment loads)
// measured 63.695 us and iteration 2 (in-block split-K=2 with the same direct
// byte loads) regressed to 69.771 us: the 16 global_load_ubyte + vmcnt chain
// per m16n16k32 step (PMC: 131,712 vmem_reads) exposes full memory latency
// with ~1.07 blocks/CU and no in-loop overlap, i.e. the direct fragment
// loader is the poison, not the grid size. Iteration 3 replaced the q_b
// (16, 2048, 2048) path with a grid-level split-K partial kernel that stages
// each 32-aligned K slice into LDS once with vectorized 16-B bulk loads
// (one barrier), then runs a zero-barrier LDS-only K loop; every block is a
// single wavefront and the grid is 128 tiles x SPLIT_K blocks (SPLIT_K=3 ->
// 384 blocks = 3.2 blocks/CU, a non-power-of-two CU-aligned occupancy probe
// from the trusted set). Exact int32 partials go to caller workspace planes
// and the combine+scale kernel runs in the timed Graph.
//
// Iteration 4 (bounded staging prefetch, kStageDepth=4) regressed to 43.876
// us while keeping LDS at 22,784 B/block (2 resident blocks/CU), so the
// staging vmem chain was NOT the lever: the LDS-resident footprint itself is.
//
// Iteration 5 (this code): B-only staging. The exact iteration-3 code object
// shows the K loop's A fragment is 8 contiguous bytes per lane (one
// ds_read2_b32 in the staged kernel; 8 consecutive global_load_ubyte in the
// direct kernel) while the B fragment is 8 scattered 16-B-strided byte loads
// (16-way LDS bank conflicts). A is therefore moved out of LDS entirely:
// per k32 step each lane issues ONE 8-B global load from the L2-resident
// row-major A (1-deep software pipeline hides the L2 latency behind the
// current step's LDS loads + mma), the 11,520-B A tile disappears, LDS drops
// from 22,784 B to 11,264 B per block, and the resident blocks/CU rise from 2
// to 5. B keeps the exact iteration-3 staged layout and byte loads.
//
// Iteration 8 (this code): HIP-only occupancy round. PMC/ISA evidence pins
// the binding occupancy limiter: 24 VGPR / 32 SGPR / 11,264 B LDS / 0 scratch
// -> LDS is the limiter at 5 resident blocks/CU (64 KiB / 11.25 KiB), while
// the grid (384 = 3.2 blocks/CU = 0.8 waves/SIMD) leaves residency half
// empty and SQ_WAIT_INST_LDS = 36/replay (~0.09/block) plus the exact loop
// body's s_waitcnt vmcnt(0) right before v_mmac (and the staging loop's 11
// serialized global_load_dwordx4 -> vmcnt(0) -> ds_write_b128 chains) show
// the shaders stall on VMEM latency, not LDS. SPLIT_K is raised 3 -> 8: the
// trusted occupancy-probe split with uniform 256-K slices (64 k32 steps / 8
// = 0 remainder), grid 128 x 8 = 1,024 blocks = 8.53 blocks/CU = 2.13
// waves/SIMD, LDS 11,264 -> 4,096 B -> 16 resident blocks/CU (residency
// never binds, zero queueing, tail factor 1.055 vs 1.25). HBM is unchanged:
// every B byte is still staged exactly once (4.19 MB/replay) and the A
// request volume is invariant in S (each 32-KB L2-hot A is re-read per
// block, 4.19 MB/replay for any S); only the partial-plane writes grow
// 393,216 -> 1,048,576 B (8 planes <= the API 16-plane workspace for this
// shape).
// ---------------------------------------------------------------------------

// Compile-time non-uniform, 32-aligned K slicing for the guarded K=2048
// shape (64 k32 steps split into SPLIT_K contiguous slices; the remainder
// steps go to the first slices so every boundary stays a multiple of the
// DUMMA tile K=32, keeping each partial an exact k-ascending int32 sum).
template <int SPLIT_K>
struct KSplit2048 {
  static constexpr int kSteps = 2048 / kDummaTileK;  // 64
  static constexpr int kBase = kSteps / SPLIT_K;
  static constexpr int kRem = kSteps % SPLIT_K;
  static constexpr int kMaxSliceSteps = kBase + (kRem > 0 ? 1 : 0);
  static constexpr int kMaxSlice = kMaxSliceSteps * kDummaTileK;
  // Closed form (no C++14 constexpr loops): slice i contributes
  // kBase + (i < kRem ? 1 : 0) k32 steps.
  static constexpr int begin(int s) {
    return (s * kBase + (s < kRem ? s : kRem)) * kDummaTileK;
  }
  static constexpr int slice(int s) {
    return (kBase + (s < kRem ? 1 : 0)) * kDummaTileK;
  }
};

template <int SPLIT_K>
__global__ __launch_bounds__(kDummaWaveSize) void
w8a8_dumma_m16_splitk_partial_kernel(
    const int8_t* __restrict__ a,
    const int8_t* __restrict__ b,
    int* __restrict__ partials,
    int n,
    int k) {
  using Split = KSplit2048<SPLIT_K>;
  // B-only staging (iteration 5): only the B tile is staged into LDS. The A
  // fragment of this compiler/DUMMA header (verified in the exact iteration-3
  // code object, profiles/.../iteration4/current-best-isa) is 8 CONTIGUOUS
  // bytes per lane: the staged path reads them with one ds_read2_b32 at
  // (lane&15)*720 + (lane>>4)*8, and the direct kernel reads them with 8
  // consecutive global_load_ubyte at (lane&15)*k + (lane>>4)*8. A therefore
  // needs no LDS round trip: one 8-B global load per lane per k32 step from
  // the L2-resident row-major A reproduces the fragment exactly. Dropping the
  // 11,520-B A tile halves the LDS footprint per block (22,784 -> 11,264 B)
  // and raises the resident blocks/CU from 2 to 5 (64 KiB LDS / 11.25 KiB),
  // while the A load latency is hidden by a 2-deep software pipeline in the
  // K loop (iteration 7). The B tile keeps the exact iteration-3 layout and
  // byte loads. Iteration 8 (occupancy round) raises SPLIT_K to 8 via the
  // launcher only: kSlice becomes a uniform 256, LDS drops to 4,096 B/block
  // and the grid becomes 1,024 blocks; the body below is unchanged.
  constexpr int kLdsStrideB = kDummaTileN;  // 16

  const int lane = static_cast<int>(threadIdx.x);  // 0..63
  const int split = static_cast<int>(blockIdx.x) % SPLIT_K;
  const int tile = static_cast<int>(blockIdx.x) / SPLIT_K;
  const int n0 = tile * kDummaTileN;
  const int kBeg = Split::begin(split);
  const int kSlice = Split::slice(split);

  __shared__ __align__(16) int8_t lds_b[Split::kMaxSlice][kLdsStrideB];

  // A fragment lane mapping (verified against the iteration-3 code object):
  // lane l owns the 8 contiguous bytes of A row (l & 15) starting at column
  // ((l >> 4) << 3) inside the k32 window, with x[e] = byte at address + e.
  // Global row-major A (row stride k) yields one 8-B (int2) load per lane per
  // step, always 8-B aligned (k, kBeg, kDummaTileK are all multiples of 8).
  const int a_row = lane & 15;
  const int a_col0 = (lane >> 4) << 3;  // 0, 8, 16, 24
  const int8_t* a_lane =
      a + static_cast<int64_t>(a_row) * k + kBeg + a_col0;

  // Iteration 11 (HIP-only consolidation): the exact iteration-10 code object
  // (profiles/.../iteration11/current-best-isa, source digest
  // 5187c0a5a308822b9914dae2c086c979f0fcda9093fa20009c201b19597794e1)
  // compiles the 2-deep A rotation below into a schedule where the load for
  // step i+2 issues at the TAIL of body i (global_load_dwordx2 at 0x4374) and
  // the body opens with a full s_waitcnt vmcnt(0) (0x4324) that drains ALL
  // outstanding loads, so the load consumed by body i's mma has only ~1 body
  // (~60-120 cycles of issue) of lookahead against an L2 round trip of
  // ~250-350 cycles: every one of the 8 bodies still exposes most of one
  // A-load round trip (~1.6-2.4k cycles of the ~3.2-3.9k cycle per-block
  // path), and the ~2.1 co-resident waves/SIMD stall in phase (264,192 VALU
  // over 18.24 us profiled ~= 2% issue busy). Uniform slices (Split::kRem ==
  // 0, true for SPLIT_K=8) therefore take a compile-time hoisted path: all 8
  // int2 A loads issue back-to-back BEFORE the B staging loop, so the
  // staging chains' serialized DRAM waits (4 x global_load_dwordx4 ->
  // vmcnt(0) -> ds_write_b128 at 0x4190-0x41C0) give every A load
  // ~2.4-3.6k cycles of slack, and the K body is fully unrolled so the
  // compiler can overlap the independent B-fragment LDS assembly of later
  // steps with the serial v_mmac chain. The B-only staging, lds_b layout,
  // k-ascending mma order and exact int32 accumulation are unchanged, so the
  // partials stay bit-identical to iteration 10 (0 mismatches; Graph replay
  // with changed contents must pass). The generic kRem>0 path keeps the
  // original rotation unchanged.
  // a_buf is function-scoped so both branches can reference it; only the
  // uniform (kRem == 0) branch fills it, and only that branch's unrolled body
  // reads it (kMaxSliceSteps >= kBase covers every SPLIT_K instantiation).
  int2 a_buf[Split::kMaxSliceSteps];
  if (Split::kRem == 0) {
#pragma unroll
    for (int i = 0; i < Split::kBase; ++i) {
      a_buf[i] = *reinterpret_cast<const int2*>(a_lane + i * kDummaTileK);
    }
  }

  // Stage only B: chunk c covers 16 contiguous bytes = exactly one tile row.
  // Iteration 17 (HIP-only consolidation round; final conditional inline-asm
  // round, raw asm still gated: plateau=false, phase=hip_only,
  // raw_inline_asm_allowed=false, skill_allowed=false): the (k, n) ==
  // (2048, 2048) weight is now the one-time tile-major pack
  // P[(t*K + kk)*16 + col] = B[kk][t*16 + col] (t = tile index), so lane c
  // reads its 16 B from packed + ((t*K + kBeg + c)*16). The 64 lanes' chunks
  // of one staging iteration are CONTIGUOUS (one 1,024-B chunk = 8
  // fully-consumed 128-B sectors) instead of 64 scattered 16-B requests
  // spread over 64 different rows' sectors (row stride n = 2048), and each
  // sector has exactly ONE consumer block (no cross-tile sharing, so an
  // evicted sector is never re-fetched: PMC l2_misses 83,350 ~= 2.5x the
  // 32,768 compulsory B sectors). The staged bytes are identical
  // (B[kBeg+c][n0..n0+16)), so lds_b, the single staging barrier, the hoisted
  // uniform A path, the unrolled k-ascending v_mmac body and the exact int32
  // partials are unchanged (bit-identical to iteration 11).
  // Iteration 18 (HIP-only consolidation round; final conditional inline-asm
  // round, raw asm still gated: plateau=false, phase=hip_only,
  // raw_inline_asm_allowed=false, skill_allowed=false): the exact
  // current-best code object (profiles/.../iteration18/current-best-isa,
  // source digest 8b6bcaa866f26094d84762db45bd259a3a6492f900b42461e60c3f4264
  // 94853c) compiles the uniform-path staging loop below into a ROLLED loop
  // with ONE global load in flight per block (global_load_dwordx4 ->
  // s_waitcnt vmcnt(0) -> ds_write_b128 at 0x4A08-0x4A20), so every block
  // pays 4 SERIALIZED cold-DRAM round trips for its 4 KiB slice while the
  // 2.13 co-resident waves/SIMD stall in phase. Pre-pack, iteration 9 batched
  // the same chains and regressed (+4%) because the 4 chains were 256
  // SCATTERED sector requests (row thrash) and the co-resident waves covered
  // the drains; post-pack each chain is 8 contiguous sectors and the 4 chains
  // tile the slice CONTIGUOUSLY (4 x 1,024 B = one 4,096-B group), so issuing
  // all four per-lane chunk loads back-to-back (one in-flight group, one
  // drain) collapses 4 serialized round trips toward ~1 while keeping the
  // DRAM row-buffer-friendly. The four chunks are NAMED locals, not a
  // runtime-indexed array: iteration 14's `int4 b_chunk[kChunksPerLane]`
  // under a runtime `c < kSlice` trip bound spilled to scratch
  // (private_segment_fixed_size 80) and regressed to 36.675 us; named
  // registers with compile-time addressing keep the loads in VGPR (arch_vgpr
  // 40 -> ~44-56; occupancy stays grid-limited at 8.53 blocks/CU, LDS 16
  // resident blocks/CU unchanged, no scratch). The staged bytes, lds_b
  // addresses, single barrier, hoisted uniform A path, unrolled k-ascending
  // v_mmac body and exact int32 partials are unchanged, so outputs stay
  // bit-identical to iteration 17 (0 mismatches; Graph replay with changed
  // contents must pass). The generic kRem>0 path keeps the rolled loop
  // (runtime slice sizes); every other uniform instantiation falls back to it.
  const int64_t b_tile_base =
      static_cast<int64_t>(n0 / kDummaTileN) * k;
  if (Split::kRem == 0) {
    // kMaxSlice is constexpr and equals kSlice on this branch (kRem == 0).
    constexpr int kChunks = Split::kMaxSlice / kDummaWaveSize;  // 4 for SPLIT_K=8
    if (kChunks == 4) {
      const int8_t* b_lane =
          b + (b_tile_base + kBeg) * kDummaTileN + lane * kDummaTileN;
      const int4 b0 = *reinterpret_cast<const int4*>(b_lane);
      const int4 b1 = *reinterpret_cast<const int4*>(
          b_lane + kDummaWaveSize * kDummaTileN);
      const int4 b2 = *reinterpret_cast<const int4*>(
          b_lane + 2 * kDummaWaveSize * kDummaTileN);
      const int4 b3 = *reinterpret_cast<const int4*>(
          b_lane + 3 * kDummaWaveSize * kDummaTileN);
      *reinterpret_cast<int4*>(&lds_b[lane][0]) = b0;
      *reinterpret_cast<int4*>(&lds_b[lane + kDummaWaveSize][0]) = b1;
      *reinterpret_cast<int4*>(&lds_b[lane + 2 * kDummaWaveSize][0]) = b2;
      *reinterpret_cast<int4*>(&lds_b[lane + 3 * kDummaWaveSize][0]) = b3;
    } else {
      for (int c = lane; c < kSlice; c += kDummaWaveSize) {
        *reinterpret_cast<int4*>(&lds_b[c][0]) =
            *reinterpret_cast<const int4*>(
                b + (b_tile_base + (kBeg + c)) * kDummaTileN);
      }
    }
  } else {
    for (int c = lane; c < kSlice; c += kDummaWaveSize) {
      *reinterpret_cast<int4*>(&lds_b[c][0]) =
          *reinterpret_cast<const int4*>(
              b + (b_tile_base + (kBeg + c)) * kDummaTileN);
    }
  }
  __syncthreads();

  DUFragment<matrix_a, kDummaTileM, kDummaTileN, kDummaTileK, signed char,
             row_major>
      a_frag;
  DUFragment<matrix_b, kDummaTileM, kDummaTileN, kDummaTileK, signed char,
             row_major>
      b_frag;
  DUFragment<accumulator, kDummaTileM, kDummaTileN, kDummaTileK, int> acc_frag;
  du_fill_fragment(acc_frag, 0);

  if (Split::kRem == 0) {
    // Hoisted uniform path: a_buf holds all kSteps int2 A fragments, loaded
    // before the B staging (every load has >= the staging chains' serialized
    // DRAM waits of slack). The fully unrolled body keeps the exact
    // k-ascending mma order of the generic path.
#pragma unroll
    for (int i = 0; i < Split::kBase; ++i) {
      const int8_t* a_bytes = reinterpret_cast<const int8_t*>(&a_buf[i]);
#pragma unroll
      for (int e = 0; e < 8; ++e) {
        a_frag.x[e] = a_bytes[e];
      }
      const int k0 = i * kDummaTileK;
      du_load_matrix_sync(b_frag, lds_b[0] + k0 * kLdsStrideB, kLdsStrideB);
      du_mma_sync(acc_frag, a_frag, b_frag, acc_frag);
    }
  } else {
    // Zero-barrier K loop: exact k-ascending int32 accumulation over
    // [kBeg, kBeg + kSlice). B fragments come from LDS (unchanged layout). A
    // fragments come from global with a 2-deep software pipeline (iteration
    // 7): three int2 values (steps i, i+1, i+2) rotate through registers and
    // the load for step i+3 issues at the top of step i's body, so every A
    // load has ~1.5 step bodies (~400-550 cycles) to arrive in instead of the
    // ~half-body window the 1-deep source degraded to in the exact
    // iteration-5 code object (flat_load issued mid-body at 0x42C0,
    // s_waitcnt vmcnt(0) at the END of the SAME body at 0x4380 before the
    // rotation movs -> each step still exposes most of the ~350-450-cycle L2
    // latency). Iteration 6 proved the LDS side of the body is NOT the
    // critical path (one ds_read_b64 replacing the 8 conflicted ds_read_u8 +
    // ~12 VALU shuffles + lgkmcnt chain changed the median by 0.03%), so the
    // per-step serial cost is the A-load wait; this rotation attacks exactly
    // that. steps is runtime (e.g. 22/21/21 for SPLIT_K=3), so the loop is
    // not unrolled and the rotation is explicit; the guards keep every load
    // inside the A slice [kBeg, kBeg + kSlice).
    const int steps = kSlice / kDummaTileK;
    int2 a_cur = *reinterpret_cast<const int2*>(a_lane);  // step 0
    int2 a_next = a_cur;
    if (1 < steps) {
      a_next =
          *reinterpret_cast<const int2*>(a_lane + kDummaTileK);  // step 1
    }
    for (int i = 0; i < steps; ++i) {
      int2 a_nxt2 = a_next;
      if (i + 2 < steps) {
        a_nxt2 = *reinterpret_cast<const int2*>(
            a_lane + (i + 2) * kDummaTileK);
      }
      const int8_t* a_bytes = reinterpret_cast<const int8_t*>(&a_cur);
#pragma unroll
      for (int e = 0; e < 8; ++e) {
        a_frag.x[e] = a_bytes[e];
      }
      const int k0 = i * kDummaTileK;
      du_load_matrix_sync(b_frag, lds_b[0] + k0 * kLdsStrideB, kLdsStrideB);
      du_mma_sync(acc_frag, a_frag, b_frag, acc_frag);
      a_cur = a_next;
      a_next = a_nxt2;
    }
  }

  // Publish the exact int32 partial to caller workspace plane `split`
  // ([SPLIT_K][M][N] row-major). Every plane element is overwritten on every
  // launch, so no workspace clear is needed and the combine kernel below is
  // stream-ordered after this store.
  int* plane =
      partials + static_cast<int64_t>(split) * kDummaTileM * n + n0;
  du_store_matrix_sync(plane, acc_frag, n, mem_row_major);
}

// Combine+scale kernel: one wavefront per 16x16 output tile; sums the
// SPLIT_K int32 planes in ascending split order (bit-identical to the scalar
// k-ascending reference), applies x_scale * weight_scale in the scalar
// epilogue order, and stores bf16. Runs in the timed Graph after the partial
// kernel on the same stream.
//
// Iteration 10 (HIP-only consolidation round): the exact iteration-8 code
// object compiled the previous 4-iteration scalar loop (idx = lane, lane+64,
// lane+128, lane+192; 8 global_load_dword per iteration, one per split
// plane, at 0x464C-0x4694 of the combine symbol) into four SERIALIZED
// load -> wait-drain -> add -> store chains: each iteration's loads issue
// only after the previous iteration's bf16 store, so a single wave pays ~4 x
// (one DRAM round trip + add chain) while the 128-block grid delivers only
// 1.07 waves/CU (no second co-resident wave to overlap the drains; PMC:
// combine 5.12 us profiled, 4,736 vmem_read and 512 vmem_write per replay).
// This rewrite is ONE pass: lane l owns the contiguous 16-B chunk (row =
// l>>2, col4 = (l&3)<<2, 4 consecutive columns) of the tile, issues all
// SPLIT_K int4 plane loads back-to-back (a single drain, ~4x fewer memory
// instructions), accumulates the SAME ascending-s int32 sum per element,
// applies the same scalar epilogue expression order (float(dot) * x_scale *
// weight_scale), and stores the 4 bf16 as two dword stores. Per-element
// values, int32 accumulation order, and float rounding are identical to the
// previous combine, so the outputs stay bit-identical (0 mismatches) and the
// Graph structure (partial then combine on the caller stream) is unchanged.
template <int SPLIT_K>
__global__ __launch_bounds__(kDummaWaveSize) void
w8a8_dumma_m16_splitk_combine_kernel(
    const int* __restrict__ partials,
    const float* __restrict__ x_scale,
    const float* __restrict__ weight_scale,
    __hip_bfloat16* __restrict__ out,
    int n) {
  const int lane = static_cast<int>(threadIdx.x);
  const int n0 = static_cast<int>(blockIdx.x) * kDummaTileN;
  const int64_t plane = static_cast<int64_t>(kDummaTileM) * n;

  // lane -> one contiguous 16-B chunk of the 16x16 tile: row = lane/4, first
  // column = (lane%4)*4. 64 lanes x 4 elements = 256 = the full tile, exactly
  // once. All chunk addresses are 16-B aligned (n, n0, col4 and plane are all
  // multiples of 4 int32).
  const int row = lane >> 2;
  const int col4 = (lane & 3) << 2;
  const int* chunk =
      partials + static_cast<int64_t>(row) * n + n0 + col4;

  int32_t dot[4] = {0, 0, 0, 0};
#pragma unroll
  for (int s = 0; s < SPLIT_K; ++s) {
    const int4 p = *reinterpret_cast<const int4*>(chunk + s * plane);
    dot[0] += p.x;
    dot[1] += p.y;
    dot[2] += p.z;
    dot[3] += p.w;
  }

  const float xs = x_scale[row];
  union {
    float f[4];
    float4 v;
  } ws;
  ws.v = *reinterpret_cast<const float4*>(weight_scale + n0 + col4);
  union {
    __hip_bfloat16 b[4];
    uint32_t u[2];
  } packed;
#pragma unroll
  for (int e = 0; e < 4; ++e) {
    const float scaled = static_cast<float>(dot[e]) * xs * ws.f[e];
    packed.b[e] = __float2bfloat16(scaled);
  }
  uint32_t* out32 = reinterpret_cast<uint32_t*>(
      out + static_cast<int64_t>(row) * n + n0 + col4);
  out32[0] = packed.u[0];
  out32[1] = packed.u[1];
}

// Identity device-to-device copy used by the bootstrap pack op. Works for any
// element type and any (K, N): unmatched shapes keep this generic fallback.
template <typename T>
__global__ void w8a8_identity_copy_kernel(
    const T* __restrict__ src,
    T* __restrict__ dst,
    int64_t numel) {
  const int64_t i =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < numel) {
    dst[i] = src[i];
  }
}

// One-time tile-major B pack for the (k, n) == (2048, 2048) q_b weight,
// outside the timed region and the Graph: P[(t*K + k)*16 + col] =
// raw[k][t*16 + col], with t = n-tile index (0..N/16-1). The 16 B of one
// (tile, k-row) stay contiguous and the 64 rows of a staging iteration become
// one contiguous 1,024-B chunk, so the split-K partial kernel stages fully
// consumed 128-B sectors instead of 64 scattered 16-B requests (iteration 17;
// see the partial kernel's staging comment). Same 4,194,304 B, same bytes, so
// the packed buffer stays graph-stable and the API contract is unchanged; the
// (n, k) == (2048, 2048) generic scalar fallback decodes this layout via the
// b_tilemajor flag. Every other (k, n) keeps the identity pack.
__global__ void w8a8_pack_b_tilemajor_kernel(
    const int8_t* __restrict__ raw,
    int8_t* __restrict__ packed,
    int k,
    int n) {
  const int t = static_cast<int>(blockIdx.x);       // n-tile 0..n/16-1
  // First k-row of this thread's 16-row chunk: threadIdx.x in 0..127 ->
  // chunks [0,16), [16,32), ..., [2032,2048), covering every k-row exactly
  // once (128 threads x 16 rows = the full K=2048).
  const int k16 = static_cast<int>(threadIdx.x) * 16;
  const int64_t src =
      static_cast<int64_t>(k16) * n + static_cast<int64_t>(t) * 16;
  const int64_t dst = (static_cast<int64_t>(t) * k + k16) * 16;
#pragma unroll
  for (int r = 0; r < 16; ++r) {
    *reinterpret_cast<int4*>(packed + dst + r * 16) =
        *reinterpret_cast<const int4*>(raw + src + r * n);
  }
}

void launch_scalar_gemm(
    const int8_t* a,
    const int8_t* b,
    const float* x_scale,
    const float* weight_scale,
    __hip_bfloat16* out,
    int m,
    int n,
    int k,
    int b_tilemajor,
    hipStream_t stream) {
  const int64_t total = static_cast<int64_t>(m) * n;
  const dim3 grid(static_cast<unsigned>(
      (total + kScalarBlockThreads - 1) / kScalarBlockThreads));
  const dim3 block(kScalarBlockThreads);
  hipLaunchKernelGGL(
      w8a8_gemm_scalar_kernel,
      grid,
      block,
      0,
      stream,
      a,
      b,
      x_scale,
      weight_scale,
      out,
      m,
      n,
      k,
      b_tilemajor);
}

// One wavefront per 16x16 output tile. N must be divisible by 16 and K by 32
// (guaranteed by the exact-shape guards that reach this launcher).
void launch_dumma_m16_direct(
    const int8_t* a,
    const int8_t* b,
    const float* x_scale,
    const float* weight_scale,
    __hip_bfloat16* out,
    int n,
    int k,
    hipStream_t stream) {
  const dim3 grid(static_cast<unsigned>(n / kDummaTileN));
  const dim3 block(kDummaWaveSize);
  hipLaunchKernelGGL(
      w8a8_dumma_m16_direct_kernel,
      grid,
      block,
      0,
      stream,
      a,
      b,
      x_scale,
      weight_scale,
      out,
      n,
      k);
}

// Grid-level split-K partial kernel + separate combine+scale kernel (both in
// the timed Graph, on the caller stream). One wavefront per (tile, split)
// block; exact int32 partials live in the caller workspace planes, so the
// partial kernel needs no cross-block synchronization and the combine kernel
// sums the planes in ascending split order. Requires K = 2048 and N divisible
// by 16 (guaranteed by the exact-shape guard that reaches this launcher).
template <int SPLIT_K>
void launch_dumma_m16_splitk(
    const int8_t* a,
    const int8_t* b,
    const float* x_scale,
    const float* weight_scale,
    __hip_bfloat16* out,
    void* workspace,
    int n,
    int k,
    hipStream_t stream) {
  auto* partials = reinterpret_cast<int*>(workspace);
  const dim3 partial_grid(
      static_cast<unsigned>(n / kDummaTileN * SPLIT_K));
  const dim3 block(kDummaWaveSize);
  hipLaunchKernelGGL(
      HIP_KERNEL_NAME(w8a8_dumma_m16_splitk_partial_kernel<SPLIT_K>),
      partial_grid,
      block,
      0,
      stream,
      a,
      b,
      partials,
      n,
      k);
  const dim3 combine_grid(static_cast<unsigned>(n / kDummaTileN));
  hipLaunchKernelGGL(
      HIP_KERNEL_NAME(w8a8_dumma_m16_splitk_combine_kernel<SPLIT_K>),
      combine_grid,
      block,
      0,
      stream,
      partials,
      x_scale,
      weight_scale,
      out,
      n);
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
  // The workspace holds the split-K int32 partial planes for the q_b
  // grid-level split path; scalar/tail paths ignore it.
  auto* out_bf16 = reinterpret_cast<__hip_bfloat16*>(out);

  // Assigned M=16 shapes run the DUMMA fast paths. q_b (K=2048, N=2048) runs
  // the grid-level split-K=8 partial kernel + combine kernel (iteration 8:
  // 1,024 blocks = 8.53 blocks/CU = 2.13 waves/SIMD, uniform 256-K slices,
  // B-only staging at 4,096 B LDS/block = 16 resident blocks/CU, A fragments
  // loaded directly from the L2-resident row-major A, workspace int32
  // partials, combine in the timed Graph). Iteration 17: the (k, n) ==
  // (2048, 2048) weight is the one-time tile-major pack P[(t*K+kk)*16+col] =
  // B[kk][t*16+col] (see w8a8_pack_b_tilemajor_kernel), so the partial
  // kernel's staging reads contiguous fully-consumed sectors; the scalar
  // fallback for this (n, k) decodes the pack via b_tilemajor. kv_b
  // (K=512, N=3584) keeps the one-wave direct kernel and its identity pack.
  // Each guard pins the exact M=16 so the paired M=2 API shape with the same
  // (N, K) still reaches the generic scalar fallback below.
  if (m == 16 && n == 2048 && k == 2048) {
    // 8 int32 partial planes of [16][2048]; the caller workspace contract
    // allocates 16 planes for this shape (2,097,152 B), so the guard is
    // defensive only.
    constexpr int64_t kRequiredWorkspace =
        8LL * 16 * 2048 * static_cast<int64_t>(sizeof(int32_t));
    if (workspace != nullptr && workspace_bytes >= kRequiredWorkspace) {
      launch_dumma_m16_splitk<8>(
          a, b, x_scale, weight_scale, out_bf16, workspace, n, k, stream);
    } else {
      // Defensive-only: the packed B must be decoded, so the direct kernel
      // (which reads the logical [K][N] layout) is not usable here; the
      // packed scalar decode is always correct.
      launch_scalar_gemm(
          a, b, x_scale, weight_scale, out_bf16, m, n, k,
          /*b_tilemajor=*/1, stream);
    }
    return;
  }
  if (m == 16 && n == 3584 && k == 512) {
    launch_dumma_m16_direct(
        a, b, x_scale, weight_scale, out_bf16, n, k, stream);
    return;
  }

  // Generic scalar fallback for every unmatched (m, n, k). The (n, k) ==
  // (2048, 2048) pair (including the paired M=2 validation shape) decodes
  // the tile-major pack; every other shape reads the logical layout.
  launch_scalar_gemm(
      a, b, x_scale, weight_scale, out_bf16, m, n, k,
      (n == 2048 && k == 2048) ? 1 : 0, stream);
}

extern "C" void launch_pack_w8a8_weight(
    const int8_t* raw_weight,
    const float* weight_scale,
    int8_t* packed_weight,
    float* packed_weight_scale,
    int k,
    int n,
    hipStream_t stream) {
  // One-time packing outside the timed region and the Graph. The (k, n) ==
  // (2048, 2048) q_b weight is repacked tile-major
  // (P[(t*K+kk)*16+col] = raw[kk][t*16+col], iteration 17); the matching
  // GEMM interpretation lives in the split-K partial staging and the
  // b_tilemajor scalar decode. Every other (K, N) shape keeps the identity
  // copy so the logical [K, N] row-major layout is preserved.
  if (k == 2048 && n == 2048) {
    const dim3 pack_grid(static_cast<unsigned>(n / 16));
    hipLaunchKernelGGL(
        w8a8_pack_b_tilemajor_kernel,
        pack_grid,
        dim3(128),
        0,
        stream,
        raw_weight,
        packed_weight,
        k,
        n);
  } else {
    const int64_t weight_numel = static_cast<int64_t>(k) * n;
    const dim3 weight_grid(static_cast<unsigned>(
        (weight_numel + kPackBlockThreads - 1) / kPackBlockThreads));
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(w8a8_identity_copy_kernel<int8_t>),
        weight_grid,
        dim3(kPackBlockThreads),
        0,
        stream,
        raw_weight,
        packed_weight,
        weight_numel);
  }

  const int64_t scale_numel = static_cast<int64_t>(n);
  const dim3 scale_grid(static_cast<unsigned>(
      (scale_numel + kPackBlockThreads - 1) / kPackBlockThreads));
  hipLaunchKernelGGL(
      HIP_KERNEL_NAME(w8a8_identity_copy_kernel<float>),
      scale_grid,
      dim3(kPackBlockThreads),
      0,
      stream,
      weight_scale,
      packed_weight_scale,
      scale_numel);
}
// @@end
