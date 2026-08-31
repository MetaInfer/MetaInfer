// @@variant shape=hy3_tp4_shared_down_proj_m4096 commit=62971c1acacf5866bcbbadc375ee87334100b9c7 added=2026-08-23
//   median_us=214.8 p90_us=215.1 tops=59.97 bandwidth_gb_s=171 speedup=20.39 baseline_us=4381
//   source=hy3-tp4-m4096-dsh-test1-bfdf3002
// MetaInfer W8A8 INT8 GEMM backend for Hygon gfx928 (K500SM_AI).
//
// This file is owned by worker_3 and provides the two stable host launch
// symbols consumed by csrc/bindings.cpp:
//   * launch_w8a8_gemm(...)         - timed, graph-safe GEMM on the caller's
//                                     HIP stream; no allocation, packing,
//                                     autotuning, or synchronization.
//   * launch_pack_w8a8_weight(...)  - optional out-of-timed-region packing;
//                                     round 14: n-major transpose pack
//                                     (packed[n][k], K-contiguous per output
//                                     column); bootstrap was an identity copy.
//
// Large-M prefill strategy (2-D macro-tile family, native INT8 DUMMA
// m16n16k32):
//   * A single templated kernel computes a (kBlockM x kBlockN) output tile
//     from (kBlockM/32) x (kBlockN/32) wavefronts, each wave owning one
//     32x32 quadrant (four 16x16 int32 accumulators) and keeping its
//     accumulation resident across the whole K loop.
//   * Three macro-tiles are instantiated - 64x64, 64x128 and 128x64 - and
//     the active one is selected at compile time (kActiveTileIndex). All
//     three share the cooperative A+B LDS staging pipeline with 16-byte
//     padded LDS row strides (A stride kStageK+16, B stride kBlockN+16).
//     The padding keeps every LDS row start 16-byte aligned (vectorized
//     int4 cooperative staging) while breaking the 128-byte bank aliasing
//     that produced ~8-way LDS conflicts in the bootstrap (PMC:
//     29,884,416 conflicts over 3,735,552 LDS instructions); the padded
//     strides keep the accepted kernel's ~2-way fragment-load conflict
//     profile.
//   * Operand feed (iteration 2): direct du_load_matrix_sync global loads
//     are rejected for this shape (they would amplify L2 requests 32-64x
//     per unique element and degrade to byte-granular strided requests),
//     so the staged path is kept and made latency-tolerant: K is staged in
//     kPrefillStageK=32 chunks into a double-buffered LDS pair while each
//     stage's global int4 loads are issued a full stage early into VGPRs
//     (software prefetch). One barrier per stage, so the vmcnt(0) wait +
//     ds_write of stage s+1 overlap the MMACs of stage s instead of
//     serializing before them (iteration-1 single buffering exposed the
//     global latency and produced 6,885,116 LDS wait instructions over
//     3,686,400 LDS instructions). Footprint 17,408 B/block -> 3 blocks/CU
//     (24 resident wavefronts), an occupancy probe from the trusted set
//     [2, 3, 4, 5, 6] that fits the 64 KiB LDS and the stage-alignment
//     constraints (kPrefillGuardK % kPrefillStageK == 0, stage % 32 == 0,
//     stage % 16 == 0 for int4 staging).
//   * B fragment transport (iteration 5): the exact-source ISA shows the
//     matrix_b fragment path as byte-granular - 16 ds_read_u8 at k-row
//     stride kBStride=80, ~40 VALU of byte assembly (v_or3_b32,
//     v_lshlrev_b32, v_or_b32_sdwa) and ~5 s_waitcnt lgkmcnt(N) groups per
//     wave per stage feeding only 4 v_mmac_i32_16x16x32_i8 - while the
//     matrix_a path is already 2 vectorized ds_read2_b32 (8 contiguous
//     bytes per lane). du_load_matrix_sync(matrix_b, row_major) fetches
//     p[(col+i)*ldm + row] (8 bytes spread over 8 k-rows), so B is now
//     staged into LDS in a transposed n-major layout (b_tile[n][k], row
//     stride kBTStride = kStageK + 8 = 40, 8-byte-aligned K runs) and
//     consumed through col_major matrix_b fragments, whose loader fetches
//     p[row*ldm + col + i] = 8 contiguous bytes per lane and lowers to the
//     same vectorized ds_read2_b32 path as A (verified against
//     /opt/dtk/include/du_mma.hpp: x[i] maps to the same B[k0+col+i][n0+row]
//     elements in both layouts and du_mma_sync passes the fragment
//     registers through LayoutB-agnostically, so the v_mmac inputs - and
//     therefore the exact int32 accumulation - are bit-identical). Staging
//     cost moves from 1 ds_write_b128 to 16 bank-spread ds_write_b8 per
//     staging thread: each int4 covers 16 consecutive n columns and all
//     sixteen bytes are scattered into the transposed tile at
//     (16*seg+j)*kBTStride + kk, so every n column is written every stage
//     (~4-way write conflicts vs ~2-way before, but B fragment reads drop
//     16 -> 2 instructions per wave per stage with no byte assembly, ~3x
//     fewer LDS instructions and ~5x fewer LDS wait groups per stage). LDS
//     footprint unchanged at 17,408 B/block.
//   * Pipeline depth (iteration 7): kPrefillStageK raised 32 -> 64. The
//     probe keeps the double-buffered structure and the transposed-B
//     staging from iteration 5 but halves the barrier count (12 -> 6 stage
//     barriers + prologue) and doubles the in-flight global bytes. LDS
//     footprint moves 17,408 -> 29,696 B/block. The PMC launch record
//     showed the code object at 74 VGPRs (arch_vgpr 80), so residency is
//     actually 1 block/CU / 8 waves - the __launch_bounds__(512, 2) pin
//     was ignored (2 blocks/CU would need <= 64 VGPRs), and stage 128
//     (iteration 9) and 3 blocks/CU at 64x64 (iteration 10) both regressed.
//   * Intra-wave ILP (iteration 11): the 2-trip kk loop is fully unrolled
//     into two independent fragment register sets (all eight ds_read2 of a
//     stage issue before the MMACs), so the second step's LDS latency
//     overlaps the first step's swizzle + MMACs instead of serializing
//     behind them. ~8 extra VGPRs (74 -> ~82) keep residency at the
//     measured 1 block/CU / 8 waves; LDS, depth, barriers, and the launch
//     config are unchanged.
//   * Vectorized B staging (iteration 14): the B transpose moves out of the
//     timed kernel into launch_pack_w8a8_weight, which now emits an n-major
//     packed weight (packed[n][k]; the API contract allows any contiguous
//     opaque packed layout and the pack runs once per weight tensor outside
//     the timed region and outside Graph capture). The timed kernel therefore
//     stages each n row's k-contiguous run with two 8-byte ds_write_b64 per
//     thread (b_tile[n][kk] addresses and the col_major fragment loads are
//     untouched, so every LDS byte and every MMAC input is identical), and
//     the 16 ds_write_b8 + ~12 VALU byte extracts of stage_b_transposed per
//     staging thread disappear from the per-stage critical path. The removed
//     byte-extract register pressure (v25..v38 in the accepted ISA) is also
//     what kept the code object at 69 VGPRs (> 64, so the
//     __launch_bounds__(512, 2) minBlocks pin never held); with it gone the
//     first clean 2 blocks/CU probe at the constant 128x64 tile is expected
//     to materialize (LDS 2 * 29,696 = 59,392 B <= 64 KiB).
//   * Fused direct fragment -> scale -> bf16 epilogue; no separate epilogue
//     kernel, no split-K.
//   * All other (m, n, k), including M < 128: scalar int8/int32 fallback.
//
// Header order is fixed by the control plane: hip_runtime first (du_mma.h is
// not self-contained before it), hip_bfloat16 second, du_mma.h last.

#include <hip/hip_runtime.h>
#include <hip/hip_bfloat16.h>
#include <du_mma.h>

#include <cstdint>

namespace {

constexpr int kTileM = 16;
constexpr int kTileN = 16;
constexpr int kTileK = 32;
constexpr int kWaveSize = 64;

// Staging stage length for the prefill pipeline. 64 = 2x the DUMMA K unit
// (iteration 7): with the 128x64 macro-tile the double-buffered footprint
// is 2*(128*80 + 64*72) = 29,696 B (A stride 80, transposed-B stride 72),
// which fits the 64 KiB LDS either way. Actual residency is 1 block/CU /
// 8 waves, capped by the 74-VGPR code object (2 blocks/CU at 512 threads
// would need <= 64 VGPRs; the __launch_bounds__(512, 2) minBlocks pin was
// ignored). Stage 64 halves the per-block barrier count vs 32 (12 -> 6
// stage barriers) and doubles the software-prefetch distance (one full
// 64-k stage of global loads in flight instead of 32 k). Padded row strides
// stay aligned: kAStride 80 % 16 == 0 for int4 staging, kBTStride 72 % 8
// == 0 for the col_major 8-byte fragment reads; the launch guard
// (k % 128 == 0) still implies k % kPrefillStageK == 0.
constexpr int kPrefillStageK = 64;

// Launch-guard divisibility is kept at 128 (unchanged from iteration 1),
// which implies k % kPrefillStageK == 0 for every dispatched shape.
constexpr int kPrefillGuardK = 128;

// Compile-time selection of the active 2-D macro-tile:
//   0 -> 64x64, 1 -> 64x128, 2 -> 128x64.
// All three instantiations are compiled so the family stays measurable; only
// the active one is launched inside the timed region.
constexpr int kActiveTileIndex = 2;
constexpr int kActiveBlockM = kActiveTileIndex == 2 ? 128 : 64;
constexpr int kActiveBlockN = kActiveTileIndex == 1 ? 128 : 64;

// LDS padding in bytes added to each staged row. 16 keeps every row start
// 16-byte aligned for int4 staging while shifting fragment rows off the same
// 32-bank phase (128-byte strides alias every row onto identical banks).
constexpr int kLdsPad = 16;

// LDS row stride of the transposed (n-major) B tile: each row holds one
// output column n with kStageK contiguous K bytes. kLdsPadB = 8 keeps every
// row start 8-byte aligned so the col_major matrix_b fragment loader can
// lower its 8 contiguous bytes per lane to a single vectorized ds_read2_b32
// (like the A path) and so the round-14 staging can land each n row's
// k-contiguous 16-byte run as two 8-byte ds_write_b64 (72 % 8 == 0; note
// 72 % 16 == 8, so a single ds_write_b128 would be 16-byte misaligned for
// odd n rows - the explicit two-half store keeps the lowering legal).
constexpr int kLdsPadB = 8;
constexpr int kBTStride = kPrefillStageK + kLdsPadB;

using namespace du::dumma;

// gfx928 int8 m16n16k32 accumulator ownership, established against
// du_store_matrix_sync: lane % 16 selects the row, lane / 16 selects the
// column mod 4, and x[i] selects columns separated by four.
template <typename AccFragment>
__device__ __forceinline__ void store_prefill_fragment(
    const AccFragment& frag,
    const float* __restrict__ x_scale,
    const float* __restrict__ weight_scale,
    __hip_bfloat16* __restrict__ out,
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
    const float scaled =
        static_cast<float>(frag.x[i]) * xs * weight_scale[col];
    out[row * n + col] = __float2bfloat16(scaled);
  }
}

// B staging (iteration 14): launch_pack_w8a8_weight emits the weight packed
// n-major (packed[n][k], one contiguous K run per output column), so the
// timed kernel stages each 16-byte int4 run straight into the transposed
// LDS tile as two 8-byte ds_write_b64 (the B_tile row address
// n_local * kBTStride + kchunk is 8-byte aligned but not 16-byte aligned,
// so a single ds_write_b128 would be illegal for odd n_local; the explicit
// two-half store keeps the lowering legal). This replaces the old 16
// byte-granular ds_write_b8 + ~12 VALU byte extracts per staging thread of
// stage_b_transposed, whose (16*seg+j)*kBTStride + kk scatter wrote every n
// column every stage. The LDS tile layout, every staged byte, the col_major
// fragment reads, and the kk/acc ordering are unchanged, so the int32
// accumulation is bit-identical.

// ---------------------------------------------------------------------------
// Large-M prefill path. One block computes a kBlockM x kBlockN output tile
// with (kBlockM/32) x (kBlockN/32) wavefronts (one 32x32 quadrant per wave).
// A[kBlockM, K] and B[K, kBlockN] are cooperatively staged into a
// double-buffered, bank-padded LDS in kStageK chunks with VGPR prefetch:
// the global int4 loads for stage s+1 are issued before the MMAC sequence
// of stage s, and the alternate LDS buffer is written right after it, so
// the vmcnt(0) wait and ds_write overlap compute. One barrier per stage.
// A stays row-major in LDS (stride 80); B is staged transposed (n-major,
// stride 72, K-contiguous) and loaded through col_major matrix_b fragments
// so both operand fragment paths use vectorized 8-byte LDS reads (iteration
// 5; the iteration-2 B path was 16 byte-granular ds_read_u8 + ~40 VALU of
// assembly per wave per stage). Since iteration 14 the B transpose happens
// at pack time (launch_pack_w8a8_weight emits packed[n][k]), so the staged
// B runs land with two 8-byte ds_write_b64 per thread instead of the 16
// ds_write_b8 + ~12 VALU byte extracts of stage_b_transposed. Each wave
// keeps four 16x16 int32
// accumulators resident over the whole K loop, then a fused scale/bf16
// store. The kk ordering (k0 outer, kk inner step kTileK) and the
// acc00/acc01/acc10/acc11 update order are unchanged from iteration 1, and
// the col_major loader places the same B[k][n] values in the same fragment
// registers as row_major, so int32 accumulation is bit-identical.
// kPrefillStageK = 64 (iteration 7): six 64-k stages (7 barriers per block
// including the prologue), one full stage of global loads in flight, and
// measured residency 1 block/CU / 8 waves (code object 69 VGPRs at
// iteration 12; the __launch_bounds__(512, 2) minBlocks pin is a floor, not
// a guarantee - round 14 expects the code object to drop to <= 64 VGPRs
// once the B byte-extract temps are gone, which would finally make 2
// blocks/CU / 16 waves / 4 waves per SIMD satisfiable at the constant
// 128x64 tile).
// ---------------------------------------------------------------------------
template <int kBlockM, int kBlockN, int kStageK>
__global__ __launch_bounds__(kBlockM * kBlockN / 1024 * kWaveSize, 2) void
w8a8_dumma_prefill_kernel(
    const int8_t* __restrict__ x_q,
    const int8_t* __restrict__ weight,
    const float* __restrict__ x_scale,
    const float* __restrict__ weight_scale,
    __hip_bfloat16* __restrict__ out,
    int m,
    int n,
    int k) {
  constexpr int kWavesN = kBlockN / 32;
  constexpr int kAStride = kStageK + kLdsPad;
  static_assert(kAStride % 16 == 0,
                "A LDS row stride must stay 16-byte aligned");
  static_assert(kBTStride % 8 == 0,
                "transposed-B LDS row stride must stay 8-byte aligned");
  static_assert(kStageK % kTileK == 0,
                "K stage must be a multiple of the DUMMA K unit");
  static_assert(kStageK % sizeof(int4) == 0,
                "K stage must keep int4 staging aligned");

  const int tid = static_cast<int>(threadIdx.x);
  const int wave = tid / kWaveSize;
  const int lane = tid % kWaveSize;
  const int wave_row = wave / kWavesN;
  const int wave_col = wave - wave_row * kWavesN;
  const int m0 = static_cast<int>(blockIdx.y) * kBlockM;
  const int n0 = static_cast<int>(blockIdx.x) * kBlockN;

  __shared__ __align__(16) int8_t a_tile[2][kBlockM * kAStride];
  // Transposed (n-major) B tile: element B[k0+kk][n] lives at
  // b_tile[n][kk], so col_major fragment loads read 8 contiguous K bytes
  // per lane (vectorized ds_read2_b32). The staged runs are written n-major
  // (packed[n][k]) as two 8-byte ds_write_b64 per thread.
  __shared__ __align__(16) int8_t b_tile[2][kBlockN * kBTStride];

  DUFragment<matrix_a, kTileM, kTileN, kTileK, signed char, row_major>
      a_frag0, a_frag1, a_frag0_1, a_frag1_1;
  DUFragment<matrix_b, kTileM, kTileN, kTileK, signed char, col_major>
      b_frag0, b_frag1, b_frag0_1, b_frag1_1;
  DUFragment<accumulator, kTileM, kTileN, kTileK, int>
      acc00, acc01, acc10, acc11;
  du_fill_fragment(acc00, 0);
  du_fill_fragment(acc01, 0);
  du_fill_fragment(acc10, 0);
  du_fill_fragment(acc11, 0);

  constexpr int kAInt4PerRow = kStageK / sizeof(int4);
  constexpr int kAInt4 = kBlockM * kAInt4PerRow;
  // B is packed n-major by launch_pack_w8a8_weight (packed[n][k], one
  // contiguous K run per output column), so a stage row holds kStageK
  // contiguous bytes per column and stages as kChunksPerBRow 16-byte int4
  // runs (two 8-byte ds_write_b64 each). kBlockN * kChunksPerBRow is 256 for
  // 64x64 / 128x64 and 512 for 64x128, so one chunk per thread fits every
  // instantiation of the 512-thread block (no loop, matching the flat
  // staging issue order the accepted kernel depends on).
  constexpr int kChunksPerBRow = kStageK / sizeof(int4);
  constexpr int kBInt4 = kBlockN * kChunksPerBRow;
  const int kStages = k / kStageK;

  // Prologue: prefetch stage 0 into VGPRs and land it in buffer 0. Every
  // source address is 16-byte aligned (k % 16 == 0 and n % 16 == 0 by the
  // launch guard) and every destination row start is 16-byte aligned.
  int4 a_pref{0, 0, 0, 0};
  int4 b_pref{0, 0, 0, 0};
  {
    if (tid < kAInt4) {
      const int row = tid / kAInt4PerRow;
      const int seg = tid - row * kAInt4PerRow;
      const int global_row = m0 + row;
      if (global_row < m) {
        a_pref = *reinterpret_cast<const int4*>(
            x_q + global_row * k + seg * sizeof(int4));
      }
    }
    if (tid < kBInt4) {
      const int n_local = tid / kChunksPerBRow;
      const int kchunk = (tid - n_local * kChunksPerBRow) * sizeof(int4);
      b_pref = *reinterpret_cast<const int4*>(
          weight + (n0 + n_local) * k + kchunk);
    }
    if (tid < kAInt4) {
      const int row = tid / kAInt4PerRow;
      const int seg = tid - row * kAInt4PerRow;
      reinterpret_cast<int4*>(a_tile[0])[row * (kAStride / sizeof(int4)) +
                                         seg] = a_pref;
    }
    if (tid < kBInt4) {
      const int n_local = tid / kChunksPerBRow;
      const int kchunk = (tid - n_local * kChunksPerBRow) * sizeof(int4);
      const int64_t* src64 = reinterpret_cast<const int64_t*>(&b_pref);
      int64_t* dst64 = reinterpret_cast<int64_t*>(
          b_tile[0] + n_local * kBTStride + kchunk);
      dst64[0] = src64[0];
      dst64[1] = src64[1];
    }
    __syncthreads();
  }

  for (int s = 0; s < kStages; ++s) {
    const int k0 = s * kStageK;
    const int buf = s & 1;

    // Issue the global loads for stage s+1 now so their latency hides
    // behind the fragment loads and MMACs below. Out-of-range A rows are
    // zero-filled (tail-M guard, matching iteration 1).
    if (s + 1 < kStages) {
      const int k1 = k0 + kStageK;
      if (tid < kAInt4) {
        const int row = tid / kAInt4PerRow;
        const int seg = tid - row * kAInt4PerRow;
        const int global_row = m0 + row;
        if (global_row < m) {
          a_pref = *reinterpret_cast<const int4*>(
              x_q + global_row * k + k1 + seg * sizeof(int4));
        } else {
          a_pref = int4{0, 0, 0, 0};
        }
      }
      if (tid < kBInt4) {
        const int n_local = tid / kChunksPerBRow;
        const int kchunk = (tid - n_local * kChunksPerBRow) * sizeof(int4);
        b_pref = *reinterpret_cast<const int4*>(
            weight + (n0 + n_local) * k + k1 + kchunk);
      }
    }

    // Consume stage s from buffer buf: four fragment loads and four MMACs
    // per wave per kTileK step. kStageK = 64 = 2 kTileK steps, now fully
    // unrolled (round 11) into two independent fragment register sets.
    // The accepted ISA (exact-source kk loop) shows each rolled trip as
    // 4 ds_read2_b32 -> 4 s_waitcnt lgkmcnt(3..0) -> ~24 VALU byte-swizzle
    // (the du_load_matrix_sync lowering) -> 4 v_mmac_i32_16x16x32_i8, with
    // VGPR reuse forcing the kk=32 loads to wait for the kk=0 MMACs, so
    // each stage serializes two load-wait-swizzle-MMAC chains per wave.
    // With both steps' eight ds_read2 issued up front into distinct
    // registers, the second step's LDS latency overlaps the first step's
    // swizzle + MMACs and the per-stage lgkmcnt wait window collapses.
    // The kk ordering (k0 outer, kk = 0 then kTileK) and the
    // acc00/acc01/acc10/acc11 update order are unchanged, so int32
    // accumulation is bit-identical. The second fragment set landed at 69
    // code-object VGPRs (measured iteration 12, fewer than the round-11
    // prediction); at 512 threads that keeps residency at 1 block/CU / 8
    // waves (2 blocks/CU would need <= 64 VGPRs), so occupancy, LDS
    // (29,696 B), stage depth (64 k), barrier count, and launch config are
    // pinned exactly where the accepted iteration-7 kernel runs.
    const int local_row = wave_row * 32;
    const int local_col = wave_col * 32;
    static_assert(kStageK == 2 * kTileK,
                  "round-11 unrolled body assumes kStageK == 2 * kTileK");
    du_load_matrix_sync(
        a_frag0, a_tile[buf] + local_row * kAStride, kAStride);
    du_load_matrix_sync(
        a_frag1, a_tile[buf] + (local_row + 16) * kAStride, kAStride);
    du_load_matrix_sync(
        b_frag0, b_tile[buf] + local_col * kBTStride, kBTStride);
    du_load_matrix_sync(
        b_frag1, b_tile[buf] + (local_col + 16) * kBTStride, kBTStride);
    // Second 32-k step (kk = kTileK): independent registers, issued while
    // the first step's data is still in flight.
    du_load_matrix_sync(
        a_frag0_1, a_tile[buf] + local_row * kAStride + kTileK, kAStride);
    du_load_matrix_sync(
        a_frag1_1, a_tile[buf] + (local_row + 16) * kAStride + kTileK,
        kAStride);
    // col_major: element B[k0+kk+k][n] at b_tile[n][kk+k], so the loader's
    // p[row*ldm + col + i] reads 8 contiguous bytes per lane.
    du_load_matrix_sync(
        b_frag0_1, b_tile[buf] + local_col * kBTStride + kTileK, kBTStride);
    du_load_matrix_sync(
        b_frag1_1, b_tile[buf] + (local_col + 16) * kBTStride + kTileK,
        kBTStride);
    du_mma_sync(acc00, a_frag0, b_frag0, acc00);
    du_mma_sync(acc01, a_frag0, b_frag1, acc01);
    du_mma_sync(acc10, a_frag1, b_frag0, acc10);
    du_mma_sync(acc11, a_frag1, b_frag1, acc11);
    du_mma_sync(acc00, a_frag0_1, b_frag0_1, acc00);
    du_mma_sync(acc01, a_frag0_1, b_frag1_1, acc01);
    du_mma_sync(acc10, a_frag1_1, b_frag0_1, acc10);
    du_mma_sync(acc11, a_frag1_1, b_frag1_1, acc11);

    // Land the prefetched stage s+1 in the alternate buffer. The compiler
    // inserts the vmcnt(0) wait here via the data dependency, so the wait
    // and the ds_write overlap the MMACs above instead of stalling before
    // them.
    if (s + 1 < kStages) {
      if (tid < kAInt4) {
        const int row = tid / kAInt4PerRow;
        const int seg = tid - row * kAInt4PerRow;
        reinterpret_cast<int4*>(a_tile[buf ^ 1])[row *
                                                 (kAStride / sizeof(int4)) +
                                                 seg] = a_pref;
      }
      if (tid < kBInt4) {
        const int n_local = tid / kChunksPerBRow;
        const int kchunk = (tid - n_local * kChunksPerBRow) * sizeof(int4);
        const int64_t* src64 = reinterpret_cast<const int64_t*>(&b_pref);
        int64_t* dst64 = reinterpret_cast<int64_t*>(
            b_tile[buf ^ 1] + n_local * kBTStride + kchunk);
        dst64[0] = src64[0];
        dst64[1] = src64[1];
      }
    }
    // One barrier per stage: makes the alternate-buffer writes visible to
    // the next stage's reads and retires this stage's reads before the
    // parity-flipped buffer is overwritten two stages later.
    __syncthreads();
  }

  const int base_row = m0 + wave_row * 32;
  const int base_col = n0 + wave_col * 32;
  store_prefill_fragment(
      acc00, x_scale, weight_scale, out, m, n,
      base_row, base_col, lane);
  store_prefill_fragment(
      acc01, x_scale, weight_scale, out, m, n,
      base_row, base_col + 16, lane);
  store_prefill_fragment(
      acc10, x_scale, weight_scale, out, m, n,
      base_row + 16, base_col, lane);
  store_prefill_fragment(
      acc11, x_scale, weight_scale, out, m, n,
      base_row + 16, base_col + 16, lane);
}

// ---------------------------------------------------------------------------
// Generic scalar int8 x int8 -> int32 fallback for unmatched (m, n, k) and
// small-M API cases. One thread per output element, coalesced along N.
// ---------------------------------------------------------------------------
__global__ __launch_bounds__(256) void w8a8_scalar_fallback_kernel(
    const int8_t* __restrict__ x_q,
    const int8_t* __restrict__ weight,
    const float* __restrict__ x_scale,
    const float* __restrict__ weight_scale,
    __hip_bfloat16* __restrict__ out,
    int m,
    int n,
    int k) {
  const int linear =
      static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int total = m * n;
  if (linear >= total) {
    return;
  }
  const int row = linear / n;
  const int col = linear - row * n;
  const int8_t* a_row = x_q + row * k;
  // The packed weight is n-major since round 14 (packed[n][k]), so the
  // column stride is k; the products and their int32 summation order are
  // unchanged, keeping the fallback bit-identical to the pre-round-14 code.
  const int8_t* b_col = weight + col * k;
  int32_t acc = 0;
  for (int kk = 0; kk < k; ++kk) {
    acc += static_cast<int32_t>(a_row[kk]) *
           static_cast<int32_t>(b_col[kk]);
  }
  const float scaled =
      static_cast<float>(acc) * x_scale[row] * weight_scale[col];
  out[linear] = __float2bfloat16(scaled);
}

// ---------------------------------------------------------------------------
// n-major transpose pack (round 14). pack_weight runs once per weight tensor
// outside the timed region and outside CUDA/HIP Graph capture, so a
// byte-granular kernel is fine: packed[col * k + row] = raw[row * n + col]
// gives every output column a contiguous K run, which the timed GEMM stages
// with two vectorized ds_write_b64 per thread instead of the 16
// byte-granular ds_write_b8 + ~12 VALU byte extracts per staging thread of
// the identity-layout path. The layout is opaque to the API contract.
// ---------------------------------------------------------------------------
__global__ __launch_bounds__(256) void w8a8_transpose_i8_kernel(
    const int8_t* __restrict__ src,
    int8_t* __restrict__ dst,
    int k,
    int n) {
  const int64_t linear =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total = static_cast<int64_t>(k) * n;
  if (linear >= total) {
    return;
  }
  const int row = static_cast<int>(linear / n);
  const int col = static_cast<int>(linear - static_cast<int64_t>(row) * n);
  dst[static_cast<int64_t>(col) * k + row] = src[linear];
}

__global__ __launch_bounds__(256) void w8a8_identity_copy_f32_kernel(
    const float* __restrict__ src,
    float* __restrict__ dst,
    int64_t numel) {
  const int64_t linear =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (linear < numel) {
    dst[linear] = src[linear];
  }
}

}  // namespace

// ---------------------------------------------------------------------------
// Host launch symbols (extern "C", consumed by csrc/bindings.cpp).
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
  (void)workspace;
  (void)workspace_bytes;
  auto* out_bf16 = reinterpret_cast<__hip_bfloat16*>(out);

  // Native INT8 DUMMA m16n16k32 tiled path for large-M prefill. All three
  // macro-tile instantiations share the exact (m, n, k) guard; the active
  // one is fixed at compile time, so the timed region never tunes. The
  // guard divisibility (k % 128 == 0) is unchanged from iteration 1 and
  // implies k % kPrefillStageK == 0 for the pipelined kernel.
  if (m >= 128 && (k % kPrefillGuardK) == 0 && (n % kActiveBlockN) == 0) {
    const dim3 grid(
        static_cast<unsigned>(n / kActiveBlockN),
        static_cast<unsigned>((m + kActiveBlockM - 1) / kActiveBlockM));
    const dim3 block(static_cast<unsigned>(
        kActiveBlockM * kActiveBlockN / 1024 * kWaveSize));
    if (kActiveTileIndex == 0) {
      hipLaunchKernelGGL(
          HIP_KERNEL_NAME(w8a8_dumma_prefill_kernel<64, 64, kPrefillStageK>),
          grid, block, 0, stream,
          a, b, x_scale, weight_scale, out_bf16, m, n, k);
    } else if (kActiveTileIndex == 1) {
      hipLaunchKernelGGL(
          HIP_KERNEL_NAME(w8a8_dumma_prefill_kernel<64, 128, kPrefillStageK>),
          grid, block, 0, stream,
          a, b, x_scale, weight_scale, out_bf16, m, n, k);
    } else {
      hipLaunchKernelGGL(
          HIP_KERNEL_NAME(w8a8_dumma_prefill_kernel<128, 64, kPrefillStageK>),
          grid, block, 0, stream,
          a, b, x_scale, weight_scale, out_bf16, m, n, k);
    }
    return;
  }

  // Generic scalar fallback for every unmatched (m, n, k) and M < 128.
  const int total = m * n;
  constexpr int kFallbackThreads = 256;
  const dim3 grid(static_cast<unsigned>(
      (total + kFallbackThreads - 1) / kFallbackThreads));
  const dim3 block(static_cast<unsigned>(kFallbackThreads));
  hipLaunchKernelGGL(
      w8a8_scalar_fallback_kernel,
      grid, block, 0, stream,
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
  constexpr int kCopyThreads = 256;
  // Round 14: n-major transpose pack (packed[n][k], K-contiguous per output
  // column). The API contract (int8_w8a8_gemm_api.py) allows any contiguous
  // opaque packed layout; the timed GEMM consumes exactly this layout, so
  // its B staging drops from 16 byte-granular ds_write_b8 + ~12 VALU byte
  // extracts per staging thread to two 8-byte ds_write_b64. The scale copy
  // below is unchanged (N-length, order-independent).
  const int64_t weight_numel = static_cast<int64_t>(k) * n;
  const dim3 weight_grid(static_cast<unsigned>(
      (weight_numel + kCopyThreads - 1) / kCopyThreads));
  hipLaunchKernelGGL(
      w8a8_transpose_i8_kernel,
      weight_grid, dim3(kCopyThreads), 0, stream,
      raw_weight, packed_weight, k, n);

  const dim3 scale_grid(static_cast<unsigned>(
      (n + kCopyThreads - 1) / kCopyThreads));
  hipLaunchKernelGGL(
      w8a8_identity_copy_f32_kernel,
      scale_grid, dim3(kCopyThreads), 0, stream,
      weight_scale, packed_weight_scale, static_cast<int64_t>(n));
}
// @@end
