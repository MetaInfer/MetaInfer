// @@variant shape=hy3_tp8_shared_down_proj_m4096 commit=426ea470e7ee02bcdbe1428e583651b7e518219e added=2026-08-27
//   median_us=171.2 p90_us=171.5 speedup=62.87 baseline_us=1.076e+04
//   source=hy3-dsh-tp8-m4096-1-0ebb994d
// MetaInfer W8A8 INT8 GEMM backend for Hygon gfx928 (K500SM_AI).
// worker_3 iteration 1 (architecture round), assigned shape:
//   hy3_tp8_shared_down_proj_m4096 : (M, N, K) = (4096, 4096, 192)
//
// This file provides the two stable host launch symbols consumed by
// csrc/bindings.cpp:
//   * launch_w8a8_gemm(...)         - timed, graph-safe GEMM on the caller's
//                                     HIP stream; no allocation, packing,
//                                     autotuning, or synchronization.
//   * launch_pack_w8a8_weight(...)  - out-of-timed-region packing. For the
//                                     exact assigned pair (k, n) == (192,
//                                     4096) this round emits an n-major
//                                     transpose pack (packed[n][k], one
//                                     contiguous K run per output column);
//                                     every other (k, n) pair keeps the
//                                     identity device-to-device copy so the
//                                     generic fallback and paired small-M
//                                     shapes stay on the raw [K, N] layout.
//
// Round 1 strategy (DUMMA throughput baseline, 2-D macro-tile family):
//   * Native INT8 DUMMA m16n16k32 tiled kernel with int32 accumulation.
//   * 2-D macro-tile family (64x64, 64x128, 128x64); the active tile is
//     selected at compile time (kActiveTileIndex = 1 -> 64x128 (round 4
//     aspect-ratio flip; rounds 1-3 ran 128x64), 512 threads
//     = 8 wavefronts, one 32x32 quadrant per wave with four 16x16 int32
//     accumulators kept resident over the whole K loop). All three tiles
//     share the templated double-buffered staging pipeline below.
//   * K = 192 is pipelined in kPrefillStageK = 64 stages (3 stages) into a
//     double-buffered LDS pair with software prefetch: the global int4
//     loads of stage s+1 are issued before the MMAC sequence of stage s and
//     the alternate LDS buffer is written right after it, so the vmcnt(0)
//     wait and ds_write overlap compute. One barrier per stage.
//   * A stays row-major in LDS (stride 88, 8-byte-aligned); B is packed
//     n-major at pack time (packed[n][k]) and staged n-major into LDS
//     (b_tile[n][k], row stride 72, K-contiguous) with two 8-byte
//     ds_write_b64 per thread, consumed through col_major matrix_b
//     fragments; both operand paths use the round-8 explicit load_frag8
//     8-byte ds_read_b64 reads (the bootstrap's row_major matrix_b path was
//     byte-granular: 16 ds_read_u8 + ~40 VALU of byte assembly per wave per
//     k-step, the measured LDS-issue bottleneck of the 634 us bootstrap).
//   * kk body fully unrolled into two independent fragment register sets
//     (kStageK == 2 * kTileK) so the second step's LDS latency overlaps the
//     first step's swizzle + MMACs.
//   * Fused direct fragment -> x_scale -> weight_scale -> bf16 epilogue,
//     using the verified gfx928 accumulator ownership (row = lane & 15,
//     col_mod4 = lane >> 4, x[i] -> columns col_mod4 + 4*i). The int32 dot
//     is exact (max |dot| = 192*127*127 << 2^31), the int32 -> float
//     conversion is exact, and the single fp32 multiply followed by one
//     bfloat16 rounding matches the CPU int64 reference bit-for-bit. The
//     col_major loader places the same B[k][n] values in the same fragment
//     registers as row_major, so int32 accumulation is bit-identical to the
//     bootstrap path.
//   * All other (m, n, k) - including every M < 128 API case and the paired
//     M=2 shape with the same (N, K) - go to a scalar int8/int32 fallback
//     that decodes the packed layout only for (k, n) == (192, 4096) and the
//     raw [K, N] layout for every other pair.
//
// Header order is fixed by the control plane: hip_runtime first (du_mma.h
// is not self-contained before it), hip_bfloat16 second, du_mma.h last.

#include <hip/hip_runtime.h>
#include <hip/hip_bfloat16.h>
#include <du_mma.h>

#include <cstdint>

namespace {

constexpr int kTileM = 16;
constexpr int kTileN = 16;
constexpr int kTileK = 32;
constexpr int kWaveSize = 64;

// Staging stage length for the prefill pipeline. 64 = 2x the DUMMA K unit:
// with the 128x64 macro-tile the double-buffered footprint is
// 2*(128*88 + 64*72) = 31,744 B; with the round-4 active 64x128 tile it is
// 2*(64*88 + 128*72) = 29,696 B (A stride 88, transposed-B stride 72; both
// footprints leave 2 blocks/CU LDS-fit at 512 threads (8 waves per block,
// 16 resident wavefronts): 59,392 B <= 64 KiB for 64x128, the same 29,696 B
// per block that round 1 ran resident at). Stage 64 keeps
// the barrier
// count low for the short K = 192 (3 stages -> 4 barriers per block
// including the prologue) while keeping one full 64-k stage of global loads
// in flight. Padded row strides stay aligned: kAStride 88 % 8 == 0 for the
// two-half int64 A staging, kBTStride 72 % 8 == 0 for the col_major 8-byte
// fragment reads and the two-half ds_write_b64 B staging. The A pad is 24
// (stride 22 dwords): the row_major m16k32 fragment maps lane l to row
// l&15, k-quarter l/16 (8 contiguous bytes per lane), so the 16 fragment
// rows read banks (22*r mod 32) - 16 distinct even bank-pairs - and every
// A ds_read2 hits the 4-phase 32-bank floor with zero conflicts. Stride 20
// dwords (pad 16) aliases rows r and r+8 onto identical bank pairs
// (20*r mod 32 has period 8), doubling each A fragment read to 8 phases;
// stride 22 is the smallest even dword stride >= 16 whose 16 multiples are
// pairwise distinct mod 32.
constexpr int kPrefillStageK = 64;

// The only assigned shape for this worker has K = 192, so the optimized
// launch guard is shape-exact on this value (k == kAssignedK). 192 = 3
// stage-64 chunks.
constexpr int kAssignedK = 192;
constexpr int kAssignedN = 4096;

// Compile-time selection of the active 2-D macro-tile:
//   0 -> 64x64, 1 -> 64x128, 2 -> 128x64.
// All three instantiations are compiled so the family stays measurable; only
// the active one is launched inside the timed region. Round 4 (tile-shape
// round) flips the active tile from 128x64 to 64x128: the first constant-
// occupancy aspect-ratio A/B in this session (same 512 threads / 8 waves /
// 2 blocks/CU / grid 2048; only the (M,N) quadrant partition and grid
// geometry (N/128, M/64) swap). The flip halves per-A-tile global re-reads
// (64 -> 32) and doubles B re-reads (32 -> 64), both 768 KiB L2-resident;
// the active footprint is 29,696 B/block (59,392 B/CU at 2 blocks <= 64 KiB,
// the same per-block footprint round 1 ran resident at).
constexpr int kActiveTileIndex = 1;
constexpr int kActiveBlockM = kActiveTileIndex == 2 ? 128 : 64;
constexpr int kActiveBlockN = kActiveTileIndex == 1 ? 128 : 64;

// LDS padding in bytes added to each staged A row. 24 keeps every row start
// 8-byte aligned for the two-half int64 staging while making the 16
// m16k32-fragment row bases pairwise distinct mod the 32 LDS banks
// (stride 22 dwords: 22*r mod 32 is distinct for r = 0..15), so the A
// fragment ds_read2 instructions run at the conflict-free 4-phase floor.
// Strides of 20 dwords (pad 16) alias rows r and r+8 onto identical bank
// pairs (8 two-way conflicts per 16-lane group -> 8 phases per read);
// strides >= 16 that are multiples of 128 B alias every row onto one bank
// phase (up to 8-16-way conflicts).
constexpr int kLdsPad = 24;

// LDS row stride of the transposed (n-major) B tile: each row holds one
// output column n with kStageK contiguous K bytes. kLdsPadB = 8 keeps every
// row start 8-byte aligned so the col_major matrix_b fragment loader can
// lower its 8 contiguous bytes per lane to a single vectorized ds_read2_b32
// and the B staging can land each n row's k-contiguous 16-byte run as two
// 8-byte ds_write_b64 (72 % 8 == 0; 72 % 16 == 8, so a single ds_write_b128
// would be 16-byte misaligned for odd n rows - the explicit two-half store
// keeps the lowering legal).
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

// ---------------------------------------------------------------------------
// Round 8 (LDS read-width round): explicit 8-byte fragment loader replacing
// the library du_load_matrix_sync for the int8 m16k32 fragments. The library
// row_major matrix_a and col_major matrix_b loaders both fetch the same
// per-lane bytes - lane l -> row (l & 15), k-quarter (l >> 4), eight
// contiguous bytes at p[row*ldm + (l>>4)*8 .. +7] - and the compiler lowers
// them to ds_read2_b32 (two 4-byte dword phases per half-wave). A
// ds_read2_b32 phase groups 32 lanes (rows 0-15 of k-quarter q with rows
// 0-15 of k-quarter q+1) into one 32-bank cycle, so the kq and kq+1 row sets
// share every phase: {22*r mod 32} and {22*r+2 mod 32} (A stride 22 dwords;
// 18*r and 18*r+2 for B stride 18) are the same 16 even banks, giving a
// 2-way conflict in every phase - 8 phases per read at ANY even dword row
// stride. This is why the exact-source PMC lds_bank_conflicts (2,162,688)
// is byte-identical between the round-5 stride-80 and round-6 stride-88
// sources. Loading the same 8 bytes as one int64 forces the 8-byte
// ds_read_b64 form, whose phases are 16-lane groups (16 lanes x 8 B = 128 B
// = one bank cycle): within each group the 16 rows land on 16 distinct
// bank-pairs (A: 22*r mod 32 distinct for r = 0..15, B: 18*r mod 32
// distinct), so every fragment read runs at the 4-phase conflict-free floor.
// The element-to-slot mapping is unchanged (f.x[0..7] = the same 8 bytes in
// the same little-endian order), the MMACs and the exact int32 k-ascending
// accumulation sequence are untouched, and the staging stores, LDS strides,
// barriers, tile, occupancy and epilogue are byte-identical.
// ---------------------------------------------------------------------------
template <typename Frag>
__device__ __forceinline__ void load_frag8(Frag& f, const int8_t* p,
                                           unsigned ldm) {
  const unsigned row = __lane_id() & 0xf;
  const unsigned kq = __lane_id() >> 4;
  const int64_t v =
      *reinterpret_cast<const int64_t*>(p + row * ldm + (kq << 3));
  reinterpret_cast<int64_t*>(f.x)[0] = v;
}

// ---------------------------------------------------------------------------
// Large-M prefill path. One block computes a kBlockM x kBlockN output tile
// with (kBlockM/32) x (kBlockN/32) wavefronts (one 32x32 quadrant per wave).
// A[kBlockM, K] and B[K, kBlockN] are cooperatively staged into a
// double-buffered, bank-padded LDS in kStageK chunks with VGPR prefetch:
// the global int4 loads for stage s+1 are issued before the MMAC sequence
// of stage s, and the alternate LDS buffer is written right after it, so
// the vmcnt(0) wait and ds_write overlap compute. One barrier per stage.
// A stays row-major in LDS (stride 88); B is packed n-major by
// launch_pack_w8a8_weight (packed[n][k], one contiguous K run per output
// column) and staged n-major (stride 72, K-contiguous) with two 8-byte
// ds_write_b64 per thread, loaded through the round-8 explicit load_frag8
// 8-byte ds_read_b64 reads (row_major A and col_major B fragments place the
// same 8 contiguous bytes per lane in the same x[0..7] slots, so both
// operand paths are vectorized 8-byte LDS reads). Each wave keeps four
// 16x16 int32 accumulators resident over the whole K loop, then a fused
// scale/bf16 store. The kk ordering (k0 outer, kk inner step kTileK) and
// the acc00/acc01/acc10/acc11 update order are unchanged from the
// bootstrap, and the loaders place the same B[k][n] values in the same
// fragment registers as row_major, so int32 accumulation is bit-identical.
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
  static_assert(kAStride % 8 == 0,
                "A LDS row stride must stay 8-byte aligned");
  static_assert(kBTStride % 8 == 0,
                "transposed-B LDS row stride must stay 8-byte aligned");
  static_assert(kStageK % kTileK == 0,
                "K stage must be a multiple of the DUMMA K unit");
  static_assert(kStageK % sizeof(int4) == 0,
                "K stage must keep int4 staging aligned");
  static_assert(kStageK == 2 * kTileK,
                "unrolled kk body assumes kStageK == 2 * kTileK");

  const int tid = static_cast<int>(threadIdx.x);
  const int wave = tid / kWaveSize;
  const int lane = tid % kWaveSize;
  const int wave_row = wave / kWavesN;
  const int wave_col = wave - wave_row * kWavesN;
  const int m0 = static_cast<int>(blockIdx.y) * kBlockM;
  const int n0 = static_cast<int>(blockIdx.x) * kBlockN;

  constexpr int kThreads = kBlockM * kBlockN / 1024 * kWaveSize;
  constexpr int kAInt4PerRow = kStageK / sizeof(int4);
  constexpr int kAInt4 = kBlockM * kAInt4PerRow;
  constexpr int kChunksPerBRow = kStageK / sizeof(int4);
  constexpr int kBInt4 = kBlockN * kChunksPerBRow;
  static_assert(kAInt4 <= kThreads && kBInt4 <= kThreads,
                "one int4 chunk per thread requires chunks <= blockDim");
  const int kStages = k / kStageK;

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

  // Prologue: prefetch stage 0 into VGPRs and land it in buffer 0. Every
  // source address is 16-byte aligned (k % 16 == 0 by the exact k guard and
  // n % 16 == 0 by the launch guard); A destination rows are 8-byte aligned
  // (stride 88) and B destination rows 8-byte aligned (stride 72), so all
  // staging stores are the two-half int64 ds_write2_b64 form.
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
      // Two 8-byte halves: kAStride 88 % 16 == 8 makes odd rows 16-byte
      // misaligned, so the single-int4 store is illegal; the int64 pair
      // lowers to one ds_write2_b64 per thread, exactly like the B staging.
      const int64_t* src64 = reinterpret_cast<const int64_t*>(&a_pref);
      int64_t* dst64 = reinterpret_cast<int64_t*>(
          a_tile[0] + row * kAStride + seg * sizeof(int4));
      dst64[0] = src64[0];
      dst64[1] = src64[1];
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
    // zero-filled (tail-M guard).
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
    // per wave per kTileK step. kStageK = 64 = 2 kTileK steps, fully
    // unrolled into two independent fragment register sets: all eight
    // 8-byte fragment reads of a stage issue before the MMACs, so the
    // second step's LDS latency overlaps the first step's MMACs.
    const int local_row = wave_row * 32;
    const int local_col = wave_col * 32;
    // Round 8: custom load_frag8 replaces du_load_matrix_sync for all eight
    // fragment reads. Both library loaders (row_major A and col_major B)
    // place the same 8 contiguous bytes per lane into the same x[0..7]
    // slots; only the LDS instruction changes, from ds_read2_b32 (2-way
    // conflict in every 32-lane dword phase at any even row stride) to the
    // 8-byte ds_read_b64 (16-lane phases, distinct bank-pairs per group ->
    // 4-phase conflict-free floor). Exact int32 accumulation is unchanged.
    load_frag8(a_frag0, a_tile[buf] + local_row * kAStride, kAStride);
    load_frag8(a_frag1, a_tile[buf] + (local_row + 16) * kAStride, kAStride);
    load_frag8(b_frag0, b_tile[buf] + local_col * kBTStride, kBTStride);
    load_frag8(b_frag1, b_tile[buf] + (local_col + 16) * kBTStride,
               kBTStride);
    // Second 32-k step (kk = kTileK): independent registers, issued while
    // the first step's data is still in flight.
    load_frag8(a_frag0_1, a_tile[buf] + local_row * kAStride + kTileK,
               kAStride);
    load_frag8(a_frag1_1,
               a_tile[buf] + (local_row + 16) * kAStride + kTileK, kAStride);
    // col_major: element B[k0+kk+k][n] at b_tile[n][kk+k], so the loader's
    // p[row*ldm + col + i] reads 8 contiguous bytes per lane.
    load_frag8(b_frag0_1, b_tile[buf] + local_col * kBTStride + kTileK,
               kBTStride);
    load_frag8(b_frag1_1,
               b_tile[buf] + (local_col + 16) * kBTStride + kTileK,
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
        const int64_t* src64 = reinterpret_cast<const int64_t*>(&a_pref);
        int64_t* dst64 = reinterpret_cast<int64_t*>(
            a_tile[buf ^ 1] + row * kAStride + seg * sizeof(int4));
        dst64[0] = src64[0];
        dst64[1] = src64[1];
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
// For the exact packed pair (k, n) == (192, 4096) the weight is n-major
// (packed[n][k], column stride k); every other pair keeps the raw logical
// [K, N] row-major layout (column stride n). The int32 summation order is
// identical either way (k-ascending), keeping the fallback bit-identical.
// ---------------------------------------------------------------------------
__global__ __launch_bounds__(256) void w8a8_scalar_fallback_kernel(
    const int8_t* __restrict__ x_q,
    const int8_t* __restrict__ weight,
    const float* __restrict__ x_scale,
    const float* __restrict__ weight_scale,
    __hip_bfloat16* __restrict__ out,
    int m,
    int n,
    int k,
    int b_col_stride) {
  const int linear =
      static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int total = m * n;
  if (linear >= total) {
    return;
  }
  const int row = linear / n;
  const int col = linear - row * n;
  const int8_t* a_row = x_q + row * k;
  // b_col_stride is 1 for the n-major packed (192, 4096) pair (each output
  // column has a contiguous K run) and n for the raw [K, N] layout.
  const int8_t* b_col = weight + col * b_col_stride;
  int32_t acc = 0;
  for (int kk = 0; kk < k; ++kk) {
    acc += static_cast<int32_t>(a_row[kk]) *
           static_cast<int32_t>(b_col[kk * b_col_stride]);
  }
  const float scaled =
      static_cast<float>(acc) * x_scale[row] * weight_scale[col];
  out[linear] = __float2bfloat16(scaled);
}

// ---------------------------------------------------------------------------
// Pack kernels for launch_pack_w8a8_weight (run once per weight tensor
// outside the timed region and outside CUDA/HIP Graph capture, so plain
// kernels are fine):
//   * w8a8_transpose_i8_kernel: n-major transpose pack for the exact
//     assigned pair (k, n) == (192, 4096): packed[col * k + row] =
//     raw[row * n + col] gives every output column a contiguous K run,
//     which the timed GEMM stages with two vectorized ds_write_b64 per
//     thread instead of byte-granular scatters.
//   * w8a8_identity_copy_i8_kernel: identity device-to-device copy for every
//     unmatched (k, n) pair (raw [K, N] layout preserved).
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

__global__ __launch_bounds__(256) void w8a8_identity_copy_i8_kernel(
    const int8_t* __restrict__ src,
    int8_t* __restrict__ dst,
    int64_t numel) {
  const int64_t linear =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (linear < numel) {
    dst[linear] = src[linear];
  }
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

  // Native INT8 DUMMA m16n16k32 tiled path for the assigned shape family.
  // The guard is shape-exact (k == kAssignedK == 192, the only assigned K),
  // keeps M >= 128 (small-M API cases fall through to the scalar path), and
  // requires N to be a multiple of the active macro-tile N (4096 always
  // is). The active macro-tile is fixed at compile time, so the timed
  // region never tunes. All three tile instantiations share the same guard.
  if (m >= 128 && k == kAssignedK && (n % kActiveBlockN) == 0) {
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
  // The packed weight is n-major only for the exact assigned pair
  // (k, n) == (192, 4096); every other pair is identity-packed, so the
  // fallback decodes b_col with column stride k for that pair and with
  // column stride n otherwise (the paired M=2 shape with the same (N, K)
  // is covered by the k-stride branch).
  const bool packed_nmajor = (k == kAssignedK && n == kAssignedN);
  const int b_col_stride = packed_nmajor ? 1 : n;
  const int total = m * n;
  constexpr int kFallbackThreads = 256;
  const dim3 grid(static_cast<unsigned>(
      (total + kFallbackThreads - 1) / kFallbackThreads));
  const dim3 block(static_cast<unsigned>(kFallbackThreads));
  hipLaunchKernelGGL(
      w8a8_scalar_fallback_kernel,
      grid, block, 0, stream,
      a, b, x_scale, weight_scale, out_bf16, m, n, k, b_col_stride);
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
  const int64_t weight_numel = static_cast<int64_t>(k) * n;
  const dim3 weight_grid(static_cast<unsigned>(
      (weight_numel + kCopyThreads - 1) / kCopyThreads));

  // Round 1: n-major transpose pack (packed[n][k], K-contiguous per output
  // column) for the exact assigned pair (k, n) == (192, 4096) only. The API
  // contract (int8_w8a8_gemm_api.py) allows any contiguous opaque packed
  // layout; the timed GEMM and the (192, 4096) fallback consume exactly
  // this layout, so the timed B staging drops from byte-granular scatters
  // to two 8-byte ds_write_b64 per thread. Every other (k, n) pair keeps
  // the identity device-to-device copy (raw [K, N] layout), so unmatched
  // shapes and paired small-M shapes on other pairs stay on the layout
  // their fallback decodes.
  if (k == kAssignedK && n == kAssignedN) {
    hipLaunchKernelGGL(
        w8a8_transpose_i8_kernel,
        weight_grid, dim3(kCopyThreads), 0, stream,
        raw_weight, packed_weight, k, n);
  } else {
    hipLaunchKernelGGL(
        w8a8_identity_copy_i8_kernel,
        weight_grid, dim3(kCopyThreads), 0, stream,
        raw_weight, packed_weight, weight_numel);
  }

  // The scale copy is unchanged (N-length, order-independent).
  const dim3 scale_grid(static_cast<unsigned>(
      (n + kCopyThreads - 1) / kCopyThreads));
  hipLaunchKernelGGL(
      w8a8_identity_copy_f32_kernel,
      scale_grid, dim3(kCopyThreads), 0, stream,
      weight_scale, packed_weight_scale, static_cast<int64_t>(n));
}
// @@end
// @@end
