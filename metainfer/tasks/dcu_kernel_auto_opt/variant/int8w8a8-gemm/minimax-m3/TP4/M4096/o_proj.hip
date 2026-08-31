// @@variant shape=minimax_tp4_o_proj_m4096 commit=ab9604c75a47252917e8d6d9c8dc7a35266de6ac added=2026-08-24
//   median_us=765.9 p90_us=766.8 speedup=38.78 baseline_us=2.97e+04
//   source=minimaxm3-dsh-tp4-m4096-1-0c2f84a9
// @@variant shape=minimax_tp4_o_proj_m4096 added=2026-08-23 iteration=1
// INT8 W8A8 GEMM HIP implementation for Hygon K500SM_AI / gfx928.
//
// Worker: worker_2 (physical GPU 2), assigned shape:
//   minimax_tp4_o_proj_m4096 : M=4096, N=6144, K=2048
//
// Iteration 20 - swizzled 8-byte-group LDS layout (bank-conflict cut).
//   * The accepted kernel's PMC (source digest 33c82f15, the official best
//       789.11 us) shows lds_bank_conflicts 28,704,768 vs lds_instructions
//       4,531,200 (~6.33 conflict cycles per fragment ds_read2 - the largest
//       remaining structural inefficiency in the accepted architecture; the
//       repaired iteration-8 128x32 class is closed on speed at iterations
//       9/10/11/15/19 ~1079-1139 us, occupancy-4 falsified, so the iteration
//       19 redirect names exactly this axis: a swizzled 64-byte B LDS layout
//       with the fragment loader reading the swizzle back).
//   * Mechanism: both LDS tiles use the padded 80-byte row stride (20 dwords);
//       bank(row, col) = (20*row + col/4) mod 32 has period 8 over the 16 rows
//       of every fragment, so lanes r and r+8 (plus the cross-k-chunk overlaps)
//       hit the same bank - 4 lanes per bank on every 8-byte fragment read.
//       Any 16-byte-aligned pad stride (dword count divisible by 4) keeps the
//       same period-8 collision, so padding alone cannot fix it. This round
//       re-lays BOTH LDS tiles at dense 64-byte rows (no pad) with the
//       bijective 8-byte-group map
//         group(row, c) = ((row & 15) + 4*(c & 3)) & 15 + 16*c + 128*(row >> 4)
//       (row = tile row, c = k 8-byte group). For any fragment load (16 rows,
//       fixed k phase) (row & 15) -> ((row & 15) + 4*(c & 3)) & 15 is a
//       permutation, so every 16-lane group reads 16 distinct LDS bank-pairs
//       and the old collision pair (r, r+8) lands 8 bank-pairs apart; the
//       worst-case lane grouping is 2-way instead of 4-way. The loader and
//       both stage writers use the same map (the writers emit two 8-byte LDS
//       stores per 16-byte global chunk at group(row, 2j) / group(row, 2j+1)).
//       Fragment values, k-byte order inside each 8-byte group, MMAC sequence,
//       int32 accumulation order (k0-outer 64, kk-inner 32), barrier count
//       (2/stage x 32), grid dim3(96,32) = 3072, element-to-lane mapping and
//       the fused scale->bf16 epilogue are bit-identical. LDS drops from
//       16,128 B to 13,056 B/block (a_tile 8,192 + b_tile 4,096 + scales 768);
//       the iteration-6 bf16 staging buffer (128 rows x 72 B = 9,216 B) now
//       spills into the dead b_tile region [8,192, 9,216) - b_tile is never
//       read after the k-loop's final barrier, so the epilogue is unchanged.
//       VGPR stays ~72 (a few extra swizzle ALU), occupancy stays 3 blocks/CU
//       (VGPR-limited), 4 blocks/CU is still impossible (4 x 256 x 73 > 65,536).
//   * Falsifiable prediction: local offline hipcc -O3 --offload-arch=gfx928
//       build reports group_segment_fixed_size 13,056 B, vgpr_count <= 85,
//       0 spills, wavefront 64, max_flat_workgroup_size 256; the profiled
//       launch stays 3 blocks/CU and lds_bank_conflicts drops from 28,704,768
//       toward the ~2-way floor (~9-10M) at unchanged lds_instructions. If the
//       LDS read bank conflicts are a real per-stage serialization, median
//       drops below the official best 789.11 us (>130.63 TOPS, p90 below
//       791.05 us) with correctness passing; if median stays >= 789.11 us with
//       conflicts near the floor, the accepted kernel is not LDS-read-bound
//       and the redirect is the epilogue/global-load path or the barrier
//       structure, NOT more LDS layout work.
//
// Iteration 7 - explicit 64-bit fragment loader (DUMMA issue grouping).
//   * The trusted gfx928 ISA of the accepted iteration-6 kernel (compile
//       cache key f364bbf5...) shows the steady-state 64-K k-loop body
//       (two unrolled m16n16k32 kk steps) holding 16 v_mmac, 12 ds_read2_b32
//       fragment loads, 7 s_waitcnt lgkmcnt gates - and 72 v_and_b32 /
//       v_or_b32_sdwa / v_or3_b32 identity byte-reassembly VALU ops
//       (~62% of the body's issue slots; 0 shifts, so bit-preserving ORs).
//       du_load_matrix_sync for int8 m16n16k32 (both matrix_a row_major and
//       matrix_b col_major) loads eight CONSECUTIVE bytes per lane into
//       x[0..7], and du_mma_sync reinterprets a.x/b.x as one 64-bit long in
//       natural byte order; the backend materializes that 64-bit MMAC
//       operand from the library's byte-wise loads by emitting the identity
//       shuffle chains, one per fragment, each gated by lgkmcnt(1) waits
//       before the consuming MMAC group (lds_wait_instructions 4.18M vs
//       lds_instructions 6.89M; valu_active = 2x valu_instructions).
//   * This round holds the accepted k-loop structure byte-for-byte
//       (single-buffered 64-K stages, n-major [N,K] B col_major fragments,
//       128x64 tile, 256 threads / 4 waves, 8 int32 accumulators, two
//       __syncthreads per stage, grid dim3(96,32) = 3072 blocks, 16,128 B
//       LDS at 3 blocks/CU, no split-K, no raw asm) and replaces only the
//       six per-kk du_load_matrix_sync calls with an explicit loader that
//       reads each lane's eight bytes as ONE 64-bit LDS read (8-byte aligned:
//       80-byte row strides and kk in {0,32} keep every fragment address
//       8-byte aligned) and stores them into the fragment's x[0..7] - the
//       exact value du_mma_sync reads. The per-lane address must reproduce
//       du_load_matrix_sync exactly: row = lane & 15, col = (lane >> 4) << 3,
//       eight consecutive bytes at p[row*ldm + col .. +7] (verified in
//       /opt/dtk/include/du_mma.hpp); the repair-1 bug was an earlier loader
//       that ignored the lane/ldm offset and gave every lane the same eight
//       base bytes (near-total mismatch, mismatch_count 25,148,019).
//       Fragment values, MMAC sequence, accumulation order and the fused
//       epilogue are bit-identical; the only change is which instructions
//       sit between the ds_read and each v_mmac (0 reassembly VALU, one
//       collapsed wait stream). No new registers are introduced (the freed
//       shuffle temps stay available to the scheduler), so VGPR stays 72
//       and occupancy stays 3 blocks/CU.
//   * Falsifiable prediction: if the identity reassembly VALU + per-group
//       lgkmcnt gates are a real issue-pipe cost, median drops below the
//       current best 1031.996 us (>99.88 TOPS, p90 below 1046.02 us) with
//       valu_instructions falling from 45.58M toward ~17-20M (72 -> ~0
//       shuffle ops per body per wave, 32 stages x 4 waves x 3072 blocks)
//       and lds_wait_instructions falling from 4.18M, at unchanged
//       lds_instructions (~6.89M), lds_bank_conflicts (~28.7M) and
//       3 blocks/CU; if the kernel is instead bound by the LDS load issue
//       itself or the per-stage barrier/global re-stage latency, median
//       stays >= 1031.996 us, falsifying the shuffle mechanism and
//       redirecting the next round to register-split kk prefetch or
//       stage-level pipelining rather than more loader surgery.
//
// Iteration 6 - fused epilogue: LDS-resident scales + coalesced store.
//   * The iteration-5 k-loop (single-buffered 64-K stages, n-major B,
//       128x64 tile, 3 blocks/CU) is held byte-for-byte; only the epilogue
//       changes.
//   * x_scale[128] and weight_scale[64] are loaded once per block into LDS
//       in the prologue (768 B; total LDS 16,128 B/block, still 3 blocks/CU).
//       The trusted ISA of iteration 5 showed the old epilogue issued 40
//       global_load_dword scale loads per thread (491K wavefront-instructions
//       kernel-wide, ~30% of all vmem_read) with s_waitcnt vmcnt(1)/vmcnt(0)
//       serialization before every fragment's multiply chain, and 33 static
//       scattered global_store_short_d16_hi 2-byte stores per thread (lane&15
//       selects the row, lane>>4 selects col%4 -> one 2-byte store per output
//       element, 64 separate sectors per store instruction, 393K store
//       instructions kernel-wide).
//   * New epilogue: after the k-loop the dead a_tile (10,240 B) is reused as
//       a two-phase bf16 staging buffer (128 rows x 72 B; 32 useful bf16 + 8 B
//       pad per row; 72 is 8 B-aligned for vector reads and its 4-byte bank
//       offset 18r is a full permutation over 16 consecutive rows, so the
//       per-lane 2-byte fragment writes are bank-conflict-free). Each wave
//       scales its eight int32 accumulator fragments (x_scale_l[row] *
//       w_scale_l[col], fused, then __float2bfloat16) while writing bf16 into
//       the staging buffer, and the block then stores each staged 128x32
//       half-tile with fully coalesced 8-byte int2 global stores (consecutive
//       lanes cover consecutive 8 B of one output row; every 32 B sector is
//       fully written). Global store instructions drop from 128 to 32 per
//       block and scale loads from 160 to 3 wavefront-instructions per block;
//       the per-element 64-bit v_mad_i64_i32 address arithmetic disappears.
//       Bit-identical output is preserved: the int32 accumulation order, the
//       fused fragment -> x_scale*weight_scale -> bf16 epilogue math, and the
//       element-to-lane mapping are unchanged (only the store path is
//       re-routed through LDS); LDS stays 16,128 B/block so occupancy remains
//       3 blocks/CU. The generic scalar fallback and the exact-shape guard
//       are untouched.
//
// Iteration 5 - B n-major pack + col_major fragment loads (reassembly cut).
//   * Large-prefill path (exact (m, n, k) == (4096, 6144, 2048)):
//       native INT8 DUMMA m16n16k32 with int32 accumulation; one 128x64
//       output tile per block; 256 threads = 4 wavefronts of 64 lanes; each
//       wave owns a 64x32 quadrant built from eight m16n16k32 fragments; the
//       block cooperatively vector-loads A[128,64] and B[64,64] into 15 KiB
//       of LDS (padded 80-byte row strides, five bank phases); 64-deep K
//       single buffering with two block barriers per stage (no split-K, no
//       raw asm, no speculative deep pipeline); fused
//       dot * x_scale[m] * weight_scale[n] epilogue stored directly as bf16
//       from the accumulator fragments.
//       The weight for this shape is packed once (outside the timed region)
//       into the n-major [N, K] layout packed[n*K + k] = raw[k*N + n], so B
//       stages are [64 n, 64 k] n-major LDS tiles: each m16n16k32 B
//       fragment's eight bytes per lane are eight consecutive k bytes and
//       the col_major loader emits plain 8-byte LDS reads instead of the
//       identity-layout per-byte ds_read_u8 gather + v_and/v_or3
//       byte-reassembly VALU (the gate_up iter-13 / qkv iter-18 lesson:
//       accepted M=4096 kernels cut fragment reassembly once B is n-major).
//       A stays row-major identity [M, K] (activations cannot be repacked).
//       Iteration 2's double-buffered 64-K staging (same identity B) was
//       already falsified at 1362 us, so occupancy-2 deeper staging is off
//       the table; this round holds the accepted single-buffered pipeline
//       and its 3 blocks/CU and only re-lays B.
//   * launch_pack_w8a8_weight packs (2048, 6144) n-major once; every other
//       (k, n) keeps the bootstrap identity device-to-device copy; packing
//       never happens inside the timed GEMM or the CUDA/HIP Graph region.
//   * Generic scalar int8/int32 fallback for every unmatched (m, n, k),
//       including small-M API cases (M=2/M=16 with the same (K, N) still
//       reach the fallback through the exact-shape guard): (k, n) ==
//       (2048, 6144) decodes the n-major packed layout, every other (K, N)
//       pair keeps the identity [K, N] layout.
//
// Graph safety: gemm_out performs no allocation, compilation, autotuning,
// packing, host/device synchronization, or default-stream launch; it runs on
// the caller-provided PyTorch HIP stream and returns the caller's `out`.

#include <hip/hip_runtime.h>
#include <hip/hip_bfloat16.h>
#include <du_mma.h>

#include <cstdint>

namespace {

using namespace du::dumma;

constexpr int kTileM = 16;
constexpr int kTileN = 16;
constexpr int kTileK = 32;  // gfx928 INT8 DUMMA primitive: m16n16k32
constexpr int kWaveSize = 64;  // gfx928 native wavefront
constexpr int kTargetM = 4096;
constexpr int kTargetN = 6144;
constexpr int kTargetK = 2048;
constexpr int kBlockM = 128;
constexpr int kBlockN = 64;
constexpr int kStageK = 64;
constexpr int kBlockThreads = 4 * kWaveSize;  // 256

// Iteration 6 epilogue staging (reuses the dead A/B LDS tiles, 12,288 B after
// the iteration-20 dense swizzle): each staged half-tile is 128 rows x 72 B =
// 9,216 B <= 12,288 B, holding 32 useful bf16 (64 B) plus 8 B pad per row.
// The staging writes spill into the dead b_tile region [8,192, 9,216); b_tile
// is never read after the k-loop's final barrier. 72 is 8 B-aligned (vector
// reads) and its 4-byte-bank offset 18*r is a full permutation over 16
// consecutive rows, so the per-lane 2-byte fragment staging writes are
// bank-conflict-free.
constexpr int kStageStride = 72;  // staging row stride in bytes
constexpr int kStageHalfBf16 = kBlockN / 2;  // 32 bf16 per staged row
constexpr int kStageChunks = (kBlockM * kStageHalfBf16) / 4;  // 1024 x 8 B

using bf16_t = hip_bfloat16;

// ---------------------------------------------------------------------------
// Iteration 6 fused epilogue, stage side: scale one m16n16k32 accumulator
// fragment (verified gfx928 int8 ownership: lane & 15 selects the row,
// lane >> 4 selects col % 4, x[i] maps to columns col%4 + 4*i) with the
// LDS-resident per-row x_scale and per-column weight_scale, convert to bf16,
// and write into the reused a_tile staging buffer (72-byte rows, 32 useful
// bf16 per row). The scale multiply order is identical to iteration 5
// (acc * x_scale[row] * weight_scale[col]), so the output is bit-identical.
// ---------------------------------------------------------------------------
template <typename AccFragment>
__device__ __forceinline__ void stage_prefill_fragment(
    const AccFragment& frag,
    int8_t* __restrict__ staging,
    const float* __restrict__ x_scale_l,
    const float* __restrict__ w_scale_l,
    int row0,
    int col0,
    int wcol0,
    int m0,
    int m,
    int lane) {
  const int row = row0 + (lane & 15);
  if (m0 + row >= m) {
    return;  // tail-M masking: padded rows never stage
  }
  const int col_mod4 = lane >> 4;
  const float xs = x_scale_l[row];
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const int col = col0 + col_mod4 + 4 * i;
    const float scaled =
        static_cast<float>(frag.x[i]) * xs * w_scale_l[wcol0 + col];
    *reinterpret_cast<bf16_t*>(staging + row * kStageStride + 2 * col) =
        __float2bfloat16(scaled);
  }
}

// ---------------------------------------------------------------------------
// Iteration 6 fused epilogue, store side: cooperative fully-coalesced store
// of one staged 128x32 half-tile (8,192 B). Each thread writes four 8-byte
// int2 chunks (q = tid, tid+256, ...); consecutive threads cover consecutive
// 8 B of one output row, so every 32 B sector is fully written and no
// per-element 64-bit address arithmetic is needed.
// ---------------------------------------------------------------------------
__device__ __forceinline__ void store_staged_half_tile(
    const int8_t* __restrict__ staging,
    bf16_t* __restrict__ out,
    int m0,
    int n0,
    int col_base,
    int m,
    int n,
    int tid) {
#pragma unroll
  for (int q = tid; q < kStageChunks; q += kBlockThreads) {
    const int row = q >> 3;  // 8 chunks (8 B each) per 64 B staging row
    const int sub = q & 7;
    if (m0 + row >= m) {
      continue;
    }
    *reinterpret_cast<int2*>(out + static_cast<int64_t>(m0 + row) * n + n0 +
                             col_base + 4 * sub) =
        *reinterpret_cast<const int2*>(staging + row * kStageStride + 8 * sub);
  }
}

// ---------------------------------------------------------------------------
// Iteration 20 swizzled 8-byte-group LDS layout. The accepted kernel's PMC
// shows 28,704,768 lds_bank_conflicts vs 4,531,200 lds_instructions (~6.33
// conflict cycles per fragment ds_read2). The cause is the padded 80-byte row
// stride (20 dwords): bank(row, col) = (20*row + col/4) mod 32 has period 8
// over the 16 rows of every fragment, so lanes r and r+8 (plus the
// cross-k-chunk overlaps) hit the same bank - 4 lanes per bank on every 8-byte
// fragment read. Any 16-byte-aligned pad stride keeps the same period-8
// collision, so padding alone cannot fix it. This round re-lays both LDS
// tiles at dense 64-byte rows with the bijective 8-byte-group map
//   group(row, c) = ((row & 15) + 4 * (c & 3)) & 15 + 16 * c + 128 * (row >> 4)
// where row is the tile row and c is the k 8-byte group. For any fragment
// load (16 rows, fixed c phase) (row & 15) -> ((row & 15) + 4 * (c & 3)) & 15
// is a permutation, so every 16-lane group reads 16 distinct LDS bank-pairs
// and the old collision pair (r, r+8) lands 8 bank-pairs apart; the worst
// case under any 16-lane grouping is 2-way instead of 4-way. The fragment
// loader and both stage writers must use the same map ("reading the swizzle
// back"): each 16-byte global chunk is written as two 8-byte LDS stores at
// group(row, 2j) and group(row, 2j+1). Fragment values, k-byte order within
// each 8-byte group, MMAC sequence, accumulation order and the fused
// epilogue are bit-identical. Iteration 7's explicit 64-bit loader (instead
// of du_load_matrix_sync's per-fragment identity v_and/v_or3 reassembly
// chains) is kept; only the address map changes.
// ---------------------------------------------------------------------------
__device__ __forceinline__ int swizzle_group(int row, int c) {
  return (((row & 15) + 4 * (c & 3)) & 15) + 16 * c + 128 * (row >> 4);
}

// Per-lane (row, k-group) position reproduces du_load_matrix_sync exactly:
// row = row0 + (lane & 15), c = (lane >> 4) + (kk >> 3), eight consecutive k
// bytes in natural byte order (row0 is always a multiple of 16, so
// row & 15 == lane & 15 and row >> 4 == row0 >> 4).
template <typename Frag>
__device__ __forceinline__ void load_frag8_swizzled(
    Frag& f, const int8_t* __restrict__ p, int lane, int row0, int kk) {
  const int row = row0 + (lane & 15);
  const int c = (lane >> 4) + (kk >> 3);
  *reinterpret_cast<long*>(f.x) =
      *reinterpret_cast<const long*>(p + swizzle_group(row, c) * 8);
}

__device__ __forceinline__ void store_chunk16_swizzled(
    int8_t* __restrict__ p, int row, int j, const int4& v) {
  *reinterpret_cast<int2*>(p + swizzle_group(row, 2 * j) * 8) =
      *reinterpret_cast<const int2*>(&v);
  *reinterpret_cast<int2*>(p + swizzle_group(row, 2 * j + 1) * 8) =
      *reinterpret_cast<const int2*>(
          reinterpret_cast<const char*>(&v) + 8);
}

// ---------------------------------------------------------------------------
// Large-M prefill: 128x64 output tile per block, four wavefronts of 64 lanes.
// Each wave owns a 64x32 quadrant (eight m16n16k32 int32 accumulators); the
// block cooperatively stages A[128,64] (identity [M, K] row-major) and
// B[64,64] (one-time n-major [N, K] packed layout, n-major LDS tile) into
// single-buffered LDS in 64-deep K stages. B fragments are loaded col_major:
// each lane's eight bytes are eight consecutive k bytes, so the loader emits
// plain 8-byte LDS reads with no per-byte gather/reassembly VALU.
// ---------------------------------------------------------------------------
__global__ __launch_bounds__(kBlockThreads) void
w8a8_dumma_prefill_128x64x64_identity_kernel(
    const int8_t* __restrict__ x_q,
    const int8_t* __restrict__ weight,
    const float* __restrict__ x_scale,
    const float* __restrict__ weight_scale,
    bf16_t* __restrict__ out,
    int m,
    int n,
    int k) {
  const int tid = static_cast<int>(threadIdx.x);
  const int wave = tid / kWaveSize;
  const int lane = tid % kWaveSize;
  const int wave_row = wave >> 1;
  const int wave_col = wave & 1;
  const int m0 = static_cast<int>(blockIdx.y) * kBlockM;
  const int n0 = static_cast<int>(blockIdx.x) * kBlockN;
  const int local_row = wave_row * 64;
  const int local_col = wave_col * 32;

  // Iteration 20: dense 64-byte-row tiles (8,192 + 4,096 B) with the
  // swizzled 8-byte-group map; the fragment loader reads the swizzle back.
  __shared__ __align__(16) int8_t a_tile[kBlockM * kStageK];
  __shared__ __align__(16) int8_t b_tile[kBlockN * kStageK];
  // Iteration 6: LDS-resident per-row / per-column scales (768 B total; the
  // kernel is at 13,056 B LDS per block = 3 blocks/CU, VGPR-limited). Loaded
  // once in the prologue, read only by the fused epilogue.
  __shared__ __align__(16) float x_scale_l[kBlockM];
  __shared__ __align__(16) float w_scale_l[kBlockN];

  DUFragment<matrix_a, kTileM, kTileN, kTileK, signed char, row_major>
      a_frag0, a_frag1;
  DUFragment<matrix_b, kTileM, kTileN, kTileK, signed char, col_major>
      b_frag0, b_frag1;
  DUFragment<accumulator, kTileM, kTileN, kTileK, int>
      acc00, acc01, acc10, acc11, acc20, acc21, acc30, acc31;
  du_fill_fragment(acc00, 0);
  du_fill_fragment(acc01, 0);
  du_fill_fragment(acc10, 0);
  du_fill_fragment(acc11, 0);
  du_fill_fragment(acc20, 0);
  du_fill_fragment(acc21, 0);
  du_fill_fragment(acc30, 0);
  du_fill_fragment(acc31, 0);

  // Prologue: cooperative vectorized staging of the first 64-K stage. Each
  // thread owns one int4 in A rows [0,64) and one in A rows [64,128), plus
  // one int4 of B[64,64] (consecutive threads cover consecutive 16-byte
  // chunks of each row, so every global load is coalesced).
  // Iteration 6: the per-row/per-column scales are also staged into LDS here
  // (threads 0..191, one float each); the first __syncthreads below makes
  // them visible to the fused epilogue.
  if (tid < kBlockM) {
    x_scale_l[tid] = x_scale[m0 + tid];
  } else if (tid < kBlockM + kBlockN) {
    w_scale_l[tid - kBlockM] = weight_scale[n0 + (tid - kBlockM)];
  }
  const int vector_byte_offset = tid * static_cast<int>(sizeof(int4));
  const int stage_row = vector_byte_offset / kStageK;  // 0..63
  const int stage_col = vector_byte_offset - stage_row * kStageK;  // 0,16,32,48
  store_chunk16_swizzled(
      a_tile, stage_row, stage_col >> 4,
      *reinterpret_cast<const int4*>(
          x_q + static_cast<int64_t>(m0 + stage_row) * k + stage_col));
  store_chunk16_swizzled(
      a_tile, stage_row + kBlockM / 2, stage_col >> 4,
      *reinterpret_cast<const int4*>(
          x_q + static_cast<int64_t>(m0 + stage_row + kBlockM / 2) * k +
          stage_col));
  // B stage is an n-major [64 n, 64 k] tile of the one-time packed layout
  // packed[n*K + k] = raw[k*N + n]: thread tid covers n row (tid>>2) and the
  // 16-byte k chunk (tid&3)*16, so the global load is coalesced along k; the
  // swizzled LDS store uses the same 8-byte-group map the fragment loader
  // reads back.
  const int b_n_row = tid >> 2;         // 0..63 n rows of the 64x64 B stage
  const int b_k_off = (tid & 3) * 16;   // 0,16,32,48 within the 64-byte k row
  store_chunk16_swizzled(
      b_tile, b_n_row, b_k_off >> 4,
      *reinterpret_cast<const int4*>(
          weight + static_cast<int64_t>(n0 + b_n_row) * k + b_k_off));
  __syncthreads();

  for (int k0 = 0; k0 < k; k0 += kStageK) {
    // Each wave consumes its 64x32 quadrant: eight m16n16k32 MMACs per kk.
#pragma unroll
    for (int kk = 0; kk < kStageK; kk += kTileK) {
      // Iteration 7: explicit 64-bit fragment loads (bit-identical values,
      // zero reassembly VALU) instead of the library's byte-wise loads; the
      // per-lane (row, k-group) position reproduces du_load_matrix_sync.
      // Iteration 20: the swizzled loader reads each lane's eight bytes from
      // swizzle_group(row, c) instead of the padded 80-byte stride (same
      // values, 16 distinct bank-pairs per 16-lane group).
      load_frag8_swizzled(b_frag0, b_tile, lane, local_col, kk);
      load_frag8_swizzled(b_frag1, b_tile, lane, local_col + kTileN, kk);
      load_frag8_swizzled(a_frag0, a_tile, lane, local_row, kk);
      load_frag8_swizzled(a_frag1, a_tile, lane, local_row + kTileM, kk);
      du_mma_sync(acc00, a_frag0, b_frag0, acc00);
      du_mma_sync(acc01, a_frag0, b_frag1, acc01);
      du_mma_sync(acc10, a_frag1, b_frag0, acc10);
      du_mma_sync(acc11, a_frag1, b_frag1, acc11);
      load_frag8_swizzled(a_frag0, a_tile, lane, local_row + 2 * kTileM, kk);
      load_frag8_swizzled(a_frag1, a_tile, lane, local_row + 3 * kTileM, kk);
      du_mma_sync(acc20, a_frag0, b_frag0, acc20);
      du_mma_sync(acc21, a_frag0, b_frag1, acc21);
      du_mma_sync(acc30, a_frag1, b_frag0, acc30);
      du_mma_sync(acc31, a_frag1, b_frag1, acc31);
    }

    // Single-buffered re-stage of the next 64-K slice: barrier (every thread
    // finished reading the current stage), coalesced vector loads into the
    // same buffers, then barrier (all loads visible before the next stage).
    const int next_k = k0 + kStageK;
    if (next_k < k) {
      __syncthreads();
      store_chunk16_swizzled(
          a_tile, stage_row, stage_col >> 4,
          *reinterpret_cast<const int4*>(
              x_q + static_cast<int64_t>(m0 + stage_row) * k + next_k +
              stage_col));
      store_chunk16_swizzled(
          a_tile, stage_row + kBlockM / 2, stage_col >> 4,
          *reinterpret_cast<const int4*>(
              x_q + static_cast<int64_t>(m0 + stage_row + kBlockM / 2) * k +
              next_k + stage_col));
      store_chunk16_swizzled(
          b_tile, b_n_row, b_k_off >> 4,
          *reinterpret_cast<const int4*>(
              weight + static_cast<int64_t>(n0 + b_n_row) * k + next_k +
              b_k_off));
      __syncthreads();
    }
  }

  // -------------------------------------------------------------------------
  // Iteration 6 fused epilogue: the dead A/B LDS tiles (12,288 B after the
  // iteration-20 dense swizzle) are reused as a two-phase bf16 staging buffer
  // (128 rows x 72 B, 32 useful bf16 per row; the top 1,024 B spill into the
  // dead b_tile region, never read after the k-loop's final barrier).
  // Phase A: waves 0 and 2 (wave_col == 0) stage the left 128x32 half-tile
  // (cols [0,32)); every thread then stores it coalesced (8-byte chunks).
  // Phase B: waves 1 and 3 (wave_col == 1) stage the right half-tile
  // (cols [32,64)) and it is stored the same way. LDS is 13,056 B/block so
  // occupancy remains 3 blocks/CU (VGPR-limited); the k-loop pipeline is
  // untouched.
  // -------------------------------------------------------------------------
  __syncthreads();  // every thread is done reading a_tile / b_tile

  const int quad_row = wave_row * kBlockM / 2;  // 0 or 64 (staging row base)
  const int wcol0 = wave_col * kStageHalfBf16;  // 0 or 32 (weight_scale base)

  if (wave_col == 0) {
    // Phase A: left half-tile, cols [0,32).
    stage_prefill_fragment(acc00, a_tile, x_scale_l, w_scale_l, quad_row,
                           0, wcol0, m0, m, lane);
    stage_prefill_fragment(acc10, a_tile, x_scale_l, w_scale_l,
                           quad_row + kTileM, 0, wcol0, m0, m, lane);
    stage_prefill_fragment(acc20, a_tile, x_scale_l, w_scale_l,
                           quad_row + 2 * kTileM, 0, wcol0, m0, m, lane);
    stage_prefill_fragment(acc30, a_tile, x_scale_l, w_scale_l,
                           quad_row + 3 * kTileM, 0, wcol0, m0, m, lane);
    stage_prefill_fragment(acc01, a_tile, x_scale_l, w_scale_l, quad_row,
                           kTileN, wcol0, m0, m, lane);
    stage_prefill_fragment(acc11, a_tile, x_scale_l, w_scale_l,
                           quad_row + kTileM, kTileN, wcol0, m0, m, lane);
    stage_prefill_fragment(acc21, a_tile, x_scale_l, w_scale_l,
                           quad_row + 2 * kTileM, kTileN, wcol0, m0, m, lane);
    stage_prefill_fragment(acc31, a_tile, x_scale_l, w_scale_l,
                           quad_row + 3 * kTileM, kTileN, wcol0, m0, m, lane);
  }
  __syncthreads();  // half-tile staged
  store_staged_half_tile(a_tile, out, m0, n0, 0, m, n, tid);
  __syncthreads();  // staging reads done before phase B overwrites a_tile

  if (wave_col == 1) {
    // Phase B: right half-tile, cols [32,64).
    stage_prefill_fragment(acc00, a_tile, x_scale_l, w_scale_l, quad_row,
                           0, wcol0, m0, m, lane);
    stage_prefill_fragment(acc10, a_tile, x_scale_l, w_scale_l,
                           quad_row + kTileM, 0, wcol0, m0, m, lane);
    stage_prefill_fragment(acc20, a_tile, x_scale_l, w_scale_l,
                           quad_row + 2 * kTileM, 0, wcol0, m0, m, lane);
    stage_prefill_fragment(acc30, a_tile, x_scale_l, w_scale_l,
                           quad_row + 3 * kTileM, 0, wcol0, m0, m, lane);
    stage_prefill_fragment(acc01, a_tile, x_scale_l, w_scale_l, quad_row,
                           kTileN, wcol0, m0, m, lane);
    stage_prefill_fragment(acc11, a_tile, x_scale_l, w_scale_l,
                           quad_row + kTileM, kTileN, wcol0, m0, m, lane);
    stage_prefill_fragment(acc21, a_tile, x_scale_l, w_scale_l,
                           quad_row + 2 * kTileM, kTileN, wcol0, m0, m, lane);
    stage_prefill_fragment(acc31, a_tile, x_scale_l, w_scale_l,
                           quad_row + 3 * kTileM, kTileN, wcol0, m0, m, lane);
  }
  __syncthreads();
  store_staged_half_tile(a_tile, out, m0, n0, kStageHalfBf16, m, n, tid);
}

// ---------------------------------------------------------------------------
// Generic scalar fallback: one thread per output element. Exact int32 dot
// over K, then fused x_scale * weight_scale, then bf16 store. Handles every
// unmatched (m, n, k), including all small-M API cases. Reads the one-time
// n-major packed [N, K] layout when (k, n) == (2048, 6144), and the
// bootstrap identity [K, N] row-major layout for every other (k, n).
// ---------------------------------------------------------------------------
__global__ __launch_bounds__(256) void w8a8_gemm_scalar_fallback_kernel(
    const int8_t* __restrict__ x_q,
    const int8_t* __restrict__ weight,
    const float* __restrict__ x_scale,
    const float* __restrict__ weight_scale,
    bf16_t* __restrict__ out,
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
  int32_t acc = 0;
  const int8_t* a_row = x_q + static_cast<int64_t>(row) * k;
  const bool nmajor_b = (k == kTargetK && n == kTargetN);
  const int8_t* b_col =
      weight + (nmajor_b ? static_cast<int64_t>(col) * k : col);
  const int64_t b_stride = nmajor_b ? 1 : n;
  for (int kk = 0; kk < k; ++kk) {
    acc += static_cast<int32_t>(a_row[kk]) *
           static_cast<int32_t>(b_col[static_cast<int64_t>(kk) * b_stride]);
  }
  const float scaled =
      static_cast<float>(acc) * x_scale[row] * weight_scale[col];
  out[linear] = __float2bfloat16(scaled);
}

// ---------------------------------------------------------------------------
// Bootstrap weight packing for every (k, n) except the assigned shape:
// identity device-to-device copy (int4 vectors plus a scalar tail for byte
// counts not a multiple of 16). The assigned (2048, 6144) uses the one-time
// n-major transpose kernel below instead. Scales are always copied identity.
// ---------------------------------------------------------------------------
__global__ __launch_bounds__(256) void w8a8_pack_identity_vec_kernel(
    const int8_t* __restrict__ raw,
    int8_t* __restrict__ packed,
    int64_t vec_count) {
  const int64_t vec =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (vec < vec_count) {
    reinterpret_cast<int4*>(packed)[vec] =
        reinterpret_cast<const int4*>(raw)[vec];
  }
}

__global__ __launch_bounds__(256) void w8a8_pack_identity_tail_kernel(
    const int8_t* __restrict__ raw,
    int8_t* __restrict__ packed,
    int64_t start,
    int64_t count) {
  const int64_t idx =
      start + static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < count) {
    packed[idx] = raw[idx];
  }
}

// One-time n-major transpose for the assigned shape (k, n) == (2048, 6144):
// packed[n*K + k] = raw[k*N + n]. Each thread owns one 16x16 (k-chunk, n-chunk)
// block and writes its 16 transposed 16-byte rows:
//   packed[(nc*16 + j2)*K + kc*16 + j1] = raw[(kc*16 + j1)*N + nc*16 + j2],
// for j1, j2 in [0, 16). This is the exact block transpose with full K*N
// coverage (the previous one-int4-per-block version left 15/16 of the packed
// rows untouched and copied each chunk untransposed). Consecutive threads read
// consecutive n chunks of one k row (coalesced 16-byte global reads); writes
// are strided by 16*K (one-time prep, never timed). Requires k % 16 == 0 and
// n % 16 == 0, true for the exact shape.
__global__ __launch_bounds__(256) void w8a8_pack_nmajor_kernel(
    const int8_t* __restrict__ raw,
    int8_t* __restrict__ packed,
    int k,
    int n) {
  const int n_chunks = n / 16;
  const int k_chunks = k / 16;
  const int total = n_chunks * k_chunks;
  const int tid = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (tid >= total) {
    return;
  }
  const int kc = tid / n_chunks;
  const int nc = tid - kc * n_chunks;
  const int64_t src_base =
      static_cast<int64_t>(kc) * 16 * n + static_cast<int64_t>(nc) * 16;
  const int64_t dst_base =
      static_cast<int64_t>(nc) * 16 * k + static_cast<int64_t>(kc) * 16;
#pragma unroll
  for (int j2 = 0; j2 < 16; ++j2) {
    __align__(16) int8_t chunk[16];
#pragma unroll
    for (int j1 = 0; j1 < 16; ++j1) {
      chunk[j1] = raw[src_base + static_cast<int64_t>(j1) * n + j2];
    }
    *reinterpret_cast<int4*>(packed + dst_base + static_cast<int64_t>(j2) * k) =
        *reinterpret_cast<const int4*>(chunk);
  }
}

__global__ __launch_bounds__(256) void w8a8_pack_scale_identity_kernel(
    const float* __restrict__ raw,
    float* __restrict__ packed,
    int count) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < count) {
    packed[idx] = raw[idx];
  }
}

}  // namespace

// ---------------------------------------------------------------------------
// Stable host launch symbols consumed by csrc/bindings.cpp.
// Both launchers are pure dispatch: no allocation, no packing, no
// synchronization, no default-stream launch; they run on the caller-provided
// PyTorch HIP stream and are CUDA/HIP-Graph safe.
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
  (void)workspace;        // no split-K: the GEMM does not use the workspace
  (void)workspace_bytes;
  auto* out_bf16 = reinterpret_cast<bf16_t*>(out);

  // Explicit dispatch. The assigned shape (M=4096, N=6144, K=2048) takes the
  // n-major-packed-B DUMMA 128x64 path; every other (m, n, k) - including
  // small-M API cases and the paired M=2/M=16 shapes with the same (K, N) -
  // takes the scalar fallback, which decodes the n-major packed layout for
  // (k, n) == (2048, 6144) and the identity layout otherwise.
  if (m == kTargetM && n == kTargetN && k == kTargetK) {
    const dim3 grid(kTargetN / kBlockN, kTargetM / kBlockM);
    const dim3 block(kBlockThreads);
    hipLaunchKernelGGL(w8a8_dumma_prefill_128x64x64_identity_kernel,
                       grid, block, 0, stream,
                       a, b, x_scale, weight_scale, out_bf16, m, n, k);
  } else {
    constexpr int kBlock = 256;
    const int64_t total = static_cast<int64_t>(m) * n;
    const dim3 grid(static_cast<unsigned>((total + kBlock - 1) / kBlock));
    const dim3 block(kBlock);
    hipLaunchKernelGGL(w8a8_gemm_scalar_fallback_kernel,
                       grid, block, 0, stream,
                       a, b, x_scale, weight_scale, out_bf16, m, n, k);
  }
  (void)hipGetLastError();
}

extern "C" void launch_pack_w8a8_weight(
    const int8_t* raw_weight,
    const float* weight_scale,
    int8_t* packed_weight,
    float* packed_weight_scale,
    int k,
    int n,
    hipStream_t stream) {
  constexpr int kBlock = 256;

  // The assigned shape (2048, 6144) is packed once into the n-major
  // [N, K] layout (packed[n*K + k] = raw[k*N + n]) so the DUMMA kernel can
  // load col_major B fragments as plain 8-byte LDS reads. Every other (k, n)
  // keeps the bootstrap identity [K, N] device-to-device copy.
  if (k == kTargetK && n == kTargetN) {
    const int64_t chunks = (static_cast<int64_t>(k) / 16) *
                           (static_cast<int64_t>(n) / 16);
    const dim3 nmajor_grid(
        static_cast<unsigned>((chunks + kBlock - 1) / kBlock));
    const dim3 block(kBlock);
    hipLaunchKernelGGL(w8a8_pack_nmajor_kernel,
                       nmajor_grid, block, 0, stream,
                       raw_weight, packed_weight, k, n);
  } else {
    // Bootstrap identity weight copy (int4 vectors plus scalar tail).
    const int64_t weight_count = static_cast<int64_t>(k) * n;
    const int64_t vec_count = weight_count / static_cast<int64_t>(sizeof(int4));
    if (vec_count > 0) {
      const dim3 weight_grid(
          static_cast<unsigned>((vec_count + kBlock - 1) / kBlock));
      const dim3 block(kBlock);
      hipLaunchKernelGGL(w8a8_pack_identity_vec_kernel,
                         weight_grid, block, 0, stream,
                         raw_weight, packed_weight, vec_count);
    }
    const int64_t copied_bytes = vec_count * static_cast<int64_t>(sizeof(int4));
    if (copied_bytes < weight_count) {
      const int64_t tail_count = weight_count - copied_bytes;
      const dim3 tail_grid(
          static_cast<unsigned>((tail_count + kBlock - 1) / kBlock));
      const dim3 block(kBlock);
      hipLaunchKernelGGL(w8a8_pack_identity_tail_kernel,
                         tail_grid, block, 0, stream,
                         raw_weight, packed_weight, copied_bytes, weight_count);
    }
  }

  // Identity scale copy for every (k, n).
  const dim3 scale_grid(static_cast<unsigned>((n + kBlock - 1) / kBlock));
  const dim3 block(kBlock);
  hipLaunchKernelGGL(w8a8_pack_scale_identity_kernel,
                     scale_grid, block, 0, stream,
                     weight_scale, packed_weight_scale, n);
  (void)hipGetLastError();
}
// @@end
// @@end
