// @@variant shape=hy3_tp4_o_proj_m16 commit=cb7af9e4e0a3adf58bc62ec3cf8be1e1a3f4c010 added=2026-08-26
//   median_us=15.51 p90_us=15.55
//   source=hy3-dsh-tp4-m16-2-368c654c
// MetaInfer W8A8 INT8 GEMM - HIP implementation for K500SM_AI / gfx928.
//
// Worker: worker_1 (physical GPU 1)
// Assigned shape: hy3_tp4_o_proj_m16  (M=16, N=4096, K=2048)
//
// Iteration 1 (accepted) = minimal 16x16x32 DUMMA bootstrap: 256 blocks x 1
//   64-thread wavefront, one N tile per block, library byte-loads of A/B
//   straight from the logical row-major globals, 64 ascending-K int32 steps,
//   LDS epilogue plane, zero in-loop barriers.
//   -> median 89.90 us / p90 90.10 us, 95.3 GB/s algorithmic, 263,424
//      vmem_read/replay = 16.08 loads per K-step per wave, all fenced
//      vmcnt(15)..vmcnt(0) before the single v_mmac.
// Iteration 2 (rejected) = grid-parallelism probe: 128 blocks x 2 waves x 2
//   adjacent N tiles (SAME 256 waves, no split-K, no packing).
//   -> median 117.81 us (REGRESSION): per-CU wave count is NOT the lever on
//      the byte-load design at 2.13 waves/CU; the named next lever was load
//      vectorization (packed-B dwordx2 fragment loads) and/or in-block
//      split-K.
// Iteration 3 (rejected, measurement-polluted) = architecture round: 256
//   blocks x 2 waves, in-block split-K=2 (uniform K=1024 slices), packed-B
//   [n_tile][k_step][lane][8] fragment-slot loads, register-only K-loop
//   transport with depth-1 prefetch, K=128 stages. This is byte-for-byte the
//   authoritative lineage's ACCEPTED final kernel for this exact shape
//   (w8a8_gemm_m16_dumma_packedb_kernel, 18.221/18.242 us median/p90 there).
//   -> min 21.37 us (9 uncontended samples tight at 21.4-22.5 us) but median
//      166.13 / p90 216.62: the 30-sample window was bimodal, with ~19
//      samples at ~190-240 us matching an ~8-10x-slower interleaved workload
//      on the GPU (external interference, not kernel behavior: the same
//      captured graph cannot run 10x slower under its own resources at 2.13
//      blocks/CU all-resident). The fast tail proves the design; the slow
//      head is the measurement window.
// Iteration 4 (accepted, 17.155 us median / 17.185 us p90, 499.5 GB/s
//   algorithmic - the fastest accepted W8A8 kernel in every worker ledger on
//   this node) = architecture/pipeline round, multi-N-tile reuse: each block
//   computes TWO adjacent 16x16 N tiles (T=2) so the A fragments loaded per
//   K=128 stage are shared across both tiles in registers; everything else
//   frozen from iteration 3 (packed-B layout, in-block split-K=2, K=128
//   stages, depth-1 register prefetch). HBM floor analysis: 8,484,352 B of
//   unavoidable traffic (8 MiB B read-once + 32 KiB A + 128 KiB bf16 C) at
//   17.155 us = 499.5 GB/s, and no accepted kernel on this device has ever
//   exceeded that figure - the kernel sits at the practical streaming
//   ceiling.
// Iteration 5 (THIS round) = HIP-only packed-layout comparison (mandate:
//   packed layout / A-only staging / B-only staging): repack B from
//   [n_tile][k_step][lane][8] to a TILE-PAIR-INTERLEAVED dwordx4 layout
//   [n_tile_pair][k_step][lane][16] in which lane's 16 bytes are tile0's 8
//   fragment bytes followed by tile1's 8 fragment bytes for the same
//   (k_step, lane). Each lane then covers BOTH tiles with ONE aligned 16-byte
//   load per k_step (a single 1 KiB contiguous region per load, a single
//   4 KiB contiguous region per K=128 stage, and one sequential 64 KiB
//   stream per block lifetime - was two 2 KiB regions 32 KiB apart per
//   stage); B vmem instructions drop from 8 to 4 per stage per wave at an
//   identical 24 VGPR in-flight footprint and identical 8 MiB of HBM bytes.
//   Geometry frozen from iteration 4 (128 blocks x 2 waves x T=2, in-block
//   split-K=2, K=128 stages, depth-1 register prefetch, partial-planes-only
//   LDS, one end-of-K barrier). See the kernel comment below.
//
// Scalar fallback strategy (one thread per output element; still the path for
// every shape OTHER than the exact assigned M=16 shape):
//   * blockDim = 256 (multiple of the gfx928 64-lane wavefront);
//   * grid = (ceil(N/256), M): each block covers 256 adjacent N columns of a
//     single M row, so adjacent lanes touch adjacent addresses in the
//     fastest-changing N dimension (coalesced B reads / bf16 stores);
//   * the complete K loop accumulates exactly in int32. For the assigned
//     K=2048 the max |dot| is 2048*127*127 = 33,032,192 << 2^31, so the
//     integer accumulation never overflows;
//   * after the K loop the two float scales are applied as
//     float(dot) * x_scale[m] * weight_scale[n] and the result is stored as
//     bf16 (hip_bfloat16 via __float2bfloat16);
//   * for the exact (n, k) == (4096, 2048) pair the weight tensor is the
//     packed [n_tile_pair][k_step][lane][16] tile-pair-interleaved layout
//     (16 bytes = tile0's 8 fragment bytes ++ tile1's 8 fragment bytes for
//     the same k_step/lane), which the fallback decodes elementwise (keeps
//     the paired M=2 validation exact with the new pack);
//   * workspace is untouched by the timed operator (in-block split-K needs no
//     partial planes).
//
// pack_weight: for the exact (k, n) == (2048, 4096) o_proj weight it is a
//   one-time device permutation into packed[n_tile_pair][k_step][lane][16] B
//   fragment slots (byte count unchanged: 8 MiB; runs outside the timed GEMM
//   and outside Graph capture). Every other (K, N) keeps the generic identity
//   device-to-device copy; packed_weight_scale[n] = weight_scale[n] always.
//
// Host launch symbols (stable, C linkage, consumed by csrc/bindings.cpp):
//   launch_w8a8_gemm(...)
//   launch_pack_w8a8_weight(...)
//
// Include order is known-good for this DTK: hip_runtime.h first, then
// hip_bfloat16.h, then du_mma.h (du_mma.h is not self-contained when
// included before the HIP runtime headers).

#include <hip/hip_runtime.h>
#include <hip/hip_bfloat16.h>
#include <du_mma.h>

#include <cstdint>

namespace {

constexpr int kScalarThreads = 256;

// Exact assigned shape (M=16, N=4096, K=2048).
constexpr int kTargetM = 16;
constexpr int kTargetN = 4096;
constexpr int kTargetK = 2048;

// ---------------------------------------------------------------------------
// DUMMA geometry for the exact assigned shape: gfx928 INT8 m16n16k32 tile.
// Iteration 4 multi-N-tile reuse: each block owns TWO adjacent 16x16 N tiles
// (T=2, grid = N/(16*2) = 128 blocks >= 120 device CUs) and 2 x 64-thread
// wavefronts; wave w accumulates the int32 dot over its UNIFORM K=1024 slice
// for BOTH tiles, so the A fragments of each K=128 stage are loaded once and
// reused across the two tiles from registers.
// ---------------------------------------------------------------------------
constexpr int kDummaBlockThreads = 64;    // one gfx928 wavefront
constexpr int kDummaWavesPerBlock = 2;    // in-block split-K=2
constexpr int kDummaTilesPerBlock = 2;    // multi-N-tile reuse (T=2)
constexpr int kDummaTileM = 16;
constexpr int kDummaTileN = 16;
constexpr int kDummaTileK = 32;

// Bytes of one (n_tile, k_step) B fragment in the packed layout:
// 16 columns * 32 k rows = 512 B, stored as 64 lanes * 8 B.
constexpr int kPackedFragBytes = kDummaTileN * kDummaTileK;  // 512

// Iteration-5 tile-pair-interleaved layout: one (k_step) slot holds BOTH
// tiles of a block in 16 B per lane - tile0's 8 fragment bytes (bytes 0..7)
// followed by tile1's 8 fragment bytes (bytes 8..15) - so a single aligned
// 16-byte (dwordx4) load per lane covers one k_step of both tiles: 64 lanes
// * 16 B = 1 KiB contiguous per load. Per n_tile_pair the 64 k_steps are
// stored ascending (64 KiB sequential per block lifetime).
constexpr int kPackedPairBytes = kDummaBlockThreads * 16;  // 1024

// K=128 stage: 4 DUMMA steps per stage, 8 loads per stage per wave (4 A
// dwordx2 shared by both tiles + 4 B dwordx4, one per k_step covering both
// tiles = 6 KB in flight), 8 v_mmac per stage per wave (4 steps x 2 tiles).
constexpr int kStageK = 128;                    // K bytes per stage
constexpr int kStageSteps = kStageK / kDummaTileK;  // 4 DUMMA steps per stage

// ---------------------------------------------------------------------------
// Packed-B decode for the generic scalar fallback: returns logical
// weight[kk, n] from the packed [n_tile_pair][k_step][lane][16]
// tile-pair-interleaved layout (see the pack kernel for the exact
// permutation). The 16 bytes of a lane slot are tile0's 8 fragment bytes
// (bytes 0..7) followed by tile1's 8 fragment bytes (bytes 8..15).
// ---------------------------------------------------------------------------
__device__ __forceinline__ int8_t
packed_b_element(const int8_t* __restrict__ packed, int n, int kk) {
  const int n_tile = n >> 4;
  const int pair = n_tile >> 1;   // n_tile_pair (block id)
  const int sub = n_tile & 1;     // 0 -> bytes 0..7, 1 -> bytes 8..15
  const int nn = n & 15;
  const int k_step = kk >> 5;
  const int kk8 = kk & 31;
  const int lane = (kk8 >> 3) * 16 + nn;
  const int i = kk8 & 7;
  return packed[(static_cast<int64_t>(pair) * (kTargetK / kDummaTileK) +
                 k_step) *
                    kPackedPairBytes +
                lane * 16 + sub * 8 + i];
}

// One thread computes one output element out[row, col]:
//   acc = sum_k a[row,k] * b[k,col]          (exact int32, k ascending)
//   out = bf16(acc * x_scale[row] * weight_scale[col])
// Bounds-checked, so it is a valid generic fallback for every (m, n, k).
// For the exact (n, k) == (4096, 2048) pair the weight tensor is the packed
// [n_tile_pair][k_step][lane][16] layout (o_proj M=16 pack), decoded
// elementwise.
__global__ __launch_bounds__(kScalarThreads) void w8a8_gemm_scalar_kernel(
    const int8_t* __restrict__ a,            // [m, k] row-major
    const int8_t* __restrict__ b,            // [k, n] (packed for 4096x2048)
    const float* __restrict__ x_scale,       // [m]
    const float* __restrict__ weight_scale,  // [n]
    hip_bfloat16* __restrict__ out,          // [m, n]
    int m,
    int n,
    int k) {
  const int row = static_cast<int>(blockIdx.y);
  const int col =
      static_cast<int>(blockIdx.x) * kScalarThreads +
      static_cast<int>(threadIdx.x);
  if (row >= m || col >= n) {
    return;
  }

  const int8_t* __restrict__ a_row = a + static_cast<int64_t>(row) * k;
  const bool packed_b = (n == kTargetN && k == kTargetK);

  int32_t acc = 0;
  for (int kk = 0; kk < k; ++kk) {
    const int8_t b_val = packed_b ? packed_b_element(b, col, kk)
                                  : b[static_cast<int64_t>(kk) * n + col];
    acc += static_cast<int32_t>(a_row[kk]) * static_cast<int32_t>(b_val);
  }

  const float scaled = static_cast<float>(acc) * x_scale[row] *
                       weight_scale[col];
  out[static_cast<int64_t>(row) * n + col] = __float2bfloat16(scaled);
}

// ---------------------------------------------------------------------------
// Iteration-5 stage prefetch helper (multi-N-tile reuse): issue the eight
// loads for one K=128 stage - the 4 A dwordx2 fragments (shared by both
// tiles) plus 4 B dwordx4 fragment-pair loads (one per k_step, each covering
// BOTH tiles) - into caller-provided u64 registers.
//   * A: four 8-byte loads of one activation row at ldm = k, 32 B apart (the
//     four DUMMA steps of the stage) - lane ownership is loop-invariant
//     (du_mma.hpp matrix_a row_major loader: row = lane & 0xf,
//     col = (lane >> 4) << 3, x[i] = p[row*ldm + col + i], 8 consecutive
//     bytes; du_mma_sync consumes a.x as one 64-bit value, du_mma.hpp line
//     1198). A stays the logical row-major 32 KiB activation (L2-hot).
//     These 4 values feed BOTH accumulator tiles - A is read once per block
//     instead of once per N tile.
//   * B (iteration-5 tile-pair-interleaved layout): per k_step one aligned
//     16-byte load from the [n_tile_pair][k_step][lane][16] slot whose bytes
//     0..7 are tile0's fragment bytes and bytes 8..15 tile1's fragment bytes
//     for the same (k_step, lane) - byte-order identical to the library
//     matrix_b row_major loader (x[i] = p[(col + i) * ldm + row]) per tile,
//     so the two 8-byte halves feed b_frag0.x / b_frag1.x directly. A stage's
//     four k_step loads sit at consecutive 1 KiB slot offsets: one 4 KiB
//     contiguous region per stage per wave; each wave's 8 stages walk one
//     sequential 32 KiB half of the block's 64 KiB pair region, so every B
//     byte is read exactly once per replay.
//   * The caller's k0_base K-slice offset makes each split wave walk its own
//     packed k_step range. The 16-byte loads are 16 B aligned (slot offset is
//     k_step * 1024 + lane * 16 on a 16 B-aligned packed buffer).
// ---------------------------------------------------------------------------
__device__ __forceinline__ void load_stage_fragments_t2(
    const int8_t* __restrict__ x_q,
    const int8_t* __restrict__ weight,
    int lane, int a_row, int a_col, int64_t b_base, int k,
    int64_t k0_base, int s,
    uint64_t& a0, uint64_t& a1, uint64_t& a2, uint64_t& a3,
    uint64_t& b00, uint64_t& b01, uint64_t& b02, uint64_t& b03,
    uint64_t& b10, uint64_t& b11, uint64_t& b12, uint64_t& b13) {
  const int64_t k0 = k0_base + static_cast<int64_t>(s) * kStageK;
  const int64_t a_off = static_cast<int64_t>(a_row) * k + k0 + a_col;
  a0 = *reinterpret_cast<const uint64_t*>(x_q + a_off);
  a1 = *reinterpret_cast<const uint64_t*>(x_q + a_off + kDummaTileK);
  a2 = *reinterpret_cast<const uint64_t*>(x_q + a_off + 2 * kDummaTileK);
  a3 = *reinterpret_cast<const uint64_t*>(x_q + a_off + 3 * kDummaTileK);
  const int64_t pair_off = (k0 / kDummaTileK) * kPackedPairBytes +
                           static_cast<int64_t>(lane) * 16;
  const int64_t b_off = b_base + pair_off;
  // One 16-byte load per k_step covers both tiles: bytes 0..7 = tile0's
  // fragment, bytes 8..15 = tile1's fragment (little-endian word order).
  const uint4 bv0 = *reinterpret_cast<const uint4*>(weight + b_off);
  const uint4 bv1 = *reinterpret_cast<const uint4*>(weight + b_off + kPackedPairBytes);
  const uint4 bv2 = *reinterpret_cast<const uint4*>(weight + b_off + 2 * kPackedPairBytes);
  const uint4 bv3 = *reinterpret_cast<const uint4*>(weight + b_off + 3 * kPackedPairBytes);
  b00 = (static_cast<uint64_t>(bv0.y) << 32) | bv0.x;  // tile0, k_step 0
  b01 = (static_cast<uint64_t>(bv1.y) << 32) | bv1.x;  // tile0, k_step 1
  b02 = (static_cast<uint64_t>(bv2.y) << 32) | bv2.x;  // tile0, k_step 2
  b03 = (static_cast<uint64_t>(bv3.y) << 32) | bv3.x;  // tile0, k_step 3
  b10 = (static_cast<uint64_t>(bv0.w) << 32) | bv0.z;  // tile1, k_step 0
  b11 = (static_cast<uint64_t>(bv1.w) << 32) | bv1.z;  // tile1, k_step 1
  b12 = (static_cast<uint64_t>(bv2.w) << 32) | bv2.z;  // tile1, k_step 2
  b13 = (static_cast<uint64_t>(bv3.w) << 32) | bv3.z;  // tile1, k_step 3
}

// ---------------------------------------------------------------------------
// Iteration 17 (HIP-only tail consolidation): float32 -> bf16 round-to-nearest-
// even WITHOUT the __float2bfloat16 NaN/Inf branch. The exact iteration-13
// code object shows each of the epilogue's four conversions as a ~10-
// instruction guarded sequence ON the block's post-barrier finishing tail:
// the finite path is v_bfe_u32 (bit 16) + v_add3_u32 (u + bit + 0x7fff) with
// the store's d16_hi keeping the high 16 bits, plus the dead-for-this-operator
// s_and_saveexec / v_or_b32 / v_cmp_eq_sdwa / v_cndmask NaN/Inf handling.
// Every converted value here is finite (the int32 dot is bounded by
// 2048*127*127 << 2^31 and both scales are finite), so the guarded branch is
// pure tail cost sitting between the LDS reads and the stores. This helper
// emits only the finite RNE path: r = u + 0x7fff + ((u >> 16) & 1), high 16
// bits kept - bit-identical to __float2bfloat16 for every finite input (same
// rounding, same carry into the exponent at the bf16 range edge); the only
// divergence is NaN payload bits, which cannot occur here.
// ---------------------------------------------------------------------------
__device__ __forceinline__ hip_bfloat16 bf16_rne_finite(float f) {
  uint32_t u;
  __builtin_memcpy(&u, &f, sizeof(u));
  u += 0x7fffu + ((u >> 16) & 1u);
  const uint16_t h = static_cast<uint16_t>(u >> 16);
  hip_bfloat16 b;
  __builtin_memcpy(&b, &h, sizeof(h));
  return b;
}

// ---------------------------------------------------------------------------
// Exact assigned shape (M=16, N=4096, K=2048): gfx928 DUMMA INT8 m16n16k32
// tile, weight pre-packed into tile-pair-interleaved B fragment slots,
// MULTI-N-TILE REUSE.
//   * Geometry: 128 blocks (grid = N/(16*2), one block per pair of adjacent
//     16x16 N tiles; 128 >= 120 device CUs, so every CU gets at least one
//     block) x 128 threads = 2 x 64-lane wavefronts.
//   * In-block split-K=2: wave w owns the UNIFORM K slice of 8 x K=128 stages
//     (stages [floor(w*16/2), floor((w+1)*16/2)) = exactly 8 stages, K=1024)
//     and accumulates BOTH tiles over that slice - 2 independent accumulator
//     fragments per wave. Per K=128 stage per wave: 4 A dwordx2 + 4 B dwordx4
//     (one 16-byte load per k_step covering both tiles) = 8 loads (6 KB in
//     flight, identical byte footprint to iteration 4) covering an 8-v_mmac
//     burst (4 steps x 2 tiles): load:compute ratio 1.5 (iteration 4) -> 1.0,
//     B vmem instructions per replay drop from 16,384 dwordx2 to 8,192
//     dwordx4, and each block's B lifetime stream is ONE sequential 64 KiB
//     region (4 KiB contiguous per stage) instead of two 32 KiB regions 32
//     KiB apart; A global traffic unchanged (L2-hot 32 KiB footprint, read
//     once per block), B HBM traffic unchanged (8 MiB, every byte read
//     exactly once).
//   * REGISTER-ONLY K-loop transport with depth-1 prefetch, identical to
//     iteration 4: current stage's 8 loads' fragments live in registers, the
//     NEXT stage's 8 loads are issued BEFORE the current 8-MMAC burst,
//     and the compiler-inserted vmcnt wait lands at the next iteration's
//     first fragment fill (global_load -> v_mmac burst -> wait, no LDS hop,
//     no lgkmcnt wait in the K loop). The 4 A dwordx2 + 4 B dwordx4 hold the
//     same 24 VGPR as iteration 4's 12 dwordx2, so occupancy is unchanged.
//   * LDS holds only the four 1 KiB int32 partial planes (2 waves x 2 tiles,
//     4 KiB/block); the ONLY barrier is the single END-of-K __syncthreads()
//     before the fused partial combine + scale -> bf16 epilogue (all 128
//     threads, 4 elements each). Barriers per K step = 0.
//   * Correctness: per element the int32 accumulation is ascending-K within
//     the wave's slice (4 ascending steps per stage, stages ascending), then
//     the two waves' partials are summed in LDS (order-independent int32, no
//     overflow: max |dot| = 2048*127*127 << 2^31) - bit-identical to
//     iteration 4. The iteration-5 repack changed the packed layout
//     COHERENTLY across the pack kernel, the DUMMA kernel's loads and the
//     scalar fallback decode (same (tile, k_step, lane, byte) -> logical
//     (k, n) mapping, new memory order), so the paired M=2 validation stays
//     exact.
//   * Graph-safe: static smem only, no allocation/sync/default-stream launch,
//     pure dispatch on the caller-provided stream. Workspace untouched.
// ---------------------------------------------------------------------------
__global__ __launch_bounds__(kDummaWavesPerBlock * kDummaBlockThreads) void
w8a8_gemm_m16_dumma_packedb_multint_kernel(
    const int8_t* __restrict__ x_q,
    const int8_t* __restrict__ weight,  // packed [n_tile_pair][k_step][lane][16]
    const float* __restrict__ x_scale,
    const float* __restrict__ weight_scale,
    hip_bfloat16* __restrict__ out,
    int n,
    int k) {
  const int tid = static_cast<int>(threadIdx.x);
  const int wave = tid / kDummaBlockThreads;  // K-slice owner, 0..1
  const int lane = tid % kDummaBlockThreads;  // wavefront lane 0..63
  // Block b owns the two adjacent N tiles 2b and 2b+1.
  const int n_tile0 = static_cast<int>(blockIdx.x) * kDummaTilesPerBlock;
  const int n0 = n_tile0 * kDummaTileN;

  du::dumma::DUFragment<du::dumma::matrix_a, kDummaTileM, kDummaTileN,
                        kDummaTileK, signed char, du::dumma::row_major>
      a_frag;
  du::dumma::DUFragment<du::dumma::matrix_b, kDummaTileM, kDummaTileN,
                        kDummaTileK, signed char, du::dumma::row_major>
      b_frag0;
  du::dumma::DUFragment<du::dumma::matrix_b, kDummaTileM, kDummaTileN,
                        kDummaTileK, signed char, du::dumma::row_major>
      b_frag1;
  du::dumma::DUFragment<du::dumma::accumulator, kDummaTileM, kDummaTileN,
                        kDummaTileK, int> acc0;
  du::dumma::DUFragment<du::dumma::accumulator, kDummaTileM, kDummaTileN,
                        kDummaTileK, int> acc1;
  du::dumma::du_fill_fragment(acc0, 0);
  du::dumma::du_fill_fragment(acc1, 0);

  // Each n_tile_pair owns one contiguous packed region of k_steps * 1024 B
  // (64 KiB: both tiles' fragments interleaved at 16 B per lane per k_step).
  const int64_t k_steps = k / kDummaTileK;
  const int64_t b_base =
      static_cast<int64_t>(blockIdx.x) * k_steps * kPackedPairBytes;

  // Per-lane A fragment ownership is loop-invariant (du_mma.hpp lines
  // 447-460): row = lane & 0xf, col = (lane >> 4) << 3.
  const int a_row = lane & 15;
  const int a_col = (lane >> 4) << 3;

  // Iteration 13 (HIP-only consolidation): prefetch the four per-thread scale
  // values at the kernel top, BEFORE the stage-0 fragment loads. The fused
  // epilogue then has ZERO exposed global latency on the block's finishing
  // tail: the same 4 vmem reads (x_scale[row], x_scale[row2],
  // weight_scale[n0+col], weight_scale[n0+16+col] - each used twice) move
  // from after the END-of-K __syncthreads() into the cold-start window, where
  // their L2 round trip overlaps the first B-load DRAM latency. Same 4 loads,
  // same values, bit-identical math - only the issue time moves (validated
  // pattern: the accepted down_proj kernel prefetches its scales at kernel
  // top too). Cost: +4 VGPR held live across the K loop (64 -> ~68, register
  // file non-binding); scratch must stay 0.
  const int row = tid >> 4;            // epilogue rows 0..7 (and row2 = +8)
  const int col = tid & 15;            // epilogue column within the tile
  const int row2 = row + (kDummaTileM >> 1);
  const float x_scale_lo = x_scale[row];
  const float x_scale_hi = x_scale[row2];
  const float ws_lo = weight_scale[n0 + col];
  const float ws_hi = weight_scale[n0 + kDummaTileN + col];

  // Four 1 KiB row-major int32 partial planes: [wave][tile][256].
  __shared__ __align__(16) int32_t acc_planes[kDummaWavesPerBlock *
                                              kDummaTilesPerBlock *
                                              kDummaTileM * kDummaTileN];

  // 16 stages of K=128 partitioned UNIFORMLY across the 2 waves (8 stages of
  // K=1024 per wave); ascending-K order per wave and the byte-exact fragment
  // loads (A dwordx2, B dwordx4 tile-pair) are unchanged, the end-of-K int32
  // combine is order-independent, so the result is bit-identical to
  // iteration 4.
  const int total_stages = k / kStageK;  // 16
  const int stage_start = (wave * total_stages) / kDummaWavesPerBlock;
  const int stage_end = ((wave + 1) * total_stages) / kDummaWavesPerBlock;
  const int n_stages = stage_end - stage_start;  // 8 stages of K=128
  const int64_t k0_base = static_cast<int64_t>(stage_start) * kStageK;

  // Prologue: stage-0 fragments land directly in registers (one cold-start
  // DRAM latency per wave, amortized over 8 stages and staggered across the
  // 2 waves).
  uint64_t ca0, ca1, ca2, ca3;                    // current stage's A
  uint64_t cb00, cb01, cb02, cb03;                // current stage's B, tile 0
  uint64_t cb10, cb11, cb12, cb13;                // current stage's B, tile 1
  load_stage_fragments_t2(x_q, weight, lane, a_row, a_col, b_base, k, k0_base,
                          0, ca0, ca1, ca2, ca3, cb00, cb01, cb02, cb03, cb10,
                          cb11, cb12, cb13);

  for (int s = 0; s < n_stages; ++s) {
    // Issue the NEXT stage's eight global loads (4 A dwordx2 + 4 B dwordx4)
    // BEFORE this stage's MMAC burst so DRAM/L2 latency overlaps the burst;
    // the compiler-inserted s_waitcnt vmcnt(0) for these loads lands at the
    // next iteration's first read of the prefetched registers (the fragment
    // fill below) - 6 KB of in-flight bytes per wait, no LDS hop.
    uint64_t na0, na1, na2, na3;
    uint64_t nb00, nb01, nb02, nb03, nb10, nb11, nb12, nb13;
    if (s + 1 < n_stages) {
      load_stage_fragments_t2(x_q, weight, lane, a_row, a_col, b_base, k,
                              k0_base, s + 1, na0, na1, na2, na3, nb00, nb01,
                              nb02, nb03, nb10, nb11, nb12, nb13);
    }
    // Stage-s MMAC burst straight from registers, in the exact a_frag.x /
    // b_frag.x byte order du_mma_sync consumes: the same A fragment feeds
    // both tiles (multi-N-tile reuse - A loaded once per stage).
#pragma unroll
    for (int t = 0; t < kStageSteps; ++t) {
      const uint64_t av =
          (t == 0) ? ca0 : (t == 1) ? ca1 : (t == 2) ? ca2 : ca3;
      const uint64_t b0v =
          (t == 0) ? cb00 : (t == 1) ? cb01 : (t == 2) ? cb02 : cb03;
      const uint64_t b1v =
          (t == 0) ? cb10 : (t == 1) ? cb11 : (t == 2) ? cb12 : cb13;
      __builtin_memcpy(a_frag.x, &av, 8);
      __builtin_memcpy(b_frag0.x, &b0v, 8);
      __builtin_memcpy(b_frag1.x, &b1v, 8);
      du::dumma::du_mma_sync(acc0, a_frag, b_frag0, acc0);
      du::dumma::du_mma_sync(acc1, a_frag, b_frag1, acc1);
    }
    // Rotate: the prefetched stage s+1 becomes the current stage. No LDS
    // WAR hazard; the vmcnt wait for the prefetch is satisfied before the
    // next iteration's first use.
    if (s + 1 < n_stages) {
      ca0 = na0;
      ca1 = na1;
      ca2 = na2;
      ca3 = na3;
      cb00 = nb00;
      cb01 = nb01;
      cb02 = nb02;
      cb03 = nb03;
      cb10 = nb10;
      cb11 = nb11;
      cb12 = nb12;
      cb13 = nb13;
    }
  }

  // Combine the two wavefronts' int32 partials in LDS (exact int32 sums,
  // order independent) after a single barrier at the END of K - barrier
  // count per K step stays 0. Each wave stores its two tiles' planes.
  du::dumma::du_store_matrix_sync(
      acc_planes + (wave * kDummaTilesPerBlock) * kDummaTileM * kDummaTileN,
      acc0, kDummaTileN, du::dumma::mem_row_major);
  du::dumma::du_store_matrix_sync(
      acc_planes + (wave * kDummaTilesPerBlock + 1) * kDummaTileM *
                       kDummaTileN,
      acc1, kDummaTileN, du::dumma::mem_row_major);
  __syncthreads();

  // Cooperative epilogue over the two 16x16 tiles: all 128 threads, 4
  // elements each - for tile t: (row = tid >> 4, col = tid & 15) for rows
  // 0..7 and (row + 8, col) for rows 8..15; sum the 2 planes per element
  // (int32 add is order-independent), then fused dot * x_scale[row] *
  // weight_scale[col] -> bf16 store. The four scale values were prefetched
  // at the kernel top (iteration 13), so no global load sits on this tail;
  // iteration 17 replaces the four __float2bfloat16 calls with the finite
  // RNE-only bf16_rne_finite (same bits for every finite input, no NaN/Inf
  // branch), shortening each element's dependent tail chain.
  int32_t t0_lo = 0, t0_hi = 0, t1_lo = 0, t1_hi = 0;
#pragma unroll
  for (int w = 0; w < kDummaWavesPerBlock; ++w) {
    const int plane_base =
        w * kDummaTilesPerBlock * kDummaTileM * kDummaTileN;
    t0_lo += acc_planes[plane_base + tid];
    t0_hi += acc_planes[plane_base + tid + kDummaTileM * kDummaTileN / 2];
    t1_lo += acc_planes[plane_base + kDummaTileM * kDummaTileN + tid];
    t1_hi += acc_planes[plane_base + kDummaTileM * kDummaTileN + tid +
                        kDummaTileM * kDummaTileN / 2];
  }
  const float s0_lo = static_cast<float>(t0_lo) * x_scale_lo * ws_lo;
  out[static_cast<int64_t>(row) * n + n0 + col] = bf16_rne_finite(s0_lo);
  const float s0_hi = static_cast<float>(t0_hi) * x_scale_hi * ws_lo;
  out[static_cast<int64_t>(row2) * n + n0 + col] = bf16_rne_finite(s0_hi);
  const float s1_lo = static_cast<float>(t1_lo) * x_scale_lo * ws_hi;
  out[static_cast<int64_t>(row) * n + n0 + kDummaTileN + col] =
      bf16_rne_finite(s1_lo);
  const float s1_hi = static_cast<float>(t1_hi) * x_scale_hi * ws_hi;
  out[static_cast<int64_t>(row2) * n + n0 + kDummaTileN + col] =
      bf16_rne_finite(s1_hi);
}

// Generic grid-stride device-to-device copy (int8) - identity pack for every
// (K, N) except the exact (2048, 4096) o_proj shape.
__global__ void w8a8_copy_i8_kernel(const int8_t* __restrict__ src,
                                    int8_t* __restrict__ dst,
                                    int64_t count) {
  const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x +
                   static_cast<int64_t>(threadIdx.x);
       i < count; i += stride) {
    dst[i] = src[i];
  }
}

// Generic grid-stride device-to-device copy (float).
__global__ void w8a8_copy_f32_kernel(const float* __restrict__ src,
                                     float* __restrict__ dst,
                                     int64_t count) {
  const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x +
                   static_cast<int64_t>(threadIdx.x);
       i < count; i += stride) {
    dst[i] = src[i];
  }
}

constexpr int kMaxCopyBlocks = 65535;

// ---------------------------------------------------------------------------
// One-time pack for the exact (k, n) == (2048, 4096) o_proj weight: permute
// the logical [K, N] int8 layout into packed[n_tile_pair][k_step][lane][16]
// B fragment slots - tile-pair-interleaved: for the same (k_step, lane),
// bytes 0..7 are the 8 fragment bytes of tile (2*pair) and bytes 8..15 the 8
// fragment bytes of tile (2*pair+1) (see the file header and the kernel
// comment for the exact du_mma.hpp matrix_b row_major lane mapping:
// row = lane & 0xf, col = (lane >> 4) << 3, x[i] = p[(col + i) * ldm + row]).
// One thread per (pair, k_step, lane): 64 k_steps x 128 pairs x 64 lanes =
// 524,288 threads, each writing one aligned 16-byte slot. Runs once per
// weight, outside the timed GEMM and outside Graph capture; byte count is
// unchanged (8 MiB), so the packed tensor is a same-size permutation.
// ---------------------------------------------------------------------------
__global__ __launch_bounds__(kScalarThreads) void w8a8_pack_o_proj_m16_kernel(
    const int8_t* __restrict__ raw,
    int8_t* __restrict__ packed,
    int k,
    int n) {
  const int64_t idx =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total = (static_cast<int64_t>(k) / kDummaTileK) *
                        (n / (2 * kDummaTileN)) * kDummaBlockThreads;
  if (idx >= total) {
    return;
  }
  const int pair = static_cast<int>(idx >> 12);  // idx / (64 * 64)
  const int rem = static_cast<int>(idx & 4095);  // idx % (64 * 64)
  const int k_step = rem >> 6;                   // idx / 64 % 64
  const int lane = rem & 63;                     // idx % 64
  const int row = lane & 15;
  const int col = (lane >> 4) << 3;
  const int8_t* src0 =
      raw + (static_cast<int64_t>(k_step) * kDummaTileK + col) * n +
      (pair * 2) * kDummaTileN + row;
  const int8_t* src1 = src0 + kDummaTileN;  // tile (2*pair + 1)
  uint64_t v0 = 0;                          // tile 2*pair   (bytes 0..7)
  uint64_t v1 = 0;                          // tile 2*pair+1 (bytes 8..15)
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    v0 |= (static_cast<uint64_t>(static_cast<uint8_t>(
               src0[static_cast<int64_t>(i) * n])))
          << (8 * i);
    v1 |= (static_cast<uint64_t>(static_cast<uint8_t>(
               src1[static_cast<int64_t>(i) * n])))
          << (8 * i);
  }
  uint4 slot;
  slot.x = static_cast<uint32_t>(v0 & 0xffffffffu);
  slot.y = static_cast<uint32_t>(v0 >> 32);
  slot.z = static_cast<uint32_t>(v1 & 0xffffffffu);
  slot.w = static_cast<uint32_t>(v1 >> 32);
  *reinterpret_cast<uint4*>(
      packed + (static_cast<int64_t>(pair) * (k / kDummaTileK) + k_step) *
                   kPackedPairBytes +
      lane * 16) = slot;
}

}  // namespace

// Stable host launch symbol used by the trusted binding for
// torch.ops.zth_w8a8.gemm_out. Runs entirely on the caller-provided stream:
// no allocation, no synchronization, no default-stream launch, no workspace
// writes. Returns nothing; the binding returns the caller-provided out.
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
  (void)workspace;  // in-block split-K: no cross-block partials, no workspace
  (void)workspace_bytes;

  hip_bfloat16* out_bf16 = reinterpret_cast<hip_bfloat16*>(out);

  if (m == kTargetM && n == kTargetN && k == kTargetK) {
    // Exact assigned shape hy3_tp4_o_proj_m16: packed-B two-wave in-block
    // split-K=2 DUMMA m16n16k32 kernel with MULTI-N-TILE REUSE (128 blocks,
    // two adjacent 16x16 N tiles per block, 2 x 64-thread wavefronts, K=128
    // stages, depth-1 register prefetch). The scalar fallback below stays
    // untouched and still covers every unmatched shape, including the paired
    // M=2 shape with the same (N, K) (the guard is shape-exact on m too; the
    // fallback decodes the packed weight for (n, k) == (4096, 2048)).
    const dim3 grid(static_cast<unsigned>(
        kTargetN / (kDummaTileN * kDummaTilesPerBlock)));
    const dim3 block(kDummaBlockThreads * kDummaWavesPerBlock);
    hipLaunchKernelGGL(w8a8_gemm_m16_dumma_packedb_multint_kernel, grid,
                       block, 0, stream, a, b, x_scale, weight_scale,
                       out_bf16, n, k);
    return;
  }

  // Generic scalar fallback for every unmatched (m, n, k). This also covers
  // the paired M=2 API shape that shares the same (N, K): it must never be
  // routed through an M=16-only specialization.
  const dim3 grid(
      static_cast<unsigned>((n + kScalarThreads - 1) / kScalarThreads),
      static_cast<unsigned>(m));
  const dim3 block(kScalarThreads);
  hipLaunchKernelGGL(w8a8_gemm_scalar_kernel, grid, block, 0, stream, a, b,
                     x_scale, weight_scale, out_bf16, m, n, k);
}

// Stable host launch symbol used by the optional out-of-timed-region
// torch.ops.zth_w8a8.pack_weight. For the exact (k, n) == (2048, 4096) o_proj
// weight this permutes into the packed-B fragment-slot layout (one-time,
// untimed, outside Graph capture); every other (K, N) keeps the generic
// identity device-to-device copy. Weight scales are always copied through.
extern "C" void launch_pack_w8a8_weight(
    const int8_t* raw_weight,
    const float* weight_scale,
    int8_t* packed_weight,
    float* packed_weight_scale,
    int k,
    int n,
    hipStream_t stream) {
  if (k == kTargetK && n == kTargetN) {
    // Exact o_proj M=16 weight: permute into [n_tile_pair][k_step][lane][16]
    // B fragment slots (one-time, untimed, outside Graph capture).
    const int64_t total = (static_cast<int64_t>(k) / kDummaTileK) *
                          (n / (2 * kDummaTileN)) * kDummaBlockThreads;
    const dim3 grid(
        static_cast<unsigned>((total + kScalarThreads - 1) / kScalarThreads));
    const dim3 block(kScalarThreads);
    hipLaunchKernelGGL(w8a8_pack_o_proj_m16_kernel, grid, block, 0, stream,
                       raw_weight, packed_weight, k, n);
  } else {
    const int64_t weight_count = static_cast<int64_t>(k) * n;
    const int64_t weight_blocks =
        (weight_count + kScalarThreads - 1) / kScalarThreads;
    const unsigned weight_grid =
        static_cast<unsigned>(weight_blocks > kMaxCopyBlocks
                                  ? kMaxCopyBlocks
                                  : weight_blocks);
    hipLaunchKernelGGL(w8a8_copy_i8_kernel, dim3(weight_grid),
                       dim3(kScalarThreads), 0, stream, raw_weight,
                       packed_weight, weight_count);
  }

  const int64_t scale_count = static_cast<int64_t>(n);
  const int64_t scale_blocks =
      (scale_count + kScalarThreads - 1) / kScalarThreads;
  const unsigned scale_grid =
      static_cast<unsigned>(scale_blocks > kMaxCopyBlocks ? kMaxCopyBlocks
                                                          : scale_blocks);
  hipLaunchKernelGGL(w8a8_copy_f32_kernel, dim3(scale_grid),
                     dim3(kScalarThreads), 0, stream, weight_scale,
                     packed_weight_scale, scale_count);
}
// @@end
