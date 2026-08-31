// @@variant shape=tp8_wo_b_m16 commit=eb0fbbad60c58b1cf09e6ceff6cb9aaa68a2c1cb added=2026-08-26
//   median_us=9.81 p90_us=9.84
//   source=hy3-dsh-tp8-m16-1-adf021ab
// MetaInfer INT8 W8A8 GEMM for gfx928 (worker_2 / tp8_wo_b_m16).
//
// Iteration 9: exact-shape DUMMA specialization for (m, n, k) == (16, 4096,
// 1024), plus the scalar generic fallback for every other shape.
//
// Iteration 9 change (prefetch round, stage-K=64 double-buffered A+B LDS
// prefetch with packed-B fragment slots): the accepted iteration-2 kernel
// (256 blocks x 128 threads, in-block split-K=2, per-wave stage-K=128,
// 17.073 us median / 17.096 us p90) is serialized on the per-stage global
// load latency. The exact iteration-2 ISA (source digest a59a4a28...) shows
// the stage loop as: 4 global_load_dwordx4 -> s_waitcnt vmcnt(3..0) (one
// wait in front of EACH staging ds_write_b128) -> 32 ds_read_u8 + 4
// ds_read2_b32 -> 4 v_mmac, with the next stage's loads issued only at the
// top of the next iteration: every one of the 4 stages per wave exposes the
// full L2 latency of its global loads with nothing to hide it. Iteration 7
// tried a depth-1 register pipeline (hold arrays + guarded rotation) and
// collapsed to 48.03 us; its exact ISA shows why: the compiler refused to
// keep the cross-iteration hold registers and materialized the rotation as
// a global-memory round trip (4 buffer_store_dwordx4 + 4
// buffer_load_dwordx4 per stage in the steady loop), i.e. MORE traffic with
// the same serialization. This round adopts the prefetch pattern validated
// on the same DTK/gfx928 by the worker_0 wqkv_a lineage (accepted 18.55 us,
// ISA-verified): per K=64 chunk, the NEXT chunk's global loads (1 x 16-B A
// dwordx4 + 2 x 8-B B dwordx2 per lane, scalar variables only - no arrays,
// no cross-iteration registers) are issued at the TOP of the iteration and
// committed to the ALTERNATE LDS buffer AFTER the current chunk's MMAC
// burst, so the compiler's vmcnt wait lands at the alternate-buffer ds_write
// and the load latency overlaps the current chunk's LDS/MMAC work instead
// of stalling in front of it. B is consumed from the one-time packed
// [n_tile][k_step][lane][8] fragment-slot layout (w8a8_pack_b_fragslot_
// kernel, produced by launch_pack_w8a8_weight outside the timed region and
// Graph capture, same byte count and graph-stable buffer), which replaces
// B's current 8x-over-fetching strided 16-B global reads with one aligned,
// coalesced dwordx2 per lane per k32 step; the scalar fallback decodes the
// pack for the paired M=2 shape. Everything else is byte-identical to the
// accepted winner: grid = N/16 = 256 blocks x 128 threads, in-block
// split-K=2 (wave w accumulates k in [w*K/2, (w+1)*K/2) k-ascending in 32-B
// DUMMA steps), LDS 10240 B/block (2 waves x 2 buffers x 2048 B of A/B
// chunks + 2 int32 partial planes -> 6 blocks/CU, residency unchanged),
// zero in-K-loop barriers, one END-of-K __syncthreads combine on wave 0 in
// ascending split order (mod-2^32 int32 addition groups exactly ->
// bit-identical, 0 mismatches), direct transposed-lane epilogue with scales
// before the bf16 store, and the scalar fallback for the paired M=2 shape
// and every unmatched (m,n,k). Graph safety is unchanged (single kernel
// launch on the caller stream inside capture, static LDS, no
// allocation/synchronization; every LDS byte is overwritten before every
// read per replay). Predicted signature if the mechanism is real: the
// steady K loop shows next-chunk global_load_dwordx4/dwordx2 BEFORE the
// current chunk's ds_reads/v_mmac with the vmcnt wait at the
// alternate-buffer ds_write_b128/b64 (vs today's vmcnt(3..0)-before-
// ds_write at stage top); vmem_read_instructions rises ~9472 -> ~13.5K (B
// loads split into narrower but coalesced dwordx2); lds_bank_conflicts
// drops (A row stride 64 B -> 8-way instead of 8-16-way; B slots
// contiguous -> conflict-free). Falsifiable gate: normal-benchmark
// median_us < 17.073 AND p90_us < 17.096 (accepted best). If flat or
// slower, the ~17 us floor is not the per-stage global-load exposure at
// 4.27 waves/CU (the co-resident wave already hides it) and the next round
// targets the launch/dispatch floor or the epilogue; the round still counts
// as a valid HIP experiment (build + correctness + graph capture).
//
// Iteration 15 change (split-epilogue round): the accepted iteration-9
// kernel (256 blocks x 128 threads, in-block split-K=2, per-wave stage-K=64
// double-buffer prefetch + packed-B, 10.695 us median / 11.145 us p90, 40
// arch VGPR / 32 SGPR / 10240 B LDS, 512 wavefronts) ends with a
// wave-0-ONLY epilogue: after the END-of-K barrier, wave 1 exits while wave
// 0 serially runs 4 combine iterations (8 ds_read2st64_b32 + 4 stores) plus
// its 5 on-demand global scale loads (1 x_scale[r] + 4 weight_scale[n0+c],
// ISA 0x4434/0x445C/0x4510/0x45E4/0x46B0) with s_waitcnt vmcnt(1)/vmcnt(0)
// between the two scale multiplies of each of the 4 unrolled output
// iterations - the block's completion time is wave 0's full epilogue path.
// Iteration 14 (scales staged into LDS at kernel top, rejected at
// 10.822/10.866) proved the epilogue tail is live: removing the 5 scale
// loads cut p90 by 0.28 us (11.145 -> 10.866) while the prologue staging
// cost pushed the median +0.127 us. This round splits the SAME combine
// across both waves instead: wave w owns output words [w*128, w*128+128)
// (i = 2*w, 2*w+1), so each lane runs 2 combine iterations instead of 4,
// each wave needs only 3 scale loads (1 x_scale + 2 weight_scale), and wave
// 1's post-barrier idle becomes useful work. Per-element math is unchanged:
// the two-term int32 sum s_partial[0][idx] + s_partial[1][idx] and the
// float chain float(acc) * x_scale[r] * weight_scale[n0+c] are computed
// exactly once, by one wave, in the same expression order -> output
// bit-identical (0 mismatches). Everything else is byte-identical to the
// accepted iteration-9 winner: grid 256 x 128 threads, split-K=2 wave
// ownership and k-ascending DUMMA order, stage-K=64 prefetch, packed-B,
// LDS 10240 B/block unchanged (6 blocks/CU capacity >= 2.13 actual, all
// 256 blocks co-resident), the single END-of-K barrier (still orders both
// planes before either wave's combine reads), and the scalar fallback for
// the paired M=2 shape and every unmatched (m,n,k). Graph safety is
// unchanged (single kernel launch on the caller stream inside capture,
// static LDS 10240 B <= 64 KiB, no allocation/synchronization; every LDS
// byte is overwritten before every read per replay). Predicted signature if
// the mechanism is real: profiled kernel duration ~8.64 -> ~8.3-8.5 us;
// epilogue per wave shows 2 unrolled iterations of (2 x ds_read2st64_b32 +
// float mul + d16 store) with only 3 global_load_dword, no s_waitcnt
// vmcnt(1) between the scale multiplies; vmem_read_instructions ~13.5K ->
// ~14.3K (wave 0's 5 epilogue loads + wave 1's new 3 = 8 per block vs 5
// today), lds_instructions ~23.3K unchanged (same reads, spread over 2
// waves), VGPR ~40 -> ~44 (wave 1 holds 3 scale floats), LDS 10240
// B/block unchanged. Falsifiable gate: normal-benchmark median_us < 10.695
// AND p90_us < 11.145 (accepted best). If flat or slower, the epilogue is
// NOT on the block's critical path (the residual tail is the launch/
// dispatch floor ~2 us above the 8.64 us profiled kernel and/or the HBM
// streaming rate at 40.6% L2 hit with B read exactly once), the next round
// targets the B-side slot swizzle or a register-only scale transport; the
// round still counts as a valid HIP experiment (build + correctness +
// graph capture).
//
// Iteration 18 repair 1 (compile fix): the paired 4-B bf16 epilogue store
// read the bf16 raw bits via __float2bfloat16(s).data, which does not exist
// on this DTK's __hip_bfloat16 (amd_hip_bf16.h stores the bits in a
// protected __x; the only raw-bit accessor is operator unsigned short()).
// Fixed by casting through static_cast<unsigned short> so overload
// resolution selects the raw-bit conversion operator and NOT the value-based
// operator unsigned int() (an exact-match candidate that would silently
// convert the scaled float value instead of reinterpreting the bf16 bits).
// Semantics unchanged: the two 16-bit bf16 patterns still land in one 32-B
// word (lo = s0, hi = s1), output bit-identical to the previous two
// 2-B scalar stores; no kernel logic, mapping, or launch geometry changed.
//
// Iteration 19 change (B chunk-slot transport consolidation): the accepted
// iteration-18 kernel (256 blocks x 128 threads, in-block split-K=2,
// per-wave stage-K=64 double-buffer prefetch + packed-B, 9.949 us median /
// 9.962 us p90, 40 arch VGPR / 32 SGPR / 10240 B LDS, kernel
// w8a8_dumma_m16_n16_2wave_dbuf_kernelILi64EEE, 512 wavefronts) has B as the
// ONLY remaining 8-B-width global stream: the exact iteration-18 ISA shows
// per chunk the next-chunk B fetched as 2 x global_load_dwordx2 (0x42C8 /
// 0x42D0, 8 B per lane, 512 B apart -> not fuseable in the current
// [n_tile][k_step][lane][8] layout) while A is already ONE
// global_load_dwordx4 (0x42C0) per lane per chunk, and B's LDS write/read
// are already compiler-fused (ds_write2st64_b64 0x43C0 / ds_read2st64_b64
// 0x42F8). Iteration 17 (register-only B transport, 11.529 us / 12.477 p90)
// proved the B LDS staging round trip is load-bearing and iteration 13
// (A-side 8-way -> 2-way bank-conflict pad, 12.30 us) proved LDS conflict
// counts are NOT the limiter, so the steady loop's remaining HIP-visible
// inefficiency is the per-chunk 2 x 512-B B request stream itself. This
// round re-packs B from [n_tile][k_step][lane][8] to
// [n_tile][k_chunk][lane][16] (k_chunk = K/64 = 16 chunks; each 16-B lane
// slot holds BOTH k32 steps of one K=64 wavefront chunk, lo = step 2ch, hi =
// step 2ch+1, identical byte count K*N = 4,194,304), so each lane fetches
// the whole chunk with ONE 16-B aligned load (BChunkSlot alignas(16) ->
// global_load_dwordx4, matching the validated worker_0 wqkv recipe "one
// 16-B cooperative load/lane/stage" and the kernel's own A int4 path), and
// the LDS side becomes one ds_write_b128 / one ds_read_b128 per chunk with
// contiguous 16-B lane slots (clean banks vs the current 4-way 8-B slot
// stream). Per-element math is unchanged: identical global bytes in
// identical k-ascending 2 x du_mma_sync order per chunk (the two k32 steps'
// fragment bytes are the same 8-B values, just regrouped into one 16-B
// slot), identical split ownership, identical ascending split sum -> int32
// bit-identical (0 mismatches). Everything else is byte-identical to the
// accepted iteration-18 winner: grid 256 x 128 threads, split-K=2 wave
// ownership and k-ascending DUMMA order, stage-K=64 A prefetch structure
// (A path untouched, ldm = kStageK = 64 unchanged), LDS 10240 B/block
// unchanged (B chunk buffer stays 1024 B per wave per buffer: 64 lanes x 16
// B), the single END-of-K __syncthreads, the iteration-18 split two-wave
// combine with one 4-B bf16 store and one 8-B float2 weight_scale load per
// lane, and the scalar fallback for the paired M=2 shape and every unmatched
// (m,n,k) (packed_b_element updated to decode the new 16-B slot layout). The
// one-time pack kernel (launch_pack_w8a8_weight, out of the timed region and
// Graph capture, same byte count, graph-stable buffer) writes the new
// layout; the paired M=2 validation decodes it via packed_b_element, so the
// packed-shape scalar path stays exact. Graph safety is unchanged (single
// kernel launch on the caller stream inside capture, static LDS 10240 B <=
// 64 KiB, no allocation/synchronization; every LDS byte is overwritten
// before every read per replay; B bytes move global -> LDS -> registers
// ordered by the existing vmcnt/lgkmcnt waits). Predicted signature if the
// mechanism is real: the steady chunk loop shows ONE next-chunk
// global_load_dwordx4 for B instead of two global_load_dwordx2 (vmem_read_
// instructions 13,312 -> ~11,264: B loads 8192 -> 4096), one
// ds_write_b128 + one ds_read_b128 per chunk replacing the st64-fused
// 8-B-slot pair (lds_instructions ~22.5K unchanged - same op count, cleaner
// banks: lds_bank_conflicts 233,472 -> ~180-200K as the B 4-way slot stream
// disappears; A conflicts remain and are known non-binding per iteration
// 13), VGPR ~40 -> ~40-42 (BChunkSlot = 4 VGPRs, same live set as the two
// uint64 today), LDS 10240 B/block unchanged, grid 256 / workgroup 128 /
// 512 waves unchanged, profiled kernel duration ~8.0 us -> ~7.8-8.0 us if
// the 1-KB per-chunk request stream improves L2/DRAM efficiency vs today's
// 2 x 512-B requests. Falsifiable gate: normal-benchmark median_us < 9.949
// AND p90_us < 9.962 (accepted best). If flat or slower, the per-chunk
// request width is NOT the residual (the kernel is already streaming B at
// ~530-550 GB/s real HBM, the highest measured rate in this lineage and
// consistent with the o_proj 8.2-MB shape's ~450-500 GB/s cap; the residual
// is the HBM wall + the ~1.9 us launch/dispatch floor - not addressable
// from HIP at this shape), and the next round documents the HIP floor; the
// round still counts as a valid HIP experiment (build + correctness + graph
// capture).
//
// DUMMA path (w8a8_dumma_m16_n16_2wave_dbuf_kernel):
//   * native gfx928 DUMMA INT8 m16n16k32 tiles, one 16x16 output tile per
//     block, two 64-thread wavefronts per block (128 threads, 2 x 64);
//   * in-block split-K=2 along K: wave 0 accumulates k in [0, K/2), wave 1
//     accumulates k in [K/2, K). Both publish their int32 partial plane to
//     LDS; one END-of-K block barrier; both waves split the plane-sum
//     combine (wave w owns output words [w*128, w*128+128)), so each
//     element's two-term int32 sum is computed exactly once by one wave
//     (int32 addition groups exactly -> bit-identical to the k-ascending
//     order) and stored scaled bf16;
//   * grid = N/16 = 256 blocks (>= 120 CUs), 512 wavefronts total
//     (~4.3 waves/CU);
//   * per-wave private double-buffered A/B stage chunks (K=64 per chunk,
//     2 chunks per buffer, 2 buffers per wave) so the K loop needs no
//     barrier at all (within-wave LDS write->read ordering is emitted by
//     the compiler as lgkmcnt waits); only the END-of-K combine barrier
//     remains (1 barrier per block);
//   * each chunk: 2 x m16n16k32 du_mma_sync in k-ascending order, with the
//     next chunk's A/B global loads prefetched into scalar registers before
//     the current chunk's MMAC burst and committed to the alternate buffer
//     after it (vmcnt wait lands at the commit);
//   * direct epilogue: transposed lane-contiguous int32 partial planes
//     (element (row, col) -> word row + 16*col), combine reads are
//     conflict-free, scales applied before the bf16 store; the 256-word
//     combine is split across both waves (2 iterations per lane each) so
//     the kernel-end critical path halves and wave 1 does not idle;
//   * B is consumed in the packed [n_tile][k_chunk][lane][16] fragment-slot
//     layout for (K, N) == (1024, 4096) (iteration-19 chunk-slot repack:
//     each 16-B lane slot holds the two k32 steps of one K=64 chunk, one
//     one-time pack in launch_pack_w8a8_weight, out of the timed
//     region/Graph); every other (K, N) keeps the identity pack.
//
// Scalar path: unchanged correctness-first fallback used by the paired M=2
// shape and every other unmatched (m, n, k); for (K, N) == (1024, 4096) it
// decodes the packed-B layout elementwise.
//
// Header order is fixed by the toolchain: HIP runtime, bf16, then du_mma
// (du_mma.h is not self-contained before the HIP runtime headers).

#include <hip/hip_runtime.h>
#include <hip/hip_bfloat16.h>
#include <du_mma.h>

#include <cstdint>

namespace {

constexpr int kScalarThreads = 128;  // multiple of the gfx928 wavefront (64)
constexpr int kPackThreads = 256;

// Fragment-slot B pack constants for the exact (K, N) == (1024, 4096) shape.
constexpr int kDummaTileK = 32;
constexpr int kPackedSlotBytes = 8;     // one lane's matrix_b fragment bytes
                                        // (one k32 step)
constexpr int kChunkSlotBytes = 16;     // one lane's full K=64 chunk bytes
                                        // (2 k32 steps, iteration-19 layout)
constexpr int kChunkBytes = 64 * kChunkSlotBytes;  // 1024 B per wavefront chunk

// One lane's 16-B B chunk slot in the [n_tile][k_chunk][lane][16] layout:
// lo = k-step 2ch fragment bytes, hi = k-step 2ch+1. alignas(16) makes the
// compiler emit 16-B vector loads/stores (global_load_dwordx4 /
// ds_read_b128 / ds_write_b128) exactly like the A int4 path; slot addresses
// are 16-B aligned by construction (b_base multiples of 1024 + lane*16).
struct alignas(16) BChunkSlot {
  uint64_t lo;
  uint64_t hi;
};

// Elementwise decode of the packed [n_tile][k_chunk][lane][16] fragment-slot
// layout for (K, N) == (1024, 4096) (iteration-19 chunk-slot repack: each
// 16-B lane slot holds one wavefront's two k32 steps of a K=64 chunk, lo =
// step 2ch, hi = step 2ch+1): returns logical weight[kk, n_col] with the
// same (tile, chunk, lane, byte) -> logical (k, n) mapping as the pack
// kernel and the DUMMA kernel. Used by the generic scalar fallback (paired
// M=2 validation) for the packed shape only.
__device__ __forceinline__ int8_t packed_b_element(
    const int8_t* __restrict__ packed, int n_col, int kk, int k) {
  const int k_chunks = k / (kDummaTileK * 2);  // 16 for K=1024
  const int nt = n_col >> 4;
  const int nn = n_col & 15;
  const int ch = kk >> 6;
  const int kk64 = kk & 63;
  const int kk32 = kk64 & 31;
  const int lane = (kk32 >> 3) * 16 + nn;
  const int i = (kk32 & 7) + ((kk64 >> 5) << 3);
  return packed[(static_cast<int64_t>(nt) * k_chunks + ch) * kChunkBytes +
                lane * kChunkSlotBytes + i];
}

// Exact-shape DUMMA kernel for (M=16, N=4096, K=1024), tp8 wo_b.
//
// Two wavefronts per block, one 16x16 N tile per block, in-block split-K=2:
// wave 0 accumulates k in [0, K/2), wave 1 accumulates k in [K/2, K). Each
// wave owns private double-buffered A/B stage chunks (stage-K=64), so the K
// loop runs without any cross-wave barrier; both waves publish their int32
// partial plane to LDS, one END-of-K block barrier orders publication, and
// both waves split the plane-sum combine (wave w owns output words
// [w*128, w*128+128)); each element's two-term int32 sum is computed exactly
// once by one wave (mod-2^32 int32 addition groups exactly, so the grouped
// sum is bit-identical to the k-ascending scalar order), scaled by
// x_scale/weight_scale, and stored bf16.
//
// Steady chunk loop (validated worker_0 prefetch pattern): the NEXT chunk's
// A/B global loads are issued into scalar registers at the top of the
// iteration (their latency is covered by the current chunk's ds_read/MMAC
// burst), and are committed to the ALTERNATE LDS buffer after the burst, so
// the compiler's vmcnt wait lands at the alternate-buffer ds_write instead
// of stalling before the current stage's LDS writes (the iteration-2 ISA
// serialization) and without the iteration-7 cross-iteration register
// rotation that the compiler spilled through global memory.
template <int kStageK>
__global__ __launch_bounds__(128) void w8a8_dumma_m16_n16_2wave_dbuf_kernel(
    const int8_t* __restrict__ a,           // [16, K] row-major int8
    const int8_t* __restrict__ packed_b,    // [n_tile][k_chunk][lane][16]
    const float* __restrict__ x_scale,      // [16, 1] fp32
    const float* __restrict__ weight_scale, // [N, 1] fp32
    hip_bfloat16* __restrict__ out,         // [16, N] bf16
    int k,
    int n) {
  constexpr int kTileM = 16;
  constexpr int kTileN = 16;
  constexpr int kTileK = 32;
  constexpr int kWaveSize = 64;
  constexpr int kSplits = 2;
  constexpr int kStageSteps = kStageK / kTileK;  // 2 k32 steps per chunk
  const int lane = threadIdx.x & (kWaveSize - 1);
  const int wave = threadIdx.x >> 6;  // 0 or 1
  const int n0 = static_cast<int>(blockIdx.x) * kTileN;
  const int ntile = static_cast<int>(blockIdx.x);

  // Per-wave private double-buffered staging: A chunk 16 x 64 (1024 B) and B
  // chunk kStageSteps x 64 x 8 (1024 B) per buffer, 2 buffers per wave
  // (2 x 2 x 2048 = 8192 B), plus 2 x 16 x 16 x 4 = 2048 B of int32 partial
  // planes -> 10240 B LDS per block (identical to the accepted iteration-2
  // kernel; 6 blocks/CU, all 256 blocks resident).
  __shared__ __align__(16) int8_t a_tile[kSplits][2][kTileM * kStageK];
  __shared__ __align__(16) int8_t b_tile[kSplits][2][kChunkBytes];
  __shared__ __align__(16) int32_t s_partial[kSplits][kTileM * kTileN];

  du::dumma::DUFragment<
      du::dumma::matrix_a, kTileM, kTileN, kTileK, signed char,
      du::dumma::row_major>
      a_frag;
  du::dumma::DUFragment<
      du::dumma::matrix_b, kTileM, kTileN, kTileK, signed char,
      du::dumma::row_major>
      b_frag;
  du::dumma::DUFragment<du::dumma::accumulator, kTileM, kTileN, kTileK, int>
      acc_frag;
  du::dumma::du_fill_fragment(acc_frag, 0);

  // A chunk roles: 4 consecutive lanes own one 64-B row (16 B each), so the
  // staging loads are 16-B vectorized and coalesced.
  const int a_row = lane >> 2;
  const int a_col16 = (lane & 3) * 16;

  const int kPerWave = k / kSplits;   // 512 for K=1024
  const int kBase = wave * kPerWave;  // 0 or 512
  const int kChunksTotal = k / kStageK;  // 16 for K=1024
  const int64_t b_base =
      static_cast<int64_t>(ntile) * kChunksTotal * kChunkBytes +
      (kBase / kStageK) * kChunkBytes + lane * kChunkSlotBytes;

  // Prologue: stage chunk 0 into buffer 0 (single wait before the loop).
  *reinterpret_cast<int4*>(a_tile[wave][0] + a_row * kStageK + a_col16) =
      *reinterpret_cast<const int4*>(a + a_row * k + kBase + a_col16);
  *reinterpret_cast<BChunkSlot*>(b_tile[wave][0] + lane * kChunkSlotBytes) =
      *reinterpret_cast<const BChunkSlot*>(packed_b + b_base);

  const int nchunks = kPerWave / kStageK;  // 8 for K=1024, K=64 chunks
  for (int c = 0; c < nchunks; ++c) {
    const int buf = c & 1;

    // Prefetch chunk c+1 from global into registers (no wait yet). The
    // values are consumed only by the alternate-buffer stores below, so the
    // compiler issues these loads before (or at worst alongside) the MMAC
    // burst and lands the vmcnt wait at the ds_write - the overlap that
    // removes the iteration-2 per-stage vmcnt(3..0)-before-ds_write
    // exposure. Scalar variables only (no arrays, no cross-iteration
    // registers), so the iteration-7 global-memory spill cannot recur.
    int4 a_v;
    BChunkSlot b_v;
    const bool has_next = (c + 1 < nchunks);
    if (has_next) {
      const int k1 = kBase + (c + 1) * kStageK;
      a_v = *reinterpret_cast<const int4*>(a + a_row * k + k1 + a_col16);
      const int64_t boff = b_base + static_cast<int64_t>(c + 1) * kChunkBytes;
      b_v = *reinterpret_cast<const BChunkSlot*>(packed_b + boff);
    }

    // Compute the current chunk from LDS: kStageSteps k-ascending k32
    // steps (unchanged DUMMA order per wave). No barrier: within-wave LDS
    // write->read ordering is guaranteed by the compiler's lgkmcnt waits,
    // and each wave only touches its own stage buffers.
    const int8_t* a_buf = a_tile[wave][buf];
    const int8_t* b_buf = b_tile[wave][buf];
    // One 16-B LDS read per lane per chunk covers BOTH k32 steps of the
    // chunk (lo = step 0, hi = step 1 of the chunk's 2-step K=64 span).
    const BChunkSlot bv =
        *reinterpret_cast<const BChunkSlot*>(b_buf + lane * kChunkSlotBytes);
#pragma unroll
    for (int j = 0; j < kStageSteps; ++j) {
      du::dumma::du_load_matrix_sync(a_frag, a_buf + j * kTileK, kStageK);
      const uint64_t b64 = j ? bv.hi : bv.lo;
      __builtin_memcpy(b_frag.x, &b64, sizeof(b64));
      du::dumma::du_mma_sync(acc_frag, a_frag, b_frag, acc_frag);
    }

    // Commit the prefetched chunk into the alternate buffer (the compiler
    // inserts the vmcnt wait here, after the MMAC burst above).
    if (has_next) {
      const int nb = (c + 1) & 1;
      *reinterpret_cast<int4*>(a_tile[wave][nb] + a_row * kStageK +
                               a_col16) = a_v;
      *reinterpret_cast<BChunkSlot*>(b_tile[wave][nb] +
                                     lane * kChunkSlotBytes) = b_v;
    }
  }

  // Publish the int32 partial plane in a transposed layout (element (row,col)
  // -> word row + 16*col). Lane L's x[i] is (row = L&15, col = (L>>4) + 4*i),
  // so it lands at word L + 64*i: consecutive words per step, conflict-free.
  const int row = lane & 15;
  const int col_mod4 = lane >> 4;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    s_partial[wave][row + kTileN * (col_mod4 + 4 * i)] = acc_frag.x[i];
  }
  __syncthreads();  // END-of-K: both partial planes visible before combine

  // Combine on BOTH waves (split in iteration 15 so the kernel-end critical
  // path halves and wave 1's post-barrier idle becomes useful work).
  // Wave w owns output words [w*128, w*128+128) = the full 16 rows x the
  // 8-column band [8*w, 8*w+8) (word idx = row + 16*col in the transposed
  // layout). Lane L owns the SAME-ROW ADJACENT-COLUMN pair
  // (r = L&15, c = 8*wave + 2*(L>>4)) and (r, c+1): words
  // A = w*128 + 32*(L>>4) + r and A+16 (an exact cover of the 128-word wave
  // span, each word read by exactly one lane). The two bf16 results land
  // side by side in out[r][n0+c .. n0+c+2), so both are emitted as ONE 4-B
  // store (was two 2-B scalar stores) and the two adjacent fp32 weight
  // scales come from ONE 8-B float2 load (was two 4-B loads); x_scale[r] is
  // hoisted (loop-invariant). Per-plane reads stay at the same 64-B word
  // separation (A, A+16) the compiler already fuses to ds_read2st64_b32,
  // with lane-contiguous 16-word spans per 16-lane group (bank spread 2-way
  // -> 4-way on this tail-only path, ~4 ns, not in the steady loop). Each
  // element's two-term int32 sum s_partial[0][idx] + s_partial[1][idx] and
  // the left-associative float chain float(acc) * x_scale[r] *
  // weight_scale[n0+c] are computed exactly once, by one lane, in the same
  // expression order as before -> output bit-identical (0 mismatches).
  const int r = lane & 15;
  const int c = 8 * wave + 2 * (lane >> 4);
  const float xs = x_scale[r];
  const float2 ws = *reinterpret_cast<const float2*>(&weight_scale[n0 + c]);
  const int idx0 = 2 * wave * kWaveSize + 32 * (lane >> 4) + r;
  const int idx1 = idx0 + 16;
  const int32_t acc0 = s_partial[0][idx0] + s_partial[1][idx0];
  const int32_t acc1 = s_partial[0][idx1] + s_partial[1][idx1];
  const float s0 = static_cast<float>(acc0) * xs * ws.x;
  const float s1 = static_cast<float>(acc1) * xs * ws.y;
  *reinterpret_cast<unsigned*>(&out[r * n + n0 + c]) =
      (static_cast<unsigned>(
           static_cast<unsigned short>(__float2bfloat16(s0)))) |
      (static_cast<unsigned>(
           static_cast<unsigned short>(__float2bfloat16(s1)))
       << 16);
}

// out[m, n] = bf16( int32(sum_k a[m,k] * b[k,n]) * x_scale[m] * weight_scale[n] )
//
// One thread per output element. Grid covers M*N; every supported (m, n, k)
// from the fixed contract reaches this kernel (generic fallback). For
// (K, N) == (1024, 4096) the weight buffer holds the packed
// [n_tile][k_step][lane][8] fragment-slot layout, which is decoded
// elementwise here so the paired M=2 validation stays exact.
__global__ __launch_bounds__(kScalarThreads) void w8a8_gemm_scalar_kernel(
    const int8_t* __restrict__ a,
    const int8_t* __restrict__ b,
    const float* __restrict__ x_scale,
    const float* __restrict__ weight_scale,
    hip_bfloat16* __restrict__ out,
    int m,
    int n,
    int k) {
  const int64_t idx =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total = static_cast<int64_t>(m) * n;
  if (idx >= total) {
    return;
  }
  const int row = static_cast<int>(idx / n);
  const int col = static_cast<int>(idx - static_cast<int64_t>(row) * n);

  // (K, N) == (1024, 4096) weights are packed into the fragment-slot layout
  // by launch_pack_w8a8_weight; every other shape keeps the logical layout.
  const bool b_packed = (k == 1024 && n == 4096);

  const int8_t* __restrict__ a_row = a + static_cast<int64_t>(row) * k;
  int32_t acc = 0;
  for (int kk = 0; kk < k; ++kk) {
    const int8_t b_val =
        b_packed ? packed_b_element(b, col, kk, k)
                 : b[static_cast<int64_t>(kk) * n + col];
    acc += static_cast<int32_t>(a_row[kk]) *
           static_cast<int32_t>(b_val);
  }
  const float scaled =
      static_cast<float>(acc) * x_scale[row] * weight_scale[col];
  out[static_cast<int64_t>(row) * n + col] = __float2bfloat16(scaled);
}

// Identity device-to-device copy (bootstrap packing layout).
__global__ void w8a8_pack_identity_kernel(
    const int8_t* __restrict__ src,
    int8_t* __restrict__ dst,
    int64_t numel) {
  const int64_t idx =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < numel) {
    dst[idx] = src[idx];
  }
}

// One-time device permutation of the raw logical [K, N] row-major weight
// into the packed [n_tile][k_chunk][lane][16] B fragment-slot layout for the
// exact (K, N) == (1024, 4096) shape (iteration-19 chunk-slot repack: each
// 16-B lane slot holds the two k32 steps of one K=64 wavefront chunk, so the
// DUMMA kernel fetches the whole chunk with ONE 16-B global load per lane
// instead of two 8-B loads). One thread owns one 16-byte lane slot; slot
// ((ntile * (k/64) + chunk) * 64 + lane) * 16 + i gets logical
// weight[chunk*64 + (i>>3)*32 + (lane>>4)*8 + (i&7)][ntile*16 + (lane&15)].
// Byte count is unchanged (K*N), so packed_weight keeps the same numel and
// graph-stable address; runs outside the timed region and Graph capture.
__global__ __launch_bounds__(kPackThreads) void
w8a8_pack_b_fragslot_kernel(const int8_t* __restrict__ raw,  // [K, N]
                            int8_t* __restrict__ packed,
                            int k,
                            int n) {
  const int64_t tid =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int k_chunks = k / (kDummaTileK * 2);
  const int64_t total = static_cast<int64_t>(n / 16) * k_chunks * 64;
  if (tid >= total) {
    return;
  }
  const int lane = static_cast<int>(tid & 63);
  const int64_t slot = tid >> 6;  // ntile * k_chunks + chunk
  const int chunk = static_cast<int>(slot % k_chunks);
  const int ntile = static_cast<int>(slot / k_chunks);
  const int nn = lane & 15;
  const int k8 = (lane >> 4) << 3;
  const int8_t* src0 =
      raw + (static_cast<int64_t>(chunk) * (2 * kDummaTileK) + k8) * n +
      ntile * 16 + nn;
  const int8_t* src1 = src0 + static_cast<int64_t>(kDummaTileK) * n;
  int8_t* dst = packed + tid * kChunkSlotBytes;
#pragma unroll
  for (int i = 0; i < kPackedSlotBytes; ++i) {
    dst[i] = src0[static_cast<int64_t>(i) * n];
    dst[kPackedSlotBytes + i] = src1[static_cast<int64_t>(i) * n];
  }
}

// Identity device-to-device copy of the per-column fp32 scales.
__global__ void w8a8_pack_scale_identity_kernel(
    const float* __restrict__ src,
    float* __restrict__ dst,
    int numel) {
  const int idx = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < numel) {
    dst[idx] = src[idx];
  }
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
  (void)workspace;
  (void)workspace_bytes;
  if (m <= 0 || n <= 0 || k <= 0) {
    return;
  }
  // Exact-shape DUMMA specialization: tp8 wo_b (M=16, N=4096, K=1024).
  // Two wavefronts per block (in-block split-K=2 along K), one 16x16 N tile
  // per block; grid = N/16 = 256 blocks >= 120 CUs -> 512 wavefronts total
  // (~4.3 waves/CU). Stage-K=64 double-buffered A+B prefetch with packed-B
  // fragment-slot loads (b is the iteration-19 packed [n_tile][k_chunk]
  // [lane][16] layout for this shape: one 16-B load per lane per K=64
  // chunk).
  if (m == 16 && n == 4096 && k == 1024) {
    constexpr int kStageK = 64;
    const dim3 grid(static_cast<unsigned>(n / 16));
    const dim3 block(128);
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(w8a8_dumma_m16_n16_2wave_dbuf_kernel<kStageK>),
        grid,
        block,
        0,
        stream,
        a,
        b,
        x_scale,
        weight_scale,
        static_cast<hip_bfloat16*>(out),
        k,
        n);
    return;
  }
  // Generic scalar fallback for every other (m, n, k), including the paired
  // M=2 shape with the same (N, K); the scalar kernel decodes the packed-B
  // layout for (k, n) == (1024, 4096).
  const int64_t total = static_cast<int64_t>(m) * n;
  const int64_t blocks = (total + kScalarThreads - 1) / kScalarThreads;
  hipLaunchKernelGGL(
      w8a8_gemm_scalar_kernel,
      dim3(static_cast<unsigned>(blocks)),
      dim3(kScalarThreads),
      0,
      stream,
      a,
      b,
      x_scale,
      weight_scale,
      static_cast<hip_bfloat16*>(out),
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
  const int64_t weight_elems = static_cast<int64_t>(k) * n;
  if (weight_elems > 0) {
    if (k == 1024 && n == 4096) {
      // Exact (K, N): pack B into the [n_tile][k_step][lane][8] fragment-
      // slot layout consumed by the iteration-9 DUMMA kernel (one-time,
      // out of the timed region and Graph capture, same byte count).
      const int64_t slots = static_cast<int64_t>(n / 16) * (k / 32) * 64;
      const unsigned grid = static_cast<unsigned>(
          (slots + kPackThreads - 1) / kPackThreads);
      hipLaunchKernelGGL(
          HIP_KERNEL_NAME(w8a8_pack_b_fragslot_kernel),
          dim3(grid),
          dim3(kPackThreads),
          0,
          stream,
          raw_weight,
          packed_weight,
          k,
          n);
    } else {
      // Bootstrap: identity device-to-device packing for every other (K, N).
      const int64_t blocks = (weight_elems + kPackThreads - 1) / kPackThreads;
      hipLaunchKernelGGL(
          w8a8_pack_identity_kernel,
          dim3(static_cast<unsigned>(blocks)),
          dim3(kPackThreads),
          0,
          stream,
          raw_weight,
          packed_weight,
          weight_elems);
    }
  }
  if (n > 0) {
    const int blocks = (n + kPackThreads - 1) / kPackThreads;
    hipLaunchKernelGGL(
        w8a8_pack_scale_identity_kernel,
        dim3(static_cast<unsigned>(blocks)),
        dim3(kPackThreads),
        0,
        stream,
        weight_scale,
        packed_weight_scale,
        n);
  }
}
// @@end
