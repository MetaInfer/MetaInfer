// @@variant shape=glm_tp8_o_proj_m16 commit=c1cfe1975c2fcb8d4bc23c86e8026afae225a4c4 added=2026-08-31
//   median_us=24 p90_us=24.02
//   source=glm5-2-dsh-tp8-m16-test1-cb4d262c
// MetaInfer INT8 W8A8 GEMM for Hygon K500SM_AI / gfx928.
//
// Iteration 10 (worker_2): HIP-only request-granularity round on the accepted
// iteration-6 baseline (the harness restored the accepted source digest
// 15eae0f3... after iteration 9's rejection). The occupancy sweep (iterations
// 7, 8: split 3 -> 27.02 us, split 1 -> 29.76 us) and the per-wait MLP probe
// (iteration 9: k128 stages, 4 KiB/wait -> 26.46 us) all falsified their
// models, leaving the 768-stream / 2 KiB-in-flight / 16-wait-per-wave
// configuration the measured memory-system sweet spot. The one untested
// dimension at that sweet spot is per-instruction request granularity: the
// iteration-6 steady state issues FOUR global_load_dwordx2 per k64 step (2 A
// halves + 2 B halves of 512 B each per wave), and iteration 9's prediction
// named the 16-B true-vector packed-B variant as the next attack if deeper
// per-wait MLP stayed flat/regressed (it regressed). Change: repack B into
// packed16[n_tile][k64_step][lane][16] -- each lane's two k32 fragment slots
// (lower half bytes 0..7, upper half bytes 8..15) side by side, a pure
// same-byte-count permutation -- so each wave's 1-KiB B slice per k64 step is
// ONE aligned 16-B-per-lane global_load_dwordx4 instead of two 8-B dwordx2
// loads. Stream count (768), per-wait in-flight bytes (2 KiB), wait count
// (16/wave), A transport, geometry, split, LDS/barrier, global bytes and the
// exact k-ascending int32 order are all unchanged.
//
// Iteration 10 repair 2: the first iteration-10 draft read the packed-B slot
// with the k64-step index missing its *kWaveSize stride (b_slots[ps0+ps]
// instead of b_slots[(ps0+ps)*kWaveSize]), so every k64 step after the first
// loaded a slot 16 B away instead of 1 KiB away and corrupted ~all outputs
// (98211/98304 mismatches). The 16-B slot layout, the one-dwordx4-per-step
// load and the k64 pipeline are all preserved; the wave's slot anchor is now
// b_base + ps0*kWaveSize + lane with in-loop index ps*kWaveSize.
//
// Iteration 6 (worker_2, physical GPU 2): HIP-only pipeline round on the
// assigned shape glm_tp8_o_proj_m16 (M=16, N=6144, K=2048). The staging-family
// decision is DIRECT (no LDS staging for A or B), argued from the iteration-5
// L2/VMEM PMC evidence: l2_misses 201,111 ~= 196,608 unique B 64-B lines
// (12,582,912 B = the whole once-read B stream) so B has ZERO reuse and an
// LDS round trip for it would only re-add the ds_write/ds_read + lgkmcnt
// chain that iteration 4 priced at 45.03 us vs the 26.83 us direct kernel;
// A's HBM footprint is ~88,960 B/replay (~0.7% of traffic, 32 KiB L2-resident
// working set) so A-only staging saves no HBM bytes and cannot remove a wait
// from the step chain (A and B loads share one vmcnt). The measured per-wave
// budget is ~28,475 cycles over 32 k32 steps (~890/step): the kernel is
// serialized per-step global-latency-wait bound (in-flight ~1 KiB/wave at
// each wait), not issue-bound and not HBM-bound (479 GB/s logical << peak).
// Change: keep the exact iteration-5 geometry, packed layout, bytes and
// int32 accumulation, but restructure the K loop from 32 one-wait k32 steps
// into 16 k64 pipeline steps -- each step = TWO m16n16k32 halves, all four
// per-lane 8-B cooperative loads (2 A + 2 B) issued together and consumed by
// the step's two serialized du_mma_sync calls, so ONE vmcnt(0) wait serves
// 2 x K; the per-wave wait count halves 32 -> 16 and the bytes in flight at
// each wait point double (2 KiB/wave), with depth-1 k64 prefetch (step j+1's
// four loads issued before step j's two-MMAC burst).
//
// History: iteration 1 established the minimal native DUMMA tile (scalar
// bootstrap 251 us -> 85.02 us median; 384 one-wave blocks, 3.2 waves/CU);
// iteration 2 (accepted, 50.16 us) doubled resident waves to 6.4/CU via
// in-block split-K=2 (384 blocks x 128 thr); iteration 3 (rejected, 51.50 us)
// probed grid-side split-K=5 at 16.0 one-wave blocks/CU and was flat -- extra
// co-resident waves are NOT the lever; iteration 4 (accepted, 45.03 us) added
// bounded depth-1 register prefetch of the global loads, which removed the
// two serialized global-latency waits per step but left the per-step LDS
// fragment-read chain (8 ds_read_u8 B byte reads at stride kBStride + packing
// VALU + ~3 ds_write, 436,224 LDS bank conflicts) as the dominant per-wave
// cost; iteration 5 (accepted, 26.83 us) implemented the packed-layout arm of
// the mandate: launch_pack_w8a8_weight permutes the exact (k,n)==(2048,6144)
// weight ONCE, outside the timed region and out of Graph capture, into the B
// fragment-slot layout packed[n_tile][k_step][lane][8] (same byte count K*N
// as the logical weight; identity copy retained for every other (K,N)); the
// DUMMA kernel then consumes A and B directly from global memory as two
// aligned 8-B vector loads per lane per k32 step (register-only K-loop
// transport with depth-1 prefetch), eliminating the LDS staging round trip,
// the 8 ds_read_u8 byte reads, the packing VALU, and every in-loop lgkmcnt
// wait.
//
// Logical contract (see int8_w8a8_gemm_api.py):
//   out[m, n] = bf16( int32_dot(x_q[m, :], raw_weight[:, n])
//                     * x_scale[m, 0] * weight_scale[n, 0] )
//
// This file provides the two stable host launch symbols:
//   launch_w8a8_gemm(...)         -- timed operator, graph-safe
//   launch_pack_w8a8_weight(...)  -- optional out-of-timed-region weight prep
//
// Iteration-5 round strategy (packed-B fragment-slot transport):
//   * the fragment lane mapping is taken from the exact DTK du_mma.hpp
//     loaders, which the accepted iteration-2/4 kernels call with the SAME
//     m16n16k32 signed-char fragment type that du_mma_sync consumes (verified
//     against the iteration-4 code object's ds_read address arithmetic):
//       matrix_a row_major: row = lane & 15, col = (lane >> 4) << 3,
//                           x[i] = p[row*ldm + col + i]  (8 consecutive B);
//       matrix_b row_major: row = lane & 15, col = (lane >> 4) << 3,
//                           x[i] = p[(col + i)*ldm + row];
//     so each lane's B fragment bytes for one k32 step are the 8 k-ascending
//     bytes  raw[(32*s + 8*(lane>>4) + i) * n + 16*t + (lane&15)], i=0..7;
//   * pack kernel: one thread per (n_tile, k_step, lane) writes the 8 raw
//     bytes (strided by n) into one aligned 8-B slot of the same-size packed
//     tensor (permutation, byte count unchanged); runs once per weight,
//     outside the timed GEMM and outside Graph capture;
//   * DUMMA kernel: grid = N/16 = 384 blocks x 128 thr = 2 wavefronts,
//     in-block split-K=2 (wave w owns k32 steps [32w, 32w+32) = K=1024 each)
//     -- the exact proven iteration-2/4 geometry and resident-wave count
//     (768 waves = 6.4/CU, all 120 CUs at 3.2 blocks/CU) is frozen; per k32
//     step each lane issues ONE aligned 8-B A load (logical row-major A:
//     a + (lane&15)*k + 32*s + 8*(lane>>4), L2-hot 32 KiB, reused by all 384
//     blocks) and ONE aligned 8-B packed-B load (fully coalesced: 64 lanes x
//     8 B = 512 B contiguous per step), with depth-1 prefetch of the NEXT
//     stage's eight loads (4 steps x 2) issued before the current 4-MMAC
//     burst (#pragma-unrolled K=128 stage), so the compiler-inserted vmcnt(0)
//     wait lands at the next stage's first fragment fill and no LDS hop /
//     lgkmcnt wait exists in the K loop;
//   * split combine unchanged: wave 0 publishes its 16x16 int32 partial into
//     the single LDS c_part plane, one END-of-K __syncthreads(), wave 1 adds
//     both halves (order-independent: no partial can overflow int32, max
//     |dot| = 2048*127*127 << 2^31) and emits the scaled bf16 output from
//     registers -- bit-identical to the exact k-ascending int32 reference;
//   * the launch is guarded by the exact (m,n,k)=(16,6144,2048) shape; every
//     other shape -- including the paired M=2 API shape with the same (N,K) --
//     keeps the scalar generic fallback, which now decodes the packed layout
//     elementwise for (n,k)==(6144,2048) and reads the logical row-major
//     weight for every other (K,N);
//   * pack_weight stays an identity device-to-device copy for unmatched
//     (K,N); the DUMMA kernel consumes packed_weight for the assigned shape.

// Known-good include order for this DTK: HIP runtime first, then
// hip_bfloat16.h, then du_mma.h (du_mma.h is not self-contained when it is
// included before the HIP runtime headers).
#include <hip/hip_runtime.h>
#include <hip/hip_bfloat16.h>
#include <du_mma.h>

#include <cstdint>

namespace {

// gfx928 wavefront is 64 lanes; every block dimension must be a multiple of
// 64. 128 threads = 2 wavefronts keeps the scalar bootstrap simple while
// still giving a straightforward grid over M*N.
constexpr int kGemmThreads = 128;
constexpr int kPackThreads = 256;

// Assigned-shape DUMMA constants.
constexpr int kTileM = 16;
constexpr int kTileN = 16;
constexpr int kTileK = 32;  // gfx928 INT8 DUMMA K depth
constexpr int kWaveSize = 64;

// Iteration-2 launch geometry: two 64-lane wavefronts per block, each owning
// a contiguous K-half of the block's single 16x16 N tile (in-block split-K=2,
// 128 threads/block, no cross-wave barrier inside the K loop).
constexpr int kWavesPerTile = 2;
constexpr int kTileThreads = kWavesPerTile * kWaveSize;

// K=64 pipeline step: two m16n16k32 halves per wait point (iteration 6).
constexpr int kPipeK = 2 * kTileK;            // 64
constexpr int kPipeHalves = kPipeK / kTileK;  // 2 halves per pipeline step

// 16-B packed-B fragment slot (iteration 10): a lane's TWO adjacent k32
// fragment 8-B slots for one k64 pipeline step, side by side (lower k32 half
// in .lo, upper in .hi), so the wave's whole 1-KiB B slice per step is ONE
// aligned 16-B-per-lane global load (global_load_dwordx4) instead of two
// 512-B dwordx2 loads. Pure same-byte-count permutation of the packed tensor.
struct alignas(16) BSlot {
  uint64_t lo;
  uint64_t hi;
};

// Round-to-nearest-even float -> bf16 bit pattern.
//
// Bit-compatible with the trusted exact reference, which converts the scaled
// fp32 value with PyTorch's BFloat16 conversion (c10 BFloat16
// round_to_bfloat16: add 0x7fff + lsb, then truncate). Timed values are
// finite (small int8 dots scaled by <= 1.0), so no inf/nan special casing is
// needed to match the reference; denormals round identically.
__device__ __forceinline__ uint16_t bf16_bits_from_float(float f) {
  union {
    float f32;
    uint32_t u32;
  } cvt;
  cvt.f32 = f;
  const uint32_t lsb = (cvt.u32 >> 16) & 1u;
  cvt.u32 += 0x7fffu + lsb;
  return static_cast<uint16_t>(cvt.u32 >> 16);
}

// One-time (out-of-timed-region, out-of-Graph) pack for the exact assigned
// weight (k,n) == (2048, 6144): permute the logical [K,N] int8 layout into
// packed[n_tile][k64_step][lane][16] B fragment slots (iteration 10), where
// for n_tile t (0..383), k64_step ps (0..31) and lane l (0..63), bytes
// 0..7 (lower k32 half) and 8..15 (upper k32 half) are
//   byte 8*h+i = raw[(64*ps + 32*h + 8*(l>>4) + i) * n + 16*t + (l&15)]
// -- exactly the bytes du_load_matrix_sync assigned to lane l's B fragment
// for the two m16n16k32 steps of the k64 pipeline step in the accepted
// iteration-2/4/6 kernels (du_mma.hpp matrix_b row_major: row = lane & 15,
// col = (lane >> 4) << 3, x[i] = p[(col + i)*ldm + row]), k-ascending in
// memory order. One thread per (t, ps, l) writes one aligned 16-B slot; byte
// count unchanged (K*N), so the packed tensor is a same-size permutation of
// the iteration-5/6 packed[n_tile][k_step][lane][8] layout (the two 8-B slots
// for k32 steps 2*ps and 2*ps+1 of the same lane become one contiguous
// 16-B slot). The identity copy remains the generic fallback for every other
// (K, N).
__global__ __launch_bounds__(kPackThreads) void w8a8_pack_o_proj_tp8_m16_kernel(
    const int8_t* __restrict__ raw,
    int8_t* __restrict__ packed,
    int k,
    int n) {
  const int64_t k64_steps = (k / kTileK) / kPipeHalves;  // 32 for k=2048
  const int64_t n_tiles = n / kTileN;                    // 384 for n=6144
  const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x +
                      static_cast<int64_t>(threadIdx.x);
  const int64_t total = n_tiles * k64_steps * kWaveSize;
  if (idx >= total) {
    return;
  }
  const int64_t t = idx / (k64_steps * kWaveSize);  // n_tile
  const int64_t rem = idx % (k64_steps * kWaveSize);
  const int64_t ps = rem / kWaveSize;  // k64 step
  const int lane = static_cast<int>(rem % kWaveSize);
  const int row = lane & 15;                 // B column within the tile
  const int col = (lane >> 4) << 3;          // k offset within each k32 half
  const int8_t* src =
      raw + (ps * kPipeK + col) * n + t * kTileN + row;
  uint64_t lo = 0;
  uint64_t hi = 0;
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    lo |= (static_cast<uint64_t>(static_cast<uint8_t>(src[i * n]))) << (8 * i);
    hi |= (static_cast<uint64_t>(static_cast<uint8_t>(src[(kTileK + i) * n])))
          << (8 * i);
  }
  BSlot slot;
  slot.lo = lo;
  slot.hi = hi;
  __builtin_memcpy(packed + idx * 16, &slot, sizeof(BSlot));
}

// 16x16x32 DUMMA tile with 2 wavefronts, in-block split-K=2, packed-B
// fragment-slot register-only transport and a k64 pipelined K loop for the
// assigned M=16 decode shape (iteration-6 pipeline round; iteration 10 moves
// the B transport to one aligned 16-B-per-lane load per k64 step).
//
// Launch geometry: grid = N/16 = 384 blocks, 128 threads (2 wavefronts) per
// block, one 16x16 output tile per block, so 384*2 = 768 independent waves
// are resident across the 120 CUs (6.4 waves/CU) -- identical to the accepted
// iteration-2/4/5 geometry. Each wave owns a contiguous K-half (K=1024 = 16
// k64 pipeline steps) of the tile:
//   * per k64 step each lane issues THREE aligned cooperative loads -- A
//     halves h=0,1 (logical row-major A: row = lane&15, col = 8*(lane>>4),
//     8 consecutive bytes per half, du_mma.hpp matrix_a row_major fragment
//     mapping) as two 8-B dwordx2, and the packed-B k64 slot
//     (packed[n_tile][k64_step][lane][16]; 64 lanes x 16 B = 1 KiB coalesced
//     per step, ONE global_load_dwordx4 instead of the two 512-B dwordx2
//     loads of iteration 6; .lo = lower k32 half, .hi = upper) -- and the
//     step's two serialized m16n16k32 du_mma_sync calls (k-ascending: lower
//     half first) consume all three loads, so ONE compiler-inserted
//     s_waitcnt vmcnt(0) serves 2 x K (2 KiB in flight per wave at the wait,
//     unchanged byte count and stream count vs iteration 6);
//   * depth-1 k64 prefetch: step j+1's three loads are issued BEFORE step
//     j's two-MMAC burst, so the wait lands at the next fill -- no LDS hop,
//     no lgkmcnt wait anywhere in the K loop; the per-wave wait count stays
//     16 (the iteration-6 sweet spot; iteration 9's 8-wait/4-KiB probe
//     regressed 26.46 us and is NOT repeated);
//   * split combine unchanged: wave 0 publishes its 16x16 int32 partial into
//     one LDS plane, one END-of-K __syncthreads(), wave 1 adds its own
//     partial (k-ascending halves; no partial sum can overflow int32, so any
//     split-sum order is bit-identical to the exact k-ascending reference)
//     and performs the direct-register scale/bf16 epilogue.
__global__ __launch_bounds__(kTileThreads) void
w8a8_gemm_m16_dumma_packedb_kernel(
    const int8_t* __restrict__ a,
    const int8_t* __restrict__ packed_b,
    const float* __restrict__ x_scale,
    const float* __restrict__ weight_scale,
    hip_bfloat16* __restrict__ out,
    int n,
    int k) {
  const int wave = static_cast<int>(threadIdx.x) / kWaveSize;
  const int lane = static_cast<int>(threadIdx.x) % kWaveSize;
  const int n_tile = static_cast<int>(blockIdx.x);

  __shared__ __align__(16) int32_t c_part[kTileM * kTileN];

  du::dumma::DUFragment<du::dumma::matrix_a, kTileM, kTileN, kTileK,
                        signed char, du::dumma::row_major>
      a_frag;
  du::dumma::DUFragment<du::dumma::matrix_b, kTileM, kTileN, kTileK,
                        signed char, du::dumma::row_major>
      b_frag;
  du::dumma::DUFragment<du::dumma::accumulator, kTileM, kTileN, kTileK, int>
      acc_frag;
  du::dumma::du_fill_fragment(acc_frag, 0);

  // Per-lane fragment ownership is loop-invariant (du_mma.hpp matrix_a /
  // matrix_b row_major loaders): row = lane & 15, col = (lane >> 4) << 3.
  const int a_row = lane & 15;
  const int a_col = (lane >> 4) << 3;

  const int64_t k_steps = k / kTileK;  // 64
  const int64_t s_per_wave = k_steps / kWavesPerTile;  // 32 k32 steps per wave
  const int64_t p_steps = s_per_wave / kPipeHalves;    // 16 k64 steps per wave
  const int64_t s0 = wave * s_per_wave;                // first step of wave
  const int64_t k64_steps = k_steps / kPipeHalves;     // 32
  const int64_t ps0 = wave * p_steps;                  // first k64 step of wave

  // Packed-B region of this block's n_tile: [n_tile * k64_steps + ps][lane][16]
  // (one aligned 16-B BSlot per lane per k64 step, iteration 10). The packed
  // slot index for (ps_global, lane) is b_base + ps_global * kWaveSize + lane
  // (pack kernel: idx = t*k64_steps*kWaveSize + ps*kWaveSize + lane). Repair 2:
  // the iteration-10 change dropped the *kWaveSize stride on the ps index and
  // read slots 16 B apart instead of 1 KiB apart after the first k64 step,
  // corrupting every output (98211/98304 mismatches). Anchor this wave's slice
  // at ps0 * kWaveSize so the in-loop index is just ps * kWaveSize.
  const int64_t b_base = n_tile * k64_steps * kWaveSize +
                         ps0 * kWaveSize + lane;  // in 16-B slots
  const BSlot* b_slots = reinterpret_cast<const BSlot*>(packed_b) + b_base;

  // Logical row-major A: lane's 8 bytes at a + a_row*k + 32*s + a_col.
  const uint64_t* a_slots =
      reinterpret_cast<const uint64_t*>(a + a_row * k + a_col);

  // Prologue: k64 step 0 (k32 steps s0, s0+1) lands directly in registers
  // (one cold-start DRAM latency per wave, amortized over 16 steps; the two
  // waves' cold starts overlap). B arrives as one 16-B slot per lane.
  uint64_t ca0 = a_slots[(s0 * kTileK) / 8];
  uint64_t ca1 = a_slots[((s0 + 1) * kTileK) / 8];
  BSlot cb;
  __builtin_memcpy(&cb, &b_slots[0], sizeof(BSlot));
  uint64_t cb0 = cb.lo;
  uint64_t cb1 = cb.hi;

  for (int64_t ps = 0; ps < p_steps; ++ps) {
    // Issue the NEXT k64 step's global loads (2 A dwordx2 + ONE B dwordx4 =
    // 2 KiB per wave in flight per wait point -- same bytes and stream count
    // as iteration 6, one instruction fewer per step) BEFORE this step's
    // two-MMAC burst so DRAM/L2 latency overlaps the burst; the
    // compiler-inserted s_waitcnt vmcnt(0) lands at the next iteration's
    // first fragment fill -- no LDS hop.
    uint64_t na0, na1;
    BSlot nb;
    if (ps + 1 < p_steps) {
      const int64_t step = s0 + (ps + 1) * kPipeHalves;
      na0 = a_slots[(step * kTileK) / 8];
      na1 = a_slots[((step + 1) * kTileK) / 8];
      __builtin_memcpy(&nb, &b_slots[(ps + 1) * kWaveSize], sizeof(BSlot));
    }
    // Step-ps MMAC pair straight from registers, k-ascending (lower k32 half
    // first), in the exact a_frag.x / b_frag.x byte order du_mma_sync consumes
    // (one 64-bit value per lane). All three loads were issued together, so
    // the single vmcnt(0) wait that precedes this fill serves the whole k64
    // step (2 x K per wait point).
    __builtin_memcpy(a_frag.x, &ca0, 8);
    __builtin_memcpy(b_frag.x, &cb0, 8);
    du::dumma::du_mma_sync(acc_frag, a_frag, b_frag, acc_frag);
    __builtin_memcpy(a_frag.x, &ca1, 8);
    __builtin_memcpy(b_frag.x, &cb1, 8);
    du::dumma::du_mma_sync(acc_frag, a_frag, b_frag, acc_frag);
    // Rotate: the prefetched k64 step ps+1 becomes the current step.
    if (ps + 1 < p_steps) {
      ca0 = na0;
      ca1 = na1;
      cb0 = nb.lo;
      cb1 = nb.hi;
    }
  }

  // Split combine + direct fragment epilogue (verified gfx928 INT8 m16n16k32
  // accumulator ownership: row = lane&15, col_mod4 = lane>>4, x[i] -> col
  // col_mod4+4*i).
  const int row = lane & 15;
  const int col_mod4 = lane >> 4;
  if (wave == 0) {
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      c_part[row * kTileN + col_mod4 + 4 * i] = acc_frag.x[i];
    }
  }
  __syncthreads();  // single END-of-K barrier (cheap, order-independent)
  if (wave == 1) {
    const float xs = x_scale[row];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int out_col = n_tile * kTileN + col_mod4 + 4 * i;
      const int32_t total =
          acc_frag.x[i] + c_part[row * kTileN + col_mod4 + 4 * i];
      const float scaled =
          static_cast<float>(total) * xs * weight_scale[out_col];
      out[row * n + out_col] = __float2bfloat16(scaled);
    }
  }
}

// Scalar INT8 W8A8 GEMM: one thread per output element.
//
//   a: [M, K] int8, row-major (x_q)
//   b: [K, N] int8, row-major (packed_weight; identity-packed in bootstrap)
//   x_scale:      [M] fp32
//   weight_scale: [N] fp32
//   out: [M, N] bf16, row-major
//
// Linear index idx = row * n + col, so consecutive lanes own consecutive N
// addresses. The full K dot is accumulated exactly in int32 (k-ascending,
// matching the exact int64 reference sum), then scaled in contract order.
// For the exact (n,k) == (6144, 2048) pair the weight tensor is the packed
// [n_tile][k64_step][lane][16] fragment-slot layout (see
// w8a8_pack_o_proj_tp8_m16_kernel), which this fallback decodes elementwise
// so the paired M=2 API shape with the same (N,K) stays byte-exact; every
// other (K,N) reads the logical row-major weight.
__global__ __launch_bounds__(kGemmThreads) void w8a8_scalar_gemm_kernel(
    const int8_t* __restrict__ a,
    const int8_t* __restrict__ b,
    const float* __restrict__ x_scale,
    const float* __restrict__ weight_scale,
    hip_bfloat16* __restrict__ out,
    int m,
    int n,
    int k) {
  const int64_t total = static_cast<int64_t>(m) * n;
  const int64_t stride =
      static_cast<int64_t>(gridDim.x) * static_cast<int64_t>(blockDim.x);
  for (int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x +
                     static_cast<int64_t>(threadIdx.x);
       idx < total; idx += stride) {
    const int row = static_cast<int>(idx / n);
    const int col = static_cast<int>(idx - static_cast<int64_t>(row) * n);
    const int8_t* a_row = a + static_cast<int64_t>(row) * k;
    const bool packed_b = (n == 6144 && k == 2048);
    int32_t acc = 0;
    if (packed_b) {
      // Decode packed[n_tile][k64_step][lane][16] (iteration 10): element
      // (kk, col) is byte 8*h+j of the 16-B slot for n_tile = col>>4,
      // k64_step = kk>>6, lane = (((kk&63)>>5)*32 + ((kk&31)>>3))*16 +
      // (col&15), where bytes 0..7 are the lower k32 half and bytes 8..15 the
      // upper half (a pure permutation of the iteration-5/6 packed layout,
      // byte count unchanged).
      const int64_t t = col >> 4;
      for (int kk = 0; kk < k; ++kk) {
        const int64_t ps = kk >> 6;
        const int64_t h = (kk >> 5) & 1;
        const int64_t c = (kk & 31) >> 3;
        const int64_t j = kk & 7;
        const int64_t slot =
            (((t * (k / kPipeK) + ps) * kWaveSize) + (c << 4) + (col & 15)) *
                16 +
            (h << 3) + j;
        acc += static_cast<int32_t>(a_row[kk]) *
               static_cast<int32_t>(b[slot]);
      }
    } else {
      const int8_t* b_col = b + col;
      for (int kk = 0; kk < k; ++kk) {
        acc += static_cast<int32_t>(a_row[kk]) *
               static_cast<int32_t>(b_col[static_cast<int64_t>(kk) * n]);
      }
    }
    const float scaled = static_cast<float>(acc) * x_scale[row] *
                         weight_scale[col];
    reinterpret_cast<uint16_t*>(out)[idx] = bf16_bits_from_float(scaled);
  }
}

// Generic elementwise device-to-device copy used by the identity pack_weight
// bootstrap. Handles every (K, N) through a grid-stride loop, so it stays
// valid as the generic fallback for unmatched (K, N) in later rounds.
template <typename T>
__global__ __launch_bounds__(kPackThreads) void w8a8_identity_pack_kernel(
    const T* __restrict__ src,
    T* __restrict__ dst,
    int64_t count) {
  const int64_t stride =
      static_cast<int64_t>(gridDim.x) * static_cast<int64_t>(blockDim.x);
  for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x +
                   static_cast<int64_t>(threadIdx.x);
       i < count; i += stride) {
    dst[i] = src[i];
  }
}

}  // namespace

// Timed operator entry point. Graph-safe: launches only on `stream`; no
// allocation, compilation, autotuning, packing, or host/device sync.
//
// The assigned shape (m,n,k) == (16, 6144, 2048) is dispatched to the
// packed-B two-wave in-block split-K=2 DUMMA kernel (384 blocks x 128
// threads, one 16x16 tile per block, register-only transport). Every other
// (m, n, k) -- including the paired M=2 API shape with the same (N, K) --
// takes the scalar generic fallback (which decodes the packed layout for
// (n,k) == (6144, 2048)).
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
  if (m == 16 && n == 6144 && k == 2048) {
    const unsigned blocks = static_cast<unsigned>(n / kTileN);
    hipLaunchKernelGGL(
        w8a8_gemm_m16_dumma_packedb_kernel, dim3(blocks), dim3(kTileThreads), 0,
        stream, a, b, x_scale, weight_scale, reinterpret_cast<hip_bfloat16*>(out),
        n, k);
    return;
  }
  const int64_t total = static_cast<int64_t>(m) * n;
  const unsigned blocks =
      static_cast<unsigned>((total + kGemmThreads - 1) / kGemmThreads);
  hipLaunchKernelGGL(
      w8a8_scalar_gemm_kernel, dim3(blocks), dim3(kGemmThreads), 0, stream,
      a, b, x_scale, weight_scale, reinterpret_cast<hip_bfloat16*>(out), m,
      n, k);
}

// Optional out-of-timed-region weight prep. For the exact (k, n) ==
// (2048, 6144) o_proj weight this permutes into the packed-B fragment-slot
// layout (one-time, untimed, outside Graph capture); every other (K, N)
// keeps the generic identity device-to-device copy. Weight scales are always
// copied through (identity [N] fp32).
extern "C" void launch_pack_w8a8_weight(
    const int8_t* raw_weight,
    const float* weight_scale,
    int8_t* packed_weight,
    float* packed_weight_scale,
    int k,
    int n,
    hipStream_t stream) {
  if (k == 2048 && n == 6144) {
    const int64_t total =
        static_cast<int64_t>(n / kTileN) * ((k / kTileK) / kPipeHalves) *
        kWaveSize;  // one thread per (n_tile, k64_step, lane) 16-B slot
    const unsigned blocks = static_cast<unsigned>(
        (total + kPackThreads - 1) / kPackThreads);
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(w8a8_pack_o_proj_tp8_m16_kernel), dim3(blocks),
        dim3(kPackThreads), 0, stream, raw_weight, packed_weight, k, n);
  } else {
    const int64_t weight_elems = static_cast<int64_t>(k) * n;
    const unsigned weight_blocks = static_cast<unsigned>(
        (weight_elems + kPackThreads - 1) / kPackThreads);
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(w8a8_identity_pack_kernel<int8_t>), dim3(weight_blocks),
        dim3(kPackThreads), 0, stream, raw_weight, packed_weight, weight_elems);
  }

  const int64_t scale_elems = n;
  const unsigned scale_blocks = static_cast<unsigned>(
      (scale_elems + kPackThreads - 1) / kPackThreads);
  hipLaunchKernelGGL(
      HIP_KERNEL_NAME(w8a8_identity_pack_kernel<float>), dim3(scale_blocks),
      dim3(kPackThreads), 0, stream, weight_scale, packed_weight_scale,
      scale_elems);
}
// @@end
