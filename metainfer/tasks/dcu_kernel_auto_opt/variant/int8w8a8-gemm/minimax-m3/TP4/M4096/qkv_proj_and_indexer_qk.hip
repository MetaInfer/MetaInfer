// @@variant shape=minimax_tp4_qkv_proj_and_indexer_qk_m4096 commit=d6cbef3829c738228f5dab5255b89c384c66c6ef added=2026-08-24
//   median_us=1082 p90_us=1089
//   source=minimaxm3-dsh-tp4-m4096-1-0c2f84a9
// MetaInfer W8A8 INT8 GEMM bootstrap for gfx928 (K500SM_AI).
//
// Worker: worker_1 (physical GPU 1)
// Assigned shape: minimax_tp4_qkv_proj_and_indexer_qk_m4096
//                 (M=4096, N=2560, K=6144)
//
// Operator contract (fixed by the control plane):
//   out[m, n] = bf16( int32_dot(x_q[m, :], raw_weight[:, n])
//                     * x_scale[m] * weight_scale[n] )
// The packed weight layout is opaque to the API: the exact assigned shape
// (K=6144, N=2560) is packed once by launch_pack_w8a8_weight outside the
// timed region into the iteration-5 fragment-interleaved swizzle (element
// (n, k) at ((k>>5)*160 + (n>>4))*512 + (((k>>3)&3)*16 + (n&15))*8 + (k&7);
// same byte count k*n as the identity copy, so allocations and graph-stable
// addresses are unchanged); every other (K, N) keeps the logical [K, N]
// row-major identity copy.
//
// Iteration 1 (DUMMA 2-D macro-tile baseline sweep, mandatory decision):
//   * The bootstrap DUMMA 64x128 baseline (this file, prior state) measured
//     1442.6 us median / 89.3 TOPS on the assigned shape (M=4096, N=2560,
//     K=6144) with PMC: 36.37M LDS instructions, 32% LDS bank conflicts,
//     40.3M LDS waits (1.11x LDS instructions, dominant stall class), 1.67M
//     VMEM reads (1.51 GB through L2), 73.5% L2 hit, 2 blocks/CU. Bottleneck
//     hypothesis: the row-major [K, N] B operand is staged into LDS but its
//     m16n16k32 B fragments are re-assembled by the generic loader as EIGHT
//     byte-granular ds_read_u8 per fragment (16-way bank aliasing with the
//     132-byte stride), which inflates the LDS instruction stream and its
//     waits. The trusted hy3 M=4096 lineage (same M and N, K=4096; accepted
//     756.7 us / 113.5 TOPS) fixed exactly this by packing the weight
//     n-major ONCE outside the timed region (packed[n][k]) and loading B
//     fragments col_major as one 8-byte contiguous ds_read2_b64 per fragment
//     (load_b_frag8), hoisting all twelve fragment loads of a 64-K stage
//     before the first v_mmac (no lgkmcnt waits inside the 16-MMAC burst),
//     and storing the epilogue as one coalesced 8-byte store per lane per
//     fragment (100% store sector efficiency). This round ports that accepted
//     lineage to the exact assigned shape (K=6144) and installs the complete
//     2-D macro-tile family (64x64, 64x128, 128x64; each 4 wavefronts x 64
//     lanes = 256 threads, one 32x32/32x64/64x32 quadrant per wave) so the
//     mandated tile sweep can flip the exact-shape dispatch in the next
//     rounds. Round 1 dispatches the exact shape to the 64x128 tile (the
//     strongest N=2560 prior evidence: qkv_proj lineage 113.5 TOPS).
//   * 64x128 packed-B kernel (iteration-1 layout; the iteration-5 swizzle
//     replaces the B tile -- see the top-of-file pack note): A staged
//     row-major into 68-byte LDS rows (17 words, odd-word bank spread,
//     4 ds_write_b32 per 16-byte vector), B staged n-major into 80-byte LDS
//     rows (20 words, five bank phases, one ds_write_b128 per 16-byte
//     vector), B fragments via load_b_frag8 (one ds_read2_b64),
//     single-buffered 64-K K stage with two barriers per stage (the accepted
//     iteration-3 pipeline), all twelve fragment loads hoisted before the
//     16-MMAC burst, coalesced epilogue. LDS/block = 64x68 + 128x80 =
//     14,592 B -> 3 resident blocks/CU (VGPR-bound at arch_vgpr <= 85, per
//     the accepted lineage). Grid (N/128)x(M/64) = 20x64 = 1280 blocks >>
//     120 CUs, no split-K, workspace unused.
//   * 64x64 and 128x64 packed-B kernels are compiled family members for the
//     mandated tile sweep (grid 40x64 = 2560 and 40x32 = 1280 blocks; LDS
//     9,472 B and 15,360 B); they still consume the n-major 80-byte-row tile
//     and are NOT dispatched for the exact shape, which uses 64x128 this
//     round and every round since.
//   * launch_pack_w8a8_weight packs exactly (k, n) == (6144, 2560) once,
//     outside the timed region and Graph capture. Iteration 1 used the
//     n-major transpose packed[n][k]; iteration 5 (packing round) replaces
//     it with the fragment-interleaved swizzle described at the top of the
//     file (same byte count k*n, so allocations and graph-stable addresses
//     are unchanged); every other (k, n) keeps the identity copy.
//   * The scalar int8/int32 fallback decodes the iteration-5 swizzled layout
//     when (k, n) == (6144, 2560) (so the paired M=2 API shape with the same
//     (N, K) stays correct) and the identity layout otherwise.
//   * Iteration 6 (epilogue round): the timed operator is already a single
//     fused kernel -- per-row x_scale, per-column weight_scale, bf16
//     conversion and the coalesced 8-byte-per-lane store all live inside
//     w8a8_dumma_prefill_64x128_packedb_kernel, with no workspace/combine
//     pass (workspace is unused; launch_w8a8_gemm dispatches exactly one
//     kernel). The round restructures the dispatched kernel's epilogue from
//     eight independent per-fragment calls into two fused 4-fragment
//     row-group calls (store_prefill_rowgroup_coalesced + transpose_frag4 +
//     pack_bf16x4): x_scale[row] and the four float4 weight_scale groups are
//     loaded once per lane before the first ds_bpermute, the four transposes
//     issue as one software-pipelined block (all xor-16 ds_bpermutes before
//     xor-32), and the four 8-byte stores issue back-to-back (still 40,960
//     vmem_write instructions, 100% sector efficiency). The int32 transpose
//     routing, per-element multiply order and bf16 rounding are byte-
//     identical to iteration 5, so stored bits stay exact. The generic
//     64x128/64x64 kernels and the compiled-but-undispatched 64x64/128x64
//     packed-B family members keep the iteration-5 epilogue untouched.
//   * Generic 64x128/64x64 DUMMA kernels (identity-packed B) are preserved
//     unchanged for every other large-M shape.
//
// The timed operator (launch_w8a8_gemm) performs no allocation, compilation,
// autotuning, packing, host/device synchronization, or default-stream
// launch: it only dispatches kernels on the caller-provided HIP stream and
// uses only the caller-provided out/workspace tensors (workspace is unused
// in this no-split-K kernel).
//
// Include order is fixed by the control plane: hip_runtime, hip_bfloat16,
// then du_mma (this DTK's du_mma.h is not self-contained otherwise).

#include <hip/hip_runtime.h>
#include <hip/hip_bfloat16.h>
#include <du_mma.h>

#include <cstdint>

namespace {

constexpr int kWaveSize = 64;
constexpr int kTileM = 16;
constexpr int kTileN = 16;
constexpr int kTileK = 32;
// 4 wavefronts; must remain a multiple of the gfx928 wavefront size (64).
constexpr int kThreadsPerBlock = 256;

// 64x64 generic kernel geometry: K stage of 128 (unpadded rows).
constexpr int kBlockM = 64;
constexpr int kBlockN = 64;
constexpr int kStageK = 128;

// 64x128 large-prefill kernel geometry: single-buffered 64-K K stage with
// odd-word padded row strides (A: 68 B = 17 words, B: 132 B = 33 words).
constexpr int kBlockM128 = 64;
constexpr int kBlockN128 = 128;
constexpr int kStageK128 = 64;
constexpr int kAStride128 = kStageK128 + 4;  // 68 bytes per A row (17 words)
constexpr int kBStride128 = kBlockN128 + 4;  // 132 bytes per B row (33 words)

// Iteration 1 (packed-B family): the exact assigned shape packs the weight
// once (outside timing) and stages B into an LDS tile. The compiled 64x64
// and 128x64 family members (not dispatched) still use the round-1 n-major
// layout: packed[n][k] staged into an n-major LDS tile with a 16-byte-aligned
// 80-byte row stride (20 words, five bank phases); every col_major
// m16n16k32 B fragment then reads its lane's 8 elements as one contiguous
// 8-byte chunk (one ds_read_b64 per fragment). 128x64 uses the same 80-byte
// stride for the A rows (16-byte-aligned staging stores, five bank phases).
// The dispatched 64x128 kernel switched to the iteration-5 fragment-
// interleaved swizzle (see kPackedBGroupBytes below) and no longer uses
// kPackedBStride. The 64x128/64x64 packed kernels keep the 68-byte A stride
// (4 ds_write_b32 per staging vector).
constexpr int kPackedBStride = 80;  // 64 data + 16 pad bytes (20 words)
constexpr int kPackedAStride128x64 = 80;  // 64 data + 16 pad (128x64 A tile)
constexpr int kPackedBK = 6144;  // exact K of the packed assigned shape
constexpr int kPackedBN = 2560;  // exact N of the packed assigned shape

// Iteration 5 (packing round): the exact-shape pack is swizzled from the
// plain n-major transpose to a fragment-interleaved layout so the dispatched
// 64x128 packed-B kernel's B tile is vector-loadable AND LDS-bank-safe end
// to end. For each 32-k x 16-n DUMMA B fragment block the pack stores the 4
// x 8-byte k-groups of the 16 n-rows n-interleaved ([k8][n][8], 128 B per
// k8, 512 B per fragment block, k32-major inside the 64-K stage). Every
// 16-byte staging vector is then 2 n-rows x 8 B of one k8 (int4 global load
// + one ds_write_b128 whose 8-lane LDS cycles each cover all 32 banks once),
// and every col_major B fragment load is one ds_read_b64 whose 16-lane group
// spans exactly one 128-B k8 block (all 32 banks exactly once per cycle ->
// conflict-free; the n-major 80-byte-row tile read the same bytes with 2-way
// conflicts). Same byte count k*n -> graph-stable packed layout.
constexpr int kPackedBGroupBytes = 512;   // 16 n x 4 k8 x 8 B per 32-k fragment block
constexpr int kPackedB32Bytes = 8 * kPackedBGroupBytes;  // 4096 B per 32-k block of the 64-K stage
constexpr int kPackedBN16 = 160;  // 2560 / 16 n-groups per 32-k block (global pack stride)

using namespace du::dumma;

// Direct accumulator epilogue for gfx928 INT8 m16n16k32 (verified against
// du_store_matrix_sync): row = lane & 15, col_mod4 = lane >> 4,
// frag.x[i] maps to columns col_mod4 + 4*i. Stores
// bf16(float(dot) * x_scale[row] * weight_scale[col]); out-of-range rows are
// masked (tail-M handling). The scale multiply order
// float(dot) * x_scale[row] * weight_scale[col] matches the reference
// (dot.float() * a_scale * b_scale.T), so bf16 bits are exact.
template <typename AccFragment>
__device__ __forceinline__ void store_prefill_fragment(
    const AccFragment& frag,
    const float* __restrict__ x_scale,
    const float* __restrict__ weight_scale,
    hip_bfloat16* __restrict__ out,
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
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const int col = base_col + col_mod4 + 4 * i;
    const float scaled = static_cast<float>(frag.x[i]) *
                         x_scale[row] * weight_scale[col];
    out[row * n + col] = __float2bfloat16(scaled);
  }
}

// Coalesced fragment store (packed-B kernels). The m16n16k32 accumulator
// lane mapping (row = lane & 15, column group c4 = lane >> 4, frag.x[i] ->
// column c4 + 4*i) gives each lane four elements strided by 4 columns, so a
// direct store is four 2-byte scalar stores per lane touching every 32-B
// sector at 25% utilization. This epilogue transposes the 4-element groups
// within each 4-lane column group (lanes r, r+16, r+32, r+48; two 2x2
// shuffle steps) so lane (r, c4) owns the four CONTIGUOUS columns
// 4*c4 .. 4*c4+3, converts to bf16, packs 4 bf16 (8 B), and issues ONE
// 8-byte store per lane per fragment (100% store sector efficiency; the
// assigned shape's vmem_write_instructions drop 163,840 -> 40,960). Only the
// int32 values are re-routed between lanes; the per-element float scale
// multiply order (float(dot) * x_scale[row] * weight_scale[col]) and bf16
// rounding are unchanged, so the stored bits are identical. The row >= m
// guard is wavefront-uniform (lane & 15 cycles the same 16 rows in every
// 16-lane group) and base_col is a multiple of 4 (multiple of 32 here), so
// the float4 weight_scale load (col0 % 4 == 0) and the 8-byte store (n even)
// are aligned. This DTK lowers __shfl_xor to ds_bpermute (LDS permute at the
// block tail, where the LDS pipe is otherwise idle); there is no global
// round trip and no staging tile.
template <typename AccFragment>
__device__ __forceinline__ void store_prefill_fragment_coalesced(
    const AccFragment& frag,
    const float* __restrict__ x_scale,
    const float* __restrict__ weight_scale,
    hip_bfloat16* __restrict__ out,
    int m,
    int n,
    int base_row,
    int base_col,
    int lane) {
  const int row = base_row + (lane & 15);
  if (row >= m) {
    return;
  }
  const int c4 = lane >> 4;  // 0..3
  const int x0 = frag.x[0];
  const int x1 = frag.x[1];
  const int x2 = frag.x[2];
  const int x3 = frag.x[3];
  // Step 1: transpose the 2x2 in-block pairs (c4 bit0 <-> element bit0).
  const int s0 = __shfl_xor(x0, 16, kWaveSize);
  const int s1 = __shfl_xor(x1, 16, kWaveSize);
  const int s2 = __shfl_xor(x2, 16, kWaveSize);
  const int s3 = __shfl_xor(x3, 16, kWaveSize);
  const bool lo = (c4 & 1) == 0;
  const int a0 = lo ? x0 : s1;
  const int a1 = lo ? s0 : x1;
  const int a2 = lo ? x2 : s3;
  const int a3 = lo ? s2 : x3;
  // Step 2: transpose the 2x2 blocks (c4 bit1 <-> element bit1).
  const int t0 = __shfl_xor(a0, 32, kWaveSize);
  const int t1 = __shfl_xor(a1, 32, kWaveSize);
  const int t2 = __shfl_xor(a2, 32, kWaveSize);
  const int t3 = __shfl_xor(a3, 32, kWaveSize);
  const bool hi = (c4 & 2) == 0;
  const int f0 = hi ? a0 : t2;
  const int f1 = hi ? a1 : t3;
  const int f2 = hi ? t0 : a2;
  const int f3 = hi ? t1 : a3;

  // Lane (r, c4) now owns columns base_col + 4*c4 .. +3 (8 B, 8-byte
  // aligned: base_col is a multiple of 4, n is even).
  const int col0 = base_col + 4 * c4;
  const float xs = x_scale[row];
  const float4 ws = *reinterpret_cast<const float4*>(weight_scale + col0);
  const float v0 = static_cast<float>(f0) * xs * ws.x;
  const float v1 = static_cast<float>(f1) * xs * ws.y;
  const float v2 = static_cast<float>(f2) * xs * ws.z;
  const float v3 = static_cast<float>(f3) * xs * ws.w;
  const uint64_t packed =
      static_cast<uint64_t>(
          static_cast<unsigned short>(__float2bfloat16(v0))) |
      (static_cast<uint64_t>(
           static_cast<unsigned short>(__float2bfloat16(v1)))
       << 16) |
      (static_cast<uint64_t>(
           static_cast<unsigned short>(__float2bfloat16(v2)))
       << 32) |
      (static_cast<uint64_t>(
           static_cast<unsigned short>(__float2bfloat16(v3)))
       << 48);
  *reinterpret_cast<uint64_t*>(out + row * n + col0) = packed;
}

// Iteration 6 (epilogue round): 4-element 4x4 lane-group transpose for one
// m16n16k32 accumulator fragment. Identical masks (xor 16, xor 32), select
// structure and lane mapping as store_prefill_fragment_coalesced (step 1
// swaps the 2x2 in-block pairs, step 2 swaps the 2x2 blocks), so the four
// returned ints are the SAME per-lane contiguous-column values; only the
// surrounding scheduling is restructured. Extracted so the fused
// store_prefill_rowgroup_coalesced can issue the transposes of a whole
// 16-row x 64-col group as one software-pipelined block.
template <typename AccFragment>
__device__ __forceinline__ void transpose_frag4(
    const AccFragment& frag,
    int lane,
    int& f0,
    int& f1,
    int& f2,
    int& f3) {
  const int c4 = lane >> 4;
  const int x0 = frag.x[0];
  const int x1 = frag.x[1];
  const int x2 = frag.x[2];
  const int x3 = frag.x[3];
  const int s0 = __shfl_xor(x0, 16, kWaveSize);
  const int s1 = __shfl_xor(x1, 16, kWaveSize);
  const int s2 = __shfl_xor(x2, 16, kWaveSize);
  const int s3 = __shfl_xor(x3, 16, kWaveSize);
  const bool lo = (c4 & 1) == 0;
  const int a0 = lo ? x0 : s1;
  const int a1 = lo ? s0 : x1;
  const int a2 = lo ? x2 : s3;
  const int a3 = lo ? s2 : x3;
  const int t0 = __shfl_xor(a0, 32, kWaveSize);
  const int t1 = __shfl_xor(a1, 32, kWaveSize);
  const int t2 = __shfl_xor(a2, 32, kWaveSize);
  const int t3 = __shfl_xor(a3, 32, kWaveSize);
  const bool hi = (c4 & 2) == 0;
  f0 = hi ? a0 : t2;
  f1 = hi ? a1 : t3;
  f2 = hi ? t0 : a2;
  f3 = hi ? t1 : a3;
}

// Scale x 4 contiguous bf16 and pack into one 8-byte store word. The
// per-element multiply order float(dot) * x_scale[row] * weight_scale[col]
// (left to right) and the bf16 rounding are identical to
// store_prefill_fragment_coalesced, so the stored bits are unchanged.
__device__ __forceinline__ uint64_t pack_bf16x4(
    int a0,
    int a1,
    int a2,
    int a3,
    float xs,
    float4 ws) {
  const float v0 = static_cast<float>(a0) * xs * ws.x;
  const float v1 = static_cast<float>(a1) * xs * ws.y;
  const float v2 = static_cast<float>(a2) * xs * ws.z;
  const float v3 = static_cast<float>(a3) * xs * ws.w;
  return static_cast<uint64_t>(
             static_cast<unsigned short>(__float2bfloat16(v0))) |
         (static_cast<uint64_t>(
              static_cast<unsigned short>(__float2bfloat16(v1)))
          << 16) |
         (static_cast<uint64_t>(
              static_cast<unsigned short>(__float2bfloat16(v2)))
          << 32) |
         (static_cast<uint64_t>(
              static_cast<unsigned short>(__float2bfloat16(v3)))
          << 48);
}

// Iteration 6 (epilogue round): fused 4-fragment epilogue for ONE 16-row
// group of a wave (the four fragments cover the group's 64 contiguous
// columns). This is the dispatched 64x128 packed-B kernel's epilogue,
// restructured from eight independent per-fragment calls into two
// row-group calls so the tail becomes one explicitly-scheduled block:
// (1) x_scale[row] and the FOUR float4 weight_scale groups are loaded into
// registers ONCE per lane BEFORE the first ds_bpermute (one L1 round trip
// for the whole group instead of relying on cross-call CSE/load sinking),
// overlapping the scale-load latency with the transpose chain; (2) the four
// independent 4x4 transposes are issued as one software-pipelined block
// (step-1 xor-16 ds_bpermutes of all fragments precede step-2 xor-32, so
// the tail's dependent LDS-latency chain is ~2 round trips, not up to 8
// serialized per-fragment chains); (3) the four 8-byte coalesced stores
// (offsets 0/32/64/96 B from obase) are issued back-to-back at the end,
// keeping 100% store-sector efficiency (same 8-byte stores per lane per
// fragment; vmem_write_instructions unchanged at 40,960). The int32
// transpose routing, per-element float multiply order
// (float(dot) * x_scale[row] * weight_scale[col]) and bf16 rounding are
// byte-identical to the iteration-5 epilogue, so output bits are exact.
template <typename F0, typename F1, typename F2, typename F3>
__device__ __forceinline__ void store_prefill_rowgroup_coalesced(
    const F0& f0,
    const F1& f1,
    const F2& f2,
    const F3& f3,
    const float* __restrict__ x_scale,
    const float* __restrict__ weight_scale,
    hip_bfloat16* __restrict__ out,
    int m,
    int n,
    int base_row,
    int base_col,
    int lane) {
  const int row = base_row + (lane & 15);
  if (row >= m) {
    return;
  }
  const int c4 = lane >> 4;
  const int col0 = base_col + 4 * c4;
  const float xs = x_scale[row];
  const float4 w0 = *reinterpret_cast<const float4*>(weight_scale + col0);
  const float4 w1 = *reinterpret_cast<const float4*>(weight_scale + col0 + 16);
  const float4 w2 = *reinterpret_cast<const float4*>(weight_scale + col0 + 32);
  const float4 w3 = *reinterpret_cast<const float4*>(weight_scale + col0 + 48);
  hip_bfloat16* const obase = out + row * n + col0;
  // Four independent transposes, issued as one pipelined block (the
  // compiler sees all 32 ds_bpermute before the first store).
  int u0, u1, u2, u3, v0, v1, v2, v3, q0, q1, q2, q3, r0, r1, r2, r3;
  transpose_frag4(f0, lane, u0, u1, u2, u3);
  transpose_frag4(f1, lane, v0, v1, v2, v3);
  transpose_frag4(f2, lane, q0, q1, q2, q3);
  transpose_frag4(f3, lane, r0, r1, r2, r3);
  // Four 8-byte coalesced stores, back-to-back (16 rows x 32 B per store
  // instruction; every 32-B sector covered by 4 lanes -> 100% efficiency).
  // Fragment j's slot is 16 columns (32 B) further along the row: obase +
  // 16*j elements.
  *reinterpret_cast<uint64_t*>(obase) = pack_bf16x4(u0, u1, u2, u3, xs, w0);
  *reinterpret_cast<uint64_t*>(obase + 16) =
      pack_bf16x4(v0, v1, v2, v3, xs, w1);
  *reinterpret_cast<uint64_t*>(obase + 32) =
      pack_bf16x4(q0, q1, q2, q3, xs, w2);
  *reinterpret_cast<uint64_t*>(obase + 48) =
      pack_bf16x4(r0, r1, r2, r3, xs, w3);
}

// Explicit 8-byte loader for the col_major m16n16k32 B fragment (accepted
// hy3 lineage, validated on this DTK). The col_major fragment mapping is
// n = lane & 15, k = 8*(lane >> 4) + i; with the n-major LDS tile at row
// stride 80 the lane's 8 elements are contiguous
// (p[row*ldm + col .. +7], 8-byte aligned because ldm=80 and col are
// multiples of 8), so this compiles to ONE ds_read2_b64 instead of 8
// byte-granular ds_read_u8 + mask/OR reassembly. The byte placement is
// identical to du_load_matrix_sync<col_major>, so the v_mmac operand
// registers receive the same values.
__device__ __forceinline__ void load_b_frag8(
    DUFragment<matrix_b, kTileM, kTileN, kTileK, signed char, col_major>& f,
    const int8_t* __restrict__ p,
    int ldm) {
  const unsigned row = static_cast<unsigned>(__lane_id()) & 0xfu;
  const unsigned col = (static_cast<unsigned>(__lane_id()) >> 4) << 3;
  const int64_t v = *reinterpret_cast<const int64_t*>(p + row * ldm + col);
  *reinterpret_cast<int64_t*>(&f.x[0]) = v;
}

// Iteration 5 (packing round): 8-byte loader for the col_major B fragment on
// the swizzled exact-shape tile. The fragment block layout is
// [k8 0..3][n 0..15][8 B] with n = lane&15 and k8 = lane>>4, so lane (n, k8)
// reads p + k8*128 + n*8: one ds_read_b64 per fragment, and each 16-lane
// group of the wavefront spans exactly one 128-B k8 block, so every LDS
// cycle touches all 32 banks exactly once (conflict-free; the n-major
// 80-byte-row tile delivered the same bytes with 2-way conflicts). The byte
// placement is identical to du_load_matrix_sync<col_major> / load_b_frag8,
// so the v_mmac operand registers and the int32 accumulation order are
// unchanged.
__device__ __forceinline__ void load_b_frag8_swz(
    DUFragment<matrix_b, kTileM, kTileN, kTileK, signed char, col_major>& f,
    const int8_t* __restrict__ p) {
  const unsigned n = static_cast<unsigned>(__lane_id()) & 0xfu;
  const unsigned k8 = static_cast<unsigned>(__lane_id()) >> 4;
  const int64_t v =
      *reinterpret_cast<const int64_t*>(p + (k8 << 7) + (n << 3));
  *reinterpret_cast<int64_t*>(&f.x[0]) = v;
}

// Large-M prefill kernel: 64x128 output tile per block, 256 threads
// (4 wavefronts), single-buffered 64-K K stage, identity-packed B. Each wave
// owns a 32x64 quadrant = eight m16n16k32 int8->int32 DUMMA accumulators.
// Serves large-M shapes with N % 128 == 0 except the exact assigned shape
// (which routes to w8a8_dumma_prefill_64x128_packedb_kernel above).
__global__ __launch_bounds__(kThreadsPerBlock) void
w8a8_dumma_prefill_64x128_kernel(
    const int8_t* __restrict__ x_q,
    const int8_t* __restrict__ weight,
    const float* __restrict__ x_scale,
    const float* __restrict__ weight_scale,
    hip_bfloat16* __restrict__ out,
    int m,
    int n,
    int k) {
  const int tid = static_cast<int>(threadIdx.x);
  const int wave = tid / kWaveSize;
  const int lane = tid % kWaveSize;
  const int wave_row = wave >> 1;  // 0 or 1
  const int wave_col = wave & 1;   // 0 or 1
  const int m0 = static_cast<int>(blockIdx.y) * kBlockM128;
  const int n0 = static_cast<int>(blockIdx.x) * kBlockN128;

  __shared__ __align__(16) int8_t a_tile[kBlockM128 * kAStride128];
  __shared__ __align__(16) int8_t b_tile[kStageK128 * kBStride128];

  DUFragment<matrix_a, kTileM, kTileN, kTileK, signed char, row_major>
      a_frag0, a_frag1;
  DUFragment<matrix_b, kTileM, kTileN, kTileK, signed char, row_major>
      b_frag0, b_frag1, b_frag2, b_frag3;
  DUFragment<accumulator, kTileM, kTileN, kTileK, int>
      acc00, acc01, acc02, acc03, acc10, acc11, acc12, acc13;
  du_fill_fragment(acc00, 0);
  du_fill_fragment(acc01, 0);
  du_fill_fragment(acc02, 0);
  du_fill_fragment(acc03, 0);
  du_fill_fragment(acc10, 0);
  du_fill_fragment(acc11, 0);
  du_fill_fragment(acc12, 0);
  du_fill_fragment(acc13, 0);

  const int num_stages = k / kStageK128;

  // Cooperative staging mapping (fixed per thread, reused every stage):
  //   A[64,64]  -> 256 int4 vectors; thread `tid` owns vector `tid`
  //                (row = tid/4, 16-B column group = (tid%4)*16), so each
  //                wavefront covers 16 rows x 64 B contiguous per row.
  //   B[64,128] -> 512 int4 vectors; thread `tid` owns vectors `tid` and
  //                `tid+256` (kk = tid/8, column group = (tid%8)*16), so
  //                each wavefront covers 8 kk rows x 128 B contiguous.
  const int a_row = tid >> 2;
  const int a_col16 = (tid & 3) << 4;
  const int b_kk0 = tid >> 3;
  const int b_col16 = (tid & 7) << 4;

  // VGPR payload for one full 64-K stage (3 x int4 = 12 VGPR), loaded just
  // before it is committed (no one-stage-ahead overlap in this control).
  int4 vA, vB0, vB1;

  // Prologue: load stage 0 into registers, commit it to the single buffer,
  // and make it visible before the first burst.
  {
    const int g_row = m0 + a_row;
    vA = (g_row < m)
             ? *reinterpret_cast<const int4*>(x_q + g_row * k + a_col16)
             : int4{0, 0, 0, 0};
    vB0 = *reinterpret_cast<const int4*>(
        weight + b_kk0 * n + n0 + b_col16);
    vB1 = *reinterpret_cast<const int4*>(
        weight + (b_kk0 + kStageK128 / 2) * n + n0 + b_col16);
    // Commit as int32 stores (skips the 4-byte pad column; 4 x ds_write_b32
    // per 16-byte vector because the odd strides are 4 B mod 16).
    int32_t* adst = reinterpret_cast<int32_t*>(
        a_tile + a_row * kAStride128 + a_col16);
    adst[0] = vA.x;
    adst[1] = vA.y;
    adst[2] = vA.z;
    adst[3] = vA.w;
    int32_t* bdst0 = reinterpret_cast<int32_t*>(
        b_tile + b_kk0 * kBStride128 + b_col16);
    bdst0[0] = vB0.x;
    bdst0[1] = vB0.y;
    bdst0[2] = vB0.z;
    bdst0[3] = vB0.w;
    int32_t* bdst1 = bdst0 + (kStageK128 / 2) * (kBStride128 / 4);
    bdst1[0] = vB1.x;
    bdst1[1] = vB1.y;
    bdst1[2] = vB1.z;
    bdst1[3] = vB1.w;
    __syncthreads();
  }

  for (int s = 0; s < num_stages; ++s) {
    // Consume the single buffer: each wave does eight m16n16k32 MMAs per
    // 32-K step over its 32x64 quadrant.
    const int local_row = wave_row * 32;
    const int local_col = wave_col * 64;
#pragma unroll
    for (int kk = 0; kk < kStageK128; kk += kTileK) {
      du_load_matrix_sync(
          a_frag0, a_tile + local_row * kAStride128 + kk, kAStride128);
      du_load_matrix_sync(
          a_frag1, a_tile + (local_row + kTileM) * kAStride128 + kk,
          kAStride128);
      du_load_matrix_sync(
          b_frag0, b_tile + kk * kBStride128 + local_col, kBStride128);
      du_load_matrix_sync(
          b_frag1, b_tile + kk * kBStride128 + local_col + kTileN,
          kBStride128);
      du_load_matrix_sync(
          b_frag2, b_tile + kk * kBStride128 + local_col + 2 * kTileN,
          kBStride128);
      du_load_matrix_sync(
          b_frag3, b_tile + kk * kBStride128 + local_col + 3 * kTileN,
          kBStride128);
      du_mma_sync(acc00, a_frag0, b_frag0, acc00);
      du_mma_sync(acc01, a_frag0, b_frag1, acc01);
      du_mma_sync(acc02, a_frag0, b_frag2, acc02);
      du_mma_sync(acc03, a_frag0, b_frag3, acc03);
      du_mma_sync(acc10, a_frag1, b_frag0, acc10);
      du_mma_sync(acc11, a_frag1, b_frag1, acc11);
      du_mma_sync(acc12, a_frag1, b_frag2, acc12);
      du_mma_sync(acc13, a_frag1, b_frag3, acc13);
    }
    // All consumption of the single buffer is complete only after every
    // wavefront passes this barrier; the stage s+1 stores below overwrite
    // that same buffer, so they must wait for it.
    __syncthreads();

    // Load stage s+1 and commit it into the single buffer. The global-load
    // latency is fully exposed here: the loads are issued after the burst
    // (nothing to overlap) and the compiler's vmcnt wait before the DS
    // stores is on the critical path. `s + 1 < num_stages` is block-uniform,
    // so the branch and its barrier are free of divergence.
    if (s + 1 < num_stages) {
      const int s1 = (s + 1) * kStageK128;
      const int g_row = m0 + a_row;
      vA = (g_row < m)
               ? *reinterpret_cast<const int4*>(x_q + g_row * k + s1 + a_col16)
               : int4{0, 0, 0, 0};
      vB0 = *reinterpret_cast<const int4*>(
          weight + (s1 + b_kk0) * n + n0 + b_col16);
      vB1 = *reinterpret_cast<const int4*>(
          weight + (s1 + b_kk0 + kStageK128 / 2) * n + n0 + b_col16);
      int32_t* adst = reinterpret_cast<int32_t*>(
          a_tile + a_row * kAStride128 + a_col16);
      adst[0] = vA.x;
      adst[1] = vA.y;
      adst[2] = vA.z;
      adst[3] = vA.w;
      int32_t* bdst0 = reinterpret_cast<int32_t*>(
          b_tile + b_kk0 * kBStride128 + b_col16);
      bdst0[0] = vB0.x;
      bdst0[1] = vB0.y;
      bdst0[2] = vB0.z;
      bdst0[3] = vB0.w;
      int32_t* bdst1 = bdst0 + (kStageK128 / 2) * (kBStride128 / 4);
      bdst1[0] = vB1.x;
      bdst1[1] = vB1.y;
      bdst1[2] = vB1.z;
      bdst1[3] = vB1.w;
      // Make the stage s+1 stores visible to every wavefront before the
      // next burst.
      __syncthreads();
    }
  }

  const int base_row = m0 + wave_row * 32;
  const int base_col = n0 + wave_col * 64;
  store_prefill_fragment(
      acc00, x_scale, weight_scale, out, m, n,
      base_row, base_col, lane);
  store_prefill_fragment(
      acc01, x_scale, weight_scale, out, m, n,
      base_row, base_col + kTileN, lane);
  store_prefill_fragment(
      acc02, x_scale, weight_scale, out, m, n,
      base_row, base_col + 2 * kTileN, lane);
  store_prefill_fragment(
      acc03, x_scale, weight_scale, out, m, n,
      base_row, base_col + 3 * kTileN, lane);
  store_prefill_fragment(
      acc10, x_scale, weight_scale, out, m, n,
      base_row + kTileM, base_col, lane);
  store_prefill_fragment(
      acc11, x_scale, weight_scale, out, m, n,
      base_row + kTileM, base_col + kTileN, lane);
  store_prefill_fragment(
      acc12, x_scale, weight_scale, out, m, n,
      base_row + kTileM, base_col + 2 * kTileN, lane);
  store_prefill_fragment(
      acc13, x_scale, weight_scale, out, m, n,
      base_row + kTileM, base_col + 3 * kTileN, lane);
}

// ---------------------------------------------------------------------------
// Packed-B family for the exact assigned shape (K=6144, N=2560).
// launch_pack_w8a8_weight stores the logical [K, N] weight once, outside the
// timed region, in the iteration-5 fragment-interleaved swizzle (the 64x64
// and 128x64 compiled-but-undispatched family members still assume the
// round-1 n-major packed[n][k] / 80-byte-row tile and are never launched by
// launch_w8a8_gemm). The dispatched 64x128 kernel stages B into the swizzled
// [k32][n16][k8][n][8] tile and consumes col_major B fragments with the
// 8-byte loader load_b_frag8_swz (one conflict-free ds_read_b64 per
// fragment). A stays row-major (68-byte LDS rows). All three kernels share
// the accepted pipeline: single-buffered 64-K K stage with two
// __syncthreads per stage, all twelve fragment loads hoisted before the
// 16-MMAC burst, fused coalesced scale/bf16 epilogue. The int32 accumulation
// order is unchanged from the generic kernels, so results are bit-identical
// to the logical [K, N] layout.
// ---------------------------------------------------------------------------

// 64x128 tile: 4 waves, 32x64 quadrant per wave, eight accumulators. This is
// the exact-shape dispatch (strongest N=2560 prior evidence: the accepted
// hy3 qkv_proj lineage at 113.5 TOPS, K=4096). Grid (20, 64) = 1280 blocks.
// LDS/block = 64x68 + 64x128 = 12,544 B (iteration 5: the swizzled B tile
// needs no padding) -> 3 blocks/CU (VGPR-bound at arch_vgpr <= 85).
__global__ __launch_bounds__(kThreadsPerBlock) void
w8a8_dumma_prefill_64x128_packedb_kernel(
    const int8_t* __restrict__ x_q,
    const int8_t* __restrict__ packed_b,
    const float* __restrict__ x_scale,
    const float* __restrict__ weight_scale,
    hip_bfloat16* __restrict__ out,
    int m,
    int n,
    int k) {
  const int tid = static_cast<int>(threadIdx.x);
  const int wave = tid / kWaveSize;
  const int lane = tid % kWaveSize;
  const int wave_row = wave >> 1;  // 0 or 1
  const int wave_col = wave & 1;   // 0 or 1
  const int m0 = static_cast<int>(blockIdx.y) * kBlockM128;
  const int n0 = static_cast<int>(blockIdx.x) * kBlockN128;

  __shared__ __align__(16) int8_t a_tile[kBlockM128 * kAStride128];
  // Swizzled B tile (iteration 5): the exact-shape pack stores the weight
  // fragment-interleaved [k32 0..1][n16 0..7][k8 0..3][n 0..15][8 B]
  // (64-K stage = 2 x 4096 B), so staging is int4 -> ds_write_b128 and the
  // col_major B fragment loads are one conflict-free ds_read_b64 each.
  __shared__ __align__(16) int8_t b_tile[kStageK128 * kBlockN128];

  DUFragment<matrix_a, kTileM, kTileN, kTileK, signed char, row_major>
      a_frag0, a_frag1;
  DUFragment<matrix_b, kTileM, kTileN, kTileK, signed char, col_major>
      b_frag0, b_frag1, b_frag2, b_frag3;
  DUFragment<accumulator, kTileM, kTileN, kTileK, int>
      acc00, acc01, acc02, acc03, acc10, acc11, acc12, acc13;
  du_fill_fragment(acc00, 0);
  du_fill_fragment(acc01, 0);
  du_fill_fragment(acc02, 0);
  du_fill_fragment(acc03, 0);
  du_fill_fragment(acc10, 0);
  du_fill_fragment(acc11, 0);
  du_fill_fragment(acc12, 0);
  du_fill_fragment(acc13, 0);

  const int num_stages = k / kStageK128;

  // A staging (row-major x_q -> 68-byte LDS rows): thread `tid` owns the
  // 16-byte k-run (a_col16..a_col16+15) of row a_row.
  const int a_row = tid >> 2;
  const int a_col16 = (tid & 3) << 4;
  // B staging on the swizzled pack: thread `tid` owns the 16-byte vectors
  // (k32=0, n16, k8, n_pair) = v = tid and (k32=1, n16, k8, n_pair) = v =
  // tid+256, i.e. bytes
  // ((k32*160 + n0/16 + n16)*512 + k8*128 + n_pair*16) of packed_b, which
  // are the 8-byte k-runs of n-rows 2*n_pair and 2*n_pair+1 in one k8 group.
  const int b_n16 = tid >> 5;      // 0..7 (n16 group inside the 128-n tile)
  const int b_k8 = (tid >> 3) & 3; // 0..3 (8-byte k8 group inside the 32-k block)
  const int b_np = tid & 7;        // 0..7 (n pair inside the 16-n group)
  // Tile-local offset of the 16-byte chunk (k32 block 0, n16 group b_n16,
  // k8 group, n pair) inside b_tile.
  const int b_off =
      b_n16 * kPackedBGroupBytes + (b_k8 << 7) + (b_np << 4);
  // Global packed_b offset of the same 16-byte chunk inside ONE 32-k block
  // (the n16-group term is already carried by the (k32g*kPackedBN16 +
  // n0_16 + b_n16) index, so it must NOT be repeated here).
  const int b_pack_off = (b_k8 << 7) + (b_np << 4);

  // VGPR payload for one full 64-K stage (3 x int4 = 12 VGPR).
  int4 vA, vB0, vB1;

  // Prologue: load stage 0 into registers, commit it to the single buffers,
  // and make it visible before the first burst.
  {
    const int g_row = m0 + a_row;
    vA = (g_row < m)
             ? *reinterpret_cast<const int4*>(x_q + g_row * k + a_col16)
             : int4{0, 0, 0, 0};
    const int n0_16 = n0 >> 4;
    vB0 = *reinterpret_cast<const int4*>(
        packed_b + (n0_16 + b_n16) * kPackedBGroupBytes + b_pack_off);
    vB1 = *reinterpret_cast<const int4*>(
        packed_b +
        (kPackedBN16 + n0_16 + b_n16) * kPackedBGroupBytes + b_pack_off);
    // Commit A as int32 stores (68-byte rows are 4 B mod 16 -> no b128).
    int32_t* adst = reinterpret_cast<int32_t*>(
        a_tile + a_row * kAStride128 + a_col16);
    adst[0] = vA.x;
    adst[1] = vA.y;
    adst[2] = vA.z;
    adst[3] = vA.w;
    // Commit B as 16-byte vector stores (all offsets are 0 mod 16): each
    // 8-lane LDS cycle covers all 32 banks exactly once (conflict-free).
    *reinterpret_cast<int4*>(b_tile + b_off) = vB0;
    *reinterpret_cast<int4*>(b_tile + kPackedB32Bytes + b_off) = vB1;
    __syncthreads();
  }

  for (int s = 0; s < num_stages; ++s) {
    // Consume the single buffer: each wave does eight m16n16k32 MMAs per
    // 32-K step over its 32x64 quadrant. All twelve fragment loads of both
    // 32-K steps are hoisted before the first v_mmac, so the LDS latency
    // concentrates in one wait and the 16-MMAC stream has no lgkmcnt waits.
    const int local_row = wave_row * 32;
    const int8_t* abase = a_tile + local_row * kAStride128;
    // B fragment j covers n16 group wave_col*4 + j of the swizzled tile
    // (512 B per 16-n fragment block); the kk=32 burst lives in k32 block 1.
    const int8_t* bbase = b_tile + (wave_col << 2) * kPackedBGroupBytes;
    du_load_matrix_sync(a_frag0, abase, kAStride128);
    du_load_matrix_sync(
        a_frag1, abase + kTileM * kAStride128, kAStride128);
    load_b_frag8_swz(b_frag0, bbase);
    load_b_frag8_swz(b_frag1, bbase + kPackedBGroupBytes);
    load_b_frag8_swz(b_frag2, bbase + 2 * kPackedBGroupBytes);
    load_b_frag8_swz(b_frag3, bbase + 3 * kPackedBGroupBytes);
    DUFragment<matrix_a, kTileM, kTileN, kTileK, signed char, row_major>
        a2_frag0, a2_frag1;
    DUFragment<matrix_b, kTileM, kTileN, kTileK, signed char, col_major>
        b2_frag0, b2_frag1, b2_frag2, b2_frag3;
    du_load_matrix_sync(a2_frag0, abase + kTileK, kAStride128);
    du_load_matrix_sync(
        a2_frag1, abase + kTileM * kAStride128 + kTileK, kAStride128);
    load_b_frag8_swz(b2_frag0, bbase + kPackedB32Bytes);
    load_b_frag8_swz(
        b2_frag1, bbase + kPackedB32Bytes + kPackedBGroupBytes);
    load_b_frag8_swz(
        b2_frag2, bbase + kPackedB32Bytes + 2 * kPackedBGroupBytes);
    load_b_frag8_swz(
        b2_frag3, bbase + kPackedB32Bytes + 3 * kPackedBGroupBytes);
    // kk = 0 burst (eight m16n16k32 MMAs).
    du_mma_sync(acc00, a_frag0, b_frag0, acc00);
    du_mma_sync(acc01, a_frag0, b_frag1, acc01);
    du_mma_sync(acc02, a_frag0, b_frag2, acc02);
    du_mma_sync(acc03, a_frag0, b_frag3, acc03);
    du_mma_sync(acc10, a_frag1, b_frag0, acc10);
    du_mma_sync(acc11, a_frag1, b_frag1, acc11);
    du_mma_sync(acc12, a_frag1, b_frag2, acc12);
    du_mma_sync(acc13, a_frag1, b_frag3, acc13);
    // kk = 32 burst (same accumulators, same int32 accumulation order).
    du_mma_sync(acc00, a2_frag0, b2_frag0, acc00);
    du_mma_sync(acc01, a2_frag0, b2_frag1, acc01);
    du_mma_sync(acc02, a2_frag0, b2_frag2, acc02);
    du_mma_sync(acc03, a2_frag0, b2_frag3, acc03);
    du_mma_sync(acc10, a2_frag1, b2_frag0, acc10);
    du_mma_sync(acc11, a2_frag1, b2_frag1, acc11);
    du_mma_sync(acc12, a2_frag1, b2_frag2, acc12);
    du_mma_sync(acc13, a2_frag1, b2_frag3, acc13);
    // All consumption of the single buffer is complete only after every
    // wavefront passes this barrier; the stage s+1 stores below overwrite
    // that same buffer.
    __syncthreads();

    // Load stage s+1 and commit it into the single buffer.
    if (s + 1 < num_stages) {
      const int s1 = (s + 1) * kStageK128;
      const int k32g = s1 >> 5;  // global 32-k block of the stage s+1 payload
      const int g_row = m0 + a_row;
      vA = (g_row < m)
               ? *reinterpret_cast<const int4*>(x_q + g_row * k + s1 + a_col16)
               : int4{0, 0, 0, 0};
      const int n0_16 = n0 >> 4;
      vB0 = *reinterpret_cast<const int4*>(
          packed_b +
          (k32g * kPackedBN16 + n0_16 + b_n16) * kPackedBGroupBytes +
          b_pack_off);
      vB1 = *reinterpret_cast<const int4*>(
          packed_b +
          ((k32g + 1) * kPackedBN16 + n0_16 + b_n16) * kPackedBGroupBytes +
          b_pack_off);
      int32_t* adst = reinterpret_cast<int32_t*>(
          a_tile + a_row * kAStride128 + a_col16);
      adst[0] = vA.x;
      adst[1] = vA.y;
      adst[2] = vA.z;
      adst[3] = vA.w;
      *reinterpret_cast<int4*>(b_tile + b_off) = vB0;
      *reinterpret_cast<int4*>(b_tile + kPackedB32Bytes + b_off) = vB1;
      // Make the stage s+1 stores visible to every wavefront before the
      // next burst.
      __syncthreads();
    }
  }

  const int base_row = m0 + wave_row * 32;
  const int base_col = n0 + wave_col * 64;
  // Iteration 6 (epilogue round): fused 4-fragment row-group epilogues.
  // acc00..acc03 cover rows base_row..base_row+15, columns base_col..+63;
  // acc10..acc13 cover the same columns one 16-row group lower. Each call
  // loads x_scale[row] and the four float4 weight_scale groups once per
  // lane before any ds_bpermute, issues the four transposes as one
  // software-pipelined block, then stores back-to-back.
  store_prefill_rowgroup_coalesced(
      acc00, acc01, acc02, acc03, x_scale, weight_scale, out, m, n,
      base_row, base_col, lane);
  store_prefill_rowgroup_coalesced(
      acc10, acc11, acc12, acc13, x_scale, weight_scale, out, m, n,
      base_row + kTileM, base_col, lane);
}

// 64x64 packed-B tile: 4 waves, 32x32 quadrant per wave, four accumulators.
// Compiled family member for the mandated tile sweep (grid 40x64 = 2560
// blocks for the assigned shape; LDS/block = 64x68 + 64x80 = 9,472 B). Not
// dispatched this round; the exact shape uses the 64x128 packed-B kernel.
__global__ __launch_bounds__(kThreadsPerBlock) void
w8a8_dumma_prefill_64x64_packedb_kernel(
    const int8_t* __restrict__ x_q,
    const int8_t* __restrict__ packed_b,
    const float* __restrict__ x_scale,
    const float* __restrict__ weight_scale,
    hip_bfloat16* __restrict__ out,
    int m,
    int n,
    int k) {
  const int tid = static_cast<int>(threadIdx.x);
  const int wave = tid / kWaveSize;
  const int lane = tid % kWaveSize;
  const int wave_row = wave >> 1;  // 0 or 1
  const int wave_col = wave & 1;   // 0 or 1
  const int m0 = static_cast<int>(blockIdx.y) * kBlockM;
  const int n0 = static_cast<int>(blockIdx.x) * kBlockN;

  __shared__ __align__(16) int8_t a_tile[kBlockM * kAStride128];
  __shared__ __align__(16) int8_t b_tile[kBlockN * kPackedBStride];

  DUFragment<matrix_a, kTileM, kTileN, kTileK, signed char, row_major>
      a_frag0, a_frag1;
  DUFragment<matrix_b, kTileM, kTileN, kTileK, signed char, col_major>
      b_frag0, b_frag1;
  DUFragment<accumulator, kTileM, kTileN, kTileK, int>
      acc00, acc01, acc10, acc11;
  du_fill_fragment(acc00, 0);
  du_fill_fragment(acc01, 0);
  du_fill_fragment(acc10, 0);
  du_fill_fragment(acc11, 0);

  const int num_stages = k / kStageK128;

  const int a_row = tid >> 2;
  const int a_col16 = (tid & 3) << 4;
  const int b_n = tid >> 2;
  const int b_k16 = (tid & 3) << 4;

  int4 vA, vB;

  // Prologue: stage 0.
  {
    const int g_row = m0 + a_row;
    vA = (g_row < m)
             ? *reinterpret_cast<const int4*>(x_q + g_row * k + a_col16)
             : int4{0, 0, 0, 0};
    vB = *reinterpret_cast<const int4*>(
        packed_b + (n0 + b_n) * k + b_k16);
    int32_t* adst = reinterpret_cast<int32_t*>(
        a_tile + a_row * kAStride128 + a_col16);
    adst[0] = vA.x;
    adst[1] = vA.y;
    adst[2] = vA.z;
    adst[3] = vA.w;
    *reinterpret_cast<int4*>(b_tile + b_n * kPackedBStride + b_k16) = vB;
    __syncthreads();
  }

  for (int s = 0; s < num_stages; ++s) {
    const int local_row = wave_row * 32;
    const int local_col = wave_col * 32;
    const int8_t* abase = a_tile + local_row * kAStride128;
    const int8_t* bbase = b_tile + local_col * kPackedBStride;
    du_load_matrix_sync(a_frag0, abase, kAStride128);
    du_load_matrix_sync(
        a_frag1, abase + kTileM * kAStride128, kAStride128);
    load_b_frag8(b_frag0, bbase, kPackedBStride);
    load_b_frag8(
        b_frag1, bbase + kTileN * kPackedBStride, kPackedBStride);
    DUFragment<matrix_a, kTileM, kTileN, kTileK, signed char, row_major>
        a2_frag0, a2_frag1;
    DUFragment<matrix_b, kTileM, kTileN, kTileK, signed char, col_major>
        b2_frag0, b2_frag1;
    du_load_matrix_sync(a2_frag0, abase + kTileK, kAStride128);
    du_load_matrix_sync(
        a2_frag1, abase + kTileM * kAStride128 + kTileK, kAStride128);
    load_b_frag8(b2_frag0, bbase + kTileK, kPackedBStride);
    load_b_frag8(
        b2_frag1, bbase + kTileN * kPackedBStride + kTileK, kPackedBStride);
    // kk = 0 burst.
    du_mma_sync(acc00, a_frag0, b_frag0, acc00);
    du_mma_sync(acc01, a_frag0, b_frag1, acc01);
    du_mma_sync(acc10, a_frag1, b_frag0, acc10);
    du_mma_sync(acc11, a_frag1, b_frag1, acc11);
    // kk = 32 burst.
    du_mma_sync(acc00, a2_frag0, b2_frag0, acc00);
    du_mma_sync(acc01, a2_frag0, b2_frag1, acc01);
    du_mma_sync(acc10, a2_frag1, b2_frag0, acc10);
    du_mma_sync(acc11, a2_frag1, b2_frag1, acc11);
    __syncthreads();

    if (s + 1 < num_stages) {
      const int s1 = (s + 1) * kStageK128;
      const int g_row = m0 + a_row;
      vA = (g_row < m)
               ? *reinterpret_cast<const int4*>(x_q + g_row * k + s1 + a_col16)
               : int4{0, 0, 0, 0};
      vB = *reinterpret_cast<const int4*>(
          packed_b + (n0 + b_n) * k + s1 + b_k16);
      int32_t* adst = reinterpret_cast<int32_t*>(
          a_tile + a_row * kAStride128 + a_col16);
      adst[0] = vA.x;
      adst[1] = vA.y;
      adst[2] = vA.z;
      adst[3] = vA.w;
      *reinterpret_cast<int4*>(b_tile + b_n * kPackedBStride + b_k16) = vB;
      __syncthreads();
    }
  }

  const int base_row = m0 + wave_row * 32;
  const int base_col = n0 + wave_col * 32;
  store_prefill_fragment_coalesced(
      acc00, x_scale, weight_scale, out, m, n,
      base_row, base_col, lane);
  store_prefill_fragment_coalesced(
      acc01, x_scale, weight_scale, out, m, n,
      base_row, base_col + kTileN, lane);
  store_prefill_fragment_coalesced(
      acc10, x_scale, weight_scale, out, m, n,
      base_row + kTileM, base_col, lane);
  store_prefill_fragment_coalesced(
      acc11, x_scale, weight_scale, out, m, n,
      base_row + kTileM, base_col + kTileN, lane);
}

// 128x64 packed-B tile: 4 waves, 64x32 quadrant per wave, eight
// accumulators (the o_proj lineage geometry). Compiled family member for the
// mandated tile sweep (grid 40x32 = 1280 blocks for the assigned shape;
// LDS/block = 128x80 + 64x80 = 15,360 B). Not dispatched this round.
__global__ __launch_bounds__(kThreadsPerBlock) void
w8a8_dumma_prefill_128x64_packedb_kernel(
    const int8_t* __restrict__ x_q,
    const int8_t* __restrict__ packed_b,
    const float* __restrict__ x_scale,
    const float* __restrict__ weight_scale,
    hip_bfloat16* __restrict__ out,
    int m,
    int n,
    int k) {
  const int tid = static_cast<int>(threadIdx.x);
  const int wave = tid / kWaveSize;
  const int lane = tid % kWaveSize;
  const int wave_row = wave >> 1;  // 0 or 1
  const int wave_col = wave & 1;   // 0 or 1
  const int m0 = static_cast<int>(blockIdx.y) * kBlockM128 * 2;
  const int n0 = static_cast<int>(blockIdx.x) * kBlockN;

  __shared__ __align__(16) int8_t a_tile[kBlockM128 * 2 * kPackedAStride128x64];
  __shared__ __align__(16) int8_t b_tile[kBlockN * kPackedBStride];

  DUFragment<matrix_a, kTileM, kTileN, kTileK, signed char, row_major>
      a_frag0, a_frag1, a_frag2, a_frag3;
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

  const int num_stages = k / kStageK128;

  // A staging: thread owns the 16-byte k-run (a_col16..+15) of A rows
  // a_row and a_row + 64 (A[128,64] = 512 int4 = 2 per thread).
  const int a_row = tid >> 2;
  const int a_col16 = (tid & 3) << 4;
  // B staging: thread owns the 16-byte k-run (b_k16..+15) of n-row b_n.
  const int b_n = tid >> 2;
  const int b_k16 = (tid & 3) << 4;

  int4 vA0, vA1, vB;

  // Prologue: stage 0.
  {
    const int g_row = m0 + a_row;
    vA0 = (g_row < m)
              ? *reinterpret_cast<const int4*>(x_q + g_row * k + a_col16)
              : int4{0, 0, 0, 0};
    vA1 = (g_row + kBlockM128 < m)
              ? *reinterpret_cast<const int4*>(
                    x_q + (g_row + kBlockM128) * k + a_col16)
              : int4{0, 0, 0, 0};
    vB = *reinterpret_cast<const int4*>(
        packed_b + (n0 + b_n) * k + b_k16);
    // 80-byte A rows are 0 mod 16: one b128 store per vector.
    *reinterpret_cast<int4*>(
        a_tile + a_row * kPackedAStride128x64 + a_col16) = vA0;
    *reinterpret_cast<int4*>(
        a_tile + (a_row + kBlockM128) * kPackedAStride128x64 + a_col16) = vA1;
    *reinterpret_cast<int4*>(b_tile + b_n * kPackedBStride + b_k16) = vB;
    __syncthreads();
  }

  for (int s = 0; s < num_stages; ++s) {
    // Each wave consumes its 64x32 quadrant: eight m16n16k32 MMAs per 32-K
    // step (4 A fragments x 2 B fragments), all fragment loads hoisted.
    const int local_row = wave_row * 64;
    const int local_col = wave_col * 32;
    const int8_t* abase = a_tile + local_row * kPackedAStride128x64;
    const int8_t* bbase = b_tile + local_col * kPackedBStride;
    du_load_matrix_sync(a_frag0, abase, kPackedAStride128x64);
    du_load_matrix_sync(
        a_frag1, abase + kTileM * kPackedAStride128x64,
        kPackedAStride128x64);
    du_load_matrix_sync(
        a_frag2, abase + 2 * kTileM * kPackedAStride128x64,
        kPackedAStride128x64);
    du_load_matrix_sync(
        a_frag3, abase + 3 * kTileM * kPackedAStride128x64,
        kPackedAStride128x64);
    load_b_frag8(b_frag0, bbase, kPackedBStride);
    load_b_frag8(
        b_frag1, bbase + kTileN * kPackedBStride, kPackedBStride);
    DUFragment<matrix_a, kTileM, kTileN, kTileK, signed char, row_major>
        a2_frag0, a2_frag1, a2_frag2, a2_frag3;
    DUFragment<matrix_b, kTileM, kTileN, kTileK, signed char, col_major>
        b2_frag0, b2_frag1;
    du_load_matrix_sync(a2_frag0, abase + kTileK, kPackedAStride128x64);
    du_load_matrix_sync(
        a2_frag1, abase + kTileM * kPackedAStride128x64 + kTileK,
        kPackedAStride128x64);
    du_load_matrix_sync(
        a2_frag2, abase + 2 * kTileM * kPackedAStride128x64 + kTileK,
        kPackedAStride128x64);
    du_load_matrix_sync(
        a2_frag3, abase + 3 * kTileM * kPackedAStride128x64 + kTileK,
        kPackedAStride128x64);
    load_b_frag8(b2_frag0, bbase + kTileK, kPackedBStride);
    load_b_frag8(
        b2_frag1, bbase + kTileN * kPackedBStride + kTileK, kPackedBStride);
    // kk = 0 burst.
    du_mma_sync(acc00, a_frag0, b_frag0, acc00);
    du_mma_sync(acc01, a_frag0, b_frag1, acc01);
    du_mma_sync(acc10, a_frag1, b_frag0, acc10);
    du_mma_sync(acc11, a_frag1, b_frag1, acc11);
    du_mma_sync(acc20, a_frag2, b_frag0, acc20);
    du_mma_sync(acc21, a_frag2, b_frag1, acc21);
    du_mma_sync(acc30, a_frag3, b_frag0, acc30);
    du_mma_sync(acc31, a_frag3, b_frag1, acc31);
    // kk = 32 burst.
    du_mma_sync(acc00, a2_frag0, b2_frag0, acc00);
    du_mma_sync(acc01, a2_frag0, b2_frag1, acc01);
    du_mma_sync(acc10, a2_frag1, b2_frag0, acc10);
    du_mma_sync(acc11, a2_frag1, b2_frag1, acc11);
    du_mma_sync(acc20, a2_frag2, b2_frag0, acc20);
    du_mma_sync(acc21, a2_frag2, b2_frag1, acc21);
    du_mma_sync(acc30, a2_frag3, b2_frag0, acc30);
    du_mma_sync(acc31, a2_frag3, b2_frag1, acc31);
    __syncthreads();

    if (s + 1 < num_stages) {
      const int s1 = (s + 1) * kStageK128;
      const int g_row = m0 + a_row;
      vA0 = (g_row < m)
                ? *reinterpret_cast<const int4*>(
                      x_q + g_row * k + s1 + a_col16)
                : int4{0, 0, 0, 0};
      vA1 = (g_row + kBlockM128 < m)
                ? *reinterpret_cast<const int4*>(
                      x_q + (g_row + kBlockM128) * k + s1 + a_col16)
                : int4{0, 0, 0, 0};
      vB = *reinterpret_cast<const int4*>(
          packed_b + (n0 + b_n) * k + s1 + b_k16);
      *reinterpret_cast<int4*>(
          a_tile + a_row * kPackedAStride128x64 + a_col16) = vA0;
      *reinterpret_cast<int4*>(
          a_tile + (a_row + kBlockM128) * kPackedAStride128x64 + a_col16) =
          vA1;
      *reinterpret_cast<int4*>(b_tile + b_n * kPackedBStride + b_k16) = vB;
      __syncthreads();
    }
  }

  const int base_row = m0 + wave_row * 64;
  const int base_col = n0 + wave_col * 32;
  store_prefill_fragment_coalesced(
      acc00, x_scale, weight_scale, out, m, n,
      base_row, base_col, lane);
  store_prefill_fragment_coalesced(
      acc01, x_scale, weight_scale, out, m, n,
      base_row, base_col + kTileN, lane);
  store_prefill_fragment_coalesced(
      acc10, x_scale, weight_scale, out, m, n,
      base_row + kTileM, base_col, lane);
  store_prefill_fragment_coalesced(
      acc11, x_scale, weight_scale, out, m, n,
      base_row + kTileM, base_col + kTileN, lane);
  store_prefill_fragment_coalesced(
      acc20, x_scale, weight_scale, out, m, n,
      base_row + 2 * kTileM, base_col, lane);
  store_prefill_fragment_coalesced(
      acc21, x_scale, weight_scale, out, m, n,
      base_row + 2 * kTileM, base_col + kTileN, lane);
  store_prefill_fragment_coalesced(
      acc30, x_scale, weight_scale, out, m, n,
      base_row + 3 * kTileM, base_col, lane);
  store_prefill_fragment_coalesced(
      acc31, x_scale, weight_scale, out, m, n,
      base_row + 3 * kTileM, base_col + kTileN, lane);
}

// Generic large-M prefill kernel: 64x64 output tile per block, K staged in
// LDS (single buffer), four waves each owning a 32x32 quadrant = four
// m16n16k32 int8->int32 DUMMA accumulators. Covers large-M shapes whose N is
// a multiple of 64 but not 128 (and K a multiple of 128).
__global__ __launch_bounds__(kThreadsPerBlock) void
w8a8_dumma_prefill_64x64_kernel(
    const int8_t* __restrict__ x_q,
    const int8_t* __restrict__ weight,
    const float* __restrict__ x_scale,
    const float* __restrict__ weight_scale,
    hip_bfloat16* __restrict__ out,
    int m,
    int n,
    int k) {
  const int tid = static_cast<int>(threadIdx.x);
  const int wave = tid / kWaveSize;
  const int lane = tid % kWaveSize;
  const int wave_row = wave >> 1;  // 0 or 1
  const int wave_col = wave & 1;   // 0 or 1
  const int m0 = static_cast<int>(blockIdx.y) * kBlockM;
  const int n0 = static_cast<int>(blockIdx.x) * kBlockN;

  __shared__ __align__(16) int8_t a_tile[kBlockM * kStageK];
  __shared__ __align__(16) int8_t b_tile[kStageK * kBlockN];

  DUFragment<matrix_a, kTileM, kTileN, kTileK, signed char, row_major>
      a_frag0, a_frag1;
  DUFragment<matrix_b, kTileM, kTileN, kTileK, signed char, row_major>
      b_frag0, b_frag1;
  DUFragment<accumulator, kTileM, kTileN, kTileK, int>
      acc00, acc01, acc10, acc11;
  du_fill_fragment(acc00, 0);
  du_fill_fragment(acc01, 0);
  du_fill_fragment(acc10, 0);
  du_fill_fragment(acc11, 0);

  for (int k0 = 0; k0 < k; k0 += kStageK) {
    // Stage A[64,128] (row-major, zero-filled past M).
    constexpr int kAVectors = kBlockM * kStageK / static_cast<int>(sizeof(int4));
    for (int vec = tid; vec < kAVectors; vec += kThreadsPerBlock) {
      const int byte_offset = vec * static_cast<int>(sizeof(int4));
      const int local_row = byte_offset / kStageK;
      const int kk = byte_offset - local_row * kStageK;
      const int global_row = m0 + local_row;
      reinterpret_cast<int4*>(a_tile)[vec] =
          global_row < m
              ? *reinterpret_cast<const int4*>(
                    x_q + global_row * k + k0 + kk)
              : int4{0, 0, 0, 0};
    }

    // Stage B[128,64] (row-major [K, N] logical weight layout).
    constexpr int kBVectors = kStageK * kBlockN / static_cast<int>(sizeof(int4));
    for (int vec = tid; vec < kBVectors; vec += kThreadsPerBlock) {
      const int byte_offset = vec * static_cast<int>(sizeof(int4));
      const int kk = byte_offset / kBlockN;
      const int col = byte_offset - kk * kBlockN;
      reinterpret_cast<int4*>(b_tile)[vec] =
          *reinterpret_cast<const int4*>(
              weight + (k0 + kk) * n + n0 + col);
    }
    __syncthreads();

    // Consume the stage: each wave does four m16n16k32 MMAs per 32-K step.
    const int local_row = wave_row * 32;
    const int local_col = wave_col * 32;
#pragma unroll
    for (int kk = 0; kk < kStageK; kk += kTileK) {
      du_load_matrix_sync(
          a_frag0, a_tile + local_row * kStageK + kk, kStageK);
      du_load_matrix_sync(
          a_frag1, a_tile + (local_row + kTileM) * kStageK + kk, kStageK);
      du_load_matrix_sync(
          b_frag0, b_tile + kk * kBlockN + local_col, kBlockN);
      du_load_matrix_sync(
          b_frag1, b_tile + kk * kBlockN + local_col + kTileN, kBlockN);
      du_mma_sync(acc00, a_frag0, b_frag0, acc00);
      du_mma_sync(acc01, a_frag0, b_frag1, acc01);
      du_mma_sync(acc10, a_frag1, b_frag0, acc10);
      du_mma_sync(acc11, a_frag1, b_frag1, acc11);
    }
    // Single buffer: make sure every wave finished reading LDS before the
    // next stage overwrites it.
    __syncthreads();
  }

  const int base_row = m0 + wave_row * 32;
  const int base_col = n0 + wave_col * 32;
  store_prefill_fragment(
      acc00, x_scale, weight_scale, out, m, n,
      base_row, base_col, lane);
  store_prefill_fragment(
      acc01, x_scale, weight_scale, out, m, n,
      base_row, base_col + kTileN, lane);
  store_prefill_fragment(
      acc10, x_scale, weight_scale, out, m, n,
      base_row + kTileM, base_col, lane);
  store_prefill_fragment(
      acc11, x_scale, weight_scale, out, m, n,
      base_row + kTileM, base_col + kTileN, lane);
}

// Generic scalar int8/int32 fallback for unmatched shapes and small-M API
// cases. One output element per grid-stride step; exact int32 accumulation
// (K <= 8192 keeps the int8 dot well within int32 range: 8192*128*128 =
// 134,217,728 < 2^31), then the fused float scale and bf16 store. The exact
// assigned shape (K=6144, N=2560) uses the iteration-5 swizzled pack (set up
// once by launch_pack_w8a8_weight; decoded per element below); every other
// (k, n) keeps the logical [K, N] row-major layout. The branch is
// grid-uniform per launch, so the paired M=2 API shape with the same (N, K)
// stays correct.
__global__ __launch_bounds__(256) void w8a8_scalar_gemm_kernel(
    const int8_t* __restrict__ a,
    const int8_t* __restrict__ b,
    const float* __restrict__ x_scale,
    const float* __restrict__ weight_scale,
    hip_bfloat16* __restrict__ out,
    int m,
    int n,
    int k) {
  const int64_t total = static_cast<int64_t>(m) * n;
  const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
  const bool packed_b = (k == kPackedBK && n == kPackedBN);
  for (int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x +
                     threadIdx.x;
       idx < total;
       idx += stride) {
    const int row = static_cast<int>(idx / n);
    const int col = static_cast<int>(idx - static_cast<int64_t>(row) * n);
    const int8_t* a_row = a + static_cast<int64_t>(row) * k;
    int32_t acc = 0;
    for (int kk = 0; kk < k; ++kk) {
      int32_t bw;
      if (packed_b) {
        // Iteration-5 swizzled exact-shape layout: element (n=col, k=kk) at
        // ((kk>>5)*160 + (col>>4))*512 + (((kk>>3)&3)*16 + (col&15))*8 +
        // (kk&7) (fragment-interleaved [k32][n16][k8][n][8]).
        const int64_t off =
            ((static_cast<int64_t>(kk >> 5) * kPackedBN16 + (col >> 4)) *
                 kPackedBGroupBytes) +
            ((((kk >> 3) & 3) * 16 + (col & 15)) << 3) + (kk & 7);
        bw = static_cast<int32_t>(b[off]);
      } else {
        bw = static_cast<int32_t>(
            b[static_cast<int64_t>(kk) * n + col]);
      }
      acc += static_cast<int32_t>(a_row[kk]) * bw;
    }
    const float scaled = static_cast<float>(acc) *
                         x_scale[row] * weight_scale[col];
    out[idx] = __float2bfloat16(scaled);
  }
}

// Identity device-to-device byte copy (pack_weight fallback, valid for any
// (K, N)).
__global__ __launch_bounds__(256) void w8a8_pack_identity_i8_kernel(
    const int8_t* __restrict__ src,
    int8_t* __restrict__ dst,
    int64_t count) {
  const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x +
                   threadIdx.x;
       i < count;
       i += stride) {
    dst[i] = src[i];
  }
}

// One-time swizzled pack of the logical [K, N] weight for the exact assigned
// shape (K=6144, N=2560), iteration 5 (packing round). Runs only from
// launch_pack_w8a8_weight, outside the timed region and outside Graph
// capture. The packed buffer keeps the same byte count (k*n) as the identity
// pack, so allocations and graph-stable addresses are unchanged. Layout:
// element (n, k) lands at
// ((k>>5)*n16g + (n>>4))*512 + (((k>>3)&3)*16 + (n&15))*8 + (k&7)
// with n16g = n>>4 (160 for the exact shape): for each 32-k x 16-n DUMMA B
// fragment block the 4 k8 groups x 16 n-rows x 8 B are stored n-interleaved
// (k8-major within the block). Each thread writes one contiguous 16-byte
// chunk (two n-rows of one k8) with a strided 2x8 gather on the read side
// (packing is outside timing, so the scalar gather is off the critical
// path). The guard guarantees n % 16 == 0.
__global__ __launch_bounds__(256) void w8a8_pack_b_swz_kernel(
    const int8_t* __restrict__ src,
    int8_t* __restrict__ dst,
    int k,
    int n) {
  const int64_t chunks = (static_cast<int64_t>(k) * n) >> 4;
  const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
  const int n16g = n >> 4;
  for (int64_t c = static_cast<int64_t>(blockIdx.x) * blockDim.x +
                   threadIdx.x;
       c < chunks;
       c += stride) {
    const int k32 = static_cast<int>(c / (static_cast<int64_t>(n16g) * 32));
    const int rem =
        static_cast<int>(c % (static_cast<int64_t>(n16g) * 32));
    const int n16 = rem >> 5;
    const int w = rem & 31;
    const int k8 = w >> 3;
    const int np = w & 7;
    const int n0 = (n16 << 4) + (np << 1);
    const int k0 = (k32 << 5) + (k8 << 3);
    int8_t* dst16 = dst + (c << 4);
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      // 16-byte chunk = [n0 row: k0..k0+7][n0+1 row: k0..k0+7] (n-major
      // within the k8 group, matching the [k8][n][8] tile layout).
      dst16[i] = src[static_cast<int64_t>(k0 + i) * n + n0];
      dst16[8 + i] = src[static_cast<int64_t>(k0 + i) * n + n0 + 1];
    }
  }
}

// Identity device-to-device float copy for the weight scales.
__global__ __launch_bounds__(256) void w8a8_pack_identity_f32_kernel(
    const float* __restrict__ src,
    float* __restrict__ dst,
    int64_t count) {
  const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x +
                   threadIdx.x;
       i < count;
       i += stride) {
    dst[i] = src[i];
  }
}

}  // namespace

// Stable host launch symbol consumed by csrc/bindings.cpp. Dispatches on the
// caller-provided stream only; never allocates, synchronizes, or touches the
// default stream.
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
  auto* out_ptr = static_cast<hip_bfloat16*>(out);

  // Packed-B family: the exact assigned shape (K=6144, N=2560) routes to
  // the swizzled-packed-B 64x128 kernel (iteration 5 layout). The guard is
  // exact (m >= 128, n == 2560, k == 6144) and sits BEFORE the generic
  // 64x128 path; every other shape keeps its existing path (generic 64x128
  // for n % 128 == 0, 64x64 for n % 64 == 0, scalar fallback otherwise) and
  // the identity-packed layout. Grid (N/128)x(M/64) = 20x64 = 1280 blocks.
  if (m >= 128 && n == kPackedBN && k == kPackedBK) {
    const dim3 grid(
        static_cast<unsigned>(n / kBlockN128),
        static_cast<unsigned>((m + kBlockM128 - 1) / kBlockM128));
    const dim3 block(static_cast<unsigned>(kThreadsPerBlock));
    hipLaunchKernelGGL(
        w8a8_dumma_prefill_64x128_packedb_kernel,
        grid,
        block,
        0,
        stream,
        a,
        b,
        x_scale,
        weight_scale,
        out_ptr,
        m,
        n,
        k);
    return;
  }

  // Native INT8 DUMMA m16n16k32 prefill path (2-D macro-tile 64x128,
  // single-buffered 64-K K stage, identity-packed B) for every other
  // large-M shape with compatible geometry.
  if (m >= 128 && (n % kBlockN128) == 0 && (k % 128) == 0) {
    const dim3 grid(
        static_cast<unsigned>(n / kBlockN128),
        static_cast<unsigned>((m + kBlockM128 - 1) / kBlockM128));
    const dim3 block(static_cast<unsigned>(kThreadsPerBlock));
    hipLaunchKernelGGL(
        w8a8_dumma_prefill_64x128_kernel,
        grid,
        block,
        0,
        stream,
        a,
        b,
        x_scale,
        weight_scale,
        out_ptr,
        m,
        n,
        k);
    return;
  }

  // Native INT8 DUMMA m16n16k32 prefill path (64x64 tile) for large-M shapes
  // whose N is not a multiple of 128 but is a multiple of 64.
  if (m >= 128 && (n % kBlockN) == 0 && (k % kStageK) == 0) {
    const dim3 grid(
        static_cast<unsigned>(n / kBlockN),
        static_cast<unsigned>((m + kBlockM - 1) / kBlockM));
    const dim3 block(static_cast<unsigned>(kThreadsPerBlock));
    hipLaunchKernelGGL(
        w8a8_dumma_prefill_64x64_kernel,
        grid,
        block,
        0,
        stream,
        a,
        b,
        x_scale,
        weight_scale,
        out_ptr,
        m,
        n,
        k);
    return;
  }

  // Scalar int8/int32 fallback for unmatched (m, n, k) and small-M API cases
  // (M < 128, including the paired M=2 API shape with the same (N,K)).
  const int64_t total = static_cast<int64_t>(m) * n;
  constexpr int kFallbackThreads = 256;
  int64_t blocks = (total + kFallbackThreads - 1) / kFallbackThreads;
  if (blocks > 4096) {
    blocks = 4096;
  }
  if (blocks < 1) {
    blocks = 1;
  }
  const dim3 grid(static_cast<unsigned>(blocks));
  const dim3 block(kFallbackThreads);
  hipLaunchKernelGGL(
      w8a8_scalar_gemm_kernel,
      grid,
      block,
      0,
      stream,
      a,
      b,
      x_scale,
      weight_scale,
      out_ptr,
      m,
      n,
      k);
}

// Stable host launch symbol consumed by csrc/bindings.cpp. For the exact
// assigned shape (K=6144, N=2560) it performs the one-time swizzled B pack
// (iteration 5: logical [K, N] -> fragment-interleaved
// [k32][n16][k8][n][8]) outside the timed region and outside Graph capture;
// every other (K, N) keeps the identity device-to-device copy, valid for
// every (K, N). The packed buffer size is k*n in both cases, so the
// allocation and graph-stable packed layout are unchanged.
extern "C" void launch_pack_w8a8_weight(
    const int8_t* raw_weight,
    const float* weight_scale,
    int8_t* packed_weight,
    float* packed_weight_scale,
    int k,
    int n,
    hipStream_t stream) {
  constexpr int kPackThreads = 256;
  const int64_t weight_count = static_cast<int64_t>(k) * n;
  if (weight_count > 0) {
    if (k == kPackedBK && n == kPackedBN) {
      const int64_t chunks = weight_count >> 4;
      int64_t blocks = (chunks + kPackThreads - 1) / kPackThreads;
      if (blocks > 4096) {
        blocks = 4096;
      }
      hipLaunchKernelGGL(
          w8a8_pack_b_swz_kernel,
          dim3(static_cast<unsigned>(blocks)),
          dim3(kPackThreads),
          0,
          stream,
          raw_weight,
          packed_weight,
          k,
          n);
    } else {
      int64_t blocks = (weight_count + kPackThreads - 1) / kPackThreads;
      if (blocks > 4096) {
        blocks = 4096;
      }
      hipLaunchKernelGGL(
          w8a8_pack_identity_i8_kernel,
          dim3(static_cast<unsigned>(blocks)),
          dim3(kPackThreads),
          0,
          stream,
          raw_weight,
          packed_weight,
          weight_count);
    }
  }
  if (n > 0) {
    int64_t blocks = (static_cast<int64_t>(n) + kPackThreads - 1) /
                     kPackThreads;
    if (blocks > 4096) {
      blocks = 4096;
    }
    hipLaunchKernelGGL(
        w8a8_pack_identity_f32_kernel,
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
// @@end
