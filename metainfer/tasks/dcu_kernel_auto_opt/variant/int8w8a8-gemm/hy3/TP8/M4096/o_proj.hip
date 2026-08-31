// @@variant shape=hy3_tp8_o_proj_m4096 commit=710edb90ef1f462cd0751356a542a14e8bbba8c6 added=2026-08-27
//   median_us=323 p90_us=325.4 speedup=43.82 baseline_us=1.415e+04
//   source=hy3-dsh-tp8-m4096-1-0ebb994d
// @@variant shape=hy3_tp8_o_proj_m4096 added=2026-08-27 (bootstrap iteration 1)
// INT8 W8A8 GEMM HIP implementation for Hygon K500SM_AI / gfx928.
//
// Worker: worker_1 (physical GPU 1), assigned shape:
//   hy3_tp8_o_proj_m4096 : M=4096, N=4096, K=1024
//
// Bootstrap strategy (iteration 1, correctness-first but a usable profiling
// baseline - a large-Prefill scalar K loop would not be):
//   * Large-prefill path (exact (m, n, k) == (4096, 4096, 1024)):
//       native INT8 DUMMA m16n16k32 with int32 accumulation; one 128x64
//       output tile per block; four wavefronts (256 threads); each wave owns
//       a 64x32 quadrant built from eight m16n16k32 accumulator fragments;
//       the block cooperatively vector-loads A[128,64] and B[64,64] into one
//       single-buffered 64-K LDS stage (15,360 B total: 128*80 + 64*80 with
//       16 B padding per row for bank skew); two __syncthreads per stage;
//       fused dot * x_scale[m] * weight_scale[n] epilogue stored directly as
//       bf16 from the accumulator fragments.  Grid dim3(64, 32) = 2048 blocks
//       dwarfs the 120 CUs, so no split-K is needed.
//       The weight is read in the packed n-major [N, K] int8 layout
//       (packed[n*K + kk] == raw[kk*N + n], a full transpose; identity
//       [K, N] layout is retained for every other (k, n)).  Every B tile of
//       one (n-tile, K-stage) is one contiguous 4 KiB vector stream in both
//       global and LDS, so each thread's B staging load/store is a single
//       16-byte-aligned int4.  launch_pack_w8a8_weight performs the
//       permutation once, outside the timed region and outside Graph
//       capture, into the same byte-count buffer (graph-stable).
//   * Generic scalar int8/int32 fallback for every unmatched (m, n, k),
//       including all small-M API cases (M=2, M=16) and M=3072 with the same
//       (K, N) = (1024, 4096); the fallback decodes the packed n-major [N, K]
//       layout for (k, n) == (1024, 4096) and the identity [K, N] layout for
//       every other (k, n).
//   * launch_pack_w8a8_weight: for (k, n) == (1024, 4096) transposes the raw
//       [K, N] weight into the [N, K] int8 layout; for any other (k, n) it
//       is an identity device-to-device copy. Scales are copied identity.
//       Packing never happens inside the timed GEMM.
//
// Iteration 6 (epilogue round): the scales and bf16 conversion were already
// fused in the compute kernel and the launcher already ignores the workspace
// (no split-K combine pass exists), so the remaining epilogue inefficiency
// is the store pattern: the m16n16k32 lane mapping (row = lane & 15,
// c4 = lane >> 4, frag.x[i] -> column c4 + 4*i) makes the direct store four
// 2-byte scalar stores per lane whose wavefront addresses touch each 32-B
// sector at 25% utilization (32 global_store_short_d16_hi per wavefront per
// block in the gfx928 ISA; vmem_write_instructions 262,144). The new
// store_prefill_fragment_coalesced transposes the 4-element groups within
// each 4-lane column group (two __shfl_xor steps, 16 then 32; this DTK
// lowers them to ds_bpermute at the block tail where the LDS pipe is idle),
// so lane (r, c4) holds the four CONTIGUOUS columns 4*c4..4*c4+3 and issues
// ONE 8-byte store per lane: 64 lanes x 8 B = 512 B per fragment per
// wavefront in 16 fully-used 32-B sectors (100% sector efficiency;
// vmem_write_instructions -> 65,536). The MMAC loop, LDS staging, barriers
// and int32 accumulation order are untouched, and the per-element scale
// multiply order plus bf16 rounding are unchanged => stored bf16 bits are
// identical (mismatch 0, max_abs_error 0.0 expected). This is the exact
// mechanism validated on the worker-29 TP4 qkv_proj lineage (accepted
// iteration 6, 770.14 -> 721.87 us).
//
// Iteration 10 (B-fragment load round): the gfx928 code object of the
// accepted kernel (419.05 us median, 81.995 TOPS, arch_vgpr 80, LDS 15,360 B,
// grid 2048, workgroup 256) shows the DUMMA B-fragment path compiled as ~32
// per-wave ds_read_u8 byte loads (the row_major int8 loader assigns
// x[i] = p[col*ldm + row + i], i.e. 8 k-rows strided by ldm per lane) plus a
// mask/OR byte-reassembly VALU chain (~6 ops per operand dword) with long
// lgkmcnt wait chains; PMC: lds_bank_conflicts 15,728,640 = 3 x (4,194,304 B
// ds_read_u8 + 1,048,576 A ds_read2_b32) and lds_wait_instructions
// 11,718,821. This round stages B n-major in LDS (b_tile[n][k], 64 rows x 80
// B - same 5,120 B) and reads each B fragment as ONE 8-byte LDS load written
// directly into the fragment storage (the same 8 bytes in the same x[0..7]
// order; validated load_fragment8 pattern from the worker-29 TP4 gate_up
// lineage iteration 13, VALU 16.08M -> 6.25M), so the per-byte loads, the
// pack VALU and their waits disappear; du_mma_sync casts b.x straight to the
// v_mmac operand, so the operand bit pattern and the int32 accumulation are
// bit-identical (mismatch 0, max_abs_error 0.0 expected). The global pack for
// (k, n) == (1024, 4096) becomes n-major [N, K] so staging stays one
// coalesced int4 per thread (same trade as the worker-29 TP4 down_proj
// iteration 14), and the scalar fallback decodes the same n-major layout.
// Tile, A staging/A fragments, barriers, staging bytes, per-accumulator MMAC
// order and the iteration-6 epilogue are untouched.
//
// Iteration 11 (repair round): the iteration-10 candidate measured the fast
// mapping (327.36 us median, 104.96 TOPS, ~43.23x vs the fixed Triton
// baseline) but failed correctness (mismatch 16,775,923, max_abs_error 86.5)
// because its pack kernel vectorized along the wrong axis: it copied
// raw[kk][n0..n0+15] (16 consecutive n at fixed k) into packed[n0][kk..kk+15]
// (which must hold 16 consecutive k at fixed n).  That wrote only the packed
// rows with n0 % 16 == 0 (93.66% of the buffer unwritten) and did misaligned
// int4 stores whenever kk % 16 != 0.  The DUMMA kernel itself is verified
// correct against the gfx928 code object: the B-fragment path compiles to
// four ds_read2_b64 at exactly b_tile[(local_col + lane&15)*80 + kk +
// (lane>>4)*8] for (kk, frag) = (0,0), (32,0), (1280,1), (1312,1) (8-byte
// offset units), the A-fragment ds_read2_b32 instructions are byte-identical
// to the accepted kernel, and the v_mmac operand order (D, A, B) is
// unchanged, so with a correct n-major pack the int32 accumulation and the
// stored bf16 bits are the reference bits (mismatch 0, max_abs_error 0.0
// expected).  This round changes ONLY w8a8_pack_o_proj_panels_kernel: each
// thread now gathers its 16-byte destination vector from 16 raw rows
// (aligned byte loads strided by N) and stores one aligned int4; the pack is
// one-time, out-of-timed-region and out-of-Graph, so the timed GEMM kernel
// (staging, load_fragment8 fragments, MMAC order, epilogue, grid 2048,
// workgroup 256, LDS 15,360 B, arch_vgpr 82, zero scratch) is bit-identical
// to iteration 10.  The exact (m, n, k) guard, the scalar fallback decode
// (packed[col*K + kk] == raw[kk*N + col]) and the identity packs are
// untouched.
//
// Iteration 12 (A-fragment load round): the iteration-11 repair was accepted
// (348.27 us median, 98.66 TOPS, mismatch 0).  The exact gfx928 code object of
// the accepted kernel still shows the library du_load_matrix_sync path for the
// four matrix_a fragments emitting, per wave per stage, 8 ds_read2_b32 (A is
// m-major in LDS, so each lane's eight k-bytes are contiguous and the compiler
// already vectorized the byte loads) followed by a ~52-instruction per-dword
// mask/OR byte-reassembly VALU chain (v_and 0xff00/0xff0000/0xff000000 +
// v_or_b32_sdwa src0_sel:BYTE_0 + v_or3_b32) plus 7 lgkmcnt waitcnts that is
// arithmetic IDENTITY on every dword (the loaded bytes reach the v_mmac A
// operand unchanged): dead issue and dead latency on the A-load -> v_mmac
// dependency path.  This round applies the exact iteration-10 mechanism
// (load_fragment8, validated on B; the worker-29 TP4 gate_up lineage measured
// VALU 16.08M -> 6.25M, and the TP4 qkv lineage accepted iteration 18 replaced
// the library row_major byte-reassembly with an explicit A-fragment loader) to
// the four A du_load_matrix_sync calls: each A fragment is filled by one
// 8-byte LDS read at a_tile[(local_row + (lane&15))*80 + kk + ((lane>>4)<<3)]
// writing the same 8 bytes to the same x[0..7] slots the row_major loader
// produced => the v_mmac A operand bit pattern and the int32 accumulation are
// bit-identical (mismatch 0, max_abs_error 0.0 expected).  Tile, staging, the
// B load_fragment8 fragments, barriers, per-accumulator MMAC order, the
// iteration-6 coalesced epilogue, the pack layout, the exact (m, n, k) guard
// and the scalar fallback are untouched.

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
constexpr int kTargetN = 4096;
constexpr int kTargetK = 1024;
constexpr int kBlockM = 128;
constexpr int kBlockN = 64;
constexpr int kStageK = 64;
constexpr int kBPad = 16;  // 64 -> 80-byte LDS row stride (five bank phases)
constexpr int kBStride = kBlockN + kBPad;
constexpr int kAStride = kStageK + kBPad;  // 64 -> 80-byte A row stride
constexpr int kBlockThreads = 4 * kWaveSize;

using bf16_t = hip_bfloat16;

// ---------------------------------------------------------------------------
// Large-M prefill: 128x64 output tile per block, four wavefronts of 64 lanes.
// Each wave owns a 64x32 quadrant (eight m16n16k32 int32 accumulators); the
// block cooperatively stages A[128,64] from x_q (row-major, stride k) and
// B[64,64] from the packed n-major [N, K] weight into a single
// 64-K LDS buffer. Two barriers per stage: one after the cooperative load,
// one before the next stage overwrites LDS. Direct fragment epilogue.
// ---------------------------------------------------------------------------
template <typename AccFragment>
__device__ __forceinline__ void store_prefill_fragment(
    const AccFragment& frag,
    const float* __restrict__ x_scale,
    const float* __restrict__ weight_scale,
    bf16_t* __restrict__ out,
    int base_row,
    int base_col,
    int m,
    int n,
    int lane) {
  // Verified gfx928 int8 m16n16k32 accumulator ownership (matches
  // du_store_matrix_sync): lane & 15 selects the row, lane >> 4 selects
  // col % 4, and x[i] maps to columns col%4 + 4*i.
  const int row = base_row + (lane & 15);
  if (row >= m) {
    return;  // tail-M masking: padded rows never write
  }
  const int col_mod4 = lane >> 4;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const int col = base_col + col_mod4 + 4 * i;
    const float scaled = static_cast<float>(frag.x[i]) *
                         x_scale[row] * weight_scale[col];
    out[static_cast<int64_t>(row) * n + col] = __float2bfloat16(scaled);
  }
}

// Iteration 6 (epilogue round): coalesced fragment store. The m16n16k32
// accumulator lane mapping (row = lane & 15, column group c4 = lane >> 4,
// frag.x[i] -> column c4 + 4*i) gives each lane four elements strided by 4
// columns, so the direct store (store_prefill_fragment) is four 2-byte
// scalar stores per lane whose wavefront addresses touch each 32-B sector at
// 25% utilization (8 B used of 32 B; 4 store instructions per fragment per
// wave, i.e. 32 global_store_short_d16_hi per wavefront per block in the
// current gfx928 ISA). This epilogue transposes the 4-element groups within
// each 4-lane column group (lanes r, r+16, r+32, r+48 -- a 4x4 transpose,
// two 2x2 steps with shfl_xor 16 then 32, one v_cndmask per element per
// step; no staging tile round trip), so lane (r, c4) ends up holding the
// four CONTIGUOUS columns 4*c4 .. 4*c4+3, converts them to bf16, packs 4
// bf16 (8 B), and writes ONE 8-byte store per lane: 64 lanes x 8 B = 512 B
// per fragment per wavefront in 16 fully-used 32-B sectors (100% store
// sector efficiency; vmem_write_instructions drops 262,144 -> 65,536 for
// this shape). This is the exact mechanism validated on the worker-29 TP4
// qkv_proj lineage (accepted iteration 6, 770.14 -> 721.87 us): this DTK's
// __shfl_xor lowers to ds_bpermute (one LDS permute per shuffle, +64 LDS
// instructions per wavefront per block -- accepted because it runs at the
// block tail where the LDS pipe is otherwise idle; there is no global round
// trip and no staging tile). Only the int32 values are re-routed between
// lanes -- the per-element scale multiply order
// (float(dot) * x_scale[row] * weight_scale[col]) and the bf16 rounding are
// unchanged, so the stored bits are identical to store_prefill_fragment.
// The row>=m guard is wavefront-uniform (all 64 lanes of a wave share the
// same 16-row window base_row + (lane & 15)), so the shuffles never mix
// active and inactive lanes; base_col is a multiple of 16 and n*2 a multiple
// of 8, so the float4 weight_scale load (col0 % 4 == 0) and the 8-byte store
// ((row*n + col0) % 4 == 0 elements -> byte offset % 8 == 0) are aligned.
template <typename AccFragment>
__device__ __forceinline__ void store_prefill_fragment_coalesced(
    const AccFragment& frag,
    const float* __restrict__ x_scale,
    const float* __restrict__ weight_scale,
    bf16_t* __restrict__ out,
    int base_row,
    int base_col,
    int m,
    int n,
    int lane) {
  const int row = base_row + (lane & 15);
  if (row >= m) {
    return;  // tail-M masking: padded rows never write (wavefront-uniform)
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
  // aligned: base_col is a multiple of 16, n*2 is a multiple of 8).
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
  *reinterpret_cast<uint64_t*>(out + static_cast<int64_t>(row) * n + col0) =
      packed;
}

// Iteration 10: direct 8-byte LDS fragment fill for matrix_b on the n-major
// [N, K] B tile. du_load_matrix_sync's int8 matrix_b loaders assign
// x[i] = p[col*ldm + row + i] (row = lane & 15, col = (lane >> 4) << 3), i.e.
// eight consecutive k-values per lane on an [N, K] tile, and du_mma_sync
// passes b.x straight to the v_mmac builtin as one packed 8-byte operand;
// the library path makes the compiler emit eight per-byte ds_read_u8 plus a
// mask/OR reassembly chain per loaded dword. Writing the same 8 bytes
// directly into the fragment storage keeps the operand bit pattern identical
// (exact int32 accumulation unchanged) and lets the compiler feed a single
// 8-byte LDS read straight to the v_mmac. This is the exact pattern
// validated on the worker-29 TP4 gate_up lineage (accepted iteration 13).
template <typename Frag>
__device__ __forceinline__ void load_fragment8(
    Frag& frag,
    const signed char* __restrict__ base,
    int stride,
    int lane) {
  const int off = (lane & 15) * stride + ((lane >> 4) << 3);
  *reinterpret_cast<unsigned long long*>(&frag.x[0]) =
      *reinterpret_cast<const unsigned long long*>(base + off);
}

__global__ __launch_bounds__(kBlockThreads) void w8a8_dumma_128x64x64_kernel(
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

  // Single-buffered 64-K stage: A[128, 80] + B[64, 80] = 15,360 B/block
  // (4 blocks/CU fit the 64 KiB LDS budget; the padded 80-byte strides are
  // 16-byte-aligned and break the 64-byte LDS bank periodicity).
  // Iteration 10: B is staged n-major b_tile[n][k] (64 n-rows x 80 B - the
  // same 5,120 B as the old [k][n] tile), so each lane's eight B-fragment
  // k-values are contiguous and load as ONE 8-byte LDS read (load_fragment8)
  // instead of eight ds_read_u8 + mask/OR pack VALU.
  __shared__ __align__(16) int8_t a_tile[kBlockM * kAStride];
  __shared__ __align__(16) int8_t b_tile[kBlockN * kBStride];

  DUFragment<matrix_a, kTileM, kTileN, kTileK, signed char, row_major>
      a_frag0, a_frag1;
  DUFragment<matrix_b, kTileM, kTileN, kTileK, signed char, row_major>
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

  // Cooperative staging: each thread owns one int4 in A rows [0,64), one in
  // A rows [64,128), and one int4 of B. B comes from the packed n-major
  // [N, K] layout (iteration 10): B[k0+stage_kv, n0+b_n] is at
  // packed[(blockIdx.x*64 + b_n)*K + k0 + stage_kv], so the whole
  // (n-tile, K-stage) B tile is one contiguous 4 KiB vector stream and the
  // thread's global load and LDS store are each a single 16-byte-aligned
  // int4. A[128,64] is 512 int4s; B[64,64] is 256 int4s.
  const int vector_byte_offset = tid * static_cast<int>(sizeof(int4));
  const int stage_row = vector_byte_offset / kStageK;
  const int stage_col = vector_byte_offset - stage_row * kStageK;
  const int b_n = tid >> 2;            // n column of this thread's B int4
  const int b_kv = (tid & 3) * 16;     // k offset (stage-local) of the B int4

  for (int k0 = 0; k0 < k; k0 += kStageK) {
    *reinterpret_cast<int4*>(a_tile + stage_row * kAStride + stage_col) =
        *reinterpret_cast<const int4*>(
            x_q + static_cast<int64_t>(m0 + stage_row) * k + k0 + stage_col);
    *reinterpret_cast<int4*>(a_tile +
                             (stage_row + kBlockM / 2) * kAStride + stage_col) =
        *reinterpret_cast<const int4*>(
            x_q + static_cast<int64_t>(m0 + stage_row + kBlockM / 2) * k +
            k0 + stage_col);
    *reinterpret_cast<int4*>(b_tile + b_n * kBStride + b_kv) =
        *reinterpret_cast<const int4*>(
            weight + (static_cast<int64_t>(blockIdx.x) * kBlockN + b_n) * k +
            k0 + b_kv);
    __syncthreads();

    // Each wave consumes its 64x32 quadrant: eight m16n16k32 MMACs per kk.
#pragma unroll
    for (int kk = 0; kk < kStageK; kk += kTileK) {
      // Iteration 10: B is n-major in LDS, so each lane's eight fragment
      // k-values are contiguous; one 8-byte load fills the fragment (the
      // same 8 bytes the old row_major loader gathered with eight ds_read_u8
      // plus mask/OR pack VALU), keeping the v_mmac operand bit pattern and
      // the int32 accumulation identical.
      load_fragment8(b_frag0, b_tile + local_col * kBStride + kk, kBStride,
                     lane);
      load_fragment8(b_frag1, b_tile + (local_col + kTileN) * kBStride + kk,
                     kBStride, lane);
      // Iteration 12: A gets the same direct 8-byte fill (load_fragment8).
      // A is m-major in LDS, so lane (r, c4) owns the eight CONSECUTIVE
      // k-values a_tile[(local_row + r)*80 + kk + c4*8 .. +7]; the code
      // object shows du_load_matrix_sync already vectorizes these into
      // ds_read2_b32 but still executes the per-dword mask/OR reconstruction
      // (~52 identity VALU + 7 lgkmcnt waits per wave per stage) before the
      // v_mmac A operands.  Writing the same 8 bytes straight into the
      // fragment storage removes that dead chain; the operand bit pattern
      // and the int32 accumulation are unchanged.
      load_fragment8(a_frag0, a_tile + local_row * kAStride + kk, kAStride,
                     lane);
      load_fragment8(a_frag1, a_tile + (local_row + kTileM) * kAStride + kk,
                     kAStride, lane);
      du_mma_sync(acc00, a_frag0, b_frag0, acc00);
      du_mma_sync(acc01, a_frag0, b_frag1, acc01);
      du_mma_sync(acc10, a_frag1, b_frag0, acc10);
      du_mma_sync(acc11, a_frag1, b_frag1, acc11);
      load_fragment8(
          a_frag0, a_tile + (local_row + 2 * kTileM) * kAStride + kk,
          kAStride, lane);
      load_fragment8(
          a_frag1, a_tile + (local_row + 3 * kTileM) * kAStride + kk,
          kAStride, lane);
      du_mma_sync(acc20, a_frag0, b_frag0, acc20);
      du_mma_sync(acc21, a_frag0, b_frag1, acc21);
      du_mma_sync(acc30, a_frag1, b_frag0, acc30);
      du_mma_sync(acc31, a_frag1, b_frag1, acc31);
    }

    // Protect the LDS buffer from the next stage's cooperative overwrite.
    __syncthreads();
  }

  const int base_row = m0 + local_row;
  const int base_col = n0 + local_col;
  // Iteration 6: coalesced epilogue store (one 8-byte store per lane per
  // fragment; scales + bf16 conversion fused in-kernel, no workspace pass).
  store_prefill_fragment_coalesced(acc00, x_scale, weight_scale, out,
                                   base_row, base_col, m, n, lane);
  store_prefill_fragment_coalesced(acc01, x_scale, weight_scale, out,
                                   base_row, base_col + kTileN, m, n, lane);
  store_prefill_fragment_coalesced(acc10, x_scale, weight_scale, out,
                                   base_row + kTileM, base_col, m, n, lane);
  store_prefill_fragment_coalesced(acc11, x_scale, weight_scale, out,
                                   base_row + kTileM, base_col + kTileN, m, n,
                                   lane);
  store_prefill_fragment_coalesced(acc20, x_scale, weight_scale, out,
                                   base_row + 2 * kTileM, base_col, m, n,
                                   lane);
  store_prefill_fragment_coalesced(acc21, x_scale, weight_scale, out,
                                   base_row + 2 * kTileM, base_col + kTileN,
                                   m, n, lane);
  store_prefill_fragment_coalesced(acc30, x_scale, weight_scale, out,
                                   base_row + 3 * kTileM, base_col, m, n,
                                   lane);
  store_prefill_fragment_coalesced(acc31, x_scale, weight_scale, out,
                                   base_row + 3 * kTileM, base_col + kTileN,
                                   m, n, lane);
}

// ---------------------------------------------------------------------------
// Generic scalar fallback: one thread per output element. Exact int32 dot
// over K, then fused x_scale * weight_scale, then bf16 store. Handles every
// unmatched (m, n, k), including all small-M API cases (M=2, M=16) and any
// M in (0, 4096] with the same (K, N). For (k, n) == (1024, 4096) it decodes
// the packed n-major [N, K] layout (raw[kk, col] == packed[col*K + kk]);
// every other (k, n) reads the identity [K, N] layout.
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
  if (k == kTargetK && n == kTargetN) {
    // Iteration 10: the packed layout for (k, n) == (1024, 4096) is n-major
    // [N, K] (packed[col*K + kk] == raw[kk*N + col]).
    for (int kk = 0; kk < k; ++kk) {
      acc += static_cast<int32_t>(a_row[kk]) *
             static_cast<int32_t>(
                 weight[static_cast<int64_t>(col) * k + kk]);
    }
  } else {
    const int8_t* b_col = weight + col;
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
// Weight packing (outside the timed region and outside Graph capture; the
// packed buffer keeps the same byte count K*N and the same allocated
// address, so the layout is graph-stable).  For the target (k, n) ==
// (1024, 4096) the raw [K, N] row-major weight is transposed into the
// n-major [N, K] layout:
//     packed[n*K + kk] == raw[kk*N + n]
// so every DUMMA B tile (one 64-column n-tile x one 64-K stage) is a single
// contiguous 4 KiB stream of 16-byte vector loads and each lane's eight
// B-fragment k-values are contiguous in LDS (one 8-byte fragment read).
// Iteration 11 repair: the n-major transpose's 16-byte destination vectors
// run along K, so each thread GATHERS its 16 bytes from 16 raw rows (byte
// loads strided by N) instead of copying a contiguous raw vector (the
// iteration-10 pack copied 16 consecutive n at fixed k, leaving 93.66% of
// the packed buffer unwritten and issuing misaligned int4 stores).  The
// pack is one-time and out-of-timed-region, so the gather cost is free.
// For every other (k, n) the pack is an identity device-to-device copy.
// Scales are copied identity in both cases.
// ---------------------------------------------------------------------------
__global__ __launch_bounds__(256) void w8a8_pack_o_proj_panels_kernel(
    const int8_t* __restrict__ raw,
    int8_t* __restrict__ packed,
    int k,
    int n) {
  // Iteration 11 repair: one thread per 16-byte vector of the packed [N, K]
  // layout, but the vector is a GATHER, not a memcpy.  The destination vector
  // packed[n*K + kk0 .. kk0+15] holds 16 CONSECUTIVE k-values of raw column n
  // (packed[n*K + kk0 + j] == raw[(kk0 + j)*N + n], j = 0..15), so its 16
  // source bytes sit in 16 DIFFERENT raw rows (strided by N).  The
  // iteration-10 pack instead vectorized along the source axis (it copied
  // raw[kk][n0..n0+15] - 16 consecutive n at fixed k - into
  // packed[n0][kk..kk+15], which must hold 16 consecutive k at fixed n);
  // that wrote only the packed rows with n0 % 16 == 0 (93.66% of the buffer
  // left unwritten, rows 1..15 of every 16 never touched) and issued
  // misaligned int4 stores whenever kk % 16 != 0.  Here each thread gathers
  // its 16 bytes with aligned byte loads and stores one aligned int4.
  const int kk_vectors = k / 16;  // 16-k vectors per packed row (k % 16 == 0)
  const int64_t total4 = static_cast<int64_t>(n) * kk_vectors;
  const int64_t linear =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (linear >= total4) {
    return;
  }
  const int kk0 = static_cast<int>(linear % kk_vectors) * 16;  // multiple of 16
  const int nn = static_cast<int>(linear / kk_vectors);  // packed row = raw col
  uint32_t d[4] = {0, 0, 0, 0};
#pragma unroll
  for (int j = 0; j < 16; ++j) {
    const uint8_t b = raw[static_cast<int64_t>(kk0 + j) * n + nn];
    d[j >> 2] |= static_cast<uint32_t>(b) << (8 * (j & 3));
  }
  int4 v;
  v.x = static_cast<int>(d[0]);
  v.y = static_cast<int>(d[1]);
  v.z = static_cast<int>(d[2]);
  v.w = static_cast<int>(d[3]);
  // Destination offset nn*K + kk0 is 16-byte aligned (K % 16 == 0 and
  // kk0 % 16 == 0), so the int4 store is aligned.
  *reinterpret_cast<int4*>(packed + static_cast<int64_t>(nn) * k + kk0) = v;
}

__global__ __launch_bounds__(256) void w8a8_pack_identity_kernel(
    const int8_t* __restrict__ raw,
    int8_t* __restrict__ packed,
    int64_t count) {
  const int64_t linear =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (linear < count) {
    packed[linear] = raw[linear];
  }
}

__global__ __launch_bounds__(256) void w8a8_pack_scale_identity_kernel(
    const float* __restrict__ raw,
    float* __restrict__ packed,
    int count) {
  const int linear = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (linear < count) {
    packed[linear] = raw[linear];
  }
}

}  // namespace

// ---------------------------------------------------------------------------
// Stable host launch symbols consumed by csrc/bindings.cpp.
// Both launchers are pure dispatch: no allocation, no packing, no
// synchronization, no default-stream launch; they run on the caller-provided
// PyTorch stream and are CUDA-Graph safe.
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
  (void)workspace;  // no split-K: the GEMM does not use the workspace
  (void)workspace_bytes;
  auto* out_bf16 = reinterpret_cast<bf16_t*>(out);

  // Explicit dispatch. The assigned shape (M=4096, N=4096, K=1024) takes the
  // DUMMA 128x64 single-buffered path; every other (m, n, k) - including
  // small-M API cases (M=2, M=16) and M=3072 with the same (K, N) - takes the
  // scalar fallback, which decodes the packed n-major [N, K] layout for
  // (k, n) == (1024, 4096) and the identity layout otherwise.
  if (m == kTargetM && n == kTargetN && k == kTargetK) {
    const dim3 grid(kTargetN / kBlockN, kTargetM / kBlockM);
    const dim3 block(kBlockThreads);
    hipLaunchKernelGGL(w8a8_dumma_128x64x64_kernel,
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
  const dim3 block(kBlock);
  if (k == kTargetK && n == kTargetN) {
    const int64_t total4 = static_cast<int64_t>(k) * n / 16;
    const dim3 pack_grid(
        static_cast<unsigned>((total4 + kBlock - 1) / kBlock));
    hipLaunchKernelGGL(w8a8_pack_o_proj_panels_kernel,
                       pack_grid, block, 0, stream,
                       raw_weight, packed_weight, k, n);
  } else {
    const int64_t weight_count = static_cast<int64_t>(k) * n;
    const dim3 weight_grid(
        static_cast<unsigned>((weight_count + kBlock - 1) / kBlock));
    hipLaunchKernelGGL(w8a8_pack_identity_kernel,
                       weight_grid, block, 0, stream,
                       raw_weight, packed_weight, weight_count);
  }

  const int64_t scale_count = n;
  const dim3 scale_grid(
      static_cast<unsigned>((scale_count + kBlock - 1) / kBlock));
  hipLaunchKernelGGL(w8a8_pack_scale_identity_kernel,
                     scale_grid, block, 0, stream,
                     weight_scale, packed_weight_scale, n);
  (void)hipGetLastError();
}
// @@end
// @@end
