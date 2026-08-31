// @@variant shape=hy3_tp8_qkv_proj_m16 commit=467827c4a33a7e06fe4b3d9871cad428d75648fe added=2026-08-27
//   median_us=23.17 p90_us=23.21
//   source=hy3-dsh-tp8-m16-2-dc53295a
// INT8 W8A8 GEMM for Hygon K500SM_AI / gfx928.
//
// Iteration 1 (DUMMA bootstrap): replace the correctness-first scalar kernel
// for the assigned shape hy3_tp8_qkv_proj_m16 (M=16, N=1280, K=4096) with the
// minimal native gfx928 DUMMA INT8 m16n16k32 tile: one 64-lane wavefront per
// block, one 16x16 output tile per block (grid = N/16 = 80 blocks), direct
// global-to-fragment loads (no LDS staging), explicit int32 accumulation, and
// a single-wave LDS epilogue with no cross-wave barrier. Measured 176.98 us.
//
// Iteration 2 (architecture round: launch geometry): keep the direct-load
// data path unchanged and change the launch geometry to 2 wavefronts per
// block (128 threads) with in-block split-K=2.  Each wavefront accumulates
// exactly one K-half (2048 = 64 m16n16k32 steps) in int32 k-ascending order,
// publishes its int32 partial to its own LDS plane, and after one END-of-K
// barrier the two wavefronts add partial 0 + partial 1 (ascending split
// order, bit-identical to the unsplit k-ascending int32 accumulation) and
// emit the scaled bf16 tile.  grid stays N/16 = 80 blocks.  Measured
// 102.12 us median / 105.59 us p90 (vs the 74.782 us Triton Graph baseline);
// the ~1.73x geometry gain shows each active CU still runs one serial
// load->wait->MMA chain per wave (64 steps after split-K=2), with the two
// chains of one block co-scheduled on the same CU.
//
// Iteration 3 (architecture round: workspace split-K pair).  Mandate: the
// unsplit grid (80 blocks) has fewer than two blocks per device CU (120 CUs)
// and K=4096 >= 1024, so implement split-K=2 plus at least one CU-aligned
// candidate (non-power-of-two allowed), writing int32 partials into the
// caller workspace and including the combine+scale kernel in the timed Graph.
// Replace the iteration-2 single-launch in-block-combine kernel with a
// workspace split-K pair (the pattern validated on the hy3 TP4 qkv lineage,
// which took the same architecture round at its iteration 3):
//
//   1. w8a8_dumma_m16_splitk_partial_kernel<SPLIT_K>: one 64-thread
//      wavefront per block, grid = (N/16 = 80 tiles, SPLIT_K K-slices).
//      Each block computes the int32 partial dot for one 16x16 output tile
//      over one 32-aligned K slice and du_store_matrix_sync's the partial
//      straight into the caller workspace.  ZERO barriers, ZERO LDS, ZERO
//      atomics in the main kernel.
//   2. w8a8_dumma_m16_combine_scale_kernel: 80 blocks x 256 threads, one
//      thread per output element; exact int32 sum of the SPLIT_K partials
//      (ascending split order), then x_scale/weight_scale multiply and the
//      bf16 (RN-even) epilogue.  Barrier-free (each thread reads only its
//      own 16x16 element's S partial values, written by the previous kernel
//      on the same stream).
//
// Both kernels launch on the caller's stream inside the timed Graph, so the
// combine cost is included in the measured operator latency.
//
// Iteration 4 (register prefetch, rejected): double-buffered a_frag/b_frag in
// registers one K step ahead.  Regressed to 120.5 us median: the exact code
// object kept the full vmcnt drain before each v_mmac, so prefetching could
// not remove the per-step global round trip from the MMA critical path.
// Iterations 5-6 were killed by the agent infrastructure (no evidence).
//
// Iteration 7 (architecture round: LDS slice staging).  The exact gfx928 code
// object of the iteration-3 kernel shows every one of the 64 K steps compiles
// to 16 narrow global_load_ubyte + byte-reassembly VALU + a full progressive
// vmcnt drain before the single v_mmac: the whole global round trip is paid
// serially per step with only ~1.3 waves/CU to hide it.  This round applies
// the TP4 qkv lineage remedy (validated 26.05 us on N=2560) scaled to this
// shape: stage the block's entire 32-aligned K slice into LDS once with
// coalesced dword global loads, one __syncthreads, then a zero-barrier
// LDS-only K loop (du_load_matrix_sync reads both fragments from LDS, so the
// MMA path sees only ~30-cycle LDS latency instead of a global round trip
// per step):
//
//   w8a8_dumma_m16_slicestage_partial_kernel<SPLIT_K>: one 64-thread
//   wavefront per block, grid = (N/16 = 80 tiles, SPLIT_K K-slices), same
//   per-tile 16x16 output and the same workspace int32 plane layout, so the
//   combine+scale kernel below is unchanged.  A (16 x slice) and B
//   (slice x 16) are staged into LDS with the same 32-aligned non-uniform
//   slice bounds and the same k-ascending int32 MMA order as iteration 3, so
//   outputs stay bit-identical (0 mismatches) and Graph capture/replay with
//   changed contents must pass.
//
// Round-8 geometry (default): split-K=16 -> grid = (80 tiles, 16 slices) =
// 1,280 one-wave blocks = 10.67 blocks/CU on the 120-CU device.  Each block
// stages exactly 256 K (A 16x256 -> 4,160 B + B 256x16 -> 5,120 B = 9,280 B
// LDS = 7 resident blocks/CU by the 64 KiB LDS) and runs exactly 8 uniform
// zero-barrier LDS MMAs (128/16 = 8, so no 11-step stragglers).  This is an
// occupancy (LDS-footprint) tune against the round-7 PMC evidence: at
// split-K=12 the profile shows grid 960 = 8 waves/CU available but
// 12,736 B LDS/block caps residency at 5 blocks/CU (62.5%), with VGPR 24 /
// scratch 0 / SGPR 32 proving registers and spills are NOT limiters -- the
// per-block LDS footprint is the binding occupancy limiter, and S=16 is the
// largest split that fits the 16-plane contract workspace (1,310,720 B
// exactly), so residency rises to 7 of 10.67 waves/CU.  A bytes are reused
// across the 80 tile blocks of each slice (1,280 x 16 x 256 = 5.24 MB total
// A reads vs 64 KB unique, L2-hot); B 16-column strips are read exactly once
// (5.24 MB unique), with the 64-B L2 sectors shared with the adjacent
// 16-column tiles, so HBM A/B traffic is byte-identical to split-K=12
// (~5.30 MB unique; no repeated HBM reads traded for occupancy -- only the
// split-K combine workspace round trip grows from 12 to 16 planes, +0.33 MB
// write + read).  vmem_read_instructions stay ~40,960/replay (slightly fewer
// than S=12's 42,240: each 256-K slice needs 32 wavefront loads vs 44).
//
// Iteration 11 (packed col-major B; the validated TP4 gate_up/down_proj
// recipe).  The accepted S=16 staged kernel's fresh PMC shows
// lds_bank_conflicts 81,920 == the 8 scalar ds_read_u8 per matrix_b fragment
// (8 steps x 1,280 blocks), i.e. every B fragment read in the zero-barrier K
// loop conflicts at the 20-B k-major pitch, while round 9 falsified
// staging-load MLP batching (58.56 us), so the K-loop B fragment reads are
// the remaining hot-path lever.  launch_pack_w8a8_weight now transposes the
// exact (k,n)==(4096,1280) weight to [N,K] n-major once, outside the timed
// region and out of Graph capture (every other (K,N) keeps the identity
// copy); the staged kernel stages its 16-column strip as 16-B int4 straight
// copies into an n-major LDS tile b_s[16][kMaxKlen+16] (row stride
// kMaxKlen+16: 16-B aligned for the int4 stores, 8-B aligned for the
// fragment b64 reads, not a multiple of 128 B -> no bank-phase aliasing), so
// each lane's 8 fragment bytes are consecutive and load with ONE ds_read_b64
// (load_b_frag_packed) instead of 8 conflicting ds_read_u8.  The direct
// S in {2,3,4,5} path reads the packed global layout through the library
// col_major loader, and the scalar fallback decodes the packed layout for
// (k,n)==(4096,1280) (covers the paired M=2 validation shape).  Fragment
// contents, the k-ascending int32 du_mma order, the slice-major int32 plane
// layout, the combine+scale kernel and the generic fallback are all
// unchanged, so outputs stay bit-identical (0 mismatches) and Graph
// capture/replay with changed contents must pass.  Per-block LDS drops
// 9,280 -> 8,512 B (65,536/8,512 = 7.70, still 7 resident/CU); A/B HBM bytes
// are unchanged (B stays read exactly once: 16-B contiguous per n-row, each
// 64-B sector read once at the uniform S=16 default).
//
// Iteration 12 (staging-load MLP batching, alignment-safe).  The fresh
// exact-source ISA of the accepted iteration-11 object shows both staging
// loops still compile to a fully serialized load -> s_waitcnt vmcnt(0) ->
// store chain (A: 16 sequential global round trips per block; B: 4), i.e.
// every block pays ~20 serial global round trips before the single barrier,
// while the K loop is already lean (2 LDS reads + 2 lgkmcnt waits + 1 v_mmac
// per step).  Iteration 11 already validated the B-side fix (16 serialized
// dword loads -> 4 aligned int4 loads as part of the accepted 40.38 ->
// 30.66 us round).  This round batches both staging loops 4-deep in source:
// 4 independent loads are issued before their 4 stores, so one vmcnt drain
// covers 4 in-flight round trips (A 16 -> 4, B 4 -> 1; ~20 -> ~5 per block).
// A stays dword-wide because kAStride = kMaxKlen+4 is not 16-B aligned --
// round 9's 58.56 us regression was misaligned int4 staging on the old
// layouts, not an MLP falsification; B stays int4 (aligned).  No layout,
// stride, fragment, slice, plane, combine, guard or fallback changes:
// outputs stay bit-identical (0 mismatches) and Graph capture/replay with
// changed contents must pass.  vmem_read/lds instruction counts are
// unchanged (25,600 / 46,080); the wait structure is what changes.
//
// The ZTH_W8A8_QKV_SPLIT_K environment override keeps its semantics but now
// selects the data path too: S in {6,8,9,12,16} run the staged kernel (their
// slices fit the 64 KiB LDS), S in {2,3,4,5} keep the iteration-3 direct-load
// kernel (S=2's 2,048-K slice alone would need 64 KiB for B, and S=3's
// 1,408-K slice does not fit either).
//
// Round-3 geometry (direct-load path, still selectable for S in
// {2,3,4,5}): split-K=2 -> 160
// one-wave zero-barrier blocks
// (160 independent wavefronts = the same total parallelism as iteration 2's
// 80 blocks x 2 waves, but with no in-block barrier, no LDS partial planes,
// and 160 independent block-level load streams that the block scheduler can
// co-resident 2-deep on 40 of the 120 CUs).  The trusted occupancy-probe
// sweep SPLIT_K in {2,3,4,5,6,8,9,12,16} is implemented as template
// instantiations and selectable at launch time with the
// ZTH_W8A8_QKV_SPLIT_K environment variable, so the control plane can
// measure split-K=2 and the CU-aligned candidates without source edits.
// For the direct path only, the CU-aligned (integer blocks/CU) candidates
// for N/16=80 tiles were S=3 -> 240 blocks = exactly 2 blocks/CU
// (non-power-of-two), S=6 -> 480 = 4/CU, S=9 -> 720 = 6/CU, S=12 -> 960 =
// 8/CU; with the iteration-7/8 routing those S values now run the staged
// kernel instead (block counts 480/720/960/1,280 for S=6/9/12/16), and
// S in {2,3,4,5} keep the direct path.  Every candidate fits the contract
// workspace (1,310,720 bytes = capacity 16 planes of 81,920 B; S=16 uses
// exactly all 16 planes) and keeps
// every K slice a multiple of the DUMMA K tile 32 (non-uniform slices differ
// by at most one 32-K step; S=16 is uniform: 128/16 = 8 steps).
//
// Exact int32 accumulation is preserved: each slice accumulates int32 over an
// ascending 32-aligned K range; slices tile [0, K) exactly once; the combine
// sums the partials in int32 ascending split order (per-tile max |dot| =
// 4096*127*127 = 66,064,384 << 2^31), so every per-tile dot total is
// bit-identical to the scalar fallback and to iterations 1-2 regardless of
// SPLIT_K.  The scale + bf16 (RN-even) epilogue runs once per output element
// in the combine kernel.
//
// Launch is guarded by the exact (m,n,k) = (16,1280,4096) shape; every other
// shape (including the paired M=2 shape with the same (N,K)) falls back to
// the generic scalar kernel below.  launch_pack_w8a8_weight is an identity
// device-to-device copy for every (K,N) except the exact pair
// (k,n)==(4096,1280), which is transposed once to an [N,K] n-major packed
// layout (iteration 11) so DUMMA col_major B fragments read 8 consecutive
// bytes per lane; the scalar fallback decodes that packed layout for the
// same pair, so every path stays correct against the packed buffer.
//
// Graph-safety: this file only launches kernels on the caller-provided stream
// (PyTorch's current HIP stream).  It performs no allocation, compilation,
// autotuning, packing, host synchronization, or device synchronization, and
// touches no buffer other than the caller-provided out/workspace pointers.
// The split-K choice is a host-side static dispatch decision (env override or
// default 16) made once per launch call, never inside the Graph replay, so
// capture/replay determinism holds.
//
// Header order is deliberate: hip_runtime.h, then hip_bfloat16.h, then
// du_mma.h (du_mma.h on this DTK is not self-contained before the HIP runtime
// headers).

#include <hip/hip_runtime.h>
#include <hip/hip_bfloat16.h>
#include <du_mma.h>

#include <cstdint>
#include <cstdlib>

namespace {

// gfx928 native wavefront is 64 lanes; every block size below is a multiple
// of 64.
constexpr int kScalarBlockThreads = 128;
constexpr int kCopyBlockThreads = 256;

// Minimal DUMMA tile constants (gfx928 INT8 support is m16n16k32).
constexpr int kDummaM = 16;
constexpr int kDummaN = 16;
constexpr int kDummaK = 32;
constexpr int kDummaThreads = 64;      // one wavefront (64 lanes) per block
constexpr int kCombineThreads = 256;   // 16x16 = 256 output elements/block

// Exact (k,n) pair whose weight launch_pack_w8a8_weight transposes to the
// [N,K] n-major packed layout and that the scalar fallback decodes as
// n-major (iteration 11; covers the paired M=2 validation shape).
constexpr int kPackedK = 4096;
constexpr int kPackedN = 1280;

// Trusted occupancy-probe split-K candidates (control plane): all fit the
// contract workspace for this shape (1,310,720 B = 16 partial planes of
// 81,920 B each) and produce 32-aligned K slices.  Block counts = 80*S:
// S=3 -> 240 blocks = exactly 2 blocks/CU (CU-aligned, non-power-of-two),
// S=6 -> 480 = 4/CU, S=9 -> 720 = 6/CU, S=12 -> 960 = 8/CU,
// S=16 -> 1,280 = 10.67/CU (16 is outside the old probe list but is the
// largest split that fits the 16-plane workspace, divides 128 K-steps
// exactly into uniform 256-K slices, and is the iteration-8 default: it
// lowers the binding occupancy limiter -- per-block LDS footprint -- from
// 12,736 B (5 resident/CU) to 9,280 B (7 resident/CU)).
constexpr int kTrustedSplitK[] = {2, 3, 4, 5, 6, 8, 9, 12, 16};

// ---------------------------------------------------------------------------
// Scalar GEMM kernel: one thread computes one output element (generic
// fallback for every unmatched shape, including the paired M=2 shape).
// ---------------------------------------------------------------------------
// Adjacent lanes own adjacent N columns (the fastest-changing dimension), so
// output stores are coalesced across each wavefront.
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
      static_cast<int64_t>(blockIdx.x) * kScalarBlockThreads + threadIdx.x;
  const int64_t total = static_cast<int64_t>(m) * n;
  if (linear >= total) {
    return;
  }
  const int row = static_cast<int>(linear / n);
  const int col = static_cast<int>(linear - static_cast<int64_t>(row) * n);

  // Complete K loop, k ascending, accumulated exactly in int32.  The maximum
  // API K (<= 6144) keeps the int8 dot well inside int32 range.
  const int8_t* a_row = a + static_cast<int64_t>(row) * k;
  // Iteration 11: the exact (k,n)==(4096,1280) pair is packed [N,K] n-major
  // by launch_pack_w8a8_weight; the fallback decodes that layout so it stays
  // correct against the packed buffer (paired M=2 validation shape included).
  // Every other (K,N) keeps the raw [K,N] row-major decode.
  const bool packed = (k == kPackedK && n == kPackedN);
  const int8_t* b_col =
      b + (packed ? static_cast<int64_t>(col) * k : col);
  const int64_t b_stride = packed ? 1 : static_cast<int64_t>(n);
  int32_t acc = 0;
  for (int kk = 0; kk < k; ++kk) {
    acc += static_cast<int32_t>(a_row[kk]) *
           static_cast<int32_t>(b_col[static_cast<int64_t>(kk) * b_stride]);
  }

  // Same float32 evaluation order as the exact reference:
  // (float(dot) * x_scale[m]) * weight_scale[n], then bf16 (RN).
  const float scaled = static_cast<float>(acc) * x_scale[row] *
                       weight_scale[col];
  out[linear] = __float2bfloat16(scaled);
}

void launch_scalar_gemm(
    const int8_t* a,
    const int8_t* b,
    const float* x_scale,
    const float* weight_scale,
    hip_bfloat16* out,
    int m,
    int n,
    int k,
    hipStream_t stream) {
  const int64_t total = static_cast<int64_t>(m) * n;
  const unsigned grid = static_cast<unsigned>(
      (total + kScalarBlockThreads - 1) / kScalarBlockThreads);
  hipLaunchKernelGGL(
      w8a8_scalar_gemm_kernel,
      dim3(grid),
      dim3(kScalarBlockThreads),
      0,
      stream,
      a,
      b,
      x_scale,
      weight_scale,
      out,
      m,
      n,
      k);
}

// ---------------------------------------------------------------------------
// Workspace split-K partial kernel (template over SPLIT_K), M == 16:
//   partial[tile, slice][m, n] = int32_dot(x_q[m, k0:k1], weight[k0:k1, n])
// ---------------------------------------------------------------------------
// Layouts (all contiguous):
//   x_q      [16, K] int8 row-major (ldm = K)
//   weight   [K, N] int8 row-major (ldm = N, identity pack layout)
//   partials [(SPLIT_K * num_tiles) * 256] int32, plane = slice*num_tiles +
//            tile, row-major 16x16 tile per plane (caller workspace)
// One block = one wavefront = one (tile, slice).  The K slice is 32-aligned
// (non-uniform: slices differ by at most one 32-K DUMMA step) and covers
// [k0, k0 + steps*32) in ascending order.  No LDS, no __syncthreads, no
// atomics: each block writes exactly one private int32 partial plane.
template <int SPLIT_K>
__global__ __launch_bounds__(kDummaThreads) void
w8a8_dumma_m16_splitk_partial_kernel(
    const int8_t* __restrict__ a,
    const int8_t* __restrict__ b,
    int32_t* __restrict__ partials,
    int n,
    int k) {
  const int tile = static_cast<int>(blockIdx.x);
  const int slice = static_cast<int>(blockIdx.y);
  const int n0 = tile * kDummaN;

  // 32-aligned non-uniform slice bounds over the K dimension.
  const int total_steps = k / kDummaK;  // 4096 / 32 = 128
  const int base = total_steps / SPLIT_K;
  const int rem = total_steps - base * SPLIT_K;
  const int steps = base + (slice < rem ? 1 : 0);
  const int k0 = (slice * base + (slice < rem ? slice : rem)) * kDummaK;

  du::dumma::DUFragment<du::dumma::matrix_a, kDummaM, kDummaN, kDummaK,
                        signed char, du::dumma::row_major>
      a_frag;
  // Iteration 11: b is the packed [N,K] n-major weight for this shape, so
  // the B fragment is col_major (each lane's 8 fragment bytes are consecutive
  // in memory -> one 8-B read instead of 8 byte reads).
  du::dumma::DUFragment<du::dumma::matrix_b, kDummaM, kDummaN, kDummaK,
                        signed char, du::dumma::col_major>
      b_frag;
  du::dumma::DUFragment<du::dumma::accumulator, kDummaM, kDummaN, kDummaK,
                        int>
      acc_frag;
  du::dumma::du_fill_fragment(acc_frag, 0);

  for (int i = 0; i < steps; ++i) {
    const int kk = k0 + i * kDummaK;
    du::dumma::du_load_matrix_sync(a_frag, a + kk, k);
    // Packed n-major row n starts at b[n*k + k0]; the library col_major
    // loader addresses p[(lane&15)*ldm + ((lane>>4)<<3) + i], i.e. n-major
    // rows of k bytes, so the tile's 16 columns live at n0..n0+15 n-rows
    // with row stride k.
    du::dumma::du_load_matrix_sync(
        b_frag, b + static_cast<int64_t>(n0) * k + kk, k);
    du::dumma::du_mma_sync(acc_frag, a_frag, b_frag, acc_frag);
  }

  // Private int32 partial plane: slice-major so a tile's S planes are
  // stride-uniform in the combine kernel.
  const int num_tiles = static_cast<int>(gridDim.x);
  du::dumma::du_store_matrix_sync(
      partials + (slice * num_tiles + tile) * (kDummaM * kDummaN),
      acc_frag,
      kDummaN,
      du::dumma::mem_row_major);
}

// Template launch helper for the partial kernel (one wavefront per block,
// grid = (num_tiles, SPLIT_K)).
template <int SPLIT_K>
void launch_splitk_partial(
    const int8_t* a,
    const int8_t* b,
    int32_t* partials,
    int n,
    int k,
    hipStream_t stream,
    int num_tiles) {
  hipLaunchKernelGGL(
      HIP_KERNEL_NAME(w8a8_dumma_m16_splitk_partial_kernel<SPLIT_K>),
      dim3(static_cast<unsigned>(num_tiles), static_cast<unsigned>(SPLIT_K)),
      dim3(static_cast<unsigned>(kDummaThreads)),
      0,
      stream,
      a,
      b,
      partials,
      n,
      k);
}

// Iteration 11 packed B-fragment loader (validated on the TP4 gate_up
// (4096,768) lineage): with n-major [N,K] storage and a col_major fragment,
// du_mma.h's matrix_b col_major address rule is x[i] = p[(lane&15)*ldm +
// ((lane>>4)<<3) + i] -- lane l's 8 fragment bytes are consecutive in
// memory, so one 8-byte LDS read per fragment replaces the 8 ds_read_u8 at
// stride-64 addresses that the row-major loader emits.  du_mma_sync consumes
// the fragment as one 64-bit value per lane, so writing the packed bytes
// straight into x[0..7] is exactly the library's own storage convention.
// p must be 8 B aligned (guaranteed: b_s is __align__(256), kBStride % 8 ==
// 0, step offsets are multiples of 32, and (lane&15)*kBStride % 8 == 0).
__device__ __forceinline__ void load_b_frag_packed(
    du::dumma::DUFragment<du::dumma::matrix_b, kDummaM, kDummaN, kDummaK,
                          signed char, du::dumma::col_major>& b_frag,
    const int8_t* p, int ldm, int lane) {
  const int row = lane & 15;
  const int col = (lane >> 4) << 3;
  const uint64_t v = *reinterpret_cast<const uint64_t*>(
      p + static_cast<int64_t>(row) * ldm + col);
  *reinterpret_cast<uint64_t*>(b_frag.x) = v;
}

// ---------------------------------------------------------------------------
// Whole-K-slice LDS staging partial kernel (M == 16, iteration 7):
//   partial[tile, slice][m, n] = int32_dot(x_q[m, k0:k1], weight[k0:k1, n])
// ---------------------------------------------------------------------------
// Same 32-aligned non-uniform K slices and the same k-ascending int32
// du_mma accumulation as w8a8_dumma_m16_splitk_partial_kernel, so the
// combine+scale kernel below sums bit-identical partials and the outputs are
// bit-identical to the accepted iteration-3 kernel.  Differences:
//   - A (16 x slice) is staged ONCE into LDS with coalesced dword global
//     loads (4 B/lane, 256 B per wavefront instruction); B (16 x slice) is
//     staged as 16-B int4 loads straight from the packed [N,K] n-major
//     weight (iteration 11) into an n-major LDS tile.
//   - One __syncthreads() after staging, then a zero-barrier LDS-only K
//     loop: du_load_matrix_sync reads the A fragment from LDS and
//     load_b_frag_packed reads each B fragment with one ds_read_b64 (~30-cycle
//     LDS latency) instead of a global round trip per K step.
// Layouts (all contiguous):
//   x_q      [16, K] int8 row-major (ldm = K)
//   weight   [K, N] int8 row-major for every (K,N) except the exact pair
//            (k,n)==(4096,1280), which is packed [N,K] n-major (element
//            (k,n) at P[n*K+k]; iteration 11)
//   partials [(SPLIT_K * num_tiles) * 256] int32, plane = slice*num_tiles +
//            tile, row-major 16x16 tile per plane (caller workspace)
// LDS: a_s[16][kMaxKlen + 4] (compile-time row stride kMaxKlen+4, an odd
// dword count -> the library's matrix_a row-major lane pattern (row =
// lane&15, 8 consecutive bytes per lane) is bank-conflict-free);
// b_s[16][kMaxKlen + 16] (iteration 11: n-major, row stride kMaxKlen+16 =
// 16-B aligned for the int4 staging stores, 8-B aligned for the col_major
// fragment b64 reads, not a multiple of 128 B -> the 16 staged n-rows do not
// alias onto one LDS bank phase; each lane's 8 fragment bytes are
// consecutive, so every B fragment loads with ONE ds_read_b64 instead of 8
// ds_read_u8).  Per-template LDS fits: S=6 22,848 B, S=8 16,704 B, S=9
// 15,680 B, S=12 11,584 B, S=16 8,512 B (all <= 64 KiB; resident blocks/CU
// by LDS: 2/3/4/5/7).  Only launched
// under the exact (m,n,k) == (16,1280,4096) guard, so total_steps = k/32 =
// 128 is exact for the compile-time LDS sizing below.
template <int SPLIT_K>
__global__ __launch_bounds__(kDummaThreads) void
w8a8_dumma_m16_slicestage_partial_kernel(
    const int8_t* __restrict__ a,
    const int8_t* __restrict__ b,
    int32_t* __restrict__ partials,
    int n,
    int k) {
  const int tile = static_cast<int>(blockIdx.x);
  const int slice = static_cast<int>(blockIdx.y);
  const int n0 = tile * kDummaN;

  // 32-aligned non-uniform slice bounds (identical formula to the direct
  // kernel: slices tile [0, K) exactly once in ascending order).
  const int total_steps = k / kDummaK;  // 4096 / 32 = 128 under the guard
  const int base = total_steps / SPLIT_K;
  const int rem = total_steps - base * SPLIT_K;
  const int steps = base + (slice < rem ? 1 : 0);
  const int k0 = (slice * base + (slice < rem ? slice : rem)) * kDummaK;
  const int klen = steps * kDummaK;

  // Compile-time LDS sizing for the longest slice this instantiation can
  // receive (base+1 steps, or base when SPLIT_K divides 128 evenly).
  constexpr int kMaxSteps = 128 / SPLIT_K + (128 % SPLIT_K == 0 ? 0 : 1);
  constexpr int kMaxKlen = kMaxSteps * kDummaK;
  __shared__ int8_t a_s[16][kMaxKlen + 4];
  // Iteration 11: B is staged n-major (16 columns x kMaxKlen+16 bytes, row
  // stride kMaxKlen+16) so col_major B fragments read 8 consecutive bytes
  // per lane (one ds_read_b64).  kMaxKlen is a multiple of 32, so the row
  // stride is 16-B aligned (int4 staging stores stay aligned), 8-B aligned
  // (fragment b64 reads stay aligned) and not a multiple of 128 B (no LDS
  // bank-phase aliasing across the 16 staged n-rows).
  __shared__ __align__(256) int8_t b_s[16][kMaxKlen + 16];

  // Row strides are compile-time constants equal to the array row strides
  // (a_s: kMaxKlen+4 bytes = odd dword count; b_s: kMaxKlen+16 bytes), so
  // the staging stores and the K-loop fragment reads always agree even for
  // short slices (klen < kMaxKlen).  The A stride stays an odd dword count
  // (bank-conflict-free row-major lane pattern); the B n-major stride keeps
  // every fragment b64 read 8 B aligned and off the single bank phase.
  constexpr int kAStride = kMaxKlen + 4;
  constexpr int kBStride = kMaxKlen + 16;

  // Stage A: 16 rows x klen bytes = 4*klen dwords; consecutive lanes copy
  // consecutive dwords of one row (coalesced 256 B per wavefront load).
  // Iteration 12 (staging MLP, alignment-safe): the fresh exact-source ISA
  // (iteration-12 profile, digest ab7fc108...) shows both staging loops
  // compile to a fully serialized load -> s_waitcnt vmcnt(0) -> store chain
  // (A: 16 sequential global round trips per block, B: 4), and the accepted
  // iteration-11 round already validated this fix on the B side (16
  // serialized dword loads -> 4 aligned int4 loads).  Both loops below batch
  // 4-deep in source: 4 independent loads are issued before their 4 stores,
  // so one vmcnt drain covers 4 in-flight round trips (A 16 -> 4, B 4 -> 1).
  // A stays dword-wide: 4-B loads AND stores are aligned at every a_s row,
  // while kAStride = kMaxKlen+4 is not 16-B aligned, so int4 A staging would
  // repeat round 9's misaligned-store regression (58.56 us) -- that round
  // falsified misaligned int4 staging, not the MLP idea.  B stays int4 (its
  // global rows are 16-B aligned and kBStride % 16 == 0).  Fragment reads,
  // LDS strides, the K loop and the int32 accumulation order are untouched,
  // so outputs stay bit-identical (0 mismatches) and Graph capture/replay
  // with changed contents must pass.  klen % 32 == 0 => a_dwords % 128 == 0,
  // so the batch loop is exact whenever klen % 128 == 0 (S=16: klen = 256 ->
  // 1024 dwords = 4 batches exactly; S=6/9/12's odd slices fall through to
  // the scalar tail, which preserves correctness for every S in the sweep).
  const int a_dwords = 4 * klen;
  const int a_dw_per_row = klen / 4;
  constexpr int kABatch = 4 * kDummaThreads;
  {
    const auto ld_a = [&](int e) {
      const int row = e / a_dw_per_row;
      return *reinterpret_cast<const int32_t*>(
          a + static_cast<int64_t>(row) * k + k0 + (e - row * a_dw_per_row) * 4);
    };
    const auto st_a = [&](int e, int32_t v) {
      const int row = e / a_dw_per_row;
      *reinterpret_cast<int32_t*>(&a_s[row][(e - row * a_dw_per_row) * 4]) = v;
    };
    int e = static_cast<int>(threadIdx.x);
    for (; e + 3 * kDummaThreads < a_dwords; e += kABatch) {
      const int32_t v0 = ld_a(e);
      const int32_t v1 = ld_a(e + kDummaThreads);
      const int32_t v2 = ld_a(e + 2 * kDummaThreads);
      const int32_t v3 = ld_a(e + 3 * kDummaThreads);
      st_a(e, v0);
      st_a(e + kDummaThreads, v1);
      st_a(e + 2 * kDummaThreads, v2);
      st_a(e + 3 * kDummaThreads, v3);
    }
    for (; e < a_dwords; e += kDummaThreads) {
      st_a(e, ld_a(e));
    }
  }

  // Stage B (iteration 11): the exact (k,n)==(4096,1280) weight is packed
  // [N,K] n-major (element (k,n) at P[n*K+k]) by launch_pack_w8a8_weight, so
  // the block's 16-column strip is 16 n-rows of klen contiguous bytes.  Each
  // lane stages 16-B int4s straight into the n-major LDS tile b_s[16][
  // kMaxKlen+16] (4 loads per lane at the uniform S=16 default: rows 0..15 x
  // 16 int4 columns).  Iteration 12: the same 4-deep batching as A, so the 4
  // aligned int4 round trips collapse to one vmcnt drain per block.
  const int k16_per_row = klen / 16;
  const int b_int4s = kDummaN * k16_per_row;
  const int4* __restrict__ b4 = reinterpret_cast<const int4*>(b);
  int4* __restrict__ lds_b4 = reinterpret_cast<int4*>(b_s);
  constexpr int kBBatch = 4 * kDummaThreads;
  {
    const auto ld_b = [&](int e) {
      const int row = e / k16_per_row;
      return b4[(static_cast<int64_t>(n0) + row) * (k >> 4) + (k0 >> 4) +
                (e - row * k16_per_row)];
    };
    const auto st_b = [&](int e, const int4& v) {
      const int row = e / k16_per_row;
      lds_b4[row * (kBStride >> 4) + (e - row * k16_per_row)] = v;
    };
    int e = static_cast<int>(threadIdx.x);
    for (; e + 3 * kDummaThreads < b_int4s; e += kBBatch) {
      const int4 v0 = ld_b(e);
      const int4 v1 = ld_b(e + kDummaThreads);
      const int4 v2 = ld_b(e + 2 * kDummaThreads);
      const int4 v3 = ld_b(e + 3 * kDummaThreads);
      st_b(e, v0);
      st_b(e + kDummaThreads, v1);
      st_b(e + 2 * kDummaThreads, v2);
      st_b(e + 3 * kDummaThreads, v3);
    }
    for (; e < b_int4s; e += kDummaThreads) {
      st_b(e, ld_b(e));
    }
  }

  __syncthreads();

  du::dumma::DUFragment<du::dumma::matrix_a, kDummaM, kDummaN, kDummaK,
                        signed char, du::dumma::row_major>
      a_frag;
  // Iteration 11: col_major B fragments match the n-major staged tile, so
  // each lane's 8 fragment bytes are consecutive (one ds_read_b64 per
  // fragment in the K loop instead of 8 ds_read_u8).
  du::dumma::DUFragment<du::dumma::matrix_b, kDummaM, kDummaN, kDummaK,
                        signed char, du::dumma::col_major>
      b_frag;
  du::dumma::DUFragment<du::dumma::accumulator, kDummaM, kDummaN, kDummaK,
                        int>
      acc_frag;
  du::dumma::du_fill_fragment(acc_frag, 0);

  // Zero-barrier LDS-only K loop: every fragment read comes from LDS, so no
  // global latency sits on the MMA critical path after the single staging
  // barrier above.
  const int lane = static_cast<int>(threadIdx.x) & 63;
  for (int i = 0; i < steps; ++i) {
    du::dumma::du_load_matrix_sync(a_frag, &a_s[0][i * kDummaK], kAStride);
    // One 8-B ds_read_b64 per B fragment (col_major: each lane's 8 fragment
    // bytes are consecutive in the n-major tile) instead of 8 ds_read_u8 at
    // the 20-B k-major pitch.
    load_b_frag_packed(b_frag, &b_s[0][i * kDummaK], kBStride, lane);
    du::dumma::du_mma_sync(acc_frag, a_frag, b_frag, acc_frag);
  }

  // Private int32 partial plane: same slice-major layout as the direct
  // kernel (plane = slice*num_tiles + tile).
  const int num_tiles = static_cast<int>(gridDim.x);
  du::dumma::du_store_matrix_sync(
      partials + (slice * num_tiles + tile) * (kDummaM * kDummaN),
      acc_frag,
      kDummaN,
      du::dumma::mem_row_major);
}

// Template launch helper for the staged partial kernel (one wavefront per
// block, grid = (num_tiles, SPLIT_K)); only instantiated for the SPLIT_K
// values whose longest slice fits the 64 KiB LDS (6, 8, 9, 12, 16).
template <int SPLIT_K>
void launch_slicestage_partial(
    const int8_t* a,
    const int8_t* b,
    int32_t* partials,
    int n,
    int k,
    hipStream_t stream,
    int num_tiles) {
  hipLaunchKernelGGL(
      HIP_KERNEL_NAME(w8a8_dumma_m16_slicestage_partial_kernel<SPLIT_K>),
      dim3(static_cast<unsigned>(num_tiles), static_cast<unsigned>(SPLIT_K)),
      dim3(static_cast<unsigned>(kDummaThreads)),
      0,
      stream,
      a,
      b,
      partials,
      n,
      k);
}

// ---------------------------------------------------------------------------
// Combine + scale kernel (M == 16): one 16x16 tile per block, one thread per
// output element.  Exact int32 sum of the SPLIT_K workspace partials
// (ascending split order), then x_scale/weight_scale scaling and the bf16
// (RN-even) store.  No barrier is needed: every thread reads only its own
// 16x16 element's S partial values (written by the previous kernel on the
// same stream).
// ---------------------------------------------------------------------------
__global__ __launch_bounds__(kCombineThreads) void
w8a8_dumma_m16_combine_scale_kernel(
    const int32_t* __restrict__ partials,
    const float* __restrict__ x_scale,
    const float* __restrict__ weight_scale,
    hip_bfloat16* __restrict__ out,
    int split_k,
    int num_tiles,
    int n) {
  const int tile = static_cast<int>(blockIdx.x);
  const int linear = static_cast<int>(threadIdx.x);  // 0..255
  const int row = linear >> 4;
  const int col = linear & 15;
  const int out_col = tile * kDummaN + col;

  int32_t sum = 0;
  for (int s = 0; s < split_k; ++s) {
    sum += partials[(s * num_tiles + tile) * (kDummaM * kDummaN) + linear];
  }

  const float scaled =
      static_cast<float>(sum) * x_scale[row] * weight_scale[out_col];
  out[row * n + out_col] = __float2bfloat16(scaled);
}

// ---------------------------------------------------------------------------
// Weight pack op (outside the timed region).
// ---------------------------------------------------------------------------
// For every (K,N) except the exact pair (k,n)==(4096,1280) the packed weight
// stays a contiguous [K, N] int8 buffer with an [N, 1] fp32 scale,
// byte-identical to the raw inputs.  For (k,n)==(4096,1280) (iteration 11)
// launch_pack_w8a8_weight transposes the raw [K,N] weight to an [N,K]
// n-major packed layout (element (kk,nn) -> dst[nn*k + kk]) so DUMMA
// col_major B fragments read 8 consecutive bytes per lane (one ds_read_b64
// instead of 8 ds_read_u8); the staged/direct DUMMA kernels and the scalar
// fallback (for the paired M=2 validation shape) all decode that layout.
__global__ __launch_bounds__(kCopyBlockThreads) void
w8a8_identity_copy_i8_kernel(
    const int8_t* __restrict__ src,
    int8_t* __restrict__ dst,
    int64_t count) {
  const int64_t linear =
      static_cast<int64_t>(blockIdx.x) * kCopyBlockThreads + threadIdx.x;
  if (linear < count) {
    dst[linear] = src[linear];
  }
}

__global__ __launch_bounds__(kCopyBlockThreads) void
w8a8_identity_copy_f32_kernel(
    const float* __restrict__ src,
    float* __restrict__ dst,
    int64_t count) {
  const int64_t linear =
      static_cast<int64_t>(blockIdx.x) * kCopyBlockThreads + threadIdx.x;
  if (linear < count) {
    dst[linear] = src[linear];
  }
}

// Iteration 11 one-time pack for the exact pair (k,n)==(4096,1280):
// transpose the raw [K,N] int8 weight to [N,K] n-major (element (kk,nn) ->
// dst[nn*k + kk]).  Runs out of the timed region and out of Graph capture;
// performance is irrelevant (single weight prep per layer load).
__global__ __launch_bounds__(kCopyBlockThreads) void
w8a8_transpose_i8_kernel(
    const int8_t* __restrict__ src,
    int8_t* __restrict__ dst,
    int k,
    int n) {
  const int64_t linear =
      static_cast<int64_t>(blockIdx.x) * kCopyBlockThreads + threadIdx.x;
  const int64_t total = static_cast<int64_t>(k) * n;
  if (linear >= total) {
    return;
  }
  const int kk = static_cast<int>(linear / n);
  const int nn = static_cast<int>(linear - static_cast<int64_t>(kk) * n);
  dst[static_cast<int64_t>(nn) * k + kk] = src[linear];
}

}  // namespace

// ---------------------------------------------------------------------------
// Stable host launch symbols (declared in csrc/bindings.cpp).
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
  auto* out_bf16 = static_cast<hip_bfloat16*>(out);

  // Exact-shape guard for the assigned shape hy3_tp8_qkv_proj_m16
  // (M=16, N=1280, K=4096).  The DUMMA specialization can never capture
  // other shapes (in particular the paired M=2 shape with the same (N, K)).
  if (m == 16 && n == 1280 && k == 4096) {
    const int num_tiles = n / kDummaN;  // 1280 / 16 = 80
    const int64_t plane_bytes =
        static_cast<int64_t>(num_tiles) * kDummaM * kDummaN *
        static_cast<int64_t>(sizeof(int32_t));  // 80 * 256 * 4 = 81,920

    // Split-K selection: default 16 (the iteration-8 staged geometry: 1,280
    // one-wave blocks = 10.67 blocks/CU; per-block LDS drops from 12,736 B
    // (5 resident/CU, the binding occupancy limiter at S=12 per PMC: grid
    // 960 = 8 waves/CU available, VGPR 24 / 0 scratch not limiting) to
    // 9,280 B = 7 resident/CU, with uniform 256-K slices = exactly 8 MMAs
    // per block; A/B HBM bytes unchanged).  The ZTH_W8A8_QKV_SPLIT_K
    // environment variable selects any trusted probe candidate
    // {2,3,4,5,6,8,9,12,16} so the sweep can be measured without source
    // edits (S=6 -> 480 blocks = 4/CU, S=9 -> 720 = 6/CU, S=12 -> 960 = 8/CU,
    // S=16 -> 1,280 = 10.67/CU, S=3 -> 240 direct-load blocks, ...).  The
    // read happens once per launch call on the host, never during Graph
    // replay, so capture/replay determinism is preserved.
    int split_k = 16;
    if (const char* env = std::getenv("ZTH_W8A8_QKV_SPLIT_K")) {
      char* end = nullptr;
      const long parsed = std::strtol(env, &end, 10);
      if (end != env && *end == '\0') {
        for (int t : kTrustedSplitK) {
          if (static_cast<int>(parsed) == t) {
            split_k = t;
            break;
          }
        }
      }
    }

    // Workspace fit: clamp to the largest trusted candidate that fits the
    // caller workspace (contract guarantees 16 planes = 1,310,720 B for this
    // shape, so the clamp never fires; kept defensive).
    int best_fit = 0;
    for (int t : kTrustedSplitK) {
      if (static_cast<int64_t>(t) * plane_bytes <= workspace_bytes) {
        best_fit = t;
      }
    }
    if (best_fit == 0) {
      launch_scalar_gemm(a, b, x_scale, weight_scale, out_bf16, m, n, k,
                         stream);
      return;
    }
    if (split_k > best_fit) {
      split_k = best_fit;
    }

    // Int32 partial planes in the caller workspace, then the combine+scale
    // kernel, both on the caller's stream inside the timed Graph.
    // Iteration-7/8 staged kernels: S in {6,8,9,12,16} (their longest slice
    // fits the 64 KiB LDS; see w8a8_dumma_m16_slicestage_partial_kernel).
    // Iteration-3 direct-load kernels: S in {2,3,4,5} (their slices exceed
    // the staging capacity).  Both families produce the same slice-major
    // int32 planes, so the combine kernel is shared and unchanged.
    int32_t* partials = static_cast<int32_t*>(workspace);
    switch (split_k) {
      case 6:
        launch_slicestage_partial<6>(a, b, partials, n, k, stream,
                                     num_tiles);
        break;
      case 8:
        launch_slicestage_partial<8>(a, b, partials, n, k, stream,
                                     num_tiles);
        break;
      case 9:
        launch_slicestage_partial<9>(a, b, partials, n, k, stream,
                                     num_tiles);
        break;
      case 12:
        launch_slicestage_partial<12>(a, b, partials, n, k, stream,
                                      num_tiles);
        break;
      case 16:
        launch_slicestage_partial<16>(a, b, partials, n, k, stream,
                                      num_tiles);
        break;
      case 2:
        launch_splitk_partial<2>(a, b, partials, n, k, stream, num_tiles);
        break;
      case 3:
        launch_splitk_partial<3>(a, b, partials, n, k, stream, num_tiles);
        break;
      case 4:
        launch_splitk_partial<4>(a, b, partials, n, k, stream, num_tiles);
        break;
      case 5:
        launch_splitk_partial<5>(a, b, partials, n, k, stream, num_tiles);
        break;
      default:
        launch_scalar_gemm(a, b, x_scale, weight_scale, out_bf16, m, n, k,
                           stream);
        return;
    }

    hipLaunchKernelGGL(
        w8a8_dumma_m16_combine_scale_kernel,
        dim3(static_cast<unsigned>(num_tiles)),
        dim3(static_cast<unsigned>(kCombineThreads)),
        0,
        stream,
        partials,
        x_scale,
        weight_scale,
        out_bf16,
        split_k,
        num_tiles,
        n);
    return;
  }

  // Generic scalar fallback for every unmatched (m, n, k).
  launch_scalar_gemm(a, b, x_scale, weight_scale, out_bf16, m, n, k, stream);
}

extern "C" void launch_pack_w8a8_weight(
    const int8_t* raw_weight,
    const float* weight_scale,
    int8_t* packed_weight,
    float* packed_weight_scale,
    int k,
    int n,
    hipStream_t stream) {
  // Out-of-timed-region weight prep: [N,K] n-major transpose pack for the
  // exact pair (k,n)==(4096,1280) (iteration 11), identity copy otherwise.
  const int64_t weight_count = static_cast<int64_t>(k) * n;
  const unsigned weight_grid = static_cast<unsigned>(
      (weight_count + kCopyBlockThreads - 1) / kCopyBlockThreads);
  if (k == kPackedK && n == kPackedN) {
    // Iteration 11: [N,K] n-major transpose pack for the exact pair so DUMMA
    // col_major B fragments read 8 consecutive bytes per lane (one
    // ds_read_b64 instead of 8 ds_read_u8).
    hipLaunchKernelGGL(
        w8a8_transpose_i8_kernel,
        dim3(weight_grid),
        dim3(kCopyBlockThreads),
        0,
        stream,
        raw_weight,
        packed_weight,
        k,
        n);
  } else {
    hipLaunchKernelGGL(
        w8a8_identity_copy_i8_kernel,
        dim3(weight_grid),
        dim3(kCopyBlockThreads),
        0,
        stream,
        raw_weight,
        packed_weight,
        weight_count);
  }

  const unsigned scale_grid = static_cast<unsigned>(
      (static_cast<int64_t>(n) + kCopyBlockThreads - 1) / kCopyBlockThreads);
  hipLaunchKernelGGL(
      w8a8_identity_copy_f32_kernel,
      dim3(scale_grid),
      dim3(kCopyBlockThreads),
      0,
      stream,
      weight_scale,
      packed_weight_scale,
      static_cast<int64_t>(n));
}
// @@end
