// @@variant shape=hy3_tp4_shared_gate_up_proj_m16 commit=f52225ed98ffd619fa1318d3eca7e56addd32f82 added=2026-08-26
//   median_us=11.31 p90_us=11.32
//   source=hy3-dsh-tp4-m16-1-7f1fb1d1
// MetaInfer W8A8 INT8 GEMM for gfx928 (K500SM_AI).
//
// worker_2  — physical GPU 2
//   assigned shape: hy3_tp4_shared_gate_up_proj_m16 (M=16, N=768, K=4096)
//
// Iteration 1 = establish the minimal 16x16x32 DUMMA tile:
//   * one 64-lane wavefront per block, one 16x16 output tile per wave,
//     explicit int32 accumulation (du_fill_fragment / du_load_matrix_sync /
//     du_mma_sync / du_store_matrix_sync from <du_mma.h>);
//   * A/B fragments are loaded straight from global memory every K step, so
//     there is no LDS staging, no double buffering, and no cross-wave
//     barrier; the accumulator fragment is drained through LDS in a
//     coalesced epilogue with the two float scales and a bf16 store;
//   * a scalar generic fallback stays reachable for every unmatched (m,n,k)
//     (including the paired M=2 API shape with the same (N,K)); the M=16
//     DUMMA path is guarded by the exact (m,n,k) triple.
//
// Iteration 2 = architecture round on grid parallelism. N=768 limits the
// independent 16x16 output tiles to 48, so no {1,2,4} waves x {1,2,4} adjacent
// N-tile geometry can cover all 120 CUs (max grid_blocks=48, max active CUs
// with one tile per wave = 48). Within that space this round packs 2
// wavefronts per block, each owning 2 adjacent 16-wide N tiles (a 16x32
// column span) that share one A fragment: grid = 768/64 = 12 blocks, 128
// threads, 2 waves per CU on 2 SIMDs, 2 independent accumulator chains per
// SIMD. Both latency-hiding mechanisms are engaged without any split-K:
//   * hardware: two co-resident wavefronts on the same CU issue their own
//     loads/MMACs while the sibling wave stalls on its fragment loads;
//   * compiler: two independent per-wave accumulator chains plus the shared
//     A fragment (1 A load + 2 B loads + 2 MMACs per K step) give the
//     scheduler real ILP, unlike iteration 1 where a single dependent chain
//     degraded to serialized global_load_ubyte -> s_waitcnt vmcnt chain ->
//     one v_mmac per step.
// A/B fragments stay direct-from-global (no LDS staging, no cross-wave
// barrier in the K loop); only the accumulator drain reuses the iteration-1
// LDS epilogue (4 KiB/block, one block-wide __syncthreads), so per-element
// int32 accumulation order (k0 ascending) is bit-identical to iteration 1.
// Measured result: median 215.7us (24 waves total on 12 CUs) vs 250.2us for
// grid=48 one-wave blocks; per-step cost stays ~1.7us/wave at K=4096 (128
// DUMMA steps), i.e. the kernel remains latency-bound with only 24 waves in
// flight on a 120-CU device, far from the 73.3us Triton Graph baseline.
//
// Iteration 3 = architecture round: split-K with combine. The unsplit grid
// (12 blocks) is below the two-blocks-per-CU latency-hiding target, so the
// K=4096 loop (128 DUMMA steps) is partitioned across S runtime-selectable
// splits:
//   * grid = 12*S blocks; blockIdx.x encodes (split = x/12, N-group = x%12);
//     every split's K range is an exact multiple of kDummaK via step-based
//     boundaries [(s*128)/S, ((s+1)*128)/S), so each block keeps the exact
//     iteration-2 inner loop (2 wavefronts, shared A fragment, 2 independent
//     accumulator chains, no cross-wave barrier in the K loop) over a shorter
//     K slice;
//   * each block drains its four int32 accumulator tiles through LDS into
//     caller-workspace plane `split` (row-major [16,n] int32, plane stride
//     16*n); accumulation stays k0 ascending inside the slice and summing the
//     S planes in split order reproduces the full-K int32 dot exactly (int32
//     addition is exact and |full dot| <= 4096*127*127 ~= 6.6e7 << 2^31), so
//     results stay bit-identical to iterations 1-2;
//   * a tiny combine kernel (96 blocks x 128 threads) then sums the S planes,
//     applies x_scale*weight_scale and stores bf16. Both kernels launch on
//     the caller stream inside the timed Graph, so the combine cost is part
//     of the measured median/P90.
// Split count is selected at launch by METAINFER_W8A8_SPLIT_K (default 16 as
// of iteration 4, legal 1..16, clamped to workspace planes and K/32 stage
// count). Trusted occupancy-probe candidates [2,3,5,7,8,10,15] give grids
// 24/36/60/84/96/120/180: S=10 is CU-aligned (120 blocks, exactly one per
// CU), S=5 lands on 60 CUs, S=15 oversubscribes to 180.
//
// Iteration 4 = architecture/pipeline round: CU-filling default grid plus
// bounded LDS prefetch. The control plane flags the S=2 default (24 blocks
// on 120 CUs) as far below the two-blocks-per-CU latency-hiding target, and
// the per-step window is load-latency bound (28 global_load_ubyte -> waitcnt
// chain -> 2 v_mmac per step; ~1.7us per step at 2 waves/CU). This round
//   * moves the default split to 16 -> grid 192 blocks x 128 threads = 384
//     waves (1.6 blocks/CU, all 120 CUs covered; combine cost of summing 16
//     int32 planes is inside the timed Graph);
//   * stages each block's whole K slice in LDS once (A 16x256 = 4 KiB, B
//     256x64 = 16 KiB, 16 B/thread vector loads), then the 8-step K loop
//     reads fragments from LDS only (du_load_matrix_sync on __shared__
//     pointers emits ds_read): zero global loads and zero barriers in the
//     loop. Per block LDS = 4 KiB drain + 20 KiB staged = 24 KiB, so two
//     blocks per CU still fit. The staged copy keeps the raw row-major
//     layout with the same ldm semantics, so every fragment holds the exact
//     iteration-3 bytes and int32 accumulation order (k0 ascending inside
//     the split, splits summed in order by combine) is bit-identical.
//   * A bytes are reused 2x per K step (one fragment feeds the wave's two
//     16-wide N-tiles); B bytes are consumed once per step from the staged
//     copy. Splits S < 16 keep the iteration-3 direct-global loop untouched.
// Falsifiable: if staging plus 384 waves does not cut the 112.6us median
// toward the ~20-40us band (and below the 73.3us Triton baseline), the
// binding cost is not per-step global latency and the next round must attack
// the staging prologue / combine / launch overheads instead.
// Measured result (accepted): median 15.65us / 206.96 GB/s / 6.43 TOPS at
// S=16. The exact-source ISA shows the remaining binding cost in the LDS-only
// K loop: per step per wavefront the row-major B fragments degrade to 16
// ds_read_u8 (8 per fragment) at stride-64 addresses (offsets 0,64,128,...)
// with per-read lgkmcnt waits, and PMC reports 328,704 LDS bank conflicts on
// 60,672 LDS instructions (~5.4 conflicts/instr) plus 88,787 LDS waits - the
// same B-fragment-aliasing / byte-reassembly signature that the gate_up
// M=4096 lineage on this worker fixed with an [N,K] pack + col_major
// fragments + an 8-consecutive-byte fragment loader (one ds_read2_b64 per
// fragment; VALU 16.08M -> 6.25M, accepted iter 13).
//
// Iteration 5 = packed-weight round. One focused layout change plus its
// minimal load-path rewiring:
//   * launch_pack_w8a8_weight transposes [K,N] -> [N,K] for the exact gate_up
//     pair (k,n)==(4096,768) (one-time, out of the timed region and out of
//     Graph capture; every other (K,N) keeps the identity copy);
//   * the B fragment is declared col_major: du_mma.h's matrix_b col_major
//     loader maps lane l to 8 CONSECUTIVE bytes (x[i] = p[(l&15)*ldm +
//     ((l>>4)<<3) + i]), so n-major storage makes each lane's fragment bytes
//     contiguous - the staged K loop replaces the 16 ds_read_u8 per step per
//     wavefront with one 8-byte load per fragment (ds_read_b64); the direct
//     S<16 path reads the packed global layout through the library col_major
//     loader;
//   * staged slices use a bank-skewed row stride 272 = 256+16 (16 B aligned
//     so the ds_write_b128 staging stays aligned; 272 % 8 == 0 so every
//     fragment b64 read is 8 B aligned), cutting the stride-256 8-way bank
//     aliasing to ~2-way on the remaining reads;
//   * the scalar generic fallback decodes the packed [N,K] layout whenever
//     (k,n)==(4096,768) (this covers the paired M=2 validation shape with the
//     same (N,K)); every other shape keeps the raw [K,N] decode.
// Per-element int32 accumulation order is unchanged (k0 ascending inside each
// split, splits summed in split order by the combine kernel), and the
// col_major loader fills the same fragment slots with the same logical (k,n)
// bytes, so results stay bit-identical to iteration 4.
// Expected: LDS instructions in the K loop drop 17 -> 3 per step per
// wavefront (A: 1 ds_read2_b32, B: 2 ds_read_b64), LDS waits and conflicts
// drop roughly 4-6x, so median should move from 15.65us toward ~10-13us.
// Falsifiable: if the median does not land below ~14us, LDS issue/latency is
// not the binding cost and the next round must attack the staging prologue /
// combine / launch overhead or the grid (finer one-wave zero-barrier grid or
// split-K probes [2,3,5,7,8,10,15], which now measure the packed direct
// path plus combine cost via METAINFER_W8A8_SPLIT_K).
//
// launch_pack_w8a8_weight is an out-of-timed-region pack op: an [N,K]
// transpose for the exact (k,n)==(4096,768) gate_up weight, identity
// device-to-device copy for every other (K,N).
//
// Iteration 8 = combine-round: specialize the S=16 combine for the default
// split count. The accepted S=16 operator (median 13.34us / 242.85 GB/s /
// 7.55 TOPS) spends ~22-25% of its time in the combine kernel (3.36us of the
// 15.04us profiled operator aggregate; GEMM kernel 11.68us). The exact-source
// ISA of the combine shows why: each thread's 16 plane loads (one scalar
// 4-B global_load_dword per plane, 3,456 vmem_read instructions per replay)
// are issued in small groups of ~4 with the per-plane address chain
// (v_mad_u64_u32 reusing the previous address register) and vmcnt waits
// serializing them, so a thread has only ~4-6 HBM round trips in flight.
// With 96 blocks x 128 threads = 192 waves and ~1.6 waves/CU the combine is
// latency-bound on those serialized round trips. This round makes ONE focused
// change: when split_k == 16 (the compile-time-known default), the combine
// loads all 16 planes into a register array with a fully unrolled loop BEFORE
// summing them, so the compiler schedules 16 independent loads in flight and
// the ascending int32 sum then consumes them. Per-element accumulation order
// (planes s=0..15 ascending, exactly the generic loop's order) is unchanged,
// so results stay bit-identical to iteration 5; the generic loop and every
// other split count are untouched. Falsifiable: if the median does not move
// by the combine's ~2-2.5us share (13.34us -> ~11-12us), the combine was not
// latency-bound on load MLP and iteration 9 must attack the GEMM kernel's
// remaining LDS conflicts / A-fragment repack VALU in the staged K loop.
//
// Include order is load-bearing on this DTK: HIP runtime first, then
// hip_bfloat16.h, then du_mma.h (not self-contained before the HIP headers).

#include <hip/hip_runtime.h>
#include <hip/hip_bfloat16.h>
#include <du_mma.h>

#include <cstdint>
#include <cstdlib>

namespace {

// gfx928 wavefront is 64 lanes; blockDim must stay a multiple of 64.
constexpr int kScalarBlockThreads = 128;
constexpr int kCopyBlockThreads = 256;

// Exact (k,n) pair that launch_pack_w8a8_weight transposes to [N,K] and that
// the scalar fallback decodes as n-major. Declared before the scalar kernel
// below (which references them for the exact gate_up pair).
constexpr int kPackedK = 4096;
constexpr int kPackedN = 768;

// Scalar INT8 dot-product GEMM, one thread per output element.
//   out[m, n] = bf16( (sum_k a[m,k] * b[k,n]) * x_scale[m] * weight_scale[n] )
// Accumulation order is exactly k0..K-1 in int32 (identical for every output
// element), so results are bit-exact against the float/int32 reference.
__global__ void w8a8_gemm_scalar_kernel(
    const int8_t* __restrict__ a,
    const int8_t* __restrict__ b,
    const float* __restrict__ x_scale,
    const float* __restrict__ weight_scale,
    hip_bfloat16* __restrict__ out,
    int m,
    int n,
    int k) {
  const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total = static_cast<int64_t>(m) * n;
  if (idx >= total) {
    return;
  }
  // Adjacent lanes map to adjacent columns in the fastest-changing N
  // dimension, so output stores (and A rows) stay coalesced.
  const int row = static_cast<int>(idx / n);
  const int col = static_cast<int>(idx - static_cast<int64_t>(row) * n);

  const int8_t* __restrict__ a_row = a + static_cast<int64_t>(row) * k;
  // Iteration 5: the exact gate_up pair (k,n)==(4096,768) is packed [N,K]
  // (n-major) by launch_pack_w8a8_weight; the fallback decodes that layout so
  // it stays correct against the packed buffer (paired M=2 validation shape
  // included). Every other (K,N) keeps the raw [K,N] row-major decode.
  const bool packed = (k == kPackedK && n == kPackedN);
  const int8_t* __restrict__ b_col =
      b + (packed ? static_cast<int64_t>(col) * k : col);
  const int64_t b_stride = packed ? 1 : static_cast<int64_t>(n);

  int32_t acc = 0;
  for (int kk = 0; kk < k; ++kk) {
    acc += static_cast<int32_t>(a_row[kk]) *
           static_cast<int32_t>(b_col[static_cast<int64_t>(kk) * b_stride]);
  }

  const float scaled =
      static_cast<float>(acc) * x_scale[row] * weight_scale[col];
  out[idx] = __float2bfloat16(scaled);
}

// ---------------------------------------------------------------------------
// Minimal gfx928 DUMMA INT8 path (assigned shape M=16, N=768, K=4096).
// Tile: 16x16x32, int32 accumulation, one 64-lane wavefront per block.
// ---------------------------------------------------------------------------

// gfx928 DUMMA INT8 m16n16k32 tile constants.
constexpr int kDummaM = 16;
constexpr int kDummaN = 16;
constexpr int kDummaK = 32;
// Iteration 2 launch geometry: 2 wavefronts per block, each owning 2
// adjacent 16-wide N tiles (a 16x32 column span) that share one A fragment.
// Block tile = 16 x 64 columns, grid = 768/64 = 12 blocks, 128 threads.
constexpr int kDummaWavesPerBlock = 2;
constexpr int kDummaTilesPerWave = 2;
constexpr int kDummaBlockThreads = kDummaWavesPerBlock * 64;      // 128
constexpr int kDummaWaveN = kDummaTilesPerWave * kDummaN;         // 32 cols/wave
constexpr int kDummaBlockN = kDummaWavesPerBlock * kDummaWaveN;   // 64 cols/block

// Iteration 3 split-K: runtime-selectable split count (1..16, matching the
// caller workspace contract WORKSPACE_MAX_SPLIT_K=16), default 16 as of
// iteration 4 (CU-filling grid + LDS-staged slice, see below). The
// combine+scale kernel uses 128 threads (multiple of wavefront 64).
constexpr int kDummaSplitDefault = 16;
constexpr int kDummaSplitMax = 16;
constexpr int kDummaCombineThreads = 128;

// Iteration 4 bounded LDS prefetch. At split_k == kDummaSplitMax every split's
// K slice is exactly 128/16 = 8 DUMMA steps = 256 rows, so the kernel stages
// the whole slice into LDS once per block (A 16x256 = 4 KiB, B 256x64 =
// 16 KiB) and the K loop then reads its fragments from LDS: no global loads
// and no barrier inside the loop (zero-barrier K loop; one __syncthreads
// after the staging copy, one in the drain epilogue). LDS per block = 4 KiB
// accumulator drain + 20 KiB staged = 24 KiB, keeping two blocks per CU
// feasible (49 KiB <= 64 KiB per CU) so the 192-block S=16 grid still covers
// all 120 CUs. Slices larger than this (S < 16) keep the iteration-3
// direct-global path.
constexpr int kStagedSliceRows = 256;

// Iteration 5 packed-weight layout. The gate_up weight (k,n)==(4096,768) is
// packed once (out of the timed region) as [N,K] n-major; B fragments are
// declared col_major so each lane's 8 fragment bytes are consecutive in
// memory and load with one ds_read_b64 instead of 8 ds_read_u8. Staged rows
// use a bank-skewed stride 272 = 256+16: 16 B aligned (ds_write_b128 staging
// stays aligned, 272 = 17*16), 8 B aligned (every fragment b64 read stays
// aligned), and not a multiple of 128 B, so rows no longer alias onto one LDS
// bank phase (stride 256 put all 16 rows on the same phase -> 8-way fragment
// conflicts).
constexpr int kStagedRowStride = 272;

// Iteration 5 packed B-fragment loader. With n-major [N,K] storage and a
// col_major fragment, du_mma.h's matrix_b col_major address rule is
// x[i] = p[(lane&15)*ldm + ((lane>>4)<<3) + i] - lane l's 8 fragment bytes
// are consecutive in memory, so one 8-byte LDS read per fragment replaces the
// 8 ds_read_u8 at stride-64 addresses that the row-major loader emits (and
// du_mma_sync consumes the fragment as one 64-bit value per lane, so writing
// the packed bytes straight into x[0..7] is exactly the library's own
// storage convention). ldm is the n-major row stride in bytes; the pointer
// must be 8 B aligned (guaranteed: staged base is __align__(256), ldm % 8 ==
// 0, tile offsets are multiples of 16, local_k is a multiple of 32).
__device__ __forceinline__ void load_b_frag_packed(
    du::dumma::DUFragment<du::dumma::matrix_b, kDummaM, kDummaN, kDummaK,
                          signed char, du::dumma::col_major>& b_frag,
    const signed char* p, int ldm, int lane) {
  const int row = lane & 15;
  const int col = (lane >> 4) << 3;
  const uint64_t v =
      *reinterpret_cast<const uint64_t*>(p + static_cast<int64_t>(row) * ldm + col);
  *reinterpret_cast<uint64_t*>(b_frag.x) = v;
}

// M=16 W8A8 GEMM with the minimal 16x16x32 DUMMA tile and runtime split-K.
// Block layout is the iteration-2 geometry (12 N-groups of 64 columns, 2
// wavefronts per block, 2 adjacent tiles per wave sharing one A fragment);
// blockIdx.x encodes (split, n_group) = (blockIdx.x / 12, blockIdx.x % 12).
// Each block computes its split's slice of the K loop (kDummaK-aligned) and
// drains the raw int32 accumulator tiles into workspace plane `split`; the
// combine kernel later sums the planes and applies the two float scales.
__global__ __launch_bounds__(kDummaBlockThreads)
void w8a8_gemm_m16_dumma_kernel(
    const int8_t* __restrict__ a,
    const int8_t* __restrict__ b,
    int32_t* __restrict__ partials,
    int n,
    int k,
    int split_k) {
  const int wave = static_cast<int>(threadIdx.x) >> 6;
  const int lane = static_cast<int>(threadIdx.x) & 63;
  const int n_groups = n / kDummaBlockN;
  const int split = static_cast<int>(blockIdx.x) / n_groups;
  const int block_n_base =
      (static_cast<int>(blockIdx.x) % n_groups) * kDummaBlockN;
  const int n_base = block_n_base + wave * kDummaWaveN;

  // Split s owns DUMMA steps [(s*total_steps)/S, ((s+1)*total_steps)/S), so
  // every split's K slice is an exact multiple of kDummaK for any legal S.
  const int total_steps = k / kDummaK;
  const int k_lo = ((split * total_steps) / split_k) * kDummaK;
  const int k_hi = (((split + 1) * total_steps) / split_k) * kDummaK;
  const int slice_rows = k_hi - k_lo;  // bytes of K owned by this block

  __shared__ __align__(16)
      int32_t acc_tile[kDummaWavesPerBlock][kDummaTilesPerWave]
                      [kDummaM * kDummaN];
  // Iteration 4/5 staged slices: A is [16, slice] row-major (reused by both
  // N-tiles of each wavefront across every K step) and B is [64, slice]
  // n-major (the block's 64-column span of the packed [N,K] weight). Both use
  // the iteration-5 bank-skewed row stride 272 so fragment reads from LDS
  // stay 8 B aligned and rows do not alias onto one bank phase.
  __shared__ __align__(256) signed char lds_a[kDummaM * kStagedRowStride];
  __shared__ __align__(256) signed char lds_b[kDummaBlockN * kStagedRowStride];

  du::dumma::DUFragment<du::dumma::matrix_a, kDummaM, kDummaN, kDummaK,
                        signed char, du::dumma::row_major>
      a_frag;
  // Iteration 5: col_major B fragments match the packed [N,K] storage so each
  // lane's 8 fragment bytes are consecutive (one ds_read_b64 per fragment in
  // the staged loop instead of 8 ds_read_u8).
  du::dumma::DUFragment<du::dumma::matrix_b, kDummaM, kDummaN, kDummaK,
                        signed char, du::dumma::col_major>
      b_frag[kDummaTilesPerWave];
  du::dumma::DUFragment<du::dumma::accumulator, kDummaM, kDummaN, kDummaK, int>
      acc_frag[kDummaTilesPerWave];

#pragma unroll
  for (int t = 0; t < kDummaTilesPerWave; ++t) {
    du::dumma::du_fill_fragment(acc_frag[t], 0);
  }

  if (slice_rows == kStagedSliceRows) {
    // ---- Iteration 4 staged path (split_k == 16). Cooperative global->LDS
    // prefetch of the whole split slice: 128 threads issue wide 16 B vector
    // loads (A: 2 int4/thread, B: 8 int4/thread), one block-wide barrier,
    // then the K loop is LDS-only: du_load_matrix_sync emits ds_read instead
    // of global_load, so no HBM latency sits on the MMAC critical path and
    // the loop is barrier-free.
    const int4* __restrict__ a4 = reinterpret_cast<const int4*>(a);
    const int4* __restrict__ b4 = reinterpret_cast<const int4*>(b);
    int4* __restrict__ lds_a4 = reinterpret_cast<int4*>(lds_a);
    int4* __restrict__ lds_b4 = reinterpret_cast<int4*>(lds_b);
    const int tid = static_cast<int>(threadIdx.x);
    // A slice: 16 rows x kStagedSliceRows bytes = 256 int4s; each thread
    // stages two int4s. All loads issue before any LDS store so the HBM
    // round trips pipeline instead of serializing behind per-load waitcnts.
    // int4 = 16 bytes, so flat int4 indices divide byte offsets by 16
    // (>>4): row stride k>>4 = 256 int4s (K=4096 bytes/row), slice column
    // offset k_lo>>4, and one int4 per a_c (16 int4s per 256 B row). The LDS
    // row stride is kStagedRowStride>>4 = 17 int4s (bank-skewed padding).
    const int a_r = tid >> 4;  // A row of the first int4 (0..7)
    const int a_c = tid & 15;  // 16 int4s per 256 B row
    int4 a0 = a4[a_r * (k >> 4) + (k_lo >> 4) + a_c];
    int4 a1 = a4[(a_r + 8) * (k >> 4) + (k_lo >> 4) + a_c];
    // B slice (iteration 5): the packed [N,K] weight is n-major, so the
    // block's 64 columns are 64 n-rows of 256 slice bytes = 1024 int4s; each
    // thread stages eight. b4 row stride = k>>4 int4s (K=4096 bytes per
    // n-row), slice column offset k_lo>>4, LDS row stride 17 int4s.
    const int b_nr = tid >> 4;  // first n-row of this thread (0..7)
    const int b_k16 = tid & 15;  // 16 int4s per 256 B slice row
    int4 b0 = b4[(static_cast<int64_t>(block_n_base) + b_nr + 0) * (k >> 4) +
                 (k_lo >> 4) + b_k16];
    int4 b1 = b4[(static_cast<int64_t>(block_n_base) + b_nr + 8) * (k >> 4) +
                 (k_lo >> 4) + b_k16];
    int4 b2 = b4[(static_cast<int64_t>(block_n_base) + b_nr + 16) * (k >> 4) +
                 (k_lo >> 4) + b_k16];
    int4 b3 = b4[(static_cast<int64_t>(block_n_base) + b_nr + 24) * (k >> 4) +
                 (k_lo >> 4) + b_k16];
    int4 b4v = b4[(static_cast<int64_t>(block_n_base) + b_nr + 32) * (k >> 4) +
                  (k_lo >> 4) + b_k16];
    int4 b5 = b4[(static_cast<int64_t>(block_n_base) + b_nr + 40) * (k >> 4) +
                 (k_lo >> 4) + b_k16];
    int4 b6 = b4[(static_cast<int64_t>(block_n_base) + b_nr + 48) * (k >> 4) +
                 (k_lo >> 4) + b_k16];
    int4 b7 = b4[(static_cast<int64_t>(block_n_base) + b_nr + 56) * (k >> 4) +
                 (k_lo >> 4) + b_k16];
    lds_a4[a_r * (kStagedRowStride >> 4) + a_c] = a0;
    lds_a4[(a_r + 8) * (kStagedRowStride >> 4) + a_c] = a1;
    lds_b4[(b_nr + 0) * (kStagedRowStride >> 4) + b_k16] = b0;
    lds_b4[(b_nr + 8) * (kStagedRowStride >> 4) + b_k16] = b1;
    lds_b4[(b_nr + 16) * (kStagedRowStride >> 4) + b_k16] = b2;
    lds_b4[(b_nr + 24) * (kStagedRowStride >> 4) + b_k16] = b3;
    lds_b4[(b_nr + 32) * (kStagedRowStride >> 4) + b_k16] = b4v;
    lds_b4[(b_nr + 40) * (kStagedRowStride >> 4) + b_k16] = b5;
    lds_b4[(b_nr + 48) * (kStagedRowStride >> 4) + b_k16] = b6;
    lds_b4[(b_nr + 56) * (kStagedRowStride >> 4) + b_k16] = b7;
    __syncthreads();

    // Fragment loads now come from LDS; ldm is the staged n-major row stride.
    // A bytes are reused 2x per step (one fragment feeds both N-tiles); B
    // bytes are consumed once per step from the staged copy, each B fragment
    // as one 8-byte ds_read_b64 via load_b_frag_packed (col_major address
    // rule). The trip count is compile-time (slice_rows == kStagedSliceRows),
    // so the scheduler can overlap the ds_read streams of consecutive steps.
#pragma unroll 2
    for (int step = 0; step < kStagedSliceRows / kDummaK; ++step) {
      const int local_k = step * kDummaK;
      du::dumma::du_load_matrix_sync(a_frag, lds_a + local_k,
                                     kStagedRowStride);
#pragma unroll
      for (int t = 0; t < kDummaTilesPerWave; ++t) {
        load_b_frag_packed(
            b_frag[t],
            lds_b + (wave * kDummaWaveN + t * kDummaN) * kStagedRowStride +
                local_k,
            kStagedRowStride, lane);
      }
#pragma unroll
      for (int t = 0; t < kDummaTilesPerWave; ++t) {
        du::dumma::du_mma_sync(acc_frag[t], a_frag, b_frag[t], acc_frag[t]);
      }
    }
  } else {
    // ---- Iteration 3 direct-global path (any other legal split count). One
    // A fragment serves both adjacent tiles of this wavefront.
#pragma unroll 2
    for (int k0 = k_lo; k0 < k_hi; k0 += kDummaK) {
      du::dumma::du_load_matrix_sync(a_frag, a + k0, k);
#pragma unroll
      for (int t = 0; t < kDummaTilesPerWave; ++t) {
        // Iteration 5: b is the packed [N,K] n-major weight for this shape,
        // so the tile's 16 columns live at (n_base + t*16)..+15 n-rows with
        // row stride k; the col_major loader addresses p[(l&15)*ldm +
        // ((l>>4)<<3) + i], i.e. n-major rows of k bytes.
        du::dumma::du_load_matrix_sync(
            b_frag[t],
            b + (static_cast<int64_t>(n_base) + t * kDummaN) * k + k0, k);
      }
#pragma unroll
      for (int t = 0; t < kDummaTilesPerWave; ++t) {
        du::dumma::du_mma_sync(acc_frag[t], a_frag, b_frag[t], acc_frag[t]);
      }
    }
  }

#pragma unroll
  for (int t = 0; t < kDummaTilesPerWave; ++t) {
    du::dumma::du_store_matrix_sync(acc_tile[wave][t], acc_frag[t], kDummaN,
                                    du::dumma::mem_row_major);
  }
  __syncthreads();

  // Drain raw int32 partials into workspace plane `split` (row-major
  // [16, n]); scaling and bf16 conversion happen in the combine kernel.
  int32_t* __restrict__ plane =
      partials + static_cast<int64_t>(split) * (kDummaM * n);
#pragma unroll
  for (int t = 0; t < kDummaTilesPerWave; ++t) {
#pragma unroll
    for (int linear = lane; linear < kDummaM * kDummaN; linear += 64) {
      const int row = linear >> 4;
      const int col = linear & 15;
      plane[static_cast<int64_t>(row) * n + n_base + t * kDummaN + col] =
          acc_tile[wave][t][linear];
    }
  }
}

// Combine + scale: out[m,n] = bf16( (sum_s partials[s][m,n]) * x_scale[m] *
// weight_scale[n] ). The S planes are summed in ascending split order; int32
// addition is exact and overflow-free (|full-K dot| <= 6.6e7 << 2^31), so the
// result is bit-identical to the single-chain k0-ascending accumulation of
// iterations 1-2. Runs on the caller stream right after the GEMM kernel, so
// its cost is inside the timed Graph.
__global__ void w8a8_gemm_m16_combine_kernel(
    const int32_t* __restrict__ partials,
    const float* __restrict__ x_scale,
    const float* __restrict__ weight_scale,
    hip_bfloat16* __restrict__ out,
    int n,
    int split_k) {
  const int total = kDummaM * n;
  const int idx = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= total) {
    return;
  }
  const int row = idx / n;
  const int col = idx - row * n;
  int32_t acc;
  if (split_k == kDummaSplitMax) {
    // Iteration 8 S=16 specialization: all 16 plane loads are independent, so
    // load them into a register array with a fully unrolled loop FIRST and
    // sum afterwards. The generic loop's exact-source ISA grouped ~4 scalar
    // 4-B loads per HBM round trip behind a serialized address chain (v_mad
    // reusing the previous address + vmcnt waits), leaving ~4-6 loads in
    // flight per thread; the unrolled array version puts 16 loads in flight
    // and collapses the combine to ~one memory round trip per wave. The sum
    // is taken in the same ascending s order as the generic loop, so every
    // element's int32 accumulation is bit-identical.
    int32_t v[kDummaSplitMax];
#pragma unroll
    for (int s = 0; s < kDummaSplitMax; ++s) {
      v[s] = partials[static_cast<int64_t>(s) * total + idx];
    }
    acc = 0;
#pragma unroll
    for (int s = 0; s < kDummaSplitMax; ++s) {
      acc += v[s];
    }
  } else {
    // Generic path: any other legal split count (runtime loop bound).
    acc = 0;
    for (int s = 0; s < split_k; ++s) {
      acc += partials[static_cast<int64_t>(s) * total + idx];
    }
  }
  const float scaled =
      static_cast<float>(acc) * x_scale[row] * weight_scale[col];
  out[idx] = __float2bfloat16(scaled);
}

// Identity device-to-device copies used by the bootstrap pack_weight op.
__global__ void w8a8_identity_copy_i8_kernel(
    const int8_t* __restrict__ src, int8_t* __restrict__ dst, int64_t numel) {
  const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < numel) {
    dst[idx] = src[idx];
  }
}

__global__ void w8a8_identity_copy_f32_kernel(
    const float* __restrict__ src, float* __restrict__ dst, int64_t numel) {
  const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < numel) {
    dst[idx] = src[idx];
  }
}

// Iteration 5 one-time pack for the exact gate_up pair (k,n)==(4096,768):
// transpose the raw [K,N] int8 weight to [N,K] n-major (element (kk,nn) ->
// dst[nn*k + kk]). Runs out of the timed region and out of Graph capture;
// performance is irrelevant (single weight prep per layer load).
__global__ void w8a8_transpose_i8_kernel(
    const int8_t* __restrict__ src, int8_t* __restrict__ dst, int k, int n) {
  const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total = static_cast<int64_t>(k) * n;
  if (idx >= total) {
    return;
  }
  const int kk = static_cast<int>(idx / n);
  const int nn = static_cast<int>(idx - static_cast<int64_t>(kk) * n);
  dst[static_cast<int64_t>(nn) * k + kk] = src[idx];
}

}  // namespace

// Timed-region GEMM entry point (called from gemm_out on PyTorch's current
// HIP stream). Must stay allocation/sync/compile-free and launch only on the
// caller-provided stream.
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
  // Shape-exact M=16 DUMMA specialization. Guarded by the exact (m,n,k)
  // triple so the paired M=2 API shape with the same (N,K) and every other
  // shape still reach the scalar fallback below.
  if (m == 16 && n == 768 && k == 4096) {
    // Iteration 3/4/5: runtime split-K selected by METAINFER_W8A8_SPLIT_K
    // (default 16, legal 1..16; clamped to workspace planes and K/32 stage
    // count). grid = 12*S blocks x 128 threads, block tile 16x64. b is the
    // [N,K] n-major packed weight (launch_pack_w8a8_weight for (4096,768));
    // at S=16 the GEMM kernel stages its 256-row K slice into LDS and runs an
    // LDS-only K loop with col_major B fragments (one ds_read_b64 per
    // fragment); at S<16 it keeps the direct-global loop on the packed
    // layout. Each block writes raw int32 partials to workspace plane
    // `split`; the combine kernel then sums the S planes, scales and stores
    // bf16. Both kernels launch on the caller stream inside the timed Graph.
    int split_k = kDummaSplitDefault;
    if (const char* env = getenv("METAINFER_W8A8_SPLIT_K")) {
      char* end = nullptr;
      const long parsed = strtol(env, &end, 10);
      if (end != env && parsed >= 1 && parsed <= kDummaSplitMax) {
        split_k = static_cast<int>(parsed);
      }
    }
    const int64_t plane_bytes = static_cast<int64_t>(16) * n * 4;
    const int max_by_ws = static_cast<int>(workspace_bytes / plane_bytes);
    const int max_by_k = k / kDummaK;
    if (split_k > max_by_ws) {
      split_k = max_by_ws;
    }
    if (split_k > max_by_k) {
      split_k = max_by_k;
    }
    if (split_k >= 1) {
      int32_t* partials = static_cast<int32_t*>(workspace);
      const int n_groups = n / kDummaBlockN;  // 12
      const dim3 grid(static_cast<unsigned>(n_groups * split_k));
      const dim3 block(kDummaBlockThreads);
      hipLaunchKernelGGL(w8a8_gemm_m16_dumma_kernel, grid, block, 0, stream,
                         a, b, partials, n, k, split_k);
      const int total_out = m * n;  // 12288
      const dim3 combine_grid(
          (total_out + kDummaCombineThreads - 1) / kDummaCombineThreads);
      const dim3 combine_block(kDummaCombineThreads);
      hipLaunchKernelGGL(w8a8_gemm_m16_combine_kernel, combine_grid,
                         combine_block, 0, stream, partials, x_scale,
                         weight_scale, static_cast<hip_bfloat16*>(out), n,
                         split_k);
      return;
    }
    // No workspace plane available: fall through to the scalar fallback.
  }

  // Scalar generic fallback: valid for every (m, n, k).
  const int64_t total = static_cast<int64_t>(m) * n;
  const dim3 grid(static_cast<unsigned>(
      (total + kScalarBlockThreads - 1) / kScalarBlockThreads));
  const dim3 block(kScalarBlockThreads);
  hipLaunchKernelGGL(w8a8_gemm_scalar_kernel, grid, block, 0, stream, a, b,
                     x_scale, weight_scale, static_cast<hip_bfloat16*>(out), m,
                     n, k);
}

// Out-of-timed-region weight packing. Iteration 5: for the exact gate_up
// pair (k,n)==(4096,768) the raw [K,N] int8 weight is transposed to an
// [N,K] n-major packed layout so DUMMA col_major B fragments read 8
// consecutive bytes per lane (one ds_read_b64 instead of 8 ds_read_u8); every
// other (K,N) keeps the identity device-to-device copy and the raw [K,N]
// decode. The [N] fp32 scales are always copied as-is. Runs outside the timed
// region and outside Graph capture.
extern "C" void launch_pack_w8a8_weight(
    const int8_t* raw_weight,
    const float* weight_scale,
    int8_t* packed_weight,
    float* packed_weight_scale,
    int k,
    int n,
    hipStream_t stream) {
  const int64_t weight_numel = static_cast<int64_t>(k) * n;
  const int64_t scale_numel = static_cast<int64_t>(n);
  if (weight_numel > 0) {
    const dim3 grid(static_cast<unsigned>(
        (weight_numel + kCopyBlockThreads - 1) / kCopyBlockThreads));
    const dim3 block(kCopyBlockThreads);
    if (k == kPackedK && n == kPackedN) {
      hipLaunchKernelGGL(w8a8_transpose_i8_kernel, grid, block, 0, stream,
                         raw_weight, packed_weight, k, n);
    } else {
      hipLaunchKernelGGL(w8a8_identity_copy_i8_kernel, grid, block, 0, stream,
                         raw_weight, packed_weight, weight_numel);
    }
  }
  if (scale_numel > 0) {
    const dim3 grid(static_cast<unsigned>(
        (scale_numel + kCopyBlockThreads - 1) / kCopyBlockThreads));
    const dim3 block(kCopyBlockThreads);
    hipLaunchKernelGGL(w8a8_identity_copy_f32_kernel, grid, block, 0, stream,
                       weight_scale, packed_weight_scale, scale_numel);
  }
}
// @@end
