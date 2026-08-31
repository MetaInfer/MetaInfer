// @@variant shape=minimax_tp8_shared_gate_up_proj_m16 commit=4358dea23da7468d3868ce258b971667a50c139e added=2026-08-28
//   median_us=17.11 p90_us=17.13
//   source=minimax-dsh-tp8-m16-1-78402260
// MetaInfer W8A8 INT8 GEMM - HIP implementation for Hygon K500SM_AI / gfx928.
//
// Iteration 1 (minimal DUMMA bootstrap): the assigned M=16 MiniMax TP8
// shared_gate_up_proj shape (16x768x6144) ran on a minimal 16x16x32 DUMMA
// tile with one 64-lane wavefront per output tile, explicit int32
// accumulation, direct row-major global fragment loads, and no LDS staging /
// no cross-wave barriers.  Measured 182.1 us median Graph replay (0.574x vs
// the 104.527 us Triton baseline): the one-wave grid of 48 blocks fills only
// 48 of 120 CUs (0.4 blocks/CU) and each block serially streams its full
// K=6144 loop, so the kernel is launch-geometry/latency bound, not ALU bound.
//
// Iteration 2 (architecture round, grid parallelism): keep the exact same
// zero-barrier one-wave direct-fragment tile, but split K into kSplitK = 5
// independent 32-aligned slices per output tile:
//   * grid = 48 N tiles x 5 K slices = 240 one-wave blocks = exactly 2
//     blocks per device CU (120 CUs) -- the two-blocks-per-CU latency-hiding
//     target of the control plane;
//   * each partial block publishes its int32 accumulator fragment to one
//     workspace plane (every (tile, slice) pair is written exactly once per
//     launch, so no workspace clear is needed);
//   * a 48-block 64-thread combine kernel sums the five planes in ascending
//     slice order (exact int32 order, bit-identical to the scalar reference),
//     applies x_scale[m] * packed_weight_scale[n], and stores bf16; both
//     kernels are enqueued on the caller stream so the combine cost is inside
//     the timed operator / Graph replay;
//   * every other shape, including the paired M=2 API shape with the same
//     (N, K) and the paired 16x6144x384 down_proj shape, keeps the scalar
//     generic fallback below;
//   * launch_pack_w8a8_weight stays an identity device-to-device copy
//     (generic fallback for every (K, N)).
//
// Iteration 3 (architecture round, occupancy probe): keep the exact
// iteration-2 architecture (zero-barrier one-wave direct-fragment tile,
// split-K partial planes + 48-block combine in the timed Graph) but raise
// kSplitK from 5 to 10, so the grid doubles from 240 to 480 one-wave blocks
// = exactly 4.0 blocks per device CU (120 CUs).  K = 6144 = 192 chunks of 32
// = 10*19 + 2, so slices 0..1 carry 20 chunks (640 elements) and slices 2..9
// carry 19 chunks (608 elements): every boundary stays a multiple of the
// DUMMA K step and the imbalance is <= 6%.  The workspace then holds 10
// int32 partial planes (491520 B), well inside the 786432 B (16-plane) the
// API allocates for this shape.  Hypothesis: at 2.0 blocks/CU the partial
// kernel is still global-load-latency bound (PMC: ~6% VALU issue
// utilization, 45.8 us perturbed partial vs 50.6 us total median), so
// doubling the independent resident one-wave streams per CU should hide more
// of that latency; the combine kernel grows from 5 to 10 plane-sums but
// stays a tiny 48-block 64-thread launch over L2-hot plane traffic.
// Measured: 45.47 us median Graph replay (2.30x), so occupancy scaling
// helped (~10%) but the direct-fragment load path remains the binding cost.
//
// Iteration 5 (load-path round): ISA on the accepted iteration-3 code object
// shows the library fragment loaders expand each k32 chunk into 16
// global_load_ubyte instructions per lane -- 8 contiguous bytes for A that
// the compiler never fuses plus 8 bytes strided by ldm for B -- followed by
// ~15 s_waitcnt vmcnt before a single v_mmac_i32_16x16x32_i8: 147456
// vmem_read_instructions over the 480-block partial launch.  Fix: keep the
// exact zero-barrier one-wave m16n16k32 tile, the 480-block split-K=10 grid,
// and the 48-block combine, but replace the byte-gather fragment loads with
// one aligned 8-byte (global_load_dwordx2) load per lane per fragment:
//   * A stays the logical row-major activation [16, K]: lane (row = lane&15,
//     col_group = (lane>>4)*8) owns A[row][k0+col_group .. k0+col_group+7],
//     which is 8 contiguous bytes (ldm = K = 6144);
//   * B is packed once, outside the timed region and outside Graph capture,
//     into the n-major transpose packed[n*K+kk] = raw[kk*n+n] for the exact
//     (k,n) == (6144,768) (same byte count and buffer, so Graph addresses
//     are unchanged; the identity copy remains the pack fallback for every
//     other (k,n)); lane (row = lane&15, col_group = (lane>>4)*8) then owns
//     packed[(n0+row)*K + k0+col_group .. +7], also 8 contiguous bytes;
//   * du_mma_sync consumes the identical byte pattern the library loader
//     would produce, so the int32 accumulation stays bit-identical
//     (k-ascending per slice, ascending-slice combine, 0 mismatches);
//   * the generic scalar fallback decodes the n-major pack for
//     (n,k) == (768,6144) (including the paired M=2 API shape with the same
//     (N,K)) and the workspace-too-small direct kernel reads B with ldm = k
//     through the same packed pointer.
// Falsifiable: partial-kernel vmem_read_instructions must drop from 147456
// to ~18432 (2 dwordx2 loads x ~19.2 chunks x 480 blocks) and the median
// must beat the 45.47 us iteration-3 best.
//
// Iteration 16 (final conditional inline-asm round -> HIP-only
// consolidation): raw asm stays forbidden (isa_policy.phase=hip_only,
// plateau=false: recent_valid_improvements_percent = [+3.44, +2.02, -4.52]
// after the rejected split-K=15 three-wave probe, so the required three
// recent valid HIP candidates within [-2%, +2%) of best do not exist), so
// this round consolidates the two-kernel operator into ONE launch with the
// qkv-proven monotonic-arrival fused combine tail (the exact pattern
// accepted on this stack for hy3_tp4_qkv_proj_m16, round 23):
//   * the partial kernel becomes template <bool FUSED>; FUSED=true adds the
//     tail: after wave 1 publishes the plane, one __syncthreads, then
//     (wave 1, lane 0) does __threadfence() + atomicAdd(&counters[tile], 1)
//     + __threadfence(); the block whose atomicAdd returns
//     (arrived % kSplitK) == kSplitK - 1 is the LAST arrival for its tile
//     and sums the tile's 10 planes in ascending slice order with the exact
//     iteration-12 combine epilogue (int4 plane loads, x_scale scalar +
//     weight_scale float4 hoisted, hip_bfloat16 conversion, 4 x 2-byte
//     stores), then writes the bf16 output;
//   * the 48-block combine kernel and its launch gap disappear from the
//     timed operator; the per-tile monotonic counters (n_tiles x int32 =
//     192 B) live in the last 192 B of the 786432-B (16-plane) contract
//     workspace (plane 15's tail; planes 0..9 are the only planes used),
//     zeroed once per workspace by a static-pointer-guarded hipMemsetAsync
//     on the caller stream before the first launch (eager warmup runs
//     before Graph capture, so the memset is never part of the captured
//     graph and every replay is exactly one kernel launch);
//   * the two-kernel path (FUSED=false + the combine kernel) remains as the
//     fallback for workspaces below the 16-plane contract, byte-identical
//     to iteration 14;
//   * per output element the int32 accumulation is still the
//     ascending-slice sum over the same bytes with the same conversion, so
//     the bf16 output is bit-identical to iteration 14 (0 mismatches
//     expected) and Graph capture/replay semantics are unchanged (one launch
//     on the caller stream, no allocation/sync/compile in the launcher).
//
// Logical operation (see int8_w8a8_gemm_api.py):
//   out[m, n] = bf16(int32_dot(x_q[m, :], packed_weight[:, n])
//                    * x_scale[m, 0] * packed_weight_scale[n, 0])
//
// Headers are included in the DTK-known-good order: hip_runtime, hip_bfloat16,
// then du_mma (this DTK's du_mma.h is not self-contained when included before
// the HIP runtime headers).
//
// The timed entry point (launch_w8a8_gemm) performs no allocation, no
// compilation, no autotuning, no weight packing, no host/device
// synchronization, and no default-stream launch: it only computes launch
// geometry and enqueues on the caller-provided stream, so it stays safe under
// torch.cuda.CUDAGraph capture and replay.

#include <hip/hip_runtime.h>
#include <hip/hip_bfloat16.h>
#include <du_mma.h>

#include <cstdint>

namespace {

// gfx928 native wavefront is 64 lanes; the block size must be a multiple of 64.
constexpr int kScalarBlockThreads = 128;
constexpr int kCopyBlockThreads = 256;

// Minimal DUMMA tile for the assigned M=16 decode shape: INT8 m16n16k32 with
// int32 accumulation, one 64-lane wavefront owning one independent fragment.
constexpr int kDummaTileM = 16;
constexpr int kDummaTileN = 16;
constexpr int kDummaTileK = 32;
constexpr int kDummaWaveThreads = 64;
// Iteration 14: the split-K partial kernel runs 128 threads = 2 wavefronts
// per block (block size stays a multiple of the 64-lane wavefront).
constexpr int kPartialBlockThreads = 2 * kDummaWaveThreads;

// Split-K geometry for the assigned MiniMax TP8 gate_up shape (iteration 3
// occupancy probe): 10 independent K slices per 16x16 output tile -> grid =
// 48 N tiles x 10 slices = 480 one-wave zero-barrier blocks = exactly 4.0
// blocks per device CU (120 CUs), doubling the iteration-2 split-K = 5 point
// (240 blocks, 2.0 blocks/CU, 50.55 us median).  K = 6144 = 192 chunks of 32
// = 10*19 + 2, so slices 0..1 carry 20 chunks (640 elements) and slices 2..9
// carry 19 chunks (608 elements); every slice boundary stays a multiple of
// the DUMMA K step and the imbalance is <= 6%.  The workspace holds kSplitK
// int32 partial planes of [16][768] (49152 B each); the Python API allocates
// 16 planes (786432 B) for this shape, so kSplitK = 10 (491520 B) always
// fits the workspace and 256-B stage alignment.
constexpr int kSplitK = 10;
constexpr int kM16N768Elems = 16 * 768;  // one int32 partial plane (12288)

// Iteration 16: the fused combine tail needs the full 16-plane contract
// workspace the Python API allocates for this shape (786432 B): the
// per-tile monotonic arrival counters live in the last n_tiles*4 B of
// plane 15's tail (planes 0..9 are the only partial planes used, so there
// is no overlap), exactly like the accepted qkv fused path.
constexpr int64_t kSkWorkspaceContractBytes =
    static_cast<int64_t>(16) * kM16N768Elems * sizeof(int32_t);  // 786432 B

// Scalar INT8 dot-product kernel: one thread -> one out[m, n] element.
//
// a: [M, K]  int8 row-major (logical activation, stride K)
// b: [K, N]  int8 row-major (identity-packed logical weight, stride N)
// x_scale: [M]     fp32
// weight_scale: [N] fp32
// out: [M, N] bf16
//
// Linear indexing keeps adjacent lanes on adjacent n, so B-column reads and
// the bf16 stores are coalesced in the fastest-changing N dimension.
__global__ __launch_bounds__(kScalarBlockThreads) void
w8a8_gemm_scalar_kernel(
    const int8_t* __restrict__ a,
    const int8_t* __restrict__ b,
    const float* __restrict__ x_scale,
    const float* __restrict__ weight_scale,
    hip_bfloat16* __restrict__ out,
    int m,
    int n,
    int k) {
  const int linear = blockIdx.x * blockDim.x + threadIdx.x;
  if (linear >= m * n) {
    return;
  }
  const int row = linear / n;
  const int col = linear - row * n;

  const int8_t* a_row = a + row * k;

  // The exact gate_up (n, k) == (768, 6144) uses the n-major packed weight
  // (packed[n*K+kk] = raw[kk*n+n], produced by launch_pack_w8a8_weight); all
  // other shapes keep the identity [K, N] row-major layout.  Decode both so
  // the generic fallback -- including the paired M=2 API shape with the same
  // (N, K) -- stays correct.
  const bool nmajor_packed = (n == 768 && k == 6144);
  const int8_t* b_col =
      nmajor_packed ? (b + static_cast<int64_t>(col) * k) : (b + col);
  const int64_t b_stride = nmajor_packed ? 1 : static_cast<int64_t>(n);

  // Exact int32 dot product.  Largest assigned K is 6144:
  // 6144 * 128 * 128 = 100,663,296 < 2^31, so the accumulation is exact and
  // cannot overflow.  int8 products are exact in int32, and the int32 sum is
  // bit-identical to the float32 dot of (A.float() @ B.float()).
  int32_t acc = 0;
  for (int i = 0; i < k; ++i) {
    acc += static_cast<int32_t>(a_row[i]) *
           static_cast<int32_t>(b_col[i * b_stride]);
  }

  // Reference: (A.float() @ B.float()) * x_scale * weight_scale.T -> bf16.
  // Both scales are applied left-associatively as in the torch reference.
  const float scaled =
      static_cast<float>(acc) * x_scale[row] * weight_scale[col];
  // hip_bfloat16's supported float conversion rounds to nearest even, which
  // matches the torch float->bfloat16 conversion.
  out[linear] = hip_bfloat16(scaled);
}

// Minimal INT8 DUMMA kernel for M=16 decode: one wave (64 lanes) per block,
// one 16x16 output tile per block, explicit int32 accumulation.  Reachable
// only under the exact (16, 768, 6144) shape guard when the workspace cannot
// hold the ten split-K partial planes, so B is the n-major packed weight
// (packed[n*K+kk] = raw[kk*n+n]) and is loaded with ldm = k.
//
// a: [16, K]  int8 row-major activation (M == 16 is enforced by the caller
//             guard; fragment loads need ldm = k)
// b: [N, K]   int8 n-major packed weight for the exact guard that is the
//             only caller (fragment loads need ldm = k)
// x_scale: [16]      fp32
// weight_scale: [N]  fp32
// out: [16, N] bf16
//
// Each K step loads a 16x32 A fragment and a 32x16 B fragment directly from
// global memory with du::dumma::du_load_matrix_sync (row_major fragments,
// 8-byte-aligned base offsets because K and the N tile origin are multiples
// of 32/16 bytes) and accumulates with du::dumma::du_mma_sync into an int32
// accumulator fragment.  No LDS, no __syncthreads, no cross-wave barrier:
// each block is a single wavefront and all stores are fragment-owner direct.
__global__ __launch_bounds__(kDummaWaveThreads) void
w8a8_gemm_dumma_m16_direct_kernel(
    const int8_t* __restrict__ a,
    const int8_t* __restrict__ b,
    const float* __restrict__ x_scale,
    const float* __restrict__ weight_scale,
    hip_bfloat16* __restrict__ out,
    int n,
    int k) {
  const int lane = static_cast<int>(threadIdx.x);  // 0..63
  const int n0 = static_cast<int>(blockIdx.x) * kDummaTileN;

  du::dumma::DUFragment<du::dumma::matrix_a, kDummaTileM, kDummaTileN,
                        kDummaTileK, signed char, du::dumma::row_major>
      a_frag;
  du::dumma::DUFragment<du::dumma::matrix_b, kDummaTileM, kDummaTileN,
                        kDummaTileK, signed char, du::dumma::row_major>
      b_frag;
  du::dumma::DUFragment<du::dumma::accumulator, kDummaTileM, kDummaTileN,
                        kDummaTileK, int>
      acc_frag;
  du::dumma::du_fill_fragment(acc_frag, 0);

  // Complete K loop accumulated in exact int32 (same values as the scalar
  // reference; K = 6144 * 128 * 128 < 2^31 so no overflow).  B is the n-major
  // packed weight for this shape, so the row_major B fragment loader with
  // ldm = k reads packed[(n0 + lane&15)*k + k0 + (lane>>4)*8 + i], exactly
  // W[k0 + (lane>>4)*8 + i][n0 + lane&15].
  for (int k0 = 0; k0 < k; k0 += kDummaTileK) {
    du::dumma::du_load_matrix_sync(a_frag, a + k0, k);
    du::dumma::du_load_matrix_sync(
        b_frag, b + static_cast<int64_t>(n0) * k + k0, k);
    du::dumma::du_mma_sync(acc_frag, a_frag, b_frag, acc_frag);
  }

  // Direct fragment epilogue.  The int32 accumulator ownership for this DTK
  // (verified against du_store_matrix_sync with mem_row_major) is:
  //   row = lane & 15, col_group = lane >> 4,
  //   acc_frag.x[i] holds output column col_group + 4*i.
  // The four 2-byte bf16 values of one row are written by lanes l, l+16,
  // l+32, l+48 (same lane&15); the whole 16x16 tile is only 512 bytes of
  // output so the scattered 2-byte stores are absorbed by L2 write combining.
  const int row = lane & 15;
  const int col_group = lane >> 4;
  const float xs = x_scale[row];
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const int col = n0 + col_group + 4 * i;
    const float scaled =
        static_cast<float>(acc_frag.x[i]) * xs * weight_scale[col];
    out[row * n + col] = hip_bfloat16(scaled);
  }
}

// Split-K partial DUMMA kernel (iteration 2 architecture, iteration 5 load
// path): identical m16n16k32 tile and zero-barrier one-wave grid as the
// iteration-1 kernel, but each block accumulates only one 32-aligned K slice
// (kSplitK total) and publishes its int32 accumulator fragment to workspace
// plane s instead of the final bf16 output.  No LDS, no __syncthreads, no
// cross-wave barrier; every (N tile, K slice) pair is written exactly once
// per launch, so no workspace clear is needed; k-ascending accumulation
// within each slice.
//
// Iteration-5 fragment transport: the library du_load_matrix_sync expands a
// chunk into 16 global_load_ubyte + ~15 vmcnt waits per lane (the known
// decode load-path poison: 147456 vmem_read_instructions over the launch).
// Replace it with one aligned 8-byte global_load_dwordx2 per lane per
// fragment, using the verified m16n16k32 int8 fragment ownership:
//   row = lane & 15, col_group = (lane >> 4) * 8,
//   a_frag.x[i] = A[row][k0 + col_group + i]           (logical [16, K],
//               8 contiguous bytes, base a + row*k + k0 + col_group);
//   b_frag.x[i] = B[k0 + col_group + i][n0 + row]      (n-major packed
//               packed[n*K+kk] = raw[kk*n+n], 8 contiguous bytes, base
//               b + (n0 + row)*k + k0 + col_group).
// Both addresses are 8-byte aligned (k0 % 32 == 0, col_group % 8 == 0,
// k == 6144) and du_mma_sync consumes the identical byte pattern the library
// loader would produce, so the int32 accumulation stays bit-identical.
//
// Iteration 8: the chunk loop is software-pipelined with depth-1 register
// prefetch (next chunk's two dwordx2 loads issued after the current v_mmac;
// see the in-kernel comment).  Same loads, same k-ascending order, same
// bit-identical int32 accumulation; only the load/compute overlap changes.
//
// Iteration 13 (HIP-only consolidation): pair two adjacent k32 chunks into
// one k64 round.  The accepted iteration-8/12 steady state is one serialized
// 2 x global_load_dwordx2 -> wait -> 1 x v_mmac chain per k32 chunk
// (19-20 wait rounds per block at ~450-530 cycles of global/L2 latency
// each), so partial-kernel time is proportional to the number of wait
// rounds.  Each round now covers TWO k32 sub-chunks with FOUR dwordx2 loads
// (sub-chunk 0 at k0, sub-chunk 1 at k0+32; the lane's 8 bytes of the two
// sub-chunks are 32 bytes apart in k, so they stay two 8-byte loads, never a
// dwordx4) prefetched one round ahead, ONE wait group, and TWO v_mmac.  The
// grid stays exactly 480 blocks / 4.0 blocks per CU and every block runs
// exactly 10 rounds (slices 0..1: 10 pairs of 64; slices 2..9: 9 pairs + one
// plain 32-row tail chunk) instead of 19-20, with every prefetch address
// bounded below k_end.  Each v_mmac consumes the identical k-ascending byte
// pattern of its k32 range in the same relative order, so the int32
// accumulation per output element (and therefore the bf16 output) is
// bit-identical to iteration 8/12.
//
// Iteration 14 (HIP-only consolidation): the iteration-13 code object shows
// the compiler wait-splits the 4-load k64 round into vmcnt(2) -> v_mmac ->
// vmcnt(0) -> v_mmac, so the second sub-chunk still exposes a full load
// latency and halving wait rounds gained only +3.4% (18.9963 -> 18.3637 us):
// at the current exactly-one-wavefront-per-SIMD grid (480 blocks x 64
// threads = 480 wavefronts over 120 CUs x 4 SIMDs) the kernel is ~99%
// memory-latency stall (~306 VALU issued per block vs ~27.7k cycles per
// block), and no HIP change can shorten a single wavefront's serial wait
// chain (counter-split waits are raw-asm-only, and the HIP plateau is not
// proven, so raw asm stays forbidden).  Raise SIMD-level concurrency
// instead: each block becomes 128 threads = 2 wavefronts; wave 0 accumulates
// chunks [0, half) of the slice and wave 1 accumulates [half, end) --
// exactly 5 k64 rounds per wavefront, perfectly balanced for both slice
// sizes (20-chunk slices: 5 pairs each; 19-chunk slices: wave 0 runs 4 pairs
// + 1 tail chunk, wave 1 runs 5 pairs) -- then wave 0 publishes its int32
// 16x16 tile to a 1 KiB LDS tile, one __syncthreads, and wave 1 adds it
// (ascending slice order: wave 0's k-range is strictly below wave 1's, and
// int32 addition is exact/associative with max |acc| ~ 1e8 < 2^31, so the
// combined sum -- and therefore the bf16 output -- is bit-identical to the
// single-wave ascending accumulation) and writes the plane.  Grid stays 480
// blocks / 4.0 blocks per CU = 960 wavefronts = exactly 2 per SIMD, so each
// SIMD can interleave a second wavefront during the load stalls; the load
// set (vmem_read 18432), plane stores (vmem_write 1920), workspace (10
// planes), the combine kernel, and every fallback are unchanged.
//
// grid = 48 N tiles x 10 K slices = 480 two-wave blocks (4.0 blocks/CU at
// 120 CUs = 2 wavefronts per SIMD); slice s bounds are computed from kSplitK
// and k so the launcher stays allocation/sync-free.
//
// Iteration 16: the kernel becomes template <bool FUSED>.  FUSED=false is
// byte-identical to the accepted iteration-14 kernel (the tail parameters
// x_scale / weight_scale / out / counters are never dereferenced).
// FUSED=true appends the qkv-proven monotonic-arrival fused combine tail
// (see the file header): wave 1 publishes the plane, one __syncthreads,
// then (wave 1, lane 0) does __threadfence() + atomicAdd(&counters[tile],
// 1) + __threadfence(); the block whose atomicAdd returns
// (arrived % kSplitK) == kSplitK - 1 is the LAST arrival for its tile and
// runs the iteration-12 combine epilogue (one aligned int4 load per plane
// per lane in ascending slice order, x_scale scalar + weight_scale float4
// hoisted, hip_bfloat16 conversion, 4 x 2-byte stores) so the bf16 output
// is bit-identical to the two-kernel path.
template <bool FUSED>
__global__ __launch_bounds__(kPartialBlockThreads) void
w8a8_gemm_dumma_m16_direct_sk_partial_kernel(
    const int8_t* __restrict__ a,
    const int8_t* __restrict__ b,
    int32_t* __restrict__ partials,  // [kSplitK][16][768] int32
    const float* __restrict__ x_scale,
    const float* __restrict__ weight_scale,
    hip_bfloat16* __restrict__ out,
    int32_t* __restrict__ counters,  // [n_tiles] monotonic arrival counts
    int n,
    int k) {
  const int wave = static_cast<int>(threadIdx.x) >> 6;  // 0 or 1
  const int lane = static_cast<int>(threadIdx.x) & 63;  // 0..63
  const int n_tiles = n / kDummaTileN;             // 48 for n == 768
  const int tile = static_cast<int>(blockIdx.x) % n_tiles;
  const int s = static_cast<int>(blockIdx.x) / n_tiles;
  const int n0 = tile * kDummaTileN;

  // 32-aligned K slice bounds for slice s (remainder spread over the first
  // slices so every boundary is a multiple of the DUMMA K step).
  const int total_chunks = k / kDummaTileK;  // 192 for k == 6144
  const int base_chunks = total_chunks / kSplitK;
  const int rem_chunks = total_chunks % kSplitK;
  const int slice_chunks = base_chunks + (s < rem_chunks ? 1 : 0);
  const int k_start = (s * base_chunks + (s < rem_chunks ? s : rem_chunks)) *
                      kDummaTileK;
  const int k_end = k_start + slice_chunks * kDummaTileK;

  // Iteration-14 per-wave chunk split: wave w accumulates chunks
  // [w*half, ...) of the slice so both wavefronts are exactly balanced
  // (20-chunk slices: 10 per wave; 19-chunk slices: 9 in wave 0, 10 in
  // wave 1) and every chunk stays k-ascending within its wave with wave 0's
  // range strictly below wave 1's, so the per-block LDS combine below
  // preserves the ascending-slice int32 order.
  const int half_chunks = slice_chunks >> 1;  // 9 or 10
  const int chunk_lo = wave * half_chunks;
  const int chunk_hi = (wave == 0) ? half_chunks : slice_chunks;
  const int k_lo = k_start + chunk_lo * kDummaTileK;
  const int k_hi = k_start + chunk_hi * kDummaTileK;
  const int wave_chunks = chunk_hi - chunk_lo;  // 9 or 10

  du::dumma::DUFragment<du::dumma::matrix_a, kDummaTileM, kDummaTileN,
                        kDummaTileK, signed char, du::dumma::row_major>
      a_frag;
  du::dumma::DUFragment<du::dumma::matrix_b, kDummaTileM, kDummaTileN,
                        kDummaTileK, signed char, du::dumma::row_major>
      b_frag;
  du::dumma::DUFragment<du::dumma::accumulator, kDummaTileM, kDummaTileN,
                        kDummaTileK, int>
      acc_frag;
  du::dumma::du_fill_fragment(acc_frag, 0);

  // Iteration-5 load path: one aligned 8-byte vector load per lane per
  // fragment (see kernel doc for the ownership mapping).  a_frag.x[0..7] and
  // b_frag.x[0..7] are 8 contiguous bytes; du_mma_sync itself reinterprets
  // each as one 64-bit operand, so the fragment registers are filled exactly
  // as the library loader would fill them.
  //
  // Iteration-8 depth-1 register prefetch: the accepted iteration-5 steady
  // state is 2 x global_load_dwordx2 -> s_waitcnt vmcnt(0) -> 1 x v_mmac per
  // chunk, so every v_mmac stalls on loads issued in the same iteration
  // (latency-bound despite 4.0 blocks/CU).  Software-pipeline the K loop
  // instead: hold the current chunk's two 8-byte words in plain locals and
  // issue chunk (k0 + kDummaTileK)'s two dwordx2 loads AFTER chunk k0's
  // v_mmac.  Each v_mmac then consumes words loaded a full iteration earlier,
  // and at the point the wait must be placed the only outstanding loads are
  // that older pair -- no loads younger than the consumed ones are in flight
  // -- so a plain vmcnt(0) covers exactly the right data and the prefetch
  // pair stays in flight across the next iteration.  The fragment registers
  // still receive the identical k-ascending byte pattern, so the int32
  // accumulation order (and therefore the bf16 output) is bit-identical to
  // iteration 5.
  const int arow = lane & 15;
  const int acol = (lane >> 4) << 3;  // {0, 8, 16, 24}: k offset within a
                                      // 32-wide chunk (lane owns 8 bytes)
  const int brow = lane & 15;         // n index within the 16-wide tile
  const int bcol = (lane >> 4) << 3;  // k offset within the 32-wide chunk
  // Iteration-13 k64 pairing, applied to each wavefront's own chunk range:
  // every wavefront runs exactly 5 wait rounds (wave_chunks 9 or 10 -> 4
  // pairs + 1 tail chunk, or 5 pairs).  Four dwordx2 words (two k32
  // sub-chunks) are software-pipelined one round ahead exactly like the
  // iteration-8 pair, so the wait before each round covers only the
  // previously issued 4-load set; all prefetch addresses stay strictly below
  // k_hi (the loop prefetches only pairs that exist, and the tail chunk is
  // loaded plainly with no prefetch).
  const int pairs = wave_chunks >> 1;       // 5 per wavefront
  const int tail_chunks = wave_chunks & 1;  // wave 0 only on odd slices
  const int k_pair_end = k_lo + pairs * 2 * kDummaTileK;
  // Prefetch pair 0: sub-chunk 0 at k_lo, sub-chunk 1 at k_lo + 32.
  long a_word0 = *reinterpret_cast<const long*>(
      a + static_cast<int64_t>(arow) * k + k_lo + acol);
  long b_word0 = *reinterpret_cast<const long*>(
      b + static_cast<int64_t>(n0 + brow) * k + k_lo + bcol);
  long a_word1 = *reinterpret_cast<const long*>(
      a + static_cast<int64_t>(arow) * k + k_lo + kDummaTileK + acol);
  long b_word1 = *reinterpret_cast<const long*>(
      b + static_cast<int64_t>(n0 + brow) * k + k_lo + kDummaTileK + bcol);
  int k0 = k_lo;
  for (; k0 + 2 * kDummaTileK < k_pair_end; k0 += 2 * kDummaTileK) {
    // Consume sub-chunk 0 (k0 .. k0+31), then sub-chunk 1 (k0+32 .. k0+63):
    // same k-ascending v_mmac sequence as two adjacent iteration-8 chunks.
    *reinterpret_cast<long*>(a_frag.x) = a_word0;
    *reinterpret_cast<long*>(b_frag.x) = b_word0;
    du::dumma::du_mma_sync(acc_frag, a_frag, b_frag, acc_frag);
    *reinterpret_cast<long*>(a_frag.x) = a_word1;
    *reinterpret_cast<long*>(b_frag.x) = b_word1;
    du::dumma::du_mma_sync(acc_frag, a_frag, b_frag, acc_frag);
    // Prefetch pair p+1 (four dwordx2 loads) after pair p's two v_mmacs, so
    // all four words are in flight one full round before consumption.
    a_word0 = *reinterpret_cast<const long*>(
        a + static_cast<int64_t>(arow) * k + (k0 + 2 * kDummaTileK) + acol);
    b_word0 = *reinterpret_cast<const long*>(
        b + static_cast<int64_t>(n0 + brow) * k +
        (k0 + 2 * kDummaTileK) + bcol);
    a_word1 = *reinterpret_cast<const long*>(
        a + static_cast<int64_t>(arow) * k + (k0 + 3 * kDummaTileK) + acol);
    b_word1 = *reinterpret_cast<const long*>(
        b + static_cast<int64_t>(n0 + brow) * k +
        (k0 + 3 * kDummaTileK) + bcol);
  }
  // Last pair: consume the words prefetched by the final loop iteration.
  *reinterpret_cast<long*>(a_frag.x) = a_word0;
  *reinterpret_cast<long*>(b_frag.x) = b_word0;
  du::dumma::du_mma_sync(acc_frag, a_frag, b_frag, acc_frag);
  *reinterpret_cast<long*>(a_frag.x) = a_word1;
  *reinterpret_cast<long*>(b_frag.x) = b_word1;
  du::dumma::du_mma_sync(acc_frag, a_frag, b_frag, acc_frag);
  // Odd-slice tail: the remaining k32 chunk (slices 2..9: 608 = 9*64 + 32)
  // at k_pair_end, loaded plainly with no prefetch (nothing follows; the
  // address is a multiple of 32 and stays below k_end).
  if (tail_chunks) {
    *reinterpret_cast<long*>(a_frag.x) = *reinterpret_cast<const long*>(
        a + static_cast<int64_t>(arow) * k + k_pair_end + acol);
    *reinterpret_cast<long*>(b_frag.x) = *reinterpret_cast<const long*>(
        b + static_cast<int64_t>(n0 + brow) * k + k_pair_end + bcol);
    du::dumma::du_mma_sync(acc_frag, a_frag, b_frag, acc_frag);
  }

  // Iteration-14 per-block int32 combine: wave 0 publishes its partial
  // 16x16 accumulator tile to a 1 KiB LDS tile, one barrier, then wave 1
  // adds wave 0's partial (ascending slice order: wave 0 holds chunks
  // [0, half), wave 1 holds [half, end); int32 addition is exact and
  // associative with max |acc| ~ 1e8 < 2^31, so the combined sum is
  // bit-identical to the single-wave ascending accumulation) and publishes
  // the plane with the same direct fragment ownership as the iteration-1
  // epilogue: row = lane & 15, col_group = lane >> 4, acc_frag.x[i] ->
  // column col_group + 4*i.  Plane strides are multiples of 256 B so every
  // store stays 4-B aligned (workspace base is 256-B aligned).
  __shared__ int32_t s_part[kDummaTileM * kDummaTileN];  // 1 KiB
  const int row = lane & 15;
  const int col_group = lane >> 4;
  if (wave == 0) {
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      s_part[row * kDummaTileN + col_group + 4 * i] = acc_frag.x[i];
    }
  }
  __syncthreads();
  if (wave == 1) {
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      acc_frag.x[i] += s_part[row * kDummaTileN + col_group + 4 * i];
    }
    int32_t* plane = partials + static_cast<int64_t>(s) * kM16N768Elems;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      plane[row * n + n0 + col_group + 4 * i] = acc_frag.x[i];
    }
  }

  // Iteration-16 fused combine tail (FUSED=true only): the 48-block combine
  // kernel and its launch gap are consolidated into the partial launch with
  // the exact qkv-proven monotonic-arrival pattern (accepted on this stack
  // for hy3_tp4_qkv_proj_m16, round 23): after the plane publish, one
  // __syncthreads, then (wave 1, lane 0) does
  // __threadfence() + atomicAdd(&counters[tile], 1) + __threadfence().
  // The block whose atomicAdd returns (arrived % kSplitK) == kSplitK - 1 is
  // the LAST arrival for its tile (counters are monotonic, zeroed once per
  // workspace by the launcher before the first launch; int32 wraparound
  // would need ~2^31/10 replays): the release fence orders this block's
  // plane store before its arrival atomic, the acquire fence orders its
  // plane reads after observing every sibling arrival, and the intervening
  // barriers broadcast the flag to the wave, so no spin loop or residency
  // assumption is needed.  The last block then sums the tile's kSplitK
  // planes in ascending slice order with the exact iteration-12 combine
  // epilogue (one aligned int4 load per plane per lane, x_scale scalar +
  // weight_scale float4 hoisted, hip_bfloat16 conversion, 4 x 2-byte
  // stores), so the bf16 output is bit-identical to iteration 14 (same
  // planes, same ascending int32 sum, same conversion and store addresses).
  if (FUSED) {
    __syncthreads();
    __shared__ int s_is_last;
    if (wave == 1 && lane == 0) {
      __threadfence();  // release: this block's plane stores are visible to
                        // the observer of this block's arrival atomic
      const int arrived = atomicAdd(&counters[tile], 1);
      __threadfence();  // acquire: this block's plane reads below (after the
                        // barrier) see every sibling store released before
                        // its arrival atomic
      s_is_last = ((arrived % kSplitK) == kSplitK - 1);
    }
    __syncthreads();
    if (s_is_last && wave == 1) {
      // Epilogue, byte-identical to w8a8_gemm_m16_n768_sk_combine_kernel:
      // lane -> (erow = lane >> 2, eq = lane & 3) owns four consecutive
      // columns ecol0 = n0 + 4*eq (16-B-aligned int4 plane reads and
      // 8-B-aligned bf16 stores, as in the iteration-12 vectorized combine).
      const int erow = lane >> 2;
      const int eq = lane & 3;
      const int ecol0 = n0 + 4 * eq;
      const int eidx = erow * n + ecol0;
      const float xs = x_scale[erow];
      const float4 ws = *reinterpret_cast<const float4*>(weight_scale + ecol0);
      int32_t eacc0 = 0, eacc1 = 0, eacc2 = 0, eacc3 = 0;
#pragma unroll
      for (int s2 = 0; s2 < kSplitK; ++s2) {
        const int4 v = *reinterpret_cast<const int4*>(
            partials + static_cast<int64_t>(s2) * kM16N768Elems + eidx);
        eacc0 += v.x;
        eacc1 += v.y;
        eacc2 += v.z;
        eacc3 += v.w;
      }
      const float es0 = static_cast<float>(eacc0) * xs * ws.x;
      const float es1 = static_cast<float>(eacc1) * xs * ws.y;
      const float es2 = static_cast<float>(eacc2) * xs * ws.z;
      const float es3 = static_cast<float>(eacc3) * xs * ws.w;
      hip_bfloat16* eo = out + eidx;
      eo[0] = hip_bfloat16(es0);
      eo[1] = hip_bfloat16(es1);
      eo[2] = hip_bfloat16(es2);
      eo[3] = hip_bfloat16(es3);
    }
  }
}

// Split-K combine kernel (iteration 2; iteration 12 vectorized plane reads):
// one 64-lane wavefront per 16x16 output tile (48 blocks).  Each lane sums
// the kSplitK partial planes in ascending slice order -- exact int32 order,
// bit-identical to the scalar reference (int32 addition is associative and
// max |acc| ~ 1.0e8 < 2^31) -- then applies x_scale[m] * weight_scale[n] and
// stores bf16.  This kernel is part of the timed operator: it is enqueued on
// the caller stream right after the partial kernel, inside the same Graph
// capture.
//
// Iteration 12 (HIP-only consolidation): the accepted iteration-2 scalar
// combine's ISA (profiled 5.28 us over the 48-block launch, 23-25% of the
// operator aggregate) is 45 global_load_dword + 26 s_waitcnt per wavefront:
// each lane reads its four output columns (col_group + 4*i, strided by 4)
// with 4 x 10 scalar dword loads and four partially-serialized wait chains.
// Remap the 64 lanes to (row = lane >> 2, quarter = lane & 3) so each lane
// owns four CONSECUTIVE columns and replaces its 40 dword loads with 10
// aligned 16-byte global_load_dwordx4 (one int4 per plane: plane stride
// 49152 B, row stride 3072 B, and quarter base 16 B are all multiples of 16,
// so every int4 address is 16-B aligned).  The ten plane loads are
// independent, so one wait group covers the set instead of 26 per-wavefront
// waits, and the per-wavefront 64-lane footprint of one load instruction is
// 16 full 64-B lines instead of 64 scattered dwords (the four lanes of a row
// wrote the full line, so every line read is complete and L2-resident).
// Scale loads are hoisted to the kernel top so they issue with the plane
// burst (the prior vectorized object issued the weight_scale dwordx4 late and
// exposed a second serialized wait after the plane chain).  Per element the
// accumulation is still the same ascending-slice int32 sum over the same
// bytes, so the bf16 output is bit-identical to iteration 2; only the load
// granularity / lane mapping / load order change.  Stores stay 4 x 2-byte
// bf16 per lane (L2 write combining absorbs them, as in the partial-kernel
// epilogue).
__global__ __launch_bounds__(kDummaWaveThreads) void
w8a8_gemm_m16_n768_sk_combine_kernel(
    const int32_t* __restrict__ partials,  // [kSplitK][16][768] int32
    const float* __restrict__ x_scale,
    const float* __restrict__ weight_scale,
    hip_bfloat16* __restrict__ out,
    int n) {
  const int lane = static_cast<int>(threadIdx.x);  // 0..63
  const int n0 = static_cast<int>(blockIdx.x) * kDummaTileN;
  const int row = lane >> 2;   // 0..15
  const int q = lane & 3;      // 0..3: 4-column quarter within the tile
  const int col0 = n0 + 4 * q; // 4 consecutive int32 columns (16-B aligned)
  const int idx = row * n + col0;

  // Scale loads hoisted to the kernel top (x_scale scalar, weight_scale as
  // one aligned float4) so they issue with the plane burst below instead of
  // trailing the plane chain with their own serialized wait.
  const float xs = x_scale[row];
  const float4 ws = *reinterpret_cast<const float4*>(weight_scale + col0);

  // One 16-byte int4 load per plane, ascending slice order; all ten loads
  // are independent, so a single wait group covers the set (see kernel doc).
  int32_t acc0 = 0, acc1 = 0, acc2 = 0, acc3 = 0;
#pragma unroll
  for (int s = 0; s < kSplitK; ++s) {
    const int4 v = *reinterpret_cast<const int4*>(
        partials + static_cast<int64_t>(s) * kM16N768Elems + idx);
    acc0 += v.x;
    acc1 += v.y;
    acc2 += v.z;
    acc3 += v.w;
  }

  // Same per-element scaling and 2-byte bf16 stores as iteration 2.
  const float scaled0 = static_cast<float>(acc0) * xs * ws.x;
  const float scaled1 = static_cast<float>(acc1) * xs * ws.y;
  const float scaled2 = static_cast<float>(acc2) * xs * ws.z;
  const float scaled3 = static_cast<float>(acc3) * xs * ws.w;
  hip_bfloat16* o = out + idx;
  o[0] = hip_bfloat16(scaled0);
  o[1] = hip_bfloat16(scaled1);
  o[2] = hip_bfloat16(scaled2);
  o[3] = hip_bfloat16(scaled3);
}

// Identity device-to-device copy for the bootstrap pack operation (int8
// weights).  Grid-stride loop: generic fallback for every (K, N).
__global__ void w8a8_pack_identity_i8_kernel(
    const int8_t* __restrict__ src,
    int8_t* __restrict__ dst,
    int64_t numel) {
  const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       idx < numel; idx += stride) {
    dst[idx] = src[idx];
  }
}

// Identity device-to-device copy for the bootstrap pack operation (fp32
// per-column weight scales).
__global__ void w8a8_pack_identity_f32_kernel(
    const float* __restrict__ src,
    float* __restrict__ dst,
    int64_t numel) {
  const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       idx < numel; idx += stride) {
    dst[idx] = src[idx];
  }
}

// N-major transpose pack (iteration 5) for the exact MiniMax TP8 gate_up
// shape (k, n) == (6144, 768): dst[n * k + kk] = src[kk * n + n].  Runs once,
// outside the timed region and outside Graph capture, into the same
// caller-owned packed_weight buffer (byte count unchanged, so captured
// addresses stay valid).  Each m16n16k32 B-fragment lane then owns 8
// contiguous bytes (see w8a8_gemm_dumma_m16_direct_sk_partial_kernel),
// enabling one aligned global_load_dwordx2 per fragment instead of 8
// strided global_load_ubyte.  The identity copy remains the fallback for
// every other (K, N).
__global__ void w8a8_pack_nmajor_i8_kernel(
    const int8_t* __restrict__ src,
    int8_t* __restrict__ dst,
    int64_t numel,
    int n,
    int k) {
  const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       idx < numel; idx += stride) {
    const int64_t n_idx = idx / k;
    const int64_t k_idx = idx - n_idx * k;
    dst[idx] = src[k_idx * n + n_idx];
  }
}

}  // namespace

// gemm_out backend entry point (called by csrc/bindings.cpp on PyTorch's
// current HIP stream).  This function must stay allocation/sync/compile-free
// and enqueue only on the caller stream: it is captured into a CUDA/HIP Graph
// and replayed with changed tensor contents.
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
  // workspace (caller-owned, allocated before Graph capture) holds the
  // kSplitK int32 partial planes used only by the exact-shape guard below
  // (plus the per-tile fused-combine arrival counters in the last 192 B of
  // the 16-plane contract workspace); its capacity is validated by the
  // Python API layer.

  hip_bfloat16* out_bf16 = static_cast<hip_bfloat16*>(out);

  // Exact-shape guard for the optimized M=16 MiniMax TP8 gate_up shape
  // (16x768x6144): iteration-16 fused single-kernel split-K = 10 DUMMA
  // m16n16k32 (480 two-wave partial blocks = 4.0 blocks/CU = 2
  // wavefronts/SIMD, one aligned 8-byte vector load per lane per fragment
  // against the n-major packed weight, per-block LDS combine, qkv-proven
  // monotonic-arrival fused combine tail replacing the 48-block combine
  // kernel), on the caller stream.  Every other
  // (m, n, k) -- including the paired M=2 API shape with the same (N, K)
  // and the paired 16x6144x384 down_proj shape -- falls through to the
  // scalar generic fallback below (which decodes the n-major pack for
  // (n, k) == (768, 6144)).
  if (m == 16 && n == 768 && k == 6144) {
    const int n_tiles = n / kDummaTileN;  // 48 tiles of 16 columns
    constexpr int64_t kSkWorkspaceBytes =
        static_cast<int64_t>(kSplitK) * kM16N768Elems * sizeof(int32_t);
    if (workspace_bytes >= kSkWorkspaceContractBytes) {
      // Iteration-16 fused single-kernel path: the combine runs inside the
      // partial kernel (last arrival per tile), so the operator is ONE
      // launch and the 48-block combine kernel + its launch gap disappear.
      // Grid stays 48 tiles x 10 slices = 480 two-wave blocks (2
      // wavefronts per SIMD).  The per-tile monotonic arrival counters
      // (n_tiles x int32 = 192 B) live in the last 192 B of the 786432-B
      // (16-plane) contract workspace (plane 15's tail; planes 0..9 are
      // the only partial planes used, so there is no overlap).  The
      // counters are zeroed once per workspace with an async memset on the
      // caller stream (static workspace-pointer guard): the first launch
      // happens during the eager warmup before Graph capture, so the
      // memset is never part of the captured graph and every replay is
      // exactly one kernel launch.
      int32_t* partials = static_cast<int32_t*>(workspace);
      int32_t* counters = reinterpret_cast<int32_t*>(
          static_cast<char*>(workspace) + kSkWorkspaceContractBytes) -
          n_tiles;
      static const void* s_fused_counters_ws = nullptr;
      if (s_fused_counters_ws != workspace) {
        hipMemsetAsync(counters, 0,
                       static_cast<size_t>(n_tiles) * sizeof(int32_t),
                       stream);
        s_fused_counters_ws = workspace;
      }
      hipLaunchKernelGGL(
          HIP_KERNEL_NAME(
              w8a8_gemm_dumma_m16_direct_sk_partial_kernel<true>),
          dim3(n_tiles * kSplitK), dim3(kPartialBlockThreads), 0, stream,
          a, b, partials, x_scale, weight_scale, out_bf16, counters, n, k);
      return;
    }
    if (workspace_bytes >= kSkWorkspaceBytes) {
      // Two-kernel path for workspaces below the 16-plane contract (cannot
      // happen through the validated API path): the iteration-14 partial
      // kernel with the fused tail compiled out (FUSED=false) + the
      // iteration-12 vectorized combine kernel, byte-identical to the
      // accepted iteration-14 behavior.  Grid = 48 tiles x 10 slices = 480
      // two-wave blocks = exactly 4.0 blocks per device CU (120 CUs) = 2
      // wavefronts per SIMD; the FUSED=false instantiation never
      // dereferences x_scale / weight_scale / out / counters.
      int32_t* partials = static_cast<int32_t*>(workspace);
      hipLaunchKernelGGL(
          HIP_KERNEL_NAME(
              w8a8_gemm_dumma_m16_direct_sk_partial_kernel<false>),
          dim3(n_tiles * kSplitK), dim3(kPartialBlockThreads), 0, stream,
          a, b, partials, x_scale, weight_scale, out_bf16, nullptr, n, k);
      hipLaunchKernelGGL(
          HIP_KERNEL_NAME(w8a8_gemm_m16_n768_sk_combine_kernel),
          dim3(n_tiles), dim3(kDummaWaveThreads), 0, stream,
          partials, x_scale, weight_scale, out_bf16, n);
      return;
    }
    // Workspace too small for the ten partial planes (cannot happen through
    // the validated API path): keep the iteration-1 direct kernel under the
    // same shape guard.
    const int blocks = n / kDummaTileN;
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(w8a8_gemm_dumma_m16_direct_kernel),
        dim3(blocks), dim3(kDummaWaveThreads), 0, stream,
        a, b, x_scale, weight_scale, out_bf16, n, k);
    return;
  }

  // Generic scalar fallback for every unmatched (m, n, k), including M=2
  // API shapes that share an assigned (N, K).
  const int threads = kScalarBlockThreads;
  const int total = m * n;
  const int blocks = (total + threads - 1) / threads;
  hipLaunchKernelGGL(
      HIP_KERNEL_NAME(w8a8_gemm_scalar_kernel),
      dim3(blocks), dim3(threads), 0, stream,
      a, b, x_scale, weight_scale, out_bf16, m, n, k);
}

// pack_weight backend entry point (called by csrc/bindings.cpp on PyTorch's
// current HIP stream, outside the timed region and outside Graph capture).
//
// Bootstrap: identity device-to-device copy of raw_weight[K, N] and
// weight_scale[N, 1].  Later Parallel-explore rounds may change only this
// function's HIP implementation (and the matching GEMM interpretation) to
// test packed layouts; the identity copy remains the generic fallback for
// unmatched (K, N).
extern "C" void launch_pack_w8a8_weight(
    const int8_t* raw_weight,
    const float* weight_scale,
    int8_t* packed_weight,
    float* packed_weight_scale,
    int k,
    int n,
    hipStream_t stream) {
  const int64_t weight_numel = static_cast<int64_t>(k) * n;
  const int weight_blocks = static_cast<int>(
      (weight_numel + kCopyBlockThreads - 1) / kCopyBlockThreads);
  const int scale_blocks =
      (n + kCopyBlockThreads - 1) / kCopyBlockThreads;

  if (k == 6144 && n == 768) {
    // Iteration-5 n-major transpose pack for the exact gate_up shape
    // (packed[n*K+kk] = raw[kk*n+n]); the matching GEMM kernels and the
    // scalar generic fallback decode this layout.  Identity copy for every
    // other (k, n).
    hipLaunchKernelGGL(
        w8a8_pack_nmajor_i8_kernel,
        dim3(weight_blocks), dim3(kCopyBlockThreads), 0, stream,
        raw_weight, packed_weight, weight_numel, n, k);
  } else {
    hipLaunchKernelGGL(
        w8a8_pack_identity_i8_kernel,
        dim3(weight_blocks), dim3(kCopyBlockThreads), 0, stream,
        raw_weight, packed_weight, weight_numel);
  }
  hipLaunchKernelGGL(
      w8a8_pack_identity_f32_kernel,
      dim3(scale_blocks), dim3(kCopyBlockThreads), 0, stream,
      weight_scale, packed_weight_scale, static_cast<int64_t>(n));
}
// @@end
