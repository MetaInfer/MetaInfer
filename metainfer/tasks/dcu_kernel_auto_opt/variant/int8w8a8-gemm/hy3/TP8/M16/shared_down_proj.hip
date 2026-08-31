// @@variant shape=hy3_tp8_shared_down_proj_m16 commit=6f8e396e4eb7b316acecbb928ea5db9f2f3ffe4d added=2026-08-27
//   median_us=7.696 p90_us=8.174
//   source=hy3-dsh-tp8-m16-2-dc53295a
// INT8 W8A8 GEMM implementation for gfx928 (K500SM_AI).
//
// Worker: worker_3, physical GPU 3.
// Assigned shape: hy3_tp8_shared_down_proj_m16 -> (M, N, K) = (16, 4096, 192).
//
// Iteration 1: minimal DUMMA bootstrap. The exact assigned shape
// (M=16, N=4096, K=192) is served by a native gfx928 DUMMA INT8
// m16n16k32 kernel with one 64-lane wavefront per block, one 16x16 output
// tile per block, and explicit int32 accumulation:
//   - A fragment: du::dumma::DUFragment<matrix_a,16,16,32,signed char,
//     row_major>, loaded directly from global memory each K step with
//     leading dimension K.
//   - B fragment: the analogous matrix_b fragment, loaded directly from
//     global memory at (k0, n0) with leading dimension N.
//   - C: du::dumma::DUFragment<accumulator,16,16,32,int>, filled to 0 and
//     accumulated by du_mma_sync over the 6 K-tiles (K=192 = 6*32).
//   - Epilogue: du_store_matrix_sync (mem_row_major) into a 1 KiB shared
//     tile, one intra-block __syncthreads (single wavefront, no cross-wave
//     barrier), then fp32 scale multiply in the reference order
//     (dot * x_scale[m] * weight_scale[n]) and a coalesced bf16 store.
//
// Iteration 2 (rejected): 2-wave/128-block geometry probe regressed to
// 18.165 us median, falsifying the launch-geometry hypothesis and pointing
// the search at the load path.
//
// Iteration 3 (this revision): load path. The iteration-1 code object showed
// the poison pattern - du_load_matrix_sync expands to ~16 global_load_ubyte
// per K=32 step with a full vmcnt drain before each single
// v_mmac_i32_16x16x32_i8 (6 serialized chains per block). This revision
// removes global traffic from the K loop entirely:
//   - A [16,192] and the block's B tile [192,16] are staged ONCE into LDS
//     with 16-B vector loads (global_load_dwordx4 -> ds_write_b128, 3 per
//     lane per tile), rows padded 192 -> 208 (kPaddedK = 192 + 16 bank-skew
//     stride, the TP4-validated down_proj stride),
//   - B is consumed from the n-major packed layout P[n*K+kk] = W[kk*N+n]
//     produced out of the timed region / Graph by launch_pack_w8a8_weight,
//     so each lane's B fragment is 8 contiguous bytes of one packed row,
//   - every m16n16k32 step then reads both fragments as one 8-B LDS read
//     (ds_read_b64) per lane and issues one v_mmac_i32_16x16x32_i8; no
//     vmcnt dependency exists inside the K loop (single wavefront, LDS only).
// The library fragment lane layout (row = lane&15, col = (lane>>4)*8) and
// byte order are reproduced exactly, so the int32 accumulation order is
// unchanged and outputs stay bit-identical to the accepted iteration-1
// kernel.
//
// Iteration 4 (this revision): architecture/pipeline round - multi-N-tile
// reuse. Each block now covers TWO adjacent 16-column N tiles (16x32 output)
// with the same single 64-lane wavefront: grid = 4096/32 = 128 blocks, still
// >= the 120 physical CUs so every CU keeps at least one block. The A tile
// [16,192] is staged ONCE per block (3 global_load_dwordx4 per lane, same as
// before) and every lane's 8-B A fragment for each of the 6 K steps is read
// from LDS ONCE and consumed by BOTH m16n16k32 MMACs (tile 0 and tile 1):
// A bytes are reused 2x per step in registers, so A global traffic and A LDS
// staging/read traffic halve vs two separate 16-col blocks. B bytes are NOT
// reused - each of the block's 32 packed-B rows is read exactly once per K
// step (8 B per lane per tile per step); B traffic per output column is
// unchanged. The two accumulator fragments are independent, so each K step
// issues two v_mmac_i32_16x16x32_i8 with no accumulator dependency between
// them (2x MMAC ILP vs the iteration-3 single serialized chain), while the
// dispatch block count halves (256 -> 128). The trusted occupancy-probe
// split candidates [2, 3] were evaluated for this axis: 2 fits (128 blocks
// >= 120 CUs, 32 divides N=4096, 16-B stage alignment preserved); 3 would
// give ceil(4096/48) = 86 blocks < 120 CUs and 48 does not divide N=4096,
// so 3 is excluded by the cover-all-CUs constraint. (The alternative reading
// of [2,3] as waves-per-block was already falsified in iteration 2:
// 2-wave/128-block regressed to 18.165 us median; 3 waves/block would give
// 86 blocks < 120 CUs.) Everything else is unchanged: same staged
// [16][208]-padded LDS layout, same 6-step k-ascending int32 accumulation
// per tile (bit-identical fragments, so outputs stay bit-identical to the
// accepted iteration-1/3 kernels), same library lane layout, same LDS
// accumulator round-trip epilogue.
//
// Iteration 11 (accepted, current best 8.4951 us): HIP-only consolidation -
// isolate the ONE untested component of the iteration-9 epilogue bundle
// (measured 10.7085 us, -1.13%, inside the +/-2% band): the 17-word (68-B)
// padded accumulator rows. Iteration 10 isolated the bundle's scale hoisting
// alone (10.7575 us, -1.58%, inside the band); this revision isolates the
// bank-conflict removal ALONE: the two 16x16 int32 tiles are stored manually
// with the library's verified fragment layout (word = (lane&15)*17 +
// (lane>>4) + 4*i, the exact mapping du_store_matrix_sync writes with stride
// 16) into 17-word rows so row r starts at bank r (17 is coprime with 32)
// instead of du_store_matrix_sync into the natural 16-word rows, where every
// row starts at bank 0 and every epilogue LDS access hits a 16-way conflict
// per phase (PMC of the accepted code object measures 32768
// lds_bank_conflicts = ~256/block, overwhelmingly from this round trip). The
// epilogue read mapping becomes row*17 + col (same values, same (row, col)
// positions); the accepted masked-branch scale loop, its in-loop x_scale
// loads (falsified as a separate axis in iteration 10), the scale order,
// output addresses, staging, K loop, int32 accumulation and the exact-shape
// guard are untouched, so outputs stay bit-identical. LDS 12032 -> 12160 B
// (still 5 blocks/CU of the 64 KiB), 0 s_barrier preserved (single
// wavefront, program-order stores then same-wave reads).
//
// Iteration 14 (this revision): HIP-only consolidation - merge the accepted
// two per-tile scale loops into ONE runtime masked loop that shares each
// pass's x_scale load across BOTH tiles. Exact-source ISA of the accepted
// object shows the epilogue as two separate runtime loops per block, each
// pass issuing ONE x_scale global_load_dword and then s_waitcnt vmcnt(0)
// before its store (0x2780/0x2798, 0x28A0/0x28B8) - 8 fully exposed L2-hot
// (64-B x_scale line) load latencies per block, and the per-pass x_scale row
// set (rows (lane>>4)+4*i) is IDENTICAL for both tiles (it does not depend
// on the tile index). The compiler cannot CSE across the two runtime loops,
// so the second tile re-loads the same four rows. The iteration-13
// straight-line rewrite (batched up-front loads + unrolled bodies) regressed
// to 11.0035 us (-22.8%) and iteration-10 register hoisting was flat on the
// old base - both falsified forms changed the per-pass wait structure and
// added live registers; this revision instead keeps the EXACT accepted
// per-pass structure (in-loop x_scale load, per-pass vmcnt(0) wait, masked
// 4-pass loop, no new registers) and only removes the cross-tile redundancy:
// one 4-pass loop, each pass loading x_scale[row_m] once and feeding both
// tiles' (acc * x_scale * weight_scale) bodies. Exposed epilogue vmcnt(0)
// waits halve 8 -> 4 per block, loop/exec-mask machinery halves, vmem read
// instructions 2432 -> 1920 (19 -> 15 loads/block: 9 staging + 4 x_scale + 2
// weight_scale), weight_scale stays LICM-hoisted (2/block), LDS 12160 B and
// 0 s_barrier unchanged. Same linear -> (row_m, col_n) mapping, same 17-word
// acc rows, same left-to-right fp32 order, same output addresses: outputs
// stay bit-identical.
//
// The scalar fallback kernel below remains the generic path for every
// unmatched (m, n, k), including the paired M=2 API shape with the same
// (N, K); it never enters the guarded DUMMA branch. Because the exact-shape
// pack is n-major, the scalar fallback decodes the packed B with a
// b_transposed flag whenever (n, k) == (4096, 192) (TP4 down_proj
// precedent); all other shapes keep the identity pack.
//
// Layout contract (logical, contiguous):
//   x_q          [M, K] int8, row-major, stride (K, 1)
//   packed_weight[K, N] int8 (identity pack for all shapes EXCEPT the exact
//                 (k, n) == (192, 4096), where it is the n-major transpose
//                 P[n*192 + kk] = W[kk*4096 + n]; same byte count, same
//                 graph-stable buffer, produced once out of the timed region)
//   x_scale      [M, 1] fp32
//   packed_weight_scale [N, 1] fp32
//   out          [M, N] bf16, row-major
//
// Mathematical reference (bit-exact for every supported shape because the
// full int8 dot fits exactly in int32 and every intermediate fp32 value is
// an exact integer below 2^24):
//   out[m, n] = bf16((int32_dot(x_q[m, :], packed_weight[:, n]))
//                    * x_scale[m] * packed_weight_scale[n])
//
// The timed operator (launch_w8a8_gemm) performs no allocation, compilation,
// autotuning, weight packing, host/device synchronization, or default-stream
// launch. It only launches kernels on the caller-provided stream and uses the
// caller-provided out tensor. Weight packing lives in the out-of-timed-region
// launch_pack_w8a8_weight (identity device-to-device copy).

#include <hip/hip_runtime.h>
#include <hip/hip_bfloat16.h>
#include <du_mma.h>

#include <cstdint>
#include <cstring>

namespace {

// gfx928 native wavefront is 64 lanes; the block size must be a multiple of 64.
constexpr int kBlockThreads = 256;

// DUMMA INT8 m16n16k32 tile constants (gfx928 native tensor core tile).
constexpr int kDummaWaveSize = 64;  // one wavefront per block
constexpr int kDummaM = 16;
constexpr int kDummaN = 16;
constexpr int kDummaK = 32;

// LDS row stride for the staged exact-shape tiles: K=192 padded by 16 to
// 208 (a 16-B-aligned non-power-of-two bank-skew stride; the same value the
// TP4 shared_down_proj lineage validated for its K=192 staging halves).
constexpr int kPaddedK = kDummaK * 6 + 16;  // 208 = 192 + 16

// ---------------------------------------------------------------------------
// Generic scalar GEMM kernel: one thread per output element.
//
// Thread mapping keeps adjacent lanes on adjacent addresses in the
// fastest-changing N dimension: threads with consecutive threadIdx.x handle
// consecutive columns of the same row, so the B column loads (stride 1) are
// fully coalesced and A row loads are broadcast within the warp.
//
// b_transposed != 0 selects the n-major packed B layout
// P[n*K + kk] = W[kk*N + n] used by the exact-shape (n, k) == (4096, 192)
// pack; the per-column load then becomes P[col*K + kk] with the same
// k-ascending int32 accumulation order (bit-identical results).
// ---------------------------------------------------------------------------
__global__ __launch_bounds__(kBlockThreads) void w8a8_gemm_scalar_kernel(
    const int8_t* __restrict__ a,           // [M, K] row-major
    const int8_t* __restrict__ b,           // [K, N] row-major, or [N, K]
                                            // n-major pack when transposed
    const float* __restrict__ x_scale,      // [M]
    const float* __restrict__ weight_scale, // [N]
    __hip_bfloat16* __restrict__ out,       // [M, N] row-major
    const int n,
    const int k,
    const int b_transposed) {
  const int col = blockIdx.x * blockDim.x + threadIdx.x;
  const int row = blockIdx.y;
  if (col >= n) {
    return;
  }

  const int8_t* a_row = a + static_cast<int64_t>(row) * k;

  // Complete K loop accumulated exactly in int32. The maximum supported K
  // keeps the exact int8 dot well inside the int32 range, so this matches the
  // fp32 reference bit-for-bit.
  int32_t acc = 0;
  if (b_transposed != 0) {
    // Packed n-major B: b[kk * n + col] == packed[col * k + kk].
    const int8_t* b_col = b + static_cast<int64_t>(col) * k;
    for (int kk = 0; kk < k; ++kk) {
      acc += static_cast<int32_t>(a_row[kk]) *
             static_cast<int32_t>(b_col[kk]);
    }
  } else {
    const int8_t* b_col = b + col;  // column n index, stride n
    for (int kk = 0; kk < k; ++kk) {
      acc += static_cast<int32_t>(a_row[kk]) *
             static_cast<int32_t>(b_col[static_cast<int64_t>(kk) * n]);
    }
  }

  // Convert to float only for the two scale multiplies (same left-to-right
  // order as the reference `(A@B) * x_scale * weight_scale.T`), then store
  // bf16 with the header's round-to-nearest-even float conversion.
  const float scaled = static_cast<float>(acc) * x_scale[row] * weight_scale[col];
  out[static_cast<int64_t>(row) * n + col] = __float2bfloat16(scaled);
}

// ---------------------------------------------------------------------------
// DUMMA INT8 m16n16k32 kernel for the exact assigned shape (M=16, N=4096,
// K=192). One 64-lane wavefront per block; each block computes TWO adjacent
// 16x16 output tiles (16x32 output, n0 = blockIdx.x * 32) for all 16 rows,
// accumulating the six 32-wide K tiles explicitly in int32 per tile.
//
// Iteration 4 multi-N-tile reuse: A and the block's two packed-B tiles are
// staged once into LDS with 16-B vector loads (3 per lane per tile); every K
// step then reads the A fragment once and each tile's B fragment as one 8-B
// LDS read (ds_read_b64) per lane and issues two independent
// v_mmac_i32_16x16x32_i8 (A fragment register-reused across both MMACs). No
// vmcnt dependency exists inside the K loop. See the file header for the
// full mechanism and the bit-identical fragment byte order.
// ---------------------------------------------------------------------------
__global__ __launch_bounds__(kDummaWaveSize) void
w8a8_gemm_dumma_m16n32k32_kernel(
    const int8_t* __restrict__ a,           // [16, K] row-major
    const int8_t* __restrict__ b,           // [N, K] n-major pack
                                            // P[n*K+kk] = W[kk*N+n]
    const float* __restrict__ x_scale,      // [16]
    const float* __restrict__ weight_scale, // [N]
    __hip_bfloat16* __restrict__ out,       // [16, N] row-major
    const int n,
    const int k) {
  const int lane = static_cast<int>(threadIdx.x) & (kDummaWaveSize - 1);
  // Two adjacent 16-column tiles per block: grid = N / 32 = 128 blocks.
  const int n0 = static_cast<int>(blockIdx.x) * (kDummaN * 2);

  // Exact-shape kernel (guard in launch_w8a8_gemm): K = 192 = 6 * 32. The
  // compile-time step count is required for full unrolling of the K loop.
  constexpr int kExactKSteps = kDummaK * 6 / kDummaK;  // 6
  if (k != kDummaK * kExactKSteps) {
    return;
  }

  __shared__ __align__(16) int8_t a_smem[kDummaM][kPaddedK];
  __shared__ __align__(16) int8_t b_smem[2][kDummaN][kPaddedK];
  __shared__ __align__(16) int32_t acc_tile[2][kDummaM * (kDummaN + 1)];

  // Cooperative global -> LDS staging with 16-B vector loads. All staged
  // tiles use the same [16][208] layout (rows padded 192 -> 208 bank skew):
  //   a_smem[m][kk]    = a[m][kk]          (A is shared by all blocks)
  //   b_smem[0][cn][kk] = P[n0 + cn][kk]        (packed B tile, rows n0..n0+15)
  //   b_smem[1][cn][kk] = P[n0 + 16 + cn][kk]  (packed B tile, rows n0+16..n0+31)
  // Each row splits into 12 chunks of 16 B; 64 lanes = 16 rows x 4 lanes per
  // row, each lane copies 3 chunks ({0,1,2}, {3,4,5}, {6,7,8} or {9,10,11}).
  const int srow = lane >> 2;                 // 0..15
  const int schunk = (lane & 3) * 3;          // 0, 3, 6, 9
  const int8_t* a_src = a + srow * k + schunk * 16;
  const int8_t* b0_src = b + (static_cast<int64_t>(n0) + srow) * k +
                         schunk * 16;
  const int8_t* b1_src = b + (static_cast<int64_t>(n0) + 16 + srow) * k +
                         schunk * 16;
  int8_t* a_dst = &a_smem[srow][schunk * 16];
  int8_t* b0_dst = &b_smem[0][srow][schunk * 16];
  int8_t* b1_dst = &b_smem[1][srow][schunk * 16];

  uint4 a_chunk[3];
  uint4 b0_chunk[3];
  uint4 b1_chunk[3];
#pragma unroll
  for (int p = 0; p < 3; ++p) {
    a_chunk[p] = *reinterpret_cast<const uint4*>(a_src + p * 16);
    b0_chunk[p] = *reinterpret_cast<const uint4*>(b0_src + p * 16);
    b1_chunk[p] = *reinterpret_cast<const uint4*>(b1_src + p * 16);
  }
#pragma unroll
  for (int p = 0; p < 3; ++p) {
    *reinterpret_cast<uint4*>(a_dst + p * 16) = a_chunk[p];
    *reinterpret_cast<uint4*>(b0_dst + p * 16) = b0_chunk[p];
    *reinterpret_cast<uint4*>(b1_dst + p * 16) = b1_chunk[p];
  }
  __syncthreads();

  du::dumma::DUFragment<du::dumma::matrix_a, kDummaM, kDummaN, kDummaK,
                        signed char, du::dumma::row_major> a_frag;
  du::dumma::DUFragment<du::dumma::matrix_b, kDummaM, kDummaN, kDummaK,
                        signed char, du::dumma::row_major> b0_frag;
  du::dumma::DUFragment<du::dumma::matrix_b, kDummaM, kDummaN, kDummaK,
                        signed char, du::dumma::row_major> b1_frag;
  du::dumma::DUFragment<du::dumma::accumulator, kDummaM, kDummaN, kDummaK,
                        int> acc0_frag;
  du::dumma::DUFragment<du::dumma::accumulator, kDummaM, kDummaN, kDummaK,
                        int> acc1_frag;
  du::dumma::du_fill_fragment(acc0_frag, 0);
  du::dumma::du_fill_fragment(acc1_frag, 0);

  // K loop: 192 / 32 = 6 DUMMA steps per tile, accumulated in the int32
  // fragments. The library fragment lane layout is row = lane&15,
  // col = (lane>>4)*8 with a.x[i] = a[row][k0+col+i] and
  // b.x[i] = b[k0+col+i][row]; all sequences are 8 contiguous bytes in the
  // staged/packed layouts, so one 8-B LDS read per lane reproduces each
  // fragment byte-identically. The A fragment is read once per step and
  // register-reused by BOTH independent MMAC chains (A bytes reused 2x per
  // step); the two accumulators have no dependency, so the two
  // v_mmac_i32_16x16x32_i8 per step pipeline against each other. All loads
  // are LDS: no vmcnt dependency and no global traffic in the loop.
  const int row = lane & 15;
  const int col = (lane >> 4) * 8;
  const int frag_off = row * kPaddedK + col;
#pragma unroll
  for (int s = 0; s < kExactKSteps; ++s) {
    const int off = frag_off + s * kDummaK;
    uint64_t a_val;
    uint64_t b0_val;
    uint64_t b1_val;
    memcpy(&a_val, a_smem[0] + off, sizeof(a_val));
    memcpy(&b0_val, b_smem[0][0] + off, sizeof(b0_val));
    memcpy(&b1_val, b_smem[1][0] + off, sizeof(b1_val));
    memcpy(a_frag.x, &a_val, sizeof(a_val));
    memcpy(b0_frag.x, &b0_val, sizeof(b0_val));
    memcpy(b1_frag.x, &b1_val, sizeof(b1_val));
    du::dumma::du_mma_sync(acc0_frag, a_frag, b0_frag, acc0_frag);
    du::dumma::du_mma_sync(acc1_frag, a_frag, b1_frag, acc1_frag);
  }

  // Materialize both 16x16 int32 tiles in shared memory with the library's
  // verified fragment layout - lane l owns acc[row = lane&15][col = (lane>>4)
  // + 4*i] for i = 0..3, the exact word mapping du_store_matrix_sync writes
  // (word = row*16 + col in the accepted 16-word rows) - but stored into
  // 17-word (68-B) padded rows so row r starts at bank r: the natural 64-B
  // row stride starts every row at bank 0, forcing a 16-way bank conflict per
  // phase (the measured 32768 lds_bank_conflicts = ~256/block on the accepted
  // code object come overwhelmingly from this round trip); a 17-word stride
  // (17 coprime with 32) spreads the 16 rows over 16 distinct banks and drops
  // every epilogue LDS access (4 ds_write2_b32 + 8 ds_read_b32 per block) to
  // the ~2-4-way bandwidth floor. Same stored int32 values, same (row, col)
  // positions, same single-wavefront program order (stores, then
  // __syncthreads that elides to 0 s_barrier, then same-wave reads), so the
  // scale loop below reads back bit-identical values.
  constexpr int kAccRowWords = kDummaN + 1;  // 17-word padded rows
  const int acc_row = lane & 15;             // fragment row
  const int acc_col4 = lane >> 4;            // fragment col group
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    acc_tile[0][acc_row * kAccRowWords + acc_col4 + 4 * i] = acc0_frag.x[i];
    acc_tile[1][acc_row * kAccRowWords + acc_col4 + 4 * i] = acc1_frag.x[i];
  }
  __syncthreads();

  // Epilogue (iteration 14): ONE runtime masked loop for both tiles. Each
  // pass loads x_scale[row_m] once (the per-pass row set (lane>>4)+4*i is
  // tile-independent) and feeds both tiles' scale/store bodies, halving the
  // exposed global-load-latency waits (8 -> 4 per block) and the loop/
  // exec-mask machinery vs the accepted two separate tile loops, while
  // keeping the exact accepted per-pass structure (in-loop x_scale load,
  // per-pass s_waitcnt vmcnt(0), no new live registers - the iteration-10
  // register hoisting and iteration-13 straight-line forms were both
  // falsified). Same linear -> (row_m, col_n) mapping, same 17-word acc
  // rows, same left-to-right (acc * x_scale * weight_scale) fp32 order and
  // same output addresses: outputs stay bit-identical.
#pragma unroll
  for (int linear = lane; linear < kDummaM * kDummaN;
       linear += kDummaWaveSize) {
    const int row_m = linear >> 4;  // / kDummaN
    const int col_n = linear & 15;  // % kDummaN
    const int acc_off = row_m * kAccRowWords + col_n;
    const float xs = x_scale[row_m];
    const float scaled0 =
        static_cast<float>(acc_tile[0][acc_off]) * xs *
        weight_scale[n0 + col_n];
    out[static_cast<int64_t>(row_m) * n + n0 + col_n] =
        __float2bfloat16(scaled0);
    const float scaled1 =
        static_cast<float>(acc_tile[1][acc_off]) * xs *
        weight_scale[n0 + kDummaN + col_n];
    out[static_cast<int64_t>(row_m) * n + n0 + kDummaN + col_n] =
        __float2bfloat16(scaled1);
  }
}

// ---------------------------------------------------------------------------
// N-major transpose pack for the exact assigned shape (k == 192, n == 4096):
//   packed[n_idx * k + kk] = weight[kk * n + n_idx]
// Runs once out of the timed region / Graph (weight preprocessing; same byte
// count, graph-stable buffer). Each 256-thread block covers one 16-column
// tile: threads 0-15 own the 16 columns and each 16-thread group copies 12
// consecutive K rows (12 * 16 = 192 = K exactly).
// ---------------------------------------------------------------------------
__global__ __launch_bounds__(kBlockThreads) void w8a8_pack_nmajor_kernel(
    const int8_t* __restrict__ w,   // [K, N] row-major raw weight
    int8_t* __restrict__ packed,    // [N, K] n-major packed weight
    const int n,
    const int k) {
  const int n0 = static_cast<int>(blockIdx.x) * 16;
  const int n_idx = n0 + (static_cast<int>(threadIdx.x) & 15);
  const int k_base = (static_cast<int>(threadIdx.x) >> 4) * 12;  // 0..180
#pragma unroll
  for (int i = 0; i < 12; ++i) {
    const int kk = k_base + i;
    packed[static_cast<int64_t>(n_idx) * k + kk] =
        w[static_cast<int64_t>(kk) * n + n_idx];
  }
}

// ---------------------------------------------------------------------------
// Identity device-to-device copy used by the bootstrap pack_weight op.
// ---------------------------------------------------------------------------
template <typename T>
__global__ __launch_bounds__(kBlockThreads) void w8a8_identity_copy_kernel(
    const T* __restrict__ src,
    T* __restrict__ dst,
    const int64_t numel) {
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < numel) {
    dst[i] = src[i];
  }
}

}  // namespace

// ---------------------------------------------------------------------------
// Timed GEMM launcher (stable host symbol required by csrc/bindings.cpp).
//
// Exact-shape guard first: only (m, n, k) == (16, 4096, 192) enters the
// assigned-shape DUMMA branch. The paired M=2 API shape with the same (N, K)
// (tp8_shared_down_proj_m2) never matches the guard and reaches the generic
// scalar path below, as required.
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

  if (m == 16 && n == 4096 && k == 192) {
    // Exact assigned shape: hy3_tp8_shared_down_proj_m16.
    // Grid: 4096 / 32 = 128 one-wave blocks, each computing a 16x32 output
    // (two adjacent 16x16 N tiles sharing one staged A tile).
    // `b` is the n-major packed weight produced by launch_pack_w8a8_weight
    // for this exact (k, n).
    const dim3 block(kDummaWaveSize);
    const dim3 grid(n / (kDummaN * 2));
    hipLaunchKernelGGL(HIP_KERNEL_NAME(w8a8_gemm_dumma_m16n32k32_kernel),
                       grid, block, 0, stream, a, b, x_scale, weight_scale,
                       out_bf16, n, k);
    return;
  }

  // Generic scalar fallback for every unmatched (m, n, k), including the
  // paired M=2 API shape with the same (N, K). Whenever (n, k) == (4096, 192)
  // the packed B is the n-major transpose, so the scalar path decodes it via
  // the b_transposed flag; every other shape keeps the identity pack.
  const dim3 block(kBlockThreads);
  const dim3 grid((n + kBlockThreads - 1) / kBlockThreads, m);
  const int b_transposed = (n == 4096 && k == 192) ? 1 : 0;
  hipLaunchKernelGGL(HIP_KERNEL_NAME(w8a8_gemm_scalar_kernel), grid, block, 0,
                     stream, a, b, x_scale, weight_scale, out_bf16, n, k,
                     b_transposed);
}

// ---------------------------------------------------------------------------
// Optional out-of-timed-region weight packing launcher (stable host symbol
// required by csrc/bindings.cpp).
//
// For the exact assigned (k, n) == (192, 4096) it packs raw_weight [K, N]
// int8 -> packed_weight [N, K] n-major transpose P[n*K+kk] = W[kk*N+n] (the
// layout consumed by the exact-shape DUMMA kernel and the b_transposed
// scalar fallback). Every other (K, N) keeps the identity device-to-device
// copy raw_weight [K, N] -> packed_weight [K, N]. weight_scale [N] fp32 ->
// packed_weight_scale [N] fp32 is always the identity copy. Both run once
// out of the timed region / Graph with graph-stable buffer addresses.
// ---------------------------------------------------------------------------
extern "C" void launch_pack_w8a8_weight(
    const int8_t* raw_weight,
    const float* weight_scale,
    int8_t* packed_weight,
    float* packed_weight_scale,
    int k,
    int n,
    hipStream_t stream) {
  const int64_t weight_numel = static_cast<int64_t>(k) * n;
  if (k == 192 && n == 4096) {
    // Exact assigned shape: n-major transpose pack (grid = N/16 tiles).
    const dim3 block(kBlockThreads);
    const dim3 grid(n / 16);
    hipLaunchKernelGGL(HIP_KERNEL_NAME(w8a8_pack_nmajor_kernel), grid, block,
                       0, stream, raw_weight, packed_weight, n, k);
  } else if (weight_numel > 0) {
    const dim3 block(kBlockThreads);
    const dim3 grid(static_cast<unsigned>(
        (weight_numel + kBlockThreads - 1) / kBlockThreads));
    hipLaunchKernelGGL(HIP_KERNEL_NAME(w8a8_identity_copy_kernel<int8_t>),
                       grid, block, 0, stream, raw_weight, packed_weight,
                       weight_numel);
  }
  if (n > 0) {
    const dim3 block(kBlockThreads);
    const dim3 grid(
        static_cast<unsigned>((n + kBlockThreads - 1) / kBlockThreads));
    hipLaunchKernelGGL(HIP_KERNEL_NAME(w8a8_identity_copy_kernel<float>), grid,
                       block, 0, stream, weight_scale, packed_weight_scale,
                       n);
  }
}
// @@end
