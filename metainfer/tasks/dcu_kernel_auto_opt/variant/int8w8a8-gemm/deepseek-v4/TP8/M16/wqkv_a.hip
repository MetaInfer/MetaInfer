// @@variant shape=tp8_wqkv_a_m16 commit=b67e227c20b3349d46020f9f52cd2484b2142ff3 added=2026-08-26
//   median_us=17.5 p90_us=17.52
//   source=hy3-dsh-tp8-m16-1-adf021ab
// MetaInfer W8A8 GEMM - worker_0 HIP implementation (gfx928).
//
// Task: hy3-dsh-tp8-m16-1-adf021ab, worker_0 owns shape
//   tp8_wqkv_a_m16: M=16, N=1536, K=4096  (physical GPU 0)
//
// Iteration 1 established the minimal gfx928 DUMMA INT8 tile for the
// assigned shape (one 64-thread wavefront per 16x16 N tile, grid=96,
// direct row-major fragment loads, direct register epilogue). It was
// accepted at 194.6 us median (vs 588.9 us scalar bootstrap), but its
// exact-source ISA is load-wait bound: per 32-wide K step it emits ~17
// global_load_ubyte + 6 global_load_dword with per-load vmcnt waits in
// front of a single v_mmac_i32_16x16x32_i8.
//
// Iteration 2 (accepted, 142.46 us median / 143.91 us p90) was the first
// mandated ARCHITECTURE round: WAVES=2 (blockDim=128), in-block split-K=2
// along K (wave 0: [0,2048), wave 1: [2048,4096)), one 16-wide N tile per
// block, grid = N/16 = 96 blocks (0.8 of 120 CUs), LDS int32 plane combine
// (2 x 16 x 20), one END-of-K barrier, direct register epilogue. Its
// exact-source ISA still shows the iteration-1 per-load vmcnt cascade
// (~17 global_load_ubyte + 6 global_load_dword per K step) in front of one
// v_mmac_i32_16x16x32_i8 per step; the second wavefront per CU recovered
// 194.6 -> 142.5 us but the kernel remains load-latency bound.
//
// Iteration 3 is the second mandated ARCHITECTURE round: the unsplit grid
// (96 blocks) has fewer than two blocks per device CU and K=4096 >= 1024,
// so the K loop is now partitioned across CROSS-BLOCK splits:
//   * partial kernel: one 64-thread wavefront per block, ZERO barriers,
//     ZERO LDS. grid = (N/16) * split_k blocks; blockIdx.x encodes
//     split-major / ntile-minor (ntile = x % ntiles, split = x / ntiles).
//     split_k is runtime-selectable in [1, min(16, K/32)] (default 2; env
//     override METAINFER_W8A8_SPLIT_K selects any trusted candidate). K is
//     tiled into 32-aligned step-based slices
//     [(s*steps)/split_k, ((s+1)*steps)/split_k) that cover [0, K) exactly
//     once ascending (non-uniform for non-power-of-two split_k, e.g.
//     split_k=5 -> 25/26/25/26/26 steps), so every split boundary stays
//     aligned to the DUMMA k32 unit. Each block keeps the unchanged
//     k-ascending du_load_matrix_sync + du_mma_sync loop over its slice
//     (per-slice int32 partials bit-identical to iterations 1-2) and
//     publishes the int32 tile with du_store_matrix_sync (mem_row_major,
//     ldm=16) into caller workspace plane (split, ntile) at
//     partials[(split*ntiles + ntile) * 16 * 16].
//   * combine+scale kernel: grid = N/16 blocks x 256 threads (4
//     wavefronts), 1 element per thread; sums the split_k planes in
//     ascending split order (bit-identical int32 to the serial k-ascending
//     accumulation - integer addition is exact and
//     |full dot| <= 4096*127*127 ~= 6.6e7 << 2^31), applies
//     x_scale[m] * weight_scale[n], stores bf16. Both kernels launch on
//     the caller stream inside the timed Graph, so the combine cost is
//     included in the measured median/P90.
//   * split_k=2 -> grid 192 blocks (1.6 blocks/CU); the CU-aligned
//     non-power-of-two candidate split_k=5 -> grid 480 = 4 x 120 CUs
//     exactly is the same code path with a different split count (trusted
//     occupancy-probe candidates {2,3,5,6,8,10,16} are all env-selectable).
//   * expected mechanism: cross-block splits multiply the resident
//     one-wave blocks per CU (each block is an independent load/MMAC
//     stream that covers sibling blocks' vmcnt stalls), the partial
//     kernel has no barrier or LDS combine on its critical path, and the
//     combine kernel is a tiny 96-block latency-tolerant tail. Exact bf16
//     equality is expected (identical int32 accumulation order:
//     k-ascending per split, ascending split sum).
//
// Iteration 4 is the mandated A-only-staging pipeline round. The
// iteration-3 partial kernel's exact-source PMC (192 blocks, split_k=2)
// shows 16 vmem-read instructions and ~63 issued VALU per 32-wide K step
// with per-load vmcnt waits in front of the single v_mmac - the same
// byte-load fragment reassembly poison as iterations 1-2 (A and B
// fragments each expand to ~8-12 scalar global loads). The chosen change
// is A-ONLY staging (B stays direct; B vectorization/packing is the later
// round): each one-wave block stages its own A slice [16, slice_k] once at
// block start into LDS with vectorized 16-byte int4 global loads and
// ds_write_b128 stores (row stride = slice_k + 16 bytes: 16-B-aligned,
// non-power-of-two bank skew, down_proj-validated rule), and the K loop
// then reads the A fragment from LDS via du_load_matrix_sync(smem, ldm =
// local_stride) - A's ~8-12 global loads + vmcnt waits per step leave the
// loop, replaced by low-latency ds_read. Default split_k is raised 2 -> 5:
// grid = 96 x 5 = 480 one-wave zero-barrier blocks = exactly 4 x 120 CUs,
// each block's stage is 16 x (832 + 16) = 13,568 B dynamic LDS so four
// blocks fit per CU (54,272 B <= 64 KiB), directly answering the
// control-plane warning that 192 blocks is below the two-blocks-per-CU
// latency-hiding target (480 = 4 blocks/CU). A bytes (64 KiB total,
// L2-hot) are still re-read once per N-tile block (96x per split pass) but
// as one vectorized burst per block instead of per-step scalar loads; B
// bytes (6.29 MiB cold) are still read exactly once per owning
// (split, ntile) block, unchanged. Exact bf16 equality is preserved: the
// staged data is the same A bytes in the same row-major order and the
// int32 accumulation order is unchanged (k-ascending per split, ascending
// split sum).
//
// Iteration 5 is the mandated HIP-only packed-weight/staging comparison
// round (chosen design: ONE PACKED LAYOUT - B fragment slots; the A-only
// LDS staging of iteration 4 stays as-is; B-only staging was not chosen
// because it would add a second ~13.3 KiB LDS slice per block and drop
// co-residency from 4 to 2 blocks/CU). The identity pack_weight bootstrap
// becomes a real one-time device permutation for the exact
// (K, N) == (4096, 1536) weight:
//   packed[n_tile][k_step][lane][8]: slot
//     ((ntile * (k/32) + step) * 64 + lane) * 8 + i
//   holds logical weight[step*32 + (lane>>4)*8 + i][ntile*16 + (lane&15)].
// Byte count is unchanged (K*N = 6,291,456 B), so the packed buffer keeps
// the caller's numel and graph-stable address; the pack runs outside the
// timed region and Graph capture (prepare_weight path).
// The pack makes each lane's matrix_b m16n16k32 row_major fragment for one
// k32 step ONE contiguous aligned 8-byte (dwordx2) global load in the exact
// du_mma.hpp byte order (x[i] = p[(col + i)*ldm + row], row = lane&15,
// col = (lane>>4) << 3; validated o_proj lineage pattern): B's per-step
// scalar byte loads + fragment reassembly + per-load vmcnt cascade (8 vmem
// instructions and ~50 VALU per step in the iteration-4 exact ISA) leave
// the K loop, which becomes ds_read (A, from LDS) -> 1 dwordx2 (B, packed)
// -> v_mmac. The DUMMA kernels and the generic scalar fallback decode the
// SAME (tile, k_step, lane, byte) -> logical (k, n) mapping (coherent
// repack, o_proj-validated rule), so the paired M=2 fallback validation
// stays exact. Geometry is unchanged from iteration 4: split_k=5, grid 480
// = 4 x 120 CUs exactly, one-wave zero-in-loop-barrier blocks, 13,568 B
// dynamic LDS per block, separate combine kernel in the timed Graph. Exact
// bf16 equality is preserved: every fragment feeds the same bytes to the
// same v_mmac in the same order, and the int32 accumulation order is
// unchanged (k-ascending per split, ascending split sum).
//
// Iteration 6 is the mandated pipeline/staging-family round: the
// iteration-5 kernel's exact-source ISA (primary kernel
// w8a8_dumma_m16n16k32_sk_astage_packedb_partial_kernel, 480 blocks, 16
// VGPR/32 SGPR) shows the steady loop as
// global_load_dwordx2 (B, packed) -> ds_read2_b32 (A, from LDS) ->
// s_waitcnt lgkmcnt(0) -> s_waitcnt vmcnt(0) -> v_mmac - one exposed
// vmcnt(0) wait in front of every v_mmac for the cold once-read B stream
// (B = 6.29 MB, read exactly once; L2 hit rate 53.3%, l2_misses 107,607),
// so the per-step critical path is B's global latency, not issue rate
// (18,528 vmem reads = exactly 1 B dwordx2 per k32 step + the one-time A
// stage; 12,288 loop steps). The chosen staging family is A+B LDS with a
// stage-K=64 DOUBLE BUFFER (the validated winner mechanism of the TP4
// reference for this exact (M=16, K=4096, N=1536) shape: 17.344 us Graph):
//   * the K slice is processed in 2-step (K=64) chunks; dynamic LDS holds
//     two ping-pong buffers of 2,560 B each (A 16x64 with a padded 96-B row
//     stride - 8-B A-fragment reads at the 2-way LDS floor vs the ~4.4
//     conflicts of the stride-848 layout - plus B 1 KiB of packed fragment
//     slots), 5,120 B per block total (constant, independent of split_k),
//     so the default split_k=5 grid 480 = 4 x 120 CUs still fits 4
//     blocks/CU (20,480 B <= 64 KiB);
//   * per chunk each lane issues 3 cooperative global loads (1 x 16-B A
//     dwordx4 + 2 x 8-B B dwordx2) into registers BEFORE the current
//     chunk's 1-2 v_mmac_i32_16x16x32_i8 and commits them to the alternate
//     LDS buffer AFTER the MMAC burst, so the vmcnt wait lands at the
//     ds_write and the cold B latency overlaps the current chunk's compute
//     (the harness ISA extraction must show next-chunk global loads before
//     the current v_mmac and the wait near the alternate-buffer ds_write;
//     the software pipeline is NOT claimed from HIP source alone);
//   * the steady K loop then has ZERO global loads: per k32 step ds_read
//     (A) -> ds_read (B) -> v_mmac. The zero-LDS direct-A packed-B kernel
//     (split_k=1 64-KiB overflow fallback) and the A-only astage packed-B
//     kernel are preserved as symbols but no longer dispatched. Exact bf16
//     equality is preserved: staged bytes are the same data fed to the same
//     v_mmac in the same order; int32 accumulation order unchanged
//     (k-ascending per split, ascending split sum).
//   * iteration-6 repair 1: the per-chunk step count was clamped to
//     kStageSteps. The original `nsteps = slice_steps - s0` is the remaining
//     count and is 3-4 for the second-to-last chunk (25/26 - 2*11 at
//     split_k=5), so the j-loop read A/B beyond the staged 64-column chunk
//     (stale LDS) and polluted every accumulator. Fix: nsteps =
//     min(kStageSteps, slice_steps - s0) -> 1 or 2, matching the staged
//     chunk width; int32 accumulation order unchanged.
//   * iteration-6 repair 2: the B slot base must index the slice's first
//     GLOBAL k32 step (k_lo/kDummaTileK) because the packed
//     [n_tile][k_step][lane][8] layout is indexed by global step (pack
//     kernel slot = ntile*k_steps + step, step in [0, K/32)). The original
//     b_slot_base pointed at global step 0 and added only the slice-relative
//     chunk/step offsets, so every split with k_lo > 0 (splits 1-4 of the
//     default split_k=5) fed B bytes of global steps [0, slice_steps) into
//     the accumulator while A was staged correctly - the dominant
//     near-total mismatch (24,538/24,576) that survived repair 1. Fix:
//     b_slot_base += (k_lo / kDummaTileK) * kPackedTileBytes; A staging
//     (a_base = k_lo) and the int32 accumulation order are unchanged.
//
// Iteration 7 is the mandated pipeline round: COMPARE SINGLE buffering with
// DOUBLE buffering (all three gates hold for this shape: K=4096 >= 1024,
// L2 hit rate 54.2% < 70%, and the doubled LDS budget 2 x 2,560 = 5,120 B
// < 48 KiB). Iteration 6 already measured and accepted the DOUBLE-buffered
// stage-K=64 arm (18.551 us median / 18.583 us p90; 480 one-wave
// zero-barrier blocks = 4 x 120 CUs; 5,120 B dynamic LDS/block; the
// per-chunk register prefetch issues chunk c+1's 3 cooperative global loads
// before chunk c's MMAC burst and the vmcnt wait lands at the post-MMAC
// alternate-buffer ds_write). This round measures the SINGLE-buffered arm:
//   * the two ping-pong 2,560-B buffers collapse into ONE 2,560-B buffer -
//     the chunk c+1 prefetch still goes through registers and commits AFTER
//     the current chunk's MMAC burst, but now to the same LDS addresses the
//     current chunk was just read from (write-after-read ordering is program
//     order in the single wave; the compiler keeps the same
//     global_load_dwordx4/dwordx2-before-v_mmac and vmcnt-wait-at-ds_write
//     placement, with the ds_write at a FIXED base instead of the
//     (c&1)*0xa00 toggle);
//   * dynamic LDS drops 5,120 -> 2,560 B per block (per CU 20,480 ->
//     10,240 B; occupancy is unchanged because the grid is 480 blocks =
//     4 blocks/CU by design and neither footprint binds the 64 KiB LDS);
//   * barriers per K step: 0 in BOTH variants (one 64-lane wave per block,
//     no cross-wave LDS producer/consumer - the pipeline synchronization is
//     s_waitcnt vmcnt/lgkmcnt only; the TP4 4-wave reference needs ~2
//     barriers per stage-K=64 chunk, i.e. ~1 per K step, for cross-wave LDS
//     visibility, which does not apply to the one-wave blocks here);
//   * launch geometry (split_k=5, grid 480, one wave/block, block-N=16),
//     K slicing, packed-B fragment-slot layout, int32 partial publish,
//     separate combine kernel in the timed Graph, and the exact bf16
//     accumulation order are all unchanged. The iteration-6 double-buffered
//     kernel stays in the source as a preserved symbol but is no longer
//     dispatched.
//
// Iteration 14 (shadow, not accepted) fused the iteration-3 combine/scale
// kernel into the iteration-7 single-buffered partial kernel as a
// qkv-proven last-arrival atomic tail (monotonic counters in the last
// counter_bytes of the caller workspace, one launch per Graph replay). It
// measured 18.382 us median / 18.409 us p90 - only +0.55% over the 18.484
// us iteration-7 shadow - proving the ~2 us partial-to-Graph gap was the
// FUSED TAIL's own fence/atomic/tail cost, not the combine kernel's
// launch+L2 round trip (consistent with the TP4 lineage where the atomic
// last-arriver fused finalize on the 4-wave design measured +1.52 us).
//
// Iteration 15 (this round, HIP-only; plateau not proven - the last three
// valid improvements were -50.78% / -6.65% / +0.92%) adopts the
// TP4-VALIDATED 4-wave design for this exact (M=16, K=4096, N=1536) local
// shape, measured at 17.344 us median / 17.616 us p90 on the same gfx928:
//   * block-N=64 (4 x wave64, 256 threads), one 16x16 quadrant per wave,
//     all four waves SHARE one staged A chunk (A logical rereads drop
//     6.4 MB -> 1.57 MB; each block stages A once per K=64 chunk instead of
//     once per 16-wide tile);
//   * stage-K=64 DOUBLE buffer, 10,240 B/block; stage i+1's global int4
//     loads issue before stage i's 2 x v_mmac per wave and commit to the
//     alternate buffer with the vmcnt wait at the ds_write; two
//     __syncthreads() per stage (pure-HIP form of the TP4 raw-first-
//     barrier sync mode - NO raw asm this round);
//   * B is staged from THIS session's packed [n_tile][k_step][lane][8]
//     fragment-slot layout (one 16-B int4 per lane = two adjacent 8-B
//     slots, one 16-B ds_write_b128 into an LDS mirror of the slot layout)
//     instead of the TP4 raw [K,N] row loads - same staged bytes, same
//     fragment reads (one ds_read_b64 per lane per k32 step);
//   * non-uniform stage-based K slicing, default split_k=10 -> grid =
//     24*10 = 240 blocks = exactly 2 x 120 CUs;
//   * the separate 96-block x 256-thread combine kernel returns to the
//     timed Graph (the TP4 lineage measured the fused finalize at +1.52 us
//     on this design, so the iteration-14 fused 1-wave kernel is now a
//     preserved symbol);
//   * exact bf16: same staged bytes, same k-ascending per-split /
//     ascending-split-sum int32 order.
//
// Iteration 16 (this round, HIP-only repair of the iteration-15 candidate):
// the iteration-15 4-wave kernel measured 60.158 us (single sample) with
// 24,555/24,576 mismatches. The SOLE correctness defect was the B-staging
// global address in w8a8_dumma_m16n16k32_4wave_sk_astage_packedb_dbuf_
// partial_kernel: each stage chunk was read as one contiguous 4,096-B
// window of the packed [n_tile][k_step][lane][8] layout, but that layout is
// indexed by the GLOBAL step (each ntile's 128 steps form a contiguous
// 64 KiB run), so a 4-ntile x 2-step chunk is NOT contiguous - only wave 0
// of chunk 0 read correct bytes. The repair changes ONLY the B global
// offsets: each thread's 16-B slot pair is addressed with the exact
// per-(ntile, step) packed offset ((4*tile64 + w)*k_steps + k_lo/32 +
// 2c + s)*512 + (tid&31)*16, and the chunk stride becomes
// kStageSteps*kPackedTileBytes = 1,024 B. The fast mapping is preserved
// byte-for-byte: same 4-wave block-N=64 stage-K=64 double-buffer pipeline,
// same one 16-B global load + one 16-B ds_write_b128 per thread per chunk
// into the unchanged slot-mirror LDS layout, same two __syncthreads() per
// stage, same fragment reads (one ds_read_b64 per lane per k32 step), same
// plane publish, same separate 96-block x 256-thread combine kernel in the
// timed Graph, same shape guards and generic fallbacks. Exact bf16:
// identical staged bytes fed to the same v_mmac in the same order; int32
// accumulation order unchanged (k-ascending per split, ascending split
// sum). Falsifiable gate vs the 18.382 us iteration-14 shadow: if the
// TP4-validated occupancy transfers once correctness is restored, the
// median lands ~17.3-17.7 us; if the 2-blocks/CU co-residency or the
// 2-barrier-per-stage sync costs more on this packed-B path, it lands at or
// above 18.382.
//
// Dispatch policy:
//   * The DUMMA fast path is reachable only for the exact assigned shape;
//     every other (m, n, k) falls back to the generic scalar kernel.
//   * launch_pack_w8a8_weight performs a real fragment-slot pack for the
//     exact (K, N) == (4096, 1536) weight and stays an identity
//     device-to-device copy for every other (K, N).
//   * The timed operator uses only the caller-provided out/workspace (the
//     split-K partial + combine path stores int32 partials in the caller
//     workspace), launches only on the caller's hipStream_t, and performs
//     no allocation, compilation, autotuning, packing, or host/device
//     synchronization. If the caller provides fewer than one int32 plane
//     (16*16*4 bytes), the previous workspace-free in-block split-K=2
//     kernel (iteration 2) keeps the fast path correct.

// Known-good include order for this DTK: HIP runtime first, then
// hip_bfloat16.h, then du_mma.h (du_mma.h is not self-contained if it is
// included before the HIP runtime headers).
#include <hip/hip_runtime.h>
#include <hip/hip_bfloat16.h>
#include <du_mma.h>

#include <cstdint>
#include <cstdlib>

namespace {

// One wavefront on gfx928 is 64 lanes; blockDim must be a multiple of 64.
constexpr int kScalarBlockThreads = 128;
constexpr int kCopyBlockThreads = 256;

// Minimal DUMMA INT8 tile: m16n16k32 with int32 accumulation.
constexpr int kDummaTileM = 16;
constexpr int kDummaTileN = 16;
constexpr int kDummaTileK = 32;
constexpr int kDummaWaveSize = 64;

// Exact assigned shape (M=16, N=1536, K=4096) used by the packed-B
// interpretation and dispatch guards.
constexpr int kTargetK = 4096;
constexpr int kTargetN = 1536;

// Iteration 5 packed-B fragment-slot layout (exact shape only):
//   packed[n_tile][k_step][lane][8]: slot
//     ((ntile * (k/32) + step) * 64 + lane) * 8 + i
//   holds logical weight[step*32 + (lane>>4)*8 + i][ntile*16 + (lane&15)] -
//   the 8 bytes of one lane's matrix_b m16n16k32 row_major fragment in the
//   exact du_mma.hpp byte order (x[i] = p[(col + i) * ldm + row], row =
//   lane&15, col = (lane>>4) << 3). Byte count is unchanged (K*N), so the
//   packed buffer keeps the caller's numel and graph-stable address.
constexpr int kPackedSlotBytes = 8;   // bytes of one lane's B fragment
constexpr int kPackedTileBytes = kDummaWaveSize * kPackedSlotBytes;  // 512

// Iteration 6 stage-K=64 double-buffer constants (exact shape only). One
// stage chunk covers 2 k32 steps (K=64): A 16 rows x 64 B row-major with a
// padded 96-B row stride (16-B-aligned non-power-of-two bank skew; 96/4 =
// 24 dwords, 24*row mod 64 spreads the 16 rows onto 16 bank phases with the
// 8-B A-fragment reads at the 2-way LDS floor, vs the ~4.4 conflicts of the
// iteration-4/5 stride-848 layout) and B 1 KiB of packed fragment slots
// (8 B per lane per step). Two ping-pong buffers = 5,120 B per block, so
// four blocks per CU (20,480 B <= 64 KiB) at the default split_k=5 grid.
constexpr int kStageSteps = 2;              // k32 steps per stage chunk
constexpr int kStageK = kDummaTileK * kStageSteps;  // 64 K-cols per chunk
constexpr int kStageAStride = 96;           // padded A row stride (bytes)
constexpr int kStageABufBytes =
    kDummaTileM * kStageAStride;                      // 1536
constexpr int kStageBBufBytes =
    kDummaWaveSize * kPackedSlotBytes * kStageSteps;  // 1024
constexpr int kStageBufBytes = kStageABufBytes + kStageBBufBytes;  // 2560
// Iteration 6 double buffer: two ping-pong buffers = 5,120 B per block.
constexpr int kStageTotalBytes = 2 * kStageBufBytes;               // 5120
// Iteration 7 single-buffer arm: ONE 2,560-B buffer per block (the sbuf
// kernel below passes kStageBufBytes as its dynamic LDS size).

// Iteration 15 4-wave block-N=64 stage-K=64 DOUBLE-buffer constants (the
// TP4-validated design for this EXACT local shape (M=16, K=4096, N=1536):
// 17.344 us median / 17.616 us p90 on the same gfx928 at split_k=10, grid
// 240 = exactly 2 x 120 CUs, 4 wave64 per block, one 16x16 quadrant per
// wave sharing one staged A chunk; see
// references/w8a8_gemm_variants.hip and the split10 reference). Per stage
// chunk (K=64): A raw [16][64] staged by wave 0 (1,024 B, stride 64) + B
// staged as a 4,096-B mirror of the packed fragment-slot layout (4 ntiles
// x 2 steps x 512 B); two ping-pong buffers = 10,240 B per block (2,560 B
// per wave-pair-of-CUs at 2 blocks/CU = 20,480 B/CU <= 64 KiB).
constexpr int k4wBlockNTiles = 4;                 // block-N = 64
constexpr int k4wBlockThreads = k4wBlockNTiles * kDummaWaveSize;  // 256
constexpr int k4wStageABytes = kDummaTileM * kStageK;             // 1024
constexpr int k4wStageBBytes =
    k4wBlockNTiles * kStageSteps * kPackedTileBytes;              // 4096
constexpr int k4wStageBufBytes = k4wStageABytes + k4wStageBBytes; // 5120
constexpr int k4wStageTotalBytes = 2 * k4wStageBufBytes;          // 10240

// ---------------------------------------------------------------------------
// Iteration 5 packed-B helpers (exact shape (K, N) == (4096, 1536) only).
// ---------------------------------------------------------------------------

// Elementwise decode of the packed [n_tile][k_step][lane][8] fragment-slot
// layout: returns logical weight[kk, n_col] with the same (tile, k_step,
// lane, byte) -> logical (k, n) mapping as the pack kernel and the DUMMA
// kernels. Used by the generic scalar fallback (M=2 paired validation).
__device__ __forceinline__ int8_t
packed_b_element(const int8_t* __restrict__ packed, int n_col, int kk) {
  const int nt = n_col >> 4;
  const int nn = n_col & 15;
  const int step = kk >> 5;
  const int kk8 = kk & 31;
  const int lane = (kk8 >> 3) * 16 + nn;
  const int i = kk8 & 7;
  return packed[(static_cast<int64_t>(nt) * (kTargetK / kDummaTileK) + step) *
                    kPackedTileBytes +
                lane * kPackedSlotBytes + i];
}

// One aligned 8-byte (dwordx2) global load per lane per k32 step from the
// packed fragment-slot layout; the loaded uint64 feeds b_frag.x directly in
// the exact du_mma.hpp matrix_b row_major byte order (validated o_proj
// lineage pattern: __builtin_memcpy of the 8 fragment bytes, same DTK).
__device__ __forceinline__ void
load_packed_b_fragment(
    du::dumma::DUFragment<du::dumma::matrix_b, kDummaTileM, kDummaTileN,
                          kDummaTileK, signed char, du::dumma::row_major>&
        b_frag,
    const int8_t* __restrict__ packed_b, int ntile, int k_steps, int step,
    int lane) {
  const int64_t off =
      (static_cast<int64_t>(ntile) * k_steps + step) * kPackedTileBytes +
      lane * kPackedSlotBytes;
  const uint64_t bv = *reinterpret_cast<const uint64_t*>(packed_b + off);
  __builtin_memcpy(b_frag.x, &bv, sizeof(bv));
}

// Generic scalar INT8 W8A8 GEMM.
//
// out[m, n] = bf16(int32_dot(a[m, :], b[:, n]) * x_scale[m] * weight_scale[n])
//
// One thread owns one output element; adjacent lanes map to adjacent n
// columns (fastest-changing N dimension), so B reads and bf16 stores are
// contiguous across lanes within a row.
__global__ __launch_bounds__(kScalarBlockThreads) void w8a8_scalar_gemm_kernel(
    const int8_t* __restrict__ a,          // [M, K] row-major int8
    const int8_t* __restrict__ b,          // [K, N] row-major int8
    const float* __restrict__ x_scale,     // [M, 1] fp32
    const float* __restrict__ weight_scale,// [N, 1] fp32
    hip_bfloat16* __restrict__ out,        // [M, N] bf16
    int m,
    int n,
    int k) {
  const int64_t tid =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total = static_cast<int64_t>(m) * n;
  if (tid >= total) {
    return;
  }
  const int row = static_cast<int>(tid / n);
  const int col = static_cast<int>(tid % n);

  const int8_t* a_row = a + static_cast<int64_t>(row) * k;
  // Iteration 5: for the exact (K=4096, N=1536) shape the weight buffer is
  // the packed [n_tile][k_step][lane][8] fragment-slot layout (see
  // w8a8_pack_b_fragslot_kernel), so the generic fallback decodes it
  // elementwise (keeps the paired M=2 validation exact with the new pack).
  const bool packed_b = (k == kTargetK && n == kTargetN);

  // Exact integer dot product, k-ascending (bit-identical to the reference
  // int64 accumulation order; the int32 accumulator cannot overflow for any
  // supported K).
  int32_t acc = 0;
  for (int kk = 0; kk < k; ++kk) {
    const int8_t b_val =
        packed_b ? packed_b_element(b, col, kk)
                 : b[static_cast<int64_t>(kk) * n + col];
    acc += static_cast<int32_t>(a_row[kk]) * static_cast<int32_t>(b_val);
  }

  // Scale exactly like the reference: (dot * x_scale[m]) * weight_scale[n].
  const float scaled = static_cast<float>(acc) * x_scale[row] *
                       weight_scale[col];
  out[tid] = hip_bfloat16(scaled);
}

// gfx928 DUMMA INT8 kernel for the assigned shape tp8_wqkv_a_m16
// (M=16, N=1536, K=4096) - iteration 2 launch geometry:
//   * blockDim = 128 = 2 wavefronts; WAVES_PER_BLOCK = 2 with in-block
//     split-K = 2: each wavefront accumulates one contiguous K half
//     (wave 0: [0, 2048), wave 1: [2048, 4096)) k-ascending in 32-wide
//     DUMMA steps, preserving the exact int32 accumulation order per split;
//   * one 16-wide N tile per block (N-TILES = 1), grid = N/16 = 96 blocks
//     (the maximum grid reachable on the N-tile axis; reported coverage is
//     96 of 120 CUs);
//   * partial int32 planes combined in LDS (2 x 16 x 16 int32 = 2 KiB) in
//     ascending split order p0 + p1 after one END-of-K barrier - bit-
//     identical to the serial k-ascending accumulation (integer addition
//     is associative), matching the validated "k-ascending per split,
//     ascending split sum" pattern;
//   * direct register epilogue on wave 0 with the verified gfx928
//     accumulator ownership (row = lane&15, col_mod4 = lane>>4,
//     acc_frag.x[i] owns column col_mod4 + 4*i): apply x_scale[m] *
//     weight_scale[n] in float and store bf16;
//   * no A/B LDS staging, no vectorized fragment loads, no inline asm -
//     the load strategy is unchanged from iteration 1 so this round
//     isolates the launch-geometry axis (expected: the second wavefront
//     per CU overlaps the per-load vmcnt waits that dominate the
//     iteration-1 ISA).
__global__ __launch_bounds__(2 * kDummaWaveSize) void
w8a8_dumma_m16n16k32_2wave_sk2_kernel(
    const int8_t* __restrict__ a,           // [16, K] row-major int8
    const int8_t* __restrict__ b,           // [K, N] row-major int8
    const float* __restrict__ x_scale,      // [16, 1] fp32
    const float* __restrict__ weight_scale, // [N, 1] fp32
    hip_bfloat16* __restrict__ out,         // [16, N] bf16
    int n,
    int k) {
  // One int32 partial plane per wave; combined in ascending split order.
  // Row stride is padded 16 -> 20 words (16-B-aligned, non-power-of-two):
  // with the natural stride the plane word index (row*16 + col) mod 64
  // aliases rows differing by 4 onto the same LDS bank phase (4-way
  // conflict on every plane access); stride 20 spreads the 16 rows of a
  // column group onto 16 distinct banks (20*row mod 64 is a permutation
  // for row in [0,16)).
  __shared__ int32_t s_part[2][kDummaTileM][kDummaTileN + 4];

  const int lane = static_cast<int>(threadIdx.x) & (kDummaWaveSize - 1);
  const int wave = static_cast<int>(threadIdx.x) >> 6;  // 0 or 1
  const int n0 = static_cast<int>(blockIdx.x) * kDummaTileN;

  du::dumma::DUFragment<du::dumma::matrix_a, kDummaTileM, kDummaTileN,
                        kDummaTileK, signed char, du::dumma::row_major>
      a_frag;
  du::dumma::DUFragment<du::dumma::matrix_b, kDummaTileM, kDummaTileN,
                        kDummaTileK, signed char, du::dumma::row_major>
      b_frag;
  du::dumma::DUFragment<du::dumma::accumulator, kDummaTileM, kDummaTileN,
                        kDummaTileK, int> acc_frag;
  du::dumma::du_fill_fragment(acc_frag, 0);

  // Each wave accumulates one contiguous K half, k-ascending in 32-wide
  // DUMMA steps. The split boundary (k/2 = 2048) is a multiple of 32, so
  // every step stays aligned to the DUMMA/staging unit.
  const int k_half = k / 2;
  const int k_lo = wave * k_half;
  for (int k0 = k_lo; k0 < k_lo + k_half; k0 += kDummaTileK) {
    du::dumma::du_load_matrix_sync(a_frag, a + k0, k);
    du::dumma::du_load_matrix_sync(b_frag, b + k0 * n + n0, n);
    du::dumma::du_mma_sync(acc_frag, a_frag, b_frag, acc_frag);
  }

  // Publish the int32 partial (k-ascending within the split).
  const int row = lane & 15;
  const int col_mod4 = lane >> 4;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    s_part[wave][row][col_mod4 + 4 * i] = acc_frag.x[i];
  }
  // One END-of-K barrier: both partial planes are complete before the sum.
  __syncthreads();

  // Wave 0 finalizes: ascending split sum p0 + p1 (bit-identical int32
  // order), scale, bf16 store.
  if (wave == 0) {
    const float xs = x_scale[row];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int col = col_mod4 + 4 * i;
      const int32_t total = s_part[0][row][col] + s_part[1][row][col];
      const float scaled = static_cast<float>(total) * xs *
                           weight_scale[n0 + col];
      out[row * n + n0 + col] = hip_bfloat16(scaled);
    }
  }
}

// Iteration 5: workspace-free in-block split-K=2 kernel with the packed-B
// fragment-slot load (same geometry and epilogue as
// w8a8_dumma_m16n16k32_2wave_sk2_kernel; B per k32 step becomes one aligned
// dwordx2 per lane instead of the direct row-major byte loads). Keeps the
// fast path correct for callers without a usable workspace once the weight
// is packed.
__global__ __launch_bounds__(2 * kDummaWaveSize) void
w8a8_dumma_m16n16k32_2wave_sk2_packedb_kernel(
    const int8_t* __restrict__ a,           // [16, K] row-major int8
    const int8_t* __restrict__ packed_b,    // [n_tile][k_step][lane][8]
    const float* __restrict__ x_scale,      // [16, 1] fp32
    const float* __restrict__ weight_scale, // [N, 1] fp32
    hip_bfloat16* __restrict__ out,         // [16, N] bf16
    int n,
    int k) {
  __shared__ int32_t s_part[2][kDummaTileM][kDummaTileN + 4];

  const int lane = static_cast<int>(threadIdx.x) & (kDummaWaveSize - 1);
  const int wave = static_cast<int>(threadIdx.x) >> 6;  // 0 or 1
  const int n0 = static_cast<int>(blockIdx.x) * kDummaTileN;

  du::dumma::DUFragment<du::dumma::matrix_a, kDummaTileM, kDummaTileN,
                        kDummaTileK, signed char, du::dumma::row_major>
      a_frag;
  du::dumma::DUFragment<du::dumma::matrix_b, kDummaTileM, kDummaTileN,
                        kDummaTileK, signed char, du::dumma::row_major>
      b_frag;
  du::dumma::DUFragment<du::dumma::accumulator, kDummaTileM, kDummaTileN,
                        kDummaTileK, int> acc_frag;
  du::dumma::du_fill_fragment(acc_frag, 0);

  const int k_half = k / 2;
  const int k_lo = wave * k_half;
  const int k_steps = k / kDummaTileK;
  const int ntile = static_cast<int>(blockIdx.x);
  for (int k0 = k_lo; k0 < k_lo + k_half; k0 += kDummaTileK) {
    du::dumma::du_load_matrix_sync(a_frag, a + k0, k);
    load_packed_b_fragment(b_frag, packed_b, ntile, k_steps, k0 / kDummaTileK,
                           lane);
    du::dumma::du_mma_sync(acc_frag, a_frag, b_frag, acc_frag);
  }

  const int row = lane & 15;
  const int col_mod4 = lane >> 4;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    s_part[wave][row][col_mod4 + 4 * i] = acc_frag.x[i];
  }
  __syncthreads();

  if (wave == 0) {
    const float xs = x_scale[row];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int col = col_mod4 + 4 * i;
      const int32_t total = s_part[0][row][col] + s_part[1][row][col];
      const float scaled = static_cast<float>(total) * xs *
                           weight_scale[n0 + col];
      out[row * n + n0 + col] = hip_bfloat16(scaled);
    }
  }
}

// gfx928 DUMMA INT8 cross-block split-K partial kernel for the assigned
// shape tp8_wqkv_a_m16 (M=16, N=1536, K=4096) - iteration 3 architecture:
//   * one 64-thread wavefront per block (blockDim = 64), ZERO barriers,
//     ZERO LDS: each block is a self-contained (k-slice, 16x16 N-tile)
//     partial;
//   * grid = (N/16) * split_k blocks; blockIdx.x encodes split-major /
//     ntile-minor (ntile = x % ntiles, split = x / ntiles), so the first
//     wave of blocks streams each split's B slice once;
//   * split_k is a runtime parameter in [1, min(16, K/32)]; K is tiled
//     into 32-aligned step-based slices
//     [(s*steps)/split_k, ((s+1)*steps)/split_k) that cover [0, K) exactly
//     once ascending (non-uniform for non-power-of-two split_k), so every
//     split boundary stays aligned to the DUMMA k32 unit;
//   * per-split accumulation is k-ascending with the unchanged
//     du_load_matrix_sync + du_mma_sync loop (per-slice int32 partials
//     bit-identical to iterations 1-2; the hardware m16n16k32 i8
//     accumulation order is unchanged);
//   * the int32 partial tile is published with du_store_matrix_sync
//     (mem_row_major, ldm = 16) into caller workspace plane (split, ntile)
//     at partials[(split*ntiles + ntile) * 16 * 16] - no LDS round trip,
//     no barrier.
__global__ __launch_bounds__(kDummaWaveSize) void
w8a8_dumma_m16n16k32_sk_partial_kernel(
    const int8_t* __restrict__ a,           // [16, K] row-major int8
    const int8_t* __restrict__ b,           // [K, N] row-major int8
    int32_t* __restrict__ partials,         // [split_k * ntiles][16][16]
    int n,
    int k,
    int split_k) {
  const int ntiles = n / kDummaTileN;
  const int ntile = static_cast<int>(blockIdx.x) % ntiles;
  const int split = static_cast<int>(blockIdx.x) / ntiles;
  const int n0 = ntile * kDummaTileN;
  const int total_steps = k / kDummaTileK;
  // Step-based 32-aligned slice boundaries (validated gate_up pattern):
  // slice s owns DUMMA steps [s*total_steps/split_k, (s+1)*total_steps/
  // split_k), tiling [0, K) exactly once ascending.
  const int k_lo =
      (static_cast<int64_t>(split) * total_steps / split_k) * kDummaTileK;
  const int k_hi =
      (static_cast<int64_t>(split + 1) * total_steps / split_k) *
      kDummaTileK;

  du::dumma::DUFragment<du::dumma::matrix_a, kDummaTileM, kDummaTileN,
                        kDummaTileK, signed char, du::dumma::row_major>
      a_frag;
  du::dumma::DUFragment<du::dumma::matrix_b, kDummaTileM, kDummaTileN,
                        kDummaTileK, signed char, du::dumma::row_major>
      b_frag;
  du::dumma::DUFragment<du::dumma::accumulator, kDummaTileM, kDummaTileN,
                        kDummaTileK, int> acc_frag;
  du::dumma::du_fill_fragment(acc_frag, 0);

  for (int k0 = k_lo; k0 < k_hi; k0 += kDummaTileK) {
    du::dumma::du_load_matrix_sync(a_frag, a + k0, k);
    du::dumma::du_load_matrix_sync(
        b_frag, b + static_cast<int64_t>(k0) * n + n0, n);
    du::dumma::du_mma_sync(acc_frag, a_frag, b_frag, acc_frag);
  }

  du::dumma::du_store_matrix_sync(
      partials +
          (static_cast<int64_t>(split) * ntiles + ntile) *
              (kDummaTileM * kDummaTileN),
      acc_frag, kDummaTileN, du::dumma::mem_row_major);
}

// Iteration 5: zero-LDS split-K partial kernel with the packed-B
// fragment-slot load (same geometry, K slicing and A direct-load path as
// w8a8_dumma_m16n16k32_sk_partial_kernel; B per k32 step becomes one
// aligned dwordx2 per lane). Used when the A-slice LDS stage would exceed
// the 64 KiB limit (split_k=1) now that the weight is packed.
__global__ __launch_bounds__(kDummaWaveSize) void
w8a8_dumma_m16n16k32_sk_directa_packedb_partial_kernel(
    const int8_t* __restrict__ a,           // [16, K] row-major int8
    const int8_t* __restrict__ packed_b,    // [n_tile][k_step][lane][8]
    int32_t* __restrict__ partials,         // [split_k * ntiles][16][16]
    int n,
    int k,
    int split_k) {
  const int ntiles = n / kDummaTileN;
  const int ntile = static_cast<int>(blockIdx.x) % ntiles;
  const int split = static_cast<int>(blockIdx.x) / ntiles;
  const int total_steps = k / kDummaTileK;
  const int k_lo =
      (static_cast<int64_t>(split) * total_steps / split_k) * kDummaTileK;
  const int k_hi =
      (static_cast<int64_t>(split + 1) * total_steps / split_k) *
      kDummaTileK;

  const int lane = static_cast<int>(threadIdx.x) & (kDummaWaveSize - 1);
  du::dumma::DUFragment<du::dumma::matrix_a, kDummaTileM, kDummaTileN,
                        kDummaTileK, signed char, du::dumma::row_major>
      a_frag;
  du::dumma::DUFragment<du::dumma::matrix_b, kDummaTileM, kDummaTileN,
                        kDummaTileK, signed char, du::dumma::row_major>
      b_frag;
  du::dumma::DUFragment<du::dumma::accumulator, kDummaTileM, kDummaTileN,
                        kDummaTileK, int> acc_frag;
  du::dumma::du_fill_fragment(acc_frag, 0);

  for (int k0 = k_lo; k0 < k_hi; k0 += kDummaTileK) {
    du::dumma::du_load_matrix_sync(a_frag, a + k0, k);
    load_packed_b_fragment(b_frag, packed_b, ntile, total_steps,
                           k0 / kDummaTileK, lane);
    du::dumma::du_mma_sync(acc_frag, a_frag, b_frag, acc_frag);
  }

  du::dumma::du_store_matrix_sync(
      partials +
          (static_cast<int64_t>(split) * ntiles + ntile) *
              (kDummaTileM * kDummaTileN),
      acc_frag, kDummaTileN, du::dumma::mem_row_major);
}

// gfx928 DUMMA INT8 cross-block split-K partial kernel for tp8_wqkv_a_m16 -
// iteration 4 architecture (A-only LDS staging). Same launch geometry as
// iteration 3 (one 64-thread wavefront per block, ZERO in-loop barriers,
// grid = (N/16) * split_k, split-major / ntile-minor blockIdx encoding,
// step-based 32-aligned K slicing) with one change on the load path:
//   * each block stages its own A slice [16, slice_k] once at block start
//     into dynamic LDS (extern __shared__): 16-byte int4 global loads
//     (16-B-aligned: k multiple of 16, k_lo multiple of 32, col16 multiple
//     of 16) + ds_write_b128 stores, row-major with padded row stride
//     local_stride = slice_k + 16 bytes (16-B-aligned non-power-of-two
//     bank skew so the 16 rows do not alias one LDS bank phase; validated
//     down_proj rule). Rows are not contiguous in global (A row stride =
//     k), so lane `chunk` owns 16 bytes of row chunk/chunks_per_row at
//     column (chunk % chunks_per_row) * 16 - consecutive lanes walk
//     consecutive 16-B chunks (coalesced), one wavefront iteration stages
//     64 chunks = 1 KiB;
//   * the K loop reads the A fragment from LDS with the unchanged
//     du_load_matrix_sync (smem pointer, ldm = local_stride in bytes), so
//     A's ~8-12 scalar global loads + per-load vmcnt waits per step leave
//     the loop (replaced by low-latency ds_read); B fragments are loaded
//     from global exactly as iteration 3 (unchanged byte-load path, B is
//     the later staging/packing round);
//   * dynamic LDS = 16 * (max_slice_k + 16) bytes is uniform per launch
//     (host computes max_slice_k = ceil(total_steps / split_k) * 32; for
//     the default split_k=5 that is 16 * 848 = 13,568 B so four blocks
//     fit per CU); each block uses its own slice_k stride;
//   * int32 partial publish unchanged (du_store_matrix_sync, plane
//     (split, ntile)). Exact bf16 equality is preserved: staged bytes are
//     the same A data in the same row-major order, and the int32
//     accumulation order is unchanged (k-ascending per split, ascending
//     split sum).
__global__ __launch_bounds__(kDummaWaveSize) void
w8a8_dumma_m16n16k32_sk_astage_partial_kernel(
    const int8_t* __restrict__ a,           // [16, K] row-major int8
    const int8_t* __restrict__ b,           // [K, N] row-major int8
    int32_t* __restrict__ partials,         // [split_k * ntiles][16][16]
    int n,
    int k,
    int split_k) {
  extern __shared__ __align__(16) signed char smem_a[];

  const int lane = static_cast<int>(threadIdx.x) & (kDummaWaveSize - 1);
  const int ntiles = n / kDummaTileN;
  const int ntile = static_cast<int>(blockIdx.x) % ntiles;
  const int split = static_cast<int>(blockIdx.x) / ntiles;
  const int n0 = ntile * kDummaTileN;
  const int total_steps = k / kDummaTileK;
  // Step-based 32-aligned slice boundaries (identical to iteration 3):
  // slice s owns DUMMA steps [s*total_steps/split_k, (s+1)*total_steps/
  // split_k), tiling [0, K) exactly once ascending.
  const int k_lo =
      (static_cast<int64_t>(split) * total_steps / split_k) * kDummaTileK;
  const int k_hi =
      (static_cast<int64_t>(split + 1) * total_steps / split_k) *
      kDummaTileK;
  const int slice_k = k_hi - k_lo;        // multiple of 32
  const int local_stride = slice_k + 16;  // padded LDS row stride (bytes)

  // One-time A-slice stage: 16 rows x slice_k bytes, row-major into LDS.
  const int chunks_per_row = slice_k / 16;
  const int total_chunks = 16 * chunks_per_row;
  const int iters = (total_chunks + kDummaWaveSize - 1) / kDummaWaveSize;
  for (int it = 0; it < iters; ++it) {
    const int chunk = it * kDummaWaveSize + lane;
    if (chunk < total_chunks) {
      const int row = chunk / chunks_per_row;
      const int col16 = (chunk % chunks_per_row) * 16;
      const int4 v = *reinterpret_cast<const int4*>(
          a + static_cast<int64_t>(row) * k + k_lo + col16);
      *reinterpret_cast<int4*>(smem_a + row * local_stride + col16) = v;
    }
  }

  du::dumma::DUFragment<du::dumma::matrix_a, kDummaTileM, kDummaTileN,
                        kDummaTileK, signed char, du::dumma::row_major>
      a_frag;
  du::dumma::DUFragment<du::dumma::matrix_b, kDummaTileM, kDummaTileN,
                        kDummaTileK, signed char, du::dumma::row_major>
      b_frag;
  du::dumma::DUFragment<du::dumma::accumulator, kDummaTileM, kDummaTileN,
                        kDummaTileK, int> acc_frag;
  du::dumma::du_fill_fragment(acc_frag, 0);

  for (int k0 = k_lo; k0 < k_hi; k0 += kDummaTileK) {
    du::dumma::du_load_matrix_sync(
        a_frag,
        reinterpret_cast<const int8_t*>(smem_a) + (k0 - k_lo),
        local_stride);
    du::dumma::du_load_matrix_sync(
        b_frag, b + static_cast<int64_t>(k0) * n + n0, n);
    du::dumma::du_mma_sync(acc_frag, a_frag, b_frag, acc_frag);
  }

  du::dumma::du_store_matrix_sync(
      partials +
          (static_cast<int64_t>(split) * ntiles + ntile) *
              (kDummaTileM * kDummaTileN),
      acc_frag, kDummaTileN, du::dumma::mem_row_major);
}

// gfx928 DUMMA INT8 cross-block split-K partial kernel for tp8_wqkv_a_m16 -
// iteration 5 architecture (A-only LDS staging + PACKED-B fragment slots).
// Same launch geometry, K slicing, A-slice LDS stage and partial publish as
// the iteration-4 astage kernel with one change on the B load path:
//   * B fragments are read from the packed [n_tile][k_step][lane][8]
//     fragment-slot layout (w8a8_pack_b_fragslot_kernel) with ONE aligned
//     8-byte (dwordx2) global load per lane per k32 step
//     (load_packed_b_fragment), in the exact du_mma.hpp matrix_b row_major
//     byte order - B's per-step scalar byte loads + fragment reassembly +
//     per-load vmcnt cascade (8 vmem instructions and ~50 VALU per step in
//     the iteration-4 exact ISA) leave the K loop, which becomes ds_read
//     (A, from LDS) -> 1 dwordx2 (B, packed) -> v_mmac;
//   * dynamic LDS is unchanged (16 x (max_slice_k + 16) B; default split_k=5
//     -> 13,568 B, four blocks/CU), int32 partial publish unchanged. Exact
//     bf16 equality is preserved: every packed fragment feeds the same bytes
//     to the same v_mmac in the same order as the iteration-4 direct loads,
//     and the int32 accumulation order is unchanged (k-ascending per split,
//     ascending split sum).
__global__ __launch_bounds__(kDummaWaveSize) void
w8a8_dumma_m16n16k32_sk_astage_packedb_partial_kernel(
    const int8_t* __restrict__ a,           // [16, K] row-major int8
    const int8_t* __restrict__ packed_b,    // [n_tile][k_step][lane][8]
    int32_t* __restrict__ partials,         // [split_k * ntiles][16][16]
    int n,
    int k,
    int split_k) {
  extern __shared__ __align__(16) signed char smem_a[];

  const int lane = static_cast<int>(threadIdx.x) & (kDummaWaveSize - 1);
  const int ntiles = n / kDummaTileN;
  const int ntile = static_cast<int>(blockIdx.x) % ntiles;
  const int split = static_cast<int>(blockIdx.x) / ntiles;
  const int total_steps = k / kDummaTileK;
  const int k_lo =
      (static_cast<int64_t>(split) * total_steps / split_k) * kDummaTileK;
  const int k_hi =
      (static_cast<int64_t>(split + 1) * total_steps / split_k) *
      kDummaTileK;
  const int slice_k = k_hi - k_lo;        // multiple of 32
  const int local_stride = slice_k + 16;  // padded LDS row stride (bytes)

  // One-time A-slice stage: 16 rows x slice_k bytes, row-major into LDS.
  const int chunks_per_row = slice_k / 16;
  const int total_chunks = 16 * chunks_per_row;
  const int iters = (total_chunks + kDummaWaveSize - 1) / kDummaWaveSize;
  for (int it = 0; it < iters; ++it) {
    const int chunk = it * kDummaWaveSize + lane;
    if (chunk < total_chunks) {
      const int row = chunk / chunks_per_row;
      const int col16 = (chunk % chunks_per_row) * 16;
      const int4 v = *reinterpret_cast<const int4*>(
          a + static_cast<int64_t>(row) * k + k_lo + col16);
      *reinterpret_cast<int4*>(smem_a + row * local_stride + col16) = v;
    }
  }

  du::dumma::DUFragment<du::dumma::matrix_a, kDummaTileM, kDummaTileN,
                        kDummaTileK, signed char, du::dumma::row_major>
      a_frag;
  du::dumma::DUFragment<du::dumma::matrix_b, kDummaTileM, kDummaTileN,
                        kDummaTileK, signed char, du::dumma::row_major>
      b_frag;
  du::dumma::DUFragment<du::dumma::accumulator, kDummaTileM, kDummaTileN,
                        kDummaTileK, int> acc_frag;
  du::dumma::du_fill_fragment(acc_frag, 0);

  for (int k0 = k_lo; k0 < k_hi; k0 += kDummaTileK) {
    du::dumma::du_load_matrix_sync(
        a_frag,
        reinterpret_cast<const int8_t*>(smem_a) + (k0 - k_lo),
        local_stride);
    load_packed_b_fragment(b_frag, packed_b, ntile, total_steps,
                           k0 / kDummaTileK, lane);
    du::dumma::du_mma_sync(acc_frag, a_frag, b_frag, acc_frag);
  }

  du::dumma::du_store_matrix_sync(
      partials +
          (static_cast<int64_t>(split) * ntiles + ntile) *
              (kDummaTileM * kDummaTileN),
      acc_frag, kDummaTileN, du::dumma::mem_row_major);
}

// gfx928 DUMMA INT8 cross-block split-K partial kernel for tp8_wqkv_a_m16 -
// iteration 6 architecture (A+B LDS staging, stage-K=64 double buffer).
// Same launch geometry, K slicing and partial publish as iteration 5 (one
// 64-thread wavefront per block, ZERO barriers, grid = (N/16) * split_k,
// split-major / ntile-minor blockIdx, step-based 32-aligned K slices) with
// the staging family changed from A-only full-slice to A+B stage-K=64
// double-buffered prefetch:
//   * the K slice is processed in 2-step (K=64) chunks; dynamic LDS holds
//     two ping-pong buffers of 2,560 B each: an A chunk region (16 rows x
//     64 cols, padded 96-B row stride) and a B chunk region (the packed
//     fragment slots of the 2 steps, 8 B per lane per step, in the exact
//     w8a8_pack_b_fragslot_kernel order);
//   * per chunk each lane issues 3 cooperative global loads (1 x 16-B A
//     dwordx4 + 2 x 8-B B dwordx2) into registers BEFORE the current
//     chunk's 1-2 v_mmac_i32_16x16x32_i8, then commits them to the
//     alternate LDS buffer AFTER the MMAC burst - the compiler's vmcnt wait
//     lands at that ds_write, overlapping the cold once-read B latency with
//     the current chunk's compute (the exact source is verified by the
//     harness ISA extraction: next-chunk global_load_dwordx4/dwordx2 before
//     the current v_mmac, vmcnt wait near the alternate-buffer
//     ds_write_b128/b64);
//   * the steady K loop then contains ZERO global loads: per k32 step it is
//     ds_read2_b32 (A, from LDS) -> ds_read_b64 (B, from LDS) -> v_mmac;
//   * dynamic LDS = 5,120 B per block (constant, independent of split_k, so
//     the old split_k=1 64-KiB A-slice overflow fallback is no longer
//     needed); at the default split_k=5 grid 480 = 4 x 120 CUs, four blocks
//     per CU = 20,480 B <= 64 KiB. The last chunk of an odd-step slice
//     stages/computes 1 step only (the A load still reads 64 cols, always
//     inside the A row [0, K));
//   * int32 partial publish unchanged. Exact bf16 equality is preserved:
//     every staged byte is the same data fed to the same v_mmac in the same
//     order, and the int32 accumulation order is unchanged (k-ascending per
//     split, ascending split sum).
__global__ __launch_bounds__(kDummaWaveSize) void
w8a8_dumma_m16n16k32_sk_astage_ab_dbuf_partial_kernel(
    const int8_t* __restrict__ a,           // [16, K] row-major int8
    const int8_t* __restrict__ packed_b,    // [n_tile][k_step][lane][8]
    int32_t* __restrict__ partials,         // [split_k * ntiles][16][16]
    int n,
    int k,
    int split_k) {
  extern __shared__ __align__(16) signed char smem[];

  const int lane = static_cast<int>(threadIdx.x) & (kDummaWaveSize - 1);
  const int ntiles = n / kDummaTileN;
  const int ntile = static_cast<int>(blockIdx.x) % ntiles;
  const int split = static_cast<int>(blockIdx.x) / ntiles;
  const int total_steps = k / kDummaTileK;
  const int k_lo =
      (static_cast<int64_t>(split) * total_steps / split_k) * kDummaTileK;
  const int k_hi =
      (static_cast<int64_t>(split + 1) * total_steps / split_k) *
      kDummaTileK;
  const int slice_steps = (k_hi - k_lo) / kDummaTileK;
  const int nchunks = (slice_steps + kStageSteps - 1) / kStageSteps;

  du::dumma::DUFragment<du::dumma::matrix_a, kDummaTileM, kDummaTileN,
                        kDummaTileK, signed char, du::dumma::row_major>
      a_frag;
  du::dumma::DUFragment<du::dumma::matrix_b, kDummaTileM, kDummaTileN,
                        kDummaTileK, signed char, du::dumma::row_major>
      b_frag;
  du::dumma::DUFragment<du::dumma::accumulator, kDummaTileM, kDummaTileN,
                        kDummaTileK, int> acc_frag;
  du::dumma::du_fill_fragment(acc_frag, 0);

  // Per-lane roles inside one chunk: A lane owns 16 B of row (lane>>2) at
  // column (lane&3)*16 (4 consecutive lanes walk one 64-B A row, coalesced
  // 16-B loads); B lane owns its 8-B fragment slot of each of the 2 steps.
  const int a_row = lane >> 2;
  const int a_col16 = (lane & 3) * 16;
  const int64_t a_base = static_cast<int64_t>(k_lo);  // + chunk*64 + col16
  // B slot base for the slice's FIRST global k32 step (k_lo/kDummaTileK):
  // the packed [n_tile][k_step][lane][8] layout is indexed by the GLOBAL
  // step (pack kernel: slot = ntile*k_steps + step, step in [0, K/32)), so
  // the slice-relative chunk/step offsets below must be added on top of the
  // split's global step offset - iteration-6 repair 2 (the original base
  // pointed at global step 0, so every split with k_lo > 0 loaded B from
  // the wrong global steps while A was staged correctly).
  const int64_t b_slot_base =
      static_cast<int64_t>(ntile) * total_steps * kPackedTileBytes +
      (k_lo / kDummaTileK) * kPackedTileBytes +
      lane * kPackedSlotBytes;  // + step*kPackedTileBytes

  // Prologue: stage chunk 0 into buffer 0 (single wait before the loop).
  {
    signed char* buf = smem;
    const int4 av = *reinterpret_cast<const int4*>(
        a + static_cast<int64_t>(a_row) * k + a_base + a_col16);
    *reinterpret_cast<int4*>(buf + a_row * kStageAStride + a_col16) = av;
    const uint64_t bv0 =
        *reinterpret_cast<const uint64_t*>(packed_b + b_slot_base);
    *reinterpret_cast<uint64_t*>(buf + kStageABufBytes +
                                 lane * kPackedSlotBytes) = bv0;
    if (1 < slice_steps) {
      const uint64_t bv1 = *reinterpret_cast<const uint64_t*>(
          packed_b + b_slot_base + kPackedTileBytes);
      *reinterpret_cast<uint64_t*>(buf + kStageABufBytes +
                                   kPackedTileBytes +
                                   lane * kPackedSlotBytes) = bv1;
    }
  }

  for (int c = 0; c < nchunks; ++c) {
    const signed char* buf = smem + (c & 1) * kStageBufBytes;

    // Prefetch chunk c+1 from global into registers (no wait yet). The
    // values are consumed only by the alternate-buffer stores below, so the
    // compiler issues these loads before (or at worst alongside) the MMAC
    // burst and lands the vmcnt wait at the ds_write - the overlap that
    // removes the iteration-5 per-step vmcnt(0)-before-mmac exposure.
    int4 a_v;
    uint64_t b_v0, b_v1;
    const bool has_next = (c + 1 < nchunks);
    if (has_next) {
      const int nc = c + 1;
      const int64_t a_off =
          a_base + static_cast<int64_t>(nc) * kStageK + a_col16;
      a_v = *reinterpret_cast<const int4*>(
          a + static_cast<int64_t>(a_row) * k + a_off);
      const int64_t boff = b_slot_base +
                           static_cast<int64_t>(nc) * kStageSteps *
                               kPackedTileBytes;
      b_v0 = *reinterpret_cast<const uint64_t*>(packed_b + boff);
      if (nc * kStageSteps + 1 < slice_steps) {
        b_v1 = *reinterpret_cast<const uint64_t*>(packed_b + boff +
                                                  kPackedTileBytes);
      }
    }

    // Compute the current chunk from LDS: 1 or 2 k-ascending k32 steps,
    // each ds_read (A) -> ds_read (B) -> v_mmac. nsteps must be clamped to
    // the chunk width: slice_steps - s0 is the remaining-step count, which
    // is > kStageSteps for the second-to-last chunk (e.g. 25 - 2*11 = 3 at
    // split_k=5, 26 - 2*11 = 4 for an even slice), where the extra steps
    // would read A/B bytes beyond the staged 64-column chunk (stale LDS).
    // Only the kStageSteps steps staged in this chunk may be computed; the
    // last chunk of an odd-step slice stages/computes 1 step only.
    const int s0 = c * kStageSteps;
    const int nsteps = (slice_steps - s0 > kStageSteps)
                           ? kStageSteps
                           : (slice_steps - s0);  // 1 or 2
    const signed char* a_buf = buf;
    const signed char* b_buf = buf + kStageABufBytes;
    for (int j = 0; j < nsteps; ++j) {
      du::dumma::du_load_matrix_sync(a_frag, a_buf + j * kDummaTileK,
                                     kStageAStride);
      const uint64_t bv = *reinterpret_cast<const uint64_t*>(
          b_buf + j * kPackedTileBytes + lane * kPackedSlotBytes);
      __builtin_memcpy(b_frag.x, &bv, sizeof(bv));
      du::dumma::du_mma_sync(acc_frag, a_frag, b_frag, acc_frag);
    }

    // Commit the prefetched chunk into the alternate buffer (the compiler
    // inserts the vmcnt wait here, after the MMAC burst above).
    if (has_next) {
      const int nc = c + 1;
      signed char* nb = smem + ((nc & 1) * kStageBufBytes);
      *reinterpret_cast<int4*>(nb + a_row * kStageAStride + a_col16) = a_v;
      const int64_t boff = b_slot_base +
                           static_cast<int64_t>(nc) * kStageSteps *
                               kPackedTileBytes;
      *reinterpret_cast<uint64_t*>(nb + kStageABufBytes +
                                   lane * kPackedSlotBytes) = b_v0;
      if (nc * kStageSteps + 1 < slice_steps) {
        *reinterpret_cast<uint64_t*>(nb + kStageABufBytes +
                                     kPackedTileBytes +
                                     lane * kPackedSlotBytes) = b_v1;
      }
    }
  }

  du::dumma::du_store_matrix_sync(
      partials +
          (static_cast<int64_t>(split) * ntiles + ntile) *
              (kDummaTileM * kDummaTileN),
      acc_frag, kDummaTileN, du::dumma::mem_row_major);
}

// gfx928 DUMMA INT8 cross-block split-K partial kernel for tp8_wqkv_a_m16 -
// iteration 7 architecture (SINGLE-buffered A+B LDS staging, stage-K=64) -
// the mandated single-vs-double buffering comparison arm. Identical to the
// accepted iteration-6 double-buffered kernel (same launch geometry, K
// slicing, per-chunk register prefetch of chunk c+1 before chunk c's MMAC
// burst, vmcnt wait placed at the post-MMAC ds_write, zero in-loop barriers,
// same packed-B fragment-slot layout and int32 partial publish) with ONE
// change: the two ping-pong 2,560-B buffers collapse into a SINGLE 2,560-B
// buffer, so
//   * dynamic LDS per block drops 5,120 -> 2,560 B (per CU 20,480 ->
//     10,240 B at the default split_k=5 grid; occupancy is grid-limited at
//     480 blocks = 4 x 120 CUs, so residency is unchanged);
//   * the chunk c+1 prefetch commits to the SAME LDS addresses chunk c was
//     just read from - the write-after-read ordering is program order
//     inside the single wave (the j-loop reads precede the commit stores),
//     the compiler keeps the same load-before-MMAC / wait-at-ds_write
//     pipeline, and the ds_write base is a fixed constant instead of the
//     (c&1)*kStageBufBytes toggle;
//   * barriers per K step stay 0 (one 64-lane wave per block, no cross-wave
//     LDS producer/consumer; sync is s_waitcnt vmcnt/lgkmcnt only - the
//     TP4 4-wave reference's ~1 barrier per K step does not apply here);
//   * the last chunk of an odd-step slice still stages/computes 1 step only
//     (nsteps clamp), and every A stage load stays inside the A row
//     [0, K). Exact bf16 equality is preserved: the staged bytes are the
//     same data fed to the same v_mmac in the same order, and the int32
//     accumulation order is unchanged (k-ascending per split, ascending
//     split sum). Falsifiable gate vs the double-buffer best: if the
//     ping-pong LDS buffer and its base toggle are NOT the mechanism behind
//     the iteration-6 win (the register prefetch + wait placement is), this
//     kernel lands within noise of 18.551 us; PMC discriminates via
//     lds_bytes 5,120 -> 2,560 with unchanged vmem_read/lds_instruction
//     counts and 0 barriers per K step in both.
__global__ __launch_bounds__(kDummaWaveSize) void
w8a8_dumma_m16n16k32_sk_astage_ab_sbuf_partial_kernel(
    const int8_t* __restrict__ a,           // [16, K] row-major int8
    const int8_t* __restrict__ packed_b,    // [n_tile][k_step][lane][8]
    int32_t* __restrict__ partials,         // [split_k * ntiles][16][16]
    int n,
    int k,
    int split_k) {
  extern __shared__ __align__(16) signed char smem[];

  const int lane = static_cast<int>(threadIdx.x) & (kDummaWaveSize - 1);
  const int ntiles = n / kDummaTileN;
  const int ntile = static_cast<int>(blockIdx.x) % ntiles;
  const int split = static_cast<int>(blockIdx.x) / ntiles;
  const int total_steps = k / kDummaTileK;
  const int k_lo =
      (static_cast<int64_t>(split) * total_steps / split_k) * kDummaTileK;
  const int k_hi =
      (static_cast<int64_t>(split + 1) * total_steps / split_k) *
      kDummaTileK;
  const int slice_steps = (k_hi - k_lo) / kDummaTileK;
  const int nchunks = (slice_steps + kStageSteps - 1) / kStageSteps;

  du::dumma::DUFragment<du::dumma::matrix_a, kDummaTileM, kDummaTileN,
                        kDummaTileK, signed char, du::dumma::row_major>
      a_frag;
  du::dumma::DUFragment<du::dumma::matrix_b, kDummaTileM, kDummaTileN,
                        kDummaTileK, signed char, du::dumma::row_major>
      b_frag;
  du::dumma::DUFragment<du::dumma::accumulator, kDummaTileM, kDummaTileN,
                        kDummaTileK, int> acc_frag;
  du::dumma::du_fill_fragment(acc_frag, 0);

  // Per-lane roles inside one chunk (unchanged from iteration 6): A lane
  // owns 16 B of row (lane>>2) at column (lane&3)*16; B lane owns its 8-B
  // fragment slot of each of the 2 steps.
  const int a_row = lane >> 2;
  const int a_col16 = (lane & 3) * 16;
  const int64_t a_base = static_cast<int64_t>(k_lo);
  // B slot base for the slice's FIRST GLOBAL k32 step (k_lo/kDummaTileK) -
  // the packed [n_tile][k_step][lane][8] layout is indexed by the GLOBAL
  // step (iteration-6 repair 2).
  const int64_t b_slot_base =
      static_cast<int64_t>(ntile) * total_steps * kPackedTileBytes +
      (k_lo / kDummaTileK) * kPackedTileBytes +
      lane * kPackedSlotBytes;  // + step*kPackedTileBytes

  // Prologue: stage chunk 0 into the single buffer (the compiler inserts
  // the vmcnt wait before the ds_write).
  {
    const int4 av = *reinterpret_cast<const int4*>(
        a + static_cast<int64_t>(a_row) * k + a_base + a_col16);
    *reinterpret_cast<int4*>(smem + a_row * kStageAStride + a_col16) = av;
    const uint64_t bv0 =
        *reinterpret_cast<const uint64_t*>(packed_b + b_slot_base);
    *reinterpret_cast<uint64_t*>(smem + kStageABufBytes +
                                 lane * kPackedSlotBytes) = bv0;
    if (1 < slice_steps) {
      const uint64_t bv1 = *reinterpret_cast<const uint64_t*>(
          packed_b + b_slot_base + kPackedTileBytes);
      *reinterpret_cast<uint64_t*>(smem + kStageABufBytes +
                                   kPackedTileBytes +
                                   lane * kPackedSlotBytes) = bv1;
    }
  }

  for (int c = 0; c < nchunks; ++c) {
    // Single buffer: the current chunk is always at the smem base.
    const signed char* buf = smem;

    // Prefetch chunk c+1 from global into registers (no wait yet). The
    // values are consumed only by the commit stores below, so the compiler
    // issues these loads before (or at worst alongside) the MMAC burst and
    // lands the vmcnt wait at the ds_write - the overlap that removes the
    // iteration-5 per-step vmcnt(0)-before-mmac exposure.
    int4 a_v;
    uint64_t b_v0, b_v1;
    const bool has_next = (c + 1 < nchunks);
    if (has_next) {
      const int nc = c + 1;
      const int64_t a_off =
          a_base + static_cast<int64_t>(nc) * kStageK + a_col16;
      a_v = *reinterpret_cast<const int4*>(
          a + static_cast<int64_t>(a_row) * k + a_off);
      const int64_t boff = b_slot_base +
                           static_cast<int64_t>(nc) * kStageSteps *
                               kPackedTileBytes;
      b_v0 = *reinterpret_cast<const uint64_t*>(packed_b + boff);
      if (nc * kStageSteps + 1 < slice_steps) {
        b_v1 = *reinterpret_cast<const uint64_t*>(packed_b + boff +
                                                  kPackedTileBytes);
      }
    }

    // Compute the current chunk from the single LDS buffer: 1 or 2
    // k-ascending k32 steps, each ds_read (A) -> ds_read (B) -> v_mmac
    // (nsteps clamped to the staged chunk width, iteration-6 repair 1).
    const int s0 = c * kStageSteps;
    const int nsteps = (slice_steps - s0 > kStageSteps)
                           ? kStageSteps
                           : (slice_steps - s0);  // 1 or 2
    const signed char* a_buf = buf;
    const signed char* b_buf = buf + kStageABufBytes;
    for (int j = 0; j < nsteps; ++j) {
      du::dumma::du_load_matrix_sync(a_frag, a_buf + j * kDummaTileK,
                                     kStageAStride);
      const uint64_t bv = *reinterpret_cast<const uint64_t*>(
          b_buf + j * kPackedTileBytes + lane * kPackedSlotBytes);
      __builtin_memcpy(b_frag.x, &bv, sizeof(bv));
      du::dumma::du_mma_sync(acc_frag, a_frag, b_frag, acc_frag);
    }

    // Commit the prefetched chunk into the single buffer (the compiler
    // inserts the vmcnt wait here, after the MMAC burst above; the write
    // follows the reads of the same buffer in program order).
    if (has_next) {
      const int nc = c + 1;
      *reinterpret_cast<int4*>(smem + a_row * kStageAStride + a_col16) =
          a_v;
      const int64_t boff = b_slot_base +
                           static_cast<int64_t>(nc) * kStageSteps *
                               kPackedTileBytes;
      *reinterpret_cast<uint64_t*>(smem + kStageABufBytes +
                                   lane * kPackedSlotBytes) = b_v0;
      if (nc * kStageSteps + 1 < slice_steps) {
        *reinterpret_cast<uint64_t*>(smem + kStageABufBytes +
                                     kPackedTileBytes +
                                     lane * kPackedSlotBytes) = b_v1;
      }
    }
  }

  du::dumma::du_store_matrix_sync(
      partials +
          (static_cast<int64_t>(split) * ntiles + ntile) *
              (kDummaTileM * kDummaTileN),
      acc_frag, kDummaTileN, du::dumma::mem_row_major);
}

// gfx928 DUMMA INT8 cross-block split-K partial kernel for tp8_wqkv_a_m16 -
// iteration 14 architecture (FUSED last-arrival combine tail): the
// iteration-7 single-buffered stage-K=64 kernel EXACTLY as measured
// (18.484 us median / 18.496 us p90 shadow; same launch geometry split_k=5
// grid 480 one-wave zero-barrier blocks = 4 x 120 CUs, same per-chunk
// register prefetch of chunk c+1 before chunk c's MMAC burst with the vmcnt
// wait at the post-MMAC fixed-base ds_write, same single 2,560-B buffer,
// same packed-B fragment-slot layout, same K slicing and int32 partial
// publish) PLUS a fused combine/scale tail: after publishing its int32
// plane, each block issues __threadfence() (device scope, orders the plane
// store), lane 0 does one atomicAdd on the ntile's monotonic arrival
// counter (counters live in the last counter_bytes of the caller
// workspace, zeroed once per distinct workspace on the first launcher call
// - the harness warmup runs OUTSIDE Graph capture - and advance by exactly
// split_k per complete launch/replay, so the (old % split_k == split_k-1)
// last-arriver test is valid forever without any in-Graph reset), and the
// LAST-ARRIVING block for the ntile (one 64-lane wave covers all 256
// elements, 4 per lane, coalesced plane reads) sums the split_k planes in
// ascending split order (bit-identical int32 order to the iteration-3
// combine kernel: k-ascending per split, ascending split sum), applies
// x_scale[row] * weight_scale[n0+col], and stores bf16 with the same
// expression. The separate 96-block x 256-thread combine kernel is REMOVED
// from the timed Graph - one kernel launch per replay. The tail's end-of-
// kernel s_barrier is the only barrier (one wave per block; accepted
// down_proj-style END-of-K combine barrier). Every staged byte and every
// v_mmac is unchanged, so partials are bit-identical and the only output-
// order change is where the ascending split sum happens; |full dot| =
// 4096*127*127 ~= 6.6e7 << 2^31. Falsifiable gate vs the 18.484 us shadow:
// if the ~2 us gap between the 16.48 us profiled partial phase and the
// Graph median is the second launch plus the combine kernel's L2-plane
// round trip (480 KB read + 48 KB write), removing them drops the median
// well below the 1% acceptance bar (predicted ~1-1.5 us); if the gap is
// actually graph dispatch overhead or profiler perturbation, the round
// lands flat; if the atomic/fence tail or its L2 traffic costs more than
// the saved launch, it lands mildly regressed - the comparison
// discriminates the mechanism either way.
// Iteration 15: the measured round landed essentially flat (18.382 us
// shadow median vs the 18.484 us iteration-7 shadow, +0.55%) - the ~2 us
// partial-to-Graph gap was the FUSED TAIL's own fence/atomic/tail cost,
// not the combine kernel's launch+L2 round trip (consistent with the TP4
// lineage, where the atomic last-arriver fused finalize on the 4-wave
// design measured +1.52 us: 18.864 vs 17.344). This kernel is now a
// PRESERVED SYMBOL: the workspace path dispatches the iteration-15
// TP4-validated 4-wave partial kernel + separate combine kernel instead.
__global__ __launch_bounds__(kDummaWaveSize) void
w8a8_dumma_m16n16k32_sk_astage_ab_sbuf_fused_partial_kernel(
    const int8_t* __restrict__ a,           // [16, K] row-major int8
    const int8_t* __restrict__ packed_b,    // [n_tile][k_step][lane][8]
    int32_t* __restrict__ partials,         // [split_k * ntiles][16][16]
    int32_t* __restrict__ counters,         // [ntiles] monotonic arrivals
    const float* __restrict__ x_scale,      // [16, 1] fp32
    const float* __restrict__ weight_scale, // [N, 1] fp32
    hip_bfloat16* __restrict__ out,         // [16, N] bf16
    int n,
    int k,
    int split_k) {
  extern __shared__ __align__(16) signed char smem[];
  __shared__ int s_arrived;  // tail arrival broadcast (one wave per block)

  const int lane = static_cast<int>(threadIdx.x) & (kDummaWaveSize - 1);
  const int ntiles = n / kDummaTileN;
  const int ntile = static_cast<int>(blockIdx.x) % ntiles;
  const int split = static_cast<int>(blockIdx.x) / ntiles;
  const int total_steps = k / kDummaTileK;
  const int k_lo =
      (static_cast<int64_t>(split) * total_steps / split_k) * kDummaTileK;
  const int k_hi =
      (static_cast<int64_t>(split + 1) * total_steps / split_k) *
      kDummaTileK;
  const int slice_steps = (k_hi - k_lo) / kDummaTileK;
  const int nchunks = (slice_steps + kStageSteps - 1) / kStageSteps;

  du::dumma::DUFragment<du::dumma::matrix_a, kDummaTileM, kDummaTileN,
                        kDummaTileK, signed char, du::dumma::row_major>
      a_frag;
  du::dumma::DUFragment<du::dumma::matrix_b, kDummaTileM, kDummaTileN,
                        kDummaTileK, signed char, du::dumma::row_major>
      b_frag;
  du::dumma::DUFragment<du::dumma::accumulator, kDummaTileM, kDummaTileN,
                        kDummaTileK, int> acc_frag;
  du::dumma::du_fill_fragment(acc_frag, 0);

  // Per-lane roles inside one chunk (unchanged from iteration 6/7): A lane
  // owns 16 B of row (lane>>2) at column (lane&3)*16; B lane owns its 8-B
  // fragment slot of each of the 2 steps.
  const int a_row = lane >> 2;
  const int a_col16 = (lane & 3) * 16;
  const int64_t a_base = static_cast<int64_t>(k_lo);
  // B slot base for the slice's FIRST GLOBAL k32 step (k_lo/kDummaTileK) -
  // the packed [n_tile][k_step][lane][8] layout is indexed by the GLOBAL
  // step (iteration-6 repair 2).
  const int64_t b_slot_base =
      static_cast<int64_t>(ntile) * total_steps * kPackedTileBytes +
      (k_lo / kDummaTileK) * kPackedTileBytes +
      lane * kPackedSlotBytes;  // + step*kPackedTileBytes

  // Prologue: stage chunk 0 into the single buffer (the compiler inserts
  // the vmcnt wait before the ds_write).
  {
    const int4 av = *reinterpret_cast<const int4*>(
        a + static_cast<int64_t>(a_row) * k + a_base + a_col16);
    *reinterpret_cast<int4*>(smem + a_row * kStageAStride + a_col16) = av;
    const uint64_t bv0 =
        *reinterpret_cast<const uint64_t*>(packed_b + b_slot_base);
    *reinterpret_cast<uint64_t*>(smem + kStageABufBytes +
                                 lane * kPackedSlotBytes) = bv0;
    if (1 < slice_steps) {
      const uint64_t bv1 = *reinterpret_cast<const uint64_t*>(
          packed_b + b_slot_base + kPackedTileBytes);
      *reinterpret_cast<uint64_t*>(smem + kStageABufBytes +
                                   kPackedTileBytes +
                                   lane * kPackedSlotBytes) = bv1;
    }
  }

  for (int c = 0; c < nchunks; ++c) {
    // Single buffer: the current chunk is always at the smem base.
    const signed char* buf = smem;

    // Prefetch chunk c+1 from global into registers (no wait yet). The
    // values are consumed only by the commit stores below, so the compiler
    // issues these loads before (or at worst alongside) the MMAC burst and
    // lands the vmcnt wait at the ds_write - the overlap that removes the
    // iteration-5 per-step vmcnt(0)-before-mmac exposure.
    int4 a_v;
    uint64_t b_v0, b_v1;
    const bool has_next = (c + 1 < nchunks);
    if (has_next) {
      const int nc = c + 1;
      const int64_t a_off =
          a_base + static_cast<int64_t>(nc) * kStageK + a_col16;
      a_v = *reinterpret_cast<const int4*>(
          a + static_cast<int64_t>(a_row) * k + a_off);
      const int64_t boff = b_slot_base +
                           static_cast<int64_t>(nc) * kStageSteps *
                               kPackedTileBytes;
      b_v0 = *reinterpret_cast<const uint64_t*>(packed_b + boff);
      if (nc * kStageSteps + 1 < slice_steps) {
        b_v1 = *reinterpret_cast<const uint64_t*>(packed_b + boff +
                                                  kPackedTileBytes);
      }
    }

    // Compute the current chunk from the single LDS buffer: 1 or 2
    // k-ascending k32 steps, each ds_read (A) -> ds_read (B) -> v_mmac
    // (nsteps clamped to the staged chunk width, iteration-6 repair 1).
    const int s0 = c * kStageSteps;
    const int nsteps = (slice_steps - s0 > kStageSteps)
                           ? kStageSteps
                           : (slice_steps - s0);  // 1 or 2
    const signed char* a_buf = buf;
    const signed char* b_buf = buf + kStageABufBytes;
    for (int j = 0; j < nsteps; ++j) {
      du::dumma::du_load_matrix_sync(a_frag, a_buf + j * kDummaTileK,
                                     kStageAStride);
      const uint64_t bv = *reinterpret_cast<const uint64_t*>(
          b_buf + j * kPackedTileBytes + lane * kPackedSlotBytes);
      __builtin_memcpy(b_frag.x, &bv, sizeof(bv));
      du::dumma::du_mma_sync(acc_frag, a_frag, b_frag, acc_frag);
    }

    // Commit the prefetched chunk into the single buffer (the compiler
    // inserts the vmcnt wait here, after the MMAC burst above; the write
    // follows the reads of the same buffer in program order).
    if (has_next) {
      const int nc = c + 1;
      *reinterpret_cast<int4*>(smem + a_row * kStageAStride + a_col16) =
          a_v;
      const int64_t boff = b_slot_base +
                           static_cast<int64_t>(nc) * kStageSteps *
                               kPackedTileBytes;
      *reinterpret_cast<uint64_t*>(smem + kStageABufBytes +
                                   lane * kPackedSlotBytes) = b_v0;
      if (nc * kStageSteps + 1 < slice_steps) {
        *reinterpret_cast<uint64_t*>(smem + kStageABufBytes +
                                     kPackedTileBytes +
                                     lane * kPackedSlotBytes) = b_v1;
      }
    }
  }

  du::dumma::du_store_matrix_sync(
      partials +
          (static_cast<int64_t>(split) * ntiles + ntile) *
              (kDummaTileM * kDummaTileN),
      acc_frag, kDummaTileN, du::dumma::mem_row_major);

  // Iteration 14 fused tail: publish this block's partial (above), fence,
  // then the last-arriving block for this ntile sums the split_k planes in
  // ascending split order (bit-identical to the iteration-3 combine
  // kernel), scales with the same expression, and stores bf16 - so the
  // timed Graph needs only this ONE kernel per replay.
  __threadfence();
  if (lane == 0) {
    s_arrived = atomicAdd(&counters[ntile], 1);
  }
  __syncthreads();
  const int arrived = s_arrived;
  if (arrived % split_k == split_k - 1) {
    const int plane = ntile * (kDummaTileM * kDummaTileN);
    const int n0 = ntile * kDummaTileN;
    int32_t total[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      total[i] = 0;
    }
    for (int s = 0; s < split_k; ++s) {
      const int32_t* plane_ptr =
          partials + (static_cast<int64_t>(s) * ntiles) *
                         (kDummaTileM * kDummaTileN) +
          plane;
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        total[i] += plane_ptr[lane + kDummaWaveSize * i];
      }
    }
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int e = lane + kDummaWaveSize * i;
      const int row = e >> 4;
      const int col = e & 15;
      const float scaled = static_cast<float>(total[i]) * x_scale[row] *
                           weight_scale[n0 + col];
      out[static_cast<int64_t>(row) * n + n0 + col] = hip_bfloat16(scaled);
    }
  }
}

// gfx928 DUMMA INT8 cross-block split-K partial kernel for tp8_wqkv_a_m16 -
// iteration 16 (REPAIR of the iteration-15 candidate; the fast mapping is
// preserved byte-for-byte). Iteration 15 measured 60.158 us (single sample)
// with 24,555/24,576 mismatches; the SOLE defect was the B-staging global
// address: each stage chunk was read as one contiguous 4,096-B window of the
// packed [n_tile][k_step][lane][8] layout starting at
// (4*tile64*k_steps + k_lo/32)*512 + tid*16 + c*4,096. The packed layout is
// indexed by the GLOBAL step ((nt*k_steps + step)*512 + lane*8, each ntile's
// 128 steps form a contiguous 64 KiB run), so a 4-ntile x 2-step chunk is
// NOT contiguous: only wave 0 of chunk 0 read correct bytes, waves 1-3 read
// ntile 4*tile64's later steps, and every chunk c>=1 read step k_lo/32+8c
// instead of k_lo/32+2c. The repair addresses each thread's 16-B slot pair
// with the exact per-(ntile, step) packed offset (same one 16-B
// global_load_dwordx4 per thread per chunk and the same 16-B ds_write_b128
// into the unchanged slot-mirror LDS layout at byte tid*16, so the fragment
// reads b_buf + (wave*2 + step)*512 + lane*8 hit exactly their own slot);
// the chunk stride becomes kStageSteps*kPackedTileBytes = 1,024 B. A
// staging, the double-buffer pipeline, both __syncthreads() per stage, the
// plane publish, the separate combine kernel, the dispatch and all
// fallbacks are byte-identical to the measured iteration-15 candidate.
// Iteration 15 architecture (TP4-validated 4-WAVE block-N=64 stage-K=64
// DOUBLE-buffer design; the TP4 lineage measured 17.344 us median / 17.616
// us p90 on this exact (M=16, K=4096, N=1536) local shape at split_k=10,
// grid 240 = exactly 2 x 120 CUs, 4 x wave64 per block, one 16x16 quadrant
// per wave sharing one staged A chunk). Differences from the TP4 winner are
// forced by this session's packed-B weight layout: B is staged from the
// packed [n_tile][k_step][lane][8] fragment slots (every lane issues one
// 16-B int4 global load = two adjacent 8-B slots of the same (ntile, step)
// pair and commits one 16-B ds_write_b128 into an LDS mirror of the slot
// layout, so the B fragment read is the same single ds_read_b64 per lane
// per k32 step as the accepted 1-wave kernels), instead of the TP4 raw
// [K,N] 16-B row loads; the A chunk is staged by wave 0 (64 lanes x 16 B =
// the full 16x64 raw A tile, row stride 64) exactly as the TP4 winner; the
// per-stage pipeline is identical (stage i+1's int4 loads issue before the
// current stage's 2 x v_mmac_i32_16x16x32_i8 per wave, the vmcnt wait
// lands at the alternate-buffer ds_write, and two __syncthreads() per
// stage - the pure-HIP form of the TP4 "raw first s_barrier + compiler-
// managed second barrier" sync mode; no raw asm per the round policy);
// partials publish to the unchanged [split][ntile][16][16] plane layout
// (per-wave 16x16 quadrant), and the separate 96-block x 256-thread
// combine kernel returns to the timed Graph (the TP4 lineage measured the
// atomic last-arriver fused finalize on this design at +1.52 us - 18.864
// vs 17.344 - so the fused tail is NOT used here; the iteration-14 fused
// 1-wave kernel stays as a preserved symbol). split_k is runtime-
// selectable (default 10; env METAINFER_W8A8_SPLIT_K), sliced in
// stage-K=64 units non-uniformly (4 x 7 + 6 x 6 stages at split10; every
// boundary stage-aligned, k-ascending per split). Dynamic LDS = 10,240 B
// per block. Exact bf16: identical staged bytes fed to the same v_mmac in
// the same order; int32 accumulation order unchanged (k-ascending per
// split, ascending split sum).
__global__ __launch_bounds__(k4wBlockThreads) void
w8a8_dumma_m16n16k32_4wave_sk_astage_packedb_dbuf_partial_kernel(
    const int8_t* __restrict__ a,           // [16, K] row-major int8
    const int8_t* __restrict__ packed_b,    // [n_tile][k_step][lane][8]
    int32_t* __restrict__ partials,         // [split_k * ntiles][16][16]
    int n,
    int k,
    int split_k) {
  extern __shared__ __align__(16) signed char smem[];
  const int tid = static_cast<int>(threadIdx.x);
  const int wave = tid >> 6;  // 0..3 -> quadrant ntile64*4 + wave
  const int lane = tid & (kDummaWaveSize - 1);
  const int ntiles64 = n / (k4wBlockNTiles * kDummaTileN);  // 24
  const int tile64 = static_cast<int>(blockIdx.x) % ntiles64;
  const int split = static_cast<int>(blockIdx.x) / ntiles64;
  // Non-uniform stage-based K slicing (stage-K=64 units, TP4 formula):
  // split10 -> 4 x 7 + 6 x 6 stages; every boundary stage-aligned.
  const int total_stages = k / kStageK;  // 64
  const int base_stages = total_stages / split_k;
  const int extra_stages = total_stages - base_stages * split_k;
  const int begin_stage = split * base_stages +
                          (split < extra_stages ? split : extra_stages);
  const int split_stages = base_stages + (split < extra_stages ? 1 : 0);
  const int k_lo = begin_stage * kStageK;

  signed char* a_tile = smem;                       // [2][1024]
  signed char* b_tile = smem + 2 * k4wStageABytes;  // [2][4096]

  du::dumma::DUFragment<du::dumma::matrix_a, kDummaTileM, kDummaTileN,
                        kDummaTileK, signed char, du::dumma::row_major>
      a_frag;
  du::dumma::DUFragment<du::dumma::matrix_b, kDummaTileM, kDummaTileN,
                        kDummaTileK, signed char, du::dumma::row_major>
      b_frag;
  du::dumma::DUFragment<du::dumma::accumulator, kDummaTileM, kDummaTileN,
                        kDummaTileK, int> acc_frag;
  du::dumma::du_fill_fragment(acc_frag, 0);

  // Staging roles: A - wave 0 (tid < 64) loads 16 B per lane, raw [16][64]
  // row stride 64; B - every thread loads one 16-B int4 = two adjacent 8-B
  // packed slots (same (ntile, step) pair, adjacent lanes), committed as
  // one 16-B ds_write_b128 to the slot-mirror LDS layout at the same byte
  // offset (each lane's later 8-B fragment read hits exactly its own slot).
  const int a_row = tid >> 2;
  const int a_col16 = (tid & 3) * 16;
  const int64_t a_base = static_cast<int64_t>(k_lo);
  // B per-thread roles: tid maps to one 16-B slot pair inside exactly ONE
  // (ntile, step) pair: pair index p = tid >> 5 (p>>1 = ntile within the
  // block's 4-tile group, p&1 = step within the K=64 chunk), 16-B
  // sub-offset (tid&31)*16. The packed [n_tile][k_step][lane][8] layout is
  // indexed by the GLOBAL step: byte ((nt*k_steps + step)*64 + lane)*8, so
  // the pair (w, s) of chunk c sits at (4*tile64 + w)*k_steps + k_lo/32 +
  // 2c + s - the 4-ntile x 2-step chunk is NOT one contiguous 4,096-B
  // global run (iteration-16 repair: the iteration-15 code read a
  // contiguous window from (4*tile64*k_steps + k_lo/32)*512 + tid*16 + c*
  // 4,096, which is only correct for the first 1,024 B; waves 1-3 consumed
  // ntile 4*tile64's later steps and every chunk c>=1 read step k_lo/32+8c
  // instead of k_lo/32+2c - the sole correctness defect of iteration 15).
  const int b_pair = tid >> 5;  // 0..7
  const int b_w = b_pair >> 1;  // ntile index within the block (0..3)
  const int b_s = b_pair & 1;   // step within the chunk (0..1)
  const int64_t b_off =  // this thread's 16-B slot pair of chunk 0
      (static_cast<int64_t>(tile64 * k4wBlockNTiles + b_w) *
           (k / kDummaTileK) +
       (k_lo / kDummaTileK) + b_s) *
          kPackedTileBytes +
      (tid & 31) * 16;  // 16-B aligned

  // Prologue: stage chunk 0 into buffer 0, then make it visible to all
  // four waves.
  if (tid < kDummaWaveSize) {
    const int4 av = *reinterpret_cast<const int4*>(
        a + static_cast<int64_t>(a_row) * k + a_base + a_col16);
    *reinterpret_cast<int4*>(a_tile + a_row * kStageK + a_col16) = av;
  }
  const int4 bv = *reinterpret_cast<const int4*>(packed_b + b_off);
  *reinterpret_cast<int4*>(b_tile + tid * 16) = bv;
  __syncthreads();

  for (int c = 0; c < split_stages; ++c) {
    const int cur = c & 1;
    const int nxt = cur ^ 1;
    const bool has_next = (c + 1 < split_stages);

    // Prefetch chunk c+1 from global into registers (no wait yet; the
    // compiler issues these before the MMAC burst and lands the vmcnt wait
    // at the alternate-buffer ds_write below - the TP4-validated overlap).
    int4 a_v{};
    int4 b_v{};
    if (has_next) {
      if (tid < kDummaWaveSize) {
        a_v = *reinterpret_cast<const int4*>(
            a + static_cast<int64_t>(a_row) * k + a_base +
            static_cast<int64_t>(c + 1) * kStageK + a_col16);
      }
      // Chunk c+1 = the same (ntile, step-in-chunk) pair two GLOBAL steps
      // later: +kStageSteps*kPackedTileBytes = 1,024 B per chunk (NOT the
      // 4,096-B contiguous-window stride of the iteration-15 bug).
      b_v = *reinterpret_cast<const int4*>(
          packed_b + b_off + static_cast<int64_t>(c + 1) * kStageSteps *
                                 kPackedTileBytes);
    }

    // Consume the current chunk: 2 k-ascending k32 steps per wave, each
    // ds_read (A, shared) -> ds_read (B, own 8-B slot) -> v_mmac.
    const signed char* a_buf = a_tile + cur * k4wStageABytes;
    const signed char* b_buf = b_tile + cur * k4wStageBBytes;
#pragma unroll
    for (int kk = 0; kk < kStageK; kk += kDummaTileK) {
      du::dumma::du_load_matrix_sync(a_frag, a_buf + kk, kStageK);
      const uint64_t bval = *reinterpret_cast<const uint64_t*>(
          b_buf + (wave * kStageSteps + (kk >> 5)) * kPackedTileBytes +
          lane * kPackedSlotBytes);
      __builtin_memcpy(b_frag.x, &bval, sizeof(bval));
      du::dumma::du_mma_sync(acc_frag, a_frag, b_frag, acc_frag);
    }

    // All four waves have consumed buffer cur (pure-HIP form of the TP4
    // raw first s_barrier).
    __syncthreads();

    // Commit the prefetched chunk into the alternate buffer (the compiler
    // inserts the vmcnt wait before these stores, after the MMAC burst).
    if (has_next) {
      if (tid < kDummaWaveSize) {
        *reinterpret_cast<int4*>(a_tile + nxt * k4wStageABytes +
                                 a_row * kStageK + a_col16) = a_v;
      }
      *reinterpret_cast<int4*>(b_tile + nxt * k4wStageBBytes + tid * 16) =
          b_v;
    }
    // The committed chunk is visible to all four waves.
    __syncthreads();
  }

  const int plane_ntile = tile64 * k4wBlockNTiles + wave;
  du::dumma::du_store_matrix_sync(
      partials +
          (static_cast<int64_t>(split) * (n / kDummaTileN) + plane_ntile) *
              (kDummaTileM * kDummaTileN),
      acc_frag, kDummaTileN, du::dumma::mem_row_major);
}

// gfx928 DUMMA INT8 combine + scale kernel for tp8_wqkv_a_m16 - iteration
// 3: sums the split_k workspace int32 planes per output element in
// ascending split order (bit-identical int32 to the serial k-ascending
// accumulation: integer addition is exact and |full dot| << 2^31),
// applies x_scale[m] * weight_scale[n], stores bf16. grid = N/16 blocks x
// 256 threads (4 wavefronts), one output element per thread: thread t owns
// element (row = t >> 4, col = t & 15) of its tile, so each wavefront
// covers 4 rows x all 16 columns with coalesced 16-wide plane reads and
// bf16 stores; no barriers, no LDS.
__global__ __launch_bounds__(4 * kDummaWaveSize) void
w8a8_dumma_m16n16k32_combine_kernel(
    const int32_t* __restrict__ partials,   // [split_k * ntiles][16][16]
    const float* __restrict__ x_scale,      // [16, 1] fp32
    const float* __restrict__ weight_scale, // [N, 1] fp32
    hip_bfloat16* __restrict__ out,         // [16, N] bf16
    int n,
    int split_k) {
  const int ntiles = n / kDummaTileN;
  const int tid = static_cast<int>(threadIdx.x);
  const int row = tid >> 4;  // 0..15
  const int col = tid & 15;  // 0..15
  const int ntile = static_cast<int>(blockIdx.x);
  const int n0 = ntile * kDummaTileN;
  const int plane = (ntile << 8);  // ntile * 16 * 16 int32 elements

  int32_t total = 0;
#pragma unroll
  for (int s = 0; s < split_k; ++s) {
    total += partials[static_cast<int64_t>(s) * ntiles * (kDummaTileM *
                                                          kDummaTileN) +
                      plane + row * kDummaTileN + col];
  }
  const float scaled = static_cast<float>(total) * x_scale[row] *
                       weight_scale[n0 + col];
  out[static_cast<int64_t>(row) * n + n0 + col] = hip_bfloat16(scaled);
}

// Generic elementwise device-to-device copy (used by the identity
// pack_weight bootstrap and valid for every (K, N)).
template <typename T>
__global__ void w8a8_device_copy_kernel(const T* __restrict__ src,
                                        T* __restrict__ dst,
                                        int64_t numel) {
  const int64_t i =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < numel) {
    dst[i] = src[i];
  }
}

// Iteration 5: one-time device permutation of the raw logical [K, N]
// row-major weight into the packed [n_tile][k_step][lane][8] B fragment-slot
// layout for the exact (K, N) == (4096, 1536) shape. One thread owns one
// 8-byte lane slot; slot ((ntile * (k/32) + step) * 64 + lane) * 8 + i gets
// logical weight[step*32 + (lane>>4)*8 + i][ntile*16 + (lane&15)]. Byte
// count is unchanged (K*N), so packed_weight keeps the same numel and
// graph-stable address; runs outside the timed region and Graph capture.
__global__ __launch_bounds__(kCopyBlockThreads) void
w8a8_pack_b_fragslot_kernel(const int8_t* __restrict__ raw,  // [K, N]
                            int8_t* __restrict__ packed,
                            int k,
                            int n) {
  const int64_t tid =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int k_steps = k / kDummaTileK;
  const int64_t total = static_cast<int64_t>(n / kDummaTileN) * k_steps *
                        kDummaWaveSize;  // 786,432 lane slots
  if (tid >= total) {
    return;
  }
  const int lane = static_cast<int>(tid & (kDummaWaveSize - 1));
  const int64_t slot = tid >> 6;  // ntile * k_steps + step
  const int step = static_cast<int>(slot % k_steps);
  const int ntile = static_cast<int>(slot / k_steps);
  const int nn = lane & 15;
  const int k8 = (lane >> 4) << 3;
  const int8_t* src =
      raw + (static_cast<int64_t>(step) * kDummaTileK + k8) * n +
      ntile * kDummaTileN + nn;
  int8_t* dst = packed + tid * kPackedSlotBytes;
#pragma unroll
  for (int i = 0; i < kPackedSlotBytes; ++i) {
    dst[i] = src[static_cast<int64_t>(i) * n];
  }
}

}  // namespace

// Host launcher for torch.ops.zth_w8a8.gemm_out. Runs only on the caller's
// stream; no allocations, no synchronization, no autotuning, no packing.
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
  if (m <= 0 || n <= 0 || k <= 0) {
    return;
  }

  // Exact-shape-guarded DUMMA fast path for the assigned shape
  // tp8_wqkv_a_m16: (M=16, N=1536, K=4096). Iteration 15 architecture:
  // the TP4-validated 4-WAVE block-N=64 stage-K=64 DOUBLE-buffered
  // cross-block split-K partial kernel (256 threads = 4 wave64, one 16x16
  // quadrant per wave sharing one staged A chunk; B staged from the packed
  // [n_tile][k_step][lane][8] fragment-slot layout) + the separate
  // 96-block x 256-thread combine/scale kernel, both launched on the
  // caller stream inside the timed Graph. Default split_k=10 -> grid =
  // 24*10 = 240 blocks = exactly 2 x 120 CUs (the TP4-validated occupancy
  // for this exact shape: 17.344 us median / 17.616 us p90 at TP4).
  // Dynamic LDS = 10,240 B per block (2 x 5,120-B ping-pong buffers),
  // independent of split_k. Int32 partial planes live in the caller-
  // provided workspace: plane (split, ntile) is 16x16 int32 at
  // partials[(split*ntiles + ntile) * 256], so split_k*96 planes total.
  if (m == 16 && n == 1536 && k == 4096) {
    const int ntiles = n / kDummaTileN;       // 96 (16x16 tiles)
    const int ntiles64 = n / (k4wBlockNTiles * kDummaTileN);  // 24
    const int64_t plane_bytes =
        static_cast<int64_t>(kDummaTileM) * kDummaTileN * sizeof(int32_t);
    if (workspace != nullptr && workspace_bytes >= plane_bytes) {
      // split_k selection: METAINFER_W8A8_SPLIT_K override for trusted
      // occupancy-probe candidates, default 10 (the TP4-validated
      // non-uniform split for this shape: grid 240 = exactly 2 x 120 CUs
      // with 24 block-N=64 tiles); clamped to [1, min(16, K/32)] and
      // workspace capacity. The staged-chunk design keeps dynamic LDS at
      // 10,240 B per block for every split_k (no 64 KiB A-slice concern).
      int split_k = 10;
      if (const char* env = getenv("METAINFER_W8A8_SPLIT_K")) {
        const long parsed = strtol(env, nullptr, 10);
        if (parsed >= 1 && parsed <= 16) {
          split_k = static_cast<int>(parsed);
        }
      }
      const int64_t capacity =
          workspace_bytes / plane_bytes / static_cast<int64_t>(ntiles);
      if (static_cast<int64_t>(split_k) > capacity) {
        split_k = static_cast<int>(capacity);
      }
      const int64_t stage_capacity = k / kStageK;
      if (static_cast<int64_t>(split_k) > stage_capacity) {
        split_k = static_cast<int>(stage_capacity);
      }
      if (split_k < 1) {
        split_k = 1;
      }

      int32_t* partials = static_cast<int32_t*>(workspace);
      // Iteration 15: the TP4-validated 4-wave block-N=64 stage-K=64
      // double-buffer partial kernel (default split_k=10 -> grid 240 =
      // exactly 2 x 120 CUs; 10,240 B dynamic LDS/block) + the separate
      // 96-block x 256-thread combine kernel, both inside the timed Graph.
      // The TP4 lineage measured the atomic last-arriver fused finalize on
      // this design at +1.52 us (18.864 vs 17.344), so the fused tail is
      // NOT used here; the iteration-14 fused 1-wave kernel stays as a
      // preserved symbol. Exact bf16: same staged bytes, same int32
      // accumulation order (k-ascending per split, ascending split sum).
      const unsigned grid = static_cast<unsigned>(ntiles64 * split_k);
      hipLaunchKernelGGL(
          w8a8_dumma_m16n16k32_4wave_sk_astage_packedb_dbuf_partial_kernel,
          dim3(grid),
          dim3(k4wBlockThreads),
          static_cast<unsigned>(k4wStageTotalBytes),
          stream,
          a,
          b,
          partials,
          n,
          k,
          split_k);
      hipLaunchKernelGGL(
          w8a8_dumma_m16n16k32_combine_kernel,
          dim3(static_cast<unsigned>(ntiles)),
          dim3(4 * kDummaWaveSize),
          0,
          stream,
          partials,
          x_scale,
          weight_scale,
          reinterpret_cast<hip_bfloat16*>(out),
          n,
          split_k);
      return;
    }
    // Caller provided no usable workspace: keep the accepted iteration-2
    // workspace-free in-block split-K=2 kernel (now with the packed-B
    // fragment-slot load) so the fast path stays correct for such callers
    // (the fixed-API contract always provides capacity for 16 planes, so
    // the harness takes the split-K path above).
    const unsigned grid = static_cast<unsigned>(n / kDummaTileN);
    hipLaunchKernelGGL(
        w8a8_dumma_m16n16k32_2wave_sk2_packedb_kernel,
        dim3(grid),
        dim3(2 * kDummaWaveSize),
        0,
        stream,
        a,
        b,
        x_scale,
        weight_scale,
        reinterpret_cast<hip_bfloat16*>(out),
        n,
        k);
    return;
  }

  // Generic scalar fallback for every other (m, n, k) (including the paired
  // M=2 shape with the same (N, K)).
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
      reinterpret_cast<hip_bfloat16*>(out),
      m,
      n,
      k);
}

// Host launcher for the optional torch.ops.zth_w8a8.pack_weight. For the
// exact (K, N) == (4096, 1536) weight it performs the iteration-5 one-time
// device permutation into the packed [n_tile][k_step][lane][8] B fragment
// slots (byte count unchanged: 6,291,456 B); for every other shape it stays
// the identity device-to-device copy. packed_weight_scale[n] =
// weight_scale[n] always. Out of the timed region and out of Graph capture.
extern "C" void launch_pack_w8a8_weight(
    const int8_t* raw_weight,
    const float* weight_scale,
    int8_t* packed_weight,
    float* packed_weight_scale,
    int k,
    int n,
    hipStream_t stream) {
  const int64_t weight_numel = static_cast<int64_t>(k) * n;
  if (weight_numel > 0) {
    if (k == kTargetK && n == kTargetN) {
      const int64_t slots =
          static_cast<int64_t>(n / kDummaTileN) * (k / kDummaTileK) *
          kDummaWaveSize;  // 786,432 lane slots of 8 B
      const unsigned grid = static_cast<unsigned>(
          (slots + kCopyBlockThreads - 1) / kCopyBlockThreads);
      hipLaunchKernelGGL(
          HIP_KERNEL_NAME(w8a8_pack_b_fragslot_kernel),
          dim3(grid),
          dim3(kCopyBlockThreads),
          0,
          stream,
          raw_weight,
          packed_weight,
          k,
          n);
    } else {
      const unsigned grid = static_cast<unsigned>(
          (weight_numel + kCopyBlockThreads - 1) / kCopyBlockThreads);
      hipLaunchKernelGGL(
          HIP_KERNEL_NAME(w8a8_device_copy_kernel<int8_t>),
          dim3(grid),
          dim3(kCopyBlockThreads),
          0,
          stream,
          raw_weight,
          packed_weight,
          weight_numel);
    }
  }
  if (n > 0) {
    const unsigned grid = static_cast<unsigned>(
        (n + kCopyBlockThreads - 1) / kCopyBlockThreads);
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(w8a8_device_copy_kernel<float>),
        dim3(grid),
        dim3(kCopyBlockThreads),
        0,
        stream,
        weight_scale,
        packed_weight_scale,
        static_cast<int64_t>(n));
  }
}
// @@end
