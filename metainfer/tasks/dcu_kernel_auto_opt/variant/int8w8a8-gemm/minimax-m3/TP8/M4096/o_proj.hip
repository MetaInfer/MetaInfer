// @@variant shape=minimax_tp8_o_proj_m4096 commit=bdaa6897e0b608089d609eaa1639ea1ea170f65f added=2026-08-29
//   median_us=397.4 p90_us=398 speedup=50.06 baseline_us=1.99e+04
//   source=minimaxm3-dsh-tp8-m4096-1-b0482833
// @@variant shape=minimax_tp8_o_proj_m4096 (iteration 2)
// INT8 W8A8 GEMM HIP implementation for Hygon K500SM_AI / gfx928.
//
// Worker: worker_2 (physical GPU 2), assigned shape:
//   minimax_tp8_o_proj_m4096 : M=4096, N=6144, K=1024
//
// Bootstrap strategy (iteration 1, correctness-first but a usable profiling
// baseline - a large-Prefill scalar K loop would not be):
//   * Large-prefill path (exact (m, n, k) == (4096, 6144, 1024)):
//       native INT8 DUMMA m16n16k32 with int32 accumulation; one 128x64
//       output tile per block; four wavefronts (256 threads); each wave owns
//       a 64x32 quadrant built from eight m16n16k32 accumulator fragments;
//       the block cooperatively vector-loads A[128,64] and B[64,64] into one
//       single-buffered 64-K LDS stage (15,360 B total: 128*80 + 64*80 with
//       16 B padding per row for bank skew); two __syncthreads per stage
//       (one after the cooperative load, one before the next stage
//       overwrites LDS - both on all-thread paths); A and B fragments are
//       read from LDS with the library du_load_matrix_sync row-major loaders
//       (B tile is the raw [K, N] layout because launch_pack_w8a8_weight is
//       the bootstrap identity device-to-device copy; later Parallel explore
//       rounds may introduce a packed layout together with matching fragment
//       loads); fused dot * x_scale[m] * weight_scale[n] epilogue stored
//       directly as bf16 from the accumulator fragments (one coalesced
//       8-byte store per lane per fragment).  Grid dim3(96, 32) = 3072
//       blocks dwarfs the 120 CUs, so no split-K is needed.
//   * Generic scalar int8/int32 fallback for every unmatched (m, n, k),
//       including all small-M API cases (M=2, M=16) and any M in (0, 4096]
//       with the same (K, N); the fallback reads the identity [K, N] weight
//       layout (packing is identity in bootstrap).
//   * launch_pack_w8a8_weight: identity device-to-device copy of the raw
//       [K, N] weight and of the [N, 1] scales for every (k, n) - bootstrap
//       only.  Packing never happens inside the timed GEMM and never inside
//       Graph capture.
//
// Iteration 2 (operand-reuse / staging round):
//   Decision: cooperative A+B LDS staging is KEPT, direct global fragment
//   loads are REJECTED.  Per 128x64 macro-tile reuse: each A element is used
//   by 64 output columns within the block (2 waves share a row-half) and
//   re-fetched by 96 N-slice blocks; each B element is used by 128 output
//   rows and re-fetched by 32 M-slice blocks.  Direct loads would move the
//   fragment reads from LDS to global/L2 with 2x fetch amplification per
//   block (A 16 KiB vs 8 KiB unique, B 8 KiB vs 4 KiB unique per stage:
//   1.2 GiB vs 603 MB total wavefront traffic) and add per-wave L2-latency
//   waits behind the two __syncthreads per stage; the worker-29 TP4 gate_up
//   measured the direct-A arm at 7.9x slower (3444.26 vs 220.71 us) - the
//   M=16 decode direct-load win does not transfer to M=4096 prefill.  The
//   binding cost is instead the LDS fragment path: the gfx928 code object
//   shows the raw-[K,N] row_major B loader compiling to eight per-byte
//   ds_read_u8 (k-strided by ldm) plus a mask/OR reassembly VALU chain on
//   both operands (PMC: 9.24M LDS instr, 23.59M bank conflicts = 2.55/instr,
//   12.46M LDS waits).  This round applies the worker-29 TP8 o_proj lineage
//   iteration-10/12 mechanism for the exact same 128x64x64 geometry: the
//   weight pack for (k, n) == (1024, 6144) becomes n-major [N, K]
//   (packed[n*K + kk] == raw[kk*N + n]; every B tile of one (n-tile,
//   K-stage) is one contiguous 4 KiB vector stream), B is staged n-major in
//   LDS (b_tile[n][k], 64 rows x 80 B - the same 5,120 B, same
//   16-byte-aligned bank-skewed stride), and BOTH operand fragments are
//   filled by load_fragment8 (one 8-byte LDS read per fragment, the same 8
//   bytes in the same x[0..7] order the library loaders produce => v_mmac
//   operand bit pattern and exact int32 accumulation unchanged; mismatch 0 /
//   max_abs_error 0.0 expected).  Tile geometry, cooperative int4 staging,
//   two-barrier single-buffered 64-K stage, per-accumulator MMAC order, the
//   coalesced epilogue, the exact (4096, 6144, 1024) guard and the scalar
//   fallback (now decoding the n-major layout for (k, n) == (1024, 6144))
//   are otherwise untouched.  Falsifiable prediction: lds_instructions and
//   lds_bank_conflicts drop sharply (8-byte fragment reads instead of 8
//   strided byte loads), lds_wait_instructions and valu_instructions drop
//   (mask/OR chains gone), and the median/P90 improve; the removal of
//   reassembly temporaries may also lower arch_vgpr 80 -> ~61-64, which
//   would move residency from 3 to 4 blocks/CU (the trusted occupancy probe
//   set [2,3,4,5]: 4 blocks x 256 threads x <=64 VGPR = 65,536 exactly
//   fits, 4 x 15,360 B LDS = 61,440 B fits 64 KiB).
//
// Iteration 6 (dead-barrier peel round; source restored to the accepted
// iteration-2 digest 92917037c97822f73ae519ba94d92219946b3fe50dafcbf351cd175a38ff2e60
// after the iteration-5 infrastructure kill):
//   The stage loop emits two __syncthreads per stage (32 per block for
//   K=1024/64 = 16 stages): barrier A after the cooperative staging writes
//   (writes -> reads) and barrier B after the consume (reads -> next stage
//   overwrites LDS).  Barrier B of the FINAL stage (k0 = 960) is dead work:
//   no successor stage overwrites LDS, and the epilogue reads only
//   accumulator registers plus global x_scale/weight_scale - it never
//   touches LDS.  This round guards that barrier with
//   `if (k0 + kStageK < k)` so the final stage runs its last MMAC group and
//   enters the epilogue one barrier earlier (32 -> 31 barriers per block,
//   -3.1%).  Everything else is untouched: 128x64 tile, 256 threads,
//   single-buffered 64-K stage (15,360 B LDS, 4 blocks/CU, arch_vgpr 64),
//   cooperative int4 A+B staging, load_fragment8 on both operands (bit-
//   identical int32 order), coalesced 8-byte epilogue, exact (4096,6144,1024)
//   guard, n-major [N,K] pack, and the scalar fallback.  The guard is
//   wavefront-uniform (k0 and k are scalars), so there is no divergence.
//   Falsifiable claim: barrier-stall time per block drops by ~1/32 of the
//   barrier cost, so median/P90 stay flat-to-improved (down_proj iter-12's
//   accepted variant dropped its dead last barrier for the same reason);
//   a >2% regression versus 449.53/450.35 us falsifies the mechanism and the
//   accepted source is restored.  Expected ISA delta: steady-state loop still
//   has 2 s_barrier for stages 0..14, the final stage path has only 1.
//
// Iteration 8 (resource round: tune waves per block, VGPR live ranges and LDS
// footprint from measured occupancy; recheck the best tile family with normal
// median/P90 measurements):
//   Measured occupancy of the current-best (iteration-6) source (digest
//   8d133a8d90204d4e4e36922249701b2ea78cad8243e89bcd9affbe17d3f667e0): 16
//   resident waves/CU is the hard ceiling - arch_vgpr 64 x 16 waves = 1,024
//   VGPRs/CU exactly (65 VGPRs -> 3 blocks/CU, the measured +6% regression of
//   iteration 4) AND LDS 4 x 15,360 B = 61,440 B (93.75% of 64 KiB).  5
//   blocks/CU is doubly unreachable (it would need VGPR <= 51 - 8 accumulator
//   chains alone are 32 VGPRs plus 12 fragment VGPRs - AND LDS <= 13,107 B),
//   and the probe set [2,3,4] is monotonically worse below 16 waves
//   (iteration 3 double-buffer at 2 blocks/CU = 8 waves: 894 us; iteration 4
//   128x96 at 3 blocks/CU = 12 waves: 477.6 us; accepted 4 blocks/CU = 16
//   waves: 446.4 us).  The one resource axis never probed AT the 16-wave
//   ceiling is waves PER BLOCK.  This round redistributes the same 16
//   waves/CU from 4 blocks x 4 waves to 2 blocks x 8 waves by doubling the
//   M-tile to 256x64 (512 threads = 8 waves of 64; each wave still owns a
//   64x32 quadrant of eight m16n16k32 int32 accumulators, so the per-wave
//   instruction stream, VGPR live ranges and per-wave LDS fragment footprint
//   are IDENTICAL by construction - the probe is a pure block-geometry
//   change).  Per-block LDS grows 15,360 -> 25,600 B (A[256,80] + B[64,80]);
//   per-CU LDS is unchanged at 51,200 B (2 x 25,600 <= 64 KiB) and per-CU
//   VGPR is unchanged at 1,024.  A staging stays 2 int4s/thread over 512
//   threads (rows [0,128) + [128,256)); B staging (256 int4s) is guarded to
//   threads 0..255 via `if (b_n < kBlockN)` so the 64-row b_tile index cannot
//   overflow.  Barriers: still 2 per stage with the iteration-6 final-stage
//   guard (31 per block), now synchronizing 8 waves instead of 4; grid 3072
//   -> 1536 blocks (16 M-slices x 96 N-slices).  Bit-identical int32
//   discipline holds (same per-accumulator kk0-then-kk1 order, same v_mmac
//   operand bit patterns -> mismatch 0 / max_abs_error 0.0 expected).
//   Everything else is untouched: 64-K single-buffered stage, cooperative
//   int4 staging, load_fragment8 on both operands, n-major [N,K] pack,
//   coalesced 8-byte epilogue, exact (4096,6144,1024) dispatch guard, scalar
//   fallback, and the (unchanged) kernel symbol name - an internal
//   anonymous-namespace symbol the harness profiler matches by name.
//   FALSIFIABLE CLAIM: at the fixed 16-wave ceiling the kernel is
//   LDS-issue/latency-bound (PMC of the same geometry: 2.10M lds_wait vs
//   749,568 vmem_read, 5.54 bank conflicts per ds_read2_b64; iteration 7's
//   prefetch probe was flat at 447.32/448.24, so LDS latency is already
//   hidden by the resident waves), therefore the only changed physical
//   variables - barrier-domain size (8 vs 4 waves per __syncthreads) and
//   block co-residency (2 vs 4 independent barrier domains per CU) - keep
//   median/P90 flat-to-improved vs 446.41/446.99 us; a >2% regression
//   (median > 455.34 or p90 > 455.93) falsifies the 8x2 geometry and the
//   iteration-6 source is restored before the next experiment.  CONFOUND
//   GUARD: if the compiled arch_vgpr lands >= 65, residency drops to 1
//   block/CU (8 waves/CU) and the probe is confounded - restore the
//   iteration-6 source and report regardless of timing.  Expected ISA
//   deltas: grid 3072 -> 1536, workgroup 256 -> 512, LDS 15,360 -> 25,600 B,
//   arch_vgpr 64; per-wave steady-state consume unchanged (6 ds_read2_b64,
//   16 v_mmac, 5 lgkmcnt waits, 31 s_barrier per block); staging: waves 0..3
//   issue 3 ds_write_b128 per stage (2 A + 1 B), waves 4..7 issue 2 (A only);
//   lds_instructions / lds_bank_conflicts / vmem_read flat per output.
//
// Iteration 16 (N-tile expansion 64 -> 128 at the measured winning
// residency; round 15 was killed by the agent infrastructure before a valid
// proposal and does NOT count - the accepted iteration-8 source is restored
// byte-identical and ONE new bounded mechanism is applied):
//   Measured state: accepted iteration-8 source (digest
//   88c9a331d5dac906a4e77c882e7fdde0bfd2f51f685730ad8ce0af49b282d66a) at
//   440.21/440.84 us, 1 block/CU x 8 resident waves (arch_vgpr 72 > 64, so
//   2 blocks/CU never fit; PMC: 2.46M lds_instructions vs 16.71M
//   lds_wait_instructions, 651K vmem_read, 83.7% L2 hit - LDS-latency/issue
//   bound, not HBM- or MMAC-bound).  Every co-residency probe on this shape
//   was falsified (iter 10 double-buffer 2 blocks/CU 669.35; iter 11
//   launch_bounds diet confounded at vgpr 71; iter 14 N=32 diet 632.94; the
//   128x64 16-wave control 446-449), and the winning direction is
//   consistently FEWER, BIGGER blocks per barrier domain (iter 8's 256x64
//   8-wave 1-block beat every 16-wave layout).  DECISION - N-tile expansion
//   64 -> 128 at the UNCHANGED 1 block/CU x 8 waves residency: kBlockN 128,
//   per-wave quadrant 64x64 = sixteen m16n16k32 int32 accumulators
//   (accumulator VGPR 32 -> 64; arch_vgpr ~72 -> ~95-110, which STRUCTURALLY
//   pins 1 block/CU x 8 waves - 2 blocks/CU would need <= 64 VGPR with 64
//   VGPR of accumulators alone, impossible), LDS 25,600 -> 30,720 B
//   (A[256,80] + B[128,80]; B row stride stays 80 = kStageK + kBPad, the
//   canonical 16-byte-aligned five-bank-phase skew, 2-way conflict floor
//   identical to the accepted 64-row B tile), B staging stays one
//   int4/thread (128 rows x 64 k = 512 int4s over exactly 512 threads - the
//   b_n < kBlockN guard becomes always-true and is elided), grid 1536 -> 768
//   (48 x 16), 2 barriers per stage unchanged (31 per block), kernel symbol,
//   n-major pack layout, exact (4096,6144,1024) dispatch guard and scalar
//   fallback unchanged.  Per-output deltas: LDS fragment reads 3 -> 2 B (A
//   read amplification 2 -> 1 at constant B; per wave per stage 12 -> 16
//   ds_read2_b64 for twice the outputs), barrier frequency halves, global
//   read traffic 1.25 -> 0.75 B/output (A staging unchanged at 16,384 B per
//   stage, B staging doubles to 8,192 B per stage = 24,576 B per block per
//   stage).  Bit-identical discipline: each of the 16 accumulators keeps the
//   same kk0-then-kk1 int32 order and each v_mmac operand reads the SAME 8
//   LDS bytes per fragment (the same addresses receive the same bytes) ->
//   mismatch 0 / max_abs_error 0.0 expected; Graph capture + changed-content
//   replay must pass.  COST ACKNOWLEDGED: grid 768/120 CUs = 6.4 blocks/CU
//   -> 7 block-waves vs 13 today (+~9% makespan ceiling from tail
//   quantization: 6 full waves + one 48-block partial wave), which must be
//   repaid by the per-output LDS-read/barrier amortization; the per-output
//   instruction reductions are the only changed physical variables.
//   FALSIFIABLE CLAIM: median/P90 flat-to-improved vs the accepted
//   440.21/440.84 us; a >2% regression (median > 449.01 or p90 > 449.66)
//   falsifies the 256x128 extension and the iteration-8 source (digest
//   88c9a331...) is restored before the next experiment.  CONFOUND GUARD:
//   the exact code object of the timed run must show vgpr_count <= 128 with
//   scratch == 0 and vgpr_spill_count == 0, grid 768, workgroup 512,
//   lds_bytes 30,720, residency 1 block/CU (any spill/scratch or residency
//   change confounds the probe - restore and report regardless of timing).
//   Expected ISA steady state per wave per stage: 16 ds_read2_b64 (8 per
//   kk) + 32 v_mmac (vs 12 + 16), still 3 global_load_dwordx4 ->
//   ds_write_b128 x3 per thread (2 A + 1 B, all 8 waves now) -> 2 s_barrier
//   (31 per block); epilogue 16 x 8-byte coalesced stores per lane per wave.
//   Expected PMC: vmem_read ~651K -> ~326K (halved - fewer blocks),
//   vmem_write ~98K flat (same output bytes), lds_instructions ~2.46M ->
//   ~1.64M (-33%), lds_bank_conflicts ~13.37M -> ~8.9M (-33%, same 2-way
//   floor), lds_wait ~16.71M -> ~11M, valu_instructions ~14.24M flat per
//   output, l2_hit_rate flat-to-slightly-lower (A-slice reuse drops 96 ->
//   48 blocks per slice).  Rejected alternatives for this round (evidence,
//   not re-litigation): raw asm (policy); any occupancy/co-residency change
//   (iters 10/11/14 and the 128x64 16-wave control all falsified or
//   confounded); one-stage-ahead global prefetch (iter 13 neutral at 442.57
//   - the global round trip is already hidden); N=96 tile (grid 1024 =
//   8.53 blocks/CU -> 9 waves, +5.5% quantization, for only -22% per-output
//   LDS reads - dominated by N=128's -33% at the same barrier amortization);
//   M-tile 512 (per-wave 128x32 keeps A at 2 B/output - strictly worse than
//   N=128's 1 B/output); stage 128 (measured-saturated axis on this
//   lineage); split-K (MxN grid dwarfs 120 CUs); 3/4/5 blocks/CU
//   (VGPR-infeasible at 512 threads).
//
// Iteration 17 (final conditional round; control plane phase hip_only,
// plateau=false, raw_inline_asm_allowed=false, skill_allowed=false - no prior
// ISA-guided round recorded a compiler limitation, and recent_valid_
// improvements_percent [-0.53, -30.45, +2.33] does not prove a plateau, so
// raw inline asm stays forbidden and this is ONE HIP-only consolidation
// change):
//   Measured state: accepted iteration-16 source (digest
//   45ca0cdc2d38470c7131dae2fa2307c04e7e650e2cec468abf3eae0e0c398d6a) at
//   430.21/430.95 us, 1 block/CU x 8 resident waves, arch_vgpr 112, grid 768.
//   The fresh PMC of the exact accepted object (iteration-17 cache) is still
//   LDS-latency-dominated: lds_wait_instructions 8.89M vs lds_instructions
//   1.87M (4.75x; the 128x64 16-wave geometry that measured flat for
//   iteration 7's read-prefetch probe had lds_wait 2.10M BELOW lds_instr
//   2.56M - latency was hidden there by 4 waves/SIMD, and that flatness does
//   NOT transfer to 2 waves/SIMD).  The exact-source ISA of the accepted
//   256x128 object (iteration-17 current-best-isa) shows the consume per kk:
//   the first group is 6x ds_read2_b64 (4 B + 2 A) with staggered lgkmcnt
//   waits interleaved into the first 8 v_mmac, but the rows-32-47 and rows-
//   48-63 A fragments of the wave's 64x64 quadrant are loaded JUST-IN-TIME
//   after that first MMAC group (ds_read2_b64 at 0x3754/0x375C, standalone
//   s_waitcnt lgkmcnt(2) at 0x3764, first acc20 v_mmac at 0x3768 - ~2
//   instructions of slack = the full LDS latency is exposed twice per stage
//   per wave).  DECISION - intra-stage A-fragment read-prefetch at the
//   16-accumulator/8-wave geometry: the rows-32-47/48-63 A fragments move
//   into dedicated fragment variables (a_frag2/a_frag3) loaded in the FIRST
//   read group (all 8 ds_read2_b64: 4 B + 4 A before any v_mmac), giving
//   those two reads ~16-32 instructions of MMAC-issue slack per kk.  LDS
//   traffic is UNCHANGED (still 8 ds_read2_b64 per kk = 16 per stage per
//   wave, the same 8 LDS bytes per fragment at the same addresses) and the
//   ISA already allocates four distinct A-fragment register groups
//   (v[88:91], v[100:103], v[104:107], v[108:111] = 16 VGPR), so arch_vgpr
//   stays ~112 (<= 128 -> 1 block/CU x 8 waves unchanged, scratch 0, no
//   spills).  Bit-identical discipline holds: each accumulator keeps its
//   kk0-then-kk1 int32 order, the v_mmac operand bit patterns are unchanged,
//   and the only delta is when the same LDS bytes enter the A-fragment
//   registers -> mismatch 0 / max_abs_error 0.0 expected; Graph capture +
//   changed-content replay must pass.  Everything else is byte-for-byte the
//   accepted iteration-16 structure: 256x128 tile, 512 threads, per-wave
//   64x64 quadrant of sixteen accumulators, single-buffered 64-K stage
//   (A[256,80] + B[128,80] = 30,720 B), 2 barriers per stage with the
//   iteration-6 final-stage peel (31 per block), cooperative int4 staging
//   with the b_n < kBlockN guard, load_fragment8 on both operands, coalesced
//   8-byte epilogue, exact (4096,6144,1024) dispatch guard, n-major [N,K]
//   pack, scalar fallback, unchanged kernel symbol.  EXPECTED DELTAS to
//   verify the mechanism: ISA steady state - all 8 ds_read2_b64 issue before
//   the first v_mmac and the standalone lgkmcnt wait at 0x3764 disappears
//   (merged into the staggered waits); PMC - lds_instructions ~1.87M FLAT,
//   lds_bank_conflicts ~8.65M FLAT (same 2-way floor), lds_wait 8.89M ->
//   ~6.5-7.5M (-15-25%), vmem_read ~454K FLAT, vmem_write ~98K FLAT,
//   valu_instructions ~13.25M FLAT; launch record - grid 768, workgroup 512,
//   lds 30,720, arch_vgpr ~112, scratch 0, residency 1 block/CU.  FALSIFIABLE
//   CLAIM: with the two just-in-time A reads per kk covered by MMAC-issue
//   slack at the measured 2-waves/SIMD residency, median/P90 stay flat-to-
//   improved vs the accepted 430.21/430.95 us; a >2% regression (median >
//   438.81 or p90 > 439.57) falsifies the read-prefetch mechanism and the
//   iteration-16 source (digest 45ca0cdc...) is restored before the next
//   experiment.  CONFOUND GUARD: the exact code object of the timed run must
//   show vgpr_count <= 128 with scratch == 0 and vgpr_spill_count == 0, grid
//   768, workgroup 512, lds_bytes 30,720, residency 1 block/CU (any
//   spill/scratch or residency change confounds the probe - restore the
//   iteration-16 source and report regardless of timing).  Rejected
//   alternatives for this round (evidence, not re-litigation): raw asm
//   (policy); every occupancy/co-residency point in the trusted probe set
//   [2,3,4,5] (VGPR-infeasible at 512 threads: 64 accumulator + 16 A + 16 B
//   fragment VGPR alone; 2 blocks/CU would need <= 64 VGPR, and every
//   measured co-residency probe was falsified - iter 10 double-buffer 669.35,
//   iter 11 diet confounded at vgpr 71, iter 14 N=32 diet 632.94, the 128x64
//   16-wave control 446-449); 128x128 at 256 threads / 2 blocks/CU x 4 waves
//   (the domain-splitting direction lost at every measurement - iter 8's
//   4x4-wave control 446.4 vs its 2x8-wave 440.2, then iter 16's 1x8-wave
//   430.2: the trend is consistently FEWER, BIGGER barrier domains, and the
//   grid 1536/240 = 6.4 quantization is identical); one-stage-ahead global
//   prefetch (iter 13 neutral-worse 442.57 - the global round trip is not
//   the binding stall); double-buffer / stage-top publish (iter 10 669.35,
//   TP4 o_proj 891.29); stride/bank-conflict probes (flat-to-worse, conflict
//   floor not on the critical path); stage 128 (saturated axis at 144-B
//   strides); stage 32 (TP4 866.60, and iter 14's stage change was fused
//   with the N=32 diet); kBlockN 192/256 (24/32 accumulators -> VGPR
//   infeasible); M-tile 512 (VGPR-infeasible); split-K (MxN grid dwarfs 120
//   CUs); epilogue restructures (vmem_write 98K is 0.2% of the stream; scale
//   staging lost on down_proj -4.9%); per-wave staging / zero barriers (TP4
//   o_proj 967.17 - duplicated staging traffic swamps barrier savings).
//
// Iteration 21 (final conditional round; control plane phase hip_only,
// plateau=false, raw_inline_asm_allowed=false, skill_allowed=false - no prior
// ISA-guided round recorded a compiler limitation, and recent_valid_
// improvements_percent [+1.29, -7.18, -0.27] does not prove a plateau, so raw
// inline asm stays forbidden and this is ONE HIP-only consolidation change):
//   Measured state: accepted iteration-17 source (digest
//   e1b870f66e8cb61f04aac92f5d669405e5a2fd6bca4a01902592b5bab6acff23) at
//   424.71/425.64 us, 1 block/CU x 8 resident waves, arch_vgpr 112, grid 768
//   (iteration 20's loop-carried one-stage-ahead global prefetch measured
//   neutral 425.86/426.97 and was NOT accepted; its exact code object shows
//   why - the compiler reverted the pipelining: the 'prefetched' loads still
//   issue immediately before the visibility barrier with their own
//   s_waitcnt vmcnt(0), so the stage-top global latency was never actually
//   covered and the +12 loop-carried VGPR bought nothing).  DECISION - remove
//   the provably-dead b_n < kBlockN staging guard (iteration-8 leftover;
//   always true at kBlockN = 128 with 512 threads: b_n = tid >> 2 is in
//   [0, 127]) and load all three staging int4s (A rows 0-127, A rows
//   128-255, B) into locals BEFORE the three ds_write_b128 publishes.  The
//   accepted object's ISA (iteration-18 current-best-isa of digest
//   e1b870f6..., vgpr 112) shows the guarded B load issues only after both A
//   writes: global_load_dwordx4 x2 -> s_waitcnt vmcnt(1) -> ds_write_b128 ->
//   vmcnt(0) -> ds_write_b128 -> s_and_saveexec/cbranch (guard) -> B
//   global_load_dwordx4 -> vmcnt(0) -> ds_write_b128 -> s_or_b64, i.e. a
//   SECOND serialized L2 round trip on the collective stage-top critical
//   path every stage (all 8 waves are barrier-locked between the WAR and the
//   visibility barrier, so the sibling wave on the SIMD cannot hide it).  A
//   stage is ~5,500 cycles at the measured 2-waves/SIMD residency, so one
//   L2 round trip per stage is a first-order term.  With the guard gone the
//   B load issues together with the A loads at the stage top and its latency
//   overlaps A's; the exec-mask dance (s_and_saveexec/s_cbranch_execz/
//   s_or_b64, 3-4 instructions per stage per wave) disappears.  Same global
//   addresses, same LDS addresses, same bytes -> mismatch 0 / max_abs_error
//   0.0 expected; Graph capture + changed-content replay must pass.  LDS
//   30,720 B unchanged, 2 barriers per stage with the iteration-6 final-stage
//   peel (31 per block), 8 ds_read2_b64 + 32 v_mmac per stage per wave
//   unchanged, grid 768, workgroup 512, arch_vgpr ~112 (<= 128 -> residency
//   stays 1 block/CU x 8 waves; the trusted occupancy split candidates
//   [2,3,4,5] stay structurally impossible: 64 accumulator + 32 fragment
//   VGPR alone at 512 threads).  EXPECTED DELTAS to verify the mechanism:
//   ISA steady state - the three global_load_dwordx4 issue back-to-back at
//   the stage top with no branch and no exec-mask, the B ds_write_b128 is
//   unconditional (staging still 3 ds_write_b128 per thread per stage), 31
//   s_barrier / 32 v_mmac / 8 ds_read2_b64 unchanged; PMC - vmem_read ~454K
//   FLAT, lds_instructions ~1.87M FLAT, lds_bank_conflicts ~8.65M FLAT (same
//   LDS layout and reads), lds_wait ~12.32M FLAT-to-lower (no LDS-side
//   change), valu_instructions flat minus the exec-mask ops; launch record -
//   grid 768, workgroup 512, lds_bytes 30,720, arch_vgpr ~112, scratch 0,
//   residency 1 block/CU.  FALSIFIABLE CLAIM: with the second serialized
//   staging L2 round trip removed from the collective stage-top critical
//   path, median/P90 improve or stay flat vs the accepted 424.71/425.64 us;
//   a >2% regression (median > 433.20 or p90 > 434.16) falsifies the
//   staging-load consolidation and the iteration-17 source (digest
//   e1b870f6...) is restored before the next experiment.  CONFOUND GUARD:
//   the exact code object of the timed run must show vgpr_count <= 128 with
//   scratch == 0 and vgpr_spill_count == 0, grid 768, workgroup 512,
//   lds_bytes 30,720, residency 1 block/CU (any spill/scratch or residency
//   change confounds the probe - restore the iteration-17 source and report
//   regardless of timing).  Rejected alternatives for this round (evidence,
//   not re-litigation): raw asm (policy); loop-carried staging prefetch
//   (iter 20 425.86 - neutral, and its code object proves the compiler
//   reverted the pipelining); double-buffer / barrier-halving / stage-top
//   publish into the idle buffer (iter 18 457.56 - falsified at this
//   residency); every occupancy/co-residency point in the trusted probe set
//   [2,3,4,5] (VGPR-infeasible at 512 threads: 64 accumulator + 32 fragment
//   VGPR alone, and every measured co-residency probe was falsified - iter
//   10 double-buffer 669.35, iter 11 diet confounded at vgpr 71, iter 14
//   N=32 diet 632.94, the 128x64 16-wave control 446-449); 128x128 at 256
//   threads / 2 blocks/CU x 4 waves (the domain-splitting direction lost at
//   every measurement); kBlockN 192/256 (24/32 accumulators -> VGPR
//   infeasible); M-tile 512 (VGPR-infeasible); stage 128 (saturated axis at
//   144-B strides); stage 32 (TP4 866.60); stride/bank-conflict probes
//   (flat-to-worse, conflict floor not on the critical path); split-K (MxN
//   grid dwarfs 120 CUs); epilogue restructures (vmem_write 98K is ~0.2% of
//   the instruction stream); per-wave staging / zero barriers (TP4 o_proj
//   967.17 - duplicated staging traffic swamps barrier savings).

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
constexpr int kTargetN = 6144;
constexpr int kTargetK = 1024;
constexpr int kBlockM = 256;  // iteration 8: 512 threads = 8 waves/block
constexpr int kBlockN = 128;  // iteration 16: N-tile 64 -> 128 (16 accums/wave)
constexpr int kStageK = 64;
constexpr int kBPad = 16;  // 64 -> 80-byte LDS row stride (five bank phases)
constexpr int kBStride = kStageK + kBPad;  // 80: B rows hold kStageK=64 k + 16 pad
constexpr int kAStride = kStageK + kBPad;  // 64 -> 80-byte A row stride
constexpr int kBlockThreads = 8 * kWaveSize;  // iteration 8: 512 = 8 waves/block

using bf16_t = hip_bfloat16;

// ---------------------------------------------------------------------------
// Coalesced fragment epilogue: the m16n16k32 accumulator lane mapping
// (row = lane & 15, column group c4 = lane >> 4, frag.x[i] -> column
// c4 + 4*i) gives each lane four elements strided by 4 columns, so the direct
// per-element store is four 2-byte scalar stores per lane at 25% sector
// utilization.  This epilogue transposes the 4-element groups within each
// 4-lane column group (lanes r, r+16, r+32, r+48: a 4x4 transpose via two
// 2x2 steps with __shfl_xor 16 then 32; no staging tile round trip), so lane
// (r, c4) ends up holding the four CONTIGUOUS columns 4*c4 .. 4*c4+3,
// converts them to bf16, packs 4 bf16 (8 B), and issues ONE 8-byte store per
// lane (100% store sector efficiency).  Only the int32 values are re-routed
// between lanes - the per-element scale multiply order
// (float(dot) * x_scale[row] * weight_scale[col]) and the bf16 rounding are
// unchanged, so the stored bits are identical to the direct store.  The
// row>=m guard is wavefront-uniform (all 64 lanes of a wave share the same
// 16-row window), the shuffles never mix active and inactive lanes, and
// base_col is a multiple of 16 with n*2 a multiple of 8, so the float4
// weight_scale load and the 8-byte store are aligned.
// ---------------------------------------------------------------------------
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

// ---------------------------------------------------------------------------
// Iteration 2: direct 8-byte LDS fragment fill (load_fragment8).  The library
// du_load_matrix_sync int8 loaders assign lane (r = lane & 15, c4 = lane >> 4)
// eight consecutive bytes x[0..7] and then make the compiler emit per-byte
// LDS reads plus a mask/OR reassembly VALU chain before the v_mmac operand.
// Writing the same 8 bytes straight into the fragment storage (one 8-byte
// LDS read) keeps the operand bit pattern identical (exact int32
// accumulation unchanged) and removes the dead byte-reassembly VALU and its
// lgkmcnt wait chains.  For the m-major A tile and the n-major B tile both
// operands are laid out so each lane's eight bytes are contiguous at
// (row)*stride + c4*8: A lane (r, c4) owns k = kk + c4*8 .. +7 of
// a_tile[(local_row + r)*80 + kk]; B lane (r, c4) owns k = kk + c4*8 .. +7
// of b_tile[(local_col + r)*80 + kk].  This is the exact pattern validated
// on the worker-29 TP8 o_proj lineage (iterations 10 and 12, 419.05 ->
// 348.27 -> 322.99 us) for the same 128x64x64 geometry.
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

// ---------------------------------------------------------------------------
// Large-M prefill: 256x128 output tile per block, eight wavefronts of 64 lanes
// (iteration 8 resource round: 512 threads = 8 waves/block; iteration 16
// N-tile round: 64 -> 128 columns per block so each wave owns a 64x64
// quadrant of sixteen m16n16k32 int32 accumulators).  Measured residency of
// the accepted 256x64 source: 1 block/CU x 8 resident waves (arch_vgpr 72 >
// 64, so 2 blocks/CU never fit; every co-resident occupancy probe on this
// shape - iteration 10 double-buffer at 2 blocks/CU, iteration 11
// launch_bounds diet at vgpr 71, iteration 14 N=32 diet at 632.94 us - was
// falsified, while the 256x64 8-wave single-domain geometry won at
// 440.21/440.84 us).  Iteration 16 keeps the winning 1 block/CU x 8 waves
// residency (sixteen int32 accumulators alone are 64 VGPR, so 2 blocks/CU is
// structurally impossible) and doubles per-block work: per-output LDS
// fragment reads drop 3 -> 2 B (A read amplification 2 -> 1 at constant B),
// per-output barrier frequency halves (still 2 barriers per stage, 31 per
// block), and per-output global read traffic drops 1.25 -> 0.75 B (A staging
// unchanged at 2 int4/thread, B staging 1 int4/thread - 128 rows x 64 k =
// 512 int4s over exactly 512 threads).  The block cooperatively stages
// A[256,64] from x_q (row-major [M, K], stride k) and B[128,64] from the
// packed n-major [N, K] weight (iteration 2: B row n is contiguous in k, so
// one (n-tile, K-stage) is one contiguous 8 KiB vector stream in both global
// and LDS) into a single-buffered 64-K LDS buffer (A[256,80] + B[128,80] =
// 30,720 B).  Two barriers per stage: one after the cooperative load, one
// before the next stage overwrites LDS (iteration 6: the final stage's
// overwrite-guard barrier is skipped because no successor stage exists - 31
// barriers per block instead of 32).  A and B fragments are filled by
// load_fragment8 (one 8-byte LDS read per fragment; iteration 2, replacing
// the library row-major loaders' per-byte ds_read_u8 + mask/OR reassembly
// VALU on both operands).  Direct fragment epilogue with fused scales + bf16
// conversion.
// ---------------------------------------------------------------------------
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
  const int local_col = wave_col * 64;  // iteration 16: 2 wave columns x 64 = 128

  // Single-buffered 64-K stage: A[256, 80] + B[128, 80] = 30,720 B/block
  // (iteration 16: N-tile 64 -> 128 at the measured 1 block/CU x 8 waves
  // residency - sixteen int32 accumulators alone are 64 VGPR, so 2 blocks/CU
  // would need <= 64 VGPR and is structurally impossible; the padded 80-byte
  // strides are 16-byte-aligned and break the 64-byte LDS bank periodicity).
  // A is m-major [256 rows, 64 k], B is n-major [128 n, 64 k] (iteration 2
  // packed layout; k is the contiguous dimension in the packed [N, K]
  // weight).
  __shared__ __align__(16) int8_t a_tile[kBlockM * kAStride];
  __shared__ __align__(16) int8_t b_tile[kBlockN * kBStride];

  DUFragment<matrix_a, kTileM, kTileN, kTileK, signed char, row_major>
      a_frag0, a_frag1, a_frag2, a_frag3;  // iter 17: 4 live A fragments
                                           // (rows 0-15/16-31/32-47/48-63) so
                                           // all 8 ds_read2_b64 issue in the
                                           // first read group
  DUFragment<matrix_b, kTileM, kTileN, kTileK, signed char, row_major>
      b_frag0, b_frag1, b_frag2, b_frag3;
  DUFragment<accumulator, kTileM, kTileN, kTileK, int>
      acc00, acc01, acc02, acc03, acc10, acc11, acc12, acc13,
      acc20, acc21, acc22, acc23, acc30, acc31, acc32, acc33;
  du_fill_fragment(acc00, 0);
  du_fill_fragment(acc01, 0);
  du_fill_fragment(acc02, 0);
  du_fill_fragment(acc03, 0);
  du_fill_fragment(acc10, 0);
  du_fill_fragment(acc11, 0);
  du_fill_fragment(acc12, 0);
  du_fill_fragment(acc13, 0);
  du_fill_fragment(acc20, 0);
  du_fill_fragment(acc21, 0);
  du_fill_fragment(acc22, 0);
  du_fill_fragment(acc23, 0);
  du_fill_fragment(acc30, 0);
  du_fill_fragment(acc31, 0);
  du_fill_fragment(acc32, 0);
  du_fill_fragment(acc33, 0);

  // Cooperative A staging: each thread owns one int4 in A rows [0,128) and
  // one in A rows [128,256): A[256,64] is 1,024 int4s = 512 threads x 2 int4s.
  const int vector_byte_offset = tid * static_cast<int>(sizeof(int4));
  const int stage_row = vector_byte_offset / kStageK;
  const int stage_col = vector_byte_offset - stage_row * kStageK;
  // Cooperative B staging from the packed n-major [N, K] weight (iteration
  // 2): the B[128 n, 64 k] tile is 512 int4s = exactly one int4 per thread
  // (iteration 16: with kBlockN = 128, b_n = tid >> 2 covers exactly the 128
  // n-rows; the iteration-8 b_n < kBlockN guard became always true then and
  // is REMOVED in iteration 21 so the B global load issues with the A loads
  // at the stage top - the compiler never elided the guard, and the accepted
  // object's ISA shows the guarded B load serialized behind the A writes).
  // The four lanes of one n-row read four consecutive 16-byte chunks of one
  // contiguous packed row (coalesced 64 B per row) and write them to one
  // contiguous 64 B LDS row; the whole (n-tile, K-stage) is one contiguous
  // 8 KiB global stream.
  const int b_n = tid >> 2;         // n row within the tile, 0..127 (512 thr)
  const int b_kv = (tid & 3) * 16;  // stage-local k offset of the B int4

  for (int k0 = 0; k0 < k; k0 += kStageK) {
    // Iteration 21: all three staging int4s are loaded into locals FIRST
    // (A rows 0-127, A rows 128-255, B), then all three are published to LDS.
    // The b_n < kBlockN guard (iteration-8 leftover, provably always true at
    // kBlockN = 128 with 512 threads: b_n = tid >> 2 is in [0, 127]) is
    // removed so the B global load issues together with the two A loads at
    // the stage top instead of behind the A ds_write_b128s (the accepted
    // object's ISA shows the guarded B load starts only after both A writes
    // issued, adding a second serialized L2 round trip to the collective
    // stage-top critical path).  Same global addresses, same LDS addresses,
    // same bytes -> bit-identical int32.
    const int4 stage_a0 = *reinterpret_cast<const int4*>(
        x_q + static_cast<int64_t>(m0 + stage_row) * k + k0 + stage_col);
    const int4 stage_a1 = *reinterpret_cast<const int4*>(
        x_q + static_cast<int64_t>(m0 + stage_row + kBlockM / 2) * k + k0 +
        stage_col);
    const int4 stage_b = *reinterpret_cast<const int4*>(
        weight + (static_cast<int64_t>(n0 + b_n) * k + k0 + b_kv));
    *reinterpret_cast<int4*>(a_tile + stage_row * kAStride + stage_col) =
        stage_a0;
    *reinterpret_cast<int4*>(a_tile +
                             (stage_row + kBlockM / 2) * kAStride +
                             stage_col) = stage_a1;
    *reinterpret_cast<int4*>(b_tile + b_n * kBStride + b_kv) = stage_b;
    __syncthreads();

    // Each wave consumes its 64x64 quadrant: sixteen m16n16k32 MMACs per kk
    // (iteration 16: N-tile 64 -> 128; the four B fragments of the wave's 64
    // columns are each reused in registers across all four M-fragments, so
    // per-output LDS fragment reads drop 3 -> 2 B).
#pragma unroll
    for (int kk = 0; kk < kStageK; kk += kTileK) {
      // Iteration 2: B is n-major in LDS, so each lane's eight fragment
      // k-values are contiguous; one 8-byte load fills the fragment (the
      // same 8 bytes the old row_major loader gathered with eight ds_read_u8
      // plus mask/OR pack VALU), keeping the v_mmac operand bit pattern and
      // the int32 accumulation identical.
      load_fragment8(b_frag0, b_tile + local_col * kBStride + kk, kBStride,
                     lane);
      load_fragment8(b_frag1, b_tile + (local_col + kTileN) * kBStride + kk,
                     kBStride, lane);
      load_fragment8(b_frag2, b_tile + (local_col + 2 * kTileN) * kBStride +
                         kk,
                     kBStride, lane);
      load_fragment8(b_frag3, b_tile + (local_col + 3 * kTileN) * kBStride +
                         kk,
                     kBStride, lane);
      // A gets the same direct 8-byte fill: A is m-major in LDS, so lane
      // (r, c4) owns the eight CONSECUTIVE k-values
      // a_tile[(local_row + r)*80 + kk + c4*8 .. +7]; the code object shows
      // du_load_matrix_sync already vectorized these into ds_read2_b32 but
      // still executed the per-dword mask/OR reconstruction before the
      // v_mmac A operands.  Writing the same 8 bytes straight into the
      // fragment storage removes that dead chain; the operand bit pattern
      // and the int32 accumulation are unchanged.
      load_fragment8(a_frag0, a_tile + local_row * kAStride + kk, kAStride,
                     lane);
      load_fragment8(a_frag1, a_tile + (local_row + kTileM) * kAStride + kk,
                     kAStride, lane);
      // Iteration 17: the rows-32-47 and rows-48-63 A fragments are loaded
      // here in the FIRST read group (dedicated a_frag2/a_frag3) instead of
      // just-in-time before the acc20/acc30 groups, so their LDS latency is
      // covered by the first 16 v_mmac of the kk (the accepted 8-wave ISA
      // exposed a standalone lgkmcnt wait immediately before acc20; the
      // flat iter-7 probe at the 16-wave geometry does not transfer to 2
      // waves/SIMD where lds_wait is 4.75x lds_instructions).  Same 8 LDS
      // bytes per fragment at the same addresses -> bit-identical int32.
      load_fragment8(
          a_frag2, a_tile + (local_row + 2 * kTileM) * kAStride + kk,
          kAStride, lane);
      load_fragment8(
          a_frag3, a_tile + (local_row + 3 * kTileM) * kAStride + kk,
          kAStride, lane);
      du_mma_sync(acc00, a_frag0, b_frag0, acc00);
      du_mma_sync(acc01, a_frag0, b_frag1, acc01);
      du_mma_sync(acc02, a_frag0, b_frag2, acc02);
      du_mma_sync(acc03, a_frag0, b_frag3, acc03);
      du_mma_sync(acc10, a_frag1, b_frag0, acc10);
      du_mma_sync(acc11, a_frag1, b_frag1, acc11);
      du_mma_sync(acc12, a_frag1, b_frag2, acc12);
      du_mma_sync(acc13, a_frag1, b_frag3, acc13);
      du_mma_sync(acc20, a_frag2, b_frag0, acc20);
      du_mma_sync(acc21, a_frag2, b_frag1, acc21);
      du_mma_sync(acc22, a_frag2, b_frag2, acc22);
      du_mma_sync(acc23, a_frag2, b_frag3, acc23);
      du_mma_sync(acc30, a_frag3, b_frag0, acc30);
      du_mma_sync(acc31, a_frag3, b_frag1, acc31);
      du_mma_sync(acc32, a_frag3, b_frag2, acc32);
      du_mma_sync(acc33, a_frag3, b_frag3, acc33);
    }

    // Protect the LDS buffer from the next stage's cooperative overwrite.
    // The FINAL stage has no successor stage and the epilogue never touches
    // LDS (it reads only accumulator registers and global scales), so this
    // guard is dead work on the last iteration: skip it there (iteration 6;
    // 32 -> 31 barriers per block).  The guard is wavefront-uniform - k0 and
    // k are scalars - so all 256 threads take the same path.
    if (k0 + kStageK < k) {
      __syncthreads();
    }
  }

  const int base_row = m0 + local_row;
  const int base_col = n0 + local_col;
  // Iteration 16: sixteen fragments per wave (4 M x 4 N of the 64x64
  // quadrant); same coalesced 8-byte store per lane per fragment.
  store_prefill_fragment_coalesced(acc00, x_scale, weight_scale, out,
                                   base_row, base_col, m, n, lane);
  store_prefill_fragment_coalesced(acc01, x_scale, weight_scale, out,
                                   base_row, base_col + kTileN, m, n, lane);
  store_prefill_fragment_coalesced(acc02, x_scale, weight_scale, out,
                                   base_row, base_col + 2 * kTileN, m, n,
                                   lane);
  store_prefill_fragment_coalesced(acc03, x_scale, weight_scale, out,
                                   base_row, base_col + 3 * kTileN, m, n,
                                   lane);
  store_prefill_fragment_coalesced(acc10, x_scale, weight_scale, out,
                                   base_row + kTileM, base_col, m, n, lane);
  store_prefill_fragment_coalesced(acc11, x_scale, weight_scale, out,
                                   base_row + kTileM, base_col + kTileN, m, n,
                                   lane);
  store_prefill_fragment_coalesced(acc12, x_scale, weight_scale, out,
                                   base_row + kTileM, base_col + 2 * kTileN,
                                   m, n, lane);
  store_prefill_fragment_coalesced(acc13, x_scale, weight_scale, out,
                                   base_row + kTileM, base_col + 3 * kTileN,
                                   m, n, lane);
  store_prefill_fragment_coalesced(acc20, x_scale, weight_scale, out,
                                   base_row + 2 * kTileM, base_col, m, n,
                                   lane);
  store_prefill_fragment_coalesced(acc21, x_scale, weight_scale, out,
                                   base_row + 2 * kTileM, base_col + kTileN,
                                   m, n, lane);
  store_prefill_fragment_coalesced(acc22, x_scale, weight_scale, out,
                                   base_row + 2 * kTileM, base_col + 2 * kTileN,
                                   m, n, lane);
  store_prefill_fragment_coalesced(acc23, x_scale, weight_scale, out,
                                   base_row + 2 * kTileM, base_col + 3 * kTileN,
                                   m, n, lane);
  store_prefill_fragment_coalesced(acc30, x_scale, weight_scale, out,
                                   base_row + 3 * kTileM, base_col, m, n,
                                   lane);
  store_prefill_fragment_coalesced(acc31, x_scale, weight_scale, out,
                                   base_row + 3 * kTileM, base_col + kTileN,
                                   m, n, lane);
  store_prefill_fragment_coalesced(acc32, x_scale, weight_scale, out,
                                   base_row + 3 * kTileM, base_col + 2 * kTileN,
                                   m, n, lane);
  store_prefill_fragment_coalesced(acc33, x_scale, weight_scale, out,
                                   base_row + 3 * kTileM, base_col + 3 * kTileN,
                                   m, n, lane);
}

// ---------------------------------------------------------------------------
// Generic scalar fallback: one thread per output element.  Exact int32 dot
// over K, then fused x_scale * weight_scale, then bf16 store.  Handles every
// unmatched (m, n, k), including all small-M API cases (M=2, M=16) and any
// M in (0, 4096] with the same (K, N).  For (k, n) == (1024, 6144) it decodes
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
    // Iteration 2: the packed layout for (k, n) == (1024, 6144) is n-major
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
// (1024, 6144) the raw [K, N] row-major weight is transposed into the
// n-major [N, K] layout:
//     packed[n*K + kk] == raw[kk*N + n]
// so every DUMMA B tile (one 64-column n-tile x one 64-K stage) is a single
// contiguous 4 KiB stream of 16-byte vector loads and each lane's eight
// B-fragment k-values are contiguous in LDS (one 8-byte fragment read).
// The transpose's 16-byte destination vectors run along K, so each thread
// GATHERS its 16 bytes from 16 raw rows (aligned byte loads strided by N)
// instead of copying a contiguous raw vector (a memcpy-style vectorized
// pack would write the wrong axis and leave most of the buffer unwritten -
// the measured correctness failure on the worker-29 TP8 o_proj lineage,
// iteration 10 -> 11 repair).  The pack is one-time and out-of-timed-region,
// so the gather cost is free.  For every other (k, n) the pack is an
// identity device-to-device copy.  Scales are copied identity in both cases.
// ---------------------------------------------------------------------------
__global__ __launch_bounds__(256) void w8a8_pack_o_proj_panels_kernel(
    const int8_t* __restrict__ raw,
    int8_t* __restrict__ packed,
    int k,
    int n) {
  // One thread per 16-byte vector of the packed [N, K] layout; the vector is
  // a GATHER, not a memcpy.  The destination vector
  // packed[n*K + kk0 .. kk0+15] holds 16 CONSECUTIVE k-values of raw column n
  // (packed[n*K + kk0 + j] == raw[(kk0 + j)*N + n], j = 0..15), so its 16
  // source bytes sit in 16 DIFFERENT raw rows (strided by N).
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

  // Explicit dispatch. The assigned shape (M=4096, N=6144, K=1024) takes the
  // DUMMA 256x128 single-buffered path (iteration 8 geometry 256x64/512
  // threads/8 waves extended by iteration 16 to 256x128 with sixteen
  // accumulators per wave; measured residency of the accepted 256x64 source
  // is 1 block/CU x 8 waves - arch_vgpr 72 - and iteration 16 keeps 1
  // block/CU, pinned structurally by 16 int32 accumulators = 64 VGPR; grid
  // (6144/128, 4096/256) = 768 blocks; the kernel symbol name is
  // intentionally unchanged - an internal anonymous-namespace symbol matched
  // by the harness profiler); every other (m, n, k) - including
  // small-M API cases (M=2, M=16) - takes the scalar fallback, which decodes
  // the packed n-major [N, K] layout for (k, n) == (1024, 6144) and the
  // identity [K, N] layout otherwise.
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
    // Iteration 2: n-major [N, K] transpose for the target (k, n) ==
    // (1024, 6144); one thread per 16-byte destination vector (gather).
    const int64_t total4 = static_cast<int64_t>(k) * n / 16;
    const dim3 pack_grid(
        static_cast<unsigned>((total4 + kBlock - 1) / kBlock));
    hipLaunchKernelGGL(w8a8_pack_o_proj_panels_kernel,
                       pack_grid, block, 0, stream,
                       raw_weight, packed_weight, k, n);
  } else {
    // Identity device-to-device copy for every other (k, n).  The packed
    // buffer keeps the same byte count K*N and the same allocated address,
    // so the layout is graph-stable.
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
