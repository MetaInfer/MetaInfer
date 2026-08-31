// @@variant shape=hy3_tp8_shared_gate_up_proj_m16 commit=2321c42b8618aae23700bc2cf3532b59c8f6c8d8 added=2026-08-27
//   median_us=10.38 p90_us=10.42
//   source=hy3-dsh-tp8-m16-2-dc53295a
// MetaInfer W8A8 INT8 GEMM - HIP implementation (gfx928 / K500SM_AI).
//
// worker_2 iteration 8: split-K=15 one-wave LDS-staged partial GEMM +
// split-15 combine for the assigned shape hy3_tp8_shared_gate_up_proj_m16:
//     M = 16, N = 384, K = 4096   (M*N = 6144 output elements)
//
// Operator contract:
//     out[m, n] = bf16( int32_dot(x_q[m, :], weight[:, n])
//                       * x_scale[m, 0] * weight_scale[n, 0] )
//
// Iteration 1 = minimal gfx928 DUMMA INT8 bootstrap (exact mandatory tile):
//   * one 64-lane wavefront per block, one 16x16 output tile per wave,
//     explicit int32 accumulation, grid = N/16 = 24 blocks, direct global
//     fragment loads, LDS-drain epilogue; measured 117.91 us median with
//     only 24 of 120 CUs active (24 waves, 1 per CU, no co-residency).
//
// Iteration 4 = accepted split-K=10 grid-parallelism probe (19.84 us):
//   * K=4096 split into 10 non-uniform 32-aligned slices, grid = 24 N-tiles
//     x 10 splits = 240 one-wave blocks (2 per CU); direct global fragment
//     loads in the K loop; separate 24-block combine kernel sums the 10
//     int32 planes in ascending split order (bit-identical int32 order);
//     both launches inside the timed Graph region.
//   * PMC: 49,152 vmem_read instructions for the partial kernel (~205 per
//     block = byte-level fragment loads + reassembly, one vmcnt wait per
//     load before each mmac), lds_instructions = 0 -> the K loop is
//     serialized on global load latency (issue/latency-bound).
//
// Iteration 6 = accepted split-K=16 LDS-staged kernel (18.73 us median /
// 19.04 us p90): block tile 16x32 (2 waves x 1 N-tile of 16 columns), 128
// thr, grid = 12 N-groups x split-K=16 = 192 blocks = 1.6 blocks/CU = 3.2
// waves/CU; whole uniform 256-row slice staged once (A 16x256 stride 272,
// B 32x256 stride 48), one __syncthreads, zero-barrier LDS-only K loop
// (8 steps x 1 v_mmac per wave); workspace = 16 int32 planes = 393,216 B =
// the caller's workspace_split_k_capacity(16,384,4096) budget; combine =
// 24 blocks x 128 thr, ascending split sum; PMC: 192 blocks < 2/CU target.
//
// Iteration 7 = measured split-K=10 one-wave grid (this file's round-7
// candidate, REJECTED by the p90 guard): block tile 16x16 (1 wavefront, 64
// thr), grid = 24 N-groups x split-K=10 = 240 one-wave blocks = exactly
// 2 blocks/CU; K=4096 -> 10 non-uniform 32-aligned slices (8 x 416 + 2 x
// 384); whole slice staged once (A stride 432, B stride 48; 26,880 B/block
// = 2 x 26,880 = 53,760 B/CU at 2 blocks, fits 64 KiB), 0 s_barrier (the
// single-wave __syncthreads is eliminated; lgkmcnt waits cover the ds
// hazard, verified in the exact gfx928 code object); 10 planes = 245,760 B;
// combine = 24 blocks x 128 thr. Measured median 15.432 us (-21% vs
// iteration 6) but p90 19.434 > best 19.040 -> p90_guard_passed = false;
// the 30 samples were bimodal (4 slow windows 19.4-23.2 us then 26 fast
// windows 15.4-15.9 us): a machine-state artifact, not kernel behavior.
//
// Iteration 8 (this file's active M=16 path) = mandatory HIP-only
// occupancy round: tune one occupancy limiter (LDS footprint -> blocks/CU)
// with PMC evidence. The split-K=10 kernel's 26,880 B LDS/block caps it at
// exactly 2 blocks/CU (3 x 26,880 = 80,640 > 64 KiB). split-K=15 (trusted
// probe set [2,5,10,15]) shortens every slice (288/256 vs 416/384 rows),
// shrinking the per-block LDS footprint to 18,688 B -> 3 x 18,688 = 56,064
// B/CU <= 64 KiB, so 3 one-wave blocks/CU become legal, and the grid =
// 24 N-groups x 15 splits = 360 one-wave blocks = EXACTLY 3 blocks/CU on
// 120 CUs (above the 2-blocks/CU latency-hiding target, integer-balanced,
// no tail wave). No repeated HBM reads: B stays once-read (1,572,864 B),
// A stays L2-hot, every block still stages its own slice exactly once.
//   * block tile 16x16 (1 wavefront x 1 N-tile), 64 thr; grid = 24 x 15 =
//     360 one-wave blocks = exactly 3 blocks/CU;
//   * K=4096 -> 15 non-uniform 32-aligned slices: 8 x 288 (9 DUMMA steps)
//     + 7 x 256 (8 steps); k0-ascending within each slice and ascending
//     split sum in the combine preserve the bit-identical int32 order;
//   * whole slice staged once into bank-skewed LDS (A 16x288 stride 304,
//     B 288x16 stride 48; 18,688 B/block), batched 16-B vector loads, one
//     __syncthreads (0 s_barrier for the 1-wave block), then a zero-barrier
//     LDS-only K loop (9/8 steps x 1 A-frag + 1 B-frag ds_read + 1 v_mmac);
//   * workspace = 15 int32 planes [15][16][384] = 368,640 B (<= the
//     caller's 393,216 B capacity; guard updated, unsplit DUMMA fallback
//     when workspace < 15 planes); combine = 24 blocks x 128 thr, all 15
//     plane loads issued into a register array before the ascending sum;
//   * total replay traffic ~2.32 MB (B 1.57 MB + planes RW 737,280 B +
//     out 12,288 B) vs 2.47 MB for the accepted S=16 kernel (-6%); vs S=10
//     the plane RW grows +245,760 B (+50%) - the combine cost of more
//     splits being benchmarked against the occupancy gain.
//
// Iteration 9 (this file's active M=16 path) = HIP-only two-launch attack.
// Round 8 measured 15.529 us median / 16.024 us p90 and its falsifiable
// branch ("if median stays >= 15.4, the next round attacks the two-launch
// structure - fused/single-kernel combine - instead of geometry") fired.
// Change: fuse the 24-block combine INTO the partial kernel as a
// last-arrival tail using the validated TP4 qkv lineage protocol - every
// block writes its 16x16 int32 plane tile, __threadfence() (release), then
// atomicAdd(&counters[tile], 1); the block whose atomicAdd returns
// kSplitK-1 (the 15th arriver of its N-group) __threadfence()s (acquire),
// sums the 15 planes in ascending split order (exact int32 order preserved
// -> outputs bit-identical), applies both scales, stores bf16, and resets
// its counter to 0 (atomicExch) so the next replay restarts from 0. The 24
// counters live in the workspace tail (96 B right after the 15 planes;
// guard updated to 368,736 B) and are zeroed once by the host before the
// first launch (a static-flag hipMemsetAsync on the caller stream - issued
// in the eager precheck, outside the timed Graph in the normal flow).
// Result: ONE kernel launch per replay (grid 360 one-wave blocks, 3/CU,
// LDS unchanged 18,688 B); the combine no longer pays a second dispatch
// and each N-group's combine tail overlaps the still-running partials of
// other groups instead of waiting for full grid completion + relaunch.
//
// Repair 1/4 (correctness): the round-9 draft let ALL 64 lanes of every
// block execute atomicAdd(&counters[tile], 1) (64 increments per block =
// 960 per tile per replay), so the kSplitK-1 condition fired on the 15th
// increment inside the FIRST-arriving block - before its 14 siblings had
// stored their planes - and the unguarded atomicExch reset made several
// blocks combine per replay. Exact check failed: 6048/6144 mismatched,
// first_mismatch (0,0) actual=0.0 (premature combiner summed unwritten,
// zero planes). Fix: exactly one arrival per block (lane 0 fence +
// atomicAdd, LDS s_is_last flag, barriers) + lane-0-only atomicExch reset,
// mirroring the validated TP4 qkv fused tail. GEMM/plane-store body and the
// ascending 15-plane sum are unchanged (bit-identical int32 order).
//
// Iteration 10 (this file's active M=16 path) = final HIP-only
// consolidation round (raw inline asm not allowed by the ISA policy:
// plateau=false, valid HIP rounds < 8). Round 9 measured 14.516 us median /
// 14.544 us p90 and its falsifiable branch ("if median stays >= 14.5 ... the
// single-wave combine tail (60 loads/lane) is the new critical path - the
// next round vectorizes the combiner's plane loads (int4) or shrinks the
// plane count instead") fired (14.516 >= 14.5). Change: vectorize ONLY the
// fused combine tail's plane loads - each lane now owns 4 CONSECUTIVE
// columns (row = lane>>2, col0 = (lane&3)*4) and reads each of the 15
// planes with one aligned int4 (16-B) load instead of 4 scalar 4-B loads
// (60 -> 15 vmem loads per lane, 4x fewer tail load instructions, same
// bytes, same L2-request footprint); weight_scale is read once as float4
// and the 4 adjacent bf16 stores merge into one 8-B store. Per-element
// int32 accumulation order is unchanged (ascending split sum per output
// element) -> outputs remain bit-identical (0 mismatches). Everything else
// is untouched: grid = 24 N-groups x 15 splits = 360 one-wave blocks = 3
// blocks/CU, per-block LDS 18,688 B, zero-barrier LDS-only K loop,
// workspace = 15 planes + 24 counters = 368,736 B, one launch per replay,
// exact (m,n,k) guard, unsplit DUMMA fallback, generic scalar fallback.
// Falsifiable: (1) median_us < 14.516 (target ~13.6-14.2) with p90 <= 14.544
// (guard vs the accepted best); (2) 0 mismatches + graph_capture_passed;
// (3) PMC: grid 360 / workgroup 64 / LDS 18,688 / s_barrier 0 unchanged,
// arch_vgpr stays ~40-60 with 0 spills, the tail portion of vmem_read drops
// ~4x (staging ~3,264 unchanged; tail 60 -> 15 loads/lane); (4) replay
// traffic unchanged ~2.07 MB. If median stays >= 14.516 with p90 <= 14.544,
// the tail was not the critical path and the kernel sits at its
// streaming/launch floor -> the round documents the plateau evidence.
//
// Iteration 11 (this file's active M=16 path) = final conditional inline-asm
// round -> mandatory HIP-only consolidation (raw asm is NOT allowed: isa_policy
// says phase=hip_only, plateau=false, raw_inline_asm_allowed=false,
// valid_hip_rounds=7 < 8 - the control plane has not confirmed a HIP plateau).
// Round 10 measured 13.925 us median / 13.972 us p90 (< 14.516 target), so the
// int4-vectorized tail branch did NOT fire and the tail is not the critical
// path anymore. The accepted kernel's fresh exact gfx928 code object exposes
// the next concrete compiler limitation: the slice staging is SERIALIZED -
// every global_load_dwordx4 is followed by s_waitcnt vmcnt(0) before its
// ds_write_b128 (load_wait0_ds_write_windows = 18 across the two slice
// instantiations, 10 long + 8 short), so each block pays ~10/8 full
// global-memory latencies one after another before the K loop's first
// ds_read (the last write is followed by lgkmcnt(0) waits). Change: in
// split15_slice_body, issue ALL staging global loads into one register array
// (int4 s_buf[10/8]) BEFORE any ds_write - the compiler is then forced to put
// the loads in flight back-to-back and move the vmcnt waits to the write
// side, collapsing the staging critical path to one memory latency. Same
// addresses, same per-lane order, same LDS layout, same __syncthreads, same
// zero-barrier LDS-only K loop, same fused last-arrival tail: the int32
// accumulation order is untouched (outputs bit-identical, 0 mismatches
// expected). Everything else is untouched: grid = 24 N-groups x 15 splits =
// 360 one-wave blocks = exactly 3 blocks/CU (trusted probe set member 15),
// per-block LDS 18,688 B (still the occupancy binder: 3 x 18,688 = 56,064 B
// <= 64 KiB), workspace = 15 int32 planes + 24 arrival counters = 368,736 B
// (<= 393,216 B budget; unsplit DUMMA fallback when smaller), one launch per
// replay, fence/atomic/barrier protocol unchanged, exact (m,n,k) guard,
// generic scalar fallback for every unmatched shape including the paired M=2.
// Falsifiable predictions: (1) median_us < 13.925 (target ~11.5-12.8) with
// p90_us <= 13.972 (guard vs the accepted best); (2) correctness 0 mismatches
// + graph_capture_passed (single-launch structure, write-in-place planes,
// self-resetting counters, no clear); (3) ISA: the first staging vmcnt(0)
// now covers ~7 outstanding global_load_dwordx4 (old kernel: exactly 1) -
// memory-level parallelism per block rises 1 -> ~7-10 and the staging
// critical path collapses from ~10 serialized latencies to ~1-2; vmcnt(0)
// 20 -> ~19, arch_vgpr ~69-75 (locally verified 69 VGPR / 40 SGPR) with 0
// spills, grid 360 / LDS 18,692 / s_barrier 0 / v_mmac 17 / ds_write_b128 18
// / ds_read2_b32 17 / ds_read_u8 136 unchanged; (4) PMC vmem_read ~4,392 and
// replay traffic ~2.06 MB unchanged (same loads, same bytes). If
// median stays >= 13.925 with p90 <= 13.972, the 3-blocks/CU overlap already
// hid the staging latency and the kernel sits at its streaming/launch floor -
// the round then documents the plateau evidence for this shape.
//
// Iteration 15 (this file's active M=16 path) = final conditional inline-asm
// round -> mandatory HIP-only consolidation (raw asm is NOT allowed: isa_policy
// says phase=hip_only, plateau=false, raw_inline_asm_allowed=false - the
// control plane has NOT confirmed a HIP plateau: the three recent valid
// improvements were -28.62% / -0.033% / +3.86% (iterations 12/13/14), the
// -28.62% being the rejected round-12 regression and +3.86% the round-14
// packed-B median gain, and no prior ISA-guided round recorded a compiler
// limitation plus target instructions). Round 14 (n-major-packed B staging
// loads only) measured median 12.512 us / p90 13.064 us - best median ever,
// rejected only by the p90 guard (13.064 > 13.052) - and the source was
// restored to the accepted iteration-11 digest. The accepted kernel's exact
// gfx928 code object exposes the remaining half of the B path: the K loop
// still reads each lane's 8 B-fragment elements as 8 ds_read_u8 from the
// k-major stride-48 LDS (4-way bank conflicted; the persistent PMC signature
// since iteration 9), while the A fragment already reads its 8 consecutive
// bytes as ONE ds_read2_b32. The TP4 gate_up/down_proj lineage validated the
// fix for exactly this ("one ds_read_b64 per step, was 8 ds_read_u8"; rule 3:
// 'Pack the cold once-read weight (B) ... into the fragment's layout').
// Change (one consolidation): complete the round-14 B-pack into a per-tile
// n-major transpose packed[tile][n][kk] = raw[kk*384 + tile*16 + n]
// ((k,n)==(4096,384) ONLY; every other (k,n) keeps the identity pack), stage
// B into an n-major LDS layout b_lds[n][k] (element (k,n) at n*304 + k;
// column stride 304 = 288 + 16 bank skew, 16-B aligned so the staging writes
// stay ds_write_b128), and load the B fragment with the library col_major
// loader: a.x[i] = p[row*304 + col + i] reads the lane's 8 elements as 8
// CONSECUTIVE bytes -> one ds_read2_b32 per step (was 8 ds_read_u8 at stride
// 48). Same elements, same x[i] register assignment (x[i] = element
// (k0+col+i, n_base+row) in both layouts), same v_mmac, so the int32
// accumulation order is untouched (k-ascending within each slice, ascending
// split sum - outputs bit-identical, 0 mismatches expected). The B staging
// loads also get the round-14 locality win: each lane holds one 16-B
// row-chunk of 16 consecutive k values of one n column (chunk c -> n = c&15,
// k-chunk = c>>4) - 64 lanes cover 16 x 64-B contiguous sectors per wave-load
// (was 64 scattered 16-B requests), sequential HBM pages, no straddles. LDS
// per block drops 18,688 -> 9,728 B (A 16x304 = 4,864 + B 16x304 = 4,864),
// so 3 x 9,728 = 29,184 B/CU << 64 KiB (LDS is no longer the occupancy
// binder, but the grid stays 360 = exactly 3 blocks/CU, so residency is
// unchanged); per-block LDS instructions drop ~4x (136 ds_read_u8 + 17
// ds_read2_b32 -> 34 ds_read2_b32), attacking the per-CU LDS-pipe pressure
// that round 13's evidence identified as the shared per-CU resource.
// Everything else is untouched: grid = 24 N-groups x 15 splits = 360
// one-wave blocks = exactly 3 blocks/CU (trusted probe-set member 15),
// workspace = 15 int32 planes + 24 arrival counters = 368,736 B (<= the
// caller's 393,216 B budget; unsplit DUMMA fallback when smaller), one launch
// per replay, fence/atomic/barrier protocol unchanged (lane-0 atomicAdd, LDS
// s_is_last flag, lane-0 atomicExch reset), exact (m,n,k) guard, generic
// scalar fallback for every unmatched shape including the paired M=2 (both
// fallbacks decode the packed layout for (k,n)==(4096,384): the scalar path
// reads element (kk, col) at (col>>4)*65536 + (col&15)*4096 + kk; the unsplit
// DUMMA path loads B with the row_major loader at ldm = K from
// packed + tile*65536 + k0, giving p[(col+i)*4096 + row] = element
// (k0+col+i, n_base+row) - identical values to the raw layout).
// Falsifiable predictions: (1) median_us < 12.995 (target ~12.0-12.7; the
// round-14 pack alone measured 12.512 and this also cuts the per-block LDS
// read instructions 8x and the per-CU LDS-pipe work ~4x) with p90_us <=
// 13.052 (guard vs the accepted best p90); (2) correctness 0 mismatches
// (bit-identical int32 order) and graph_capture_passed (single-launch
// structure unchanged, write-in-place planes, self-resetting counters, pack
// out-of-timed/out-of-Graph, no clear); (3) PMC/ISA: grid_blocks 360 /
// workgroup_size 64 / s_barrier 0 / v_mmac 17 / ds_write_b128 18 unchanged,
// ds_read_u8 136 -> 0, ds_read2_b32 17 -> 34 (A + B), group_segment_fixed_size
// 18,692 -> ~9,732, arch_vgpr ~69 +/- a few with 0 spills, occupancy 3
// blocks/CU (now grid-bound), lds_instructions ~31,632 -> ~20k,
// lds_bank_conflicts ~63,744 -> ~15-25k (the B u8 4-way reads disappear),
// vmem_read_instructions ~4,392 unchanged (same loads, same bytes);
// (4) memory_traffic.total_bytes_per_operator_replay unchanged ~2.06 MB (no
// HBM delta). If median stays >= 12.995 with p90 <= 13.052, the LDS pipe was
// not on the critical path either (the kernel is at its launch/memory floor)
// and this final round documents the plateau evidence for this shape instead
// of claiming a gain.
//
// Iteration 17 (this file's active M=16 path) = HIP-only B-pack ORDER
// consolidation (isa_policy: phase=hip_only, plateau=false,
// raw_inline_asm_allowed=false, skill_allowed=false - the three recent valid
// HIP rounds are NOT all within [-2%, +2%): -0.033% / +3.86% / +8.35%, so the
// plateau is not proven and one more HIP-only change is mandated; the
// round-16 attempt died in the agent infrastructure before a valid proposal,
// so this round returns to the accepted iteration-15 source (digest
// b39a985c...) and starts ONE new bounded experiment). Round 15 (n-major B
// pack + n-major LDS) measured median 11.994 us / p90 12.293 us - accepted -
// and left the remaining half of the B memory path on the table: the n-major
// pack places chunk (n, k-chunk) at n*4096 + k-chunk*16, so each staged B
// wave-load touches 16 x 64-B sectors at 4,096-B stride (16 scattered L2
// requests / up to 16 DRAM pages) - the round-14 contiguous-load locality
// was traded away to buy the one-ds_read2_b32 B fragment read. Change (one
// consolidation, pack ORDER only): the pack kernel now emits the SAME 16-B
// chunks (16 consecutive k of one n column) in ascending (slice, n, k-chunk)
// order - chunk index = slice_chunk_base(s) + n*cps(s) + k-chunk, byte
// address = tile*65536 + chunk*16 - so each staged wave-load (64 consecutive
// chunks) covers 1,024 CONTIGUOUS bytes (sequential L2 sectors and
// sequential DRAM pages; was 16 scattered 64-B sectors at 4,096-B stride).
// Chunk content, the n-major LDS destination b_lds[n][k] (n*304 + k), the
// one-ds_read2_b32 B fragment read, the v_mmac count, the zero-barrier K
// loop and the fused last-arrival tail are all UNTOUCHED: every staged byte
// and every fragment register is identical to iteration 15, so the int32
// accumulation order is bit-identical (0 mismatches expected). The
// long-slice staging write index changes from (c&15, c>>4) to (c/18, c%18)
// (the short 256-row instantiation keeps (c>>4, c&15); divisions resolved
// by the compiler, staging path only); the generic scalar fallback and the
// unsplit DUMMA fallback decode the new pack order through a shared
// packed384_byte_offset helper (per-element gather in the unsplit path,
// fallback-only: x[i] = element (k0 + (lane>>4)*8 + i, n_base + (lane&15)),
// the same fragment registers as the library row_major loader).
// Everything else is untouched: grid = 24 N-groups x 15 splits = 360
// one-wave blocks = exactly 3 blocks/CU (trusted probe-set member 15),
// per-block LDS 9,732 B, workspace = 15 int32 planes + 24 arrival counters
// = 368,736 B (<= 393,216 B budget; unsplit DUMMA fallback when smaller),
// one launch per replay, fence/atomic/barrier protocol unchanged, exact
// (m,n,k) guard, generic scalar fallback for every unmatched shape
// including the paired M=2.
// Falsifiable predictions: (1) median_us < 11.994 (target ~11.2-11.8; the B
// staging wave-loads become fully sequential, removing the 16-scattered-
// sector DRAM page pattern) with p90_us <= 12.293 (guard vs the accepted
// best p90); (2) correctness 0 mismatches (bit-identical int32 order) and
// graph_capture_passed (single-launch structure unchanged, write-in-place
// planes, self-resetting counters, pack out-of-timed/out-of-Graph, no
// clear); (3) PMC/ISA: grid_blocks 360 / workgroup_size 64 / lds_bytes
// 9,732 / s_barrier 0 / v_mmac 17 / ds_read2_b32 34 / ds_write_b128 18
// unchanged, arch_vgpr ~69-72 with 0 spills, vmem_read ~4,392 unchanged
// (same loads, same bytes), the B staging address stream becomes sequential
// ((k_base + c)*16 per chunk) so l2_misses ~32,321 may drop (fewer sector
// straddles / page effects) while lds_bank_conflicts may shift (the
// long-slice staging write bank pattern changes; instruction count and
// bytes identical); (4) memory_traffic.total_bytes_per_operator_replay
// unchanged ~2.06 MB (same bytes, better DRAM locality). If median stays
// >= 11.994 with p90 <= 12.293, the B staging DRAM pattern was not on the
// critical path either (the kernel is at its launch/latency floor) and this
// round documents the next plateau datum for this shape.
//
// Iteration 21 (this file's active M=16 path) = HIP-only, ONE consolidation:
// A register-only transport (the validated TP4 o_proj lineage recipe
// "register-only K-loop transport with depth-1 prefetch"; raw asm is NOT
// allowed: isa_policy says phase=hip_only, plateau=false,
// raw_inline_asm_allowed=false, skill_allowed=false - the three recent valid
// HIP improvements +3.86% / +8.35% / +5.35% are not within [-2%, +2%), so the
// HIP plateau is not proven and one more HIP-only change is mandated; rounds
// 18/19/20 died in the agent infrastructure before valid proposals, so this
// round returns to the accepted iteration-17 source (digest
// 88ea95a832fbc79e4d625b2aa22df3a0d533d6eb2c6aff08529c7e60a92942c4, verified
// sha256-identical before editing) and starts ONE new bounded experiment - no
// round-18/19/20 partial candidate was repaired or replayed). The accepted
// kernel (11.385 us median / 11.954 us p90) still stages the small L2-hot A
// slice (16 x 288/256 int8) through LDS: per block that is 5/4 16-B global
// loads + 5/4 ds_write_b128 + 9/8 ds_read2_b32 (2-way bank conflicted,
// structural for any 16-B-aligned stride) that the K loop then reads back.
// Every block reads its own A rows exactly once, so the LDS round trip only
// adds instructions and a serialization point; the A values are 64 KiB total
// and L2-resident (re-read by all 360 blocks), so direct global fragment
// loads are cheap. Change (one consolidation): (1) split15_slice_body no
// longer stages A; each lane preloads its own 8-B fragment slice for EVERY
// DUMMA step into a per-lane register array a_slice[kStepsT] (9/8
// global_load_dwordx2 per lane, issued back-to-back BEFORE any ds_write -
// the iteration-11 register-array batching pattern - so the A slice costs
// ONE memory latency and zero LDS instructions); (2) the K loop consumes the
// A fragments from registers (du_mma_sync(acc, a_slice[s], b_frag, acc)) -
// same elements per lane per step as the LDS row_major loader (verified
// against du_mma.hpp: x[i] = p[row*ldm + col + i], row = lane&15, col =
// (lane>>4)*8, ldm = k), so the int32 accumulation order is untouched
// (k-ascending per split, ascending split sum -> outputs bit-identical, 0
// mismatches expected); (3) the shared a_lds array is deleted -> per-block
// LDS drops 9,732 -> 4,868 B (B 4,864 + s_is_last 4). B staging, the B
// col_major LDS reads, the zero-barrier structure (one __syncthreads before
// the B reads, unchanged), the slice-major pack, the fused last-arrival tail
// and every fallback are UNTOUCHED. Everything else is unchanged: grid = 24
// N-groups x 15 splits = 360 one-wave blocks = exactly 3 blocks/CU (trusted
// probe-set member 15; LDS is no longer the binder at 3 x 4,868 = 14,604
// B/CU), workspace = 15 int32 planes + 24 arrival counters = 368,736 B (<=
// the caller's 393,216 B budget; unsplit DUMMA fallback when smaller), one
// launch per replay, fence/atomic/barrier protocol unchanged (lane-0
// atomicAdd, LDS s_is_last flag, lane-0 atomicExch reset), exact (m,n,k)
// guard, generic scalar fallback for every unmatched shape including the
// paired M=2 (the packed384_byte_offset decode is untouched).
// Falsifiable predictions: (1) median_us < 11.385 (target ~10.8-11.3; the
// per-block staging loses the A LDS round trip and the K loop loses 9/8
// LDS reads) with p90_us <= 11.954 (guard vs the accepted best p90);
// (2) correctness 0 mismatches (bit-identical int32 order; M=2 paired
// fallback exact via the packed decode) and graph_capture_passed (single-
// launch structure unchanged, write-in-place planes, self-resetting
// counters, no clear); (3) PMC/ISA: grid_blocks 360 / workgroup_size 64 /
// s_barrier 0 / v_mmac 17 unchanged; group_segment_fixed_size 9,732 ->
// ~4,868 B; ds_write_b128 18 -> ~9/8 (B staging only); ds_read2_b32 34 ->
// ~17 (B only); ds_read_u8 stays 0; the A slice appears as 9/8 batched
// global_load_dwordx2 before the first B ds_write (explicit int2 loads; if
// instead the ISA shows the A loads sunk into the K loop or scalarized to
// global_load_ubyte, the compiler refused the batching/vectorization and
// this round documents that finding); arch_vgpr ~69 -> ~85-95 (a_slice
// holds 18/16 extra VGPR) with 0 spills (per-wave ~90 x 64 = 5,760 VGPR <<
// 65,536/CU, occupancy stays 3 blocks/CU = grid-bound); lds_instructions
// ~10,128 -> ~5-6k and lds_bank_conflicts ~29,184 -> ~15k (the A 2-way
// fragment-read conflicts disappear); vmem_read_instructions ~4,392 ->
// ~10,464 (A loads become 8-B per step instead of 16-B staged chunks - SAME
// bytes, more instructions, all issued in one batch); (4)
// memory_traffic.total_bytes_per_operator_replay unchanged ~2.06 MB (same
// A+B bytes; A stays L2-hot). If median stays >= 11.385 with p90 <= 11.954,
// the A LDS round trip was not on the critical path either (the kernel is
// at its launch/latency floor with 3 waves/CU) and this round documents the
// next plateau datum for this shape.
//
// The generic scalar path remains the fallback for every unmatched
// (m,n,k), including the paired M=2 shape with the same (N,K); the M=16
// DUMMA path is guarded by the exact (m,n,k) triple.
//
// Host symbols (matched against csrc/bindings.cpp):
//     launch_w8a8_gemm(...)
//     launch_pack_w8a8_weight(...)
//
// Known-good include order for this DTK: HIP runtime first, then the HIP
// bfloat16 header, then du_mma.h.

#include <hip/hip_runtime.h>
#include <hip/hip_bfloat16.h>
#include <du_mma.h>

#include <cstdint>

namespace {

// Byte offset of element (k, col) inside the exact (k,n)==(4096,384)
// slice-major-chunk weight pack (iteration 17; see
// w8a8_pack_weight_gateup_slicemajor_kernel): each 16-B chunk holds 16
// CONSECUTIVE k values of one n column, and chunks are stored in ascending
// (slice, n, k-chunk) order. With tile = col>>4, n_local = col&15 and
// kchunk = k>>4:
//   kchunk < 144 (the 8 long 288-k slices): s = kchunk/18, kk' = kchunk%18,
//     chunk base = s*288, chunks per n = 18;
//   kchunk >= 144 (the 7 short 256-k slices): c2 = kchunk-144,
//     s = 8 + c2/16, kk' = c2%16, chunk base = 2304 + (c2/16)*256,
//     chunks per n = 16;
//   offset = tile*65536 + (chunk_base + n_local*cps + kk')*16 + (k & 15).
// Fallback paths only (the timed M=16 staging path uses the same mapping
// implicitly through the pack's chunk order). 144 = 8 long slices * 18
// k-chunks per 288-k slice.
__device__ __forceinline__ int64_t packed384_byte_offset(int k, int col) {
  const int64_t tile = static_cast<int64_t>(col) >> 4;  // col / 16
  const int64_t n_local = static_cast<int64_t>(col) & 15;
  const int kchunk = k >> 4;
  int64_t base;
  int kk_p, cps;
  if (kchunk < 144) {
    const int s = kchunk / 18;
    base = static_cast<int64_t>(s) * 288;
    kk_p = kchunk % 18;
    cps = 18;
  } else {
    const int c2 = kchunk - 144;
    const int s = 8 + c2 / 16;
    base = 2304 + static_cast<int64_t>(c2 / 16) * 256;
    kk_p = c2 % 16;
    cps = 16;
  }
  return tile * 65536 + (base + n_local * cps + kk_p) * 16 + (k & 15);
}

// Scalar W8A8 GEMM: one thread computes one output element (m, n).
//
// Adjacent threads map to adjacent N columns (the fastest-changing dimension
// of both `out` and the [K, N] weight), so global reads coalesce in N.
// The K loop is executed exactly once per output element in ascending k and
// accumulated in int32; every assigned K keeps the exact int8 dot within
// int32 range (max |dot| = 128*128*4096 = 67,108,864 < 2^31).  The result is
// converted to float only for `dot * x_scale[m] * weight_scale[n]`, then
// stored as bfloat16, matching the contract reference exactly.
//
// gfx928: wavefront = 64, blockDim must be a multiple of 64 (128 here).
__global__ __launch_bounds__(128) void w8a8_gemm_scalar_kernel(
    const int8_t* __restrict__ a,        // x_q       [M, K] row-major
    const int8_t* __restrict__ b,        // weight    [K, N] row-major
    const float* __restrict__ x_scale,   //           [M, 1] contiguous
    const float* __restrict__ weight_scale,  //      [N, 1] contiguous
    hip_bfloat16* __restrict__ out,      //           [M, N] row-major
    int m,
    int n,
    int k) {
  const int64_t total = static_cast<int64_t>(m) * n;
  const int64_t idx =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= total) {
    return;
  }

  const int row = static_cast<int>(idx / n);
  const int col = static_cast<int>(idx - static_cast<int64_t>(row) * n);

  const int8_t* __restrict__ a_row = a + static_cast<int64_t>(row) * k;
  // For (k,n)==(4096,384) the weight buffer holds the per-tile
  // slice-major-chunk pack (iteration 17; see packed384_byte_offset); every
  // other (k,n) keeps the raw [K, N] row-major layout (identity pack).
  // Decode so this fallback stays exact for the paired M=2 shape and all
  // unmatched m.
  const bool packed_384 = (k == 4096 && n == 384);
  const int64_t b_row_stride = packed_384 ? 0 : static_cast<int64_t>(n);

  int32_t acc = 0;
#pragma unroll 8
  for (int kk = 0; kk < k; ++kk) {
    const int8_t b_val =
        packed_384 ? b[packed384_byte_offset(kk, col)]
                   : b[static_cast<int64_t>(col) +
                       static_cast<int64_t>(kk) * b_row_stride];
    acc += static_cast<int32_t>(a_row[kk]) * static_cast<int32_t>(b_val);
  }

  // Reference order: (float(dot) * x_scale[m]) * weight_scale[n], then bf16.
  // hip_bfloat16's float constructor performs the supported RNE conversion.
  const float scaled = static_cast<float>(acc) * x_scale[row] *
                       weight_scale[col];
  out[idx] = hip_bfloat16(scaled);
}

// ---------------------------------------------------------------------------
// Minimal gfx928 DUMMA INT8 path (assigned shape M=16, N=384, K=4096).
// Tile: 16x16x32, int32 accumulation, one 64-lane wavefront per block,
// one 16x16 output tile per wave (grid = N/16 = 24 blocks).
// ---------------------------------------------------------------------------

// gfx928 DUMMA INT8 m16n16k32 tile constants.
constexpr int kDummaM = 16;
constexpr int kDummaN = 16;
constexpr int kDummaK = 32;
constexpr int kDummaBlockThreads = 64;  // one wavefront per block

// M=16 W8A8 GEMM with the minimal 16x16x32 DUMMA tile: one wavefront owns one
// independent 16x16 output tile. A (x_q [16, K] row-major) and B (weight
// [K, N] row-major) fragments are loaded directly from global memory every
// DUMMA step with the library loaders; the int32 accumulator is explicit and
// drained through LDS in a coalesced epilogue that applies x_scale and
// weight_scale before the bf16 store. No LDS staging, no double buffering,
// and no cross-wave barrier in the K loop (a single wavefront per block;
// the one __syncthreads in the epilogue is reached by every block thread).
__global__ __launch_bounds__(kDummaBlockThreads) void
w8a8_gemm_m16_dumma_kernel(
    const int8_t* __restrict__ a,        // x_q          [16, K] row-major
    const int8_t* __restrict__ b,        // weight       [K, N] row-major
    const float* __restrict__ x_scale,   //              [16, 1] contiguous
    const float* __restrict__ weight_scale,  //         [N, 1] contiguous
    hip_bfloat16* __restrict__ out,      //              [16, N] row-major
    int n,
    int k) {
  const int lane = static_cast<int>(threadIdx.x);  // 0..63
  const int n_base = static_cast<int>(blockIdx.x) * kDummaN;

  __shared__ __align__(16) int32_t acc_tile[kDummaM * kDummaN];

  du::dumma::DUFragment<du::dumma::matrix_a, kDummaM, kDummaN, kDummaK,
                        signed char, du::dumma::row_major>
      a_frag;
  du::dumma::DUFragment<du::dumma::matrix_b, kDummaM, kDummaN, kDummaK,
                        signed char, du::dumma::row_major>
      b_frag;
  du::dumma::DUFragment<du::dumma::accumulator, kDummaM, kDummaN, kDummaK, int>
      acc_frag;
  du::dumma::du_fill_fragment(acc_frag, 0);

  // K is divisible by kDummaK for every assigned shape. A fragment rows are
  // x_q rows (ldm = k); B fragment rows are weight k-rows (ldm = n), with the
  // tile's 16 columns at n_base. The explicit int32 accumulation is
  // k0-ascending over exact 32-element dot products, so the final int32 total
  // is bit-identical to the scalar chain (integer arithmetic is exact).
  // Iteration 17: this fallback is only reachable for (m,n,k)==(16,384,4096),
  // whose weight buffer holds the per-tile slice-major-chunk pack. The
  // library row_major loader (a.x[i] = p[(col+i)*ldm + row], row = lane&15,
  // col = (lane>>4)*8) cannot decode that non-linear order, so the B
  // fragment is filled with per-element gathers from the pack: x[i] =
  // element (k0 + (lane>>4)*8 + i, n_base + (lane&15)) - the SAME fragment
  // registers the raw-layout loader would produce (the round-15 n-major
  // decode asserted the same mapping). Fallback path only - never on the
  // timed M=16 split-K kernel.
  const bool packed_384 = (k == 4096 && n == 384);
#pragma unroll 4
  for (int k0 = 0; k0 < k; k0 += kDummaK) {
    du::dumma::du_load_matrix_sync(a_frag, a + k0, k);
    if (packed_384) {
      const int n_local = lane & 15;
      const int k_off = (lane >> 4) * 8;
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        b_frag.x[i] =
            b[packed384_byte_offset(k0 + k_off + i, n_base + n_local)];
      }
    } else {
      du::dumma::du_load_matrix_sync(
          b_frag, b + static_cast<int64_t>(n_base >> 4) * (kDummaN * k) + k0,
          k);
    }
    du::dumma::du_mma_sync(acc_frag, a_frag, b_frag, acc_frag);
  }

  du::dumma::du_store_matrix_sync(acc_tile, acc_frag, kDummaN,
                                  du::dumma::mem_row_major);
  __syncthreads();

  // Coalesced epilogue: each lane drains 256/64 = 4 accumulator elements,
  // applies the per-row/per-column float scales and stores bf16.
#pragma unroll
  for (int linear = lane; linear < kDummaM * kDummaN; linear += 64) {
    const int row = linear >> 4;
    const int col = linear & 15;
    const float scaled = static_cast<float>(acc_tile[linear]) *
                         x_scale[row] * weight_scale[n_base + col];
    out[static_cast<int64_t>(row) * n + n_base + col] = hip_bfloat16(scaled);
  }
}

// ---------------------------------------------------------------------------
// Iteration 8 (HIP round): split-K=15 one-wave LDS-staged partial GEMM +
// split-15 combine (mandatory occupancy round - one limiter: LDS footprint
// -> blocks/CU; trusted probe set [2,5,10,15]).
//
// Round 7 measured the split-K=10 one-wave grid (240 blocks = exactly
// 2 blocks/CU, 26,880 B LDS/block = 53,760 B/CU) at 15.43 us median but was
// rejected by the p90 guard (19.43 > 19.04; bimodal samples - machine
// state, not kernel behavior). Its 26,880 B/block LDS caps occupancy at
// exactly 2 blocks/CU (3 x 26,880 = 80,640 B > 64 KiB). split-K=15 shortens
// every slice (288/256 vs 416/384 rows), shrinking the per-block LDS
// footprint to 18,688 B -> 3 x 18,688 = 56,064 B/CU <= 64 KiB, so 3 one-wave
// blocks/CU become legal, and the grid = 24 N-groups x 15 splits = 360
// one-wave blocks = EXACTLY 3 blocks/CU on 120 CUs (above the 2-blocks/CU
// latency-hiding target, integer-balanced, no tail wave). B stays once-read
// from HBM and A stays L2-hot (no repeated global reads - each block stages
// its own slice exactly once).
//   * block tile 16x16 (1 wavefront x 1 N-tile), 64 thr; grid = 24 x 15 =
//     360 one-wave blocks = exactly 3 blocks/CU;
//   * K=4096 -> 15 non-uniform 32-aligned slices: 8 x 288 (9 DUMMA steps)
//     + 7 x 256 (8 steps); k0-ascending within each slice, ascending split
//     sum -> bit-identical int32 order (0 mismatches expected);
//   * whole slice staged once into bank-skewed LDS (A 16x288 stride 304,
//     B 288x16 stride 48; 18,688 B/block), batched 16-B vector loads, one
//     __syncthreads (emitted as 0 s_barrier for the 1-wave block), then a
//     zero-barrier LDS-only K loop (9/8 steps x 1 A-frag + 1 B-frag ds_read
//     + 1 v_mmac, 9/8 v_mmac per block);
//   * workspace = 15 int32 planes [15][16][384] = 368,640 B (<= the
//     caller's 393,216 B workspace_split_k_capacity budget; every launch
//     overwrites every partial tile - no workspace clear);
//   * combine kernel: 24 blocks x 128 thr, all 15 plane loads issued into a
//     register array before the ascending split sum (exact int32 order),
//     then (float(acc) * x_scale[row]) * weight_scale[col] -> bf16;
//   * replay traffic ~2.32 MB (B 1.57 MB + planes RW 737,280 B + out 12,288
//     B) vs 2.47 MB for the accepted S=16 kernel (-6%); vs S=10 the plane
//     RW grows +245,760 B (+50%) - the combine cost of more splits is the
//     explicit trade being benchmarked against the occupancy gain.
// ---------------------------------------------------------------------------

constexpr int kSplitK = 15;              // 15 planes; workspace 368,640 B
constexpr int kNumNGroups = 24;          // N=384 / 16 (partial block tiles)
constexpr int kNumLongSplits = 8;        // 8 x 288-row slices + 7 x 256-row
constexpr int kSliceLong = 288;          // 9 x kDummaK steps (non-uniform)
constexpr int kSliceShort = 256;         // 8 x kDummaK steps
constexpr int kStepsLong = kSliceLong / kDummaK;   // 9
constexpr int kStepsShort = kSliceShort / kDummaK; // 8
constexpr int kPartialBlockThreads = 64; // one wavefront per block
constexpr int kCounterInts = kNumNGroups; // one arrival counter per N-group
constexpr int kStagedAStride = 304;      // 288 + 16 bank skew (16-B aligned)
// Iteration 15: B is staged n-major (element (k,n) at n*kStagedBStride + k,
// the transpose of the A staging convention), so the column stride must hold
// the longest k-slice (288) plus bank skew; 304 = 288 + 16 keeps every
// staging ds_write_b128 and every 8-byte fragment read 16-/8-B aligned.
constexpr int kStagedBStride = 304;      // 288 + 16 bank skew (16-B aligned)

// Partial GEMM: block (tile, split) computes one 16x16 output tile for the
// split's contiguous K slice and stores the int32 partial to
// partials[split][row][n]. The whole slice is staged into LDS once with
// batched 16-B vector loads, one barrier, then a zero-barrier LDS-only K
// loop. k0-ascending within the slice; the combine kernel sums slices in
// ascending split order, so the total int32 accumulation order is
// bit-identical to the scalar chain. The slice body is a compile-time
// template (kSliceKT, kStepsT) so the staging loops and the K loop are
// fully unrolled; the grid-uniform `split < kNumLongSplits` branch picks the
// 288- or 256-row instantiation for each block (one launch total).
template <int kSliceKT, int kStepsT>
__device__ __forceinline__ void split15_slice_body(
    const int8_t* __restrict__ a,        // x_q       [16, K] row-major
    const int8_t* __restrict__ b,        // weight    [K, N] row-major
    int32_t* __restrict__ partials,      //           [kSplitK][16][N] int32
    int n,
    int k,
    int split,
    int tile,
    int n_base,
    int k_base,
    int8_t (*b_lds)[kStagedBStride]) {
  const int tid = static_cast<int>(threadIdx.x);

  // Iteration 21 (HIP-only, A register-only transport; the validated TP4
  // o_proj lineage recipe "register-only K-loop transport"). The A slice is
  // small (16 x kSliceKT = 4,608 / 4,096 B, L2-hot 64-KiB activation) and
  // every block reads its own rows exactly once, so staging it through LDS
  // only buys the LDS round trip: per block it costs kALoads 16-B global
  // loads + kALoads ds_write_b128 + kStepsT ds_read2_b32 (2-way bank
  // conflicted). This round removes the A LDS stage entirely: each lane
  // loads its own 8-B fragment slice for EVERY DUMMA step into a per-lane
  // register array a_slice[kStepsT] (the library matrix_a row_major loader
  // with ldm = k reads x[i] = p[row*k + k_base + s*kDummaK + col + i], row =
  // lane&15, col = (lane>>4)*8 - the SAME 8 elements per lane per step the
  // LDS version read, so every fragment register is identical and the int32
  // accumulation order is bit-identical). All kStepsT A loads are issued in
  // one register-array block BEFORE any ds_write (the iteration-11 batching
  // pattern that forced the staging loads back-to-back in the accepted code
  // object), so the A slice costs one memory latency and zero LDS
  // instructions; the K loop then consumes the fragments straight from
  // registers. Guarded lanes are impossible (kStepsT * 64 lanes * 8 B =
  // kSliceKT * 16 B exactly, no tail).
  du::dumma::DUFragment<du::dumma::matrix_a, kDummaM, kDummaN, kDummaK,
                        signed char, du::dumma::row_major>
      a_slice[kStepsT];
  // Explicit vector loads: the library loader assigns the 8 bytes
  // individually, which LLVM scalarizes to 8 global_load_ubyte per step per
  // lane from GLOBAL memory (verified in the round-21 draft code object:
  // 136 ubyte loads = the old "direct fragment byte-load" poison). The
  // fragment's 8 elements are 8 CONSECUTIVE bytes of one row (m = lane&15)
  // at k = k_base + s*kDummaK + (lane>>4)*8, 8-B aligned (k_base, kDummaK
  // and the col offset are all multiples of 8; x_q is tensor-aligned), so
  // one explicit int2 load per lane per step forces ONE global_load_dwordx2
  // and the __builtin_memcpy (8 bytes, register-to-register) fills the same
  // fragment registers the LDS row_major loader produced.
  const int a_row = tid & 15;               // m index (library loader row)
  const int a_col = (tid >> 4) * 8;         // k offset (library loader col)
  const int8_t* a_base =
      a + static_cast<int64_t>(a_row) * k + k_base + a_col;
#pragma unroll
  for (int s = 0; s < kStepsT; ++s) {
    const int2 raw = *reinterpret_cast<const int2*>(a_base + s * kDummaK);
    __builtin_memcpy(a_slice[s].x, &raw, 8);
  }

  // Stage the B slice (kSliceKT rows x 16 cols) with coalesced 16-B vector
  // loads, then one barrier (B only - the A stage is gone). B chunks
  // (iteration 17) come from the per-tile slice-major-chunk pack (see
  // w8a8_pack_weight_gateup_slicemajor_kernel): chunk c holds 16 CONSECUTIVE
  // k values of one n column (n = c/18 long, c/16 short; k-chunk = c%18 /
  // c%16) and the chunks are stored in ascending c order, so the slice's
  // chunks occupy the CONTIGUOUS byte range [(k_base + c)*16] of the tile -
  // 64 lanes cover 1,024 CONTIGUOUS bytes per wave-load (the round-14
  // locality win). Same chunk content, same n-major LDS destination, same
  // per-lane order -> the fragment reads and the int32 accumulation stay
  // bit-identical.
  //
  // Iteration 11: issue ALL staging global loads into one register array
  // BEFORE any ds_write. The accepted kernel's exact gfx928 code object
  // serialized the slice staging: every global_load_dwordx4 was followed by
  // s_waitcnt vmcnt(0) before its ds_write_b128 (load_wait0_ds_write_windows
  // = 18 across the two slice instantiations, 10 long + 8 short), so each
  // block's staging paid ~10 (long) / 8 (short) full global-memory latencies
  // one after another before the K loop could issue its first ds_read
  // (lgkmcnt(0)). Holding the 16-B chunks in a register array first forces
  // the compiler to issue the loads back-to-back (all in flight
  // concurrently) and move the vmcnt waits to the write side, collapsing the
  // staging critical path to one memory latency. Guarded lanes (the unrolled
  // tail beyond the chunk count) write a zero placeholder so no
  // uninitialized register is ever consumed.
  constexpr int kChunksPerBRow = kSliceKT / 16;          // 18 / 16 (per n column)
  constexpr int kBLoads = (kSliceKT + 63) / 64;          // 5 / 4
  int4 s_buf[kBLoads];
#pragma unroll
  for (int i = 0; i < kBLoads; ++i) {
    const int c = tid + 64 * i;
    // Iteration 17: slice-major-chunk pack - chunk c (n = c/kChunksPerBRow,
    // k-chunk = c%kChunksPerBRow) lives at byte (k_base + c)*16 of the
    // tile, so a wave-load reads 1,024 contiguous bytes.
    s_buf[i] = (c < kSliceKT)
                   ? *reinterpret_cast<const int4*>(
                         b + static_cast<int64_t>(tile) * (kDummaN * k) +
                         (static_cast<int64_t>(k_base) + c) * 16)
                   : int4{0, 0, 0, 0};
  }
#pragma unroll
  for (int i = 0; i < kBLoads; ++i) {
    const int c = tid + 64 * i;
    if (c < kSliceKT) {
      // n-major LDS destination (iteration 15, UNCHANGED): element (k,n) at
      // n*kStagedBStride + k. Chunk c maps to (n = c/kChunksPerBRow,
      // k-chunk = c%kChunksPerBRow): for the 256-row instantiation this is
      // the same (c&15, c>>4) as iteration 15; for the 288-row
      // instantiation the chunk order is n-major (18 chunks per column) and
      // the division is resolved by the compiler (staging path only). Still
      // one ds_write_b128 per 16-B chunk.
      const int b_row = c / kChunksPerBRow;
      *reinterpret_cast<int4*>(
          &b_lds[b_row][(c - b_row * kChunksPerBRow) * 16]) = s_buf[i];
    }
  }

  __syncthreads();

  // The block's one wavefront owns the whole 16x16 tile. Zero-barrier K
  // loop: kStepsT steps of kDummaK=32; the A fragment comes from the
  // preloaded register array a_slice[s] (iteration 21, no LDS), the B
  // fragment from b_lds (iteration 15: col_major on the n-major layout, ldm
  // 304), one v_mmac per step (9 or 8 v_mmac per block). The col_major
  // loader computes b.x[i] = p[row*304 + col + i] (row = n_local = lane&15,
  // col = k offset (lane>>4)*8), i.e. the lane's 8 elements B[k0+col+i]
  // [n_base+row] are 8 CONSECUTIVE bytes of one n column -> ONE ds_read2_b32
  // per step (was 8 ds_read_u8 at stride 48). x[i] gets the same element as
  // the old row_major loader, so the fragment registers (and the int32
  // accumulation order) are bit-identical.
  du::dumma::DUFragment<du::dumma::matrix_b, kDummaM, kDummaN, kDummaK,
                        signed char, du::dumma::col_major>
      b_frag;
  du::dumma::DUFragment<du::dumma::accumulator, kDummaM, kDummaN, kDummaK, int>
      acc_frag;
  du::dumma::du_fill_fragment(acc_frag, 0);

#pragma unroll
  for (int s = 0; s < kStepsT; ++s) {
    const int k0 = s * kDummaK;
    du::dumma::du_load_matrix_sync(b_frag, &b_lds[0][k0], kStagedBStride);
    du::dumma::du_mma_sync(acc_frag, a_slice[s], b_frag, acc_frag);
  }

  // Direct int32 partial store: partials[split][row][n_base .. n_base+15].
  // Every launch overwrites every partial tile, so no workspace clear is
  // needed before capture or between replays.
  int32_t* partial_base =
      partials + static_cast<int64_t>(split) * kDummaM * n + n_base;
  du::dumma::du_store_matrix_sync(partial_base, acc_frag, n,
                                  du::dumma::mem_row_major);
}

// Fused last-arrival combine tail (iteration 9): removes the separate
// 24-block combine kernel -> exactly ONE kernel launch per replay.
//
// Protocol (the validated TP4 qkv lineage pattern):
//   * every block writes its 16x16 int32 plane tile, then __threadfence()
//     (release) and atomicAdd(&counters[tile], 1);
//   * the block whose atomicAdd returns kSplitK-1 is the 15th (last)
//     arriver of its N-group: it __threadfence()s (acquire), sums the 15
//     planes in ascending split order (exact int32 order), applies both
//     scales, stores bf16, and resets its counter to 0 (atomicExch) so the
//     next replay restarts from 0. The host zeroes the 24 counters once
//     before the first launch; the self-reset keeps every later replay
//     deterministic (stream-ordered, no racing increments).
__device__ __forceinline__ void split15_combine_tail(
    const int32_t* __restrict__ partials,    // [kSplitK][16][N] int32
    const float* __restrict__ x_scale,       // [16, 1] contiguous
    const float* __restrict__ weight_scale,  // [N, 1] contiguous
    hip_bfloat16* __restrict__ out,          // [16, N] row-major
    int32_t* __restrict__ counters,          // [kNumNGroups] monotonic
    int tile,
    int n) {
  const int lane = static_cast<int>(threadIdx.x);   // 0..63 (one wavefront)
  const int n_base = tile * kDummaN;
  const int plane = kDummaM * n;                    // int32 stride per split

  // Repair 1/4 (iteration 9): exactly ONE arrival per block. Only lane 0
  // fences, atomically increments the tile counter and publishes the result
  // through LDS; the other 63 lanes must NOT atomicAdd. The round-9 draft had
  // all 64 lanes incrementing -> 64 x 15 = 960 increments per tile per
  // replay, so the 15th increment (arrived == kSplitK-1) fired inside the
  // FIRST-arriving block, before its 14 siblings had stored their planes:
  // the premature combiner summed mostly-zero planes (first_mismatch
  // actual=0.0 at (0,0)) and the unguarded atomicExch reset made several
  // blocks combine per replay (6048/6144 mismatched). This mirrors the
  // validated TP4 qkv fused tail: tid-0 fence + atomicAdd, LDS flag, barriers.
  __syncthreads();
  __shared__ int s_is_last;
  if (lane == 0) {
    __threadfence();                                // release plane stores
    const int arrived = atomicAdd(&counters[tile], 1);
    __threadfence();                                // acquire other planes
    s_is_last = (arrived == kSplitK - 1) ? 1 : 0;
  }
  __syncthreads();
  if (s_is_last == 0) {
    return;                                         // not the last arriver
  }

  // 256 outputs / 64 lanes = 4 CONSECUTIVE columns per lane (row = lane>>2,
  // col0 = (lane & 3) * 4): one aligned int4 (16-B) plane load per split
  // replaces the round-9 loop's 4 scalar 4-B loads per element, cutting the
  // tail's vmem load instruction count 4x (60 -> 15 per lane) with no byte
  // or L2-request increase (each 16-B load is fully consumed; the 4 lanes
  // of one row cover a contiguous 64-B segment). The per-element int32
  // accumulation order is UNCHANGED: for each output element the 15 plane
  // values are still summed in ascending split order, so the outputs stay
  // bit-identical to the scalar chain. weight_scale is read once as a
  // float4 (n_base + col0 is 16-B aligned) and the 4 adjacent bf16 stores
  // merge into one 8-B store. Fence/atomic/barrier protocol untouched.
  const int row = lane >> 2;                 // 0..15 (4 lanes per row)
  const int col0 = (lane & 3) * 4;           // 0,4,8,12 (16-B aligned)
  const int32_t* p = partials + row * n + n_base + col0;
  const float4 ws =
      *reinterpret_cast<const float4*>(weight_scale + n_base + col0);
  const float xs = x_scale[row];
  int32_t acc0 = 0, acc1 = 0, acc2 = 0, acc3 = 0;
#pragma unroll
  for (int s = 0; s < kSplitK; ++s) {
    const int4 v = *reinterpret_cast<const int4*>(
        p + static_cast<int64_t>(s) * plane);
    acc0 += v.x;
    acc1 += v.y;
    acc2 += v.z;
    acc3 += v.w;
  }
  hip_bfloat16* o = out + static_cast<int64_t>(row) * n + n_base + col0;
  o[0] = hip_bfloat16(static_cast<float>(acc0) * xs * ws.x);
  o[1] = hip_bfloat16(static_cast<float>(acc1) * xs * ws.y);
  o[2] = hip_bfloat16(static_cast<float>(acc2) * xs * ws.z);
  o[3] = hip_bfloat16(static_cast<float>(acc3) * xs * ws.w);

  // All kSplitK arrivals of this replay have happened; restart for the next
  // replay (the kernel's completion on the stream orders this reset before
  // the next replay's first atomicAdd). One lane is enough; all 15 blocks
  // have already passed their atomicAdd, so no racing increment survives.
  if (lane == 0) {
    atomicExch(&counters[tile], 0);
  }
}

__global__ __launch_bounds__(kPartialBlockThreads) void
w8a8_gemm_m16_dumma_splitk15_lds_staged_partial_kernel(
    const int8_t* __restrict__ a,        // x_q       [16, K] row-major
    const int8_t* __restrict__ b,        // weight    [K, N] row-major
    int32_t* __restrict__ partials,      //           [kSplitK][16][N] int32
    const float* __restrict__ x_scale,   //           [16, 1] contiguous
    const float* __restrict__ weight_scale,  //      [N, 1] contiguous
    hip_bfloat16* __restrict__ out,      //           [16, N] row-major
    int32_t* __restrict__ counters,      //           [kNumNGroups] monotonic
    int n,
    int k) {
  const int tile = static_cast<int>(blockIdx.x) / kSplitK;   // 0..kNumNGroups-1
  const int split = static_cast<int>(blockIdx.x) % kSplitK;  // 0..kSplitK-1
  const int n_base = tile * kDummaN;
  const int k_base =
      (split < kNumLongSplits)
          ? split * kSliceLong
          : kNumLongSplits * kSliceLong + (split - kNumLongSplits) * kSliceShort;

  // Iteration 15: n-major B stage, element (k,n) at n*kStagedBStride + k
  // (16 n columns x 304 B = 4,864 B; the k-slice length is carried by the
  // template instantiation, not by the shared array). Iteration 21: the A
  // slice is no longer staged in LDS (register-only transport), so the
  // per-block LDS is B + the combine flag only.
  __shared__ __align__(16) int8_t b_lds[kDummaN][kStagedBStride];

  if (split < kNumLongSplits) {
    split15_slice_body<kSliceLong, kStepsLong>(a, b, partials, n, k, split,
                                               tile, n_base, k_base, b_lds);
  } else {
    split15_slice_body<kSliceShort, kStepsShort>(a, b, partials, n, k, split,
                                                 tile, n_base, k_base, b_lds);
  }

  // Fused combine tail (iteration 9): one launch per replay; the last
  // arriver of each N-group sums the 15 planes and stores bf16.
  split15_combine_tail(partials, x_scale, weight_scale, out, counters, tile,
                       n);
}

// Identity device-to-device weight pack (iteration 1: the DUMMA kernel reads
// B with the raw [K, N] row-major layout, so identity packing is exact).
// Kept valid for every (K, N) including unmatched shapes; the exact
// (k,n)==(4096,384) shape uses the per-tile slice-major-chunk pack below
// instead (iteration 17).
__global__ void w8a8_pack_weight_identity_kernel(
    const int8_t* __restrict__ src,
    int8_t* __restrict__ dst,
    int64_t total) {
  const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       i < total; i += stride) {
    dst[i] = src[i];
  }
}

// Per-tile slice-major-chunk weight pack for the exact (k,n)==(4096,384)
// shape (iteration 17): each 16-B chunk holds 16 CONSECUTIVE k values of one
// n column, and the 4,096 chunks of a tile are stored in ascending (slice,
// n, k-chunk) order:
//   chunk index = slice_chunk_base(s) + n_local*chunks_per_slice(s) + kk'
// with the 8 long 288-k slices (s < 8: base s*288, 18 chunks per n) followed
// by the 7 short 256-k slices (s >= 8: base 8*288 + (s-8)*256, 16 chunks per
// n); byte address = tile*65536 + chunk_index*16, chunk content =
// raw[k*384 + tile*16 + n_local] for k = k_base(s) + kk'*16 .. +15 (k_base
// numerically equals the chunk base: 288 k and 288 chunks per long slice).
// This is the iteration-15 n-major pack with the same chunk CONTENT but
// reordered memory (n*4096 + kk*16 -> slice-major chunk order), so a staged
// wave-load (64 consecutive chunks) covers 1,024 CONTIGUOUS bytes
// (sequential L2 sectors and sequential DRAM pages; was 16 x 64-B sectors
// at 4,096-B stride) while the n-major LDS destination, the
// one-ds_read2_b32 B fragment read and the int32 accumulation order are all
// untouched (outputs bit-identical to iteration 15). One-time,
// out-of-timed-region, out-of-Graph, same byte count and same buffer
// (graph-stable addresses unchanged). Every other (K, N) keeps the identity
// pack. Fallbacks decode this order via packed384_byte_offset.
__global__ void w8a8_pack_weight_gateup_slicemajor_kernel(
    const int8_t* __restrict__ src,
    int8_t* __restrict__ dst) {
  const int64_t idx =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  constexpr int64_t kTotal = 4096 * 384;  // 1,572,864
  if (idx >= kTotal) {
    return;
  }
  const int64_t tile = idx >> 16;         // / 65536
  const int64_t rem = idx - (tile << 16); // % 65536 (byte in tile)
  const int64_t chunk = rem >> 4;         // 16-B chunk index in tile
  const int64_t in_chunk = rem & 15;      // byte within the chunk
  int64_t base;                           // chunk index of the slice start
  int64_t n_local, kk_p;
  if (chunk < 8 * 288) {                  // long slice (18 chunks per n)
    const int64_t r = chunk % 288;
    base = (chunk / 288) * 288;
    n_local = r / 18;
    kk_p = r % 18;
  } else {                                // short slice (16 chunks per n)
    const int64_t c2 = chunk - 8 * 288;
    const int64_t r = c2 % 256;
    base = 8 * 288 + (c2 / 256) * 256;
    n_local = r / 16;
    kk_p = r % 16;
  }
  // base == k_base (k index of the slice start) numerically: 288 k / 288
  // chunks per long slice, 256 / 256 per short slice.
  const int64_t k = base + kk_p * 16 + in_chunk;
  dst[idx] = src[k * 384 + tile * 16 + n_local];
}

// Identity device-to-device scale pack (bootstrap).
__global__ void w8a8_pack_scale_identity_kernel(
    const float* __restrict__ src,
    float* __restrict__ dst,
    int total) {
  const int stride = gridDim.x * blockDim.x;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < total; i += stride) {
    dst[i] = src[i];
  }
}

}  // namespace

// Timed GEMM entry point (inside the CUDA/HIP Graph region).
//
// Graph-safety contract: no allocation, no compilation, no autotuning, no
// weight packing, no host/device synchronization, no default-stream launch.
// Launches only on the caller-provided stream and uses only caller-provided
// tensors. The iteration-9 split-K=15 path uses the caller's workspace as the
// int32 partial planes (write-in-place every launch, no clear needed) plus 24
// arrival counters in the workspace tail and issues exactly ONE kernel per
// replay (LDS-staged partial GEMM with a fused last-arrival combine tail);
// the iteration-1 unsplit kernel is the workspace-too-small fallback.
//
// Exact-shape specializations are guarded HERE, before the generic scalar
// fallback, so the paired M=2 API shape with the same (N, K) and every other
// shape still reach the scalar path below.
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
  const int64_t total = static_cast<int64_t>(m) * n;
  if (total <= 0) {
    return;
  }

  // Exact-shape M=16 specialization (iteration 9): split-K=15 LDS-staged
  // partial GEMM with a FUSED last-arrival combine tail -> exactly one
  // launch per replay (the round-8 falsifiable branch fired: median
  // 15.53 >= 15.4, so the two-launch structure is attacked instead of
  // geometry). grid = 24 N-groups (16x16 tiles) x 15 K-slices = 360 one-wave
  // blocks = exactly 3 blocks/CU on 120 CUs; K=4096 -> 15 non-uniform
  // 32-aligned slices (8 x 288 = 9 DUMMA steps + 7 x 256 = 8 steps); each
  // block stages its whole slice in LDS once (18,688 B/block -> 3 x 18,688 =
  // 56,064 B/CU <= 64 KiB), runs a zero-barrier LDS-only K loop, stores its
  // int32 plane, then the 15th arriver of its N-group sums the 15 planes in
  // ascending k order, applies both scales and stores bf16. The 24 arrival
  // counters live in the 96 B right after the 15 planes and are zeroed once
  // before the first launch (outside the timed Graph in the normal
  // eager-first flow). The iteration-1 unsplit DUMMA kernel remains the
  // fallback when the caller's workspace cannot hold 15 partial planes plus
  // the counters (368,640 + 96 B <= the workspace_split_k_capacity budget of
  // 393,216 B); the paired M=2 API shape with the same (N,K) and every
  // other shape still reach the scalar fallback below.
  if (m == 16 && n == 384 && k == 4096) {
    constexpr int64_t kPartialBytes =
        static_cast<int64_t>(kSplitK) * kDummaM * 384 * 4;   // 368,640 B
    constexpr int64_t kCounterBytes =
        static_cast<int64_t>(kCounterInts) * sizeof(int32_t);  // 96 B
    if (workspace != nullptr &&
        workspace_bytes >= kPartialBytes + kCounterBytes) {
      auto* partials = reinterpret_cast<int32_t*>(workspace);
      int32_t* counters = partials + kSplitK * kDummaM * 384;
      // Zero the 24 arrival counters once before the first launch. Issued on
      // the caller stream in the eager precheck, so the captured Graph
      // contains no memset node; if the first call is captured, the memset
      // becomes a replayable Graph node (still exact). The kernel's
      // self-reset keeps every later replay deterministic.
      static bool counters_zeroed = false;
      if (!counters_zeroed) {
        counters_zeroed = true;
        hipMemsetAsync(counters, 0, kCounterBytes, stream);
      }
      hipLaunchKernelGGL(
          w8a8_gemm_m16_dumma_splitk15_lds_staged_partial_kernel,
          dim3(kNumNGroups * kSplitK),
          dim3(kPartialBlockThreads),
          0,
          stream,
          a,
          b,
          partials,
          x_scale,
          weight_scale,
          reinterpret_cast<hip_bfloat16*>(out),
          counters,
          n,
          k);
    } else {
      hipLaunchKernelGGL(w8a8_gemm_m16_dumma_kernel, dim3(n / kDummaN),
                         dim3(kDummaBlockThreads), 0, stream, a, b, x_scale,
                         weight_scale, reinterpret_cast<hip_bfloat16*>(out),
                         n, k);
    }
    return;
  }

  constexpr int kBlockThreads = 128;  // multiple of the gfx928 wavefront (64)
  const int grid = static_cast<int>((total + kBlockThreads - 1) / kBlockThreads);

  hipLaunchKernelGGL(
      w8a8_gemm_scalar_kernel,
      dim3(grid),
      dim3(kBlockThreads),
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

// Optional out-of-timed-region weight packing. Iteration 1: identity
// device-to-device copy of raw_weight -> packed_weight and weight_scale ->
// packed_weight_scale, valid for any (K, N) and any unmatched shape (the
// DUMMA kernel reads the raw [K, N] row-major layout). Iteration 17: the
// exact (k,n)==(4096,384) shape is packed per-tile slice-major-chunk
// instead (see w8a8_pack_weight_gateup_slicemajor_kernel: same 16-B chunks
// as iteration 15, chunk order (slice, n, k-chunk) so staged wave-loads are
// 1,024-B contiguous); every other (K, N) keeps the identity copy. Same
// byte count, same buffer, one-time and outside the timed Graph region.
extern "C" void launch_pack_w8a8_weight(
    const int8_t* raw_weight,
    const float* weight_scale,
    int8_t* packed_weight,
    float* packed_weight_scale,
    int k,
    int n,
    hipStream_t stream) {
  const int64_t total = static_cast<int64_t>(k) * n;
  if (total > 0) {
    constexpr int kPackBlock = 256;
    const int grid = static_cast<int>((total + kPackBlock - 1) / kPackBlock);
    if (k == 4096 && n == 384) {
      hipLaunchKernelGGL(w8a8_pack_weight_gateup_slicemajor_kernel,
                         dim3(grid), dim3(kPackBlock), 0, stream, raw_weight,
                         packed_weight);
    } else {
      hipLaunchKernelGGL(
          w8a8_pack_weight_identity_kernel,
          dim3(grid),
          dim3(kPackBlock),
          0,
          stream,
          raw_weight,
          packed_weight,
          total);
    }
  }
  if (n > 0) {
    constexpr int kScaleBlock = 64;
    const int grid = (n + kScaleBlock - 1) / kScaleBlock;
    hipLaunchKernelGGL(
        w8a8_pack_scale_identity_kernel,
        dim3(grid),
        dim3(kScaleBlock),
        0,
        stream,
        weight_scale,
        packed_weight_scale,
        n);
  }
}
// @@end
