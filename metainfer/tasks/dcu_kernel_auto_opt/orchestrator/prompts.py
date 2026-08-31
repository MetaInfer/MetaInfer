"""Prompt templates for coordination and kernel implementation agents."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, Optional


HARNESS_PATH = (
    Path(__file__).resolve().parent.parent / "assets" / "w8a8_bench.py"
)


def split_k_candidate_set(
    shape: Dict[str, Any],
    *,
    cu_count: int,
    max_split: int = 16,
) -> list[int]:
    """Suggest occupancy probes; callers must still benchmark outside them."""
    try:
        n = int(shape["N"])
        k = int(shape["K"])
        cus = int(cu_count)
    except (KeyError, TypeError, ValueError):
        return []
    if n <= 0 or k < 64 or cus <= 0 or max_split < 2:
        return []
    stage_count = k // 64
    candidates: set[int] = {2}
    for waves_per_block in (1, 2, 3, 4, 6, 8):
        n_blocks = math.ceil(n / (16 * waves_per_block))
        for resident_batches in (1, 2):
            ideal = cus * resident_batches / n_blocks
            candidates.update((math.floor(ideal), math.ceil(ideal)))
    return sorted(
        split for split in candidates
        if 2 <= split <= min(max_split, stage_count)
    )


def _shapes_table(shapes: Dict[str, Dict[str, Any]]) -> str:
    return "\n".join(
        "| {sid} | {m} | {n} | {k} |".format(
            sid=sid,
            m=params.get("M", "?"),
            n=params.get("N", "?"),
            k=params.get("K", "?"),
        )
        for sid, params in shapes.items()
    )


def shape_balanced_assignment(
    shapes: Dict[str, Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    """Deterministically balance exact shapes across up to four workers."""
    shape_work: Dict[str, int] = {}
    for shape_id, params in shapes.items():
        try:
            work = (
                2
                * int(params["M"])
                * int(params["N"])
                * int(params["K"])
            )
        except (KeyError, TypeError, ValueError):
            work = 1
        shape_work[shape_id] = work

    worker_count = min(4, len(shapes))
    bins: list[tuple[int, list[str]]] = [
        (0, []) for _ in range(worker_count)
    ]
    for shape_id in sorted(
        shapes, key=lambda name: (-shape_work[name], name)
    ):
        index = min(
            range(worker_count), key=lambda item: (bins[item][0], item)
        )
        current_work, current_shapes = bins[index]
        bins[index] = (
            current_work + shape_work[shape_id],
            current_shapes + [shape_id],
        )
    return {
        f"worker_{gpu}": {"gpu": gpu, "shapes": bins[gpu][1]}
        for gpu in range(worker_count)
        if bins[gpu][1]
    }


def w8a8_strategy_guidance(
    shapes: Dict[str, Dict[str, Any]],
) -> str:
    """Return gfx928 strategy guidance tailored to the assigned M values."""
    m_values = {
        int(params["M"])
        for params in shapes.values()
        if "M" in params
    }
    if m_values and max(m_values) <= 4:
        return """This is a small-M decode lane (M <= 4). Start with a
bandwidth-conscious scalar/vector path, not DUMMA by reflex:
- map adjacent wavefront lanes to adjacent N columns so B[k, n] loads are
  coalesced at each K step;
- use blockDim 128 or 256 (a multiple of the gfx928 wavefront size 64);
- stage/reuse the tiny A tile in LDS or registers while accumulating int32;
- if staging A in LDS, stage the whole tiny A matrix once and use at most one
  synchronization before compute; never add a barrier for every K tile;
- let one thread or lane compute one or a small fixed number of output columns;
- do not use split-K in bootstrap. Consider it later only when measured grid
  parallelism is insufficient and include reduction/combine cost."""
    if m_values == {16}:
        return """This is an M=16 lane. The primary candidate is native gfx928
DUMMA INT8 m16n16k32:
- include `<du_mma.h>` and use the exact installed DTK API below:
  `du::dumma::DUFragment<du::dumma::matrix_a,16,16,32,signed char,du::dumma::row_major>`
  for A, the analogous `matrix_b` fragment for B, and
  `du::dumma::DUFragment<du::dumma::accumulator,16,16,32,int>` for C;
- call `du::dumma::du_fill_fragment`, `du::dumma::du_load_matrix_sync`,
  `du::dumma::du_mma_sync`, and `du::dumma::du_store_matrix_sync`;
- `row_major` is a fragment layout type, while the store layout argument is
  `du::dumma::mem_row_major`; do not use CUDA-like unprefixed function names;
- one 64-thread wavefront owns each independent DUMMA fragment/tile;
- K is divisible by 32 and N by 16 for every assigned shape;
- use int8 A/B fragments and int32 accumulation, then apply x_scale[m] and
  packed_weight_scale[n] before bf16 store;
- choose 1, 2, or 4 wavefronts per block only after checking N-grid
  parallelism, LDS use, and register pressure;
- use coalesced global-to-LDS loads and a layout accepted by the installed
  du_mma.h. Do not invent CUDA WMMA/PTX APIs."""
    if m_values and min(m_values) >= 128:
        return """This is a large-prefill lane (M >= 128). Treat it as a
throughput GEMM, not a decode kernel:
- use native gfx928 INT8 DUMMA m16n16k32 and build larger 2-D macro-tiles from
  multiple wavefront-owned fragments;
- sweep M/N block tiles and waves per block; start from 64x64, 64x128 and
  128x64 output tiles, then select using measured VGPR, LDS and occupancy;
- cooperatively vector-load A and packed B into LDS with padded/swizzled
  layouts, reuse both operands across adjacent fragments, and compare single
  versus double buffering using measured memory stalls;
- keep int32 accumulation resident through the full K loop, fuse scale and
  bf16 conversion into the final store, and avoid a separate epilogue;
- do not use split-K by default: the MxN grid already exposes substantial
  parallelism. Test it only for an evidenced K-pipeline imbalance and include
  workspace reduction cost;
- report achieved TOPS and HBM traffic together with median/P90 so compute,
  LDS, and memory bottlenecks are distinguished."""
    return """This lane contains different M regimes. Implement explicit
shape/M dispatch: use a coalesced scalar/LDS path for M <= 4 and evaluate
native gfx928 DUMMA INT8 m16n16k32 for M=16. For M>=128, use a throughput
DUMMA macro-tile with 2-D A/B reuse and tune tile/LDS/occupancy. Do not force
one launch geometry onto all regimes."""


def w8a8_round_strategy(
    shape: Dict[str, Any],
    iteration: int,
    history: list[Dict[str, Any]] | None = None,
    pmc_evidence: Dict[str, Any] | None = None,
    max_iterations: int = 10,
    isa_policy: Dict[str, Any] | None = None,
) -> str:
    """Choose an architecture-first round plan from measured evidence."""
    history = history or []
    pmc_evidence = pmc_evidence or {}
    isa_policy = isa_policy or {}

    faster_wrong = [
        record for record in history
        if record.get("correctness_passed") is False
        and isinstance(record.get("speedup"), (int, float))
        and float(record["speedup"]) > 1.0
    ]
    if faster_wrong:
        candidate = max(
            faster_wrong, key=lambda record: float(record["speedup"])
        )
        return (
            "Highest priority: repair the faster but incorrect candidate from "
            f"iteration {candidate.get('iteration')} (measured speedup "
            f"{candidate.get('speedup')}x). Read its archived source at "
            f"`{candidate.get('artifact_dir')}` and preserve the fast mapping. "
            "Fix only the smallest correctness defect: signed int8 unpacking, "
            "tail bounds, scale indexing, bf16 conversion, or a race. Do not "
            "replace it with an unrelated architecture."
        )

    if history:
        latest = history[-1]
        failure = str(latest.get("failure_reason") or "").lower()
        if any(token in failure for token in (
            "timeout", "timed out", "killed", "no result", "exit 143",
        )):
            return (
                "The preceding attempt failed in the agent infrastructure. "
                "Return to the accepted best source and start a new bounded "
                "experiment; do not repair or replay a partially written "
                "candidate. This failure does not count as a completed "
                "optimization round."
            )
        if latest.get("build_success") is False:
            return (
                "Repair the immediately preceding candidate from iteration "
                f"{latest.get('iteration')} at `{latest.get('artifact_dir')}`. "
                "Keep its strategy and make only the minimum compile/API/syntax "
                "correction; do not start another redesign this round."
            )

    phase = str(isa_policy.get("phase") or "hip_only")
    if phase == "isa_guided_hip":
        completed = int(isa_policy.get("valid_isa_guided_rounds") or 0)
        return (
            "ISA-guided HIP round. This is successful ISA experiment "
            f"{completed + 1} of at least 2. Select one measured memory or "
            "compute bottleneck, compare the exact primary-kernel ISA with "
            "the preceding code object, and make one HIP/DUMMA/intrinsic "
            "code-shaping change. Raw inline asm remains forbidden. Record "
            "a compiler limitation only when the before/after binary proves "
            "it and name the exact target instructions."
        )
    if phase == "conditional_inline_asm":
        return (
            "Conditional inline-asm experiment. Target only the compiler "
            "limitation and exact instructions verified by the immediately "
            "preceding ISA-guided HIP round. Keep the asm block minimal, "
            "preserve complete constraints/clobbers, and reject it unless "
            "the candidate ISA, exact correctness, median, P90, and resources "
            "all validate. Do not write raw global/buffer/flat loads or MMAC."
        )

    small_m = {
        1: "Vectorize contiguous K loads with exact signed-int8 semantics.",
        2: (
            "Increase instruction-level parallelism with independent int32 "
            "accumulators or adjacent N outputs; avoid per-K-tile LDS barriers."
        ),
        3: (
            "Stage all of tiny A once with at most one barrier, or tune unroll "
            "one step if whole-A staging is not cheaper."
        ),
        4: (
            "Change one launch variable only: waves per block, N columns per "
            "wave, or unroll factor."
        ),
        5: (
            "HIP-only memory round: change one vector-load width, contiguous "
            "N mapping, or whole-A reuse decision. Raw inline asm is forbidden."
        ),
        6: (
            "HIP-only pipeline round: reduce one dependency chain or barrier "
            "using ordinary HIP/intrinsics. Raw inline asm is forbidden."
        ),
        7: (
            "HIP-only resource round: tune one block size, unroll factor, or "
            "live range while preserving coalescing. Raw inline asm is forbidden."
        ),
        8: (
            "HIP-only consolidation round: revisit the fastest correct archived "
            "mapping and make one final architecture/codegen improvement. Raw "
            "inline asm is forbidden."
        ),
    }
    m16 = {
        1: (
            "Establish a minimal 16x16x32 DUMMA tile with the exact API below, "
            "one wave per output tile and explicit int32 accumulation. If the "
            "seed already has a correct DUMMA kernel, preserve it and instead "
            "test the smallest one-wave-per-block, one-N-tile geometry with no "
            "cross-wave barrier; do not spend the round reimplementing it."
        ),
        2: (
            "Architecture round: measure grid parallelism before polishing. "
            "Explore one complete launch geometry among 1/2/4 waves per block "
            "and 1/2/4 adjacent N tiles. Prefer enough independent blocks to "
            "cover at least all device CUs; report grid_blocks, waves_per_block "
            "and estimated_active_cus in proposal.json."
        ),
        3: (
            "Architecture round: if the unsplit grid has fewer than two "
            "blocks per device CU and K >= 1024, implement and measure "
            "split-K=2 plus at least one CU-aligned candidate (which may be "
            "non-power-of-two), or test a one-wave zero-barrier geometry that "
            "reaches the same parallelism. Write int32 partials into the "
            "caller workspace and include the combine+scale kernel in the "
            "timed Graph."
        ),
        4: (
            "Architecture/pipeline round: explore one of multi-N-tile reuse, "
            "A-only staging, or bounded register/LDS prefetch. Retain enough "
            "blocks to cover all CUs, state which A/B bytes are reused, and "
            "measure whether the change improves normal median/P90."
        ),
        5: (
            "HIP-only packed-weight/staging round: compare one packed layout, "
            "A-only staging, or B-only staging design. Raw inline asm is forbidden."
        ),
        6: (
            "Pipeline round: choose exactly one staging family from direct, "
            "A-only LDS, B-only LDS, or A+B LDS using L2/VMEM evidence. Use "
            "coalesced 8- or 16-byte cooperative loads and report HBM/LDS byte "
            "changes; do not claim asynchronous overlap without evidence."
        ),
        7: (
            "Pipeline round: compare single buffering with double buffering "
            "only when K>=1024, L2 hit rate is below 70%, and the doubled LDS "
            "budget stays below 48 KiB. Count barriers per K step."
        ),
        8: (
            "HIP-only resource round: tune one occupancy limiter using actual "
            "PMC evidence: waves per block, VGPR live range, LDS footprint, "
            "or spill removal. Do not trade repeated HBM reads for occupancy."
        ),
        9: (
            "Late ISA-diagnosis round. Only if the control-plane plateau gate "
            "is open, use one selected ISA Skill and trusted disassembly to "
            "shape compiler output through HIP/DUMMA/intrinsics. Raw inline "
            "asm remains forbidden. Otherwise continue HIP-only exploration."
        ),
        10: (
            "Final conditional inline-asm round. Raw asm is allowed only when "
            "the control plane confirms a HIP plateau and the prior ISA-guided "
            "round recorded one concrete compiler limitation plus target "
            "instructions. Otherwise make one HIP-only consolidation change."
        ),
    }
    large_m = {
        1: (
            "Establish a correct DUMMA throughput baseline using a 2-D "
            "macro-tile. Benchmark 64x64, 64x128, and 128x64 block tiles; "
            "record waves per block, VGPRs, LDS bytes, occupancy and TOPS."
        ),
        2: (
            "Operand-reuse round: compare direct loads with cooperative "
            "A+B LDS staging. Quantify A/B reuse per macro-tile and use "
            "vectorized coalesced global loads with a bank-safe LDS layout."
        ),
        3: (
            "Pipeline round: compare single and double buffering across K "
            "tiles. Retain double buffering only when ISA/PMC evidence shows "
            "reduced VMEM stalls without harmful LDS or occupancy growth."
        ),
        4: (
            "Tile-shape round: tune M-tile versus N-tile aspect ratio for "
            "this exact M/N/K, balancing B reuse, A reuse and enough blocks "
            "to occupy every CU. Do not inherit decode launch geometry."
        ),
        5: (
            "Packing round: test one weight packing/swizzle that makes each "
            "DUMMA B tile vector-loadable and LDS-bank-safe. Include packing "
            "outside timing and validate the graph-stable packed layout."
        ),
        6: (
            "Epilogue round: fuse per-row and per-column scales, bf16 "
            "conversion and the final coalesced store into the compute "
            "kernel; remove any unnecessary workspace/combine pass."
        ),
        7: (
            "Compute-pipeline round: tune DUMMA issue grouping, prefetch "
            "distance and accumulator independence using ISA stall evidence. "
            "Raw inline asm remains forbidden."
        ),
        8: (
            "Resource round: tune waves per block, VGPR live ranges and LDS "
            "footprint from measured occupancy. Recheck the best tile family "
            "with normal median/P90 measurements."
        ),
        9: m16[9],
        10: m16[10],
    }
    m = int(shape.get("M", 0))
    if m >= 128:
        portfolio = large_m
    elif m < 16:
        portfolio = small_m
    else:
        portfolio = m16
    late_start = max(9, max_iterations - 1)
    if m < 16:
        if iteration < late_start:
            return portfolio.get(iteration, portfolio[8])
        return portfolio[9] if iteration < max_iterations else portfolio[10]
    if iteration < late_start:
        decision = portfolio.get(iteration, portfolio[8])
    else:
        decision = portfolio[9] if iteration < max_iterations else portfolio[10]
    grid_blocks = pmc_evidence.get("grid_blocks")
    cu_count = pmc_evidence.get("device_cu_count")
    split_candidates = split_k_candidate_set(
        shape,
        cu_count=int(cu_count) if isinstance(cu_count, (int, float)) else 0,
    )
    if iteration >= 2 and split_candidates:
        decision += (
            " Trusted occupancy-probe split candidates for the measured CU "
            f"count are {split_candidates}. They include non-power-of-two "
            "values where useful, are not a whitelist, and must fit the "
            "workspace and stage-alignment constraints. Explore outside this "
            "set when evidence supports it."
        )
    if (
        iteration >= 2
        and isinstance(grid_blocks, (int, float))
        and isinstance(cu_count, (int, float))
        and grid_blocks < 2 * cu_count
    ):
        decision += (
            f" Trusted control-plane warning: current grid has {grid_blocks} "
            f"blocks for {cu_count} CUs, below the two-blocks-per-CU latency-"
            "hiding target. Before micro-optimization, benchmark a finer "
            "one-wave zero-barrier grid or multiple legal split-K candidates "
            "including combine cost."
        )
    return decision


def generate_kernel_prompt(
    *,
    operator: str,
    dtype: str,
    shapes: Dict[str, Dict[str, Any]],
    hardware: str,
    kernel_language: str,
    source_dir: Path,
    harness_path: Path,
    api_contract_path: Path | None = None,
    iteration: int = 0,
    prev_failure: Optional[str] = None,
    fixed_assignment: Dict[str, Dict[str, Any]] | None = None,
) -> str:
    """Prompt the main Agent to coordinate, never implement, kernels."""
    contract_path = api_contract_path or (
        source_dir / "int8_w8a8_gemm_api.py"
    )
    failure_block = ""
    if prev_failure:
        failure_block = f"""
## Previous attempt failed

The last coordination attempt failed with:

{prev_failure}

Correct only `proposal.json`. Do not respond by writing implementation code.
"""
    assignment = (
        fixed_assignment
        if fixed_assignment is not None
        else shape_balanced_assignment(shapes)
    )
    assignment_example = json.dumps(assignment, indent=2)
    assignment_rule = (
        "The control plane has already selected the authoritative "
        "`gpu_assignment` shown in the deliverable. Preserve it exactly; "
        "do not rebalance, regroup, add, or remove shapes."
        if fixed_assignment is not None
        else (
            "Use the deterministic shape-balanced `gpu_assignment` shown "
            "in the deliverable."
        )
    )
    return f"""You are the MAIN COORDINATOR for a DCU kernel optimization task.

## Your responsibilities

1. Read the immutable Python API contract at `{contract_path}`.
2. Read the trusted Generate preflight evidence at
   `{source_dir / "generation_preflight.json"}`.
3. Inspect the staged correctness harness at `{harness_path}`, the build
   scaffold, `profile_pmc.sh`, hardware, and complete shape list below.
4. Confirm that the repository contains no HIP implementation.
5. Review or preserve the GPU assignment as instructed below.
6. Write only `{source_dir / "proposal.json"}`.

## Hard role boundary

You are not a kernel implementation agent. Do not write, edit, rename, or
delete any HIP, C++, CUDA, Python backend, setup/build, test, benchmark, or
public API file. In particular, do not create `.hip`, `.cu`, `.cpp`,
`w8a8_backend.py`, or `setup.py`. Do not compile or run the correctness
harness. The control plane has staged only the immutable API and build/loader
scaffolding. After assignment, child implementation Agents create their
initial HIP kernels from scratch during Parallel explore, and the control
plane validates them.

The API contract is user-owned and immutable. The repository scaffold and
interface have already been prepared by the control plane.
The fixed public call is `w8a8_gemm_out(...)`; its backend operation is
`torch.ops.zth_w8a8.gemm_out`.
The trusted scaffold contains `w8a8_bench.py` and `profile_pmc.sh`. Do not
modify or execute them. The control plane has already run the harness's CPU
PyTorch-reference self-test, probed GPU visibility, checked PMC script syntax,
verified the real hipprof command, and recorded those facts in
`generation_preflight.json`. Actual kernel correctness and PMC collection are
deferred until a child Agent creates a kernel during Parallel explore.

## GPU assignment policy

## Task context

- Operator: {operator}
- Dtype: {dtype}
- Hardware: {hardware}
- Kernel language for child agents: {kernel_language}
- Child workers: between 1 and 4 non-empty workers
- Mapping: worker_N→physical GPU N; one to four non-empty workers
- Assignment instruction: {assignment_rule}

| ID | M | N | K |
|---|---|---|---|
{_shapes_table(shapes)}

The assignment unit is one exact shape ID. Different M variants of the same
logical operator may be assigned to different workers; for example,
`m4_wqkv_a` and `m16_wqkv_a` may use different shape-specific HIP kernels.
The fixed public API remains one interface, and final synthesis dispatches to
the selected kernel by shape. Every target shape must appear exactly once.
{failure_block}
## Deliverable

Write strict JSON to `{source_dir / "proposal.json"}`:

```json
{{
  "iteration": {iteration},
  "generated": false,
  "hypothesis": "brief shape grouping and load-balance rationale",
  "profile_evidence": {{
    "hardware": "{hardware}",
    "coordination_only": true
  }},
  "profiling_plan": {{
    "script": "profile_pmc.sh",
    "mode": "hipprof_pmc_csv",
    "trigger": "usable DUMMA bootstrap, accepted best, or late ISA decision",
    "reuse_when_source_digest_matches": true,
    "skip_scalar_bootstrap": true,
    "acceptance_timing": "unprofiled_cuda_graph_replay_median_p90"
  }},
  "scaffold_review": {{
    "preflight_file": "generation_preflight.json",
    "preflight_status": "passed",
    "harness_reference_self_test_passed": true,
    "gpu_probe_passed": true,
    "cudagraph_available": true,
    "python_graph_wrapper_staged": true,
    "pmc_script_checked": true,
    "no_hip_implementation": true
  }},
  "gpu_assignment": {assignment_example}
}}
```

Every target shape must appear exactly once. Every emitted worker must receive
at least one shape. Use up to four GPUs; a subset task may use fewer when its
operator-family grouping does not provide four independent units of work.
Do not change any other file.
"""


def bootstrap_worker_prompt(
    *,
    worker_id: str,
    gpu: int,
    shapes: Dict[str, Dict[str, Any]],
    hardware: str,
    kernel_language: str,
    source_dir: Path,
    harness_path: Path,
    api_contract_path: Path,
    attempt: int,
    prev_failure: Optional[str] = None,
) -> str:
    """Prompt one child to generate a new initial HIP implementation."""
    del harness_path
    failure_block = ""
    if prev_failure:
        failure_block = f"""
## Previous attempt failed

{prev_failure}

Make the smallest source-only correction for the reported issue before
returning it to the trusted control plane. Do not inspect the machine,
toolchain, network, or unrelated files.
"""
    has_large_prefill = any(
        int(shape.get("M", 0)) >= 128 for shape in shapes.values()
    )
    if has_large_prefill:
        bootstrap_strategy = """
Bootstrap is iteration 0 and remains correctness-first, but a large-Prefill
scalar K loop is not a usable profiling baseline. For every assigned shape
with M >= 128, create a simple native INT8 DUMMA m16n16k32 tiled kernel with
int32 accumulation. Use enough M/N tiles to expose CU parallelism; keep the
first implementation direct or single-buffered and avoid split-K, raw asm,
or speculative deep pipelines.

When variant reference code is present, you may read and adapt measured
implementations as reference. Variants live in a staged tree under
`references/` organized as `<operator+dtype>/<model>/<TP>/<M>/<operator>.hip`
(for example `references/int8w8a8-gemm/hy3/TP4/M4096/o_proj.hip`); the
legacy flat file `references/w8a8_gemm_variants.hip` is also available.
Navigate the tree with `ls`/`find` to the directory matching your exact
operator/dtype, model, TP, and M family, then read only the specific operator
file you need. Variants are neither a whitelist nor a restriction: take only
the minimum code needed for a correct starting point and continue exploring
freely in later rounds. Do not claim their measurements until the trusted
control plane revalidates this source. Reusing a variant never locks you into
it: adapt freely, combine ideas, or ignore it and start fresh — every
accepted kernel must still pass the full correctness/Graph/median-P90/
resource validation.

Keep one simple scalar int8/int32 fallback for unmatched shapes and small-M
API cases. Shapes with M < 128 may use that scalar fallback for bootstrap.
"""
        bootstrap_path = "dumma_prefill_with_scalar_fallback"
    else:
        bootstrap_strategy = """
Bootstrap is iteration 0, not a performance round. For every assigned shape,
including M=16, implement one simple scalar int8 dot-product kernel:

- map one thread to one output element or one/few adjacent N columns;
- use blockDim 128 or 256 and a straightforward grid over M*N;
- compute the complete K loop exactly in int32, then apply the two float
  scales and store bf16;
- prioritize code that compiles and passes exact correctness.

Do not use DUMMA, split-K, double buffering, complicated LDS layouts, inline
assembly, or speculative performance machinery for this small-M bootstrap.
Those belong to measured optimization iterations after correctness passes.
"""
        bootstrap_path = "scalar_correctness"
    return f"""You are CHILD kernel implementation agent `{worker_id}` in
Parallel explore. You own physical GPU {gpu} and only these assigned shapes:

| ID | M | N | K |
|---|---|---|---|
{_shapes_table(shapes)}

The main coordinator has already fixed the API, repository, hardware, and
shape/GPU assignment. This is a new task, not continuation mode: no HIP
implementation has been supplied. Create the initial HIP kernel for the
assigned shapes from scratch. Only the staged optional variant file may be
adapted as a reference under the rules below; do not copy another task repo.

## Immutable interface

Read `{api_contract_path}` and preserve it byte-for-byte. Implement its private
backend contract without creating a second public API.

## Required implementation

The control plane owns these staged scaffold files under `{source_dir}`:

- `w8a8_backend.py`
- `w8a8_graph.py`
- `setup.py`
- `csrc/bindings.cpp`

You own and must create `csrc/w8a8_gemm_hip.hip`. Do not assume an existing
compiled kernel is present.

The trusted binding already registers both `gemm_out` and the optional
out-of-timed-region `pack_weight` operation. Your HIP file must provide these
two stable host launch symbols:

- `launch_w8a8_gemm(..., void* workspace, int64_t workspace_bytes,
  int m, int n, int k, hipStream_t stream)`;
- `launch_pack_w8a8_weight(raw_weight, weight_scale, packed_weight,
  packed_weight_scale, int k, int n, hipStream_t stream)`.

Bootstrap `launch_pack_w8a8_weight` as an identity device-to-device copy.
Later Parallel explore rounds may change only its HIP implementation and the
matching GEMM interpretation to test packed layouts. The main coordinator and
Generate phase never write the HIP implementation.

Register `torch.ops.zth_w8a8.gemm_out`, return the exact caller-provided `out`
tensor, use PyTorch's current HIP stream, and preserve graph-safe behavior.
The trusted harness rejects the bootstrap unless this fixed Python API can be
captured on a non-default stream with `torch.cuda.CUDAGraph`, replayed for
exact correctness, and called from Python through `w8a8_graph.py`.
The mathematical reference is:
`(A.float() @ B.float()) * x_scale * weight_scale.T`, converted to bfloat16.

The timed operator must perform no allocation, compilation, autotuning,
weight packing, host synchronization, device synchronization, or default-
stream launch. It may use only the caller-provided `out` and `workspace`.
Weight packing, if implemented, belongs in the optional `pack_weight` op
outside the timed region.

`w8a8_backend.py` must keep the control-plane entry point
`load_extension()`. It uses `torch.utils.cpp_extension.load(...,
is_python_module=False)` because the operator is registered through
`TORCH_LIBRARY`. Never replace it with an import of a prebuilt `.so` or an
installed package.

## Hardware

- Target: {hardware}
- Language: {kernel_language}
- Compile target: gfx928
- Visible physical GPU: {gpu}
- Native wavefront size: 64
- LDS capacity: 64 KiB per CU
- INT8 DUMMA support is m16n16k32 with int32 accumulation
{failure_block}

## Correctness-first bootstrap strategy

{bootstrap_strategy}

Include `<hip/hip_bfloat16.h>`, use `hip_bfloat16`, and use its supported
float conversion; alternatively use the existing proven manual uint16 store
without switching type families.

For every path:

1. Include the installed headers in this known-good order:
   `<hip/hip_runtime.h>`, then `<hip/hip_bfloat16.h>`, then `<du_mma.h>`.
   This DTK's `du_mma.h` is not self-contained when included before the HIP
   runtime headers. Do not include or invent `du_mma_common.h`; it is not an
   installed public header.
2. Keep adjacent lanes on adjacent addresses in the fastest-changing N
   dimension. Audit every hard-coded warp value: gfx928 wavefront is 64, not
   32; blockDim must be a multiple of 64.
3. Accumulate the integer dot product in int32. The maximum assigned K keeps
   the exact int8 dot within int32 range. Convert to float only for
   `dot * x_scale[m] * packed_weight_scale[n]`, then store bf16.
4. Use `hipLaunchKernelGGL` on PyTorch's current HIP stream. Never use
   stream 0, `hipDeviceSynchronize`, `hipStreamSynchronize`, or a temporary
   allocation in `gemm_out`.
5. Never use NVIDIA `wmma`, `mma.sync`, PTX, warp=32 masks, FP8, or INT4 on
   gfx928.
6. Keep all barriers on paths reached by every thread in the block. Budget
   LDS before choosing single/double buffering; bootstrap should prefer one
   correct, understandable buffer over speculative complexity.
7. Support every assigned shape through explicit dispatch where launch
   geometry differs. Do not specialize for only the first shape.
8. Keep a scalar generic fallback for every unmatched `(m,n,k)`. Later
   optimization rounds will specialize only an exact shape and the accepted
   object will be linked directly into the final extension. In particular,
   an M=16 specialization must remain guarded so the paired M=2 API shape
   with the same `(N,K)` still reaches the scalar fallback correctly. Keep
   `launch_pack_w8a8_weight` valid for unmatched `(K,N)` as well; identity
   device-to-device packing is its generic fallback.

## Execution boundary

Do not run the benchmark or correctness harness. Do not invoke Docker, SSH,
Skill tools, pip, apt, conda, network access, package installation,
environment activation, filesystem-wide searches such as `find /`, or
PyTorch/CUDA environment probes. Read only the immutable API, named scaffold
files, and optional `references/w8a8_gemm_variants.hip`. The trusted control plane already owns the correct
DTK/PyTorch environment and will compile, run exact correctness, and measure
median/P90 after you return. Your job is source implementation only.

You may inspect the immutable API and scaffold source read-only. Preserve
`w8a8_backend.py`, `setup.py`, and `csrc/bindings.cpp`. You may change only
`csrc/w8a8_gemm_hip.hip` plus `proposal.json`. Do not edit tests, the harness,
cache/build artifacts, or files outside this worktree.

Then write `{source_dir / "proposal.json"}` as strict JSON:

```json
{{
  "iteration": {attempt},
  "generated": true,
  "hypothesis": "specific initial kernel strategy chosen for these M/N/K shapes",
  "profile_evidence": {{
    "worker_id": "{worker_id}",
    "physical_gpu": {gpu},
    "shapes_targeted": {json.dumps(list(shapes))},
    "path": "{bootstrap_path}",
    "block_threads": 128,
    "tile_m": 0,
    "tile_n": 0,
    "tile_k": 0,
    "lds_bytes": 0,
    "split_k": 1,
    "expected_bottleneck": "memory, compute, launch, or occupancy",
    "risk": "main correctness or performance risk",
    "validation_owner": "trusted_control_plane"
  }},
  "files_changed": [
    "csrc/w8a8_gemm_hip.hip"
  ]
}}
```
"""
