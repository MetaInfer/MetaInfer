"""Prompt templates for evolve-kernel sub-agents.

Eight-phase kernel optimization flow:
  A: Generate Correctness Harness
  B: Review Correctness Harness (adversarial)
  C: Generate Performance Harness
  D: Review Performance Harness (adversarial)
  E: Select Kernel (programmatic, no agent prompt needed)
  F: Optimize Kernel
  G: Verify Correctness (run harness)
  H: Measure Performance + Complexity → Update Library

Each builder returns a rendered prompt string.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional


def _render_req(req: Dict[str, Any]) -> str:
    from metainfer.orchestrator.requirements import req_summary_lines
    return "\n".join(req_summary_lines(req))


# ========================================================================== #
# A: Generate Correctness Harness
# ========================================================================== #

def gen_correctness_harness_prompt(
    req: Dict[str, Any],
    iter_dir: Path,
    kernel_code: str,
    kernel_fn_name: str,
    ref_kernel_path: str,
    iteration: int,
    review_feedback: Optional[str] = None,
    logs_dir: Optional[Path] = None,
) -> str:
    feedback_block = ""
    if review_feedback:
        feedback_block = f"""
# Previous Review Feedback (MUST ADDRESS)
The last correctness harness was REJECTED. Fix these issues:

{review_feedback}
"""

    return f"""You are the **CORRECTNESS HARNESS GENERATOR** for GPU kernel optimization, iteration #{iteration}.

# Task
Create a comprehensive correctness test harness for a Triton GPU kernel.

# Original Kernel (reference)
```python
{kernel_code}
```
Function name: `{kernel_fn_name}`

{feedback_block}

# Deliverable
Write ONE file: `{iter_dir}/correctness_harness.py`

The harness must:
1. Load the REFERENCE kernel from `{ref_kernel_path}` and the EVOLVED kernel from a path given as command-line arg `sys.argv[1]`
2. Generate diverse test inputs that cover ALL code paths in the kernel:
   - Different sizes: small (1-16), medium (64-256), large (512-4096)
   - Even AND uneven dimensions
   - Edge cases: size=1, power-of-2 sizes, non-power-of-2 sizes
   - Different dtypes if the kernel has dtype-dependent paths
   - If the kernel has internal conditional branches (if/else), create inputs that exercise BOTH paths
3. Run both kernels on the SAME inputs
4. Compare outputs using element-wise max error + NaN/Inf checks
5. Print ONE JSON line to stdout: {{"passed": true/false, "error": "...", "total": N, "passed_count": N, "results": [...]}}

**Critical:** Use CUDA events (`torch.cuda.Event`) for synchronization. Always `torch.cuda.synchronize()` before and after kernel calls.

# JSON output contract
```json
{{"passed": true, "error": null, "total": 10, "passed_count": 10, "results": [{{"index": 0, "name": "small_even", "passed": true, "error": null, "details": {{"max_error": 1e-7}}}}]}}
```

# Constraints
- The harness must be a STANDALONE Python script (can run as `python3 correctness_harness.py <evolved_kernel.py>`)
- Use `importlib.util` to load both kernels dynamically
- Tolerance for floating-point: 1e-3 relative OR 1e-5 absolute
- Handle the case where the evolved kernel CRASHES (catch exceptions, report as failure)

Write the complete harness file now.
"""


# ========================================================================== #
# B: Review Correctness Harness
# ========================================================================== #

def review_correctness_harness_prompt(
    req: Dict[str, Any],
    iter_dir: Path,
    kernel_code: str,
    harness_code: str,
    iteration: int,
    logs_dir: Optional[Path] = None,
) -> str:
    return f"""You are the **CORRECTNESS HARNESS REVIEWER** for iteration #{iteration}.

Your job: ADVERSARIALLY review the generated correctness harness. Find what it MISSES.

# Original Kernel
```python
{kernel_code}
```

# Generated Correctness Harness
```python
{harness_code}
```

# Review Checklist (be adversarial — assume it's insufficient unless proven otherwise)
1. **Code path coverage**: Does the test input set cover EVERY branch in the kernel?
   - Identify each `if`/`else` branch in the kernel
   - Verify at least one test case targets each branch
   - If there are boundary conditions (e.g., `K % BLOCK_SIZE == 0`), are BOTH paths tested?
2. **Edge cases**: Are extremes covered?
   - Minimum size (M=1, N=1, K=1)
   - Power-of-2 sizes
   - Non-power-of-2 (prime numbers ideally)
   - Very large sizes that might cause OOM
3. **Dtype coverage**: If the kernel has dtype-specific paths, are all tested?
4. **Tolerance**: Are error tolerances appropriate (not too loose)?
5. **Crash handling**: Does the harness catch kernel crashes and report them cleanly?
6. **Synchronization**: Does it use proper CUDA sync between kernel calls?

# Output
Write `{iter_dir}/correctness_review.md` with:
- **Verdict**: PASS (harness is complete) or FAIL (needs fixes)
- **Issues found**: List each gap with specific example inputs that are missing
- **Suggested fixes**: What the harness generator should add/change
- **Confidence**: 1-5 (how sure are you this harness will catch broken kernels)

If you find ANY gap, verdict MUST be FAIL. Be strict.
"""


# ========================================================================== #
# C: Generate Performance Harness
# ========================================================================== #

def gen_perf_harness_prompt(
    req: Dict[str, Any],
    iter_dir: Path,
    kernel_code: str,
    kernel_fn_name: str,
    ref_kernel_path: str,
    iteration: int,
    review_feedback: Optional[str] = None,
    logs_dir: Optional[Path] = None,
) -> str:
    feedback_block = ""
    if review_feedback:
        feedback_block = f"""
# Previous Review Feedback (MUST ADDRESS)
The last performance harness was REJECTED. Fix these issues:

{review_feedback}
"""

    return f"""You are the **PERFORMANCE HARNESS GENERATOR** for GPU kernel optimization, iteration #{iteration}.

# Task
Create a performance measurement harness for a Triton GPU kernel.

# Original Kernel (reference)
```python
{kernel_code}
```
Function name: `{kernel_fn_name}`

{feedback_block}

# Deliverable
Write ONE file: `{iter_dir}/perf_harness.py`

The harness must:
1. Load kernels: REFERENCE from `{ref_kernel_path}`, EVOLVED from `sys.argv[1]`
2. Generate realistic benchmark input sizes
3. Use INTERLEAVED A/B timing to cancel thermal drift:
   - For each test size, alternate: ref → evo → ref → evo → ... (20 pairs)
   - Use `torch.cuda.Event(enable_timing=True)` for precise GPU timing
4. Warmup: 5 warmup runs before timing
5. Print ONE JSON line to stdout with performance results

# JSON output contract
```json
{{"passed": true, "ref_median_ms": 0.5, "evo_median_ms": 0.4, "overall_speedup": 1.25, "per_case": {{"size_256x256": {{"ref_median_ms": 0.5, "evo_median_ms": 0.4, "speedup": 1.25}}}}, "num_cases": 3}}
```

# Critical
- Synchronize GPU before AND after each kernel call
- Use MEDIAN not mean for timing (outlier-resistant)
- Warmup runs go before timing
- Handle kernel crashes gracefully

Write the complete harness file now.
"""


# ========================================================================== #
# D: Review Performance Harness
# ========================================================================== #

def review_perf_harness_prompt(
    req: Dict[str, Any],
    iter_dir: Path,
    kernel_code: str,
    harness_code: str,
    iteration: int,
    logs_dir: Optional[Path] = None,
) -> str:
    return f"""You are the **PERFORMANCE HARNESS REVIEWER** for iteration #{iteration}.

Adversarially review the generated performance harness for methodological flaws.

# Original Kernel
```python
{kernel_code}
```

# Generated Performance Harness
```python
{harness_code}
```

# Review Checklist (be adversarial)
1. **Timing methodology**: Is it using CUDA events with proper sync? No CPU-only timing?
2. **Warmup**: Are there enough warmup runs (≥5) before measurement?
3. **Interleaving**: Does it interleave ref and evo measurements to cancel thermal drift?
4. **Sample size**: Enough iterations (≥20) for statistical significance?
5. **Input sizes**: Are benchmark sizes realistic for production workloads?
6. **Confounds**: Are there any system effects (cache warmup, GPU boost clock, memory allocation) that could bias results?
7. **Aggregation**: Uses median (not mean) for timing aggregation?

# Output
Write `{iter_dir}/perf_review.md` with:
- **Verdict**: PASS or FAIL
- **Issues found**: Each methodological flaw
- **Suggested fixes**
- **Confidence**: 1-5

Be strict — any methodological issue → FAIL.
"""


# ========================================================================== #
# F: Optimize Kernel
# ========================================================================== #

def optimize_kernel_prompt(
    req: Dict[str, Any],
    iter_dir: Path,
    original_kernel_code: str,
    current_kernel_code: str,
    kernel_fn_name: str,
    current_exec_time_ms: float,
    current_complexity: float,
    best_exec_time_ms: float,
    best_kernel_code: str,
    optimization_history: str,
    failure_feedback: Optional[str] = None,
    iteration: int = 0,
    logs_dir: Optional[Path] = None,
    headroom_bottleneck: str = "",
    headroom_bw_util_pct: float = 0.0,
    headroom_compute_util_pct: float = 0.0,
    headroom_pct: float = 0.0,
) -> str:
    fb = ""
    if failure_feedback:
        fb = f"""
# PREVIOUS ATTEMPT FAILED CORRECTNESS
The last optimization was REJECTED by the correctness harness. Fix this:

{failure_feedback}
"""

    best_section = ""
    if best_kernel_code != current_kernel_code:
        best_section = f"""
# Best Kernel So Far (exec_time={best_exec_time_ms:.4f}ms)
```python
{best_kernel_code}
```
"""

    # Build headroom context for the optimizer to guide escalation
    headroom_section = ""
    if headroom_bottleneck:
        headroom_blk = (
            f"## Headroom Analysis\n"
            f"- **Bottleneck:** {headroom_bottleneck}\n"
            f"- **BW usage:** {headroom_bw_util_pct:.0f}% of peak HBM\n"
            f"- **Compute usage:** {headroom_compute_util_pct:.0f}% of peak TFLOPS\n"
            f"- **Estimated remaining headroom:** {headroom_pct:.0f}%\n"
        )
        if headroom_bottleneck == "near_optimal" or headroom_pct < 10:
            headroom_blk += (
                "\n**⚠ CRITICAL: This kernel is at the hardware roofline.** "
                "Further Triton-level tuning is unlikely to yield meaningful gains. "
                "You MUST attempt a radically different approach:\n"
                "- Rewrite the entire kernel as a hand-tuned HIP C++ kernel with inline GCN assembly\n"
                "- Drop Triton entirely — direct HIP gives control over register allocation, "
                "shared memory banks, and instruction scheduling that Triton cannot match\n"
                "- Output BOTH `optimized_kernel.cpp` (HIP kernel) AND `optimized_kernel.py` "
                "(Python wrapper with same function signature for harness compatibility)\n"
            )
        elif headroom_bw_util_pct < 30 and headroom_compute_util_pct < 30:
            headroom_blk += (
                "\n**⚠ Kernel is inefficient.** Both bandwidth and compute utilization are low. "
                "Consider rewriting in HIP C++ with direct register control and SMEM management "
                "to overcome the Triton codegen overhead.\n"
            )
        headroom_section = headroom_blk

    mode = req.get("optimizer_mode", "Triton (standard)")
    if isinstance(mode, list):
        mode = mode[0] if mode else "Triton (standard)"
    # Normalize: form sends display labels, not internal keys
    if mode in ("HIP C++ (from scratch)", "hip_cpp", "hip"):
        mode = "hip_cpp"
    else:
        mode = "triton"  # default: Triton

    return _build_optimizer_prompt(
        mode=mode,
        iteration=iteration,
        original_kernel_code=original_kernel_code,
        current_kernel_code=current_kernel_code,
        current_exec_time_ms=current_exec_time_ms,
        current_complexity=current_complexity,
        best_exec_time_ms=best_exec_time_ms,
        best_kernel_code=best_kernel_code,
        best_section=best_section,
        failure_block=fb,
        optimization_history=optimization_history,
        iter_dir=iter_dir,
        kernel_fn_name=kernel_fn_name,
        headroom_section=headroom_section,
    )


def _build_optimizer_prompt(
    mode: str,
    iteration: int,
    original_kernel_code: str,
    current_kernel_code: str,
    current_exec_time_ms: float,
    current_complexity: float,
    best_exec_time_ms: float,
    best_kernel_code: str,
    best_section: str,
    failure_block: str,
    optimization_history: str,
    iter_dir: Path,
    kernel_fn_name: str,
    headroom_section: str = "",
) -> str:
    if mode == "hip_cpp":
        return _hip_cpp_optimizer_prompt(
            iteration=iteration,
            original_kernel_code=original_kernel_code,
            current_kernel_code=current_kernel_code,
            current_exec_time_ms=current_exec_time_ms,
            best_section=best_section,
            failure_block=failure_block,
            optimization_history=optimization_history,
            iter_dir=iter_dir,
            kernel_fn_name=kernel_fn_name,
            headroom_section=headroom_section,
        )
    else:  # triton (default)
        return _triton_optimizer_prompt(
            iteration=iteration,
            original_kernel_code=original_kernel_code,
            current_kernel_code=current_kernel_code,
            current_exec_time_ms=current_exec_time_ms,
            current_complexity=current_complexity,
            best_section=best_section,
            failure_block=failure_block,
            optimization_history=optimization_history,
            iter_dir=iter_dir,
            kernel_fn_name=kernel_fn_name,
            headroom_section=headroom_section,
        )


# ========================================================================== #
# Mode: Triton (standard — default)
# ========================================================================== #


def _triton_optimizer_prompt(
    iteration: int,
    original_kernel_code: str,
    current_kernel_code: str,
    current_exec_time_ms: float,
    current_complexity: float,
    best_section: str,
    failure_block: str,
    optimization_history: str,
    iter_dir: Path,
    kernel_fn_name: str,
    headroom_section: str = "",
) -> str:
    return f"""You are the **KERNEL OPTIMIZER** for GPU kernel optimization, iteration #{iteration}.

# Goal
Optimize the Triton kernel to run FASTER on the GPU while maintaining numerical correctness.

# Original (Reference) Kernel
```python
{original_kernel_code}
```

# Current Kernel to Optimize (exec_time={current_exec_time_ms:.4f}ms, complexity={current_complexity:.2f})
```python
{current_kernel_code}
```
{best_section}
{failure_block}
{headroom_section}

# Optimization History
{optimization_history}

# Optimization Strategy: MEMORY-FIRST, THEN COMPUTE

Most GPU kernels are memory-bound. Fix memory access patterns FIRST,
then fill idle compute slots SECOND. For AMD DCU gfx928 (warp_size=64, 120 CUs).

## Phase 1 — Memory Access Optimization (do this first!)
1. **Coalesce global memory**: Adjacent threads in a warp → adjacent addresses.
2. **Eliminate shared memory bank conflicts**: Pad SMEM to avoid same-bank collisions.
3. **Maximize data reuse**: Larger BLOCK_SIZE_M/N increases FLOPs/byte.
4. **Vectorized loads**: 128-bit loads (4×32-bit) reduce load count 4×.
5. **Double-buffering**: num_stages >= 2 to overlap load with compute.
6. **Prefetching**: Load first iteration before the loop.

## Phase 2 — Compute Optimization (after memory is fixed)
7. **Dual-issue**: gfx928 dual-issues scalar + vector in same cycle.
8. **ILP**: Process 2-4 output elements per thread.
9. **Reduce conversions**: Fuse dot→int32→fp32→scale path.
10. **Loop unrolling**: tl.constexpr with EVEN_K for compile-time unrolling.
11. **Warp count**: Adjust `num_warps` (4/8) to balance occupancy vs registers.
12. **Reduced BLOCK_SIZE_K**: Smaller K tiles (64/128) → more wavefronts in flight.

# Constraints
- MUST maintain the same function signature as the original
- MUST produce numerically equivalent results (within 1e-3 tolerance)
- Pure Triton — NO inline assembly, NO CUDA/HIP code
- Target hardware: AMD DCU gfx928, warp_size=64, 120 CUs
- Write the COMPLETE optimized kernel file

# Deliverable
Write the COMPLETE optimized kernel to `{iter_dir}/optimized_kernel.py`.

```python
import torch
import triton
import triton.language as tl

@triton.jit
def {kernel_fn_name}(...):
    # Your optimized implementation
    ...
```
Write ONLY the optimized kernel file.
"""


# ========================================================================== #
# Mode: HIP C++ (full rewrite)
# ========================================================================== #


def _hip_cpp_optimizer_prompt(
    iteration: int,
    original_kernel_code: str,
    current_kernel_code: str,
    current_exec_time_ms: float,
    best_section: str,
    failure_block: str,
    optimization_history: str,
    iter_dir: Path,
    kernel_fn_name: str,
    headroom_section: str = "",
) -> str:
    return f"""You are the **KERNEL OPTIMIZER** for GPU kernel optimization, iteration #{iteration}.

# Goal
Rewrite the kernel as a HIP C++ kernel with a Python wrapper for MAXIMUM performance
on AMD DCU gfx928. HIP C++ gives direct control over register allocation, shared memory,
and instruction scheduling — no Triton codegen overhead.

# Original (Reference) Triton Kernel — analyze its algorithm, inputs, outputs
```python
{original_kernel_code}
```

# Current Implementation (exec_time={current_exec_time_ms:.4f}ms)
```python
{current_kernel_code}
```
{best_section}
{failure_block}
{headroom_section}

# Optimization History
{optimization_history}

# Step 1: Analyze the Triton Kernel
Before writing any HIP code, carefully analyze the original kernel to identify:
1. **Algorithm**: What computation does this kernel do? (GEMM, attention, reduction, convolution, element-wise, etc.)
2. **Input tensors**: Names, shapes, dtypes — trace through the function body to understand each argument
3. **Output tensor**: What dtype and shape does the kernel produce?
4. **Key computation pattern**: The inner loop structure, accumulation pattern, and any non-trivial logic
5. **Function signature**: The exact Python function signature — you MUST replicate this exactly in the wrapper

# Step 2: Map to HIP C++ for gfx928 (warp_size=64, 120 CUs, peak HBM ~1.2 TB/s)

## General HIP Kernel Structure
```cpp
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hip/hip_bf16.h>

// Launch grid configuration
// gridDim.x = ceil(N / BLOCK_N), gridDim.y = ceil(M / BLOCK_M)
// blockDim.x = warp_size * num_warps_per_block
// Typical: blockDim = 256 (4 warps), BLOCK_M=128, BLOCK_N=128

__global__ void kernel_name(
    // Map each Triton tensor pointer to a HIP pointer
    // Use appropriate types: float, __half, __hip_bfloat16, int8_t, etc.
) {{
    // Thread/work-item indexing
    int tid = threadIdx.x;
    int wid = tid / warpSize;             // warp index within block
    int lane = tid % warpSize;            // lane within warp (0-63)

    // Block-level output tile: blockIdx.y → M tile, blockIdx.x → N tile
    // Each thread computes a fragment of the output tile

    // Shared memory declarations (use extern __shared__ or fixed size)
    // Accumulator in registers (fp32 or int32 depending on algorithm)

    // Main loop over reduction dimension
    for (int k = 0; k < K; k += BLOCK_K) {{
        // 1. Cooperative global→shared load (vectorized, coalesced)
        // 2. __syncthreads()
        // 3. Compute tile operation
        // 4. __syncthreads()
    }}

    // Epilogue: activation, scaling, type conversion, etc.
    // Cooperative store to global memory (coalesced)
}}
```

## Key Optimizations for gfx928
1. **Vectorized loads**: Use `__builtin_amdgcn_global_load_dwordx4` or `float4`/`uint4` for 16-byte loads — critical for HBM BW utilization
2. **Shared memory banking**: Pad SMEM arrays by 1 element to avoid bank conflicts on gfx928's 32-bank LDS
3. **Register blocking**: Each thread computes an 4×4 or 8×8 output tile to amortize load instructions — use #pragma unroll
4. **Double buffering**: Two SMEM buffers with async copy (`__builtin_amdgcn_sched_barrier`) to overlap loads and compute
5. **Assembly intrinsics**: `__builtin_amdgcn_mfma_*` for matrix ops, `__builtin_amdgcn_ds_*` for LDS ops, `__builtin_amdgcn_s_barrier` for fine-grained sync
6. **Occupancy management**: Target 4+ waves/CU (waves_per_cu = (120 * 4) / num_workgroups). Keep VGPR ≤ 128, LDS ≤ 32KB per block
7. **Coalesced memory access**: Ensure adjacent threads access adjacent addresses for all global load/store operations
8. **ILP (Instruction-Level Parallelism)**: Interleave independent operations to hide latency without consuming VGPRs

# Step 3: Write the TWO deliverable files

## File 1: `{iter_dir}/optimized_kernel.cpp`
Complete, self-contained HIP C++ kernel:
- `#include <hip/hip_runtime.h>` and any needed fp16/bf16 headers
- `__global__` kernel function with the actual algorithm from Step 1
- Host-side launch function (`void launch_kernel(...)`) that computes grid/block dims and calls the kernel
- HIP error checking wrapper (`HIP_CHECK(call)`)
- Use `extern "C"` for the host function so it can be called from Python via ctypes

## File 2: `{iter_dir}/optimized_kernel.py`
Python wrapper using `torch.utils.cpp_extension.load_inline()`:
```python
import os
import torch
import torch.utils.cpp_extension

# JIT-compile the HIP kernel at import time
_src_dir = os.path.dirname(os.path.abspath(__file__))
_cpp_path = os.path.join(_src_dir, "optimized_kernel.cpp")
_cpp_source = open(_cpp_path).read()

_hip_module = torch.utils.cpp_extension.load_inline(
    name="evolved_hip_kernel",
    cpp_sources=[_cpp_source],
    functions=["launch_kernel"],  # the host-side C function name
    extra_cflags=["-O3", "-ffast-math"],
    with_cuda=True,  # PyTorch uses CUDA/HIP interchangeably
    verbose=False,
)

# IMPORTANT: Replicate the ORIGINAL kernel's function name and signature EXACTLY.
# The correctness/perf harness calls this function by name.
# Convert PyTorch tensors to raw pointers, call launch_kernel, return output.
def {kernel_fn_name}(...):
    # ... same signature as original ...
    raise NotImplementedError("Replace with actual implementation")
```

## Critical Constraints
- **SAME function name**: The Python wrapper MUST export a function named `{kernel_fn_name}` — the test harness imports it by this name
- **SAME function signature**: Arguments, return type, and semantics MUST match the original kernel — the harness calls it with the same inputs
- **Numerical equivalence**: Results must match the reference within 1e-3 relative tolerance
- **Self-contained**: The `.py` file reads the `.cpp` file at import time and JIT-compiles it — no manual build step
- **Error handling**: Check all HIP calls with `hipGetErrorString` and raise Python RuntimeError on failure

Write BOTH files now. Do NOT omit either file.
"""


# ========================================================================== #
# Complexity Evaluation
# ========================================================================== #

def evaluate_complexity_prompt(
    kernel_code: str,
    iteration: int = 0,
    logs_dir: Optional[Path] = None,
) -> str:
    return f"""You are a **KERNEL COMPLEXITY EVALUATOR**. Rate the COMPLEXITY of a GPU kernel.

# Kernel Code
```python
{kernel_code}
```

# Complexity Rating Scale (0.0 = simplest, 1.0 = most complex)
Rate these dimensions and give an overall score:

1. **Code length** (0.0-1.0): lines of code relative to a typical kernel
2. **Control flow** (0.0-1.0): number of if/else branches, loops, conditions
3. **Memory access** (0.0-1.0): complex addressing patterns, shared memory usage
4. **Register pressure** (0.0-1.0): hint at high register usage from local variables
5. **Overall** (0.0-1.0): weighted average (30% code length, 30% control flow, 25% memory, 15% registers)

# Output Format
Respond with ONLY a JSON object on the last line:
```json
{{"code_length": 0.5, "control_flow": 0.3, "memory_access": 0.4, "register_pressure": 0.6, "overall_complexity": 0.45}}
```

Lower complexity = easier to further optimize → better for the kernel library selection.
"""


# ========================================================================== #
# Retrospective
# ========================================================================== #

def retrospective_prompt(
    req: Dict[str, Any],
    iter_dir: Path,
    iteration: int,
    exec_time_ms: float = 0.0,
    prev_exec_time_ms: float = 0.0,
    complexity: float = 0.5,
    speedup: float = 1.0,
    library_size: int = 0,
    added_to_library: bool = False,
    review_feedback: Optional[str] = None,
    logs_dir: Optional[Path] = None,
) -> str:
    added = "YES — kernel entered top-10 library" if added_to_library else "NO — did not beat library threshold"
    return f"""You are the **RETROSPECTIVE WRITER** for iteration #{iteration}.

# Iteration Results
- Exec time: {exec_time_ms:.4f}ms (prev: {prev_exec_time_ms:.4f}ms, speedup: {speedup:.2f}x)
- Complexity: {complexity:.2f}
- Added to library: {added}
- Library size: {library_size}

# Task
Write `{logs_dir}/retrospective.md`:
- **Goal**: what was attempted this iteration
- **What changed**: specific optimization tried
- **Perf result**: {exec_time_ms:.4f}ms (Δ from {prev_exec_time_ms:.4f}ms)
- **Why perf moved**: analysis of why the speed changed
- **Complexity**: why {complexity:.2f}
- **Library status**: {added}
- **Next steps**: what to try in the next iteration

Keep under 400 words. Do NOT modify code.
"""


def failure_retrospective_prompt(
    req: Dict[str, Any],
    iter_dir: Path,
    iteration: int,
    failure_reason: Optional[str] = None,
    failed_phase: Optional[str] = None,
    logs_dir: Optional[Path] = None,
) -> str:
    fr = failure_reason or "(unknown)"
    fp = failed_phase or "(unknown)"
    return f"""You are the **POSTMORTEM WRITER** for FAILED iteration #{iteration}.

Write `{logs_dir}/retrospective.md`:
- **Failed phase**: {fp}
- **Failure**: {fr}

Sections: What was tried, Where it broke, Root cause analysis, What next iter should do differently.
Keep under 300 words. Do NOT modify code.
"""


# ========================================================================== #
# Kernel function name extraction (not a prompt, but a helper)
# ========================================================================== #

def kernel_fn_name_from_code(kernel_code: str) -> str:
    """Extract the kernel function name from Triton kernel code.

    Looks for ``def <name>(...)`` after a ``@triton.jit`` decorator.
    Falls back to ``triton_kernel``.
    """
    import re
    # Match @triton.jit followed by def function_name
    m = re.search(r'@triton\.jit\s*\n\s*def\s+(\w+)', kernel_code)
    if m:
        return m.group(1)
    # Fallback: find any def after @triton.jit on same line
    m = re.search(r'@triton\.jit\s+def\s+(\w+)', kernel_code)
    if m:
        return m.group(1)
    # Last resort: first function definition that looks like a kernel
    m = re.search(r'def\s+(\w*kernel\w*)', kernel_code, re.IGNORECASE)
    if m:
        return m.group(1)
    m = re.search(r'def\s+(\w+)', kernel_code)
    if m:
        return m.group(1)
    return "triton_kernel"
