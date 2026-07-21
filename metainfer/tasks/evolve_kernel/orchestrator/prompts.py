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
{fb}

# Optimization History
{optimization_history}

# Optimization Guide (Triton-specific, for DCU/AMD GPU)
1. **Tile size tuning**: Adjust BLOCK_SIZE_M/N/K. Larger tiles = more parallelism but more register pressure.
2. **Thread coarsening**: Have each thread compute multiple output elements.
3. **Memory coalescing**: Ensure adjacent threads access adjacent memory.
4. **Software pipelining**: Use `num_stages` > 2 to overlap compute and memory.
5. **Warp count**: Adjust `num_warps` to balance occupancy vs register usage.
6. **Precision**: Use `tl.float16` or `tl.bfloat16` compute types where safe.
7. **Loop ordering**: Reorder loops for better memory access patterns.
8. **Reduce shared memory**: Use less SMEM to increase occupancy.

# Constraints
- MUST maintain the same function signature as the original
- MUST produce numerically equivalent results (within 1e-3 tolerance)
- MUST remain a Triton kernel (no CUDA/HIP inline assembly)
- Test on the ACTUAL hardware — detect GPU type before optimizing
- Write the COMPLETE optimized kernel file (not a diff)

# Deliverable
Write the COMPLETE optimized kernel to `{iter_dir}/optimized_kernel.py`.
Include the FULL kernel with all imports, the `@triton.jit` decorated function, and any helper functions.

```python
# optimized_kernel.py — complete, runnable Triton kernel file
import torch
import triton
import triton.language as tl

@triton.jit
def {kernel_fn_name}(...):
    # Your optimized implementation
    ...
```

Write ONLY the optimized kernel file. Do NOT modify any other files.
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
