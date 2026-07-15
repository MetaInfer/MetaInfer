"""Prompt templates for opt-kernel sub-agents.

Concise prompts — the bare minimum to drive the ABCDEF loop for GPU kernel
optimization. Each builder returns a rendered prompt string.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional


NOTEBOOKS_HINT = """Knowledge base: `notebooks/` directory. Use Glob + Read tools to consult
relevant reference designs. Read only the files matching this iteration's task."""

HARDWARE_DETECTION_MANDATE = """# MANDATORY: detect actual GPU hardware BEFORE writing any code
The `target_hardware` field in the requirements is what the USER selected — it may
be aspirational, inaccurate, or for a different machine. Your kernel MUST target the
HARDWARE THAT IS ACTUALLY PRESENT on this machine, not what the user typed.

Before writing a single line of kernel code, RUN these detection commands and report
what you find:

```bash
# Detect GPU vendor and model
if command -v rocm-smi >/dev/null; then
    echo "=== ROCm platform ==="
    rocm-smi --showproductname 2>/dev/null || rocm-smi -a 2>/dev/null | grep -i 'Card Series' | head -4
    echo "HIP version:" && hipconfig --full 2>/dev/null | grep 'HIP version' | head -1
elif command -v nvidia-smi >/dev/null; then
    echo "=== NVIDIA platform ==="
    nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader
else
    echo "=== No GPU tool found ==="
    lspci | grep -i -E 'vga|3d|display|accelerator' || echo "unknown"
fi

# Check available toolchain
echo "=== Toolchain ==="
which hipcc 2>/dev/null || echo "no hipcc"
which nvcc 2>/dev/null || echo "no nvcc"
python3 -c "import triton; print('Triton', triton.__version__)" 2>/dev/null || echo "no Triton"
python3 -c "import torch; print('PyTorch', torch.__version__, 'CUDA', torch.version.cuda)" 2>/dev/null || echo "no PyTorch"
```

Rules:
1. **Run this FIRST** in your first Bash turn. Do not skip it.
2. **Report the actual hardware** in your plan.md / first commit message.
3. **If actual hardware differs from target_hardware**, write the kernel for the
   ACTUAL hardware, but note the discrepancy in plan.md. The requirements field is
   a user aspiration; the real hardware is the build target.
4. **Use the toolchain that actually exists**: if `hipcc` is present, write HIP
   kernels. If `nvcc` is present, write CUDA kernels. If Triton is available,
   prefer Triton for portability. Never assume a toolchain that isn't installed.
5. **Verify after compilation**: run a trivial smoke test (allocate, copy, launch a
   tiny kernel) to confirm the runtime works before attempting the full kernel."""



def _render_req(req: Dict[str, Any]) -> str:
    lines = [f"- task_type: {req.get('task_type', '?')}",
             f"- task_id: {req.get('task_id', '?')}"]
    _skip = {"task_type", "task_id", "raw_request", "answers"}
    for k, v in req.items():
        if k in _skip:
            continue
        if isinstance(v, (list, tuple)):
            v = ", ".join(str(x) for x in v) if v else "(none)"
        lines.append(f"- {k}: {v}")
    answers = req.get("answers") or {}
    for k, v in answers.items():
        if isinstance(v, (list, tuple)):
            v = ", ".join(str(x) for x in v) if v else "(none)"
        lines.append(f"- {k}: {v}")
    return "\n".join(lines)


def plan_prompt(
    req: Dict[str, Any],
    iter_dir: Path,
    notebooks_dir: Path,
    iteration: int,
    prev_failures: Optional[str] = None,
    review_feedback: Optional[str] = None,
    perf_plan: Optional[str] = None,
    logs_dir: Optional[Path] = None,
) -> str:
    pf = prev_failures or "(none — first iteration)"
    rf = f"\n\n# Previous iteration's review\n{review_feedback}" if review_feedback else ""
    pp = f"\n\n# Previous perf plan\n{perf_plan}" if perf_plan else ""
    return f"""You are the **PLANNER** for GPU kernel optimization, iteration #{iteration}.

# Task requirements
{_render_req(req)}

# Working directory
{iter_dir}

# Knowledge base
{NOTEBOOKS_HINT}
Path: {notebooks_dir}

# Previous failures
{pf}{rf}{pp}

# Deliverables
Write exactly two files in `{iter_dir}`:
1. `plan.md` — work plan: goal, file-by-file work items, test plan, perf targets, risks
2. `test_spec.md` — test spec: inputs, expected outputs, JSON output shape

### Kernel correctness: agent-written test.sh
For correctness, the implementer will write `test.sh`. The test MUST output a
JSON line: `{{"passed": true/false, "perf": {{...}}, "notes": "..."}}`.

Do NOT write code. Planning only.

{HARDWARE_DETECTION_MANDATE}
"""


def implement_prompt(
    req: Dict[str, Any],
    iter_dir: Path,
    notebooks_dir: Path,
    iteration: int,
    prev_failure: Optional[str] = None,
    review_feedback: Optional[str] = None,
    perf_plan: Optional[str] = None,
    logs_dir: Optional[Path] = None,
) -> str:
    pf = prev_failure or "(none)"
    rf = f"\n\n# Review from previous iteration\n{review_feedback}" if review_feedback else ""
    pp = f"\n\n# Perf plan to execute\n{perf_plan}" if perf_plan else ""
    return f"""You are the **IMPLEMENTER** for GPU kernel optimization, iteration #{iteration}.

# Task requirements
{_render_req(req)}

# Working directory
{iter_dir}

# Plan
Read `{iter_dir}/plan.md` and `{iter_dir}/test_spec.md`.

# Knowledge base
{NOTEBOOKS_HINT}
Path: {notebooks_dir}

# Previous failure
{pf}
Fix this FIRST if a failure is shown.{rf}{pp}

# Deliverables
1. The kernel code described in the plan, inside `{iter_dir}`.
2. A test script at `{iter_dir}/test.sh` (bash, executable):
   - Compile and run the kernel, compare output against a correctness reference.
   - On success print: `{{"passed": true, "perf": {{"<metric>": <number>}}, "notes": "..."}}`
   - On failure print: `{{"passed": false, "error": "...", "traceback": "..."}}`
   - Exit 0 in all cases.

{HARDWARE_DETECTION_MANDATE}

# Mandatory self-test
Before declaring B done, run `bash {iter_dir}/test.sh` yourself. If it doesn't
pass, fix the kernel before handing off to C.
"""


def implement_redo_prompt(
    req: Dict[str, Any],
    iter_dir: Path,
    notebooks_dir: Path,
    iteration: int,
    prev_failure: Optional[str] = None,
    logs_dir: Optional[Path] = None,
) -> str:
    return f"""Your previous B (implement) turn did not produce a working kernel.
You're in a resumed session — do NOT re-bootstrap.

# What went wrong
{prev_failure or "(no detail)"}

# Working directory
{iter_dir}

Fix the issue with minimal edits. Re-run `bash test.sh` to confirm before exiting.
"""


def write_test_harness_prompt(
    req: Dict[str, Any],
    iter_dir: Path,
    notebooks_dir: Path,
    iteration: int,
) -> str:
    return f"""You are the **TEST HARNESS WRITER** for iteration #{iteration}.

Write `{iter_dir}/test.sh` that compiles + runs the kernel in `{iter_dir}` and
compares against a reference implementation.

Output contract:
- success → `{{"passed": true, "perf": {{"<metric>": <number>}}, "notes": "..."}}`
- failure → `{{"passed": false, "error": "...", "traceback": "..."}}`
- exit 0 always.

Knowledge base: {notebooks_dir}
"""


def review_prompt(
    req: Dict[str, Any],
    iter_dir: Path,
    notebooks_dir: Path,
    iteration: int,
    outcome: Optional[str] = None,
    failure: Optional[str] = None,
    perf: Optional[Dict[str, float]] = None,
    logs_dir: Optional[Path] = None,
) -> str:
    oc = f"C outcome: **{outcome}**" if outcome else "C outcome: (not provided)"
    fb = f"\nFailure:\n```\n{failure}\n```" if failure else ""
    pf = f"\nPerf: {json.dumps(perf)}" if perf else ""
    review_path = (logs_dir / "review.md") if logs_dir else iter_dir / "review.md"
    return f"""You are the **REVIEWER** for iteration #{iteration}.

# Task
{_render_req(req)}

# Test outcome
{oc}{fb}{pf}

# Working directory
{iter_dir}

Read the code, plan, and test output. Write `{review_path}`:
- **Verdict**: PASS / NEEDS_FIX (advisory)
- **Root cause** (if C failed): cite specific error
- **Issues**: list with file:line and fixes
- **Suggestions for next iteration**: prioritized
- **Confidence**: 1-5

Do NOT modify code. Review only.
"""


def c_repair_prompt(
    req: Dict[str, Any],
    iter_dir: Path,
    notebooks_dir: Path,
    iteration: int,
    attempt: int,
    max_attempts: int,
    failure: Optional[str],
    logs_dir: Optional[Path] = None,
) -> str:
    remaining = max(0, max_attempts - attempt)
    return f"""You are the **C-STEP DEBUGGER** for iteration #{iteration}, attempt {attempt}/{max_attempts} ({remaining} remaining).

C failed with:
```
{failure or "(no detail)"}
```

# Working directory
{iter_dir}

Fix ONE root cause with minimal edits. Verify locally. Stop after the fix.
Do NOT re-run the full test suite — the orchestrator does that.
"""


def c_repair_followup_prompt(
    iteration: int,
    attempt: int,
    max_attempts: int,
    new_failure: Optional[str],
    logs_dir: Path,
) -> str:
    remaining = max(0, max_attempts - attempt)
    return f"""C-step re-run still FAILED. Attempt {attempt}/{max_attempts} ({remaining} remaining).

New failure:
```
{new_failure or "(no detail)"}
```

You're in a resumed session. Fix ONE root cause. Minimal edits. Stop after the fix.
"""


def perf_test_prompt(
    req: Dict[str, Any],
    iter_dir: Path,
    notebooks_dir: Path,
    iteration: int,
    review_feedback: Optional[str] = None,
    logs_dir: Optional[Path] = None,
) -> str:
    return f"""You are the **PERF TESTER** for iteration #{iteration}.

The kernel PASSED correctness. Measure its performance.

# Working directory
{iter_dir}

# Deliverables
1. Write `{iter_dir}/perf.sh` — benchmarks the kernel with realistic shapes,
   warmup, multiple runs. Outputs `perf_report.json`.
2. Run it and produce:
   ```json
   {{"tokens_per_sec": <float>, "ms_per_op": <float>, "gflops": <float>,
     "bandwidth_gb_s": <float>, "notes": "..."}}
   ```
3. Print one JSON line to stdout: `{{"passed": true, "perf": {{...}}, "notes": "..."}}`

Do NOT modify the kernel. Benchmark only.
"""


def perf_plan_prompt(
    req: Dict[str, Any],
    iter_dir: Path,
    notebooks_dir: Path,
    iteration: int,
    last_perf: Optional[Dict[str, float]] = None,
    review_feedback: Optional[str] = None,
    logs_dir: Optional[Path] = None,
) -> str:
    lp = json.dumps(last_perf) if last_perf else "(read perf_report.json)"
    return f"""You are the **PERF PLANNER** for iteration #{iteration}.

# Working directory
{iter_dir}

# Last measured perf
{lp}

Read `{iter_dir}/perf_report.json` and write `{iter_dir}/perf_plan.md`:
1. Current perf baseline
2. Bottleneck analysis (kernel bound? memory bandwidth? launch overhead?)
3. Planned optimizations (prioritized, with expected impact + confidence)
4. What NOT to do

Do NOT modify code. Planning only.
"""


def retrospective_prompt(
    req: Dict[str, Any],
    iter_dir: Path,
    notebooks_dir: Path,
    iteration: int,
    this_perf: Optional[Dict[str, float]] = None,
    prev_perf: Optional[Dict[str, float]] = None,
    review_feedback: Optional[str] = None,
    e_ok: bool = True,
    e_error: Optional[str] = None,
    logs_dir: Optional[Path] = None,
    goal: Optional[str] = None,
) -> str:
    tp = json.dumps(this_perf) if this_perf else "(none)"
    pp = json.dumps(prev_perf) if prev_perf else "(none)"
    es = "E PASSED" if e_ok else f"E FAILED: {e_error or 'unknown'}"
    gl = goal or "(no goal)"
    return f"""You are the **RETROSPECTIVE WRITER** for iteration #{iteration}.

Write `{logs_dir}/retrospective.md`:
- **Goal**: {gl}
- **E status**: {es}
- **Perf this iter**: {tp}
- **Perf prev iter**: {pp}

Read `{iter_dir}/plan.md` and the code for context. Synthesize, don't dump.

Sections: Goal, What changed, Perf vs prev, Why perf moved, Review summary, Caveats.
Keep under 600 words. Do NOT modify code.
"""


def failure_retrospective_prompt(
    req: Dict[str, Any],
    iter_dir: Path,
    notebooks_dir: Path,
    iteration: int,
    failure_reason: Optional[str] = None,
    failed_phase: Optional[str] = None,
    phase_attempts: Optional[int] = None,
    logs_dir: Optional[Path] = None,
    goal: Optional[str] = None,
) -> str:
    fr = failure_reason or "(unknown)"
    fp = failed_phase or "(unknown)"
    gl = goal or "(no goal)"
    return f"""You are the **POSTMORTEM WRITER** for FAILED iteration #{iteration}.

Write `{logs_dir}/retrospective.md`:
- **Goal**: {gl}
- **Failed phase**: {fp}
- **Failure**: {fr}

Read the plan, code, and diagnostic logs. Sections: What was tried, Where it broke,
Recovery attempted, Why recovery didn't work, What next iter should do differently.
Keep under 600 words. Do NOT modify code.
"""
