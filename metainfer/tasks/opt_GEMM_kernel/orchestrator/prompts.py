"""Prompts for the six-phase GEMM optimization loop.

The evaluator is intentionally absent from these prompts. Agents receive the
contract and sanitized feedback, never the system-owned command bundle or
held-out case definitions.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional


BOUNDARY = """## Trust boundary

- You are a candidate generator, not the judge.
- Never create or modify correctness, benchmark, score, or pass/fail scripts.
- Never search for the system evaluator or held-out shape definitions.
- Treat files outside the current iteration and the supplied notebooks as out of scope.
- Public feedback is evidence; a self-authored speedup claim is not evidence.
"""


def _requirements(req: Dict[str, Any]) -> str:
    public = dict(req)
    public.pop("evaluator_bundle", None)
    public.pop("weight_bundle", None)
    return json.dumps(public, indent=2, ensure_ascii=False)


def plan_prompt(
    req: Dict[str, Any],
    iter_dir: Path,
    notebooks_dir: Path,
    iteration: int,
    champion: Dict[str, Any],
    prior_feedback: Optional[Dict[str, Any]],
) -> str:
    feedback = json.dumps(prior_feedback or {}, indent=2, ensure_ascii=False)
    return f"""You are planning iteration {iteration} of a GEMM kernel optimization.

Working directory: {iter_dir}
Read-only knowledge base: {notebooks_dir}
Current champion: {json.dumps(champion, ensure_ascii=False)}

Frozen user requirements (the private evaluator path is deliberately omitted):
```json
{_requirements(req)}
```

Previous sanitized evaluator feedback:
```json
{feedback}
```

Inspect the current `submission/`, the frozen public shapes, and the latest
per-shape evidence before reading optimization notes. Treat notebooks as
historical evidence, known constraints, and candidate hypotheses—not as an
exhaustive search space, a required dispatch recipe, or a substitute for
analyzing the current source and hardware. Notebook absolute timings are not
cross-machine service-level targets. If `perf_plan.md` exists, evaluate its
hypothesis against the current Champion, operator latency, dispatch breakdown,
and PMC/resource evidence rather than applying it mechanically.

Write the new `plan.md` in the iteration directory. Select one evidence-backed,
bounded, measurable, reversible change; novelty is not a goal. State the exact
affected shape IDs, numerical latency expectation, source-level mechanism,
expected counter/resource movement, named control shapes expected to remain
unchanged, risks, and rollback rule. Valid directions include reducing summed GPU operator
time, reducing dispatch/reduction work, improving HBM or L2 efficiency,
reducing VGPR/AGPR/SGPR/LDS/scratch pressure, or improving parallelism when the
available evidence supports it. Do not edit `submission/` in this phase.

{BOUNDARY}
"""


def implement_prompt(
    req: Dict[str, Any],
    submission_dir: Path,
    notebooks_dir: Path,
    iteration: int,
) -> str:
    return f"""You are the implementer for GEMM optimization iteration {iteration}.

Your working directory is the only writable deliverable:
{submission_dir}

The iteration plan is at `../plan.md`. The knowledge base is at:
{notebooks_dir}

Implement the plan by editing files under the current `submission/` directory
only. Preserve the submission entry-point contract described in the notebooks.
The system owns CMake, nvcc/hipcc, GPU architecture flags, the build command,
and every profiler command/counter group. Do not create or replace
CMakeLists.txt/build.sh, invoke a different compiler, or launch a profiler.
Keep `submission.yaml` valid: it may list relative source
and include paths and request only BuildProfile-allowlisted options. Do not add
test or benchmark harnesses intended to replace the system evaluator. Finish
with a short `CHANGELOG.md` inside submission describing the exact change.

Public requirements:
```json
{_requirements(req)}
```

{BOUNDARY}
"""


def review_prompt(
    iter_dir: Path,
    notebooks_dir: Path,
    iteration: int,
    feedback: Dict[str, Any],
) -> str:
    return f"""You are the D-phase correctness reviewer for GEMM optimization iteration {iteration}.

Iteration directory: {iter_dir}
Knowledge base: {notebooks_dir}

Sanitized system evidence:
```json
{json.dumps(feedback, indent=2, ensure_ascii=False)}
```

Write `review.md` in the iteration directory. Review the build and correctness
evidence only. Explain a compile/correctness failure precisely, or record why
the candidate is safe to advance to E_perf_test. Do not claim a speedup before
E runs, and do not edit `submission/`.

{BOUNDARY}
"""


def perf_plan_prompt(
    iter_dir: Path,
    notebooks_dir: Path,
    iteration: int,
    feedback: Dict[str, Any],
    promotion: Dict[str, Any],
) -> str:
    return f"""You are the F-phase performance planner for GEMM optimization iteration {iteration}.

Iteration directory: {iter_dir}
Knowledge base: {notebooks_dir}
The D-phase correctness review is at `review.md`.

Sanitized system performance evidence:
```json
{json.dumps(feedback, indent=2, ensure_ascii=False)}
```

Champion decision:
```json
{json.dumps(promotion, indent=2, ensure_ascii=False)}
```

Write `perf_plan.md` in the iteration directory. Start from the current source
and every public shape's hipprof GPU operator time. For multi-dispatch calls,
interpret total operator latency and the per-kernel contribution breakdown;
do not treat the longest kernel or PMC replay duration as latency. Correlate
regressions with HBM read/write traffic and bandwidth, L2 behavior,
VGPR/AGPR/SGPR, LDS, scratch, dispatch count, and occupancy/wave information
only when the profiler actually reports it.

Identify every failed shape and use reported dispersion/CV and any boundary
retest to distinguish noise from a plausible bottleneck. Rank up to three
evidence-backed hypotheses, then recommend one bounded, measurable, reversible
next optimization. Novelty is not a goal: prefer the strongest measured
evidence even when the direction is already documented. Include numerical
latency expectations for affected shapes, named control shapes expected to stay
unchanged, expected counter/resource changes, and an explicit rollback rule. Do not edit
`submission/`; the next A/B phases execute the new plan.

{BOUNDARY}
"""


def repair_prompt(
    req: Dict[str, Any], submission_dir: Path, iteration: int,
    feedback: Dict[str, Any],
) -> str:
    return f"""You are making the single allowed build/correctness repair for GEMM iteration {iteration}.

Writable submission directory: {submission_dir}
Sanitized compiler/correctness evidence:
```json
{json.dumps(feedback, indent=2, ensure_ascii=False)}
```

Make only the smallest source change needed to address the evidenced failure.
Do not introduce a new optimization, broaden the original plan, change the ABI,
or modify any evaluator/profiler file. Update `CHANGELOG.md` with the repair.
If the evidence is insufficient, leave source unchanged and explain why there.

Public requirements:
```json
{_requirements(req)}
```

{BOUNDARY}
"""


def with_human_guidance(prompt: str, items: List[Dict[str, Any]]) -> str:
    """Append live user steering while reasserting the evaluator boundary."""
    if not items:
        return prompt
    rendered = "\n\n".join(
        f"### Guidance {index}\n{str(item.get('text') or '').strip()}"
        for index, item in enumerate(items, 1)
    )
    return f"""{prompt}

## Live optimization guidance

The task owner submitted the following optimization ideas while the task was
running. Treat them as high-priority hypotheses within the public GEMM
contract, not as instructions to bypass independent analysis. Inspect the
current code, target shapes, hipprof operator timing, dispatch breakdown, PMC
evidence, and hardware constraints before applying them. If an idea is unsafe,
incompatible with the ABI, or contradicted by measurements, explain that in
the plan/changelog instead of silently forcing it.

{rendered}

Optimization guidance may influence candidate generation, but it may never change or
bypass compilation, correctness, benchmark, scoring, or champion gates.

{BOUNDARY}
"""
