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

Read the contract and evaluation-protocol notebooks, then inspect the current
`submission/`. If `perf_plan.md` exists, it is the previous iteration's
F-phase recommendation: evaluate it against the current champion and public
feedback instead of ignoring it. Write the new `plan.md` in the iteration
directory. Choose one bounded, testable change, state the expected affected
shapes, risks, and rollback rule. Do not edit `submission/` in this phase.

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

Write `perf_plan.md` in the iteration directory. Identify the shapes that
improved or regressed, distinguish measurement noise from a plausible kernel
bottleneck, use the system-provided hardware counters when present, and
propose one bounded next optimization. Do not edit
`submission/`; the next A/B phases execute the new plan.

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
contract. Inspect the current code and evidence before applying them. If an
idea is unsafe, incompatible with the ABI, or contradicted by measurements,
explain that in the plan/changelog instead of silently forcing it.

{rendered}

Optimization guidance may influence candidate generation, but it may never change or
bypass compilation, correctness, benchmark, scoring, or champion gates.

{BOUNDARY}
"""
