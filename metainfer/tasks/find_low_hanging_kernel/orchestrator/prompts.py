"""Prompt templates for find-low-hanging-kernel sub-agents.

Every prompt that touches user-provided inputs (model dir, framework source,
trace file, startup log) starts with a READ-ONLY INPUTS block reminding the
agent never to modify those paths.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional


READONLY_BANNER = """# READ-ONLY INPUTS — DO NOT MODIFY
The following paths are user-provided inputs. You MUST NOT create, modify,
or delete any file under these paths. All writes you produce must land in
your assigned workdir. If you believe a file is wrong, note it in your
report — do not "fix" it.

"""


def _readonly_block(form: Dict[str, Any]) -> str:
    lines: List[str] = []
    for key, label in (
        ("model_dir", "Model directory"),
        ("framework_source_dir", "Framework source tree"),
        ("trace_file", "Chrome tracing file"),
        ("startup_log", "Startup log"),
    ):
        v = (form.get(key) or "").strip()
        if v:
            lines.append(f"- {label}: {v}")
    if not lines:
        return ""
    return READONLY_BANNER + "\n".join(lines) + "\n\n"


def _render_form_summary(form: Dict[str, Any]) -> str:
    lines: List[str] = []
    for k, v in form.items():
        if v in (None, ""):
            continue
        lines.append(f"- {k}: {v}")
    return "\n".join(lines) if lines else "(no form fields populated)"


# --------------------------------------------------------------------------- #
# Step 1 — three cross-validation agents
# --------------------------------------------------------------------------- #


_STEP1_ROLE_INSTRUCTIONS = {
    "arch_tracer": (
        "You are the **architecture tracer**. Starting from config.json in the "
        "model directory, identify the model architecture (e.g. Llama-style "
        "decoder transformer, MoE variant, etc.). Then read the inference "
        "framework source to locate every operator call site in the forward "
        "pass — input embedding, attention, MLP/RMSNorm/LayerNorm, RoPE, "
        "final logits, kv-cache operations, sampling. For each call site "
        "capture file:line:symbol and the kernel name it ultimately invokes "
        "(if you can tell from the framework's dispatch / autotuner)."
    ),
    "quant_tracer": (
        "You are the **quantization tracer**. Identify how the model is "
        "quantized from config.json (quant_method, bits, group_size, "
        "symmetric/asymmetric, etc.). Then locate the framework source code "
        "that LOADS these quantized weights: the weight loader, the "
        "dequantize / W8A16 / W4A16 kernel selection logic, the scales & "
        "zeros tensors. Capture file:line:symbol for each. Flag any weight-"
        "loading fallback paths that could activate if the primary path "
        "isn't built."
    ),
    "runtime_tracer": (
        "You are the **runtime-resolution tracer**. Your job is to confirm "
        "WHICH code paths actually executed at launch time, given the CLI "
        "args + env vars + startup log. Hunt for fallback impls: framework "
        "X often has CUDA / Triton / eager / fallback variants gated on "
        "env vars or compile-time flags — record which variant ran, with "
        "evidence from the startup log or env vars. Note explicitly any "
        "case where the active config bypasses the 'main' implementation."
    ),
}


def step1_agent_prompt(
    *,
    role: str,
    form: Dict[str, Any],
    workdir: Path,
    inputs_snapshot_dir: Path,
) -> str:
    instructions = _STEP1_ROLE_INSTRUCTIONS.get(role)
    if instructions is None:
        raise ValueError(f"unknown step-1 role {role!r}")
    snapshot_hint = (
        f"- Inputs snapshot (copies of small user inputs): {inputs_snapshot_dir}\n"
        if inputs_snapshot_dir else ""
    )
    return f"""{_readonly_block(form)}# Your role
{instructions}

# Inputs available to you
- Your workdir (write your report here as `report.md`): {workdir}
{snapshot_hint}
- Form fields provided at launch time:
{_render_form_summary(form)}

# Output
Write your findings to `{workdir / "report.md"}`. Structure:

## Findings
- (your role-specific findings, with file:line:symbol references where possible)

## Evidence
- (quote the exact lines from the framework source or config.json that back
  each finding; include file paths)

## Confidence: (high | medium | low)
## Open questions / uncertainties
- (things you couldn't pin down — the synthesizer agent will weigh these)
"""


def step1_synthesis_prompt(
    *,
    form: Dict[str, Any],
    reports: Dict[str, Path],
    out_path: Path,
) -> str:
    """Cross-validate the three independent reports into one memory file."""
    reports_block = "\n\n".join(
        f"## Report from `{role}` ({path})\n"
        f"(read this file from disk — do not paste it here)"
        for role, path in reports.items()
    )
    return f"""{_readonly_block(form)}# Your role
You are the **synthesizer** for Step 1. Three independent analysis agents each
produced a report on the inference framework + model you're investigating,
each from a different angle: architecture, quantization, and runtime-resolution.

Read all three reports from disk and cross-validate them against each other and
against the framework source. Your job is to produce ONE consolidated, trusted
memory file that downstream steps (Step 2 source-mapping, Step 3 graph build,
Step 3 graph validation) will rely on.

# Reports to read
{reports_block}

# Required output
Write your synthesis to `{out_path}` with exactly these sections, in order:

## Model architecture
(consolidated view; include hidden_size, num_layers, num_attention_heads,
num_kv_heads, intermediate_size, vocab_size, max_seq_len, etc. — anything
relevant to tensor shapes later)

## Operator call sites
A markdown table with columns: `Step | Operator | Kernel name | Source location | Notes`.
One row per distinct operator call site in the forward pass. Source location
must be `file:line (symbol)`.

## Quantization loading
How weights are loaded, dequantized, and dispatched. Include file:line refs.

## Runtime-resolved code paths
Which implementation variants actually run under the user's CLI/env config.
Cite evidence (env var values, startup-log lines, framework feature flags).

## Disagreements & confidence
Where the three agents disagreed. For each disagreement: which view did you
pick and why? Which findings are HIGH confidence vs MEDIUM vs LOW?

## Open uncertainties
Unresolved questions to flag for downstream steps.
"""


# --------------------------------------------------------------------------- #
# Step 2 — deterministic parse + three cross-validation agents
# --------------------------------------------------------------------------- #


_STEP2_ROLE_INSTRUCTIONS = {
    "stat_analyst": (
        "You are the **statistics analyst**. Working ONLY from "
        "`trace_parsed.json`, rank the kernels by total time. Flag every "
        "kernel with high std/mean (bimodal_suspect=true) — these are "
        "candidates for 'one kernel name, multiple call sites with different "
        "shapes'. For each flagged kernel, also note the histogram shape."
    ),
    "source_mapper": (
        "You are the **source mapper**. For each kernel name in the trace, "
        "look up the corresponding source-level operator call sites in the "
        "Step 1 memory file. When a kernel name appears at MULTIPLE call "
        "sites, enumerate every one. When you cannot confidently map a "
        "kernel name, say so explicitly rather than guessing."
    ),
    "tp_shape_analyst": (
        "You are the **TP / shape analyst**. For each kernel call site, "
        "derive the ACTUAL tensor shapes that hit the GPU, given the CLI "
        "args (especially TP rank + world size). Pay special attention to: "
        "(a) weights that participate in TP sharding (column-parallel / "
        "row-parallel), (b) batch and seq-len dims (these are dynamic — "
        "use symbolic names like B, M). Flag any case where the same kernel "
        "name is invoked at different shapes across call sites."
    ),
}


def step2_agent_prompt(
    *,
    role: str,
    form: Dict[str, Any],
    workdir: Path,
    trace_parsed_path: Path,
    step1_memory_path: Path,
) -> str:
    instructions = _STEP2_ROLE_INSTRUCTIONS.get(role)
    if instructions is None:
        raise ValueError(f"unknown step-2 role {role!r}")
    return f"""{_readonly_block(form)}# Your role
{instructions}

# Inputs available to you
- Your workdir (write your report here as `report.md`): {workdir}
- Deterministic trace parse output (READ THIS, not the raw trace): {trace_parsed_path}
- Step 1 consolidated memory: {step1_memory_path}
- Framework source dir (read-only): {form.get("framework_source_dir")!s}
- Model dir (read-only): {form.get("model_dir")!s}
- Form fields:
{_render_form_summary(form)}

# Output
Write your findings to `{workdir / "report.md"}`. Be exhaustive about evidence
(file:line refs, exact numbers from trace_parsed.json). End with a
`## Confidence` section.
"""


def step2_synthesis_prompt(
    *,
    form: Dict[str, Any],
    reports: Dict[str, Path],
    out_path: Path,
) -> str:
    reports_block = "\n\n".join(
        f"## Report from `{role}` ({path})"
        for role, path in reports.items()
    )
    return f"""{_readonly_block(form)}# Your role
You are the **synthesizer** for Step 2. Three independent analysts examined
the trace_parsed.json from different angles: statistics, source mapping, and
TP-aware shape derivation. Cross-validate their reports against the trace
data and the Step 1 memory.

# Reports to read
{reports_block}

# Required output → write to `{out_path}`

## Kernel statistics ranking
Markdown table sorted by total_us descending. Columns:
`Rank | Kernel name | Cat | Count | Mean us | Std us | Total us | P99 us | Bimodal?`

## Source mapping table
Markdown table. Columns:
`Kernel name | Call site (file:line symbol) | Shape on this call site | Mean us at this call site`

## Disambiguated multi-call-site kernels
For each kernel name that appears at >1 call site (or with >1 distinct shape),
list the call sites and the per-call-site stats. This is the critical output
that prevents Step 3 from conflating kernels of the same name.

## CUDA graph + CPU-stack notes
What the trace_parsed.json flags say, and what that means for confidence in
per-call-site attribution. (CUDA Graph capture often collapses per-launch
attribution; CPU stack presence enables much better attribution.)

## Open uncertainties
"""


# --------------------------------------------------------------------------- #
# Step 3a — graph build
# --------------------------------------------------------------------------- #


def step3_build_prompt(
    *,
    form: Dict[str, Any],
    workdir: Path,
    step1_memory_path: Path,
    step2_memory_path: Path,
    out_graph_path: Path,
) -> str:
    return f"""{_readonly_block(form)}# Your role
You are the **execution-flow graph builder**. Read the Step 1 and Step 2
memory files and emit a single JSON file representing the inference pass as a
directed acyclic graph of computational steps. This is conceptually what an
AI compiler does — but you're doing it by reading source + logs + trace.

# Inputs available to you
- Your workdir: {workdir}
- Step 1 consolidated memory: {step1_memory_path}
- Step 2 consolidated memory: {step2_memory_path}
- Framework source dir (read-only): {form.get("framework_source_dir")!s}
- Model dir (read-only): {form.get("model_dir")!s}
- Form fields:
{_render_form_summary(form)}

# Output → write to `{out_graph_path}`
A single JSON object with EXACTLY this shape (see schema below). Set
`schema_version: 1`. Use one node per distinct (operator, call_site,
shape_signature) — Step 2's disambiguation work tells you when the same kernel
name needs to be split into multiple nodes.

```json
{{
  "schema_version": 1,
  "metadata": {{
    "task_id": "...",
    "model": "...",
    "tp_size": <int>,
    "vars": {{"B": "batch", "M": "seq_len", "...": "..."}}
  }},
  "nodes": [
    {{
      "id": "n01",
      "role": "<short human-readable purpose, e.g. 'RMSNorm'>",
      "operator": "<kernel name as seen in trace_parsed.json>",
      "source_ref": {{"file": "<path>", "line": <int>, "symbol": "<name>"}},
      "inputs":  [{{"name": "x", "dtype": "fp16", "shape": ["B", "M", 4096]}}],
      "outputs": [{{"name": "y", "dtype": "fp16", "shape": ["B", "M", 4096]}}],
      "stats": {{
        "count": <int>, "mean_us": <float>, "std_us": <float>,
        "total_us": <float>, "p99_us": <float>
      }},
      "confidence": "high"
    }}
  ],
  "edges": [{{"from": "n01", "to": "n02", "label": "y"}}]
}}
```

# Rules (very important)
1. **Shape variables**: declare every dynamic dim in `metadata.vars` (keys are
   short uppercase names like `B`, `M`, `H`, `D`; values are human descriptions).
   Reference them consistently everywhere — never use a literal where a
   variable is meant.
2. **TP sharding**: when a weight is TP-sharded, the per-rank shape becomes
   e.g. `[out_features/tp_size, in_features]`. Reflect this in the node's
   input/output tensors.
3. **Entry/exit nodes**: tag the first node of the forward pass with
   `role: "entry"` and the last with `role: "exit"` so the integrity check
   doesn't flag them as isolated.
4. **Disambiguated kernels**: if Step 2's "Disambiguated multi-call-site
   kernels" section lists the same kernel name at multiple call sites with
   different shapes, emit SEPARATE nodes (one per call site).
5. **Stats**: copy mean_us / count / std_us / total_us / p99_us verbatim from
   the Step 2 source-mapping table for THIS call site. Don't blend across
   call sites.
6. **Confidence**: set to `high` only when Step 1 + Step 2 + your source read
   all agree. Use `medium` / `low` otherwise.
7. Output ONLY the JSON object — no commentary, no markdown fence. The next
   step parses this file directly.
"""


# --------------------------------------------------------------------------- #
# Generic synthesis follow-up (re-uses step1/step2 prompts when needed)
# --------------------------------------------------------------------------- #


def failure_postmortem_prompt(
    *, phase: str, error: str, out_path: Path
) -> str:
    return f"""# Postmortem

The find-low-hanging-kernel pipeline failed in phase `{phase}`.

Write a concise postmortem to `{out_path}` covering:
- What the pipeline was doing when it failed
- The error message (cleaned up if it contains paths)
- One or two hypotheses for the root cause
- Suggested next steps for a re-run

Error:
```
{error[:2000]}
```
"""
