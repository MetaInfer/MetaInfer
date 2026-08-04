"""LLM prompts for sglang_trace_analyze.

Two prompt families:
1. **mapping_sanity_check** — validate the kernel-to-model-structure mapping.
2. **optimization_hints** — generate actionable optimization suggestions from
   the full analysis (kernel tables + overlap + fuse results).
"""

from __future__ import annotations


def mapping_sanity_check_prompt(
    mapping_json: str,
    model_config_json: str,
    gpu_label: str,
) -> str:
    """Prompt for LLM to sanity-check a kernel → model-structure mapping."""
    return f"""You are a GPU inference optimization expert. Review the following
kernel-to-model-structure mapping that was auto-generated from a torch
profiler trace's call stacks.

## Model config.json
```json
{model_config_json}
```

## Auto-generated mapping (excerpt — full file too large, this is the first
200 entries sorted by GPU time)
```json
{mapping_json}
```

## GPU
{gpu_label}

## Tasks
1. For each mapping entry, rate its confidence: "high" (call stack clearly
   points to a known layer/op), "medium" (plausible but ambiguous), or
   "low" (likely wrong — kernel name and call stack don't match expected
   pattern). If you're uncertain about a model architecture detail, search
   the web for the model's architecture documentation before rating.
2. Flag any kernel that appears to be mapped to the wrong layer type
   (e.g. a MoE kernel mapped to a dense layer, or an attention kernel
   mapped to an FFN layer).
3. Flag missing mappings — kernel names that appear in the trace but have
   no clear model-layer assignment.
4. Return a JSON object with this schema:
   {{
     "entries": [
       {{
         "kernel_name": "...",
         "confidence": "high|medium|low",
         "issues": ["..."]  // empty list if none
       }}
     ],
     "summary": {{
       "high_count": N,
       "medium_count": N,
       "low_count": N,
       "overall_assessment": "..."
     }}
   }}
"""


def optimization_hints_prompt(
    kernel_tables_summary: str,
    overlap_summary: str,
    fuse_summary: str,
    gpu_label: str,
    model_name: str,
) -> str:
    """Prompt for LLM to generate optimization hints from analysis results."""
    return f"""You are a GPU inference optimization expert. Review the profiling
analysis below and generate actionable optimization suggestions.

## Model
{model_name}

## GPU
{gpu_label}

## Kernel Hotspot Summary (top kernels by GPU time across all batch sizes)
{kernel_tables_summary}

## Overlap Analysis
{overlap_summary}

## Fuse Pattern Matches
{fuse_summary}

## Tasks
1. Identify the single biggest bottleneck and explain why it dominates.
2. List 3-5 concrete optimization directions, ordered by estimated impact.
   For each: what to change, why it helps, and estimated saving (%).
3. Note any surprising or counter-intuitive findings (e.g. a kernel that
   should be fast but is unexpectedly slow).
4. Return a JSON object with this schema:
   {{
     "bottleneck": {{
       "kernel_or_pattern": "...",
       "reason": "...",
       "impact_pct": N
     }},
     "suggestions": [
       {{
         "title": "...",
         "what_to_change": "...",
         "why": "...",
         "estimated_saving_pct": N,
         "difficulty": "low|medium|high",
         "category": "fuse|overlap|kernel_replace|config_tune|other"
       }}
     ],
     "surprises": ["..."]
   }}
"""
