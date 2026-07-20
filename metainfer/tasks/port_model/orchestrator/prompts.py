"""Prompt templates for the port-model orchestrator.

Each major phase gets its own builder function. Every prompt begins with a
READ-ONLY banner listing directories the agent must never write to.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict

from metainfer.orchestrator.requirements import req_field, req_summary_lines

# ---------------------------------------------------------------------------
# Shared banners
# ---------------------------------------------------------------------------

READONLY_BANNER = """\
# ⛔ READ-ONLY INPUTS

You MUST NOT create, modify, or delete any file under these directories:

  - MODEL_DIR = {model_dir}
  - SOURCE_FRAMEWORK_DIR = {source_framework_dir}

{extra_readonly}

All your writes MUST land in your own workdir: {workdir}

The *only* exception to read-only is TARGET_FRAMEWORK_DIR = {target_framework_dir}
which you ARE allowed to modify (and only that directory — not model_dir, not
source_framework_dir).
"""

P4_WRITABLE_BANNER = """\
# ⛔ READ-ONLY INPUTS (same as before)

You MUST NOT create, modify, or delete any file under:

  - MODEL_DIR = {model_dir}
  - SOURCE_FRAMEWORK_DIR = {source_framework_dir}

All memory / analysis writes go to your workdir: {workdir}

# ✅ WRITABLE

You ARE expected to modify files under TARGET_FRAMEWORK_DIR:
  - TARGET_FRAMEWORK_DIR = {target_framework_dir}
"""


# ---------------------------------------------------------------------------
# Step 1: Model architecture analysis
# ---------------------------------------------------------------------------

def p1_model_analysis_prompt(
    *,
    form: Dict,
    workdir: Path,
) -> str:
    model_dir = req_field(form, "model_dir") or ""
    source_fw = req_field(form, "source_framework_dir") or ""
    target_fw = req_field(form, "target_framework_dir") or ""

    return READONLY_BANNER.format(
        model_dir=model_dir,
        source_framework_dir=source_fw,
        target_framework_dir=target_fw,
        extra_readonly="",
        workdir=workdir,
    ) + f"""

# Task: Analyze the model architecture

Your ONLY job in this phase is to read config files and produce a structured
analysis document. DO NOT write any code. DO NOT modify any framework files.

## Steps

1. Read the model config at `{model_dir}/config.json`. Extract:
   - ``architectures`` / ``model_type``
   - Hidden size, intermediate size, num layers, num attention heads, num KV heads
   - Activation function, norm type (LayerNorm / RMSNorm)
   - Position encoding (RoPE / ALiBi / learned)
   - Vocab size, max position embeddings
   - Quantization config if present (bits, group_size, desc_act, sym)

2. List the weight files under `{model_dir}/`. Note the format (safetensors / pytorch / npz).
   Check if there's an index file and note which layers map to which files.

3. Identify special architectural features:
   - **MoE** (Mixture of Experts): shared experts, routing, top-k
   - **VLM** (Vision-Language Model): vision encoder, projector, image processor
   - **SSM** (Mamba / RWKV / linear attention)
   - **Sliding window / sparse attention**
   - **QKV bias** (some models have bias on QKV projections)
   - **Tied embeddings** (input embedding == output head weight)
   - **GQA** (Grouped Query Attention): num_kv_heads vs num_attention_heads
   - **MLA** (Multi-head Latent Attention, e.g. DeepSeek-V3)

4. Derive per-tensor shapes. For each block in the forward pass, list the
   input/output tensor shapes. Use symbolic dims (B=batch, M=seq_len, H=hidden,
   N=num_heads, V=vocab, …). Also list the TP-sharded dims (e.g. hidden_size
   divided by tp_size for column-parallel linear).

## Output

Write your analysis to `{workdir}/p1_model_analysis.md`. Use this structure:

```markdown
# Model Architecture: <model_name>

## Summary
- Model type: <...>
- Architecture family: <...>
- Parameter count: <...> B

## Architecture Details
(nested sections: Layers, Attention, Normalization, Position Encoding, …)

## Special Features
(MoE / VLM / SSM / … — each with a dedicated subsection)

## Quantization
(dtype, group_size, format, scales/zeros layout — if applicable)

## Weight Layout
(how safetensors files are organized, per-layer grouping)

## Shape Table
| Step | Input shapes | Output shapes | TP-sharded? |
|---|---|---|---|
| Embedding | [B, M] | [B, M, H] | No |
| ... | ... | ... | ... |

## Tokenizer
(vocab size, special tokens, chat template if applicable)

## VLM Details (if applicable)
(vision encoder arch, projector type, image_size, patch_size, …)
```
"""


# ---------------------------------------------------------------------------
# Step 2: Source framework analysis
# ---------------------------------------------------------------------------

def p2_source_analysis_prompt(
    *,
    form: Dict,
    workdir: Path,
    p1_path: Path,
) -> str:
    model_dir = req_field(form, "model_dir") or ""
    source_fw = req_field(form, "source_framework_dir") or ""
    target_fw = req_field(form, "target_framework_dir") or ""
    model_name = Path(model_dir).name

    return READONLY_BANNER.format(
        model_dir=model_dir,
        source_framework_dir=source_fw,
        target_framework_dir=target_fw,
        extra_readonly="",
        workdir=workdir,
    ) + f"""

# Task: Analyze the source framework's model registration

The **source framework** at `{source_fw}` already supports model `{model_name}`.
Your job: understand HOW it's registered so we can replicate the pattern in the
target framework.

## Pre-reading

Before analysis, read the Step-1 model architecture document at:
  `{p1_path}`

This tells you what the model looks like — now find the corresponding code.

## Steps

1. **Find the model entry point.** Search `{source_fw}` for files/classes/functions
   that reference the model name, architecture type, or config class. Common entry
   points:
   - vLLM: ``vllm/model_executor/models/<name>.py`` — a class inheriting
     ``SupportsPP`` / ``SupportsLoRA`` / etc.
   - SGLang: ``python/sglang/srt/models/<name>.py``
   - Transformers: ``transformers/src/transformers/models/<name>/modeling_<name>.py``

2. **Trace the full registration chain:**
   - Model class → what it inherits from
   - Custom layers (FusedMLP, custom attention, custom norm, custom MoE)
   - Weight loading: ``load_weights()`` or equivalent — trace the mapping from
     safetensors keys to layer parameters
   - Forward call chain: ``forward()`` → attention → MLP → norm → residual

3. **List every custom op/layer** the model uses that goes beyond standard
   PyTorch nn.Module. For each: file path, class name, purpose, key implementation
   details.

4. **VLM handling** (if applicable): how the vision encoder is integrated, image
   preprocessing pipeline, projector/connector between vision and text towers.

5. **Configuration:** how the model config is parsed — HuggingFace
   ``AutoConfig.from_pretrained()`` → ``<Model>Config``, or a custom config
   class.

## Output

Write your analysis to `{workdir}/p2_source_analysis.md`. Structure:

```markdown
# Source Framework Analysis

## Model Entry Point
(file, class, inheritance chain)

## Registration Chain
(step-by-step: config → model class → layers → forward)

## Custom Ops / Layers
| File | Class | Purpose | Key details |
|---|---|---|---|
| ... | ... | ... | ... |

## Weight Loading
(mapping from safetensors keys to layer parameter names, special cases)

## Forward Pass
(call chain: embedding → [layer_n: attention → MLP → norm] → head)

## VLM Integration (if applicable)
(vision encoder, projector, image preprocessing)

## Config Parsing
(how the framework reads model_dir/config.json)
```
"""


# ---------------------------------------------------------------------------
# Step 3: Target framework analysis
# ---------------------------------------------------------------------------

def p3_target_analysis_prompt(
    *,
    form: Dict,
    workdir: Path,
    p1_path: Path,
    p2_path: Path,
) -> str:
    model_dir = req_field(form, "model_dir") or ""
    source_fw = req_field(form, "source_framework_dir") or ""
    target_fw = req_field(form, "target_framework_dir") or ""
    fw_type = req_field(form, "target_framework_type") or "unknown"

    return READONLY_BANNER.format(
        model_dir=model_dir,
        source_framework_dir=source_fw,
        target_framework_dir=target_fw,
        extra_readonly="",
        workdir=workdir,
    ) + f"""

# Task: Analyze the target framework and plan the port

The **target framework** at `{target_fw}` is `{fw_type}`. You need to figure out
how to add support for our model to it. This phase is analysis + planning only —
do NOT write any framework code yet.

## Pre-reading

1. Read the Step-1 model architecture: `{p1_path}`
2. Read the Step-2 source framework analysis: `{p2_path}`

## Steps

1. **Find a template model.** Search `{target_fw}` for an existing model that is
   architecturally similar (same attention type, similar layer structure). This
   will be your primary reference/template. Document:
   - Template model name + file path
   - Why it's a good match

2. **Understand the registration pattern.** For the template model, trace:
   - Where and how the model class/function is registered (registry dict /
     decorator / ``__init__.py`` / ``MODEL_REGISTRY``)
   - What base class or interface it must implement
   - Required methods: ``load_model()``, ``get_attention()``, ``forward()``, etc.

3. **Map model features to framework features.** For each non-trivial aspect of
   our model (from P1), determine what framework mechanism handles it:
   - Quantized weights → does the framework have a quantized linear layer?
   - MoE → does the framework have MoE pipeline support?
   - VLM → how does the framework handle multimodal inputs? (vision tower,
     ``multi_modal_data``, image processor pipeline)
   - Custom ops → does the framework allow custom ops, or must everything use
     framework-provided layers?
   - GQA / MLA → how do existing models handle non-standard attention?

4. **List every file that needs to be created or modified.** Be precise — full
   relative paths under `{target_fw}`.

5. **Identify risks and unknowns.** What could go wrong? What framework internals
   might not be obvious from reading the source?

## Output

Write your analysis to `{workdir}/p3_target_analysis.md`. Structure:

```markdown
# Target Framework Analysis ({fw_type})

## Template Model
(model_name, file_path, why it's a good match)

## Registration Pattern
(how models are registered, what base class to use, required methods)

## Feature Mapping
| Model feature (from P1) | Framework mechanism | Notes |
|---|---|---|
| Quantization | ... | ... |
| MoE | ... | ... |
| ... | ... | ... |

## Files to Create
(path / description)

## Files to Modify
(path / description / what changes)

## Risks & Unknowns
(what might fail, what needs investigation during implementation)
```
"""


# ---------------------------------------------------------------------------
# Step 4: Implementation
# ---------------------------------------------------------------------------

def p4_implement_prompt(
    *,
    form: Dict,
    workdir: Path,
    p1_path: Path,
    p2_path: Path,
    p3_path: Path,
    prev_test_failure: str | None = None,
) -> str:
    model_dir = req_field(form, "model_dir") or ""
    source_fw = req_field(form, "source_framework_dir") or ""
    target_fw = req_field(form, "target_framework_dir") or ""
    hw = req_field(form, "target_hardware") or "unknown"

    repair_section = ""
    if prev_test_failure:
        repair_section = f"""\
# ⚠️ REPAIR MODE

The previous implementation produced incorrect results. Here is the test
failure report. Fix the issues described below:

```
{prev_test_failure}
```

Read the full test results at `{workdir}/../test/test_results.json` for details.
"""

    return P4_WRITABLE_BANNER.format(
        model_dir=model_dir,
        source_framework_dir=source_fw,
        target_framework_dir=target_fw,
        workdir=workdir,
    ) + f"""

# Task: Implement model support in the target framework

Add support for this model to the target framework at `{target_fw}`.

## Pre-reading (REQUIRED)

1. Model architecture: `{p1_path}`
2. Source framework reference: `{p2_path}`
3. Target framework plan: `{p3_path}`

Read ALL THREE before writing any code.

## Requirements

Target hardware: **{hw}**

### What to implement

Based on P3's analysis, create/modify files in `{target_fw}` to add model
support. Follow the framework's existing conventions EXACTLY — do not invent
new patterns.

### Must include:

1. **Model registration file** — the file that tells the framework about this
   model. Follow the template model's pattern.

2. **Custom layers** (if needed) — any layer that doesn't exist in the framework
   already. Reuse framework layers where possible; only write custom code when
   the model has genuinely unique ops.

3. **Weight loading** — the mapping from safetensors keys to layer parameters.
   Must handle:
   - Standard Linear / Embedding / LayerNorm / RMSNorm weights
   - QKV fused weights (if the model packs them)
   - MoE expert weights (if applicable)
   - Quantized weights (if applicable)
   - Vision encoder weights (if VLM)

4. **Forward pass** — the model's forward method, respecting the framework's
   conventions (e.g. vLLM's ``forward()`` must handle ``AttnMetadata``).

5. **VLM support** (if applicable) — register a ``MultiModalProcessor`` or
   equivalent, wire the vision encoder + projector, handle image preprocessing.

### Must NOT:

- Modify any file outside `{target_fw}`
- Touch `{model_dir}` or `{source_fw}`
- Introduce new pip dependencies
- Remove or break existing model support in the framework
- Copy-paste large code blocks from the source framework (learn from it, don't
  steal from it — write original code that respects the target's architecture)

### Self-check (run BEFORE declaring done):

1. **Import check**: can Python import the new module without errors?
   ```
   cd {target_fw} && python -c "from <new_module> import <ModelClass>; print('import ok')"
   ```

2. **Config parse**: can the framework load the model config from `{model_dir}`?

3. **Weight keys**: print the expected safetensors keys from your weight loader
   and compare against actual files in `{model_dir}/`. Every key must map.

## Output

1. Modify/create files in `{target_fw}`.
2. Write a summary of changes to `{workdir}/p4_changes.md`.
3. Generate a patch file:
   ```
   cd {target_fw} && git diff > {workdir}/../diff/model_port.patch
   ```
{repair_section}
"""
