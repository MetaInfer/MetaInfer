"""Prompt strings for the calc-value orchestrator's agents.

Every prompt repeats the **non-negotiable** constraint::

    DO NOT modify, write to, or rename any file under the user's
    ``model_dir`` or ``framework_source_dir``. Those trees are READ-ONLY.
    All output goes into your workdir as text/JSON only.

The prompts are intentionally long and repetitive on this point because
LLMs are known to "helpfully" tidy up user files (re-save configs,
normalize whitespace) if not firmly warned off.
"""

from __future__ import annotations

# --------------------------------------------------------------------------- #
# Universal preamble
# --------------------------------------------------------------------------- #

READONLY_WARNING = """\
CRITICAL CONSTRAINT — read carefully before doing anything:

1. You are operating in a sandboxed workdir. The user has granted you
   READ-ONLY access to two external directories:

     MODEL_DIR  = {model_dir}
     FRAMEWORK  = {framework_dir}

2. You MUST NOT modify, create, rename, move, or delete any file under
   MODEL_DIR or FRAMEWORK. These trees are sacred:

   * No "fixing typos" in config.json.
   * No "normalizing whitespace" in framework source.
   * No "saving a parsed copy" back into the model dir.
   * No writing artifacts into either tree.

3. The orchestrator (a Python script, not an LLM) will verify this by
   checking modification times on every file under MODEL_DIR and
   FRAMEWORK after your run. Any touched file → task aborts.

4. ALL of your output goes in your workdir as JSON or text files. Read
   from the external trees, write only to your workdir.

If you are tempted to "just edit one config" — STOP. Write a note in
your output describing what you WOULD change, but do not change it.

Acknowledge this constraint in your first sentence: "I will not modify
any file under MODEL_DIR or FRAMEWORK."
"""


# --------------------------------------------------------------------------- #
# Step 1 — code analysis from 2 angles
# --------------------------------------------------------------------------- #

STEP1_OUTPUT_SCHEMA = """\
Write your findings as a JSON object with EXACTLY this top-level shape
(add more detail in nested fields if useful, but keep the top-level keys
stable so the orchestrator can merge your output with the other agents'):

{
  "architecture_summary": {
    "architecture": "<transformers class name from config.json>",
    "num_layers": <int>,
    "hidden_size": <int>,
    "num_attention_heads": <int>,
    "num_key_value_heads": <int>,    // 0 if not GQA
    "intermediate_size": <int>,
    "vocab_size": <int>,
    "context_length": <int>,
    "quantization": "<fp16 | bf16 | fp8 | gptq | awq | gguf | other-spec>",
    "evidence": {
      "config_file": "<path relative to MODEL_DIR>",
      "config_lines": "<line range like '1-40'>",
      "notes": "<free-form short justification>"
    }
  },
  "framework_entry_points": [
    // Where does the framework's model code LIVE (file:line)? List 1-5
    // key files/classes/functions that define the model + forward pass.
    {
      "purpose": "<short: e.g. 'Model class definition'>",
      "file": "<path relative to FRAMEWORK>",
      "lines": "<line range>",
      "symbol": "<class or function name>"
    }
  ],
  "operator_calls": [
    // The KEY outputs: every meaningful operator the forward pass
    // invokes. Each entry MUST say which code path is taken under the
    // user's actual cmdline + env (not the default fallback path).
    {
      "node_id_hint": "<short id: embedding | layer_N_attention | ...>",
      "purpose": "<what this op does>",
      "op": "<primary op: e.g. nn.Embedding / scaled_dot_product_attention / matmul / rmsnorm / ...>",
      "source_ref": {"file": "<relative to FRAMEWORK>", "lines": "<range>"},
      "active_under": "<what makes THIS code path active given cmdline/env; e.g. 'default' | 'TP>1' | 'VLLM_ATTENTION_BACKEND=FLASH_ATTN' | ...>",
      "fallback_avoided": "<if a fallback exists in source but is NOT taken under user's flags, name it; else null>",
      "evidence": "<one-sentence why you're sure this is the actual runtime path>"
    }
  ],
  "quantization_load": {
    "approach": "<none | gptq | awq | fp8 | int4 | ...>",
    "loader_file": "<path relative to FRAMEWORK>",
    "loader_lines": "<range>",
    "applies_to": "<which weights / modules>",
    "evidence": "<short justification>"
  },
  "tp_behavior": {
    "summary": "<how weights get split across tensor-parallel ranks, or 'not applicable' if TP=1>",
    "split_weights": ["<weight name pattern, e.g. 'q_proj'>", "..."],
    "nonsplit_weights": ["<weight name pattern, e.g. 'embedding'>", "..."],
    "evidence": "<file:line where the TP split happens>"
  },
  "uncertainties": [
    // Anything you are NOT sure about. The orchestrator will compare
    // your output with 2 other agents' outputs; honest uncertainties
    // help the merge resolve disagreements.
    "<short statement>"
  ]
}

Write this JSON object to a file named ``output.json`` in your workdir
using the Write tool. The orchestrator reads ONLY that file — your
natural-language response text is archived as ``response.txt`` for human
review (a short summary of what you found is welcome there) but is NOT
parsed for the structured data.

If you cannot determine a field, use `null` for objects/strings, `0` for
ints. Never invent numbers — say null and add an entry to
`uncertainties`.
"""


STEP1_AGENT_A_PROMPT = """\
You are analyzing a deep-learning model and its inference framework to
identify the exact code paths used during a forward pass.

STRATEGY: TOP-DOWN — start from config.json.

Steps:
1. Read MODEL_DIR/config.json. Extract architecture, layer count,
   hidden size, head counts, intermediate size, vocab, context length,
   and any quantization config field.
2. From the architecture name (e.g. LlamaForCausalLM), find the
   matching model class in FRAMEWORK (look for a model registry, a
   `@supports_*` decorator, an `_MODELS` dict, or a file under
   `framework_dir/model_executor/models/`).
3. Read the model's `forward()` method. Enumerate each operator call
   in order: embedding lookup, attention, MLP, norms, unembedding.
4. For each operator, identify the FILE and LINE where it is invoked,
   and the FILE and LINE of the underlying kernel/compute call.
5. Cross-check quantization: if config.json has a `quantization_config`
   block, find the matching loader in FRAMEWORK.
6. Identify TP-relevant weights: search for `--tensor-parallel-size` /
   `RowParallelLinear` / `ColumnParallelLinear` / `split_weight` in the
   framework.

Be precise about line numbers. Cite the specific lines you read.

{readonly}

CLI ARGS the user gave (these affect which code paths are active):
{cmdline}

ENV VARS the user gave (these affect which code paths are active):
{env_block}

{output_schema}
"""


STEP1_AGENT_B_PROMPT = """\
You are analyzing a deep-learning model and its inference framework to
identify the exact code paths used during a forward pass.

STRATEGY: BOTTOM-UP — start from the CLI flags + env vars the user
provided, and trace WHICH code branches they activate.

Steps:
1. From the user's CLI args, identify: tensor-parallel size, pipeline
   parallel size, attention backend selection, dtype, quantization
   flags, max-model-len, seed, etc.
2. From the env vars, identify: attention backend env overrides,
   CUDA_VISIBLE_DEVICES (affects TP routing), NCCL settings, any
   framework-specific feature flags.
3. For each flag/env var that has a code-level effect, find the
   `if/else` branch or the dispatcher that selects the code path. Cite
   file:line.
4. CRITICAL — there are often FALLBACK paths in framework code
   ("if has_flash_attn: ... else: slow_attention"). For each operator,
   verify which branch is taken under the user's actual flags/env.
   The "default" branch is RARELY what actually runs.
5. List every operator actually invoked at runtime, with the
   file:line of the invocation AND the file:line of the active
   implementation.

Pay special attention to:
  * Attention backend selection (FLASH_ATTN / XFORMERS / SPLIT_DECODE / ...)
  * Quantization loader dispatch (gptq vs awq vs fp8 vs marlin vs ...)
  * TP-aware vs TP-unaware weights
  * Feature flags that gate kernel selection

{readonly}

CLI ARGS:
{cmdline}

ENV VARS:
{env_block}

{output_schema}
"""


STEP1_AGENT_C_PROMPT = """\
You are analyzing a deep-learning model and its inference framework to
identify the exact code paths used during a forward pass.

STRATEGY: WEIGHT-DRIVEN — start from how the model's weight files are
loaded and trace backwards into the model code.

Steps:
1. List the weight files in MODEL_DIR (`.safetensors`, `.bin`, etc.).
   Read their index file (e.g. `model.safetensors.index.json`) to
   enumerate weight names + shapes.
2. Find where the framework LOADS these weights: search for
   `safetensors_open`, `load_safetensors`, `torch.load`,
   `initialize_weights`, `hf_model_weights_iterator`, etc. in FRAMEWORK.
   Cite the file:line.
3. For each weight-load call, find what tensor-parallel / quantization
   transform happens DURING load (e.g. `qkv_weight` may be split by
   `num_attention_heads`, gptq weights may go through `pack()`).
4. From the weight names, identify which operators consume them
   (e.g. `.q_proj.weight` → attention query projection; `.mlp.gate.weight`
   → SwiGLU gate; `.embed_tokens.weight` → embedding lookup).
5. For each quantized weight set, identify the dequantize-on-the-fly
   vs pre-dequantize code path actually used under the user's flags.

This strategy is the strongest signal for QUANTIZED models, where the
load path often differs from the unquantized reference implementation.

{readonly}

CLI ARGS:
{cmdline}

ENV VARS:
{env_block}

{output_schema}
"""


STEP1_DISAGREEMENT_PROMPT = """\
You previously analyzed a model + framework from a specific angle.
Other agents analyzing from different angles produced DIFFERENT findings
on some critical fields. Re-examine those fields and either defend your
answer or correct it.

Disputed fields (with the other agents' values):

{disputes}

Your previous output (for reference):

{your_prev}

Strategy:
1. Re-read the relevant files at the cited line numbers.
2. For each disputed field, decide: was your prior answer correct
   (then defend with evidence), or were the others correct (then
   update your answer with their evidence)?
3. Output the SAME JSON schema as before, fully filled in (not just
   the disputed fields). The orchestrator will re-merge.

{readonly}

CLI ARGS:
{cmdline}

ENV VARS:
{env_block}

{output_schema}
"""


# --------------------------------------------------------------------------- #
# Step 0 — rough single-pass estimate
# --------------------------------------------------------------------------- #

STEP0_ROUGH_PROMPT = """\
You are doing a FAST rough-pass estimate of an LLM's theoretical FLOPs
and HBM traffic per forward pass. Goal: produce SOMETHING reasonable
within minutes that the user can look at while the detailed audit runs.
Do NOT get bogged down in edge cases or precise kernel accounting.

STRATEGY: CONFIG-DRIVEN — read MODEL_DIR/config.json, extract the
headline constants, derive back-of-envelope formulas for each major
operator type, and emit one simplified calc.py per node.

Steps:
1. Read MODEL_DIR/config.json. Extract: architecture, num_layers (N),
   hidden_size (H), num_attention_heads, num_key_value_heads (0 if not
   GQA), head_dim (= H / num_attention_heads), intermediate_size (I),
   vocab_size (V), context_length, quantization config (bits, group_size).
   For DeepSeek-style MoE: also extract n_routed_experts (E),
   num_experts_per_tok (K), moe_intermediate_size, routed_scaling_factor.
2. Identify the operator nodes a single forward pass invokes. Use a
   STANDARD DECODER-ONLY LLM SKELETON — you do NOT need to read the
   framework's forward() in detail. The standard nodes per layer:
     - input_layernorm / rmsnorm
     - attention.q_proj, k_proj, v_proj (or QKV fused)
     - attention.rope (rotary embedding)
     - attention.score = Q @ K^T
     - attention.context = score @ V
     - attention.o_proj
     - post_attention_layernorm / rmsnorm
     - mlp.gate_proj, mlp.up_proj (or gate_up fused)
     - mlp.activation (silu/swiglu)
     - mlp.down_proj
   For MoE layers (DeepSeek-V2+): replace the 3 MLP linears with:
     - moe.router_gate (linear over hidden_size to n_experts)
     - moe.routed_experts (K selected experts, each gate/up/down)
     - moe.shared_expert (if config has a shared_expert)
   Plus the embedding lookup + final norm + lm_head.
3. For EACH node, write a SIMPLIFIED calc.py that derives FLOPs and HBM
   bytes from the config constants + (batch_size, seq_len). Return TWO
   phases per call — prefill (process all S tokens) and decode (1 new
   token with S cached). Use the STANDARD textbook formulas:
     - Linear [in,out] @ [out]:
         prefill FLOPs = 2*B*S*in*out; bytes = weight + 2*B*S*in*dtype + 2*B*S*out*dtype
         decode  FLOPs = 2*B*1*in*out; bytes = weight + 2*B*1*in*dtype + 2*B*1*out*dtype
       (weight bytes are the same in both — read once per call regardless
       of token count)
     - RMSNorm:
         prefill FLOPs ~ 5*B*S*H; bytes ~ 3*B*S*H*dtype
         decode  FLOPs ~ 5*B*1*H; bytes ~ 3*B*1*H*dtype
     - RoPE:
         prefill FLOPs ~ 6*B*S*H; bytes ~ 2*B*S*H*dtype
         decode  FLOPs ~ 6*B*1*H; bytes ~ 2*B*1*H*dtype
     - Attention:
         prefill score = 2*B*Hh*Hd*S*S; context = 2*B*Hh*Hd*S*S
         decode  score = 2*B*Hh*Hd*1*S; context = 2*B*Hh*Hd*1*S
         decode  bytes MUST include reading all S cached K and V:
                 + 2 * B*Hh*Hd*S*dtype  (read K, read V)
                 + 2 * B*Hh*Hd*1*dtype  (write new K, write new V)
       (this KV-cache read is what makes decode memory-bound at large S)
     - MoE routed_experts: only K experts active per token,
         prefill FLOPs = 6*K*H*I*B*S; decode FLOPs = 6*K*H*I*B*1
         weight bytes = sum over ALL E experts (full table in HBM), same
         in both phases.
   DO NOT include fine-grained terms (e.g. scaling-factor multiply,
   weighted-combine scatter-add, dequantization overhead). Keep it
   crude — the detailed audit will refine.
4. Write each node's calc.py to its own file using the Write tool:
       {workdir}/per_node/<section>__<node_id>.py
   Where <section> is one of: input, layer (or moe_layer for MoE
   sections), output. Use the compound filename so two sections with
   same-named nodes don't collide.
   Each file MUST define `def calc(batch_size, seq_len) -> dict` returning
   {{"prefill": {{"tflops", "access_gb"}}, "decode": {{"tflops", "access_gb"}}}}
   (per the contract below). Decode covers generating 1 new token with
   `seq_len` tokens already in KV cache — include the KV cache read in
   decode bytes for attention nodes (this is the dominant decode cost).
5. Write a manifest JSON to {workdir}/rough_graph.json listing every
   node you produced. Schema:
   {{
     "sections": [
       {{"id": "input", "kind": "input", "repeat_count": 1,
         "graph": {{"nodes": [{{"id": "embedding", "op": "embedding", "compound": "input__embedding"}}]}}}},
       {{"id": "layer", "kind": "layer_template", "repeat_count": <N from config>,
         "graph": {{"nodes": [{{...}}, ...]}}}},
       {{"id": "output", "kind": "output", "repeat_count": 1, ...}}
     ],
     "config_summary": {{... a copy of the key config constants ...}},
     "notes": "rough pass; refined by detailed audit later"
   }}

CRITICAL — speed over precision. If you are unsure about an exact
formula, use the textbook version and add a comment noting the
uncertainty. The detailed audit (which runs after you) will refine the
numbers. Your goal is to get a defensible number on screen FAST.

{readonly}

CLI ARGS the user gave:
{cmdline}

ENV VARS the user gave:
{env_block}

{calc_contract}

Output your workdir as: {workdir}
Write per_node/*.py and rough_graph.json there.
"""


# --------------------------------------------------------------------------- #
# Step 2 — execution graph build + per-node validation
# --------------------------------------------------------------------------- #

STEP2_OUTPUT_SCHEMA = """\
Output a SECTIONED graph as a JSON object with EXACTLY this top-level shape:

{{
  "sections": [
    {{
      "id": "<stable snake_case id, unique within sections>",
      "kind": "input" | "layer_template" | "output",
      "description": "<short human-readable: what this section covers>",
      "applies_to": [<int>, ...],
      "repeat_count": <int>,
      "graph": {{
        "nodes": [<node>, ...],
        "edges": [{{"from": "<node_id>", "to": "<node_id>"}}]
      }}
    }},
    ...
  ],
  "inter_section_edges": [
    {{"from_section": "<id>", "to_section": "<id>"}}
  ]
}}

Per-node shape (inside each section's ``graph.nodes``):

{{
  "id": "<stable short id, snake_case, unique WITHIN its section>",
  "purpose": "<one-sentence: what this step does>",
  "op": "<primary operator, e.g. nn.Embedding | scaled_dot_product_attention
          | matmul | rmsnorm | softmax | rotary_emb | layernorm | ...>",
  "source_ref": {{
    "file": "<path relative to FRAMEWORK, or null if pure mathematical>",
    "lines": "<range like '120-145', or null>"
  }},
  "inputs": [
    {{"name": "token_ids", "shape": ["batch_size", "seq_len"], "dtype": "int64"}}
  ],
  "outputs": [
    {{"name": "hidden_states", "shape": ["batch_size", "seq_len", "hidden_size"],
     "dtype": "float16"}}
  ],
  "duration_ms_hint": 0.0,
  "notes": "<optional short string>"
}}

SECTION STRUCTURE — read carefully, this is the whole point of the redesign:

A real model's forward pass is NOT 1000+ unique operators. It is a few
STAGES that repeat (per-layer dense / per-layer MoE / per-layer shared
expert) sandwiched between non-repeating ends (embedding, sampling).
GROUP identical layers into ONE ``layer_template`` section; do NOT
enumerate every layer.

Canonical layout:

* ONE ``input`` section (id="embedding" or similar) — embedding lookup
  and any pre-layer preprocessing. NOT repeated.
* ONE OR MORE ``layer_template`` sections — one per DISTINCT layer
  structure. Each represents ONE occurrence of that layer; set
  ``repeat_count`` to the number of actual layers in the model that
  follow this structure, and ``applies_to`` to the list of layer
  indices. The downstream FLOPs/bytes aggregator multiplies the
  section's per-occurrence numbers by ``repeat_count``.
* ONE ``output`` section (id="sampling" or similar) — final norm,
  LM head / unembedding, any post-layer logits processing. NOT repeated.

Examples:

* Llama-style (all layers identical): 3 sections — input + 1
  layer_template (repeat_count=num_layers, applies_to=[0..num_layers-1])
  + output.
* DeepSeek-V3 (dense layers 0-2, MoE layers 3-60): 4 sections — input
  + dense_layer (repeat_count=3, applies_to=[0,1,2]) + moe_layer
  (repeat_count=58, applies_to=[3..60]) + output.
* Qwen3 MoE (similar dense/MoE split): same shape as DeepSeek-V3.
* GPT-style with a single MLP type: 3 sections like Llama.

If the model has only ONE layer type, you have exactly 3 sections. If
it has two distinct layer types, you have 4. And so on. Do NOT split a
single layer type into per-layer sections just to be safe — that
defeats the purpose.

``inter_section_edges`` describes the order sections execute in. A
linear model has N-1 edges for N sections; branching/parallel topologies
can have more.

CRITICAL rules for shapes (same as before):
* Use SYMBOLIC variable names for non-constant dims: ``batch_size``,
  ``seq_len``, ``hidden_size``, ``num_heads``, ``head_dim``,
  ``kv_seq_len``, ``intermediate_size``, ``vocab_size``, ``num_layers``.
  Use the SAME symbol everywhere it refers to the same quantity.
* For TP-split weights, write the per-rank shape and use a suffix like
  ``num_heads_per_rank`` or ``hidden_size_per_rank`` for the post-split
  dim. Add a note in ``notes`` explaining the split.
* For each operator whose shape changes between input and output, the
  shape math MUST be consistent across the input of the next node and
  the output of the previous one — within a section AND across section
  boundaries (the last node of an upstream section feeds the first node
  of the downstream section).

Cover at minimum (across all sections combined):
* embedding lookup (input section)
* per-layer attention: QKV projections, attention proper, output
  projection (in each layer_template section that has attention)
* per-layer MLP: gate/up/down projections, activation
* per-layer norms: pre/post attention, pre/post MLP
* final norm + LM head / unembedding (output section)
* any quantization-related dequantize steps that are explicit operators
  in the framework (not weight loading)

Write this JSON object to a file named ``graph.json`` in your workdir
using the Write tool. The orchestrator reads ONLY that file — your
natural-language response text is archived as ``response.txt`` for human
review (a short summary of what you built is welcome there) but is NOT
parsed for the structured data. Since the sectioned schema is compact
(a 60-layer model has ~10–20 nodes per template, not 1000+), you can
usually Write the JSON directly; but if helpful, write a Python
generator script (``build_graph.py``) and run it via Bash to emit
``graph.json``.
"""


STEP2_BUILD_PROMPT = """\
You are building an execution-flow graph for a single forward pass of an
LLM, similar to what an AI compiler would emit but at the operator-call
granularity.

Inputs at your disposal:
* MODEL_DIR  = {model_dir}     (read-only — see the constraint below)
* FRAMEWORK  = {framework_dir} (read-only)
* memory.json (a consensus memory produced by 3 prior analysis agents):

```json
{memory_json}
```

Strategy:
1. Read memory.json's `architecture_summary` for layer count, hidden size,
   head counts, intermediate size, vocabulary size, etc. Use these as
   your symbolic constants.
2. Read memory.json's `operator_calls` for the actual operator sequence
   (these were cross-validated by 2 agents).
3. Inspect the model's layer definitions in FRAMEWORK. Identify how many
   DISTINCT layer types exist. A "distinct" layer type is one whose
   operator sequence differs structurally — e.g. a dense MLP layer vs a
   MoE layer are distinct; two MoE layers with the same topology but
   different expert counts are also distinct. Layers that differ only
   in their constant values (same shape, same op sequence) are NOT
   distinct.
4. Build the sectioned graph:
   (a) ONE ``input`` section for embedding lookup + any pre-layer
       preprocessing.
   (b) ONE ``layer_template`` section per distinct layer type. Set
       ``repeat_count`` to the number of actual model layers that
       follow this structure, and ``applies_to`` to their indices.
       Each template's inner graph represents ONE occurrence of that
       layer.
   (c) ONE ``output`` section for the final norm + LM head + any
       post-layer logits processing.
   (d) ``inter_section_edges`` to express the execution order
       (typically: input → first layer template → ... → output).
5. For each operator inside each section, derive the input/output tensor
   shapes symbolically using ``batch_size`` and ``seq_len`` as the only
   free variables. Config constants (``hidden_size``, ``num_heads``,
   ``intermediate_size``, ``vocab_size``, ``num_layers``) come from
   memory.json's architecture_summary.
6. DO NOT enumerate every layer. A 60-layer model with 2 distinct layer
   types produces a graph with ~3 sections and ~20–40 nodes total —
   not 1000+ nodes.

Your output is graph.json (the sectioned schema). It will be:
* structurally validated (per-section orphan check, field-presence
  check, repeat_count/applies_to consistency, inter-section endpoints)
* per-node validated by separate agents that cross-check against source
* used downstream to compute per-node FLOPs and memory traffic. For
  ``layer_template`` sections the per-occurrence numbers are multiplied
  by ``repeat_count`` to get the section's contribution.

{readonly}

CLI ARGS (these affect active code paths):
{cmdline}

ENV VARS:
{env_block}

{output_schema}
"""


STEP2_VALIDATE_NODE_PROMPT = """\
You are validating ONE node of an execution-flow graph for an LLM
forward pass. Decide whether the node's description, operator, source
reference, and tensor shapes are consistent with the actual framework
source code and the model's config.

The node (in JSON):

```json
{node_json}
```

Its neighbors WITHIN the same section (so you can check shape consistency
across in-section edges; cross-section neighbors are the orchestrator's
job via ``inter_section_edges``):

```json
{neighbors_json}
```

The section this node belongs to (so you can reason about whether the
node's role makes sense — e.g. a node inside a ``layer_template``
section with ``repeat_count=58`` represents ONE of 58 identical layers;
its shapes should be per-occurrence, not aggregated):

```json
{section_json}
```

The consensus memory from earlier analysis (for reference):

```json
{memory_json}
```

What to check:
1. PURPOSE/OP: Does the framework source at the cited file:lines actually
   implement this operator? If the file/lines don't match the operator,
   verdict = "reject".
2. SHAPES: Are the input/output tensor shapes consistent across the
   edges (i.e. this node's input matches the upstream node's output)?
   If a symbol like `hidden_size` shows up with two different values in
   the same chain, verdict = "reject".
3. ACTIVE PATH: Is this the operator that actually runs under the
   user's cmdline + env (not a fallback path)? Cross-check against the
   memory's `operator_calls`.
4. COMPLETENESS: Are all required shape dims present (no `null` for a
   dim that should be numeric/symbolic)?

Output a JSON verdict:

{{
  "verdict": "pass" | "reject",
  "reason": "<one paragraph justification citing file:line>",
  "suggested_fix": "<if reject, what should change in the node's JSON; else null>"
}}

Write this JSON object to a file named ``verdict.json`` in your workdir
using the Write tool. The orchestrator reads ONLY that file — your
natural-language response text is archived as ``response.txt`` for human
review only and is NOT parsed.

{readonly}

CLI ARGS:
{cmdline}

ENV VARS:
{env_block}
"""


STEP2_FIX_PROMPT = """\
You previously produced an execution-flow graph. Several nodes failed
per-node validation (the validator agents said their descriptions or
shapes don't match the framework source).

Your current graph.json:

```json
{graph_json}
```

Failed-node verdicts:

{verdicts}

Your job: produce an UPDATED graph.json that fixes every flagged node.
Keep passing nodes unchanged. Preserve the same JSON shape and the same
node ids where possible (only change an id if the node itself was
mis-identified).

{readonly}

CLI ARGS:
{cmdline}

ENV VARS:
{env_block}

{output_schema}
"""


# --------------------------------------------------------------------------- #
# Step 3 — per-node FLOPs / mem-traffic calc scripts
# --------------------------------------------------------------------------- #

STEP3_CALC_FUNC_CONTRACT = """\
Your output MUST be a single Python file that defines a top-level function:

    def calc(batch_size: int, seq_len: int) -> dict:

The function must return a dict with EXACTLY two phase keys, each holding
a {{tflops, access_gb}} pair:

    {{
      "prefill": {{"tflops": <float>, "access_gb": <float>}},
      "decode":  {{"tflops": <float>, "access_gb": <float>}},
    }}

PHASE SEMANTICS:

* **prefill**: the model processes the full `seq_len` tokens in one forward
  pass (the prompt). All weights, all token activations, no KV cache reuse.

* **decode**: the model generates ONE new token given a KV cache that
  already contains `seq_len` prior tokens. Decode FLOPs cover ONLY the
  single new token's compute. Decode HBM bytes MUST include:
    - weight reads (always — same total weight bytes as prefill per call,
      independent of token count)
    - new token's input/output activations (B*1 vectors, not B*S)
    - READ of the S cached K and V tensors from HBM (this is the whole
      reason decode is memory-bound — at large S the KV cache read dwarfs
      everything else)
    - WRITE of the new K and V being appended to the cache

QUICK DECODE FORMULAS BY OP TYPE (H=hidden, I=intermediate, Hh=num heads,
Hd=head dim, S=seq_len, B=batch, E=n_experts, K=top_k):

* Linear / matmul (Q proj, K/V proj, O proj, MLP gate/up/down, lm_head):
    FLOPs  = 2 * B * 1 * in * out
    HBM    = weight_bytes + B*1*in*dtype + B*1*out*dtype
  (vs prefill: prefill replaces `1` with `S`)

* Attention score = Q @ K^T (decode):
    FLOPs  = 2 * B * Hh * Hd * 1 * S        # Q is [B,Hh,1,Hd], K is [B,Hh,S,Hd]
    HBM-Q  = B * Hh * 1 * Hd * dtype        # new Q
    HBM-K  = B * Hh * S * Hd * dtype        # READ all S cached K — this dominates

* Attention context = score @ V (decode):
    FLOPs  = 2 * B * Hh * 1 * S * Hd
    HBM-V  = B * Hh * S * Hd * dtype        # READ all S cached V

* Attention K/V proj output write:
    HBM-write = 2 * B * Hh * 1 * Hd * dtype # new K and V appended to cache

* RMSNorm / layernorm (decode):
    FLOPs ~ 5 * B * 1 * H
    HBM   ~ 3 * B * 1 * H * dtype (read x, read weight, write y)

* MoE routed experts (decode, top-K routing):
    FLOPs  = B * 1 * 6 * K * H * I          # only K experts per token
    HBM    = weight_bytes_of_selected_experts + B*1*H*dtype + B*1*I*dtype
  (vs prefill: prefill multiplies both FLOPs and activation bytes by S,
   but weight bytes are the same per call)

GENERAL RULES:
* `1 GB = 1024**3 bytes` (binary GB / GiB).
* `1 TFLOP = 1e12 FLOPs`.
* For matmul-shaped ops: FLOPs = 2 * (output_elements) when the matmul
  is fully connected, or 2 * M * N * K for shape (M, K) x (K, N).
* For a single source read: access = element_count * bytes_per_element.
  Typical bytes_per_element: fp16/bf16 = 2, fp32 = 4, int8 = 1, fp8 = 1,
  int4 = 0.5.
* The function MUST be deterministic and side-effect-free (no network,
  no file I/O, no random).
* Import from `math` if needed; do NOT import torch / numpy / external libs.
* The orchestrator will import your file as a module and call calc()
  at the canonical shape (batch_size=1, seq_len=512) to verify it.
  The WebUI may also call it at other user-chosen shapes on demand.
  It must NOT print anything.
* Numeric constants baked in from the model config MUST be hardcoded
  (read from your understanding of memory.json); only batch_size and
  seq_len come from the function arguments.

Example skeleton (dense MLP up-projection):

```python
def calc(batch_size: int, seq_len: int):
    H = 4096
    out_dim = 4 * H
    weight_bytes = H * out_dim * 2  # fp16 weight, read once per call

    # Prefill: process all S tokens.
    pre_flops = 2 * batch_size * seq_len * H * out_dim
    pre_bytes = (weight_bytes
                 + batch_size * seq_len * H * 2      # fp16 input
                 + batch_size * seq_len * out_dim * 2)  # fp16 output

    # Decode: 1 new token.
    dec_flops = 2 * batch_size * 1 * H * out_dim
    dec_bytes = (weight_bytes
                 + batch_size * 1 * H * 2
                 + batch_size * 1 * out_dim * 2)
    # NOTE: this is a linear op — no KV cache. Attention nodes have an
    # extra + batch_size*Hh*S*Hd*2*2 term in decode bytes (read K and V).

    return {{
        "prefill": {{"tflops": pre_flops / 1e12,
                     "access_gb": pre_bytes / (1024**3)}},
        "decode":  {{"tflops": dec_flops / 1e12,
                     "access_gb": dec_bytes / (1024**3)}},
    }}
```

Write your file as calc.py.
"""


STEP3_WRITER_PROMPTS = [
    # Writer 0: algorithmic / operator-semantic angle.
    """\
You are writing a Python function to compute the theoretical FLOPs and
global-memory traffic of ONE operator node in an LLM forward pass.

STRATEGY: ALGORITHMIC — derive the formula from the operator's semantics
(matrix multiply, attention, norm, etc.).

The node (from the validated execution-flow graph):

```json
{node_json}
```

The consensus memory (for config constants like hidden_size, num_heads,
etc.):

```json
{memory_json}
```

Reason about the operator's compute pattern:
* For GEMM-shaped ops (linear layers, embeddings-as-matmul): FLOPs = 2*M*N*K.
* For attention proper (pre-fill): FLOPs = 2*B*H*sq*sk*d + 2*B*H*sq*d*sk,
  mem = reads of Q,K,V (each B*H*sq*d*2 bytes) + output write.
* For norms: FLOPs ~ 5*elements (rmsnorm) or 7*elements (layernorm),
  mem = 2*reads+1*write of the activations.
* For softmax: FLOPs ~ 5*elements (subtract max, exp, sum, divide),
  mem = 2*read+1*write.

State your assumptions as comments in the file. Be precise.

{readonly}

{calc_contract}

Write your Python source to a file named ``calc.py`` in your workdir
using the Write tool. The orchestrator imports ONLY that file — your
natural-language response text is archived as ``response.txt`` for human
review only and is NOT executed. Reasoning comments belong inside
``calc.py`` as Python comments, not in the response text.
""",
    # Writer 1: source-driven / shape-inference angle.
    """\
You are writing a Python function to compute the theoretical FLOPs and
global-memory traffic of ONE operator node in an LLM forward pass.

STRATEGY: SOURCE-DRIVEN — start from the framework source at the cited
file:lines in the node, derive the actual tensor shapes that flow
through the operator, then compute FLOPs and bytes from those shapes.

The node:

```json
{node_json}
```

Consensus memory:

```json
{memory_json}
```

Steps:
1. Read the framework source at `source_ref.file` lines `source_ref.lines`.
2. Identify the input tensor shapes and the output tensor shape of THIS
   operator (use the node's `inputs`/`outputs` as a hint, but verify
   against source).
3. Compute:
   * FLOPs based on the operation (matmul, elementwise, reduction, etc.)
   * Bytes read = sum of input tensor volumes * dtype_bytes
   * Bytes written = sum of output tensor volumes * dtype_bytes
4. Bake the constants (hidden_size, num_heads, etc.) into the function
   as hardcoded literals from memory.json.

Important: this strategy tends to be MOST accurate for "boring" ops
(like linear layers and norms) where the framework source unambiguously
defines the shape math.

{readonly}

{calc_contract}

Write your Python source to a file named ``calc.py`` in your workdir
using the Write tool. The orchestrator imports ONLY that file; your
response text is archived as ``response.txt`` and is NOT executed.
""",
]


STEP3_FIX_PROMPT = """\
You previously wrote a calc() function for ONE operator node, but its
output disagreed with the other independent agent's calc() function on
the cartesian product of (batch_size, seq_len). Reconcile.

The node:

```json
{node_json}
```

Your previous calc.py:

```python
{your_script}
```

Mismatches (the orchestrator's deterministic checker ran both scripts
at the canonical shape and found these disagreements; "a0"=your output,
"a1" is the other agent's output):

{mismatches}

Steps:
1. Re-derive the FLOPs and access_gb formulas for this node.
2. If your prior answer was correct (the OTHER agent is wrong),
   KEEP your formula but make it more rigorous — the comparison is
   relative-tolerance based; rounding noise under 1e-6 should already
   pass.
3. If your prior answer was wrong, identify the bug (e.g. used fp32
   bytes for fp16 weights, mis-counted the matmul axes, forgot the
   output write, double-counted the input read) and FIX it.

CRITICAL: You are STILL working in isolation. Do NOT try to look up or
guess what the other agent wrote. The mismatches above show their
NUMBERS only, not their formulas. Make your formula correct on its own
merits.

{readonly}

{calc_contract}

Write your updated Python source to ``calc.py`` in your workdir via the
Write tool. The orchestrator imports ONLY that file.
"""


STEP3_VIZ_BUILDER_PROMPT = """\
You are generating an interactive HTML visualization of an LLM forward
pass's theoretical FLOPs and memory-traffic breakdown.

The task_id is "{task_id}". The compute endpoint is:

  {compute_url}?batch_size=<b>&seq_len=<s>

CRITICAL: In your HTML JavaScript, embed the task_id and compute base
URL as HARDCODED constants — do NOT try to parse them from
window.location or URL query parameters. The page is served inside an
<iframe> at a path-based URL (no ?task_id= in the query string).
Use this pattern at the top of your <script>:

  // Fall back to server-injected globals if you forgot to hardcode.
  const TASK_ID = "{task_id}" || window.METAINFER_TASK_ID;
  const COMPUTE_URL = "{compute_url}" || window.METAINFER_COMPUTE_URL;

Also listen for postMessage to detect the API base when embedded in
the WebUI (copy this block exactly):

  let API_BASE = "";
  window.addEventListener("message", (ev) => {{
    if (ev.data && ev.data.metainfer_api_base) {{
      API_BASE = ev.data.metainfer_api_base;
    }}
  }});

Then construct the fetch URL as:
  API_BASE + COMPUTE_URL + "?batch_size=" + b + "&seq_len=" + s

or if API_BASE is empty (standalone testing):
  COMPUTE_URL + "?batch_size=" + b + "&seq_len=" + s

The execution-flow graph is SECTIONED (validated):

```json
{graph_json}
```

Schema reminder: the top-level object has ``sections`` and
``inter_section_edges``. Each section has ``id``, ``kind`` (``input`` /
``layer_template`` / ``output``), ``repeat_count`` (N for
``layer_template``, 1 for ``input`` / ``output``), ``applies_to`` (the
list of layer indices a template represents), and an inner ``graph``
with ``nodes`` / ``edges`` representing ONE occurrence.

The final calc scripts are in this directory (one per compound
``<section_id>__<node_id>.py``, each exposes
calc(batch_size, seq_len) -> {{"prefill": {{"tflops", "access_gb"}},
"decode": {{"tflops", "access_gb"}}}} — these are PER-OCCURRENCE
numbers; multiply by section.repeat_count for the section's total
contribution):

  {calc_dir}

Produce a single self-contained HTML file with these requirements:

1. Render the graph as STACKED SECTION CARDS, one per section. Each
   card has:
   * A header: section id, kind badge (input/layer_template/output),
     and a "× N" badge for ``layer_template`` sections with
     ``repeat_count > 1``.
   * Inside the card: the section's nodes as SVG boxes (or a table).
     Each node box has the node id, op name, a category color
     (attention=red, mlp=orange, norm=green, quant=cyan,
     embedding=blue, other=gray), and per-instance TFLOPs / GB badges
     that get FILLED IN at runtime via JS.
   * A per-section subtotal line showing the section's total TFLOPs
     and GB (= per-instance × repeat_count).
2. Between section cards, draw an arrow indicating execution order
   (driven by ``inter_section_edges``).
3. Two input controls at the top: batch_size and seq_len (both default
   1), and a "Recalculate" button. Each per-node badge and the totals
   bar must show BOTH prefill and decode numbers side by side (e.g. two
   TFLOPs columns: "prefill.tf" and "decode.tf").
4. The Recalculate button fetches (using the TASK_ID/COMPUTE_URL/API_BASE
   constants declared above):
     COMPUTE_URL + "?batch_size=" + b + "&seq_len=" + s
   (prefixed with API_BASE when available). The response is JSON keyed
   by compound id with both phases
   (``per_compound["<section_id>__<node_id>"] = {{"prefill": {{"tflops", "access_gb"}},
   "decode": {{"tflops", "access_gb"}}}}``).
   JS fills in the per-instance badges AND multiplies by repeat_count
   for the section subtotals and the grand totals.
5. A totals bar at the bottom: show TWO sets of totals — one for
   prefill, one for decode. Each set: sum of TFLOPs (per-instance ×
   repeat_count across all sections), sum of GB, and arithmetic
   intensity (TFLOPs / GB). Decode arithmetic intensity should be
   visibly lower than prefill (decode is memory-bound).
6. Style: dark background (#0e1117), light text (#e6edf3), monospace
   font for numbers, compact (12-14px base). Match the MetaInfer WebUI
   palette.
7. NO external resources — no CDN, no Google Fonts. Inline <style> and
   <script> only.
8. Single file.

Write your HTML to a file named ``viz.html`` in your workdir via the
Write tool. The orchestrator reads ONLY that file; your response text
is archived as ``response.txt`` for human review only and is NOT
parsed. The HTML will be served via an <iframe> from the WebUI, so
make it self-contained.
"""
