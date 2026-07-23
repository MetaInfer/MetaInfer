# Qwen3 Compute Graph Notes for a C++ Inference Framework

This note summarizes `src/models/qwen3.cpp` in llama.cpp as an implementation reference for building a small C++ inference framework.

The important answer first: `src/models/qwen3.cpp` is the Qwen3 forward compute-graph builder. It describes the per-token/per-layer math graph. It is not the tokenizer, model loader, runtime scheduler, sampler, or backend kernel implementation.

## Where This Fits

Runtime call chain:

```text
example / server
  -> llama_decode(ctx, batch)
  -> llama_context::decode()
  -> llama_context::process_ubatch()
  -> llama_model::build_graph()
  -> llm_build_qwen3 constructor in src/models/qwen3.cpp
  -> ggml graph execution by backend scheduler
```

Key files:

- `src/models/qwen3.cpp`: Qwen3 forward graph.
- `src/llama-model.cpp`: architecture dispatch, hparam reading, tensor allocation and tensor names.
- `src/llama-graph.cpp`: reusable graph helpers for RMSNorm, LoRA matmul, FFN, attention, inputs.
- `src/llama-context.cpp`: decode runtime, batching, KV cache preparation, graph execution.
- `src/llama-kv-cache.cpp`: KV cache allocation, slot finding, KV read/write.

## Architecture Summary

Qwen3 in this file is a decoder-only Transformer with:

- token embedding input
- repeated decoder blocks
- pre-attention RMSNorm
- Q, K, V projections
- per-head Q RMSNorm and K RMSNorm before RoPE
- RoPE on Q and K
- grouped-query attention if `n_head_kv < n_head`
- attention output projection
- residual connection
- pre-FFN RMSNorm
- SwiGLU-style FFN: `down(silu(gate(x)) * up(x))`
- residual connection
- final RMSNorm
- output projection / lm head

This Qwen3 path is non-MoE. Qwen3-MoE has a separate graph builder in `src/models/qwen3moe.cpp`.

## Required Hparams

The graph uses these hparams through `llm_graph_context`:

| Name | Meaning |
| --- | --- |
| `n_embd` | hidden size |
| `n_layer` | number of decoder blocks |
| `n_head` | number of query heads |
| `n_head_kv` | number of key/value heads |
| `n_embd_head_k` | per-head Q/K dimension |
| `n_embd_head_v` | per-head V dimension |
| `n_rot` | RoPE dimension |
| `n_ff` | FFN intermediate size |
| `f_norm_rms_eps` | RMSNorm epsilon |
| `freq_base`, `freq_scale` | RoPE frequency parameters |
| `n_ctx_orig`, `rope_type` | RoPE/YARN related context parameters |

Qwen3 asserts:

```cpp
n_embd_head_v == n_embd_head_k
n_embd_head_k == n_rot
```

So a minimal framework can assume:

```text
head_dim = n_embd_head_k = n_embd_head_v = n_rot
q_dim    = n_head    * head_dim
kv_dim   = n_head_kv * head_dim
```

## Tensor Names and Shapes

Tensor names for Qwen3 are registered under `LLM_ARCH_QWEN3` in `src/llama-arch.cpp`.

llama.cpp tensor shapes below follow the source allocation order. The effective matmul direction is handled by `ggml_mul_mat(weight, x)`.

Global tensors:

| Tensor | Name in GGUF | Shape in llama.cpp allocation | Required |
| --- | --- | --- | --- |
| token embedding | `token_embd.weight` | `{n_embd, n_vocab}` | yes |
| output norm | `output_norm.weight` | `{n_embd}` | yes |
| lm head | `output.weight` | `{n_embd, n_vocab}` | optional, can tie to token embedding |
| rerank/classifier head | `cls.output.weight` | `{n_embd, n_cls_out}` | optional |

上表的 `output.weight optional` 是通用 llama.cpp 架构兼容行为，不是当前固定模型的 Loader 契约。当前 Qwen3-8B 的 `tie_word_embeddings=false`，所以 `formats/gguf/qwen3_loader.md` 必须要求独立 `output.weight`；缺失时不能回退到 token embedding。

Per-layer tensors, for layer `i`:

| Tensor | Name in GGUF | Shape in llama.cpp allocation | Required |
| --- | --- | --- | --- |
| attention input norm | `blk.%d.attn_norm.weight` | `{n_embd}` | yes |
| Q projection | `blk.%d.attn_q.weight` | `{n_embd, n_head * head_dim}` | yes |
| K projection | `blk.%d.attn_k.weight` | `{n_embd, n_head_kv * head_dim}` | yes |
| V projection | `blk.%d.attn_v.weight` | `{n_embd, n_head_kv * head_dim}` | yes |
| attention output projection | `blk.%d.attn_output.weight` | `{n_head * head_dim, n_embd}` | yes |
| per-head Q norm | `blk.%d.attn_q_norm.weight` | `{head_dim}` | yes |
| per-head K norm | `blk.%d.attn_k_norm.weight` | `{head_dim}` | yes |
| FFN input norm | `blk.%d.ffn_norm.weight` | `{n_embd}` | yes |
| FFN gate projection | `blk.%d.ffn_gate.weight` | `{n_embd, n_ff}` | yes |
| FFN up projection | `blk.%d.ffn_up.weight` | `{n_embd, n_ff}` | yes |
| FFN down projection | `blk.%d.ffn_down.weight` | `{n_ff, n_embd}` | yes |

Unlike some LLaMA paths, this Qwen3 graph does not use Q/K/V/output biases.

## Forward Graph

### Inputs

The graph builder creates these logical inputs:

- token ids or precomputed embeddings via `build_inp_embd()`
- positions via `build_inp_pos()`
- KV attention inputs via `build_attn_inp_kv()`
- output token row ids via `build_inp_out_ids()`

`build_attn_inp_kv()` creates:

- K cache write indices
- V cache write indices
- attention mask over the KV cache

These are graph inputs because they change every decode step.

### Embedding

```text
inpL = token_embedding[token_ids]
```

If direct embeddings are supplied instead of token ids, `inpL` is the supplied embedding matrix.

### One Decoder Layer

For each layer `il`:

```text
inpSA = inpL

cur = RMSNorm(inpL, attn_norm[il])

Q = matmul(wq[il], cur)
K = matmul(wk[il], cur)
V = matmul(wv[il], cur)

Q = reshape(Q, [head_dim, n_head,    n_tokens])
K = reshape(K, [head_dim, n_head_kv, n_tokens])
V = reshape(V, [head_dim, n_head_kv, n_tokens])

Q = RMSNorm(Q, attn_q_norm[il])
K = RMSNorm(K, attn_k_norm[il])

Q = RoPE(Q, positions)
K = RoPE(K, positions)

attn_out = AttentionWithKVCache(Q, K, V)
cur = matmul(wo[il], attn_out)

ffn_inp = cur + inpSA

cur = RMSNorm(ffn_inp, ffn_norm[il])
up   = matmul(ffn_up[il],   cur)
gate = matmul(ffn_gate[il], cur)
cur  = matmul(ffn_down[il], silu(gate) * up)

inpL = cur + ffn_inp
```

`AttentionWithKVCache` expands to:

```text
write K into KV cache
write V into KV cache
read K cache view
read V cache view
apply causal/KV mask
softmax(QK^T / sqrt(head_dim))
multiply by V
project by wo
```

In llama.cpp, writing K/V is part of the ggml graph through `cpy_k()` and `cpy_v()`, implemented with row-indexed set operations. A small standalone framework can implement this as explicit KV cache writes before calling an attention kernel.

### Last Layer Output Selection

On the final layer only:

```text
if inp_out_ids exists:
    cur   = get_rows(cur,   inp_out_ids)
    inpSA = get_rows(inpSA, inp_out_ids)
```

This optimization computes logits only for requested output tokens, commonly just the last token during decoding.

### Final Output

```text
hidden = RMSNorm(inpL, output_norm)
logits = matmul(output, hidden)
```

`hidden` is stored as `res->t_embd`, and logits as `res->t_logits`.

## Attention Details

Qwen3 uses:

```text
kq_scale = 1 / sqrt(head_dim)
```

It does not read `hparams.f_attention_scale` in this graph path.

Q and K normalization happens before RoPE:

```text
Q = q_norm(Q)
Q = rope(Q)

K = k_norm(K)
K = rope(K)
```

This is an important Qwen3 difference from the simpler LLaMA graph. If you omit `attn_q_norm` / `attn_k_norm`, outputs will be wrong.

If `n_head_kv < n_head`, use GQA:

```text
query heads: n_head
kv heads:    n_head_kv
group size:  n_head / n_head_kv
```

Each group of query heads attends to the corresponding KV head.

## KV Cache Expectations

The Qwen3 graph assumes a decoder KV cache.

Minimum KV cache design for your framework:

```cpp
struct KVCacheLayer {
    Tensor K; // [max_seq_len, n_head_kv, head_dim] or equivalent
    Tensor V; // [max_seq_len, n_head_kv, head_dim] or equivalent
};

struct KVCache {
    std::vector<KVCacheLayer> layers;
    int current_pos;
};
```

Prefill:

```text
for prompt tokens:
    compute Q/K/V for all prompt positions
    write K/V into cache positions [0, prompt_len)
    attention reads positions [0, current_position]
```

Decode:

```text
for one new token:
    compute Q/K/V for current position
    write K/V into cache[current_pos]
    attention reads cache positions [0, current_pos]
    sample next token
```

llama.cpp has a more flexible cell-based KV cache for batching and multiple sequences. A first inference framework does not need to copy that design. Start with a dense per-layer KV cache.

## Minimal C++ Framework Checklist

For Qwen3-only generation, implement:

1. GGUF or custom weight loader for the tensors listed above.
2. Tokenizer compatible with the model vocabulary.
3. Tensor abstraction with at least FP16/BF16/FP32 storage and matmul.
4. RMSNorm:

```text
y = x * rsqrt(mean(x^2) + eps) * weight
```

5. Linear projection:

```text
y = x @ W
```

Adapt orientation to your storage format.

6. RoPE for Q and K over `head_dim`.
7. GQA attention with causal mask and KV cache.
8. SwiGLU FFN:

```text
ffn(x) = down(silu(gate(x)) * up(x))
```

9. Residual connections in the exact order shown above.
10. Final RMSNorm and lm head.
11. Sampler, at least greedy first.

## Pseudocode

```cpp
Tensor forward_qwen3(TokenBatch tokens, KVCache & kv, int pos0) {
    Tensor x = embedding(tokens);

    for (int l = 0; l < n_layer; ++l) {
        Tensor residual = x;

        Tensor h = rms_norm(x, layer[l].attn_norm);

        Tensor q = linear(h, layer[l].wq);
        Tensor k = linear(h, layer[l].wk);
        Tensor v = linear(h, layer[l].wv);

        q = reshape(q, {n_tokens, n_head,    head_dim});
        k = reshape(k, {n_tokens, n_head_kv, head_dim});
        v = reshape(v, {n_tokens, n_head_kv, head_dim});

        q = rms_norm_per_head(q, layer[l].attn_q_norm);
        k = rms_norm_per_head(k, layer[l].attn_k_norm);

        apply_rope(q, pos0);
        apply_rope(k, pos0);

        kv.layers[l].write(pos0, k, v);

        Tensor a = gqa_attention(q, kv.layers[l].K, kv.layers[l].V, causal_mask, 1.0f / sqrt(head_dim));
        a = linear(a, layer[l].wo);

        x = residual + a;

        residual = x;
        h = rms_norm(x, layer[l].ffn_norm);

        Tensor up   = linear(h, layer[l].ffn_up);
        Tensor gate = linear(h, layer[l].ffn_gate);
        Tensor ffn  = linear(silu(gate) * up, layer[l].ffn_down);

        x = residual + ffn;
    }

    x = rms_norm(x, output_norm);
    return linear(x, output);
}
```

## What Not To Copy At First

Do not start by copying all of llama.cpp:

- LoRA support through `build_lora_mm()`
- control vectors through `build_cvec()`
- multi-backend graph scheduler
- graph reuse
- cell-based multi-sequence KV allocator
- embedding/rerank pooling paths
- speculative decoding
- server slots

Those are framework features, not required for a first Qwen3 forward path.

## Practical Reading Map

Read in this order:

1. `src/models/qwen3.cpp`: exact Qwen3 layer order.
2. `src/llama-model.cpp`: Qwen3 tensor allocation and dispatch.
3. `src/llama-arch.cpp`: Qwen3 tensor names.
4. `src/llama-graph.cpp`: helper implementations for RMSNorm, FFN, attention, inputs.
5. `src/llama-kv-cache.cpp`: how llama.cpp stores and masks KV.
6. `examples/simple/simple.cpp`: how an application drives prefill/decode/sample.

For your own framework, the useful abstraction boundary is:

```text
ModelLoader -> Qwen3Model -> RuntimeContext -> KVCache -> BackendOps -> Sampler
```

`Qwen3Model::forward()` should implement the graph in this note.
