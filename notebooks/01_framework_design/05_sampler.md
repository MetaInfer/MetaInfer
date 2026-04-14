# Sampler - Token Sampling Strategies

## Core Concept

After the model produces logits (unnormalized scores for each vocabulary token), the sampler selects the next token. This is where generation quality meets generation control.

## Sampling Pipeline

```
logits → temperature scaling → top-k filtering → top-p (nucleus) filtering → softmax → sample
```

### Step 1: Temperature Scaling
```python
logits = logits / temperature
# temperature < 1.0: sharper distribution (more deterministic)
# temperature = 1.0: no change
# temperature > 1.0: flatter distribution (more random)
# temperature = 0.0: greedy (argmax)
```

### Step 2: Top-K Filtering
```python
if top_k > 0:
    # Keep only the top_k highest logits, set rest to -inf
    top_k_values, _ = torch.topk(logits, top_k)
    threshold = top_k_values[:, -1]
    logits[logits < threshold] = -inf
```

### Step 3: Top-P (Nucleus) Filtering
```python
if top_p < 1.0:
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
    # Remove tokens with cumulative probability above threshold
    mask = cumulative_probs - sorted_probs > top_p
    sorted_logits[mask] = -inf
    # Scatter back to original positions
    logits = sorted_logits.scatter(1, sorted_indices, sorted_logits)
```

### Step 4: Sample
```python
probs = torch.softmax(logits, dim=-1)
next_token = torch.multinomial(probs, num_samples=1)
```

## Implementations Across Projects

### nano-vllm: Compiled Sampler
Uses `@torch.compile` for kernel fusion:
```python
@torch.compile(dynamic=True)
def sample(logits, temperatures):
    logits = logits / temperatures.unsqueeze(1)
    probs = torch.softmax(logits, dim=-1)
    # Gumbel-max trick for efficient sampling
    q = torch.empty_like(probs).exponential_(1)
    return (probs / q).argmax(dim=-1)
```

The **Gumbel-max trick** avoids the expensive `torch.multinomial` by using the equivalence:
```
argmax(log(probs) + gumbel_noise) == multinomial(probs)
```
Where gumbel noise = -log(-log(uniform)), which is equivalent to sampling from exponential distribution and dividing.

### nano-sglang: Batch Sampling with FSM
Supports per-request sampling parameters and constrained decoding:
```python
def sample(logits, batch):
    for i, req in enumerate(batch.reqs):
        # Apply FSM mask for constrained decoding
        if req.regex_fsm:
            allowed = req.regex_fsm.allowed_token_ids(req.fsm_state)
            logits[i, ~allowed] = -inf

        # Per-request temperature
        logits[i] /= req.sampling_params.temperature

    # Batch top-p/top-k filtering
    apply_top_k(logits, batch.top_k_values)
    apply_top_p(logits, batch.top_p_values)

    probs = torch.softmax(logits, dim=-1)
    next_tokens = torch.multinomial(probs, num_samples=1)
    return next_tokens
```

### mini-sglang: GPU-Side Sampler
```python
class Sampler:
    def __call__(self, logits, batch):
        # Greedy (temperature = 0)
        if all_greedy:
            return logits.argmax(dim=-1)

        # Stochastic
        logits /= temperatures
        if has_top_k:
            apply_top_k(logits, top_k_values)
        if has_top_p:
            apply_top_p(logits, top_p_values)
        probs = torch.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1).squeeze(1)
```

## Greedy Decoding Shortcut

When temperature = 0 or greedy mode is requested:
```python
if temperature == 0:
    next_token = logits.argmax(dim=-1)
    # Skip all filtering and probability computation
```

This is significantly faster than the full sampling pipeline.

## Design Template

A minimal sampler needs:
1. **Temperature scaling** (always)
2. **Greedy shortcut** for temperature=0
3. **`torch.multinomial`** or Gumbel-max for stochastic sampling

Optional enhancements:
- Top-k / top-p filtering (common user request)
- Per-request sampling parameters (needed for serving)
- `@torch.compile` for kernel fusion (measurable speedup)
- Constrained decoding integration (FSM mask application)
- Repetition penalty, frequency penalty, presence penalty
- Min-p sampling
