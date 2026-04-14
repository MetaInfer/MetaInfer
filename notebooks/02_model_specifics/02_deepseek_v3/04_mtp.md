# DeepSeek V3 - Multi-Token Prediction (MTP)

## Core Concept

Multi-Token Prediction adds additional "prediction heads" that predict tokens beyond just the next one. During inference, these heads serve as a built-in speculative decoding mechanism.

## Architecture

```
Main Model Forward Pass:
    hidden_states[layer_L]  ← Final hidden state from main model
        ↓
MTP Head 1 (predicts token at position t+2):
    ├── eh_proj: concat(embed(token_t+1), hidden[L]) → combined
    ├── RMSNorm → normalized
    ├── TransformerLayer → mtp_hidden_1
    └── LM Head → logits for position t+2

MTP Head 2 (predicts token at position t+3):
    ├── eh_proj: concat(embed(token_t+2), mtp_hidden_1) → combined
    ├── RMSNorm → normalized
    ├── TransformerLayer → mtp_hidden_2
    └── LM Head → logits for position t+3
```

## Key Design Choices

1. **Embedding-Hidden Projection (eh_proj)**: Combines the embedding of the predicted token with the hidden state from the previous stage
2. **Shared LM Head**: The MTP heads typically share the main model's LM head weights
3. **Lightweight Layers**: Each MTP head uses only 1 transformer layer (vs 32+ in the main model)

## Speculative Decoding with MTP

### Draft Phase
```python
def speculative_draft(main_hidden, last_token):
    draft_tokens = [last_token]
    h = main_hidden

    for mtp_head in mtp_heads:
        # Predict next token using MTP head
        combined = mtp_head.eh_proj(concat(embed(draft_tokens[-1]), h))
        h = mtp_head.transformer_layer(combined)
        logits = mtp_head.lm_head(h)
        draft_token = sample(logits)
        draft_tokens.append(draft_token)

    return draft_tokens[1:]  # N draft tokens
```

### Verify Phase
```python
def speculative_verify(input_tokens, draft_tokens):
    # Run main model on all tokens at once (as if prefill)
    all_tokens = concat(input_tokens, draft_tokens)
    all_logits = main_model.forward(all_tokens)

    # Verify each draft token
    accepted = []
    for i, draft in enumerate(draft_tokens):
        main_prediction = sample(all_logits[input_len + i])
        if main_prediction == draft:
            accepted.append(draft)
        else:
            accepted.append(main_prediction)
            break  # Stop at first rejection

    return accepted
```

## Impact on Inference Framework

1. **Hidden State Capture**: The main model must expose intermediate hidden states to feed into MTP heads
2. **Extra Forward Pass**: MTP heads add a small forward pass per speculative step (but much cheaper than the main model)
3. **Verification Batching**: Need to batch the draft tokens with the main model for verification
4. **Acceptance Rate**: Higher acceptance → more speedup; typically 70-90% for well-trained MTP heads
5. **Memory**: MTP heads add minimal memory (1 layer each + shared LM head)

## When to Use

- **Beneficial**: Low-latency single-request scenarios where TTPT (time per token) matters
- **Less beneficial**: High-throughput batch scenarios where GPU is already saturated
- **Trade-off**: Adds complexity to the inference loop; only worth it when acceptance rate is high
