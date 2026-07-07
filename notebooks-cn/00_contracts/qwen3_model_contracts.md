# Qwen3 模型完整层级 — API 契约

> 关联 notebooks: `04_parallel_strategies/qwen_dense_tp_implementation_guide.md`

## 概述

Qwen3-8B 的 TP=4 完整类层级 — 从 Config 到 Top-Level Model。
源实现文件: `engine/models/qwen.py`

---

## Class Hierarchy

### QwenTPConfig

```python
@dataclass
class QwenTPConfig:
    model_dir: Path
    # ⚠️ 以下默认值为 Qwen3-8B 示例，不是规范。所有值应从 config.json 动态读取
    # Factory 方法 _load_qwen_tp_config() 会覆盖这些默认值
    hidden_size: int = 4096        # 示例值，实际从 config.json 读取
    intermediate_size: int = 12288  # 示例值
    num_hidden_layers: int = 36     # 示例值
    num_attention_heads: int = 32   # 示例值
    num_key_value_heads: int = 8    # 示例值
    head_dim: int = 128             # 示例值
    vocab_size: int = 151936        # 示例值
    rms_norm_eps: float = 1e-06     # 示例值
    rope_theta: float = 1000000.0   # 示例值
    max_position_embeddings: int = 40960  # 示例值
    tie_word_embeddings: bool = False
```

**Factory**: `_load_qwen_tp_config(model_dir)` → `AutoConfig.from_pretrained(model_dir, trust_remote_code=True)` → QwenTPConfig(...)

**head_dim fallback**: `if not hasattr(cfg, 'head_dim'): cfg.head_dim = cfg.hidden_size // cfg.num_attention_heads`

### QwenAttentionTP

**Constructor**: `__init__(self, cfg)`

**属性列表**:
```
self.total_num_heads = cfg.num_attention_heads        # 32 (全量)
self.total_num_kv_heads = cfg.num_key_value_heads      # 8 (全量)
self.num_heads = cfg.num_attention_heads // tp_size    # 8 (per-rank)
self.num_kv_heads = cfg.num_key_value_heads // tp_size # 2 (per-rank)
self.head_dim = cfg.head_dim                           # 128
self.q_size = self.num_heads * self.head_dim           # 1024
self.kv_size = self.num_kv_heads * self.head_dim       # 256
self.scaling = self.head_dim ** -0.5                   # 0.08839
self.qkv_proj = QKVColumnParallelLinear(...)           # ⚠️ 非 q_proj/k_proj/v_proj
self.o_proj = RowParallelLinear(...)                   # ⚠️ 非 out_proj
self.q_norm = RMSNorm(self.head_dim, eps)              # ⚠️ Qwen3 特有 per-head norm
self.k_norm = RMSNorm(self.head_dim, eps)              # ⚠️ Qwen3 特有 per-head norm
self._kv_block_size = 256
self._key_cache = None; self._value_cache = None; self._block_table = None
self._kv_len_gpu: Buffer[1] int32
self._slot_mapping_decode: Buffer[1] int64
self._cos_sin_cache_cpu / self._cos_sin_cache_gpu
```

### QwenMLPTP

**Constructor**: `__init__(self, cfg)`

**属性列表**:
```
self.gate_up_proj = MergedColumnParallelLinear(...)   # ⚠️ 非 gate_proj + up_proj
self.down_proj = RowParallelLinear(...)                # ⚠️ 非 down_proj (命名正确但注意类型)
```

### QwenDecoderLayerTP

**Constructor**: `__init__(self, cfg)`

**属性列表**:
```
self.self_attn = QwenAttentionTP(cfg)                  # ⚠️ 非 attention
self.mlp = QwenMLPTP(cfg)
self.input_layernorm = RMSNorm(cfg.hidden_size, eps)   # ⚠️ 非 ln_1
self.post_attention_layernorm = RMSNorm(cfg.hidden_size, eps)  # ⚠️ 非 ln_2
```

### QwenForCausalLMTP

**Constructor**: `__init__(self, cfg, device, dtype)`

**属性列表**:
```
self.embed_tokens = VocabParallelEmbedding(vocab_size, hidden_size)
self.layers = nn.ModuleList([QwenDecoderLayerTP(cfg) for _ in range(num_hidden_layers)])
self.norm = RMSNorm(cfg.hidden_size, eps)
self.lm_head = ParallelLMHead(hidden_size, vocab_size, gather_output=True)
```

---

## Top-Level Forward (模型入口)

```python
def forward(self, input_ids, past_key_values=None, position_offset=0, max_seq_len=None):
    # max_seq_len: 若未指定，从 self.cfg.max_position_embeddings 推导
    hidden_states = self.embed_tokens(input_ids)  # [B, S, hidden_size]
    seq_len = input_ids.shape[1]
    positions = torch.arange(position_offset, position_offset + seq_len, device=input_ids.device, dtype=torch.long)
    residual = None
    is_decode = past_key_values is not None
    
    if not is_decode:  # === prefill ===
        for i, layer in enumerate(self.layers):
            hidden_states, residual = layer.forward(
                hidden_states, positions, layer_cache=None, max_seq_len=max_seq_len, residual=residual)
        hidden_states, _ = self.norm(hidden_states, residual)
        kv_lens = None
    else:  # === decode ===
        for i, layer in enumerate(self.layers):
            kv_len = past_key_values[i]
            hidden_states, residual = layer.forward_decode(
                hidden_states, positions, kv_len, max_seq_len=max_seq_len, residual=residual)
        # Batch read kv_lens AFTER all layers (MUST be outside compiled region)
        kv_lens = [int(l.self_attn._kv_len_gpu[0].item()) for l in self.layers]
        hidden_states, _ = self.norm(hidden_states, residual)
    
    logits = self.lm_head(hidden_states)  # [B, S, vocab_size]
    return logits, kv_lens
```

---

## Construction Chain (5 步构造序列)

```
1. cfg = _load_qwen_tp_config(model_dir)           # config.json → QwenTPConfig
2. model = QwenForCausalLMTP(cfg, device, dtype)
3. model.load_weights(model_dir)                    # see weight_loading_contracts.md
4. model.eval()
5. init_custom_ar(device=device)                     # after model on GPU, before first forward
```

> **⚠️ forward/weight 耦合约束**：步骤 2（模型构造）中定义的所有 `nn.Parameter`/`nn.Module` 属性（特别是 QKV projection、q_norm/k_norm、gate_up_proj），其命名和 weight shape 必须在步骤 3（load_weights）中与 HF 权重 key 兼容。不能在 Phase 5/6 写完 forward 逻辑后再在 Phase 7 调整属性定义——两者必须同步设计。交叉引用：`weight_loading_contracts.md` §HF Key Mapping、`AGENT_SKILL.md` §1 规则 11。

---

## Prefill/Decode Dispatch

```
is_prefill = (past_key_values is None)
  → True:  layer.forward() → qkv_proj → rotary → flash_attn_varlen_func(causal=True) → KV write → o_proj → mlp
  → False: layer.forward_decode() → qkv_proj → Q/K norm → rotary → KV write(index_copy_) → flash_attn_with_kvcache → o_proj
```

---

## 陷阱与反模式

- **FM-016**: RMSNorm 计算顺序 — `self.weight * x.to(bf16)` 非 `.float()` 路径
- **FM-017**: forward 后缺失 `hs = hs + res` — residual 合并必须手动执行
- **NAMING-001**: `.self_attn` 非 `.attention`
- **NAMING-002**: `.qkv_proj` 非 `.q_proj`
- **NAMING-003**: `.gate_up_proj` 非 `.gate_proj`
- **NAMING-004**: `.input_layernorm` 非 `.ln_1`
- **NAMING-005**: `.post_attention_layernorm` 非 `.ln_2`
- **NAMING-006**: `.q_norm` / `.k_norm` — Qwen3 特有 per-head QK norm，HF 权重有对应 key
- **RESIDUAL-001**: vLLM-style DecoderLayer 返回 `(mlp_out, residual)` 二元组 — caller 负责合并
- **SCOPE-NOTE**: 当前覆盖 nocompile, B=1, TP=4。不覆盖 torch.compile, CUDA Graph, B>1
