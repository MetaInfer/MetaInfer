# 模型规格与失败模式库

> 关联 notebooks: `07_improvementPlan/bugfix.md`, `06_experience/*`

---

## 已验证模型参考

> **⚠️ 重要：以下维度表是特定模型的已验证示例值，不是规范。所有维度必须从 `config.json` 动态读取。**
> 这些表格展示的是"某次构建中 config.json 的实际内容"——它们是验证参考，用于确认给定模型配置下 per-rank 维度的正确计算方式。

### Qwen3-8B (已验证示例)

#### 全量值 (来自该模型 config.json)

| 参数 | 值 |
|------|-----|
| hidden_size | 4096 |
| intermediate_size | 12288 |
| num_attention_heads | 32 |
| num_key_value_heads | 8 |
| num_hidden_layers | 36 |
| head_dim | 128 |
| vocab_size | 151936 |
| max_position_embeddings | 40960 |
| rope_theta | 1000000.0 |
| rms_norm_eps | 1e-06 |

#### TP=4 Per-Rank 维度 (示例：计算方式)

| 参数 | 计算方式 | 示例值 (Qwen3-8B, TP=4) |
|------|---------|----------------------|
| hidden per rank | hidden_size / tp_size | 1024 |
| intermediate per rank | intermediate_size / tp_size | 3072 |
| attn_heads per rank | num_attention_heads / tp_size | 8 |
| kv_heads per rank | num_key_value_heads / tp_size | 2 |
| qkv_weight per rank | [q_size+2*kv_size, hidden_size] | [1536, 4096] |
| gate_up_weight per rank | [2*intermediate/tp, hidden_size] | [6144, 4096] |
| max_blocks | max_position_embeddings / block_size | 160 (40960/256) |

---

## 全局约束 (Global Primitives Constraints)

### rmsnorm_precision_law

- RMSNorm 使用 vLLM CUDA kernel (`rms_norm` / `fused_add_rms_norm`)
- Kernel 内部 fp32 计算，调用方确保 out 预分配 + input contiguous
- 禁止手写 PyTorch 逐元素实现 (性能低 ~6x 且无 fused kernel 优化)

### tp_linear_load_no_double_shard

- ColumnParallel/RowParallel/VocabParallel 的 `load_weight_shard` 必须先检查 `shape == self.weight.shape`
- 若已为 shard → 直接 copy_
- 仅在全量权重时才按 tp_rank 切片

### hf_baseline_test_oom_guard

- torchrun 多进程对齐测试时，HF 基准模型保持 CPU 驻留
- 仅在 forward 时将 input_ids 放到 GPU
- 必要时 `torch.cuda.empty_cache()`

### debug_evidence_before_guessing

- 先做证据链，再推断根因
- 数值崩塌故障 → 至少采集 3 层中间 tensor 的 shape/dtype/statistics
- 通信故障 → 先确认 `is_tp_enabled()` 和 dist 初始化状态

---

## Platform Detection (平台自动检测)

启动时必须执行（此检测结果被所有平台相关契约文件交叉引用）:
1. `torch.cuda.get_device_name(0)` → GPU 品牌 (NVIDIA/AMD/Iluvatar)
2. `torch.cuda.get_device_capability(0)` → SM/Compute Unit 版本
3. 通信后端版本: `torch.distributed.get_backend_version()`
4. 根据结果路由:
   - **NVIDIA**: CustomAR 使用 IPC handle + NCCL all_reduce
   - **AMD (ROCm)**: CustomAR 不可用，fallback RCCL all_reduce
   - **DCU (Iluvatar)**: CustomAR 不可用，使用 torch.cuda.comm 或 fallback
5. 各平台相关契约（kernel/communication/engine）应引用此检测结果，而非硬编码平台假设

---

## Failure Mode Library

从 10 个有效改动点的踩坑经验标准化提取。每条: symptom → check → fix。

### FM-001: TP Embedding 双重切片
- **Symptom**: VocabParallelEmbedding 加载报错 `size mismatch (0)`
- **Check**: `_load_tensor(split_dim=0)` 已按 tp_rank 取本地分片，load_weight_shard 是否二次切片
- **Fix**: load_weight_shard 增加分支: 若传入张量首维已等于 local_vocab_size → 直接 copy；否则按全量词表范围切片

### FM-002: RMSNorm contiguous 约束
- **Symptom**: rms_norm 输出数值异常或 CUDA error
- **Check**: 传入 rms_norm 的 4D QK norm tensor (view 结果) 是否 `.is_contiguous()`
- **Fix**: 调用前加 `.contiguous(): rms_norm(out, x.contiguous(), weight, eps)`

### FM-003: fused_add_rms_norm 跨层 weight
- **Symptom**: post_mlp 的 fused_add_rms_norm 使用了错误的 weight → 输出异常
- **Check**: 确认所有 4 处 fused_add_rms_norm 调用的 weight 均为 `self.xxx.weight` (本层内)
- **Fix**: 所有调用均使用本层 `self.input_layernorm.weight` 或 `self.post_attention_layernorm.weight`

### FM-004: CosSinCache 格式 + 显存
- **Symptom**: rotary_embedding 输出与 PyTorch RoPE 不一致；或 36×8MB=288MB 显存浪费
- **Check**: cache shape 应为 `[max_pos, head_size]=[40960,128]` 而非 `[2*head_size]`；是否模块级共享
- **Fix**: `make_cos_sin_cache(max_pos, head_size)`；`_cos_sin_cache_registry` dict 共享，lazy 移到 GPU

### FM-005: CustomAR gloo ProcessGroup
- **Symptom**: `dist.all_gather_object` 报错 (NCCL 不支持 object collectives)
- **Fix**: `gloo_group = dist.new_group(backend='gloo')`

### FM-006: QKV weight 拼接索引
- **Symptom**: QKV 切片后 Q/K/V 维度与 num_heads 不匹配
- **Fix**: load_weight_shard 按 `[0:q_size]`, `[q_size:q_size+kv_size]`, `[q_size+kv_size:]` 三段复制

### FM-007: RoPE Neox vs GPT-J 错配
- **Symptom**: 短句重复、局部复读、退化输出 (非乱码，有一定语义但质量极差)
- **Check**: Qwen3 必须 `is_neox=True` (前后分半 rotate_half)；DeepSeek 用 `is_neox=False` (GPT-J 奇偶交错)
- **Fix**: 根据模型系列选择正确的 is_neox 值

### FM-008: paged KV block_size
- **Symptom**: flash_attn_with_kvcache 报错
- **Check**: `_kv_block_size >= 256`
- **Fix**: 使用 `block_size=256` (flash_attn_with_kvcache 硬性最低要求)

### FM-009: compiled region 内 .item()
- **Symptom**: SIGABRT (torch.compile fullgraph=True 内 CPU sync)
- **Fix**: kv_len `.item()` 移到 forward() 外部 — batch 读取所有层后统一 sync
- **nocompile note**: 当前 nocompile 模式此问题不适用，但 `.item()` 仍需在 forward() 外层

### FM-016: RMSNorm 计算顺序
- **Symptom**: 单卡乱码，所有组件单测 diff=0，但 Q norm diff ≈ 0.0625 或 K norm diff ≈ 1.0
- **Fix**: `(self.weight.float() * x_f).to(bf16)` → `self.weight * x_f.to(bf16)` (HF 权重已针对此精度路径训练)

### FM-017: forward 缺失 residual 合并
- **Symptom**: top-1 token 为 `!` 且 logit=0.0，所有组件单测 PASS
- **Fix**: forward 36 层循环后加 `hs = hs + res` (vLLM-style DecoderLayer 返回分离二元组)

### FM-018: RowParallel _row_slice 未除 tp_size
- **Symptom**: TP=4 权重复载: `shape mismatch loaded=[4096, 0], model=[4096, 1024]`
- **Fix**: `_row_slice(weight, hidden_size // tp_size)` 而非 `hidden_size`

### FM-019: CustomAR rank_data 尺寸/GC
- **Symptom**: TP=4 forward 时 `CUBLAS_STATUS_EXECUTION_FAILED`，崩溃位置不稳定
- **Fix**: rank_data 增至 16MB + CustomARHandle 持有 rank_data 引用防止 GC

---

## E2E 输出乱码回溯流程

```
Phase 10 E2E 乱码
  ├─ 单卡 TP=1 也乱码？
  │   ├─ Q/K norm 与 HF 参考有偏差？ → FM-016 (Phase 5)
  │   ├─ top-1 token 异常 (如 '!' + logit=0)？ → FM-017 (Phase 6)
  │   └─ 模型正常加载但 forward 不报错？ → Bug 6: q_norm/k_norm 遗漏 (Phase 7)
  ├─ 仅 TP>1 乱码？
  │   ├─ 权重复载 shape mismatch？ → FM-018 (Phase 7)
  │   └─ CUBLAS crash 位置不稳定？ → FM-019 (Phase 2)
  └─ 短句重复/局部复读（非完全乱码）？
      └─ FM-007: RoPE Neox vs GPT-J 错配 (Phase 5)
```

---

## Qwen3.6-27B (已验证示例 — Qwen3.5 Hybrid Architecture)

<!-- source: evolution/evo-001, timestamp=2026-07-03 -->

### 全量值 (来自 config.json)

| 参数 | 值 |
|------|-----|
| architectures | Qwen3_5ForConditionalGeneration |
| model_type | dense_hybrid (48 linear_attention + 16 full_attention) |
| hidden_size | 5120 |
| intermediate_size | 17408 |
| num_attention_heads | 24 |
| num_key_value_heads | 4 |
| head_dim | 256 |
| num_hidden_layers | 64 |
| vocab_size | 248320 |
| max_position_embeddings | 262144 |
| rope_theta | 10000000.0 |
| partial_rotary_factor | 0.25 |
| rotary_dim | 64 |
| mrope_section | [11, 11, 10] |
| rms_norm_eps | 1e-06 |
| norm_type | Qwen3_5RMSNorm (weight=zeros, (1+w)*x) |
| attention_type | Hybrid: GatedDeltaNet (48 layers) + FullAttention with MRoPE (16 layers) |
| ffn_type | SwiGLU (silu) |
| linear_key_head_dim | 128 |
| linear_num_key_heads | 16 |
| linear_value_head_dim | 128 |
| linear_num_value_heads | 48 |
| linear_conv_kernel_dim | 4 |
| attn_output_gate | true (sigmoid gate from q_proj second half) |
| weight_key_prefix | model.language_model.* (NOT model.*) |
| config_nesting | text_config { ... } (top-level config wraps text_config) |

### TP=4 Per-Rank 维度

| 参数 | 值 |
|------|-----|
| hidden per rank | 1280 |
| intermediate per rank | 4352 |
| attn_heads per rank | 6 (full_attention) |
| kv_heads per rank | 1 |
| k_heads per rank (linear) | 4 |
| v_heads per rank (linear) | 12 |
| qkv_weight per rank (linear) | [2560, 5120] |
| q_size per rank (full_attn) | 1536 |
| q_double_size per rank (full_attn) | 3072 |

### RMSNorm 变体对照

| 变体 | weight init | 公式 | 使用位置 |
|------|------------|------|---------|
| RMSNorm (标准) | ones | x*rsqrt*w | FullAttention Q/K norms |
| Qwen3_5RMSNorm | zeros | x*rsqrt*(1+w) | decoder norms + final norm |
| Qwen3_5RMSNormGated | ones | x*rsqrt*w*silu(gate) | GatedDeltaNet output |
