# Qwen3 TP 全部有效改动点知识追溯

> 从 4 卡 TP 拉通后，Qwen3 的全部有效改动 = 7 个 kernel 替换 + 3 个框架层改动。
> 每个改动点均记录：参考源码路径、最终实现方法、踩坑经验、文档位置。

---

## 第一层：Kernel 替换（7 个）

### Stage 1 — rms_norm + fused_add_rms_norm

| 维度 | 记录 |
|------|------|
| **参考源码** | `vllm/_custom_ops.py:420-423` → `vllm_wrappers.py` 新建 |
| **最终实现** | `RMSNorm.forward(x)` → `rms_norm`，`RMSNorm.forward(x, residual)` → `fused_add_rms_norm`。vLLM 风格 residual chaining，residual 由 model forward loop 在层间传递 |
| **踩坑 1** | `rms_norm` 要求 contiguous input → 调用前加 `.contiguous()` |
| **踩坑 2** | `fused_add_rms_norm` 第二处需要**下一层**的 `input_layernorm.weight`（跨层依赖） → 采纳 vLLM residual chaining 模式 |
| **文档位置** | `kernel_replacement_plan.md` Stage 1 实施报告，commit `8f811bd` |
| **经验记录** | `06_experience/p5_kernel_fusion_experience.md` |

### Stage 2 — silu_and_mul + MergedColumnParallelLinear

| 维度 | 记录 |
|------|------|
| **参考源码** | `vllm/model_executor/layers/activation.py` class `SiluAndMul.forward_cuda` → `vllm_wrappers.py` |
| **最终实现** | `MergedColumnParallelLinear(gate+up)` → `empty_like` → `silu_and_mul(out, gate_up)` → `down_proj` |
| **踩坑 1** | `MergedColumnParallelLinear` 未从 `engine.tp_layers` 导出 → `__init__.py` 加 import |
| **踩坑 2** | 合并 GEMM vs separate GEMM 有 1-2 ULP 差异 → bf16 tiling 噪声，端到端输出不受影响 |
| **文档位置** | `kernel_replacement_plan.md` Stage 2 实施报告，commit `daea0b7` |

### Stage 3 — rotary_embedding + cos_sin_cache

| 维度 | 记录 |
|------|------|
| **参考源码** | `vllm/_custom_ops.py:400-410`（kernel）+ `vllm/model_executor/layers/rotary_embedding/base.py:76-84`（cache 构造） |
| **最终实现** | `make_cos_sin_cache`（模块级共享）→ flatten q,k → `rotary_embedding(in-place)` → unflatten |
| **踩坑 1** | cos_sin_cache 格式：`[max_pos, 2*head_size]`（错误）→ `[max_pos, head_size]`（正确，vLLM kernel 内部分半为 cos/sin，自行处理 NeoX 重复） |
| **踩坑 2** | 每层创建 cache 36×8MB=288MB → 模块级 `_cos_sin_cache_registry` dict 共享 |
| **踩坑 3** | CPU 创建 → GPU 使用 → lazy 首次 forward 时移到 GPU（`_cos_sin_cache_gpu`） |
| **文档位置** | `kernel_replacement_plan.md` Stage 3 实施报告 + Stage 0 §0.2，commit `2c2b794` |

### Stage 4 — CustomAR P2P 替换 NCCL

| 维度 | 记录 |
|------|------|
| **参考源码** | `vllm/_custom_ops.py:640-680` → `custom_ar.py` 新建 + `distributed.py` |
| **最终实现** | `init_custom_ar` 创建 gloo 组（IPC 句柄交换）→ 分配 P2P 共享 buffer → `ops.init_custom_ar` + `ops.register_buffer`。`all_reduce_sum` 透明切换到 `ops.all_reduce` |
| **踩坑 1** | CustomAR 需要非 NCCL ProcessGroup 做 IPC handle exchange → 创建 gloo `dist.new_group()` |
| **踩坑 2** | `dist.all_gather_object` 需要 gloo backend → 使用 gloo 组传递 IPC handles |
| **踩坑 3** | 单 GPU（world_size=1）无需初始化 → `__init__` 中提前 return |
| **文档位置** | `kernel_replacement_plan.md` Stage 4 实施报告，commit `d554fc2` |

### Stage 5 — SDPA enable_gqa

| 维度 | 记录 |
|------|------|
| **参考源码** | PyTorch 2.9+ 原生 `F.scaled_dot_product_attention(enable_gqa=True)` |
| **最终实现** | 去掉 `repeat_interleave`，K/V 保持 `[B, num_kv_heads, kv_len, D]` 不广播，SDPA 内部处理 GQA |
| **踩坑 1** | `enable_gqa=True` 要求格式 `(B, H, S, D)` 且 `Hq > Hkv` |
| **文档位置** | `kernel_replacement_plan.md` Stage 5 实施报告，commit `95b7595` |

### Stage 6 — KV Cache 写入（认定为已最优）

| 维度 | 记录 |
|------|------|
| **参考源码** | nano-sglang 的 Python slice、sglang 的 JIT CUDA StoreKernel、vLLM 的 `reshape_and_cache_flash_kernel` |
| **最终实现** | 保持 Python slice 不变（`k_buf[:, kv_len:kv_len+seqlen] = k`），decode B=1 连续写入场景下已最优 |
| **踩坑** | 无——经过 4 个参考工程横评确认，不需要改动 |
| **文档位置** | `kernel_replacement_plan.md` Stage 6 组装说明 |

### Stage 7 — QKV 合并投影

| 维度 | 记录 |
|------|------|
| **参考源码** | vLLM `QKVParallelLinear`（继承 `ColumnParallelLinear`，单次 GEMM → split → Q/K/V） → `linear.py` |
| **最终实现** | `QKVColumnParallelLinear`：merged weight `[q_size+2*kv_size, H]` → `F.linear` → `.split([q_size, kv_size, kv_size])` |
| **踩坑 1** | QKV weight 拼接需要正确分配索引 → `load_weight_shard` 按 `[0:q_size]`, `[q_size:q_size+kv_size]`, `[q_size+kv_size:]` 三段复制，配合 KV head replication |
| **文档位置** | `kernel_replacement_plan.md` Stage 7 实施报告，commit `2931779` |

---

## 第二层：框架层改动（3 个）

### P0 — 增量 KV Cache 解码

| 维度 | 记录 |
|------|------|
| **参考源码** | 自研，参考 HuggingFace `past_key_values` 模式 |
| **最终实现** | TP attention 层内置 KV cache（`_key_cache`/`_value_cache`），prefill 时缓存 KV，decode 时拼接后只算新 token |
| **踩坑 1** | `is_causal=False` 在 prefill 破坏因果注意力 → `is_causal = (past_key_values is None)` |
| **踩坑 2** | DeepSeek MLA 的 k_pe 需要在 decode 时对所有位置重新 RoPE → 缓存 raw_k_pe（RoPE 前），decode 时拼接全量位置重算 |
| **文档位置** | `improvement_plan.md` §P0 实现记录 |
| **经验记录** | `06_experience/01_task10_tp_qwen_debug_experience.md`（Embedding 分片、RoPE rotate_half、RMSNorm fp32 等 6 类 TP 调试问题） |

### P2 — torch.compile kernel fusion

| 维度 | 记录 |
|------|------|
| **参考源码** | PyTorch `torch.compile(mode='default')` |
| **最终实现** | 每层 `self_attn` + `mlp` 分别 `torch.compile(mode='default')`，decode 路径固定 shape（全 buffer + attn_mask）避免重编译 |
| **踩坑 1** | `mode='reduce-overhead'` 内部 CUDA Graph 与 KV cache buffer 跨步复用冲突 → `RuntimeError` → 回退到 `mode='default'` |
| **踩坑 2** | 动态切片 `k[:, :kv_len]` 形状每步变化 → 改全 buffer + attn_mask |
| **踩坑 3** | MoE 模块有 `.item()` GPU→CPU 同步 → 编译会 graph break，不编译 MoE |
| **文档位置** | `improvement_plan.md` §P2 |
| **经验记录** | `06_experience/p5_kernel_fusion_experience.md` |

### P3-FA — Flash Attention 集成

| 维度 | 记录 |
|------|------|
| **参考源码** | `flash_attn` package（pybind11），`flash_attn_varlen_func` |
| **最终实现** | Qwen: prefill+decode 都用 `flash_attn_varlen_func`，消除 permute 和 attn_mask |
| **踩坑 1** | 方案 A（切片 KV + SDPA 无 mask）→ -20%，torch.compile 动态 shape 重编译 |
| **踩坑 2** | 方案 B（切片 KV + SDPA + 去 attn compile）→ -12%，切片开销 > 消除 mask 收益 |
| **踩坑 3** | 方案 C（V-padding + flash_attn_varlen_func）→ -15%，F.pad/unpad 开销 |
| **踩坑 4** | DeepSeek QK headdim=192 < 256，V padding 后输入为 full buffer |
| **文档位置** | `improvement_plan.md` §P3-FA，`flashattn.md` |
| **经验记录** | `06_experience/03_flash_attn_profiling_analysis.md`（P2 基线与 FA kernel 的 profiling trace 分析） |

---

## 知识追溯总览

| # | 改动点 | 参考源码 | 最终实现文件 | 踩坑数 | 所属计划文档 |
|---|--------|---------|-------------|--------|------------|
| 1 | rms_norm + fused_add_rms_norm | `vllm/_custom_ops.py:420-423` | `vllm_wrappers.py`, `qwen.py` | 2 | kernel_replacement_plan.md Stage 1 |
| 2 | silu_and_mul + MergedColumnParallelLinear | `vllm/.../activation.py` | `vllm_wrappers.py`, `linear.py`, `qwen.py` | 2 | kernel_replacement_plan.md Stage 2 |
| 3 | rotary_embedding + cos_sin_cache | `vllm/_custom_ops.py:400-410`, `vllm/.../base.py:76-84` | `vllm_wrappers.py`, `qwen.py` | 3 | kernel_replacement_plan.md Stage 3 |
| 4 | CustomAR P2P 替换 NCCL | `vllm/_custom_ops.py:640-680` | `custom_ar.py`, `distributed.py` | 3 | kernel_replacement_plan.md Stage 4 |
| 5 | SDPA enable_gqa | PyTorch 2.9+ 原生 | `qwen.py` | 1 | kernel_replacement_plan.md Stage 5 |
| 6 | KV Cache 写入（已最优） | nano-sglang/vLLM 横评 | `qwen.py`（不变） | 0 | kernel_replacement_plan.md Stage 6 |
| 7 | QKV 合并投影 | vLLM `QKVParallelLinear` | `linear.py`, `qwen.py` | 1 | kernel_replacement_plan.md Stage 7 |
| 8 | 增量 KV Cache 解码 | 自研（HuggingFace 参考） | `qwen.py`, `deepseek_v2.py`, `llm_engine.py` | 2 | improvement_plan.md §P0 |
| 9 | torch.compile kernel fusion | PyTorch `torch.compile(mode='default')` | `qwen.py`, `deepseek_v2.py` | 3 | improvement_plan.md §P2 |
| 10 | Flash Attention 集成 | `flash_attn` package | `qwen.py`, `deepseek_v2.py` | 4 | improvement_plan.md §P3-FA |

> 全部 10 个改动点的参考源码路径、最终方法、踩坑经验——均可从 `kernel_replacement_plan.md`、`improvement_plan.md`、`06_experience/` 中找到。
