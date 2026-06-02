# 蓝图重构完备性漏洞报告

> **审计对象**：`inference_blueprint.json` (v2.3.0) + `AGENT_SKILL.md`
> **审计目标**：独立重构 Qwen3-8B TP=4 调度层 + 推理引擎
> **审计日期**：2026-05-26
> **排除范围**：所有 `deepseek_v2_v3_mla_moe` 相关节点

---

## 1. 🔴 Reconstructability Gaps (重构死锁漏洞项)

以下空洞将导致全新 Agent 在无现有源码参考下必定写出崩溃或死锁代码。

### GAP-1: Scheduler 与 BlockManager 的物理耦合链路缺失

| 属性 | 值 |
|------|-----|
| **JSON Path** | `framework_layer.data_flow_contracts.scheduler_to_runner` |
| **严重等级** | 🔴 Critical |

**问题描述**：

图谱定义了 `can_allocate` 和 `can_append_one_more` 的公式，但 Agent 无法得知 Scheduler 如何获取 `num_free_blocks`。BlockManager 的 API spec 中仅有 `allocate(seq, num_blocks)`、`free(block_id)`、`may_append(seq)`，缺少 `num_free_blocks` 属性或 `get_free_blocks()` 查询接口。Scheduler 如何感知 BlockManager 的内部状态是一个黑盒。

此外，`max_num_batched_tokens` 公式为 `max(1, num_free_blocks * block_size)`，但 `block_size` 存在双轨歧义——Scheduler 应使用 `block_size=16`（框架层）还是 `block_size=256`（TP Runner）？对于 TP Runner 路径，`_dual_track_note` 声明了 BlockManager 不参与，但 Scheduler 仍需要某种方式感知可调度 token 上限。

**必须补充的规格**：
```
BlockManager.get_num_free_blocks() -> int  # 缺失的 API
Scheduler.max_num_batched_tokens 的计算中 block_size 的选取规则:
  - RealModelRunner 路径: block_size = 16
  - TP Runner 路径: block_size = 256 (且不使用 BlockManager，直接查 KVMemoryPool)
```

---

### GAP-2: flash_attn_varlen_func 的 cu_seqlens 构造方法完全缺失

| 属性 | 值 |
|------|-----|
| **JSON Path** | `framework_layer.data_flow_contracts.flash_attention_integration_contract.prefill_path` |
| **严重等级** | 🔴 Critical |

**问题描述**：

图谱在 `flash_attention_integration_contract.prefill_path` 中指定了 kernel 调用签名：
```python
flash_attn_varlen_func(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, causal=True)
```
但完全没有给出 `cu_seqlens_q` 和 `cu_seqlens_k` 的构造方法。对于 prefill 中多个不同长度序列的 ragged batch，`cu_seqlens_q` 必须是前缀和形式的 `[0, len1, len1+len2, ...]` tensor。Agent 不知道：
- 从哪些 Sequence 字段获取各序列长度
- 在 TP Runner 中（块大小 256，一批 prefill token 可能跨多个 block），kv 是 paged 而非 contiguous 时，`cu_seqlens_k` 如何计算

**必须补充的规格**：
```python
# cu_seqlens_q 构造伪代码
cu_seqlens_q = torch.zeros(len(batch) + 1, dtype=torch.int32, device='cuda')
cu_seqlens_q[1:] = torch.tensor([seq.seq_len() for seq in batch]).cumsum(0).to(device)
# cu_seqlens_k 在 paged KV 场景的构造逻辑（等长或不等长）
```

---

### GAP-3: decode 路径的 position 和 max_seq_len 参数来源不明

| 属性 | 值 |
|------|-----|
| **JSON Path** | `framework_layer.data_flow_contracts.tp_layer_interface_contracts.qwen3_tp_model_interfaces.decode_forward_pattern` |
| **严重等级** | 🔴 Critical |

**问题描述**：

`decode_forward_pattern` 描述了 `layer.forward_decode(hidden_states, pos, kv_len, residual=residual)` 的签名，但没有说明：
- `pos` 从哪里来（每个序列的当前位置？是 `kv_len` 本身吗？）
- `max_seq_len` 参数在 `runner_decode_tensors._tp_runner_actual_path` 中被引用为 `layer.forward_decode(..., kv_len, max_seq_len, residual=residual)` 但在 `decode_forward_pattern` 中未出现——两个位置的函数签名不一致
- `residual` 初始值是什么（首次调用时为 `None`？为 `hidden_states.clone()` ？）

**必须补充的规格**：
```python
# decode_forward_pattern 统一签名
layer.forward_decode(
    hidden_states,      # [1, 1, hidden_size]
    positions,          # [1] int64 — 当前 token 位置 = kv_len
    kv_len,             # int — 当前 KV cache 长度
    max_seq_len,        # int — 序列最大长度 (block_table 分配上限)
    residual=None       # Tensor or None — 首次调用为 None
) -> (hidden_states, residual)
```

---

### GAP-4: prefill 阶段 paged KV 写入的 slot_mapping 构造算法缺失

| 属性 | 值 |
|------|-----|
| **JSON Path** | `framework_layer.data_flow_contracts.paged_kv_cache_contract.prefill_kv_write` |
| **严重等级** | 🔴 Critical |

**问题描述**：

`prefill_kv_write` 仅描述为 "`torch.arange(num_blocks) → block_table → 按 slot_mapping 顺序写入 _key_cache/_value_cache (index_copy_)`"。Agent 面临以下未知：
- `slot_mapping` 的具体计算方法（每个 prefill token 映射到 `block_id * block_size + offset_in_block`）
- prefill 可能跨多个 block，slot_mapping 是一个 1D tensor 包含所有 token 的槽位
- `index_copy_` 的 dim 参数（应在 dim=0 上写入）
- `torch.arange(num_blocks)` 填入 block_table 后，多余的 block_table 条目如何处理（保持为 0？填充为 -1？）

**必须补充的规格**：
```python
# prefill slot_mapping 构造伪代码
num_tokens = input_ids.shape[1]
num_blocks = (num_tokens + block_size - 1) // block_size
block_table[:num_blocks] = torch.arange(num_blocks)  # 或从 BlockManager 分配
slot_mapping = torch.zeros(num_tokens, dtype=torch.int64)
for i in range(num_tokens):
    slot_mapping[i] = block_table[i // block_size] * block_size + (i % block_size)
_key_cache.index_copy_(0, slot_mapping.to(device), k_flat)
```

---

### GAP-5: KVMemoryPool 与 BlockManager 的分工边界模糊

| 属性 | 值 |
|------|-----|
| **JSON Path** | `framework_layer.components[1]` (KVMemoryPool) + `framework_layer.components[2]` (BlockManager) |
| **严重等级** | 🔴 Critical |

**问题描述**：

KVMemoryPool 角色为 "管理 KV 逻辑块容量、块分配/释放与可扩展性检查"；BlockManager 角色为 "分页块分配器；维护 free/used block 集合"——两个组件的职责高度重叠。Agent 无法判断：
- 分配/释放操作的真正入口是 KVMemoryPool 还是 BlockManager？
- `num_free_blocks` 应该从哪个组件查询？
- BlockManager 的 `allocate` 返回 `list[int]`（block IDs），KVMemoryPool 是否也返回 block IDs？两者如何协调？

**必须补充的规格**：
```
KVMemoryPool: 负责显存预算计算 (estimate_num_blocks) 与 GPU 占位张量创建
BlockManager: 负责运行时块分配/释放/prefix caching，持有 free_pool 与 used_blocks
Scheduler → BlockManager.get_num_free_blocks() → 调度决策
Scheduler → BlockManager.allocate() → 获取 block_ids → 写入 Sequence.block_table
```

---

### GAP-6: CUDA Graph all_reduce_sum 自定义算子注册骨架缺失

| 属性 | 值 |
|------|-----|
| **JSON Path** | `framework_layer.data_flow_contracts.tp_layer_interface_contracts.cuda_graph_execution_contract` |
| **严重等级** | 🔴 Critical |

**问题描述**：

图谱声明了 TP=4 CUDA Graph 的阻塞原因是 "Dynamo RNG 重编译崩溃"，修复方案为 "sglang 切图: torch.fx.split_module 在 all_reduce_sum 处拆分 FX 图"。但 `all_reduce_sum` 的 `@torch.library.custom_op` 注册签名、Fake Tensor 形状推导逻辑完全缺失。Agent 不知道：
- `all_reduce_sum` custom op 的完整注册代码骨架
- `register_fake` 实现中 FakeTensor 的形状推导规则
- `split_module` 的具体切分点和切分后的子图组装方式

**必须补充的规格**：
```python
@torch.library.custom_op("meta_infer::all_reduce_sum", mutates_args=())
def all_reduce_sum(x: torch.Tensor) -> torch.Tensor:
    ...

@torch.library.register_fake("meta_infer::all_reduce_sum")
def _all_reduce_sum_fake(x):
    return torch.empty_like(x)
```

---

### GAP-7: Sequence.block_table 的固定容量与动态扩展语义未定义

| 属性 | 值 |
|------|-----|
| **JSON Path** | `framework_layer.data_flow_contracts.paged_kv_cache_contract.kv_cache_format.block_table` |
| **严重等级** | 🔴 High |

**问题描述**：

`block_table` 定义为 `[1, max_blocks] int32 — 固定 shape`，但 `max_blocks` 的值如何确定？Agent 可能不知道 `max_blocks = ceil(max_model_len / block_size)`。此外，"固定 shape，prefill 时分配，decode 时不变" 与 `can_append_one_more` 的扩展语义矛盾——如果 decode 时可以扩展 block_table，它的 shape 怎么是"固定不变"的？

**必须补充的规格**：
```python
max_blocks = (config.max_position_embeddings + block_size - 1) // block_size
# block_table 预分配为 max_blocks，有效条目数为 num_blocks
# decode 扩展时在 block_table 的 [num_blocks] 位置追加新 block_id
```

---

### GAP-8: decode 路径中 kv_len 的跨层共享与更新时序未定义

| 属性 | 值 |
|------|-----|
| **JSON Path** | `framework_layer.data_flow_contracts.paged_kv_cache_contract.decode_kv_write` + `decode_kv_len_reading` |
| **严重等级** | 🔴 High |

**问题描述**：

`decode_kv_write` 描述：`_kv_len_gpu[0] += 1`。但关键问题未说明：
- `_kv_len_gpu` 是 attention 层的一个属性，但 decode step 需要在**所有层** forward 完成后才 batch 读取 `_kv_len_gpu[0].item()`。那么在某一层内 `_kv_len_gpu += 1` 后，同一 decode step 的后续层应该读到的 kv_len 是多少？是已自增还是未自增？
- 这涉及 KV cache 写入的 slot 计算——第 N 层写入的 slot 应该基于当前的 `_kv_len_gpu` 值，该值是否在每层 attention forward 后自增？如果是，那 36 层的 `_kv_len_gpu` 应为同一值（在所有层写入后统一 +1）还是各自独立维护？
- 当前描述暗示每层维护独立的 `_kv_len_gpu`，且所有层写入同一 slot（因为所有层的 `_kv_len_gpu` 在 decode step 开始时相同），但这一点没有明确说明。

**必须补充的规格**：
```
每层维护独立的 _kv_len_gpu，初始值相同（均为当前 KV 长度）。
decode step 中：
  各层 forward_decode 内部: slot = _kv_len_gpu[0]; 写入 KV; _kv_len_gpu[0] += 1
  所有层完成后: 统一 .item() 读取任意层 _kv_len_gpu（它们值相同，均为 kv_len + 1）
```

---

### GAP-9: QwenMLPTP 中 silu_and_mul 的输入/输出维度链路未完整闭合

| 属性 | 值 |
|------|-----|
| **JSON Path** | `framework_layer.data_flow_contracts.tp_layer_interface_contracts.qwen3_kernel_contracts.silu_and_mul` |
| **严重等级** | 🔴 High |

**问题描述**：

`silu_and_mul` 的 constraint 描述 "input 为 MergedColumnParallelLinear 输出 [B, S, 2*intermediate/tp]（前 gate 后 up）"。但 `intermediate/tp` 中的 `intermediate` 是指 `intermediate_size`（Qwen3-8B 为 12800）还是经过 GQA 调整后的值？Agent 不知道 `intermediate_size` 这个具体数字，因为 blueprint 中没有任何模型具体维度参数（如 Qwen3-8B 的 `hidden_size=4096, intermediate_size=12800, num_hidden_layers=36` 等）。没有这些数字，Agent 无法写出正确的张量形状代码。

**必须补充的规格**：
```json
"qwen3_8b_model_dims": {
    "hidden_size": 4096,
    "intermediate_size": 12800,
    "num_attention_heads": 32,
    "num_key_value_heads": 8,
    "num_hidden_layers": 36,
    "head_dim": 128,
    "vocab_size": 151936,
    "max_position_embeddings": 32768,
    "rope_theta": 1000000.0
}
```

---

### GAP-10: rotary_embedding 的 cos_sin_cache 创建时机与 GPU 迁移逻辑缺失

| 属性 | 值 |
|------|-----|
| **JSON Path** | `framework_layer.data_flow_contracts.tp_layer_interface_contracts.qwen3_kernel_contracts.rotary_embedding` |
| **严重等级** | 🔴 High |

**问题描述**：

"cos_sin_cache_strategy" 描述了模块级 `_cos_sin_cache_registry` dict 共享、CPU 创建、lazy 首次 forward 移到 GPU。但 Agent 不知道：
- cos_sin_cache 的创建函数签名和调用位置（是在 `__init__` 还是首次 `forward`？）
- lazy GPU 迁移的具体触发条件（检查 `_cos_sin_cache_gpu is None`？检查 device 属性？）
- `rotary_embedding` wrapper 函数的完整签名（输入/输出、in-place 语义）

**必须补充的规格**：
```python
def _get_cos_sin_cache(max_position_embeddings, head_dim, device):
    key = (max_position_embeddings, head_dim)
    if key not in _cos_sin_cache_registry:
        cache = make_cos_sin_cache(max_position_embeddings, head_dim)  # CPU
        _cos_sin_cache_registry[key] = cache
    cache = _cos_sin_cache_registry[key]
    if _cos_sin_cache_gpu is None:
        _cos_sin_cache_gpu = cache.to(device)
    return _cos_sin_cache_gpu
```

---

## 2. 🟡 Ambiguous Descriptions (信息熵不足警告项)

### AMB-1: BlockManager 哈希算法未指定

| 属性 | 值 |
|------|-----|
| **JSON Path** | `framework_layer.components[2].api_spec.compute_hash` |
| **严重等级** | 🟡 Medium |

"prefix caching — 相同前缀共享 KV block"，仅给出签名 `compute_hash(token_ids: tuple[int,...]) -> int`。Agent 可能使用 Python 内置 `hash()`（跨进程不稳定），或实现 rolling hash（不知道具体参数），或简单拼接。缺少具体哈希策略（如 `hash(tuple(token_ids[-block_size:]))` 做滑动窗口哈希？），以及 `hash_to_block_id` 映射表的结构定义。

---

### AMB-2: decode 路径 prefill/decode 混合批次的组装规则缺失

| 属性 | 值 |
|------|-----|
| **JSON Path** | `framework_layer.data_flow_contracts.scheduler_to_runner.schedule_algorithm` |
| **严重等级** | 🟡 Medium |

`phase1_prefill` 和 `phase2_decode` 被描述为互斥的二阶段，但 Agent 不知道：同一 batch 内是否允许 prefill 序列和 decode 序列混合？如果允许，`max_num_batched_tokens` 是否对 prefill 和 decode 分别计算？`sampling` 如何区分 prefill token（取 `[:, -1, :]`）和 decode token（取 `[:, 0, :]`）？

---

### AMB-3: fused_add_rms_norm 残差链的层间权重拓扑不透明

| 属性 | 值 |
|------|-----|
| **JSON Path** | `framework_layer.data_flow_contracts.tp_layer_interface_contracts.qwen3_kernel_contracts.fused_add_rms_norm` |
| **严重等级** | 🟡 Medium |

描述为 "post_mlp 调用使用下一层 input_layernorm.weight"，但 "下一层" 是字面意义的 layer[i+1] 还是 layer[i] 自己的 post_attention_layernorm 或 mlp 后的 norm？vLLM 的 residual chaining 模式（residual 作为传播载体，每层内两次 fused_add_rms_norm）没有以伪代码明确：
```python
# 正确的模式（缺失于图谱）:
residual = hidden_states  # 或 None
hidden_states, residual = fused_add_rms_norm(hidden_states, residual, self.input_layernorm.weight, eps)
# ... attention ...
hidden_states, residual = fused_add_rms_norm(hidden_states, residual, self.post_attention_layernorm.weight, eps)
# ... mlp ...
```

---

### AMB-4: torch.compile swap_mechanism 的实现细节缺失

| 属性 | 值 |
|------|-----|
| **JSON Path** | `framework_layer.data_flow_contracts.torch_compile_contract.forward_decode_design.swap_mechanism` |
| **严重等级** | 🟡 Medium |

描述为 "CUDA_GRAPH=1 时 _setup_cuda_graph_piecewise 将 layer.forward_decode 替换为 torch.compile(forward_decode_graph)"。Agent 不知道该替换发生在哪个层级：在 QwenDecoderLayerTP 的 `__init__`？在 ModelRunner 的 `__init__`？使用 `setattr(layer, 'forward_decode', ...)` 或 `types.MethodType`？

---

### AMB-5: custom_ar_all_reduce 的 Gloo IPC handle exchange 协议未展开

| 属性 | 值 |
|------|-----|
| **JSON Path** | `framework_layer.data_flow_contracts.tp_layer_interface_contracts.qwen3_kernel_contracts.custom_ar_all_reduce` |
| **严重等级** | 🟡 Medium |

描述为 "需要 gloo ProcessGroup 做 IPC handle exchange"，但 Agent 不知道 exchange 的具体数据格式、调用顺序、以及超时/错误处理。FM-005 提到 `dist.all_gather_object` 需要 gloo backend，但没有 exchange 的伪代码。

---

### AMB-6: flash_attn_with_kvcache custom op 的注册模板不完整

| 属性 | 值 |
|------|-----|
| **JSON Path** | `framework_layer.data_flow_contracts.flash_attention_integration_contract.decode_path.custom_op_registration` |
| **严重等级** | 🟡 Medium |

描述为 `@torch.library.custom_op('meta_infer::flash_attn_with_kvcache', mutates_args=()) + register_fake`，但没有给出：
- 函数参数列表和类型
- `register_fake` 的返回值形状推导
- 导入方式（`from engine.kernels.custom_ops import flash_attn_with_kvcache_op`）

---

### AMB-7: Prefill 中 kv_len 的含义在 paged 和 contiguous 场景下不同

| 属性 | 值 |
|------|-----|
| **JSON Path** | `framework_layer.data_flow_contracts.tp_layer_interface_contracts.qwen3_tp_model_interfaces.attention` |
| **严重等级** | 🟡 Medium |

图谱提到 decode 使用 `flash_attn_with_kvcache`，但 prefill 使用 `flash_attn_varlen_func`。prefill 时 `_kv_len_gpu` 值是多少？是 0？等于 prefill token 数？它如何与 `flash_attn_varlen_func` 的 `cu_seqlens_k` 关联？prefill 的 KV 写入后 `_kv_len_gpu` 是否更新？

---

### AMB-8: 双轨块大小隔离缺乏代码级强制执行

| 属性 | 值 |
|------|-----|
| **JSON Path** | `framework_layer.components[0]._dual_track_note` |
| **严重等级** | 🟡 Medium |

AGENT_SKILL.md 用 "严禁 BlockManager API 接入 TP Runner" 约束 Agent，但 blueprint JSON 中没有相应的接口隔离设计（如抽象基类、assertion 守卫、或类型系统强制）。Agent 可能在 TP Runner 的 prefill 中错误调用 `BlockManager.allocate()` 而得不到任何运行时错误提示。

---

### AMB-9: ModelRunner 中 RealModelRunner vs TP Runner 的路由分支逻辑未展开

| 属性 | 值 |
|------|-----|
| **JSON Path** | `framework_layer.components[3].tp_runner_actual_flow` |
| **严重等级** | 🟡 Low |

`tp_runner_actual_flow` 描述了 TP Runner 的实际流程（prefill → model.forward → layer.forward → decode → sample），但没有说明在 `LLMEngine.step()` 中如何从 `Scheduler` 输出过渡到 `ModelRunner` 调用。`input_ids` 的 batch 组装（padding/stacking）逻辑是 Agent 必须自行推断的。

---

### AMB-10: KV 占位张量 (placeholder) 初始化未指定

| 属性 | 值 |
|------|-----|
| **JSON Path** | `framework_layer.components[1]` (KVMemoryPool: "提供可选 GPU KV 占位张量") + `paged_kv_cache_contract.kv_cache_format` |
| **严重等级** | 🟡 Low |

TP Runner 使用 paged KV cache (`_key_cache[num_blocks, 256, num_kv_heads, head_dim]`)，但这些 tensor 是在哪里创建的？KVMemoryPool 还是 QwenAttentionTP 的 `__init__`？初始值是什么（全零？`torch.empty`？）？Agent 可能分配错误 dtype 或 device。

---

## 3. 🟢 Verdict (最终图纸判决)

### 信息完备度量化评估

| 系统模块 | 完备度 | 关键缺口数 |
|---------|--------|-----------|
| 推理框架调度层 (Scheduler / BlockManager / KVMemoryPool) | **55%** | GAP-1, GAP-5, GAP-7, AMB-1, AMB-2 |
| 框架层增量修改 (P0 KV / P2 Compile / P3-FA) | **60%** | GAP-2, GAP-3, GAP-4, AMB-3, AMB-6, AMB-7 |
| Kernel 层 7 大标品替换 | **70%** | GAP-10, AMB-5 |
| CUDA Graph 静态执行契约 | **40%** | GAP-6, AMB-4 |
| 模型维度具体参数 (hidden_size 等) | **0%** | GAP-9 |

### 总体完备度：**48%**

### 最终结论：**不允许直接交付给全新 Agent 闭环开工**

当前图谱在概念层面记录了架构决策和关键约束，但缺乏足够的物理参数、伪代码骨架和控制流细节，使 Agent 能够"照图施工"。特别是以下结构性缺陷无法仅靠 Agent 推理弥补。

### Top 3 阻断项（必须修复）

| 排名 | 阻断项 | 理由 |
|------|--------|------|
| **#1** | **GAP-9: 缺少 Qwen3-8B 具体模型维度参数** | Agent 连 `hidden_size=4096`、`num_hidden_layers=36` 都不知道，无法写出任何一行正确的张量形状代码。这是"无法开工"级别的阻断。 |
| **#2** | **GAP-2 + GAP-4: flash_attn_varlen_func 的 cu_seqlens 构造 + prefill slot_mapping 算法** | 这两个是 prefill 路径的核心数据准备逻辑，涉及从 Sequence.block_table 到 ragged tensor 的完整转换链路。缺少它们，prefill 路径 100% 会 shape mismatch 崩溃。 |
| **#3** | **GAP-6: CUDA Graph 自定义算子注册骨架** | TP=4 CUDA Graph 是整个 Stage D 的核心目标，没有 `all_reduce_sum` 的 custom_op 注册 + register_fake 骨架，Agent 不可能独立实现 split_graph 方案。 |

### 修复优先级建议

1. **P0 (阻塞开工)**：在 blueprint 中新增 `"qwen3_8b_model_dims"` 节点，补充全部模型维度参数。
2. **P0 (阻塞 prefill)**：在 `flash_attention_integration_contract.prefill_path` 中补充 `cu_seqlens_q`/`cu_seqlens_k` 构造伪代码，在 `paged_kv_cache_contract.prefill_kv_write` 中补充 `slot_mapping` 计算方法。
3. **P0 (阻塞 CUDA Graph)**：在 `cuda_graph_execution_contract` 中补充 `all_reduce_sum` custom op 注册 + register_fake 完整代码模板。
4. **P1 (高优先级)**：明确 KVMemoryPool/BlockManager 的职责边界和 Scheduler 的数据查询链路（GAP-1, GAP-5）。
5. **P2 (中优先级)**：补充 `fused_add_rms_norm` 残差链伪代码、decode forward 统一签名、`compute_hash` 具体算法。

---

*审计完成。当前蓝图是从"有经验的开发者对照已有源码维护系统"的参考手册，而非"从零重构"的施工图纸。要升级为后者，至少需要补充以上 Top 3 阻断项中的物理参数与伪代码骨架。*
