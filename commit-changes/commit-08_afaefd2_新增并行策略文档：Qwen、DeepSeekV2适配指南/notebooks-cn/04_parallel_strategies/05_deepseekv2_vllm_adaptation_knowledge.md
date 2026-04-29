# DeepSeekV2 在 vLLM 中的适配知识（TP 视角）

本文将 `vllm/model_executor/models/deepseek_v2.py` 中相对 HF 直接加载的核心改造整理为 4 大类，并给出：

- 知识点说明
- 参考源码路径
- 与 TP 直接相关的关键代码片段（简要）

用于后续在极简框架实现 DeepSeekV2 的 TP 版本（尤其 TP=4）。

---

## 1) 并行层被替换成 vLLM 并行层

## 1.1 知识点

- 普通 `nn.Linear` 被替换为 TP 语义明确的层：
  - `QKVParallelLinear`
  - `MergedColumnParallelLinear`
  - `RowParallelLinear`
  - `VocabParallelEmbedding` / `ParallelLMHead`
- 这些层本身内置了：
  - 按 `tp_size` 分片参数形状
  - 前向时 `all_reduce`/`all_gather` 协议
  - `weight_loader` 分片加载接口

## 1.2 参考源码路径

- 模型使用侧：
  - `vllm-v0.15.1-dev/vllm/model_executor/models/deepseek_v2.py`
- 层定义侧：
  - `vllm-v0.15.1-dev/vllm/model_executor/layers/linear.py`
  - `vllm-v0.15.1-dev/vllm/model_executor/layers/vocab_parallel_embedding.py`
- 参数加载协议：
  - `vllm-v0.15.1-dev/vllm/model_executor/parameter.py`

## 1.3 关键代码（TP）

```python
# deepseek_v2.py
self.qkv_proj = QKVParallelLinear(...)
self.o_proj = RowParallelLinear(...)
self.gate_up_proj = MergedColumnParallelLinear(...)
self.down_proj = RowParallelLinear(...)
```

```python
# deepseek_v2.py
tp_size = get_tensor_model_parallel_world_size()
self.num_heads = self.total_num_heads // tp_size
if self.total_num_kv_heads >= tp_size:
    assert self.total_num_kv_heads % tp_size == 0
else:
    assert tp_size % self.total_num_kv_heads == 0
self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
```

```python
# layers/linear.py (RowParallelLinear 语义)
if self.reduce_results and self.tp_size > 1:
    output_ = tensor_model_parallel_all_reduce(output_parallel)
```

---

## 2) 注意力/KV 缓存改成 vLLM 运行时协议

## 2.1 知识点

- 不走 HF 直接 `past_key_values` 使用方式，而是对接 vLLM 运行时注意力与 KV cache 抽象。
- 关键点：
  - `Attention(...)` 统一接入 vLLM attention backend
  - `KVCacheSpec` / `MLAAttentionSpec` 声明缓存布局
  - 对 MLA 路径有专门结构（DeepSeekV2/3 特有）

## 2.2 参考源码路径

- `vllm-v0.15.1-dev/vllm/model_executor/models/deepseek_v2.py`
- 关联接口：
  - `vllm-v0.15.1-dev/vllm/v1/kv_cache_interface.py`

## 2.3 关键代码（TP + KV）

```python
# deepseek_v2.py
from vllm.v1.kv_cache_interface import KVCacheSpec, MLAAttentionSpec

def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec:
    return MLAAttentionSpec(...)
```

```python
# deepseek_v2.py
self.attn = Attention(
    self.num_heads,
    self.head_dim,
    self.scaling,
    num_kv_heads=self.num_kv_heads,
    cache_config=cache_config,
    ...
)
```

```python
# deepseek_v2.py
qkv, _ = self.qkv_proj(hidden_states)
q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
attn_output = self.attn(q, k, v)
output, _ = self.o_proj(attn_output)
```

---

## 3) MoE 路径改成 FusedMoE + EP/TP 组协同

## 3.1 知识点

- MoE 不再是纯 Python 逐专家前向拼接，而是用 `SharedFusedMoE`（fused kernel 路径）。
- 同时引入 EP（Expert Parallel）与 TP 组信息：
  - `tp_rank/tp_size`
  - `ep_group/ep_rank/ep_size`
- 这使 MoE 在大模型推理服务中可扩展且高性能。

## 3.2 参考源码路径

- `vllm-v0.15.1-dev/vllm/model_executor/models/deepseek_v2.py`
- 关联 MoE 层：
  - `vllm-v0.15.1-dev/vllm/model_executor/layers/fused_moe/`
- 分布式组接口：
  - `vllm-v0.15.1-dev/vllm/distributed/parallel_state.py`

## 3.3 关键代码（TP/EP + MoE）

```python
# deepseek_v2.py
self.tp_size = get_tensor_model_parallel_world_size()
self.tp_rank = get_tensor_model_parallel_rank()
self.ep_group = get_ep_group().device_group
self.ep_rank = get_ep_group().rank_in_group
self.ep_size = self.ep_group.size()
```

```python
# deepseek_v2.py
self.gate = GateLinear(...)
self.experts = SharedFusedMoE(...)
```

```python
# deepseek_v2.py
expert_params_mapping = SharedFusedMoE.make_expert_params_mapping(
    self,
    ckpt_gate_proj_name="gate_proj",
    ckpt_down_proj_name="down_proj",
    ckpt_up_proj_name="up_proj",
    num_experts=self.config.n_routed_experts + ...,
    num_redundant_experts=self.num_redundant_experts,
)
```

---

## 4) 权重加载不是 `from_pretrained`，而是映射+分片+专家专用 loader

## 4.1 知识点

- vLLM 的模型层参数不是简单按 HF 名字一一 `copy_`，而是通过：
  1. 名字映射（例如 q/k/v 融合到 qkv）
  2. shard_id 指导切片
  3. 专家参数映射（MoE）
  4. 参数对象的 `weight_loader(...)` 执行最终切分加载
- 这套机制是 TP 可工作的关键，HF 默认加载路径不具备这类统一协议。

## 4.2 参考源码路径

- 模型特化加载逻辑：
  - `vllm-v0.15.1-dev/vllm/model_executor/models/deepseek_v2.py`（`load_weights`）
- 通用 loader：
  - `vllm-v0.15.1-dev/vllm/model_executor/model_loader/default_loader.py`
  - `vllm-v0.15.1-dev/vllm/model_executor/model_loader/sharded_state_loader.py`
  - `vllm-v0.15.1-dev/vllm/model_executor/model_loader/weight_utils.py`
- 参数侧分片行为：
  - `vllm-v0.15.1-dev/vllm/model_executor/parameter.py`

## 4.3 关键代码（映射+分片）

```python
# deepseek_v2.py
stacked_params_mapping = [
    ("gate_up_proj", "gate_proj", 0),
    ("gate_up_proj", "up_proj", 1),
    ("qkv_proj", "q_proj", "q"),
    ("qkv_proj", "k_proj", "k"),
    ("qkv_proj", "v_proj", "v"),
]
```

```python
# deepseek_v2.py
for param_name, weight_name, shard_id in stacked_params_mapping:
    if weight_name not in name:
        continue
    name = name.replace(weight_name, param_name)
    param = params_dict[name]
    weight_loader = param.weight_loader
    weight_loader(param, loaded_weight, shard_id)
```

```python
# deepseek_v2.py (MoE 专家权重专用加载)
weight_loader = typing.cast(Callable[..., bool], param.weight_loader)
success = weight_loader(
    param, weight_to_load, name_mapped,
    shard_id=shard_id, expert_id=expert_id, return_success=True,
)
```

```python
# model_loader/weight_utils.py
def sharded_weight_loader(shard_axis: int):
    def loader(param, loaded_weight):
        tp_rank = get_tensor_model_parallel_rank()
        shard_size = param.data.shape[shard_axis]
        start_idx = tp_rank * shard_size
        loaded_weight = loaded_weight.narrow(shard_axis, start_idx, shard_size)
        return default_weight_loader(param, loaded_weight)
```

---

## 5) 与 HF 直接加载的本质差异总结

- HF 直接加载：  
  - 目标是“模型语义正确 + 通用可用”，默认不负责 TP/EP 服务化细节。
- vLLM DeepSeekV2 适配：  
  - 目标是“可并行切分 + 可服务化调度 + 可高吞吐执行”，所以必须有：
    - TP-aware 层定义
    - KV/cache 运行时协议
    - FusedMoE 与 EP/TP 组协同
    - 统一 `weight_loader` 映射/分片/专家加载机制

---

## 6) 给后续“极简框架接入 TP=4”的直接建议

落地优先顺序：
1. 先对齐 `layers/linear.py` 的 TP 语义（QKV/Row/MergedColumn）。
2. 再实现 `parameter.py` 类似的 `weight_loader` 协议。
3. 参考 `deepseek_v2.py::load_weights` 做名字映射和专家加载。
4. 最后再考虑 kernel 侧替换优化（Triton/Cutlass/FlashInfer）。

这能保证先跑通 TP 正确性，再逐步优化性能。

