# Qwen Dense 模型 TP 切分实现知识手册（面向当前极简框架）

本文目标：从 `nano-sglang` 与 `mini-sglang` 中抽取可直接迁移到当前 `meta-infer/engine` 的 **Qwen Dense（非 MoE）TP 实现知识**，并给出源码路径与关键代码片段。  
适用范围：`tp_size = N`，`N ∈ {1,2,4,8}`（同一套代码参数化，不按 N 分叉代码）。

---

## 1. TP 总体设计（先立协议，再改模型）

### 1.1 统一结论
- TP 核心不是“把模型包一层”，而是同时改 4 件事：
  - 进程组初始化（rank/size/group）
  - 线性层与 embedding/lm_head 的分片形状
  - 权重加载时按同样规则切片
  - 前向中的通信（`all_reduce` / `all_gather`）
- Qwen Dense 可直接复用 LLaMA/Qwen2/Qwen3 的通用 TP 模式：  
  `QKV并行 + O/Down行并行 + Embedding/LMHead词表并行`。

### 1.2 关键参考路径
- `meta-infer/ref_projects/nano-sglang/python/sglang/srt/managers/router/model_runner.py`
- `meta-infer/ref_projects/nano-sglang/python/sglang/srt/parallel_utils/parallel_state.py`
- `meta-infer/ref_projects/mini-sglang/python/minisgl/engine/engine.py`
- `meta-infer/ref_projects/mini-sglang/python/minisgl/distributed/info.py`
- `meta-infer/ref_projects/mini-sglang/python/minisgl/distributed/impl.py`

### 1.3 关键代码（初始化）
```python
# nano-sglang: model_runner.py
torch.distributed.init_process_group(
    backend="nccl",
    world_size=self.tp_size,
    rank=self.tp_rank,
    init_method=f"tcp://127.0.0.1:{self.nccl_port}",
)
initialize_model_parallel(tensor_model_parallel_size=self.tp_size)
```

```python
# mini-sglang: engine.py
set_tp_info(rank=config.tp_info.rank, size=config.tp_info.size)
torch.distributed.init_process_group(
    backend="nccl",  # 或 gloo + pynccl
    rank=config.tp_info.rank,
    world_size=config.tp_info.size,
    init_method=config.distributed_addr,
)
```

---

## 2. Qwen Dense 的层切分规则（最关键）

Qwen2/Qwen3 在 mini-sglang 中直接用了统一算子：
- `LinearQKVMerged`
- `LinearOProj`
- `LinearColParallelMerged`（gate/up）
- `LinearRowParallel`（down）
- `VocabParallelEmbedding`
- `ParallelLMHead`

参考路径：
- `meta-infer/ref_projects/mini-sglang/python/minisgl/models/qwen2.py`
- `meta-infer/ref_projects/mini-sglang/python/minisgl/models/qwen3.py`
- `meta-infer/ref_projects/mini-sglang/python/minisgl/models/utils.py`
- `meta-infer/ref_projects/mini-sglang/python/minisgl/layers/linear.py`
- `meta-infer/ref_projects/mini-sglang/python/minisgl/layers/embedding.py`
- `meta-infer/ref_projects/mini-sglang/python/minisgl/layers/attention.py`

### 2.1 Attention：QKV 与 KV 头复制/切分
```python
# minisgl/layers/linear.py
local_num_qo = div_even(num_qo_heads, tp_info.size)
local_num_kv = div_even(num_kv_heads, tp_info.size, allow_replicate=True)
local_osize = (local_num_qo + 2 * local_num_kv) * head_dim
```

```python
# minisgl/layers/attention.py
self.num_qo_heads = div_even(num_qo_heads, tp_size)
self.num_kv_heads = div_even(num_kv_heads, tp_size, allow_replicate=True)
q, k, v = qkv.split([self.qo_attn_dim, self.kv_attn_dim, self.kv_attn_dim], dim=-1)
```

知识点：
- `num_qo_heads` 必须能被 `tp_size` 整除。
- `num_kv_heads` 若小于 `tp_size`，采用复制（GQA 场景常见）；否则按头切分。

### 2.2 MLP：gate/up 列并行，down 行并行
```python
# minisgl/models/utils.py
self.gate_up_proj = LinearColParallelMerged(
    config.hidden_size, [config.intermediate_size, config.intermediate_size], has_bias=False
)
self.down_proj = LinearRowParallel(config.intermediate_size, config.hidden_size, has_bias=False)
```

知识点：
- 列并行（输出切分）常用于 FFN 扩维支路；
- 行并行（输入切分）常用于投回 hidden_size，结果需要跨 rank 求和。

### 2.3 O 投影与 RowParallel 的通信
```python
# minisgl/layers/linear.py
y = F.linear(x, self.weight, self.bias)
if self._tp_size > 1:
    y = self._comm.all_reduce(y)
```

知识点：
- `o_proj` 与 `down_proj` 通常是“每卡算部分，再 all_reduce 得到完整输出”。

---

## 3. Embedding / LM Head 的词表并行

参考路径：
- `meta-infer/ref_projects/mini-sglang/python/minisgl/layers/embedding.py`
- `meta-infer/ref_projects/nano-sglang/python/sglang/srt/layers/vocab_parallel_embedding.py`
- `meta-infer/ref_projects/nano-sglang/python/sglang/srt/layers/logits_processor.py`

### 3.1 Embedding：按 vocab 维切分 + all_reduce 合并
```python
# minisgl/layers/embedding.py
self.num_embeddings_tp = div_ceil(num_embeddings, self.tp_size)
start_idx = self.num_embeddings_tp * tp_rank
finish_idx = min(start_idx + self.num_embeddings_tp, num_embeddings)
self.vocab_range = (start_idx, finish_idx - start_idx)
...
return self._comm.all_reduce(y) if self.tp_size > 1 else y
```

### 3.2 LM Head：每卡算局部 logits，再 all_gather 拼接
```python
# minisgl/layers/embedding.py
logits = F.linear(x, module.weight, self.bias)
output_tensor = self._comm.all_gather(logits)
...
return output_tensor[:, : self.num_embeddings]
```

```python
# nano-sglang/logits_processor.py
last_logits = torch.matmul(last_hidden, weight.T)
if self.tp_size > 1:
    last_logits = tensor_model_parallel_all_gather(last_logits)
last_logits = last_logits[:, : self.config.vocab_size]
```

知识点：
- TP 下采样前要先全局 vocab 聚合（`all_gather`），再 `[:vocab_size]` 裁掉 padding。

---

## 4. 权重加载：切分规则必须与层并行规则完全一致

参考路径（最关键）：
- `meta-infer/ref_projects/mini-sglang/python/minisgl/models/weight.py`
- `meta-infer/ref_projects/nano-sglang/python/sglang/srt/models/llama2.py`

### 4.1 mini-sglang：统一 sharding + merge
```python
# minisgl/models/weight.py
_SPLIT_DIM_0 = [".q_proj", ".k_proj", ".v_proj", ".gate_proj", ".up_proj"]
_SPLIT_DIM_1 = [".o_proj", ".down_proj"]
...
if any(key.count(sub) for sub in _SPLIT_DIM_0):
    return value.chunk(n, dim=0)[r].clone()
elif any(key.count(sub) for sub in _SPLIT_DIM_1):
    return value.chunk(n, dim=1)[r].clone()
```

```python
# minisgl/models/weight.py: 融合加载
_MERGE_GROUPS = {
    ".q_proj": (".qkv_proj", ("q", "k", "v")),
    ".k_proj": (".qkv_proj", ("q", "k", "v")),
    ".v_proj": (".qkv_proj", ("q", "k", "v")),
    ".gate_proj": (".gate_up_proj", ("gate", "up")),
    ".up_proj": (".gate_up_proj", ("gate", "up")),
}
```

### 4.2 nano-sglang：参数对象挂 `weight_loader`
```python
# nano-sglang/llama2.py
stacked_params_mapping = [
    ("qkv_proj", "q_proj", "q"),
    ("qkv_proj", "k_proj", "k"),
    ("qkv_proj", "v_proj", "v"),
    ("gate_up_proj", "gate_proj", 0),
    ("gate_up_proj", "up_proj", 1),
]
...
weight_loader = param.weight_loader
weight_loader(param, loaded_weight, shard_id)
```

知识点：
- “层怎么并行”与“权重怎么切片”必须一一对应，否则 shape 或语义错。
- Qwen Dense 推荐直接复用这套：
  - q/k/v -> 合并到 `qkv_proj`
  - gate/up -> 合并到 `gate_up_proj`
  - o/down -> 输入维切分
  - embed/lm_head -> vocab 切分

---

## 5. KV Cache 与 TP 的尺寸联动（Dense 也必须做）

参考路径：
- `meta-infer/ref_projects/nano-sglang/python/sglang/srt/managers/router/model_runner.py`
- `meta-infer/ref_projects/mini-sglang/python/minisgl/engine/engine.py`

关键代码：
```python
# nano-sglang/model_runner.py
head_num = self.model_config.num_key_value_heads // self.tp_size
...
self.token_to_kv_pool = TokenToKVPool(
    ...,
    head_num=self.model_config.num_key_value_heads // self.tp_size,
)
```

```python
# mini-sglang/engine.py
cache_per_page = (
    2 * head_dim
    * div_even(num_kv_heads, tp_size, allow_replicate=True)
    * page_size * dtype_size * num_layers
)
```

知识点：
- TP 后每卡 KV 头数变化，直接影响显存估算和池大小；
- 若 `num_kv_heads < tp_size`，按复制逻辑估算本卡 KV 大小。

---

## 6. 给当前 `meta-infer` 的最小可落地改造清单（Qwen Dense TP）

建议改造顺序：

1) **并行上下文**
- 新增 `tp_size/tp_rank` 参数与进程组初始化（`nccl`）。
- 提供 `all_reduce/all_gather` 包装函数（单卡旁路）。

2) **TP 线性层**
- 在 `meta-infer/engine` 增加：
  - `LinearQKVMerged`
  - `LinearColParallelMerged`
  - `LinearRowParallel`
  - `LinearOProj`
- 前向通信规则对齐 mini/nano。

3) **Qwen Dense 模型定义**
- 新建 QwenDense 模型类（先对齐 Qwen3 结构）：
  - Attention 使用 `qkv_proj + o_proj`
  - MLP 使用 `gate_up_proj + down_proj`
  - Embedding / LMHead 使用词表并行。

4) **权重加载器**
- 按 `q/k/v` 与 `gate/up` 融合映射；
- 按 dim0/dim1/vocab 做 rank 切片；
- 对 `num_kv_heads < tp_size` 加复制切片逻辑。

5) **KV Pool 与 Logits**
- KV pool 容量估算改为依赖 local kv-head；
- logits 在 TP>1 时 `all_gather` 后再采样。

---

## 7. 实施时的高频坑位检查

- `num_attention_heads % tp_size != 0`：直接报错，不能静默降级。
- `num_kv_heads` 与 `tp_size` 的关系必须走“切分/复制”二分逻辑。
- `lm_head` 采样前未 `all_gather`：会导致词表不完整、采样错误。
- 权重 merge 后再 shard / 先 shard 再 merge 顺序不一致：会错形状或错语义。
- KV 缓存估算仍用全局 `num_kv_heads`：会高估内存并导致调度错误。

---

## 8. Qwen Dense TP 参考源码总索引

### nano-sglang
- `meta-infer/ref_projects/nano-sglang/python/sglang/srt/managers/router/model_runner.py`
- `meta-infer/ref_projects/nano-sglang/python/sglang/srt/parallel_utils/parallel_state.py`
- `meta-infer/ref_projects/nano-sglang/python/sglang/srt/layers/linear.py`
- `meta-infer/ref_projects/nano-sglang/python/sglang/srt/layers/vocab_parallel_embedding.py`
- `meta-infer/ref_projects/nano-sglang/python/sglang/srt/layers/logits_processor.py`
- `meta-infer/ref_projects/nano-sglang/python/sglang/srt/models/llama2.py`
- `meta-infer/ref_projects/nano-sglang/python/sglang/srt/models/mixtral.py`（主要借鉴其 TP 通信与 KV/GQA 处理模式）

### mini-sglang
- `meta-infer/ref_projects/mini-sglang/python/minisgl/engine/engine.py`
- `meta-infer/ref_projects/mini-sglang/python/minisgl/distributed/info.py`
- `meta-infer/ref_projects/mini-sglang/python/minisgl/distributed/impl.py`
- `meta-infer/ref_projects/mini-sglang/python/minisgl/layers/linear.py`
- `meta-infer/ref_projects/mini-sglang/python/minisgl/layers/embedding.py`
- `meta-infer/ref_projects/mini-sglang/python/minisgl/layers/attention.py`
- `meta-infer/ref_projects/mini-sglang/python/minisgl/models/qwen2.py`
- `meta-infer/ref_projects/mini-sglang/python/minisgl/models/qwen3.py`
- `meta-infer/ref_projects/mini-sglang/python/minisgl/models/utils.py`
- `meta-infer/ref_projects/mini-sglang/python/minisgl/models/weight.py`

---

如果下一步你要我直接改 `meta-infer/engine` 加入 Qwen Dense TP，我会按本手册第 6 节的顺序逐项落地，并优先先跑通 `tp=1 -> tp=2` 的一致性测试，再扩展到 `tp=4/8`。
