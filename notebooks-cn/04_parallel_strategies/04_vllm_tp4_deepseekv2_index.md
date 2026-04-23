# vLLM 中 TP 切分与模型加载知识索引（面向 DeepSeekV2 `TP=4`）

目标：基于你指定的目录  
- `meta-infer/ref_projects/vllm/vllm/model_executor/model_loader`  
- `meta-infer/ref_projects/vllm/vllm/model_executor/layers`  
- `meta-infer/ref_projects/vllm/vllm/model_executor/kernels`  
以及这些文件直接 import 的 TP 相关依赖，整理出一份可直接指导后续在极简框架实现 DeepSeekV2 `TP=4` 的索引文档。

---

## 1. 三个目录的职责边界（先建立心智模型）

- **`model_loader/`**：解决“从 checkpoint 到本 rank 参数张量”的问题（下载、遍历 shard、按 TP rank 切分、`weight_loader` 调度）。
- **`layers/`**：解决“模型结构如何按 TP 切分与通信”的问题（`ColumnParallel`、`RowParallel`、`QKVParallel`、`VocabParallel`、MoE 归约等）。
- **`kernels/`**：解决“本 rank 上算子怎么高效执行”的问题（Triton/Cutlass/FlashInfer 等线性核）；通常**不直接管理 TP 通信组**。

---

## 2. `model_executor/model_loader`：必须吃透的文件

## 2.1 通用加载与 `weight_loader`

- `model_loader/base_loader.py`  
  - 抽象基类与 `load_weights` 入口（约 `37-64`）。

- `model_loader/default_loader.py`  
  - 默认加载路径，最终 `model.load_weights(...)`（约 `368-381`）。  
  - EP/DP/PCP/TP 混合时的 rank 展平逻辑（约 `334-357`）。

- `model_loader/weight_utils.py`  
  - `default_weight_loader`（`1361+`）  
  - `row_parallel_weight_loader`（`1382-1394`）  
  - `sharded_weight_loader(shard_axis)`（`1400-1410`）  
  这是 TP shard 切分最直接的模板：`tp_rank * shard_size` + `narrow(...)`。

## 2.2 预切分 checkpoint（最适合大模型 TP）

- `model_loader/sharded_state_loader.py`  
  - 文档说明：每个 worker 只读自己 shard（`29-34`）。  
  - `load_weights` 按 `get_tensor_model_parallel_rank()` 拼文件模式（`110-123`）。  
  - 仅支持 pre-sharded checkpoint（`130-135`）。

## 2.3 量化加载中 TP 切分逻辑

- `model_loader/bitsandbytes_loader.py`  
  - 引入 `get_tensor_model_parallel_rank/world_size`（文件头 import）。  
  - 按模块类型分类 row/column/unsharded（`492+`）。  
  - 实际 shard 索引计算（`338-417` 一段）。

## 2.4 其他与 TP 强关联

- `model_loader/tensorizer_loader.py` 与 `tensorizer.py`  
  - `tensor_parallel_size > 1` 时要求 tensorizer URI/shard 约束（如 `tensorizer.py` 约 `312+`）。

- `model_loader/ep_weight_filter.py`  
  - DP+EP 时专家权重过滤（注释已明确和 `FusedMoE.weight_loader` 配合）。

---

## 3. `model_executor/layers`：TP 结构定义与权重装载协议

## 3.1 TP 线性层主文件（核心）

- `layers/linear.py`
  - `ColumnParallelLinear`：`406+`
  - `MergedColumnParallelLinear`：`603+`
  - `QKVParallelLinear`：`965+`
  - `RowParallelLinear`：`1371+`

关键点：
- 列并行切输出维，行并行切输入维并在必要时 `all_reduce`。
- `QKVParallelLinear` 内置 KV head 复制场景处理（当 `num_kv_heads < tp_size`）。
- 这些层在 `weight_loader` 流程中依赖参数对象的 `load_*` 方法。

## 3.2 词表切分

- `layers/vocab_parallel_embedding.py`
  - `VocabParallelEmbedding`：`192+`
  - `ParallelLMHead`：`503+`
  - 词表范围切分、mask、`tensor_model_parallel_all_reduce/all_gather` 协议。

## 3.3 DeepSeekV2（目标模型）直接参考

- `model_executor/models/deepseek_v2.py`（虽然不在你列的三个目录中，但它正是这些层/loader 的汇合点）
  - 头部 import 说明依赖关系（`39-84`）：  
    `vllm.distributed`、`layers.linear`、`vocab_parallel_embedding`、`weight_utils`、`SharedFusedMoE` 等。
  - Attention TP 维度逻辑与 QKV/OProj（`130-191`）。
  - MLP 中 `MergedColumnParallelLinear` + `RowParallelLinear`（`211-238`）。
  - MoE 中 `tp_size/tp_rank` 与 EP 组（`250-258`）。
  - `load_weights`（`1470-1687`）是 DeepSeekV2 TP 加载最关键实现：  
    - `stacked_params_mapping` 融合映射（`1474-1499`）  
    - `expert_params_mapping`（`1503-1515`）  
    - 通用 stack 参数 `weight_loader(param, loaded_weight, shard_id)`（`1536-1571`）  
    - 专家参数按 expert-aware loader 装载（`1620-1655`）  
    - fallback 到 `default_weight_loader`（`1679-1683`）。

---

## 4. import 依赖里必须看的 TP 文件

## 4.1 分布式 TP API（被 layers/model_loader 共同依赖）

- `vllm/distributed/parallel_state.py`
  - `get_ep_group()`：`1254-1260`
  - `get_tensor_model_parallel_world_size()`：`1827-1829`
  - `get_tensor_model_parallel_rank()`：`1832-1834`

- `vllm/distributed/communication_op.py`
  - `tensor_model_parallel_all_reduce`：`12-14`
  - `tensor_model_parallel_all_gather`：`17-21`

## 4.2 参数对象与 weight_loader 协议

- `model_executor/parameter.py`
  - `BasevLLMParameter` 绑定 `tp_rank/tp_size`：`41-67`
  - `_ColumnvLLMParameter.load_column_parallel_weight`：`148-154`
  - `_ColumnvLLMParameter.load_merged_column_weight`：`156-177`
  - `_ColumnvLLMParameter.load_qkv_weight`：`178-201`
  - `RowvLLMParameter`（行并行参数）：`204+`

这是把“layer 的 shard 语义”落实到“参数切片复制”的桥梁。

---

## 5. `model_executor/kernels`：与 TP 的真实关系

- `kernels/linear/__init__.py`：线性核注册与选择（`22+` 大段 import、`301+` 选择函数）。
- `kernels/linear/scaled_mm/ScaledMMLinearKernel.py`：核接口抽象（`55+`）。
- `kernels/linear/scaled_mm/triton.py`：某具体核实现（主要关注本 rank 张量处理，不管理 TP group）。

结论：
- kernel 文件一般不写 `get_tp_group().all_reduce()` 这类组通信；  
- TP 语义主要在 `layers`/`parameter`/`distributed`，kernel 只消费已经分片好的权重与输入。

---

## 6. 面向 DeepSeekV2 `TP=4` 的最小实现映射（给极简框架）

把当前极简框架接入 TP=4 时，按下面映射抄：

1. **分布式接口层**  
   参考：`distributed/parallel_state.py` + `communication_op.py`  
   实现 `get_tp_rank/get_tp_size/all_reduce/all_gather`。

2. **并行线性层**  
   参考：`layers/linear.py`（`Column/Merged/QKV/Row`）  
   至少实现：  
   - QKV 切分（含 KV replication）  
   - O/down 投影 all-reduce。

3. **词表并行**  
   参考：`layers/vocab_parallel_embedding.py`。

4. **权重加载协议**  
   参考：`parameter.py` + `weight_utils.py` + `deepseek_v2.py::load_weights`。  
   核心：  
   - `stacked_params_mapping`（q/k/v、gate/up 合并）  
   - `expert_params_mapping`（MoE 专家参数映射）  
   - `weight_loader(param, weight, shard_id, ...)`。

5. **MoE 处理**  
   参考：`deepseek_v2.py` 中 `SharedFusedMoE` 路径（`1503+` 与 `1620+`）。  
   第一版可先 TP + 本地专家映射 + 必要 all-reduce，再逐步补 EP/高级调度。

---

## 7. 建议阅读顺序（最快落地 TP=4）

1. `model_executor/models/deepseek_v2.py`（先看 `load_weights` 与 Attention/MLP/MoE 结构）  
2. `model_executor/layers/linear.py`  
3. `model_executor/parameter.py`  
4. `model_loader/weight_utils.py` + `sharded_state_loader.py`  
5. `layers/vocab_parallel_embedding.py`  
6. `distributed/parallel_state.py` + `communication_op.py`  
7. `kernels/linear/*`（最后看，确认算子后端）

---

## 8. 结论

若目标是“后续可直接在极简框架实现 DeepSeekV2 `TP=4`”，最关键不是先改 kernel，而是先把这三层打通：

- **TP 通信与 rank 状态（distributed）**
- **并行层语义与参数装载协议（layers + parameter）**
- **DeepSeekV2 特定的映射加载（models/deepseek_v2.py::load_weights）**

这些都已在上述路径给出明确索引，可作为直接实施清单。

