# DeepSeekV2（MoE）在极简框架中实现 TP=N 的知识指南（N∈{1,2,4,8}）

目标：整理 nano-sglang 与 mini-sglang 中**可直接迁移**到当前极简推理框架的 TP/MoE 关键机制，支持后续实现 **DeepSeekV2 Chat 16B 的 TP=N 切分**（`N ∈ {1,2,4,8}`）。

---

## 1. 先定边界：TP=N（N∈{1,2,4,8}）里哪些必须做，哪些可后做

### 必做（第一阶段）
- 初始化分布式（4 个进程，1 rank 1 卡），拿到 `tp_rank/tp_size`。
- 线性层切分：`ColumnParallel`、`RowParallel`、`QKVParallel`。
- Embedding/LM Head 词表维切分（forward all-reduce / all-gather）。
- 权重加载按 TP shard 切分（含 QKV fused 和 MoE 专家参数）。
- MoE 路由后输出在 TP 组内归约（all-reduce）。

### 可后做（第二阶段）
- PyNCCL 优化通信、CUDA Graph、FlashInfer 特化路径。
- 更复杂并行（DP/PP/EP 组合）与层级调度。

---

## 2. nano-sglang：TP 与 MoE 的核心实现点

## 2.1 TP 进程组与通信原语

### 关键源码
- `meta-infer/ref_projects/nano-sglang/python/sglang/srt/parallel_utils/parallel_state.py`
  - `initialize_model_parallel()`：`19-85`
  - `get_tensor_model_parallel_world_size/rank`：`111-128`
  - `tensor_model_parallel_all_reduce`：`188-199`
  - `tensor_model_parallel_all_gather`：`201-227`
- `meta-infer/ref_projects/nano-sglang/python/sglang/srt/parallel_utils/utils.py`
  - `divide`：`17-21`
  - `split_tensor_along_last_dim`：`24-49`

### 可迁移结论
- TP 框架最小接口就是：`tp_rank/tp_size + all_reduce + all_gather + split_last_dim`。
- 极简工程可先只做 tensor-parallel group，不启 pipeline group。

---

## 2.2 线性层 TP 切分（最重要）

### 关键源码
- `meta-infer/ref_projects/nano-sglang/python/sglang/srt/layers/linear.py`
  - `ColumnParallelLinear`：`138-232`
  - `MergedColumnParallelLinear`：`235-333`
  - `QKVParallelLinear`：`336-470`
  - `RowParallelLinear`：`473-595`

### 可迁移规则
- **ColumnParallel**：切输出维；forward 默认不通信（可选 gather）。
- **RowParallel**：切输入维；forward 后 `all_reduce`。
- **QKVParallel**：
  - `num_heads = total_q_heads // tp_size`
  - 当 `total_kv_heads < tp_size` 时，KV head 复制（`num_kv_head_replicas` 逻辑，`380-385`）。
  - 权重加载通过 `loaded_shard_id in ["q","k","v"]` 逐段切（`441-463`）。

---

## 2.3 词表切分（Embedding/LM Head）

### 关键源码
- `meta-infer/ref_projects/nano-sglang/python/sglang/srt/layers/vocab_parallel_embedding.py`
  - `VocabParallelEmbedding`：`35-109`
  - `ParallelLMHead`：`112-150`

### 可迁移规则
- Embedding：每 rank 只保留 vocab shard，前向把越界 token mask 为 0，再 `all_reduce` 合并。
- LM Head：每 rank 仅算 local vocab logits，最终 `all_gather` 还原全词表 logits（或只在 rank0 gather）。

---

## 2.4 MoE 与 TP 的耦合

### 关键源码
- `meta-infer/ref_projects/nano-sglang/python/sglang/srt/models/mixtral.py`
  - `MixtralMoE` 初始化与专家分配：`73-115`
  - MoE forward + `tensor_model_parallel_all_reduce`：`116-137`
  - 注意力里 QKV/OProj TP：`140-215`
  - `load_weights` 中 QKV shard 与专家过滤：`339-380`

### 可迁移规则
- 专家按 rank 划分（`np.array_split(..., tp_size)[rank]`，`91-93`）。
- 路由（gate）可 replicated，expert FFN 只算本地专家，再对结果做 TP all-reduce。
- 加载权重时要**跳过非本 rank 专家**（`375-376`），否则参数名不匹配或显存浪费。

---

## 2.5 TP 初始化入口（管理器侧）

### 关键源码
- `meta-infer/ref_projects/nano-sglang/python/sglang/srt/managers/router/model_runner.py`
  - `init_process_group` + warmup all_reduce + `initialize_model_parallel`：`225-237`
  - 模型构建与加载入口：`247-294`
  - KV 池 head_num 按 TP 缩小：`303-320`

### 可迁移规则
- 先 init 分布式，再构建模型并加载 shard。
- KV cache 容量估算中 `num_kv_heads` 必须用 `// tp_size`（或允许复制时做映射）。

---

## 3. mini-sglang：更“工程化”的 TP/MoE 参考

## 3.1 TP 状态与通信抽象

### 关键源码
- `meta-infer/ref_projects/mini-sglang/python/minisgl/distributed/info.py`
  - `DistributedInfo` / `set_tp_info` / `get_tp_info`：`6-31`
- `meta-infer/ref_projects/mini-sglang/python/minisgl/distributed/impl.py`
  - `TorchDistributedImpl`：`25-41`
  - `DistributedCommunicator`：`63-70`
  - `enable_pynccl_distributed`：`73-90`

### 可迁移规则
- 把通信封装成 `comm.all_reduce/all_gather`，业务层不直接写 `dist.*`，后续替换 PyNCCL 更容易。

---

## 3.2 TP 线性层（简洁版本）

### 关键源码
- `meta-infer/ref_projects/mini-sglang/python/minisgl/layers/linear.py`
  - `_LinearTPImpl`：`13-33`
  - `LinearColParallelMerged`：`56-69`
  - `LinearQKVMerged`：`71-89`
  - `LinearOProj`：`91-107`
  - `LinearRowParallel`：`109-127`

### 可迁移规则
- mini-sglang 把维度映射写得非常清晰，适合作为极简工程模板。
- `LinearQKVMerged` 支持 `allow_replicate=True` 的 KV head 复制（`83`）。

---

## 3.3 MoE 层与路由后归约

### 关键源码
- `meta-infer/ref_projects/mini-sglang/python/minisgl/layers/moe.py`
  - `MoELayer`：`9-43`
  - forward 后 `all_reduce`：`45-59`
- `meta-infer/ref_projects/mini-sglang/python/minisgl/moe/fused.py`
  - `FusedMoe.forward`：`230-257`

### 可迁移规则
- MoE 参数按 TP 切的是中间维（`intermediate_size_per_partition`，`33-43`）。
- 每 rank 跑 fused MoE backend 后，再 all-reduce 汇总 hidden states。

---

## 3.4 Qwen3-MoE 组装方式（可映射到 DeepSeekV2）

### 关键源码
- `meta-infer/ref_projects/mini-sglang/python/minisgl/models/qwen3_moe.py`
  - 模型组装入口：`18-80`
- `meta-infer/ref_projects/mini-sglang/python/minisgl/models/utils.py`
  - `MoEMLP`（gate replicated + experts）：`53-76`
  - `RopeAttn`（`LinearQKVMerged` + `LinearOProj`）：`79-124`

### 可迁移规则
- DeepSeekV2 可沿用同样模式：Attention 走 QKV/O TP；MoE 走 gate + experts + reduce。

---

## 3.5 权重切分/合并加载（非常关键）

### 关键源码
- `meta-infer/ref_projects/mini-sglang/python/minisgl/models/weight.py`
  - 切分规则常量：`13-31`
  - `_shard_tensor`（dim0/dim1/vocab/QKV-KV复制）：`34-53`
  - merge 规则（`q_proj/k_proj/v_proj -> qkv_proj`，`gate/up -> gate_up_proj`）：`16-30`、`55-60`
  - 流式加载主循环：`75-124`
  - MoE expert stack：`111-119`

### 可迁移规则
- 对极简工程最有价值：先在加载期完成**切分+融合**，推理时模块只消费“就位”参数。
- DeepSeekV2 TP=N（`N ∈ {1,2,4,8}`）也应优先采用这种“加载时整形”策略，而不是 forward 时频繁 reshape/cat。

---

## 4. 面向 DeepSeekV2 TP=N（N∈{1,2,4,8}）的直接实现蓝图

以下是基于上述源码可直接落地到 `meta-infer/engine` 的最小方案。

1. **分布式启动**
   - 启动 `N` 进程（`N ∈ {1,2,4,8}`），设置 `LOCAL_RANK/RANK/WORLD_SIZE`。
   - 在引擎初始化里做 `init_process_group("nccl", world_size=N, rank=rank)`。
   - 按 nano-sglang `parallel_state.py` 建 TP 组 API（可先不做 PP）。

2. **替换线性层抽象**
   - 将当前相关投影层替换为：
     - `ColumnParallelLinear`
     - `QKVParallelLinear`（支持 KV replication）
     - `RowParallelLinear`
   - 规则照搬 nano/mini 的维度切分。

3. **Embedding/LMHead TP 化**
   - vocab 按 rank 切 shard。
   - embedding forward：mask + all_reduce。
   - lm_head forward：local logits + all_gather（或 rank0 gather）。

4. **MoE TP 路径**
   - gate 保持 replicated（最简稳定）。
   - expert FFN 参数按 TP 分片（优先中间维）。
   - 本地 expert 计算后 `all_reduce` 汇总。

5. **权重加载（最关键）**
   - 参考 mini `models/weight.py`：
     - 按 key 规则 dim0/dim1 切分；
     - q/k/v 合并到 qkv；
     - gate/up 合并；
     - MoE experts stack；
   - 参考 nano mixtral `load_weights`：
     - `weight_loader(param, tensor, shard_id)`；
     - 非本 rank 专家跳过。

6. **KV 缓存参数**
   - `num_kv_heads` 用 `div_even(..., tp_size, allow_replicate=True)` 逻辑。
   - cache bytes/token 估算同步使用本 rank kv_heads。

7. **校验策略（TP=N）**
   - 在 `tp ∈ {1,2,4,8}` 下，同 prompt 的输出长度、eos 行为一致。
   - 数值允许微差，但 top-1/采样统计应稳定。
   - 先单 batch，再多 batch，再长上下文。

---

## 5. 你后续实现时优先照抄的文件清单（按重要性）

1. `nano-sglang/.../layers/linear.py`
2. `mini-sglang/..../models/weight.py`
3. `nano-sglang/..../models/mixtral.py`
4. `mini-sglang/.../layers/moe.py` + `.../moe/fused.py`
5. `nano-sglang/.../parallel_utils/parallel_state.py`
6. `mini-sglang/.../distributed/impl.py`
7. `mini-sglang/..../models/utils.py`（RopeAttn / MoEMLP 组装）

---

## 6. 注意事项（针对 DeepSeekV2）

- DeepSeekV2 是 MoE + GQA/MLA 结构，**KV head 与 Q head 不对称**：必须处理 KV replication。
- 不建议第一版就做复杂 EP（expert parallel）；先在 TP 内跑通 MoE 并 all-reduce。
- 当前极简框架若仍使用 HF `AutoModelForCausalLM` 直接 forward，则无法直接插入上述 TP 层；需要先进入“可控模型层”路径（类似 mini/nano 的自定义模型组装 + load_weights）。

---

*说明：本文只整理你指定目录及其直接 import 关联的 TP 切分知识，目的是支持后续在当前工程实现 DeepSeekV2 TP=N（`N ∈ {1,2,4,8}`）。*
