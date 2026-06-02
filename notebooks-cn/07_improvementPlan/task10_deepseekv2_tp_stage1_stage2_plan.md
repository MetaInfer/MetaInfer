# Task10：DeepSeekV2 TP 适配计划（阶段一 + 阶段二）

本文只定义两件事：
- 阶段一：`engine/tp_layers/moe.py` 的 `ExpertParallelMoE` 骨架（简化 EP via AllReduce）。
- 阶段二：`engine/models/deepseek_v2.py` 中 `DeepseekAttentionTP` 的 MLA 结构定义。

不包含阶段三权重加载实现，不改现有源码。

---

## 0. 目标与边界

### 0.1 当前目标（必须实现）
- 优先支持 **DeepSeekV2** 的 TP（目标先跑通 `tp_size=4`）。
- 结构上预留对 **DeepSeekV3** 的可扩展位（不在本阶段实现 MTP/NSA/DeepEP）。
- 仅输出“代码骨架设计文档”，不落地实现。

### 0.2 明确不做（本轮禁止）
- 不实现阶段三（`load_weights` 惰性切片/OOM 防护）。
- 不引入全量 EP all-to-all 通信（只做简化版本地专家 + all_reduce 汇总）。
- 不做 fused kernel/FlashMLA/FP8 优化。

---

## 1. 设计总原则（从 Qwen TP 经验继承）

来自 `notebooks-cn/06_experience/01_task10_tp_qwen_debug_experience.md` 的硬约束，直接套用到 DeepSeekV2：

1. **模型真源对齐优先**  
   Attention/RoPE/Norm 必须以官方实现语义为准，不能“近似实现”。
2. **不要重复切片**  
   参数加载层与模块内部层，必须约定“输入是全量还是本地 shard”，避免二次切分导致空张量。
3. **先正确再优化**  
   先用可验证路径（单样本/低并发）跑通，再做 batched/fused 优化。
4. **RoPE 旋转风格必须显式标注**  
   本任务固定用 Neox 风格（前后分半），禁止混入奇偶交错风格。
5. **Norm 数值路径要稳定**  
   注意力/归一化涉及的关键数值路径优先 fp32 计算后再 cast 回。

---

## 2. 阶段一：`engine/tp_layers/moe.py` 骨架

## 2.1 文件与导出目标
- 新建文件：`engine/tp_layers/moe.py`
- 导出：
  - `ExpertParallelMoEConfig`
  - `ExpertParallelMoE`
  - （可选）`partition_experts_for_rank`

## 2.2 配置结构（建议）

```python
@dataclass
class ExpertParallelMoEConfig:
    hidden_size: int
    intermediate_size: int
    num_experts: int
    top_k: int
    tp_size: int
    tp_rank: int
    # DeepSeekV2 先不做 shared experts 的细节，保留字段方便 V3 扩展
    num_shared_experts: int = 0
    score_function: str = "softmax"  # 或 sigmoid
    route_scale: float = 1.0
```

说明：
- `num_experts` 以模型配置为准（文档中可用 160 作为示例，但实现时不得写死）。
- `top_k` 默认按 DeepSeekV2 配置读取。
- `num_shared_experts` 先保留扩展位，阶段一可不启用。

## 2.3 模块拓扑

### Router（Gate，Replicated）
- 每个 rank 保持同一份 router 权重。
- 对输入 `hidden_states[B, T, H]` 计算 `router_logits[B, T, E]`。
- 每个 token 做 top-k：得到 `topk_idx[B, T, K]` 与 `topk_weight[B, T, K]`。

### Experts（按专家 ID 均分到 TP ranks）
- 将 `[0, num_experts)` 均匀切成 `tp_size` 份。
- 当前 rank 仅实例化 `local_expert_ids` 对应专家模块。
- 专家建议结构（骨架）：
  - `w1/gate_up`（可合并） + 激活
  - `w2/down`

## 2.4 前向语义（简化 EP via AllReduce）

输入：`hidden_states: [B, T, H]`  
输出：`moe_out: [B, T, H]`

步骤：
1. Router 在本 rank 计算 top-k（各 rank相同）。
2. 初始化 `local_out = zeros_like(hidden_states)`。
3. 遍历 token 的 top-k 专家：
   - 若专家属于本 rank：执行该专家前向，乘路由权重，累加到 `local_out`。
   - 若专家不属于本 rank：跳过（本 rank 贡献 0）。
4. `global_out = all_reduce_sum(local_out)`，得到完整 MoE 输出。
5. 返回 `global_out`。

该方案优点：
- 简单、易验证，且与当前 TP 库一致（只依赖 all_reduce）。
- 不需要 all-to-all 或复杂 token dispatch。

代价：
- 有效算力利用率不如完整 EP；后续可演进到 all-to-all。

## 2.5 骨架接口定义（建议）

```python
class ExpertParallelMoE(nn.Module):
    def __init__(self, cfg: ExpertParallelMoEConfig):
        ...
        self.tp_rank = get_tp_rank()
        self.tp_size = get_tp_size()
        self.local_expert_ids = partition_experts_for_rank(...)
        self.router = nn.Linear(cfg.hidden_size, cfg.num_experts, bias=False)  # replicated
        self.experts = nn.ModuleDict({...})  # only local experts

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 1) router -> topk
        # 2) local experts accumulate
        # 3) all_reduce_sum
        return out
```

## 2.6 必备防错点（阶段一）

1. **top-k 一致性**  
   所有 rank 必须基于同一输入算出同一 `topk_idx`，否则 all_reduce 后结果无意义。
2. **专家归属边界**  
   `expert_id` 到 `rank` 的映射函数必须单测覆盖（首尾/非整除场景）。
3. **空专家负载**  
   某 rank 某步可能一个 token 都没命中本地专家，仍需正确参与 all_reduce。
4. **dtype 统一**  
   router logits、topk 权重、专家输出 dtype 策略要一致，避免 rank 间数值偏差放大。

---

## 3. 阶段二：`engine/models/deepseek_v2.py` 的 MLA 结构定义

## 3.1 文件与导出目标
- 新建文件：`engine/models/deepseek_v2.py`
- 先定义（仅结构）：
  - `DeepseekV2TPConfig`
  - `DeepseekAttentionTP`
  - `DeepseekDecoderLayerTP`（仅骨架）
  - `DeepseekForCausalLMTP`（仅骨架，`load_weights` 留空到阶段三）

## 3.2 配置字段（最小集）

```python
@dataclass
class DeepseekV2TPConfig:
    hidden_size: int
    num_attention_heads: int
    num_key_value_heads: int
    q_lora_rank: int
    kv_lora_rank: int
    qk_nope_head_dim: int
    qk_rope_head_dim: int
    v_head_dim: int
    rope_theta: float
    rms_norm_eps: float
    num_hidden_layers: int
    # MoE 相关（供 Decoder 层装配）
    n_routed_experts: int
    num_experts_per_tok: int
```

说明：
- 这些字段来自 MLA/MoE 所需最小维度，不应复用 Qwen 的 Dense 配置假设。
- 为后续兼容 V3，保留扩展字段接口（如 grouped_topk 参数）。

## 3.3 `DeepseekAttentionTP`：切分规则（硬约束）

### 绝对不能切（Replicated）
- `q_a_proj`
- `kv_a_proj_with_mqa`

理由：它们是 `hidden_size -> lora_rank` 的降维瓶颈，按头维切分会破坏潜向量语义与后续 cache 结构。

### 必须按 TP 切分
- `q_b_proj`：`ColumnParallelLinear`（输出头维切分）
- `kv_b_proj_with_mqa`：`ColumnParallelLinear`（输出头维切分）
- `o_proj`：`RowParallelLinear`（输入按 rank 分片，输出 all_reduce）

## 3.4 MLA 前向数据流（结构定义）

对输入 `x[B, T, H]`：

1. **Q 路径（分两段）**
   - `q_latent = q_a_proj(x)`（replicated）
   - `q_full_local = q_b_proj(q_latent)`（column parallel，得到本 rank 的本地头）
   - 切分为 `q_nope` 与 `q_pe`。

2. **KV 路径（压缩表示）**
   - `kv_latent_plus_pe = kv_a_proj_with_mqa(x)`（replicated）
   - 切分：
     - `c_kv`（latent，维度 `kv_lora_rank`）
     - `k_pe`（rope token 部分，维度 `qk_rope_head_dim` 或其派生）
   - `kv_full_local = kv_b_proj_with_mqa(c_kv)`（column parallel）
   - 从中取 `k_nope` 与 `v_local`。

3. **Decoupled RoPE（只作用在 pe 段）**
   - 对 `q_pe` 和 `k_pe` 做旋转。
   - RoPE 使用 Neox（前后分半）旋转风格。
   - 最终拼接：
     - `q_local = concat(q_nope, q_pe_rot)`
     - `k_local = concat(k_nope, k_pe_rot)`

4. **Attention + 输出投影**
   - 本 rank 执行本地 heads attention 得 `attn_out_local`。
   - `o_proj(attn_out_local)` 通过 `RowParallelLinear` 汇聚。

## 3.5 KV Cache 定义（仅结构）

MLA 缓存只保留压缩态：
- `c_kv`（latent）
- `k_pe`（少量 rope 分量）

不缓存：
- 展开后的全量 `K` / `V` heads。

建议预留数据结构：
```python
class DeepseekMLAKVCache:
    # per layer per token:
    # latent: [kv_lora_rank]
    # k_pe:   [qk_rope_head_dim]
    ...
```

## 3.6 与阶段一 `ExpertParallelMoE` 的装配关系

- `DeepseekDecoderLayerTP` 的 MLP 位改为：
  - Dense 层：保留常规分支接口（便于回退）
  - MoE 层：接 `ExpertParallelMoE`（阶段一产物）
- 这样阶段二完成后，模型结构上可表达：
  - MLA Attention（TP）
  - MoE（简化 EP via all_reduce）

---

## 4. 对 DeepSeekV3 的前向兼容位（只留接口）

尽管本轮仅做 V2，需预留 V3 扩展点：

1. **MoE 路由策略可插拔**
   - `route_impl: Literal["topk", "grouped_topk"]`
2. **共享专家接口**
   - `num_shared_experts` 字段与独立分支
3. **Attention 模式枚举**
   - `attn_impl: Literal["mla", "mla_absorb"]`
4. **可选 MTP/NSA 插槽**
   - Decoder 输出留钩子，不在本阶段实现逻辑

---

## 5. 实施顺序（仅阶段一+二）

1. 新建 `engine/tp_layers/moe.py`：先让 `ExpertParallelMoE` 在随机输入下可跑通前向和 all_reduce。
2. 新建 `engine/models/deepseek_v2.py`：先定义 `DeepseekAttentionTP` 与相关 config/class skeleton。
3. 将 `DeepseekDecoderLayerTP` 串起 Attention + MoE 的模块图（不接真实权重）。
4. 只做结构层级 smoke test（shape/device/通信可达），不做真实模型质量对齐。

---

## 6. 本文交付物检查清单

- [x] 只覆盖阶段一与阶段二。
- [x] 明确 `moe.py` 中 `ExpertParallelMoE` 的通信策略（本地专家 + all_reduce）。
- [x] 明确 MLA 的“不可切/必须切”矩阵。
- [x] 明确 Decoupled RoPE 与 Neox 风格要求。
- [x] 明确 KV cache 只存 latent + rope 分量。
- [x] 继承 Qwen TP 已踩坑的防错规则，避免重复错误。

