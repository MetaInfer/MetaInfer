# 改进计划：模型兼容性与Loader扩展

状态：proposed  
来源：当前生成框架以Qwen3-1.7B Dense专用实现为主；SRC-LLAMA、SRC-MLC、SRC-SAFE。  
前置Contract：`00_contracts/qwen3_model_contracts.md`、
`00_contracts/weight_loading_contracts.md`、`02_model_specifics/01_transformer_models.md`。

## 1. 问题

“支持Qwen3”可能只表示一个Checkpoint能启动。真正的兼容性需要把Architecture、Config、
Tokenizer、Weight Mapping、Operator、DType、KV Layout和Service Capability分开声明。
继续用模型名条件分支扩展Dense、MoE和Hybrid，会让Loader与ModelRunner不可验证。

## 2. 目标和非目标

目标是建立Capability-driven Model Registry，先稳定Qwen3 Dense的多尺寸，再按明确
Contract增加MoE和Hybrid。每个模型由配置事实选择Implementation，不靠目录名猜测。

非目标：不宣称任意Transformer自动兼容；不以运行到第一个Token作为支持完成；不在
模型类中直接解析HTTP或调用具体HIP Library。

## 3. Capability矩阵

每个已注册实现声明：

```text
architecture_ids
config_schema_version和必需字段
supported_weight_dtypes / quantization_descriptors
attention_kind / rope_layout / position_encoding
q_heads / kv_heads / head_dim约束
mlp_kind / activation / moe_routing
kv_layout_ids
tp_shard_rules
operator_capabilities
sampling / streaming / max_context能力
validated_checkpoint_revisions
```

Runtime只选择所有条件满足的实现。部分满足时输出逐项Unsupported原因，不选择“最近似”
实现。

## 4. 实施阶段

### C0：分离Loader层次

- Container Reader只负责安全读取Metadata和Byte Range；
- Model Config Parser负责Architecture字段与约束；
- Weight Mapper把Canonical Parameter映射到源Tensor并声明Transform/Shard；
- Device Loader负责Storage、Upload和事务发布；
- Model Builder只消费已验证Config和Tensor Registry。

### C1：Qwen3 Dense多尺寸

- 使用Tiny Fixture和至少两个真实Dense尺寸验证没有1.7B常量；
- Hidden、Layer、Head、KV Head、Intermediate、Vocab和Context全部来自Config；
- Tokenizer Special ID和Chat Template来自Checkpoint并独立验证；
- Missing/Extra Tensor策略明确，Tied Embedding不能猜测；
- TP=1的Full Precision成为后续模型的Golden Baseline。

### C2：Qwen3 MoE

- 新增Router、Top-K Expert、Normalize、Capacity和Expert Tensor Mapping；
- 先用Host/Tiny Native Reference验证Routing Tie和Token Dispatch；
- Correctness Path可逐Expert GEMM，性能路径再做Grouped GEMM；
- Dense和MoE共享基础Attention/Norm，但不通过空Expert伪装同一模型；
- TP/EP策略未验证时保持单卡或明确Unsupported。

### C3：Qwen3.5/Hybrid

- 根据Layer Type构建不同Block，维护Attention/State Space/Hybrid State；
- MRoPE、Sliding Window或特殊KV Group拥有独立Layout ID；
- 每类Layer有Tiny Step和状态连续性测试；
- 不把Full Attention Paged KV规则直接用于不同状态模型；
- 完整模型能力取所有Layer能力交集。

### C4：注册与扩展

- `ModelRegistry`使用Architecture ID和Version匹配Factory；
- 每个实现提供Config Validator、Weight Spec、Operator Requirements和Tests；
- 新模型不修改HTTP/Scheduler/KV公共语义，除非先扩展Contract；
- Build Manifest列出编译进入Binary和本次实际选择的实现；
- 未验证Checkpoint不进入“supported”列表。

## 5. Loader安全和事务

- Safetensors Header Length、JSON Shape、Offset、乘法和Range全部Checked；
- Shard Index必须覆盖所有必需Tensor且没有冲突来源；
- Transform、Transpose、Concat、Shard和Quant Repack逐项记录目标Range；
- 所有权重验证和Upload成功后一次发布Model；
- 失败释放Host Mapping、Staging、Device Buffer和Stream，不留下半初始化Ready状态。

## 6. 本地验收

- Config Property/Fuzz覆盖未知Architecture、缺字段、极值和整除错误；
- Tiny Safetensors覆盖Truncate、Overlap、Wrong DType、Wrong Shape和Missing Tensor；
- 每个模型先比较Embedding、单Layer、Prefill、Decode Logits，再运行文本Oracle；
- 两个Qwen3 Dense尺寸通过，证明不存在固定Layer/Hidden/Vocab常量；
- MoE Routing和Hybrid State使用独立Golden Fixture；
- 未支持模型在设备大分配前返回精确Capability Report；
- Loader并行或TP分片后Tensor Hash/元素范围守恒。

## 7. 性能验收

模型兼容改造首先不回退现有Qwen3 Dense。报告Load Time、Host/Device峰值、Prefill、
Decode和Manifest Size。抽象层不得引入每Token虚调用查找、字符串Map或重复Shape验证；
初始化后把Dispatch解析为稳定Function/Object边界。

## 8. 风险与停止条件

- 模型名相似不代表RoPE、QKV Layout或Chat Template相同。
- 上游框架支持列表只能用于发现候选字段，不能作为本卡片验证证据。
- 没有真实Checkpoint、Reference Logits或目标Operator时保持proposed/blocked。
- 通用化不得牺牲错误可诊断性；每个Unsupported报告缺少的确切Capability。
