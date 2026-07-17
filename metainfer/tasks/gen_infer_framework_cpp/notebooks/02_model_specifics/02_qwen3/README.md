# Qwen3原生C++模型知识

本目录按Checkpoint架构选择文档，不能同时混用多个变体。

## 路由

| Checkpoint特征 | 必读文档 |
|---|---|
| 标准Dense Decoder | `01_dense.md`、`04_model_loader.md` |
| `num_experts > 0`等MoE字段 | `02_moe.md`、`04_model_loader.md` |
| 混合FullAttention/GatedDeltaNet层 | `03_qwen3_5_hybrid.md`、`04_model_loader.md` |

共同强制契约为`00_contracts/qwen3_model_contracts.md`和
`00_contracts/weight_loading_contracts.md`。

## 识别规则

模型类型以`config.json`中的Architecture和结构字段为事实来源，卡片
`model_family`用于约束/交叉检查。字段冲突必须在分配权重前失败，不能因为Agent
只实现Dense就忽略Expert或Hybrid字段。

## 共同完成条件

- Checkpoint Tokenizer和Chat Template Token ID Golden通过；
- 必需权重全部加载且Shape/DType/Shard报告完整；
- 单Layer、Prefill和Decode与Reference达到阈值；
- 目标加速器执行真实热路径；
- Greedy与Seeded Sampling语义都通过Oracle；
- SIGTERM、取消和失败不泄漏设备资源。

