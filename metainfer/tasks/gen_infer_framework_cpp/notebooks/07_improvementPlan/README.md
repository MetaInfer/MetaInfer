# C++推理框架改进计划

本目录保存已经完成初步根因分析、值得实施但尚未在目标环境全部验收的计划。
计划不能覆盖`00_contracts/`中的正确性和安全要求，也不能把上游项目的功能声明
当作海光DTK/HIP上的可用能力。

## 1. 计划索引

| 文件 | 主题 | 首要证据 |
|---|---|---|
| `00_sources_and_governance.md` | 上游版本、许可证、证据等级和引用规则 | 固定Commit与License |
| `01_sampling_and_generation.md` | Greedy、Seeded Random、Top-K/Top-P、Penalty | 旧任务随机路径退化为Greedy |
| `02_continuous_batching.md` | 独立Engine Loop、Token Budget、Fairness | Iteration 7 Batched Decode收益 |
| `03_paged_kv_and_prefix_cache.md` | Block Pool、Paged KV、Prefix Cache | KV OOM与碎片问题 |
| `04_hip_operator_optimization.md` | HIP Kernel、hipBLAS、Fusion、Dispatch | 低GPU利用率与大量GEMM Dispatch |
| `05_tensor_parallel_rccl.md` | TP、Collective、Rank Lifecycle | TP仍缺真实多卡闭环 |
| `06_quantization_and_weight_formats.md` | 格式识别、量化Loader、Kernel、质量门槛 | 量化Checkpoint不受支持 |
| `07_native_service_streaming.md` | SSE、取消、背压、连接生命周期 | Event接口和Streaming不完整 |
| `08_profiling_and_benchmarking.md` | Native Trace、GPU Event、回归归因 | Review与实测结论曾不一致 |
| `09_model_compatibility_and_loader.md` | Qwen3能力矩阵、Loader扩展、模型注册 | 当前实现仍是单模型专用路径 |

## 2. 状态与证据

每份计划使用以下状态：

```text
proposed              外部设计和本地现象支持，尚未批准实施
accepted              已选入某轮Plan并冻结验收标准
in_progress           正在修改，必须绑定任务和Iteration
blocked               有明确外部阻塞证据
validated             已通过本地Oracle，等待迁移为Experience
superseded            被指定的新Canonical计划替代
```

外部项目只能提供设计假设。计划至少经过Compile Probe、Focused Native Test、真实
Checkpoint C Oracle和可重复Benchmark后，才允许把结果写入`06_experience/`。

## 3. 使用顺序

1. A阶段先读相关Contract和当前任务证据；只有增量目标与本目录某计划匹配时才读它。
2. B阶段以已接受的`plan.md`为准，不能为了追求上游功能扩大本轮范围。
3. C/D阶段只使用本地日志、Binary和Oracle定位问题；外部实现不是正确性证据。
4. E/G/F阶段从`08_profiling_and_benchmarking.md`选择测量方法，再执行一个可归因改动。
5. Retrospective把未解决事实放入`08_issues/`，把已决定动作更新到本目录。

## 4. 禁止事项

- 任务Agent不得浏览、复制或Vendor现有框架源码；本目录是维护侧已提炼的自包含输入。
- 不得把CUDA、TensorRT、上游ROCm的新API直接写成DTK强制实现。
- 不得仅凭同名参数宣称语义兼容，例如“INT4”“Paged Attention”或“Streaming”。
- 不得引用没有Commit、License和适用边界的网上代码片段。
- 不得把尚未复现的外部Issue伪装成本项目已观察问题。

完成并通过Oracle后，将结果移动或重写为`06_experience/`；形成长期不变量时再更新
Contract。一个主题只保留一份Canonical计划，历史差异由Git保存。
