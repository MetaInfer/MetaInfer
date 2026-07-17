# 外部来源与知识治理

状态：accepted（知识库治理规则，不代表任何运行时功能已实现）  
快照日期：2026-07-17

## 1. 目的

外部框架用于发现成熟的设计问题、接口边界和验证方法，不用于向生成任务复制实现。
本文件固定调研快照，使计划在上游继续变化后仍可审计。任务Agent只读本知识库，
不得重新打开这些仓库补充实现细节。

## 2. 固定来源

| ID | 项目与固定版本 | License | 只借鉴什么 | 不能推导什么 |
|---|---|---|---|---|
| SRC-LLAMA | [llama.cpp@0bd0ec6](https://github.com/ggml-org/llama.cpp/tree/0bd0ec60998d0f71ec45471b633bf2403ac81956) | MIT | 原生Runtime边界、采样能力、量化格式治理、Server与性能诊断 | GGUF与Safetensors等价；其GPU Backend可直接用于DTK |
| SRC-VLLM | [vLLM@17fdd42](https://github.com/vllm-project/vllm/tree/17fdd421009092a569e9ed22346b932e55824fb5) | Apache-2.0 | Paged KV、Prefix Cache、Continuous Batching、并行与指标语义 | 允许引入其Scheduler；CUDA Kernel能在HIP编译 |
| SRC-TRT | [TensorRT-LLM@ed1a0b9](https://github.com/NVIDIA/TensorRT-LLM/tree/ed1a0b9bfad2074b49452aa0597ee9a476cb47a5) | Apache-2.0，部分第三方文件另有License | In-flight Batching、Executor、KV、Sampling、量化和并行的验收维度 | TensorRT、CUDA Graph或NCCL在DTK存在 |
| SRC-SGLANG | [SGLang@bbd2a3f](https://github.com/sgl-project/sglang/tree/bbd2a3fe4a267b1e5a2a49792a2273e2e519d881) | Apache-2.0 | Radix/Prefix Cache、Cache-aware Scheduling、PD解耦的设计问题 | 宣称支持AMD就等于支持本机海光设备 |
| SRC-MLC | [MLC-LLM@a2bcc5c](https://github.com/mlc-ai/mlc-llm/tree/a2bcc5c86678b72a86b7aadc29b643a5ce63c747) | Apache-2.0 | 模型编译、Weight Packaging、量化配置与Runtime Manifest | 必须引入TVM；其CUDA量化模式可在DTK运行 |
| SRC-ROCM-EX | [ROCm Examples@41dd746](https://github.com/ROCm/rocm-examples/tree/41dd7463e65e230af913db75d48a1d6c0dcff6bc) | MIT | HIP Stream/Event、内存、Profiler、hipBLAS和RCCL的最小Probe形态 | 上游最新Header与DTK Header相同 |
| SRC-HIPBLAS | [hipBLAS@23b26a0](https://github.com/ROCm/hipBLAS/tree/23b26a0093345264e7387481cbe01d1e1ae55fda) | MIT，附带第三方Attribution | GEMM/GEMMEx、Datatype、Leading Dimension和Batched接口验证 | 某算法、Compute Type或Lt扩展必然存在 |
| SRC-RCCL | [rocm-systems RCCL@27b4e4d](https://github.com/ROCm/rocm-systems/tree/27b4e4dd4438e205c3c9163efe4084b890bbb08e/projects/rccl) | RCCL组件复合License，见其LICENSE.txt | Collective API、Communicator、Rank、Group和故障诊断 | DTK提供同路径、同SONAME或全部Data Type |
| SRC-RCCL-OLD | [独立RCCL@57e5868](https://github.com/ROCm/rccl/tree/57e58688f44c77076ad536ef1f6b68741fc6e694) | 复合License | 历史兼容证据 | 继续作为当前维护入口；该仓库已声明retired |
| SRC-SAFE | [safetensors@6eb4dc9](https://github.com/safetensors/safetensors/tree/6eb4dc9a28ebce297606e0f4836bbf28839cacef) | Apache-2.0 | 文件格式、Header和Offset安全边界 | 量化Tensor的模型语义或Backend Kernel |

固定链接中的短Hash只用于显示，完整Hash才是权威版本。刷新来源必须单独提交，记录
为什么升级、哪些结论变化以及旧计划是否仍成立。

## 3. 采用的公开设计证据

- [llama.cpp Token Generation Performance Tips](https://github.com/ggml-org/llama.cpp/blob/0bd0ec60998d0f71ec45471b633bf2403ac81956/docs/development/token_generation_performance_tips.md)：设备Offload和CPU线程也必须由运行证据确认。
- [vLLM Paged Attention](https://github.com/vllm-project/vllm/blob/17fdd421009092a569e9ed22346b932e55824fb5/docs/design/paged_attention.md)：Block Table与Paged Kernel的数据访问问题。
- [vLLM Prefix Caching](https://github.com/vllm-project/vllm/blob/17fdd421009092a569e9ed22346b932e55824fb5/docs/design/prefix_caching.md)：Full Block Hash、Reference Count、Eviction和租户隔离。
- [TensorRT-LLM Paged Attention与IFB](https://github.com/NVIDIA/TensorRT-LLM/blob/ed1a0b9bfad2074b49452aa0597ee9a476cb47a5/docs/source/features/paged-attention-ifb-scheduler.md)：请求在执行中动态加入/退出的验收维度。
- [TensorRT-LLM Sampling](https://github.com/NVIDIA/TensorRT-LLM/blob/ed1a0b9bfad2074b49452aa0597ee9a476cb47a5/docs/source/features/sampling.md)：请求级采样参数与批内差异。
- [SGLang固定版本README](https://github.com/sgl-project/sglang/blob/bbd2a3fe4a267b1e5a2a49792a2273e2e519d881/README.md)：RadixAttention、Continuous Batching、Chunked Prefill与多种并行是独立能力。
- [MLC-LLM Quantization Configuration](https://github.com/mlc-ai/mlc-llm/blob/a2bcc5c86678b72a86b7aadc29b643a5ce63c747/docs/compilation/configure_quantization.rst)：Weight-only与Weight-Activation量化需要不同配置和校准证据。
- [ROCm HIP Examples](https://github.com/ROCm/rocm-examples/tree/41dd7463e65e230af913db75d48a1d6c0dcff6bc)：Event、Stream、Stream Ordered Allocation与Collective Probe。
- [hipBLAS Datatype Support](https://github.com/ROCm/hipBLAS/blob/23b26a0093345264e7387481cbe01d1e1ae55fda/docs/reference/data-type-support.rst)：Datatype支持必须按API与Backend验证。
- [RCCL API Reference](https://github.com/ROCm/rocm-systems/tree/27b4e4dd4438e205c3c9163efe4084b890bbb08e/projects/rccl/docs/api-reference)：Collective、Datatype和环境变量的上游证据。

## 4. 证据等级

```text
E0  只有外部文档或设计推断
E1  目标服务器Header/Library/Compile/Link Probe通过
E2  Tiny Native Test在目标设备通过
E3  真实Checkpoint、完整C Oracle和故障注入通过
E4  固定负载多次Benchmark证明收益且无正确性回退
```

`07_improvementPlan`允许E0/E1；`08_issues`至少需要本地可观察证据；
`06_experience`至少需要E3。性能结论必须达到E4。E0内容不能进入Contract成为硬性
平台能力声明。

## 5. 每个外部建议的记录格式

```text
Source ID和完整Commit
原始问题类别
提炼后的接口或不变量
没有复制源码的说明
CUDA/ROCm/DTK差异
目标服务器Probe
本地正确性Oracle
本地性能基线与结果
License/Attribution处理
结论: reject | defer | accept | validate
```

## 6. License与实现边界

概念、公开API语义和测试思想可以重写为本项目设计。任何实际代码复用都超出本卡片
默认授权边界，必须先修改生成任务政策、记录文件级来源和License，并经过独立Review。
即使License允许，也不能绕过“从零生成原生框架”的任务约束。

## 7. DTK适配规则

1. 先读取`00_contracts/hardware_profile_contracts.md`记录的本机事实。
2. 对Header、Symbol、Data Type、Stream语义和错误行为分别Probe。
3. 使用Adapter隔离上游API差异，不把版本判断散落在Model Layer。
4. 缺少能力时返回Unsupported或选择本知识库已验证的Native Baseline。
5. 不允许因为上游AMD GPU通过就宣称Hygon GPU通过。
