# 改进计划：Profiling、Benchmark与性能归因

状态：accepted（方法计划；具体优化仍需单独接受）  
来源：旧任务Iteration 5-7；SRC-LLAMA、SRC-ROCM-EX、SRC-HIPBLAS、SRC-RCCL、
SRC-VLLM。  
前置Contract：`00_contracts/profiling_contracts.md`、
`06_profiling/00_overview.md`、`06_profiling/01_native_trace.md`。

## 1. 当前证据

旧任务曾出现D Review在E阶段测量前判断“没有性能提升”，随后E实测Iteration 7吞吐
提升154.7%。也曾把错误率、不同总Token和不同完成请求混入轮次比较。这说明性能流程
必须强制“先建立可比基线，再归因”，Review不能用代码直觉替代设备测量。

## 2. 目标

- 默认关闭且低开销的Native Region Trace；
- CPU Wall、GPU Event、外部Profiler和端到端请求指标分层；
- 固定Workload、Build和环境的Before/After比较；
- 每个性能结论绑定Artifact、Region、改动和正确性Oracle；
- Agent无法得到证据时输出下一步测量计划，不编造Kernel结论。

## 3. Region和Metric标准

```text
request.parse / tokenize / admission / queue_wait
scheduler.build_step / kv.reserve / kv.commit
model.prefill / model.decode / layer.N
operator.gemm / attention / norm_rope / mlp / sample
collective.allreduce / broadcast / barrier_wait
response.detokenize / queue / socket_write
shutdown.drain / profile.flush
```

每个Region携带Request/Step/Rank/Device/Batch/Token Count的非敏感关联ID。不得把Prompt
文本、Token原文或认证Header写入Trace。

## 4. 实施阶段

### M0：可比性Manifest

- 记录Git状态/源码摘要、Compiler、Flags、DTK/HIP、Library、Model Revision和DType；
- 记录设备可见性、频率/功耗工具可用性、其他已观察GPU进程；
- 固定Prompt集合、输入/输出Token、Warmup、Concurrency、Request Rate和Seed；
- 结果不满足相同配置时标记Non-comparable，不计算百分比提升。

### M1：内部Trace

- RAII Region写入每线程Buffer，默认关闭时只保留一次可预测分支；
- 使用Monotonic Clock，GPU工作由Event关联，不用CPU提交时间冒充Kernel时间；
- Multi-rank文件名和事件包含Rank，时间基准差异明确记录；
- SIGTERM和正常退出都Flush到临时文件后Atomic Rename；
- Buffer满采用有计数的Drop或Chunk Flush，不无限增长。

### M2：GPU Event Microbenchmark

- Event在同一Stream包围目标Operator，Warmup后重复并报告Median/P95；
- 不在计时窗口执行Allocation、首次JIT/Library Init或文件I/O；
- 每个Shape验证输出，防止编译器或错误路径产生虚假快速结果；
- Operator时间与端到端Decode时间交叉核对；
- Event API和Resolution由本机DTK Probe确认。

### M3：外部Profiler

- 先探测rocprof、ROCTX或Vendor Tool的实际版本和参数；
- `serve.sh`只在`METAINFER_PROFILE=1`时包裹Native Binary；
- 工具不存在时内部Trace和Event仍工作，服务不因可选Profiler失败；
- Artifact、命令、Exit Code、版本和时间窗口写入Perf Report；
- 不把上游ROCm命令行硬编码为所有DTK版本通用。

### M4：端到端矩阵

至少测量Concurrency 1/4/16，并按能力增加Prompt Length、Output Length、Paged KV命中、
TP Size和Sampling模式。每格报告请求数、完成数、错误数、输入/输出Token、Wall Time、
Token/s、TTFT、ITL、P50和P99。

## 5. 归因Gate

一个性能结论必须同时满足：

1. 改动前后完整C Oracle通过；
2. Workload和Manifest可比较；
3. 至少多次测量并报告离散程度；
4. 一个Region、调用次数、同步或资源指标解释方向；
5. 没有通过减少Token、增加错误、跳过功能或错误计时制造收益；
6. 回退实验或Feature Flag能恢复旧表现。

如果端到端提升而Microbenchmark无变化，应检查Batch Size、Queue、Overlap和调用次数；
如果Microbenchmark提升但端到端无变化，应降低该算子优先级而不是继续盲目融合。

## 6. 产物

```text
perf_report.json          稳定Schema的可比摘要
runtime_manifest.json     Build、Model、Hardware和Capability
native-trace-rankN.json   内部Region
gpu-event-summary.json    Shape级统计
perf-review.md            证据、归因、风险和下一实验
```

Artifact路径由Oracle指定，禁止覆盖前一轮文件。Report引用文件Hash或Size，零字节Trace
视为失败。

## 7. 本地验收

- Profiling关闭时与无Hook基线无稳定可测开销；
- 开启后Region嵌套、Thread/Rank和Step关联正确；
- SIGTERM、Buffer满和Profiler缺失仍产生可诊断结果；
- 一个已知Sleep/Kernel Fixture的CPU/GPU时间分层正确；
- 比较器拒绝不同Model、Token Count、Build或错误率的虚假对比；
- Perf Planner引用真实Artifact后才能提出优化优先级。

## 8. 风险

- GPU利用率采样是粗粒度旁证，不能替代Event/Trace。
- 外部Profiler会改变Timing，吞吐基线和Profile Run分开。
- 首次运行包含Load、Warmup和Library初始化，不进入稳态Decode统计。
- 不设置固定“必须优化N轮”；正确性通过且无可归因高价值目标时应结束。
