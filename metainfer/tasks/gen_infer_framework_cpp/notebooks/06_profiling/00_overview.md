# C++ Profiling章节

Profiling只在正确性通过后用于定位瓶颈。Contract为
`00_contracts/profiling_contracts.md`，实现和分析按以下顺序阅读：

1. `01_native_trace.md`：框架内部Region、Artifact和优化闭环；
2. `02_roctx_rocprof.md`：HIP/DTK平台Range与外部Profiler；
3. `03_gpu_event_benchmark.md`：算子/Step的设备时间和统计方法。

## 原则

- 默认关闭，不影响稳态性能；
- E阶段先产生固定负载的可比数据，G/F再分析；
- 每个数字带模型、Commit、Build、硬件、DType、Batch和并发配置；
- CPU Wall Time、GPU Event Time和外部Profiler时间分开；
- 优化后必须重新运行完整C Oracle；
- 无Profile证据时只能提出测量计划，不能宣称性能提升。

## 产物

`perf_report.json`保存稳定摘要，Trace/CSV/厂商Artifact放在指定Profile目录，
`perf-review.md`引用具体文件和Region。Artifact缺失是可观测性问题，不允许伪造
Kernel结论。

