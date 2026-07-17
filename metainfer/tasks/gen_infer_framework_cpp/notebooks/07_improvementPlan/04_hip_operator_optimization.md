# 改进计划：HIP算子、hipBLAS与Dispatch优化

状态：proposed  
来源：旧任务Iteration 5-7；SRC-ROCM-EX、SRC-HIPBLAS、SRC-LLAMA。  
前置Contract：`00_contracts/operator_contracts.md`、
`03_operators/01_attention_ops.md`、`03_operators/05_hip_blas_backend.md`、
`06_profiling/03_gpu_event_benchmark.md`。

## 1. 当前证据

Iteration 6删除多余Stream Sync后吞吐只提升约11.6%；Iteration 7通过并行Attention、
Fused Norm/RoPE和Batched Decode把Concurrency 16提升约154.7%。但GPU利用率仍约
13.4%，每Token仍有大量小M GEMM/Kernel Dispatch。结论是：优化必须从Trace和调用
次数出发，不能只凭“Attention通常最慢”排序。

## 2. 目标和非目标

目标是在保持独立可验证Baseline的前提下，减少Host Dispatch、同步、临时分配和无效
Memory Traffic，并建立DTK能力表驱动的Operator Dispatch。

非目标：不移植CUDA Kernel，不假设Wave Size，不用设备名称硬编码Feature，也不在
没有中间Tensor对比时一次融合完整Decoder Layer。

## 3. 能力探针

在优化前生成`operator-capabilities.json`：

```text
HIP architecture和warpSize
max_threads、shared_memory、register相关限制
FP16/BF16 load/store/atomic/intrinsic编译与运行
hipBLAS GEMM/GEMMEx支持的A/B/C/compute dtype
transpose、leading dimension、batched/strided-batched行为
Stream/Event和异步错误传播
可选ROCTX、Graph、Lt或Vendor Extension
```

Probe使用Tiny输入校验数值和错误码。只Link成功不算支持；上游ROCm最新结果不能覆盖
本机DTK Probe。

## 4. 实施阶段

### O0：算子账本

- 为Prefill/Decode记录Operator、Shape、DType、调用次数、CPU Launch和GPU Event；
- 标出每Token的`hipMalloc`、Memcpy、Stream Sync和隐式D2H；
- 用总和与端到端Wall Time交叉检查，避免重复累计嵌套Region；
- 从占比和可消除调用数选择一个优化目标。

### O1：hipBLAS Adapter校正

- 集中处理Row/Column Major、Transpose、Leading Dimension和Compute Type；
- 对实际Qwen Shape建立正确性与性能矩阵，不凭单个Square GEMM决定策略；
- Handle与Workspace按Device/Stream复用，禁止每次创建销毁；
- Algorithm选择若可用，必须按Capability和Shape缓存并提供回退。

### O2：无语义融合

优先选择容易逐项验证的融合：

- Residual Add + RMSNorm；
- Q/K Norm + RoPE；
- Bias/Add/Activation（模型确实存在时）；
- Gate + Up Projection的权重布局与Batched GEMM；
- Sampling的候选筛选流水线。

每次只融合一个边界，保留Unfused Path和中间Tensor Golden Test。融合不能改变Accumulation
DType、RoPE Layout、Mask或Broadcast语义。

### O3：Attention分派

- Tiny Reference覆盖Prefill、Decode、GQA、Causal Mask和Paged KV；
- 先优化Decode Reduction和Memory Coalescing，再评估Online Softmax/Tiled Prefill；
- Kernel Dispatch由Head Dim、KV Length、Block Size、DType和Capability决定；
- Unsupported Shape走已验证Native Baseline，不走Host模型计算。

### O4：Batch和图执行

- 先通过Continuous Batching增大GEMM的M，再考虑Graph/Capture；
- 固定地址、Shape Bucket、Workspace和Collective必须满足后才能Capture；
- Capture失败或Shape未命中时有正常异步路径；
- DTK Graph API未通过Probe时计划终止，不做伪接口。

## 5. 正确性验收

- 所有优化算子与Host FP32/Tiny Native Reference比较多组随机和边界Shape；
- Prefill与逐Token Decode在同一Prompt上Logits一致；
- Fused与Unfused路径的中间或最终Tensor在DType容差内一致；
- Stream非默认、连续调用、错误注入和多请求Workspace隔离通过；
- Sanitizer可覆盖的Host边界无越界，Device错误在正确同步点上报；
- 完整Qwen3 C Oracle、Seeded Sampling和长Context回归通过。

## 6. 性能验收

每个改动报告Before/After：

```text
Operator Shape和DType
CPU launch count / token
GPU event time / call和/ token
Host sync和allocation count
Concurrency 1/4/16吞吐、P50、P99、完成率
功耗/利用率（工具可用时）
```

低于测量噪声的变化不宣称收益。端到端变快但单算子变慢时必须解释Batch、Overlap或
调用次数变化。正确性回退立即停止，不用模型输出偶然通过覆盖Tensor差异。

## 7. 风险与回滚

- 小M GEMM可能受Dispatch而非算力限制，优先增加Batch或合并Projection。
- 上游hipBLAS的Datatype表不代表Vendor Fork，Feature必须运行时Manifest可见。
- 过度融合增加Register/Shared Memory压力，保留按Shape回退。
- Unfused Baseline在优化验证完成前不可删除；它是Repair Agent定位问题的最后边界。
