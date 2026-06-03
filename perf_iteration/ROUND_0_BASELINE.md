# 第0轮 — 基线基准测试

## 吞吐率（torchrun TP=4, 32 tokens, temperature=0）

| 引擎 | 耗时 | 吞吐率 | 差距 |
|--------|------|-----------|-----|
| meta-infer（原型） | 0.593s | **54.0 tok/s** | 基准 |
| agent-engine（生成代码） | 0.804s | **39.8 tok/s** | **-26.3%** |

## Profiler 关键差异（Self CUDA Time）

| 指标 | meta-infer | agent-engine | 差异 |
|--------|-----------|-------------|-------|
| **Self CUDA 总计** | 185.1ms | 1,155ms | 通信操作的 profiler 失真 |
| aten::mm（GEMM） | 98.5ms | 99.5ms | ~一致 |
| cutlass gemm | 37.9ms | 39.0ms | ~一致 |
| **flash_fwd_splitkv_combine** | **5.3ms** | **30.9ms** | **+25.6ms（5.8倍）** |
| **cudaDeviceGetAttribute** | 未出现 | **15.6ms（4,644次调用）** | **新增** |
| **cudaFuncSetAttribute** | 未出现 | **4.2ms（1,152次调用）** | **新增** |
| **cudaLaunchKernel** | 未进入前列 | **9.8ms（20,286次调用）** | **新增** |
| **record_param_comms** | 1.1ms | **7.0ms** | **+5.9ms（6.4倍）** |
| **nccl:all_gather** | 2.9ms | **7.2ms** | **+4.3ms（2.5倍）** |
| aten::contiguous | 未出现 | 0.3ms（72次调用） | 新增 |
| aten::split_with_sizes | 未出现 | 0.5ms（1,152次调用） | 新增 |
| aten::item CPU同步 | ~2.2ms | ~2.2ms | ~一致 |

## 已识别的问题

1. **ISSUE-1**：flash_fwd_splitkv_combine 慢了5.8倍（30.9ms vs 5.3ms）—— 可能是 page_block_size 或 KV head 数不同导致
2. **ISSUE-2**：cudaDeviceGetAttribute 15.6ms —— 热路径上不必要的 CUDA 运行时查询
3. **ISSUE-3**：cudaFuncSetAttribute 4.2ms —— 动态 kernel 属性设置
4. **ISSUE-4**：cudaLaunchKernel 开销 9.8ms —— 过多的 kernel 启动
5. **ISSUE-5**：record_param_comms 高出6.4倍 —— custom_op 中类似 DDP 的开销
6. **ISSUE-6**：nccl:all_gather 高出2.5倍 —— embedding/LM head 通信开销
7. **ISSUE-7**：aten::contiguous 72次不必要的调用 —— 张量格式转换
8. **ISSUE-8**：aten::split_with_sizes 1152次调用 —— QKV 投影分割模式
