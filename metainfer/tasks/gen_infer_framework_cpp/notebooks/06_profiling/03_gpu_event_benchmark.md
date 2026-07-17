# GPU Event Benchmark

## 1. 为什么使用Event

CPU Wall Time包含排队、Launch和其他请求，不能直接代表Kernel时间。算子和Step
Microbenchmark在同一Device/Stream上记录Start/Stop Event，并在读取结果前等待
Stop完成。

```cpp
Result<double> MeasureGpuMilliseconds(BackendStream stream,
                                      const std::function<Status()>& enqueue);
```

Event创建、记录、同步和Elapsed API都必须检查错误。跨Stream测量需要显式Event
依赖，不能比较没有共同时间线的Event。

## 2. Benchmark协议

固定模型、Shape、DType、设备、Clock/共享主机状态和Implementation ID。先Warmup，
再执行足够重复次数；报告Median、P90/P99、Min和样本数。首轮JIT/Module Load不
计入稳态，但单独报告冷启动。

## 3. 范围

- Micro：单GEMM、Norm、RoPE、Attention、Sampling；
- Step：完整Prefill或Decode Device工作；
- End-to-end：HTTP Wall Time、TTFT、ITL和Tokens/s。

三类数字不可互相替代。优化Kernel但Host/Scheduler变慢时，最终吞吐可能下降。

## 4. 同步与缓存

Benchmark不得在每次生产调用中同步；同步只存在测量Harness。输入Buffer在重复间
保持有效，输出有消费或校验以避免错误路径。每个候选实现先与Reference比数值，
再计时。

## 5. GPU遥测

同时记录每设备Memory、Utilization、Power和活动时间序列。多卡报告Assigned和
Active Device均值，不能用所有可见Idle设备稀释利用率。共享主机上的其他进程
只记录，不终止。

## 6. 回归判定

比较同协议基线，写明绝对值、百分比、方差和正确性结果。样本不足或环境变化时
结论为Inconclusive。性能Gate不得接受只有平均值且没有原始配置的报告。

