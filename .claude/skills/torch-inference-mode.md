# 技能：推理引擎的 torch.inference_mode()

## 模式

在构建 PyTorch 推理引擎时，始终为模型的 `forward()` 和 `forward_decode()` 方法添加 `@torch.inference_mode()` 装饰器。

```python
@torch.inference_mode()
def forward_decode(self, input_ids, positions, kv_len, max_seq_len):
    ...
```

## 为什么重要

**`model.eval()` 不够。** PyTorch 的 `eval()` 模式只影响有状态的层（Dropout、BatchNorm）。它不会禁用 autograd 追踪。没有 `torch.inference_mode()` 时，每个张量操作——matmul、add、norm、attention——都会递增版本计数器并附加 autograd 元数据。这些开销在计算图中不可见，但会表现为：

- profiler 中的 `aten::clone` 调用（autograd 创建中间张量的副本）
- `cudaLaunchKernel` CPU 开销（autograd 包装 kernel 启动）
- `cudaDeviceGetAttribute` 调用（autograd 在创建张量时查询设备属性）
- trace 中的 `GeneratedBackwardFor*` 条目（autograd 构建反向图，即使从未使用）

**在 TP=4 Qwen3-8B 推理引擎上的实际影响（32-token 生成）：**
- 使用 `@torch.inference_mode()`：**0.477s，67.1 tok/s**
- 不使用：**0.716s，44.7 tok/s**
- 差距：一行装饰器带来 **+50% 吞吐率提升**

## `torch.inference_mode()` vs `torch.no_grad()`

使用 `torch.inference_mode()`——它在各方面都更优：

| 特性 | `torch.no_grad()` | `torch.inference_mode()` |
|---------|-------------------|--------------------------|
| 禁用梯度计算 | 是 | 是 |
| 禁用版本计数器递增 | 否 | 是 |
| 禁用 autograd 元数据追踪 | 否 | 是 |
| 张量操作可有 `requires_grad=True` | 是（易出错） | 否（会报错） |
| 性能 | 较慢 | 更快 |

## 何时应用

1. 任何仅用于推理的模型 `forward()` 方法
2. 任何独立的 `forward_decode()` 或仅 decode 的路径
3. 模型 runner 的 `run()` 方法（作为安全网）
4. 在尽可能外层的作用域应用——如果整个 `run()` 方法被装饰，内部方法也会被覆盖

将装饰器直接放在方法**定义**上，而非在调用处包装，这样意图明确且不会被意外移除。

## 如何识别这个问题

在 PyTorch profiler trace（`torch.profiler.profile()`）中，在热 decode 路径中寻找以下信号：

1. **`aten::clone` 调用次数过高**——autograd clone 出现在每一层、每一步。如果 clone 调用远超预期（如32次，对应每层第0层的 residual），说明 autograd 在克隆中间量。
2. **`cudaLaunchKernel` CPU 时间过高**——autograd 包装 kernel 启动，增加 CPU 开销。
3. **`cudaDeviceGetAttribute`**——autograd 在内部创建张量时查询设备属性。
4. **`GeneratedBackwardFor*`**——反向图正在被构建，尽管从未调用 `backward()`。
5. **CPU 总时间 > 2倍端到端耗时**——过多的 CPU 时间表明 autograd 在做簿记工作。

使用以下命令检查：
```bash
# 在 profiler 输出中查找这些模式
grep -E "aten::clone|GeneratedBackwardFor|cudaLaunchKernel|cudaDeviceGetAttribute" key_avg.txt
```

## 验证

1. 添加装饰器前后各运行一次基准测试
2. 检查 profiler：`aten::clone` 调用应从数千次降到接近零
3. `cudaLaunchKernel` CPU 时间应从前15名中消失
4. CPU 总时间应显著下降
5. 输出正确性必须验证（装饰器不应改变计算结果）

## 常见陷阱

对需要梯度计算的方法（如训练、微调）应用 `@torch.inference_mode()` 会出错。对于混合用途的模型，仅在推理专用方法（如 `forward_decode()`）上使用装饰器，如果 `forward()` 也用于训练则保持不加。

## 反模式

```python
# 错误：仅依赖 model.eval()
model.eval()
output = model(input_ids)  # autograd 仍在追踪！

# 错误：在调用处包装（容易遗忘）
with torch.inference_mode():
    output = model(input_ids)

# 正确：装饰器直接放在方法上
@torch.inference_mode()
def forward(self, ...):
    ...
```

## 总结

一行装饰器。+50% 吞吐率。始终为 PyTorch 推理 forward 方法添加 `@torch.inference_mode()`。
