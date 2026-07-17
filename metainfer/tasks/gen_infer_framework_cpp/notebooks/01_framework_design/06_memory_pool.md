# Memory Pool：设备内存预算与复用

先读：`00_contracts/memory_contracts.md`和
`01_framework_design/08_tensor_ownership.md`。

## 1. 分工

Memory Pool管理可复用物理Storage和Workspace；Paged KV Manager管理请求到KV
Block的逻辑映射。两者不能合并成一个无类型全局Allocator。模型权重是长生命周期
Storage，不进入临时Workspace复用。

## 2. 内存预算

启动时按设备分别计算：

```text
usable = free_at_start * configured_fraction
       - runtime_reserve
       - communication_reserve
weights + persistent_metadata + kv_pool + workspace_peak <= usable
```

预算必须使用实际Local Weight字节、DType和模型配置；不能把多卡空闲显存相加后
作为单地址空间。预算与最终分配差异写入Runtime Manifest。

## 3. Pool接口

```cpp
class WorkspacePool {
 public:
  Result<WorkspaceLease> Acquire(std::size_t bytes,
                                 std::size_t alignment,
                                 BackendStream stream);
  Status ReleaseAfter(WorkspaceLease lease, BackendEvent completion);
  PoolStats Stats() const;
};
```

Lease唯一拥有一次借用，移动后源对象失效。释放必须等待使用它的最后Event。
不同DType可以共享原始Storage，但每次构造TensorView时重新验证Size、Alignment、
Shape和Stride。

## 4. Size Class与碎片

基线实现使用有限的Size Class或Best-fit空闲块，并记录Requested/Allocated/Peak
字节。不要在每个Decode Step调用`hipMalloc/hipFree`。超过阈值的特殊Workspace
可以独占分配，但必须在请求或Step结束后有界释放。

## 5. KV与临时Buffer

KV Pool在模型加载后根据剩余预算一次性或分段分配，Block Size和每Token字节由
模型Head配置推导。Logits、QKV、Attention Scratch和Collective Staging使用
Workspace Pool。任何Workspace不得覆盖仍被异步Kernel读取的KV或激活。

## 6. OOM策略

分配失败先返回`ResourceExhausted`和Pool统计，由Scheduler选择拒绝、延后或降低
Batch；Backend不得静默切换到Host内存或CPU执行。OOM后的Pool元数据必须保持
一致，后续较小请求仍可运行。

## 7. 并发与关闭

Pool元数据由单Engine线程拥有，或使用明确锁保护；Destructor只在所有Stream
停止后执行。Shutdown顺序为停止Admission、取消/Drain、等待Events、释放
Workspace/KV、释放Weights、销毁Library Handle/Stream、最后销毁Device Context。

## 8. 测试

- Alignment、零字节、溢出和越界View；
- Event完成前Buffer不复用；
- 多Size Class反复分配后统计守恒；
- 注入OOM后无元数据损坏；
- KV容量公式与真实分配字节一致；
- 连续一千个Decode Step不产生逐步增长；
- SIGTERM和Backend错误后设备分配回到基线。

