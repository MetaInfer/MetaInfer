# 原生内存与生命周期契约

> 权威级别：Host/Device Allocation、Tensor View、Workspace、KV Storage 的强制契约。

## 1. 所有权模型

Owning Allocation 必须由 Move-only RAII 类型管理，Tensor View 必须是非拥有引用：

```cpp
class DeviceBuffer {
 public:
  DeviceBuffer() = default;
  DeviceBuffer(DeviceBuffer&& other) noexcept;
  DeviceBuffer& operator=(DeviceBuffer&& other) noexcept;
  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;
  ~DeviceBuffer();

  void* data() const noexcept { return ptr_; }
  std::size_t bytes() const noexcept { return bytes_; }
  int device() const noexcept { return device_; }
  explicit operator bool() const noexcept { return ptr_ != nullptr; }

 private:
  void* ptr_ = nullptr;
  std::size_t bytes_ = 0;
  int device_ = -1;
};

struct TensorView {
  void* data = nullptr;             // non-owning
  DType dtype = DType::kUnknown;
  std::vector<std::int64_t> shape;
  std::vector<std::int64_t> strides;
  std::size_t storage_bytes = 0;
  int device = -1;
};
```

禁止把 `hipMalloc` 返回指针散落到各 Layer 中手工释放。

## 2. 强制规则

- **MEM-001**：每次设备/Host Allocation 只有一个拥有者和一个释放路径。
- **MEM-002**：分配失败返回结构化错误，禁止把 Null Pointer 传入 Kernel/BLAS。
- **MEM-003**：Element Count、Byte Size、Offset 计算必须检查整数溢出。
- **MEM-004**：Debug/Test 构建必须校验 View 的 Rank、Shape、Stride、Alignment 和边界。
- **MEM-005**：Allocator、Stream 和 Event 生命周期必须长于所有异步使用者。
- **MEM-006**：Destructor 不得在 Steady-state Hot Path 隐式执行 Device-wide Synchronize。
- **MEM-007**：Weight Storage 初始化后只读；KV、Workspace、Host Staging 使用不同预算/池。

推荐集中实现安全乘法：

```cpp
Result<std::size_t> CheckedBytes(Span<const std::int64_t> shape,
                                 DType dtype) {
  std::size_t elements = 1;
  for (std::int64_t dim : shape) {
    if (dim < 0) return InvalidArgument("negative tensor dimension");
    if (dim != 0 && elements > std::numeric_limits<std::size_t>::max() /
                                 static_cast<std::size_t>(dim)) {
      return OutOfRange("tensor element count overflow");
    }
    elements *= static_cast<std::size_t>(dim);
  }
  return CheckedMultiply(elements, SizeOf(dtype));
}
```

## 3. 异步安全

C++ 对象析构不代表 GPU 工作已完成。Buffer 复用/释放之前必须满足以下至少一种条件：

- 后续操作与之前操作位于同一 Stream 且有顺序保证；
- 通过 Event 建立依赖；
- 在明确的调试/关闭边界执行同步。

Workspace 推荐使用 Lease：

```cpp
class WorkspaceLease {
 public:
  TensorView view() const;
  Status ReleaseAfter(BackendEvent completion);

 private:
  WorkspacePool* pool_ = nullptr;  // pool outlives lease
  std::uint32_t slot_ = 0;
};
```

`WorkspaceLease` 析构时不能在设备仍使用该 Buffer 的情况下直接归还。

## 4. 每 Rank 内存预算

启动时按“每个可见设备”分别计算：

```text
total/free memory
  - 本 Rank 权重分片
  - 持久化 Backend/BLAS Workspace
  - 通信 Buffer
  - 安全余量
  = 可用于 KV Block Pool 的预算
```

四张 16 GiB 卡是四个独立地址空间，不能创建一个 64 GiB `DeviceBuffer`。TP 只会根据切分策略让不同 Rank 持有不同权重/状态。

## 5. KV Cache 事务规则

每个物理 Block 至少记录 Allocated/Free 状态和 Generation/Owner 信息，防止旧 Block Table 引用已经复用的 Block。

跨 Layer/Rank 分配必须是事务：

```cpp
Result<BlockReservation> ReserveBlocks(std::uint32_t count);
Status Commit(BlockReservation&& reservation, RequestId owner);
void Rollback(BlockReservation&& reservation) noexcept;
Status FreeRequest(RequestId owner);  // idempotent
```

要么所有必需 Block 成功保留并提交，要么一个都不改变请求状态。

## 6. 验证要求

必须覆盖：

- Move Constructor/Assignment 和析构；
- 第 N 次分配注入失败；
- Byte Count 溢出、越界 View；
- 异步 Buffer 过早复用；
- Decode 中取消请求；
- Block 耗尽与回滚；
- Engine 反复构造/销毁后显存和 Block 数稳定；
- 可用时运行 Host Sanitizer 和厂商 Device Memory Checker。
