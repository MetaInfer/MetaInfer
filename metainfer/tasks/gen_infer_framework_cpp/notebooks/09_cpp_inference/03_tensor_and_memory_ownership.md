# Tensor 元数据与显存所有权

先读：`00_contracts/cpp/cpp_memory_contracts.md`。

本模块要解决三个问题：谁拥有内存、谁只是 View、GPU 异步工作完成前什么时候可以复用。

## 1. DeviceBuffer：唯一所有权

```cpp
class DeviceBuffer {
 public:
  DeviceBuffer() = default;
  ~DeviceBuffer() { Reset(); }

  DeviceBuffer(DeviceBuffer&& other) noexcept { MoveFrom(std::move(other)); }
  DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
    if (this != &other) {
      Reset();
      MoveFrom(std::move(other));
    }
    return *this;
  }

  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;

  static Result<DeviceBuffer> Allocate(std::size_t bytes, int device) {
    if (bytes == 0) return InvalidArgument("zero-byte device allocation");
    HIP_RETURN_IF_ERROR(hipSetDevice(device));
    void* ptr = nullptr;
    const hipError_t status = hipMalloc(&ptr, bytes);
    if (status != hipSuccess) {
      return BackendError("hipMalloc", hipGetErrorString(status), device, bytes);
    }
    return DeviceBuffer(ptr, bytes, device);
  }

  void* data() const noexcept { return ptr_; }
  std::size_t bytes() const noexcept { return bytes_; }
  int device() const noexcept { return device_; }

 private:
  DeviceBuffer(void* ptr, std::size_t bytes, int device)
      : ptr_(ptr), bytes_(bytes), device_(device) {}

  void Reset() noexcept {
    if (ptr_ != nullptr) {
      // Destructor 不能抛异常；失败写入诊断日志/计数器。
      hipSetDevice(device_);
      const hipError_t status = hipFree(ptr_);
      RecordDestructorErrorIfAny(status, "hipFree", device_);
    }
    ptr_ = nullptr;
    bytes_ = 0;
    device_ = -1;
  }

  void MoveFrom(DeviceBuffer&& other) noexcept {
    ptr_ = std::exchange(other.ptr_, nullptr);
    bytes_ = std::exchange(other.bytes_, 0);
    device_ = std::exchange(other.device_, -1);
  }

  void* ptr_ = nullptr;
  std::size_t bytes_ = 0;
  int device_ = -1;
};
```

生产实现还需要处理 Backend Shutdown 顺序：Runtime 必须晚于所有 DeviceBuffer 销毁。

## 2. TensorStorage 与 TensorView

```cpp
struct TensorView {
  void* data = nullptr;
  DType dtype = DType::kUnknown;
  std::vector<std::int64_t> shape;
  std::vector<std::int64_t> strides;
  std::size_t storage_bytes = 0;
  int device = -1;
};

class TensorStorage {
 public:
  static Result<TensorStorage> Allocate(Span<const std::int64_t> shape,
                                        DType dtype,
                                        int device);
  TensorView view();
  const TensorView view() const;

 private:
  DeviceBuffer buffer_;
  DType dtype_;
  std::vector<std::int64_t> shape_;
  std::vector<std::int64_t> strides_;
};
```

View 不拥有内存。需要跨异步任务保存 View 时，必须由上层保证 Storage 生命周期或使用明确的 Shared Storage Handle，禁止隐式 `shared_ptr<void>`。

## 3. Shape/Stride/Offset 校验

```cpp
Result<void*> TensorView::ByteAddress(Span<const std::int64_t> index) const {
  if (index.size() != shape.size()) {
    return InvalidArgument("tensor index rank mismatch");
  }
  std::uint64_t element_offset = 0;
  for (std::size_t axis = 0; axis < index.size(); ++axis) {
    if (index[axis] < 0 || index[axis] >= shape[axis]) {
      return OutOfRange("tensor index out of bounds");
    }
    ASSIGN_OR_RETURN(element_offset,
      CheckedAdd(element_offset,
                 CheckedMultiply(index[axis], strides[axis])));
  }
  ASSIGN_OR_RETURN(std::size_t byte_offset,
                   CheckedMultiply(element_offset, SizeOf(dtype)));
  if (byte_offset + SizeOf(dtype) > storage_bytes) {
    return OutOfRange("tensor view exceeds storage");
  }
  return static_cast<std::byte*>(data) + byte_offset;
}
```

Hot Path Kernel 不应逐元素调用此函数；它用于 Parser、Debug Validation 和构建元数据阶段。

## 4. Stream/Event 与 WorkspaceLease

```cpp
struct WorkspaceSlot {
  DeviceBuffer buffer;
  BackendEvent reusable_after;
  bool leased = false;
};

class WorkspacePool {
 public:
  Result<WorkspaceLease> Acquire(std::size_t bytes,
                                 int device,
                                 BackendStream stream);
  Status Release(std::uint32_t slot, BackendEvent completion);

 private:
  std::mutex mutex_;
  std::vector<WorkspaceSlot> slots_;
  std::size_t capacity_bytes_ = 0;
};
```

`Acquire` 只能复用已完成 Event 的 Slot。池达到上限时返回 `ResourceExhausted` 或等待有界时间，不能无界 `hipMalloc`。

典型使用：

```cpp
ASSIGN_OR_RETURN(auto workspace,
                 pool.Acquire(required_bytes, device, stream));
RETURN_IF_ERROR(LaunchAttention(input, workspace.view(), output, stream));
ASSIGN_OR_RETURN(auto done, backend.RecordEvent(stream));
RETURN_IF_ERROR(workspace.ReleaseAfter(std::move(done)));
```

## 5. 初始化与回滚顺序

```text
选择设备并创建 Backend Context
-> 读取/校验 Model Metadata 和 Weight Index（Host）
-> 计算每 Rank 内存预算
-> 分配/上传不可变权重
-> 校验 Weight Sample/Checksum/Finite
-> 创建有界 Workspace Pool
-> 用剩余预算创建 KV Block Pool
```

每一步返回 `Result<T>`。局部对象利用 RAII 自动回滚，禁止使用多个 `goto cleanup` 手动维护几十个指针。

## 6. 内存预算代码骨架

```cpp
Result<MemoryBudget> BuildMemoryBudget(const DeviceMemoryInfo& info,
                                       std::uint64_t weight_bytes,
                                       std::uint64_t communication_bytes,
                                       double safety_fraction) {
  if (safety_fraction < 0.05 || safety_fraction > 0.5) {
    return InvalidArgument("unsafe memory safety fraction");
  }
  const auto safety = static_cast<std::uint64_t>(info.total_bytes * safety_fraction);
  ASSIGN_OR_RETURN(auto persistent,
                   CheckedAdd(weight_bytes, communication_bytes, safety));
  if (persistent >= info.free_bytes) {
    return ResourceExhausted("model shard and runtime buffers do not fit this rank");
  }
  return MemoryBudget{
      .persistent_bytes = persistent,
      .kv_pool_bytes = info.free_bytes - persistent,
  };
}
```

C++17 不支持 Designated Initializer；实际代码使用构造函数或逐字段赋值。示例重点是 Checked Arithmetic 和 Per-rank Budget。

## 7. 测试清单

```cpp
TEST(DeviceBuffer, MoveLeavesSourceEmpty);
TEST(DeviceBuffer, FailedAllocationReturnsStatus);
TEST(TensorView, RejectsOverflowAndOutOfBounds);
TEST(WorkspacePool, DoesNotReuseBeforeEventCompletion);
TEST(WorkspacePool, RespectsCapacityLimit);
TEST(EngineMemory, PartialInitializationReleasesAllBuffers);
```

此外需要 Device Test：异步 Kernel 尚未完成时尝试复用、反复构造/销毁 Engine、注入第 N 次分配失败、KV Exhaustion 后显存/Block 计数恢复。
