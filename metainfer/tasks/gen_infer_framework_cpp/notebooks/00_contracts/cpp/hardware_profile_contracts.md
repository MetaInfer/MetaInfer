# 原生硬件画像契约（Native Hardware Profile Contract）

> 权威级别：`gen-infer-framework-cpp` 的强制契约。

Orchestrator 必须在 A 阶段开始前生成 `hardware_profile.json`。该文件描述的是“当前任务进程实际能看到和使用的硬件环境”，而不是根据用户选择的产品名称推测出来的配置。

Agent 不得用经验值覆盖画像中的事实，也不得因为画像缺少某个字段就自行假设该能力存在。

## 1. 四种身份必须分开保存

```text
requested.target_hardware       用户期望的部署目标，例如 Hygon Z200SM
detected.pci_devices            PCI 总线识别出的物理板卡和 PCI ID
detected.smi_devices            驱动管理工具返回的产品/兼容名称
detected.hip_architectures      rocminfo 返回的 HIP 编译架构，例如 gfx***
```

这四个值可能不完全一致。例如 PCI 可能显示 `Z200SM`，DTK 管理层可能显示兼容名称 `Z100SM`。此时必须同时保留两个名称：

- 物理产品识别优先看 PCI 证据；
- HIP 编译参数只允许使用 `rocminfo`/HIP Runtime 返回的架构；
- 禁止把 `Z200SM` 之类的营销名称直接转换或猜测成 `gfx*`。

## 2. 必需字段

画像至少应包含：

- `schema_version`、生成时间、主机名、操作系统和 CPU 架构；
- 用户选择的目标硬件、后端和显式分配设备；
- `CUDA_VISIBLE_DEVICES`、`HIP_VISIBLE_DEVICES`、`ROCR_VISIBLE_DEVICES`；
- PCI BDF、Vendor ID、Device ID、Revision、产品名称；
- 物理设备数量与当前任务可见设备数量；
- 每卡显存、驱动版本、HIP 架构；
- `hipconfig`、`hipcc`、Clang、CMake 的路径和版本；
- 已发现的 HIP、HSA、BLAS、Collective/通信库；
- `/dev/kfd` 与 `/dev/dri/renderD*` 的读写权限；
- 兼容性状态、warning、blocker；
- 每条探测命令的返回码、受限长度 stdout/stderr。

推荐的精简结构示例：

```json
{
  "schema_version": 1,
  "requested": {
    "target_hardware": "Hygon Z200SM",
    "accelerator_backend": "Hygon DTK / HIP",
    "assigned_devices": "0,1,2,3"
  },
  "visibility": {
    "HIP_VISIBLE_DEVICES": "0,1,2,3",
    "ROCR_VISIBLE_DEVICES": "0,1,2,3"
  },
  "detected": {
    "vendor_family": "hygon",
    "physical_device_count": 4,
    "device_count": 4,
    "hip_architectures": ["<rocminfo-result>"]
  },
  "validation": {
    "status": "compatible_with_warnings",
    "runnable": true,
    "warnings": ["PCI and SMI product names differ"],
    "blockers": []
  }
}
```

示例中的架构占位符不能被复制为真实编译参数。

## 3. C++ 侧消费接口

生成框架应把解析后的硬件能力收敛到一个只读结构，而不是让各模块重复读取环境变量：

```cpp
struct DeviceIdentity {
  int logical_index = -1;
  std::string pci_bdf;
  std::string pci_product_name;
  std::string smi_product_name;
  std::string hip_architecture;
  std::uint64_t total_memory_bytes = 0;
};

struct HardwareCapabilities {
  std::string vendor_family;
  std::string backend;
  std::vector<DeviceIdentity> visible_devices;
  bool kfd_accessible = false;
  bool hip_runtime_available = false;
  bool blas_available = false;
  bool collectives_available = false;
};
```

该结构应在进程启动后只初始化一次，并作为 `const HardwareCapabilities&` 传给 Backend、ModelLoader 和 TP 初始化逻辑。

## 4. 强制规则

- **HP-001**：HIP 架构参数只能来自画像或用户显式覆盖；显式覆盖必须写入 Build Manifest。
- **HP-002**：生成程序必须继承设备可见性，禁止主动扩大到隐藏设备。
- **HP-003**：TP Size 不得超过可见且可访问的设备数量。
- **HP-004**：`/dev/kfd`/render node 无权限属于环境 blocker，禁止回退 CPU。
- **HP-005**：目标型号、厂商、后端不匹配时，必须在日志和 Oracle 中显式失败。
- **HP-006**：未知能力只能选择通用实现并运行编译/执行探针，不能按“可能支持”处理。
- **HP-007**：探测过程禁止 `sudo`、`chmod`、驱动重置、设备重置和在线安装软件。
- **HP-008**：SMI 能看到宿主机全部卡时，不能把物理卡数误当成当前任务可用卡数。

## 5. 多人共享服务器安全规则

GPU 占用不是进程所有权证据。只有同时满足以下条件，才允许终止进程：

1. PID 由当前 MetaInfer 任务明确创建并登记；
2. PID 尚未被复用，启动时间/命令仍匹配；
3. 进程属于当前 Unix UID；
4. 先发送 SIGTERM，超时后才允许处理同一个 PID。

进程名、cwd、端口、显存占用、打开 render node 都不能单独作为清理依据。

## 6. 验收条件

在编译框架前，至少运行一个原生 HIP Probe：

```cpp
int count = 0;
HIP_CHECK(hipGetDeviceCount(&count));
for (int i = 0; i < count; ++i) {
  hipDeviceProp_t prop{};
  HIP_CHECK(hipGetDeviceProperties(&prop, i));
  // 输出逻辑设备、架构、显存和线程/共享内存限制到 JSON 日志。
}
```

在宣称 TP 可用前，必须在“确切分配的 Rank/设备”上执行一次 Broadcast/AllReduce Smoke Test，并保存编译命令、运行结果和设备映射。
