# 硬件探测与能力门控

基础探测由 Orchestrator 中的确定性代码完成，并写入 `hardware_profile.json`。Agent 的职责是读取事实，并为本次实现运行“小而有界”的二次编译/执行 Probe，而不是重新发明硬件身份。

## 1. 只读探测来源

```text
lspci -D -nn                   PCI 产品、BDF、Vendor/Device/Revision ID
rocminfo                       HIP Runtime Agent 与编译架构
rocm-smi --show*               驱动别名、显存、拓扑、利用率
nvidia-smi --query-gpu=...     NVIDIA 身份、显存、驱动
hipconfig --full               HIP 配置
hipcc/clang++/cmake --version  工具链
ldconfig -p                    Runtime/BLAS/Collective Library
/dev/kfd、renderD* stat        当前用户设备权限
Visible Device 环境变量       管理员/调度器分配范围
```

禁止执行 Reset、Clock/Power 修改、Firmware、Permission 修改、Package Install。

## 2. 身份优先级

```text
用户选择                决定目标支持范围
PCI                     识别物理板卡/ID
SMI                     识别驱动状态，名称可能是兼容别名
rocminfo architecture    决定 HIP 编译目标
hipGetDeviceProperties   决定 Runtime Launch/Resource 参数
```

不要压缩成一个没有来源的 `gpu_name`。

建议 C++ 读取精简画像：

```cpp
struct HardwareProfile {
  std::string requested_hardware;
  std::string backend;
  std::vector<std::string> hip_architectures;
  std::vector<int> visible_devices;
  int physical_device_count = 0;
  bool kfd_readable = false;
  bool kfd_writable = false;
  std::vector<std::string> blockers;
};

Result<HardwareProfile> LoadHardwareProfile(const std::filesystem::path& path);
```

Parser 必须对缺失字段、类型错误和未知 `schema_version` 给出明确错误。

## 3. 可见设备与逻辑编号

多人共享服务器上，`rocm-smi` 可能展示宿主机全部四张卡，但当前任务只被分配两张。程序只允许使用：

```text
HIP_VISIBLE_DEVICES
ROCR_VISIBLE_DEVICES
CUDA_VISIBLE_DEVICES
```

逻辑设备 0 是“当前任务可见列表中的第一张卡”，不一定是宿主机物理 0 号卡。

```cpp
Result<std::vector<int>> ParseVisibleDeviceList(std::string_view text) {
  if (text.empty() || text == "-1") return std::vector<int>{};
  std::vector<int> result;
  for (std::string_view part : Split(text, ',')) {
    ASSIGN_OR_RETURN(int index, ParseNonNegativeInt(Trim(part)));
    result.push_back(index);
  }
  RETURN_IF_ERROR(RejectDuplicates(result));
  return result;
}
```

生成服务不得设置更宽的可见性变量。显式 `assigned_devices` 已由 Orchestrator 写入环境，子进程只继承。

## 4. 原生 HIP Device Probe

下面是 Probe 的核心流程示意：

```cpp
#include <hip/hip_runtime.h>

Result<DeviceProbeResult> ProbeDevice(int logical_device) {
  HIP_RETURN_IF_ERROR(hipSetDevice(logical_device));

  hipDeviceProp_t prop{};
  HIP_RETURN_IF_ERROR(hipGetDeviceProperties(&prop, logical_device));

  constexpr std::size_t kCount = 256;
  std::array<float, kCount> host{};
  std::iota(host.begin(), host.end(), 0.0f);

  ASSIGN_OR_RETURN(DeviceBuffer device,
                   DeviceBuffer::Allocate(kCount * sizeof(float), logical_device));
  HIP_RETURN_IF_ERROR(hipMemcpy(device.data(), host.data(), device.bytes(),
                                hipMemcpyHostToDevice));
  LaunchAddOne(static_cast<float*>(device.data()), kCount, /*stream=*/nullptr);
  HIP_RETURN_IF_ERROR(hipGetLastError());
  HIP_RETURN_IF_ERROR(hipMemcpy(host.data(), device.data(), device.bytes(),
                                hipMemcpyDeviceToHost));

  for (std::size_t i = 0; i < kCount; ++i) {
    if (host[i] != static_cast<float>(i) + 1.0f) {
      return BackendError("HIP probe result mismatch");
    }
  }
  return DeviceProbeResult::From(prop);
}
```

该 Probe 同时证明：Runtime 可访问、当前用户权限正常、Memory Copy/Kernel Launch 可用。仅看到 `hipcc --version` 不足以证明计算可用。

## 5. 能力门控

把能力表示为“布尔值 + 证据”：

```cpp
struct CapabilityEvidence {
  bool supported = false;
  std::string probe_name;
  std::string compiler_output;
  std::string runtime_output;
};

struct BackendCapabilities {
  CapabilityEvidence bf16_gemm;
  CapabilityEvidence peer_access;
  CapabilityEvidence collectives;
  CapabilityEvidence profiling_ranges;
};
```

分派时：

```cpp
if (caps.bf16_gemm.supported) {
  return CreateVendorBlasBf16Gemm(...);
}
return Unsupported("BF16 GEMM probe did not pass on this DTK installation");
```

未知不等于 False 也不等于 True。未知能力要么继续运行 Probe，要么报告 Unsupported。

## 6. P2P/TP Probe

```cpp
for (int src = 0; src < visible_count; ++src) {
  for (int dst = 0; dst < visible_count; ++dst) {
    int can_access = 0;
    HIP_RETURN_IF_ERROR(hipDeviceCanAccessPeer(&can_access, src, dst));
    matrix[src][dst] = can_access != 0;
  }
}
```

P2P 支持只是通信策略输入，不能代替 Collective Smoke Test。TP=4 必须在四个实际 Rank 上验证 Broadcast/AllReduce。

## 7. 不匹配处理

- 厂商/Backend 不匹配：阻止 Build/Run；
- PCI=Z200SM、SMI=Z100SM：保留两者并 Warning，编译仍看 Runtime Architecture；
- 无 HIP Architecture：禁止架构专用编译，要求 Probe/Override；
- `/dev/kfd`/render node 无权限：环境 blocker，禁止 CPU 回退；
- 可见设备少于 TP Size：加载模型前失败；
- BLAS/Collective 缺失：选择有 Contract 的 Native 替代或明确 Unsupported。

## 8. 必测异常

```text
hardware_profile.json 不存在/损坏/schema 不支持
Visible Device 字符串非法或重复
SMI 与 PCI 名称冲突
hipGetDeviceCount=0
hipSetDevice/hipMalloc Permission Denied
请求 TP=4 但只看到 2 张卡
Header 可编译但 Runtime Library 无法加载
```
