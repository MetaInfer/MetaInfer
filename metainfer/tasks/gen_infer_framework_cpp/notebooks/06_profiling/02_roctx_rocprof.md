# ROCTX与rocprof/厂商Profiler

## 1. 能力探测

启动前使用`hardware_profile.json`和只读Probe确认ROCTX Library、rocprof或DTK
等价工具的实际路径与版本。工具不存在时保留内部Trace和GPU Event，不阻止服务；
不得下载或安装系统组件。

## 2. Range设计

原生代码用RAII Range标记Admission、Tokenize、Schedule、Prefill、Decode、Layer、
Attention、MLP、Collective、Sampling和Response。Range名称稳定且包含Step/Layer/
Rank等小型标识，不能把Prompt或用户文本写入Trace。

```cpp
class ProfileRange {
 public:
  explicit ProfileRange(std::string_view name);
  ~ProfileRange();
  ProfileRange(const ProfileRange&) = delete;
};
```

ROCTX实现由Compile/Runtime Capability保护；禁用时编译为空操作。

## 3. 外部启动

Profiling启用时，`serve.sh`或Oracle可在原生Binary外包一层已探测Profiler。命令、
Environment、Exit Code、Artifact路径和工具版本写入Manifest。不要把某个上游
rocprof参数硬编码为所有DTK版本都支持。

## 4. 多Rank与退出

每Rank独立Artifact，文件名含Rank、PID和时间。SIGTERM先停止新工作，再结束
Profiler/Flush Artifact。父Launcher等待工具Wrapper和Native Child，避免只杀
Wrapper留下模型进程。

## 5. 分析

优先检查Range间空洞、Host Launch间隔、Memcpy、同步、Collective和Top Kernel。
必须将观察与固定Benchmark对应；不能从一次启动/权重加载Trace推断稳态Decode。

## 6. 测试

Profiler关闭零Artifact；启用产生非空支持格式；Range成对；多Rank不覆盖；
SIGTERM可读；缺工具正常服务；Profiler失败不泄漏Child进程。

