# RCCL/HIP Collective封装

本文面向Hygon DTK/HIP和兼容ROCm栈。具体Header、符号、版本和功能必须由目标
环境Probe，不能假定上游最新RCCL API存在。

## 1. 抽象边界

模型层只依赖`Collective`接口，不包含厂商Handle。实现至少提供Broadcast、
AllReduce、AllGather、Barrier/Health Probe和Abort。Unsupported操作在初始化时
暴露Capabilities。

```cpp
struct CollectiveCapabilities {
  bool broadcast;
  bool all_reduce;
  bool all_gather;
  bool all_to_all;
  bool async_error_query;
};
```

## 2. 初始化

每个Rank先选择Device、创建Communication Stream，再通过Rendezvous交换唯一ID并
创建Communicator。初始化设置有限超时；任一Rank失败时所有已创建Communicator
进入Abort/Destroy路径。Rank 0监听HTTP之前必须完成小Buffer Probe。

## 3. 顺序与类型

所有Rank按完全一致的Sequence、Kind、Count、DType和Root调用Collective。封装层
记录递增Sequence和摘要，Hang时可以比较最后成功记录。C++ DType到厂商DType的
映射集中维护；BF16等类型只有Compile/Runtime Probe通过才启用。

## 4. Stream与Buffer

Collective在调用方提供的Communication Stream上排队。Compute结果通过Event
等待，通信结果再通过Event交给Compute。Buffer在通信Completion前不得复用。
In-place/Out-of-place语义必须与厂商API匹配并有重叠检测。

## 5. 错误传播

同步返回值立即检查；异步错误在Step边界轮询。任意Rank检测错误后，先停止新
Step，记录首错，再Abort Communicator并通知Launcher协调退出。不能让其他Rank
无限等待，也不能在错误后继续返回HTTP成功。

## 6. 共享主机安全

只使用卡片分配并通过Visibility暴露的设备。不得Reset设备、修改权限或终止未由
当前Launcher创建的进程。端口/Rendezvous冲突选择新资源，不清理其他用户任务。

## 7. 测试

1/2/4 Rank的Broadcast、AllReduce和AllGather；非方形字节数和多DType；故意
Count/Sequence不一致的有界诊断；单Rank退出；Compute/Communication Event依赖；
重复初始化/销毁；所有期望设备活动遥测。

