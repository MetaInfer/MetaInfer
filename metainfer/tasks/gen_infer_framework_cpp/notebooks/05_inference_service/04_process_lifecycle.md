# 原生服务进程生命周期

## 1. serve.sh边界

`serve.sh`只解析Port/Model路径、选择已有Build或执行一次有界Build，最后使用
`exec`启动原生Binary。它不得后台化、启动Python Model Worker、每请求编译或
清理未知PID。Oracle拥有服务进程组生命周期。

## 2. Signal模型

Signal Handler只执行Async-signal-safe操作，例如设置原子标志或写Self-pipe。
真正的停止逻辑在主循环：

```text
SIGTERM
-> stop admission/listener
-> mark cancellation/drain
-> stop scheduling new device steps
-> wait bounded in-flight events
-> flush profile
-> close connections
-> release request/KV/workspace
-> release model/backend
-> exit
```

第二次Signal或超时可升级为快速退出，但仍只处理当前任务拥有的资源。

## 3. 多Rank

Launcher转发停止到精确Child PID。Rank 0停止Admission后广播Shutdown；任一Rank
Backend失败触发Collective Abort。父进程必须Reap全部Child并返回首个失败状态。
Worker不得因Rank 0连接关闭而永久阻塞。

## 4. 失败分类

配置/模型错误在监听前退出非零；端口占用不得杀占用者；请求错误只影响请求；
不可恢复Backend错误使服务停止并返回非零。Destructor不抛异常，清理错误记录但
不覆盖首个根因。

## 5. 测试

加载前/加载中/Ready/Decode阶段SIGTERM；第二次Signal；端口冲突；父进程异常；
单Rank错误；客户端连接存在时Drain；Profile Flush；退出后无Child、监听FD和
设备资源残留。

