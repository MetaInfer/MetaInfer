# C++推理框架经验库

本目录只保存已经在C++卡片中复现、修复并通过对应Oracle的经验。未解决现象放入
`08_issues/`，尚未执行的方案放入`07_improvementPlan/`。

每份经验必须记录：

```text
标题与日期
模型/Checkpoint
硬件、Backend、SDK和Build
症状与最小复现
错误假设和排除证据
根因
文件/API级修复
正确性与性能验证
可推广不变量
不适用范围
```

只有已经验证的可推广不变量才提升到`00_contracts/`。不得复制Python、CUDA或
其他任务的修复并标记为C++/HIP已验证经验。

