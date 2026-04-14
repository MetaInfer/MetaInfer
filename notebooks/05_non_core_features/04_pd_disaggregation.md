# Prefill-Decode Disaggregation (PD 分离)

## 概述

PD 分离将推理的两个阶段部署在不同的 GPU 实例上：
- **Prefill 实例**：处理输入 prompt，计算密集型，高算力利用率
- **Decode 实例**：逐 token 生成，内存带宽密集型，高吞吐量

这允许针对不同阶段的特性独立选择硬件和扩缩容策略。

## 为什么是可选但重要的

- **可选**：单机单卡场景不需要，混合 prefill+decode 即可
- **重要**：大规模部署中，PD 分离可提升 40-60% 的整体吞吐，是生产环境的标配

## 架构设计

```
                    ┌─────────────────┐
                    │  Request Router  │  (Gateway / Load Balancer)
                    └────────┬────────┘
                             │
              ┌──────────────┼──────────────┐
              ▼                             ▼
    ┌──────────────────┐          ┌──────────────────┐
    │  Prefill Instance │          │  Decode Instance  │
    │  (KV Producer)    │          │  (KV Consumer)    │
    │                   │   KV     │                   │
    │  1. Receive prompt│ Transfer │  4. Receive KV    │
    │  2. Run prefill   │ ──────→  │  5. Start decode  │
    │  3. Send KV cache │          │  6. Generate tokens│
    └──────────────────┘          └──────────────────┘
```

### 核心组件

| 组件 | Prefill 实例 | Decode 实例 |
|------|-------------|-------------|
| Scheduler | 调度 prefill 任务，完成后触发 KV 发送 | 监听 KV 到达，构建 prebuilt batch |
| KV Transfer | KVSender：序列化并发送 KV 数据 | KVReceiver：接收并写入本地 KV pool |
| Forward Pass | 完整 prefill forward + 第一个 token | 仅 decode forward（跳过 prefill） |
| Memory Pool | 临时持有 KV 直到传输完成 | 预分配空间接收远端 KV |

## 协调机制

### Bootstrap 握手 (sglang 方案)
```python
# Prefill 侧：启动一个小型 TCP/gRPC server
class PrefillBootstrapQueue:
    def wait_for_decode_ready(self, req_id):
        """等待 decode 实例确认已预分配空间"""
        metadata = self.recv_metadata(req_id)  # 包含 decode 侧的内存地址
        return metadata

# Decode 侧：预分配 KV 空间，发送元数据给 prefill
class DecodePreallocQueue:
    def preallocate_and_notify(self, req_id, num_tokens):
        pages = self.cache_manager.allocate_pages(num_tokens)
        self.send_metadata(req_id, pages.gpu_addresses)
```

### KV Lookup Buffer (vllm 方案)
```python
# vllm 使用 KVConnector 抽象层
class KVConnectorBase_V1:
    # Scheduler 侧方法
    def get_num_new_matched_tokens(self, request) -> int:
        """检查远端 cache 是否有该请求的 KV 数据"""

    def update_state_after_alloc(self, request, blocks):
        """分配 block 后更新连接器状态"""

    # Worker 侧方法
    def start_load_kv(self, metadata):
        """异步开始从远端加载 KV 数据"""

    def wait_for_layer_load(self, layer_id):
        """在 attention 计算前等待该层 KV 数据到达"""

    def save_kv_layer(self, layer_id):
        """将该层 KV 数据异步发送到远端"""
```

## Scheduler 变化

### Prefill 侧
```python
class PrefillScheduler(BaseScheduler):
    def process_batch_result(self, batch, results):
        # 标准后处理
        super().process_batch_result(batch, results)

        # === PD 集成点 ===
        for req in batch.finished_prefill_reqs:
            # 触发 KV 发送（异步）
            self.kv_sender.send(req.kv_pages, req.metadata)
            # 加入 inflight 队列等待传输确认
            self.inflight_queue.append(req)

    def poll_inflight(self):
        """检查 KV 传输是否完成"""
        for req in self.inflight_queue:
            if self.kv_sender.is_complete(req):
                self.inflight_queue.remove(req)
                # 释放本地 KV 内存
                self.cache_manager.free(req.kv_pages)
```

### Decode 侧
```python
class DecodeScheduler(BaseScheduler):
    def get_next_batch(self):
        # === PD 集成点 ===
        # 检查是否有新的 KV 数据到达
        ready_reqs = self.kv_receiver.get_ready_requests()
        for req in ready_reqs:
            # 构建 prebuilt batch（跳过 prefill forward）
            req.cached_len = req.total_prompt_len  # 所有 token 已有 KV
            self.decode_queue.append(req)

        return super().get_next_batch()
```

## KV 传输细节

### 传输内容
```python
@dataclass
class KVTransferPayload:
    request_id: str
    kv_data: List[Tensor]      # 每层的 KV 张量
    page_indices: List[int]     # 物理页索引
    num_tokens: int             # token 数量
    metadata: Dict              # 采样参数等
```

### 逐层传输（与计算重叠）
```python
# 高级优化：边计算边传输
class OverlappedKVSender:
    def send_layer(self, layer_id, kv_data):
        """在第 L 层 forward 完成后立即发送该层 KV"""
        # 不等所有层完成，实现 pipeline 重叠
        self.transport.async_send(layer_id, kv_data)
```

## 集成到生成代码的方式

### 最小化集成（3 个钩子）

在标准推理框架基础上，只需添加 3 个集成点：

**钩子 1：Prefill 完成后发送 KV**
```python
# 在 scheduler.postprocess() 中
if self.is_prefill_instance and req.prefill_complete:
    self.kv_sender.send_async(req.kv_pages, req.metadata)
```

**钩子 2：Decode 侧接收 KV**
```python
# 在 scheduler.schedule() 开始时
new_reqs = self.kv_receiver.poll_ready()
for req in new_reqs:
    req.skip_prefill = True
    self.waiting_queue.append(req)
```

**钩子 3：跳过 prefill forward**
```python
# 在 model_runner.run() 中
if batch.skip_prefill:
    # KV 已通过网络传入，无需再做 prefill forward
    # 直接进入 decode 模式
    return self.run_decode(batch)
```

### 配置参数
```python
@dataclass
class PDConfig:
    mode: str = "normal"             # "normal" | "prefill" | "decode"
    transfer_backend: str = "nccl"   # "nccl" | "mooncake" | "nixl" | "tcp"
    ib_device: str = ""              # InfiniBand 设备名（RDMA）
    prefill_url: str = ""            # Prefill 实例地址（decode 侧配置）
    decode_url: str = ""             # Decode 实例地址（prefill 侧配置）
```

## 源码参考

| 项目 | 关键文件 |
|------|---------|
| sglang | `srt/disaggregation/prefill.py`, `srt/disaggregation/decode.py` |
| vllm | `distributed/kv_transfer/kv_connector/v1/base.py` |
| vllm | `config/kv_transfer.py` |
