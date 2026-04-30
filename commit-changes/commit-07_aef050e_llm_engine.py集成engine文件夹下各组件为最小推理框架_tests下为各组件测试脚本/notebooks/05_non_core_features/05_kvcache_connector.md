# KVCache Connector（KV 缓存传输连接器）

## 概述

KVCache Connector 是 KV 缓存在不同 GPU/节点之间传输的抽象层。它是 PD 分离、KV 缓存共享、请求迁移等高级功能的基础设施。

## 为什么是可选但重要的

- **可选**：单机部署不需要跨节点 KV 传输
- **重要**：PD 分离、多实例缓存共享、弹性扩缩容都依赖它

## 连接器接口

### vllm 设计（最完整的抽象）

vllm 将连接器分为 Scheduler 侧和 Worker 侧两组方法：

```python
class KVConnectorBase_V1(ABC):
    """KV 连接器的标准接口"""

    # ============ Scheduler 侧（运行在调度进程） ============

    @abstractmethod
    def get_num_new_matched_tokens(self, request) -> int:
        """查询远端是否有该请求的 KV 缓存
        返回可复用的 token 数量（类似 prefix cache hit）"""

    @abstractmethod
    def update_state_after_alloc(self, request, block_ids):
        """本地 block 分配完成后，更新连接器状态
        （记录哪些 block 需要从远端加载）"""

    @abstractmethod
    def build_connector_meta(self, scheduler_output) -> bytes:
        """序列化传输元数据，发送给 Worker 侧"""

    @abstractmethod
    def request_finished(self, request, block_ids) -> bool:
        """请求完成时调用，返回 delay_free_blocks=True
        表示 block 不能立即释放（需等传输完成）"""

    # ============ Worker 侧（运行在 GPU Worker 进程） ============

    @abstractmethod
    def register_kv_caches(self, kv_caches: List[Tensor]):
        """注册本地 KV 缓存张量，提取 GPU 物理地址"""

    @abstractmethod
    def start_load_kv(self, metadata):
        """异步开始加载 KV 数据（非阻塞）"""

    @abstractmethod
    def wait_for_layer_load(self, layer_id):
        """等待指定层的 KV 数据加载完成（可被 attention 前调用）"""

    @abstractmethod
    def save_kv_layer(self, layer_id):
        """异步保存指定层的 KV 数据到远端"""

    @abstractmethod
    def get_finished(self) -> List[str]:
        """查询已完成传输的请求 ID 列表"""
```

### sglang 设计（存储导向）

```python
class BaseKVConnector(ABC):
    """sglang 的 KV 连接器，偏存储接口风格"""

    @abstractmethod
    def batch_get_v1(self, keys, local_pages, remote_pages):
        """批量从远端获取 KV 数据到本地 pages"""

    @abstractmethod
    def batch_set_v1(self, keys, local_pages):
        """批量将本地 KV 数据存储到远端"""

    @abstractmethod
    def batch_exists(self, keys) -> List[bool]:
        """检查远端是否存在指定 key 的 KV 数据"""

    @abstractmethod
    def register_mem_pool_host(self, host_kv_cache):
        """注册本地内存池"""
```

## 传输协议

### 支持的后端

| 后端 | 协议 | 特点 | 适用场景 |
|------|------|------|---------|
| NCCL (P2P) | NVLink / PCIe | 最低延迟，需要同机或 NVSwitch | 同机多卡 PD |
| NIXL | RDMA (UCX) / TCP | 高性能跨节点，零拷贝 | 跨节点 PD |
| Mooncake | RDMA / TCP | 分布式 KV 存储，自带路由 | 大规模集群 |
| TCP | TCP Socket | 最通用，性能最低 | 开发调试 |
| 共享文件系统 | NFS/FUSE | 持久化 KV 缓存 | 离线预计算 |

### 传输数据格式

```python
# 物理地址计算
def get_physical_addr(kv_tensor, block_id, block_size):
    """将逻辑 block_id 转换为 GPU 物理地址"""
    base_addr = kv_tensor.data_ptr()
    block_bytes = block_size * num_heads * head_dim * dtype_size * 2  # K+V
    return base_addr + block_id * block_bytes

# 元数据（通过 msgpack 序列化）
@dataclass
class TransferMetadata:
    engine_id: str              # 发送方实例 ID
    request_id: str             # 请求 ID
    block_ids: List[int]        # 要传输的 block 列表
    num_tokens: int             # token 数量
    remote_block_ids: List[int] # 接收方预分配的 block 列表
```

## 地址翻译与内存布局

### 逻辑 Block → 物理地址
```python
class KVAddressTranslator:
    def __init__(self, kv_caches, block_size, layout="NHD"):
        # kv_caches: List[Tensor]，每层一个，shape 取决于 layout
        # NHD: [num_blocks, 2, num_heads, block_size, head_dim]
        # HND: [num_blocks, 2, block_size, num_heads, head_dim]
        self.base_addrs = [kv.data_ptr() for kv in kv_caches]
        self.block_bytes = self._calc_block_bytes()

    def translate(self, layer_id, block_id):
        return self.base_addrs[layer_id] + block_id * self.block_bytes
```

### 异构 TP 支持
当 prefill 和 decode 使用不同的 TP 度时，需要额外的切分/合并：
```python
# 例：Prefill TP=4, Decode TP=2
# Prefill 每个 rank 有 8 个 KV head
# Decode 每个 rank 有 16 个 KV head
# 需要将 Prefill 的 2 个 rank 的 KV 合并发送给 Decode 的 1 个 rank
tp_ratio = prefill_tp_size // decode_tp_size
for decode_rank in range(decode_tp_size):
    combined_kv = concat([
        prefill_ranks[decode_rank * tp_ratio + i].kv
        for i in range(tp_ratio)
    ])
    send_to_decode(decode_rank, combined_kv)
```

## 集成到生成代码的方式

### 接口层设计

生成代码时，应提供一个简洁的连接器接口：

```python
class KVConnector(ABC):
    """最小化 KV 连接器接口"""

    @abstractmethod
    def send_kv(self, request_id: str, kv_pages: List[int],
                dest_addr: str) -> Future:
        """异步发送 KV 数据到目标地址"""

    @abstractmethod
    def recv_kv(self, request_id: str, local_pages: List[int]) -> Future:
        """异步接收 KV 数据到本地 pages"""

    @abstractmethod
    def is_ready(self, request_id: str) -> bool:
        """检查传输是否完成"""
```

### 具体后端实现模板

```python
class NCCLKVConnector(KVConnector):
    def __init__(self, kv_caches, nccl_group):
        self.kv_caches = kv_caches
        self.group = nccl_group

    def send_kv(self, request_id, kv_pages, dest_rank):
        for layer_id, kv in enumerate(self.kv_caches):
            data = kv[kv_pages]  # Gather KV data
            dist.send(data, dst=dest_rank, group=self.group)

    def recv_kv(self, request_id, local_pages, src_rank):
        for layer_id, kv in enumerate(self.kv_caches):
            buffer = torch.empty_like(kv[local_pages])
            dist.recv(buffer, src=src_rank, group=self.group)
            kv[local_pages] = buffer
```

### 与调度器的集成

```python
# 在 scheduler 中添加连接器相关逻辑
class Scheduler:
    def __init__(self, ..., kv_connector=None):
        self.kv_connector = kv_connector

    def postprocess(self, batch, tokens):
        super().postprocess(batch, tokens)
        if self.kv_connector and self.mode == "prefill":
            for req in batch.finished:
                self.kv_connector.send_kv(req.id, req.kv_pages, req.dest)

    def schedule(self):
        if self.kv_connector and self.mode == "decode":
            ready = [r for r in self.pending if self.kv_connector.is_ready(r.id)]
            self.decode_queue.extend(ready)
        return super().schedule()
```

## 源码参考

| 项目 | 关键文件 |
|------|---------|
| vllm | `distributed/kv_transfer/kv_connector/v1/base.py` |
| vllm | `distributed/kv_transfer/kv_connector/v1/nixl/` |
| vllm | `distributed/kv_transfer/kv_connector/v1/mooncake/` |
| sglang | `srt/connector/base_connector.py` |
| sglang | `srt/mem_cache/storage/nixl/hicache_nixl.py` |
