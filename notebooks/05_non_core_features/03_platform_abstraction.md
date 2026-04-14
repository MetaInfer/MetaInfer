# 平台抽象层（可抽离功能）

## 1. 多平台支持的复杂性

### 1.1 支持的平台

成熟框架支持多种硬件平台：

| 平台 | 厂商 | 特点 |
|------|------|------|
| CUDA | NVIDIA | 最成熟，生态最好 |
| ROCm | AMD | 开源，兼容CUDA |
| TPU | Google | 专用AI加速器 |
| XPU | Intel | GPU和专用加速器 |
| NPU | 华为等 | 国产AI芯片 |
| MPS | Apple | Mac GPU |
| CPU | 通用 | 最广泛支持 |

### 1.2 复杂性体现

```python
# vLLM platforms/ 目录
PLATFORMS = {
    "cuda": CudaPlatform,
    "rocm": RocmPlatform,
    "tpu": TpuPlatform,
    "xpu": XpuPlatform,
    "cpu": CpuPlatform,
}

# 代码中大量的平台判断
if is_cuda():
    from vllm.attention.backends.flash_attn import FlashAttentionBackend
elif is_rocm():
    from vllm.attention.backends.rocm_attn import ROCmAttentionBackend
elif is_xpu():
    from vllm.attention.backends.xpu_attn import XPUAttentionBackend
# ...
```

## 2. 平台抽象层设计

### 2.1 Platform基类

```python
class Platform(ABC):
    """平台抽象基类"""
    
    @property
    @abstractmethod
    def device_name(self) -> str:
        """设备名称 (cuda, rocm, etc.)"""
        pass
    
    @property
    @abstractmethod
    def device_type(self) -> DeviceType:
        """设备类型"""
        pass
    
    @abstractmethod
    def get_attn_backend(self, ...) -> type:
        """获取Attention后端"""
        pass
    
    @abstractmethod
    def get_communication_backend(self) -> str:
        """获取通信后端 (nccl, gloo, etc.)"""
        pass
    
    @staticmethod
    @abstractmethod
    def get_device_memory() -> tuple[int, int]:
        """获取设备内存 (free, total)"""
        pass
```

### 2.2 具体平台实现

```python
class CudaPlatform(Platform):
    """NVIDIA CUDA平台"""
    
    @property
    def device_name(self) -> str:
        return "cuda"
    
    def get_attn_backend(self, ...):
        return FlashAttentionBackend
    
    def get_communication_backend(self) -> str:
        return "nccl"
    
    @staticmethod
    def get_device_memory():
        return torch.cuda.mem_get_info()

class RocmPlatform(Platform):
    """AMD ROCm平台"""
    
    @property
    def device_name(self) -> str:
        return "cuda"  # ROCm兼容CUDA API
    
    def get_attn_backend(self, ...):
        return ROCmAttentionBackend
    
    def get_communication_backend(self) -> str:
        return "nccl"  # RCCL

class CpuPlatform(Platform):
    """CPU平台"""
    
    @property
    def device_name(self) -> str:
        return "cpu"
    
    def get_attn_backend(self, ...):
        return CPUAttentionBackend
    
    def get_communication_backend(self) -> str:
        return "gloo"
```

## 3. 平台差异点

### 3.1 Attention后端

| 平台 | 可用后端 | 最佳选择 |
|------|----------|----------|
| CUDA | FlashAttention, FlashInfer, Triton | FlashAttention |
| ROCm | CK (Composable Kernel), Triton | CK |
| TPU | Pallas | Pallas |
| CPU | PyTorch Native | torch.scaled_dot_product_attention |

### 3.2 通信后端

| 平台 | 通信库 | NCCL兼容 |
|------|--------|----------|
| CUDA | NCCL | 原生 |
| ROCm | RCCL | 兼容NCCL API |
| TPU | PyTorch XLA | 特殊 |
| CPU | Gloo | 无GPU |

### 3.3 其他差异

| 特性 | CUDA | ROCm | TPU | CPU |
|------|------|------|-----|-----|
| CUDA Graph | ✓ | ✓ | ✗ | ✗ |
| FP8 | Hopper+ | MI300+ | ✗ | ✗ |
| Paged Attention | ✓ | ✓ | ✓ | ✓ |
| Quantization | 丰富 | 有限 | 有限 | 有限 |

## 4. 平台检测与选择

### 4.1 自动检测

```python
def detect_platform() -> Platform:
    """自动检测当前平台"""
    if torch.cuda.is_available():
        if torch.version.cuda:
            return CudaPlatform()
        elif torch.version.hip:
            return RocmPlatform()
    elif hasattr(torch, 'xpu') and torch.xpu.is_available():
        return XpuPlatform()
    else:
        return CpuPlatform()
```

### 4.2 运行时选择

```python
# 全局平台实例
_current_platform: Platform = None

def get_platform() -> Platform:
    global _current_platform
    if _current_platform is None:
        _current_platform = detect_platform()
    return _current_platform

def set_platform(platform: Platform):
    global _current_platform
    _current_platform = platform
```

## 5. 条件分支模式

### 5.1 散布的条件判断

```python
# 在各处散布的平台判断
def some_function():
    platform = get_platform()
    
    if platform.device_name == "cuda":
        # CUDA特定逻辑
        torch.cuda.synchronize()
    elif platform.device_name == "rocm":
        # ROCm特定逻辑
        torch.cuda.synchronize()  # ROCm兼容
    elif platform.device_name == "cpu":
        # CPU特定逻辑
        pass
```

### 5.2 更好的模式：委托

```python
# 使用委托模式减少条件分支
class Platform:
    def synchronize(self):
        """同步设备"""
        pass

class CudaPlatform(Platform):
    def synchronize(self):
        torch.cuda.synchronize()

class CpuPlatform(Platform):
    def synchronize(self):
        pass  # CPU不需要同步

# 使用
platform.synchronize()
```

## 6. 抽离建议

### 6.1 精简框架策略

**策略：只支持CUDA**

```python
# 直接使用CUDA API，不做平台抽象
def allocate_memory(size):
    return torch.empty(size, device="cuda")

def synchronize():
    torch.cuda.synchronize()

def get_memory_info():
    return torch.cuda.mem_get_info()
```

### 6.2 如果需要多平台

如果确实需要支持多平台：

```python
# 最小化平台抽象
class Platform:
    @staticmethod
    def get_memory():
        return torch.cuda.mem_get_info()  # ROCm兼容

# 通过配置而非运行时检测
PLATFORM = "cuda"  # 编译时确定
```

## 7. 复杂性量化

### 7.1 代码量

| 平台 | 特定代码量 |
|------|------------|
| CUDA | 核心 |
| ROCm | ~2K行 |
| TPU | ~3K行 |
| XPU | ~2K行 |
| NPU | ~5K行 |
| CPU | ~1K行 |

### 7.2 维护成本

- 每个平台需要独立测试
- 不同平台的性能特性不同
- 新功能需要适配所有平台

## 8. 总结

### 8.1 核心观点

多平台支持增加了框架复杂度，但对于专用推理框架：

1. 目标平台通常确定
2. 平台抽象层是过度设计
3. 直接使用特定平台API更简洁

### 8.2 精简建议

对于精简框架：
1. 只支持CUDA（或单一目标平台）
2. 移除平台抽象层
3. 直接调用平台特定API
