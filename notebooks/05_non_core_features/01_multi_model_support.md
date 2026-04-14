# 多模型支持（可抽离功能）

## 1. 多模型支持的复杂性

### 1.1 复杂性来源

成熟框架支持大量模型带来巨大复杂性：

```
vLLM: 274+ 模型实现
SGLang: 90+ 模型实现

每种模型可能有：
- 不同的架构细节
- 不同的Attention类型
- 不同的RoPE实现
- 不同的MoE结构
- 不同的位置编码
```

### 1.2 复杂性体现

```python
# vLLM model_executor/models/registry.py
_TEXT_GENERATION_MODELS = {
    "LlamaForCausalLM": ("llama", "LlamaForCausalLM"),
    "MistralForCausalLM": ("llama", "LlamaForCausalLM"),  # 复用
    "Qwen2ForCausalLM": ("qwen2", "Qwen2ForCausalLM"),
    "Qwen2MoeForCausalLM": ("qwen2_moe", "Qwen2MoeForCausalLM"),
    "DeepseekV2ForCausalLM": ("deepseek_v2", "DeepseekV2ForCausalLM"),
    "MixtralForCausalLM": ("mixtral", "MixtralForCausalLM"),
    "GemmaForCausalLM": ("gemma", "GemmaForCausalLM"),
    # ... 274+ 映射
}
```

## 2. 模型注册表模式

### 2.1 注册表设计

```python
# 模型注册表
_MODEL_REGISTRY = {}

def register_model(arch_name: str):
    """模型注册装饰器"""
    def decorator(model_cls):
        _MODEL_REGISTRY[arch_name] = model_cls
        return model_cls
    return decorator

@register_model("LlamaForCausalLM")
class LlamaForCausalLM(nn.Module):
    ...

@register_model("MistralForCausalLM")
class MistralForCausalLM(nn.Module):
    ...

def get_model_class(architecture: str) -> type:
    """根据架构名获取模型类"""
    if architecture not in _MODEL_REGISTRY:
        raise ValueError(f"Unknown architecture: {architecture}")
    return _MODEL_REGISTRY[architecture]
```

### 2.2 动态模型加载

```python
def load_model(config: ModelConfig):
    """根据配置动态加载模型"""
    architectures = config.architectures
    
    for arch in architectures:
        if arch in _MODEL_REGISTRY:
            model_cls = _MODEL_REGISTRY[arch]
            return model_cls(config)
    
    raise ValueError(f"No supported architecture in {architectures}")
```

## 3. 模型差异处理

### 3.1 差异点

| 模型 | RoPE | Attention | MLP | 特殊 |
|------|------|-----------|-----|------|
| LLaMA | 标准 | GQA | SwiGLU | 无 |
| Mistral | 标准 | Sliding Window | SwiGLU | 滑动窗口 |
| Qwen | 变体 | GQA | SwiGLU | 特殊RoPE |
| DeepSeek | 变体 | MLA | MoE | MLA |
| Gemma | 变体 | GQA | GeGLU | 特殊RoPE |

### 3.2 条件分支处理

```python
# 复杂框架中的条件分支
class Attention(nn.Module):
    def __init__(self, config):
        if config.model_type == "llama":
            self.rotary_emb = LlamaRotaryEmbedding(...)
        elif config.model_type == "gemma":
            self.rotary_emb = GemmaRotaryEmbedding(...)
        elif config.model_type == "qwen":
            self.rotary_emb = QwenRotaryEmbedding(...)
        # ...
        
        if hasattr(config, "sliding_window"):
            self.sliding_window = config.sliding_window
        else:
            self.sliding_window = None
```

### 3.3 Mixin模式

```python
# 使用Mixin分离差异
class LlamaAttentionMixin:
    def get_rope(self, config):
        return LlamaRotaryEmbedding(config)

class GemmaAttentionMixin:
    def get_rope(self, config):
        return GemmaRotaryEmbedding(config)

class Attention(LlamaAttentionMixin, nn.Module):
    def __init__(self, config):
        self.rotary_emb = self.get_rope(config)
```

## 4. 抽离建议

### 4.1 精简框架策略

**策略1：只支持一种模型**

```python
# 只支持LLaMA架构
class Model(nn.Module):
    """固定为LLaMA架构，不做模型选择"""
    def __init__(self, config):
        self.embed_tokens = nn.Embedding(...)
        self.layers = nn.ModuleList([LlamaDecoderLayer(...)])
        self.norm = RMSNorm(...)
        self.lm_head = nn.Linear(...)
```

**策略2：通过配置切换**

```python
# 通过配置文件而非代码分支
@dataclass
class ModelConfig:
    model_type: str = "llama"  # 固定，不暴露给用户
    hidden_size: int = 4096
    # ...
```

### 4.2 可抽离的功能

| 功能 | 抽离方式 | 影响 |
|------|----------|------|
| 模型注册表 | 移除 | 减少动态派发 |
| 多架构支持 | 移除 | 只保留一种实现 |
| 特殊RoPE | 外置 | 减少条件分支 |
| 滑动窗口 | 移除 | 简化Attention |

## 5. 模型适配器模式（可选）

如果需要支持有限模型：

```python
class ModelAdapter(ABC):
    """模型适配器基类"""
    @abstractmethod
    def create_model(self, config) -> nn.Module: ...
    
    @abstractmethod
    def load_weights(self, model, state_dict): ...

class LlamaAdapter(ModelAdapter):
    def create_model(self, config):
        return LlamaForCausalLM(config)
    
    def load_weights(self, model, state_dict):
        # LLaMA特定的权重加载逻辑
        ...

# 使用
ADAPTERS = {
    "llama": LlamaAdapter(),
    "mistral": MistralAdapter(),
}

def create_model(config):
    adapter = ADAPTERS[config.model_type]
    return adapter.create_model(config)
```

## 6. 复杂性量化

### 6.1 代码量对比

| 框架 | 模型文件数 | 模型代码行数 |
|------|------------|--------------|
| nano-vllm | 1 | ~200 |
| vLLM | 274 | ~50K |
| SGLang | 90 | ~20K |

### 6.2 维护成本

- 每种模型需要独立测试
- 权重加载需要针对不同格式适配
- 不同模型的特殊配置需要维护

## 7. 总结

### 7.1 核心观点

多模型支持是推理框架臃肿的主要原因之一。对于专用推理框架：

1. **固定模型架构**：只支持一种模型
2. **移除注册表**：不需要动态模型选择
3. **简化配置**：不暴露模型类型参数

### 7.2 如果需要扩展

如果未来需要支持新模型，建议：

1. 创建新的精简框架实例
2. 而非扩展现有框架
3. 保持每个框架的简洁性
