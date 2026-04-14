# Guided Decoding（约束解码 / 结构化输出）

## 概述

Guided Decoding 在 token 采样时施加约束，确保模型输出符合特定格式（JSON Schema、正则表达式、上下文无关文法等）。核心机制是在每步采样前，根据当前状态计算允许的 token 集合，屏蔽不合法的 token。

## 为什么是可选但重要的

- **可选**：不是所有场景都需要结构化输出
- **重要**：API 服务中 JSON mode / function calling 是核心功能，没有 guided decoding 就无法保证输出格式正确

## 核心架构

```
                                ┌──────────────────┐
用户请求 (JSON Schema/Regex) →  │  Grammar Compiler │ → 编译为 FSM/Grammar
                                └────────┬─────────┘
                                         │ (cached)
                                         ▼
每个 decode step:              ┌──────────────────┐
  当前 FSM 状态 + 词表  →      │  Bitmask Generator │ → 允许的 token bitmask
                                └────────┬─────────┘
                                         │
                                         ▼
                                ┌──────────────────┐
  原始 logits + bitmask  →     │  Logits Masking    │ → 屏蔽后的 logits → Sampler
                                └────────┬─────────┘
                                         │
                                         ▼
  采样得到的 token  →          ┌──────────────────┐
                                │  State Transition  │ → 更新 FSM 状态
                                └──────────────────┘
```

## Grammar 引擎

### 支持的引擎

| 引擎 | 特点 | 性能 |
|------|------|------|
| XGrammar | 最高性能，原生 bitmask 支持 | 最快（推荐） |
| Outlines | 成熟稳定，FSM-based | 中等 |
| LLGuidance | 功能丰富，支持复杂文法 | 中等 |

### XGrammar 使用模式（推荐）

```python
import xgrammar

class XGrammarEngine:
    def __init__(self, tokenizer):
        # 将 tokenizer 的词表编译为 XGrammar 的内部格式
        tokenizer_info = xgrammar.TokenizerInfo.from_huggingface(tokenizer)
        self.compiler = xgrammar.GrammarCompiler(tokenizer_info)
        self.cache = {}  # 编译缓存

    def compile(self, schema_or_regex, mode="json_schema"):
        """将 schema/regex 编译为 grammar"""
        cache_key = (mode, schema_or_regex)
        if cache_key not in self.cache:
            if mode == "json_schema":
                grammar = self.compiler.compile_json_schema(schema_or_regex)
            elif mode == "regex":
                grammar = self.compiler.compile_regex(schema_or_regex)
            elif mode == "ebnf":
                grammar = self.compiler.compile_ebnf(schema_or_regex)
            self.cache[cache_key] = grammar
        return self.cache[cache_key]

    def create_matcher(self, grammar):
        """创建一个有状态的匹配器（每个请求一个）"""
        return xgrammar.GrammarMatcher(grammar)
```

### Bitmask 生成与应用

```python
class GuidedDecodingSampler:
    def __init__(self, vocab_size):
        self.vocab_size = vocab_size
        # 预分配 bitmask 缓冲区
        self.bitmask_buffer = torch.zeros(
            max_batch_size,
            (vocab_size + 31) // 32,  # 每 32 个 token 一个 int32
            dtype=torch.int32,
            device='cuda'
        )

    def apply_grammar_mask(self, logits, batch):
        """在采样前应用语法约束"""
        for i, req in enumerate(batch.reqs):
            if req.grammar_matcher is not None:
                # XGrammar 高效填充 bitmask
                req.grammar_matcher.fill_next_token_bitmask(
                    self.bitmask_buffer[i]
                )

        # 用 Triton kernel 批量应用 bitmask
        apply_bitmask_inplace(logits, self.bitmask_buffer[:len(batch)])
        return logits
```

### Triton Bitmask 应用 Kernel

```python
@triton.jit
def apply_bitmask_kernel(logits_ptr, bitmask_ptr, vocab_size, BLOCK_SIZE: tl.constexpr):
    """将 bitmask 为 0 的位置对应的 logits 设为 -inf"""
    batch_idx = tl.program_id(0)
    token_offset = tl.program_id(1) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = token_offset < vocab_size

    # 读取 bitmask 对应的 bit
    bitmask_idx = token_offset // 32
    bit_pos = token_offset % 32
    bitmask_val = tl.load(bitmask_ptr + batch_idx * bitmask_stride + bitmask_idx, mask=mask)
    is_allowed = (bitmask_val >> bit_pos) & 1

    # 屏蔽不允许的 token
    logits = tl.load(logits_ptr + batch_idx * vocab_size + token_offset, mask=mask)
    logits = tl.where(is_allowed == 1, logits, float('-inf'))
    tl.store(logits_ptr + batch_idx * vocab_size + token_offset, logits, mask=mask)
```

## 状态管理

### 每请求状态
```python
class GuidedRequest:
    def __init__(self, grammar, matcher):
        self.grammar = grammar       # 编译好的 grammar（可共享）
        self.matcher = matcher       # 有状态的匹配器（每请求独立）

    def accept_token(self, token_id):
        """token 被采样后，推进 FSM 状态"""
        self.matcher.accept_token(token_id)

    def is_terminated(self):
        """检查 grammar 是否已到达终止状态"""
        return self.matcher.is_terminated()

    def rollback(self, num_tokens):
        """回滚状态（用于投机解码的 rejection）"""
        self.matcher.rollback(num_tokens)
```

### 与投机解码的交互
Guided decoding 必须与投机解码正确配合：
```python
def speculative_with_grammar(draft_tokens, target_logits, req):
    accepted = []
    for i, draft_tok in enumerate(draft_tokens):
        # 在 accept 前检查 grammar
        if req.matcher.is_allowed(draft_tok):
            req.matcher.accept_token(draft_tok)
            accepted.append(draft_tok)
        else:
            # Grammar 不允许该 token → 直接拒绝
            req.matcher.rollback(i)  # 回滚所有已 accept 的
            # 从 target logits 中重采样（已施加 grammar mask）
            new_token = sample_with_grammar(target_logits[i], req.matcher)
            accepted.append(new_token)
            break
    return accepted
```

## 支持的约束格式

### JSON Schema
```python
# 用户请求
{
    "response_format": {
        "type": "json_schema",
        "json_schema": {
            "name": "user_info",
            "schema": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "age": {"type": "integer"}
                },
                "required": ["name", "age"]
            }
        }
    }
}

# 编译为 grammar
grammar = compiler.compile_json_schema(schema_string)
```

### 正则表达式
```python
# 用户请求
{"guided_regex": r"\d{4}-\d{2}-\d{2}"}  # 日期格式

grammar = compiler.compile_regex(r"\d{4}-\d{2}-\d{2}")
```

### EBNF 文法
```python
# 自定义文法
grammar_str = """
root ::= "{"  key_value ("," key_value)* "}"
key_value ::= "\"" key "\"" ":" value
key ::= [a-zA-Z_]+
value ::= "\"" [^"]* "\"" | number
number ::= [0-9]+
"""
grammar = compiler.compile_ebnf(grammar_str)
```

### 选项列表
```python
# 限制输出为几个固定选项之一
choices = ["yes", "no", "maybe"]
# 内部转换为 EBNF: root ::= "yes" | "no" | "maybe"
grammar = compiler.compile_choices(choices)
```

## 集成到生成代码的方式

### 修改采样流程

在 Sampler 的 forward 方法中插入一个钩子：

```python
class Sampler:
    def __init__(self, vocab_size, grammar_engine=None):
        self.grammar_engine = grammar_engine

    def forward(self, logits, batch):
        # === Guided Decoding 集成点 ===
        if self.grammar_engine:
            logits = self.grammar_engine.apply_grammar_mask(logits, batch)

        # 正常采样流程
        logits = logits / temperatures
        if has_top_k:
            apply_top_k(logits)
        probs = softmax(logits)
        tokens = multinomial(probs)

        # === 状态更新集成点 ===
        if self.grammar_engine:
            for i, (req, token) in enumerate(zip(batch.reqs, tokens)):
                if req.grammar_matcher:
                    req.grammar_matcher.accept_token(token.item())

        return tokens
```

### 修改请求初始化

```python
class RequestHandler:
    def create_request(self, params):
        req = Request(...)

        # === Guided Decoding 集成点 ===
        if params.json_schema:
            grammar = self.grammar_engine.compile(params.json_schema, "json_schema")
            req.grammar_matcher = self.grammar_engine.create_matcher(grammar)
        elif params.regex:
            grammar = self.grammar_engine.compile(params.regex, "regex")
            req.grammar_matcher = self.grammar_engine.create_matcher(grammar)

        return req
```

### 需要修改的组件

| 组件 | 修改内容 |
|------|---------|
| Sampler | 添加 bitmask 应用逻辑（在 softmax 前） |
| Request | 添加 `grammar_matcher` 字段 |
| Scheduler postprocess | 添加 `accept_token()` 调用 |
| API Server | 解析 `response_format` / `guided_json` 参数 |
| 配置 | 选择 grammar 引擎 |

### 配置参数
```python
@dataclass
class GuidedDecodingConfig:
    backend: str = "xgrammar"    # "xgrammar" | "outlines" | "none"
    max_cached_grammars: int = 256  # LRU 缓存大小
    async_compile: bool = True      # 异步编译（不阻塞调度）
```

## 源码参考

| 项目 | 关键文件 |
|------|---------|
| sglang | `srt/constrained/grammar_manager.py` |
| sglang | `srt/constrained/xgrammar_backend.py` |
| sglang | `srt/constrained/triton_ops/bitmask_ops.py` |
| vllm | `v1/structured_output/backend_xgrammar.py` |
| vllm | `v1/worker/gpu/structured_outputs.py` |
| nano-sglang | `srt/constrained/fsm.py` (简化版 FSM) |
