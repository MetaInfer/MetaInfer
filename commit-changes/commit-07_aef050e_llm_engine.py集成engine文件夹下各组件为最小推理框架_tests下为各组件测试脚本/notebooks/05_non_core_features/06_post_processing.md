# 后处理逻辑（Post-Processing）

## 概述

后处理涵盖 token 采样之后到最终返回用户之间的所有逻辑：增量解码（detokenization）、停止条件检测、流式输出、结果格式化等。

## 为什么是可选但重要的

- **可选**：离线批量推理可以直接返回 token ID，由调用方自行解码
- **重要**：任何在线 serving 场景都需要完整的后处理，特别是流式输出和停止条件检测

## 增量解码（Incremental Detokenization）

### 核心问题
Token 到文本的转换不是简单的一对一映射：
1. **子词拼接**：BPE token 可能是词的一部分，需要和相邻 token 合并
2. **UTF-8 多字节**：一个中文字符可能横跨多个 token，部分解码会产生乱码
3. **特殊 token**：`<s>`, `</s>`, `<pad>` 等不应出现在输出中

### 解决方案

```python
class IncrementalDetokenizer:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.prev_tokens = []      # 已解码的 token 历史
        self.prev_text = ""        # 已输出的文本
        self.output_offset = 0     # 已发送给客户端的文本偏移

    def add_token(self, token_id) -> str:
        """添加一个新 token，返回可安全输出的增量文本"""
        self.prev_tokens.append(token_id)

        # 解码所有 token 得到完整文本
        full_text = self.tokenizer.decode(
            self.prev_tokens,
            skip_special_tokens=True
        )

        # 计算新增文本
        new_text = full_text[len(self.prev_text):]
        self.prev_text = full_text

        # 安全检查：不输出可能不完整的尾部
        safe_text = self._find_safe_boundary(new_text)
        return safe_text
```

### 安全边界检测（避免截断 UTF-8）

```python
def _find_safe_boundary(self, text):
    """找到可以安全输出的文本边界"""
    if not text:
        return ""

    # 如果以换行结尾，一定安全
    if text.endswith("\n"):
        return text

    # 如果末尾是 replacement character (�)，说明 UTF-8 不完整
    if text.endswith("\ufffd"):
        return text[:-1] if len(text) > 1 else ""

    # 如果末尾不是空格且不是标点，可能是部分词
    # 保守策略：持有最后一个 "词" 直到确认完整
    last_space = text.rfind(" ")
    if last_space > 0:
        return text[:last_space + 1]

    return text  # 单个 token 的情况，直接输出
```

### vllm 的高性能方案
```python
# 使用 HuggingFace tokenizers 库的 DecodeStream（Rust 实现）
from tokenizers import DecodeStream

class FastDetokenizer:
    def __init__(self, tokenizer):
        self.stream = tokenizer.decode_stream(skip_special_tokens=True)

    def add_token(self, token_id) -> str:
        try:
            return self.stream.step(token_id) or ""
        except Exception:
            # Fallback: reset stream on error
            self.stream = self.tokenizer.decode_stream(skip_special_tokens=True)
            return ""
```

## 停止条件检测

### 停止类型
```python
class FinishReason(Enum):
    NONE = "none"         # 未完成
    STOP = "stop"         # 遇到 stop_token 或 stop_string
    LENGTH = "length"     # 达到 max_tokens
    EOS = "eos"           # 遇到 EOS token
    ABORT = "abort"       # 被系统终止
```

### 检测逻辑
```python
def check_finished(req, new_token_id, decoded_text):
    # 1. EOS token 检测
    if new_token_id in req.eos_token_ids:
        return FinishReason.EOS

    # 2. 最大长度检测
    if req.output_len >= req.max_tokens:
        return FinishReason.LENGTH

    # 3. Stop string 检测
    for stop_str in req.stop_strings:
        if stop_str in decoded_text:
            return FinishReason.STOP

    return FinishReason.NONE
```

### 部分 Stop String 匹配（关键！）

流式输出时，必须处理 stop string 可能被分割在两次输出之间的情况：

```python
class StopStringChecker:
    def __init__(self, stop_strings):
        self.stop_strings = stop_strings
        # 最大可能的部分匹配长度
        self.buffer_length = max(len(s) for s in stop_strings) - 1

    def get_safe_output(self, new_text, full_text):
        """返回可安全输出的文本（扣留可能的部分匹配）"""
        # 扣留尾部 buffer_length 个字符
        safe_end = len(full_text) - self.buffer_length
        if safe_end <= self.last_output_end:
            return ""  # 没有新的安全文本

        safe_text = full_text[self.last_output_end:safe_end]
        self.last_output_end = safe_end
        return safe_text

    def check_full_match(self, full_text):
        """检查是否有完整的 stop string 匹配"""
        for stop_str in self.stop_strings:
            idx = full_text.find(stop_str)
            if idx >= 0:
                # 截断到 stop string 开始位置
                return full_text[:idx], FinishReason.STOP
        return full_text, FinishReason.NONE
```

## 流式输出

### SSE (Server-Sent Events) 协议
```python
async def stream_response(request, engine):
    """生成 SSE 格式的流式响应"""
    yield "data: " + json.dumps({
        "id": request.id,
        "choices": [{"delta": {"role": "assistant"}, "index": 0}],
    }) + "\n\n"

    async for token_text in engine.generate_stream(request):
        yield "data: " + json.dumps({
            "id": request.id,
            "choices": [{"delta": {"content": token_text}, "index": 0}],
        }) + "\n\n"

    yield "data: [DONE]\n\n"
```

### 与推理循环的集成
```python
# 在 scheduler 的 postprocess 中
def postprocess(self, batch, tokens):
    for req, token in zip(batch.reqs, tokens):
        req.output_tokens.append(token)

        # 增量解码
        new_text = req.detokenizer.add_token(token)

        # 检查停止条件
        finish = check_finished(req, token, req.full_text)

        if req.stream:
            # 流式：检查部分 stop string，发送安全文本
            safe_text = req.stop_checker.get_safe_output(new_text, req.full_text)
            if safe_text:
                req.output_queue.put(safe_text)

        if finish != FinishReason.NONE:
            req.finish_reason = finish
            req.status = FINISHED
            if req.stream:
                req.output_queue.put(None)  # 结束信号
```

## LogProbs 收集

```python
class LogProbsCollector:
    def collect(self, logits, sampled_token, top_k=5):
        """收集采样的 log probability 和 top-k 候选"""
        log_probs = F.log_softmax(logits, dim=-1)

        # 采样 token 的 logprob
        token_logprob = log_probs[sampled_token].item()

        # Top-k 候选
        top_values, top_indices = log_probs.topk(top_k)
        top_logprobs = [
            {"token": tokenizer.decode([idx]), "logprob": val.item()}
            for idx, val in zip(top_indices, top_values)
        ]

        return {
            "token": tokenizer.decode([sampled_token]),
            "logprob": token_logprob,
            "top_logprobs": top_logprobs,
        }
```

## 集成到生成代码的方式

### 最小集成（3 层）

```
Scheduler postprocess → 增量解码 + 停止检测
         ↓
Detokenizer (独立进程或线程) → 文本安全边界处理
         ↓
API Server → SSE/JSON 格式化输出
```

### 进程分离模式
对于高吞吐场景，detokenization 应放在独立进程：
```python
# 主进程（Scheduler）
def postprocess(batch, tokens):
    # 只发送 token ID，不做字符串处理
    send_to_detokenizer([(req.id, token) for req, token in zip(batch, tokens)])

# 独立进程（Detokenizer）
def detokenize_loop():
    while True:
        batch = recv_from_scheduler()
        for req_id, token in batch:
            text = detokenizers[req_id].add_token(token)
            if text:
                send_to_frontend(req_id, text)
```

## 源码参考

| 项目 | 关键文件 |
|------|---------|
| sglang | `srt/managers/detokenizer_manager.py` |
| vllm | `v1/engine/detokenizer.py` |
| vllm | `v1/engine/output_processor.py` |
| mini-sglang | `tokenizer/detokenize.py` |
| sglang | `srt/entrypoints/openai/serving_chat.py` |
