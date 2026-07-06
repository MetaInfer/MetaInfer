# 端到端输出乱码 & 4 卡 TP 错误 — 根因分析与修复记录

> 日期: 2026-06-07
> 模型: Qwen3-8B (4096 hidden, 36 layers, 32 heads, 8 KV heads, 12288 intermediate)
> 环境: A800-SXM4-80GB × 4, PyTorch 2.x, CUDA 12.x

---

## Part I: 端到端输出乱码（单卡可复现）

单卡 TP=1 推理时，模型输出为乱码（如 `"而非，木地板木地板木地板無しさん"`），与 HF baseline 输出 `"（ ） A：建筑与园林结合 B：建筑与自然结合 C：建筑与山水结合 D：建筑"` 完全不匹配。

### 根因 1: RMSNorm 计算顺序与 HF 参考实现不一致

**影响组件**: `engine/models/qwen.py` — `RMSNorm` 类（Q 头/K 头的 QKNorm）

**问题描述**:

引擎的 RMSNorm 使用 `(self.weight.float() * x_f).to(dtype=input_dtype)` — 在 fp32 下乘 weight 再转 bf16。HuggingFace `Qwen3RMSNorm` 使用 `self.weight * x_f.to(input_dtype)` — 先转 bf16 再乘 weight。

两种顺序在随机张量上差异为 0（单测无法发现），但在真实 Q/K 投影输出上产生系统性偏差：

| 检查点 | 差异值 | 说明 |
|--------|--------|------|
| Q norm 输出 | **0.0625** | 恰好 = 2⁻⁴，可疑的二进制整数 |
| K norm 输出 | **1.0** | 恰好 = 2⁰ |

**根本原因**: 模型权重是在 HF 的精度路径下训练的——权重参数已针对 "先 cast bf16 再乘 weight" 这种特定的舍入点做了优化。使用"更精确"的 fp32 乘法路径反而产生更差的结果，因为它偏离了训练精度路径。

**HF Qwen3RMSNorm 的正确计算顺序**:
```
input → float32 → variance → rsqrt → normalize → .to(input_dtype) → * weight(bf16)
```

**错误路径**:
```
input → float32 → variance → rsqrt → normalize → * weight(float) → .to(input_dtype)
```

**修复** (文件: `engine/models/qwen.py`, 第 119 行):

```python
# 修复前:
out = (self.weight.float() * x_f).to(dtype=input_dtype)

# 修复后:
out = self.weight * x_f.to(dtype=input_dtype)
```

### 根因 2: 最终 LayerNorm 缺少 residual 累积

**影响组件**: `engine/models/qwen.py` — `QwenForCausalLMTP.forward_prefill()` 和 `forward_decode()`

**问题描述**:

RMSNorm 修复后，单卡输出仍然完全错误（top-1 token 为 `!` 且 logit=0.0），尽管所有单组件测试均通过（embedding diff=0, input layernorm diff=0, QKV diff=0, Q/K norm diff=0, RoPE diff=0）。

根因在于 residual chain 的语义差异：

| 实现 | DecoderLayer 返回值 | 最终 norm 输入 |
|------|---------------------|----------------|
| HF `Qwen3DecoderLayer` | `hidden_states = residual + mlp_output`（residual 已 bake in） | 完整 hidden state |
| 引擎 `QwenDecoderLayerTP` | `(mlp_out, res)` — mlp_out 不含 residual，res 单独累积 | **仅有最后层 mlp_out** ← BUG |

引擎代码在 36 层循环后对 `hs`（最后一层的 mlp_out）直接做 `self.norm(hs)`，而未加上累积在 `res` 中的 36 层 residual。这导致最终 norm 的输入只包含最后一层的 MLP 输出，缺失了全部 residual 信息。

**修复** (文件: `engine/models/qwen.py`, `forward_prefill` 和 `forward_decode`):

```python
# 修复前:
for layer in self.layers:
    hs, res = layer(hs, positions, residual=res)
hs, _ = self.norm(hs)          # ← hs = 最后层 mlp_out，缺少 residual

# 修复后:
for layer in self.layers:
    hs, res = layer(hs, positions, residual=res)
assert res is not None
hs = hs + res                   # 重建完整 hidden state
hs, _ = self.norm(hs)           # 对完整 hidden state 做 norm
```

两处修复（`forward_prefill` 和 `forward_decode`）必须同时应用。

### 单卡验证结果

```
OUTPUT:   '（ ） A：建筑与园林结合 B：建筑与自然结合 C：建筑与山水结合 D：建筑'
EXPECTED: '（ ） A：建筑与园林结合 B：建筑与自然结合 C：建筑与山水结合 D：建筑'
MATCH:    True
```

---

## Part II: 4 卡 TP 错误（TP=4 特有）

单卡修复后，TP=4 (tensor parallelism) 推理失败，涉及两个独立问题。

### 根因 3: o_proj 权重复载的 RowParallel 分片尺寸错误

**影响组件**: `engine/models/qwen.py` — `QwenForCausalLMTP.load_weights()` 第 1264 行

**问题描述**:

`_row_slice` 用于 RowParallelLinear（如 `o_proj`）时，size 参数应传 `in_features // tp_size`，但实际传了完整的 `in_features`。

```python
# 修复前:
layer_o[layer_idx] = _row_slice(weight, hidden_size)      # hidden_size = 4096

# 修复后:
layer_o[layer_idx] = _row_slice(weight, hidden_size // tp_size)  # = 1024
```

`_row_slice` 在 dim=1 上按 `[tp_rank * size : (tp_rank+1) * size]` 切片。当 size=4096（全尺寸）时：

| Rank | 切片范围 | 结果形状 | 预期形状 | 状态 |
|------|---------|---------|---------|------|
| 0 | `[:, 0:4096]` | [4096, 4096] | [4096, 1024] | 形状不匹配 → `RuntimeError` |
| 1 | `[:, 4096:8192]` | [4096, 0] | [4096, 1024] | 空张量 |
| 2 | `[:, 8192:12288]` | [4096, 0] | [4096, 1024] | 空张量 |
| 3 | `[:, 12288:16384]` | [4096, 0] | [4096, 1024] | 空张量 |

错误信息:
```
RuntimeError: Layer 0 o_proj shape mismatch: loaded=[4096, 0], model=[4096, 1024]
```

**教训**: `_col_slice` 的调用者通常自然地使用 per-rank 尺寸（`q_size`, `kv_size`, `inter_per_rank` 都已经是 `÷ tp_size` 的），但 `_row_slice` 的调用者容易误传全尺寸。写 TP 权重复载代码时必须逐行验证 `_row_slice` 的 size 参数已除以 `tp_size`。

### 根因 4: CustomAR rank_data 工作区尺寸不足 + GC 释放

**影响组件**: `engine/tp_layers/distributed.py` — `init_custom_ar()` 和 `CustomARHandle`

**问题描述**:

o_proj 分片修复后，TP=4 推理在模型 forward 时崩溃，报错:

```
RuntimeError: CUDA error: CUBLAS_STATUS_EXECUTION_FAILED when calling 'cublasGemmEx'
```

**关键诊断信号**: 崩溃位置在多次运行间**不稳定**:
- 第一次运行: 崩溃在 `QwenMLPTP.forward` → `self.down_proj(out)` → `RowParallelLinear.forward` (gate_up_proj 后)
- 第二次运行: 崩溃在 `QwenMLPTP.forward` → `self.gate_up_proj(x)` → `MergedColumnParallelLinear.forward` (gate_up_proj 本身)

崩溃位置会变化排除了形状不匹配的可能。对比 `inference-agent-system/engine/tp_layers/distributed.py` 发现两个独立差异:

**差异 1: `rank_data` 大小 (硬件缓冲区越界)**

| 代码库 | `max_size` | `rank_data` 大小 |
|--------|-----------|------------------|
| inference-agent-system | 16 MB | `torch.empty(max_size, ...)` = 16 MB |
| Konwldge (修复前) | 16 MB | `torch.empty(8*1024*1024, ...)` = 8 MB |

`rank_data` 是 `ops.init_custom_ar()` 的内部工作区。CustomAR 内核期望该区域至少 `max_size` 字节，但只有 8 MB——写入溢出会污染相邻 CUDA 内存。

**差异 2: `rank_data` 生命周期 (Python GC use-after-free)**

| 代码库 | `CustomARHandle` 是否持有 `rank_data` 引用 |
|--------|------------------------------------------|
| inference-agent-system | ✅ `self.rank_data = rank_data` |
| Konwldge (修复前) | ❌ 不持有 — `rank_data` 在 `init_custom_ar()` 返回后被 GC |

`ops.init_custom_ar()` 内部保存了 `rank_data` 的裸指针。当 Python tensor 被 GC 时，其 CUDA 内存被释放，但内核仍通过裸指针访问 → use-after-free，写入已释放内存（可能已被模型权重重新分配）。

**两个差异必须同时修复**:

```python
# init_custom_ar — 修复 rank_data 大小:
rank_data = torch.empty(
    max_size, dtype=torch.uint8, device=f"cuda:{local_rank}"  # 16 MB
)

# CustomARHandle.__init__ — 添加 rank_data 引用:
self._rank_data = rank_data  # 防止 GC，内核持有此内存的裸指针

# CustomARHandle 构造:
_custom_ar_handle = CustomARHandle(
    ptr=ptr,
    buf_ptrs=buf_ptrs,
    buf_sz_bytes=max_size,
    rank_data=rank_data,  # 传入并被持有
)
```

### TP=4 验证结果

```
TP=4 OUTPUT:   '（ ） A：建筑与园林结合 B：建筑与自然结合 C：建筑与山水结合 D：建筑'
TP=4 EXPECTED: '（ ） A：建筑与园林结合 B：建筑与自然结合 C：建筑与山水结合 D：建筑'
TP=4 MATCH:    True
TP=4 GREEDY ALIGN: PASS
```

---

## 总结: 四类根因一览

| # | 根因 | 分类 | 影响 | 修复文件 | 关键教训 |
|---|------|------|------|----------|----------|
| 1 | RMSNorm 计算顺序与参考实现不一致 | 精度路径差异 | 单卡/多卡乱码 | `engine/models/qwen.py` RMSNorm | 随机张量测试无法发现精度路径差异——必须用真实模型数据验证 |
| 2 | 最终 norm 缺失 residual 累积 | 架构语义差异 | 单卡/多卡乱码 | `engine/models/qwen.py` forward_prefill/decode | vLLM-style residual chain 与 HF-style 返回值语义不同——不能直接对比 `hs` 变量 |
| 3 | `_row_slice` 未除以 `tp_size` | TP 分片逻辑 | 仅 TP>1 | `engine/models/qwen.py` load_weights | `_row_slice` 的 size 参数容易漏除 `tp_size`，`_col_slice` 则天然正确 |
| 4 | CustomAR rank_data 尺寸不足 + GC 释放 | GPU 内存安全 (use-after-free + overflow) | 仅 TP>1 | `engine/tp_layers/distributed.py` init_custom_ar + CustomARHandle | 对比参考实现的每个细节——不仅看做了什么，还要看**没做什么**（不持有引用） |

## 涉及文件

| 文件 | 修改 |
|------|------|
| `engine/models/qwen.py` | 根因 1: RMSNorm.forward() 计算顺序；根因 2: forward_prefill/decode 加 `hs = hs + res`；根因 3: `_row_slice(weight, hidden_size // tp_size)` |
| `engine/tp_layers/distributed.py` | 根因 4: CustomAR rank_data 尺寸 8MB→16MB + CustomARHandle 持有 rank_data 引用防 GC |
| `scripts/test_phase10_greedy_align.sh` | 改进错误处理：Python 路径检测、退出码检查、stderr 可见、GPU 数量验证 |
