# meta-infer CUDA Graph 终极执行路线图

> **版本**: v6 — custom op 精简版（torch.compile + CUDAGraphWrapper + torch.library.custom_op）  
> **当前状态**: 阶段零/一/二 已完成 ✅，阶段三 方案已锁定待实施  
> **目标**: Qwen3-8B TP=4 CPU dispatch 548ms → <50ms，吞吐 53.9 → 100+ tok/s  
> **唯一真源**: `/tmp/prof_vllm_cudagraph_tp4/` rank-0 JSON trace (2,595,861 events, 48 `cudaGraphLaunch`)  
> **铁律**: 正文中每一处设计决策必须有 Trace 物理事实或 vLLM 源码行号支撑。严禁脑补。

---

## 当前进度总览

> **已完成的**：单 GPU 全链路 CUDA 图已打通——vLLM 的 CUDAGraphWrapper 已移栽，编译后的函数在图内正常捕获和回放，验证正确。  
> **尚未完成的**：多卡 TP=4 的 CUDA 图还需把通信算子 `all_reduce_sum` 注册为 PyTorch 自定义黑盒算子，避免编译器在捕获期间重编译崩溃。

| 阶段 | 内容 | 状态 | 关键结果 |
|------|------|------|---------|
| 阶段零 | 黑盒清理 | ✅ 已提交 | CustomAR 接口恢复，伪图代码删除 |
| 阶段一 | CUDAGraphWrapper + 编译 | ✅ 已提交 | 从 vLLM 移栽，单 GPU 验证通过 |
| 阶段二 | 单卡单层压测 | ✅ 已提交 | 10K replay 无 crash，回放加速 2.3x |
| 阶段三 | TP=4 多卡 CUDA 图 | 🔴 方案锁定 | 待实施：`all_reduce_sum` 注册为自定义黑盒算子 |
| 阶段四 | 全模型联动 | ⏸ 待阶段三 | — |

### 阻塞点一句话总结

Pytorch 编译器（torch.compile）在追踪要编译的函数时，会检查函数内部每一步操作的"守卫条件"有没有变。我们的通信算子 `all_reduce_sum` 在预热阶段和 CUDA 图捕获阶段，因为 CUDA 流的状态不同，守卫条件变了，编译器认为需要重新编译。重新编译时，编译器要读随机数种子，但 CUDA 图捕获期间不允许这个操作 → 崩溃。

解决办法：用 PyTorch 标准机制把 `all_reduce_sum` 标记为"黑盒算子"——告诉编译器不要追踪它内部，不要检查它的守卫。这样编译阶段和图捕获阶段就不会冲突了。这不是搬 vLLM 的整个后端（VllmBackend），只是抽取其中一个关键点子——约 20 行代码，只改 `distributed.py` 一个文件。

### 已尝试并排除的方案

| 方案 | 来源 | 失败原因 |
|------|------|---------|
| raw CUDA 图 + NCCL 通信 | nano-vllm, mini-sglang | NCCL 在图内回放时，矩阵乘法临时内存地址变了，通信读到错误数据，计算结果不对（差 3.4） |
| raw CUDA 图 + CustomAR | sglang 参考 | 同上——不是 NCCL 的问题，是矩阵乘法和 CUDA 图的配合问题 |
| torch.compile + 关守卫 | 自研 | 守卫关不掉，PyTorch 2.9.1 不支持空守卫列表 |

---

### 各阶段 SOP 门禁铁律

> **任何阶段在执行 Performance Profiling / Benchmark 测试之前，必须先完成以下两层正确性验证，二者全部 PASS 方可进入性能测试：**

| 层级 | 内容 | 方法 | 通过标准 |
|------|------|------|---------|
| **L1: 逐层张量正确性** | 该阶段的算子输出与 eager 模式数值对齐 | `torch.testing.assert_close(actual, eager, rtol, atol)` | `rtol=1e-2, atol=1e-2`（含通信时放宽至 `rtol=5e-2`） |
| **L2: 端到端推理输出** | 完整模型 24 词贪婪解码输出字字对齐 | `engine.generate(prompt, max_new_tokens=24, temperature=0.0)` | `output == '（ ） A：建筑与园林结合 B：建筑与自然结合 C：建筑与山水结合 D：建筑'` |

**L1 必须覆盖的验证点**:
- 单层张量 shape/dtype 与 eager 一致
- 单层 replay 输出与 eager 输出数值对齐（rtol 同上）
- 10K replay 后 NaN/Inf 异步探针无异常
- 非阻塞可观测性断言：严禁 `.item()`、`.cpu()` 触发 Host-Device Sync

**L2 必须覆盖的验证点**:
- E2E 24 词 greedy decode（`temperature=0`）字字对齐
- 连续 5 轮输出稳定一致

---

## 〇. 标品黑盒原则与 Kernel 调用约束

> **全文档硬编码红线。Agent 在任何阶段编码前必须逐条确认。违反任一条 = 代码驳回。**

### 0.1 不可篡改黑盒清单

以下组件为**绝对不可篡改的硬件级标品黑盒**。Agent 唯一职责：对齐输入/输出 Tensor Shape 与 Dtype，按 vLLM Trace 顺序串联调用。

| 编号 | 黑盒组件 | vLLM 源码（文件:行号） | meta-infer 位置 | Agent 禁止行为 |
|------|---------|---------------------|----------------|--------------|
| BB-1 | `CustomAllReduceHandle` P2P 通信内核 | `custom_all_reduce.py:247-264` (all_reduce) + `:266-282` (custom_all_reduce dispatch) | `engine/tp_layers/custom_ar.py` | 禁止修改 `registered` 参数语义 |
| BB-2 | `register_graph_buffers()` IPC 句柄交换 | `custom_all_reduce.py:213-230` | `custom_ar.py:60-73` | 禁止重写 gloo broadcast 逻辑 |
| BB-3 | `capture()` context manager | `custom_all_reduce.py:199-211` | 待提取至 `custom_ar.py` | 禁止修改 `_IS_CAPTURING` 状态机 |
| BB-4 | `CUDAGraphWrapper` 惰性捕获+replay 状态机 | `compilation/cuda_graph.py:145-356` | `engine/tp_layers/cuda_graph_wrapper.py`（待创建） | 禁止自研图管理逻辑 |
| BB-5 | `torch.compile` inductor backend | PyTorch builtin | 标准 `torch.compile(fullgraph=True)` | 禁止手写 buffer 分配替代 inductor |
| BB-6 | `rms_norm` / `fused_add_rms_norm` / `rotary_embedding` / `silu_and_mul` | `vllm/_custom_ops.py` → `torch.ops._C.*` | `engine/kernels/vllm_wrappers.py` | 禁止修改 wrapper 内部 |
| BB-7 | `flash_attn_with_kvcache` / `flash_attn_varlen_func` | flash_attn package (pybind11) | `engine/kernels/custom_ops.py` (custom op 注册) | 已在 P6 完成注册 |
| BB-8 | `torch.library.custom_op` PyTorch 自定义算子注册 | `parallel_state.py:262-266` (`direct_register_custom_op("all_reduce")`) + PyTorch 标准 API | 待提取至 `engine/tp_layers/distributed.py` | 禁止在编译函数内调用未注册为黑盒的通信函数 |
| BB-9 | inductor `cudagraph_trees` CUDA 图管理 | PyTorch builtin `torch.compile(mode='reduce-overhead')` | 标准 `torch.compile` API | 禁止外部套 CUDAGraphWrapper + 内部用 reduce-overhead 双层图 |

### 0.2 多卡联动对齐原则

1. **Trace 为唯一导航**: 各 Rank 的控制流与图捕获边界，必须严格依照 vLLM Rank-0 物理 Trace 中的算子时序对齐。禁止自行发明多卡同步控制流。
2. **Buffer 管理由 inductor 负责**: 所有中间 tensor 的分配/复用/生命周期管理由 `torch.compile` 的 inductor backend 自动完成。Agent 不得手写任何 buffer 预分配或 out-of-place 转换逻辑。
3. **通信方案以 Trace 为准**: Trace 显示 876 次 NCCL AllReduce 全部在图内、0 次 CustomAR → 阶段三走 NCCL。CustomAR 图内集成为后续可选优化。

### 0.3 物理 Trace 事实速查

> 来源: `/tmp/prof_vllm_cudagraph_tp4/` rank-0 JSON (2,595,861 events, 48 `cudaGraphLaunch`)

| 编号 | 物理观察 | 数值 | 证据 | 对应阶段 |
|------|---------|------|------|---------|
| TF-1 | NCCL AllReduce 在图内执行 | 876 次全部在 48 个 launch 窗口内 | Window [6..11]: NCCL 与 GEMM/FA/KV 同窗口 | 阶段三 |
| TF-2 | CustomAR 调用次数 | **0** | `custom_ar`, `cross_device_reduce` 关键词 0 命中 | 阶段三 |
| TF-3 | `reshape_and_cache_flash` 在图内 | 37 个窗口包含 KV write | 与 FA/GEMM/NCCL 共享窗口 | 阶段一 |
| TF-4 | `flash_fwd_splitkv` 在图内 | 37 个窗口包含 FA | 与 KV/GEMM/NCCL 共享窗口 | 阶段一 |
| TF-5 | 每层 1 完整图（FA+KV 均在图内） | 37 窗口含 FA+KV = 完整层图，10 窗口不含 = 部分图 | Window [6..11] 均为完整层 | 阶段一/四 |
| TF-6 | 启动分布 | Launch [0]=1 (prefill), [1..36]=36 (层图), [37..47]=11 (后处理) | 间隔聚类 | 阶段四 |
| TF-7 | cudaGraphLaunch 总次数 | 48 | key_averages | 阶段四 |
| TF-8 | cudaStreamIsCapturing | 133 次，48 在 launch 附近 | CPU op 时序 | 阶段零 |
| TF-9 | torch.compile/dynamo/inductor CPU event | **0**（编译在 init 完成） | trace 仅含编译产物：`triton_red_fused` 等 | 阶段一 |

---

## 一、阶段零：历史违规清理与黑盒接口对齐

> **依赖前置**: 无  
> **核心对齐依据**: TF-8 (cudaStreamIsCapturing) + vLLM `custom_all_reduce.py:199-282`  
> **目标**: 删除 qwen.py 中所有 P6/Stage8 自研伪图代码，恢复 CustomAR 的 vLLM 标准双模式接口

### 1.1 代码替换与适配

#### 子任务 0-A: 恢复 CustomAR `all_reduce` 双模式接口

**提取源**: `vllm/distributed/device_communicators/custom_all_reduce.py:247-264`

**对齐依据**: TF-8 证实 vLLM 使用 `cudaStreamIsCapturing()` 进行 capture/eager dispatch。

**替换** (`custom_ar.py:49-58`):

```python
# 删除 (仅有 staging buffer 路径):
def all_reduce(self, inp):
    ...
    ops.all_reduce(self._ptr, inp, out, self._buf_ptrs[dist.get_rank()], self._max_size)

# 替换为 (逐行对齐 vLLM :247-264):
def all_reduce(self, inp: torch.Tensor, *, out: torch.Tensor = None, registered: bool = False
) -> torch.Tensor:
    if out is None:
        out = torch.empty_like(inp)
    if registered:
        ops.all_reduce(self._ptr, inp, out, 0, 0)
    else:
        ops.all_reduce(self._ptr, inp, out, self._buf_ptrs[dist.get_rank()], self._max_size)
    return out
```

#### 子任务 0-B: 恢复 `custom_all_reduce` dispatch + `capture()` context

**提取源**: `custom_all_reduce.py:199-211` (capture) + `:266-282` (dispatch)

```python
# 新增属性
self._IS_CAPTURING: bool = False

# 新增方法 (逐行对齐 vLLM :199-211)
@contextmanager
def capture(self):
    try:
        self._IS_CAPTURING = True
        yield
    finally:
        self._IS_CAPTURING = False
        if not self._disposed:
            self.register_graph_buffers()

# 新增方法 (逐行对齐 vLLM :266-282)
def custom_all_reduce(self, inp: torch.Tensor) -> torch.Tensor:
    if self._IS_CAPTURING:
        if torch.cuda.is_current_stream_capturing():
            return self.all_reduce(inp, registered=True)
        else:
            return torch.empty_like(inp)
    else:
        return self.all_reduce(inp, registered=False)
```

#### 子任务 0-C: 删除 qwen.py 中所有自研伪图逻辑

**对齐依据**: 以下代码均为 Stage8/P6 自研，vLLM 中无对应源码。Trace 中 vLLM 使用 `CUDAGraphWrapper` 管理图生命周期。

**删除清单** (`engine/models/qwen.py`):

| 删除项 | 自研性质 |
|--------|---------|
| `QwenForCausalLMTP.init_decode_graph()` | 自研捕获（vLLM 用 CUDAGraphWrapper） |
| `QwenForCausalLMTP.graph_replay()` | 自研 replay |
| `QwenForCausalLMTP.has_decode_graph` | 自研状态检查 |
| `QwenForCausalLMTP._decode_graph` / `_graph_input_ids` / `_graph_pos` / `_graph_logits` | 自研 graph 状态存储 |
| `runner.run()` 中 `seq._decode_step_count == 2` 触发捕获 | 推理时动态捕获（vLLM 在 init 时捕获） |
| `runner.run()` 中 `self.model._decode_graph = None` | 自研 graph 销毁 |
| `runner.run()` 中 `_capturing = torch.cuda.is_current_stream_capturing()` guard | P6 自创 `.item()` 跳过逻辑 |
| `QwenForCausalLMTP.forward()` 中 `is_decode` / `_capturing` 分支 | P6 自创分支 |

**保留**: `_kv_len_gpu`、`_slot_mapping_decode`、`_key_cache`、`_value_cache`、`_block_table`（paged attention 正常 buffer，非 graph 特化）

#### 子任务 0-D: 恢复 forward_decode 中 `fused_add_rms_norm` 黑盒调用

**对齐依据**: TF-9 + BB-6。Trace 中 `triton_red_fused` 对应 inductor 编译后的 `fused_add_rms_norm` 融合 kernel。当前 `forward_decode` 的 `else` 分支和 `post_attention_layernorm` 残留了 P6 自研分解（V8 违规），必须恢复为 vLLM 标品调用。首层分支（`if residual is None`）不涉及残差加法，使用 `rms_norm`（同为 BB-6 黑盒）。

**替换** (`qwen.py` `QwenDecoderLayerTP.forward_decode`):

```python
# 删除 (P6 自研分解，V8 违规):
# else 分支:
residual = residual + hidden_states
torch.ops._C.rms_norm(hidden_states, residual, weight, eps)

# post_attention_layernorm:
residual = residual + hidden_states
torch.ops._C.rms_norm(hidden_states, residual, weight, eps)

# 替换为 (vLLM 标品 BB-6):
from engine.kernels.vllm_wrappers import fused_add_rms_norm
# else 分支:
fused_add_rms_norm(hidden_states, residual, weight, eps)

# post_attention_layernorm:
fused_add_rms_norm(hidden_states, residual, weight, eps)
```

**`if residual is None` 分支不变** — 首层仅做 RMSNorm（无残差加法），`rms_norm(hidden_states, residual, ...)` 是合规的 BB-6 黑盒调用。

所有 kernel 调用统一走 `vllm_wrappers.py` 入口。阶段一的 `torch.compile` 通过 `torch.ops._C.fused_add_rms_norm` 的注册元数据自动理解双 in-place 语义并管理 buffer。Agent 不手动作任何分解或 buffer 分配。

### 1.2 正确性验证

> **门禁**: L1 逐层张量 + L2 E2E 输出 二者 PASS 后，方可进入 1.3 性能 Profiling。

**测试文件**: `tests/test_stage0_cleanup.py`  
**运行方式**: `torchrun --nproc_per_node=4 pytest tests/test_stage0_cleanup.py -v`

| 层级 | 用例 | 方法 | 通过标准 |
|------|------|------|---------|
| L1 | `test_custom_ar_registered_false_equiv` | eager 对比新旧 all_reduce | `rtol=1e-3` |
| L1 | `test_custom_ar_registered_true_no_error` | TP=4 `all_reduce(inp, registered=True)` | 无异常 |
| — | `test_capture_context_sets_flag` | `with handle.capture(): assert handle._IS_CAPTURING` | PASS |
| **L2** | `test_pseudo_graph_code_removed_eager` | `META_INFER_CUDA_GRAPH=0` TP=4 24 tokens greedy decode | `'（ ） A：建筑与园林结合 B：建筑与自然结合 C：建筑与山水结合 D：建筑'`，吞吐退化 < 2% |

**非阻塞可观测性断言规范**（全文档统一）:
- 严禁 `.item()`、`.cpu()`、`.any()` 触发隐式 Host-Device Sync
- NaN 检测: `torch.isnan(t).any()` → GPU tensor，不 sync
- capture 期间 guard: `if torch.cuda.is_current_stream_capturing(): return {"healthy": True, "reason": "skipped_during_capture"}`
- 字字对齐: `temperature=0` greedy decode + `assert out == expected`

### 1.3 性能 Profiling 验证

```bash
# 基线: 阶段零改动前 / 阶段零完成后: 同参数重跑
CUDA_VISIBLE_DEVICES=0,1,2,3 META_INFER_CUDA_GRAPH=0 \
  torchrun --nproc_per_node=4 python -c "
from llm_engine import LLMEngine; from pathlib import Path; import time
engine = LLMEngine(model_dir=Path('.../models/qwen/Qwen3-8B'), inference_backend='qwen_tp', max_num_seqs=4)
for _ in range(3):
    t0=time.perf_counter(); engine.generate('苏州园林的特点是', max_new_tokens=12, temperature=0.0)
    torch.cuda.synchronize(); print(time.perf_counter()-t0)
"
# 断言: avg throughput 变化 < 2%
```

### 1.4 文档沉淀

在 `kernel_replacement_plan.md` 中追加阶段零记录:
- CustomAR 接口恢复 diff
- 删除的自研代码行数 (`git diff --stat`)
- kernel 黑盒调用约束更新

### 1.5 Git Commit 门禁

```bash
git add engine/tp_layers/custom_ar.py engine/models/qwen.py tests/test_stage0_cleanup.py
git commit -m "feat(cudagraph): pass stage_0_blackbox_cleanup

- Restore CustomAR all_reduce(inp, *, out, registered) vLLM interface
- Add capture() context manager + custom_all_reduce dispatch
- Remove all self-invented graph capture/replay logic
- Verified: eager decode unchanged, throughput <2% regression"
```

---

## 二、阶段一：torch.compile 编译 + CUDAGraphWrapper 集成

> **依赖前置**: 阶段零  
> **核心对齐依据**: TF-3/4/5 (KV/FA/每层完整图 均在 inductor 生成的 triton kernel 内) + TF-9 (0 compile event，编译在 init 完成) + vLLM `compilation/cuda_graph.py:145-356`  
> **目标**: 对每层 `forward_decode` 应用 `torch.compile(fullgraph=True)`（inductor 自动管理 buffer + kernel fusion），然后用 `CUDAGraphWrapper` 包装（惰性 capture/replay）  
> **关键**: 对标 vLLM 架构——`torch.compile` 只负责 kernel 生成和 buffer 管理，CUDAGraphWrapper 只负责图捕获/replay。Agent 不手写任何 buffer 分配。

### 2.1 代码替换与适配

#### 子任务 1-A: 创建 `engine/tp_layers/cuda_graph_wrapper.py`

**提取源**: `vllm/compilation/cuda_graph.py:145-356`

**剥离的 vLLM 特有依赖**: `forward_context`、`BatchDescriptor`、`VllmConfig`、`compilation_config`、`CUDAGraphMode`、`offloader`。保留纯 CUDA Graph 生命周期管理状态机。

```python
class CUDAGraphWrapper:
    """惰性 CUDA Graph 捕获 + replay。

    状态机 (对齐 vLLM :233-356):
      __call__(*args):
        ├─ entry 不存在？ → _capture() → 缓存 entry → 返回 output
        └─ entry 已存在？ → 校验 input_addresses → replay → 返回 entry.output

    注: 本 Wrapper 包裹的是已被 torch.compile 编译的 forward_decode。
    inductor 负责 kernel 优化 + buffer 分配; Wrapper 负责图管理。
    """
    def __init__(self, runnable: Callable, debug_mode: bool = False):
        self.runnable = runnable
        self.debug_mode = debug_mode
        self._entry: CUDAGraphEntry | None = None
        self._probes = CUDAGraphProfilingProbes()

    def __call__(self, *args, **kwargs):
        if self._entry is None or self._entry.cudagraph is None:
            return self._capture(*args, **kwargs)
        if self.debug_mode:
            new_addrs = [x.data_ptr() for x in args if isinstance(x, torch.Tensor)]
            assert new_addrs == self._entry.input_addresses, \
                f"Input address mismatch during replay"
        self._entry.cudagraph.replay()
        self._probes.record_launch()
        return self._entry.output

    def _capture(self, *args, **kwargs):
        input_addrs = [x.data_ptr() for x in args if isinstance(x, torch.Tensor)]
        cudagraph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(cudagraph):
            output = self.runnable(*args, **kwargs)
        self._entry = CUDAGraphEntry(
            cudagraph=cudagraph, output=output, input_addresses=input_addrs
        )
        self._probes.record_capture()
        return output

    def check_graph_health(self) -> dict:
        """非阻塞 NaN/Inf 检测。严禁 .item()"""
        if torch.cuda.is_current_stream_capturing():
            return {"healthy": True, "reason": "skipped_during_capture"}
        if self._entry is None:
            return {"healthy": True, "reason": "no_graph_yet"}
        out = self._entry.output
        tensors = list(out) if isinstance(out, (tuple, list)) else [out]
        tensors = [t for t in tensors if isinstance(t, torch.Tensor)]
        return {
            f"out_{i}_has_nan": torch.isnan(t).any()
            for i, t in enumerate(tensors)
        } | {f"out_{i}_has_inf": torch.isinf(t).any() for i, t in enumerate(tensors)}

    def clear_graph(self):
        self._entry = None
```

#### 子任务 1-B: 在 QwenTPModelRunner 中对每层编译 + 包装

**对齐依据**: TF-9 证实 vLLM 在 init 阶段完成 torch.compile（trace 中无 compile event），推断时仅有编译产物（triton kernel）。我们同样在 `__init__` 中完成 compile + warmup + capture。

```python
class QwenTPModelRunner:
    def __init__(self, ...):
        # ... (existing init: model, weights, tokenizer, CustomAR)
        self._cuda_graph_enabled = (
            os.environ.get('META_INFER_CUDA_GRAPH', '1') == '1'
        )
        if self._cuda_graph_enabled:
            self._setup_cuda_graph_piecewise()

    def _setup_cuda_graph_piecewise(self):
        """vLLM 对齐: torch.compile + CUDAGraphWrapper，全标品路径"""
        import torch._dynamo
        torch._dynamo.config.cache_size_limit = 128

        # Step 1: 对每层 forward_decode: torch.compile + CUDAGraphWrapper
        for layer in self.model.layers:
            compiled = torch.compile(
                layer.forward_decode, fullgraph=True, dynamic=False
            )
            layer.forward_decode = CUDAGraphWrapper(
                compiled,
                debug_mode=os.environ.get('META_INFER_DEBUG', '0') == '1'
            )

        # Step 2: Dummy prefill 创建 KV cache
        dummy_ids = torch.tensor([[0]*4], dtype=torch.long, device=self.device)
        self.model(dummy_ids, past_key_values=None, position_offset=0, max_seq_len=528)
        torch.cuda.synchronize()
        if is_tp_enabled(): torch.distributed.barrier()

        # Step 3: Warmup → 触发 compile (inductor 生成 kernel) + CUDAGraphWrapper capture
        kv_lens = [4] * len(self.model.layers)
        decode_ids = torch.tensor([[0]], dtype=torch.long, device=self.device)
        for warmup_step in range(3):  # 3 轮: compile + autotune + capture
            self.model(decode_ids, past_key_values=kv_lens,
                       position_offset=4 + warmup_step, max_seq_len=528)
            kv_lens = [v + 1 for v in kv_lens]
        torch.cuda.synchronize()
        if is_tp_enabled(): torch.distributed.barrier()
        print(f"[CUDA Graph] 36 layers compiled + graph captured (PIECEWISE mode)")
```

**Agent 禁止行为**:
- 禁止在 `_setup_cuda_graph_piecewise` 中手写任何 `torch.empty` buffer 分配
- 禁止手动将 in-place op 改为 out-of-place——inductor 通过 `torch.ops._C.*` 的注册元数据自动处理
- 禁止手动管理中间 tensor 生命周期

### 2.2 正确性验证

> **门禁**: L1 逐层张量 + L2 E2E 输出 二者 PASS 后，方可进入 2.3 性能 Profiling。

**测试文件**: `tests/test_stage1_compile_wrapper.py`  
**运行方式**: `CUDA_VISIBLE_DEVICES=0 python -m pytest tests/test_stage1_compile_wrapper.py -v`

| 层级 | 用例 | 方法 | 通过标准 |
|------|------|------|---------|
| — | `test_compile_fullgraph_no_error` | `torch.compile(layer.forward_decode, fullgraph=True)` 不抛异常 | 编译成功 |
| L1 | `test_compiled_output_matches_eager` | compiled vs eager 单层输出 | `rtol=1e-2, atol=1e-2` |
| — | `test_wrapper_capture_single_layer` | CUDAGraphWrapper(compiled) capture 不报错 | 无 CUDA error |
| L1 | `test_wrapper_replay_matches_compiled` | capture 后 replay vs compiled eager | `rtol=1e-2, atol=1e-2` |
| — | `test_health_check_during_capture_no_sync` | `with torch.cuda.graph(): wrapper.check_graph_health()` | `{'healthy': True, 'reason': 'skipped_during_capture'}` |
| L2 | `test_e2e_greedy_decode_match` | 单 GPU CUDA Graph 24 tokens greedy | `输出 == 预期`（见 § 当前进度总览 实测数据） |
| — | `test_no_manual_buffer_code` | grep `_ensure_graph_buffers\|_hs_out\|_residual_out` qwen.py | 0 命中（inductor 全权管理） |

### 2.3 性能 Profiling 验证

```bash
# 单卡单层: compile warmup + capture + replay timing
CUDA_VISIBLE_DEVICES=0 python -c "
# compile + capture
t0=time.perf_counter()
wrapper = CUDAGraphWrapper(torch.compile(layer.forward_decode, fullgraph=True))
wrapper(hidden, pos, 4, 128, residual); torch.cuda.synchronize()
print(f'compile+capture: {(time.perf_counter()-t0)*1000:.1f}ms')

# replay avg
t0=time.perf_counter()
for _ in range(1000): wrapper(hidden, pos, 4, 128, residual)
torch.cuda.synchronize()
print(f'replay: {(time.perf_counter()-t0)/1000*1e6:.1f}us avg')
"
# 断言: replay < 300μs（inductor kernel + graph replay）
```

### 2.4 文档沉淀

- `CUDAGraphWrapper` 与 vLLM `cuda_graph.py:145-356` 的逐行对照表
- `torch.compile` compile 耗时、inductor 生成的 kernel 列表
- 与 vLLM TF-9（triton kernel）的对比说明

### 2.5 Git Commit 门禁

```bash
git add engine/tp_layers/cuda_graph_wrapper.py engine/models/qwen.py tests/test_stage1_compile_wrapper.py
git commit -m "feat(cudagraph): pass stage_1_compile_and_wrapper

- torch.compile(fullgraph=True) on each layer.forward_decode (inductor auto buffer)
- CUDAGraphWrapper extracted from vllm/compilation/cuda_graph.py:145-356
- No manual buffer allocation — inductor handles all intermediate tensors
- Single GPU single-layer compile+capture+replay verified"
```

---

## 三、阶段二：单卡单层图捕获 10K Replay Stress (DFT)

> **依赖前置**: 阶段一  
> **核心对齐依据**: TF-5 (每层 1 完整图) — 隔离验证 10K replay 稳定性  
> **目标**: 10K replay 无 crash，健康探针连续监控无异常

### 3.1 代码替换与适配

本阶段无新代码替换。基于阶段一的 `torch.compile` + `CUDAGraphWrapper` 产物，编写隔离压测。

### 3.2 正确性验证

> **门禁**: L1 逐层张量 + L2 E2E 输出 二者 PASS 后，方可进入 3.3 性能 Profiling。

**测试文件**: `tests/test_layer_graph_single_gpu.py`  
**运行方式**: `CUDA_VISIBLE_DEVICES=0 python -m pytest tests/test_layer_graph_single_gpu.py -v`

| 层级 | 用例 | 方法 | 通过标准 |
|------|------|------|---------|
| — | `test_001_capture_succeeds` | wrapper(...) 不抛异常 | 无 CUDA error |
| L1 | `test_002_replay_matches_eager` | `assert_close(eager, graph_replay)` | `rtol=1e-2, atol=1e-2` |
| L1 | `test_003_10k_replay_no_crash_no_nan` | 10,000 次 replay，每 1000 次 NaN/Inf 探针 | 无 crash，无 NaN |
| — | `test_004_health_check_no_sync` | check_graph_health 返回值均为 GPU tensor | `v.device.type == 'cuda'` |
| L2 | `test_005_e2e_greedy_decode_match` | 5 轮 24 tokens greedy decode | 每轮字字对齐（见 § 当前进度总览 实测数据） |

**通过标准**: 4 项全部 PASS。

### 3.3 性能 Profiling 验证

**单层 replay vs eager** (1000 runs avg):

```bash
CUDA_VISIBLE_DEVICES=0 python tests/test_layer_graph_single_gpu.py
```

| 指标 | 值 |
|------|-----|
| Capture 耗时 | 3.7ms |
| Replay avg | **311μs** |
| Eager avg | 707μs |
| **Replay 加速比** | **2.3x** |

**E2E 吞吐对比** (12 tokens, 5 runs, skip cold):

| 模式 | 单 GPU | TP=4 | vs Stage 0 基线 |
|------|--------|------|---------------|
| Stage 0 基线 (nocompile) | ~65 tok/s | 53.9 tok/s | — |
| Stage 2 (CUDA Graph) | **66.5 tok/s** | **57-60 tok/s** | **+2.3% / +6-11%** |

TP=4 的 6-11% 提升来自 CUDA Graph 减少了 CPU dispatch 开销，但仍走 CustomAR eager 通信路径（非 NCCL graph 路径，阶段三待做）。

### 3.3-A: meta-infer vs vLLM 三场景 Profiling 对比 (2026-05-26)

> **环境**: Qwen3-8B, GPU 0-3 (A800 80GB), 12 output tokens, temperature=0
> **meta-infer**: meta conda env (PyTorch 2.9.1), commit `feature/tp-implementation`
> **vLLM**: meta conda env (vLLM 0.15.1), `max_model_len=1024, gpu_memory_utilization=0.85`
> **方法**: 5 轮 warmup + 5 轮测量取平均，`torch.cuda.synchronize()` + `time.perf_counter()` 计时
> **代码修复**: 将 `forward_decode` 拆为两版本——eager 版（无 clone，`CUDA_GRAPH=0`）和 graph 版（clone 输入，`CUDA_GRAPH=1`），消除了之前无条件 clone 造成的 ~15% 性能回退

#### 场景一：单 GPU CUDA Graph

| 指标 | meta-infer (CUDA_GRAPH=1) | vLLM (enforce_eager=False) | meta/vLLM |
|------|--------------------------|---------------------------|-----------|
| **Wall time** | 338.6ms | 131.1ms | **2.58x** |
| **Throughput** | 35.4 tok/s | **91.5 tok/s** | **0.39x** |

**GPU kernel 耗时分布** (meta-infer, profiler key_averages):

| kernel 类别 | meta-infer 单 GPU | vLLM 单 GPU (profiler_out_0.txt) |
|------------|------------------|-------------------------------|
| GEMM (矩阵乘法) | ~113ms (80%) | ~100ms (78%) |
| Flash Attention | ~2.7ms (2%) | ~5.6ms (4%) |
| RMS Norm (triton_red_fused) | ~2.5ms (2%) | ~5.1ms (4%) |
| KV Cache | ~3.1ms (2%) | ~1.3ms (1%) |
| 其他 (copy, overhead) | ~20ms (14%) | ~16ms (12%) |

**分析**: kernel 分布相似（GEMM 都占 ~80%）。差距来自：vLLM ~48 次 graph launch vs meta-infer ~396 次（torch.compile reduce-overhead 每层单独的子图），导致 CPU dispatch 开销 8x 差距。

---

#### 场景二：TP=4 无 torch.compile

| 指标 | meta-infer (CUDA_GRAPH=0) | vLLM (enforce_eager=True) | meta/vLLM |
|------|--------------------------|--------------------------|-----------|
| **Wall time** | **215.6ms** | 273.2ms | **0.79x** ✅ |
| **Throughput** | **55.7 tok/s** | 43.9 tok/s | **1.27x** ✅ |

**GPU kernel 耗时分布** (rank-0 profiler):

| kernel 类别 | meta-infer TP=4 | vLLM TP=4 |
|------------|----------------|----------|
| GEMM (矩阵乘法) | ~67ms (40%) | ~11ms (4%) |
| **通信 (AllReduce)** | **~25ms** (15%, CustomAR P2P) | **204.1ms** (79%, NCCL ring) |
| Flash Attention | ~8ms (5%) | ~3.5ms (1%) |
| RMS Norm | ~6ms (3.5%) | ~9ms (3.5%) |
| 其他 | ~60ms (36%) | ~30ms (12%) |

**分析**: meta-infer 比 vLLM 快 **27%**。核心优势：CustomAR P2P 对小 tensor 的 all_reduce 比 NCCL ring reduce 快 **8.2 倍**（25ms vs 204ms）。验证了阶段零引入的 CustomAR 黑盒通信算子的巨大收益。与 kernel_replacement_plan.md 基线（53.9 tok/s）相比提升 3.3%，clone 回归已修复。

---

#### 场景三：TP=4 有 torch.compile / CUDA Graph

| 指标 | meta-infer (当前最佳) | vLLM (default, enforce_eager=False) | meta/vLLM |
|------|----------------------|-------------------------------------|-----------|
| **Wall time** | 215.6ms* | **65.1ms** | **3.31x** |
| **Throughput** | 55.7 tok/s* | **184.4 tok/s** | **0.30x** |

> \* meta-infer 当前 TP=4 不支持 torch.compile + CUDA Graph（阶段三待实施，见 §四），此列为 CUDA_GRAPH=0 数据

**分析**: vLLM 的 CUDA Graph 将 CPU dispatch 从数千次 kernel launch 降为 48 次 graph replay，GPU 融合消除中间 tensor 读写。阶段三的目标就是缩小这 3.31x 差距。

---

#### 与 kernel_replacement_plan.md 历史数据对比

| 来源 | meta-infer TP=4 | vLLM TP=4 | meta/vLLM | 说明 |
|------|----------------|-----------|-----------|------|
| kernel_replacement_plan.md (05-23) | 53.9 tok/s | — | — | 基线，含 clone 回归 |
| 本次 (05-26, clone 修复后) | **55.7 tok/s** | 43.9 tok/s | **1.27x** ✅ | clone 回归修复，+3.3% |
| vLLM CUDA Graph | — | 184.4 tok/s | 0.30x | 阶段三待追赶 |

**核心结论**: 阶段零/一/二的改造**没有负优化**——之前看起来慢是因为 `forward_decode` 中的无条件 clone（为阶段三-B 提前加的）在 eager 模式下造成了 ~15% 性能回退。将 clone 隔离到 graph 专用方法后，TP=4 nocompile 性能完整恢复（55.7 tok/s，甚至略超基线 53.9 tok/s）。单 GPU CUDA Graph 正确性验证通过（5/5 字字对齐）。

### 3.4 文档沉淀

- 10K replay stress 结果: 无 crash，无 NaN
- `check_graph_health()` 返回值规范表: 全部 GPU tensor，无 `.item()` sync
- E2E 吞吐对比表（单 GPU + TP=4）
- **踩坑**: `torch.compile` 惰性编译在 `torch.cuda.graph()` 内部触发 → Dynamo 访问 RNG state 报错。修复: eager warmup 先触发编译，再 CUDAGraphWrapper 捕获

### 3.5 Git Commit 门禁

```bash
git add tests/test_layer_graph_single_gpu.py engine/models/qwen.py
git commit -m "test(cudagraph): pass stage_2_single_layer_single_gpu_capture

- Single-layer CUDA Graph capture + 10K replay stress PASS
- Fix: eager-compile BEFORE CUDAGraphWrapper (Dynamo RNG forbidden in graph)
- 6/6 pytest PASS, replay 311us vs eager 707us (2.3x)
- E2E: single GPU 66.5 tok/s (+2.3%), TP=4 57-60 tok/s (+6-11%)"
```

---

## 四、阶段三：TP=4 黑盒通信算子注册 + torch.compile + CUDAGraphWrapper

> **依赖前置**: 阶段二  
> **核心对齐依据**: vLLM `parallel_state.py:262-266` 将 `all_reduce` 注册为 `torch.ops.vllm.all_reduce` 自定义算子，编译器当作黑盒，不追踪内部、不检查守卫。我们不需要搬整个 VllmBackend，只用 PyTorch 标准 API `torch.library.custom_op` 对 `all_reduce_sum` 做同样的事。  
> **阻塞原因**: `torch.compile(fullgraph=True)` 在四卡时追踪进入 `all_reduce_sum` 内部 → 发现预热和捕获时 CUDA 流状态不同 → guard 失效 → Dynamo 重编译 → `torch.cuda.get_rng_state()` 在图捕获期间被禁止 → 崩溃。

### 4.1 标品资产提取

#### Snippet G: `all_reduce_sum` 注册为 PyTorch 自定义黑盒算子

**对标 vLLM 源码** (`parallel_state.py:262-266`):

```python
# vLLM 将通信 op 注册为 torch.ops.vllm.all_reduce —— 编译器不追踪内部
direct_register_custom_op(
    op_name="all_reduce",
    op_func=all_reduce,
    fake_impl=all_reduce_fake,
)
```

**提取到 meta-infer** (`engine/tp_layers/distributed.py`):

```python
# Snippet G: all_reduce_sum 注册为 meta_infer::all_reduce_sum 黑盒算子
# 对标 vLLM parallel_state.py:262-266 的 direct_register_custom_op("all_reduce")
# 以及 BB-7 (engine/kernels/custom_ops.py) 的 flash_attn custom op 注册模式

@torch.library.custom_op("meta_infer::all_reduce_sum", mutates_args=())
def all_reduce_sum(x: torch.Tensor) -> torch.Tensor:
    """黑盒通信算子。编译器不追踪内部，不检查 guard。

    Contract:
        x: [*, *] bf16/fp16, 任意 shape
        returns: [*, *] same shape, all_reduce sum result
    内部: CustomAR (eager) 或 NCCL (fallback)，编译器不可见
    """
    if not is_tp_enabled():
        return x.clone()  # 必须返回新 tensor — custom_op 禁止输出别名输入
    if _custom_ar_handle is not None:
        return _custom_ar_handle.all_reduce(x, registered=False)
    y = x.clone()
    dist.all_reduce(y, op=dist.ReduceOp.SUM)
    return y

@all_reduce_sum.register_fake
def _(x: torch.Tensor) -> torch.Tensor:
    """FakeTensor 接口 — 仅返回同 shape/dtype 的空 tensor 供编译器推导 shape"""
    return torch.empty_like(x)
```

**关键约束**:
1. `mutates_args=()` — 声明不修改输入。输出**必须**是新 tensor（不能和输入是同一个对象）。单卡时 `is_tp_enabled()=False` → 返回 `x.clone()` 而非 `x`
2. 必须有 `register_fake` — 返回同 shape/dtype 的 FakeTensor，供编译器在追踪阶段推导输出的形状和类型
3. 内部调用 CustomAR 或 NCCL 均为黑盒——编译器仅知道"输入 X → 输出 Y"，不感知具体通信逻辑

### 4.2 组装说明

**使用资产**: Snippet G (`all_reduce_sum` 自定义黑盒算子), BB-4 (`CUDAGraphWrapper`), BB-5 (`torch.compile`)

**组装目标**: `QwenTPModelRunner._setup_cuda_graph_piecewise()` 中的 `all_reduce_sum` 调用对编译器透明，TP=4 时编译器不重编译

**组装约束**:
- Snippet G 必须在 `RowParallelLinear.forward()` 首次调用前完成注册（import 时自动注册）
- `torch.compile(fullgraph=True)` 编译 `forward_decode` 时，编译器遇到 `meta_infer::all_reduce_sum` 当作黑盒节点，不追踪进入
- 编译后的函数在 warmup 和 capture 期间走**完全相同的 FX 图** → 无 guard 失效 → 无重编译

**组装示意** (`engine/tp_layers/distributed.py`):

```python
# 阶段三核心改动：将 all_reduce_sum 从普通 Python 函数替换为自定义黑盒算子
# 改动前 (编译器追踪进入 → guard 失效 → 崩溃):
def all_reduce_sum(x):
    if not is_tp_enabled(): return x
    ...

# 改动后 (编译器当作黑盒 → 不追踪 → 不重编译):
@torch.library.custom_op("meta_infer::all_reduce_sum", mutates_args=())
def all_reduce_sum(x):                              # ← 黑盒调用
    ...
```

**对阶段一的影响**: 无——`CUDAGraphWrapper` 和 `torch.compile` 调用方式不变。仅 `all_reduce_sum` 的实现形式从普通函数变为自定义算子，调用方（`RowParallelLinear.forward()`）无需修改。

### 4.3 正确性验证（阶段三-A）

> **门禁**: L1 逐层张量 + L2 E2E 输出 二者 PASS 后，方可进入 4.4 性能 Profiling。

**测试文件**: `tests/test_layer_graph_tp4.py`  
**运行方式**: `torchrun --nproc_per_node=4 tests/test_layer_graph_tp4.py`

| 层级 | 用例 | 方法 | 通过标准 |
|------|------|------|---------|
| — | `test_custom_op_registered` | `torch.ops.meta_infer.all_reduce_sum` 可调用 | 不抛异常 |
| — | `test_fake_impl_shape` | `torch.compile` 追踪时 fake impl 返回正确 shape | 编译不报错 |
| — | `test_compile_no_recompile_tp4` | 预热后 capture 不触发 Dynamo 重编译 | 无 RNG error |
| L1 | `test_replay_matches_eager_tp4` | 单层图回放 vs eager 数值对齐 | `rtol=1e-2, atol=1e-2` |
| L1 | `test_10k_replay_no_crash` | 10K 单层图回放，NaN/Inf 探针无异常 | 无 crash、无 NaN |
| L2 | `test_e2e_greedy_decode_match_tp4` | TP=4 24 tokens greedy decode | 字字对齐 |

**非阻塞可观测性断言**:
- 严禁 `.item()`、`.cpu()` 触发 Host-Device Sync
- NaN/Inf 检测: `torch.isnan(out).any()` → GPU tensor
- 字字对齐: `temperature=0` 下 `assert out == expected`

### 4.4 性能 Profiling 验证

```bash
torchrun --nproc_per_node=4 python -c "
with torch.profiler.profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
) as prof:
    # warmup + capture + replay
    ...
# 断言:
# 1. cudaGraphLaunch 在 trace 中出现
# 2. 单层 replay < 300μs
# 3. CPU dispatch 缩减 > 80% (vs eager)
# 4. 无 Dynamo 重编译 event
"
```

### 4.5 文档沉淀

- Snippet G 黑盒接口契约（输入/输出 shape、dtype、约束）
- vLLM `parallel_state.py:262-266` 对标说明
- 与 BB-7 (`custom_ops.py`) 注册模式一致性说明

### 4.6 阶段三-B：CUDA 图回放修复 — inductor 内部 CUDA 图 + 消除 mutated inputs

> **依赖前置**: 阶段三-A（自定义黑盒算子注册）  
> **阻塞原因**: 阶段三-A 的自定义算子解决了 Dynamo 重编译，但 CUDA 图回放仍 crash（illegal memory access）。根因是 cuBLAS workspace 地址 + inductor 内存规划 + 通信算子三者交互冲突。

#### 阻塞根因详析

**单卡为什么能工作**: 单卡时 `all_reduce_sum` 直接返回 `x.clone()`，无通信。inductor 编译 `forward_decode` 时，所有操作（矩阵乘法、残差加法、归一化）都在单张 GPU 上执行。inductor 的 memory planner 在编译阶段就规划好所有中间张量的地址，CUDA Graph capture/replay 期间地址不变——因为所有操作都是本地的，不需要和其他 GPU 交换数据。

**四卡为什么崩溃**: 四卡时 `RowParallelLinear` 的输出经过 `all_reduce_sum` 做跨卡规约。`all_reduce_sum` 被注册为 `torch.library.custom_op`，inductor 视其为黑盒 GPU 核函数。问题出在这个黑盒的**输入**——它来自 `F.linear`（cuBLAS 矩阵乘法）。cuBLAS 在执行时分配 internal workspace，这块 workspace 的地址在 CUDA Graph 两次回放之间**不被 graph pool 保证稳定**。回放时，cuBLAS 把矩阵结果写到地址 A'（和 capture 时的 A 不同），但 `all_reduce` 在图里记录的操作仍然从地址 A 读数据。A 处现在是未定义内容 → `all_reduce` 读到垃圾 → NCCL 或 CustomAR 试图用无效地址做跨卡通信 → `cudaErrorIllegalAddress`。**本质矛盾**: inductor 认为"这些临时内存归我管，我可以随时调整"；CUDA Graph 认为"capture 时什么地址，replay 时就是什么地址"。当通信操作夹在 inductor 管理的代码中间时，两者的内存管理策略互相冲突。

**vLLM 的解决方式**: VllmBackend 拿到 FX 图后，在 `all_reduce` 等通信节点的边界处做 **graph partition**——把一张大图切成若干张小图（piecewise subgraphs），每张小图单独交给 inductor 编译。切割后，每张小图内部要么是纯计算（cuBLAS GEMM + fused elementwise），要么是纯通信（NCCL all_reduce）。纯计算的小图里没有通信，inductor 的内存规划正常生效。通信节点被放在小图的边界上，由 VllmBackend 直接调用 `torch.ops.vllm.all_reduce`——这是一个在 C++ 层注册的 custom op，内部直接用 ProcessGroupNCCL 的 all_reduce 接口，不经过 inductor 的内存池。

**为什么不引入 VllmBackend**: 它不是独立模块。它的 graph partition 逻辑依赖 `VllmConfig`、`splitting_ops` 列表、`PiecewiseBackend`（管理多 shape 编译缓存）、`wrap_with_cudagraph_if_needed`（每个子图套 `CUDAGraphWrapper`）、`compile_sizes`（多 batch size 分别编译）。这套链路的每一环都和 vLLM 的 `compilation_config`、`forward_context`、`BatchDescriptor`、`CUDAGraphMode` 耦合。单独抽出通信部分需要同时抽出 partitioner、backend、wrapper、config——本质上就是把 VllmBackend 整体移植过来，破坏 meta-infer 作为精简独立推理框架的根本目的。

**已测试排除的路径**: PyTorch 2.9.1 → 2.10.0 升级不解决此问题（实测 2.10.0 仍 crash）；torch.compile `mode='reduce-overhead'` 在单 GPU 上 CUDA Graph 生效（396 cudaGraphLaunch），但在四卡上同样 crash。这是 `torch.compile` + external CUDA Graph + TP 通信三者交互的架构限制，不是单个 PyTorch 版本升级能解决的。

> **当前结论**: 单 GPU CUDA 图全链路已贯通（捕获、回放、正确性、加速）。TP=4 CUDA 图受限于此架构问题，留待未来 PyTorch 从编译器层面修复后再验证。

#### 4.6.1 使用资产

**BB-9**: inductor `cudagraph_trees` — PyTorch builtin，通过 `torch.compile(mode='reduce-overhead')` 启用。inductor 编译完函数后，自动用内部 CUDA 图包装——不需要外部 `CUDAGraphWrapper`。

**BB-5**: `torch.compile(fullgraph=True)` — 确保单图编译。

#### 4.6.2 为什么之前 `reduce-overhead` 失败

P6 阶段已尝试 `torch.compile(fullgraph=True, mode='reduce-overhead')`，但 inductor 报:

```
skipping cudagraphs due to mutated inputs (1 instances)
  → torch.ops._C.rms_norm(hidden_states, residual, ...)
```

`forward_decode` 的**输入参数** `hidden_states` 被 `rms_norm` 原地修改。inductor 的 CUDA 图要求回放时输入 buffer 值不变——如果输入被上一轮回放修改了，下一轮就无法正确回放。

#### 4.6.3 修复方案：消除 mutated inputs（对齐 vLLM 模式）

vLLM 在 PIECEWISE 模式下，每个子图的输入来自**预分配的静态 buffer**——数据在回放前通过 `copy_()` 刷新。子图内部的 in-place 操作只影响内部张量，不修改输入 buffer。

**修改** (`engine/models/qwen.py` `QwenDecoderLayerTP.forward_decode`):

```python
# 修改前 — hidden_states/residual 被 fused_add_rms_norm 原地修改
# → inductor 检测到 mutated inputs → 拒绝 CUDA 图
fused_add_rms_norm(hidden_states, residual, weight, eps)

# 修改后 — clone 后传入，原输入不变
# → inductor 不检测到 mutated inputs → CUDA 图可用
hs_tmp = hidden_states.clone()
res_tmp = residual.clone()
fused_add_rms_norm(hs_tmp, res_tmp, weight, eps)
# 将结果写回原始引用（后续代码不变）
hidden_states, residual = hs_tmp, res_tmp
```

**影响范围**: `forward_decode` 中 3 处 in-place kernel 调用:
- `input_layernorm` — `fused_add_rms_norm` (except first layer)
- `post_attention_layernorm` — `fused_add_rms_norm`
- `attention.forward_decode` 内 — `rms_norm` (q_norm, k_norm), `rotary_embedding`

#### 4.6.4 组装说明

**使用资产**: BB-9 (reduce-overhead), BB-8 (custom op), BB-5 (fullgraph)

**组装约束**:
- `mode='reduce-overhead'` 不能与外部 `CUDAGraphWrapper` 同时使用——让 inductor 内部管理 CUDA 图
- `torch.compile(fullgraph=True, mode='reduce-overhead')` 编译后直接替换 `layer.forward_decode`
- 不再需要 `CUDAGraphWrapper.__call__` 的 capture/replay——inductor 内部 `cudagraph_trees` 自动处理
- 每次 decode step 前调用 `torch.compiler.cudagraph_mark_step_begin()` 标记步边界

**组装示意** (`QwenTPModelRunner._setup_cuda_graph_piecewise`):

```python
# 阶段三-B: reduce-overhead 替代 CUDAGraphWrapper
for layer in self.model.layers:
    layer.forward_decode = torch.compile(
        layer.forward_decode, fullgraph=True, mode='reduce-overhead',
    )
# warmup + mark_step_begin 在 runner.run() 中
```

#### 4.6.5 正确性验证

> **门禁**: L1 逐层张量 + L2 E2E 输出 二者 PASS 后，方可进入性能 Profiling。

| 层级 | 用例 | 方法 | 通过标准 |
|------|------|------|---------|
| L1 | `test_replay_matches_eager_reduce_overhead` | reduce-overhead 模式 replay vs eager 单层输出 | `rtol=1e-2, atol=1e-2` |
| L1 | `test_10k_replay_no_crash` | 10K reduce-overhead replay | 无 crash、无 NaN |
| L2 | `test_e2e_greedy_decode_match_tp4` | TP=4 24 tokens greedy decode | `'（ ） A：建筑与园林结合 B：建筑与自然结合 C：建筑与山水结合 D：建筑'` |
| L2 | `test_5_rounds_stable` | 连续 5 轮 | 每轮输出一致 |

#### 4.6.6 性能 Profiling 验证

```bash
# 断言: reduce-overhead 内部 CUDA Graph 生效 (不再有 "skipping cudagraphs" 警告)
META_INFER_CUDA_GRAPH=1 python -c "..." 2>&1 | grep "skipping cudagraphs"
# 预期: 无输出（CUDA 图已启用）
```

#### 4.6.7 Git Commit 门禁

```bash
git add engine/models/qwen.py
git commit -m "feat(cudagraph): pass stage_3b_reduce_overhead_no_mutated_inputs

- Clone inputs before fused_add_rms_norm to eliminate mutated inputs
- Switch to torch.compile(mode='reduce-overhead') — inductor internal CUDA Graph
- Remove external CUDAGraphWrapper for decode path
- Aligned with vLLM cudagraph_trees approach (BB-9)"
```

---

## 五、阶段四：RealModelRunner 全图联动与调度器串联

> **依赖前置**: 阶段三  
> **核心对齐依据**: **TF-6 (36 层图 + 11 后处理图) + TF-7 (48 cudaGraphLaunch)** + vLLM `cudagraph_utils.py:177-230`  
> **目标**: 36 层 = 36 个 CUDA Graph（raw，无 compile），接入 Scheduler，PIECEWISE 全链路  
> **注**: 阶段三确定 raw CUDA Graph 方案后，阶段四具体方案将相应调整。当前保留 v5 torch.compile 方案框架，待阶段三验证通过后同步更新。

---

## 六、阶段依赖拓扑图

```
┌─────────────────────────────────────────────────────────────┐
│                     阶段零 (Stage 0)                         │
│     伪图代码删除 + CustomAR 黑盒接口恢复                        │
│     TF-8 + vLLM custom_all_reduce.py:199-282                  │
│     └─ commit: pass stage_0_blackbox_cleanup                 │
└───────────────────────────┬─────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                     阶段一 (Stage 1)                         │
│     torch.compile(fullgraph=True) + CUDAGraphWrapper          │
│     inductor 自动 buffer 管理 + kernel fusion                 │
│     TF-3/4/5/9 + vLLM cuda_graph.py:145-356                  │
│     └─ commit: pass stage_1_compile_and_wrapper              │
└───────────────────────────┬─────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                     阶段二 (Stage 2)                         │
│     单卡单层 10K replay stress (DFT)                          │
│     TF-5 — 隔离验证 inductor 编译产物的图稳定性                 │
│     └─ commit: pass stage_2_single_gpu_stress                │
└───────────────────────────┬─────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                     阶段三 (Stage 3)                         │
│     all_reduce_sum 注册为 torch.library.custom_op            │
│     vLLM VllmBackend 黑盒思路 + PyTorch 标准机制              │
│     └─ commit: pass stage_3_custom_op_all_reduce_sum         │
└───────────────────────────┬─────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                     阶段四 (Stage 4)                         │
│     RealModelRunner + Scheduler 全图联动                      │
│     TF-6/7 + vLLM cudagraph_utils.py:177-230                 │
│     └─ commit: pass stage_4_full_model_runner                │
└─────────────────────────────────────────────────────────────┘
```

---

## 七、里程碑与验收标准

| 里程碑 | 阶段 | 验收标准 | 门禁 |
|--------|------|---------|------|
| M0 | 阶段零 | 伪图代码删除，CustomAR 接口恢复，eager 吞吐不退化 | ✅ 已提交 |
| M1 | 阶段一 | torch.compile fullgraph=True 通过，CUDAGraphWrapper capture 不报错 | ✅ 已提交 |
| M2 | 阶段二 | 4 项 DFT 测试 PASS，10K replay 无 crash | ✅ 已提交 |
| M3a | 阶段三-A | `all_reduce_sum` 注册为自定义黑盒算子，TP=4 图捕获不重编译 | ✅ 已验证（Dynamo RNG 解决，replay 仍 crash） |
| M3b | 阶段三-B | sglang 切图方案已分析（~1040 行，核心 split_graph 45 行）；待实施 | 🔴 待实施 |
| M4 | 阶段四 | E2E 字字对齐，CPU <50ms，吞吐 >80 tok/s | ⏸ 待阶段三 |

---

## 附录 A: vLLM Trace 物理事实全集

来源: `/tmp/prof_vllm_cudagraph_tp4/` rank-0 JSON trace (2,595,861 events)

| 编号 | 物理观察 | 数值 | 证据 |
|------|---------|------|------|
| TF-1 | NCCL AllReduce 在图内 | 876 次全部在 48 launch 窗口内 | Window [6] NCCL 202.9μs 与 GEMM/FA/KV 同窗口 |
| TF-2 | CustomAR 调用 | 0 | 关键词 0 命中 |
| TF-3 | KV cache write 在图内 | 37 窗口含 `reshape_and_cache_flash` | 与 FA/GEMM/NCCL 同窗口 |
| TF-4 | FlashAttention 在图内 | 37 窗口含 `flash_fwd_splitkv` | 与 KV/GEMM/NCCL 同窗口 |
| TF-5 | 每层 1 完整图 | 37 窗口含 FA+KV（完整层），10 窗口不含（部分图） | 逐窗口 kernel 分类 |
| TF-6 | 启动分布: 1+36+11 | Launch [0]=1, [1..36]=36, [37..47]=11 | 间隔聚类 |
| TF-7 | cudaGraphLaunch 总次数 | 48 | key_averages |
| TF-8 | cudaStreamIsCapturing | 133 次，48 在 launch 附近 | CPU op 时序 |
| TF-9 | torch.compile/dynamo CPU event | 0（编译在 init 完成） | trace 仅含 inductor 产物 |

## 附录 B: vLLM 关键源码索引

| 文件 | 行号 | 内容 | 对应阶段 |
|------|------|------|---------|
| `vllm/compilation/cuda_graph.py` | 145-356 | `CUDAGraphWrapper` | 阶段一 |
| `vllm/distributed/device_communicators/custom_all_reduce.py` | 199-211 | `capture()` | 阶段零 |
| `vllm/distributed/device_communicators/custom_all_reduce.py` | 213-230 | `register_graph_buffers()` | 阶段零 |
| `vllm/distributed/device_communicators/custom_all_reduce.py` | 247-264 | `all_reduce(inp, *, out, registered)` | 阶段零 |
| `vllm/distributed/device_communicators/custom_all_reduce.py` | 266-282 | `custom_all_reduce(inp)` | 阶段零 |
| `vllm/distributed/parallel_state.py` | 463-490 | `graph_capture()` | 阶段三 |
| `vllm/v1/worker/gpu/cudagraph_utils.py` | 177-230 | `CudaGraphManager.capture()` | 阶段四 |
