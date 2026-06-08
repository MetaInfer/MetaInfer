# Phase 9 发现的 Bug 修复完整记录

Phase 9 (引擎集成) 首次触发端到端 `generate()` 流程，暴露了前 8 个 Phase 在单卡测试中未被发现的 TP=4 运行时错误。共修复 7 个 bug（Bug 8 为 Phase 10 benchmark 脚本问题，另见 PHASE9_BUG_FIXES.md）。

## 共同模式

所有 7 个 bug 的共同特征：**前序 Phase 的 spec-review 和 verification 都 PASS 了**，但 Phase 9 一跑 generate() 就炸。原因：

1. 单卡测试不覆盖 TP=4 路径（Bug 2, 7）
2. 模块测试不覆盖完整 forward 链（Bug 4, 5）
3. 随机权重测试不依赖真实模型文件（Bug 1, 6）
4. 无 GPU forward 的测试不检查 device 放置（Bug 3）

**教训**：每个 Phase 的测试覆盖面和真实 E2E 场景之间存在 gap。Phase 9 是这个 gap 的"清算时刻"。

---

## Bug 1: float32 全量模型显存爆炸

- **症状**: TP=4 加载 Qwen3-8B 模型时，rank 0 显存分配 31661 MB，远超单卡容量。8B 参数 × 4 bytes (float32) = ~32 GB。
- **发现过程**: Phase 9 首次完整加载 Qwen3-8B 真实模型权重并执行 generate()。之前 Phase 7 测试只验证了权重加载的正确性和显存使用（3.83 GB/rank），因为测试使用随机权重或立即退出，未触发全量 float32 分配。
- **复现**:
  ```bash
  python -c "
  from engine.models.qwen import QwenTPConfig, QwenForCausalLMTP
  cfg = QwenTPConfig.from_model_dir('${MODEL_DIR}/qwen/Qwen3-8B')
  model = QwenForCausalLMTP(cfg)  # 无 device 参数 → 不执行 to(dtype=bf16)
  # → rank0 31661 MB OOM
  "
  ```
- **根因**: `QwenForCausalLMTP.__init__` 有 `dtype=torch.bfloat16` 默认参数，但 `if device is not None: self.to(device, dtype)` 分支在 `device=None` 时不执行 `.to(dtype=bf16)`，模型保持 PyTorch 默认 float32。同时调用方 [llm_engine.py:52](llm_engine.py#L52) 未显式传递 dtype 参数。
  ```python
  # 修复前 (qwen.py:461):
  def __init__(self, cfg, device=None, dtype=torch.bfloat16):
      ...
      if device is not None:
          self.to(device=device, dtype=dtype)  # device=None 时不执行 → float32
  ```
- **修复**:
  ```python
  # engine/models/qwen.py:461-471 (修复后)
  def __init__(self, cfg, device=None, dtype=torch.bfloat16):
      super().__init__()
      self.cfg = cfg
      self.embed_tokens = VocabParallelEmbedding(cfg.vocab_size, cfg.hidden_size)
      self.layers = nn.ModuleList(
          [QwenDecoderLayerTP(cfg) for _ in range(cfg.num_hidden_layers)])
      self.norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
      self.lm_head = ParallelLMHead(cfg.vocab_size, cfg.hidden_size)
      if device is not None:
          self.to(device=device, dtype=dtype)  # dtype=bf16 强制转换
  ```
  同时在 [llm_engine.py:52](llm_engine.py#L52) 调用方显式传递 dtype：
  ```python
  self.model = QwenForCausalLMTP(self.cfg, device=device, dtype=dtype)
  ```
- **为什么单卡/前序 Phase 测试没发现**: Phase 7 测试用随机权重/小尺寸，不分配 8B 参数 → 不会 OOM。Phase 7 verification 只检查了签名存在性，未验证调用方实际传递了 dtype。
- **关联 Phase**: Phase 7 (权重加载与模型构造)
- **关联文件**:
  - [engine/models/qwen.py:461-471](engine/models/qwen.py#L461)
  - [llm_engine.py:52](llm_engine.py#L52)

---

## Bug 2: init_tp_distributed 无 WORLD_SIZE guard

- **症状**: 单 GPU 非 torchrun 启动 (`python script.py`) 时，进程永久 hang 在 `dist.init_process_group(backend="nccl", init_method="env://")`，等待不存在的 NCCL master 进程。
- **发现过程**: Phase 9 引入 `LLMEngine` 后，首次出现非 torchrun 启动的端到端场景（如单 GPU `generate()` 测试）。`test_phase9_generate_single_gpu.sh` 初次运行时直接 hang。
- **复现**:
  ```bash
  python -c "
  from llm_engine import LLMEngine
  engine = LLMEngine('${MODEL_DIR}/qwen/Qwen3-8B', inference_backend='qwen_tp')
  # → 永久 hang
  "
  ```
- **根因**: [llm_engine.py:232](llm_engine.py#L232) `init_tp_distributed()` 在 `WORLD_SIZE <= 1` 时未做 guard 直接调用。`dist.init_process_group` 在单进程环境下等待不存在的 NCCL master。
  ```python
  # 修复前 (llm_engine.py:232):
  if not is_tp_enabled():
      init_tp_distributed()  # 无 WORLD_SIZE guard → 单进程 hang
  ```
- **修复**: 添加 `_world_size > 1` 条件：
  ```python
  # llm_engine.py:232-234 (修复后)
  _world_size = int(_os.environ.get('WORLD_SIZE', '1'))
  if _world_size > 1 and not is_tp_enabled():
      init_tp_distributed()
  ```
- **为什么单卡/前序 Phase 测试没发现**: Phase 2-8 测试全部通过 `torchrun --nproc_per_node=4` 启动，不会触发单进程路径。Phase 9 首次出现非 torchrun 场景。
- **关联 Phase**: Phase 2 (TP 通信初始化)
- **关联文件**:
  - [llm_engine.py:231-234](llm_engine.py#L231)
  - [engine/tp_layers/distributed.py:203-215](engine/tp_layers/distributed.py#L203)

---

## Bug 3: input_ids_tensor 无 device 参数

- **症状**:
  ```
  RuntimeError: Expected all tensors to be on the same device, but found at least two devices, cuda:0 and cpu!
  ```
  在 `QwenTPModelRunner.run()` prefill 路径中，`torch.cat([s.input_ids_tensor() ...])` 返回 CPU tensor，与其他 CUDA tensor 不兼容。
- **发现过程**: Phase 9 首次将 Phase 8 的 Sequence/Scheduler 与 QwenTPModelRunner 串联。`Sequence.input_ids_tensor()` 在 Phase 8 测试中只在 CPU 上使用，但 Phase 9 需要与 GPU 上的 embedding weight 交互。
- **复现**:
  ```python
  from engine.structs import Sequence
  seq = Sequence(request_id='test', input_ids=[1,2,3])
  t = seq.input_ids_tensor()  # device=None → CPU tensor
  # 后续 torch.cat 或 CUDA 操作 → RuntimeError
  ```
- **根因**: [engine/structs.py:61](engine/structs.py#L61) `input_ids_tensor(device=None)` 在 `device=None` 时 `torch.tensor(..., device=None)` 创建的 tensor 在 CPU 上。Phase 8 设计假设所有测试不涉及 GPU forward。
  ```python
  # 修复前 (structs.py:61):
  def input_ids_tensor(self, device=None):
      return torch.tensor([self.input_ids], dtype=torch.long, device=device)
  # device=None → CPU tensor
  ```
- **修复**:
  1. `input_ids_tensor(device=None)` 接口不变，但调用方 [llm_engine.py:117](llm_engine.py#L117) 显式传 `device=self.device`：
     ```python
     input_ids = torch.cat([s.input_ids_tensor(device=self.device) for s in seqs], dim=1)
     ```
  2. [llm_engine.py:486](llm_engine.py#L486) `Sequence(...)` 构造时添加 `device=self.device`：
     ```python
     seq = Sequence(
         request_id=req_id, input_ids=input_ids,
         block_size=self.block_size, ...
         device=self.device)  # 修复: 添加 device 参数
     ```
- **为什么单卡/前序 Phase 测试没发现**: Phase 8 测试 (`test_phase8_sequence_scheduler.py`) 只测 Sequence 状态管理和 scheduler 逻辑，不需要 GPU forward → 所有 tensor 在 CPU 上。
- **关联 Phase**: Phase 8 (框架外壳 — Sequence/Scheduler)
- **关联文件**:
  - [engine/structs.py:61-62](engine/structs.py#L61)
  - [llm_engine.py:117, 486](llm_engine.py#L117)

---

## Bug 4: RMSNorm 条件返回类型不统一

- **症状**:
  ```python
  hidden_states, _ = self.norm(hidden_states, residual)
  # ValueError: not enough values to unpack (expected 2, got 1)
  ```
  影响 7 处调用点：QwenForCausalLMTP.forward (3 处)、QwenAttentionTP (4 处 q_norm/k_norm)。
- **发现过程**: Phase 9 首次通过 QwenForCausalLMTP.forward 完整 prefill/decode 流程，触发了 `self.norm(...)`、`self.q_norm(...)`、`self.k_norm(...)` 调用。前序 Phase 的 QwenDecoderLayerTP 使用裸 kernel 函数绕过此问题。
- **复现**:
  ```python
  from engine.models.qwen import RMSNorm
  import torch
  norm = RMSNorm(hidden_size=4096, eps=1e-6)
  x = torch.randn(1, 4096, device='cuda')
  out, _ = norm(x)  # residual=None 分支返回单 Tensor → ValueError
  ```
- **根因**: Phase 5 实现 `RMSNorm.forward` 时，两个分支都返回单个 Tensor，但蓝图契约要求统一返回 2-tuple `(out, residual)` 以供 `hidden_states, _ = self.norm(...)` 解包。
  ```python
  # 修复前 (qwen.py:72-79):
  def forward(self, x, residual=None):
      if residual is None:
          out = torch.empty_like(x)
          rms_norm(out, x.contiguous(), self.weight, self.eps)
          return out           # ← Bug: 返回单 Tensor
      else:
          fused_add_rms_norm(x.contiguous(), residual.contiguous(), self.weight, self.eps)
          return x             # ← Bug: 返回单 Tensor
  ```
- **修复**: [engine/models/qwen.py:72-79](engine/models/qwen.py#L72) 统一返回 2-tuple：
  ```python
  # 修复后:
  def forward(self, x, residual=None):
      if residual is None:
          out = torch.empty_like(x)
          rms_norm(out, x.contiguous(), self.weight, self.eps)
          return out, None        # ← 返回 2-tuple
      else:
          fused_add_rms_norm(x.contiguous(), residual.contiguous(), self.weight, self.eps)
          return x, residual      # ← 返回 2-tuple
  ```
  同时在 QwenAttentionTP 中 4 处 `q_norm/k_norm` 调用改用 tuple unpacking：
  ```python
  # 修复前: q = self.q_norm(q)
  # 修复后: q_flat, _ = self.q_norm(q_flat)
  ```
- **为什么单卡/前序 Phase 测试没发现**: QwenDecoderLayerTP 使用裸 kernel 函数 `rms_norm()` / `fused_add_rms_norm()`（从 `engine.kernels.vllm_wrappers` 导入），**不走 RMSNorm.forward**。只有 QwenForCausalLMTP.forward 和 QwenAttentionTP 调用 `self.norm(...)` / `self.q_norm(...)` / `self.k_norm(...)`，而这些调用链是 Phase 9 首次激活的。
- **关联 Phase**: Phase 5 (Attention/KV Cache — RMSNorm 定义于此 Phase)
- **关联文件**:
  - [engine/models/qwen.py:72-79](engine/models/qwen.py#L72) (RMSNorm.forward)
  - [engine/models/qwen.py:188-189, 248-249, 502, 513, 543](engine/models/qwen.py#L188) (7 处调用点)

---

## Bug 5: flash_attn 2.8.3 API 签名不兼容

- **症状**: `flash_attn_with_kvcache` 使用 positional args 调用时参数错位，导致 CUDA kernel 参数 mismatch 或静默数值错误。
- **发现过程**: Phase 9 环境安装了 flash_attn 2.8.3，其 `flash_attn_with_kvcache` 签名在中间增加了 `softmax_scale` 等新参数。Phase 5 的 positional arg 写法与新签名不兼容。真实引擎 `meta-infer/engine/kernels/custom_ops.py` 使用 keyword args，验证了修复方向。
- **根因**: flash_attn 不同版本间 `flash_attn_with_kvcache` 参数顺序有变化。修复前全部使用 positional args：
  ```python
  # 修复前 (qwen.py:266-269):
  out = flash_attn_with_kvcache(
      q_attn, self._key_cache, self._value_cache,
      self._kv_len_gpu, self._block_table,
      self.scaling, causal=False)          # 全部 positional → 错位风险
  ```
- **修复**: [engine/models/qwen.py:269-274](engine/models/qwen.py#L269) 全部改用 keyword args：
  ```python
  # 修复后:
  out = flash_attn_with_kvcache(
      q_attn, self._key_cache, self._value_cache,
      cache_seqlens=self._kv_len_gpu,      # ← keyword arg
      block_table=self._block_table,       # ← keyword arg
      softmax_scale=self.scaling,          # ← keyword arg
      causal=False)
  ```
  与真实引擎 `meta-infer/engine/kernels/custom_ops.py:16-28` 一致。
- **为什么单卡/前序 Phase 测试没发现**: Phase 5 测试时可能使用了不同版本 flash_attn，API 签名与 2.8.3 不同。
- **关联 Phase**: Phase 5 (Attention — flash_attn_with_kvcache 调用)
- **关联文件**:
  - [engine/models/qwen.py:269-274](engine/models/qwen.py#L269)
  - `/home/honglin/meta-infer/engine/kernels/custom_ops.py:16-28` (参考实现)

---

## Bug 6: q_norm/k_norm 权重未加载

- **症状**: `engine.generate('苏州园林的特点是', max_new_tokens=24, temperature=0.0)` 输出乱码/随机 token，而非正确的中文。
  修复后正确输出: `（ ） A：建筑与园林结合 B：建筑与自然结合 C：建筑与山水结合 D：建筑`
- **发现过程**: Phase 9 首次执行完整 E2E generate()。模型正常加载、显存正常、forward 不报错，但贪婪解码输出与基线不匹配。通过逐层对比 logits 定位到 attention 层的 Q/K 被错误的 norm 权重污染。
- **复现**:
  ```bash
  bash scripts/test_phase9_generate_single_gpu.sh
  # 修复前: 输出乱码 token → exit 1
  # 修复后: 输出正确中文 → PASS
  ```
- **根因**: `_dispatch_weight` (qwen.py) 处理了 12 个标准 HF key 映射（q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj, input_layernorm, post_attention_layernorm, embed_tokens, norm, lm_head），但**遗漏了 Qwen3 特有的** `self_attn.q_norm.weight` 和 `self_attn.k_norm.weight`。
  
  `QwenAttentionTP.__init__` 定义了：
  ```python
  self.q_norm = RMSNorm(self.head_dim, cfg.rms_norm_eps)
  self.k_norm = RMSNorm(self.head_dim, cfg.rms_norm_eps)
  ```
  但 `_dispatch_weight` 没有对应的 case，导致 `q_norm.weight` 和 `k_norm.weight` 保持随机初始值 (torch.ones)，Q/K 张量被错误的 norm 权重污染 → logits 完全错误 → 输出垃圾 token。
- **修复**: [engine/models/qwen.py:657-660](engine/models/qwen.py#L657) 新增 2 行 dispatch：
  ```python
  elif component == 'self_attn.q_norm.weight':
      layer.self_attn.q_norm.weight.data.copy_(full)
  elif component == 'self_attn.k_norm.weight':
      layer.self_attn.k_norm.weight.data.copy_(full)
  ```
  PHASE9_VERIFICATION_REPORT.md 验证：
  ```
  新增: model.layers.{i}.self_attn.q_norm.weight → layers.{i}.self_attn.q_norm.weight ✅
  新增: model.layers.{i}.self_attn.k_norm.weight → layers.{i}.self_attn.k_norm.weight ✅
  ```
- **为什么单卡/前序 Phase 测试没发现**:
  - Phase 7 spec-review 的 key mapping 表格只有 12 条，**不包含** q_norm/k_norm
  - Phase 7 测试只验证了显存使用和数据加载，未执行端到端 generate()
  - Phase 5/6 测试用随机权重，不依赖真实模型文件
  - **只有 Phase 9 E2E generate() 才暴露权重缺失。这是最隐蔽的 bug：模型正常加载、显存正常、forward 不报错，但输出全错。**
- **关联 Phase**: Phase 7 (权重加载 — `_dispatch_weight`)
- **关联文件**:
  - [engine/models/qwen.py:141-142](engine/models/qwen.py#L141) (q_norm/k_norm 定义)
  - [engine/models/qwen.py:657-660](engine/models/qwen.py#L657) (dispatch 修复)

---

## Bug 7: CustomAR buf_ptrs[0] → TP=4 "buffer address not registered"

**唯一仅在 TP=4 多卡测试中才暴露的 bug。**

- **症状**: TP=4 `generate()` 在 `all_reduce_sum` → `ops.all_reduce` 处崩溃:
  ```
  RuntimeError: buffer address 140245722988544 is not registered! (rank 1/2/3 同时崩溃)
  ```
  rank 0 正常（因为 rank 0 的 `buf_ptrs[0]` 恰好等于 `buf_ptrs[dist.get_rank()]`），rank 1/2/3 全部崩溃。
- **发现过程**:
  1. TP=4 torchrun 启动 → CustomAR init 成功（"CustomAR initialized" 正常打印）
  2. `engine.generate()` → prefill 阶段 `embed_tokens.forward()` → `all_reduce_sum(x)` → `_custom_ar_handle.all_reduce(x, registered=False)` → `ops.all_reduce(...)` → RuntimeError
  3. 对比真实引擎 `meta-infer/engine/tp_layers/custom_ar.py:76` 使用 `self._buf_ptrs[dist.get_rank()]`
  4. Agent 代码 [engine/tp_layers/distributed.py:132](engine/tp_layers/distributed.py#L132) 使用 `self.buf_ptrs[0]`
- **复现**:
  ```bash
  torchrun --nproc_per_node=4 python -c "
  from llm_engine import LLMEngine
  engine = LLMEngine('${MODEL_DIR}/qwen/Qwen3-8B', inference_backend='qwen_tp')
  engine.generate('苏州园林的特点是', max_new_tokens=24, temperature=0.0)
  # → Rank 1/2/3: RuntimeError: buffer address not registered!
  "
  ```
- **根因**:
  ```python
  # Agent 的 distributed.py:132（错误）
  reg_buf = self.buf_ptrs[0]    # 始终用 rank 0 的 buffer 地址

  # 真实 engine/tp_layers/custom_ar.py:76（正确）
  reg_buf = self._buf_ptrs[dist.get_rank()]  # 当前 rank 的 buffer 地址
  ```
  `registered=False` 时 P2P kernel 先将输入 tensor 拷贝到**本 rank** 的 staging buffer，再做跨 rank all_reduce。Rank 3 传了 rank 0 的 buffer 地址，和 `ops.register_buffer(ptr, buf_ptrs)` 时 rank 3 注册的地址（`buf_ptrs[3]`）不一致 → `buffer address not registered`。
- **修复**: [engine/tp_layers/distributed.py:132](engine/tp_layers/distributed.py#L132)：
  ```python
  # 错误:
  reg_buf = self.buf_ptrs[0] if self.buf_ptrs else 0

  # 正确:
  reg_buf = self.buf_ptrs[dist.get_rank()]  # must be THIS rank's buffer
  ```
- **为什么单卡/Phase 2 测试没发现**:
  - 单卡: `world_size == 1` → `init_custom_ar` 直接 return → `_custom_ar_handle is None` → `all_reduce_sum` 走 NCCL fallback，不走 CustomAR P2P 路径
  - Phase 2 `test_phase2_custom_ar_init.sh`: 测了 `init_custom_ar` 成功 + NCCL all_reduce 数值正确，但**没有测 `ops.all_reduce` 的实际 CustomAR P2P 调用路径**
- **蓝图知识缺口**: `custom_ar_all_reduce.constraint` 未说明 `reg_buf` 必须是 `buf_ptrs[rank]`
- **关联 Phase**: Phase 2 (TP 通信)
- **关联文件**:
  - [engine/tp_layers/distributed.py:132](engine/tp_layers/distributed.py#L132)
  - `/home/honglin/meta-infer/engine/tp_layers/custom_ar.py:76` (参考实现)

---

## Bug 修复时间线

```
Phase 1-8: 模块构建 + 单卡测试（全部 PASS）
    ↓
Phase 9: 首次端到端 generate()
    ↓
    ├── Bug 1: float32 OOM → 修 qwen.py + llm_engine.py
    ├── Bug 2: 单进程 hang → 修 llm_engine.py
    ├── Bug 3: CPU/CUDA mismatch → 修 structs.py + llm_engine.py
    ├── Bug 4: tuple unpack error → 修 RMSNorm.forward + 7 处调用
    ├── Bug 5: flash_attn API → 修 qwen.py keyword args
    ├── Bug 6: 垃圾输出 → 修 _dispatch_weight 2 行
    └── Bug 7: custom_ar TP=4 crash → 修 distributed.py buf_ptrs[rank]
    ↓
Phase 9 全部修复后 → generate() 输出正确中文 → 继续 Phase 10-11
```
