# Phase 9 发现的 Bug 修复记录

Phase 9 (引擎集成) 首次触发端到端 generate() 流程，暴露了前 8 个 Phase 在单卡测试中未被发现的 TP=4 运行时错误。共修复 7 个 bug。

---

## Bug 7: CustomAR buf_ptrs[0] → TP=4 运行时 "buffer address not registered"

- **症状**: TP=4 `generate()` 在 `all_reduce_sum` → `ops.all_reduce` 处崩溃:
  ```
  RuntimeError: buffer address 140245722988544 is not registered! (rank 1/2/3 同时崩溃)
  ```
  rank 0 正常（因为 rank 0 的 `buf_ptrs[0]` 恰好等于 `buf_ptrs[dist.get_rank()]`），rank 1/2/3 全部崩溃。

- **发现过程**:
  1. TP=4 torchrun 启动 → CustomAR init 成功（"CustomAR initialized" 正常打印）
  2. `engine.generate()` → prefill 阶段 `embed_tokens.forward()` → `all_reduce_sum(x)` → `_custom_ar_handle.all_reduce(x, registered=False)` → `ops.all_reduce(...)` → RuntimeError
  3. 对比真实引擎 `meta-infer/engine/tp_layers/custom_ar.py:76` 使用 `self._buf_ptrs[dist.get_rank()]`
  4. Agent 代码 `engine/tp_layers/distributed.py:132` 使用 `self.buf_ptrs[0]`

- **根因**:
  ```python
  # Agent 的 distributed.py:132（错误）
  reg_buf = self.buf_ptrs[0]    # 始终用 rank 0 的 buffer 地址

  # 真实 engine/tp_layers/custom_ar.py（正确）
  reg_buf = self._buf_ptrs[dist.get_rank()]  # 当前 rank 的 buffer 地址
  ```
  `registered=False` 时 P2P kernel 先将输入 tensor 拷贝到**本 rank** 的 staging buffer，再做跨 rank all_reduce。Rank 3 传了 rank 0 的 buffer 地址，和 `ops.register_buffer(ptr, buf_ptrs)` 时 rank 3 注册的地址（`buf_ptrs[3]`）不一致 → `buffer address not registered`。

- **为什么单卡/Phase 2 测试没发现**:
  - 单卡: `world_size == 1` → `init_custom_ar` 直接 return → `_custom_ar_handle is None` → `all_reduce_sum` 走 NCCL fallback，不走 CustomAR P2P 路径
  - Phase 2 `test_phase2_custom_ar_init.sh`: 测了 `init_custom_ar` 成功 + NCCL all_reduce 数值正确，但**没有测 `ops.all_reduce` 的实际 CustomAR P2P 调用路径**

- **修复**:
  ```python
  # engine/tp_layers/distributed.py:132
  # 错误:
  reg_buf = self.buf_ptrs[0] if self.buf_ptrs else 0

  # 正确:
  reg_buf = self.buf_ptrs[dist.get_rank()]  # must be THIS rank's buffer
  ```

- **真实引擎对照**: `meta-infer/engine/tp_layers/custom_ar.py` 第 76 行:
  ```python
  ops.all_reduce(
      self._ptr, inp, out,
      self._buf_ptrs[dist.get_rank()], self._max_size,
  )
  ```

- **关联 Phase**: Phase 2 (TP 通信)
- **蓝图知识缺口**: `custom_ar_all_reduce.constraint` 未说明 `reg_buf` 必须是 `buf_ptrs[rank]`

---

## 其余 Bug 摘要

| # | Bug | 症状 | 修复 | Phase |
|---|-----|------|------|-------|
| 1 | float32 全量模型 | rank0 31661 MB | `to(dtype=bf16)` | 7 |
| 2 | init_tp_distributed 无 guard | 单进程 hang | `WORLD_SIZE <= 1: return` | 2 |
| 3 | input_ids_tensor 无 device | CPU/CUDA RuntimeError | device 参数 | 8 |
| 4 | RMSNorm 条件返回类型 | tuple unpack ValueError | 统一 2-tuple | 5 |
| 5 | flash_attn 2.8.3 API 签名 | positional arg 错位 | keyword args | 5 |
| 6 | q_norm/k_norm 未加载 | generate() 垃圾输出 | 2 行 dispatch 添加 | 7 |

**Bug 7 是其中唯一仅在 TP=4 多卡测试中才暴露的 bug。**

---

## Bug 8: agent-engine OpenAI server SIGABRT on startup (benchmark script cleanup issue)

- **症状**: bench_compare.sh 中 meta-infer → agent-engine 切换时，agent server 报 SIGABRT "exitcode: -6"
- **根因**: bench_compare.sh 仅用 `kill -9 PID` 杀 meta-infer server 主进程，torchrun worker 子进程（4 个 `python openai_tp_server.py`）未被杀死，GPU 显存未释放。agent server 启动时 `torch.cuda.set_device` 或 `dist.init_process_group` 因显存不足触发 SIGABRT
- **修复**: bench_compare.sh 每个 engine 结束后增加 `pkill -9 -f "torchrun.*openai_tp_server"`（杀所有 torchrun 子进程）+ `sleep 5`（等 GPU 显存释放）
- **验证**: agent server 单独 `torchrun --nproc_per_node=4 openai_tp_server.py` 启动正常，/health 和 /v1/completions 均返回正确响应
- **关联 Phase**: Phase 10 (OpenAI server)
