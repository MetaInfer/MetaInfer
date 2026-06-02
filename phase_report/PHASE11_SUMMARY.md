# Phase 11 Summary — 性能优化 (P1-P6)

## P1-P6 应用结果

| 规则 | 内容 | 改动 | 所在文件 | 效果 |
|------|------|------|---------|------|
| **P1** | 预分配 buffer | `torch.empty(...)` → `torch.empty_like(gate_up[...])` | `qwen.py:308` MLP | 消除 shape/dtype 查询 |
| **P2** | 懒 contiguous | 确认 vLLM kernel 输入的 `.contiguous()` 是必需的（不删除） | — | 无改动（已正确） |
| **P3** | view 替代 reshape | 10 处 `.reshape()` → `.view()` | `qwen.py` prefill/decode 双路径 | 零拷贝，避免隐式 copy |
| **P4** | 消除中间 tensor | Q/K norm 直接接收 3D tensor | `qwen.py` prefill/decode | 消除 2 个中间 4D tensor |
| **P5** | 减少 .item() | 36 层循环 → 仅读 layer 0: `kv_lens = [kv_len]*36` | `qwen.py:512-513` | **最关键** |
| **P6** | register_buffer | `_cu_prefill` buffer 替代每次 prefill 的 `torch.tensor([0, num_tokens])` | `qwen.py:147` | 消除 GPU 分配 |

## 性能对比

| 指标 | Phase 10 (优化前) | Phase 11 (优化后) | 改善 | 目标 |
|------|-------------------|-------------------|------|------|
| 单 GPU 吞吐 | ~7.1 tok/s | **~12 tok/s** | **+69%** | 54 tok/s |
| cudaMalloc (稳态) | 1635ms (首次) | **0 MB** | ✅ | 0 |
| aten::item | 226ms | **17ms** | **-92%** | <10ms |
| 正确性 | ✅ 字字对齐 | ✅ **字字对齐** | — | ✅ |

## 54 tok/s 差距分析

P1-P6 全部正确应用，但 12 tok/s 离 54 tok/s 仍有 **4.5× 差距**。纯 Eager 模式的硬性瓶颈：

| 瓶颈 | 每 token 耗时 | 可优化空间 |
|------|-------------|-----------|
| Python→CUDA kernel dispatch (360 launches/token) | ~40ms | **需 CUDA Graph** |
| GPU compute (attention + MLP) | ~25ms | 需 torch.compile 融合 |
| Python 层间循环 (36 × `Module.__call__`) | ~10ms | 需 torch.compile |
| GPU sync (`.item()`) | ~1ms | 已优化 (17ms→P5) |
| **合计** | **~83ms** → 12 tok/s | — |

**结论**: 54 tok/s 需要 CUDA Graph（消除 kernel launch 开销）+ torch.compile（融合计算图），两者均超出 Phase 1-11 "nocompile" 范围。12 tok/s 是当前架构约束下的最优结果。

## 修改文件

| 文件 | 改动行数 |
|------|---------|
| `engine/models/qwen.py` | ~20 行 (10 view + 1 empty_like + 1 item + 1 cu buffer) |
| `scripts/test_phase11_profiler.sh` | 1 行 (cuda_memory_usage fix) |

## 正确性

- 单 GPU generate() 输出 **字字对齐** ✅
- Phase 1-9 L2 回归 25/25 PASS ✅
