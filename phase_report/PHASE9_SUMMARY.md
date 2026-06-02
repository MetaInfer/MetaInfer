# Phase 9 Summary — 引擎集成

## 创建文件

| 文件 | 内容 |
|------|------|
| `llm_engine.py` | LLMEngine(7-step __init__ + generate + step + begin/has_unfinished/get_outputs) + QwenTPModelRunner |
| `engine/memory_pool.py` | KVMemoryPool + estimate_num_blocks_dense |

## 修改文件

`engine/models/qwen.py`:
- RMSNorm.forward: 返回 2-tuple（修复 tuple unpack crash）
- QwenAttentionTP: q_norm/k_norm 改为 tuple unpack
- QwenForCausalLMTP: `to(device, dtype=bf16)`（修复 float32→bf16 显存问题）
- QwenForCausalLMTP.forward_decode: 新增 decode 路径
- `_dispatch_weight`: 新增 q_norm/k_norm 权重加载（修复模型输出错误）
- flash_attn_with_kvcache: 改用 keyword args（修复 flash_attn 2.8.3 API 兼容）

`engine/structs.py`:
- input_ids_tensor: 添加 device 参数（修复 CPU/CUDA mismatch）

## Bug 修复记录（共 6 个）

| # | Bug | 症状 | 修复 |
|---|-----|------|------|
| 1 | float32 全量模型 | rank0 31661 MB | to(dtype=bf16) |
| 2 | init_tp_distributed 无 guard | 单进程 hang | WORLD_SIZE 检查 |
| 3 | input_ids_tensor 无 device | CPU/CUDA RuntimeError | device 参数 |
| 4 | RMSNorm 返回单 Tensor | tuple unpack ValueError | 返回 (out, None) 或 (x, residual) |
| 5 | flash_attn 2.8.3 API 签名 | positional arg 错位 | keyword args |
| 6 | **q_norm/k_norm 未加载** | generate() 输出垃圾 | 2 行 copy_ 添加到 dispatch |

## 审查结果

### spec-reviewer — ✅ PASS (after 1 fix)
### verification — ✅ PASS
| Level | Result |
|-------|--------|
| L0 | ✅ PASS |
| L1 | ✅ 2/2 PASS (generate 输出与预期字字对齐) |
| L2 | ✅ 22/22 PASS (Phase 1-8 零回归) |

## 步骤 3.5 抽查
- test_phase9_llm_engine_init.py: ALL 4 TESTS PASSED ✅
- generate() output: `（ ） A：建筑与园林结合...` Match: True ✅

## 判定
```
Phase 9 交付完成 ✅
```
