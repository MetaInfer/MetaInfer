# meta-infer Stage 1 优化总结

## 测试环境

| 项目 | 配置 |
|------|------|
| 模型 | DeepSeek-V2-Lite-Chat (~16B MoE) / Qwen3-8B |
| GPU | NVIDIA A800 80GB PCIe × 4, TP=4 |
| 压测参数 | ROUNDS=5, STEPS=8, REQUEST_RATE=4, MAX_CONCURRENCY=1 |

---

## 已完成优化一览

| Phase | 模型 | 吞吐 (tok/s) | 变化 | 说明 |
|-------|------|-------------|------|------|
| P7a | DeepSeek | 8.49→8.87 | +4.5% | TP all_reduce 去 bf16→fp32→bf16 dtype 转换 |
| P3-Triton | DeepSeek | 12.75→13.08 | +2.6% | Triton MLA decode kernel 替换 FA2 V-padding |
| P5a | Qwen3 | 12.58→12.76 | +1.4% | gate_up_proj 合并为单次 GEMM |
| P5b | DeepSeek | 13.08→13.20 | +0.9% | MoE GPU-side expert_map hybrid |
| P6 | DeepSeek | 13.20→13.42 | +1.7% | 预分配 position 索引 buffer |
| P6 | Qwen3 | 12.76→13.03 | +2.1% | 同上 |

## DeepSeek-V2-Lite 全链路

```
2.15 tok/s (baseline, 全量重算)
    ↓ P0: 增量 KV Cache  (8.49, +3.95x)
    ↓ P2: torch.compile   (12.75, +45.5%)
    ↓ P3-Triton: MLA kernel (13.08, +2.6%)
    ↓ P5b: MoE GPU map    (13.20, +0.9%)
    ↓ P6: position buffer (13.42, +1.7%)

最终: 2.15 → 12.57 tok/s (5.85x) — 波动范围内与峰值 13.42 一致
vLLM 基准: 36.94 tok/s (差距 2.94x)
```

## Qwen3-8B 全链路

| 阶段 | 吞吐 (tok/s) | 说明 |
|------|-------------|------|
| P0 (KV Cache) | **23.51** | SDPA slice KV, 无 compile |
| P2 (torch.compile) | 5.95 | compile + full buffer + attn_mask — 严重回退 |
| P5a (flash_attn+fused MLP) | 16.57 | flash_attn_varlen_func |
| P6 (+position buf) | 16.45 | 波动范围内 |
| **decode→SDPA, 去 P5a/P6/compile** | **22.25** | 接近 P0 峰值 |
| **vLLM 基准** | **36.92** | — |

**结论**: Qwen3-8B decode 最优路径是 slice KV + permute + SDPA auto-dispatch，无 torch.compile。FA2 full buffer 方案反而是负优化（-30%）。当前回退后恢复到 22.25 tok/s。

---

## 代码文件清单

| 文件 | Phase | 变更 |
|------|-------|------|
| `engine/kernels/triton_mla_decode.py` | P3-Triton | 新增 Triton MLA decode kernel |
| `engine/models/deepseek_v2.py` | P3-Triton, P6 | MLA 权重提取, 统一 KV cache, position buffer |
| `engine/models/qwen.py` | P5a, P6 | gate_up_proj 合并 GEMM, position buffer |
| `engine/tp_layers/linear.py` | P5a | 新增 MergedColumnParallelLinear |
| `engine/tp_layers/moe.py` | P5b | GPU-side expert_map + hybrid forward |
| `engine/tp_layers/distributed.py` | P7a | all_reduce 去 dtype 转换 |

---

## 阻塞/延后的优化

| Phase | 状态 | 阻塞原因 |
|-------|------|---------|
| P2 (CUDA Graph) | 阻塞 | Qwen: tensor 索引 KV 写 vs Python slice 数值发散。DeepSeek: MoE `.item()` + Triton 动态分配 |
| 完整 Fused MoE | 延后 | batch=1 时 nonzero+index_add_ 开销 > `.item()`，需 batch>1 |

## 下一优先级

1. **P7b Custom AllReduce** — profiling 显示 NCCL AllReduce 占 GPU 61.2%，用 P2P 内存映射替代 ring allreduce，预计 +15-25%
2. **P4 Continuous Batching** — 解除 batch=1 约束，使 P5b 的 batched MoE 路径生效
