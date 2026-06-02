# 实验基线表

> 记录 mac-engine 推理引擎开发的实验过程与性能对比。

## 1. 硬件环境

| 项目 | 值 |
|------|-----|
| 芯片 | Apple M5 Pro |
| GPU 核心 | 20 |
| 统一内存 | 48 GB |
| 操作系统 | macOS 26.4.1 |
| 深度学习框架 | mlx=0.31.2, mlx-lm=0.31.3, vllm-metal=0.1.0, torch=2.12.0, transformers=5.8.0, modelscope=1.36.3 |

## 2. 基准实验 (Baseline: vllm-metal / MLX backend)

> 实验模型: Qwen3-8B (safetensors, ~15GB) — 与上游 CUDA 分支一致，方便横向对比。

### 2.1 单次推理

| 模型 | 输入长度 | 输出长度 | TTFT (ms) | TPOT (ms/tok) | 总耗时 (s) | 吞吐 (tok/s) | 内存 (GB) |
|------|---------|---------|-----------|---------------|-----------|-------------|-----------|
| Qwen3-8B | 11 | 256 | 237.8 | 55.6 | 14.42 | 17.8 | 7.3 |

### 2.2 并发压测

| 模型 | 并发数 | 请求数 | 总吞吐 (tok/s) | Mean TTFT (ms) | Mean TPOT (ms/tok) | 内存 (GB) |
|------|--------|--------|---------------|----------------|-------------------|-----------|
| Qwen3-8B | 8 | 8 | 14.7 | - | - | 7.6 |

> 注: vllm-metal server 当前版本为串行处理，并发场景下总吞吐低于单次推理 (竞争 MLX GPU 资源)。Mean TTFT/TPOT 因 server 不支持 streaming 无法直接测量。

## 3. 自研引擎对比 (目标: ≥70% baseline = 12.5 tok/s)

> **注意**: 引擎使用纯 `mlx.nn` 从头实现(不含 `mlx_lm`)。模型加载、架构、KV Cache、采样均为自主实现。
> 性能低于 mlx_lm 基线属正常现象，优化空间在后续 phase。

| 实验编号 | 日期 | Phase | 吞吐 (tok/s) | vs baseline | TTFT (ms) | TPOT (ms/tok) | 内存 (GB) | 正确性 | 备注 |
|---------|------|-------|-------------|-------------|-----------|---------------|-----------|--------|------|
| E01 | 0602 | Phase 0 | 4.9 | 27.5% | 130.7 | 205.2 | 4.6 | ✅ | 全量重算, 无 KV cache, 首 token="Machine" |
| E02 | 0602 | Phase 1 | 9.3 | 52.2% | 134.0 | 107.5 | 5.9 | ✅ | KV cache 增量解码, ~2x vs Phase 0 |
| E03 | 0602 | Phase 2 | 9.3 (s) / 9.3 (x4) | 52.2% | - | - | 3.1 | ✅ | 调度器 + round-robin, 并发与单次持平 |
| E04 | 0602 | Phase 1+ | 9.0 | 50.6% | 135.7 | 111.5 | 2.8 | ✅ | mx.compile sample + pre-alloc KVCache (中性, 瓶颈在 model forward) |
| E05 | 0602 | Phase 1+ | 9.0 | 50.6% | 254.7 | 110.1 | 2.7 | ✅ | mx.async_eval stream pipeline (中性, MLX 0.31.2 async_eval 无实际加速) |
| E06 | 0602 | Phase 1 | 18.0 | 101.1% | 155.1 | 55.1 | 13.4 | ✅ | **dtype 修复**: mx.load 原生 bf16 + decode 无 mask + 动态 KVCache, 超 mlx_lm 基线 |

> E06 根因分析: 之前所有实验 (E01-E05) 因 `weights.py` 中 bfloat16→float32 转换，模型以 float32 运行 (30.5 GB)。E06 修复三个问题:
> 1. `mx.load()` 直接加载 safetensors 保持原生 bf16 (15.3 GB)
> 2. decode 步 (L=1) 不创建 mask — 消除 float32 mask 与 bf16 attention 的 dtype 冲突
> 3. 移除 KVCache 的 float16 预分配 — 消除 bf16↔f16 转换开销
> 正确性验证: "The capital of France is" → " Paris", 首 token "Machine" 均与 golden 一致。

## 4. 正确性验证记录

| 验证编号 | 日期 | Golden 来源 | 测试用例数 | 通过数 | 状态 | 备注 |
|---------|------|-----------|----------|--------|------|------|
| V01 | 2026-06-02 | vllm-metal v0.1.0 (MLX backend) | 7 | 6 | ⚠️ 部分通过 | edge_empty 因 MLX 不支持空 prompt 失败 |
| V02 | 2026-06-02 | Phase 0/1 手动验证 | 1 | 1 | ✅ | "The capital of France is" → " Paris", 首 token 匹配 |

## 5. 优化方向记录

| 方向 | 优先级 | 预期收益 | 状态 |
|------|--------|---------|------|
| KV Cache 增量解码 | P0 | 2.0x (vs P0) | ✅ 已完成 (Phase 1, 9.3 tok/s) |
| 批量推理调度器 | P1 | 并发支持 | ✅ 已完成 (Phase 2, 9.3 tok/s @ x4) |
| mx.async_eval stream pipeline | P2 | 1.5-1.8x decode 提升 | ⚠️ 已尝试 (E05, 中性) — MLX 0.31.2 上 async_eval+stream 对自定义模型无提速 |
| mx.compile 包装 model forward | P2 | 1.5-2.0x decode 提升 | ⬜ 待开始 (需解决 mutable cache 兼容性) |
| bfloat16 dtype | P1 | 1.5-2.0x | ✅ 已完成 (E06, 18.0 tok/s = 101% baseline) — 根因: float32 权重 + f16 KVCache + f32 mask 三重 dtype 冲突 |
| 固定形状 KV Cache | P3 | 1.1-1.2x 提升 | ⬜ 待开始 |
| Triton/Metal kernel | P4 | 2-3x 提升 | ⬜ 待开始 (参考上游 CUDA 分支) |

---
日期: 2026-06-02

## 附录: 引擎代码与脚本

引擎源码: `subprojects/mac-engine/src/`
- `model.py` — Qwen3 模型架构 (纯 mlx.nn, 188 行)
- `kv_cache.py` — 自定义 KV Cache
- `weights.py` — safetensors 权重加载
- `tokenizer.py` — Tokenizer 封装 (transformers)
- `sampler.py` — Greedy/temperature 采样
- `engine_v0.py` — Phase 0: 全量重算, 4.9 tok/s
- `engine_v1.py` — Phase 1: KV cache, 9.3 tok/s
- `engine_v2.py` — Phase 2: 调度器 + 批量, 9.3 tok/s @ x4

基准测试: `subprojects/mac-engine/scripts/`
- `ref_bench_vllm_metal.py` — vllm-metal 参考基线
- `bench_engine.py` — 自研引擎 bench (--phase 0/1/2)
- `verify_correctness.py` — 正确性验证

Golden outputs: `subprojects/mac-engine/tests/golden_outputs/golden_outputs.json`
