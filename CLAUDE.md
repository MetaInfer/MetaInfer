# meta-infer 推理引擎优化 — Claude Code 工作指南

## 1. 项目目标

将自研推理引擎 `meta-infer` 的吞吐率从 baseline 2.15 tok/s 优化至接近 vLLM 水平。
当前已实现 Qwen3-8B TP=4 nocompile **55.7 tok/s**（~26x vs baseline），与 vLLM CUDA Graph（166.8 tok/s）差距来自 CUDA Graph 未在 TP=4 启用。

## 2. 优化阶段总览

| 阶段 | 内容 | 状态 |
|------|------|------|
| Stage 1-7 (kernel_replacement_plan.md) | 7 个 vLLM 黑盒 kernel 替换 | ✅ 全部完成 |
| Stage A (cuda_graph_plan.md) | 伪图代码清理、CustomAR 接口恢复 | ✅ 已完成 |
| Stage B | CUDAGraphWrapper + torch.compile 单 GPU | ✅ 已完成 |
| Stage C | 单 GPU 10K replay 压测 + clone 回归修复 | ✅ 已完成 |
| Stage D (阶段三) | TP=4 CUDA Graph（all_reduce_sum 入图） | 🔴 待实施 |
| P4 Continuous Batching | 代码已提交，未激活 | ⏸ 暂停 |

## 2. 代码位置

- **根目录**: `/home/honglin/meta-infer`
- **模型文件**:
  - `engine/models/qwen.py` — Qwen3 模型（GQA attention）
  - `engine/models/deepseek_v2.py` — DeepSeek-V2 模型（MLA attention）
  - `engine/scheduler.py` — 调度器
  - `engine/memory_pool.py` — KV 缓存池
  - `engine/tp_layers/` — Tensor Parallel 层
  - `llm_engine.py` — 推理引擎主循环
  - `openai_tp_server.py` — OpenAI 兼容 HTTP 服务
- **文档**:
  - `notebooks-cn/07_improvementPlan/improvement_plan.md` — 完整优化计划与结果
  - `notebooks-cn/07_improvementPlan/.checkpoint` — 当前进度 checkpoint
  - `notebooks-cn/07_improvementPlan/flashattn.md` — Flash Attention 集成详细方案
- **模型权重**:
  - `/home/honglin/models/qwen/Qwen3-8B`
  - `/home/honglin/models/deepseek-ai/DeepSeek-V2-Lite-Chat`
- **参考工程**: `ref_projects/vllm/` — vLLM 0.15.1 源码

## 3. Git 状态

- **分支**: `feature/tp-implementation`
- **最新 commit**: `6f061a2` — P3-FA: DeepSeek V-padding + flash_attn_varlen_func
- **关键历史 commits**:
  - `bb7689d` — P3-FA 初始版本（Qwen flash_attn + DeepSeek sliced SDPA）
  - `a339817` — P2: torch.compile kernel fusion + 固定形状 attention
  - `e887856` — P3: enable_gqa=True
  - `f1ff0a3` — P4: Continuous Batching（已提交但未激活）

## 4. 已完成的优化阶段

| Phase | 内容 | 吞吐 (tok/s) | 状态 |
|-------|------|-------------|------|
| Baseline | 全量重算，无 KV cache | 2.15 | - |
| P0 | 增量 KV Cache 解码 | 8.49 | 完成 |
| P1 | Batched MoE | - | 回滚（batch=1 无收益） |
| P2 | torch.compile + 固定形状 attention | 12.75 | 完成 |
| P3 | SDPA enable_gqa=True | - | 完成 |
| P3-FA | Flash Attention 集成 | - | 正确性通过，性能有回退 |
| P3-Triton | Triton MLA Decode Kernel | 13.08 | 正确性通过，+2.6% vs P2 |
| P4 | Continuous Batching | - | 代码已提交，未激活 |
| P5 | TP 通信优化 | 8.87 | 完成 |

## 5. 已解决：P3-FA Flash Attention

Qwen3 的 Flash Attention 集成已完成——prefill 使用 `flash_attn_varlen_func`，decode 使用 `flash_attn_with_kvcache`（paged attention, block_size=256）。DeepSeek-V2 使用 Triton MLA decode kernel。

## 6. 测试环境与脚本

### 6.1 Conda 环境

- **环境名**: `meta`
- **路径**: `/home/honglin/miniconda3/envs/meta/`
- **关键包**: flash_attn 2.8.3, vLLM 0.15.1, PyTorch 2.9.1+cu128
- **激活方式**: `eval "$(/home/honglin/miniconda3/bin/conda shell.bash hook)" && conda activate meta`
- **或者直接用 PATH**: `export PATH=/home/honglin/miniconda3/envs/meta/bin:$PATH`

### 6.2 正确性测试

```bash
# Qwen3-8B 单进程测试
export PATH=/home/honglin/miniconda3/envs/meta/bin:$PATH
PYTHONPATH=/home/honglin/meta-infer:$PYTHONPATH CUDA_VISIBLE_DEVICES=0 python -c "
import os; os.environ['META_INFER_LOG_RANK0_ONLY'] = '1'
from llm_engine import LLMEngine; from pathlib import Path
engine = LLMEngine(model_dir=Path('/home/honglin/models/qwen/Qwen3-8B'), inference_backend='qwen_tp', max_num_seqs=4)
out = engine.generate('苏州园林的特点是', max_new_tokens=24, temperature=0.0)
print(f'Qwen: {out!r}')
"
# 预期输出: '（ ） A：建筑与园林结合 B：建筑与自然结合 C：建筑与山水结合 D：建筑'

# DeepSeek-V2-Lite TP=4 测试
cat > /tmp/test_deepseek.py << 'EOF'
import os
os.environ['META_INFER_LOG_RANK0_ONLY'] = '1'
from llm_engine import LLMEngine
from pathlib import Path
engine = LLMEngine(model_dir=Path('/home/honglin/models/deepseek-ai/DeepSeek-V2-Lite-Chat'), inference_backend='deepseek_tp', max_num_seqs=4)
out = engine.generate('苏州园林的特点是', max_new_tokens=24, temperature=0.0)
if int(os.environ.get('RANK', '0')) == 0:
    print(f'DeepSeek: {out!r}')
EOF
PYTHONPATH=/home/honglin/meta-infer:$PYTHONPATH CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 /tmp/test_deepseek.py
# 预期输出: '怎样的？\n\n苏州园林的特点是怎样的？\n\n苏州园林是中国传统园林的代表之一，以其精巧的'
```

### 6.3 性能基准测试

```bash
export PATH=/home/honglin/miniconda3/envs/meta/bin:$PATH

# DeepSeek-V2-Lite TP=4 基准测试（meta-infer only）
SKIP_VLLM=1 CUDA_VISIBLE_DEVICES=4,5,6,7 TP_SIZE=4 ROUNDS=10 STEPS=8 REQUEST_RATE=4 MAX_CONCURRENCY=1 \
  MODEL_DIR=/home/honglin/models/deepseek-ai/DeepSeek-V2-Lite-Chat \
  bash meta-infer/run_compare_metainfer_vllm.sh dsv2

# 对比 vLLM（gpu_memory_utilization=0.15 避免 OOM）
CUDA_VISIBLE_DEVICES=4,5,6,7 TP_SIZE=4 ROUNDS=10 STEPS=8 REQUEST_RATE=4 MAX_CONCURRENCY=1 \
  MODEL_DIR=/home/honglin/models/deepseek-ai/DeepSeek-V2-Lite-Chat \
  VLLM_GPU_MEM_UTIL=0.15 \
  bash meta-infer/run_compare_metainfer_vllm.sh dsv2

# Qwen3-8B 测试
SKIP_VLLM=1 CUDA_VISIBLE_DEVICES=4,5,6,7 TP_SIZE=4 ROUNDS=10 STEPS=8 REQUEST_RATE=4 MAX_CONCURRENCY=1 \
  MODEL_DIR=/home/honglin/models/qwen/Qwen3-8B \
  bash meta-infer/run_compare_metainfer_vllm.sh qwen
```

### 6.4 关键指标

- **Output throughput (tok/s)** — 主要衡量标准
- **Mean TTFT (ms)** — 首 token 延迟
- **Duration (s)** — 总耗时
- **correct_rate** — 输出正确率（应为 0.0，因为使用中文 prompt 不在标准测试集中）

## 7. 代码改动记录

### 7.1 engine/models/qwen.py

- 添加 `from flash_attn import flash_attn_varlen_func`
- Prefill: `flash_attn_varlen_func(q_fa, k_fa, v_fa, cu, cu, seqlen, kv_len, causal=True)`
- Decode: `flash_attn_varlen_func(q_fa, k_fa, v_fa, cu_q, cu_k, seqlen, kv_len, causal=False)`
- 消除 `.permute(0,2,1,3)` 和 `attn_mask`

### 7.2 engine/models/deepseek_v2.py

- 添加 `from flash_attn import flash_attn_varlen_func`
- Prefill: `flash_attn_varlen_func` + V-padding + causal=True
- Decode: full buffer + `flash_attn_varlen_func` + `cu_seqlens_k` + V-padding
- V-padding: `F.pad(v_buf[0], [0, qk_head_dim - v_head_dim])`
- 输出 unpad: `out[:, :, :v_head_dim]`
- 保持 `torch.compile` on attention and MLP

### 7.3 run_compare_metainfer_vllm.sh

- 添加 `VLLM_GPU_MEM_UTIL` 环境变量（默认 0.9）
- vllm serve 命令添加 `--gpu-memory-utilization "$VLLM_GPU_MEM_UTIL"`

## 8. 工作流程约定

1. **Bash 命令必须先 cd**: `cd /home/honglin/meta-infer &&` 或使用 `bash meta-infer/...`
2. **Python 环境**: 使用 meta conda 环境（有 flash_attn 2.8.3）
3. **GPU 选择**: 优先使用 GPU 4-7（通常有更多空闲显存），TP=4
4. **模型**: DeepSeek-V2-Lite-Chat 为主要测试模型，Qwen3-8B 为辅助验证
5. **Commit 规范**: 每次改动后更新 `improvement_plan.md` 和 `.checkpoint`，然后 git commit
6. **不要杀死其他用户进程**: 检查 `nvidia-smi` 选择空闲 GPU

## 9. 下一步行动

1. **阶段三**: TP=4 CUDA Graph——引入 sglang 切图方案（在 all_reduce_sum 处拆分 FX 图，通信 eager 执行不入图），见 `cuda_graph_plan.md` §四
2. 单 GPU CUDA Graph 已可用（`META_INFER_CUDA_GRAPH=1`），当前 35.4 tok/s（reduce-overhead 方案）
3. 高并发场景需要 P4 Continuous Batching（代码已提交未激活）
