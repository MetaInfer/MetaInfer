# meta-infer 推理引擎优化 — Claude Code 工作指南

## 1. 项目目标

将自研推理引擎 `meta-infer` 的吞吐率从 baseline 2.15 tok/s 优化至接近 vLLM 水平。
当前已完成 5.93x 提升（2.15 → 12.75 tok/s），正在进行 Flash Attention 集成。

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

## 5. 当前待解决问题：P3-FA Flash Attention

### 5.1 目标

将 decode 路径的 `F.scaled_dot_product_attention` 替换为 `flash_attn_varlen_func`，
消除 `attn_mask` 分配开销，利用 flash kernel 的 IO-aware tiling 加速。

### 5.2 当前实现状态

**Qwen (已完成，无回退)**:
- Prefill + Decode 均使用 `flash_attn_varlen_func` 直接调用
- 3D ragged 格式 `[total_tokens, nheads, headdim]`，无需 permute
- 代码位置: `engine/models/qwen.py` `QwenAttentionTP.forward()`

**DeepSeek-V2 (P3-Triton MLA Decode)**:
- Prefill: FA2 + V-padding（不变）
- Decode: Triton MLA kernel — `engine/kernels/triton_mla_decode.py`
- 统一 KV cache: `[B, max_seq_len, 1, kv_lora_rank + rope_dim]` 存储 `[c_kv | k_pe_rope]`
- Q 投影: `q_nope @ W_UK_T → q_nope_proj` (潜空间)
- 输出扩展: `out_latent @ W_UV → out` (v_head_dim 空间)
- Scaling: `1/sqrt(qk_head_dim) = 1/sqrt(192)` (与 vLLM 一致)
- k_pe 在存入 cache 前应用 RoPE（关键修复）
- 代码位置: `engine/models/deepseek_v2.py` `DeepseekAttentionTP.forward()`

### 5.3 尝试过的方案与回退原因

| 方案 | DeepSeek 策略 | 结果 | 回退原因 |
|------|-------------|------|---------|
| A | 切片 KV + SDPA 无 mask | -20% | torch.compile 动态 shape 重编译 |
| B | 切片 KV + SDPA + 去 attn compile | -12% | 切片开销 > 消除 attn_mask 收益 |
| C (当前) | V-padding + flash_attn_varlen_func | -15% | F.pad/unpad 开销 + full buffer 浪费 |

### 5.4 关键技术约束

- **GPU**: A800 (SM80, Ampere) → 仅支持 FA2（FA3 需 SM90 Hopper）
- **FA2 最大 headdim**: 256
- **DeepSeek-V2-Lite MLA**: QK headdim=192 (< 256), V headdim=128, K/V 不同需 padding
- **DeepSeek-V2 全量版**: QK headdim=576 (> 256), 需 FA3/FlashMLA（A800 不支持）
- **torch.compile**: `mode='default'`（kernel fusion only），支持动态 shape 但会重编译

### 5.5 待验证

1. **V-padding 性能**: 之前测试 GPU 只有 6-20GB 空闲，无法公平对比。需要 GPU 有 30GB+ 空闲时重新测试
2. **vLLM 对比**: `run_compare_metainfer_vllm.sh` 已添加 `VLLM_GPU_MEM_UTIL` 参数，可设置 `VLLM_GPU_MEM_UTIL=0.15` 避免 OOM
3. **其他优化方向**: 如果 V-padding 仍有回退，可考虑：
   - 自定义 Triton kernel（跳过 padding 计算）
   - 使用 PyTorch SDPA 但预分配 attn_mask 缓冲区（避免每步分配）
   - 等 FA3 支持 SM80 或使用 FlashInfer

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

1. 等 GPU 4-7 有 30GB+ 空闲时，跑 V-padding 性能基准
2. 如果 V-padding 仍有回退，考虑：
   - 预分配 attn_mask 缓冲区（避免每步分配开销）
   - 自定义 Triton kernel
   - 等 FlashInfer 支持
3. P4 Continuous Batching 代码已提交（commit f1ff0a3），可随时激活
