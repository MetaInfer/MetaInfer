# QwenPaw-Flash-2B 推理引擎开发日志

## 目标

为 `agentscope-ai/QwenPaw-Flash-2B` (Qwen3.5 Dense 2.72B) 构建高性能本地推理引擎。

## 模型架构

- Qwen3.5 Dense：24 层（18 层 Gated DeltaNet 线性注意力 + 6 层标准 softmax 注意力）
- **无 MoE**：Dense MLP，intermediate_size=6144
- hidden_size=2048, head_dim=256, partial_rotary_factor=0.25
- GQA: full_attn 8 heads / 2 kv_heads; linear_attn 16 key_heads / 16 value_heads
- RMSNorm：`input_layernorm` 使用 `(1 + weight)` 格式，`linear_attn.norm` (RMSNormGated) 使用 `weight * x` 格式（不加 1）
- Full attention 层有 sigmoid 输出门控
- tie_word_embeddings=true（lm_head 共享 embed_tokens 权重）

## 已完成

### Step 1: 知识库建立

- `notebooks/02_model_specifics/04_qwen3_5_moe/01_gated_delta_net.md` — Gated DeltaNet 数学推导、递归更新规则、状态管理
- `notebooks/02_model_specifics/04_qwen3_5_moe/02_moe_routing.md` — MoE 路由细节、权重命名映射
- `notebooks/02_model_specifics/04_qwen3_5_moe/README.md` — 架构概述与规格表

### Step 2: 模型实现 (`engine/20260507/mac_qwen/model.py`)

| 组件 | 说明 |
|------|------|
| `Qwen35MoeConfig` + `load_config()` | 从 `config.json` 的 `text_config` 读取混合注意力配置，`is_moe` 属性区分 MoE/Dense |
| `RMSNorm` / `RMSNormGated` | Qwen3.5 专用归一化（1+weight 格式、silu 门控） |
| `apply_partial_rope()` | 只对 head_dim 的 25%（64 维）应用旋转位置编码 |
| `Qwen35MoeFullAttention` | sigmoid 输出门控 + GQA + SDPA + 预分配 KV cache |
| `Qwen35MoeGatedDeltaNet` | 因果卷积（含 SiLU 激活）+ QK L2 norm + Gated DeltaNet 递归状态管理 |
| `Qwen35MoeMLP` | Dense MLP（gate_proj + up_proj + down_proj，silu 激活） |
| `Qwen35MoeSparseMoeBlock` | MoE 专家路由（仅 `is_moe=True` 时使用） |
| `load_weights()` | 自动检测 `model.language_model.` / `model.` 前缀，支持单 shard 和多 shard，Dense/MoE 分支加载 |
| `Qwen35MoeModelRunner` | 管理每序列的 recurrent state 和 KV cache |
| `LayerState` + `create_layer_states()` | 混合注意力的状态分配（full: KV cache, linear: recurrent + conv） |

### Step 3: 引擎入口 (`engine/20260507/mac_qwen/engine.py`)

- 自包含推理引擎，不修改现有 `llm_engine.py`
- 支持 CUDA / MPS / CPU 自动选择
- Prefill + decode 循环，流式输出支持

### Step 4: 模型下载 (`engine/20260507/mac_qwen/downloader.py`)

- `huggingface_hub.snapshot_download` 断点续传
- 支持 `HF_ENDPOINT` 环境变量切换镜像
- 默认模型 ID：`agentscope-ai/QwenPaw-Flash-2B`

### Step 5: OpenAI 兼容 API 服务 (`engine/20260507/mac_qwen/openai_server.py`)

- `/v1/chat/completions` — 带 chat template 的聊天接口
- `/v1/completions` — 纯文本补全接口
- `/health`、`/v1/models` — 健康检查和模型列表
- 支持 streaming（SSE）和非 streaming 模式
- MPS 采样修复：logits 移至 CPU 后再做 multinomial 采样

### 验证

- ruff lint + format 全部通过
- 模块 `importlib` 导入正常（目录名含数字前缀需 `importlib.import_module`）
- 现有 11 个单元测试全部通过
- 端到端推理：HF transformers logits 对比 cosine similarity > 0.9999（4 个 prompt 全部 > 0.9999，top-1 token 完全匹配）
- OpenAI 接口冒烟测试 10/10 全部通过（temperature=0.6，输出质量优秀：中文诗歌、数学推理、角色扮演均正常）
- Decode 性能基准（MPS, float16）：~25 tok/s

### Bug 修复

1. **q_proj view 形状错误**: `attn_output_gate=True` 时 q_proj 输出 2x 维度，view 需要用 `head_dim * 2`
2. **kwargs 解包导致状态丢失**: `**kwargs` 解包创建新字典，layer 内修改不传回。改为直接传递 dict 引用
3. **torch.empty() 全零**: Mac 上 `torch.empty()` 产生全零张量，添加 `_init_weights()` 随机初始化
4. **Conv1d 缺少 SiLU 激活**: Gated DeltaNet 的因果卷积后需要 `F.silu()`，与 HF transformers 实现对齐
5. **QK L2 norm 缺失**: 线性注意力需要对 Q、K 做 L2 归一化（`F.normalize`），HF 使用 `use_qk_l2norm_in_kernel=True`
6. **Query 缺少 `1/sqrt(d_k)` 缩放**: 修正 Gated DeltaNet 输出幅度，与 HF `scale = 1 / (k_head_dim ** 0.5)` 对齐
7. **linear_attn 权重前缀错误**: safetensors 中线性注意力层使用 `linear_attn` 前缀而非 `self_attn`
8. **单 shard 加载失败**: 无 `model.safetensors.index.json` 时直接从 `model.safetensors` 构建 weight_map
9. **权重前缀检测不全**: `_detect_prefix` 增加 `model.` 前缀检测，`_w` 函数增加 fallback
10. **`top_p_sample` 返回排序索引而非 token ID**: `torch.multinomial` 返回 sorted 数组中的位置，需用 `sorted_indices.gather()` 映射回原始 token ID（`engine/sampler.py` 全局修复）
11. **`apply_chat_template` 返回 BatchEncoding**: 需用 `result["input_ids"]` 提取 token ID 列表
12. **Gated DeltaNet recurrent state float32 计算**: MPS float16 下 decode 超过 ~30 步后输出退化（重复 `**`、循环文本、乱码）。修复：`torch_recurrent_gated_delta_rule` 内部升级为 float32 计算，`create_layer_states` 中 linear attention 的 recurrent state 初始分配为 float32，输出前转回模型 dtype (fp16)。效果：长文本输入用例质量明显提升（从完全循环变成有意义的摘要），但其他中长输出用例仍然退化（空行重复、文本循环），说明还有其他精度瓶颈
13. **截断 thinking tokens 失败（负优化，已回退）**: 尝试在 chat template 末尾注入 `<think\>\n\n</think\>\n\n` 让模型跳过思考。失败原因：模型在 prompt 中看到 `<think\></think\>` 后会跳过思考直接回答，但去掉 prompt 中的 thinking tokens 后模型会自己生成 `<think\>` 进入 thinking mode，导致输出全是思考文本。结论：不应截断 thinking tokens，保留原始 chat template
14. **skip_thinking 过滤失败（负优化，已回退）**: 在 `_generate_tokens` 中添加 skip_thinking 过滤，等看到 `</think\>` token 后才开始输出。失败原因：(a) 模型在 prompt 中看到 `<think\></think\>` 时跳过思考直接回答，第一个 token 就不是 `</think\>`，导致 thinking_done 永远为 False；(b) 去掉 prompt 中的 thinking tokens 后模型会自己生成 thinking 内容，但 max_tokens 在 thinking 阶段也被消耗完，导致输出为空
15. **RMSNormGated 使用了错误的 weight 格式（关键修复）**: `RMSNormGated` 使用了 `(1 + weight)` 格式（与 layer RMSNorm 一致），但 HF transformers 中 `Qwen3_5RMSNormGated` 使用的是 `weight * x`（不加 1）。这导致 linear attention 层的输出幅度偏差约 2 倍（our norm=5.07 vs HF norm=3.17），进而导致所有后续层的 hidden states 发散。修复后 prefill logits cosine similarity 从 0.845 提升到 >0.9999，所有 4 个测试 prompt 的 top-1 token 完全匹配

## 常用命令

```bash
# 下载模型（通过环境变量设置镜像源）
HF_ENDPOINT=https://hf-mirror.com python engine/20260507/mac_qwen/download_model.py --local-dir ~/.cache/qwen35-moe

# 命令行推理
uv run python engine/20260507/mac_qwen/engine.py --prompt "你好" --model-dir ~/.cache/modelscope/hub/models/AgentScope/QwenPaw-Flash-2B --stream

# OpenAI 兼容 API 服务
uv run python -m engine.20260507.mac_qwen.openai_server \
  --model-dir ~/.cache/modelscope/hub/models/AgentScope/QwenPaw-Flash-2B \
  --port 8000

# 通过冒烟测试
pytest tests/test_openai_smoke.py -v

# Lint & Format
ruff check . --fix && ruff format .

# 单元测试
pytest tests/test_scheduler.py tests/test_sequence.py tests/test_memory.py tests/test_prefix_cache.py -v
```

## 待完成

### Step 6: 正确性验证

- [x] 对比 HF transformers 多 prompt 的 logits（余弦相似度 > 0.9999，4 个 prompt 全部通过）
- [x] 验证 decode 多步后 recurrent state 精度不退化（50 步 greedy decode，cosine > 0.9999，top-1 100% 匹配）
- [x] 逐层 hidden norm 对比（24 层全部 cos > 0.9999，norm ratio = 1.000，与 HF 完全一致）
- [x] hidden_norm "发散" 问题已确认是 RMSNormGated bug 导致，修复后不存在发散。norm 从 2.53→129.61 (x51.3) 是模型正常行为（HF 也完全一致）

### Step 7: 精度优化（已完成）

- [x] **Gated DeltaNet recurrent state float32 计算**：`torch_recurrent_gated_delta_rule` 内部已升级为 float32，recurrent state 初始分配为 float32
- [x] **Chunk parallel prefill kernel**：实现了 `torch_chunk_gated_delta_rule`（基于 HF transformers 参考实现）
- [x] **RMSNormGated weight 格式修复**：修复后 prefill logits cosine similarity 从 0.845 → >0.9999
- [x] **验证 decode 多步精度**：CPU float32 下 50 步 greedy decode，cosine > 0.9999，top-1 100%
- [x] **冒烟测试质量验证**：10/10 通过，输出质量优秀（诗歌、数学、角色扮演均正常）

### Step 8: 性能优化

- [x] **MLP gate_up_proj 合并**：将独立的 `gate_proj` + `up_proj` 合并为 `gate_up_proj`，减少 decode 路径的矩阵乘法次数。权重加载时 `torch.cat([gate_w, up_w], dim=0)` 合并
- [x] **Linear attention fused in_proj**：将 `in_proj_qkv` + `in_proj_z` + `in_proj_b` + `in_proj_a` 四个独立线性层合并为单个 `in_proj`（2048→8224），减少 decode 路径 4 次 matmul → 1 次。权重加载时 `torch.cat([qkv_w, z_w, b_w, a_w], dim=0)` 合并
- [x] **Recurrent kernel bmm 优化**: 用 `torch.bmm` 替代 Gated DeltaNet recurrent kernel 中的手动 broadcast+sum，减少中间张量分配。decode 速度从 23.2 → 27.8 tok/s（+20%）
- [x] **Single-token decode fast path**: 特化 S=1 的 recurrent kernel（`torch_recurrent_gated_delta_rule_single`），去掉循环和多余 indexing/reshape
- [x] **Full attention fused QKV projection**: 将 q_proj + k_proj + v_proj 合并为单个 `qkv_proj`（2048→9216），减少 decode 路径 matmul 次数
- [x] **RoPE inv_freq 预计算**: `_precompute_rope_inv_freq` 预计算 RoPE 逆频率，避免每层每次 decode 重复计算 `arange` 和 `pow`
- [x] **Decode 循环状态管理优化**: `openai_server.py` decode 循环中 kw_list 只构建一次，每步 in-place 更新 position_ids 和 layer state，避免 `states_to_kwargs` 的重复调用
- [x] **A_log exp 缓存**: `Qwen35MoeGatedDeltaNet` 中缓存 `exp(A_log)` 避免每步重复 exp 计算
- [x] **top-p 采样 top-k 预筛选**: `engine/sampler.py` 中 `top_p_sample` 先取 top-1024 候选再做 top-p 排序，避免对 248320 vocab 全量排序。采样时间从 12.68ms → 0.88ms（-93%），decode 速度从 28 → 45 tok/s（+64%）。这是本轮最大性能提升
- [x] **性能基准**（MPS, float16）：decode ~45 tok/s, prefill 512 tokens ~0.33s (1548 tok/s)
  - Model forward: 96.0% | Sampling: 4.0%
- [x] **Usage token 计数优化**: `_generate_tokens` 直接跟踪 output token 数，避免 `_batch_chat`/`_batch_compl` 重新 encode 输出文本计算 token 数。yield 改为三元组 `(text, reason, token_count)`
- [ ] **`torch.compile` decode 路径**：编译 decode 路径（单 token）的模型前向来消除 Python 开销
  - 尝试 aot_eager 和 inductor 两个 backend，均失败
  - aot_eager: (a) full attention 层的 KV cache in-place 赋值 `key_cache[:, cache_len:cache_len+S] = k` 不兼容图捕获；(b) linear attention 的 chunk kernel 中 transpose/contiguous 在 MPS 上触发 weakref 错误
  - inductor: MPS 后端 PythonDispatcher dispatch 错误
  - 结论：PyTorch 2.11 的 `torch.compile` 对 MPS 后端支持不够成熟，暂不可用
- [ ] **多序列 batch decode**：支持多请求并行推理
- [ ] **MPS 特定优化**：验证 `torch.backends.mps` 路径，优化 memory pool

## 失败经验总结

### Thinking tokens 处理（Bug #13, #14）

QwenPaw-Flash-2B 的 chat template 包含 `<think\>` 相关 token。尝试操控这些 token 来跳过模型思考阶段均失败：

1. **截断 thinking tokens**: 模型会自行生成 `<think\>` 进入 thinking mode，截断无效
2. **skip_thinking 过滤**: 过滤 `</think\>` 之前的 token，但模型可能直接跳过思考（第一个 token 不是 `</think\>`），导致永远等不到标记而输出为空

**结论**: 保留原始 chat template，不做任何 thinking token 相关的特殊处理。模型的 thinking 行为由 chat template 和采样参数（temperature、top_p）自然控制。

### RoPE 预计算表在 MPS 上负优化

CPU 上 RoPE cos/sin 预计算表查找比直接计算快 107x，但在 MPS 上 fancy indexing (`tensor[:, pos_tensor]`) 的 kernel launch 开销抵消了计算节省，实测 decode 速度从 43.4 → 38.5 tok/s（-11%）。已回退，保留直接计算 + inv_freq 预缓存的方式。

**原因**: MPS 后端对 fancy indexing 的实现效率低，每次索引操作都需要启动新的 Metal kernel，而直接计算（element-wise 乘法 + cos/sin）可以被 MPS 的 compute graph 优化合并。

**结论**: 在 MPS 上避免使用 tensor indexing，优先使用 element-wise 操作和 compute-bound 的数学运算。
