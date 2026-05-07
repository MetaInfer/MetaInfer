# QwenPaw-Flash-2B 推理引擎开发日志

## 目标

为 `agentscope-ai/QwenPaw-Flash-2B` (Qwen3.5 Dense 2.72B) 构建高性能本地推理引擎。

## 模型架构

- Qwen3.5 Dense：24 层（18 层 Gated DeltaNet 线性注意力 + 6 层标准 softmax 注意力）
- **无 MoE**：Dense MLP，intermediate_size=6144
- hidden_size=2048, head_dim=256, partial_rotary_factor=0.25
- GQA: full_attn 8 heads / 2 kv_heads; linear_attn 16 key_heads / 16 value_heads
- RMSNorm 使用 `(1 + weight)` 格式
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
- 端到端推理：HF transformers logits 对比 cosine similarity > 0.99
- OpenAI 接口冒烟测试通过（temperature=0.6，输出连贯中文）

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

- 对比 HF transformers 多 prompt 的 logits（余弦相似度 > 0.999）
- 验证 decode 多步后 recurrent state 精度不退化

### Step 7: 精度优化（进行中）

- [x] **Gated DeltaNet recurrent state float32 计算**：`torch_recurrent_gated_delta_rule` 内部已升级为 float32，recurrent state 初始分配为 float32。长文本用例改善明显，但中长输出仍有退化（空行重复、文本循环）。**仍需进一步排查其他精度瓶颈**
- [ ] 对比 HF transformers 多 prompt 的 logits（余弦相似度 > 0.999）
- [ ] **排查剩余精度退化**：float32 修复仅改善了部分用例，退化模式从 `**` 重复变为空行重复和文本循环。可能的瓶颈方向：(a) KV cache（softmax attention 层）的 fp16 累积误差；(b) MLP 层中间激活的精度损失；(c) MPS 后端特定算子的数值精度问题。需要逐层对比 HF 输出定位

### Step 8: 性能优化

- [ ] Gated DeltaNet prefill 路径：实现 chunk parallel 计算替代逐 token 循环
- [ ] MoE expert batch：token 路由到同一 expert 时合并 matmul（如切换回 MoE 模型）
- [ ] `torch.compile` 编译 MoE routing + expert 计算图
- [ ] 多序列 batch decode
- [ ] MPS 特定优化（`torch.backends.mps` 路径验证）

## 失败经验总结

### Thinking tokens 处理（Bug #13, #14）

QwenPaw-Flash-2B 的 chat template 包含 `<think\>` 相关 token。尝试操控这些 token 来跳过模型思考阶段均失败：

1. **截断 thinking tokens**: 模型会自行生成 `<think\>` 进入 thinking mode，截断无效
2. **skip_thinking 过滤**: 过滤 `</think\>` 之前的 token，但模型可能直接跳过思考（第一个 token 不是 `</think\>`），导致永远等不到标记而输出为空

**结论**: 保留原始 chat template，不做任何 thinking token 相关的特殊处理。模型的 thinking 行为由 chat template 和采样参数（temperature、top_p）自然控制。
