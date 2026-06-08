# Phase 7-8：权重加载 + 框架外壳

## 触发词

`/phase7-8`

## 角色

你是主 Agent。按 CLAUDE.md 的 spawn 协议，依次完成 Phase 7 → Phase 8，每 Phase 独立走完整串行路径。

---

## Phase 7：权重加载

### 任务

- QwenTPConfig dataclass：从 `config.json` 动态读取所有字段（**严禁硬编码**）
- QwenForCausalLMTP.__init__：embed_tokens → layers[QwenDecoderLayerTP]×36 → norm → lm_head
- HF key mapping：QKV cat 顺序 **Q-K-V**（严禁 K-Q-V）；Gate-Up cat 顺序 **gate-up**
- `load_weights()`：safetensors.index.json → safe_open → double_shard_guard → barrier() + init_custom_ar()
- 每 rank 显存 < 全量模型/N

### Phase 7 Scripts

`test_phase7_qwen_tp_config.py` + `test_phase7_hf_key_mapping.py` + `test_phase7_weight_loading.sh`

### 知识映射

- Blueprint：`class_hierarchy.QwenTPConfig` → `class_hierarchy.QwenForCausalLMTP` → `construction_chain` → `qwen_hf_key_mapping`（12 个 HF key→attr 映射）→ `qwen3_8b_model_dims`
- ref_docs：`kernel_replacement_plan.md` §四

### 高发错误

1. **QKV cat 顺序 K-Q-V** → 必须 Q-K-V
2. **double_shard_guard 失效** → 预切片权重被二次切片
3. **每卡加载全量** → OOM
4. **q_norm/k_norm weight dispatch 遗漏** → logits 全错但不 crash（Bug 6）

---

## Phase 8：框架外壳

### 任务

Phase 8 组件与模型层物理独立，可从 nano-vllm 抄入但须按重载清单修改：

- Sequence：input_ids/output_ids/block_table(list+Tensor 双轨)/status(WAITING→PREFILL→DECODE→FINISHED/REJECTED)
- Scheduler：schedule(num_free) 先 prefill 后 decode + REJECTED。**删除 preempt() 逻辑**
- Sampler：temperature=0.0 greedy + top_p。TP 协议：rank 0 采样 + dist.broadcast
- BlockManager：TP 降级 no-op。allocate/free 空方法

### Phase 8 Scripts

`test_phase8_sequence_scheduler.py` + `test_phase8_sampler_tp.py`

### 知识映射

- Blueprint：`components[0] Scheduler` → `components[4] Sampler` → `components[5] Sequence` → `components[2] BlockManager`
- ref_docs：`02_scheduler.md`, `03_kv_cache.md`, `05_sampler.md`, `01_architecture.md`, `07_request_lifecycle.md`
- ref_code：`ref_projects/nano-vllm/nanovllm/engine/`（⚠️ 删除 preempt L66-69）

### 高发错误

1. **preempt() 未删除** → running.pop() 序列丢失
2. **block_size 硬编码 16** → TP 路径未注入 256
3. **TP 各 rank 独立采样** → KV 不同步 → NCCL 崩溃

---

## 关键约束

- 每个 Phase 独立走完整 impl→spec→verify 流程
- Phase 7 verif L2：重跑 Phase 1-6 全部脚本
- Phase 8 verif L2：重跑 Phase 1-7 全部脚本
