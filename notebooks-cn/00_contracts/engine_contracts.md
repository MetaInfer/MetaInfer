# 引擎集成 — API 契约

> 关联 notebooks: `01_framework_design/01_architecture.md`, `01_framework_design/07_request_lifecycle.md`

## 概述

LLMEngine (7 步构造 + 5 步 generate while-loop) + ModelRunner (prefill/decode 分发)。
源实现文件: `llm_engine.py`

---

## LLMEngine.__init__ — 7 步构造序列

```python
class LLMEngine:
    def __init__(self, model_dir, inference_backend="qwen_tp", max_num_seqs=256, block_size=256, temperature=0.0):
        # Step 1: device setup (portable pattern: device = torch.device(f'cuda:{local_rank}'))
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        
        # Step 2: init dist if needed
        if not dist.is_initialized() and "RANK" in os.environ:
            dist.init_process_group(backend="nccl")  # NVIDIA→NCCL, AMD→RCCL (自动检测)
        
        # Step 3: Load model
        cfg = QwenTPConfig.from_model_dir(model_dir, tp_size, tp_rank)
        model = QwenForCausalLMTP(cfg)
        model = model.to(torch.bfloat16).cuda()
        model.load_weights(model_dir)
        model.eval()
        self.runner = ModelRunner(model)
        
        # Step 4: EOS token — 从 tokenizer 配置读取，禁止硬编码
        self.eos_token_id = self.tokenizer.eos_token_id
        
        # Step 5: Estimate KV blocks
        self.num_kv_blocks = self._estimate_kv_blocks(cfg)
        
        # Step 6: KVMemoryPool — skipped for TP (KV managed by QwenAttentionTP)
        
        # Step 7: Scheduler
        self.scheduler = Scheduler(block_size=self.block_size, max_num_seqs=max_num_seqs)
        # block_size injection: TP path = 256
```

### _estimate_kv_blocks 公式

```python
total_mem = torch.cuda.get_device_properties(rank).total_memory  # NVIDIA: CUDA, AMD: ROCm (兼容层), DCU: 需适配
allocated = torch.cuda.memory_allocated(rank)
free_mem = total_mem - allocated - 2 * 1024**3  # reserve 2GB

bytes_per_block = 2 * block_size * num_kv_heads_local * head_dim * num_layers * 2  # 2(K+V) * bf16
num_blocks = max(1, int(free_mem / bytes_per_block))
```

---

## LLMEngine.generate() — 5 步 While-Loop

```python
def generate(self, prompts: str | list[str], max_tokens: int = 256, max_new_tokens: int = None, temperature: float = 0.0) -> list[str] | str:
    # Step 1: Enqueue prompts → create Sequence objects
    for prompt in prompts:
        input_ids = tokenize(prompt)
        seq = Sequence(seq_id=i, prompt, input_ids, max_tokens, eos_token_id)
        self.scheduler.add(seq)
    
    # Step 2-4: Main loop
    for step in range(max_steps):
        # Step 2: Schedule
        prefill_seqs, decode_seqs = self.scheduler.schedule(self._get_num_free_blocks())
        if not prefill_seqs and not decode_seqs: break
        
        # Step 3: Run model (prefill + decode in one step)
        self.runner.run(prefill_seqs, decode_seqs)
        
        # Step 4: Postprocess + finish check
        for s in prefill_seqs: self.scheduler.postprocess(s)
        if all(s.is_finished for s in seqs): break
    
    # Step 5: Decode outputs
    return [self._decode_output(s) for s in seqs]
```

---

## ModelRunner

```python
class ModelRunner:
    def __init__(self, model):
        self.model = model
        self._device = next(model.parameters()).device
    
    def run(self, prefill_seqs: list[Sequence], decode_seqs: list[Sequence]) -> torch.Tensor:
        # 一次 step: prefill 新序列 + decode 运行序列
        # 返回 [total_batch] sampled token IDs
        
        sampled = []
        if prefill_seqs:
            sampled.append(self._run_prefill(prefill_seqs))
        if decode_seqs:
            sampled.append(self._run_decode(decode_seqs))
        return torch.cat(sampled, dim=0) if sampled else torch.empty(0)
    
    def _run_prefill(self, seqs) -> Tensor:
        # 构造 ragged batch input_ids [total_tokens]
        # forward(model, is_prefill=True) → logits [1, total_tokens, vocab]
        # 提取每序列最后 token 的 logits → greedy sample
        # 转换序列 status: PREFILL → DECODE
    
    def _run_decode(self, seqs) -> Tensor:
        # 每个 decode seq 的输入为 [1] (最后一个 token)
        # forward(model, past_key_values=kv_lens, is_decode=True)
        # greedy sample → append_token → EOS/max_tokens 检测 → finish
```

### get_num_free_blocks

```python
def _get_num_free_blocks(self) -> int:
    """CPU-side counter (O2: no GPU .item() sync)"""
    total_blocks = model.layers[0].attention._max_blocks
    blocks_used = (self._cpu_kv_len + self.block_size - 1) // self.block_size
    return max(0, total_blocks - blocks_used)
```

---

## 引擎主循环完整流程

```
generate(prompt)
  → _enqueue(prompts) → Sequence objects
  → while not all_finished:
      → schedule(num_free_blocks) → (prefill_seqs, decode_seqs)
      → runner.run(prefill_seqs, decode_seqs)
        → _run_prefill: model(input_ids, is_prefill=True) → logits → sample → PREFILL→DECODE
        → _run_decode: model(input_ids, past_key_values=kv_lens) → logits → sample → finish check
      → postprocess(seqs) → update output_ids, check EOS/max_tokens
      → finish_check → break if all done
  → decode_outputs → return string(s)
```

---

## 陷阱与反模式

- **Bug 2 (Phase 9)**: init_tp_distributed 无 WORLD_SIZE guard → 单进程启动挂死
- **Bug 3 (Phase 9)**: input_ids tensor 无 device 参数 → 留在 CPU
- **Bug 1 (Phase 9)**: float32 OOM → 模型未转 bf16
- **ENGINE-001**: TP 路径 Scheduler block_size 必须注入为 256
- **ENGINE-002**: num_free_blocks TP 路径来自 runner，HF 路径来自 BlockManager
- **ENGINE-003**: BlockManager TP 路径 allocate/free 为 no-op
- **ENGINE-004**: 超长 prompt REJECTED — 防止永久 WAITING 死循环
- **ENGINE-005**: get_num_free_blocks 使用 CPU 侧计数器 (避免 GPU .item() sync)
