"""
Phase 9 — LLMEngine + QwenTPModelRunner: engine integration.

Blueprint contracts:
  - framework_layer.components[6] LLMEngine.full_api_surface
  - framework_layer.components[3] ModelRunner.tp_runner_actual_flow
  - framework_layer.data_flow_contracts.scheduler_tp_runner_bridge

Architecture:
  LLMEngine is the system orchestrator — routes inference_backend, creates Runner,
  estimates KV pool, initializes Scheduler, and drives the generate/step loop.
  It is the glue layer between Scheduler and ModelRunner.

  QwenTPModelRunner wraps QwenForCausalLMTP with a tokenizer, exposes run()
  for prefill/decode and get_num_free_blocks() for scheduler capacity queries.
"""

import os
import json
import torch
from pathlib import Path

from engine.memory_pool import KVMemoryPool
from engine.structs import Sequence, SeqStatus
from engine.sampler import tp_sample
from engine.scheduler import Scheduler
from engine.block_manager import BlockManager


# ================================================================
# QwenTPModelRunner
# ================================================================

class QwenTPModelRunner:
    """TP Model Runner for Qwen3 models.

    Wraps QwenForCausalLMTP with tokenizer, exposes run() for prefill/decode
    dispatch and get_num_free_blocks() for scheduler integration.

    Blueprint contract:
      components[3] ModelRunner.tp_runner_actual_flow.run_method_impl
      scheduler_tp_runner_bridge.num_free_blocks_source.TP_Runner.impl
    """

    def __init__(self, model_dir, device, dtype=torch.bfloat16):
        from engine.models.qwen import QwenTPConfig, QwenForCausalLMTP
        from transformers import AutoTokenizer

        self.device = device
        self.dtype = dtype
        self.cfg = QwenTPConfig.from_model_dir(model_dir)
        self.model = QwenForCausalLMTP(self.cfg, device=device, dtype=dtype)
        self.model.load_weights()
        self.model.eval()

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(model_dir), trust_remote_code=True)
        # eos_token_id fallback chain: tokenizer > config attr > Qwen3 default
        if self.tokenizer.eos_token_id is not None:
            self.eos_token_id = self.tokenizer.eos_token_id
        elif hasattr(self.cfg, 'eos_token_id'):
            self.eos_token_id = self.cfg.eos_token_id
        else:
            self.eos_token_id = 151645  # Qwen3 default

        self.max_seq_len = self.cfg.max_position_embeddings

    def get_num_free_blocks(self):
        """Return number of free blocks available.

        Blueprint contract:
          scheduler_tp_runner_bridge.num_free_blocks_source.TP_Runner.impl
          kv_len = model.layers[0].self_attn._kv_len_gpu[0].item()
          max_blocks = config.max_position_embeddings // 256
          allocated = (kv_len + 255) // 256
          return max_blocks - allocated
        """
        kv_len = self.model.layers[0].self_attn._kv_len_gpu[0].item()
        max_blocks = self.cfg.max_position_embeddings // 256
        allocated = (kv_len + 255) // 256
        return max_blocks - allocated

    def run(self, seqs, is_prefill, temperature=0.0, top_p=1.0):
        """Execute prefill or decode for a batch of sequences.

        Blueprint contract:
          tp_runner_actual_flow.run_method_impl — prefill/decode dispatch.

        Prefill:
          - Ragged concatenation of input_ids: [1, total_tokens]
          - Positions from 0 for each sequence
          - Calls model.forward() with past_key_values=None

        Decode:
          - Single token per sequence: [B, 1]
          - Position = current kv_len
          - Calls model.forward_decode()

        Sampling:
          - tp_sample() handles rank-0-only sampling + broadcast for TP

        Args:
            seqs: list[Sequence]
            is_prefill: True for prefill batch, False for decode batch
            temperature: sampling temperature (0.0 = greedy)
            top_p: nucleus sampling threshold (1.0 = disabled)

        Returns:
            list[int] of next tokens per sequence
        """
        if not seqs:
            return []

        if is_prefill:
            # Ragged concatenation: [1, total_tokens]
            input_ids = torch.cat([s.input_ids_tensor(device=self.device) for s in seqs], dim=1)
            positions = torch.cat(
                [torch.arange(s.seq_len(), device=self.device) for s in seqs])
            # Use existing forward() with past_key_values=None for prefill
            logits, _ = self.model.forward(
                input_ids, past_key_values=None, max_seq_len=self.max_seq_len)
            for s in seqs:
                s.kv_len = s.seq_len()
        else:
            # Decode: single token per sequence
            kv_lens = [s.kv_len for s in seqs]
            input_ids = torch.tensor(
                [[s.output_ids[-1]] for s in seqs],
                dtype=torch.long, device=self.device)
            positions = torch.tensor(
                [kv_lens[0]], dtype=torch.long, device=self.device)
            logits = self.model.forward_decode(
                input_ids, positions=positions,
                kv_len=kv_lens[0], max_seq_len=self.max_seq_len)
            for s in seqs:
                s.kv_len = self.model.layers[0].self_attn._kv_len_gpu[0].item()

        tokens = tp_sample(logits[:, -1, :], temperature, top_p)
        return tokens


# ================================================================
# RealModelRunner (HF fallback — stub for future extension)
# ================================================================

class RealModelRunner:
    """HF fallback model runner.

    This is a stub for the HF inference path. Full implementation requires
    contiguous KV cache + BlockManager integration, which is outside the
    current TP-focused scope.

    Blueprint contract:
      components[3] ModelRunner — HF path uses model(input_ids, use_cache=False).
    """

    def __init__(self, model_dir, device, dtype=torch.bfloat16):
        from transformers import AutoTokenizer
        import json

        self.device = device
        self.dtype = dtype
        self.model_dir = Path(model_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(model_dir), trust_remote_code=True)
        self.eos_token_id = self.tokenizer.eos_token_id or 151645

        # Load config
        with open(self.model_dir / 'config.json') as f:
            config = json.load(f)
        # Minimal config attrs needed by LLMEngine
        self.cfg = type('Config', (), {
            'num_hidden_layers': config['num_hidden_layers'],
            'num_key_value_heads': config.get('num_key_value_heads', config['num_attention_heads']),
            'head_dim': config.get('head_dim', config['hidden_size'] // config['num_attention_heads']),
            'max_position_embeddings': config.get('max_position_embeddings', 40960),
        })()
        self.max_seq_len = self.cfg.max_position_embeddings

        raise NotImplementedError(
            "RealModelRunner (HF path) is not implemented in this scope. "
            "Use inference_backend='qwen_tp' for TP runner.")


# ================================================================
# LLMEngine
# ================================================================

class LLMEngine:
    """System orchestrator: routes backend, creates Runner, drives generate/step loop.

    Blueprint contract:
      components[6] LLMEngine.full_api_surface

    Two API surfaces:
      1. Single-shot: generate(prompt, ...) -> str
      2. Step-based: begin_generation() + step() + get_generation_outputs()
         (for OpenAI server integration)

    Internal flow (generate):
      enqueue -> while-loop(schedule -> run -> postprocess -> finish_check) -> decode

    __init__ 7-step flow:
      1. Set device from LOCAL_RANK
      2. Route inference_backend (tp -> auto-detect Qwen/DeepSeek)
      3. Create Runner (QwenTPModelRunner / RealModelRunner)
      4. Set eos_token_id from runner
      5. Determine block_size (256 TP, 16 HF)
      6. Estimate KV blocks
      7. Create KVMemoryPool + Scheduler
    """

    def __init__(self, model_dir, inference_backend='hf', block_size=None,
                 mem_utilization=0.85, reserve_bytes=2 * 1024**3,
                 max_num_seqs=4, max_num_batched_tokens=4096):
        # ---- Step 1: Set device ----
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        torch.cuda.set_device(local_rank)
        self.device = torch.device(f'cuda:{local_rank}')
        self.dtype = torch.bfloat16

        # ---- Step 2: Route backend ----
        if inference_backend == 'tp':
            inference_backend = self._select_tp_backend(model_dir)
        self.inference_backend = inference_backend

        # ---- Step 3: Create Runner ----
        if inference_backend == 'qwen_tp':
            from engine.tp_layers.distributed import init_tp_distributed, is_tp_enabled
            import os as _os
            _world_size = int(_os.environ.get('WORLD_SIZE', '1'))
            if _world_size > 1 and not is_tp_enabled():
                init_tp_distributed()
            self.runner = QwenTPModelRunner(model_dir, self.device, self.dtype)
        elif inference_backend == 'deepseek_tp':
            raise NotImplementedError(
                "DeepSeek TP runner is not implemented in this scope. "
                "Use inference_backend='qwen_tp' or 'hf'.")
        elif inference_backend == 'hf':
            self.runner = RealModelRunner(model_dir, self.device, self.dtype)
        else:
            raise ValueError(f"Unknown inference_backend: {inference_backend}")

        # ---- Step 4: Set eos_token_id ----
        self.eos_token_id = self.runner.eos_token_id

        # ---- Step 5: Determine block_size ----
        if inference_backend in ('qwen_tp', 'deepseek_tp'):
            self.block_size = 256  # flash_attn_with_kvcache hard requirement
        else:
            self.block_size = block_size if block_size is not None else 16

        # ---- Step 6: Estimate KV blocks ----
        self.reserve_bytes = reserve_bytes
        self.mem_utilization = mem_utilization
        num_blocks = self._estimate_kv_blocks()

        # ---- Step 7: Create KVMemoryPool + Scheduler ----
        self.memory_pool = KVMemoryPool(
            num_blocks, self.block_size,
            self.runner.cfg.num_hidden_layers,
            self.runner.cfg.num_key_value_heads,
            self.runner.cfg.head_dim)

        self.scheduler = Scheduler(
            max_num_seqs=max_num_seqs,
            max_num_batched_tokens=max_num_batched_tokens,
            eos_token_id=self.eos_token_id)

        # Inject block_size and max_blocks into scheduler
        # (blueprint scheduler_tp_runner_bridge.llm_engine_block_size_injection)
        self.scheduler._block_size = self.block_size
        self.scheduler._max_blocks = self.runner.cfg.max_position_embeddings // self.block_size

        # Active generation sequences (used by step-based API)
        self._active_gen_seqs = []

    # ----------------------------------------------------------------
    # Backend routing
    # ----------------------------------------------------------------

    @staticmethod
    def _select_tp_backend(model_dir):
        """Auto-detect TP backend from config.json architectures.

        Blueprint contract:
          full_api_surface.__init__._select_tp_backend
        """
        cfg_path = os.path.join(model_dir, 'config.json')
        with open(cfg_path) as f:
            cfg = json.load(f)
        arch = cfg['architectures'][0]
        if 'Qwen' in arch:
            return 'qwen_tp'
        if 'Deepseek' in arch or 'DeepSeek' in arch:
            return 'deepseek_tp'
        raise ValueError(
            f"Unknown architecture: {arch}. "
            f"Cannot auto-detect TP backend from {cfg_path}")

    # ----------------------------------------------------------------
    # KV block estimation
    # ----------------------------------------------------------------

    def _estimate_kv_blocks(self):
        """Estimate number of KV blocks based on free GPU memory.

        Blueprint contract:
          full_api_surface.__init__._estimate_kv_blocks.dense_pseudocode

        Uses Dense formula (Qwen3 path):
          K+V per token = layers * kv_heads * head_dim * 2 * elem_bytes

        Note: MLA (DeepSeek) formula is not implemented — DeepSeek path uses
        a different KV structure (KV cache + absorb).
        """
        free_bytes, total = torch.cuda.mem_get_info(self.device)
        cfg = self.runner.cfg
        return KVMemoryPool.estimate_num_blocks_dense(
            free_bytes, self.reserve_bytes, self.mem_utilization,
            cfg.num_hidden_layers, cfg.num_key_value_heads, cfg.head_dim,
            self.block_size)

    # ----------------------------------------------------------------
    # Single-shot generate
    # ----------------------------------------------------------------

    def generate(self, prompt, max_new_tokens=256, temperature=0.0, top_p=1.0):
        """Generate completion for a single prompt.

        Blueprint contract:
          full_api_surface.generate — 6-step flow.

        Flow:
          1. Enqueue: tokenizer.encode -> Sequence -> scheduler.add
          2. While-loop: schedule -> run -> postprocess -> finish check

        Args:
            prompt: str — input text
            max_new_tokens: int — max tokens to generate
            temperature: float — sampling temperature (0.0 = greedy)
            top_p: float — nucleus sampling threshold

        Returns:
            str — decoded completion text
        """
        # 1. Enqueue
        seq = self._enqueue(
            [prompt], max_new_tokens, temperature, top_p,
            request_ids=['gen-0'])[0]
        self._active_gen_seqs = [seq]

        # 2. While-loop
        while not self.scheduler.is_finished():
            num_free = self._get_num_free_blocks()
            batch, is_prefill = self.scheduler.schedule(num_free)
            if not batch:
                break
            tokens = self.runner.run(batch, is_prefill, temperature, top_p)
            self.scheduler.postprocess(batch, is_prefill, tokens)

        # 3. Decode output
        return self.runner.tokenizer.decode(
            seq.output_ids, skip_special_tokens=True)

    # ----------------------------------------------------------------
    # Step-based API (for OpenAI server)
    # ----------------------------------------------------------------

    def begin_generation(self, prompts, max_new_tokens, temperature, top_p):
        """Enqueue multiple prompts for step-by-step generation.

        Blueprint contract:
          full_api_surface.begin_generation

        Args:
            prompts: list[str] — input texts
            max_new_tokens: int
            temperature: float
            top_p: float
        """
        self._active_gen_seqs = self._enqueue(
            prompts, max_new_tokens, temperature, top_p,
            request_ids=[f'gen-{i}' for i in range(len(prompts))])

    def has_unfinished_requests(self):
        """Check if any sequences are still active.

        Blueprint contract:
          full_api_surface.has_unfinished_requests
        """
        return not self.scheduler.is_finished()

    def step(self, temperature=0.0, top_p=1.0):
        """Advance generation by one scheduling step.

        Blueprint contract:
          full_api_surface.step — 5-step flow.

        Returns:
            list[Sequence] — sequences that finished in this step
        """
        num_free = self._get_num_free_blocks()
        batch, is_prefill = self.scheduler.schedule(num_free)
        finished = []

        if batch:
            tokens = self.runner.run(batch, is_prefill, temperature, top_p)
            self.scheduler.postprocess(batch, is_prefill, tokens)

            # Check for finished sequences
            for seq in batch:
                if seq.status == SeqStatus.FINISHED:
                    finished.append(seq)
                elif self._check_finish(seq):
                    self._finish_cleanup(seq)
                    finished.append(seq)

        # Also handle sequences that reached max_tokens without a batch
        for seq in list(self.scheduler.running):
            if self._check_finish(seq):
                self._finish_cleanup(seq)
                if seq not in finished:
                    finished.append(seq)

        return finished

    def get_generation_outputs(self):
        """Return decoded texts for all active sequences.

        Blueprint contract:
          full_api_surface.get_generation_outputs
        """
        return [
            self.runner.tokenizer.decode(
                s.output_ids, skip_special_tokens=True)
            for s in self._active_gen_seqs
        ]

    # ----------------------------------------------------------------
    # Internal helpers
    # ----------------------------------------------------------------

    def _get_num_free_blocks(self):
        """Get num_free_blocks based on inference_backend.

        Blueprint contract:
          scheduler_tp_runner_bridge.num_free_blocks_source.interface
          TP path: runner.get_num_free_blocks()
          HF path: BlockManager.get_num_free_blocks()
        """
        if self.inference_backend in ('qwen_tp', 'deepseek_tp'):
            return self.runner.get_num_free_blocks()
        else:
            # HF path: BlockManager (not implemented in this scope)
            # Fallback: return a large number
            return self.scheduler._max_blocks

    def _enqueue(self, prompts, max_new_tokens, temperature, top_p,
                 request_ids=None):
        """Encode prompts and add them to the scheduler.

        Blueprint contract:
          full_api_surface._enqueue

        Flow:
          tokenizer.encode -> Sequence(input_ids, sampling_params) ->
          seq.block_size = self.block_size -> scheduler.add(seq)

        Returns:
            list[Sequence]
        """
        if request_ids is None:
            request_ids = [f'gen-{i}' for i in range(len(prompts))]

        sequences = []
        for prompt, req_id in zip(prompts, request_ids):
            input_ids = self.runner.tokenizer.encode(prompt)
            seq = Sequence(
                request_id=req_id,
                input_ids=input_ids,
                block_size=self.block_size,
                max_model_len=self.runner.max_seq_len,
                max_blocks=self.scheduler._max_blocks,
                device=self.device)
            seq.max_tokens = max_new_tokens
            seq.temperature = temperature
            seq.top_p = top_p
            self.scheduler.add(seq)
            sequences.append(seq)

        return sequences

    def _check_finish(self, seq):
        """Check if a sequence should be finished.

        Conditions:
          - EOS token generated
          - max_tokens reached
        """
        if not seq.output_ids:
            return False
        last_token = seq.output_ids[-1]
        if self.eos_token_id is not None and last_token == self.eos_token_id:
            return True
        if seq.max_tokens > 0 and seq.num_completion_tokens >= seq.max_tokens:
            return True
        return False

    def _finish_cleanup(self, seq):
        """Transition sequence to FINISHED and release resources.

        Blueprint contract:
          full_api_surface.generate._finish_check_and_cleanup
        """
        if seq.status != SeqStatus.FINISHED:
            seq.transition_to(SeqStatus.FINISHED)
        self.scheduler._release(seq)
