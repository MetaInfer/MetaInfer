"""
Phase 9 — LLMEngine: system orchestrator that glues Scheduler + ModelRunner.

7-step init (per blueprint full_api_surface):
  1. torch.cuda.set_device + init_tp_distributed()
  2. _select_tp_backend(): detect Qwen/DeepSeek from config.json → qwen_tp/deepseek_tp
  3. Create TPModelRunner (TP path) or HFModelRunner (HF fallback)
  4. eos_token_id from tokenizer or config
  5. _estimate_kv_blocks(): max_position_embeddings // block_size
  6. BlockManager (tp_mode=True for TP path → no-op allocate/free)
  7. Scheduler(block_size=256 for TP, 16 for HF)

5-step generate() while-loop:
  1. _enqueue(prompts) → Sequence(status=WAITING)
  2. begin_generation() → join waiting queue
  3. while has_unfinished_requests(): step()
  4. Collect outputs
  5. return get_generation_outputs()

CRITICAL-01 (scheduler_tp_runner_bridge):
  - TP path: block_size=256 injected to Scheduler, num_free from runner.get_num_free_blocks()
  - HF path: block_size=16, num_free from BlockManager.get_num_free_blocks()
  - BlockManager(tp_mode=True): allocate/free are no-ops, real blocks via torch.arange in QwenAttentionTP

All signatures must match inference_blueprint.json
  > components[6] LLMEngine
  > data_flow_contracts.scheduler_tp_runner_bridge
"""

import os
import json
from pathlib import Path
from typing import List, Optional, Union

import torch

# Suppress vLLM debug logs that pollute stdout during model loading
if "VLLM_LOGGING_LEVEL" not in os.environ:
    os.environ["VLLM_LOGGING_LEVEL"] = "ERROR"
# Suppress RCCL/NCCL INFO messages that pollute stdout under torchrun
if "NCCL_DEBUG" not in os.environ:
    os.environ["NCCL_DEBUG"] = "WARN"

from engine.framework.sequence import Sequence, SequenceStatus
from engine.framework.scheduler import Scheduler, ScheduleResult
from engine.framework.block_manager import BlockManager
from engine.framework.model_runner import TPModelRunner, RunnerOutput
from engine.tp_layers.distributed import init_tp_distributed


class LLMEngine:
    """System orchestrator: routes backend → creates Runner → drives generate loop.

    Public API:
      generate(prompt, max_new_tokens, ...) → str | list[str]
      step(temperature, top_p) → list[Sequence]
      begin_generation(seqs) → None
      has_unfinished_requests() → bool
      get_generation_outputs() → list[str]
    """

    def __init__(
        self,
        model_dir: Union[str, Path],
        tp_size: int = 1,
        inference_backend: str = "qwen_tp",
        max_num_seqs: int = 4,
    ):
        model_dir = Path(model_dir)

        # ==================================================================
        # Step 1: Device + TP distributed init
        # ==================================================================
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        self.device = torch.device(f"cuda:{local_rank}")
        self.dtype = torch.bfloat16
        init_tp_distributed()

        # ==================================================================
        # Step 2: Backend selection
        # ==================================================================
        self.inference_backend = inference_backend
        if inference_backend in (None, "tp", ""):
            self.inference_backend = self._select_tp_backend(model_dir)

        # ==================================================================
        # Step 3: Create ModelRunner
        # ==================================================================
        if self.inference_backend in ("qwen_tp", "deepseek_tp"):
            self.runner = TPModelRunner(model_dir, tp_size=tp_size)
            self.block_size = 256  # flash_attn_with_kvcache requires >= 256
        else:
            raise ValueError(
                f"Unsupported inference_backend: {self.inference_backend}. "
                f"Currently only 'qwen_tp' and 'deepseek_tp' are supported."
            )

        # ==================================================================
        # Step 4: EOS token
        # ==================================================================
        self.eos_token_id = self.runner.tokenizer.eos_token_id

        # ==================================================================
        # Step 5: Estimate KV blocks
        # ==================================================================
        # TP path: max_position_embeddings // block_size
        # (40960 // 256 = 160 for Qwen3-8B, verified by physical config.json)
        max_blocks = self._estimate_kv_blocks()

        # ==================================================================
        # Step 6: BlockManager (TP path → tp_mode=True, no-op allocate/free)
        # ==================================================================
        self.block_manager = BlockManager(
            num_blocks=max_blocks,
            tp_mode=(self.inference_backend in ("qwen_tp", "deepseek_tp")),
            block_size=self.block_size,
        )

        # ==================================================================
        # Step 7: Scheduler (block_size injected per backend)
        # ==================================================================
        self.scheduler = Scheduler(
            block_size=self.block_size, max_blocks=max_blocks
        )

        # Internal state
        self._waiting: List[Sequence] = []
        self._running: List[Sequence] = []
        self._active_gen_seqs: List[Sequence] = []
        self.max_num_seqs = max_num_seqs

    # ------------------------------------------------------------------
    # Step 2 helper: _select_tp_backend
    # ------------------------------------------------------------------

    @staticmethod
    def _select_tp_backend(model_dir: Path) -> str:
        """Read config.json architectures[0] and route to backend name.

        Qwen2/Qwen3 → 'qwen_tp'
        DeepseekV2/DeepseekV3 → 'deepseek_tp'
        else → raise ValueError
        """
        config_path = model_dir / "config.json"
        with open(config_path) as f:
            cfg = json.load(f)
        arch = cfg["architectures"][0]
        if "Qwen" in arch:
            return "qwen_tp"
        elif "Deepseek" in arch or "DeepSeek" in arch:
            return "deepseek_tp"
        else:
            raise ValueError(
                f"Unknown architecture: {arch}. "
                f"Expected Qwen2/Qwen3 or DeepseekV2/DeepseekV3."
            )

    # ------------------------------------------------------------------
    # Step 5 helper: _estimate_kv_blocks
    # ------------------------------------------------------------------

    def _estimate_kv_blocks(self) -> int:
        """Estimate number of KV cache blocks.

        TP path: max_position_embeddings // block_size
        (e.g. 40960 // 256 = 160 for Qwen3-8B).
        """
        max_pos = self.runner.model.cfg.max_position_embeddings
        return max_pos // self.block_size

    # ==================================================================
    # Public API: generate
    # ==================================================================

    def generate(
        self,
        prompts: Union[str, List[str]],
        max_new_tokens: int = 256,
        temperature: float = 0.0,
        top_p: Optional[float] = None,
    ) -> Union[str, List[str]]:
        """Generate text from prompt(s).

        5-step while-loop per blueprint full_api_surface.generate.flow:
          1. _enqueue(prompts, ...) → Sequence objects (status=WAITING)
          2. begin_generation() → join scheduler waiting queue
          3. while has_unfinished_requests(): step()
          4. Collect outputs
          5. return get_generation_outputs()

        Args:
            prompts: Single prompt string or list of prompt strings.
            max_new_tokens: Maximum number of tokens to generate per prompt.
            temperature: 0.0 = greedy (argmax), > 0 = temperature sampling.
            top_p: Nucleus sampling threshold (None = 1.0 = disabled).

        Returns:
            Generated text string (single prompt) or list of strings (batch).
        """
        # Normalize prompts
        if isinstance(prompts, str):
            prompts = [prompts]
            single_prompt = True
        else:
            single_prompt = False

        # Validate prompts: reject empty strings (would cause FPE in GPU kernels)
        for p in prompts:
            if not p or not p.strip():
                raise ValueError(
                    f"Empty prompt is not allowed: {p!r}. "
                    f"Prompt must contain at least one non-whitespace character."
                )

        # Reset state for fresh generation
        self._active_gen_seqs.clear()
        self._waiting.clear()
        self._running.clear()

        # Step 1: Enqueue
        seqs = self._enqueue(prompts, max_new_tokens, temperature, top_p)

        # Step 2: Begin generation
        self.begin_generation(seqs)

        # Step 3: While-loop
        while self.has_unfinished_requests():
            self.step(temperature, top_p)

        # Step 4-5: Collect + return
        outputs = self.get_generation_outputs()
        return outputs[0] if single_prompt else outputs

    # ==================================================================
    # Public API: begin_generation
    # ==================================================================

    def begin_generation(self, seqs: List[Sequence]) -> None:
        """Add sequences to the waiting queue.

        Called after _enqueue to register newly created Sequence objects
        with the scheduler.  Also transitions status to WAITING.

        Args:
            seqs: List of Sequence objects (from _enqueue).
        """
        for seq in seqs:
            seq.status = SequenceStatus.WAITING
            self._waiting.append(seq)

    # ==================================================================
    # Public API: has_unfinished_requests
    # ==================================================================

    def has_unfinished_requests(self) -> bool:
        """Check whether any active sequence is not yet FINISHED/REJECTED.

        Returns:
            True if waiting or running queues are non-empty.
        """
        # Clean running queue: remove already-FINISHED sequences
        self._running = [
            s for s in self._running if s.status not in (
                SequenceStatus.FINISHED, SequenceStatus.REJECTED
            )
        ]
        return bool(self._waiting) or bool(self._running)

    # ==================================================================
    # Public API: step
    # ==================================================================

    def step(
        self,
        temperature: float = 0.0,
        top_p: Optional[float] = None,
    ) -> List[Sequence]:
        """Single scheduling + forward + postprocess step.

        Flow:
          1. num_free = runner.get_num_free_blocks() (TP) or block_manager (HF)
          2. result = scheduler.schedule(waiting, running, num_free)
          3. runner.run(batch_seqs, is_prefill, ...) → next_tokens
          4. scheduler.postprocess(batch_seqs, is_prefill, next_tokens)
          5. State transitions: WAITING→DECODE, FINISHED removal, rejected cleanup

        Args:
            temperature: Sampling temperature.
            top_p: Nucleus sampling threshold.

        Returns:
            List of sequences that completed (FINISHED) in this step.
        """
        # 1. Get num_free_blocks (route per CRITICAL-01)
        if self.inference_backend in ("qwen_tp", "deepseek_tp"):
            num_free = self.runner.get_num_free_blocks()
        else:
            num_free = self.block_manager.get_num_free_blocks()

        # 2. Schedule
        result = self.scheduler.schedule(
            self._waiting, self._running, num_free
        )

        # No batch to execute — cleanup and return
        if not result.batch:
            self._cleanup_rejected(result.rejected)
            return []

        is_prefill = result.is_prefill
        batch_seqs = result.batch

        # 3. Run model forward
        output = self.runner.run(
            batch_seqs,
            is_prefill=is_prefill,
            temperature=temperature,
            top_p=top_p,
        )

        # 4. Postprocess (advance state, check termination)
        self.scheduler.postprocess(batch_seqs, is_prefill, output.next_tokens)

        # 5. State transitions
        if is_prefill:
            # Move from waiting→running (prefill→decode transition)
            for seq in batch_seqs:
                if seq in self._waiting:
                    self._waiting.remove(seq)
                if seq not in self._running:
                    self._running.append(seq)

        # Collect finished sequences
        finished: List[Sequence] = []
        still_running: List[Sequence] = []
        for seq in self._running:
            if seq.status == SequenceStatus.FINISHED:
                finished.append(seq)
            else:
                still_running.append(seq)
        self._running = still_running

        # Cleanup rejected sequences
        self._cleanup_rejected(result.rejected)

        return finished

    # ==================================================================
    # Public API: get_generation_outputs
    # ==================================================================

    def get_generation_outputs(self) -> List[str]:
        """Decode generated token sequences to text.

        Returns:
            List of decoded strings, one per active generation sequence.
        """
        outputs: List[str] = []
        for seq in self._active_gen_seqs:
            text = self.runner.tokenizer.decode(
                seq.output_ids, skip_special_tokens=True
            )
            outputs.append(text)
        return outputs

    # ==================================================================
    # Internal: _enqueue
    # ==================================================================

    def _enqueue(
        self,
        prompts: List[str],
        max_new_tokens: int,
        temperature: float,
        top_p: Optional[float],
        request_ids: Optional[List[str]] = None,
    ) -> List[Sequence]:
        """Tokenize prompts and create Sequence objects.

        Args:
            prompts: List of prompt strings.
            max_new_tokens: Max tokens to generate per prompt.
            temperature: Sampling temperature.
            top_p: Nucleus sampling threshold.
            request_ids: Optional custom request IDs.

        Returns:
            List of Sequence objects (status=WAITING).
        """
        seqs: List[Sequence] = []
        for i, prompt in enumerate(prompts):
            if not prompt or not prompt.strip():
                raise ValueError(
                    f"Empty prompt is not allowed: {prompt!r}. "
                    f"Prompt must contain at least one non-whitespace character."
                )
            token_ids = self.runner.tokenizer.encode(
                prompt, add_special_tokens=True
            )
            seq = Sequence(
                input_ids=token_ids,
                max_output_len=max_new_tokens,
                block_size=self.block_size,
                max_blocks=self._estimate_kv_blocks(),
                device=self.device,
            )
            seq.sampling_params = {
                "max_tokens": max_new_tokens,
                "temperature": temperature,
                "top_p": top_p if top_p is not None else 1.0,
            }
            seq.status = SequenceStatus.WAITING
            seqs.append(seq)
            self._active_gen_seqs.append(seq)

        return seqs

    # ==================================================================
    # Internal: cleanup_rejected
    # ==================================================================

    def _cleanup_rejected(self, rejected: List[Sequence]) -> None:
        """Remove rejected sequences from waiting queue and active set."""
        for seq in rejected:
            if seq in self._waiting:
                self._waiting.remove(seq)
            if seq in self._active_gen_seqs:
                self._active_gen_seqs.remove(seq)
