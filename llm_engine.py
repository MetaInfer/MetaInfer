from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from engine.kv_specs import hf_deepseek_v2_kv_bytes_per_token
from engine.memory_pool import KVMemoryPool
from engine.sampler import sample_next_tokens
from engine.scheduler import Scheduler
from engine.structs import Sequence, SequenceStatus


MODEL_DIR = Path("/data/xinference/cache/deepseek-v2-chat-pytorch-16b")


@dataclass
class SamplingParams:
    max_tokens: int
    temperature: float = 0.0
    top_p: float | None = None


class RealModelRunner:
    """
    HF 真实模型前向。
    当前使用 use_cache=False 的整段重算路径，避免部分 transformers 与远程 modeling_deepseek
    中 DynamicCache API 不一致导致的 past_key_values 错误；K/V 物理占用由 KVMemoryPool 按 MLA 维度预留。
    """

    def __init__(self, model_dir: str | Path, device: torch.device, dtype: torch.dtype) -> None:
        self.model_dir = str(model_dir)
        self.device = device
        self.dtype = dtype

        print(f"[ModelRunner] loading tokenizer from: {self.model_dir}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_dir, trust_remote_code=True, local_files_only=True
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        print(f"[ModelRunner] tokenizer eos={self.tokenizer.eos_token_id}, pad={self.tokenizer.pad_token_id}")

        print(f"[ModelRunner] loading model dtype={self.dtype}, device={self.device}")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_dir,
            dtype=self.dtype,
            trust_remote_code=True,
            local_files_only=True,
        )
        self.model.to(self.device)
        self.model.eval()
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"[ModelRunner] model loaded, parameters={total_params:,}")
        cfg = self.model.config
        kv_b = hf_deepseek_v2_kv_bytes_per_token(cfg, self.dtype)
        print(f"[ModelRunner] MLA KV bytes/token (HF materialized K/V)≈{kv_b} (heads={cfg.num_attention_heads})")
        print(
            f"[ModelRunner] config: q_head={cfg.qk_nope_head_dim}+{cfg.qk_rope_head_dim}, "
            f"v_head={cfg.v_head_dim}, layers={cfg.num_hidden_layers} (MoE 不占 KV)"
        )

    @torch.inference_mode()
    def run(
        self,
        seqs: list[Sequence],
        *,
        is_prefill: bool,
        temperature: float,
        top_p: float | None,
    ) -> list[int]:
        if not seqs:
            return []
        next_tokens: list[int] = []
        if is_prefill:
            for seq in seqs:
                ids = torch.tensor([seq.token_ids], dtype=torch.long, device=self.device)
                out = self.model(input_ids=ids, use_cache=False, return_dict=True)
                logits = out.logits[0, -1, :]
                tid = int(sample_next_tokens(logits.unsqueeze(0), temperature=temperature, top_p=top_p).item())
                next_tokens.append(tid)
                print(f"[ModelRunner] prefill req={seq.request_id} len={seq.total_tokens} first_token={tid}")
        else:
            max_len = max(len(seq.token_ids) for seq in seqs)
            input_ids: list[list[int]] = []
            mask: list[list[int]] = []
            for seq in seqs:
                toks = seq.token_ids
                pad_len = max_len - len(toks)
                input_ids.append([self.tokenizer.pad_token_id] * pad_len + toks)
                mask.append([0] * pad_len + [1] * len(toks))
            ids = torch.tensor(input_ids, dtype=torch.long, device=self.device)
            m = torch.tensor(mask, dtype=torch.long, device=self.device)
            out = self.model(input_ids=ids, attention_mask=m, use_cache=False, return_dict=True)
            logits = out.logits[:, -1, :]
            next_tokens = sample_next_tokens(logits, temperature=temperature, top_p=top_p).detach().cpu().tolist()
            print(f"[ModelRunner] decode batch={len(seqs)} max_len={max_len} next_tokens={next_tokens}")
        return next_tokens


class LLMEngine:
    def __init__(
        self,
        model_dir: str | Path = MODEL_DIR,
        *,
        block_size: int = 16,
        mem_utilization: float = 0.85,
        reserve_bytes: int = 2 * 1024**3,
        max_num_seqs: int = 4,
        max_num_batched_tokens: int = 4096,
    ) -> None:
        self.model_dir = Path(model_dir)
        self.block_size = block_size
        self.mem_utilization = mem_utilization
        self.reserve_bytes = reserve_bytes

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            self.dtype = torch.bfloat16
        else:
            self.device = torch.device("cpu")
            self.dtype = torch.float32
        print(f"[LLMEngine] device={self.device}, dtype={self.dtype}")

        self.runner = RealModelRunner(self.model_dir, self.device, self.dtype)
        self.eos_token_id = self.runner.tokenizer.eos_token_id

        num_blocks = self._estimate_kv_blocks()
        print(f"[LLMEngine] KV pool setup: block_size={self.block_size}, num_blocks={num_blocks}")
        self.memory_pool = KVMemoryPool(
            num_blocks=num_blocks,
            block_size=self.block_size,
            hf_config=self.runner.model.config,
            dtype=self.dtype,
            device=self.device,
            reserve_physical_kv=True,
        )
        self.scheduler = Scheduler(
            memory_pool=self.memory_pool,
            max_num_seqs=max_num_seqs,
            max_num_batched_tokens=max_num_batched_tokens,
        )

    def _estimate_kv_blocks(self) -> int:
        if self.device.type != "cuda":
            print("[LLMEngine] CUDA not available, fallback KV blocks=256")
            return 256

        free_bytes, total_bytes = torch.cuda.mem_get_info(device=self.device)
        print(
            f"[LLMEngine] gpu mem: free={free_bytes/1024**3:.2f}GB total={total_bytes/1024**3:.2f}GB"
        )
        cfg = self.runner.model.config
        return KVMemoryPool.estimate_num_blocks(
            cfg,
            block_size=self.block_size,
            dtype=self.dtype,
            free_bytes=free_bytes,
            reserve_bytes=self.reserve_bytes,
            mem_utilization=self.mem_utilization,
        )

    def _enqueue(
        self,
        prompts: list[str],
        max_new_tokens: int,
        temperature: float,
        top_p: float | None,
    ) -> list[Sequence]:
        seqs: list[Sequence] = []
        for i, prompt in enumerate(prompts):
            token_ids = self.runner.tokenizer.encode(prompt, add_special_tokens=True)
            seq = Sequence(
                request_id=f"req-{i}",
                input_ids=token_ids,
                sampling_params={"max_tokens": max_new_tokens, "temperature": temperature, "top_p": top_p},
            )
            seq.block_size = self.block_size
            self.scheduler.add_request(seq)
            seqs.append(seq)
            print(f"[LLMEngine] enqueue {seq.request_id}: prompt_len={len(token_ids)}")
        return seqs

    def _all_finished(self, seqs: list[Sequence]) -> bool:
        return all(seq.status == SequenceStatus.FINISHED for seq in seqs)

    def _finish_check_and_cleanup(self, seq: Sequence) -> bool:
        max_tokens = int(seq.sampling_params.get("max_tokens", 0))
        gen_len = len(seq.output_ids)
        reached_max = gen_len >= max_tokens
        reached_eos = bool(seq.output_ids and seq.output_ids[-1] == self.eos_token_id)
        if not (reached_max or reached_eos):
            return False
        seq.transition_to(SequenceStatus.FINISHED)
        if seq in self.scheduler.running:
            self.scheduler.running.remove(seq)
        self.memory_pool.free_sequence(seq)
        print(
            f"[LLMEngine] finish {seq.request_id}: gen_len={gen_len}, "
            f"reason={'eos' if reached_eos else 'max_tokens'}, free_blocks={self.memory_pool.num_free_blocks}"
        )
        return True

    def generate(
        self,
        prompt: str | list[str],
        max_new_tokens: int,
        temperature: float = 0.0,
        top_p: float | None = None,
    ) -> str | list[str]:
        prompts = [prompt] if isinstance(prompt, str) else prompt
        t0 = time.time()
        seqs = self._enqueue(prompts, max_new_tokens=max_new_tokens, temperature=temperature, top_p=top_p)

        step = 0
        while True:
            step += 1
            batch, is_prefill = self.scheduler.schedule()
            print(
                f"[LLMEngine] step={step} phase={'prefill' if is_prefill else 'decode'} "
                f"batch={len(batch)} waiting={len(self.scheduler.waiting)} running={len(self.scheduler.running)} "
                f"free_blocks={self.memory_pool.num_free_blocks}"
            )
            if not batch:
                if self._all_finished(seqs):
                    break
                for seq in seqs:
                    if seq.status != SequenceStatus.FINISHED and len(seq.output_ids) >= max_new_tokens:
                        self._finish_check_and_cleanup(seq)
                if self._all_finished(seqs):
                    break
                raise RuntimeError("Scheduler returned empty batch before all sequences finished")

            if is_prefill:
                first_tokens = self.runner.run(
                    batch, is_prefill=True, temperature=temperature, top_p=top_p
                )
                self.scheduler.postprocess(batch, is_prefill=True, generated_tokens=first_tokens)
                continue

            next_tokens = self.runner.run(batch, is_prefill=False, temperature=temperature, top_p=top_p)
            self.scheduler.postprocess(batch, is_prefill=False, generated_tokens=next_tokens)
            for seq in batch:
                if seq.status == SequenceStatus.RUNNING_DECODE:
                    self._finish_check_and_cleanup(seq)
            if self._all_finished(seqs):
                break

        out_texts = []
        for seq in seqs:
            text = self.runner.tokenizer.decode(seq.output_ids, skip_special_tokens=True)
            out_texts.append(text)
            print(f"[LLMEngine] output {seq.request_id}: {text!r}")
        print(f"[LLMEngine] generate done in {time.time()-t0:.2f}s")
        return out_texts[0] if isinstance(prompt, str) else out_texts
