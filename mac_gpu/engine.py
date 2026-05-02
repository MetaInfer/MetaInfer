"""
Mac GPU (MPS) 推理引擎入口。简化版 LLMEngine，不含 TP 和 CUDA 逻辑。
"""

from __future__ import annotations

from mac_gpu.memory_pool import MPSMemoryPool
from mac_gpu.model_runner import MPSModelRunner
from mac_gpu.scheduler import Scheduler
from mac_gpu.structs import Sequence, SequenceStatus


class MacGPUEngine:
    def __init__(
        self,
        model_name_or_path: str,
        block_size: int = 16,
        max_num_seqs: int = 4,
        max_num_batched_tokens: int = 2048,
        mem_utilization: float = 0.80,
    ) -> None:
        self.block_size = block_size

        # 加载模型到 MPS
        self.runner = MPSModelRunner(model_name_or_path)
        self.eos_token_id = self.runner.eos_token_id

        # 估算模型权重占用
        model_bytes = sum(p.numel() * p.element_size() for p in self.runner.model.parameters())

        # 估算 KV 块数量
        num_blocks = MPSMemoryPool.estimate_num_blocks(
            hf_config=self.runner.model.config,
            block_size=block_size,
            model_bytes=model_bytes,
            mem_utilization=mem_utilization,
        )
        print(
            f"[MacGPUEngine] KV pool: block_size={block_size}, num_blocks={num_blocks}, "
            f"model_bytes={model_bytes / 1024**3:.2f}GB"
        )

        self.memory_pool = MPSMemoryPool(num_blocks=num_blocks, block_size=block_size)
        self.scheduler = Scheduler(
            memory_pool=self.memory_pool,
            max_num_seqs=max_num_seqs,
            max_num_batched_tokens=max_num_batched_tokens,
        )

    def generate(
        self,
        prompt: str | list[str],
        max_new_tokens: int,
        temperature: float = 0.0,
        top_p: float | None = None,
    ) -> str | list[str]:
        prompts = [prompt] if isinstance(prompt, str) else prompt

        # 入队
        seqs: list[Sequence] = []
        for i, p in enumerate(prompts):
            token_ids = self.runner.tokenizer.encode(p, add_special_tokens=True)
            seq = Sequence(
                request_id=f"req-{i}",
                input_ids=token_ids,
                sampling_params={"temperature": temperature, "top_p": top_p},
            )
            seq.block_size = self.block_size
            self.scheduler.add_request(seq)
            seqs.append(seq)

        # 调度循环
        step = 0
        while not all(s.status == SequenceStatus.FINISHED for s in seqs):
            batch, is_prefill = self.scheduler.schedule()

            if not batch:
                # 没有可调度的序列，检查是否达到 max_tokens
                all_done = True
                for seq in seqs:
                    if seq.status != SequenceStatus.FINISHED:
                        if len(seq.output_ids) >= max_new_tokens:
                            self._finish_seq(seq)
                        else:
                            all_done = False
                if all_done:
                    break
                continue

            next_tokens = self.runner.run(batch, is_prefill)
            self.scheduler.postprocess(batch, is_prefill, generated_tokens=next_tokens)
            step += 1

            # 检查完成条件
            for seq in batch:
                if seq.status == SequenceStatus.RUNNING_DECODE:
                    self._finish_check(seq, max_new_tokens)

        # 解码输出
        results = [
            self.runner.tokenizer.decode(s.output_ids, skip_special_tokens=True) for s in seqs
        ]
        return results[0] if isinstance(prompt, str) else results

    def _finish_check(self, seq: Sequence, max_tokens: int) -> None:
        if len(seq.output_ids) >= max_tokens:
            self._finish_seq(seq)
            return
        if seq.output_ids and seq.output_ids[-1] == self.eos_token_id:
            self._finish_seq(seq)

    def _finish_seq(self, seq: Sequence) -> None:
        if seq.status == SequenceStatus.FINISHED:
            return
        seq.transition_to(SequenceStatus.FINISHED)
        if seq in self.scheduler.running:
            self.scheduler.running.remove(seq)
        self.memory_pool.free_sequence(seq)
        seq.past_key_values = None  # 释放 KV cache 显存
