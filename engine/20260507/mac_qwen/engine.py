"""Qwen3.5 MoE 自包含推理引擎。

用法:
    python -m engine.20260507.mac_qwen.engine --prompt "你好" --model-dir /path/to/model
"""

from __future__ import annotations

import argparse
import importlib
import time
from pathlib import Path

import torch

from engine.structs import Sequence, SequenceStatus

# 目录名 20260507 以数字开头，无法常规 import
_model_mod = importlib.import_module("engine.20260507.mac_qwen.model")
Qwen35MoeModelRunner = _model_mod.Qwen35MoeModelRunner
states_to_kwargs = _model_mod.states_to_kwargs
update_states_from_kwargs = _model_mod.update_states_from_kwargs


def _select_device() -> tuple[torch.device, torch.dtype]:
    if torch.cuda.is_available():
        return torch.device("cuda"), torch.bfloat16
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps"), torch.float16
    return torch.device("cpu"), torch.float32


class Qwen35MoeEngine:
    """自包含的 Qwen3.5 MoE 推理引擎。"""

    def __init__(
        self,
        model_dir: str | Path,
        *,
        max_seq_len: int = 8192,
        max_new_tokens: int = 512,
    ):

        device, dtype = _select_device()
        print(f"[Engine] device={device}, dtype={dtype}")

        self.runner = Qwen35MoeModelRunner(
            model_dir=model_dir,
            device=device,
            dtype=dtype,
            max_seq_len=max_seq_len,
        )
        self.max_new_tokens = max_new_tokens
        self.max_seq_len = max_seq_len

    def generate(
        self,
        prompt: str,
        *,
        max_new_tokens: int | None = None,
        temperature: float = 0.6,
        top_p: float = 0.95,
        stream: bool = False,
    ) -> str:
        max_tokens = max_new_tokens or self.max_new_tokens
        tokenizer = self.runner.tokenizer

        token_ids = tokenizer.encode(prompt, add_special_tokens=True)
        print(f"[Engine] prompt_len={len(token_ids)}")

        seq = Sequence(
            request_id="req-0",
            input_ids=token_ids,
            sampling_params={"max_tokens": max_tokens, "temperature": temperature, "top_p": top_p},
        )
        self.runner.init_sequence(seq.request_id)

        # Prefill
        seq.transition_to(SequenceStatus.RUNNING_PREFILL)
        ids = torch.tensor([seq.token_ids], dtype=torch.long, device=self.runner.device)
        position_ids = torch.arange(
            seq.total_tokens, device=self.runner.device, dtype=torch.long
        ).unsqueeze(0)

        states = self.runner._seq_states[seq.request_id]
        kwargs_list = states_to_kwargs(states)
        for kw in kwargs_list:
            kw["position_ids"] = position_ids

        logits = self.runner.model(ids, kwargs_list)
        from engine.sampler import sample_next_tokens as _sampler

        first_token = int(_sampler(logits[:, -1, :], temperature=temperature, top_p=top_p).item())
        seq.append_token(first_token)
        update_states_from_kwargs(states, kwargs_list)
        seq.transition_to(SequenceStatus.RUNNING_DECODE)

        if stream:
            print(tokenizer.decode([first_token], skip_special_tokens=True), end="", flush=True)

        # Decode loop
        t0 = time.time()
        for step in range(1, max_tokens):
            decode_ids = torch.tensor(
                [[seq.output_ids[-1]]], dtype=torch.long, device=self.runner.device
            )
            current_pos = seq.total_tokens - 1
            pos_ids = torch.tensor([[current_pos]], device=self.runner.device, dtype=torch.long)

            kwargs_list = states_to_kwargs(states)
            for kw in kwargs_list:
                kw["position_ids"] = pos_ids

            logits = self.runner.model(decode_ids, kwargs_list)
            next_token = int(
                _sampler(logits[:, -1, :], temperature=temperature, top_p=top_p).item()
            )
            seq.append_token(next_token)
            update_states_from_kwargs(states, kwargs_list)

            if stream:
                text = tokenizer.decode([next_token], skip_special_tokens=True)
                print(text, end="", flush=True)

            if next_token == self.runner.eos_token_id:
                break
            if seq.total_tokens >= self.max_seq_len:
                break

        self.runner.cleanup_sequence(seq.request_id)
        output = tokenizer.decode(seq.output_ids, skip_special_tokens=True)
        elapsed = time.time() - t0
        gen_len = len(seq.output_ids)
        tps = gen_len / max(elapsed, 0.001)
        print(f"\n[Engine] done: {gen_len} tokens in {elapsed:.2f}s ({tps:.1f} tok/s)")
        return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Qwen3.5 MoE 推理引擎")
    parser.add_argument("--prompt", required=True, help="输入提示词")
    parser.add_argument("--model-dir", required=True, help="模型目录路径")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-seq-len", type=int, default=8192)
    parser.add_argument("--stream", action="store_true", help="流式输出")
    args = parser.parse_args()

    engine = Qwen35MoeEngine(
        model_dir=args.model_dir,
        max_seq_len=args.max_seq_len,
        max_new_tokens=args.max_new_tokens,
    )
    output = engine.generate(
        args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        stream=args.stream,
    )
    if not args.stream:
        print(output)


if __name__ == "__main__":
    main()
