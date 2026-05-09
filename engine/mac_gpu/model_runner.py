"""
Apple MPS 模型运行器：基于 HuggingFace AutoModelForCausalLM + KV Cache。
使用 PyTorch 原生 SDPA（MPS 后端支持），无需 Flash Attention。
"""

from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from engine.sampler import sample_next_tokens
from engine.structs import Sequence


class MPSModelRunner:
    def __init__(self, model_name_or_path: str) -> None:
        if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
            raise RuntimeError("MPS backend not available on this device")

        self.device = torch.device("mps")
        self.dtype = torch.float16

        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        print(f"[MPSModelRunner] Loading model {model_name_or_path} ...")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=self.dtype,
            trust_remote_code=True,
        ).to(self.device)
        self.model.eval()
        print(
            f"[MPSModelRunner] Model loaded: device={self.device}, dtype={self.dtype}, "
            f"params={sum(p.numel() for p in self.model.parameters()) / 1e6:.1f}M"
        )

    @property
    def eos_token_id(self) -> int:
        return self.tokenizer.eos_token_id

    @torch.inference_mode()
    def run_prefill(self, seqs: list[Sequence]) -> list[int]:
        """处理完整 prompt，缓存 KV，返回每个序列的第一个生成 token。"""
        next_tokens = []
        for seq in seqs:
            ids = torch.tensor([seq.token_ids], dtype=torch.long, device=self.device)
            out = self.model(input_ids=ids, use_cache=True, return_dict=True)
            seq.past_key_values = out.past_key_values
            logits = out.logits[0, -1, :].unsqueeze(0)
            tid = sample_next_tokens(
                logits,
                temperature=seq.sampling_params.get("temperature", 0.0),
                top_p=seq.sampling_params.get("top_p"),
            ).item()
            next_tokens.append(int(tid))
        return next_tokens

    @torch.inference_mode()
    def run_decode(self, seqs: list[Sequence]) -> list[int]:
        """Decode 步骤：用 KV Cache，每步只传入新 token + 缓存的 past_key_values。"""
        next_tokens = []
        for seq in seqs:
            new_id = torch.tensor([[seq.output_ids[-1]]], dtype=torch.long, device=self.device)
            seq_len = len(seq.input_ids) + len(seq.output_ids) - 1
            attn_mask = torch.ones([1, seq_len + 1], dtype=torch.long, device=self.device)

            out = self.model(
                input_ids=new_id,
                attention_mask=attn_mask,
                past_key_values=seq.past_key_values,
                use_cache=True,
                return_dict=True,
            )
            seq.past_key_values = out.past_key_values
            logits = out.logits[:, -1, :]

            tid = sample_next_tokens(
                logits,
                temperature=seq.sampling_params.get("temperature", 0.0),
                top_p=seq.sampling_params.get("top_p"),
            ).item()
            next_tokens.append(int(tid))
        return next_tokens

    def run(self, seqs: list[Sequence], is_prefill: bool) -> list[int]:
        if is_prefill:
            return self.run_prefill(seqs)
        return self.run_decode(seqs)
