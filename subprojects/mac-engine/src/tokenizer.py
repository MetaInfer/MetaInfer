# Tokenizer wrapper — wraps transformers.AutoTokenizer, zero mlx_lm dependency.
"""Tokenizer that loads from standard HuggingFace tokenizer files."""

from __future__ import annotations

from pathlib import Path


class Tokenizer:
    """Thin wrapper around transformers.AutoTokenizer."""

    def __init__(self, tokenizer_path: str) -> None:
        from transformers import AutoTokenizer

        model_dir = Path(tokenizer_path)
        if model_dir.is_file():
            model_dir = model_dir.parent

        self._tok = AutoTokenizer.from_pretrained(
            str(model_dir),
            trust_remote_code=True,
            use_fast=True,
        )

    def encode(self, text: str) -> list[int]:
        tokens = self._tok.encode(text)
        if isinstance(tokens, list):
            if tokens and isinstance(tokens[0], list):
                tokens = tokens[0]
        return tokens  # type: ignore[no-any-return]

    def decode(self, ids: list[int]) -> str:
        return self._tok.decode(ids, skip_special_tokens=True)  # type: ignore[no-any-return]

    @property
    def eos_token_id(self) -> int | None:
        return self._tok.eos_token_id

    @property
    def vocab_size(self) -> int:
        return self._tok.vocab_size
