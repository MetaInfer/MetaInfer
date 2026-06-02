# Phase 2: 批量推理 + 调度器
"""Phase 2 inference engine. No mlx_lm dependency.

Round-robin scheduler: prefills requests sequentially, then decodes
all running requests one token each per step.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any

import mlx.core as mx

from .kv_cache import KVCache, make_kv_cache
from .model import Qwen3Config
from .sampler import temperature_sample
from .tokenizer import Tokenizer
from .weights import load_qwen3_model


@dataclass
class Request:
    req_id: str
    prompt: str
    max_tokens: int
    temperature: float = 0.0
    token_ids: list[int] = field(default_factory=list)
    output_tokens: list[int] = field(default_factory=list)
    cache: list[KVCache] | None = None
    last_logits: Any = None
    status: str = "waiting"


class Scheduler:
    """Round-robin scheduler."""

    def __init__(self, model: Any, tokenizer: Tokenizer, n_layers: int) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.n_layers = n_layers
        self.waiting: list[Request] = []
        self.running: list[Request] = []

    def add_request(self, req: Request) -> None:
        req.token_ids = self.tokenizer.encode(req.prompt)
        self.waiting.append(req)

    def _prefill_one(self, req: Request) -> None:
        req.cache = make_kv_cache(self.n_layers)
        input_ids = mx.array([req.token_ids])
        req.last_logits = self.model(input_ids, cache=req.cache)
        req.status = "running"

    def _decode_one(self, req: Request) -> None:
        logits = req.last_logits
        next_id = temperature_sample(logits[0, -1, :], req.temperature)
        req.output_tokens.append(next_id)
        if len(req.output_tokens) >= req.max_tokens:
            req.status = "finished"
            req.last_logits = None
            return
        next_input = mx.array([[next_id]])
        req.last_logits = self.model(next_input, cache=req.cache)

    def step(self) -> None:
        while self.waiting:
            req = self.waiting.pop(0)
            self._prefill_one(req)
            self.running.append(req)
        still_running = []
        for req in self.running:
            if req.status == "finished":
                continue
            self._decode_one(req)
            if req.status == "running":
                still_running.append(req)
        self.running = still_running

    def all_finished(self) -> bool:
        return not self.waiting and not self.running


class InferenceEngine:
    """Phase 2: batch inference engine with scheduler."""

    def __init__(self) -> None:
        self.model: Any = None
        self.config: Qwen3Config | None = None
        self.tokenizer: Tokenizer | None = None
        self.scheduler: Scheduler | None = None

    def load_model(self, model_path: str) -> None:
        self.model, self.config = load_qwen3_model(model_path)
        self.tokenizer = Tokenizer(model_path)
        n_layers = len(self.model.layers)
        self.scheduler = Scheduler(self.model, self.tokenizer, n_layers)

    def generate(self, prompt: str, max_tokens: int = 64, temperature: float = 0.0):
        if self.scheduler is None:
            msg = "Model not loaded"
            raise RuntimeError(msg)
        req = Request(
            req_id=uuid.uuid4().hex[:8],
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        self.scheduler.add_request(req)
        while req.status != "finished":
            self.scheduler.step()
        for tok_id in req.output_tokens[:max_tokens]:
            yield self.tokenizer.decode([tok_id])

    def generate_batch(
        self, prompts: list[str], max_tokens: int = 64, temperature: float = 0.0,
    ) -> list[list[int]]:
        if self.scheduler is None:
            msg = "Model not loaded"
            raise RuntimeError(msg)
        requests = []
        for prompt in prompts:
            req = Request(
                req_id=uuid.uuid4().hex[:8],
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            self.scheduler.add_request(req)
            requests.append(req)
        while not self.scheduler.all_finished():
            self.scheduler.step()
        return [r.output_tokens[:max_tokens] for r in requests]
