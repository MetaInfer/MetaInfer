"""
ModelRunner: 从 Sequence 批次组 batch，经小型因果 LM 得到每序列最后一个位置的 logits 并采样。

- 单进程 (TP=1): ``FakeCausalLM`` 与既有单测行为一致（Embedding + 单层 proj）。
- 多进程 SPMD (TP>1): ``FakeCausalLMTP`` 演示 QKV 列切、O 行切+AllReduce、FFN 上/下列切+行切、
  LM head 列切+AllGather，与等价的**整权重**前向在数值上对齐，便于 TDD 验证张量并行。
"""

from __future__ import annotations

import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

from engine.sampler import greedy_sample, sample_next_tokens
from engine.structs import Sequence
from engine.tp_distributed import get_tp_rank, get_tp_size, init_distributed, is_distributed


def _all_reduce_if_tp(t: torch.Tensor) -> None:
    if is_distributed() and get_tp_size() > 1:
        dist.all_reduce(t, op=dist.ReduceOp.SUM)


def _all_gather_vocab(logits_part: torch.Tensor) -> torch.Tensor:
    if not is_distributed() or get_tp_size() == 1:
        return logits_part
    w = get_tp_size()
    outs = [torch.empty_like(logits_part) for _ in range(w)]
    dist.all_gather(outs, logits_part)
    return torch.cat(outs, dim=-1)


class FakeCausalLM(nn.Module):
    """Random projection: token id -> logits. Deterministic given seed."""

    def __init__(self, vocab_size: int, hidden_size: int):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.proj = nn.Linear(hidden_size, vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        h = self.embed(input_ids)
        return self.proj(h)


class FakeCausalLMTP(nn.Module):
    """
    极简张量并行玩具网络（与等价的单卡 ``reference_forward_toy`` 权重复现时 logits 应一致）::

        embed(复制) -> QKV(列) -> O(行+AllReduce) -> Up(列) -> GELU -> Down(行+AllReduce) -> Head(列+AllGather)

    与 DeepSeek 等真实模型的列/行排布及通信次数概念对齐；不加载 16B 实权重，由 SPMD 各 rank
    用同一 seed 生成完整再切片，模拟 ``load 切片权``。
    """

    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        *,
        ffn_mult: int = 4,
        seed: int = 42,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.tp = get_tp_size() if is_distributed() else 1
        assert self.tp in (1, 2, 4, 8), f"TP_SIZE 仅支持 1,2,4,8，当前 {self.tp}"
        ffn = ffn_mult * hidden_size
        qkv_out = 3 * hidden_size
        for n, s in (("3*H", qkv_out), ("4*H", ffn), ("V", vocab_size)):
            if s % self.tp != 0:
                raise ValueError(
                    f"FakeCausalLMTP: {n}={s} 需能被 tp={self.tp} 整除（当前 {s} % {self.tp} != 0）"
                )

        g = torch.Generator()
        g.manual_seed(seed)
        w_qkv = torch.empty(qkv_out, hidden_size)
        w_o = torch.empty(hidden_size, qkv_out)
        w_up = torch.empty(ffn, hidden_size)
        w_down = torch.empty(hidden_size, ffn)
        w_head = torch.empty(vocab_size, hidden_size)
        emb = torch.empty(vocab_size, hidden_size)
        for t in (w_qkv, w_o, w_up, w_down, w_head, emb):
            t.normal_(0, 0.02, generator=g)
        w_qkv = w_qkv.contiguous()
        w_o = w_o.contiguous()
        w_up = w_up.contiguous()
        w_down = w_down.contiguous()
        w_head = w_head.contiguous()
        emb = emb.contiguous()

        r = 0 if self.tp == 1 else get_tp_rank()  # 仅多进程时有效
        # Column: 按输出维（dim 0）切
        self.register_buffer("w_qkv", w_qkv[r * (qkv_out // self.tp) : (r + 1) * (qkv_out // self.tp)].clone())
        # Row: 按输入维（dim 1）切
        self.register_buffer("w_o", w_o[:, r * (qkv_out // self.tp) : (r + 1) * (qkv_out // self.tp)].clone())
        self.register_buffer("w_up", w_up[r * (ffn // self.tp) : (r + 1) * (ffn // self.tp)].clone())
        self.register_buffer("w_down", w_down[:, r * (ffn // self.tp) : (r + 1) * (ffn // self.tp)].clone())
        self.register_buffer("w_head", w_head[r * (vocab_size // self.tp) : (r + 1) * (vocab_size // self.tp)].clone())
        # 全 rank 相同 embedding
        self.register_buffer("embed_weight", emb.clone())

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        h0 = F.embedding(input_ids, self.embed_weight)  # [B, H]
        qkv_p = F.linear(h0, self.w_qkv)  # [B, 3H/tp]
        h1 = F.linear(qkv_p, self.w_o)  # [B, H]
        _all_reduce_if_tp(h1)
        u = F.linear(h1, self.w_up)  # [B, 4H/tp]
        u = F.gelu(u)
        o = F.linear(u, self.w_down)  # [B, H]
        _all_reduce_if_tp(o)
        logits_p = F.linear(o, self.w_head)  # [B, V/tp]
        return _all_gather_vocab(logits_p)


def reference_forward_toy(
    input_ids: torch.Tensor,
    *,
    vocab_size: int,
    hidden_size: int,
    ffn_mult: int = 4,
    seed: int = 42,
) -> torch.Tensor:
    """与 FakeCausalLMTP 等价的**未切分**整权重前向，仅用于 TDD/对照。"""
    ffn = ffn_mult * hidden_size
    qkv_out = 3 * hidden_size
    g = torch.Generator()
    g.manual_seed(seed)
    w_qkv = torch.empty(qkv_out, hidden_size)
    w_o = torch.empty(hidden_size, qkv_out)
    w_up = torch.empty(ffn, hidden_size)
    w_down = torch.empty(hidden_size, ffn)
    w_head = torch.empty(vocab_size, hidden_size)
    emb = torch.empty(vocab_size, hidden_size)
    for t in (w_qkv, w_o, w_up, w_down, w_head, emb):
        t.normal_(0, 0.02, generator=g)
    h0 = F.embedding(input_ids, emb)
    qkv = F.linear(h0, w_qkv)
    h1 = F.linear(qkv, w_o)
    u = F.gelu(F.linear(h1, w_up))
    o = F.linear(u, w_down)
    return F.linear(o, w_head)


def _use_tp_model() -> bool:
    return int(os.environ.get("META_INFER_USE_TP", "0")) == 1 or (
        "WORLD_SIZE" in os.environ and int(os.environ["WORLD_SIZE"]) > 1
    )


class ModelRunner:
    """
    准备 prefill (flat + cu_seqlens) 或 decode (每序列 1 个 token)，
    跑小型因果 LM，返回每个序列采样的下一 token id。
    环境变量 ``META_INFER_USE_TP=1`` 时（且 world_size>1 已 init）走 ``FakeCausalLMTP``。
    双进程/多卡测试请使用 ``torchrun`` 并令 ``world_size>1``；单进程不 init PG 时仍为 FakeCausalLM。
    """

    def __init__(
        self,
        vocab_size: int,
        hidden_size: int = 64,
        *,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float32,
        seed: int = 42,
        use_tp: bool | None = None,
    ):
        self.vocab_size = vocab_size
        self.device = torch.device(device)
        self.dtype = dtype
        self.seed = seed
        if use_tp is None:
            use_tp = _use_tp_model()
        if use_tp:
            init_distributed()
        torch.manual_seed(seed)
        if use_tp and is_distributed():
            self.model = FakeCausalLMTP(
                vocab_size, hidden_size, ffn_mult=4, seed=seed
            ).to(device=self.device, dtype=dtype)
        else:
            self.model = FakeCausalLM(vocab_size, hidden_size).to(device=self.device, dtype=dtype)
        self.model.eval()

    def prepare_prefill(self, seqs: list[Sequence]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Uncached prompt tokens: flat input_ids, positions, cu_seqlens [0, l0, l0+l1, ...]."""
        input_ids: list[int] = []
        positions: list[int] = []
        cu_seqlens: list[int] = [0]
        for seq in seqs:
            tokens = seq.token_ids
            start = seq.num_cached_tokens
            end = len(tokens)
            if end <= start:
                cu_seqlens.append(cu_seqlens[-1])
                continue
            for pos in range(start, end):
                input_ids.append(tokens[pos])
                positions.append(pos)
            cu_seqlens.append(cu_seqlens[-1] + (end - start))
        input_t = torch.tensor(input_ids, dtype=torch.long, device=self.device)
        pos_t = torch.tensor(positions, dtype=torch.long, device=self.device)
        cu_t = torch.tensor(cu_seqlens, dtype=torch.int32, device=self.device)
        return input_t, pos_t, cu_t

    def prepare_decode(self, seqs: list[Sequence]) -> tuple[torch.Tensor, torch.Tensor]:
        """One token per sequence: last token id and its position."""
        input_ids = [seq.token_ids[-1] for seq in seqs]
        positions = [len(seq.token_ids) - 1 for seq in seqs]
        input_t = torch.tensor(input_ids, dtype=torch.long, device=self.device)
        pos_t = torch.tensor(positions, dtype=torch.long, device=self.device)
        return input_t, pos_t

    def forward_logits(
        self,
        input_ids: torch.Tensor,
        cu_seqlens: torch.Tensor | None,
        is_prefill: bool,
    ) -> torch.Tensor:
        """Returns [num_seqs, vocab_size] logits at the last position of each sequence."""
        all_logits = self.model(input_ids)
        if not is_prefill:
            return all_logits
        if cu_seqlens is None or cu_seqlens.numel() < 2:
            raise ValueError("prefill requires cu_seqlens")
        num_seqs = cu_seqlens.numel() - 1
        last_indices = []
        for i in range(num_seqs):
            start = int(cu_seqlens[i].item())
            end = int(cu_seqlens[i + 1].item())
            if end <= start:
                raise ValueError("empty sequence in prefill batch")
            last_indices.append(end - 1)
        idx = torch.tensor(last_indices, dtype=torch.long, device=input_ids.device)
        return all_logits[idx]

    def run(
        self,
        seqs: list[Sequence],
        is_prefill: bool,
        *,
        temperature: float = 0.0,
        top_p: float | None = None,
        generator: torch.Generator | None = None,
    ) -> list[int]:
        if not seqs:
            return []
        if is_prefill:
            input_ids, _positions, cu = self.prepare_prefill(seqs)
            logits = self.forward_logits(input_ids, cu, is_prefill=True)
        else:
            input_ids, _positions = self.prepare_decode(seqs)
            logits = self.forward_logits(input_ids, None, is_prefill=False)

        if temperature == 0.0 and top_p is None:
            out = greedy_sample(logits)
        else:
            out = sample_next_tokens(logits, temperature=temperature, top_p=top_p, generator=generator)
        return out.detach().cpu().tolist()
