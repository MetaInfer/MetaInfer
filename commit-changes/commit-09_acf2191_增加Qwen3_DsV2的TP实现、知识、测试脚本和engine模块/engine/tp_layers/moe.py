from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from engine.tp_layers.distributed import all_reduce_sum, get_tp_rank, get_tp_size


@dataclass
class ExpertParallelMoEConfig:
    hidden_size: int
    intermediate_size: int
    num_experts: int
    top_k: int
    routed_scaling_factor: float = 1.0
    score_function: str = "softmax"


def partition_experts_for_rank(num_experts: int, tp_size: int, tp_rank: int) -> list[int]:
    base = num_experts // tp_size
    rem = num_experts % tp_size
    start = tp_rank * base + min(tp_rank, rem)
    size = base + (1 if tp_rank < rem else 0)
    return list(range(start, start + size))


class _ExpertMLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class ExpertParallelMoE(nn.Module):
    """
    简化 EP: router 全量复制；每 rank 仅计算本地专家；最后 all_reduce 求和聚合。
    """

    def __init__(self, cfg: ExpertParallelMoEConfig):
        super().__init__()
        self.cfg = cfg
        self.tp_rank = get_tp_rank()
        self.tp_size = get_tp_size()
        self.local_expert_ids = partition_experts_for_rank(cfg.num_experts, self.tp_size, self.tp_rank)

        # Router replicated on all ranks.
        self.gate = nn.Linear(cfg.hidden_size, cfg.num_experts, bias=False)
        self.experts = nn.ModuleDict(
            {str(i): _ExpertMLP(cfg.hidden_size, cfg.intermediate_size) for i in self.local_expert_ids}
        )

    def _router_topk(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        bsz, seqlen, hidden = hidden_states.shape
        logits = F.linear(
            hidden_states.reshape(-1, hidden).float(),
            self.gate.weight.float(),
            None,
        )
        if self.cfg.score_function == "sigmoid":
            scores = torch.sigmoid(logits)
        else:
            scores = torch.softmax(logits, dim=-1, dtype=torch.float32)
        topk_weight, topk_idx = torch.topk(scores, k=self.cfg.top_k, dim=-1, sorted=False)
        topk_weight = topk_weight * self.cfg.routed_scaling_factor
        return topk_idx, topk_weight.to(hidden_states.dtype)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        hidden_states: [B, T, H]
        return: [B, T, H]
        """
        bsz, seqlen, hidden = hidden_states.shape
        flat = hidden_states.reshape(-1, hidden)
        topk_idx, topk_weight = self._router_topk(hidden_states)

        local_out = torch.zeros_like(flat)
        topk_idx_flat = topk_idx.reshape(-1, self.cfg.top_k)
        topk_weight_flat = topk_weight.reshape(-1, self.cfg.top_k)

        for token_i in range(flat.shape[0]):
            x = flat[token_i : token_i + 1]
            for k in range(self.cfg.top_k):
                expert_id = int(topk_idx_flat[token_i, k].item())
                key = str(expert_id)
                if key not in self.experts:
                    continue
                expert = self.experts[key]
                w = topk_weight_flat[token_i, k]
                local_out[token_i : token_i + 1] += expert(x) * w

        global_out = all_reduce_sum(local_out)
        return global_out.view(bsz, seqlen, hidden)

