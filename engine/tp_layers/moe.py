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
    EP: router replicated; each rank owns a subset of experts; all_reduce combines.
    P5b: GPU-side expert_map eliminates .item() — follows vLLM's deadlock-free pattern.
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

        # P5b: GPU-side expert_map [num_global_experts] → local_idx or -1
        # This tensor lives on GPU and is registered so .to(device) moves it.
        self._expert_map: torch.Tensor | None = None

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

    def _ensure_expert_map(self, device: torch.device) -> torch.Tensor:
        if self._expert_map is None or self._expert_map.device != device:
            m = torch.full((self.cfg.num_experts,), -1, dtype=torch.int32, device=device)
            for local_idx, global_eid in enumerate(self.local_expert_ids):
                m[global_eid] = local_idx
            self._expert_map = m
        return self._expert_map

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        hidden_states: [B, T, H]
        return: [B, T, H]
        """
        bsz, seqlen, hidden = hidden_states.shape
        flat = hidden_states.reshape(-1, hidden)
        num_tokens = flat.shape[0]
        topk_idx, topk_weight = self._router_topk(hidden_states)

        local_out = torch.zeros_like(flat)
        topk_idx_flat = topk_idx.reshape(-1, self.cfg.top_k)  # [N, top_k]
        topk_weight_flat = topk_weight.reshape(-1, self.cfg.top_k)

        # P5b: hybrid strategy based on token count
        #   num_tokens > 4 → batched GPU-side (nonzero+index_add_, no .item())
        #   num_tokens ≤ 4 → per-token .item() loop (lower overhead for decode)
        if num_tokens > 4:
            expert_map = self._ensure_expert_map(flat.device)
            local_eids = expert_map[topk_idx_flat.long()]  # [N, top_k], -1=not local
            for local_idx, global_eid in enumerate(self.local_expert_ids):
                mask = (local_eids == local_idx)
                if not mask.any():
                    continue
                token_idx, k_idx = mask.nonzero(as_tuple=True)
                weights = topk_weight_flat[token_idx, k_idx].unsqueeze(-1)
                expert_out = self.experts[str(global_eid)](flat[token_idx])
                local_out.index_add_(0, token_idx, expert_out * weights)
        else:
            for token_i in range(num_tokens):
                x = flat[token_i : token_i + 1]
                for k in range(self.cfg.top_k):
                    expert_id = int(topk_idx_flat[token_i, k].item())
                    key = str(expert_id)
                    if key not in self.experts:
                        continue
                    w = topk_weight_flat[token_i, k]
                    local_out[token_i : token_i + 1] += self.experts[key](x) * w

        global_out = all_reduce_sum(local_out)
        return global_out.view(bsz, seqlen, hidden)

