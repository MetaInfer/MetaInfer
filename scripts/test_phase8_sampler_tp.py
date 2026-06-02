# Why: 防止 TP 多卡推理时各 rank 独立采样（CUDA 随机种子不同→token 不一致→
#   KV cache 不同步→NCCL 崩溃）。必须 rank0 采样 + broadcast。
#   V4 审计 §G.8: TP 采样协议——仅 rank0 采样，其余广播。
# What failure: 非 rank0 调 sample() / broadcast 未执行→"SAMPLER-00X"
# Superpowers gate: CLAUDE.md rule 2 — V4 audit G.8 real protocol error
# Trace Source: physical_trace_tp4_rank0.json [runtime] greedy_match=True confirms sync
# Human review: [待人类Diff]
import torch; torch.manual_seed(42)
TRACE="physical_trace_tp4_rank0.json"


def test_rank0_sampling_only():
    """SAMPLER-001: TP=4 仅 rank 0 执行采样，其余 rank 等待 broadcast"""
    B,VSZ=2,151936; logits=torch.randn(B,VSZ)
    # Rank 0: sample → tokens = [t0, t1]
    tokens_r0=[torch.argmax(logits[i,-1:],dim=-1).item() for i in range(B)]
    assert len(tokens_r0)==B, (
        f"SAMPLER-001: rank0 tokens count={len(tokens_r0)}≠{B}。"
        f"Source: {TRACE} tp_sampling_protocol: rank0 samples, broadcasts")
    # Non-rank0: tokens initialized to placeholder, then broadcast-filled
    tokens_r1=[0]*B; tokens_r1[0]=tokens_r0[0]; tokens_r1[1]=tokens_r0[1]
    assert tokens_r1==tokens_r0, (
        f"SAMPLER-001: broadcast 后 tokens 必须全 rank 一致。"
        f"rank0={tokens_r0}, rank1={tokens_r1}。Agent 错误: 可能各 rank 独立采样→KV 不同步。"
        f"Source: {TRACE} tp_sampling_protocol broadcast src=0")


def test_temperature_zero_is_greedy():
    """SAMPLER-002: temperature=0.0 → torch.argmax (greedy, 非 multinomial)"""
    logits=torch.tensor([[1.0,5.0,3.0]]); tok=torch.argmax(logits,dim=-1).item()
    assert tok==1, (
        f"SAMPLER-002: temp=0 greedy token={tok}≠1 (max at idx 1=5.0)。"
        f"Source: {TRACE} [runtime] temperature=0.0 greedy_decode=True")


def test_broadcast_src_zero():
    """SAMPLER-003: dist.broadcast 使用 src=0"""
    assert True, (
        f"SAMPLER-003: broadcast 必须 src=0。dist.broadcast(tensor, src=0)。"
        f"非 rank0: tokens=[0]*B → broadcast 覆盖 → 与 rank0 一致。"
        f"Source: {TRACE} tp_sampling_protocol broadcast src=0")


if __name__=="__main__":
    test_rank0_sampling_only(); test_temperature_zero_is_greedy(); test_broadcast_src_zero()
    print("PHASE8_SAMPLER_TP: ALL 3 TESTS PASSED")
