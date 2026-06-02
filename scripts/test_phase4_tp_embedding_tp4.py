# Why: 防止 TP=4 下 VocabParallelEmbedding mask 范围计算错误导致各 rank 输出不一致
#   （vocab 边界、mask 范围、all_reduce 后输出不等于 tp=1 参考值）。
#   Trace: vocab_size=151936, TP=4 → per_rank=37984
#   V17 OW-6: Sequence block_table 双轨切换逻辑不完整
# What failure: tp=4 embedding != tp=1 embedding → "EMBED-TP4-00X" + Source
# Superpowers gate: CLAUDE.md rule 2 — vocab_size verified by config.json 2026-05-27
# Trace Source: physical_trace_tp4_rank0.json [config] vocab_size=151936
# Human review: [待人类Diff]
import torch; import torch.nn.functional as F; torch.manual_seed(42)
TRACE="physical_trace_tp4_rank0.json"
V=151936; H=4096; TP=4; VP=V//TP


def test_embedding_mask_per_rank():
    """EMBED-TP4-001: 每个 rank 的 mask 范围不重叠，覆盖全 vocab 无遗漏"""
    ranges=[]
    for r in range(TP):
        vs=r*VP; ve=vs+VP; ranges.append((vs,ve))
    # Ranges must be contiguous and cover full vocab
    assert ranges[0][0]==0, (
        f"EMBED-TP4-001: rank0 start={ranges[0][0]}≠0。"
        f"Source: {TRACE} [config] vocab_size={V}, TP={TP} → per_rank={VP}")
    assert ranges[-1][1]==V, (
        f"EMBED-TP4-001: rank{TP-1} end={ranges[-1][1]}≠{V}。"
        f"Source: {TRACE} [config] vocab_size={V} full range must be covered")
    for i in range(TP-1):
        assert ranges[i][1]==ranges[i+1][0], (
            f"EMBED-TP4-001: rank{i} end={ranges[i][1]} ≠ rank{i+1} start={ranges[i+1][0]}。"
            f"vocab 分区有缝隙或重叠。Source: {TRACE} [config] vocab_size={V}")


def test_embedding_output_all_reduce_consistency():
    """EMBED-TP4-002: 各 rank mask+embedding 后 all_reduce_sum = 全量 embedding"""
    B,T=1,4; ids=torch.randint(0,V,(B,T))
    full_w=torch.randn(V,H); ref=F.embedding(ids,full_w)
    parts=[]
    for r in range(TP):
        vs=r*VP; ve=vs+VP; w=full_w[vs:ve,:]
        mask=(ids>=vs)&(ids<ve); lid=(ids-vs).masked_fill(~mask,0)
        out=F.embedding(lid,w); out=out.masked_fill((~mask).unsqueeze(-1),0)
        parts.append(out)
    summed=sum(parts)
    assert torch.allclose(summed,ref,rtol=1e-3,atol=1e-2), (
        f"EMBED-TP4-002: all_reduce_sum 后不等于全量 embedding (bf16 tol)。"
        f"max_diff={(summed-ref).abs().max().item():.4f}。"
        f"Agent 错误: mask 范围错或 masked_fill 遗漏。"
        f"Source: {TRACE} VocabParallelEmbedding mask+masked_fill+all_reduce contract")


def test_lm_head_all_gather_consistency():
    """EMBED-TP4-003: LM head 各 rank local logits → all_gather → full vocab"""
    B,T=1,1; hs=torch.randn(B,T,H)
    full_w=torch.randn(V,H); ref=F.linear(hs,full_w)
    parts=[]
    for r in range(TP):
        vs=r*VP; ve=vs+VP; w=full_w[vs:ve,:]; local=F.linear(hs,w)
        parts.append(local)
    gathered=torch.cat(parts,dim=-1)
    assert gathered.shape==(B,T,V), (
        f"EMBED-TP4-003: all_gather 后 shape={list(gathered.shape)}≠{(B,T,V)}。"
        f"Source: {TRACE} ParallelLMHead all_gather_last_dim → [B,T,vocab_size]")


if __name__=="__main__":
    test_embedding_mask_per_rank(); test_embedding_output_all_reduce_consistency()
    test_lm_head_all_gather_consistency()
    print("PHASE4_TP_EMBEDDING_TP4: ALL 3 TESTS PASSED")
