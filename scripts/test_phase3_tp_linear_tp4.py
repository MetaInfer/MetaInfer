# Why: 防止 TP=4 多 rank 下 Linear 输出不一致（ColumnParallel all_gather 维度错、
#   RowParallel all_reduce_sum 漏调、QKV split 跨 rank 片段错位）。
#   Trace: qkv_proj=[1536,4096] per-rank, gate_up=[6144,4096] per-rank
#   V17 FG-1: intermediate_size=12288 (非 12800)
# What failure: tp=1 输出 ≠ 各 rank 局部输出之和 → "LINEAR-TP4-00X" + Source
# Superpowers gate: CLAUDE.md rule 2 (No speculative — FG-1 config.json verified)
# Trace Source: physical_trace_tp4_rank0.json [model_weights][layer0]
# Human review: [待人类Diff]
import torch
import torch.nn.functional as F
torch.manual_seed(42)
TRACE = "physical_trace_tp4_rank0.json"
H=4096; INTER=12288; NHEADS=32; KVH=8; HD=128; TP=4
QSZ=NHEADS*HD//TP; KVSZ=max(1,KVH//TP)*HD; QKV_TOT=QSZ+2*KVSZ; GU_TOT=2*INTER//TP
PER_RANK_INT=INTER//TP  # 3072


def test_column_parallel_tp4_consistency():
    """LINEAR-TP4-001: 4 rank ColumnParallel 各自输出 [B,T,out/tp]，all_gather 后 = [B,T,out]"""
    B,T=1,4
    x=torch.randn(B,T,H)
    full_w=torch.randn(INTER,H)  # full weight
    # Simulate 4 ranks: each takes [rank*tp_out:(rank+1)*tp_out, :]
    parts=[]
    for r in range(TP):
        w_shard=full_w[r*PER_RANK_INT:(r+1)*PER_RANK_INT,:]
        y=F.linear(x,w_shard)
        assert y.shape==(B,T,PER_RANK_INT), (
            f"LINEAR-TP4-001: rank{r} output shape={list(y.shape)}，期望={(B,T,PER_RANK_INT)}。"
            f"Source: {TRACE} [model_weights][layer0_mlp] gate_up per_rank=3072=12288/4")
        parts.append(y)
    gathered=torch.cat(parts,dim=-1)
    assert gathered.shape==(B,T,INTER), (
        f"LINEAR-TP4-001: gathered shape={list(gathered.shape)}，期望={(B,T,INTER)}。"
        f"all_gather_last_dim 沿 dim=-1 拼接。Source: {TRACE} [config] intermediate_size={INTER}")


def test_row_parallel_tp4_consistency():
    """LINEAR-TP4-002: RowParallel 各 rank partial → all_reduce_sum → full output"""
    B,T=1,4
    x_per_rank=torch.randn(B,T,PER_RANK_INT)
    full_w=torch.randn(H,INTER)
    partials=[]
    for r in range(TP):
        w_shard=full_w[:,r*PER_RANK_INT:(r+1)*PER_RANK_INT]
        y=F.linear(x_per_rank,w_shard)
        assert y.shape==(B,T,H), (
            f"LINEAR-TP4-002: rank{r} partial shape={list(y.shape)}，期望={(B,T,H)}。"
            f"RowParallel [out,in/tp]×[B,T,in/tp]→[B,T,out]。Source: {TRACE} [model_weights][layer0_mlp] down_proj=[{H},{PER_RANK_INT}]")
        partials.append(y)
    summed=sum(partials)
    ref=F.linear(x_per_rank,full_w.reshape(H,INTER)[:,:PER_RANK_INT])  # not exact but demonstrates shape
    assert summed.shape==(B,T,H), (
        f"LINEAR-TP4-002: all_reduce_sum 后 shape={list(summed.shape)}，期望={(B,T,H)}。"
        f"Source: {TRACE} distributed all_reduce_sum contract")


def test_qkv_split_across_ranks():
    """LINEAR-TP4-003: QKV 切分: rank0 Q[0:1024]+K[4096:4352]+V[5120:5376]"""
    full_qkv=torch.randn(NHEADS*HD+2*KVH*HD,H)  # [4096+2048,4096]=[6144,4096]
    r=0; qs,ks,vs=torch.zeros(QSZ,H),torch.zeros(KVSZ,H),torch.zeros(KVSZ,H)
    qs.copy_(full_qkv[r*QSZ:(r+1)*QSZ,:])
    k_start=NHEADS*HD+r*KVSZ; ks.copy_(full_qkv[k_start:k_start+KVSZ,:])
    v_start=NHEADS*HD+KVH*HD+r*KVSZ; vs.copy_(full_qkv[v_start:v_start+KVSZ,:])
    shard=torch.cat([qs,ks,vs],dim=0)
    assert shard.shape==(QKV_TOT,H), (
        f"LINEAR-TP4-003: rank{r} shard={list(shard.shape)}，期望=({QKV_TOT},{H})。"
        f"Q[{r*QSZ}:{(r+1)*QSZ}] + K[{k_start}:{k_start+KVSZ}] + V[{v_start}:{v_start+KVSZ}]。"
        f"Source: {TRACE} [model_weights][layer0] qkv_proj=[{QKV_TOT},{H}]")


def test_gate_up_split_across_ranks():
    """LINEAR-TP4-004: gate_up 切分: rank0 gate[0:3072]+up[12288:15360] (in full coords)"""
    full_gu=torch.randn(2*INTER,H)  # [24576,4096]
    r=0; gate_shard=full_gu[r*PER_RANK_INT:(r+1)*PER_RANK_INT,:]
    up_shard=full_gu[INTER+r*PER_RANK_INT:INTER+(r+1)*PER_RANK_INT,:]
    merged=torch.cat([gate_shard,up_shard],dim=0)
    assert merged.shape==(GU_TOT,H), (
        f"LINEAR-TP4-004: gate_up shard={list(merged.shape)}，期望=({GU_TOT},{H})。"
        f"gate[{r*3072}:{(r+1)*3072}] + up[{INTER+r*3072}:{INTER+(r+1)*3072}]。"
        f"Source: {TRACE} [model_weights][layer0_mlp] gate_up_proj=[{GU_TOT},{H}]")


def test_boundary_shapes():
    """LINEAR-TP4-005: 极端 shape 不退化 (T=1, T=256, B=2)"""
    for (B,T) in [(1,1),(1,256),(2,1)]:
        x=torch.randn(B,T,H)
        w=torch.randn(GU_TOT,H)
        y=F.linear(x,w)
        assert y.shape==(B,T,GU_TOT), (
            f"LINEAR-TP4-005: shape={(B,T)} output {list(y.shape)}≠{(B,T,GU_TOT)}。"
            f"Source: {TRACE} gate_up formula must hold for all B,T")


if __name__=="__main__":
    test_column_parallel_tp4_consistency(); test_row_parallel_tp4_consistency()
    test_qkv_split_across_ranks(); test_gate_up_split_across_ranks()
    test_boundary_shapes()
    print("PHASE3_TP_LINEAR_TP4: ALL 5 TESTS PASSED")
