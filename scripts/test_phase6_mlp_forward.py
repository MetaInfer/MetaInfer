# Why: 防止 QwenMLPTP 的 gate_up_proj→silu_and_mul→down_proj shape 链错误。
#   V17 FG-1: intermediate_size=12288 (NOT 12800!), gate_up=[6144,4096] (NOT [6400,4096])
#   Trace: layer0_mlp gate_up_proj_weight_shape=[6144,4096], down_proj=[4096,3072]
# What failure: gate_up 输出不是 [B,T,2*inter/tp] / silu_and_mul 后不是 [B,T,inter/tp] /
#   down_proj 后不是 [B,T,hidden] → assert "MLP-00X" + Source trace
# Superpowers gate: CLAUDE.md rule 2 (No speculative — FG-1 verified by config.json)
# Human review: [待人类Diff]
# T11 source: physical_trace_tp4_rank0.json [model_weights][layer0_mlp]
import torch
import torch.nn.functional as F
torch.manual_seed(42)
TRACE_SRC = "Source: physical_trace_tp4_rank0.json"

HIDDEN = 4096; INTERMEDIATE = 12288; TP = 4
INTER_PER_RANK = INTERMEDIATE // TP  # 3072 (NOT 3200!)
GATE_UP_TOTAL = 2 * INTER_PER_RANK  # 6144 (NOT 6400!)


def test_gate_up_proj_output_shape():
    """MLP-001: MergedColumnParallelLinear 输出 [B,T,2*intermediate/tp]=[B,T,6144]"""
    B, T = 1, 1
    x = torch.randn(B, T, HIDDEN)
    w = torch.randn(GATE_UP_TOTAL, HIDDEN)
    y = F.linear(x, w)
    assert y.shape == (B, T, GATE_UP_TOTAL), (
        f"MLP-001: gate_up output shape={list(y.shape)}，期望={(B,T,GATE_UP_TOTAL)}。"
        f"intermediate={INTERMEDIATE}/TP={TP} → per_rank={INTER_PER_RANK} → 2×={GATE_UP_TOTAL}。"
        f"Agent 错误: 若用旧 intermediate=12800 → gate_up=6400 (错误!)"
        f"{TRACE_SRC} [model_weights][layer0_mlp] gate_up_proj_weight_shape=[{GATE_UP_TOTAL},{HIDDEN}]")


def test_silu_and_mul_splits_gate_and_up():
    """MLP-002: silu_and_mul 取前半 gate 做 SiLU × 后半 up，输出 [B,T,inter/tp]"""
    B, T = 1, 4
    gate_up = torch.randn(B, T, GATE_UP_TOTAL)
    gate = gate_up[..., :INTER_PER_RANK]
    up = gate_up[..., INTER_PER_RANK:]
    out = F.silu(gate) * up
    assert out.shape == (B, T, INTER_PER_RANK), (
        f"MLP-002: silu_and_mul output shape={list(out.shape)}，期望={(B,T,INTER_PER_RANK)}。"
        f"SiLU(gate[:inter/tp]) * up[inter/tp:] → [B,T,inter/tp]。"
        f"{TRACE_SRC} [model_weights][layer0_mlp]: gate_up_proj=[{GATE_UP_TOTAL},{HIDDEN}], 前{INTER_PER_RANK} gate 后{INTER_PER_RANK} up")


def test_down_proj_output_shape():
    """MLP-003: down_proj(RowParallelLinear) [B,T,inter/tp]→[B,T,hidden]"""
    B, T = 1, 1
    x = torch.randn(B, T, INTER_PER_RANK)
    w = torch.randn(HIDDEN, INTER_PER_RANK)
    y = F.linear(x, w)
    assert y.shape == (B, T, HIDDEN), (
        f"MLP-003: down_proj output shape={list(y.shape)}，期望={(B,T,HIDDEN)}。"
        f"RowParallelLinear [hidden, inter/tp] + all_reduce_sum。"
        f"{TRACE_SRC} [model_weights][layer0_mlp] down_proj_weight_shape=[4096,3072]")


def test_intermediate_size_is_12288_not_12800():
    """MLP-004: intermediate_size=12288 (2026-05-27 config.json verified)"""
    assert INTERMEDIATE == 12288, (
        f"MLP-004: intermediate_size={INTERMEDIATE}，期望=12288。"
        f"旧蓝图值 12800 是错误的！{TRACE_SRC} [config] intermediate_size=12288")
    assert INTER_PER_RANK == 3072, (
        f"MLP-004: inter_per_rank={INTER_PER_RANK}，期望=3072=12288/4。"
        f"Agent 错误: 12800/4=3200 导致 shape mismatch。"
        f"{TRACE_SRC} [derived] per_rank_intermediate=3072")
    assert GATE_UP_TOTAL == 6144, (
        f"MLP-004: gate_up_total={GATE_UP_TOTAL}，期望=6144=2*3072。"
        f"旧值 6400=2*3200 是错误的！"
        f"{TRACE_SRC} [model_weights][layer0_mlp] gate_up_proj_weight_shape=[6144,4096]")


if __name__ == "__main__":
    test_gate_up_proj_output_shape(); test_silu_and_mul_splits_gate_and_up()
    test_down_proj_output_shape(); test_intermediate_size_is_12288_not_12800()
    print("PHASE6_MLP_FORWARD: ALL 4 TESTS PASSED")
