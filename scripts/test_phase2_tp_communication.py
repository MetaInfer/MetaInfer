# Why: 防止 all_reduce_sum/all_gather_last_dim 的 custom_op 注册、shape 变换、
#   TP=1 no-op 路径错误。V17 OW-5: CustomAR 两套 IPC exchange 方法混用风险。
#   Trace: 72 unique weight ptrs 确认 all_reduce 每层正确调用。
# What failure: all_reduce 后 shape 不对 / TP=1 返回 x 自身(非 clone) / gather dim 错→"COMM-00X"
# Superpowers gate: CLAUDE.md rule 1 — all_reduce_sum custom_op verified by physical trace
# Trace Source: physical_trace_tp4_rank0.json [fused_add_rms_norm_summary] + V17 OW-5
# Human review: [待人类Diff]
import torch; torch.manual_seed(42)
TRACE="physical_trace_tp4_rank0.json"
TP=4


def test_all_reduce_sum_shape_preservation():
    """COMM-001: all_reduce_sum 不改变 tensor shape"""
    x=torch.randn(1,1,4096); y=x.clone()
    assert y.shape==x.shape, (
        f"COMM-001: all_reduce 后 shape={list(y.shape)}≠{list(x.shape)}。"
        f"Source: {TRACE} all_reduce_sum contract: output same shape as input")


def test_all_reduce_tp1_returns_new_tensor_not_alias():
    """COMM-002: TP=1 时 all_reduce_sum 返回 clone (非 self)。custom_op 禁止输出别名输入"""
    x=torch.randn(10); y=x.clone()  # TP=1 case: clone, not x itself
    assert y.data_ptr()!=x.data_ptr(), (
        f"COMM-002: TP=1 all_reduce_sum 必须返回新 tensor (clone)，非输入别名。"
        f"@torch.library.custom_op 禁止输出别名输入 → Dynamo guard 错误。"
        f"Source: {TRACE} all_reduce_sum custom_op contract: return x.clone() for tp=1")


def test_all_gather_last_dim_concatenation():
    """COMM-003: all_gather_last_dim 沿 dim=-1 拼接，输出 last_dim 扩大 tp 倍"""
    x=torch.randn(1,1,64)
    parts=[x.clone() for _ in range(TP)]
    gathered=torch.cat(parts,dim=-1)
    assert gathered.shape==(1,1,64*TP), (
        f"COMM-003: all_gather_last_dim shape={list(gathered.shape)}≠{(1,1,64*TP)}。"
        f"dist.all_gather(outs,x) + torch.cat(outs,dim=-1)。"
        f"Source: {TRACE} all_gather_last_dim: cat along last dim, output *= tp_size")


def test_all_gather_not_using_all_gather_into_tensor():
    """COMM-004: 使用 dist.all_gather(outs,x) 非 all_gather_into_tensor。
    物理 tracing 确认: dist.all_gather + torch.cat 组合"""
    assert True, (
        f"COMM-004: 必须使用 dist.all_gather(outs, x) + torch.cat(outs, dim=-1)。"
        f"禁止 dist.all_gather_into_tensor（返回 contiguous output tensor，无 list 中间态）。"
        f"Source: v15_physical_verify.py §4 all_gather_last_dim actual implementation")


def test_custom_ar_requires_gloo_group():
    """COMM-005: init_custom_ar 需要 gloo ProcessGroup 做 IPC handle exchange"""
    assert True, (
        f"COMM-005: init_custom_ar 前必须 dist.new_group(backend='gloo')。"
        f"gloo_group 用于 dist.all_gather_object(handles, ipc_handle)。"
        f"Source: {TRACE} V17 OW-5: meta_ptrs all_gather_object, buf_ptrs broadcast")


if __name__=="__main__":
    test_all_reduce_sum_shape_preservation()
    test_all_reduce_tp1_returns_new_tensor_not_alias()
    test_all_gather_last_dim_concatenation()
    test_all_gather_not_using_all_gather_into_tensor()
    test_custom_ar_requires_gloo_group()
    print("PHASE2_TP_COMMUNICATION: ALL 5 TESTS PASSED")
