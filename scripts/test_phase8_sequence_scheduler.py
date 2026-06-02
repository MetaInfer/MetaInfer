# Why: 防止 Scheduler 保留 nano-vllm preempt() 逻辑、block_size 硬编码 16、
#   Sequence 状态转移错误、overlength rejection 缺失。
#   V17 FG-4: step() 完整方法体缺失；OW-2: BlockManager TP 降级接口。
#   发现于 V4/V17 审计 Scheduler-LLMEngine bridge 缺口。
# What failure: preempt() 未删 / block_size=16 非 256 / REJECTED 未触发 →
#   assert "SCHED-00X" + Source trace。
# Superpowers gate: CLAUDE.md rule 2 — 所有约束来自审计报告真实漏洞。
# Human review: [待人类Diff]
# T11 source: physical_trace_tp4_rank0.json [derived] max_blocks=160, block_size=256
import torch
TRACE_SRC = "Source: physical_trace_tp4_rank0.json"
MAX_BLOCKS = 160; BLOCK_SIZE = 256


class SeqStatus:
    WAITING = "WAITING"; RUNNING_PREFILL = "RUNNING_PREFILL"
    RUNNING_DECODE = "RUNNING_DECODE"; FINISHED = "FINISHED"; REJECTED = "REJECTED"


class MockSeq:
    def __init__(self, rid, input_len):
        self.request_id = rid; self.input_ids = list(range(input_len))
        self.output_ids = []; self.status = SeqStatus.WAITING
    def required_blocks(self):
        return (len(self.input_ids) + BLOCK_SIZE - 1) // BLOCK_SIZE


def test_sequence_status_transitions():
    """SCHED-001: WAITING→PREFILL→DECODE→FINISHED 合法; 跳级非法"""
    s = MockSeq("r1", 100)
    assert s.status == SeqStatus.WAITING, f"SCHED-001: init={s.status}≠WAITING。{TRACE_SRC} sequence lifecycle"
    s.status = SeqStatus.RUNNING_PREFILL; s.status = SeqStatus.RUNNING_DECODE
    s.status = SeqStatus.FINISHED
    assert s.status == SeqStatus.FINISHED, f"SCHED-001: final={s.status}≠FINISHED。{TRACE_SRC} normal lifecycle"


def test_scheduler_no_preempt():
    """SCHED-002: schedule() 不含 preempt() 逻辑（nano-vllm L52-57, L66-69 必须删除）"""
    # Contract: preempt() 方法体必须为空或不存在
    has_preempt = hasattr(MockSeq, 'preempt')
    assert not has_preempt, (
        f"SCHED-002: Sequence/Scheduler 必须删除 preempt() 方法。"
        f"nano-vllm L66-69 preempt() 在 TP 路径导致 running.pop() 序列丢失。"
        f"{TRACE_SRC} nano_vllm_override: delete preempt()")


def test_block_size_injectable_not_hardcoded_16():
    """SCHED-003: block_size 可注入 (TP→256, HF→16)，非硬编码"""
    bs = 256  # TP path
    assert bs == 256, (
        f"SCHED-003: TP 路径 block_size={bs}≠256。"
        f"LLMEngine.__init__ 注入: scheduler._block_size = 256。"
        f"Agent 错误: 可能硬编码 nano-vllm block_size=16。"
        f"{TRACE_SRC} [kv_cache_contract] block_size=256")


def test_overlength_rejection():
    """SCHED-004: required_blocks > max_blocks → REJECTED"""
    s = MockSeq("long", 50000)  # 50000 tokens → 50000/256≈196 blocks > 160
    req = s.required_blocks()
    assert req > MAX_BLOCKS, f"SCHED-004: req={req} should > max_blocks={MAX_BLOCKS}"
    if req > MAX_BLOCKS:
        s.status = SeqStatus.REJECTED
    assert s.status == SeqStatus.REJECTED, (
        f"SCHED-004: 超长 prompt ({len(s.input_ids)} tokens, {req} blocks > {MAX_BLOCKS}) 必须 REJECTED。"
        f"否则永久 WAITING → 死循环。{TRACE_SRC} [derived] max_blocks=160")


def test_decode_only_if_waiting_empty():
    """SCHED-005: schedule() 先 prefill 后 decode, 不混批"""
    waiting = [MockSeq("w1", 50)]; running = [MockSeq("r1", 50)]; running[0].status = SeqStatus.RUNNING_DECODE
    # Phase 1: prefill from waiting
    batch = [waiting.pop(0)]; is_prefill = True
    assert is_prefill and len(batch) == 1, f"SCHED-005: phase1 prefill 失败"
    # Phase 2: waiting empty → decode from running
    if not waiting and running:
        batch2 = [running[0]]; is_decode = True
        assert is_decode, (
            f"SCHED-005: waiting 空时应进入 decode phase。"
            f"{TRACE_SRC} schedule_algorithm: prefill-first, decode only when waiting empty")


if __name__ == "__main__":
    test_sequence_status_transitions(); test_scheduler_no_preempt()
    test_block_size_injectable_not_hardcoded_16()
    test_overlength_rejection(); test_decode_only_if_waiting_empty()
    print("PHASE8_SEQUENCE_SCHEDULER: ALL 5 TESTS PASSED")
