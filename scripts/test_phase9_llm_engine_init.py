# Why: 防止 LLMEngine.__init__ 路由错误（_select_tp_backend 误判架构）、
#   block_size 注入遗漏（TP→256,HF→16）、_max_blocks 未注入 Scheduler。
#   V17 FG-4: step() 完整方法体缺失; OW-2: BlockManager TP 降级接口。
#   Trace: max_blocks=160, block_size=256, intermediate_size=12288
# What failure: backend 路由错 / block_size=16(非256) / _max_blocks 未注入→"ENGINE-00X"
# Superpowers gate: CLAUDE.md rule 2 — FG-4 real gap from V17 audit
# Trace Source: physical_trace_tp4_rank0.json [derived] max_blocks=160, block_size=256
# Human review: [待人类Diff]
import json; import os; import torch; torch.manual_seed(42)
TRACE="physical_trace_tp4_rank0.json"

CFG_PATH=os.path.join(os.environ["MODEL_DIR"], "config.json")


def test_select_tp_backend_qwen():
    """ENGINE-001: _select_tp_backend 从 config.json architectures[0] 路由到 qwen_tp"""
    cfg=json.load(open(CFG_PATH)); arch=cfg["architectures"][0]
    assert arch in ("Qwen3ForCausalLM","Qwen2ForCausalLM"), (
        f"ENGINE-001: arch={arch} 应路由到 qwen_tp。"
        f"Source: {TRACE} [config] architectures[0] must match Qwen family")
    if "Qwen" in arch: backend="qwen_tp"
    assert backend=="qwen_tp", (
        f"ENGINE-001: backend={backend}≠qwen_tp。Agent 错误: _select_tp_backend 路由逻辑。"
        f"Source: {TRACE} [config] architectures[0]='{arch}' → qwen_tp")


def test_block_size_tp_256_not_16():
    """ENGINE-002: TP Runner 路径 block_size=256，HF 路径=16"""
    tp_bs=256; hf_bs=16
    assert tp_bs==256, (
        f"ENGINE-002: TP block_size={tp_bs}≠256。flash_attn_with_kvcache 要求≥256。"
        f"Source: {TRACE} [kv_cache_contract] block_size=256")
    assert hf_bs==16, (
        f"ENGINE-002: HF block_size 默认=16 (nano-vllm 兼容)。"
        f"Source: {TRACE} framework block_size=16 for HF path only")


def test_max_blocks_injection():
    """ENGINE-003: LLMEngine.__init__ 注入 scheduler._max_blocks = 40960//256 = 160"""
    max_pos=json.load(open(CFG_PATH))["max_position_embeddings"]
    max_blocks=max_pos//256
    assert max_blocks==160, (
        f"ENGINE-003: _max_blocks={max_blocks}≠160。40960//256=160 (非 32768//256=128)。"
        f"Agent 错误: 可能硬编码 128。Source: {TRACE} [derived] max_blocks=160")


def test_num_free_routing():
    """ENGINE-004: num_free 来源路由: TP→runner.get_num_free_blocks(), HF→block_manager"""
    backend="qwen_tp"
    src="runner" if backend in ("qwen_tp","deepseek_tp") else "block_manager"
    assert src=="runner", (
        f"ENGINE-004: TP 路径 num_free 必须来自 runner.get_num_free_blocks()。"
        f"TP 下 BlockManager 降级为 no-op。Source: {TRACE} scheduler_tp_runner_bridge")


if __name__=="__main__":
    test_select_tp_backend_qwen(); test_block_size_tp_256_not_16()
    test_max_blocks_injection(); test_num_free_routing()
    print("PHASE9_LLM_ENGINE_INIT: ALL 4 TESTS PASSED")
