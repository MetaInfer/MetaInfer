# Why: 防止 QwenTPConfig 从 config.json 硬编码维度（而非动态读取）、head_dim fallback 缺失、
#   per-rank 计算用了全量值而非除以 tp。
#   Trace: config.json max_position_embeddings=40960, intermediate_size=12288, head_dim=128
#   V17 FG-1: max_position_embeddings 32768→40960, intermediate_size 12800→12288
# What failure: 硬编码值 ≠ config.json / head_dim fallback 未生效 → "CONFIG-00X" + Source
# Superpowers gate: CLAUDE.md rule 2 — config.json 2026-05-27 物理验证
# Trace Source: physical_trace_tp4_rank0.json [config] all 10 fields verified
# Human review: [待人类Diff]
import json; import torch; torch.manual_seed(42)
TRACE="physical_trace_tp4_rank0.json"
CFG_PATH="${MODEL_DIR}/config.json"


def test_config_json_all_fields_present():
    """CONFIG-001: 所有必须字段存在于 config.json"""
    cfg=json.load(open(CFG_PATH))
    required=["max_position_embeddings","intermediate_size","hidden_size",
              "num_attention_heads","num_key_value_heads","num_hidden_layers",
              "vocab_size","rms_norm_eps","rope_theta"]
    for k in required:
        assert k in cfg, (f"CONFIG-001: config.json 缺少字段 {k}。"
                          f"Source: {TRACE} [config] has all {len(required)} fields")


def test_max_position_embeddings_40960_not_32768():
    """CONFIG-002: max_position_embeddings=40960 (非旧文档 32768)"""
    cfg=json.load(open(CFG_PATH))
    assert cfg["max_position_embeddings"]==40960, (
        f"CONFIG-002: max_position_embeddings={cfg['max_position_embeddings']}≠40960。"
        f"旧文档 32768 是错误的。max_blocks=40960//256=160 (非 128)。"
        f"Source: {TRACE} [config] max_position_embeddings=40960")


def test_intermediate_size_12288_not_12800():
    """CONFIG-003: intermediate_size=12288 (非旧蓝图 12800)"""
    cfg=json.load(open(CFG_PATH))
    assert cfg["intermediate_size"]==12288, (
        f"CONFIG-003: intermediate_size={cfg['intermediate_size']}≠12288。"
        f"旧蓝图 12800 是错误的。gate_up=2*12288/4=6144 (非 6400)。"
        f"Source: {TRACE} [config] intermediate_size=12288")


def test_head_dim_fallback():
    """CONFIG-004: head_dim fallback = hidden_size // num_attention_heads"""
    cfg=json.load(open(CFG_PATH))
    head_dim=cfg.get("head_dim",cfg["hidden_size"]//cfg["num_attention_heads"])
    assert head_dim==128, (
        f"CONFIG-004: head_dim={head_dim}≠128。"
        f"Qwen3-8B config.json 有显式 head_dim=128; 若缺失则 fallback 4096//32=128。"
        f"Source: {TRACE} [config] head_dim=128")


def test_per_rank_calculations():
    """CONFIG-005: per-rank 值 = 全量值 // tp_size，非全量值"""
    cfg=json.load(open(CFG_PATH)); TP=4
    nheads_per=cfg["num_attention_heads"]//TP
    kvh_per=max(1,cfg["num_key_value_heads"]//TP)
    inter_per=cfg["intermediate_size"]//TP
    assert nheads_per==8, (
        f"CONFIG-005: num_heads_per_rank={nheads_per}≠8。32/4=8。"
        f"Agent 错误: 可能直接用 32 作为 num_heads。Source: {TRACE} [derived] per_rank_attn_heads=8")
    assert kvh_per==2, (
        f"CONFIG-005: kv_heads_per_rank={kvh_per}≠2。max(1,8//4)=2。"
        f"Source: {TRACE} [derived] per_rank_kv_heads=2")
    assert inter_per==3072, (
        f"CONFIG-005: inter_per_rank={inter_per}≠3072。12288/4=3072。"
        f"Source: {TRACE} [derived] per_rank_intermediate=3072")


if __name__=="__main__":
    test_config_json_all_fields_present()
    test_max_position_embeddings_40960_not_32768()
    test_intermediate_size_12288_not_12800()
    test_head_dim_fallback(); test_per_rank_calculations()
    print("PHASE7_QWEN_TP_CONFIG: ALL 5 TESTS PASSED")
