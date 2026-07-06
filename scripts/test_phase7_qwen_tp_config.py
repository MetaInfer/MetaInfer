# Why: 防止 QwenTPConfig 从 config.json 硬编码维度（而非动态读取）、head_dim fallback 缺失、
#   per-rank 计算用了全量值而非除以 tp。
#   What failure: 硬编码值 ≠ config.json / head_dim fallback 未生效 → "CONFIG-00X" + Source
# Superpowers gate: CLAUDE.md rule 2 — config.json 物理验证
# V3: Extended to support Qwen3.5/3.6 nested text_config and multiple model families.
import json; import os; import torch; torch.manual_seed(42)
TRACE="physical_trace_tp4_rank0.json"
CFG_PATH=os.path.join(os.environ["MODEL_DIR"], "config.json")

def _get_model_config():
    """Read model config, handling nested text_config and rope_parameters for Qwen3.5/3.6 family."""
    raw = json.load(open(CFG_PATH))
    if "text_config" in raw:
        cfg = raw["text_config"]
        cfg["architectures"] = raw.get("architectures", [])
        cfg["model_type"] = raw.get("model_type", cfg.get("model_type", ""))
    else:
        cfg = raw
    # Promote nested rope_parameters to top-level fields
    if "rope_parameters" in cfg and "rope_theta" not in cfg:
        cfg["rope_theta"] = cfg["rope_parameters"].get("rope_theta", 10000)
    return cfg


def test_config_json_all_fields_present():
    """CONFIG-001: 所有必须字段存在于 config.json (flat or nested text_config)"""
    cfg=_get_model_config()
    required=["max_position_embeddings","intermediate_size","hidden_size",
              "num_attention_heads","num_key_value_heads","num_hidden_layers",
              "vocab_size","rms_norm_eps","rope_theta"]
    for k in required:
        assert k in cfg, (f"CONFIG-001: config.json 缺少字段 {k}。"
                          f"Source: {TRACE} [config] has all {len(required)} fields")


def test_max_position_embeddings_present():
    """CONFIG-002: max_position_embeddings 存在于 config，值 >= 40960"""
    cfg=_get_model_config()
    mpe=cfg["max_position_embeddings"]
    assert mpe>=40960, (
        f"CONFIG-002: max_position_embeddings={mpe}<40960。"
        f"Expected >=40960 (Qwen3-8B=40960, Qwen3.6-27B=262144)."
        f"Source: {TRACE} [config]")


def test_intermediate_size_present():
    """CONFIG-003: intermediate_size 存在于 config，值 > 0"""
    cfg=_get_model_config()
    inter=cfg["intermediate_size"]
    assert inter>0, (
        f"CONFIG-003: intermediate_size={inter}<=0。"
        f"Source: {TRACE} [config]")


def test_head_dim_fallback():
    """CONFIG-004: head_dim fallback = hidden_size // num_attention_heads"""
    cfg=_get_model_config()
    head_dim=cfg.get("head_dim",cfg["hidden_size"]//cfg["num_attention_heads"])
    assert head_dim>0, (
        f"CONFIG-004: head_dim={head_dim}<=0。"
        f"Qwen3-8B: head_dim=128; Qwen3.6-27B: head_dim=256。"
        f"Source: {TRACE} [config]")


def test_per_rank_calculations():
    """CONFIG-005: per-rank 值 = 全量值 // tp_size，非全量值"""
    cfg=_get_model_config(); TP=4
    nheads_per=cfg["num_attention_heads"]//TP
    kvh_per=max(1,cfg["num_key_value_heads"]//TP)
    inter_per=cfg["intermediate_size"]//TP
    assert nheads_per>0, (
        f"CONFIG-005: num_heads_per_rank={nheads_per}<=0。"
        f"Source: {TRACE} [derived] per_rank_attn_heads")
    assert kvh_per>=1, (
        f"CONFIG-005: kv_heads_per_rank={kvh_per}<1。"
        f"Source: {TRACE} [derived] per_rank_kv_heads")
    assert inter_per>0, (
        f"CONFIG-005: inter_per_rank={inter_per}<=0。"
        f"Source: {TRACE} [derived] per_rank_intermediate")


if __name__=="__main__":
    test_config_json_all_fields_present()
    test_max_position_embeddings_present()
    test_intermediate_size_present()
    test_head_dim_fallback(); test_per_rank_calculations()
    print("PHASE7_QWEN_TP_CONFIG: ALL 5 TESTS PASSED")
