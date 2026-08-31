from __future__ import annotations

import pytest

from ..orchestrator import variant_store as vs
from ..orchestrator.variant_store import (
    derive_variant_meta,
    dtype_slug,
    model_slug,
    operator_type_slug,
    parse_shape_meta,
)


def test_slugs():
    assert operator_type_slug("Quantized GEMM") == "gemm"
    assert operator_type_slug("RMSNorm / LayerNorm") == "rmsnorm"
    assert dtype_slug("INT8 W8A8") == "int8w8a8"
    assert dtype_slug("FP16 / BF16") == "fp16bf16"
    assert model_slug("Hy3 (Hunyuan 3)") == "hy3"
    assert model_slug("DeepSeek V4 Flash") == "deepseek-v4"
    assert model_slug("MiniMax M3") == "minimax-m3"
    assert model_slug("GLM5.2") == "glm52"


def test_parse_shape_meta():
    assert parse_shape_meta("hy3_tp4_o_proj_m4096") == {
        "tp": 4, "operator": "o_proj", "m": 4096,
    }
    assert parse_shape_meta("hy3_tp4_shared_gate_up_proj_m4096") == {
        "tp": 4, "operator": "shared_gate_up_proj", "m": 4096,
    }
    assert parse_shape_meta("tp4_wqkv_a_m2") == {
        "tp": 4, "operator": "wqkv_a", "m": 2,
    }


def test_derive_variant_meta():
    answers = {"operator": "Quantized GEMM", "dtype": "INT8 W8A8", "model": "Hy3 (Hunyuan 3)"}
    meta = derive_variant_meta(answers, "hy3_tp4_o_proj_m4096")
    assert meta["family"] == "int8w8a8-gemm"
    assert meta["model"] == "hy3"
    assert meta["tp"] == 4
    assert meta["operator_name"] == "o_proj"
    assert meta["m"] == 4096


def test_variant_path_matches_requested_layout(tmp_path, monkeypatch):
    monkeypatch.setattr(vs, "variant_root", lambda: tmp_path)
    answers = {"operator": "Quantized GEMM", "dtype": "INT8 W8A8", "model": "Hy3 (Hunyuan 3)"}
    meta = derive_variant_meta(answers, "hy3_tp4_o_proj_m4096")
    path = vs.variant_path(meta)
    assert path == tmp_path / "int8w8a8-gemm" / "hy3" / "TP4" / "M4096" / "o_proj.hip"


def test_add_variant_writes_and_replaces(tmp_path, monkeypatch):
    monkeypatch.setattr(vs, "variant_root", lambda: tmp_path)
    answers = {"operator": "Quantized GEMM", "dtype": "INT8 W8A8", "model": "Hy3 (Hunyuan 3)"}
    meta = derive_variant_meta(answers, "hy3_tp4_o_proj_m4096")
    r1 = vs.add_variant(
        meta=meta, kernel_source="KERNEL_OLD", commit="abc",
        metrics={"median_us": 803.9, "logical_tops": 85.48, "speedup": 24.73},
        source_task="task-1",
    )
    assert r1["action"] == "added"
    target = vs.variant_path(meta)
    assert target.is_file()
    assert "KERNEL_OLD" in target.read_text()
    assert "@@variant shape=hy3_tp4_o_proj_m4096" in target.read_text()
    assert "median_us=803.9" in target.read_text()

    r2 = vs.add_variant(
        meta=meta, kernel_source="KERNEL_NEW", commit="def",
        metrics={"median_us": 800.0},
        source_task="task-1",
    )
    assert r2["action"] == "updated"
    assert r2["backup"] is not None
    text = target.read_text()
    assert "KERNEL_NEW" in text
    assert "KERNEL_OLD" not in text


def test_list_variant_index_walks_tree(tmp_path, monkeypatch):
    monkeypatch.setattr(vs, "variant_root", lambda: tmp_path)
    answers = {"operator": "Quantized GEMM", "dtype": "INT8 W8A8", "model": "Hy3 (Hunyuan 3)"}
    vs.add_variant(
        meta=derive_variant_meta(answers, "hy3_tp4_o_proj_m4096"),
        kernel_source="A", commit="x", metrics={},
    )
    vs.add_variant(
        meta=derive_variant_meta(answers, "hy3_tp4_qkv_proj_m4096"),
        kernel_source="B", commit="y", metrics={},
    )
    index = vs.list_variant_index()
    shapes = {item["shape"] for item in index}
    assert shapes == {"hy3_tp4_o_proj_m4096", "hy3_tp4_qkv_proj_m4096"}
    o_proj = next(item for item in index if item["shape"] == "hy3_tp4_o_proj_m4096")
    assert o_proj["family"] == "int8w8a8-gemm"
    assert o_proj["model"] == "hy3"
    assert o_proj["tp"] == "TP4"
    assert o_proj["m"] == "M4096"
    assert o_proj["operator"] == "o_proj"


def test_parse_variant_header_extracts_metrics():
    text = (
        "// @@variant shape=hy3_tp4_o_proj_m16 commit=abc123 added=2026-08-26\n"
        "//   median_us=18.22 p90_us=18.24 speedup=3.0\n"
        "//   source=hy3-dsh-tp4-m16-1-7f1fb1d1\n"
        "// body comments like K=4096 must not surface\n"
        "// @@end\n"
        "kernel source...\n"
    )
    parsed = vs._parse_variant_header(text)
    assert parsed["shape"] == "hy3_tp4_o_proj_m16"
    assert parsed["commit"] == "abc123"
    assert parsed["source"] == "hy3-dsh-tp4-m16-1-7f1fb1d1"
    assert parsed["median_us"] == 18.22
    assert parsed["p90_us"] == 18.24
    assert parsed["speedup"] == 3.0
    assert "K" not in parsed
    assert "shape" not in parsed or parsed["shape"] != "m16"


def test_list_variant_index_surfaces_header_metrics(tmp_path, monkeypatch):
    monkeypatch.setattr(vs, "variant_root", lambda: tmp_path)
    target = (
        tmp_path / "int8w8a8-gemm" / "hy3" / "TP4" / "M16" / "qkv_proj.hip"
    )
    target.parent.mkdir(parents=True)
    target.write_text(
        "// @@variant shape=hy3_tp4_qkv_proj_m16 commit=abc added=2026-08-26\n"
        "//   median_us=26.05 p90_us=26.09\n"
        "//   source=task-1\n"
        "// @@end\nkernel\n",
        encoding="utf-8",
    )
    index = vs.list_variant_index()
    assert len(index) == 1
    v = index[0]
    assert v["shape"] == "hy3_tp4_qkv_proj_m16"
    assert v["median_us"] == 26.05
    assert v["p90_us"] == 26.09
    assert v["source"] == "task-1"
    assert v["commit"] == "abc"
    assert v["operator"] == "qkv_proj"


def _add_with_guard(tmp_path, monkeypatch, median_us):
    monkeypatch.setattr(vs, "variant_root", lambda: tmp_path)
    answers = {"operator": "Quantized GEMM", "dtype": "INT8 W8A8", "model": "Hy3 (Hunyuan 3)"}
    meta = derive_variant_meta(answers, "hy3_tp4_qkv_proj_m16")
    return vs.add_variant(
        meta=meta,
        kernel_source="KERNEL",
        commit="c",
        metrics={"median_us": median_us},
        source_task="task-x",
        reject_slower_than_existing=True,
    )


def test_reject_slower_replacement_guard(tmp_path, monkeypatch):
    assert _add_with_guard(tmp_path, monkeypatch, 26.05)["action"] == "added"
    # faster candidate replaces
    assert _add_with_guard(tmp_path, monkeypatch, 24.30)["action"] == "updated"
    # equal candidate replaces
    assert _add_with_guard(tmp_path, monkeypatch, 24.30)["action"] == "updated"
    # strictly slower candidate is rejected
    with pytest.raises(ValueError, match="rejecting slower variant replacement"):
        _add_with_guard(tmp_path, monkeypatch, 30.0)
    # the rejected replacement did not overwrite the variant
    text = vs.variant_path(
        derive_variant_meta(
            {"operator": "Quantized GEMM", "dtype": "INT8 W8A8", "model": "Hy3 (Hunyuan 3)"},
            "hy3_tp4_qkv_proj_m16",
        )
    ).read_text()
    assert "median_us=24.3" in text or "median_us=24.30" in text


def test_reject_slower_guard_skips_when_median_unknown(tmp_path, monkeypatch):
    monkeypatch.setattr(vs, "variant_root", lambda: tmp_path)
    answers = {"operator": "Quantized GEMM", "dtype": "INT8 W8A8", "model": "Hy3 (Hunyuan 3)"}
    meta = derive_variant_meta(answers, "hy3_tp4_o_proj_m16")
    vs.add_variant(
        meta=meta, kernel_source="OLD", commit="a",
        metrics={}, source_task="task-1",
    )
    # existing has no median in header; candidate median present -> no guard
    result = vs.add_variant(
        meta=meta, kernel_source="NEW", commit="b",
        metrics={"median_us": 999.0}, source_task="task-2",
        reject_slower_than_existing=True,
    )
    assert result["action"] == "updated"
    assert "NEW" in vs.variant_path(meta).read_text()
