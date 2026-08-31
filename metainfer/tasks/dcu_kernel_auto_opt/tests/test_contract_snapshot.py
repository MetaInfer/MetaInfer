from __future__ import annotations

import hashlib
import json

import pytest

from ..orchestrator.api_contracts import (
    OperatorAPIContract,
    default_optimization_shapes,
    resolve_operator_api,
    validate_contract_shapes,
)
from ..orchestrator.gen_and_opt_pipeline import _task_local_api_contract


def _api_text(shape_id: str, m: int) -> str:
    return (
        "DEFAULT_OPTIMIZATION_SHAPES = ("
        f"{{'id': '{shape_id}', 'M': {m}, 'N': 16, 'K': 32}},"
        ")\n"
        "def _check_target_shape(m, n, k):\n"
        "    if not (1 <= m <= 3072 and n == 16 and k == 32):\n"
        "        raise ValueError('unsupported')\n"
    )


def test_live_contract_accepts_hy3_tp4_shapes():
    contract = resolve_operator_api("Quantized GEMM", "INT8 W8A8")
    shapes = {
        "qkv": {
            "tp_size": 4,
            "operator": "qkv_proj",
            "M": 4096,
            "N": 2560,
            "K": 4096,
        },
        "o": {
            "tp_size": 4,
            "operator": "o_proj",
            "M": 4096,
            "N": 4096,
            "K": 2048,
        },
        "gate_up": {
            "tp_size": 4,
            "operator": "shared_gate_up_proj",
            "M": 4096,
            "N": 768,
            "K": 4096,
        },
        "down": {
            "tp_size": 4,
            "operator": "shared_down_proj",
            "M": 4096,
            "N": 4096,
            "K": 384,
        },
    }

    validate_contract_shapes(contract, shapes)


def test_live_contract_rejects_unregistered_hy3_tp4_variant():
    contract = resolve_operator_api("Quantized GEMM", "INT8 W8A8")
    shapes = {
        "qkv": {
            "tp_size": 4,
            "operator": "qkv_proj",
            "M": 4096,
            "N": 2576,
            "K": 4096,
        }
    }

    with pytest.raises(ValueError, match="requires \\(K, N\\) in"):
        validate_contract_shapes(contract, shapes)


def test_live_contract_accepts_model_catalog_tp8_m4096_shapes():
    # Model-catalog TP8 large-prefill boundary (2026-08-27): Hy3 / MiniMax M3
    # / GLM5.2 TP8 operators are optimizable at M=4096 ("Selected shapes
    # only"); DeepSeek TP8 defaults keep the original three M values.
    contract = resolve_operator_api("Quantized GEMM", "INT8 W8A8")
    shapes = {
        "hy3_tp8_qkv_proj_m4096": {
            "tp_size": 8, "operator": "qkv_proj",
            "M": 4096, "N": 1280, "K": 4096,
        },
        "hy3_tp8_o_proj_m4096": {
            "tp_size": 8, "operator": "o_proj",
            "M": 4096, "N": 4096, "K": 1024,
        },
        "hy3_tp8_shared_gate_up_proj_m4096": {
            "tp_size": 8, "operator": "shared_gate_up_proj",
            "M": 4096, "N": 384, "K": 4096,
        },
        "hy3_tp8_shared_down_proj_m4096": {
            "tp_size": 8, "operator": "shared_down_proj",
            "M": 4096, "N": 4096, "K": 192,
        },
        "minimax_tp8_qkv_proj_m4096": {
            "tp_size": 8, "operator": "qkv_proj",
            "M": 4096, "N": 1280, "K": 6144,
        },
        "minimax_tp8_qkv_proj_and_indexer_qk_m4096": {
            "tp_size": 8, "operator": "qkv_proj_and_indexer_qk",
            "M": 4096, "N": 1536, "K": 6144,
        },
        "minimax_tp8_o_proj_m4096": {
            "tp_size": 8, "operator": "o_proj",
            "M": 4096, "N": 6144, "K": 1024,
        },
        "minimax_tp8_shared_gate_up_proj_m4096": {
            "tp_size": 8, "operator": "shared_gate_up_proj",
            "M": 4096, "N": 768, "K": 6144,
        },
        "minimax_tp8_shared_down_proj_m4096": {
            "tp_size": 8, "operator": "shared_down_proj",
            "M": 4096, "N": 6144, "K": 384,
        },
        "glm52_tp8_fused_qkv_a_proj_m4096": {
            "tp_size": 8, "operator": "fused_qkv_a_proj",
            "M": 4096, "N": 2624, "K": 6144,
        },
        "glm52_tp8_q_b_proj_m4096": {
            "tp_size": 8, "operator": "q_b_proj",
            "M": 4096, "N": 2048, "K": 2048,
        },
        "glm52_tp8_kv_b_proj_m4096": {
            "tp_size": 8, "operator": "kv_b_proj",
            "M": 4096, "N": 3584, "K": 512,
        },
        "glm52_tp8_o_proj_m4096": {
            "tp_size": 8, "operator": "o_proj",
            "M": 4096, "N": 6144, "K": 2048,
        },
        "glm52_tp8_shared_gate_up_proj_m4096": {
            "tp_size": 8, "operator": "shared_gate_up_proj",
            "M": 4096, "N": 512, "K": 6144,
        },
        "glm52_tp8_shared_down_proj_m4096": {
            "tp_size": 8, "operator": "shared_down_proj",
            "M": 4096, "N": 6144, "K": 256,
        },
    }
    validate_contract_shapes(contract, shapes)


def test_default_optimization_shapes_unchanged_by_model_tp8_boundary():
    # Adding the model-catalog TP8 M=4096 boundary must not change the
    # DeepSeek-only default workload or its serial-validation fallback scope.
    from ..api.int8w8a8gemm.int8_w8a8_gemm_api import (
        DEFAULT_OPTIMIZATION_SHAPES,
        MODEL_TP8_EXTRA_OPTIMIZATION_M_VALUES,
    )

    assert MODEL_TP8_EXTRA_OPTIMIZATION_M_VALUES == (4096,)
    assert len(DEFAULT_OPTIMIZATION_SHAPES) == 42
    assert all(
        str(shape["id"]).startswith("tp") for shape in DEFAULT_OPTIMIZATION_SHAPES
    )


def test_live_contract_accepts_minimax_m3_tp4_shapes():
    # Regression for minimaxm3-dsh-tp4-m4096-1-0c2f84a9: the MiniMax M3 TP4
    # M=4096 workload stopped in prepare because TP=4 qkv_proj (K, N)
    # (6144, 2304) was outside the fixed API contract. The model catalog
    # (frontend + baseline table) must be mirrored by the contract.
    contract = resolve_operator_api("Quantized GEMM", "INT8 W8A8")
    shapes = {
        "qkv": {
            "tp_size": 4,
            "operator": "qkv_proj",
            "M": 4096,
            "N": 2304,
            "K": 6144,
        },
        "qkv_indexer": {
            "tp_size": 4,
            "operator": "qkv_proj_and_indexer_qk",
            "M": 4096,
            "N": 2560,
            "K": 6144,
        },
        "o": {
            "tp_size": 4,
            "operator": "o_proj",
            "M": 4096,
            "N": 6144,
            "K": 2048,
        },
        "gate_up": {
            "tp_size": 4,
            "operator": "shared_gate_up_proj",
            "M": 4096,
            "N": 1536,
            "K": 6144,
        },
        "down": {
            "tp_size": 4,
            "operator": "shared_down_proj",
            "M": 4096,
            "N": 6144,
            "K": 768,
        },
    }

    validate_contract_shapes(contract, shapes)


def test_live_contract_rejects_unregistered_minimax_tp4_variant():
    contract = resolve_operator_api("Quantized GEMM", "INT8 W8A8")
    shapes = {
        "qkv": {
            "tp_size": 4,
            "operator": "qkv_proj",
            "M": 4096,
            "N": 2320,
            "K": 6144,
        }
    }

    with pytest.raises(ValueError, match="requires \\(K, N\\) in"):
        validate_contract_shapes(contract, shapes)


def test_task_contract_does_not_follow_live_api_updates(tmp_path):
    live = tmp_path / "live.py"
    task = tmp_path / "task"
    task.mkdir()
    snapshot = task / "api.py"
    snapshot.write_text(_api_text("old", 16), encoding="utf-8")
    digest = hashlib.sha256(snapshot.read_bytes()).hexdigest()
    (task / "scaffold_manifest.json").write_text(
        json.dumps({"control_plane_files": {"api.py": digest}}),
        encoding="utf-8",
    )
    live.write_text(_api_text("old", 16), encoding="utf-8")
    origin = OperatorAPIContract(
        operator="Quantized GEMM",
        dtype="INT8 W8A8",
        source=live,
        destination_name="api.py",
    )

    frozen = _task_local_api_contract(origin, task)
    live.write_text(_api_text("new", 3072), encoding="utf-8")

    assert default_optimization_shapes(frozen) == [
        {"id": "old", "M": 16, "N": 16, "K": 32}
    ]


def test_task_contract_rejects_snapshot_digest_drift(tmp_path):
    source = tmp_path / "api.py"
    source.write_text(_api_text("old", 16), encoding="utf-8")
    (tmp_path / "scaffold_manifest.json").write_text(
        json.dumps({"control_plane_files": {"api.py": "wrong"}}),
        encoding="utf-8",
    )
    origin = OperatorAPIContract(
        operator="Quantized GEMM",
        dtype="INT8 W8A8",
        source=source,
        destination_name="api.py",
    )

    with pytest.raises(RuntimeError, match="digest mismatch"):
        _task_local_api_contract(origin, tmp_path)
