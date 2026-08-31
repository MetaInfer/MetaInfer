from __future__ import annotations

import hashlib
import json

import pytest

from ..orchestrator.api_contracts import (
    OperatorAPIContract,
    default_optimization_shapes,
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
