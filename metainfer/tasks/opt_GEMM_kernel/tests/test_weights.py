from __future__ import annotations

import json

import pytest

from ..orchestrator.evaluator.spec import SpecError
from ..orchestrator.evaluator.weights import FrozenWeightBundle


def test_weight_directory_is_snapshotted_and_detects_mutation(tmp_path):
    source = tmp_path / "model_weights"
    source.mkdir()
    (source / "info.json").write_text(json.dumps({"w": {"shape": [1]}}), encoding="utf-8")
    (source / "w.bin").write_bytes(b"x")

    frozen = FrozenWeightBundle.materialize(source, tmp_path / "state" / "system_weights")
    assert frozen.root != source
    frozen.verify()
    (frozen.root / "w.bin").write_bytes(b"y")
    with pytest.raises(SpecError, match="changed"):
        frozen.verify()


def test_weight_directory_requires_info_json(tmp_path):
    source = tmp_path / "model_weights"
    source.mkdir()
    with pytest.raises(SpecError, match="info.json"):
        FrozenWeightBundle.materialize(source, tmp_path / "frozen")
