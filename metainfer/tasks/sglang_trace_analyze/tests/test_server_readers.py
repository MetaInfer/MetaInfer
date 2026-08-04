"""Server state reader tests."""

import json
import tempfile
from pathlib import Path

from ..server._state_readers import (
    read_batch_detail,
    read_hints,
    read_mapping,
    read_summary,
)


def test_read_summary():
    with tempfile.TemporaryDirectory() as td:
        state_dir = Path(td)
        analysis = state_dir / "analysis"
        analysis.mkdir()
        (analysis / "summary.json").write_text(
            json.dumps({"model": "test", "batches": []})
        )
        result = read_summary(state_dir)
        assert result is not None
        assert result["model"] == "test"


def test_read_summary_missing():
    with tempfile.TemporaryDirectory() as td:
        assert read_summary(Path(td)) is None


def test_read_mapping():
    with tempfile.TemporaryDirectory() as td:
        state_dir = Path(td)
        analysis = state_dir / "analysis"
        analysis.mkdir()
        (analysis / "mapping.json").write_text(
            json.dumps({"entries": [{"kernel_name": "test"}]})
        )
        result = read_mapping(state_dir)
        assert len(result["entries"]) == 1


def test_read_hints():
    with tempfile.TemporaryDirectory() as td:
        state_dir = Path(td)
        analysis = state_dir / "analysis"
        analysis.mkdir()
        (analysis / "hints.json").write_text(
            json.dumps({"bottleneck": {"kernel_or_pattern": "triton_gemm"}})
        )
        result = read_hints(state_dir)
        assert result["bottleneck"]["kernel_or_pattern"] == "triton_gemm"


def test_read_batch_detail():
    with tempfile.TemporaryDirectory() as td:
        state_dir = Path(td)
        batch_dir = state_dir / "analysis" / "batches" / "bs_8" / "decode"
        batch_dir.mkdir(parents=True)
        (batch_dir / "kernel_table.json").write_text(json.dumps({"kernels": []}))
        (batch_dir / "overlap.json").write_text(json.dumps({"gaps": []}))
        (batch_dir / "fuse.json").write_text(json.dumps({"matches": []}))

        result = read_batch_detail(state_dir, 8, "decode")
        assert result is not None
        assert result["kernel_table"]["kernels"] == []
        assert result["overlap"]["gaps"] == []


def test_read_batch_detail_missing():
    with tempfile.TemporaryDirectory() as td:
        result = read_batch_detail(Path(td), 8, "decode")
        assert result is None
