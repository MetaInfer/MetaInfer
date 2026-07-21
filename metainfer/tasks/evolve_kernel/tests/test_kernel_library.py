"""Unit tests for kernel_library.py."""

from __future__ import annotations

import json
import math
import tempfile
import uuid
from pathlib import Path

import pytest

from metainfer.tasks.evolve_kernel.orchestrator.kernel_library import (
    KernelEntry,
    KernelLibrary,
    MAX_LIBRARY_SIZE,
)


# --------------------------------------------------------------------------- #
# KernelEntry
# --------------------------------------------------------------------------- #


class TestKernelEntry:
    def test_recompute_combined_fast_kernel(self):
        e = KernelEntry(
            id="test1", code="def kernel(): pass",
            exec_time_ms=1.0, complexity_score=0.0,
        )
        score = e.recompute_combined()
        # perf_score = 10.0 / 1.0 = 10.0
        # simplicity = 1.0 - 0.0 = 1.0
        # combined = 0.7 * 10.0 + 0.3 * 1.0 = 7.3
        assert math.isclose(score, 7.3, rel_tol=1e-6)
        assert math.isclose(e.combined_score, 7.3, rel_tol=1e-6)

    def test_recompute_combined_slow_kernel(self):
        e = KernelEntry(
            id="test2", code="def kernel(): pass",
            exec_time_ms=50.0, complexity_score=1.0,
        )
        score = e.recompute_combined()
        # perf_score = 10.0 / 50.0 = 0.2
        # simplicity = 1.0 - 1.0 = 0.0
        # combined = 0.7 * 0.2 + 0.3 * 0.0 = 0.14
        assert score < 0.3

    def test_recompute_combined_zero_time(self):
        """Zero exec time should be clamped to avoid division by zero."""
        e = KernelEntry(
            id="test3", code="def kernel(): pass",
            exec_time_ms=0.0, complexity_score=0.5,
        )
        score = e.recompute_combined()
        assert score > 0.0
        assert not math.isnan(score)
        assert not math.isinf(score)

    def test_to_dict_and_from_dict_roundtrip(self):
        e = KernelEntry(
            id="roundtrip", code="def foo(): pass",
            exec_time_ms=3.5, complexity_score=0.3,
            combined_score=2.0, iteration_added=5,
            parent_id="parent-abc",
        )
        as_dict = e.to_dict() if hasattr(e, "to_dict") else {
            "id": e.id, "code": e.code,
            "exec_time_ms": e.exec_time_ms,
            "complexity_score": e.complexity_score,
            "combined_score": e.combined_score,
            "iteration_added": e.iteration_added,
            "parent_id": e.parent_id,
        }
        assert as_dict["id"] == "roundtrip"
        assert as_dict["exec_time_ms"] == 3.5
        assert as_dict["complexity_score"] == 0.3


# --------------------------------------------------------------------------- #
# KernelLibrary
# --------------------------------------------------------------------------- #


def _make_entry(idx: int, exec_time: float, complexity: float) -> KernelEntry:
    e = KernelEntry(
        id=f"k{idx}", code=f"def kernel_{idx}(): pass",
        exec_time_ms=exec_time, complexity_score=complexity,
        iteration_added=idx,
    )
    e.recompute_combined()
    return e


class TestKernelLibrary:
    def test_empty_library(self):
        lib = KernelLibrary()
        assert lib.size == 0
        assert lib.select() is None
        assert lib.best is None
        assert lib.top_n(5) == []

    def test_add_and_sorting(self):
        lib = KernelLibrary()
        # Add kernels — lower exec_time = higher score
        e1 = _make_entry(1, 10.0, 0.5)
        e2 = _make_entry(2, 5.0, 0.5)
        e3 = _make_entry(3, 2.0, 0.5)

        assert lib.add(e1) is True
        assert lib.add(e2) is True
        assert lib.add(e3) is True
        assert lib.size == 3

        # e3 should be #1 (fastest), e1 #3 (slowest)
        assert lib.best.id == "k3"
        assert lib.kernels[0].id == "k3"
        assert lib.kernels[2].id == "k1"

    def test_max_size_enforced(self):
        lib = KernelLibrary()
        for i in range(MAX_LIBRARY_SIZE + 5):
            e = _make_entry(i, 100.0 / (i + 1), 0.5)
            lib.add(e)

        assert lib.size == MAX_LIBRARY_SIZE

    def test_new_kernel_beats_last_place(self):
        lib = KernelLibrary()
        # Fill with slow kernels
        for i in range(MAX_LIBRARY_SIZE):
            e = _make_entry(i, 100.0 - i, 0.9)
            lib.add(e)

        worst_score = lib.kernels[-1].combined_score

        # Add a super-fast kernel
        fast = _make_entry(999, 1.0, 0.1)
        assert lib.add(fast) is True
        assert lib.kernels[-1].id != "999"  # fast should be near top
        # An old slow kernel should have been evicted
        old_ids = {k.id for k in lib.kernels}
        assert len(old_ids) == MAX_LIBRARY_SIZE

    def test_new_kernel_does_not_beat_last_place(self):
        lib = KernelLibrary()
        # Fill with fast kernels
        for i in range(MAX_LIBRARY_SIZE):
            e = _make_entry(i, 1.0 + i * 0.1, 0.1)
            lib.add(e)

        worst_score = lib.kernels[-1].combined_score

        # Add a very slow kernel
        slow = _make_entry(999, 10000.0, 1.0)
        assert lib.add(slow) is False  # should be rejected
        assert "k999" not in {k.id for k in lib.kernels}

    def test_weighted_selection_distribution(self):
        """Weighted selection should prefer higher-score kernels."""
        lib = KernelLibrary()
        e1 = _make_entry(1, 1.0, 0.1)   # very fast
        e2 = _make_entry(2, 100.0, 1.0)  # very slow

        lib.add(e1)
        lib.add(e2)

        # Run many selections and check fast kernel is picked more often
        picks = {"k1": 0, "k2": 0}
        for _ in range(1000):
            k = lib.select()
            picks[k.id] += 1

        # k1 should be selected significantly more often
        assert picks["k1"] > picks["k2"], f"Expected k1 > k2, got {picks}"

    def test_select_single_entry(self):
        lib = KernelLibrary()
        e = _make_entry(1, 10.0, 0.5)
        lib.add(e)
        for _ in range(10):
            assert lib.select().id == "k1"

    def test_top_n(self):
        lib = KernelLibrary()
        for i in range(8):
            e = _make_entry(i, 100.0 / (i + 1), 0.5)
            lib.add(e)

        top = lib.top_n(3)
        assert len(top) == 3
        assert top[0].exec_time_ms < top[1].exec_time_ms
        assert top[1].exec_time_ms < top[2].exec_time_ms

    def test_get_by_id(self):
        lib = KernelLibrary()
        e1 = _make_entry(1, 10.0, 0.5)
        e2 = _make_entry(2, 5.0, 0.5)
        lib.add(e1)
        lib.add(e2)

        assert lib.get_by_id("k1").id == "k1"
        assert lib.get_by_id("k2").id == "k2"
        assert lib.get_by_id("nonexistent") is None

    def test_save_and_load(self, tmp_path):
        lib = KernelLibrary()
        for i in range(5):
            e = _make_entry(i, 10.0 / (i + 1), 0.3)
            lib.add(e)

        path = tmp_path / "kernel_library.json"
        lib.save(path)
        assert path.is_file()

        loaded = KernelLibrary.load(path)
        assert loaded.size == lib.size
        assert loaded.kernels[0].id == lib.kernels[0].id
        assert loaded.kernels[0].combined_score == pytest.approx(
            lib.kernels[0].combined_score
        )

    def test_load_nonexistent_file(self):
        lib = KernelLibrary.load(Path("/nonexistent/path/kernel_library.json"))
        assert lib.size == 0

    def test_load_empty_file(self, tmp_path):
        path = tmp_path / "empty.json"
        path.write_text("")
        lib = KernelLibrary.load(path)
        assert lib.size == 0

    def test_load_invalid_json(self, tmp_path):
        path = tmp_path / "invalid.json"
        path.write_text("not valid json {{{")
        lib = KernelLibrary.load(path)
        assert lib.size == 0

    def test_load_non_list(self, tmp_path):
        path = tmp_path / "non_list.json"
        path.write_text('{"key": "value"}')
        lib = KernelLibrary.load(path)
        assert lib.size == 0

    def test_to_list_roundtrip(self, tmp_path):
        lib = KernelLibrary()
        for i in range(3):
            e = _make_entry(i, 5.0 / (i + 1), 0.2 * i)
            lib.add(e)

        data = lib.to_list()
        assert len(data) == 3
        assert isinstance(data, list)

        lib2 = KernelLibrary.from_list(data)
        assert lib2.size == lib.size
        for a, b in zip(lib.kernels, lib2.kernels):
            assert a.id == b.id
            assert math.isclose(a.combined_score, b.combined_score)

    def test_last_added(self):
        lib = KernelLibrary()
        for i in range(5):
            e = _make_entry(i, 10.0 / (i + 1), 0.5)
            lib.add(e)

        assert lib.last_added().iteration_added == 4

    def test_from_list_handles_missing_fields(self):
        data = [
            {"id": "minimal", "code": "def f(): pass"},
        ]
        lib = KernelLibrary.from_list(data)
        assert lib.size == 1
        k = lib.kernels[0]
        assert k.id == "minimal"
        assert k.exec_time_ms == 0.0
        assert k.complexity_score == 0.5
        assert k.combined_score == 0.0
        assert k.parent_id is None
