"""Unit tests for harness.py — template generation and JSON parsing."""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from pathlib import Path
from unittest import mock

import pytest

from metainfer.tasks.evolve_kernel.orchestrator.harness import (
    _extract_json,
    build_correctness_harness_template,
    build_perf_harness_template,
    CORRECTNESS_HARNESS_TEMPLATE,
    PERF_HARNESS_TEMPLATE,
    run_correctness_test,
    run_perf_test,
)


# --------------------------------------------------------------------------- #
# Template rendering
# --------------------------------------------------------------------------- #


class TestTemplateRendering:
    def test_correctness_harness_template_fill(self):
        result = build_correctness_harness_template(
            kernel_fn_name="my_gemm_kernel",
            ref_kernel_path="/path/to/original_kernel.py",
        )
        assert "my_gemm_kernel" in result
        assert "/path/to/original_kernel.py" in result
        assert "importlib.util" in result
        assert 'def load_kernel' in result
        assert 'def main()' in result
        assert 'sys.argv[1]' in result

    def test_correctness_harness_template_custom_bodies(self):
        result = build_correctness_harness_template(
            kernel_fn_name="test_kernel",
            ref_kernel_path="/ref.py",
            test_inputs_body="    test_cases = [{'name': 'test1', 'M': 64}]",
            ref_runner_body="    output = kernel_fn(**test_case)",
            evo_runner_body="    output = kernel_fn(**test_case)",
            comparator_body="    return True, None, {'max_err': 0.0}",
        )
        assert "    test_cases = [{'name': 'test1', 'M': 64}]" in result
        assert "    return True, None, {'max_err': 0.0}" in result

    def test_perf_harness_template_fill(self):
        result = build_perf_harness_template(
            kernel_fn_name="my_kernel",
            ref_kernel_path="/ref.py",
            warmup=10,
            repeat=50,
        )
        assert "my_kernel" in result
        assert "/ref.py" in result
        assert "warmup=10" in result
        assert "repeat=50" in result
        assert "torch.cuda.Event" in result
        assert "overall_speedup" in result

    def test_harness_template_is_valid_python(self):
        """The filled templates should have valid Python syntax."""
        result = build_correctness_harness_template(
            kernel_fn_name="test_fn",
            ref_kernel_path="/tmp/ref.py",
            test_inputs_body="    test_cases = [{'name': 'case1'}]",
        )
        # Should compile without syntax errors
        try:
            compile(result, "<correctness_harness>", "exec")
        except SyntaxError as e:
            pytest.fail(f"Correctness harness template has syntax error: {e}")

        result = build_perf_harness_template(
            kernel_fn_name="test_fn",
            ref_kernel_path="/tmp/ref.py",
        )
        try:
            compile(result, "<perf_harness>", "exec")
        except SyntaxError as e:
            pytest.fail(f"Perf harness template has syntax error: {e}")

    def test_template_no_placeholder_leftovers(self):
        """After filling, no unfilled template placeholders should remain."""
        result = build_correctness_harness_template(
            kernel_fn_name="test_fn",
            ref_kernel_path="/tmp/ref.py",
        )
        import re
        # Only check for the specific template placeholders we expect to be filled
        # (the generated code may contain f-string and dict literals with {braces})
        unfilled = re.findall(
            r'\{(kernel_fn_name|ref_kernel_path|test_inputs_body|ref_runner_body|evo_runner_body|comparator_body)\}',
            result,
        )
        assert unfilled == [], f"Unfilled placeholders: {unfilled}"

        result = build_perf_harness_template(
            kernel_fn_name="test_fn",
            ref_kernel_path="/tmp/ref.py",
        )
        unfilled = re.findall(
            r'\{(kernel_fn_name|ref_kernel_path|bench_inputs_body|runner_body|warmup|repeat)\}',
            result,
        )
        assert unfilled == [], f"Unfilled perf placeholders: {unfilled}"


# --------------------------------------------------------------------------- #
# _extract_json
# --------------------------------------------------------------------------- #


class TestExtractJson:
    def test_single_json_line(self):
        text = 'some debug output\n{"passed": true, "exec_time_ms": 1.5}\nmore output'
        result = _extract_json(text)
        assert result is not None
        assert result["passed"] is True
        assert result["exec_time_ms"] == 1.5

    def test_json_not_on_last_line(self):
        """JSON can be anywhere — we scan from the end."""
        text = '{"passed": true, "perf": {"x": 1}}\nsome trailing text'
        result = _extract_json(text)
        assert result is not None
        assert result["passed"] is True

    def test_multiple_json_lines_picks_last_valid(self):
        text = (
            '{"passed": false, "error": "test1"}\n'
            '{"passed": true, "error": null}\n'
        )
        result = _extract_json(text)
        assert result is not None
        assert result["passed"] is True

    def test_no_json(self):
        text = "just some log output\nno JSON here"
        result = _extract_json(text)
        assert result is None

    def test_invalid_json_skipped(self):
        text = '{"passed": true, invalid json here}\n{"passed": true, "valid": true}'
        result = _extract_json(text)
        assert result is not None
        assert result["valid"] is True

    def test_empty_string(self):
        assert _extract_json("") is None

    def test_nested_json_object(self):
        text = '{"passed": false, "results": [{"name": "case1", "passed": false, "error": "mismatch"}]}'
        result = _extract_json(text)
        assert result is not None
        assert result["passed"] is False
        assert len(result["results"]) == 1

    def test_json_with_escaped_chars(self):
        text = '{"passed": false, "error": "max error: 0.001\\n tensor mismatch"}'
        result = _extract_json(text)
        assert result is not None
        assert "max error" in result["error"]

    def test_full_harness_output(self):
        """Parse the kind of output a real correctness harness produces."""
        text = """
Running test case: small_even...
Running test case: medium_uneven...
Running test case: large...

{"passed": true, "error": null, "total": 3, "passed_count": 3, "results": [{"index": 0, "name": "small_even", "passed": true, "error": null}, {"index": 1, "name": "medium_uneven", "passed": true, "error": null}, {"index": 2, "name": "large", "passed": true, "error": null}]}
"""
        result = _extract_json(text)
        assert result is not None
        assert result["passed"] is True
        assert result["total"] == 3
        assert result["passed_count"] == 3

    def test_perf_harness_output(self):
        text = """
Benchmarking...
{"passed": true, "ref_median_ms": 0.523, "evo_median_ms": 0.401, "overall_speedup": 1.304, "per_case": {"size_256": {"ref_median_ms": 0.5, "evo_median_ms": 0.4, "speedup": 1.25}}, "num_cases": 3}
"""
        result = _extract_json(text)
        assert result is not None
        assert result["passed"] is True
        assert result["overall_speedup"] == 1.304


# --------------------------------------------------------------------------- #
# Harness execution (integration — with mocked subprocess)
# --------------------------------------------------------------------------- #


class TestRunCorrectnessTest:
    def test_passed(self):
        harness = Path("/fake/harness.py")
        kernel = Path("/fake/kernel.py")
        output = '{"passed": true, "total": 5, "passed_count": 5, "results": []}\n'

        with mock.patch("subprocess.run") as m_run:
            m_run.return_value = mock.MagicMock(
                stdout=output, stderr="", returncode=0,
            )
            passed, result = run_correctness_test(harness, kernel, timeout_s=10)

        assert passed is True
        assert result["passed"] is True

    def test_failed_with_error(self):
        harness = Path("/fake/harness.py")
        kernel = Path("/fake/kernel.py")
        output = '{"passed": false, "error": "max_err=0.5 exceeds 1e-3", "results": [{"name": "case1", "passed": false}]}\n'

        with mock.patch("subprocess.run") as m_run:
            m_run.return_value = mock.MagicMock(
                stdout=output, stderr="", returncode=0,
            )
            passed, result = run_correctness_test(harness, kernel)

        assert passed is False
        assert "max_err" in result["error"]

    def test_timeout(self):
        harness = Path("/fake/harness.py")
        kernel = Path("/fake/kernel.py")

        with mock.patch("subprocess.run") as m_run:
            m_run.side_effect = subprocess.TimeoutExpired(cmd="python3", timeout=5)
            passed, result = run_correctness_test(harness, kernel, timeout_s=5)

        assert passed is False
        assert "timed out" in result["error"].lower()

    def test_no_valid_json(self):
        harness = Path("/fake/harness.py")
        kernel = Path("/fake/kernel.py")

        with mock.patch("subprocess.run") as m_run:
            m_run.return_value = mock.MagicMock(
                stdout="just logs\nno json here\n", stderr="", returncode=0,
            )
            passed, result = run_correctness_test(harness, kernel)

        assert passed is False
        assert "no parseable json" in result["error"].lower()

    def test_stderr_json_fallback(self):
        harness = Path("/fake/harness.py")
        kernel = Path("/fake/kernel.py")

        with mock.patch("subprocess.run") as m_run:
            m_run.return_value = mock.MagicMock(
                stdout="garbage", stderr='{"passed": true, "total": 1}\n', returncode=1,
            )
            passed, result = run_correctness_test(harness, kernel)

        assert passed is True


class TestRunPerfTest:
    def test_passed_with_timing(self):
        harness = Path("/fake/perf_harness.py")
        kernel = Path("/fake/kernel.py")
        output = '{"passed": true, "ref_median_ms": 0.5, "evo_median_ms": 0.4, "overall_speedup": 1.25}\n'

        with mock.patch("subprocess.run") as m_run:
            m_run.return_value = mock.MagicMock(
                stdout=output, stderr="", returncode=0,
            )
            ok, result = run_perf_test(harness, kernel)

        assert ok is True
        assert result["passed"] is True
        assert result["evo_median_ms"] == 0.4
        assert result["overall_speedup"] == 1.25

    def test_timeout(self):
        harness = Path("/fake/perf_harness.py")
        kernel = Path("/fake/kernel.py")

        with mock.patch("subprocess.run") as m_run:
            m_run.side_effect = subprocess.TimeoutExpired(cmd="python3", timeout=5)
            ok, result = run_perf_test(harness, kernel, timeout_s=5)

        assert ok is False
        assert "timed out" in result["error"].lower()
