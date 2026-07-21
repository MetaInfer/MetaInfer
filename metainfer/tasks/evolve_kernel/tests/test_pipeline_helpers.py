"""Unit tests for pipeline helper functions and prompts."""

from __future__ import annotations

import json
import math
import tempfile
from pathlib import Path

import pytest

from metainfer.tasks.evolve_kernel.orchestrator.pipeline import (
    _parse_complexity_from_text,
    _parse_complexity_from_events,
    _try_parse_complexity_json,
    _failure_outcome,
)
from metainfer.tasks.evolve_kernel.orchestrator.prompts import (
    kernel_fn_name_from_code,
    _render_req,
)
from metainfer.tasks.evolve_kernel.orchestrator.phases import OK, LOGIC_FAIL, INFRA_FAIL


# --------------------------------------------------------------------------- #
# Complexity parsing
# --------------------------------------------------------------------------- #


class TestParseComplexityJson:
    def test_valid_json_with_overall_complexity(self):
        assert _try_parse_complexity_json(
            '{"overall_complexity": 0.45, "code_length": 0.5}'
        ) == pytest.approx(0.45)

    def test_only_overall_complexity(self):
        assert _try_parse_complexity_json(
            '{"overall_complexity": 0.8}'
        ) == pytest.approx(0.8)

    def test_clamp_to_range(self):
        # Should clamp to [0.0, 1.0]
        assert _try_parse_complexity_json('{"overall_complexity": 2.5}') == pytest.approx(1.0)
        assert _try_parse_complexity_json('{"overall_complexity": -0.5}') == pytest.approx(0.0)

    def test_missing_key(self):
        assert _try_parse_complexity_json('{"other_key": 0.5}') is None

    def test_invalid_json(self):
        assert _try_parse_complexity_json("not json") is None

    def test_empty_string(self):
        assert _try_parse_complexity_json("") is None

    def test_non_dict_json(self):
        assert _try_parse_complexity_json("[1, 2, 3]") is None

    def test_complexity_is_string(self):
        """Non-numeric values should be ignored."""
        assert _try_parse_complexity_json('{"overall_complexity": "0.5"}') is None

    def test_none_input(self):
        assert _try_parse_complexity_json(None) is None


class TestParseComplexityFromText:
    def test_json_code_block(self):
        text = """Here is the evaluation:
```json
{"overall_complexity": 0.45, "code_length": 0.5}
```
"""
        assert _parse_complexity_from_text(text) == pytest.approx(0.45)

    def test_json_code_block_no_lang_tag(self):
        text = """```
{"overall_complexity": 0.3}
```
"""
        assert _parse_complexity_from_text(text) == pytest.approx(0.3)

    def test_bare_json_object(self):
        text = 'The kernel complexity is:\n{"overall_complexity": 0.67}'
        assert _parse_complexity_from_text(text) == pytest.approx(0.67)

    def test_entire_text_is_json(self):
        text = '{"overall_complexity": 0.25}'
        assert _parse_complexity_from_text(text) == pytest.approx(0.25)

    def test_multiple_json_objects_first_wins(self):
        text = '{"other": 1}\n{"overall_complexity": 0.5}\n{"overall_complexity": 0.9}'
        # The regex finds the first bare object containing "overall_complexity"
        assert _parse_complexity_from_text(text) == pytest.approx(0.5)

    def test_no_complexity_in_text(self):
        text = "This text has no complexity score."
        assert _parse_complexity_from_text(text) is None

    def test_empty_text(self):
        assert _parse_complexity_from_text("") is None

    def test_complexity_in_nested_text(self):
        text = """Analysis complete.

The kernel is moderately complex with several code paths.

Final score: {"overall_complexity": 0.55, "code_length": 0.6, "control_flow": 0.4}
"""
        assert _parse_complexity_from_text(text) == pytest.approx(0.55)


class TestParseComplexityFromEvents:
    def test_finds_in_events_file(self, tmp_path):
        events_file = tmp_path / "events.jsonl"
        events_file.write_text(
            '{"type": "assistant", "message": {"content": [{"type": "text", "text": "'
            '{"overall_complexity": 0.72}"}]}}\n'
        )
        result = _parse_complexity_from_events(events_file)
        assert result == pytest.approx(0.72)

    def test_nonexistent_file(self):
        result = _parse_complexity_from_events(Path("/nonexistent/events.jsonl"))
        assert result is None

    def test_empty_file(self, tmp_path):
        events_file = tmp_path / "empty.jsonl"
        events_file.write_text("")
        result = _parse_complexity_from_events(events_file)
        assert result is None

    def test_no_complexity_in_events(self, tmp_path):
        events_file = tmp_path / "events.jsonl"
        events_file.write_text(
            '{"type": "system"}\n{"type": "result", "result": "done"}\n'
        )
        result = _parse_complexity_from_events(events_file)
        assert result is None


# --------------------------------------------------------------------------- #
# Failure outcome
# --------------------------------------------------------------------------- #


class TestFailureOutcome:
    def test_infra_mode(self):
        assert _failure_outcome("infra") == INFRA_FAIL

    def test_logic_mode(self):
        assert _failure_outcome("logic") == LOGIC_FAIL

    def test_none_mode(self):
        assert _failure_outcome(None) == LOGIC_FAIL

    def test_unknown_mode(self):
        assert _failure_outcome("unknown") == LOGIC_FAIL


# --------------------------------------------------------------------------- #
# Prompt helpers
# --------------------------------------------------------------------------- #


class TestKernelFnNameFromCode:
    def test_standard_triton_jit_pattern(self):
        code = """
import torch
import triton

@triton.jit
def my_gemm_kernel(a_ptr, b_ptr, c_ptr, M, N, K):
    ...
"""
        assert kernel_fn_name_from_code(code) == "my_gemm_kernel"

    def test_single_line_decorator(self):
        code = "@triton.jit\ndef fused_attention_kernel(q, k, v):\n    pass\n"
        assert kernel_fn_name_from_code(code) == "fused_attention_kernel"

    def test_no_triton_decorator_finds_kernel_named_function(self):
        code = "def triton_kernel(a, b):\n    pass\n"
        assert kernel_fn_name_from_code(code) == "triton_kernel"

    def test_no_kernel_in_name(self):
        code = "def foo():\n    pass\n"
        assert kernel_fn_name_from_code(code) == "foo"

    def test_empty_code(self):
        assert kernel_fn_name_from_code("") == "triton_kernel"

    def test_multiple_functions_picks_decorated(self):
        code = """
def helper():
    pass

@triton.jit
def actual_kernel():
    pass

def another():
    pass
"""
        assert kernel_fn_name_from_code(code) == "actual_kernel"

    def test_kernel_suffix_match(self):
        code = "def conv_kernel():\n    pass\n"
        assert kernel_fn_name_from_code(code) == "conv_kernel"
