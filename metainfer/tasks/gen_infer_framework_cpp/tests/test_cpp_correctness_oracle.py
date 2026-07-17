from unittest.mock import patch

from metainfer.tasks.gen_infer_framework_cpp.orchestrator.oracles.correctness import (
    _continuous_batching_check,
    _deterministic_repeat_check,
    _paged_kv_boundary_check,
    _run_protocol_checks,
    _seeded_sampling_check,
    _streaming_check,
)


def _ok_response(text):
    return text, 200, 0.01, None


def test_deterministic_repeat_requires_identical_content():
    with patch(
        "metainfer.tasks.gen_infer_framework_cpp.orchestrator.oracles.correctness._send_request",
        side_effect=[_ok_response("same"), _ok_response("same"), _ok_response("different")],
    ):
        result = _deterministic_repeat_check(1234)

    assert result.judge_verdict == "fail"
    assert result.gating == "hard"


def test_seeded_sampling_rejects_parameters_that_are_ignored():
    with patch(
        "metainfer.tasks.gen_infer_framework_cpp.orchestrator.oracles.correctness._send_request",
        return_value=_ok_response("always greedy"),
    ):
        result = _seeded_sampling_check(1234)

    assert result.judge_verdict == "fail"
    assert "parameters appear ignored" in result.judge_reason


def test_seeded_sampling_accepts_reproducible_seeded_variation():
    def send(_port, cfg, timeout_s):
        del timeout_s
        mode = "narrow" if cfg["top_p"] < 0.1 else "wide"
        return _ok_response(f"sample-for-{cfg['seed']}-{mode}")

    with patch(
        "metainfer.tasks.gen_infer_framework_cpp.orchestrator.oracles.correctness._send_request",
        side_effect=send,
    ):
        result = _seeded_sampling_check(1234)

    assert result.judge_verdict == "pass"


def test_seeded_sampling_rejects_ignored_top_p():
    def send(_port, cfg, timeout_s):
        del timeout_s
        return _ok_response(f"sample-for-{cfg['seed']}")

    with patch(
        "metainfer.tasks.gen_infer_framework_cpp.orchestrator.oracles.correctness._send_request",
        side_effect=send,
    ):
        result = _seeded_sampling_check(1234)

    assert result.judge_verdict == "fail"
    assert "top_p narrows" in result.judge_reason


def test_streaming_check_is_enabled_only_when_requested():
    fake = _deterministic_repeat_check
    with (
        patch(
            "metainfer.tasks.gen_infer_framework_cpp.orchestrator.oracles.correctness._deterministic_repeat_check",
            return_value=fake,
        ),
        patch(
            "metainfer.tasks.gen_infer_framework_cpp.orchestrator.oracles.correctness._seeded_sampling_check",
            return_value=fake,
        ),
        patch(
            "metainfer.tasks.gen_infer_framework_cpp.orchestrator.oracles.correctness._streaming_check",
            return_value=fake,
        ) as stream_check,
    ):
        without_stream = _run_protocol_checks(1, {"features": []})
        with_stream = _run_protocol_checks(1, {"features": ["Streaming responses"]})

    assert len(without_stream) == 2
    assert len(with_stream) == 3
    stream_check.assert_called_once_with(1)


def test_feature_specific_protocol_checks_are_routed():
    fake = _deterministic_repeat_check
    with (
        patch(
            "metainfer.tasks.gen_infer_framework_cpp.orchestrator.oracles.correctness._deterministic_repeat_check",
            return_value=fake,
        ),
        patch(
            "metainfer.tasks.gen_infer_framework_cpp.orchestrator.oracles.correctness._seeded_sampling_check",
            return_value=fake,
        ),
        patch(
            "metainfer.tasks.gen_infer_framework_cpp.orchestrator.oracles.correctness._paged_kv_boundary_check",
            return_value=fake,
        ) as paged_check,
        patch(
            "metainfer.tasks.gen_infer_framework_cpp.orchestrator.oracles.correctness._continuous_batching_check",
            return_value=fake,
        ) as batching_check,
    ):
        results = _run_protocol_checks(
            1, {"features": ["Paged KV cache", "Continuous batching"]},
        )

    assert len(results) == 4
    paged_check.assert_called_once_with(1)
    batching_check.assert_called_once_with(1)


def test_paged_kv_boundary_requires_a_successful_long_context_response():
    with patch(
        "metainfer.tasks.gen_infer_framework_cpp.orchestrator.oracles.correctness._send_request",
        return_value=_ok_response("bounded response"),
    ):
        result = _paged_kv_boundary_check(1234)

    assert result.judge_verdict == "pass"
    assert result.gating == "hard"


def test_continuous_batching_requires_concurrent_determinism():
    with patch(
        "metainfer.tasks.gen_infer_framework_cpp.orchestrator.oracles.correctness._send_request",
        side_effect=[
            _ok_response("same"), _ok_response("same"),
            _ok_response("different"), _ok_response("same"),
        ],
    ):
        result = _continuous_batching_check(1234)

    assert result.judge_verdict == "fail"


def test_streaming_rejects_one_buffered_content_chunk():
    with patch(
        "metainfer.tasks.gen_infer_framework_cpp.orchestrator.oracles.correctness._send_stream_request",
        return_value=(
            "all at once", 200, 0.01, None, "text/event-stream", True, 1,
        ),
    ):
        result = _streaming_check(1234)

    assert result.judge_verdict == "fail"
    assert "at least two" in result.judge_reason


def test_streaming_accepts_multiple_incremental_content_chunks():
    with patch(
        "metainfer.tasks.gen_infer_framework_cpp.orchestrator.oracles.correctness._send_stream_request",
        return_value=(
            "two chunks", 200, 0.01, None, "text/event-stream", True, 2,
        ),
    ):
        result = _streaming_check(1234)

    assert result.judge_verdict == "pass"
