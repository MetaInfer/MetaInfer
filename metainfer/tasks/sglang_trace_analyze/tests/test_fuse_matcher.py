"""Fuse matcher tests."""

from ..orchestrator.fuse_matcher import _match_consecutive, match_fuse_patterns


def test_match_consecutive_found():
    names = ["abc", "rms_norm", "triton_gemm", "add"]
    pattern = ["rms_norm", "gemm"]
    result = _match_consecutive(names, pattern)
    assert result == ["rms_norm", "triton_gemm"]


def test_match_consecutive_not_found():
    names = ["abc", "rms_norm", "add"]
    pattern = ["rms_norm", "gemm"]
    result = _match_consecutive(names, pattern)
    assert result == []


def test_match_consecutive_short_list():
    names = ["abc"]
    pattern = ["a", "b"]
    result = _match_consecutive(names, pattern)
    assert result == []


def test_match_fuse_patterns_with_known_kernels():
    kernels = [
        {"kernel_name": "abc"},
        {"kernel_name": "triton_gemm"},
        {"kernel_name": "ncclAllReduce"},
        {"kernel_name": "triton_gemm"},
    ]
    matches = match_fuse_patterns(kernels)
    # The "nccl_allreduce + gemm (no overlap)" pattern should fire
    pattern_names = [m["pattern"] for m in matches]
    assert "nccl_allreduce + gemm (no overlap)" in pattern_names
