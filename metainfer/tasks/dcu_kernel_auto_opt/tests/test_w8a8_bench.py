from __future__ import annotations

from pathlib import Path

import torch

import pytest

from ..assets.w8a8_bench import (
    exact_w8a8_reference,
    reference_cache_path,
    save_reference_cache,
    validate_profile_protocol,
    w8a8_seed,
)


def test_exact_reference_uses_integer_dot_before_scaling():
    a = torch.tensor([[127, 127, 127], [-127, 1, 2]], dtype=torch.int8)
    b = torch.tensor(
        [[127, -127], [127, 3], [127, 4]], dtype=torch.int8
    )
    a_scale = torch.tensor([[0.5], [0.25]], dtype=torch.float32)
    b_scale = torch.tensor([[0.125], [0.75]], dtype=torch.float32)

    actual = exact_w8a8_reference(a, b, a_scale, b_scale)
    dot = torch.tensor(
        [
            [3 * 127 * 127, 127 * (-127 + 3 + 4)],
            [-127 * 127 + 127 + 2 * 127, 127 * 127 + 3 + 8],
        ],
        dtype=torch.int64,
    )
    expected = (
        dot.float() * a_scale * b_scale.T
    ).to(torch.bfloat16)

    assert torch.equal(actual, expected)


def test_profile_protocol_requires_one_post_marker_replay():
    validate_profile_protocol(True, 0, 1, 1)
    with pytest.raises(ValueError, match="profile-only requires"):
        validate_profile_protocol(True, 0, 1, 2)
    validate_profile_protocol(False, 100, 30, 100)


def test_w8a8_seed_is_deterministic_and_shape_scoped():
    assert w8a8_seed(4096, 2304, 6144) == 20260724 + 4096 + 2304 + 6144
    assert w8a8_seed(4096, 2304, 6144) == w8a8_seed(4096, 2304, 6144)
    assert w8a8_seed(4096, 2304, 6144) != w8a8_seed(4096, 2304, 6145)


def test_reference_cache_path_matches_benchmark_naming():
    path = reference_cache_path(4096, 2304, 6144, Path("/tmp/cache"))
    assert path == Path("/tmp/cache") / "exact-int64-v1-m4096-n2304-k6144.pt"


def test_save_reference_cache_roundtrip(tmp_path):
    reference = torch.tensor(
        [[1.5, -2.5], [3.0, 4.5]], dtype=torch.bfloat16
    )
    # Parent directory deliberately does not exist yet: serial validation
    # uses a fresh cache dir (final/cache/references) and the save must
    # create it, or torch.save fails with "Parent directory does not exist".
    path = tmp_path / "cache" / "references" / "exact-int64-v1-m2-n2-k2.pt"
    save_reference_cache(reference, path)
    assert path.is_file()
    loaded = torch.load(path, map_location="cpu", weights_only=True)
    assert torch.equal(loaded, reference)
