from __future__ import annotations

import pytest

try:
    import torch
except ModuleNotFoundError:
    torch = None

from ..assets.w8a8_bench import (
    exact_w8a8_reference,
    validate_profile_protocol,
)


@pytest.mark.skipif(torch is None, reason="PyTorch is not installed")
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
