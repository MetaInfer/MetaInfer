"""Tests for :mod:`metainfer.orchestrator.requirements`.

The canonical requirements.json schema is flat (form answers spread to
top level). The nested ``form`` / ``answers`` sub-dicts are legacy
fallbacks for old task files on disk. These tests pin both paths so
future schema changes are intentional.
"""

from __future__ import annotations

from metainfer.orchestrator.requirements import (
    req_field,
    req_field_float,
    req_field_int,
)


def test_req_field_flat_canonical():
    """WebUI writes flat — that's the canonical form."""
    req = {"task_id": "t1", "target_model": "/data/m", "max_iterations": "50"}
    assert req_field(req, "target_model") == "/data/m"
    assert req_field(req, "max_iterations") == "50"
    assert req_field(req, "task_id") == "t1"


def test_req_field_legacy_form_nested():
    """Old test fixtures / hand-edited files may use ``form: {...}``."""
    req = {"task_id": "t1", "form": {"target_model": "/legacy"}}
    assert req_field(req, "target_model") == "/legacy"


def test_req_field_legacy_answers_nested():
    """Some early code wrote ``answers: {...}``."""
    req = {"task_id": "t1", "answers": {"target_model": "/legacy2"}}
    assert req_field(req, "target_model") == "/legacy2"


def test_req_field_flat_beats_nested():
    """If both flat and nested exist (shouldn't happen, but if it does),
    flat wins — flat is the canonical form."""
    req = {
        "target_model": "/flat",
        "form": {"target_model": "/nested-form"},
        "answers": {"target_model": "/nested-answers"},
    }
    assert req_field(req, "target_model") == "/flat"


def test_req_field_default_on_missing():
    assert req_field({}, "nope") is None
    assert req_field({}, "nope", default=42) == 42


def test_req_field_handles_non_dict():
    """Defensive: a corrupted file shouldn't crash the helper."""
    assert req_field(None, "x", default="d") == "d"  # type: ignore[arg-type]
    assert req_field("not a dict", "x", default="d") == "d"  # type: ignore[arg-type]


def test_req_field_int_coerces_string():
    """Form widgets often emit numbers as strings."""
    assert req_field_int({"max_iterations": "50"}, "max_iterations") == 50


def test_req_field_int_default_on_garbage():
    assert req_field_int({"max_iterations": "unbounded"}, "max_iterations", default=10) == 10
    assert req_field_int({}, "max_iterations", default=10) == 10


def test_req_field_float_coerces():
    assert req_field_float({"cost": "50.5"}, "cost") == 50.5
    assert req_field_float({}, "cost", default=1.0) == 1.0


def _main() -> None:
    tests = [
        ("test_req_field_flat_canonical", test_req_field_flat_canonical),
        ("test_req_field_legacy_form_nested", test_req_field_legacy_form_nested),
        ("test_req_field_legacy_answers_nested", test_req_field_legacy_answers_nested),
        ("test_req_field_flat_beats_nested", test_req_field_flat_beats_nested),
        ("test_req_field_default_on_missing", test_req_field_default_on_missing),
        ("test_req_field_handles_non_dict", test_req_field_handles_non_dict),
        ("test_req_field_int_coerces_string", test_req_field_int_coerces_string),
        ("test_req_field_int_default_on_garbage", test_req_field_int_default_on_garbage),
        ("test_req_field_float_coerces", test_req_field_float_coerces),
    ]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"  PASS  {name}")
        except AssertionError as e:
            failed += 1
            print(f"  FAIL  {name}: {e}")
        except Exception as e:
            failed += 1
            print(f"  ERROR {name}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    _main()
