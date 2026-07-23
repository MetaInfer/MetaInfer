"""Stage-7 capability combination matrix."""

from metainfer.tasks.gen_cpp_infer_framework.orchestrator.validation_matrix import (
    run_validation_matrix,
)


def test_full_validation_matrix_passes():
    report = run_validation_matrix()
    assert report["cases_total"] == 14
    assert report["cases_passed"] == 14, report["cases"]
    assert report["passed"] is True
