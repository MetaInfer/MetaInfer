"""Layered correctness-oracle control-flow tests."""

from __future__ import annotations

import json
from pathlib import Path

from metainfer.tasks.gen_cpp_infer_framework.orchestrator.capabilities import (
    resolve_capabilities,
)
from metainfer.tasks.gen_cpp_infer_framework.orchestrator.oracles import correctness


FIXTURES = Path(__file__).parent / "fixtures" / "requirements"


class _Proc:
    def poll(self):
        return None


def _request(name: str = "base_q8.json"):
    req = json.loads((FIXTURES / name).read_text(encoding="utf-8"))
    req["resolved_requirements"] = resolve_capabilities(req)
    return req


def _artifacts(iter_dir: Path) -> None:
    (iter_dir / "src").mkdir(parents=True)
    (iter_dir / "include").mkdir()
    (iter_dir / "build").mkdir()
    (iter_dir / "CMakeLists.txt").write_text("project(test)\n", encoding="utf-8")
    (iter_dir / "build.sh").write_text("#!/bin/bash\n", encoding="utf-8")
    (iter_dir / "serve.sh").write_text("#!/bin/bash\n", encoding="utf-8")
    (iter_dir / "serve.sh").chmod(0o755)
    (iter_dir / "src" / "main.cpp").write_text("int main(){}\n", encoding="utf-8")
    (iter_dir / "src" / "engine.cpp").write_text("// engine\n", encoding="utf-8")
    (iter_dir / "src" / "model_loader.cpp").write_text(
        "auto data_offset = align_up(tensor_info_end, general.alignment);\n"
        "auto file_offset = data_offset + tensor.offset;\n",
        encoding="utf-8",
    )


def _stages(report_dir: Path):
    return json.loads(
        (report_dir / "oracle-stages.json").read_text(encoding="utf-8")
    )


def _patch_success_path(monkeypatch, calls, *, models_payload=None) -> None:
    payload = {} if models_payload is None else models_payload
    monkeypatch.setattr(correctness, "materialize_hardware_binding", lambda *_args: None)
    monkeypatch.setattr(correctness, "execution_environment", lambda *_args: {})
    monkeypatch.setattr(
        correctness,
        "_load_cases",
        lambda _req: [{
            "id": "short-correctness",
            "prompt": "return ok",
            "expected_keywords": ["ok"],
            "max_tokens": 4,
        }],
    )
    monkeypatch.setattr(
        correctness,
        "_run_build_check",
        lambda *_args, **_kwargs: (
            calls.__setitem__("build", calls["build"] + 1) or True,
            None,
        ),
    )
    monkeypatch.setattr(
        correctness,
        "_run_numeric_check",
        lambda *_args, **_kwargs: (
            calls.__setitem__("numeric", calls["numeric"] + 1) or True,
            None,
            {"passed": True, "cases": []},
        ),
    )
    monkeypatch.setattr(correctness, "_pick_free_port", lambda: 12345)

    def start(*_args, **_kwargs):
        calls["server"] += 1
        return _Proc()

    monkeypatch.setattr(correctness, "_start_server", start)
    monkeypatch.setattr(
        correctness, "_wait_healthy", lambda *_args, **_kwargs: (True, None)
    )
    monkeypatch.setattr(correctness, "_fetch_models_payload", lambda _port: payload)

    def send(*_args, **_kwargs):
        calls["requests"] += 1
        return "ok response", 200, 0.01, None

    monkeypatch.setattr(correctness, "_send_request", send)
    monkeypatch.setattr(
        correctness,
        "_kill_server",
        lambda _proc: calls.__setitem__("kills", calls["kills"] + 1),
    )


def _calls():
    return {"build": 0, "numeric": 0, "server": 0, "requests": 0, "kills": 0}


def test_c0_failure_stops_before_build_numeric_and_server(tmp_path: Path):
    calls = _calls()
    report_dir = tmp_path / "reports"
    result = correctness.InferFrameworkOracle().run(
        iter_dir=tmp_path / "missing",
        req=_request(),
        report_dir=report_dir,
    )

    assert result.passed is False
    report = _stages(report_dir)
    assert [stage["id"] for stage in report["stages"]] == ["C0_artifacts"]
    assert report["stages"][0]["passed"] is False
    assert report["full_oracle_completed"] is False
    assert calls == _calls()


def test_c1_failure_stops_before_numeric_and_server(tmp_path: Path, monkeypatch):
    calls = _calls()
    iter_dir = tmp_path / "iteration"
    iter_dir.mkdir()
    _artifacts(iter_dir)
    _patch_success_path(monkeypatch, calls)
    monkeypatch.setattr(
        correctness,
        "_run_build_check",
        lambda *_args, **_kwargs: (
            calls.__setitem__("build", calls["build"] + 1) or False,
            "compile failed",
        ),
    )

    result = correctness.InferFrameworkOracle().run(
        iter_dir=iter_dir, req=_request(), report_dir=tmp_path / "reports"
    )

    assert result.passed is False
    assert calls["build"] == 1
    assert calls["numeric"] == 0
    assert calls["server"] == 0
    assert [stage["id"] for stage in _stages(tmp_path / "reports")["stages"]] == [
        "C0_artifacts", "C1_build",
    ]


def test_c2_failure_stops_before_server(tmp_path: Path, monkeypatch):
    calls = _calls()
    iter_dir = tmp_path / "iteration"
    iter_dir.mkdir()
    _artifacts(iter_dir)
    _patch_success_path(monkeypatch, calls)
    monkeypatch.setattr(
        correctness,
        "_run_numeric_check",
        lambda *_args, **_kwargs: (
            calls.__setitem__("numeric", calls["numeric"] + 1) or False,
            "numeric failed",
            None,
        ),
    )

    result = correctness.InferFrameworkOracle().run(
        iter_dir=iter_dir, req=_request(), report_dir=tmp_path / "reports"
    )

    assert result.passed is False
    assert calls["build"] == 1
    assert calls["numeric"] == 1
    assert calls["server"] == 0
    assert [stage["id"] for stage in _stages(tmp_path / "reports")["stages"]] == [
        "C0_artifacts", "C1_build", "C2_numeric",
    ]


def test_runtime_target_failure_skips_full_cases_and_judge(
    tmp_path: Path, monkeypatch,
):
    calls = _calls()
    iter_dir = tmp_path / "iteration"
    iter_dir.mkdir()
    _artifacts(iter_dir)
    _patch_success_path(monkeypatch, calls)
    monkeypatch.setattr(correctness, "_fetch_models_payload", lambda _port: None)

    result = correctness.InferFrameworkOracle().run(
        iter_dir=iter_dir,
        req=_request(),
        report_dir=tmp_path / "reports",
        repair_route={"route_id": "http_or_lifecycle", "signature": "sig"},
    )

    report = _stages(tmp_path / "reports")
    assert result.passed is False
    assert calls["server"] == 1
    assert calls["requests"] == 0
    assert report["target_route"] == "http_or_lifecycle"
    assert report["stages"][-1]["id"] == "C3_targeted"
    assert report["stages"][-1]["passed"] is False
    assert report["full_oracle_completed"] is False


def test_target_pass_reuses_server_then_runs_full_c4(tmp_path: Path, monkeypatch):
    calls = _calls()
    iter_dir = tmp_path / "iteration"
    iter_dir.mkdir()
    _artifacts(iter_dir)
    _patch_success_path(monkeypatch, calls)

    result = correctness.InferFrameworkOracle().run(
        iter_dir=iter_dir,
        req=_request(),
        report_dir=tmp_path / "reports",
        repair_route={"route_id": "model_path_or_serve_args"},
    )

    report = _stages(tmp_path / "reports")
    assert result.passed is True
    assert calls["server"] == 1
    assert calls["kills"] == 1
    assert calls["requests"] == 1
    assert [stage["id"] for stage in report["stages"]] == [
        "C0_artifacts", "C1_build", "C2_numeric", "C3_targeted", "C4_full",
    ]
    assert report["stages"][-1]["passed"] is True
    assert report["full_oracle_completed"] is True


def test_numeric_target_reuses_c2_and_starts_only_full_server(
    tmp_path: Path, monkeypatch,
):
    calls = _calls()
    iter_dir = tmp_path / "iteration"
    iter_dir.mkdir()
    _artifacts(iter_dir)
    _patch_success_path(monkeypatch, calls)

    result = correctness.InferFrameworkOracle().run(
        iter_dir=iter_dir,
        req=_request(),
        report_dir=tmp_path / "reports",
        repair_route={"route_id": "numeric_or_nonfinite"},
    )

    report = _stages(tmp_path / "reports")
    assert result.passed is True
    assert calls["numeric"] == 1
    assert calls["server"] == 1
    assert report["stages"][3]["id"] == "C3_targeted"
    assert "reused the passing C2" in report["stages"][3]["detail"]
    assert report["full_oracle_completed"] is True


def test_completed_but_failing_c4_is_not_reported_as_incomplete(
    tmp_path: Path, monkeypatch,
):
    calls = _calls()
    iter_dir = tmp_path / "iteration"
    iter_dir.mkdir()
    _artifacts(iter_dir)
    _patch_success_path(monkeypatch, calls)

    def fail_request(*_args, **_kwargs):
        calls["requests"] += 1
        return "server error", 503, 0.01, "HTTP 503"

    monkeypatch.setattr(correctness, "_send_request", fail_request)

    result = correctness.InferFrameworkOracle().run(
        iter_dir=iter_dir, req=_request(), report_dir=tmp_path / "reports"
    )

    report = _stages(tmp_path / "reports")
    assert result.passed is False
    assert report["stages"][-1]["id"] == "C4_full"
    assert report["stages"][-1]["passed"] is False
    assert report["full_oracle_completed"] is True
