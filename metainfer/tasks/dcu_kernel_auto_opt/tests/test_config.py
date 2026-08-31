from __future__ import annotations

import pytest

from ..orchestrator.config import (
    DSH_DEFAULT_MODEL_ID,
    dsh_model_id,
    load_config,
    resolve_claude_bin,
)
from ..orchestrator.gpu_binding import (
    bind_worker_gpu,
    hide_gpus_from_control_plane,
)


def _req(shape_config: str):
    return {
        "answers": {
            "execution_mode": "Mock (no GPU)",
            "shape_config": shape_config,
            "mock_iterations": "2",
            "minimum_improvement_percent": 1.0,
        }
    }


def test_static_assignments_parse():
    cfg = load_config(_req("""
shapes:
  - {id: m2, M: 2, N: 16, K: 32}
  - {id: m16, M: 16, N: 16, K: 32}
assignments:
  worker_0: {gpu: 0, shapes: [m2]}
  worker_1: {gpu: 1, shapes: [m16]}
"""))
    assert set(cfg.shapes) == {"m2", "m16"}
    assert [a.gpu for a in cfg.assignments] == [0, 1]
    assert cfg.assignment_mode == "manual"
    assert cfg.claude_model == "claude-opus-5"


def test_new_task_can_select_sonnet():
    req = _req("""
shapes:
  - {id: m2, M: 2, N: 16, K: 32}
assignments:
  worker_0: {gpu: 0, shapes: [m2]}
""")
    req["answers"]["claude_model"] = "Sonnet"

    assert load_config(req).claude_model == "claude-sonnet-5"


def test_unknown_claude_model_is_rejected():
    req = _req("""
shapes:
  - {id: m2, M: 2, N: 16, K: 32}
assignments:
  worker_0: {gpu: 0, shapes: [m2]}
""")
    req["answers"]["claude_model"] = "default"

    with pytest.raises(ValueError, match="agent_model must be one of"):
        load_config(req)


def test_dsh_framework_resolves_wrapper_model(monkeypatch):
    monkeypatch.setenv("DSH_AGENT_MODEL", "deepseek/deepseek-v4-flash-0731")
    req = _req("""
shapes:
  - {id: m2, M: 2, N: 16, K: 32}
assignments:
  worker_0: {gpu: 0, shapes: [m2]}
""")
    req["answers"]["agent_framework"] = "dsh"
    req["answers"]["agent_model"] = "deepseek-v4-flash"

    cfg = load_config(req)
    assert cfg.agent_framework == "dsh"
    assert cfg.claude_model == "deepseek/deepseek-v4-flash-0731"


def test_dsh_default_model_when_agent_model_missing(monkeypatch):
    monkeypatch.setenv("DSH_AGENT_MODEL", "deepseek/deepseek-v4-flash-0731")
    req = _req("""
shapes:
  - {id: m2, M: 2, N: 16, K: 32}
assignments:
  worker_0: {gpu: 0, shapes: [m2]}
""")
    req["answers"]["agent_framework"] = "dsh"

    assert load_config(req).claude_model == DSH_DEFAULT_MODEL_ID


def test_dsh_rejects_ccb_only_model(monkeypatch):
    monkeypatch.setenv("DSH_AGENT_MODEL", "deepseek/deepseek-v4-flash-0731")
    req = _req("""
shapes:
  - {id: m2, M: 2, N: 16, K: 32}
assignments:
  worker_0: {gpu: 0, shapes: [m2]}
""")
    req["answers"]["agent_framework"] = "dsh"
    req["answers"]["agent_model"] = "Opus"

    with pytest.raises(ValueError, match="framework 'dsh'"):
        load_config(req)


def test_unknown_agent_framework_is_rejected():
    req = _req("""
shapes:
  - {id: m2, M: 2, N: 16, K: 32}
assignments:
  worker_0: {gpu: 0, shapes: [m2]}
""")
    req["answers"]["agent_framework"] = "bogus"

    with pytest.raises(ValueError, match="agent_framework must be one of"):
        load_config(req)


def test_resolve_claude_bin_per_framework():
    # dsh -> the bundled ccb-compatible DSH wrapper.
    dsh_bin = resolve_claude_bin("dsh")
    assert dsh_bin.endswith("bridge/dsh/dsh_agent.py")
    # Explicit override always wins.
    assert resolve_claude_bin("dsh", explicit="/opt/custom/agent") == "/opt/custom/agent"
    # ccb defaults to the env / "ccb" (METAINFER_CLAUDE_BIN unset here).
    assert resolve_claude_bin("ccb") == "ccb"


def test_dsh_model_id_honors_env(monkeypatch):
    monkeypatch.setenv("DSH_AGENT_MODEL", "deepseek/deepseek-v4-flash")
    assert dsh_model_id() == "deepseek/deepseek-v4-flash"
    monkeypatch.delenv("DSH_AGENT_MODEL")
    assert dsh_model_id() == DSH_DEFAULT_MODEL_ID


def test_explicit_manual_mode_requires_assignments():
    req = _req("""
assignment_mode: manual
shapes:
  - {id: m2, M: 2, N: 16, K: 32}
""")
    req["answers"]["execution_mode"] = (
        "Generate & optimize (auto-create kernel repo)"
    )
    with pytest.raises(ValueError, match="assignments"):
        load_config(req)


def test_explicit_ai_mode_can_omit_generate_assignments():
    req = _req("""
assignment_mode: ai
shapes:
  - {id: m2, M: 2, N: 16, K: 32}
""")
    req["answers"]["execution_mode"] = (
        "Generate & optimize (auto-create kernel repo)"
    )
    cfg = load_config(req)
    assert cfg.assignment_mode == "ai"
    assert cfg.assignments[0].shape_ids == ["m2"]


def test_subset_scope_is_preserved():
    req = _req("""
shape_scope: subset
assignment_mode: manual
shapes:
  - {id: m2, M: 2, N: 16, K: 32}
assignments:
  worker_2: {gpu: 2, shapes: [m2]}
""")
    cfg = load_config(req)
    assert cfg.shape_scope == "subset"
    assert list(cfg.shapes) == ["m2"]


def test_shape_cannot_be_assigned_twice():
    with pytest.raises(ValueError, match="assigned more than once"):
        load_config(_req("""
shapes:
  - {id: m2, M: 2}
assignments:
  worker_0: {gpu: 0, shapes: [m2]}
  worker_1: {gpu: 1, shapes: [m2]}
"""))


def test_gpu_cannot_be_shared():
    with pytest.raises(ValueError, match="GPU 0"):
        load_config(_req("""
shapes:
  - {id: m2, M: 2}
  - {id: m16, M: 16}
assignments:
  worker_0: {gpu: 0, shapes: [m2]}
  worker_1: {gpu: 0, shapes: [m16]}
"""))


def test_worker_gpu_binding_uses_one_filter_only():
    env = {"ROCR_VISIBLE_DEVICES": "3", "UNCHANGED": "yes"}
    visible = bind_worker_gpu(env, 2)
    assert visible == {"HIP_VISIBLE_DEVICES": "2"}
    assert env["HIP_VISIBLE_DEVICES"] == "2"
    assert "ROCR_VISIBLE_DEVICES" not in env
    assert env["UNCHANGED"] == "yes"


def test_control_plane_hides_gpus():
    env = {"ROCR_VISIBLE_DEVICES": "0"}
    hide_gpus_from_control_plane(env)
    assert env == {"HIP_VISIBLE_DEVICES": ""}


def test_real_smoke_mode_is_accepted():
    req = _req("""
shapes:
  - {id: m2, M: 2, N: 16, K: 32}
assignments:
  worker_0: {gpu: 0, shapes: [m2]}
""")
    req["answers"]["execution_mode"] = "Real agents + DCU (smoke harness)"
    assert load_config(req).execution_mode.startswith("Real agents")


def test_real_smoke_ignores_legacy_repo_field():
    req = _req("""
shapes:
  - {id: m2, M: 2, N: 16, K: 32}
assignments:
  worker_0: {gpu: 0, shapes: [m2]}
""")
    req["answers"].update({
        "execution_mode": "Real agents + DCU (smoke harness)",
        "target_repo_path": ">=1.2x baseline",
    })
    assert load_config(req).target_repo_path is None


def test_legacy_workers_shape_config_is_normalized():
    cfg = load_config(_req("""
workers:
  worker_0:
    gpu: 0
    shapes:
      - {id: m2, op: gemm, M: 2, N: 16, K: 32}
  worker_1:
    gpu: 1
    shapes:
      - {id: m16, op: gemm, M: 16, N: 16, K: 32}
"""))
    assert set(cfg.shapes) == {"m2", "m16"}
    assert cfg.shapes["m2"].params["op"] == "gemm"
    assert [(item.worker_id, item.gpu, item.shape_ids) for item in cfg.assignments] == [
        ("worker_0", 0, ["m2"]),
        ("worker_1", 1, ["m16"]),
    ]


def test_real_w8a8_mode_requires_absolute_repo():
    req = _req("""
shapes:
  - {id: m2, M: 2, N: 16, K: 32}
assignments:
  worker_0: {gpu: 0, shapes: [m2]}
""")
    # Use an existing directory so it passes the exists() check.
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        req["answers"].update({
            "execution_mode": "Real INT8 W8A8 GEMM",
            "target_repo_path": td,
        })
        cfg = load_config(req)
        assert cfg.execution_mode == "Real INT8 W8A8 GEMM"
        assert str(cfg.target_repo_path) == td


def test_real_w8a8_mode_accepts_missing_repo_field():
    """W8A8 mode accepts missing target_repo_path (agent auto-generates)."""
    req = _req("""
shapes:
  - {id: m2, M: 2, N: 16, K: 32}
assignments:
  worker_0: {gpu: 0, shapes: [m2]}
""")
    req["answers"]["execution_mode"] = "Real INT8 W8A8 GEMM"
    cfg = load_config(req)
    assert cfg.execution_mode == "Real INT8 W8A8 GEMM"
    assert cfg.target_repo_path is None
