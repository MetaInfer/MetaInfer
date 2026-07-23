"""Deterministic C-step failure classification tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from metainfer.tasks.gen_cpp_infer_framework.orchestrator.capabilities import (
    resolve_capabilities,
)
from metainfer.tasks.gen_cpp_infer_framework.orchestrator.failure_routing import (
    DEFAULT_FAILURE_PLAYBOOK,
    DEFAULT_FAILURE_ROUTES,
    FailureRoutingError,
    classify_failure,
)


FIXTURES = Path(__file__).parent / "fixtures" / "requirements"


def _req(name: str = "full_f16.json"):
    req = json.loads((FIXTURES / name).read_text(encoding="utf-8"))
    req["resolved_requirements"] = resolve_capabilities(req)
    return req


@pytest.mark.parametrize(
    ("failure", "route_id"),
    [
        ("C++ build failed with rc=2", "build_or_link"),
        ("Error: model file not found: /models/qwen.gguf", "model_path_or_serve_args"),
        ("vocab_size mismatch: expected 151936, got 0", "gguf_metadata_or_tensor"),
        ("kv_cache allocation failed after weight upload", "vram_or_kv_allocation"),
        ("C0.1 numeric tests failed: paged_attention", "numeric_or_nonfinite"),
        ("world_size does not match required TP size", "tensor_parallel"),
        ("paged attention block table index is invalid", "paged_kv"),
        ("continuous batching sequence isolation failed", "continuous_batching"),
        ("HTTP 500 from /v1/models", "http_or_lifecycle"),
        ("request TimeoutError: timed out", "generation_timeout"),
        ("7/7 hard cases failed with gibberish", "model_output"),
    ],
)
def test_real_failure_shapes_route_deterministically(failure: str, route_id: str):
    classified = classify_failure(failure, _req())

    assert classified.route_id == route_id
    assert classified.signature.startswith(route_id + ":")
    assert classified.targeted_checks
    assert classified.likely_files
    assert classified.required_documents
    assert classified.root_cause_checks
    assert classified.evidence_required


def test_signature_ignores_volatile_paths_ports_and_numbers():
    first = classify_failure(
        "server did not become healthy on port 35821\n"
        "--- server.stderr.log ---\n"
        "WARNING: skipping unrecognized metadata key general.name\n"
        "insufficient VRAM after weight upload: 0 MiB free, 512 MiB required "
        "for /models/a.gguf",
        _req(),
    )
    second = classify_failure(
        "server did not become healthy on port 58845\n"
        "--- server.stderr.log ---\n"
        "WARNING: skipping unrecognized metadata key general.file_type\n"
        "insufficient VRAM after weight upload: 0 MiB free, 650 MiB required "
        "for /different/path/b.gguf",
        _req(),
    )

    assert first.route_id == "vram_or_kv_allocation"
    assert second.route_id == first.route_id
    assert second.signature == first.signature


def test_signature_distinguishes_different_causes_in_same_category():
    kv_allocation = classify_failure(
        "server did not become healthy\n"
        "--- server.stderr.log ---\nkv_cache allocation failed",
        _req(),
    )
    weight_pressure = classify_failure(
        "server did not become healthy\n"
        "--- server.stderr.log ---\ninsufficient VRAM after weight upload",
        _req(),
    )

    assert kv_allocation.route_id == weight_pressure.route_id
    assert kv_allocation.signature != weight_pressure.signature


def test_capability_specific_route_is_inactive_when_feature_is_unselected():
    classified = classify_failure(
        "tensor parallel world_size startup failed", _req("base_q8.json")
    )

    assert classified.route_id == "unclassified"


def test_failure_route_registry_rejects_missing_matchers(tmp_path: Path):
    invalid = tmp_path / "failure_routes.yaml"
    invalid.write_text(
        "schema_version: 1\n"
        "defaults:\n"
        "  debugger_timeout_s: 10\n"
        "  likely_files: []\n"
        "  targeted_checks: []\n"
        "routes:\n"
        "  - id: broken\n"
        "    match_any: []\n",
        encoding="utf-8",
    )

    with pytest.raises(FailureRoutingError, match="match_any cannot be empty"):
        classify_failure("failure", _req(), routes_path=invalid)


def test_default_failure_routes_file_is_task_local():
    assert DEFAULT_FAILURE_ROUTES.name == "failure_routes.yaml"
    assert DEFAULT_FAILURE_ROUTES.parent.name == "gen_cpp_infer_framework"
    assert DEFAULT_FAILURE_ROUTES.is_file()
    assert DEFAULT_FAILURE_PLAYBOOK.name == "failure_playbook.yaml"
    assert DEFAULT_FAILURE_PLAYBOOK.parent == DEFAULT_FAILURE_ROUTES.parent
    assert DEFAULT_FAILURE_PLAYBOOK.is_file()


def test_failure_playbook_covers_every_route_and_registered_document():
    routes = yaml.safe_load(DEFAULT_FAILURE_ROUTES.read_text(encoding="utf-8"))
    playbook = yaml.safe_load(
        DEFAULT_FAILURE_PLAYBOOK.read_text(encoding="utf-8")
    )
    routing = yaml.safe_load(
        (DEFAULT_FAILURE_ROUTES.parent / "notebooks" / "routing.yaml").read_text(
            encoding="utf-8"
        )
    )
    route_ids = {str(item["id"]) for item in routes["routes"]}
    assert set(playbook["routes"]) == route_ids

    known_documents = set(routing["documents"])
    referenced = set(playbook["defaults"]["required_documents"])
    referenced.update(playbook["defaults"]["reference_templates"])
    for entry in playbook["routes"].values():
        referenced.update(entry.get("required_documents", []))
        referenced.update(entry.get("reference_templates", []))
        for documents in entry.get("combination_documents", {}).values():
            referenced.update(documents)
        for templates in entry.get("capability_templates", {}).values():
            referenced.update(templates)
    assert referenced <= known_documents


def test_numeric_playbook_requires_real_capacity_evidence():
    classified = classify_failure(
        "C0.1 numeric tests failed: missing required numeric case "
        "kv_capacity_contract",
        _req(),
    )

    assert classified.route_id == "numeric_or_nonfinite"
    assert "capability-checklists" in classified.required_documents
    assert "scheduler-block-manager-template" in classified.reference_templates
    assert any(
        "name-only PASS" in check for check in classified.root_cause_checks
    )
    assert any(
        "exact required case ID" in evidence
        for evidence in classified.evidence_required
    )

    base = classify_failure(
        "C0.1 numeric tests failed: rms_norm", _req("base_q8.json")
    )
    assert "scheduler-block-manager-template" not in base.reference_templates
    assert "tp-sharding-template" not in base.reference_templates


def test_playbook_combination_documents_follow_frozen_scope():
    full = classify_failure("TP2 allreduce failed", _req())
    tp_only = classify_failure("TP2 allreduce failed", _req("tp2_f16.json"))

    assert "tp-paged-kv-contract" in full.required_documents
    assert "paged-continuous-state-machine" in full.required_documents
    assert "tp-paged-kv-contract" not in tp_only.required_documents
    assert "paged-continuous-state-machine" not in tp_only.required_documents
