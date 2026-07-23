"""Deterministic C-step failure classification and repair routing."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from functools import lru_cache
import hashlib
import re
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import yaml

from .capabilities import resolved_from_request


TASK_DIR = Path(__file__).resolve().parent.parent
DEFAULT_FAILURE_ROUTES = TASK_DIR / "failure_routes.yaml"
DEFAULT_FAILURE_PLAYBOOK = TASK_DIR / "failure_playbook.yaml"


class FailureRoutingError(ValueError):
    """Raised when the task-local failure-routing registry is malformed."""


@dataclass(frozen=True)
class FailureClassification:
    route_id: str
    category: str
    signature: str
    matched_term: str
    debugger_timeout_s: int
    knowledge_context: str
    likely_files: Tuple[str, ...]
    targeted_checks: Tuple[str, ...]
    required_documents: Tuple[str, ...]
    reference_templates: Tuple[str, ...]
    root_cause_checks: Tuple[str, ...]
    evidence_required: Tuple[str, ...]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def classify_failure(
    failure: Optional[str],
    req: Optional[Dict[str, Any]] = None,
    *,
    routes_path: Path = DEFAULT_FAILURE_ROUTES,
    playbook_path: Path = DEFAULT_FAILURE_PLAYBOOK,
) -> FailureClassification:
    """Return the first active route matching a C-step failure."""
    registry = _load_registry(str(routes_path.resolve()))
    text = str(failure or "")
    folded = text.casefold()
    active_capabilities, active_combinations = _active_scope(req or {})
    defaults = _mapping(registry.get("defaults"), "defaults")
    playbook = _load_playbook(str(playbook_path.resolve()))
    playbook_defaults = _mapping(playbook.get("defaults"), "playbook.defaults")
    playbook_routes = _mapping(playbook.get("routes"), "playbook.routes")
    configured_route_ids = {
        str(_mapping(route, "route").get("id") or "")
        for route in _sequence(registry.get("routes"), "routes")
    }
    playbook_route_ids = {str(route_id) for route_id in playbook_routes}
    if configured_route_ids != playbook_route_ids:
        raise FailureRoutingError(
            "failure playbook routes do not match failure_routes.yaml: "
            f"missing={sorted(configured_route_ids - playbook_route_ids)}, "
            f"unknown={sorted(playbook_route_ids - configured_route_ids)}"
        )

    selected: Mapping[str, Any] = defaults
    selected_priority = -1
    matched_term = ""
    for index, raw_route in enumerate(_sequence(registry.get("routes"), "routes")):
        route = _mapping(raw_route, f"routes[{index}]")
        capability_filter = set(
            _strings(route.get("capabilities_any", []), f"routes[{index}].capabilities_any")
        )
        if capability_filter and not capability_filter.intersection(active_capabilities):
            continue
        matching_terms = [
            term
            for term in _strings(
                route.get("match_any", []), f"routes[{index}].match_any"
            )
            if term.casefold() in folded
        ]
        if not matching_terms:
            continue
        priority = _non_negative_int(
            route.get("priority", 0), f"routes[{index}].priority"
        )
        if priority > selected_priority:
            selected = route
            selected_priority = priority
            matched_term = max(matching_terms, key=len)

    route_id = str(selected.get("id") or "unclassified")
    playbook_route = _mapping(
        playbook_routes.get(route_id, {}), f"playbook.routes.{route_id}"
    )
    required_documents = list(_strings(
        playbook_defaults.get("required_documents", []),
        "playbook.defaults.required_documents",
    ))
    required_documents.extend(_strings(
        playbook_route.get("required_documents", []),
        f"playbook.routes.{route_id}.required_documents",
    ))
    combination_documents = _mapping(
        playbook_route.get("combination_documents", {}),
        f"playbook.routes.{route_id}.combination_documents",
    )
    for combination_id in active_combinations:
        required_documents.extend(_strings(
            combination_documents.get(combination_id, []),
            f"playbook.routes.{route_id}.combination_documents.{combination_id}",
        ))
    reference_templates = [
        *_strings(
            playbook_defaults.get("reference_templates", []),
            "playbook.defaults.reference_templates",
        ),
        *_strings(
            playbook_route.get("reference_templates", []),
            f"playbook.routes.{route_id}.reference_templates",
        ),
    ]
    capability_templates = _mapping(
        playbook_route.get("capability_templates", {}),
        f"playbook.routes.{route_id}.capability_templates",
    )
    for capability_id in sorted(active_capabilities):
        reference_templates.extend(_strings(
            capability_templates.get(capability_id, []),
            f"playbook.routes.{route_id}.capability_templates.{capability_id}",
        ))
    signature_basis = (
        f"{route_id}:{matched_term.casefold()}:{_normalize_failure(text)}"
    )
    digest = hashlib.sha256(signature_basis.encode("utf-8")).hexdigest()[:16]
    return FailureClassification(
        route_id=route_id,
        category=str(selected.get("category") or defaults.get("category") or "unclassified"),
        signature=f"{route_id}:{digest}",
        matched_term=matched_term,
        debugger_timeout_s=_positive_int(
            selected.get("debugger_timeout_s", defaults.get("debugger_timeout_s", 900)),
            "debugger_timeout_s",
        ),
        knowledge_context=str(
            selected.get("knowledge_context")
            or defaults.get("knowledge_context")
            or ""
        ),
        likely_files=_strings(
            selected.get("likely_files", defaults.get("likely_files", [])),
            "likely_files",
        ),
        targeted_checks=_strings(
            selected.get("targeted_checks", defaults.get("targeted_checks", [])),
            "targeted_checks",
        ),
        required_documents=_dedupe(required_documents),
        reference_templates=_dedupe(reference_templates),
        root_cause_checks=_dedupe([
            *_strings(
                playbook_defaults.get("root_cause_checks", []),
                "playbook.defaults.root_cause_checks",
            ),
            *_strings(
                playbook_route.get("root_cause_checks", []),
                f"playbook.routes.{route_id}.root_cause_checks",
            ),
        ]),
        evidence_required=_dedupe([
            *_strings(
                playbook_defaults.get("evidence_required", []),
                "playbook.defaults.evidence_required",
            ),
            *_strings(
                playbook_route.get("evidence_required", []),
                f"playbook.routes.{route_id}.evidence_required",
            ),
        ]),
    )


def _active_scope(req: Dict[str, Any]) -> Tuple[set[str], Tuple[str, ...]]:
    try:
        resolved = resolved_from_request(req)
    except Exception:  # noqa: BLE001 - classification must never hide a failure
        return set(), ()
    active = set(str(item) for item in resolved.get("required_capabilities", []))
    active.update(str(item) for item in resolved.get("allowed_capabilities", []))
    combinations = tuple(
        str(item) for item in resolved.get("active_combination_contracts", [])
    )
    return active, combinations


def _normalize_failure(value: str) -> str:
    """Remove volatile paths, ports, counters, and diagnostic tails."""
    text = value.casefold()
    for marker in (
        "\ndiagnostic files",
        "\n--- server.",
        "\n--- cpp-build",
        "\nfailing test cases",
    ):
        text = text.split(marker, 1)[0]
    text = re.sub(r"/[\w.@+,:=~-]+(?:/[\w.@+,:=~-]+)+", "<path>", text)
    text = re.sub(r"\b[0-9a-f]{8}-[0-9a-f-]{27,}\b", "<uuid>", text)
    text = re.sub(r"\b0x[0-9a-f]+\b", "<hex>", text)
    text = re.sub(r"\b\d+(?:\.\d+)?\b", "<n>", text)
    return " ".join(text.split())[:800]


@lru_cache(maxsize=4)
def _load_registry(path: str) -> Dict[str, Any]:
    registry_path = Path(path)
    try:
        raw = yaml.safe_load(registry_path.read_text(encoding="utf-8")) or {}
    except OSError as exc:
        raise FailureRoutingError(f"cannot read failure routes: {registry_path}") from exc
    if not isinstance(raw, dict) or raw.get("schema_version") != 1:
        raise FailureRoutingError(f"unsupported failure routes schema: {registry_path}")
    defaults = _mapping(raw.get("defaults"), "defaults")
    _positive_int(defaults.get("debugger_timeout_s", 0), "defaults.debugger_timeout_s")
    _strings(defaults.get("likely_files", []), "defaults.likely_files")
    _strings(defaults.get("targeted_checks", []), "defaults.targeted_checks")
    for index, raw_route in enumerate(_sequence(raw.get("routes"), "routes")):
        route = _mapping(raw_route, f"routes[{index}]")
        if not str(route.get("id") or "").strip():
            raise FailureRoutingError(f"routes[{index}].id is required")
        if not _strings(route.get("match_any", []), f"routes[{index}].match_any"):
            raise FailureRoutingError(f"routes[{index}].match_any cannot be empty")
        _positive_int(
            route.get("debugger_timeout_s", defaults.get("debugger_timeout_s")),
            f"routes[{index}].debugger_timeout_s",
        )
        _non_negative_int(route.get("priority", 0), f"routes[{index}].priority")
        _strings(route.get("capabilities_any", []), f"routes[{index}].capabilities_any")
        _strings(
            route.get("likely_files", defaults.get("likely_files", [])),
            f"routes[{index}].likely_files",
        )
        _strings(
            route.get("targeted_checks", defaults.get("targeted_checks", [])),
            f"routes[{index}].targeted_checks",
        )
    return raw


@lru_cache(maxsize=4)
def _load_playbook(path: str) -> Dict[str, Any]:
    playbook_path = Path(path)
    try:
        raw = yaml.safe_load(playbook_path.read_text(encoding="utf-8")) or {}
    except OSError as exc:
        raise FailureRoutingError(
            f"cannot read failure playbook: {playbook_path}"
        ) from exc
    if not isinstance(raw, dict) or raw.get("schema_version") != 1:
        raise FailureRoutingError(
            f"unsupported failure playbook schema: {playbook_path}"
        )
    defaults = _mapping(raw.get("defaults"), "playbook.defaults")
    for field in (
        "required_documents", "reference_templates", "root_cause_checks",
        "evidence_required",
    ):
        _strings(defaults.get(field, []), f"playbook.defaults.{field}")
    routes = _mapping(raw.get("routes"), "playbook.routes")
    for route_id, raw_entry in routes.items():
        entry = _mapping(raw_entry, f"playbook.routes.{route_id}")
        for field in (
            "required_documents", "reference_templates", "root_cause_checks",
            "evidence_required",
        ):
            _strings(
                entry.get(field, []), f"playbook.routes.{route_id}.{field}"
            )
        combinations = _mapping(
            entry.get("combination_documents", {}),
            f"playbook.routes.{route_id}.combination_documents",
        )
        for combination_id, documents in combinations.items():
            _strings(
                documents,
                f"playbook.routes.{route_id}.combination_documents.{combination_id}",
            )
        capability_templates = _mapping(
            entry.get("capability_templates", {}),
            f"playbook.routes.{route_id}.capability_templates",
        )
        for capability_id, templates in capability_templates.items():
            _strings(
                templates,
                f"playbook.routes.{route_id}.capability_templates.{capability_id}",
            )
    return raw


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise FailureRoutingError(f"{name} must be a mapping")
    return value


def _sequence(value: Any, name: str) -> Sequence[Any]:
    if not isinstance(value, (list, tuple)):
        raise FailureRoutingError(f"{name} must be a list")
    return value


def _strings(value: Any, name: str) -> Tuple[str, ...]:
    return tuple(str(item) for item in _sequence(value, name))


def _dedupe(values: Sequence[str]) -> Tuple[str, ...]:
    return tuple(dict.fromkeys(str(value) for value in values))


def _positive_int(value: Any, name: str) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise FailureRoutingError(f"{name} must be an integer") from exc
    if parsed <= 0:
        raise FailureRoutingError(f"{name} must be positive")
    return parsed


def _non_negative_int(value: Any, name: str) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise FailureRoutingError(f"{name} must be an integer") from exc
    if parsed < 0:
        raise FailureRoutingError(f"{name} must be non-negative")
    return parsed


__all__ = [
    "DEFAULT_FAILURE_PLAYBOOK",
    "DEFAULT_FAILURE_ROUTES",
    "FailureClassification",
    "FailureRoutingError",
    "classify_failure",
]
