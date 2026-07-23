"""Deterministic knowledge routing for inference-framework agents.

The router turns frozen task requirements, the current agent role, and
failure/review context into a required reading set plus a bounded optional
set.  The manifest owns policy; this module only validates and applies it.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

import yaml

from metainfer.orchestrator.requirements import req_field
from .capabilities import capability_knowledge_for_role, resolved_from_request


_ROUTING_FILE = "routing.yaml"


class KnowledgeRoutingError(ValueError):
    """Raised when the knowledge manifest is invalid or references bad paths."""


@dataclass(frozen=True)
class KnowledgeDocument:
    id: str
    path: str
    summary: str
    kind: str = "contract"


@dataclass(frozen=True)
class KnowledgeRoute:
    role: str
    required: Tuple[KnowledgeDocument, ...]
    optional: Tuple[KnowledgeDocument, ...]
    optional_limit: int
    notes: Tuple[str, ...] = ()


def resolve_knowledge_route(
    req: Dict[str, Any],
    notebooks_dir: Path,
    *,
    role: str,
    context: Optional[str] = None,
    required_document_ids: Sequence[str] = (),
) -> KnowledgeRoute:
    """Resolve the reading route for one agent prompt.

    Resolution order is additive: role baseline, matching requirement routes,
    then matching context routes. Duplicate document ids keep their first
    position. Required documents always win over optional documents.
    """
    manifest = _load_manifest(str((notebooks_dir / _ROUTING_FILE).resolve()))
    documents = _parse_documents(manifest, notebooks_dir)
    role_routes = _mapping(manifest.get("role_routes"), "role_routes")
    _validate_manifest_routes(manifest, documents, role_routes)
    if role not in role_routes:
        raise KnowledgeRoutingError(f"unknown knowledge-routing role: {role!r}")

    required: list[str] = []
    optional: list[str] = []
    notes: list[str] = []
    context_text = (context or "").casefold()
    base = _mapping(role_routes[role], f"role_routes.{role}")
    _extend_route(base, required, optional, notes)
    required.extend(str(document_id) for document_id in required_document_ids)
    # A C debugger with an explicit failure has a narrower job than an
    # implementer: context routes add only the capability documents relevant
    # to that failure. Loading every selected capability contract here made a
    # memory or build repair re-read TP, Paged KV, and batching documents.
    if role != "debugger" or not context_text:
        required.extend(capability_knowledge_for_role(req, role))
    optional_limit = _non_negative_int(
        base.get("optional_limit", 0), f"role_routes.{role}.optional_limit"
    )

    for index, selector in enumerate(_sequence(manifest.get("requirement_routes", []), "requirement_routes")):
        selector_map = _mapping(selector, f"requirement_routes[{index}]")
        if not _requirement_matches(req, selector_map):
            continue
        additions = _mapping(selector_map.get("add", {}), f"requirement_routes[{index}].add")
        role_addition = additions.get(role)
        if role_addition is not None:
            _extend_route(
                _mapping(role_addition, f"requirement_routes[{index}].add.{role}"),
                required,
                optional,
                notes,
            )

    resolved = resolved_from_request(req)
    active_capabilities = set(resolved.get("required_capabilities", []))
    active_capabilities.update(resolved.get("allowed_capabilities", []))
    if context_text:
        for index, context_route in enumerate(_sequence(manifest.get("context_routes", []), "context_routes")):
            route_map = _mapping(context_route, f"context_routes[{index}]")
            roles = tuple(str(value) for value in _sequence(route_map.get("roles", []), f"context_routes[{index}].roles"))
            if roles and role not in roles:
                continue
            capability_filter = {
                str(value)
                for value in _sequence(
                    route_map.get("capabilities_any", []),
                    f"context_routes[{index}].capabilities_any",
                )
            }
            if capability_filter and not capability_filter.intersection(active_capabilities):
                continue
            patterns = tuple(
                str(value).casefold()
                for value in _sequence(route_map.get("match_any", []), f"context_routes[{index}].match_any")
            )
            if patterns and not any(pattern in context_text for pattern in patterns):
                continue
            _extend_route(route_map, required, optional, notes)

    required_ids = _dedupe(required)
    required_set = set(required_ids)
    optional_ids = tuple(
        document_id
        for document_id in _dedupe(optional)
        if document_id not in required_set
    )
    _validate_document_ids(documents, required_ids + optional_ids, role)

    return KnowledgeRoute(
        role=role,
        required=tuple(documents[document_id] for document_id in required_ids),
        optional=tuple(documents[document_id] for document_id in optional_ids),
        optional_limit=optional_limit,
        notes=tuple(_dedupe(notes)),
    )


def render_knowledge_route(
    req: Dict[str, Any],
    notebooks_dir: Path,
    *,
    role: str,
    context: Optional[str] = None,
    required_document_ids: Sequence[str] = (),
) -> str:
    """Render a resolved route as an agent-facing mandatory prompt section."""
    route = resolve_knowledge_route(
        req,
        notebooks_dir,
        role=role,
        context=context,
        required_document_ids=required_document_ids,
    )
    lines = [
        "# Deterministic knowledge route (MANDATORY)",
        "The orchestrator resolved this route from the frozen requirements,",
        "your current role, and the available failure/review context.",
        "Read every REQUIRED item before making decisions or edits. Required",
        "items do not count against the optional-reading limit.",
        "",
        "## Required reading",
    ]
    if route.required:
        lines.extend(_render_document(doc, notebooks_dir) for doc in route.required)
    else:
        lines.append("- (none for this role)")

    lines.extend(["", f"## Optional reading (choose at most {route.optional_limit})"])
    if route.optional:
        lines.extend(_render_document(doc, notebooks_dir) for doc in route.optional)
    else:
        lines.append("- (none)")

    if route.notes:
        lines.extend(["", "## Routing notes"])
        lines.extend(f"- {note}" for note in route.notes)

    lines.extend(
        [
            "",
            "After the required set, use `Glob notebooks/**/*.md` only if the",
            "current problem still needs an optional document. Do not re-read a",
            "file already present in this resumed agent session.",
        ]
    )
    return "\n".join(lines)


def _render_document(document: KnowledgeDocument, notebooks_dir: Path) -> str:
    return (
        f"- `{document.id}`: `{notebooks_dir / document.path}` "
        f"({document.kind}) - {document.summary}"
    )


@lru_cache(maxsize=8)
def _load_manifest(path: str) -> Dict[str, Any]:
    manifest_path = Path(path)
    if not manifest_path.is_file():
        raise KnowledgeRoutingError(f"knowledge routing manifest not found: {manifest_path}")
    raw = yaml.safe_load(manifest_path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict) or raw.get("schema_version") != 1:
        raise KnowledgeRoutingError(
            f"unsupported knowledge routing schema in {manifest_path}"
        )
    return raw


def _parse_documents(
    manifest: Mapping[str, Any], notebooks_dir: Path
) -> Dict[str, KnowledgeDocument]:
    raw_documents = _mapping(manifest.get("documents"), "documents")
    documents: Dict[str, KnowledgeDocument] = {}
    root = notebooks_dir.resolve()
    for document_id, raw in raw_documents.items():
        entry = _mapping(raw, f"documents.{document_id}")
        relative = Path(str(entry.get("path", "")))
        if not relative.as_posix() or relative.is_absolute() or ".." in relative.parts:
            raise KnowledgeRoutingError(
                f"documents.{document_id}.path must stay inside notebooks/: {relative}"
            )
        resolved = (root / relative).resolve()
        if root not in resolved.parents or not resolved.is_file():
            raise KnowledgeRoutingError(
                f"documents.{document_id}.path does not exist: {resolved}"
            )
        documents[str(document_id)] = KnowledgeDocument(
            id=str(document_id),
            path=relative.as_posix(),
            summary=str(entry.get("summary", "")).strip(),
            kind=str(entry.get("kind", "contract")).strip() or "contract",
        )
    return documents


def _requirement_matches(req: Dict[str, Any], selector: Mapping[str, Any]) -> bool:
    field = str(selector.get("field", "")).strip()
    if not field:
        raise KnowledgeRoutingError("requirement route is missing field")
    expected = {
        str(value).casefold()
        for value in _sequence(selector.get("values", []), f"requirement route {field}.values")
    }
    actual = req_field(req, field)
    if isinstance(actual, (list, tuple, set, frozenset)):
        values = {str(value).casefold() for value in actual}
    elif actual is None:
        values = set()
    else:
        values = {str(actual).casefold()}
    return bool(expected.intersection(values))


def _extend_route(
    route: Mapping[str, Any],
    required: list[str],
    optional: list[str],
    notes: list[str],
) -> None:
    required.extend(str(value) for value in _sequence(route.get("required", []), "required"))
    optional.extend(str(value) for value in _sequence(route.get("optional", []), "optional"))
    notes.extend(str(value) for value in _sequence(route.get("notes", []), "notes"))


def _validate_document_ids(
    documents: Mapping[str, KnowledgeDocument], ids: Iterable[str], role: str
) -> None:
    unknown = [document_id for document_id in ids if document_id not in documents]
    if unknown:
        raise KnowledgeRoutingError(
            f"knowledge route for {role!r} references unknown documents: {unknown}"
        )


def _validate_manifest_routes(
    manifest: Mapping[str, Any],
    documents: Mapping[str, KnowledgeDocument],
    role_routes: Mapping[str, Any],
) -> None:
    known_roles = {str(role) for role in role_routes}
    for role, raw_route in role_routes.items():
        route = _mapping(raw_route, f"role_routes.{role}")
        ids = tuple(str(value) for value in _sequence(route.get("required", []), "required"))
        ids += tuple(str(value) for value in _sequence(route.get("optional", []), "optional"))
        _validate_document_ids(documents, ids, str(role))
        _non_negative_int(route.get("optional_limit", 0), f"role_routes.{role}.optional_limit")

    for index, selector in enumerate(_sequence(manifest.get("requirement_routes", []), "requirement_routes")):
        route = _mapping(selector, f"requirement_routes[{index}]")
        if not str(route.get("field", "")).strip():
            raise KnowledgeRoutingError(f"requirement_routes[{index}] is missing field")
        _sequence(route.get("values", []), f"requirement_routes[{index}].values")
        additions = _mapping(route.get("add", {}), f"requirement_routes[{index}].add")
        for role, raw_addition in additions.items():
            if role not in known_roles:
                raise KnowledgeRoutingError(
                    f"requirement_routes[{index}] references unknown role: {role}"
                )
            addition = _mapping(raw_addition, f"requirement_routes[{index}].add.{role}")
            ids = tuple(str(value) for value in _sequence(addition.get("required", []), "required"))
            ids += tuple(str(value) for value in _sequence(addition.get("optional", []), "optional"))
            _validate_document_ids(documents, ids, str(role))

    for index, raw_context in enumerate(_sequence(manifest.get("context_routes", []), "context_routes")):
        route = _mapping(raw_context, f"context_routes[{index}]")
        roles = tuple(str(value) for value in _sequence(route.get("roles", []), f"context_routes[{index}].roles"))
        unknown_roles = [role for role in roles if role not in known_roles]
        if unknown_roles:
            raise KnowledgeRoutingError(
                f"context_routes[{index}] references unknown roles: {unknown_roles}"
            )
        _sequence(route.get("match_any", []), f"context_routes[{index}].match_any")
        ids = tuple(str(value) for value in _sequence(route.get("required", []), "required"))
        ids += tuple(str(value) for value in _sequence(route.get("optional", []), "optional"))
        _validate_document_ids(documents, ids, f"context_routes[{index}]")
        _sequence(
            route.get("capabilities_any", []),
            f"context_routes[{index}].capabilities_any",
        )


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise KnowledgeRoutingError(f"{name} must be a mapping")
    return value


def _sequence(value: Any, name: str) -> Sequence[Any]:
    if not isinstance(value, (list, tuple)):
        raise KnowledgeRoutingError(f"{name} must be a list")
    return value


def _non_negative_int(value: Any, name: str) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise KnowledgeRoutingError(f"{name} must be an integer") from exc
    if parsed < 0:
        raise KnowledgeRoutingError(f"{name} must be non-negative")
    return parsed


def _dedupe(values: Iterable[str]) -> Tuple[str, ...]:
    return tuple(dict.fromkeys(values))
