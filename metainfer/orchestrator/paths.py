"""Path resolution for the MetaInfer orchestrator.

Pluggable task layout (post task-package refactor)::

    MetaInfer/                          ← repo / install root
    ├── pyproject.toml
    ├── metainfer/
    │   ├── __init__.py
    │   ├── orchestrator/              ← THIS package (shared framework)
    │   │   └── paths.py               ← this file
    │   ├── tasks/                     ← task packages (one per task type)
    │   │   └── <task_pkg>/            ← orchestrator/ + server/ + static/ + tests/ + form.yaml
    │   ├── web/
    │   └── static/

Each task type is fully self-contained: its form schema lives at
``metainfer/tasks/<task_pkg>/form.yaml``, its knowledge base (if any)
lives at ``metainfer/tasks/<task_pkg>/notebooks/`` (resolved by the
task package itself, not by this module), and its label / description
/ detail view live on its :class:`metainfer.server.registry.WebPlugin`.
The framework never hardcodes the identity of any single task package.

All paths derive from ``__file__`` — no walk-up search, no env vars,
no hardcoded absolute paths.
"""

from __future__ import annotations

from pathlib import Path


def repo_root() -> Path:
    """Absolute path to the repo / install root.

    ``paths.py`` lives at ``<repo>/metainfer/orchestrator/paths.py``. The
    repo root is therefore ``Path(__file__).resolve().parents[2]``.
    """
    return Path(__file__).resolve().parents[2]
