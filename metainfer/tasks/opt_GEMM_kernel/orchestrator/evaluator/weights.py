"""Freeze the task-author-owned weight directory used by the GEMM harness."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path

from .spec import SpecError


_MANIFEST = ".weights-manifest.json"


def _tree_digest(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*")):
        if path.name == _MANIFEST:
            continue
        rel = path.relative_to(root).as_posix()
        digest.update(rel.encode("utf-8"))
        if path.is_symlink():
            raise SpecError(f"weight bundle may not contain symlinks: {rel}")
        if path.is_file():
            with path.open("rb") as stream:
                for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                    digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class FrozenWeightBundle:
    """A private directory snapshot excluded from optimizer workspaces."""

    root: Path
    digest: str

    @classmethod
    def materialize(cls, source: Path, destination: Path) -> "FrozenWeightBundle":
        manifest_path = destination / _MANIFEST
        if destination.exists():
            try:
                raw = json.loads(manifest_path.read_text(encoding="utf-8"))
                frozen = cls(destination, str(raw["sha256"]))
            except (OSError, TypeError, KeyError, json.JSONDecodeError) as exc:
                raise SpecError(f"invalid frozen weight manifest: {manifest_path}") from exc
            frozen.verify()
            return frozen

        source = source.expanduser().resolve()
        if not source.is_dir():
            raise SpecError(f"Weight directory must be a directory: {source}")
        if not (source / "info.json").is_file():
            raise SpecError(f"Weight directory has no info.json: {source}")

        destination.parent.mkdir(parents=True, exist_ok=True)
        tmp = destination.with_name(destination.name + ".tmp")
        if tmp.exists():
            shutil.rmtree(tmp)
        shutil.copytree(source, tmp, symlinks=False)
        digest = _tree_digest(tmp)
        (tmp / _MANIFEST).write_text(
            json.dumps({"schema_version": 1, "sha256": digest}, indent=2),
            encoding="utf-8",
        )
        os.replace(tmp, destination)
        return cls(destination, digest)

    def verify(self) -> None:
        if not self.root.is_dir() or not (self.root / "info.json").is_file():
            raise SpecError(f"frozen weight directory is incomplete: {self.root}")
        actual_digest = _tree_digest(self.root)
        if actual_digest != self.digest:
            raise SpecError(
                f"frozen weight directory changed: expected {self.digest}, got {actual_digest}"
            )
