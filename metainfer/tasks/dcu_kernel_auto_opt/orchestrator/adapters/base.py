"""Operator-agnostic seam between orchestration and a concrete kernel."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict

from ..config import ShapeSpec


@dataclass(frozen=True)
class AdapterResult:
    success: bool
    metrics: Dict[str, float] = field(default_factory=dict)
    evidence: Dict[str, Any] = field(default_factory=dict)
    error: str | None = None


class KernelAdapter(ABC):
    """A trusted adapter owned by MetaInfer, not by optimization agents."""

    requires_gpu = True

    @abstractmethod
    def describe_environment(self) -> Dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def prepare(self, workspace: Path) -> AdapterResult:
        raise NotImplementedError

    @abstractmethod
    def build(self, workspace: Path) -> AdapterResult:
        raise NotImplementedError

    @abstractmethod
    def correctness(self, workspace: Path, shape: ShapeSpec) -> AdapterResult:
        raise NotImplementedError

    @abstractmethod
    def benchmark(
        self, workspace: Path, shape: ShapeSpec, *, iteration: int = 0
    ) -> AdapterResult:
        raise NotImplementedError

    @abstractmethod
    def profile(self, workspace: Path, shape: ShapeSpec) -> AdapterResult:
        raise NotImplementedError
