"""Kernel library for the evolve-kernel optimization loop.

Maintains a ranked pool of kernels (max 10). Each kernel has:
  - exec_time_ms: measured execution time (lower is better)
  - complexity_score: 0-1 agent-assessed complexity (lower = simpler = better)
  - combined_score: weighted combination, higher is better

Selection from the library is weighted-random by combined_score, giving
preference to kernels that are both fast AND simple (easier to further
optimize).
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional


MAX_LIBRARY_SIZE = 10


@dataclass
class KernelEntry:
    """One kernel in the optimization library."""
    id: str
    code: str
    exec_time_ms: float = 0.0
    complexity_score: float = 0.5  # 0=simplest, 1=most complex
    combined_score: float = 0.0
    iteration_added: int = 0
    parent_id: Optional[str] = None

    def recompute_combined(self) -> float:
        """Recompute combined_score from exec_time and complexity.

        Higher combined_score = better. Formula:
          perf_score = 1.0 / max(exec_time_ms, 1e-6)  (normalized by 10ms baseline)
          simplicity_bonus = 1.0 - complexity_score
          combined = perf_score * 0.7 + simplicity_bonus * 0.3
        """
        perf = 10.0 / max(self.exec_time_ms, 1e-6)  # 10ms baseline → 1.0
        simplicity = 1.0 - self.complexity_score
        self.combined_score = 0.7 * perf + 0.3 * simplicity
        return self.combined_score


class KernelLibrary:
    """Ranked pool of kernels driving the optimization loop.

    Thread-safe for single-orchestrator use (no concurrent access).
    Persisted to ``kernel_library.json`` in the workspace directory.
    """

    def __init__(self, kernels: Optional[List[KernelEntry]] = None) -> None:
        self._kernels: List[KernelEntry] = list(kernels or [])
        self._sort()

    def _sort(self) -> None:
        self._kernels.sort(key=lambda k: k.combined_score, reverse=True)
        # Trim to max size
        if len(self._kernels) > MAX_LIBRARY_SIZE:
            self._kernels = self._kernels[:MAX_LIBRARY_SIZE]

    @property
    def kernels(self) -> List[KernelEntry]:
        return list(self._kernels)

    @property
    def size(self) -> int:
        return len(self._kernels)

    @property
    def best(self) -> Optional[KernelEntry]:
        return self._kernels[0] if self._kernels else None

    def add(self, entry: KernelEntry) -> bool:
        """Add a kernel. Returns True if it was added (beats or fills a slot).

        If library is at capacity, the new kernel only enters if its
        combined_score beats the lowest-ranked kernel's score.
        """
        entry.recompute_combined()

        if self.size < MAX_LIBRARY_SIZE:
            self._kernels.append(entry)
            self._sort()
            return True

        # Full — only add if it beats the current last place
        if entry.combined_score > self._kernels[-1].combined_score:
            self._kernels.append(entry)
            self._sort()
            return True

        return False

    def select(self) -> Optional[KernelEntry]:
        """Weighted-random selection. Returns None if library is empty.

        Weight = combined_score (higher = more likely to be selected).
        If all scores are zero or negative, uniform random.
        """
        if not self._kernels:
            return None
        if self.size == 1:
            return self._kernels[0]

        weights = [max(k.combined_score, 1e-9) for k in self._kernels]
        total = sum(weights)
        if total <= 0:
            return random.choice(self._kernels)

        # Weighted random selection
        r = random.random() * total
        cumulative = 0.0
        for k, w in zip(self._kernels, weights):
            cumulative += w
            if r <= cumulative:
                return k

        return self._kernels[-1]  # fallback

    def get_by_id(self, kernel_id: str) -> Optional[KernelEntry]:
        for k in self._kernels:
            if k.id == kernel_id:
                return k
        return None

    def top_n(self, n: int = 5) -> List[KernelEntry]:
        return self._kernels[:n]

    def to_list(self) -> List[Dict[str, Any]]:
        return [asdict(k) for k in self._kernels]

    @classmethod
    def from_list(cls, data: List[Dict[str, Any]]) -> "KernelLibrary":
        kernels = [
            KernelEntry(
                id=d["id"],
                code=d["code"],
                exec_time_ms=d.get("exec_time_ms", 0.0),
                complexity_score=d.get("complexity_score", 0.5),
                combined_score=d.get("combined_score", 0.0),
                iteration_added=d.get("iteration_added", 0),
                parent_id=d.get("parent_id"),
            )
            for d in data
        ]
        return cls(kernels)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.to_list(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: Path) -> "KernelLibrary":
        if not path.is_file():
            return cls()
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(data, list):
                return cls()
            return cls.from_list(data)
        except (json.JSONDecodeError, OSError, KeyError):
            return cls()

    def last_added(self) -> Optional[KernelEntry]:
        """Return the most recently added kernel (highest iteration_added)."""
        if not self._kernels:
            return None
        return max(self._kernels, key=lambda k: k.iteration_added)
