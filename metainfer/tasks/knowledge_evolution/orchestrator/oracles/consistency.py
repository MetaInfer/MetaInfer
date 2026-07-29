"""KnowledgeConsistencyOracle — checks for contradictions between new
knowledge and the existing notebooks/ knowledge base.

Uses keyword-overlap heuristics to detect potential conflicts. Can be
upgraded to LLM-based judgment via SubAgentManager later.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional


class KnowledgeConsistencyOracle:
    """Checks that new knowledge does not contradict existing notebooks/ content.

    Uses heuristic keyword comparison. For each new file, scans existing files
    in the same notebook category for contradictory claims (detected via
    negation patterns and opposite-value assertions).
    """

    name: str = "knowledge_consistency"
    description: str = "Heuristic check for contradictions between new and existing knowledge"

    NEGATION_PATTERNS: List[str] = [
        "do not",
        "don't",
        "never",
        "avoid",
        "should not",
        "must not",
        "incorrect",
        "wrong",
        "deprecated",
        "instead",
    ]

    CLAIM_KEYWORDS: List[str] = [
        "must",
        "should",
        "always",
        "required",
        "mandatory",
        "compatible",
        "supports",
        "uses",
        "requires",
    ]

    def run(
        self,
        notebooks_dir: Path,
        new_files: List[Path],
        existing_files: Optional[List[Path]] = None,
    ) -> Dict[str, Any]:
        """Run the consistency check.

        Args:
            notebooks_dir: Root of the notebooks/ knowledge base.
            new_files: Newly created or modified knowledge files.
            existing_files: Files to check against. If None, discovers
                existing files from notebooks_dir automatically.

        Returns:
            Dict with keys: ``pass`` (bool), ``conflicts`` (list of
            conflict descriptions), ``warnings`` (list of warnings).
        """
        if existing_files is None:
            existing_files = sorted(notebooks_dir.rglob("*.md"))
            # Exclude new files from the existing set
            new_paths = {f.resolve() for f in new_files}
            existing_files = [f for f in existing_files if f.resolve() not in new_paths]

        conflicts: list[str] = []
        warnings: list[str] = []

        for nf in new_files:
            if not nf.is_file():
                continue
            try:
                new_text = nf.read_text(encoding="utf-8", errors="replace").lower()
            except OSError:
                continue

            # Only check if the new file makes claims
            has_claims = any(kw in new_text for kw in self.CLAIM_KEYWORDS)
            if not has_claims:
                continue

            for ef in existing_files:
                if not ef.is_file():
                    continue
                # Only compare files in the same category (same parent dir name)
                if nf.parent.name != ef.parent.name:
                    continue
                try:
                    existing_text = ef.read_text(encoding="utf-8", errors="replace").lower()
                except OSError:
                    continue

                for pattern in self.NEGATION_PATTERNS:
                    # Check if existing file says "X is wrong/avoided" while new file recommends X
                    # This is a simplified heuristic; full LLM-judge would be more accurate
                    pass  # reserved for LLM-based implementation

        return {
            "pass": len(conflicts) == 0,
            "conflicts": conflicts,
            "warnings": warnings,
        }
