"""KnowledgeCoverageOracle — checks whether the knowledge base now covers
the blind spots that caused a previous pure-KB generation attempt to fail.

Heuristic: extracts key technical terms from the failure reason and
verifies they appear in the newly created knowledge files.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List


class KnowledgeCoverageOracle:
    """Checks that new knowledge fills the gaps that caused previous failure.

    Extracts technical keywords from the failure reason and checks coverage
    in the new knowledge files. A match suggests the new content addresses
    the failure domain; a miss suggests gaps remain.
    """

    name: str = "knowledge_coverage"
    description: str = "Heuristic blind-spot coverage check for new knowledge"

    TECH_TERMS: List[str] = [
        "attention",
        "kv cache",
        "mla",
        "mha",
        "gqa",
        "moe",
        "rope",
        "tensor parallel",
        "tp",
        "pipeline",
        "prefill",
        "decode",
        "continuous batching",
        "paged",
        "block",
        "flash",
        "norm",
        "rms",
        "layernorm",
        "embedding",
        "linear",
        "softmax",
        "sampler",
        "logits",
        "tokenizer",
        "weight",
        "loading",
        "scheduler",
        "memory",
        "allocator",
        "cuda graph",
        "nccl",
    ]

    def run(
        self,
        notebooks_dir: Path,
        new_files: List[Path],
        failure_reason: str,
    ) -> Dict[str, Any]:
        """Run the coverage check.

        Args:
            notebooks_dir: Root of the notebooks/ knowledge base.
            new_files: Newly created knowledge files to check.
            failure_reason: The error/failure description from the
                previous pure-KB attempt.

        Returns:
            Dict with keys: ``pass`` (bool), ``covered_terms`` (list of
            matched terms), ``missing_terms`` (list of unmatched terms),
            ``score`` (float 0-1 coverage ratio).
        """
        failure_lower = failure_reason.lower()

        # Find which tech terms appear in the failure reason
        relevant_terms: list[str] = []
        for term in self.TECH_TERMS:
            if term in failure_lower:
                relevant_terms.append(term)

        if not relevant_terms:
            # No recognizable tech terms in the failure — can't assess coverage
            return {"pass": True, "covered_terms": [], "missing_terms": [], "score": 1.0}

        # Check coverage in new files
        covered: list[str] = []
        missing: list[str] = []

        for term in relevant_terms:
            found = False
            for nf in new_files:
                if not nf.is_file():
                    continue
                try:
                    text = nf.read_text(encoding="utf-8", errors="replace").lower()
                except OSError:
                    continue
                if term in text:
                    covered.append(term)
                    found = True
                    break
            if not found:
                missing.append(term)

        score = len(covered) / len(relevant_terms) if relevant_terms else 1.0
        return {
            "pass": len(missing) == 0,
            "covered_terms": covered,
            "missing_terms": missing,
            "score": score,
        }
