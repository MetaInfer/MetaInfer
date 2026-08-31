"""Real multi-agent optimization pipeline for the extracted W8A8 GEMM."""

from __future__ import annotations

import json
import hashlib
import os
import shutil
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict

from metainfer.orchestrator.state import StateStore
from metainfer.orchestrator.subagent_manager import AgentSpec, SubAgentManager

from . import phases
from .api_contracts import W8A8_API_FILENAME, file_digest
from .config import (
    ROUND_ACCEPTANCE_IMPROVEMENT_PERCENT,
    OptimizerConfig,
    WorkerAssignment,
    load_config,
)
from .gpu_binding import bind_worker_gpu
from .experience_store import load_verified_experience
from .guidance import claim_next_guidance
from .isa_analysis import (
    analyze_inline_asm_source,
    evaluate_inline_asm_gate,
    inspect_gfx928_object,
)
from .prompts import w8a8_round_strategy, w8a8_strategy_guidance
from .pmc_profile import (
    add_unprofiled_bandwidth,
    parse_memory_traffic_csv,
    parse_pmc_csv,
)
from .real_pipeline import _last_json, _run, _safe, _status
from .result_store import SCHEMA_VERSION, append_jsonl, write_json
from .skill_store import generate_merged_skill, generate_worker_skill
from .w8a8_baselines import fixed_triton_graph_baseline


HARNESS = (
    Path(__file__).resolve().parent.parent / "assets" / "w8a8_bench.py"
)

_REQUIRED_FILE_SETS = (
    # Current generated-repository contract.
    (
        "int8_w8a8_gemm_api.py",
        "w8a8_backend.py",
        "w8a8_bench.py",
        "setup.py",
        "csrc/bindings.cpp",
        "csrc/w8a8_gemm_hip.hip",
    ),
    # Legacy optimize-existing-repository contract.
    (
        "w8a8_gemm.py",
        "csrc/bindings.cpp",
        "csrc/w8a8_gemm_hip.hip",
    ),
)
_SOURCE_ONLY_AGENT_ARGS = [
    "--disallowedTools",
    "Bash,Skill,WebFetch,WebSearch",
]
_ISA_AGENT_ARGS = [
    "--disallowedTools",
    "Bash,WebFetch,WebSearch",
]
_MAX_IN_ROUND_REPAIRS = 4
_MAX_INFRASTRUCTURE_RECOVERY_ROUNDS = 5
_MAX_PHASE_EXTENSION_ROUNDS = 8
_REQUIRED_VALID_ISA_GUIDED_ROUNDS = 2
_PLATEAU_MAX_REGRESSION_PERCENT = 2.0
_SHADOW_MIN_IMPROVEMENT_PERCENT = 0.3
_AGENT_TIMEOUT_S = 900
_AGENT_STUCK_TIMEOUT_S = 600
_UNSUPPORTED_SKILL_CLAIMS = (
    "scalar is optimal",
    "dumma is unavailable",
    "does not support int8 dumma",
    "lacks int8",
    "hiplaunchkernelggl cannot",
    "template kernels cannot",
)


def isa_round_policy(
    *,
    iteration: int,
    max_iterations: int,
    history: list[Dict[str, Any]],
) -> Dict[str, Any]:
    """Gate ISA Skills and raw asm behind completed HIP-only exploration."""
    required_hip_rounds = max(1, max_iterations - 2)
    base = {
        "phase": "hip_only",
        "skill_allowed": False,
        "raw_inline_asm_allowed": False,
        "plateau": False,
        "max_iterations": max_iterations,
        "required_valid_hip_rounds": required_hip_rounds,
        "required_valid_isa_guided_rounds": _REQUIRED_VALID_ISA_GUIDED_ROUNDS,
        "reason": "At least eight HIP-only rounds are required.",
    }
    valid = [
        record for record in history
        if record.get("build_success") is True
        and record.get("correctness_passed") is True
        and (record.get("metrics") or {}).get("graph_capture_passed") is True
    ]
    valid_hip = [
        record for record in valid
        if (record.get("isa_policy") or {}).get("phase", "hip_only")
        == "hip_only"
    ]
    if len(valid_hip) < required_hip_rounds:
        return {
            **base,
            "valid_hip_rounds": len(valid_hip),
            "reason": (
                f"Require {required_hip_rounds} completed correct HIP-only "
                "experiments; infrastructure and invalid-candidate failures "
                f"do not count. Current valid HIP rounds: {len(valid_hip)}."
            ),
        }

    recent = valid_hip[-3:]
    improvements = [
        float((record.get("acceptance") or {}).get(
            "improvement_percent", float("inf")
        ))
        for record in recent
    ]
    plateau = (
        len(recent) == 3
        and all(
            -_PLATEAU_MAX_REGRESSION_PERCENT <= value < 2.0
            for value in improvements
        )
    )
    if not plateau:
        return {
            **base,
            "reason": (
                "HIP plateau is not proven: require three recent valid "
                "HIP candidates within [-2%, +2%) of the then-current best, "
                "after the required HIP exploration. Large regressions do "
                "not prove a plateau. Continue HIP-only work."
            ),
            "recent_valid_improvements_percent": improvements,
        }

    policy = {
        **base,
        "phase": "isa_guided_hip",
        "skill_allowed": True,
        "plateau": True,
        "reason": (
            "HIP plateau proven. Complete two valid ISA-guided HIP rounds. "
            "One selected ISA Skill may guide each HIP/DUMMA/intrinsic "
            "code-shaping change."
        ),
        "recent_valid_improvements_percent": improvements,
    }
    valid_isa = [
        record for record in valid
        if (record.get("isa_policy") or {}).get("phase")
        == "isa_guided_hip"
    ]
    policy["valid_isa_guided_rounds"] = len(valid_isa)
    if len(valid_isa) < _REQUIRED_VALID_ISA_GUIDED_ROUNDS:
        return policy

    previous = valid_isa[-1]
    isa_plan = previous.get("isa_optimization") or {}
    limitation_confirmed = (
        isa_plan.get("compiler_limitation_confirmed") is True
        and isinstance(isa_plan.get("target_instructions"), list)
        and bool(isa_plan.get("target_instructions"))
        and (previous.get("candidate_isa") or {}).get("available") is True
    )
    if not limitation_confirmed:
        return {
            **policy,
            "reason": (
                "HIP plateau is proven, but the prior ISA-guided round did "
                "not confirm a compiler limitation with target instructions "
                "and trusted candidate ISA. The two-round ISA requirement is "
                "complete, but raw asm remains forbidden."
            ),
        }
    return {
        **policy,
        "phase": "conditional_inline_asm",
        "raw_inline_asm_allowed": True,
        "verified_target_instructions": list(
            isa_plan.get("target_instructions") or []
        ),
        "reason": (
            "Final-round raw asm gate is open for one minimal block targeting "
            "the compiler limitation verified in the preceding round."
        ),
    }


def phase_extension_reason(
    *,
    max_iterations: int,
    history: list[Dict[str, Any]],
) -> str | None:
    """Return the successful phase still owed after the nominal budget."""
    valid = [
        record for record in history
        if record.get("build_success") is True
        and record.get("correctness_passed") is True
        and (record.get("metrics") or {}).get("graph_capture_passed") is True
    ]
    required_hip = max(1, max_iterations - 2)
    valid_hip = [
        record for record in valid
        if (record.get("isa_policy") or {}).get("phase", "hip_only")
        == "hip_only"
    ]
    if len(valid_hip) < required_hip:
        return (
            f"need {required_hip - len(valid_hip)} more valid HIP-only "
            "experiment(s)"
        )

    policy = isa_round_policy(
        iteration=len(history) + 1,
        max_iterations=max_iterations,
        history=history,
    )
    if policy.get("plateau") is not True:
        return "need a bounded three-experiment HIP plateau"

    valid_isa = [
        record for record in valid
        if (record.get("isa_policy") or {}).get("phase")
        == "isa_guided_hip"
    ]
    if len(valid_isa) < _REQUIRED_VALID_ISA_GUIDED_ROUNDS:
        return (
            "need "
            f"{_REQUIRED_VALID_ISA_GUIDED_ROUNDS - len(valid_isa)} more valid "
            "ISA-guided HIP experiment(s)"
        )

    previous_plan = valid_isa[-1].get("isa_optimization") or {}
    limitation_confirmed = (
        previous_plan.get("compiler_limitation_confirmed") is True
        and isinstance(previous_plan.get("target_instructions"), list)
        and bool(previous_plan.get("target_instructions"))
        and (valid_isa[-1].get("candidate_isa") or {}).get("available") is True
    )
    valid_inline = [
        record for record in valid
        if (record.get("isa_policy") or {}).get("phase")
        == "conditional_inline_asm"
    ]
    if limitation_confirmed and not valid_inline:
        return "need one conditional inline-asm experiment"
    return None


def pmc_profile_decision(
    *,
    iteration: int,
    history: list[Dict[str, Any]],
    source_uses_dumma: bool,
    isa_policy: Dict[str, Any],
) -> Dict[str, Any]:
    """Profile only when counters can change the next optimization decision."""
    if isa_policy.get("skill_allowed"):
        return {
            "profile": True,
            "reason": "late-round ISA or plateau decision requires fresh PMC",
        }
    if iteration == 1:
        if source_uses_dumma:
            return {
                "profile": True,
                "reason": "usable DUMMA bootstrap needs one initial profile",
            }
        return {
            "profile": False,
            "reason": (
                "skip PMC for scalar bootstrap; first establish a validated "
                "DUMMA/current-best implementation"
            ),
        }
    if history and history[-1].get("accepted") is True:
        return {
            "profile": True,
            "reason": "the preceding round established a new official best",
        }
    return {
        "profile": False,
        "reason": (
            "current best is unchanged; reuse matching prior counters and "
            "reserve fresh PMC for accepted-best or late ISA decisions"
        ),
    }


def _cached_pmc_evidence(
    profile_root: Path,
    source_digest: str,
) -> Dict[str, Any] | None:
    candidates = sorted(
        profile_root.glob("iteration*/pmc.json"),
        key=lambda path: path.stat().st_mtime_ns,
        reverse=True,
    )
    for path in candidates:
        try:
            evidence = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if (
            evidence.get("available") is not False
            and evidence.get("source_hip_digest") == source_digest
        ):
            return {
                **evidence,
                "reused": True,
                "reused_from": str(path),
            }
    return None


def _compact_metrics_for_prompt(metrics: Dict[str, Any]) -> Dict[str, Any]:
    """Keep decision signals while leaving full samples in the fact ledger."""
    keys = (
        "median_us", "p90_us", "min_us", "max_us", "passed",
        "graph_capture_passed", "mismatch_count", "max_abs_error",
        "logical_tops", "algorithmic_bandwidth_gb_s", "shape",
        "official_best_median_us", "official_best_p90_us",
        "shadow_candidate_active", "correctness_passed_in_precheck",
    )
    compact = {key: metrics.get(key) for key in keys if key in metrics}
    fallback = metrics.get("paired_m2_fallback_validation")
    if isinstance(fallback, dict):
        compact["paired_m2_fallback_validation"] = {
            key: fallback.get(key)
            for key in (
                "passed", "graph_capture_passed", "mismatch_count",
                "median_us", "p90_us", "shape",
            )
            if key in fallback
        }
    return compact


def _compact_pmc_for_prompt(evidence: Dict[str, Any]) -> Dict[str, Any]:
    """Retain actionable PMC/ISA facts without repeating raw artifacts."""
    compact = {
        key: evidence.get(key)
        for key in (
            "available", "reused", "reused_from", "skipped", "skip_reason",
            "error", "shape", "source_hip_digest", "compile_cache_key",
            "primary_kernel_name", "profiled_duration_us", "grid_blocks",
            "workgroup_size", "lds_bytes", "scratch_bytes", "arch_vgpr",
            "accum_vgpr", "sgpr", "wave_size", "counters", "l2_hits",
            "l2_misses", "l2_hit_rate_percent", "device_cu_count",
            "interpretation_guard",
        )
        if key in evidence
    }
    memory = evidence.get("memory_traffic")
    if isinstance(memory, dict):
        compact["memory_traffic"] = {
            key: memory.get(key)
            for key in (
                "available", "read_bytes_per_operator_replay",
                "write_bytes_per_operator_replay",
                "total_bytes_per_operator_replay",
                "counter_derived_operator_hbm_bandwidth_gb_s", "reason",
            )
            if key in memory
        }
    isa = evidence.get("isa")
    if isinstance(isa, dict):
        compact["isa"] = {
            key: isa.get(key)
            for key in (
                "available", "kernel_name", "instruction_counts",
                "waitcnt_expressions", "resources",
                "profiled_launch_resources", "key_instruction_excerpt",
                "error", "interpretation_guard",
            )
            if key in isa
        }
    return compact


def is_infrastructure_failure(reason: str | None) -> bool:
    normalized = str(reason or "").lower()
    return any(token in normalized for token in (
        "timeout", "timed out", "killed", "no result", "exit 143",
    ))


def evaluate_candidate_acceptance(
    *,
    passed: bool,
    metrics: Dict[str, Any],
    best_metrics: Dict[str, Any],
    minimum_improvement_percent: float,
    shadow_metrics: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """Apply stable median and no-P90-regression acceptance gates."""
    candidate_us = float(metrics.get("median_us") or float("inf"))
    best_us = float(best_metrics["median_us"])
    candidate_p90 = float(metrics.get("p90_us") or candidate_us)
    best_p90 = float(best_metrics.get("p90_us") or best_us)
    improvement = (best_us / candidate_us - 1.0) * 100.0
    p90_guard_passed = candidate_p90 <= best_p90
    accepted = (
        passed
        and candidate_us < best_us
        and improvement >= minimum_improvement_percent
        and p90_guard_passed
    )
    shadow_us = float(
        (shadow_metrics or {}).get("median_us") or float("inf")
    )
    shadow_p90 = float(
        (shadow_metrics or {}).get("p90_us") or shadow_us
    )
    improves_shadow = (
        candidate_us < shadow_us and candidate_p90 <= shadow_p90
    )
    shadow_eligible = (
        passed
        and not accepted
        and _SHADOW_MIN_IMPROVEMENT_PERCENT <= improvement
        < minimum_improvement_percent
        and p90_guard_passed
        and improves_shadow
    )
    return {
        "accepted": accepted,
        "candidate_us": candidate_us,
        "best_us": best_us,
        "candidate_p90_us": candidate_p90,
        "best_p90_us": best_p90,
        "improvement_percent": improvement,
        "p90_guard_passed": p90_guard_passed,
        "shadow_eligible": shadow_eligible,
        "shadow_base_us": (
            shadow_us if shadow_us != float("inf") else None
        ),
        "improves_shadow": improves_shadow,
    }


def evaluate_final_target(
    *,
    baseline: Dict[str, Any],
    validation: Dict[str, Any],
    target_improvement_percent: float,
) -> Dict[str, Any]:
    """Evaluate the user target only against the fixed baseline at report time."""
    validation_shapes = validation.get("shapes", validation)
    shapes: Dict[str, Any] = {}
    for shape_id, baseline_record in baseline.items():
        final_record = validation_shapes.get(shape_id) or {}
        final_metrics = final_record.get("metrics", final_record)
        baseline_value = baseline_record.get("median_us")
        final_value = final_metrics.get("median_us")
        baseline_us = float(baseline_value) if baseline_value else None
        final_us = float(final_value) if final_value else None
        improvement = (
            (baseline_us / final_us - 1.0) * 100.0
            if baseline_us is not None and final_us is not None else None
        )
        shapes[shape_id] = {
            "baseline_us": baseline_us,
            "final_us": final_us,
            "improvement_percent": improvement,
            "target_met": (
                final_record.get("passed") is True
                and improvement is not None
                and improvement >= target_improvement_percent
            ),
        }
    return {
        "target_improvement_percent": target_improvement_percent,
        "semantics": "final validated result versus fixed baseline",
        "all_shapes_met": bool(shapes) and all(
            record["target_met"] for record in shapes.values()
        ),
        "shapes": shapes,
    }


def experiment_fact_ledger(worker_root: Path) -> list[Dict[str, Any]]:
    """Return control-plane facts suitable for Skill authoring."""
    facts: list[Dict[str, Any]] = []
    for path in sorted((worker_root / "runs").glob("*/experiments.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                record = json.loads(line)
            except ValueError:
                continue
            if not isinstance(record, dict):
                continue
            facts.append({
                "shape_id": record.get("shape_id"),
                "iteration": record.get("iteration"),
                "accepted": record.get("accepted"),
                "build_success": record.get("build_success"),
                "correctness_passed": record.get("correctness_passed"),
                "metrics": record.get("metrics") or {},
                "architecture": record.get("architecture") or {},
                "baseline_us": record.get("baseline_us"),
                "speedup": record.get("speedup"),
                "p90_guard_passed": record.get("p90_guard_passed"),
                "failure_category": (
                    "compile_or_agent_failure"
                    if record.get("build_success") is False
                    else None
                ),
            })
    return facts


def validate_skill_draft(
    content: str,
    facts: list[Dict[str, Any]],
) -> None:
    """Reject common unsupported conclusions before they become memory."""
    lowered = content.lower()
    violations = [
        claim for claim in _UNSUPPORTED_SKILL_CLAIMS
        if claim in lowered
    ]
    if violations:
        raise ValueError(
            "skill draft contains unsupported conclusions: "
            f"{violations}"
        )
    if facts and not any(item.get("accepted") for item in facts):
        markers = (
            "no accepted",
            "none accepted",
            "zero accepted",
            "0 accepted",
            "no candidate was accepted",
            "no optimization was accepted",
        )
        if not any(marker in lowered for marker in markers):
            raise ValueError(
                "skill draft does not clearly state that no candidate was "
                "accepted"
            )


def archive_iteration_candidate(
    source: Path,
    iteration_dir: Path,
    extra_paths: list[str] | None = None,
) -> list[str]:
    """Keep an immutable-looking copy of implementation code for one round.

    The operator API contract deliberately remains shared in ``source``.
    Candidate implementation and HIP files are copied before a rejected
    working tree is restored or a later agent overwrites it.
    """
    relative_paths = {
        "w8a8_gemm.py",
        "w8a8_backend.py",
        "setup.py",
        "csrc/bindings.cpp",
        *(extra_paths or []),
    }
    for path in source.rglob("*.hip"):
        if ".git" not in path.parts:
            relative_paths.add(path.relative_to(source).as_posix())

    archived = []
    for relative in sorted(relative_paths):
        if (
            relative in {W8A8_API_FILENAME, "proposal.json"}
            or relative.startswith("references/")
        ):
            continue
        candidate = source / relative
        if not candidate.is_file():
            continue
        destination = iteration_dir / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(candidate, destination)
        archived.append(relative)
    return archived


def candidate_iteration_destination(
    workspace_dir: Path,
    assignment: WorkerAssignment,
    shape_id: str,
    iteration: int,
) -> Path | None:
    """Return the stable kernel-repo path for one immutable round snapshot."""
    candidate_dir = (
        workspace_dir / "main" / "candidates" / assignment.worker_id
    )
    if not candidate_dir.is_dir() or candidate_dir.is_symlink():
        return None
    if len(assignment.shape_ids) == 1:
        return candidate_dir / f"iteration{iteration}"
    return candidate_dir / shape_id / f"iteration{iteration}"


def publish_iteration_candidate(
    iteration_dir: Path,
    destination: Path,
) -> None:
    """Copy one archived round into the user-visible kernel repository."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(iteration_dir, destination, dirs_exist_ok=True)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def compiled_kernel_object(worker_root: Path) -> Path:
    """Return the exact HIP object produced by the trusted worker benchmark."""
    current_build = worker_root / "cache" / "current_build.json"
    if current_build.is_file():
        try:
            metadata = json.loads(current_build.read_text(encoding="utf-8"))
            extension_name = str(metadata["extension_name"])
            expected = (
                worker_root / "cache" / "torch" / extension_name
                / "w8a8_gemm_hip.cuda.o"
            )
            if expected.is_file():
                return expected
        except (KeyError, TypeError, ValueError, OSError):
            pass
    expected = (
        worker_root / "cache" / "torch" / "metainfer_w8a8_backend"
        / "w8a8_gemm_hip.cuda.o"
    )
    if expected.is_file():
        return expected
    matches = sorted(
        (worker_root / "cache" / "torch").glob(
            "**/w8a8_gemm_hip.cuda.o"
        ),
        key=lambda path: path.stat().st_mtime_ns,
        reverse=True,
    )
    if not matches:
        raise FileNotFoundError(
            "trusted benchmark did not leave a compiled "
            "w8a8_gemm_hip.cuda.o"
        )
    return matches[0]


def snapshot_accepted_kernel_artifact(
    *,
    worker_root: Path,
    shape_id: str,
    shape: Dict[str, Any],
    metrics: Dict[str, Any],
    commit: str,
    isa_evidence: Dict[str, Any] | None = None,
    isa_dir: Path | None = None,
) -> Dict[str, Any]:
    """Freeze the benchmarked HIP source and its exact compiled object."""
    source_hip = worker_root / "source" / "csrc" / "w8a8_gemm_hip.hip"
    object_file = compiled_kernel_object(worker_root)
    destination = worker_root / "accepted" / _safe(shape_id)
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True)
    archived_hip = destination / "kernel.hip"
    archived_object = destination / "kernel.cuda.o"
    shutil.copy2(source_hip, archived_hip)
    shutil.copy2(object_file, archived_object)
    build_ninja = object_file.parent / "build.ninja"
    if build_ninja.is_file():
        shutil.copy2(build_ninja, destination / "build.ninja")
    archived_isa: list[str] = []
    if isa_dir is not None:
        for name in ("gfx928.co", "metadata.txt", "isa.txt"):
            candidate = isa_dir / name
            if candidate.is_file():
                shutil.copy2(candidate, destination / name)
                archived_isa.append(name)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "shape_id": shape_id,
        "shape": {
            key: int(shape[key]) for key in ("M", "N", "K")
        },
        "commit": commit,
        "source": str(archived_hip.relative_to(worker_root)),
        "object": str(archived_object.relative_to(worker_root)),
        "source_sha256": _sha256_file(archived_hip),
        "object_sha256": _sha256_file(archived_object),
        "compile_target": "gfx928",
        "compiler_contract": (
            "trusted torch cpp_extension worker build; exact accepted object"
        ),
        "launch_abi": "launch_w8a8_gemm",
        "pack_abi": "launch_pack_w8a8_weight",
        "metrics": {
            key: metrics[key]
            for key in (
                "median_us", "p90_us", "min_us", "max_us",
                "graph_capture_passed", "mismatch_count", "max_abs_error",
            )
            if key in metrics
        },
        "isa_artifacts": archived_isa,
        "isa_evidence": {
            key: value for key, value in (isa_evidence or {}).items()
            if key != "artifact_paths"
        },
    }
    write_json(destination / "manifest.json", manifest)
    return manifest


def _check_required_files(repo: Path) -> bool:
    """Return True for either supported complete kernel-repo layout."""
    return any(
        all((repo / name).is_file() for name in required_files)
        for required_files in _REQUIRED_FILE_SETS
    )


class W8A8Runner:
    def __init__(self, worker_root: Path, gpu: int) -> None:
        self.worker_root = worker_root
        self.source = worker_root / "source"
        staged_harness = self.source / "w8a8_bench.py"
        self.harness = (
            staged_harness if staged_harness.is_file() else HARNESS
        )
        self.env = dict(os.environ)
        bind_worker_gpu(self.env, gpu)
        self.env.update({
            "MAX_JOBS": "2",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTORCH_ROCM_ARCH": "gfx928",
            "TORCH_EXTENSIONS_DIR": str(worker_root / "cache" / "torch"),
            "TRITON_CACHE_DIR": str(worker_root / "cache" / "triton"),
            "XDG_CACHE_HOME": str(worker_root / "cache" / "xdg"),
            "TMPDIR": str(worker_root / "cache" / "tmp"),
        })
        for key in (
            "TORCH_EXTENSIONS_DIR", "TRITON_CACHE_DIR", "XDG_CACHE_HOME",
            "TMPDIR",
        ):
            Path(self.env[key]).mkdir(parents=True, exist_ok=True)

    def _prepare_compile_cache(self) -> Dict[str, Any]:
        """Stage immutable content-addressed compiler inputs for Ninja reuse."""
        inputs = [self.source / "csrc" / "bindings.cpp"]
        prebuilt = sorted((self.source / "prebuilt").glob("*.o"))
        dispatch = self.source / "csrc" / "w8a8_dispatch.cpp"
        if prebuilt:
            inputs.extend([dispatch, *prebuilt])
        else:
            inputs.append(self.source / "csrc" / "w8a8_gemm_hip.hip")
        missing = [str(path) for path in inputs if not path.is_file()]
        if missing:
            raise FileNotFoundError(
                f"compile inputs are missing: {missing}"
            )
        digest = hashlib.sha256()
        for path in inputs:
            relative = path.relative_to(self.source)
            digest.update(str(relative).encode("utf-8"))
            digest.update(b"\0")
            digest.update(bytes.fromhex(_sha256_file(path)))
        build_key = digest.hexdigest()
        cache_source = (
            self.worker_root / "cache" / "compile_sources" / build_key
        )
        if not cache_source.is_dir():
            temporary = cache_source.with_name(
                f"{cache_source.name}.tmp-{os.getpid()}"
            )
            if temporary.exists():
                shutil.rmtree(temporary)
            for path in inputs:
                destination = temporary / path.relative_to(self.source)
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, destination)
            write_json(temporary / "manifest.json", {
                "build_key": build_key,
                "inputs": {
                    str(path.relative_to(self.source)): _sha256_file(path)
                    for path in inputs
                },
                "arch": "gfx928",
                "flags": ["-O3", "--offload-arch=gfx928"],
            })
            temporary.replace(cache_source)
        extension_name = f"metainfer_w8a8_backend_{build_key[:24]}"
        self.env["METAINFER_W8A8_COMPILE_SOURCE_DIR"] = str(cache_source)
        self.env["METAINFER_W8A8_BUILD_KEY"] = build_key
        metadata = {
            "build_key": build_key,
            "extension_name": extension_name,
            "compile_source_dir": str(cache_source),
        }
        write_json(self.worker_root / "cache" / "current_build.json", metadata)
        return metadata

    def probe(self) -> Dict[str, Any]:
        result = _run([
            "python3", str(self.harness), "--source", str(self.source),
            "--m", "1", "--n", "1", "--k", "1", "--probe",
        ], cwd=self.source, env=self.env, timeout=60)
        return _last_json(result.stdout)

    def benchmark(
        self,
        shape: Dict[str, Any],
        *,
        warmups: int | None = None,
        samples: int | None = None,
        replays_per_sample: int | None = None,
        check_correctness: bool = True,
    ) -> Dict[str, Any]:
        try:
            m = int(shape["M"])
            n = int(shape["N"])
            k = int(shape["K"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("W8A8 shapes require integer M, N and K") from exc
        command = [
            "python3", str(self.harness), "--source", str(self.source),
            "--m", str(m), "--n", str(n), "--k", str(k),
            "--reference-cache-dir",
            str(self.worker_root / "cache" / "references"),
        ]
        if warmups is not None:
            command.extend(["--warmups", str(warmups)])
        if samples is not None:
            command.extend(["--samples", str(samples)])
        if replays_per_sample is not None:
            command.extend([
                "--replays-per-sample", str(replays_per_sample)
            ])
        if not check_correctness:
            command.append("--skip-correctness")
        build = self._prepare_compile_cache()
        result = _run(
            command, cwd=self.source, env=self.env, timeout=900
        )
        metrics = _last_json(result.stdout)
        metrics["compile_cache_key"] = build["build_key"]
        return metrics

    def profile_pmc(
        self,
        shape: Dict[str, Any],
        output_dir: Path,
    ) -> Dict[str, Any]:
        """Profile the current trusted source once and return compact PMC data."""
        try:
            m = int(shape["M"])
            n = int(shape["N"])
            k = int(shape["K"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("W8A8 shapes require integer M, N and K") from exc
        script = self.source / "profile_pmc.sh"
        if not script.is_file():
            raise FileNotFoundError(
                f"trusted PMC script is missing: {script}"
            )
        output_dir.mkdir(parents=True, exist_ok=True)
        build = self._prepare_compile_cache()
        _run(
            [
                "bash",
                str(script),
                str(self.harness),
                str(self.source),
                str(m),
                str(n),
                str(k),
                str(output_dir),
            ],
            cwd=self.source,
            env=self.env,
            timeout=900,
        )
        evidence = parse_pmc_csv(output_dir / "pmc.csv")
        read_csv = output_dir / "pmc-read.csv"
        write_csv = output_dir / "pmc-write.csv"
        if read_csv.is_file() and write_csv.is_file():
            evidence["memory_traffic"] = parse_memory_traffic_csv(
                read_csv, write_csv
            )
        else:
            evidence["memory_traffic"] = {
                "available": False,
                "reason": (
                    "This repository predates trusted HBM traffic "
                    "profiling. General PMC remains valid; do not make an "
                    "HBM bandwidth claim."
                ),
            }
        evidence["shape"] = {"M": m, "N": n, "K": k}
        evidence["source_hip_digest"] = file_digest(
            self.source / "csrc" / "w8a8_gemm_hip.hip"
        )
        evidence["compile_cache_key"] = build["build_key"]
        evidence["csv_path"] = str(output_dir / "pmc.csv")
        evidence["memory_csv_paths"] = {
            "read": str(read_csv),
            "write": str(write_csv),
        }
        return evidence

    def inspect_isa(
        self,
        output_dir: Path,
        *,
        kernel_names: list[str] | None = None,
        primary_kernel_name: str | None = None,
    ) -> Dict[str, Any]:
        """Audit the exact object most recently built by the trusted runner."""
        evidence = inspect_gfx928_object(
            compiled_kernel_object(self.worker_root),
            output_dir,
            kernel_names=kernel_names,
            primary_kernel_name=primary_kernel_name,
        )
        evidence["source_hip_digest"] = file_digest(
            self.source / "csrc" / "w8a8_gemm_hip.hip"
        )
        return evidence


class RealW8A8OptimizationPipeline:
    def __init__(
        self,
        *,
        req: Dict[str, Any],
        state_dir: Path,
        workspace_dir: Path,
        store: StateStore,
        manager: SubAgentManager,
    ) -> None:
        self.req = req
        self.state_dir = state_dir
        self.workspace_dir = workspace_dir
        self.store = store
        self.manager = manager
        self._current_phase = phases.PREPARE
        self._progress_lock = threading.Lock()
        self._reported_iteration = 0
        self._worker_failures: Dict[str, Dict[str, Any]] = {}

    def _phase(self, phase: str, **payload: Any) -> None:
        self._current_phase = phase
        self.store.update_run(current_phase=phase)
        self.store.append_timeline("phase_start", {"phase": phase, **payload})

    def run(self, *, dry_run: bool = False) -> Dict[str, Any]:
        task_id = str(self.req.get("task_id", "task"))
        self.store.init_or_resume(task_id)
        self.store.update_run(
            current_iteration=0,
            finished=False,
            final_status=None,
            last_outcome=None,
            last_transition_label=None,
            notes=[],
        )
        started = time.time()
        try:
            self._phase(phases.PREPARE)
            config = load_config(self.req)
            self._validate_contract(config)
            self._prepare_worktrees(config, task_id)
            plan = self._plan(config)
            write_json(self.workspace_dir / "plan.json", plan)
            if dry_run:
                return plan

            self._phase(phases.BASELINE)
            baseline = self._parallel_baseline(config)

            self._phase(phases.EXPLORE, workers=len(config.assignments))
            workers = self._parallel_agents(config, baseline)

            self._phase(phases.SYNTHESIZE)
            completed_assignments = [
                item for item in config.assignments
                if item.worker_id in workers
            ]
            merged_skill = self._author_merged_skill(
                config, completed_assignments
            )

            self._phase(phases.VALIDATE)
            validation = self._serial_validate(config, workers)

            self._phase(phases.REPORT)
            final_target = evaluate_final_target(
                baseline=baseline,
                validation=validation,
                target_improvement_percent=(
                    config.minimum_improvement_percent
                ),
            )
            report = {
                "schema_version": SCHEMA_VERSION,
                "task_id": task_id,
                "task_type": "dcu-kernel-auto-opt",
                "mode": "real-int8-w8a8-gemm",
                "started_at": started,
                "finished_at": time.time(),
                "duration_s": round(time.time() - started, 4),
                "config": plan,
                "baseline": baseline,
                "workers": workers,
                "worker_failures": self._worker_failures,
                "merged_skill": merged_skill,
                "final_validation": validation,
                "final_target": final_target,
                "real_gpu_used": True,
                "target_repo_modified": False,
                "status": (
                    "partial_success" if self._worker_failures else "success"
                ),
            }
            write_json(self.workspace_dir / "final_report.json", report)
            self.store.update_run(
                current_iteration=config.mock_iterations,
                current_phase=phases.FINISHED,
                finished=True,
                final_status="success",
                last_outcome="ok",
                last_transition_label="real W8A8 optimization complete",
            )
            self.store.append_timeline(
                "orchestrator_success",
                {
                    "workers": len(workers),
                    "real_gpu_used": True,
                    "operator": "int8_w8a8_gemm",
                },
            )
            return report
        except Exception as exc:
            draft.unlink(missing_ok=True)
            self.store.append_timeline(
                "orchestrator_error", {"error": repr(exc)}
            )
            self.store.update_run(
                current_phase=self._current_phase,
                finished=True,
                final_status="stopped",
                last_outcome="infra_fail",
                notes=[str(exc)],
            )
            raise

    @staticmethod
    def _validate_contract(config: OptimizerConfig) -> None:
        if config.operator != "Quantized GEMM":
            raise ValueError(
                "Real INT8 W8A8 GEMM mode requires operator=Quantized GEMM"
            )
        if config.dtype != "INT8 W8A8":
            raise ValueError(
                "Real INT8 W8A8 GEMM mode requires dtype=INT8 W8A8"
            )
        for shape in config.shapes.values():
            for key in ("M", "N", "K"):
                if key not in shape.params:
                    raise ValueError(f"{shape.id} is missing {key}")

    def _prepare_worktrees(
        self, config: OptimizerConfig, task_id: str
    ) -> None:
        self.workspace_dir.mkdir(parents=True, exist_ok=True)
        seed = self.workspace_dir / "main"
        if not (seed / ".git").exists():
            source = config.target_repo_path
            repo_exists = (
                source is not None
                and source.is_dir()
                and _check_required_files(source)
            )
            if repo_exists:
                assert source is not None
                shutil.copytree(
                    source,
                    seed,
                    dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns(
                        ".git", "__pycache__", "*.pyc", "profiles",
                        "build", ".pytest_cache",
                    ),
                )
                _run(["git", "init"], cwd=seed)
                _run(["git", "config", "user.name", "MetaInfer Agent"], cwd=seed)
                _run([
                    "git", "config", "user.email", "metainfer@localhost",
                ], cwd=seed)
                _run(["git", "add", "."], cwd=seed)
                _run([
                    "git", "commit", "-m", "seed extracted INT8 W8A8 GEMM",
                ], cwd=seed)
            else:
                raise RuntimeError(
                    "The optimize-existing-repo mode requires a complete "
                    "kernel repository. Use Generate + Optimize mode when "
                    "child agents should create kernels from the fixed API."
                )
        for assignment in config.assignments:
            root = self.workspace_dir / "workers" / assignment.worker_id
            for name in ("build", "cache", "logs", "runs", "artifacts"):
                (root / name).mkdir(parents=True, exist_ok=True)
            source = root / "source"
            if not source.exists():
                branch = f"agent/{_safe(task_id)}/{assignment.worker_id}"
                _run([
                    "git", "worktree", "add", "-b", branch,
                    str(source), "HEAD",
                ], cwd=seed)
        for name in ("shared_baseline", "final_validation", "skills"):
            (self.workspace_dir / name).mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _plan(config: OptimizerConfig) -> Dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "execution_mode": config.execution_mode,
            "operator": "INT8 W8A8 GEMM",
            "dtype": config.dtype,
            "hardware": config.hardware,
            "kernel_language": config.kernel_language,
            "claude_model": config.claude_model,
            "max_iterations": config.mock_iterations,
            "minimum_improvement_percent": (
                config.minimum_improvement_percent
            ),
            "minimum_improvement_semantics": (
                "final validated result versus fixed baseline"
            ),
            "round_acceptance_improvement_percent": (
                ROUND_ACCEPTANCE_IMPROVEMENT_PERCENT
            ),
            "shape_scope": config.shape_scope,
            "assignment_mode": config.assignment_mode,
            "target_repo_path": (
                str(config.target_repo_path) if config.target_repo_path else None
            ),
            "harness": "trusted PyTorch FP32-accumulate W8A8 reference",
            "contract": {
                "A": "int8[M,K], row-major contiguous",
                "B": "int8[K,N], row-major contiguous",
                "x_scale": "float32[M,1]",
                "weight_scale": "float32[N,1]",
                "output": "bfloat16[M,N]",
            },
            "shapes": [
                {"id": shape.id, **shape.params}
                for shape in config.shapes.values()
            ],
            "assignments": [
                {
                    "worker_id": item.worker_id,
                    "gpu": item.gpu,
                    "shapes": item.shape_ids,
                }
                for item in config.assignments
            ],
            "real_gpu_used": True,
        }

    def _parallel_baseline(
        self, config: OptimizerConfig
    ) -> Dict[str, Dict[str, Any]]:
        output: Dict[str, Dict[str, Any]] = {}

        def run_one(
            assignment: WorkerAssignment,
        ) -> Dict[str, Dict[str, Any]]:
            root = self.workspace_dir / "workers" / assignment.worker_id
            runner = W8A8Runner(root, assignment.gpu)
            _status(
                root, assignment, state="building",
                iteration=0, shape_id=None,
            )
            probe = runner.probe()
            if probe.get("visible_devices") != 1:
                raise RuntimeError(
                    f"{assignment.worker_id} sees "
                    f"{probe.get('visible_devices')} GPUs"
                )
            _status(
                root, assignment, state="baseline",
                iteration=0, shape_id=None, probe=probe,
            )
            result: Dict[str, Dict[str, Any]] = {}
            for shape_id in assignment.shape_ids:
                shape = config.shapes[shape_id].params
                metrics = runner.benchmark(shape)
                if not metrics.get("passed"):
                    raise RuntimeError(
                        f"baseline correctness failed: {shape_id}"
                    )
                result[shape_id] = fixed_triton_graph_baseline(
                    shape_id,
                    shape,
                    bootstrap_metrics=metrics,
                )
            return result

        with ThreadPoolExecutor(max_workers=len(config.assignments)) as pool:
            futures = {
                pool.submit(run_one, item): item
                for item in config.assignments
            }
            for future in as_completed(futures):
                assignment = futures[future]
                try:
                    output.update(future.result())
                except Exception as exc:
                    error = str(exc)
                    state = (
                        "timed_out"
                        if any(
                            token in error.lower()
                            for token in (
                                "timeout", "timed out", "stuck", "killed",
                            )
                        )
                        else "failed"
                    )
                    failure = {
                        "worker_id": assignment.worker_id,
                        "physical_gpu": assignment.gpu,
                        "shape_ids": assignment.shape_ids,
                        "state": state,
                        "stage": "baseline",
                        "error": error,
                        "timestamp": time.time(),
                    }
                    self._worker_failures[assignment.worker_id] = failure
                    root = (
                        self.workspace_dir / "workers"
                        / assignment.worker_id
                    )
                    _status(
                        root, assignment, state=state,
                        iteration=0, shape_id=None, error=error,
                    )
                    write_json(root / "failure.json", failure)
                    self.store.append_timeline("worker_failed", failure)
        completed_workers = {
            assignment.worker_id
            for assignment in config.assignments
            if assignment.worker_id not in self._worker_failures
        }
        minimum_completed = 1
        if len(completed_workers) < minimum_completed:
            raise RuntimeError(
                "parallel baseline has fewer than "
                f"{minimum_completed} successful GPU workers "
                f"({len(completed_workers)}/{len(config.assignments)} completed)"
            )
        write_json(
            self.workspace_dir / "shared_baseline" / "results.json",
            {"schema_version": SCHEMA_VERSION, "shapes": output},
        )
        return output

    def _parallel_agents(
        self,
        config: OptimizerConfig,
        baseline: Dict[str, Dict[str, Any]],
    ) -> Dict[str, Any]:
        output: Dict[str, Any] = {}
        active_assignments = [
            item for item in config.assignments
            if item.worker_id not in self._worker_failures
        ]
        with ThreadPoolExecutor(
            max_workers=max(1, len(active_assignments))
        ) as pool:
            futures = {
                pool.submit(
                    self._run_worker, config, assignment, baseline
                ): assignment
                for assignment in active_assignments
            }
            for future in as_completed(futures):
                assignment = futures[future]
                root = self.workspace_dir / "workers" / assignment.worker_id
                try:
                    result = future.result()
                except Exception as exc:
                    error = str(exc)
                    state = (
                        "timed_out"
                        if any(
                            token in error.lower()
                            for token in ("timeout", "timed out", "stuck", "killed")
                        )
                        else "failed"
                    )
                    failure = {
                        "worker_id": assignment.worker_id,
                        "physical_gpu": assignment.gpu,
                        "shape_ids": assignment.shape_ids,
                        "state": state,
                        "error": error,
                        "timestamp": time.time(),
                    }
                    self._worker_failures[assignment.worker_id] = failure
                    _status(
                        root, assignment, state=state,
                        iteration=0, shape_id=None, error=error,
                    )
                    write_json(root / "failure.json", failure)
                    self.store.append_timeline("worker_failed", failure)
                    continue
                _status(
                    root, assignment, state="skill_writing",
                    iteration=config.mock_iterations, shape_id=None,
                )
                result["skill"] = self._author_worker_skill(config, assignment)
                _status(
                    root, assignment, state="completed",
                    iteration=config.mock_iterations, shape_id=None,
                )
                output[assignment.worker_id] = result
                self.store.append_timeline(
                    "worker_complete",
                    {
                        "worker_id": assignment.worker_id,
                        "skill": result["skill"]["name"],
                    },
                )
        minimum_completed = 1
        if len(output) < minimum_completed:
            raise RuntimeError(
                "parallel explore has fewer than "
                f"{minimum_completed} successful GPU workers "
                f"({len(output)}/{len(config.assignments)} completed)"
            )
        if self._worker_failures:
            self.store.append_timeline(
                "parallel_explore_degraded",
                {
                    "completed_workers": sorted(output),
                    "ignored_workers": sorted(self._worker_failures),
                    "policy": "continue when at least one worker completes",
                },
            )
        return dict(sorted(output.items()))

    def _author_worker_skill(
        self,
        config: OptimizerConfig,
        assignment: WorkerAssignment,
    ) -> Dict[str, Any]:
        """Ask the finished GPU worker to distill its five-round evidence."""
        root = self.workspace_dir / "workers" / assignment.worker_id
        draft = root / "source" / "worker_skill_draft.md"
        draft.unlink(missing_ok=True)
        fact_ledger = experiment_fact_ledger(root)
        prompt = f"""You are the evidence writer for {assignment.worker_id}.
The optimization rounds are complete. Read `{root / 'result.json'}` and every
`experiments.jsonl` below `{root / 'runs'}`. Write a concise reusable Markdown
skill body to `{draft}`.

The following control-plane fact ledger is authoritative:

```json
{json.dumps(fact_ledger, ensure_ascii=False, indent=2)}
```

Do not add YAML frontmatter and do not modify any other file. Include:
- exact hardware, dtype, operator and shape scope;
- accepted and rejected changes with measured evidence;
- the best reusable optimization rules;
- correctness/performance pitfalls and non-generalizable conclusions;
- how a later agent should validate a transferred idea.

Separate measured facts from hypotheses. Missing or failed measurements must
not be presented as successful optimizations. A compile error proves only that
candidate failed to compile; it does not prove a dtype/API/hardware feature is
unsupported. Do not call a baseline optimal unless an accepted-candidate
search establishes that fact. Do not claim template HIP kernels cannot be
launched; `HIP_KERNEL_NAME(...)` is valid with `hipLaunchKernelGGL`.
"""
        prompt_file = root / "logs" / "worker-skill.prompt.txt"
        prompt_file.write_text(prompt, encoding="utf-8")
        name = f"{assignment.worker_id}-skill"
        self.store.append_timeline(
            "worker_skill_launch", {"worker_id": assignment.worker_id}
        )
        try:
            self.manager.launch(AgentSpec(
                name=name,
                role="dcu_worker_skill_writer",
                prompt_file=prompt_file,
                workdir=root / "source",
                log_dir=root / "logs",
                timeout_s=300,
                stuck_timeout_s=300,
                max_retries=0,
                extra_args=list(_SOURCE_ONLY_AGENT_ARGS),
            ))
            agent_result = self.manager.result(name)
            if (
                agent_result is None
                or not agent_result.success
                or not draft.is_file()
            ):
                raise RuntimeError(
                    agent_result.error
                    if agent_result is not None and agent_result.error
                    else "skill writer did not produce a draft"
                )
            content = draft.read_text(encoding="utf-8", errors="replace")
            if not content.strip():
                raise RuntimeError("skill writer produced an empty draft")
            validate_skill_draft(content, fact_ledger)
            draft.unlink(missing_ok=True)
            self.store.append_timeline(
                "worker_skill_success", {"worker_id": assignment.worker_id}
            )
            return generate_worker_skill(
                config=config,
                assignment=assignment,
                workspace_dir=self.workspace_dir,
                agent_draft=content,
            )
        except Exception as exc:
            draft.unlink(missing_ok=True)
            self.store.append_timeline(
                "worker_skill_fallback",
                {"worker_id": assignment.worker_id, "error": str(exc)},
            )
            return generate_worker_skill(
                config=config,
                assignment=assignment,
                workspace_dir=self.workspace_dir,
            )

    def _author_merged_skill(
        self,
        config: OptimizerConfig,
        assignments: list[WorkerAssignment],
    ) -> Dict[str, Any]:
        """Let the main agent summarize only the workers with usable evidence."""
        workdir = (
            self.workspace_dir / "workers"
            / assignments[0].worker_id / "source"
        )
        draft = workdir / "main_agent_skill_draft.md"
        draft.unlink(missing_ok=True)
        worker_skills = sorted(
            (self.workspace_dir / "skills" / "pending").glob("*/SKILL.md")
        )
        merged_facts = [
            fact
            for assignment in assignments
            for fact in experiment_fact_ledger(
                self.workspace_dir / "workers" / assignment.worker_id
            )
        ]
        prompt = f"""You are the main DCU optimization synthesis agent.
Read these completed worker skills:
{json.dumps([str(path) for path in worker_skills], ensure_ascii=False, indent=2)}

Unavailable GPU workers:
{json.dumps(self._worker_failures, ensure_ascii=False, indent=2)}

Authoritative control-plane fact ledger:
```json
{json.dumps(merged_facts, ensure_ascii=False, indent=2)}
```

Write one concise Markdown skill body to `{draft}`. Do not add YAML frontmatter
and do not modify any other file. Summarize shape routing, accepted evidence,
rejected ideas, integration/validation rules, and fallback behavior. Clearly
mark unavailable shapes as unoptimized; never infer success from a timed-out
or failed worker. Compiler errors are candidate failures, not proof that
DUMMA, a dtype, or a HIP API is unsupported. Never call the baseline optimal
when no candidate was accepted.
"""
        prompt_file = self.workspace_dir / "skills" / "main-agent.prompt.txt"
        prompt_file.write_text(prompt, encoding="utf-8")
        self.store.append_timeline(
            "main_skill_launch",
            {
                "completed_workers": [item.worker_id for item in assignments],
                "ignored_workers": sorted(self._worker_failures),
            },
        )
        try:
            self.manager.launch(AgentSpec(
                name="main-skill-synthesis",
                role="dcu_main_skill_synthesizer",
                prompt_file=prompt_file,
                workdir=workdir,
                log_dir=self.workspace_dir / "skills" / "logs",
                timeout_s=420,
                stuck_timeout_s=420,
                max_retries=0,
                extra_args=list(_SOURCE_ONLY_AGENT_ARGS),
            ))
            agent_result = self.manager.result("main-skill-synthesis")
            if (
                agent_result is None
                or not agent_result.success
                or not draft.is_file()
            ):
                raise RuntimeError(
                    agent_result.error
                    if agent_result is not None and agent_result.error
                    else "main skill synthesizer did not produce a draft"
                )
            content = draft.read_text(encoding="utf-8", errors="replace")
            if not content.strip():
                raise RuntimeError("main skill synthesizer produced an empty draft")
            validate_skill_draft(content, merged_facts)
            draft.unlink(missing_ok=True)
            self.store.append_timeline("main_skill_success", {})
            return generate_merged_skill(
                config=config,
                assignments=assignments,
                workspace_dir=self.workspace_dir,
                agent_draft=content,
                failed_workers=self._worker_failures,
            )
        except Exception as exc:
            self.store.append_timeline(
                "main_skill_fallback", {"error": str(exc)}
            )
            return generate_merged_skill(
                config=config,
                assignments=assignments,
                workspace_dir=self.workspace_dir,
                failed_workers=self._worker_failures,
            )

    def _run_worker(
        self,
        config: OptimizerConfig,
        assignment: WorkerAssignment,
        baseline: Dict[str, Dict[str, Any]],
    ) -> Dict[str, Any]:
        root = self.workspace_dir / "workers" / assignment.worker_id
        source = root / "source"
        runner = W8A8Runner(root, assignment.gpu)
        probe = runner.probe()
        result: Dict[str, Any] = {
            "worker_id": assignment.worker_id,
            "physical_gpu": assignment.gpu,
            "branch": (
                f"agent/{_safe(self.req.get('task_id', 'task'))}/"
                f"{assignment.worker_id}"
            ),
            "worktree_created": True,
            "mode": "real-int8-w8a8-gemm",
            "gpu_probe": probe,
            "shapes": {},
        }
        for shape_id in assignment.shape_ids:
            shape = config.shapes[shape_id]
            verified_experience = load_verified_experience(
                (
                    config.target_repo_path.parent
                    if config.target_repo_path is not None else None
                ),
                shape.params,
                exclude_repo=config.target_repo_path,
            )
            comparison_baseline = baseline[shape_id]
            best_metrics = comparison_baseline.get(
                "bootstrap_metrics", comparison_baseline
            )
            comparison_target = {
                key: value
                for key, value in comparison_baseline.items()
                if key != "bootstrap_metrics"
            }
            best_commit = _run(
                ["git", "rev-parse", "HEAD"], cwd=source
            ).stdout.strip()
            artifact_manifest_path = (
                root / "accepted" / _safe(shape_id) / "manifest.json"
            )
            if not artifact_manifest_path.is_file():
                snapshot_accepted_kernel_artifact(
                    worker_root=root,
                    shape_id=shape_id,
                    shape=shape.params,
                    metrics=best_metrics,
                    commit=best_commit,
                )
            experiments_path = root / "runs" / shape_id / "experiments.jsonl"
            iteration = 1
            attempt_limit = config.mock_iterations
            infrastructure_recoveries = 0
            phase_extensions = 0
            shadow_metrics: Dict[str, Any] | None = None
            shadow_path = (
                root / "shadow" / _safe(shape_id) / "w8a8_gemm_hip.hip"
            )
            session_path = (
                root / "sessions" / f"{_safe(shape_id)}.json"
            )
            shape_session_id: str | None = None
            if session_path.is_file():
                try:
                    stored_session = json.loads(
                        session_path.read_text(encoding="utf-8")
                    )
                    candidate_session = stored_session.get("session_id")
                    if isinstance(candidate_session, str) and candidate_session:
                        shape_session_id = candidate_session
                except (OSError, ValueError, TypeError):
                    shape_session_id = None
            while iteration <= attempt_limit:
                working_metrics = shadow_metrics or best_metrics
                guidance = claim_next_guidance(
                    self.state_dir / "guidance",
                    assignment.worker_id,
                    iteration,
                )
                source_hip_path = source / "csrc" / "w8a8_gemm_hip.hip"
                round_source_digest = file_digest(source_hip_path)
                source_text = source_hip_path.read_text(encoding="utf-8")
                history: list[Dict[str, Any]] = []
                if experiments_path.is_file():
                    for line in experiments_path.read_text(
                        encoding="utf-8"
                    ).splitlines():
                        try:
                            record = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        if isinstance(record, dict):
                            history.append(record)
                isa_policy = isa_round_policy(
                    iteration=iteration,
                    max_iterations=config.mock_iterations,
                    history=history,
                )
                pmc_decision = pmc_profile_decision(
                    iteration=iteration,
                    history=history,
                    source_uses_dumma=(
                        "du_mma_sync" in source_text
                        or "DUFragment" in source_text
                    ),
                    isa_policy=isa_policy,
                )
                profile_dir = (
                    root / "profiles" / shape_id / f"iteration{iteration}"
                )
                if pmc_decision["profile"]:
                    _status(
                        root, assignment, state="profiling_current_best",
                        iteration=iteration, shape_id=shape_id, probe=probe,
                    )
                    try:
                        pmc_evidence = runner.profile_pmc(
                            shape.params, profile_dir
                        )
                        pmc_evidence["device_cu_count"] = probe.get(
                            "multi_processor_count", 0
                        )
                        add_unprofiled_bandwidth(
                            pmc_evidence, working_metrics.get("median_us")
                        )
                    except Exception as exc:
                        pmc_evidence = {
                            "available": False,
                            "error": str(exc),
                            "shape": shape.params,
                            "source_hip_digest": round_source_digest,
                            "interpretation_guard": (
                                "PMC collection failed. Do not infer a "
                                "bottleneck from algorithmic_bandwidth_gb_s; "
                                "it is not measured HBM."
                            ),
                        }
                    try:
                        isa_evidence = runner.inspect_isa(
                            profile_dir / "current-best-isa",
                            kernel_names=[
                                str(item.get("kernel_name") or "")
                                for item in pmc_evidence.get(
                                    "profiled_kernels", []
                                )
                                if item.get("kernel_name")
                            ],
                            primary_kernel_name=str(
                                pmc_evidence.get("primary_kernel_name") or ""
                            ) or None,
                        )
                        launches = {
                            str(item.get("kernel_name") or ""): item
                            for item in pmc_evidence.get(
                                "profiled_kernels", []
                            )
                        }
                        for item in isa_evidence.get(
                            "profiled_kernels", []
                        ):
                            launch = launches.get(
                                str(item.get("kernel_name") or "")
                            )
                            if launch is not None:
                                item["profiled_launch_resources"] = {
                                    key: launch.get(key) for key in (
                                        "grid_blocks", "workgroup_size",
                                        "lds_bytes", "scratch_bytes",
                                        "arch_vgpr", "accum_vgpr", "sgpr",
                                        "wave_size",
                                    )
                                }
                        primary_launch = launches.get(
                            str(isa_evidence.get("kernel_name") or "")
                        )
                        if primary_launch is not None:
                            isa_evidence["profiled_launch_resources"] = {
                                key: primary_launch.get(key) for key in (
                                    "grid_blocks", "workgroup_size",
                                    "lds_bytes", "scratch_bytes", "arch_vgpr",
                                    "accum_vgpr", "sgpr", "wave_size",
                                )
                            }
                        pmc_evidence["isa"] = isa_evidence
                    except Exception as exc:
                        pmc_evidence["isa"] = {
                            "available": False,
                            "error": str(exc),
                            "interpretation_guard": (
                                "Current-best ISA extraction failed. Do not "
                                "invent instruction-level claims."
                            ),
                        }
                else:
                    cached = _cached_pmc_evidence(
                        root / "profiles" / shape_id,
                        round_source_digest,
                    )
                    pmc_evidence = cached or {
                        "available": False,
                        "shape": shape.params,
                        "source_hip_digest": round_source_digest,
                        "skipped": True,
                        "skip_reason": pmc_decision["reason"],
                        "interpretation_guard": (
                            "Fresh PMC was intentionally skipped. Use only "
                            "normal benchmark metrics until an accepted best "
                            "or late ISA decision triggers profiling."
                        ),
                    }
                write_json(profile_dir / "pmc.json", pmc_evidence)
                self.store.append_timeline(
                    (
                        "worker_pmc_profile"
                        if pmc_decision["profile"]
                        else "worker_pmc_skipped"
                    ),
                    {
                        "worker_id": assignment.worker_id,
                        "physical_gpu": assignment.gpu,
                        "shape_id": shape_id,
                        "iteration": iteration,
                        "available": pmc_evidence.get("available", False),
                        "path": str(profile_dir / "pmc.json"),
                        "error": pmc_evidence.get("error"),
                        "reason": pmc_decision["reason"],
                        "reused": bool(pmc_evidence.get("reused")),
                    },
                )
                _status(
                    root, assignment, state="agent_running",
                    iteration=iteration, shape_id=shape_id, probe=probe,
                    pmc_available=pmc_evidence.get("available", False),
                )
                proposal_path = source / "proposal.json"
                proposal_path.unlink(missing_ok=True)
                baseline_inline_asm = analyze_inline_asm_source(
                    source_text
                )
                contract_path = source / W8A8_API_FILENAME
                contract_digest = file_digest(contract_path)
                prompt_evidence = dict(pmc_evidence)
                if not isa_policy["skill_allowed"]:
                    prompt_evidence.pop("isa", None)
                prompt_metrics = dict(working_metrics)
                prompt_metrics["official_best_median_us"] = best_metrics.get(
                    "median_us"
                )
                prompt_metrics["official_best_p90_us"] = best_metrics.get(
                    "p90_us"
                )
                prompt_metrics["shadow_candidate_active"] = (
                    shadow_metrics is not None
                )
                prompt = self._worker_prompt(
                    assignment, shape_id, shape.params, prompt_metrics,
                    root, iteration, guidance, history, prompt_evidence,
                    verified_experience, comparison_target,
                    isa_policy=isa_policy,
                    continuation=shape_session_id is not None,
                )
                prompt_file = root / "logs" / (
                    f"{shape_id}-iteration-{iteration}.prompt.txt"
                )
                prompt_file.write_text(prompt, encoding="utf-8")
                agent_name = (
                    f"{assignment.worker_id}-{shape_id}-iter{iteration}"
                )
                spec = AgentSpec(
                    name=agent_name,
                    role="dcu_w8a8_gemm_worker",
                    prompt_file=prompt_file,
                    workdir=source,
                    log_dir=root / "logs",
                    timeout_s=_AGENT_TIMEOUT_S,
                    stuck_timeout_s=_AGENT_STUCK_TIMEOUT_S,
                    max_retries=0,
                    extra_args=list(
                        _ISA_AGENT_ARGS
                        if isa_policy["skill_allowed"]
                        else _SOURCE_ONLY_AGENT_ARGS
                    ),
                    env_overrides=runner.env,
                    resume_session_id=shape_session_id,
                )
                self.store.append_timeline(
                    "agent_launch",
                    {
                        "name": agent_name,
                        "worker_id": assignment.worker_id,
                        "physical_gpu": assignment.gpu,
                        "operator": "int8_w8a8_gemm",
                        "session_continuation": shape_session_id is not None,
                    },
                )
                agent_result = None
                proposal: Dict[str, Any] = {}
                failure_reason: str | None = None
                compiled = False
                agent_ready = False
                repair_records: list[Dict[str, Any]] = []
                try:
                    self.manager.launch(spec)
                    agent_result = self.manager.result(agent_name)
                    if agent_result is not None and agent_result.session_id:
                        shape_session_id = agent_result.session_id
                        write_json(session_path, {
                            "worker_id": assignment.worker_id,
                            "shape_id": shape_id,
                            "session_id": shape_session_id,
                            "last_iteration": iteration,
                            "updated_at": time.time(),
                        })
                    if agent_result is None or not agent_result.success:
                        raise RuntimeError(
                            f"{agent_name} failed: "
                            f"{agent_result.error if agent_result else 'no result'}"
                        )
                    if not proposal_path.is_file():
                        raise RuntimeError(
                            f"{agent_name} did not write proposal.json"
                        )
                    proposal = json.loads(
                        proposal_path.read_text(encoding="utf-8")
                    )
                    if not isinstance(proposal, dict):
                        raise RuntimeError(
                            f"{agent_name} proposal.json must be an object"
                        )
                    agent_ready = True
                except Exception as exc:
                    failure_reason = str(exc)

                def changed_paths() -> list[str]:
                    return [
                        line for line in _run(
                            ["git", "diff", "--name-only"], cwd=source
                        ).stdout.splitlines()
                        if line and line != "proposal.json"
                    ]

                def preflight() -> list[str]:
                    if (
                        contract_digest is not None
                        and file_digest(contract_path) != contract_digest
                    ):
                        raise RuntimeError(
                            f"agent modified immutable API contract "
                            f"{W8A8_API_FILENAME}"
                        )
                    paths = changed_paths()
                    if file_digest(source_hip_path) == round_source_digest:
                        raise RuntimeError(
                            "agent made no new kernel change relative to the "
                            "current official/shadow working source"
                        )
                    if not paths:
                        raise RuntimeError(
                            "agent made no tracked kernel change"
                        )
                    unexpected = sorted(
                        path for path in paths
                        if path != "csrc/w8a8_gemm_hip.hip"
                    )
                    if unexpected:
                        raise RuntimeError(
                            "agent changed control-plane-owned extension "
                            f"infrastructure: {unexpected}; only "
                            "csrc/w8a8_gemm_hip.hip may change"
                        )
                    candidate_inline_asm = analyze_inline_asm_source(
                        source_hip_path.read_text(encoding="utf-8")
                    )
                    old_asm = set(
                        baseline_inline_asm.get(
                            "raw_instruction_fingerprints"
                        ) or []
                    )
                    new_asm = set(
                        candidate_inline_asm.get(
                            "raw_instruction_fingerprints"
                        ) or []
                    ) - old_asm
                    if new_asm:
                        if not isa_policy["raw_inline_asm_allowed"]:
                            raise RuntimeError(
                                "raw inline asm is forbidden by this round's "
                                f"ISA policy: {isa_policy['reason']}"
                            )
                        isa_plan = proposal.get("isa_optimization")
                        if not isinstance(isa_plan, dict):
                            raise RuntimeError(
                                "new raw inline asm requires an "
                                "isa_optimization object in proposal.json"
                            )
                        if isa_plan.get("strategy") != "inline_asm":
                            raise RuntimeError(
                                "new raw inline asm requires "
                                "isa_optimization.strategy=inline_asm"
                            )
                        targets = isa_plan.get("target_instructions")
                        if not isinstance(targets, list) or not targets:
                            raise RuntimeError(
                                "new raw inline asm requires non-empty "
                                "isa_optimization.target_instructions"
                            )
                    if int(shape.params.get("M", 0)) == 16:
                        architecture = proposal.get("architecture")
                        required_architecture_fields = {
                            "family",
                            "grid_blocks",
                            "waves_per_block",
                            "tiles_per_block",
                            "split_k",
                            "estimated_active_cus",
                            "packed_layout",
                            "staging",
                            "vector_load_bytes",
                            "barriers_per_k_step",
                        }
                        if not isinstance(architecture, dict):
                            raise RuntimeError(
                                "M=16 proposal.json must contain an "
                                "architecture object"
                            )
                        missing_architecture = sorted(
                            required_architecture_fields - architecture.keys()
                        )
                        if missing_architecture:
                            raise RuntimeError(
                                "M=16 proposal architecture is missing "
                                f"fields: {missing_architecture}"
                            )
                    return paths

                metrics: Dict[str, Any] = {}
                session_id = (
                    getattr(agent_result, "session_id", None)
                    if agent_result is not None else None
                )
                for repair_index in range(_MAX_IN_ROUND_REPAIRS + 1):
                    if not agent_ready:
                        break
                    if failure_reason is None:
                        try:
                            preflight()
                            _status(
                                root,
                                assignment,
                                state="validating_candidate",
                                iteration=iteration,
                                shape_id=shape_id,
                                probe=probe,
                                repair=repair_index,
                            )
                            validation_metrics = runner.benchmark(
                                shape.params,
                                warmups=0,
                                samples=1,
                                replays_per_sample=1,
                            )
                            compiled = True
                            metrics = validation_metrics
                            if validation_metrics.get("passed"):
                                # The exact same source and deterministic
                                # inputs just passed the CPU-int64 reference.
                                # The stable timing pass must not repeat that
                                # expensive reference, especially for large M.
                                metrics = runner.benchmark(
                                    shape.params,
                                    check_correctness=False,
                                )
                                if (
                                    metrics.get("compile_cache_key")
                                    != validation_metrics.get(
                                        "compile_cache_key"
                                    )
                                ):
                                    raise RuntimeError(
                                        "timing source changed after exact "
                                        "correctness precheck"
                                    )
                                metrics["correctness_passed_in_precheck"] = True
                                if int(shape.params.get("M", 0)) == 16:
                                    paired_fallback = {
                                        **shape.params,
                                        "M": 2,
                                    }
                                    fallback_metrics = runner.benchmark(
                                        paired_fallback,
                                        warmups=0,
                                        samples=1,
                                        replays_per_sample=1,
                                    )
                                    metrics[
                                        "paired_m2_fallback_validation"
                                    ] = fallback_metrics
                                    if (
                                        not fallback_metrics.get("passed")
                                        or fallback_metrics.get(
                                            "graph_capture_passed"
                                        ) is not True
                                    ):
                                        raise RuntimeError(
                                            "paired M=2 fallback failed for "
                                            f"{shape_id}: "
                                            f"{json.dumps(fallback_metrics)}"
                                        )
                                failure_reason = None
                                break
                            failure_reason = (
                                "exact correctness failed: "
                                f"{json.dumps(validation_metrics)}"
                            )
                        except Exception as exc:
                            failure_reason = str(exc)

                    if repair_index >= _MAX_IN_ROUND_REPAIRS:
                        break
                    repair_number = repair_index + 1
                    _status(
                        root,
                        assignment,
                        state="repairing_candidate",
                        iteration=iteration,
                        shape_id=shape_id,
                        probe=probe,
                        repair=repair_number,
                        max_repairs=_MAX_IN_ROUND_REPAIRS,
                    )
                    proposal_path.unlink(missing_ok=True)
                    repair_prompt = self._repair_prompt(
                        assignment=assignment,
                        shape_id=shape_id,
                        shape=shape.params,
                        root=root,
                        iteration=iteration,
                        repair=repair_number,
                        error=failure_reason or "unknown validation failure",
                        metrics=metrics,
                        pmc_evidence=prompt_evidence,
                        isa_policy=isa_policy,
                    )
                    repair_prompt_file = root / "logs" / (
                        f"{shape_id}-iteration-{iteration}-"
                        f"repair-{repair_number}.prompt.txt"
                    )
                    repair_prompt_file.write_text(
                        repair_prompt, encoding="utf-8"
                    )
                    repair_name = (
                        f"{assignment.worker_id}-{shape_id}-iter{iteration}-"
                        f"repair{repair_number}"
                    )
                    self.store.append_timeline(
                        "worker_repair_launch",
                        {
                            "name": repair_name,
                            "worker_id": assignment.worker_id,
                            "physical_gpu": assignment.gpu,
                            "shape_id": shape_id,
                            "iteration": iteration,
                            "repair": repair_number,
                            "max_repairs": _MAX_IN_ROUND_REPAIRS,
                        },
                    )
                    repair_failure = failure_reason
                    try:
                        self.manager.launch(AgentSpec(
                            name=repair_name,
                            role="dcu_w8a8_gemm_repair",
                            prompt_file=repair_prompt_file,
                            workdir=source,
                            log_dir=root / "logs",
                            timeout_s=_AGENT_TIMEOUT_S,
                            stuck_timeout_s=_AGENT_STUCK_TIMEOUT_S,
                            max_retries=0,
                            extra_args=list(
                                _ISA_AGENT_ARGS
                                if isa_policy["skill_allowed"]
                                else _SOURCE_ONLY_AGENT_ARGS
                            ),
                            env_overrides=runner.env,
                            resume_session_id=session_id,
                        ))
                        repair_result = self.manager.result(repair_name)
                        if (
                            repair_result is None
                            or not repair_result.success
                        ):
                            raise RuntimeError(
                                f"{repair_name} failed: "
                                f"{repair_result.error if repair_result else 'no result'}"
                            )
                        if repair_result.session_id:
                            session_id = repair_result.session_id
                            shape_session_id = session_id
                            write_json(session_path, {
                                "worker_id": assignment.worker_id,
                                "shape_id": shape_id,
                                "session_id": shape_session_id,
                                "last_iteration": iteration,
                                "last_repair": repair_number,
                                "updated_at": time.time(),
                            })
                        if not proposal_path.is_file():
                            raise RuntimeError(
                                f"{repair_name} did not write proposal.json"
                            )
                        repair_proposal = json.loads(
                            proposal_path.read_text(encoding="utf-8")
                        )
                        if not isinstance(repair_proposal, dict):
                            raise RuntimeError(
                                f"{repair_name} proposal.json must be an object"
                            )
                        if repair_proposal.get("hypothesis"):
                            proposal["last_repair_hypothesis"] = (
                                repair_proposal["hypothesis"]
                            )
                        if isinstance(
                            repair_proposal.get("architecture"), dict
                        ):
                            proposal["architecture"] = (
                                repair_proposal["architecture"]
                            )
                        if isinstance(
                            repair_proposal.get("isa_optimization"), dict
                        ):
                            proposal["isa_optimization"] = (
                                repair_proposal["isa_optimization"]
                            )
                        failure_reason = None
                        repair_agent_ready = True
                    except Exception as exc:
                        failure_reason = str(exc)
                        repair_agent_ready = False
                    repair_records.append({
                        "repair": repair_number,
                        "input_failure": repair_failure,
                        "agent_failure": failure_reason,
                        "session_id": session_id,
                    })
                    if not repair_agent_ready:
                        break

                changed = changed_paths()

                iteration_dir = (
                    root / "iterations" / shape_id
                    / f"iteration{iteration}"
                )
                candidate_inline_asm = analyze_inline_asm_source(
                    source_hip_path.read_text(encoding="utf-8")
                )
                candidate_isa: Dict[str, Any] = {
                    "available": False,
                    "error": "candidate did not compile",
                }
                if compiled:
                    try:
                        candidate_isa = runner.inspect_isa(
                            iteration_dir / "isa"
                        )
                    except Exception as exc:
                        candidate_isa = {
                            "available": False,
                            "error": str(exc),
                        }
                inline_asm_gate = evaluate_inline_asm_gate(
                    before=baseline_inline_asm,
                    after=candidate_inline_asm,
                    proposal=proposal,
                    isa_evidence=candidate_isa,
                    raw_inline_asm_allowed=bool(
                        isa_policy.get("raw_inline_asm_allowed")
                    ),
                    verified_target_instructions=(
                        list(isa_policy.get("verified_target_instructions") or [])
                        if isa_policy.get("raw_inline_asm_allowed")
                        else None
                    ),
                )
                if not inline_asm_gate["passed"]:
                    gate_error = (
                        "inline asm acceptance gate failed: "
                        + "; ".join(inline_asm_gate["reasons"])
                    )
                    failure_reason = (
                        f"{failure_reason}; {gate_error}"
                        if failure_reason else gate_error
                    )

                _status(
                    root, assignment, state="recording_result",
                    iteration=iteration, shape_id=shape_id, probe=probe,
                )
                passed = (
                    bool(metrics.get("passed"))
                    and metrics.get("graph_capture_passed") is True
                    and inline_asm_gate["passed"]
                )
                acceptance = evaluate_candidate_acceptance(
                    passed=passed,
                    metrics=metrics,
                    best_metrics=best_metrics,
                    minimum_improvement_percent=(
                        ROUND_ACCEPTANCE_IMPROVEMENT_PERCENT
                    ),
                    shadow_metrics=shadow_metrics,
                )
                candidate_us = acceptance["candidate_us"]
                accepted = acceptance["accepted"]
                shadow_eligible = acceptance["shadow_eligible"]
                p90_guard_passed = acceptance["p90_guard_passed"]
                experiment = {
                    "schema_version": SCHEMA_VERSION,
                    "worker_id": assignment.worker_id,
                    "iteration": iteration,
                    "shape_id": shape_id,
                    "shape": shape.params,
                    "hypothesis": (
                        proposal.get("hypothesis")
                        or f"Round {iteration} failed before a valid proposal."
                    ),
                    "changes": changed,
                    "profile_evidence": (
                        proposal.get("profile_evidence") or {}
                    ),
                    "architecture": proposal.get("architecture") or {},
                    "isa_optimization": (
                        proposal.get("isa_optimization") or {}
                    ),
                    "isa_policy": isa_policy,
                    "candidate_isa": candidate_isa,
                    "inline_asm_source": candidate_inline_asm,
                    "inline_asm_gate": inline_asm_gate,
                    "build_success": compiled,
                    "correctness_passed": passed,
                    "metrics": {
                        key: metrics[key] for key in (
                            "median_us", "p90_us", "min_us", "max_us",
                            "logical_tops", "algorithmic_bytes",
                            "algorithmic_bandwidth_gb_s",
                            "latency_samples_us", "max_abs_error",
                            "mismatch_count", "first_mismatch",
                            "graph_capture_passed", "timing_mode",
                            "python_callable", "python_graph_api",
                            "graph_error",
                        ) if key in metrics
                    },
                    "pmc_evidence": pmc_evidence,
                    "in_round_repairs": repair_records,
                    "baseline_us": baseline[shape_id]["median_us"],
                    "baseline_kind": baseline[shape_id].get(
                        "baseline_kind", "measured_bootstrap"
                    ),
                    "speedup": (
                        round(
                            float(baseline[shape_id]["median_us"])
                            / candidate_us,
                            6,
                        )
                        if candidate_us != float("inf") else 0.0
                    ),
                    "accepted": accepted,
                    "shadow_eligible": shadow_eligible,
                    "shadow_base_active": shadow_metrics is not None,
                    "p90_guard_passed": p90_guard_passed,
                    "acceptance": acceptance,
                    "beats_baseline": (
                        passed
                        and candidate_us
                        < float(baseline[shape_id]["median_us"])
                    ),
                    "commit": None,
                    "failure_reason": failure_reason,
                    "manual_guidance": (
                        guidance["text"] if guidance else None
                    ),
                    "guidance_id": guidance["id"] if guidance else None,
                    "agent_session_id": (
                        getattr(agent_result, "session_id", None)
                    ),
                    "timestamp": time.time(),
                }
                candidate_files = archive_iteration_candidate(
                    source, iteration_dir, changed
                )
                experiment["artifact_dir"] = str(
                    iteration_dir.relative_to(root)
                )
                experiment["candidate_files"] = candidate_files
                candidate_destination = candidate_iteration_destination(
                    self.workspace_dir,
                    assignment,
                    shape_id,
                    iteration,
                )
                if candidate_destination is not None:
                    experiment["candidate_repo_dir"] = str(
                        candidate_destination
                    )
                proposal_path.unlink(missing_ok=True)
                if accepted:
                    experiment["shadow_promoted"] = shadow_metrics is not None
                    _run(["git", "add", "-u"], cwd=source)
                    _run([
                        "git", "commit", "-m",
                        f"{shape_id}: accept iteration {iteration}",
                    ], cwd=source)
                    best_commit = _run(
                        ["git", "rev-parse", "HEAD"], cwd=source
                    ).stdout.strip()
                    experiment["commit"] = best_commit
                    best_metrics = metrics
                    shadow_metrics = None
                    snapshot_accepted_kernel_artifact(
                        worker_root=root,
                        shape_id=shape_id,
                        shape=shape.params,
                        metrics=best_metrics,
                        commit=best_commit,
                        isa_evidence=candidate_isa,
                        isa_dir=iteration_dir / "isa",
                    )
                elif shadow_eligible:
                    shadow_path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(source_hip_path, shadow_path)
                    shadow_metrics = metrics
                    experiment["shadow_candidate"] = {
                        "source": str(shadow_path),
                        "median_us": metrics.get("median_us"),
                        "p90_us": metrics.get("p90_us"),
                        "official_best_median_us": best_metrics.get(
                            "median_us"
                        ),
                        "policy": (
                            "experimental base only; official best is unchanged "
                            "until cumulative improvement reaches the normal "
                            "acceptance threshold"
                        ),
                    }
                else:
                    _run(["git", "restore", "."], cwd=source)
                    if shadow_metrics is not None and shadow_path.is_file():
                        shutil.copy2(shadow_path, source_hip_path)
                write_json(
                    iteration_dir / "iteration.json", experiment
                )
                append_jsonl(experiments_path, experiment)
                if candidate_destination is not None:
                    publish_iteration_candidate(
                        iteration_dir, candidate_destination
                    )
                if failure_reason:
                    self.store.append_timeline(
                        "worker_iteration_failed",
                        {
                            "worker_id": assignment.worker_id,
                            "physical_gpu": assignment.gpu,
                            "shape_id": shape_id,
                            "iteration": iteration,
                            "error": failure_reason,
                            "next_iteration": (
                                iteration + 1
                                if iteration < attempt_limit
                                else None
                            ),
                            "policy": (
                                "record failed round, restore best commit, "
                                "and continue"
                            ),
                        },
                    )
                if (
                    is_infrastructure_failure(failure_reason)
                    and infrastructure_recoveries
                    < _MAX_INFRASTRUCTURE_RECOVERY_ROUNDS
                ):
                    infrastructure_recoveries += 1
                    attempt_limit += 1
                    self.store.append_timeline(
                        "worker_iteration_recovered",
                        {
                            "worker_id": assignment.worker_id,
                            "physical_gpu": assignment.gpu,
                            "shape_id": shape_id,
                            "failed_iteration": iteration,
                            "replacement_iteration": attempt_limit,
                            "recovery": infrastructure_recoveries,
                            "max_recoveries": (
                                _MAX_INFRASTRUCTURE_RECOVERY_ROUNDS
                            ),
                        },
                    )
                elif (
                    iteration >= attempt_limit
                    and not is_infrastructure_failure(failure_reason)
                ):
                    extension_reason = phase_extension_reason(
                        max_iterations=config.mock_iterations,
                        history=[*history, experiment],
                    )
                    if (
                        extension_reason is not None
                        and phase_extensions < _MAX_PHASE_EXTENSION_ROUNDS
                    ):
                        phase_extensions += 1
                        attempt_limit += 1
                        self.store.append_timeline(
                            "worker_phase_extended",
                            {
                                "worker_id": assignment.worker_id,
                                "physical_gpu": assignment.gpu,
                                "shape_id": shape_id,
                                "completed_iteration": iteration,
                                "replacement_iteration": attempt_limit,
                                "reason": extension_reason,
                                "extension": phase_extensions,
                                "max_extensions": _MAX_PHASE_EXTENSION_ROUNDS,
                            },
                        )
                with self._progress_lock:
                    if iteration > self._reported_iteration:
                        self.store.update_run(current_iteration=iteration)
                        self._reported_iteration = iteration
                iteration += 1
            discarded_shadow = None
            if shadow_metrics is not None:
                discarded_shadow = {
                    "median_us": shadow_metrics.get("median_us"),
                    "p90_us": shadow_metrics.get("p90_us"),
                    "source": str(shadow_path),
                    "reason": (
                        "cumulative gain did not reach the official acceptance "
                        "threshold before the phase budget ended"
                    ),
                }
                _run(["git", "restore", "."], cwd=source)
            result["shapes"][shape_id] = {
                "shape_id": shape_id,
                "candidate": best_commit,
                "metrics": best_metrics,
                "discarded_shadow": discarded_shadow,
                "artifact": json.loads(
                    artifact_manifest_path.read_text(encoding="utf-8")
                ),
            }
        _status(
            root, assignment, state="optimization_complete",
            iteration=iteration - 1, shape_id=None, probe=probe,
        )
        write_json(root / "result.json", result)
        return result

    @staticmethod
    def _repair_prompt(
        *,
        assignment: WorkerAssignment,
        shape_id: str,
        shape: Dict[str, Any],
        root: Path,
        iteration: int,
        repair: int,
        error: str,
        metrics: Dict[str, Any],
        pmc_evidence: Dict[str, Any],
        isa_policy: Dict[str, Any] | None = None,
    ) -> str:
        isa_policy = isa_policy or {
            "skill_allowed": False,
            "raw_inline_asm_allowed": False,
            "reason": "HIP-only repair round",
        }
        isa_repair = (
            "If the candidate introduced raw inline asm, use the installed "
            "`hygon-gfx928-memory-isa` or `hygon-gfx928-compute-isa` Skill "
            "as appropriate, and repair the smallest constraint, clobber, "
            "wait, EXEC, lane-layout, or operand defect. It is valid to "
            "replace unsafe asm with HIP/DUMMA code."
            if isa_policy.get("skill_allowed")
            else "Do not use ISA Skills or add raw inline asm in this repair."
        )
        isa_repair_schema = (
            '''  "isa_optimization": {
    "skill": "hygon-gfx928-memory-isa or hygon-gfx928-compute-isa",
    "evidence_level": "binary or experiment",
    "bottleneck": "memory, compute, or mixed",
    "strategy": "hip_codegen, intrinsic, or inline_asm",
    "compiler_limitation_confirmed": false,
    "current_isa_findings": [],
    "target_instructions": [],
    "expected_isa_change": "specific observable code-object change",
    "constraint_and_clobber_risks": []
  },
'''
            if isa_policy.get("skill_allowed") else ""
        )
        return f"""Continue the same gfx928 W8A8 optimization round for
{assignment.worker_id}, shape {shape_id}: {json.dumps(shape)}.

This is in-round repair {repair}/{_MAX_IN_ROUND_REPAIRS} for iteration
{iteration}. The trusted control plane compiled or checked your current HIP
code and returned:

```text
{error[-6000:]}
```

Structured validation data:

```json
{json.dumps(metrics, indent=2, sort_keys=True)}
```

The pre-edit PMC evidence remains:

```json
{json.dumps(pmc_evidence, indent=2, sort_keys=True)}
```

Repair the current `csrc/w8a8_gemm_hip.hip` in place. Preserve the proposed
mapping and performance idea; make the smallest change that fixes the reported
compile or exact-correctness defect. Use `mismatch_count` and
`first_mismatch` to distinguish tail/index/scale/signed-unpack/race errors.
{isa_repair}
Do not redesign the kernel, change infrastructure, add files, run commands, or
run the harness. The trusted control plane will revalidate after you return.
If `graph_capture_passed` is false, use `graph_error` to remove allocation,
synchronization, host callbacks, or default-stream launches from the timed
operator while preserving the current HIP computation.

Write strict JSON to `{root / 'source' / 'proposal.json'}` with:

```json
{{
  "iteration": {iteration},
  "repair": {repair},
  "hypothesis": "specific root cause and minimal repair",
  "architecture": {{
    "family": "preserve the current family",
    "grid_blocks": 0,
    "waves_per_block": 1,
    "tiles_per_block": 1,
    "split_k": 1,
    "estimated_active_cus": 0,
    "packed_layout": "identity or exact layout name",
    "staging": "direct, A_only, B_only, or A_and_B",
    "vector_load_bytes": 0,
    "barriers_per_k_step": 0
  }},
{isa_repair_schema}
  "files_changed": ["csrc/w8a8_gemm_hip.hip"]
}}
```
"""

    @staticmethod
    def _worker_prompt(
        assignment: WorkerAssignment,
        shape_id: str,
        shape: Dict[str, Any],
        best: Dict[str, Any],
        root: Path,
        iteration: int,
        guidance: Dict[str, Any] | None,
        history: list[Dict[str, Any]] | None = None,
        pmc_evidence: Dict[str, Any] | None = None,
        verified_experience: list[Dict[str, Any]] | None = None,
        comparison_baseline: Dict[str, Any] | None = None,
        isa_policy: Dict[str, Any] | None = None,
        continuation: bool = False,
    ) -> str:
        history = history or []
        verified_experience = verified_experience or []
        pmc_evidence = pmc_evidence or {
            "available": False,
            "error": "PMC evidence was not supplied",
        }
        comparison_baseline = comparison_baseline or best
        isa_policy = isa_policy or isa_round_policy(
            iteration=iteration,
            max_iterations=10,
            history=history,
        )
        if not isa_policy["skill_allowed"]:
            pmc_evidence = dict(pmc_evidence)
            pmc_evidence.pop("isa", None)
        guidance_text = (
            guidance["text"] if guidance else "(none; decide independently)"
        )
        history_summary = [
            {
                "iteration": record.get("iteration"),
                "accepted": record.get("accepted"),
                "build_success": record.get("build_success"),
                "correctness_passed": record.get("correctness_passed"),
                "metrics": _compact_metrics_for_prompt(
                    record.get("metrics") or {}
                ),
                "baseline_us": record.get("baseline_us"),
                "speedup": record.get("speedup"),
                "hypothesis": record.get("hypothesis"),
                "isa_optimization": (
                    record.get("isa_optimization") or {}
                ),
                "candidate_isa": {
                    "available": (record.get("candidate_isa") or {}).get(
                        "available"
                    ),
                    "instruction_counts": (
                        record.get("candidate_isa") or {}
                    ).get("instruction_counts"),
                    "resources": (record.get("candidate_isa") or {}).get(
                        "resources"
                    ),
                },
                "inline_asm_gate": record.get("inline_asm_gate") or {},
                "failure_reason": str(
                    record.get("failure_reason") or ""
                )[-1200:],
                "artifact_dir": record.get("artifact_dir"),
            }
            for record in history[-5:]
        ]
        if not isa_policy["skill_allowed"]:
            for record in history_summary:
                record.pop("isa_optimization", None)
                record.pop("candidate_isa", None)
                record.pop("inline_asm_gate", None)
        isa_section = ""
        isa_discipline = (
            "6. ISA Skills and raw inline asm are disabled for this round. "
            "Use HIP, DUMMA APIs, and ordinary compiler intrinsics only."
        )
        skill_boundary = (
            "The Skill tool is disabled for this HIP-only round."
        )
        isa_schema = ""
        change_dimensions = (
            "memory_mapping, tile, block, lds, registers, dumma, or split_k"
        )
        if isa_policy["skill_allowed"]:
            raw_rule = (
                "One minimal raw inline-asm block is permitted by the "
                "control-plane gate."
                if isa_policy["raw_inline_asm_allowed"]
                else "Raw inline asm remains forbidden in this round."
            )
            isa_section = f"""
## Late-round ISA evidence and selected Skills

The nested `isa` object in PMC evidence is generated by the trusted control
plane from the exact current-best gfx928 object. It is audit evidence, not
proof of a bottleneck. The read-only Skill tool is allowed only for one of:
- `hygon-gfx928-memory-isa` for VMEM/LDS/waitcnt/barrier evidence;
- `hygon-gfx928-compute-isa` for VALU/DPP/MMAC evidence.

Select only the skill matching one measured bottleneck. Do not use either
skill to answer DUMMA C++ API questions. {raw_rule}
"""
            isa_discipline = (
                "6. Inspect only the trusted ISA excerpt relevant to one "
                "bottleneck. Prefer a HIP/DUMMA/intrinsic code-shaping change. "
                + raw_rule + " Raw global/buffer/flat loads and raw MMAC remain "
                "forbidden. Record whether a compiler limitation is actually "
                "confirmed; do not infer lane mappings or clobbers."
            )
            skill_boundary = (
                "The read-only Skill tool may be used only for the selected "
                "memory or compute ISA skill named above."
            )
            isa_schema = '''  "isa_optimization": {
    "skill": "hygon-gfx928-memory-isa or hygon-gfx928-compute-isa",
    "evidence_level": "binary or experiment",
    "bottleneck": "memory, compute, or mixed",
    "strategy": "hip_codegen, intrinsic, decline, or inline_asm",
    "compiler_limitation_confirmed": false,
    "current_isa_findings": ["facts visible in trusted ISA only"],
    "target_instructions": [],
    "expected_isa_change": "observable candidate disassembly change",
    "constraint_and_clobber_risks": [],
    "decline_reason": "why instruction-level work is not justified"
  },
'''
            change_dimensions += ", isa_memory, isa_compute, or inline_asm"
        round_strategy = w8a8_round_strategy(
            shape, iteration, history, pmc_evidence,
            max_iterations=int(isa_policy.get("max_iterations") or 10),
            isa_policy=isa_policy,
        )
        prompt_best = _compact_metrics_for_prompt(best)
        prompt_baseline = _compact_metrics_for_prompt(comparison_baseline)
        prompt_pmc = _compact_pmc_for_prompt(pmc_evidence)
        if continuation:
            return f"""Continue the existing shape-specialized gfx928 W8A8
optimization session for {assignment.worker_id}, shape {shape_id}:
{json.dumps(shape, sort_keys=True)}. This is iteration {iteration}.

The immutable API, graph/current-stream contract, exact int32 accumulation,
wavefront=64 rules, source ownership, optional reference freedom, acceptance
threshold, and proposal schema from the first turn remain unchanged. Do not
reread unchanged scaffold files. Inspect only the current HIP diff/sections
needed for this decision and the fact-ledger paths below.

Current official/shadow metrics:
```json
{json.dumps(prompt_best, indent=2, sort_keys=True)}
```

Fixed Triton comparison baseline:
```json
{json.dumps(prompt_baseline, indent=2, sort_keys=True)}
```

Mandatory decision for this round:
{round_strategy}

ISA policy:
```json
{json.dumps(isa_policy, indent=2, sort_keys=True)}
```

Current-best PMC/ISA summary (fresh or exact-source cached):
```json
{json.dumps(prompt_pmc, indent=2, sort_keys=True)}
```

Recent trusted experiments:
```json
{json.dumps(history_summary[-3:], indent=2, sort_keys=True)}
```

Full evidence remains available without being repeated in this prompt:
- `{root / 'runs' / shape_id / 'experiments.jsonl'}`
- `{root / 'profiles' / shape_id}`
- `{root / 'iterations' / shape_id}`
- `{root / 'source' / 'csrc' / 'w8a8_gemm_hip.hip'}`

Human guidance: {guidance_text}

Make one falsifiable, focused HIP change. Do not run the harness, profiler,
Docker, SSH, package tools, or broad searches. Preserve exact shape guards and
the generic fallback. Raw asm and ISA Skills follow only the policy above.
Write strict JSON to `{root / 'source' / 'proposal.json'}` with the unchanged
first-turn schema, including iteration={iteration}, hypothesis,
profile_evidence, architecture, optional isa_optimization when required, and
files_changed=["csrc/w8a8_gemm_hip.hip"].
"""
        return f"""You are {assignment.worker_id}, a shape-specialized native
HIP optimization worker for gfx928 INT8 W8A8 GEMM.
You own physical GPU {assignment.gpu} and the branch in {root / 'source'}.
Optimize only shape {shape_id}: {json.dumps(shape)}.

## Immutable timed contract

The immutable operator contract is defined by `int8_w8a8_gemm_api.py`:
- A: contiguous row-major int8 [M,K]
- B: contiguous row-major int8 [K,N]
- x_scale: float32 [M,1]
- weight_scale: float32 [N,1]
- output: bfloat16 [M,N]
- logical dot: int32 accumulation, then float scaling and bf16 store
- timed API: w8a8_gemm_out(..., out, workspace), returning the same out storage
- backend op: torch.ops.zth_w8a8.gemm_out
- no allocation, compilation, autotuning, packing, host/device sync, or
  default-stream launch inside the timed API
- use PyTorch's current HIP stream and caller-owned out/workspace only
- the trusted binding passes `workspace.data_ptr()` and workspace byte count
  to `launch_w8a8_gemm`; split-K partials and their combine kernel must use
  only this storage and their total Graph replay time is the candidate time
- the trusted binding calls `launch_pack_w8a8_weight` before capture;
  packing may be optimized in HIP but must never occur in `gemm_out`
- `csrc/w8a8_gemm_hip.hip` must preserve the binding-owned launch ABI:
  `launch_w8a8_gemm(..., void* workspace, int64_t workspace_bytes,
  int m, int n, int k, hipStream_t stream)` and
  `launch_pack_w8a8_weight(raw_weight, weight_scale, packed_weight,
  packed_weight_scale, int k, int n, hipStream_t stream)`
- every candidate is captured on a non-default stream and timed exclusively
  through `torch.cuda.CUDAGraph.replay()`; Graph capture failure is a
  correctness failure, not a performance result

## Optional implementation reference

`references/w8a8_gemm_variants.hip`, when present, is a read-only collection
of previously measured implementation variants. It is not compiled by the
default build, not a seed, and not a strategy whitelist. You may adapt,
combine, or ignore it; continue exploring any legal HIP/DUMMA architecture
that improves this exact shape. Never edit the reference file, and never
claim its measurements for the current source without revalidation.

Current best HIP metrics: {json.dumps(prompt_best)}.
If `shadow_candidate_active` is true, the checked-out HIP source is a
provisional sub-1% improvement. Build on it, but treat
`official_best_median_us`/`official_best_p90_us` as the acceptance baseline.
The control plane will promote the accumulated source only after the normal
>=1% median threshold and P90 guard pass; otherwise it restores the official
best at the end.
Fixed comparison baseline: {json.dumps(prompt_baseline)}.
The baseline is the user-supplied Triton decode Graph replay latency. It is
not the bootstrap HIP source and it is not PMC evidence. Improve the current
HIP source iteratively; report speedup against this fixed Graph baseline.
Human guidance for this round: {guidance_text}
Accepted/rejected history, when present:
`{root / 'runs' / shape_id / 'experiments.jsonl'}`.

## Mandatory decision for this round

{round_strategy}

Control-plane ISA policy:

```json
{json.dumps(isa_policy, indent=2, sort_keys=True)}
```

## Trusted prior-round evidence

```json
{json.dumps(history_summary, indent=2, sort_keys=True)}
```

## Trusted PMC evidence for the current best source

The control plane supplies fresh counters only for a usable DUMMA bootstrap,
a newly accepted best, or a late plateau/ISA decision. Otherwise it reuses
evidence only when the exact source digest matches, or marks PMC skipped.
Use available counters and resource fields below to choose your change.
Profiled latency is perturbed and must not be used as the acceptance timing.

```json
{json.dumps(prompt_pmc, indent=2, sort_keys=True)}
```

{isa_section}

## Verified exact-shape experience from earlier tasks

This ledger is generated from trusted `iteration.json` records, never from
free-form Skill prose. Treat measured fields and classification as facts.
Treat `proposed_change_not_verified_fact` only as a hypothesis.

```json
{json.dumps(verified_experience, indent=2, sort_keys=True)}
```

## Shape-specific starting point

{w8a8_strategy_guidance({shape_id: shape})}

## One-round optimization discipline

1. Read the current source and prior experiment history. You may read archived
   snapshots below `{root / 'iterations' / shape_id}` as read-only evidence.
   Follow the mandatory decision above and do not repeat an already rejected change.
2. State one falsifiable bottleneck hypothesis from the current metrics and
   launch geometry. Architecture rounds may implement one complete architecture
   family (including its required main/combine or pack/GEMM pair); polish rounds
   change one conceptual factor only. Keep all HIP implementation and required
   launch dispatch inside
   `csrc/w8a8_gemm_hip.hip`.
3. Preserve gfx928 invariants:
   - wavefront=64; blockDim is 64/128/256, never warp-32 logic;
   - DUMMA INT8 support is m16n16k32 with int32 accumulation;
   - installed DTK uses `<du_mma.h>` and namespace `du::dumma`;
   - preserve the current compiling bf16 representation. When a typed bf16
     pointer is needed, include `<hip/hip_bfloat16.h>` and use
     `hip_bfloat16`; do not switch between bf16 type families speculatively;
   - no NVIDIA WMMA, `mma.sync`, PTX, FP8, or INT4;
   - all `__syncthreads()` calls are reached by every block thread;
   - coalesced lanes follow contiguous N addresses;
   - single-buffer LDS must fit 64 KiB; use double buffering only when the
     two buffers fit about 48 KiB, leaving occupancy headroom.
4. Do not choose split-K merely because K is large. Require too few N/M tiles
   to occupy the measured device CUs. Scan legal CU-aligned candidates,
   including non-power-of-two splits, within the caller workspace capacity;
   the suggested set is not a whitelist. Count the combine kernel in total
   latency and accept only measured median/P90 improvements.
5. Do not force occupancy with `launch_bounds` unless the source already has
   resource evidence. Extra VGPR spill, barriers, or repeated HBM reads can
   erase any occupancy gain.
   Treat `algorithmic_bandwidth_gb_s` from the normal benchmark as an
   effective minimum-bytes rate, not measured HBM traffic. Use
   `memory_traffic.counter_derived_operator_hbm_bandwidth_gb_s` for HBM traffic
   claims; it sums every kernel in the operator replay before combining the
   request counters with the separate unprofiled whole-operator median.
   Never use `profiled_duration_us` for performance acceptance.
{isa_discipline}
7. Preserve exact correctness, current-stream behavior, graph safety, and all
   assigned-shape dispatch paths. Keep every optimized launch behind an exact
   `(m,n,k)` guard and preserve the scalar generic fallback for unmatched
   shapes, especially the paired M=2 shape with the same `(N,K)`. The trusted
   control plane archives and links the exact accepted object, so do not
   replace the fallback with an unconditional shape-specialized launch.
   Guard packed-weight specializations by exact `(k,n)` and retain identity
   packing for unmatched pairs.
   Never modify or rename
   `int8_w8a8_gemm_api.py`. The control plane also owns
   `w8a8_backend.py`, `setup.py`, `profile_pmc.sh`, and
   `csrc/bindings.cpp`; do not edit them.

## Execution boundary

Do not run the harness, benchmark, profiler, Docker, SSH, pip, apt, conda,
package installation, network access, or filesystem-wide searches
such as `find /`, or environment probes. Use the authoritative DTK facts in
this prompt. {skill_boundary} The trusted control plane will compile, validate exact output,
measure median/P90, and accept or
restore your diff after you return. Do not modify tests, caches, build output,
or files outside this worktree. Do not add files.

Inspect `README.md`, `w8a8_backend.py`, `csrc/bindings.cpp`, and
`csrc/w8a8_gemm_hip.hip`, make the focused tracked-source change, inspect the
diff, and then write `{root / 'source' / 'proposal.json'}` as strict JSON:

```json
{{
  "iteration": {iteration},
  "hypothesis": "one falsifiable bottleneck hypothesis and the focused change",
    "profile_evidence": {{
    "observed_best": {json.dumps(prompt_best)},
    "path": "scalar_lds or dumma_m16n16k32",
    "change_dimension": "{change_dimensions}",
    "block_threads": 64,
    "tile_m": 0,
    "tile_n": 0,
    "tile_k": 0,
    "lds_bytes": 0,
    "double_buffer": false,
    "split_k": 1,
    "expected_effect": "which measured metric should improve and why",
    "risk": "correctness, occupancy, bank conflict, or combine-overhead risk",
      "validation_owner": "trusted_control_plane"
  }},
  "architecture": {{
    "family": "direct, split_k, multi_n_tile, persistent, packed_weight, or staged",
    "grid_blocks": 0,
    "waves_per_block": 1,
    "tiles_per_block": 1,
    "split_k": 1,
    "estimated_active_cus": 0,
    "packed_layout": "identity or exact layout name",
    "staging": "direct, A_only, B_only, or A_and_B",
    "vector_load_bytes": 0,
    "barriers_per_k_step": 0
  }},
{isa_schema}
  "files_changed": ["actual tracked source paths changed this round"]
}}
```
"""

    def _serial_validate(
        self, config: OptimizerConfig, workers: Dict[str, Any]
    ) -> Dict[str, Any]:
        results: Dict[str, Any] = {}
        for assignment in config.assignments:
            if assignment.worker_id not in workers:
                continue
            runner = W8A8Runner(
                self.workspace_dir / "workers" / assignment.worker_id,
                assignment.gpu,
            )
            for shape_id in assignment.shape_ids:
                metrics = runner.benchmark(config.shapes[shape_id].params)
                if not metrics.get("passed"):
                    raise RuntimeError(
                        f"serial validation failed: {shape_id}"
                    )
                winner = workers[assignment.worker_id]["shapes"][shape_id]
                results[shape_id] = {
                    "passed": True,
                    "worker_id": assignment.worker_id,
                    "physical_gpu": assignment.gpu,
                    "candidate": winner["candidate"],
                    "metrics": metrics,
                    "serial": True,
                }
        write_json(
            self.workspace_dir / "final_validation" / "results.json",
            {
                "schema_version": SCHEMA_VERSION,
                "shapes": results,
                "real_gpu_used": True,
            },
        )
        return results
