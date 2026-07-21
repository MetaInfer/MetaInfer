"""Loading, validation and freezing of the system evaluator bundle."""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

import yaml


class SpecError(ValueError):
    pass


_PHASES = ("correctness", "benchmark")


@dataclass(frozen=True)
class CommandSpec:
    argv: List[str]
    timeout_s: int


@dataclass(frozen=True)
class AcceptanceSpec:
    min_weighted_speedup: float = 1.0
    noise_threshold: float = 0.01
    max_critical_regression: float = 0.03
    require_all_cases: bool = True


@dataclass(frozen=True)
class BenchmarkCaseSpec:
    id: str
    weight: float = 1.0
    critical: bool = False
    shape: Optional[Dict[str, int]] = None
    flops: Optional[float] = None
    bytes: Optional[float] = None


@dataclass(frozen=True)
class KernelTaskSpec:
    name: str
    public_contract: Dict[str, Any]
    commands: Mapping[str, CommandSpec]
    correctness_case_ids: List[str]
    benchmark_cases: List[BenchmarkCaseSpec]
    benchmark_protocol: Dict[str, Any]
    private_case_ids: List[str] = field(default_factory=list)
    acceptance: AcceptanceSpec = field(default_factory=AcceptanceSpec)
    schema_version: int = 2

    @classmethod
    def load(cls, path: Path) -> "KernelTaskSpec":
        try:
            raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        except (OSError, yaml.YAMLError) as exc:
            raise SpecError(f"cannot load evaluator spec {path}: {exc}") from exc
        if not isinstance(raw, dict):
            raise SpecError("task.yaml must contain a mapping")
        if raw.get("schema_version") != 2:
            raise SpecError("evaluator task.yaml must use schema_version=2")
        name = str(raw.get("name") or "").strip()
        if not name:
            raise SpecError("task.yaml requires a non-empty name")

        public_contract = _public_contract(raw.get("public_contract"))

        commands_raw = raw.get("commands")
        if not isinstance(commands_raw, dict):
            raise SpecError("task.yaml requires commands mapping")
        commands: Dict[str, CommandSpec] = {}
        for phase in _PHASES:
            item = commands_raw.get(phase)
            if not isinstance(item, dict):
                raise SpecError(f"commands.{phase} must be a mapping")
            argv = item.get("argv")
            if not isinstance(argv, list) or not argv or not all(isinstance(v, str) and v for v in argv):
                raise SpecError(f"commands.{phase}.argv must be a non-empty string list")
            timeout_s = int(item.get("timeout_s", 600))
            if timeout_s < 1 or timeout_s > 86_400:
                raise SpecError(f"commands.{phase}.timeout_s must be in [1, 86400]")
            commands[phase] = CommandSpec(list(argv), timeout_s)

        cases = raw.get("cases") or {}
        if not isinstance(cases, dict):
            raise SpecError("cases must be a mapping")
        benchmark = _benchmark_cases(cases.get("benchmark"))
        include_benchmark = bool(cases.get("correctness_include_benchmark", False))
        correctness = _unique_ids(
            cases.get("correctness", []),
            "cases.correctness",
            allow_empty=include_benchmark,
        )
        if include_benchmark:
            correctness = [case.id for case in benchmark] + correctness
            if len(set(correctness)) != len(correctness):
                raise SpecError("cases.correctness duplicates a benchmark case id")
        protocol = raw.get("benchmark_protocol")
        if not isinstance(protocol, dict):
            raise SpecError("benchmark_protocol must be a mapping")
        try:
            warmup = int(protocol["warmup"])
            samples = int(protocol["samples"])
            timer = str(protocol["timer"]).strip()
        except (KeyError, TypeError, ValueError) as exc:
            raise SpecError("benchmark_protocol requires warmup, samples, and timer") from exc
        if warmup < 1 or samples < 3 or not timer:
            raise SpecError("benchmark protocol requires warmup>=1, samples>=3, and timer")
        protocol = {**protocol, "warmup": warmup, "samples": samples, "timer": timer}
        private = _unique_ids(cases.get("private", []), "cases.private", allow_empty=True)
        unknown_private = sorted(set(private) - set(correctness))
        if unknown_private:
            raise SpecError(f"private cases must also be correctness cases: {unknown_private}")

        acc_raw = raw.get("acceptance") or {}
        if not isinstance(acc_raw, dict):
            raise SpecError("acceptance must be a mapping")
        acceptance = AcceptanceSpec(
            min_weighted_speedup=float(acc_raw.get("min_weighted_speedup", 1.0)),
            noise_threshold=float(acc_raw.get("noise_threshold", 0.01)),
            max_critical_regression=float(acc_raw.get("max_critical_regression", 0.03)),
            require_all_cases=bool(acc_raw.get("require_all_cases", True)),
        )
        if not math.isfinite(acceptance.min_weighted_speedup) or acceptance.min_weighted_speedup <= 0:
            raise SpecError("min_weighted_speedup must be positive")
        if not math.isfinite(acceptance.noise_threshold) or not 0 <= acceptance.noise_threshold < 1:
            raise SpecError("noise_threshold must be in [0, 1)")
        if not math.isfinite(acceptance.max_critical_regression) or not 0 <= acceptance.max_critical_regression < 1:
            raise SpecError("max_critical_regression must be in [0, 1)")
        return cls(
            name=name,
            public_contract=public_contract,
            commands=commands,
            correctness_case_ids=correctness,
            benchmark_cases=benchmark,
            benchmark_protocol=protocol,
            private_case_ids=private,
            acceptance=acceptance,
            schema_version=2,
        )

    @property
    def benchmark_case_ids(self) -> List[str]:
        return [case.id for case in self.benchmark_cases]

    def agent_contract(self) -> Dict[str, Any]:
        """Public evaluator contract supplied to candidate-generating agents."""
        private = set(self.private_case_ids)
        shapes = [
            {
                "id": case.id,
                "shape": case.shape,
                "weight": case.weight,
                "critical": case.critical,
            }
            for case in self.benchmark_cases
            if case.id not in private
        ]
        return {
            **self.public_contract,
            "benchmark_shapes": shapes,
            "benchmark_protocol": dict(self.benchmark_protocol),
        }


def _public_contract(value: Any) -> Dict[str, Any]:
    if not isinstance(value, dict):
        raise SpecError("task.yaml requires a public_contract mapping")
    contract = dict(value)
    for key in ("dtype", "layout", "abi"):
        item = contract.get(key)
        if not isinstance(item, dict) or not item:
            raise SpecError(f"public_contract.{key} must be a non-empty mapping")
    entrypoint = str(contract["abi"].get("entrypoint") or "").strip()
    if not entrypoint:
        raise SpecError("public_contract.abi.entrypoint is required")
    contract["abi"] = {**contract["abi"], "entrypoint": entrypoint}
    return contract


def _unique_ids(value: Any, label: str, *, allow_empty: bool = False) -> List[str]:
    if not isinstance(value, list) or (not value and not allow_empty):
        raise SpecError(f"{label} must be {'a' if allow_empty else 'a non-empty'} list")
    ids: List[str] = []
    for item in value:
        case_id = str(item.get("id") if isinstance(item, dict) else item).strip()
        if not case_id:
            raise SpecError(f"{label} contains an empty id")
        ids.append(case_id)
    if len(set(ids)) != len(ids):
        raise SpecError(f"{label} contains duplicate ids")
    return ids


def _benchmark_cases(value: Any) -> List[BenchmarkCaseSpec]:
    if isinstance(value, dict):
        return _benchmark_matrix(value)
    if not isinstance(value, list) or not value:
        raise SpecError("cases.benchmark must be a non-empty list or matrix mapping")
    cases: List[BenchmarkCaseSpec] = []
    for item in value:
        if isinstance(item, str):
            case = BenchmarkCaseSpec(item)
        elif isinstance(item, dict):
            case_id = str(item.get("id") or "").strip()
            try:
                weight = float(item.get("weight", 1.0))
            except (TypeError, ValueError) as exc:
                raise SpecError(f"benchmark case {case_id!r} has invalid weight") from exc
            if not case_id or not math.isfinite(weight) or weight <= 0:
                raise SpecError("benchmark case id and positive weight are required")
            shape = _benchmark_shape(item, case_id)
            if shape is None:
                raise SpecError(
                    f"benchmark case {case_id!r} requires shape metadata"
                )
            flops = _optional_positive_number(item.get("flops"), f"benchmark case {case_id!r} flops")
            transferred = _optional_positive_number(
                item.get("bytes"), f"benchmark case {case_id!r} bytes"
            )
            if flops is None and shape is not None:
                flops = float(
                    2 * shape["m"] * shape["n"] * shape["k"] * shape["batch"]
                )
            case = BenchmarkCaseSpec(
                case_id,
                weight,
                bool(item.get("critical", False)),
                shape,
                flops,
                transferred,
            )
        else:
            raise SpecError("benchmark cases must be strings or mappings")
        if case.shape is None:
            raise SpecError(
                f"benchmark case {case.id!r} requires shape metadata"
            )
        cases.append(case)
    if len({case.id for case in cases}) != len(cases):
        raise SpecError("cases.benchmark contains duplicate ids")
    return cases


def _benchmark_matrix(value: Dict[str, Any]) -> List[BenchmarkCaseSpec]:
    """Expand a compact M-by-workload matrix into ordinary scored cases."""
    matrix = value.get("matrix")
    if not isinstance(matrix, dict):
        raise SpecError("cases.benchmark mapping requires matrix")
    try:
        m_values = [int(item) for item in matrix["m_values"]]
        large_m = int(matrix.get("large_m", 4096))
        small_total = float(matrix.get("small_m_total_weight", 0.5))
        large_weight = float(matrix.get("large_m_weight", 0.5))
    except (KeyError, TypeError, ValueError) as exc:
        raise SpecError("benchmark matrix has invalid M values or weights") from exc
    if not m_values or len(set(m_values)) != len(m_values) or any(m <= 0 for m in m_values):
        raise SpecError("benchmark matrix m_values must be unique positive integers")
    small_values = [m for m in m_values if m != large_m]
    if large_m not in m_values or not small_values:
        raise SpecError("benchmark matrix must contain large_m and at least one small M")
    if not math.isfinite(small_total) or small_total <= 0:
        raise SpecError("benchmark matrix small_m_total_weight must be positive")
    if not math.isfinite(large_weight) or large_weight <= 0:
        raise SpecError("benchmark matrix large_m_weight must be positive")
    critical_m = {int(item) for item in matrix.get("critical_m", [1, large_m])}
    workloads = matrix.get("workloads")
    if not isinstance(workloads, list) or not workloads:
        raise SpecError("benchmark matrix workloads must be a non-empty list")

    cases: List[BenchmarkCaseSpec] = []
    for workload in workloads:
        if not isinstance(workload, dict):
            raise SpecError("benchmark matrix workload must be a mapping")
        workload_id = str(workload.get("id") or "").strip()
        try:
            n = int(workload["n"])
            k = int(workload["k"])
            batch = int(workload.get("batch", 1))
            workload_weight = float(workload.get("weight", 1.0))
        except (KeyError, TypeError, ValueError) as exc:
            raise SpecError(f"benchmark matrix workload {workload_id!r} is invalid") from exc
        if not workload_id or min(n, k, batch) <= 0 or workload_weight <= 0:
            raise SpecError(f"benchmark matrix workload {workload_id!r} is invalid")
        for m in m_values:
            shape = {"m": m, "n": n, "k": k, "batch": batch}
            weight = workload_weight * (
                large_weight if m == large_m else small_total / len(small_values)
            )
            transferred = float(batch * (m * k + k * n + 4 * m + 4 * n + 2 * m * n))
            cases.append(
                BenchmarkCaseSpec(
                    id=f"{workload_id}-m{m}",
                    weight=weight,
                    critical=m in critical_m,
                    shape=shape,
                    flops=float(2 * m * n * k * batch),
                    bytes=transferred,
                )
            )
    if len({case.id for case in cases}) != len(cases):
        raise SpecError("benchmark matrix expands to duplicate case ids")
    return cases


def _benchmark_shape(item: Dict[str, Any], case_id: str) -> Optional[Dict[str, int]]:
    raw = item.get("shape")
    if raw is None and any(key in item for key in ("m", "n", "k", "batch")):
        raw = {key: item.get(key) for key in ("m", "n", "k", "batch")}
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise SpecError(f"benchmark case {case_id!r} shape must be a mapping")
    try:
        shape = {
            "m": int(raw["m"]),
            "n": int(raw["n"]),
            "k": int(raw["k"]),
            "batch": int(raw.get("batch", 1)),
        }
    except (KeyError, TypeError, ValueError) as exc:
        raise SpecError(
            f"benchmark case {case_id!r} shape requires positive m, n, k, and batch"
        ) from exc
    if any(value <= 0 for value in shape.values()):
        raise SpecError(
            f"benchmark case {case_id!r} shape requires positive m, n, k, and batch"
        )
    return shape


def _optional_positive_number(value: Any, label: str) -> Optional[float]:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise SpecError(f"{label} must be a positive finite number") from exc
    if not math.isfinite(number) or number <= 0:
        raise SpecError(f"{label} must be a positive finite number")
    return number


def bundle_digest(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*")):
        if path.name in {"__pycache__", ".pytest_cache"} or "__pycache__" in path.parts:
            continue
        rel = path.relative_to(root).as_posix()
        digest.update(rel.encode("utf-8"))
        if path.is_symlink():
            raise SpecError(f"evaluator bundle may not contain symlinks: {rel}")
        if path.is_file():
            digest.update(path.read_bytes())
    return digest.hexdigest()


@dataclass
class FrozenEvaluatorBundle:
    root: Path
    digest: str
    spec: KernelTaskSpec

    @classmethod
    def materialize(cls, source: Path, destination: Path) -> "FrozenEvaluatorBundle":
        manifest_path = destination / ".bundle-manifest.json"
        if destination.exists():
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                expected_digest = str(manifest["sha256"])
            except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
                raise SpecError(f"invalid frozen evaluator manifest: {manifest_path}") from exc
            frozen = cls(destination, expected_digest, KernelTaskSpec.load(destination / "task.yaml"))
            frozen.verify()
            return frozen

        source = source.expanduser().resolve()
        if not source.is_dir():
            raise SpecError(f"evaluator_bundle is not a directory: {source}")
        if not (source / "task.yaml").is_file():
            raise SpecError(f"evaluator bundle has no task.yaml: {source}")

        destination.parent.mkdir(parents=True, exist_ok=True)
        tmp = destination.with_name(destination.name + ".tmp")
        if tmp.exists():
            shutil.rmtree(tmp)
        shutil.copytree(source, tmp)
        digest = bundle_digest(tmp)
        (tmp / ".bundle-manifest.json").write_text(
            json.dumps({"schema_version": 1, "sha256": digest}, indent=2), encoding="utf-8"
        )
        os.replace(tmp, destination)
        spec = KernelTaskSpec.load(destination / "task.yaml")
        return cls(destination, digest, spec)

    def verify(self) -> None:
        actual = bundle_digest(self.root)
        # The manifest itself is added after the source digest. Recompute a
        # source-equivalent digest by temporarily excluding that one file.
        manifest = self.root / ".bundle-manifest.json"
        if manifest.exists():
            actual = _digest_without_manifest(self.root)
        if actual != self.digest:
            raise SpecError(
                f"frozen evaluator bundle changed: expected {self.digest}, got {actual}"
            )


def _digest_without_manifest(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*")):
        if path.name == ".bundle-manifest.json" or "__pycache__" in path.parts or path.name == ".pytest_cache":
            continue
        rel = path.relative_to(root).as_posix()
        digest.update(rel.encode("utf-8"))
        if path.is_symlink():
            raise SpecError(f"evaluator bundle may not contain symlinks: {rel}")
        if path.is_file():
            digest.update(path.read_bytes())
    return digest.hexdigest()
