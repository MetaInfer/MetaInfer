from __future__ import annotations

import json
import sys
from pathlib import Path

import yaml


def make_bundle(root: Path, *, speedup: float = 1.25) -> Path:
    root.mkdir(parents=True)
    spec = {
        "schema_version": 2,
        "name": "unit-gemm",
        "public_contract": {
            "operation": "C = alpha * A @ B + beta * C",
            "dtype": {"a": "fp16", "b": "fp16", "accumulation": "fp32", "c": "fp16"},
            "layout": {
                "a": "row_major", "b": "row_major", "c": "row_major",
                "trans_a": False, "trans_b": False,
            },
            "abi": {
                "entrypoint": "launch_gemm",
                "signature": "launch_gemm(A, B, C, M, N, K, stream)",
            },
        },
        "commands": {
            phase: {"argv": [sys.executable, "evaluate.py"], "timeout_s": 30}
            for phase in ("correctness", "benchmark")
        },
        "cases": {
            "correctness": ["public", "heldout"],
            "private": ["heldout"],
            "benchmark": [
                {
                    "id": "small", "weight": 3, "critical": True,
                    "shape": {"m": 2, "n": 3, "k": 4, "batch": 1},
                    "bytes": 100,
                },
                {
                    "id": "large", "weight": 1, "critical": False,
                    "shape": {"m": 4, "n": 4, "k": 4, "batch": 2},
                    "bytes": 200,
                },
            ],
        },
        "benchmark_protocol": {"warmup": 10, "samples": 100, "timer": "fake"},
        "acceptance": {
            "min_weighted_speedup": 1.01,
            "noise_threshold": 0.01,
            "max_critical_regression": 0.03,
            "require_all_cases": True,
        },
    }
    (root / "task.yaml").write_text(yaml.safe_dump(spec), encoding="utf-8")
    candidate_ms = 1.0 / speedup
    script = f'''import json, os
phase = os.environ["METAINFER_EVALUATION_PHASE"]
role = os.environ["METAINFER_EVALUATION_ROLE"]
path = os.environ["METAINFER_REPORT_PATH"]
if phase == "correctness":
    report = {{"passed": True, "cases": [
        {{"id": "public", "passed": True}},
        {{"id": "heldout", "passed": True}}
    ]}}
else:
    latency = 1.0 if role == "baseline" else {candidate_ms}
    report = {{"passed": True, "methodology": {{"warmup": 10, "samples": 100, "timer": "fake"}}, "cases": [
        {{"id": "small", "latency_ms": latency}},
        {{"id": "large", "latency_ms": latency}}
    ]}}
with open(path, "w", encoding="utf-8") as f:
    json.dump(report, f)
'''
    (root / "evaluate.py").write_text(script, encoding="utf-8")
    return root


class FakeManager:
    def __init__(self):
        self.results = {}
        self.shutdown_called = False
        self.prompts = {}

    def launch(self, spec):
        from metainfer.orchestrator.subagent_manager import AgentResult
        self.prompts[spec.role] = spec.prompt_file.read_text(encoding="utf-8")
        if spec.role == "planner":
            (spec.workdir / "plan.md").write_text("# plan\n", encoding="utf-8")
        elif spec.role == "implementer":
            (spec.workdir / "kernel.cpp").write_text("// optimized\n", encoding="utf-8")
            (spec.workdir / "CHANGELOG.md").write_text("optimized\n", encoding="utf-8")
        elif spec.role == "reviewer":
            (spec.workdir / "review.md").write_text("# review\n", encoding="utf-8")
        elif spec.role == "perf_planner":
            (spec.workdir / "perf_plan.md").write_text("# perf plan\n", encoding="utf-8")
        self.results[spec.name] = AgentResult(
            name=spec.name,
            role=spec.role,
            success=True,
            returncode=0,
            duration_s=0.01,
            attempts=1,
        )

    def result(self, name):
        return self.results.get(name)

    def shutdown(self):
        self.shutdown_called = True


class FakeBuilder:
    def __init__(self, fingerprint="fake-build-v1"):
        from types import SimpleNamespace
        self.profile = SimpleNamespace(fingerprint=fingerprint)

    def verify(self):
        return None

    def build(self, submission_dir, build_dir):
        from ..orchestrator.build import BuildResult
        build_dir.mkdir(parents=True, exist_ok=True)
        report = {
            "passed": True,
            "build_fingerprint": self.profile.fingerprint,
            "artifacts": [],
        }
        (build_dir / "compile-report.json").write_text(
            json.dumps(report), encoding="utf-8"
        )
        return BuildResult(True, build_dir, report)


class FakeProfiler:
    def __init__(self):
        from types import SimpleNamespace
        self.profile = SimpleNamespace(
            fingerprint="fake-profiler-v1", required=True,
        )

    def run(self, artifact_dir, output_dir, *, role):
        from ..orchestrator.profiler import ProfileResult
        report = {
            "passed": True,
            "profile_id": "hygon-k100-gfx928",
            "gpu_arch": "gfx928",
            "tool": "rocprofv3",
            "profile_fingerprint": self.profile.fingerprint,
            "counter_groups": [["SQ_WAVES"]],
            "cases": [{
                "id": "small", "vgpr_count": 32, "lds_bytes": 4096,
                "l2_hit_pct": 88.0, "compute_busy_pct": 75.0,
                "measured_bandwidth_gbps": 640.0,
            }],
        }
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / f"{role}-hardware-profile.json").write_text(
            json.dumps(report), encoding="utf-8"
        )
        return ProfileResult(True, report)
