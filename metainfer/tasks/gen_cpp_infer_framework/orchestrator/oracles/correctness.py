"""Inference-framework oracle.

Artifact contract (what the agent must produce in ``iter_dir/``):

    serve.sh PORT
        Bash script, executable. Starts the inference framework's HTTP
        server on the given port. Must:
        * expose an OpenAI-compatible API:
            POST /v1/chat/completions
            POST /v1/completions
            GET  /v1/models    (optional, used for health-check)
        * BLOCK in the foreground (no daemonize) — the oracle owns the
          process lifecycle.
        * honor deterministic decoding (temperature=0, fixed seed) where
          possible.

What this oracle does:

1. Picks a free localhost port.
2. Launches ``serve.sh PORT`` as a subprocess (in a new process group).
3. Polls ``GET /v1/models`` (falls back to ``POST /v1/chat/completions``
   with a trivial prompt) until the server is up or startup_timeout_s.
4. For each canned case in ``prompts.yaml``:
   * Sends an OpenAI-style request via :mod:`urllib` (stdlib, zero deps).
   * Captures status code, latency, response body.
5. Batches all (prompt, response) pairs into one judge sub-agent and parses
   the verdicts.
6. Aggregates: pass iff every case passes (with at least one judge_mode=llm).
7. Kills the server (SIGTERM → SIGKILL).
8. Writes ``oracle-report.json`` to ``report_dir``.

The oracle is **immutable** from the agent's perspective: it lives inside
the MetaInfer package, never inside ``iter_dir/``. Agents cannot edit it.
"""

from __future__ import annotations

import concurrent.futures
import json
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Set, Tuple

import yaml

from metainfer.orchestrator.requirements import req_field
from ..hardware import (
    HardwareProfileError,
    execution_environment,
    materialize_hardware_binding,
)
from ..acceptance import (
    AcceptanceContract,
    compile_suite_results,
    extract_runtime_metadata,
    validate_implementation_artifacts,
    validate_runtime_metadata,
)
from metainfer.orchestrator.oracles.base import Oracle, OracleCaseResult, OracleResult
from metainfer.orchestrator.oracles.judge import JudgeInput, run_judge_batch


PROMPTS_FILE = Path(__file__).parent / "data" / "correctness_cases.yaml"


# How long to wait for serve.sh's HTTP server to become healthy (i.e.
# finish loading model weights and answer /v1/models with non-503).
# Default is generous: on non-NVIDIA accelerators (Hygon DCU, etc.)
# loading a 7-8B model can take several minutes; even larger models
# (70B, MoE) can take 10+ minutes. Override per-task with the env var
# METAINFER_ORACLE_STARTUP_TIMEOUT_S if a model is known to be slow.
DEFAULT_STARTUP_TIMEOUT_S = 600

ORACLE_STAGES_FILE = "oracle-stages.json"

_SERVER_TARGET_ROUTES = {
    "model_path_or_serve_args",
    "gguf_metadata_or_tensor",
    "vram_or_kv_allocation",
    "http_or_lifecycle",
}
_GENERATION_TARGET_ROUTES = {"generation_timeout", "model_output"}
_CAPABILITY_TARGET_CASES = {
    "tensor_parallel": {
        "capability-runtime-metadata",
        "capability-tp-paged-cb-integration",
    },
    "paged_kv": {
        "capability-runtime-metadata",
        "capability-paged-kv-long-context",
    },
    "continuous_batching": {
        "capability-runtime-metadata",
        "capability-continuous-batching-concurrency",
    },
}


class _OracleStageReport:
    """Persist the latest layered-oracle attempt as it progresses."""

    def __init__(
        self,
        report_dir: Path,
        repair_route: Optional[Mapping[str, Any]],
    ) -> None:
        route = repair_route if isinstance(repair_route, Mapping) else {}
        self.path = report_dir / ORACLE_STAGES_FILE
        self.payload: Dict[str, Any] = {
            "schema_version": 1,
            "target_route": str(route.get("route_id") or "") or None,
            "target_category": str(route.get("category") or "") or None,
            "target_signature": str(route.get("signature") or "") or None,
            "stages": [],
            "full_oracle_completed": False,
        }
        self._write()

    @property
    def target_route(self) -> Optional[str]:
        value = self.payload.get("target_route")
        return str(value) if value else None

    def finish(
        self,
        stage_id: str,
        *,
        passed: bool,
        started_at: float,
        detail: Optional[str] = None,
        skipped: bool = False,
        full_oracle_completed: bool = False,
    ) -> None:
        stage: Dict[str, Any] = {
            "id": stage_id,
            "passed": bool(passed),
            "duration_s": round(max(0.0, time.time() - started_at), 3),
        }
        if detail:
            stage["detail"] = detail
        if skipped:
            stage["skipped"] = True
        self.payload["stages"].append(stage)
        if full_oracle_completed:
            self.payload["full_oracle_completed"] = True
        self._write()

    def _write(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.payload, indent=2) + "\n", encoding="utf-8")
        os.replace(tmp, self.path)


def _resolve_startup_timeout_s() -> int:
    """Pick the startup timeout, honoring env override.

    Order: METAINFER_ORACLE_STARTUP_TIMEOUT_S env var → default.
    Clamped to [30, 3600] so a fat-finger doesn't under/overshoot.
    """
    raw = os.environ.get("METAINFER_ORACLE_STARTUP_TIMEOUT_S")
    if raw:
        try:
            v = int(raw)
            return max(30, min(3600, v))
        except ValueError:
            pass
    return DEFAULT_STARTUP_TIMEOUT_S


class InferFrameworkOracle(Oracle):
    task_type = "gen-cpp-infer-framework"

    # ------------------------------------------------------------------ #
    # Public entry
    # ------------------------------------------------------------------ #

    def run(
        self,
        *,
        iter_dir: Path,
        req: Dict[str, Any],
        report_dir: Path,
        timeout_s: int = 600,
        manager=None,
        repair_route: Optional[Mapping[str, Any]] = None,
    ) -> OracleResult:
        report_dir.mkdir(parents=True, exist_ok=True)
        stages = _OracleStageReport(report_dir, repair_route)
        serve_sh = iter_dir / "serve.sh"
        build_sh = iter_dir / "build.sh"

        stage_started = time.time()
        artifact_errors = list(validate_implementation_artifacts(iter_dir, req))
        if not build_sh.is_file():
            artifact_errors.insert(0, f"no build.sh at {build_sh}")
        if not serve_sh.is_file() and not any(
            "serve.sh" in error for error in artifact_errors
        ):
            artifact_errors.insert(0, f"no serve.sh at {serve_sh}")
        if artifact_errors:
            reason = "C0 artifact validation failed: " + "; ".join(artifact_errors)
            stages.finish(
                "C0_artifacts", passed=False, started_at=stage_started,
                detail=reason,
            )
            return self._fail(report_dir, reason)
        try:
            materialize_hardware_binding(req, iter_dir)
            hardware_env = execution_environment(req, iter_dir)
        except HardwareProfileError as exc:
            reason = f"hardware profile error: {exc}"
            stages.finish(
                "C0_artifacts", passed=False, started_at=stage_started,
                detail=reason,
            )
            return self._fail(report_dir, reason)

        cases_cfg = _load_cases(req)
        if not cases_cfg:
            reason = "no test cases configured"
            stages.finish(
                "C0_artifacts", passed=False, started_at=stage_started,
                detail=reason,
            )
            return self._fail(report_dir, reason)
        stages.finish(
            "C0_artifacts", passed=True, started_at=stage_started,
            detail="static artifact and task binding checks passed",
        )

        model_dir = req_field(req, "target_model")
        stage_started = time.time()
        ok, build_err = _run_build_check(
            build_sh, report_dir, model_dir=model_dir,
            extra_env=hardware_env, timeout_s=min(timeout_s, 900)
        )
        if not ok:
            reason = build_err or "C++ build failed"
            stages.finish(
                "C1_build", passed=False, started_at=stage_started,
                detail=reason,
            )
            return self._fail(report_dir, reason)
        stages.finish(
            "C1_build", passed=True, started_at=stage_started,
            detail="incremental build passed",
        )

        stage_started = time.time()
        numeric_ok, numeric_err, numeric_summary = _run_numeric_check(
            iter_dir / "build" / "qwen3_numeric_tests",
            report_dir / "numeric-test-report.json",
            iter_dir=iter_dir,
            report_dir=report_dir,
            req=req,
            extra_env=hardware_env,
            timeout_s=min(timeout_s, 180),
        )
        if not numeric_ok:
            reason = numeric_err or "C0.1 numeric tests failed"
            stages.finish(
                "C2_numeric", passed=False, started_at=stage_started,
                detail=reason,
            )
            return self._fail(report_dir, reason)
        stages.finish(
            "C2_numeric", passed=True, started_at=stage_started,
            detail="required operator numeric cases passed",
        )

        port = _pick_free_port()
        proc = None
        models_payload: Optional[Dict[str, Any]] = None
        metadata_errors: Tuple[str, ...] = ()
        targeted_probe_cases: List[OracleCaseResult] = []

        def ensure_server() -> Optional[str]:
            nonlocal proc, models_payload, metadata_errors
            if proc is not None:
                return None
            try:
                proc = _start_server(
                    serve_sh, port, report_dir, model_dir=model_dir,
                    extra_env=hardware_env,
                )
            except Exception as exc:  # noqa: BLE001 - preserve stage evidence
                return f"server failed to launch: {exc!r}"
            startup_to = _resolve_startup_timeout_s()
            healthy, error = _wait_healthy(
                port, proc,
                startup_timeout_s=startup_to,
                report_dir=report_dir,
            )
            if not healthy:
                tail = _server_log_tail(report_dir, max_chars=1200)
                message = (
                    f"server did not become healthy within {startup_to}s: {error}"
                )
                if tail:
                    message += (
                        "\n--- server.stderr.log (tail) ---\n"
                        f"{tail}"
                    )
                return message
            models_payload = _fetch_models_payload(port)
            metadata_errors = validate_runtime_metadata(models_payload, req)
            return None

        try:
            stage_started = time.time()
            target_route = stages.target_route
            if target_route is None:
                stages.finish(
                    "C3_targeted", passed=True, started_at=stage_started,
                    detail="no repaired failure route supplied", skipped=True,
                )
            elif target_route == "build_or_link":
                stages.finish(
                    "C3_targeted", passed=True, started_at=stage_started,
                    detail="targeted build route reused the passing C1 result",
                )
            elif target_route == "numeric_or_nonfinite":
                stages.finish(
                    "C3_targeted", passed=True, started_at=stage_started,
                    detail="targeted numeric route reused the passing C2 result",
                )
            elif target_route in _SERVER_TARGET_ROUTES:
                error = ensure_server()
                if error is None and models_payload is None:
                    error = "targeted /v1/models probe did not return a JSON object"
                if error:
                    stages.finish(
                        "C3_targeted", passed=False, started_at=stage_started,
                        detail=error,
                    )
                    return self._fail(report_dir, error)
                stages.finish(
                    "C3_targeted", passed=True, started_at=stage_started,
                    detail=(
                        f"targeted {target_route} server startup, health, and "
                        "/v1/models checks passed"
                    ),
                )
            elif target_route in _CAPABILITY_TARGET_CASES:
                error = ensure_server()
                if error:
                    stages.finish(
                        "C3_targeted", passed=False, started_at=stage_started,
                        detail=error,
                    )
                    return self._fail(report_dir, error)
                targeted_probe_cases = _run_capability_probes(
                    port, req, models_payload=models_payload,
                    metadata_errors=metadata_errors,
                    only_case_ids=_CAPABILITY_TARGET_CASES[target_route],
                )
                failed_target_probes = [
                    case for case in targeted_probe_cases
                    if case.judge_verdict != "pass"
                ]
                if not targeted_probe_cases:
                    reason = (
                        f"targeted route {target_route} has no active capability probe"
                    )
                    stages.finish(
                        "C3_targeted", passed=False, started_at=stage_started,
                        detail=reason,
                    )
                    return self._fail(report_dir, reason)
                if failed_target_probes:
                    reason = "targeted capability check failed: " + "; ".join(
                        f"{case.case_id}: {case.judge_reason or case.error or 'failed'}"
                        for case in failed_target_probes
                    )
                    stages.finish(
                        "C3_targeted", passed=False, started_at=stage_started,
                        detail=reason,
                    )
                    return self._fail(report_dir, reason)
                stages.finish(
                    "C3_targeted", passed=True, started_at=stage_started,
                    detail=(
                        "targeted capability probes passed: "
                        + ", ".join(case.case_id for case in targeted_probe_cases)
                    ),
                )
            elif target_route in _GENERATION_TARGET_ROUTES:
                error = ensure_server()
                if error:
                    stages.finish(
                        "C3_targeted", passed=False, started_at=stage_started,
                        detail=error,
                    )
                    return self._fail(report_dir, error)
                smoke_cfg = dict(cases_cfg[0])
                smoke_cfg["id"] = f"targeted-{target_route}"
                smoke_cfg["max_tokens"] = min(
                    4, max(1, int(smoke_cfg.get("max_tokens", 4)))
                )
                smoke_result = _send_request(port, smoke_cfg, timeout_s=60)
                smoke_error = _request_error_detail(smoke_result)
                if smoke_error is None and not smoke_result[0].strip():
                    smoke_error = "targeted generation returned an empty response"
                if smoke_error:
                    reason = f"targeted {target_route} check failed: {smoke_error}"
                    stages.finish(
                        "C3_targeted", passed=False, started_at=stage_started,
                        detail=reason,
                    )
                    return self._fail(report_dir, reason)
                stages.finish(
                    "C3_targeted", passed=True, started_at=stage_started,
                    detail=f"targeted {target_route} short generation passed",
                )
            else:
                stages.finish(
                    "C3_targeted", passed=True, started_at=stage_started,
                    detail=f"route {target_route} has no deterministic narrow check",
                    skipped=True,
                )

            stage_started = time.time()
            error = ensure_server()
            if error:
                stages.finish(
                    "C4_full", passed=False, started_at=stage_started,
                    detail=error,
                )
                return self._fail(report_dir, error)

            completed_probe_ids = {
                case.case_id for case in targeted_probe_cases
            }
            remaining_probe_ids = (
                _required_capability_probe_ids(req) - completed_probe_ids
            )
            probe_cases = list(targeted_probe_cases)
            if remaining_probe_ids:
                probe_cases.extend(_run_capability_probes(
                    port, req, models_payload=models_payload,
                    metadata_errors=metadata_errors,
                    only_case_ids=remaining_probe_ids,
                ))
            probe_cases.insert(0, OracleCaseResult(
                case_id="numeric-operator-contract",
                prompt="C0.1 independent HIP operator numeric tests",
                response=json.dumps(numeric_summary or {}, sort_keys=True)[:2000],
                judge_verdict="pass",
                judge_reason="all required numeric cases passed without skips",
                gating="hard",
            ))

            # Hit each case
            case_results_raw: List[Tuple[Dict[str, Any], str, Optional[int], float, Optional[str]]] = []
            t_start = time.time()
            for cfg in cases_cfg:
                resp, status, elapsed, err = _send_request(port, cfg, timeout_s=60)
                case_results_raw.append((cfg, resp, status, elapsed, err))
                if time.time() - t_start > timeout_s:
                    break
            if len(case_results_raw) != len(cases_cfg):
                reason = (
                    "full correctness request set did not complete within "
                    f"{timeout_s}s: attempted {len(case_results_raw)}/{len(cases_cfg)}"
                )
                stages.finish(
                    "C4_full", passed=False, started_at=stage_started,
                    detail=reason,
                )
                return self._fail(report_dir, reason)

            # Build judge inputs (only judge cases that produced a response)
            judge_inputs: List[JudgeInput] = []
            preliminary: List[OracleCaseResult] = []
            for cfg, resp, status, elapsed, err in case_results_raw:
                gating = cfg.get("gating") or "hard"
                if err is not None or status is None or status >= 500:
                    # infra-level failure — no point judging.
                    # NOTE: even "soft" cases escalate to hard when the
                    # failure is infrastructural (HTTP 5xx / timeout). A
                    # model getting the wrong number is soft; the server
                    # being unreachable is a real framework regression.
                    if gating == "soft":
                        gating = "hard"
                    preliminary.append(OracleCaseResult(
                        case_id=cfg["id"], prompt=cfg["prompt"], response=resp,
                        elapsed_s=elapsed, http_status=status, error=err,
                        judge_verdict="error",
                        judge_reason=f"http error: status={status} err={err}",
                        gating=gating,
                    ))
                else:
                    preliminary.append(None)  # placeholder, filled below
                    judge_inputs.append(JudgeInput(
                        case_id=cfg["id"],
                        user_prompt=cfg["prompt"],
                        model_response=resp,
                        expected_keywords=cfg.get("expected_keywords") or [],
                        # Stash gating on JudgeInput? Judge doesn't see
                        # it. We'll re-stamp from cfg after the verdict
                        # comes back, keyed by case_id.
                    ))

            # Build cfg lookup by case_id so we can stamp gating onto
            # judge results (JudgeInput doesn't carry it through).
            gating_by_id = {cfg["id"]: (cfg.get("gating") or "hard")
                            for cfg, *_ in case_results_raw}

            # Run judge sub-agent (if manager available)
            judged: List[OracleCaseResult] = []
            if manager is not None and judge_inputs:
                judged = run_judge_batch(
                    manager=manager, cases=judge_inputs,
                    workdir=report_dir, log_dir=report_dir,
                    timeout_s=max(120, min(300, timeout_s)),
                    judge_name="infer-framework-judge",
                )
                judge_mode = "llm"
            else:
                # Heuristic-only path (no manager passed — e.g. dry-run)
                from metainfer.orchestrator.oracles.judge import heuristic_verdict
                judged = []
                for ji in judge_inputs:
                    v = heuristic_verdict(ji)
                    judged.append(OracleCaseResult(
                        case_id=ji.case_id, prompt=ji.user_prompt,
                        response=ji.model_response, elapsed_s=0.0,
                        judge_verdict=v["verdict"], judge_reason=v["reason"],
                    ))
                judge_mode = "heuristic"
            if len(judged) != len(judge_inputs):
                reason = (
                    "full correctness judge returned an incomplete verdict set: "
                    f"{len(judged)}/{len(judge_inputs)}"
                )
                stages.finish(
                    "C4_full", passed=False, started_at=stage_started,
                    detail=reason,
                )
                return self._fail(report_dir, reason)

            # Merge: replace placeholders with judged results
            final_cases: List[OracleCaseResult] = []
            ji_idx = 0
            for entry in preliminary:
                if entry is None:
                    # augment judged entry with http status + timing + gating
                    j = judged[ji_idx]; ji_idx += 1
                    # find raw timing
                    raw = next((c for c in case_results_raw if c[0]["id"] == j.case_id), None)
                    if raw:
                        j.http_status = raw[2]
                        j.elapsed_s = raw[3]
                    j.gating = gating_by_id.get(j.case_id, "hard")
                    final_cases.append(j)
                else:
                    final_cases.append(entry)

            # Capability probes are immutable, deterministic checks owned by
            # this oracle. They are never sent to the LLM judge and therefore
            # cannot be softened by a model-quality verdict.
            final_cases.extend(probe_cases)

            # Aggregate — SOFT-GATE POLICY:
            # Only HARD cases gate the pass verdict. SOFT cases (e.g.
            # arith-basic on an 8B model) are still recorded + surfaced
            # in failure_reason for visibility, but their failure alone
            # doesn't flip passed=False. This avoids burning repair
            # budget on model-quality limitations that no framework
            # change can fix.
            total = len(final_cases)
            passed = sum(1 for c in final_cases if c.judge_verdict == "pass")
            hard_cases = [c for c in final_cases if c.gating != "soft"]
            soft_cases = [c for c in final_cases if c.gating == "soft"]
            hard_failed = [c for c in hard_cases if c.judge_verdict != "pass"]
            soft_failed = [c for c in soft_cases if c.judge_verdict != "pass"]
            hard_total = len(hard_cases)
            hard_passed = len(hard_cases) - len(hard_failed)
            # All HARD cases pass (and at least one exists) → oracle
            # passes. Soft failures are noted in `reason` for visibility.
            all_passed = hard_total > 0 and len(hard_failed) == 0

            # Perf: average first-token-ish latency (rough proxy)
            avg_latency = (sum(c.elapsed_s for c in final_cases) / total) if total else 0.0
            perf = {
                "oracle_avg_http_latency_ms": round(avg_latency * 1000, 2),
                "oracle_cases_total": float(total),
                "oracle_cases_passed": float(passed),
                "oracle_hard_total": float(hard_total),
                "oracle_hard_passed": float(hard_passed),
                "oracle_soft_total": float(len(soft_cases)),
                "oracle_soft_passed": float(len(soft_cases) - len(soft_failed)),
            }

            # Build failure_reason. Hard failures gate the verdict; soft
            # failures are surfaced as an advisory tail so the user sees
            # model-quality limitations without the oracle failing on them.
            reason = None
            if not all_passed:
                hard_bits = [f"{c.case_id}={c.judge_verdict}"
                             for c in hard_failed[:5]]
                reason = (
                    f"{len(hard_failed)}/{hard_total} hard cases failed: "
                    f"{', '.join(hard_bits)}"
                )
            if soft_failed:
                soft_bits = [f"{c.case_id}={c.judge_verdict}"
                             for c in soft_failed[:5]]
                advisory = (
                    f"{len(soft_failed)}/{len(soft_cases)} soft cases "
                    f"failed (advisory only, model-quality probe, does NOT "
                    f"block the verdict): {', '.join(soft_bits)}"
                )
                if reason:
                    reason = reason + "; " + advisory
                else:
                    reason = advisory

            result = OracleResult(
                passed=all_passed,
                failure_reason=reason,
                perf=perf,
                cases=final_cases,
                notes=f"server on port {port}; judge_mode={judge_mode}",
                judge_mode=judge_mode,
                report_path=str(report_dir / "oracle-report.json"),
            )
            contract = AcceptanceContract.from_request(req)
            probe_verdicts = {
                case.case_id: case.judge_verdict for case in probe_cases
            }
            probe_ids = set(probe_verdicts)
            baseline_hard_passed = all(
                case.judge_verdict == "pass"
                for case in final_cases
                if case.case_id not in probe_ids and case.gating != "soft"
            )
            suite_results = compile_suite_results(
                contract,
                baseline_passed=baseline_hard_passed,
                probe_verdicts=probe_verdicts,
            )
            acceptance = {
                "required_capabilities": list(
                    contract.required_capabilities
                ),
                "required_suites": list(
                    contract.correctness_suites
                ),
                "suite_results": list(suite_results),
                "probes": [
                    {
                        "id": case.case_id,
                        "verdict": case.judge_verdict,
                        "reason": case.judge_reason,
                        "error": case.error,
                    }
                    for case in probe_cases
                ],
                "passed": all(item["passed"] for item in suite_results),
            }
            report_payload = result.to_dict()
            report_payload["acceptance"] = acceptance
            (report_dir / "oracle-report.json").write_text(
                json.dumps(report_payload, indent=2), encoding="utf-8")
            stages.finish(
                "C4_full", passed=all_passed, started_at=stage_started,
                detail=reason or "full correctness oracle passed",
                full_oracle_completed=True,
            )
            return result

        finally:
            if proc is not None:
                _kill_server(proc)

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    def _fail(self, report_dir: Path, reason: str) -> OracleResult:
        result = OracleResult(
            passed=False, failure_reason=reason, judge_mode="disabled",
            report_path=str(report_dir / "oracle-report.json"),
        )
        report_dir.mkdir(parents=True, exist_ok=True)
        (report_dir / "oracle-report.json").write_text(
            json.dumps(result.to_dict(), indent=2), encoding="utf-8")
        return result


# --------------------------------------------------------------------------- #
# Server lifecycle
# --------------------------------------------------------------------------- #


def _pick_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _run_build_check(
    build_sh: Path,
    report_dir: Path,
    *,
    model_dir: Optional[str] = None,
    extra_env: Optional[Dict[str, str]] = None,
    timeout_s: int = 900,
) -> Tuple[bool, Optional[str]]:
    """Run the C++ build before booting serve.sh."""
    out_path = report_dir / "cpp-build.stdout.log"
    err_path = report_dir / "cpp-build.stderr.log"
    env = dict(os.environ)
    if model_dir:
        env["MODEL_DIR"] = str(model_dir)
    if extra_env:
        env.update(extra_env)
    env.setdefault("OMP_NUM_THREADS", "8")
    try:
        with open(out_path, "wb") as out_fp, open(err_path, "wb") as err_fp:
            proc = subprocess.run(
                ["bash", str(build_sh)],
                cwd=str(build_sh.parent),
                env=env,
                stdout=out_fp,
                stderr=err_fp,
                timeout=max(60, int(timeout_s)),
                check=False,
            )
    except subprocess.TimeoutExpired:
        return False, f"C++ build timed out after {timeout_s}s"
    except OSError as exc:
        return False, f"C++ build failed to launch: {exc!r}"
    if proc.returncode != 0:
        tail = _build_log_tail(report_dir)
        msg = f"C++ build failed with rc={proc.returncode}"
        if tail:
            msg += f"\n--- cpp-build logs (tail) ---\n{tail}"
        return False, msg
    return True, None


def _run_numeric_check(
    numeric_bin: Path,
    numeric_report: Path,
    *,
    iter_dir: Path,
    report_dir: Path,
    req: Dict[str, Any],
    extra_env: Optional[Dict[str, str]] = None,
    timeout_s: int = 180,
) -> Tuple[bool, Optional[str], Optional[Dict[str, Any]]]:
    """Run the generated independent operator suite before loading weights."""
    if not numeric_bin.is_file():
        return False, f"C0.1 numeric test binary is missing: {numeric_bin}", None
    stdout_path = report_dir / "numeric-test.stdout.log"
    stderr_path = report_dir / "numeric-test.stderr.log"
    env = dict(os.environ)
    if extra_env:
        env.update(extra_env)
    try:
        with stdout_path.open("wb") as stdout_fp, stderr_path.open("wb") as stderr_fp:
            completed = subprocess.run(
                [str(numeric_bin), "--report", str(numeric_report)],
                cwd=str(iter_dir),
                env=env,
                stdout=stdout_fp,
                stderr=stderr_fp,
                timeout=max(30, int(timeout_s)),
                check=False,
            )
    except subprocess.TimeoutExpired:
        return False, f"C0.1 numeric tests timed out after {timeout_s}s", None
    except OSError as exc:
        return False, f"C0.1 numeric tests failed to launch: {exc}", None

    try:
        report = json.loads(numeric_report.read_text(encoding="utf-8"))
    except OSError as exc:
        return False, f"C0.1 numeric report is missing: {exc}", None
    except json.JSONDecodeError as exc:
        return False, f"C0.1 numeric report is invalid JSON: {exc}", None
    errors = _validate_numeric_report(report, req)
    if completed.returncode != 0:
        errors.insert(0, f"numeric binary exited with rc={completed.returncode}")
    if errors:
        tail = _numeric_log_tail(report_dir)
        reason = "C0.1 numeric tests failed: " + "; ".join(errors)
        if tail:
            reason += f"\n--- numeric test logs (tail) ---\n{tail}"
        return False, reason, report if isinstance(report, dict) else None
    return True, None, report


def _required_numeric_cases(req: Dict[str, Any]) -> Tuple[str, ...]:
    contract = AcceptanceContract.from_request(req)
    cases = [
        "cast_fp32_to_fp16",
        "rms_norm",
        "per_head_rms_norm",
        "rope_neox",
        "kv_write",
        "prefill_gqa",
        "swiglu",
        "greedy",
    ]
    weight_format = str(contract.parameters.get("weight_format", "") or "").casefold()
    target_model = str(contract.parameters.get("target_model", "") or "").casefold()
    if "q8" in weight_format or "q8_0" in target_model:
        cases.extend(("dequant_q8_0", "q8_embedding", "q8_linear"))
    else:
        cases.append("f16_linear")
    if contract.requires("paged_kv_cache"):
        cases.append("paged_attention")
    if contract.requires("continuous_batching"):
        cases.append("packed_sequence_isolation")
    if contract.requires("paged_kv_cache") or contract.requires("continuous_batching"):
        cases.append("kv_capacity_contract")
    if contract.requires("tensor_parallelism"):
        cases.extend(("tp_collective", "tp_sharded_linear"))
    return tuple(cases)


def _validate_numeric_report(report: Any, req: Dict[str, Any]) -> List[str]:
    if not isinstance(report, dict):
        return ["numeric report root must be an object"]
    errors = []
    if report.get("passed") is not True:
        errors.append("numeric report passed is not true")
    raw_cases = report.get("cases")
    if not isinstance(raw_cases, list):
        return [*errors, "numeric report cases must be a list"]
    by_id = {
        str(item.get("id")): item
        for item in raw_cases
        if isinstance(item, dict) and isinstance(item.get("id"), str)
    }
    for case_id in _required_numeric_cases(req):
        case = by_id.get(case_id)
        if case is None:
            errors.append(f"missing required numeric case {case_id}")
        elif case.get("passed") is not True:
            state = "skipped" if case.get("skipped") is True else "failed"
            errors.append(f"required numeric case {case_id} {state}")
    return errors


def _numeric_log_tail(report_dir: Path, *, max_chars: int = 2000) -> str:
    chunks = []
    for name in ("numeric-test.stderr.log", "numeric-test.stdout.log"):
        path = report_dir / name
        try:
            content = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        if content.strip():
            chunks.append(f"--- {name} ---\n{content[-max_chars:]}")
    return "\n".join(chunks)[-max_chars * 2:]


def _start_server(
    serve_sh: Path, port: int, report_dir: Path,
    model_dir: Optional[str] = None,
    extra_env: Optional[Dict[str, str]] = None,
) -> subprocess.Popen:
    # Kill any orphan process holding GPU VRAM before booting. A previous
    # crashed experiment (implementer's self-test, prior iteration's C
    # oracle, etc.) can leave model weights stranded in VRAM; our serve.sh
    # then OOMs at load time and the C step fails with a confusing "out
    # of memory" that has nothing to do with the agent's code.
    from metainfer.orchestrator.gpu_preflight import preflight_gpu
    preflight_gpu(label="c-oracle")

    log_fp = open(report_dir / "server.stdout.log", "wb")
    err_fp = open(report_dir / "server.stderr.log", "wb")
    env = dict(os.environ)
    env["METAINFER_ORACLE_PORT"] = str(port)
    # Surface the real model path to the agent's serve.sh. Without this,
    # serve.sh has no way to know where the weights live (the oracle only
    # passes the port as $1), and any sane defensive fallback inside
    # serve.sh ends up serving mock responses — every C step then fails.
    if model_dir:
        env["MODEL_DIR"] = str(model_dir)
    if extra_env:
        env.update(extra_env)
    # Best-effort determinism hints
    env.setdefault("PYTHONHASHSEED", "0")
    env.setdefault("OMP_NUM_THREADS", "8")
    return subprocess.Popen(
        ["bash", str(serve_sh), str(port)],
        stdout=log_fp, stderr=err_fp,
        cwd=str(serve_sh.parent),
        env=env, start_new_session=True,
    )


def _wait_healthy(
    port: int,
    proc: subprocess.Popen,
    *,
    startup_timeout_s: int,
    report_dir: Optional[Path] = None,
) -> Tuple[bool, Optional[str]]:
    """Poll /v1/models until the server is ready, the deadline expires,
    OR the underlying serve.sh process dies (then we bail immediately
    rather than waiting out the full deadline — much faster feedback
    when load crashes instead of just being slow).

    Treats 503 as "still loading, keep waiting" (NOT a fatal error),
    since the framework legitimately returns 503 while weights are
    loading into VRAM. Anything in [200, 500) is considered healthy.
    """
    deadline = time.time() + startup_timeout_s
    url_models = f"http://127.0.0.1:{port}/v1/models"
    url_chat = f"http://127.0.0.1:{port}/v1/chat/completions"
    last_err: Optional[str] = None
    last_poll = time.time()
    while time.time() < deadline:
        # Bail early if serve.sh died — no point polling a dead process.
        rc = proc.poll()
        if rc is not None:
            tail = ""
            if report_dir is not None:
                tail = _server_log_tail(report_dir, max_chars=800)
            return False, (
                f"serve.sh exited (rc={rc}) before becoming healthy. "
                + (f"stderr tail:\n{tail}" if tail else "")
            )
        # Try /v1/models first (cheap)
        try:
            req = urllib.request.Request(url_models, headers={"Accept": "application/json"})
            with urllib.request.urlopen(req, timeout=5) as resp:
                if 200 <= resp.status < 500:
                    return True, None
        except urllib.error.HTTPError as e:
            # 404 / 405 means the server is up but doesn't implement /v1/models;
            # fall through to the chat-completions probe.
            if e.code in (404, 405, 401):
                if _probe_chat(url_chat):
                    return True, None
            # 503 = server up but model still loading — keep polling,
            # don't treat as fatal. Other 5xx → record and keep polling
            # (could be transient; server just bound the port).
            last_err = f"HTTP {e.code}"
        except urllib.error.URLError as e:
            last_err = f"URLError: {e.reason!r}"
        except Exception as e:  # noqa: BLE001
            last_err = f"{type(e).__name__}: {e}"
        last_poll = time.time()
        time.sleep(1.0)
    return False, last_err


def _fetch_models_payload(port: int) -> Optional[Dict[str, Any]]:
    """Fetch /v1/models once after health polling for capability evidence."""
    url = f"http://127.0.0.1:{port}/v1/models"
    try:
        req = urllib.request.Request(url, headers={"Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
        obj = json.loads(raw)
        return obj if isinstance(obj, dict) else None
    except Exception:  # noqa: BLE001 - metadata is reported as a probe failure
        return None


def _run_capability_probes(
    port: int,
    req: Dict[str, Any],
    *,
    models_payload: Optional[Dict[str, Any]],
    metadata_errors: Tuple[str, ...],
    only_case_ids: Optional[Set[str]] = None,
) -> List[OracleCaseResult]:
    """Run behavior probes selected by the frozen capability contract."""
    contract = AcceptanceContract.from_request(req)
    probes: List[OracleCaseResult] = []

    def selected(case_id: str) -> bool:
        return only_case_ids is None or case_id in only_case_ids

    if any(contract.requires(cap) for cap in (
        "tensor_parallelism", "paged_kv_cache", "continuous_batching"
    )) and selected("capability-runtime-metadata"):
        probes.append(OracleCaseResult(
            case_id="capability-runtime-metadata",
            prompt="GET /v1/models runtime capability metadata",
            response=json.dumps(models_payload or {}, sort_keys=True)[:2000],
            judge_verdict="pass" if not metadata_errors else "fail",
            judge_reason="runtime metadata matches frozen contract"
            if not metadata_errors else "; ".join(metadata_errors),
            gating="hard",
        ))

    if contract.requires("paged_kv_cache") and selected(
        "capability-paged-kv-long-context"
    ):
        block_size = int(
            contract.capability_parameters.get("paged_kv_cache", {}).get("block_size", 16)
            or 16
        )
        prompt = "paged kv lifecycle probe " * max(8, block_size * 2)
        results = [
            _send_request(
                port,
                {"id": "paged-kv-long-context", "prompt": prompt, "max_tokens": 8},
                timeout_s=120,
            )
            for _ in range(2)
        ]
        errors = [
            detail
            for result in results
            if (detail := _request_error_detail(result)) is not None
        ]
        probes.append(OracleCaseResult(
            case_id="capability-paged-kv-long-context",
            prompt="two long-context requests exercising KV block allocation and reuse",
            judge_verdict="pass" if not errors else "fail",
            judge_reason="long-context requests completed" if not errors else "; ".join(errors),
            gating="hard",
        ))

    if contract.requires("continuous_batching") and selected(
        "capability-continuous-batching-concurrency"
    ):
        max_concurrency = int(
            contract.capability_parameters.get("continuous_batching", {}).get(
                "max_concurrency", 4
            ) or 4
        )
        count = max(1, min(max_concurrency, 8))
        request_cfgs = [
            {
                "id": f"continuous-batching-sequence-{index}",
                "prompt": (
                    f"Sequence isolation key {index}: "
                    f"summarize the distinct number {1009 + index * 97} briefly."
                ),
                "max_tokens": 8,
            }
            for index in range(count)
        ]
        # Deterministic greedy output for each distinct prompt is captured
        # before concurrency.  A shared/cleared generation map, mixed packed
        # prefill, or cross-request KV alias then becomes an observable output
        # mismatch instead of passing merely because every socket returned 200.
        baselines = [
            _send_request(port, cfg, 120) for cfg in request_cfgs
        ]
        with concurrent.futures.ThreadPoolExecutor(max_workers=count) as pool:
            futures = [
                pool.submit(_send_request, port, cfg, 120)
                for cfg in request_cfgs
            ]
            results = [future.result() for future in futures]
        errors = _request_errors(baselines, label="sequential baseline")
        errors.extend(_request_errors(results, label="concurrent request"))
        if not errors:
            for index, (baseline, concurrent_result) in enumerate(zip(baselines, results)):
                baseline_text = baseline[0]
                concurrent_text = concurrent_result[0]
                if not baseline_text.strip():
                    errors.append(f"sequence {index} returned empty deterministic baseline")
                elif concurrent_text != baseline_text:
                    errors.append(
                        f"sequence {index} changed under concurrency: "
                        f"baseline={baseline_text[:80]!r}, concurrent={concurrent_text[:80]!r}"
                    )
        observed_payload = _fetch_models_payload(port) or models_payload
        observed_metadata = extract_runtime_metadata(observed_payload) or {}
        observed_batch_size = observed_metadata.get("max_observed_batch_size")
        required_batch_size = min(2, count)
        if not isinstance(observed_batch_size, (int, float)):
            errors.append(
                "runtime metadata is missing numeric max_observed_batch_size "
                "after the concurrency probe"
            )
        elif int(observed_batch_size) < required_batch_size:
            errors.append(
                f"runner max_observed_batch_size={observed_batch_size}, expected "
                f">={required_batch_size}; concurrent HTTP success alone does not "
                "prove Continuous Batching"
            )
        probes.append(OracleCaseResult(
            case_id="capability-continuous-batching-concurrency",
            prompt=f"{count} concurrent requests with stable sequence isolation",
            judge_verdict="pass" if not errors else "fail",
            judge_reason=(
                f"{count} distinct concurrent outputs matched their deterministic "
                "sequential baselines"
            )
            if not errors else "; ".join(errors),
            gating="hard",
        ))

    # The full combination gets an additional concurrent long-context probe;
    # passing each individual feature is not enough to claim their integration.
    if contract.has_combination(
        "tp_with_paged_kv_and_continuous_batching"
    ) and selected("capability-tp-paged-cb-integration"):
        block_size = int(
            contract.capability_parameters.get("paged_kv_cache", {}).get("block_size", 16)
            or 16
        )
        prompt = "tp paged kv continuous batching integration probe " * max(8, block_size)
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            futures = [
                pool.submit(
                    _send_request, port,
                    {"id": "tp-paged-cb-integration", "prompt": prompt, "max_tokens": 8},
                    120,
                )
                for _ in range(2)
            ]
            results = [future.result() for future in futures]
        errors = [
            detail
            for result in results
            if (detail := _request_error_detail(result)) is not None
        ]
        probes.append(OracleCaseResult(
            case_id="capability-tp-paged-cb-integration",
            prompt="TP + Paged KV + Continuous Batching integration probe",
            judge_verdict="pass" if not errors else "fail",
            judge_reason="combined concurrent long-context requests completed"
            if not errors else "; ".join(errors),
            gating="hard",
        ))
    return probes


def _required_capability_probe_ids(req: Dict[str, Any]) -> Set[str]:
    """Return every immutable runtime probe required by this contract."""
    contract = AcceptanceContract.from_request(req)
    case_ids: Set[str] = set()
    if any(contract.requires(cap) for cap in (
        "tensor_parallelism", "paged_kv_cache", "continuous_batching"
    )):
        case_ids.add("capability-runtime-metadata")
    if contract.requires("paged_kv_cache"):
        case_ids.add("capability-paged-kv-long-context")
    if contract.requires("continuous_batching"):
        case_ids.add("capability-continuous-batching-concurrency")
    if contract.has_combination("tp_with_paged_kv_and_continuous_batching"):
        case_ids.add("capability-tp-paged-cb-integration")
    return case_ids


def _request_errors(
    results: List[Tuple[str, Optional[int], float, Optional[str]]],
    *,
    label: str,
) -> List[str]:
    return [
        f"{label} {index}: {detail}"
        for index, result in enumerate(results)
        if (detail := _request_error_detail(result)) is not None
    ]


def _request_error_detail(
    result: Tuple[str, Optional[int], float, Optional[str]],
) -> Optional[str]:
    """Keep a bounded response-body clue when an HTTP probe fails."""
    body, status, _elapsed, error = result
    if error is None and status is not None and status < 400:
        return None
    detail = error or f"HTTP {status}"
    snippet = " ".join(body.split())[:240]
    if snippet:
        detail += f"; response={snippet!r}"
    return detail


def _server_log_tail(report_dir: Path, *, max_chars: int = 1200) -> str:
    """Best-effort tail of server.stderr.log (preferred) + stdout.
    Returned as a single string truncated to ``max_chars``. Used to
    surface a useful clue in the oracle failure message — without this
    the agent only sees 'HTTP 503' / 'Connection refused' and has to
    dig through files manually.
    """
    chunks: List[str] = []
    for name in ("server.stderr.log", "server.stdout.log"):
        p = report_dir / name
        if not p.exists():
            continue
        try:
            data = p.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        if not data.strip():
            continue
        chunks.append(f"--- {name} (tail) ---\n{data[-max_chars:]}")
    out = "\n".join(chunks)
    return out[-(max_chars * 2):] if out else out


def _build_log_tail(report_dir: Path, *, max_chars: int = 2000) -> str:
    chunks: List[str] = []
    for name in ("cpp-build.stderr.log", "cpp-build.stdout.log"):
        p = report_dir / name
        if not p.exists():
            continue
        try:
            data = p.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        if not data.strip():
            continue
        chunks.append(f"--- {name} (tail) ---\n{data[-max_chars:]}")
    out = "\n".join(chunks)
    return out[-(max_chars * 2):] if out else out


def _probe_chat(url: str) -> bool:
    body = json.dumps({
        "model": "probe", "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 1, "temperature": 0,
    }).encode("utf-8")
    try:
        req = urllib.request.Request(
            url, data=body,
            headers={"Content-Type": "application/json", "Accept": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            return 200 <= resp.status < 500
    except Exception:  # noqa: BLE001
        return False


def _kill_server(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        try:
            proc.terminate()
        except Exception:  # noqa: BLE001
            return
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except Exception:  # noqa: BLE001
            try:
                proc.kill()
            except Exception:  # noqa: BLE001
                pass


# --------------------------------------------------------------------------- #
# HTTP request to the framework under test (stdlib only — no third-party deps)
# --------------------------------------------------------------------------- #


def _send_request(
    port: int, cfg: Dict[str, Any], timeout_s: int
) -> Tuple[str, Optional[int], float, Optional[str]]:
    """Send an OpenAI-style chat completion request. Returns
    (extracted_text, http_status, elapsed_s, error_or_None)."""
    url = f"http://127.0.0.1:{port}/v1/chat/completions"
    payload = {
        "model": cfg.get("model", "default"),
        "messages": [{"role": "user", "content": cfg["prompt"]}],
        "max_tokens": int(cfg.get("max_tokens", 256)),
        "temperature": float(cfg.get("temperature", 0.0)),
        "stream": False,
    }
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url, data=body,
        headers={"Content-Type": "application/json", "Accept": "application/json"},
        method="POST",
    )
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
            elapsed = time.time() - t0
            text = _extract_text(raw)
            return text, resp.status, elapsed, None
    except urllib.error.HTTPError as e:
        elapsed = time.time() - t0
        try:
            err_body = e.read().decode("utf-8", errors="replace")
        except Exception:  # noqa: BLE001
            err_body = ""
        return err_body, e.code, elapsed, f"HTTP {e.code}"
    except urllib.error.URLError as e:
        return "", None, time.time() - t0, f"URLError: {e.reason!r}"
    except Exception as e:  # noqa: BLE001
        return "", None, time.time() - t0, f"{type(e).__name__}: {e}"


def _extract_text(raw_body: str) -> str:
    """Extract the assistant message text from an OpenAI-format response.

    Tolerates minor shape variations (e.g. choices[0].message.content vs
    choices[0].text for /v1/completions-style).
    """
    try:
        obj = json.loads(raw_body)
    except json.JSONDecodeError:
        return raw_body[:2000]
    choices = obj.get("choices") if isinstance(obj, dict) else None
    if not isinstance(choices, list) or not choices:
        return ""
    first = choices[0]
    if not isinstance(first, dict):
        return ""
    msg = first.get("message")
    if isinstance(msg, dict) and isinstance(msg.get("content"), str):
        return msg["content"]
    if isinstance(first.get("text"), str):
        return first["text"]
    return ""


# --------------------------------------------------------------------------- #
# Case loading
# --------------------------------------------------------------------------- #


def _load_cases(req: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Load canned cases from prompts.yaml, optionally overridden by user.

    A user can drop a custom ``prompts.yaml`` into
    ``<state_dir>/oracle/prompts.yaml`` to replace or extend the defaults.
    """
    cases: List[Dict[str, Any]] = []
    # 1. defaults
    if PROMPTS_FILE.exists():
        data = yaml.safe_load(PROMPTS_FILE.read_text(encoding="utf-8")) or []
        cases.extend([c for c in data if isinstance(c, dict) and "id" in c and "prompt" in c])
    # 2. user overrides — look in req answers for a path, else skip
    custom = req_field(req, "oracle_prompts_path")
    if custom:
        cp = Path(custom)
        if cp.exists():
            data = yaml.safe_load(cp.read_text(encoding="utf-8")) or []
            cases = [c for c in data if isinstance(c, dict) and "id" in c and "prompt" in c]
    return cases
