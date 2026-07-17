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
from typing import Any, Dict, List, Optional, Tuple

from ..capabilities import normalize_features

import yaml

from metainfer.orchestrator.oracles.base import Oracle, OracleCaseResult, OracleResult
from metainfer.orchestrator.oracles.judge import JudgeInput, run_judge_batch
from ..task_types import is_cpp_framework_task
from .native_validation import (
    collect_native_process_evidence,
    hardware_validation_errors,
    native_accelerator_errors,
    validate_cpp_artifacts,
    write_native_evidence,
)


PROMPTS_FILE = Path(__file__).parent / "data" / "correctness_cases.yaml"


# How long to wait for serve.sh's HTTP server to become healthy (i.e.
# finish loading model weights and answer /v1/models with non-503).
# Default is generous: on non-NVIDIA accelerators (Hygon DCU, etc.)
# loading a 7-8B model can take several minutes; even larger models
# (70B, MoE) can take 10+ minutes. Override per-task with the env var
# METAINFER_ORACLE_STARTUP_TIMEOUT_S if a model is known to be slow.
DEFAULT_STARTUP_TIMEOUT_S = 600


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
    task_type = "gen-infer-framework-cpp"

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
    ) -> OracleResult:
        report_dir.mkdir(parents=True, exist_ok=True)
        serve_sh = iter_dir / "serve.sh"
        if not serve_sh.exists():
            return self._fail(report_dir, f"no serve.sh at {serve_sh}")
        if is_cpp_framework_task(req.get("task_type", "")):
            native_errors = validate_cpp_artifacts(iter_dir)
            native_errors.extend(hardware_validation_errors(req))
            if native_errors:
                return self._fail(
                    report_dir,
                    "native C++ preflight failed: " + "; ".join(native_errors),
                )

        port = _pick_free_port()
        cases_cfg = _load_cases(req)
        if not cases_cfg:
            return self._fail(report_dir, "no test cases configured")

        proc = None
        try:
            # Pass the real model path (if captured in requirements) to
            # serve.sh via env var so a weakly-written serve.sh still finds
            # the weights instead of falling into mock mode.
            model_dir = req.get("target_model") or (req.get("answers") or {}).get("target_model")
            proc = _start_server(serve_sh, port, report_dir, model_dir=model_dir)
            startup_to = _resolve_startup_timeout_s()
            ok, err = _wait_healthy(
                port, proc,
                startup_timeout_s=startup_to,
                report_dir=report_dir,
            )
            if not ok:
                tail = _server_log_tail(report_dir, max_chars=1200)
                msg = f"server did not become healthy within {startup_to}s: {err}"
                if tail:
                    msg += (
                        "\n--- server.stderr.log (tail) ---\n"
                        f"{tail}"
                    )
                return self._fail(report_dir, msg)

            if is_cpp_framework_task(req.get("task_type", "")):
                native_evidence = collect_native_process_evidence(proc.pid, iter_dir)
                write_native_evidence(
                    report_dir / "native-runtime-evidence.json",
                    native_evidence,
                )
                native_errors = list(native_evidence["errors"])
                native_errors.extend(native_accelerator_errors(req, native_evidence))
                if native_errors:
                    return self._fail(
                        report_dir,
                        "native runtime validation failed: "
                        + "; ".join(native_errors),
                    )

            # Hit each case
            case_results_raw: List[Tuple[Dict[str, Any], str, Optional[int], float, Optional[str]]] = []
            t_start = time.time()
            for cfg in cases_cfg:
                resp, status, elapsed, err = _send_request(port, cfg, timeout_s=60)
                case_results_raw.append((cfg, resp, status, elapsed, err))
                if time.time() - t_start > timeout_s:
                    break

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

            # Protocol/parameter semantics are framework correctness, not
            # model knowledge, so they are evaluated deterministically by
            # the oracle rather than delegated to the LLM judge.
            final_cases.extend(_run_protocol_checks(port, req))

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
                notes=(
                    f"server on port {port}; judge_mode={judge_mode}; "
                    "protocol checks include deterministic and seeded "
                    "stochastic decoding"
                ),
                judge_mode=judge_mode,
                report_path=str(report_dir / "oracle-report.json"),
            )
            (report_dir / "oracle-report.json").write_text(
                json.dumps(result.to_dict(), indent=2), encoding="utf-8")
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


def _start_server(
    serve_sh: Path, port: int, report_dir: Path,
    model_dir: Optional[str] = None,
) -> subprocess.Popen:
    # Observe occupancy before booting. This may be a shared accelerator
    # host, so the oracle must never evict an unregistered process.
    from ..gpu_preflight import preflight_gpu
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
    if "top_p" in cfg:
        payload["top_p"] = float(cfg["top_p"])
    if "seed" in cfg:
        payload["seed"] = int(cfg["seed"])
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


def _run_protocol_checks(port: int, req: Dict[str, Any]) -> List[OracleCaseResult]:
    checks = [
        _deterministic_repeat_check(port),
        _seeded_sampling_check(port),
    ]
    features = set(normalize_features(req.get("features")))
    if "paged kv cache" in features:
        checks.append(_paged_kv_boundary_check(port))
    if "continuous batching" in features:
        checks.append(_continuous_batching_check(port))
    if "streaming responses" in features:
        checks.append(_streaming_check(port))
    return checks


def _paged_kv_boundary_check(port: int) -> OracleCaseResult:
    # Long enough to cross ordinary KV block sizes while remaining bounded on
    # the small checkpoint used by diagnostic runs. This validates external
    # block-boundary semantics; E owns memory-efficiency evidence.
    prompt = " ".join(["alpha beta gamma delta"] * 96)
    cfg = {
        "prompt": prompt,
        "max_tokens": 8,
        "temperature": 0.0,
        "seed": 17,
    }
    text, status, elapsed, error = _send_request(port, cfg, timeout_s=120)
    passed = error is None and status == 200 and bool(text)
    return OracleCaseResult(
        case_id="protocol-paged-kv-boundary",
        prompt="bounded repeated-token long-context probe",
        response=text,
        judge_verdict="pass" if passed else "fail",
        judge_reason=(
            "a bounded long-context request crossed KV block boundaries"
            if passed else
            "the selected Paged KV feature failed a bounded long-context request"
        ),
        elapsed_s=elapsed,
        http_status=status,
        error=error,
        gating="hard",
    )


def _continuous_batching_check(port: int) -> OracleCaseResult:
    cfg = {
        "prompt": "Give one short sentence about a lighthouse.",
        "max_tokens": 24,
        "temperature": 0.0,
        "seed": 29,
    }
    t0 = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        futures = [
            pool.submit(_send_request, port, cfg, 120)
            for _ in range(4)
        ]
        observations = [future.result() for future in futures]
    texts = [item[0] for item in observations]
    errors = [item[3] for item in observations if item[3]]
    passed = (
        not errors
        and all(item[1] == 200 for item in observations)
        and bool(texts[0])
        and len(set(texts)) == 1
    )
    return OracleCaseResult(
        case_id="protocol-continuous-batching-isolation",
        prompt=cfg["prompt"],
        response=texts[0] if texts else "",
        judge_verdict="pass" if passed else "fail",
        judge_reason=(
            "four concurrent deterministic requests completed with isolated, identical output"
            if passed else
            "concurrent deterministic requests failed, were empty, or contaminated one another"
        ),
        elapsed_s=time.time() - t0,
        http_status=observations[-1][1] if observations else None,
        error="; ".join(str(error) for error in errors) or None,
        gating="hard",
    )


def _deterministic_repeat_check(port: int) -> OracleCaseResult:
    cfg = {
        "prompt": "Give one short sentence about a compass.",
        "max_tokens": 32,
        "temperature": 0.0,
        "seed": 123,
    }
    observations = [_send_request(port, cfg, timeout_s=60) for _ in range(3)]
    texts = [item[0] for item in observations]
    errors = [item[3] for item in observations if item[3]]
    passed = not errors and bool(texts[0]) and len(set(texts)) == 1
    return OracleCaseResult(
        case_id="protocol-deterministic-repeat",
        prompt=cfg["prompt"],
        response=texts[0] if texts else "",
        judge_verdict="pass" if passed else "fail",
        judge_reason=(
            "temperature=0 with a fixed seed produced three byte-identical responses"
            if passed else
            "temperature=0/fixed-seed responses were empty, failed, or not identical"
        ),
        elapsed_s=sum(item[2] for item in observations),
        http_status=observations[-1][1] if observations else None,
        error="; ".join(str(error) for error in errors) or None,
        gating="hard",
    )


def _seeded_sampling_check(port: int) -> OracleCaseResult:
    base = {
        "prompt": (
            "Write a vivid two-sentence description of an imaginary city at dawn."
        ),
        "max_tokens": 64,
        "temperature": 1.2,
        "top_p": 0.95,
    }
    seeds = (101, 101, 202, 303)
    observations = [
        _send_request(port, {**base, "seed": seed}, timeout_s=60)
        for seed in seeds
    ]
    narrow_top_p = _send_request(
        port, {**base, "seed": seeds[0], "top_p": 0.01}, timeout_s=60,
    )
    texts = [item[0] for item in observations]
    errors = [item[3] for item in [*observations, narrow_top_p] if item[3]]
    same_seed_reproducible = len(texts) >= 2 and texts[0] == texts[1] and bool(texts[0])
    different_seed_varies = len(set(texts[1:])) >= 2
    top_p_changes_output = bool(narrow_top_p[0]) and narrow_top_p[0] != texts[0]
    passed = (
        not errors
        and same_seed_reproducible
        and different_seed_varies
        and top_p_changes_output
    )
    reason = (
        "same seed reproduced exactly, different seeds changed stochastic output, "
        "and a narrow top_p changed the same-seed output"
        if passed else
        "sampling parameters appear ignored or unseeded: expected same-seed "
        "reproducibility, at least two outputs across different seeds, and a "
        "different output when top_p narrows from 0.95 to 0.01"
    )
    return OracleCaseResult(
        case_id="protocol-seeded-stochastic-sampling",
        prompt=base["prompt"],
        response=texts[0] if texts else "",
        judge_verdict="pass" if passed else "fail",
        judge_reason=reason,
        elapsed_s=sum(item[2] for item in observations) + narrow_top_p[2],
        http_status=narrow_top_p[1],
        error="; ".join(str(error) for error in errors) or None,
        gating="hard",
    )


def _streaming_check(port: int) -> OracleCaseResult:
    prompt = "Reply with five words about the moon."
    text, status, elapsed, error, content_type, done, content_chunks = (
        _send_stream_request(
            port,
            {
                "prompt": prompt,
                "max_tokens": 24,
                "temperature": 0.0,
                "seed": 7,
            },
            timeout_s=60,
        )
    )
    passed = (
        error is None
        and status == 200
        and "text/event-stream" in content_type.casefold()
        and bool(text)
        and content_chunks >= 2
        and done
    )
    return OracleCaseResult(
        case_id="protocol-sse-streaming",
        prompt=prompt,
        response=text,
        judge_verdict="pass" if passed else "fail",
        judge_reason=(
            "stream=true returned multiple incremental SSE content chunks "
            "followed by [DONE]"
            if passed else
            "stream=true did not return at least two non-empty incremental "
            "text/event-stream chunks ending in [DONE]"
        ),
        elapsed_s=elapsed,
        http_status=status,
        error=error,
        gating="hard",
    )


def _send_stream_request(
    port: int, cfg: Dict[str, Any], timeout_s: int,
) -> Tuple[str, Optional[int], float, Optional[str], str, bool, int]:
    url = f"http://127.0.0.1:{port}/v1/chat/completions"
    payload = {
        "model": "default",
        "messages": [{"role": "user", "content": cfg["prompt"]}],
        "max_tokens": int(cfg.get("max_tokens", 64)),
        "temperature": float(cfg.get("temperature", 0.0)),
        "stream": True,
        "seed": int(cfg.get("seed", 0)),
    }
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
        method="POST",
    )
    t0 = time.time()
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            content_type = response.headers.get("Content-Type", "")
            chunks: List[str] = []
            done = False
            for raw_line in response:
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    done = True
                    break
                try:
                    obj = json.loads(data)
                    choice = (obj.get("choices") or [{}])[0]
                    delta = choice.get("delta") or {}
                    content = delta.get("content")
                    if isinstance(content, str) and content:
                        chunks.append(content)
                except (ValueError, TypeError, IndexError, AttributeError):
                    continue
            return (
                "".join(chunks), response.status, time.time() - t0,
                None, content_type, done, len(chunks),
            )
    except urllib.error.HTTPError as exc:
        return "", exc.code, time.time() - t0, f"HTTP {exc.code}", "", False, 0
    except Exception as exc:  # noqa: BLE001
        return (
            "", None, time.time() - t0, f"{type(exc).__name__}: {exc}",
            "", False, 0,
        )


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
    custom = (req.get("answers") or {}).get("oracle_prompts_path")
    if custom:
        cp = Path(custom)
        if cp.exists():
            data = yaml.safe_load(cp.read_text(encoding="utf-8")) or []
            cases = [c for c in data if isinstance(c, dict) and "id" in c and "prompt" in c]
    return cases
