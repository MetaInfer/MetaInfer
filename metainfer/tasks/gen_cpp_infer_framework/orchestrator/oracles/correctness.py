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

import yaml

from ..hardware import (
    HardwareProfileError,
    execution_environment,
    materialize_hardware_binding,
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
    ) -> OracleResult:
        report_dir.mkdir(parents=True, exist_ok=True)
        serve_sh = iter_dir / "serve.sh"
        build_sh = iter_dir / "build.sh"
        if not build_sh.exists():
            return self._fail(report_dir, f"no build.sh at {build_sh}")
        if not serve_sh.exists():
            return self._fail(report_dir, f"no serve.sh at {serve_sh}")

        try:
            materialize_hardware_binding(req, iter_dir)
            hardware_env = execution_environment(req, iter_dir)
        except HardwareProfileError as exc:
            return self._fail(report_dir, f"hardware profile error: {exc}")

        port = _pick_free_port()
        cases_cfg = _load_cases(req)
        if not cases_cfg:
            return self._fail(report_dir, "no test cases configured")

        model_dir = req.get("target_model") or (req.get("answers") or {}).get("target_model")
        ok, build_err = _run_build_check(
            build_sh, report_dir, model_dir=model_dir,
            extra_env=hardware_env, timeout_s=min(timeout_s, 900)
        )
        if not ok:
            return self._fail(report_dir, build_err or "C++ build failed")

        proc = None
        try:
            # Pass the real model path (if captured in requirements) to
            # serve.sh via env var so a weakly-written serve.sh still finds
            # the weights instead of falling into mock mode.
            proc = _start_server(
                serve_sh, port, report_dir, model_dir=model_dir,
                extra_env=hardware_env,
            )
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
                if err is not None or status is None or status >= 500:
                    # infra-level failure — no point judging
                    preliminary.append(OracleCaseResult(
                        case_id=cfg["id"], prompt=cfg["prompt"], response=resp,
                        elapsed_s=elapsed, http_status=status, error=err,
                        judge_verdict="error",
                        judge_reason=f"http error: status={status} err={err}",
                    ))
                else:
                    preliminary.append(None)  # placeholder, filled below
                    judge_inputs.append(JudgeInput(
                        case_id=cfg["id"],
                        user_prompt=cfg["prompt"],
                        model_response=resp,
                        expected_keywords=cfg.get("expected_keywords") or [],
                    ))

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
                from ....oracles.judge import heuristic_verdict
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
                    # augment judged entry with http status + timing
                    j = judged[ji_idx]; ji_idx += 1
                    # find raw timing
                    raw = next((c for c in case_results_raw if c[0]["id"] == j.case_id), None)
                    if raw:
                        j.http_status = raw[2]
                        j.elapsed_s = raw[3]
                    final_cases.append(j)
                else:
                    final_cases.append(entry)

            # Aggregate
            total = len(final_cases)
            passed = sum(1 for c in final_cases if c.judge_verdict == "pass")
            failed_cases = [c for c in final_cases if c.judge_verdict != "pass"]
            all_passed = total > 0 and passed == total

            # Perf: average first-token-ish latency (rough proxy)
            avg_latency = (sum(c.elapsed_s for c in final_cases) / total) if total else 0.0
            perf = {
                "oracle_avg_http_latency_ms": round(avg_latency * 1000, 2),
                "oracle_cases_total": float(total),
                "oracle_cases_passed": float(passed),
            }

            reason = None
            if not all_passed:
                bits = [f"{c.case_id}={c.judge_verdict}" for c in failed_cases[:5]]
                reason = f"{len(failed_cases)}/{total} cases failed: {', '.join(bits)}"

            result = OracleResult(
                passed=all_passed,
                failure_reason=reason,
                perf=perf,
                cases=final_cases,
                notes=f"server on port {port}; judge_mode={judge_mode}",
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


def _run_build_check(
    build_sh: Path,
    report_dir: Path,
    *,
    model_dir: Optional[str] = None,
    extra_env: Optional[Dict[str, str]] = None,
    timeout_s: int = 900,
) -> Tuple[bool, Optional[str]]:
    """Run the C++ build before booting serve.sh.

    Keeping this in the immutable oracle gives the agent a clear failure
    class: compile/link errors are reported as build failures, not as a
    vague server-health timeout.
    """
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
    from ....gpu_preflight import preflight_gpu
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
    custom = (req.get("answers") or {}).get("oracle_prompts_path")
    if custom:
        cp = Path(custom)
        if cp.exists():
            data = yaml.safe_load(cp.read_text(encoding="utf-8")) or []
            cases = [c for c in data if isinstance(c, dict) and "id" in c and "prompt" in c]
    return cases
